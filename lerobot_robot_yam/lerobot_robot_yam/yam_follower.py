"""YAM follower with i2rt-authentic motor-chain behavior."""

import logging
import time
from functools import cached_property
from typing import Optional

import numpy as np

from lerobot.cameras.utils import make_cameras_from_configs
from yam_common.dm.dm_driver import (
    CanInterface,
    DMChainCanInterface,
    EncoderChain,
    PassiveEncoderReader,
    ReceiveMode,
)
from lerobot.processor import RobotAction, RobotObservation
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from lerobot.robots.robot import Robot

from .config_yam_follower import YAMFollowerRobotConfig
from yam_common.mujoco_kdl import get_yam_mujoco_kdl
from yam_common.motor_chain_robot import MotorChainRobot
from yam_common.utils import GripperType

logger = logging.getLogger(__name__)


class YAMFollower(Robot):
    """
    Robot implementation for YAM arm with DM motors.

    This class mirrors i2rt/robots/get_robot.py + motor_chain_robot.py.
    """

    config_class = YAMFollowerRobotConfig
    name = "yam_follower"

    def __init__(self, config: YAMFollowerRobotConfig):
        """
        Initialize YAM Follower robot.

        Args:
            config: Robot configuration
        """
        super().__init__(config)
        self.config = config

        self._motor_chain_robot: Optional[MotorChainRobot] = None
        self._motor_chain: Optional[DMChainCanInterface] = None
        self._arm_joints = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "wrist_yaw"]
        self._gripper_joint = "gripper"
        self._motor_names = self._build_motor_names()
        self.cameras = make_cameras_from_configs(config.cameras)

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def _motors_ft(self) -> dict[str, type]:
        """Motor feature types for observations/actions."""
        return {f"{motor}.pos": float for motor in self._motor_names}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        """Camera feature shapes for observations."""
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3)
            for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        """Features returned by get_observation()."""
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        """Features expected by send_action()."""
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        """Check if robot is fully connected."""
        return self._motor_chain_robot is not None and all(cam.is_connected for cam in self.cameras.values())

    @property
    def is_calibrated(self) -> bool:
        """Check if robot is calibrated."""
        return True

    # =========================================================================
    # Connection
    # =========================================================================

    def connect(self, calibrate: bool = True) -> None:
        """
        Connect to robot hardware.

        This method:
        1. Connects to motor bus (enables motors)
        2. Runs calibration if needed
        3. Connects cameras
        4. Starts gravity compensation thread
        5. Initializes gripper limiter and smoothing

        Args:
            calibrate: If True, run calibration if not already calibrated
        """
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self._motor_chain_robot = self._build_motor_chain_robot()
        try:
            for cam in self.cameras.values():
                cam.connect()
        except Exception as exc:
            for cam in self.cameras.values():
                try:
                    cam.disconnect()
                except Exception:
                    pass
            if self._motor_chain_robot is not None:
                self._enter_zero_torque_mode_safely(reason="camera connect failed", exc=exc)
                self._motor_chain_robot.close()
                self._motor_chain_robot = None
            raise

        logger.info(f"{self} connected.")

    def calibrate(self) -> None:
        """No-op calibration (handled internally by motor chain)."""
        return

    def configure(self) -> None:
        """
        Configure robot after connection.

        Sets up operating mode and PID parameters.
        """
        pass

    # =========================================================================
    # Observation and Action
    # =========================================================================

    def get_observation(self) -> RobotObservation:
        # lerobot compatible method
        """Return LeRobot-style observation dict."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        obs = self._motor_chain_robot.get_observations()
        joint_pos = np.concatenate([obs["joint_pos"], obs.get("gripper_pos", np.array([]))])
        lerobot_vals = self._lerobot_from_i2rt(joint_pos)
        obs_dict = {f"{name}.pos": float(val) for name, val in zip(self._motor_names, lerobot_vals)}
        for cam_key, cam in self.cameras.items():
            obs_dict[cam_key] = cam.async_read()
        return obs_dict

    def send_action(self, action: RobotAction) -> RobotAction:
        # lerobot compatible method
        """Send LeRobot-style action dict."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        lerobot_array, i2rt_array = self._prepare_lerobot_command(action, log_clamp=True)
        try:
            self._motor_chain_robot.command_joint_pos(i2rt_array)
        except Exception as exc:
            self._enter_zero_torque_mode_safely(reason="send_action error", exc=exc)
            raise
        return {f"{motor}.pos": float(val) for motor, val in zip(self._motor_names, lerobot_array)}

    # i2rt-compatible helpers for direct scripts
    def get_observations(self) -> dict[str, np.ndarray]:
        """Return i2rt-style observation dict."""
        if self._motor_chain_robot is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        return self._motor_chain_robot.get_observations()

    def get_joint_pos(self) -> np.ndarray:
        """Return i2rt-style joint position vector (radians; gripper in [0,1])."""
        if self._motor_chain_robot is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        return self._motor_chain_robot.get_joint_pos()

    def command_joint_pos(self, joint_pos: np.ndarray) -> None:
        """Command i2rt-style joint position vector (radians; gripper in [0,1])."""
        if self._motor_chain_robot is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        try:
            self._motor_chain_robot.command_joint_pos(joint_pos)
        except Exception as exc:
            self._enter_zero_torque_mode_safely(reason="command_joint_pos error", exc=exc)
            raise

    def zero_torque_mode(self) -> None:
        """Enter i2rt zero-torque mode."""
        if self._motor_chain_robot is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        self._motor_chain_robot.zero_torque_mode()

    def close(self) -> None:
        """Close the i2rt motor chain robot."""
        if self._motor_chain_robot is None:
            return
        self._enter_zero_torque_mode_safely(reason="close", exc=None)
        self._motor_chain_robot.close()
        self._motor_chain_robot = None
        for cam in self.cameras.values():
            cam.disconnect()

    # =========================================================================
    # Disconnect
    # =========================================================================

    def disconnect(self) -> None:
        """
        Disconnect from robot.

        This method:
        1. Stops gravity compensation thread
        2. Disconnects motor bus (with zero-G mode)
        3. Disconnects cameras
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if self._motor_chain_robot is not None:
            self._enter_zero_torque_mode_safely(reason="disconnect", exc=None)
            self._motor_chain_robot.close()
            self._motor_chain_robot = None

        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def set_zero_gravity_mode(self) -> None:
        """
        Enter zero-gravity mode.

        In this mode, motors are enabled but don't resist movement.
        Gravity compensation (if enabled) still applies.
        """
        if self._motor_chain_robot is not None:
            self._motor_chain_robot.zero_torque_mode()
        logger.info(f"{self} entered zero-G mode")

    def set_position_control_mode(self) -> None:
        """
        Enter position control mode.

        Exits zero-G mode and applies configured KP/KD gains.
        """
        if self._motor_chain_robot is not None:
            self._motor_chain_robot.update_kp_kd(
                kp=np.array([self.config.kp_gains[m] for m in self._motor_names]),
                kd=np.array([self.config.kd_gains[m] for m in self._motor_names]),
            )
        logger.info(f"{self} entered position control mode")

    def _enter_zero_torque_mode_safely(self, reason: str, exc: Optional[Exception]) -> None:
        """Try to enter zero-torque mode and log failures."""
        if self._motor_chain_robot is None:
            return
        try:
            self._motor_chain_robot.zero_torque_mode()
        except Exception as zero_exc:
            logger.exception(
                "Failed to enter zero-torque mode (%s): %s",
                reason,
                zero_exc,
            )
        else:
            if exc is not None:
                logger.warning(
                    "Entered zero-torque mode after %s: %s",
                    reason,
                    exc,
                )
            logger.warning(
                "Zero-torque mode active. Move the arm to a safe rest position before exit."
            )

    def _lerobot_from_i2rt(self, i2rt_joint_pos: np.ndarray) -> np.ndarray:
        """Map i2rt joint positions to LeRobot normalized values."""
        lerobot_vals = []
        for name, pos in zip(self._motor_names, i2rt_joint_pos):
            if name == self._gripper_joint:
                lerobot_vals.append(float(max(0.0, min(1.0, pos))) * 100.0)
            else:
                lo, hi = self.config.joint_limits[name]
                if hi <= lo:
                    lerobot_vals.append(0.0)
                    continue
                t = (pos - lo) / (hi - lo)
                t = max(0.0, min(1.0, t))
                lerobot_vals.append(t * 200.0 - 100.0)
        return np.array(lerobot_vals, dtype=np.float32)

    def _i2rt_from_lerobot(self, lerobot_vals: np.ndarray) -> np.ndarray:
        """Map LeRobot normalized values to i2rt joint positions."""
        i2rt_vals = []
        for name, val in zip(self._motor_names, lerobot_vals):
            if name == self._gripper_joint:
                i2rt_vals.append(max(0.0, min(1.0, float(val) / 100.0)))
            else:
                lo, hi = self.config.joint_limits[name]
                t = (float(val) + 100.0) / 200.0
                t = max(0.0, min(1.0, t))
                i2rt_vals.append(lo + t * (hi - lo))
        return np.array(i2rt_vals, dtype=np.float32)

    def _prepare_lerobot_command(
        self, action: RobotAction, log_clamp: bool
    ) -> tuple[np.ndarray, np.ndarray]:
        """Prepare lerobot and i2rt commands with safety step limits."""
        goal_pos = {key.removesuffix(".pos"): float(val) for key, val in action.items() if key.endswith(".pos")}
        current_i2rt = self._motor_chain_robot.get_joint_pos()
        current_lerobot = self._lerobot_from_i2rt(current_i2rt)
        lerobot_vals = []
        for name, current_val in zip(self._motor_names, current_lerobot):
            raw = goal_pos.get(name, None)
            if raw is None or not np.isfinite(raw):
                lerobot_vals.append(float(current_val))
                continue
            if name == self._gripper_joint:
                clamped = max(0.0, min(100.0, float(raw)))
            else:
                clamped = max(-100.0, min(100.0, float(raw)))
            lerobot_vals.append(clamped)
        lerobot_array = np.array(lerobot_vals, dtype=np.float32)
        delta = lerobot_array - current_lerobot
        max_steps_array = np.full(
            len(self._motor_names),
            self.config.lerobot_max_step,
            dtype=np.float32,
        )
        if self._gripper_joint in self._motor_names:
            max_steps_array[self._motor_names.index(self._gripper_joint)] = (
                self.config.lerobot_gripper_max_step
            )
        clamped_delta = np.clip(delta, -max_steps_array, max_steps_array)
        if log_clamp and np.any(np.not_equal(delta, clamped_delta)):
            logger.warning("LeRobot action step limited for safety.")
        lerobot_array = current_lerobot + clamped_delta
        i2rt_array = self._i2rt_from_lerobot(lerobot_array)
        return lerobot_array, i2rt_array

    def _build_motor_names(self) -> list[str]:
        """Return ordered motor names based on gripper type."""
        gripper_type = GripperType.from_string(self.config.gripper_type)
        with_gripper = gripper_type not in (GripperType.YAM_TEACHING_HANDLE, GripperType.NO_GRIPPER)
        motor_names = list(self._arm_joints)
        if with_gripper:
            motor_names.append(self._gripper_joint)
        return motor_names

    def _build_motor_chain_robot(self) -> MotorChainRobot:
        """Construct i2rt-authentic motor chain robot."""
        gripper_type = GripperType.from_string(self.config.gripper_type)
        with_gripper = gripper_type not in (GripperType.YAM_TEACHING_HANDLE, GripperType.NO_GRIPPER)
        with_teaching_handle = gripper_type == GripperType.YAM_TEACHING_HANDLE
        motor_list = [
            (1, "DM4340"),
            (2, "DM4340"),
            (3, "DM4340"),
            (4, "DM4310"),
            (5, "DM4310"),
            (6, "DM4310"),
        ]
        motor_offsets = [self.config.motor_offsets[name] for name in self._arm_joints]
        motor_directions = [self.config.motor_directions[name] for name in self._arm_joints]

        if with_gripper:
            motor_list.append((7, gripper_type.get_motor_type()))
            motor_offsets.append(self.config.motor_offsets[self._gripper_joint])
            motor_directions.append(self.config.motor_directions[self._gripper_joint])

        joint_limits = np.array([self.config.joint_limits[name] for name in self._arm_joints])

        # First pass: read current positions and adjust offsets by +/- 2*pi
        motor_chain = DMChainCanInterface(
            motor_list,
            motor_offsets,
            motor_directions,
            self.config.port,
            motor_chain_name="yam_real",
            receive_mode=ReceiveMode.p16,
            start_thread=False,
        )
        motor_states = motor_chain.read_states()
        motor_chain.close()

        for idx, motor_state in enumerate(motor_states):
            motor_position = motor_state.pos
            if motor_position < -np.pi:
                extra_offset = -2 * np.pi
            elif motor_position > np.pi:
                extra_offset = +2 * np.pi
            else:
                extra_offset = 0.0
            motor_offsets[idx] += extra_offset

        time.sleep(0.5)

        def get_encoder_chain(can_interface: CanInterface) -> EncoderChain:
            passive_encoder_reader = PassiveEncoderReader(can_interface)
            return EncoderChain([0x50E], passive_encoder_reader)

        motor_chain = DMChainCanInterface(
            motor_list,
            motor_offsets,
            motor_directions,
            self.config.port,
            motor_chain_name="yam_real",
            receive_mode=ReceiveMode.p16,
            get_same_bus_device_driver=get_encoder_chain if with_teaching_handle else None,
            use_buffered_reader=False,
        )

        if self.config.mujoco_xml_path:
            xml_path = self.config.mujoco_xml_path
        else:
            xml_path = gripper_type.get_xml_path()

        kp_names = self._motor_names
        kp = np.array([self.config.kp_gains[name] for name in kp_names])
        kd = np.array([self.config.kd_gains[name] for name in kp_names])
        if with_gripper:
            gripper_kp, gripper_kd = gripper_type.get_motor_kp_kd()
            kp[-1] = gripper_kp
            kd[-1] = gripper_kd

        return MotorChainRobot(
            motor_chain=motor_chain,
            xml_path=xml_path,
            use_gravity_comp=self.config.use_gravity_compensation,
            gravity_comp_factor=self.config.gravity_comp_factor,
            joint_limits=joint_limits,
            kp=kp,
            kd=kd,
            zero_gravity_mode=self.config.zero_gravity_mode,
            gripper_index=6 if with_gripper else None,
            gripper_limits=gripper_type.get_gripper_limits() if with_gripper else None,
            enable_gripper_calibration=gripper_type.get_gripper_needs_calibration() if with_gripper else False,
            gripper_type=gripper_type,
            limit_gripper_force=self.config.limit_gripper_force,
        )


class YAMFollowerRobot(YAMFollower):
    """Alias class for Robot config auto-discovery."""

    pass
