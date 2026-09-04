"""YAM follower plugin that delegates hardware control to yam_common.YAMArm."""

import logging
from functools import cached_property

import numpy as np

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.processor import RobotAction, RobotObservation
from lerobot.robots.robot import Robot
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from yam_common import YAMArm, YAMArmAlreadyConnectedError, YAMArmConfig

from .config_yam_follower import YAMFollowerRobotConfig

logger = logging.getLogger(__name__)


def yam_arm_config_from_follower(config: YAMFollowerRobotConfig) -> YAMArmConfig:
    """Copy follower plugin fields onto the hardware-only YAMArmConfig."""
    return YAMArmConfig(
        port=config.port,
        bitrate=config.bitrate,
        bustype=config.bustype,
        motor_offsets=dict(config.motor_offsets),
        motor_directions=dict(config.motor_directions),
        kp_gains=dict(config.kp_gains),
        kd_gains=dict(config.kd_gains),
        joint_limits=dict(config.joint_limits),
        gripper_limits=tuple(config.gripper_limits),
        use_gravity_compensation=config.use_gravity_compensation,
        gravity_comp_factor=config.gravity_comp_factor,
        mujoco_xml_path=config.mujoco_xml_path,
        gripper_type=config.gripper_type,
        zero_gravity_mode=config.zero_gravity_mode,
        shutdown_zero_gravity_wait_for_enter=config.shutdown_zero_gravity_wait_for_enter,
        limit_gripper_force=config.limit_gripper_force,
        lerobot_max_step=config.lerobot_max_step,
        lerobot_gripper_max_step=config.lerobot_gripper_max_step,
    )


class YAMFollower(Robot):
    """
    Robot implementation for YAM arm with DM motors.

    Hardware construction, mapping, and control-loop ownership live on YAMArm.
    This class keeps the LeRobot robot/camera plugin surface.
    """

    config_class = YAMFollowerRobotConfig
    name = "yam_follower"

    def __init__(self, config: YAMFollowerRobotConfig):
        super().__init__(config)
        self.config = config
        self._arm = YAMArm(yam_arm_config_from_follower(config))
        self._motor_names = [key.removesuffix(".pos") for key in self._arm.action_keys]
        self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self._motor_names}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3)
            for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return self._arm.is_connected and all(cam.is_connected for cam in self.cameras.values())

    @property
    def is_calibrated(self) -> bool:
        return True

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        try:
            self._arm.connect()
        except YAMArmAlreadyConnectedError as exc:
            raise DeviceAlreadyConnectedError(f"{self} already connected") from exc

        try:
            for cam in self.cameras.values():
                cam.connect()
        except Exception as exc:
            for cam in self.cameras.values():
                try:
                    cam.disconnect()
                except Exception:
                    pass
            if self._arm.is_connected:
                try:
                    self._arm.emergency_cleanup()
                except Exception:
                    logger.exception("Failed to disconnect YAM arm after camera connect failure")
            raise

        logger.info(f"{self} connected.")

    def calibrate(self) -> None:
        return

    def configure(self) -> None:
        pass

    def get_observation(self) -> RobotObservation:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        obs_dict = dict(self._arm.get_observation())
        for cam_key, cam in self.cameras.items():
            obs_dict[cam_key] = cam.async_read()
        return obs_dict

    def send_action(self, action: RobotAction) -> RobotAction:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        return self._arm.send_action(action)

    def get_observations(self) -> dict[str, np.ndarray]:
        if not self._arm.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        return self._arm.get_observations()

    def get_joint_pos(self) -> np.ndarray:
        if not self._arm.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        return self._arm.get_joint_pos()

    def command_joint_pos(self, joint_pos: np.ndarray) -> None:
        if not self._arm.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        self._arm.command_joint_pos(joint_pos)

    def zero_torque_mode(self) -> None:
        if not self._arm.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        self._arm.zero_torque()

    def close(self) -> None:
        if self._arm.is_connected:
            self._arm.close()
        for cam in self.cameras.values():
            cam.disconnect()

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if self._arm.is_connected:
            self._arm.disconnect()

        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")

    def set_zero_gravity_mode(self) -> None:
        if self._arm.is_connected:
            self._arm.zero_torque()
        logger.info(f"{self} entered zero-G mode")

    def set_position_control_mode(self) -> None:
        if self._arm.is_connected:
            self._arm.update_kp_kd(
                kp=np.array([self.config.kp_gains[m] for m in self._motor_names]),
                kd=np.array([self.config.kd_gains[m] for m in self._motor_names]),
            )
        logger.info(f"{self} entered position control mode")


class YAMFollowerRobot(YAMFollower):
    """Alias class for Robot config auto-discovery."""

    pass
