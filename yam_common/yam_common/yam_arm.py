"""Lightweight YAM hardware controller that does not import LeRobot or torch."""

from __future__ import annotations

import logging
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np

from yam_common.dm.dm_driver import (
    CanInterface,
    DMChainCanInterface,
    EncoderChain,
    PassiveEncoderReader,
    ReceiveMode,
    probe_motors,
)
from yam_common.motor_chain_robot import MotorChainRobot
from yam_common.utils import GripperType

logger = logging.getLogger(__name__)

ARM_JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "wrist_yaw",
)
GRIPPER_JOINT = "gripper"
ACTION_KEYS = tuple(f"{name}.pos" for name in (*ARM_JOINT_NAMES, GRIPPER_JOINT))

DEFAULT_KP_GAINS = {
    "shoulder_pan": 80.0,
    "shoulder_lift": 80.0,
    "elbow_flex": 80.0,
    "wrist_flex": 40.0,
    "wrist_roll": 10.0,
    "wrist_yaw": 10.0,
    "gripper": 20.0,
}
DEFAULT_KD_GAINS = {
    "shoulder_pan": 5.0,
    "shoulder_lift": 5.0,
    "elbow_flex": 5.0,
    "wrist_flex": 1.5,
    "wrist_roll": 1.5,
    "wrist_yaw": 1.5,
    "gripper": 0.5,
}
DEFAULT_JOINT_LIMITS = {
    "shoulder_pan": (-2.767, 3.28),
    "shoulder_lift": (-0.15, 3.8),
    "elbow_flex": (-0.15, 3.28),
    "wrist_flex": (-1.72, 1.72),
    "wrist_roll": (-1.72, 1.72),
    "wrist_yaw": (-2.24, 2.24),
}
DEFAULT_GRIPPER_LIMITS = (0.0, -2.7)
DEFAULT_MOTOR_OFFSETS = {
    "shoulder_pan": 0.0,
    "shoulder_lift": 0.0,
    "elbow_flex": 0.0,
    "wrist_flex": 0.0,
    "wrist_roll": 0.0,
    "wrist_yaw": 0.0,
    "gripper": 0.0,
}
DEFAULT_MOTOR_DIRECTIONS = {
    "shoulder_pan": 1,
    "shoulder_lift": 1,
    "elbow_flex": 1,
    "wrist_flex": 1,
    "wrist_roll": 1,
    "wrist_yaw": 1,
    "gripper": 1,
}


class YAMArmError(RuntimeError):
    """Base error for the hardware-only YAM controller."""


class YAMArmNotConnectedError(YAMArmError):
    """Public API called before connect or after disconnect."""


class YAMArmAlreadyConnectedError(YAMArmError):
    """connect() called while the arm is already connected."""


class YAMArmUnhealthyError(YAMArmError):
    """Background DM or gravity-comp control loop has died."""


@dataclass
class YAMArmConfig:
    """Plain hardware config for a single YAM follower arm."""

    port: str = "can0"
    bitrate: int = 1_000_000
    bustype: str = "socketcan"
    motor_offsets: dict[str, float] = field(default_factory=lambda: DEFAULT_MOTOR_OFFSETS.copy())
    motor_directions: dict[str, int] = field(default_factory=lambda: DEFAULT_MOTOR_DIRECTIONS.copy())
    kp_gains: dict[str, float] = field(default_factory=lambda: DEFAULT_KP_GAINS.copy())
    kd_gains: dict[str, float] = field(default_factory=lambda: DEFAULT_KD_GAINS.copy())
    joint_limits: dict[str, tuple[float, float]] = field(default_factory=lambda: DEFAULT_JOINT_LIMITS.copy())
    gripper_limits: tuple[float, float] = DEFAULT_GRIPPER_LIMITS
    use_gravity_compensation: bool = True
    gravity_comp_factor: float = 1.3
    mujoco_xml_path: Optional[str] = None
    gripper_type: str = "crank_4310"
    zero_gravity_mode: bool = True
    shutdown_zero_gravity_wait_for_enter: bool = False
    limit_gripper_force: float = 50.0
    lerobot_max_step: float = 5.0
    lerobot_gripper_max_step: float = 5.0
    rest_pose: Optional[tuple[float, ...]] = None

    @property
    def arm_joint_names(self) -> tuple[str, ...]:
        return ARM_JOINT_NAMES

    def __post_init__(self) -> None:
        names = motor_names_for_config(self)
        for label, mapping in (
            ("motor_offsets", self.motor_offsets),
            ("motor_directions", self.motor_directions),
            ("kp_gains", self.kp_gains),
            ("kd_gains", self.kd_gains),
        ):
            missing = [name for name in names if name not in mapping]
            if missing:
                raise ValueError(f"{label} missing keys: {missing}")
            for name in names:
                if not np.isfinite(float(mapping[name])):
                    raise ValueError(f"{label}[{name!r}] must be finite")
        for name in names:
            if int(self.motor_directions[name]) not in (-1, 1):
                raise ValueError(f"motor_directions[{name!r}] must be 1 or -1")
        for name in ARM_JOINT_NAMES:
            if name not in self.joint_limits:
                raise ValueError(f"joint_limits missing keys: ['{name}']")
            lo, hi = self.joint_limits[name]
            if not (np.isfinite(float(lo)) and np.isfinite(float(hi))):
                raise ValueError(f"joint_limits[{name!r}] must be finite")
            if not float(lo) < float(hi):
                raise ValueError(f"joint_limits[{name!r}] must be ordered lo < hi")
        if GRIPPER_JOINT in names:
            open_pos, closed_pos = self.gripper_limits
            if not (np.isfinite(float(open_pos)) and np.isfinite(float(closed_pos))):
                raise ValueError("gripper_limits must be finite")
        if self.rest_pose is not None:
            if len(self.rest_pose) != len(names):
                raise ValueError(f"rest_pose must have {len(names)} values")
            if not all(np.isfinite(float(value)) for value in self.rest_pose):
                raise ValueError("rest_pose values must be finite")


def motor_names_for_config(config: YAMArmConfig) -> list[str]:
    gripper_type = GripperType.from_string(config.gripper_type)
    names = list(ARM_JOINT_NAMES)
    if gripper_type not in (GripperType.YAM_TEACHING_HANDLE, GripperType.NO_GRIPPER):
        names.append(GRIPPER_JOINT)
    return names


def normalize_from_physical(
    i2rt_joint_pos: np.ndarray,
    config: YAMArmConfig,
) -> dict[str, float]:
    """Map i2rt joint positions to LeRobot-normalized `{joint}.pos` values."""
    values: dict[str, float] = {}
    for name, pos in zip(motor_names_for_config(config), i2rt_joint_pos):
        if name == GRIPPER_JOINT:
            values[f"{name}.pos"] = float(max(0.0, min(1.0, pos))) * 100.0
            continue
        lo, hi = config.joint_limits[name]
        if hi <= lo:
            values[f"{name}.pos"] = 0.0
            continue
        t = max(0.0, min(1.0, (float(pos) - lo) / (hi - lo)))
        values[f"{name}.pos"] = t * 200.0 - 100.0
    return values


def physical_from_normalized(
    normalized: Mapping[str, float],
    config: YAMArmConfig,
) -> np.ndarray:
    """Map LeRobot-normalized `{joint}.pos` values to i2rt joint positions."""
    physical = []
    for name in motor_names_for_config(config):
        val = float(normalized[f"{name}.pos"])
        if name == GRIPPER_JOINT:
            physical.append(max(0.0, min(1.0, val / 100.0)))
            continue
        lo, hi = config.joint_limits[name]
        t = max(0.0, min(1.0, (val + 100.0) / 200.0))
        physical.append(lo + t * (hi - lo))
    return np.array(physical, dtype=np.float64)


def prepare_normalized_action(
    action: Mapping[str, float],
    current_normalized: Mapping[str, float],
    config: YAMArmConfig,
    *,
    log_clamp: bool = False,
) -> dict[str, float]:
    """Clamp range and per-cycle step, returning the action that will be performed."""
    names = motor_names_for_config(config)
    prepared: dict[str, float] = {}
    for name in names:
        key = f"{name}.pos"
        raw = action.get(key, None)
        current = float(current_normalized[key])
        if raw is None or not np.isfinite(raw):
            prepared[key] = current
            continue
        if name == GRIPPER_JOINT:
            prepared[key] = max(0.0, min(100.0, float(raw)))
        else:
            prepared[key] = max(-100.0, min(100.0, float(raw)))

    current_array = np.array([current_normalized[f"{name}.pos"] for name in names], dtype=np.float64)
    prepared_array = np.array([prepared[f"{name}.pos"] for name in names], dtype=np.float64)
    max_steps = np.full(len(names), config.lerobot_max_step, dtype=np.float64)
    if GRIPPER_JOINT in names:
        max_steps[names.index(GRIPPER_JOINT)] = config.lerobot_gripper_max_step
    delta = prepared_array - current_array
    clamped_delta = np.clip(delta, -max_steps, max_steps)
    if log_clamp and np.any(np.not_equal(delta, clamped_delta)):
        logger.warning("LeRobot action step limited for safety.")
    limited = current_array + clamped_delta
    return {f"{name}.pos": float(val) for name, val in zip(names, limited)}


class YAMArm:
    """Owns MotorChainRobot / DM CAN hardware for one YAM follower arm."""

    def __init__(
        self,
        config: Optional[YAMArmConfig] = None,
        *,
        robot_factory: Optional[Callable[..., Any]] = None,
        chain_factory: Optional[Callable[..., Any]] = None,
    ) -> None:
        self.config = config or YAMArmConfig()
        self._robot_factory = robot_factory
        self._chain_factory = chain_factory
        self._robot: Any = None
        self._cached_obs: Optional[dict[str, np.ndarray]] = None
        self._cached_error: Optional[BaseException] = None

    @property
    def action_keys(self) -> tuple[str, ...]:
        return tuple(f"{name}.pos" for name in motor_names_for_config(self.config))

    @property
    def observation_keys(self) -> tuple[str, ...]:
        return self.action_keys

    @property
    def is_connected(self) -> bool:
        return self._robot is not None

    def connect(self) -> None:
        if self.is_connected:
            raise YAMArmAlreadyConnectedError(f"{self} already connected")
        self._cached_error = None
        try:
            if self._robot_factory is not None:
                self._robot = self._robot_factory(config=self.config)
            else:
                self._validate_ready_to_open_can()
                self._robot = self._build_motor_chain_robot()
            self._refresh_cache()
            self.health_check()
        except Exception:
            if self._robot is not None:
                self.emergency_cleanup()
            raise

    def disconnect(self) -> None:
        if self._robot is None:
            raise YAMArmNotConnectedError(f"{self} is not connected.")
        self._shutdown_robot(wait_for_enter=True)

    def close(self) -> None:
        if self._robot is None:
            return
        self._shutdown_robot(wait_for_enter=True)

    def emergency_cleanup(self) -> None:
        """Zero-torque and close CAN without interactive shutdown waiting."""
        if self._robot is None:
            return
        self._shutdown_robot(wait_for_enter=False)

    def get_observation(self) -> dict[str, float]:
        self._ensure_healthy()
        return normalize_from_physical(self.get_joint_pos(), self.config)

    def send_action(self, action: Mapping[str, float]) -> dict[str, float]:
        self._ensure_healthy()
        current = normalize_from_physical(self.get_joint_pos(), self.config)
        performed = prepare_normalized_action(action, current, self.config, log_clamp=True)
        try:
            self._robot.command_joint_pos(physical_from_normalized(performed, self.config))
        except Exception as exc:
            self._enter_zero_torque_mode_safely(reason="send_action error", exc=exc)
            raise
        self._refresh_cache()
        return performed

    def hold_current_pose(self) -> None:
        self.command_joint_pos(self.get_joint_pos())

    def command_joint_pos(self, joint_pos: np.ndarray | Sequence[float]) -> None:
        self._ensure_healthy()
        try:
            self._robot.command_joint_pos(np.asarray(joint_pos, dtype=np.float64))
        except Exception as exc:
            self._enter_zero_torque_mode_safely(reason="command_joint_pos error", exc=exc)
            raise
        self._refresh_cache()

    def command_rest(self, rest_pose: Optional[Sequence[float]] = None) -> None:
        pose = rest_pose if rest_pose is not None else self.config.rest_pose
        if pose is None:
            raise ValueError(f"{self}: rest_pose is not configured")
        self.command_joint_pos(pose)

    def zero_torque(self) -> None:
        self._ensure_healthy()
        self._robot.zero_torque_mode()

    def get_joint_pos(self) -> np.ndarray:
        self._ensure_healthy()
        return self._robot.get_joint_pos()

    def get_observations(self) -> dict[str, np.ndarray]:
        self._ensure_healthy()
        obs = self._robot.get_observations()
        self._cached_obs = obs
        return obs

    def get_telemetry(self) -> dict[str, Any]:
        error = self._current_error()
        obs = self._cached_obs
        if obs is None and self._robot is not None and error is None:
            try:
                obs = self._robot.get_observations()
                self._cached_obs = obs
            except Exception as exc:
                error = exc
                self._cached_error = exc
        if obs is None:
            obs = {}
        return {
            "joint_pos": obs.get("joint_pos"),
            "gripper_pos": obs.get("gripper_pos"),
            "joint_vel": obs.get("joint_vel"),
            "joint_eff": obs.get("joint_eff"),
            "connected": self.is_connected,
            "healthy": error is None and self.is_connected,
            "error": error,
        }

    def health_check(self) -> None:
        self._ensure_healthy()

    def update_kp_kd(self, kp: np.ndarray, kd: np.ndarray) -> None:
        self._ensure_healthy()
        self._robot.update_kp_kd(kp, kd)

    def _current_error(self) -> Optional[BaseException]:
        if self._robot is None:
            return self._cached_error
        robot_error = getattr(self._robot, "control_loop_error", None)
        if robot_error is not None:
            self._cached_error = robot_error
            return robot_error
        return self._cached_error

    def _ensure_connected(self) -> None:
        if self._robot is None:
            raise YAMArmNotConnectedError(f"{self} is not connected.")

    def _ensure_healthy(self) -> None:
        self._ensure_connected()
        error = self._current_error()
        if error is not None:
            raise YAMArmUnhealthyError("control loop is not running") from error
        raise_if_unhealthy = getattr(self._robot, "raise_if_unhealthy", None)
        if raise_if_unhealthy is not None:
            try:
                raise_if_unhealthy()
            except Exception as exc:
                self._cached_error = exc
                raise YAMArmUnhealthyError("control loop is not running") from exc

    def _refresh_cache(self) -> None:
        if self._robot is None:
            return
        try:
            self._cached_obs = self._robot.get_observations()
        except Exception:
            pass

    def _shutdown_robot(self, *, wait_for_enter: bool = True) -> None:
        robot = self._robot
        try:
            self._enter_zero_torque_mode_safely(reason="disconnect", exc=None)
            if wait_for_enter:
                self._hold_zero_gravity_before_shutdown()
            if robot is not None:
                robot.close()
        finally:
            self._robot = None

    def _enter_zero_torque_mode_safely(self, reason: str, exc: Optional[Exception]) -> None:
        if self._robot is None:
            return
        try:
            self._robot.zero_torque_mode()
        except Exception as zero_exc:
            logger.exception("Failed to enter zero-torque mode (%s): %s", reason, zero_exc)
        else:
            if exc is not None:
                logger.warning("Entered zero-torque mode after %s: %s", reason, exc)
            logger.warning("Zero-torque mode active. Move the arm to a safe rest position before exit.")

    def _hold_zero_gravity_before_shutdown(self) -> None:
        if self._robot is None:
            return
        if not bool(self.config.shutdown_zero_gravity_wait_for_enter):
            return
        if sys.stdin is not None and sys.stdin.isatty():
            logger.warning(
                "Zero-G active. Move to a safe rest position, then press ENTER to finish shutdown."
            )
            try:
                input()
            except Exception:
                logger.exception("Failed while waiting for ENTER; continuing shutdown.")
        else:
            logger.error("Zero-G active. No TTY detected; holding indefinitely before shutdown.")
            while True:
                time.sleep(1.0)

    def _validate_ready_to_open_can(self) -> None:
        if self.config.mujoco_xml_path:
            xml_path = os.path.expanduser(self.config.mujoco_xml_path)
            if not os.path.isfile(xml_path):
                raise FileNotFoundError(f"MuJoCo XML not found: {xml_path}")
        elif self.config.use_gravity_compensation:
            GripperType.from_string(self.config.gripper_type).get_xml_path()

    def _abandon_chain(self, chain: Any) -> None:
        try:
            zeros = np.zeros(len(chain))
            chain.set_commands(zeros, pos=zeros, vel=zeros, kp=zeros, kd=zeros, get_state=False)
        except Exception:
            logger.exception("Failed to zero raw chain during connect rollback")
        try:
            chain.close(disable_motors=True)
        except Exception:
            logger.exception("Failed to close raw chain during connect rollback")

    def _build_chain(self, **kwargs: Any) -> Any:
        factory = self._chain_factory or DMChainCanInterface
        kwargs.setdefault("bustype", self.config.bustype)
        kwargs.setdefault("bitrate", self.config.bitrate)
        kwargs.setdefault("channel", self.config.port)
        return factory(**kwargs)

    def _build_motor_chain_robot(self) -> MotorChainRobot:
        gripper_type = GripperType.from_string(self.config.gripper_type)
        with_gripper = gripper_type not in (GripperType.YAM_TEACHING_HANDLE, GripperType.NO_GRIPPER)
        with_teaching_handle = gripper_type == GripperType.YAM_TEACHING_HANDLE
        names = motor_names_for_config(self.config)
        motor_list = [
            (1, "DM4340"),
            (2, "DM4340"),
            (3, "DM4340"),
            (4, "DM4310"),
            (5, "DM4310"),
            (6, "DM4310"),
        ]
        motor_offsets = [self.config.motor_offsets[name] for name in ARM_JOINT_NAMES]
        motor_directions = [self.config.motor_directions[name] for name in ARM_JOINT_NAMES]
        if with_gripper:
            motor_list.append((7, gripper_type.get_motor_type()))
            motor_offsets.append(self.config.motor_offsets[GRIPPER_JOINT])
            motor_directions.append(self.config.motor_directions[GRIPPER_JOINT])

        joint_limits = np.array([self.config.joint_limits[name] for name in ARM_JOINT_NAMES])
        active_chain: Any = None
        try:
            motor_chain = self._build_chain(
                motor_list=motor_list,
                motor_offset=motor_offsets,
                motor_direction=motor_directions,
                channel=self.config.port,
                bitrate=self.config.bitrate,
                bustype=self.config.bustype,
                motor_chain_name="yam_real",
                receive_mode=ReceiveMode.p16,
                start_thread=False,
            )
            active_chain = motor_chain
            motor_states = motor_chain.read_states()
            motor_chain.close(disable_motors=False)
            active_chain = None

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
                return EncoderChain([0x50E], PassiveEncoderReader(can_interface))

            motor_chain = self._build_chain(
                motor_list=motor_list,
                motor_offset=motor_offsets,
                motor_direction=motor_directions,
                channel=self.config.port,
                bitrate=self.config.bitrate,
                bustype=self.config.bustype,
                motor_chain_name="yam_real",
                receive_mode=ReceiveMode.p16,
                get_same_bus_device_driver=get_encoder_chain if with_teaching_handle else None,
                use_buffered_reader=False,
            )
            active_chain = motor_chain

            if self.config.mujoco_xml_path:
                xml_path = self.config.mujoco_xml_path
            elif self.config.use_gravity_compensation:
                xml_path = gripper_type.get_xml_path()
            else:
                xml_path = None

            kp = np.array([self.config.kp_gains[name] for name in names])
            kd = np.array([self.config.kd_gains[name] for name in names])
            robot = MotorChainRobot(
                motor_chain=motor_chain,
                xml_path=xml_path,
                use_gravity_comp=self.config.use_gravity_compensation,
                gravity_comp_factor=self.config.gravity_comp_factor,
                joint_limits=joint_limits,
                kp=kp,
                kd=kd,
                zero_gravity_mode=self.config.zero_gravity_mode,
                gripper_index=6 if with_gripper else None,
                gripper_limits=self.config.gripper_limits if with_gripper else None,
                enable_gripper_calibration=gripper_type.get_gripper_needs_calibration() if with_gripper else False,
                gripper_type=gripper_type,
                limit_gripper_force=self.config.limit_gripper_force,
            )
            active_chain = None
            return robot
        except Exception:
            if active_chain is not None:
                self._abandon_chain(active_chain)
            raise

    def __repr__(self) -> str:
        return f"YAMArm(port={self.config.port!r})"


__all__ = [
    "ACTION_KEYS",
    "ARM_JOINT_NAMES",
    "GRIPPER_JOINT",
    "YAMArm",
    "YAMArmAlreadyConnectedError",
    "YAMArmConfig",
    "YAMArmError",
    "YAMArmNotConnectedError",
    "YAMArmUnhealthyError",
    "normalize_from_physical",
    "physical_from_normalized",
    "prepare_normalized_action",
    "probe_motors",
]
