"""
Utility classes for YAM Follower robot.

This module contains:
- GripperForceLimiter: Protects gripper from excessive force (from i2rt/robots/utils.py)
- EMAFilter: Exponential moving average filter for smoothing (from i2rt/scripts/zmq_follower.py)
- JointMapper: Maps between normalized and physical joint spaces (from i2rt/robots/utils.py)

Source files:
- i2rt/robots/utils.py: GripperForceLimiter, JointMapper, GripperType
- i2rt/scripts/zmq_follower.py: EMA smoothing logic
"""

import logging
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from functools import partial
from typing import Callable, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# GRIPPER TYPE CONFIGURATION
# Source: i2rt/robots/utils.py:GripperType
# =============================================================================

class GripperType(Enum):
    """
    Gripper type enumeration with associated parameters.

    Source: i2rt/robots/utils.py:GripperType
    """
    CRANK_4310 = "crank_4310"       # 4310 motor with crank mechanism
    LINEAR_3507 = "linear_3507"     # 3507 motor with linear actuator
    LINEAR_4310 = "linear_4310"     # 4310 motor with linear actuator
    YAM_TEACHING_HANDLE = "yam_teaching_handle"  # Passive teaching handle
    NO_GRIPPER = "no_gripper"       # No gripper attached

    @classmethod
    def from_string(cls, name: str) -> "GripperType":
        """Convert string name to GripperType."""
        name_map = {
            "crank_4310": cls.CRANK_4310,
            "linear_3507": cls.LINEAR_3507,
            "linear_4310": cls.LINEAR_4310,
            "yam_teaching_handle": cls.YAM_TEACHING_HANDLE,
            "no_gripper": cls.NO_GRIPPER,
        }
        if name not in name_map:
            raise ValueError(f"Unknown gripper type: {name}. Valid types: {list(name_map.keys())}")
        return name_map[name]

    def get_gripper_limits(self) -> Optional[Tuple[float, float]]:
        """
        Get gripper limits (open, closed) in radians.

        Source: i2rt/robots/utils.py:GripperType.get_gripper_limits()

        Returns:
            (open_pos, closed_pos) or None if not applicable
        """
        if self == GripperType.CRANK_4310:
            return (0.0, -2.7)  # open, closed
        elif self in [GripperType.LINEAR_3507, GripperType.LINEAR_4310]:
            return None  # Requires calibration
        elif self in [GripperType.YAM_TEACHING_HANDLE, GripperType.NO_GRIPPER]:
            return None
        return None

    def get_gripper_needs_calibration(self) -> bool:
        return self.needs_calibration()

    def needs_calibration(self) -> bool:
        """
        Check if gripper needs runtime calibration.

        Source: i2rt/robots/utils.py:GripperType.get_gripper_needs_calibration()
        """
        return self in [GripperType.LINEAR_3507, GripperType.LINEAR_4310]

    def get_motor_type(self) -> str:
        """
        Get motor model string for this gripper.

        Source: i2rt/robots/utils.py:GripperType.get_motor_type()
        """
        if self in [GripperType.CRANK_4310, GripperType.LINEAR_4310]:
            return "DM4310"
        elif self == GripperType.LINEAR_3507:
            return "DM3507"
        elif self == GripperType.YAM_TEACHING_HANDLE:
            return ""  # No motor
        else:
            raise ValueError(f"Unknown motor type for gripper: {self}")

    def get_xml_path(self) -> str:
        from .mujoco_kdl import get_yam_mujoco_kdl

        return get_yam_mujoco_kdl(self.value).xml_path

    def get_kp_kd(self) -> Tuple[float, float]:
        """
        Get KP/KD gains for gripper motor.

        Source: i2rt/robots/utils.py:GripperType.get_motor_kp_kd()
        """
        if self in [GripperType.CRANK_4310, GripperType.LINEAR_4310]:
            return (20.0, 0.5)
        elif self == GripperType.LINEAR_3507:
            return (10.0, 0.3)
        elif self == GripperType.YAM_TEACHING_HANDLE:
            return (-1.0, -1.0)  # No control
        else:
            raise ValueError(f"Unknown KP/KD for gripper: {self}")

    def get_motor_kp_kd(self) -> Tuple[float, float]:
        """i2rt-compatible alias."""
        return self.get_kp_kd()


# =============================================================================
# GRIPPER STATE
# =============================================================================

@dataclass
class GripperState:
    """Current state of the gripper for force limiting."""
    target_pos: float           # Target position (radians)
    current_pos: float          # Current position (radians)
    current_vel: float          # Current velocity (rad/s)
    current_effort: float       # Current effort/torque (Nm)
    current_normalized_pos: float  # Current position in [0, 1]
    target_normalized_pos: float   # Target position in [0, 1]
    last_command_pos: float     # Last commanded position (radians)


# =============================================================================
# GRIPPER FORCE LIMITER
# Source: i2rt/robots/utils.py:GripperForceLimiter
# =============================================================================

def linear_gripper_force_torque_map(
    motor_stroke: float,
    gripper_stroke: float,
    gripper_force: float,
    current_angle: float,
) -> float:
    """
    Force-to-torque mapping for linear gripper.

    Source: i2rt/robots/utils.py:linear_gripper_force_torque_map()

    Args:
        motor_stroke: Motor stroke in radians
        gripper_stroke: Gripper stroke in meters
        gripper_force: Required gripper force in Newtons
        current_angle: Current motor angle (unused for linear)

    Returns:
        Required motor torque in Nm
    """
    # force = torque * motor_stroke / gripper_stroke
    return gripper_force * gripper_stroke / motor_stroke


def crank_gripper_force_torque_map(
    gripper_close_angle: float,
    gripper_open_angle: float,
    gripper_stroke: float,
    motor_reading_to_crank_angle: Callable[[float], float],
    current_angle: float,
    gripper_force: float,
) -> float:
    """
    Force-to-torque mapping for zero-linkage crank gripper (YAM style).

    Source: i2rt/robots/utils.py:zero_linkage_crank_gripper_force_torque_map()

    This computes the motor torque needed to achieve a given gripper force
    at the current crank angle, accounting for the mechanical advantage
    of the crank mechanism.

    Args:
        gripper_close_angle: Crank angle at closed position (radians)
        gripper_open_angle: Crank angle at open position (radians)
        gripper_stroke: Linear gripper displacement (meters)
        motor_reading_to_crank_angle: Function to convert motor reading to crank angle
        current_angle: Current motor angle reading (radians)
        gripper_force: Required gripper force (Newtons)

    Returns:
        Required motor torque (Nm)
    """
    # Convert motor reading to crank angle
    crank_angle = motor_reading_to_crank_angle(current_angle)

    # Compute crank radius from geometry
    crank_radius = gripper_stroke / (2 * (np.cos(gripper_close_angle) - np.cos(gripper_open_angle)))

    # Gradient of gripper position w.r.t. crank angle (mechanical advantage)
    grad_gripper_position = crank_radius * np.sin(crank_angle)

    # Required torque via work-energy equivalence
    target_torque = gripper_force * grad_gripper_position
    return target_torque


class GripperForceLimiter:
    """
    Gripper force limiter that detects clogging and limits force.

    Replicates i2rt/robots/utils.py:GripperForceLimiter exactly.

    When the gripper is pushing against an object (high effort, low velocity),
    this class detects the "clog" condition and adjusts the target position
    to limit the applied force.

    The detection logic:
    - Track average effort over a sliding time window
    - If average effort > threshold AND velocity < threshold -> clogged
    - When clogged, compute target position to achieve max_force
    - Apply EMA smoothing to adjusted position

    The unclogging logic:
    - If user wants to open (target > current) OR effort drops -> unclogged
    """

    def __init__(
        self,
        max_force: float,
        kp: float,
        gripper_type: GripperType = GripperType.CRANK_4310,
        clog_force_threshold: float = 0.5,
        clog_speed_threshold: float = 0.2,
        average_torque_window: float = 0.1,  # seconds
        debug: bool = False,
    ):
        """
        Initialize gripper force limiter.

        Source: i2rt/robots/utils.py:GripperForceLimiter.__init__()

        Args:
            max_force: Maximum allowed gripper force (Newtons)
            kp: KP gain of the gripper motor (for position adjustment)
            gripper_type: Type of gripper (affects force-torque mapping)
            clog_force_threshold: Effort threshold to detect clog (Nm)
            clog_speed_threshold: Velocity threshold to detect clog (rad/s)
            average_torque_window: Time window for averaging torque (seconds)
            debug: Enable debug logging
        """
        self.max_force = max_force
        self.kp = kp
        self.gripper_type = gripper_type
        self.clog_force_threshold = clog_force_threshold
        self.clog_speed_threshold = clog_speed_threshold
        self.average_torque_window = average_torque_window
        self.debug = debug

        self._is_clogged = False
        self._adjusted_pos: Optional[float] = None
        self._effort_history: deque = deque(maxlen=1000)

        # Get gripper-specific parameters
        # Source: i2rt/robots/utils.py:GripperType.get_gripper_limiter_params()
        self._setup_force_torque_map()

    def _setup_force_torque_map(self) -> None:
        """Set up force-to-torque mapping function based on gripper type."""
        if self.gripper_type == GripperType.CRANK_4310:
            # Crank parameters from i2rt
            self.direction_sign = 1.0
            self.force_torque_map = partial(
                crank_gripper_force_torque_map,
                gripper_close_angle=8.0 / 180.0 * np.pi,
                gripper_open_angle=170.0 / 180.0 * np.pi,
                gripper_stroke=0.071,  # meters
                motor_reading_to_crank_angle=lambda x: (-x + 0.174),
                gripper_force=self.max_force,
            )
        elif self.gripper_type in [GripperType.LINEAR_3507, GripperType.LINEAR_4310]:
            self.direction_sign = 1.0
            self.force_torque_map = partial(
                linear_gripper_force_torque_map,
                motor_stroke=6.57,
                gripper_stroke=0.096,
                gripper_force=self.max_force,
            )
        else:
            # No force limiting for teaching handle or no gripper
            self.direction_sign = 1.0
            self.force_torque_map = lambda current_angle: 0.0

    def _compute_target_torque(self, state: GripperState) -> Optional[float]:
        """
        Compute target torque if clogged, None if not clogged.

        Source: i2rt/robots/utils.py:GripperForceLimiter.compute_target_gripper_torque()
        """
        current_speed = state.current_vel

        # Get recent effort history
        if len(self._effort_history) == 0:
            return None

        history_ts, history_effort = zip(*self._effort_history)
        history_ts = np.array(history_ts)
        history_effort = np.array(history_effort)

        # Filter to recent window
        cutoff_time = time.time() - self.average_torque_window
        valid_idx = history_ts > cutoff_time
        if not np.any(valid_idx):
            return None

        average_effort = np.abs(np.mean(history_effort[valid_idx]))

        if self.debug:
            logger.debug(f"GripperForceLimiter: avg_effort={average_effort:.3f}, vel={current_speed:.3f}")

        # Check clog state transitions
        if self._is_clogged:
            # Check if we want to open (unclog condition)
            # 0 = closed, 1 = open in normalized space
            if (state.current_normalized_pos < state.target_normalized_pos) or average_effort < 0.2:
                self._is_clogged = False
                if self.debug:
                    logger.debug("GripperForceLimiter: unclogged")
        else:
            # Check if we're clogging
            if average_effort > self.clog_force_threshold and np.abs(current_speed) < self.clog_speed_threshold:
                self._is_clogged = True
                if self.debug:
                    logger.debug("GripperForceLimiter: clog detected")

        if self._is_clogged:
            # Compute target torque to achieve max force
            target_torque = self.force_torque_map(current_angle=state.current_pos) + 0.3  # friction compensation
            return target_torque
        else:
            return None

    def update(self, state: GripperState | dict) -> float:
        """
        Update force limiter and return adjusted target position.

        Source: i2rt/robots/utils.py:GripperForceLimiter.update()

        Args:
            state: Current gripper state

        Returns:
            Adjusted target position (radians) - either original target or limited position
        """
        if isinstance(state, dict):
            state = GripperState(
                target_pos=state["target_qpos"],
                current_pos=state["current_qpos"],
                current_vel=state["current_qvel"],
                current_effort=state["current_eff"],
                current_normalized_pos=state["current_normalized_qpos"],
                target_normalized_pos=state["target_normalized_qpos"],
                last_command_pos=state["last_command_qpos"],
            )

        # Record effort history
        current_time = time.time()
        self._effort_history.append((current_time, state.current_effort))

        # Check if clogged and compute target torque
        target_torque = self._compute_target_torque(state)

        if target_torque is not None:
            # Clogged - compute position adjustment to achieve target torque
            command_sign = np.sign(state.target_pos - state.current_pos) * self.direction_sign

            # Estimate zero-effort position from current state
            current_zero_eff_pos = (
                state.last_command_pos - command_sign * np.abs(state.current_effort) / self.kp
            )

            # Compute target position to achieve target torque
            target_pos = current_zero_eff_pos + command_sign * np.abs(target_torque) / self.kp

            if self.debug:
                logger.debug(f"GripperForceLimiter: clogged, adjusting pos to {target_pos:.4f}")

            # Apply EMA smoothing to adjusted position
            alpha = 0.1
            if self._adjusted_pos is None:
                self._adjusted_pos = target_pos
            else:
                self._adjusted_pos = (1 - alpha) * self._adjusted_pos + alpha * target_pos

            return self._adjusted_pos
        else:
            # Not clogged - return original target
            if self.debug and self._adjusted_pos is not None:
                logger.debug("GripperForceLimiter: not clogged, returning target")
            self._adjusted_pos = state.current_pos
            return state.target_pos

    def reset(self) -> None:
        """Reset force limiter state."""
        self._is_clogged = False
        self._adjusted_pos = None
        self._effort_history.clear()


def detect_gripper_limits(
    motor_chain,
    gripper_index: int = 6,
    test_torque: float = 0.2,
    max_duration: float = 2.0,
    position_threshold: float = 0.01,
    check_interval: float = 0.1,
) -> list[float]:
    """
    Detect gripper limits by applying test torques and monitoring position changes.

    Ported from i2rt/robots/utils.py:detect_gripper_limits.
    """
    logger = logging.getLogger(__name__)
    positions: list[float] = []
    num_motors = len(motor_chain.motor_list)
    zero_torques = np.zeros(num_motors)

    motor_direction = motor_chain.motor_direction[gripper_index]

    initial_states = motor_chain.read_states()
    init_torque = np.array([state.eff for state in initial_states])
    initial_pos = initial_states[gripper_index].pos
    positions.append(initial_pos)
    logger.info(f"Gripper calibration starting from position: {initial_pos:.4f}")

    for direction in [1, -1]:
        logger.info(f"Testing gripper direction: {direction}")
        test_torques = init_torque.copy()
        test_torques[gripper_index] = direction * test_torque

        start_time = time.time()
        last_pos = None
        position_stable_count = 0

        while time.time() - start_time < max_duration:
            motor_chain.set_commands(torques=test_torques)
            time.sleep(check_interval)

            states = motor_chain.read_states()
            current_pos = states[gripper_index].pos
            positions.append(current_pos)

            if last_pos is not None:
                pos_change = abs(current_pos - last_pos)
                if pos_change < position_threshold:
                    position_stable_count += 1
                else:
                    position_stable_count = 0
                if position_stable_count >= 3:
                    logger.info(f"Gripper limit detected: pos={current_pos:.4f}")
                    break

            last_pos = current_pos

        time.sleep(0.3)

    min_pos = min(positions)
    max_pos = max(positions)

    if motor_direction > 0:
        detected_limits = [max_pos, min_pos]
    else:
        detected_limits = [min_pos, max_pos]

    logger.info(f"Motor direction: {motor_direction}, detected limits: {detected_limits}")
    return detected_limits


# =============================================================================
# EMA FILTER
# Source: i2rt/scripts/zmq_follower.py smoothing logic
# =============================================================================

class EMAFilter:
    """
    Exponential Moving Average filter for smoothing positions.

    Replicates the smoothing from i2rt/scripts/zmq_follower.py:
        smoothed_qpos = alpha * target_qpos + (1 - alpha) * smoothed_qpos

    This reduces jitter from noisy leader position readings.
    """

    def __init__(self, alpha: float = 0.2, initial_value: Optional[np.ndarray] = None):
        """
        Initialize EMA filter.

        Args:
            alpha: Smoothing factor in (0, 1].
                   Higher = more responsive (less smoothing)
                   Lower = smoother (more lag)
                   alpha=1.0 means no smoothing (output = input)
                   Typical values: 0.1-0.3 for smooth tracking
            initial_value: Initial filter state. If None, first input is used.
        """
        self.alpha = np.clip(alpha, 0.01, 1.0)
        self._state: Optional[np.ndarray] = initial_value

    def update(self, value: np.ndarray) -> np.ndarray:
        """
        Update filter with new value and return smoothed output.

        Args:
            value: New input value (position array)

        Returns:
            Smoothed output value
        """
        value = np.asarray(value)

        if self._state is None:
            self._state = value.copy()
            return self._state

        # EMA formula: output = alpha * input + (1 - alpha) * previous_output
        self._state = self.alpha * value + (1.0 - self.alpha) * self._state
        return self._state.copy()

    def reset(self, value: Optional[np.ndarray] = None) -> None:
        """Reset filter state."""
        self._state = value.copy() if value is not None else None

    @property
    def value(self) -> Optional[np.ndarray]:
        """Current filter state."""
        return self._state.copy() if self._state is not None else None


# =============================================================================
# JOINT MAPPER
# Source: i2rt/robots/utils.py:JointMapper
# =============================================================================

class JointMapper:
    """
    Maps between normalized [0, 1] command space and physical joint space.

    Source: i2rt/robots/utils.py:JointMapper

    This is primarily used for gripper mapping where:
    - Command space: [0, 1] where 0=closed, 1=open
    - Physical space: actual motor radians

    For non-remapped joints, values pass through unchanged.
    """

    def __init__(
        self,
        index_range_map: Dict[int, Tuple[float, float]],
        total_dofs: int,
    ):
        """
        Initialize joint mapper.

        Args:
            index_range_map: Dict mapping joint index to (min, max) physical range
                             e.g., {6: (0.0, -2.7)} for gripper at index 6
            total_dofs: Total number of joints including gripper
        """
        self.empty = len(index_range_map) == 0
        self.total_dofs = total_dofs

        if not self.empty:
            self.joints_one_hot = np.zeros(total_dofs, dtype=bool)
            self.joint_limits = []

            for idx, (start, end) in index_range_map.items():
                self.joints_one_hot[idx] = True
                self.joint_limits.append((start, end))

            self.joint_limits = np.array(self.joint_limits)
            self.joint_range = self.joint_limits[:, 1] - self.joint_limits[:, 0]

    def to_physical_space(self, command_pos: np.ndarray) -> np.ndarray:
        """
        Convert from command space [0, 1] to physical space (radians).

        Source: i2rt/robots/utils.py:JointMapper.to_robot_joint_pos_space()
        """
        if self.empty:
            return command_pos

        command_pos = np.asarray(command_pos, order="C")
        result = command_pos.copy()

        # For remapped joints: physical = normalized * range + min
        needs_remapping = command_pos[self.joints_one_hot]
        needs_remapping = needs_remapping * self.joint_range + self.joint_limits[:, 0]
        result[self.joints_one_hot] = needs_remapping

        return result

    # i2rt compatibility aliases
    def to_robot_joint_pos_space(self, command_joint_pos: np.ndarray) -> np.ndarray:
        return self.to_physical_space(command_joint_pos)

    def to_command_space(self, physical_pos: np.ndarray) -> np.ndarray:
        """
        Convert from physical space (radians) to command space [0, 1].

        Source: i2rt/robots/utils.py:JointMapper.to_command_joint_pos_space()
        """
        if self.empty:
            return physical_pos

        result = physical_pos.copy()

        # For remapped joints: normalized = (physical - min) / range
        needs_remapping = physical_pos[self.joints_one_hot]
        needs_remapping = (needs_remapping - self.joint_limits[:, 0]) / self.joint_range
        result[self.joints_one_hot] = needs_remapping

        return result

    def to_command_joint_pos_space(self, robot_joint_pos: np.ndarray) -> np.ndarray:
        return self.to_command_space(robot_joint_pos)

    def to_physical_velocity(self, command_vel: np.ndarray) -> np.ndarray:
        """
        Convert velocity from command space to physical space.

        Source: i2rt/robots/utils.py:JointMapper.to_robot_joint_vel_space()
        """
        if self.empty:
            return command_vel

        result = command_vel.copy()
        needs_remapping = command_vel[self.joints_one_hot]
        needs_remapping = needs_remapping * self.joint_range
        result[self.joints_one_hot] = needs_remapping

        return result

    def to_robot_joint_vel_space(self, command_joint_vel: np.ndarray) -> np.ndarray:
        return self.to_physical_velocity(command_joint_vel)

    def to_command_velocity(self, physical_vel: np.ndarray) -> np.ndarray:
        """
        Convert velocity from physical space to command space.

        Source: i2rt/robots/utils.py:JointMapper.to_command_joint_vel_space()
        """
        if self.empty:
            return physical_vel

        result = physical_vel.copy()
        needs_remapping = physical_vel[self.joints_one_hot]
        needs_remapping = needs_remapping / self.joint_range
        result[self.joints_one_hot] = needs_remapping

        return result

    def to_command_joint_vel_space(self, robot_joint_vel: np.ndarray) -> np.ndarray:
        return self.to_command_velocity(robot_joint_vel)
