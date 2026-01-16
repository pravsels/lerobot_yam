"""
Configuration for YAM Follower robot.

This configuration replicates i2rt/robots/get_robot.py parameters exactly.

Source files:
- i2rt/robots/get_robot.py: Motor list, KP/KD, joint limits
- i2rt/robots/utils.py: GripperType defaults
"""

from dataclasses import dataclass, field
from typing import Optional

from lerobot.cameras import CameraConfig
from lerobot.motors import Motor, MotorNormMode
from lerobot.robots.config import RobotConfig


# =============================================================================
# DEFAULT CONFIGURATION VALUES FROM i2rt
# Source: i2rt/robots/get_robot.py
# =============================================================================

# KP gains per joint - EXACTLY from i2rt/robots/get_robot.py line 52, 56
DEFAULT_KP_GAINS = {
    "shoulder_pan": 80.0,
    "shoulder_lift": 80.0,
    "elbow_flex": 80.0,
    "wrist_flex": 40.0,
    "wrist_roll": 10.0,
    "wrist_yaw": 10.0,
    "gripper": 20.0,  # From GripperType.CRANK_4310.get_motor_kp_kd()
}

# KD gains per joint - EXACTLY from i2rt/robots/get_robot.py line 53, 56
DEFAULT_KD_GAINS = {
    "shoulder_pan": 5.0,
    "shoulder_lift": 5.0,
    "elbow_flex": 5.0,
    "wrist_flex": 1.5,
    "wrist_roll": 1.5,
    "wrist_yaw": 1.5,
    "gripper": 0.5,  # From GripperType.CRANK_4310.get_motor_kp_kd()
}

# Joint limits in radians - from i2rt/robots/get_robot.py lines 47-49
# Note: i2rt adds +-0.15 buffer to original limits
DEFAULT_JOINT_LIMITS = {
    "shoulder_pan": (-2.767, 3.28),     # Original: [-2.617, 3.13] + 0.15 buffer
    "shoulder_lift": (-0.15, 3.8),      # Original: [0, 3.65] + 0.15 buffer
    "elbow_flex": (-0.15, 3.28),        # Original: [0.0, 3.13] + 0.15 buffer
    "wrist_flex": (-1.72, 1.72),        # Original: [-1.57, 1.57] + 0.15 buffer
    "wrist_roll": (-1.72, 1.72),        # Original: [-1.57, 1.57] + 0.15 buffer
    "wrist_yaw": (-2.24, 2.24),         # Original: [-2.09, 2.09] + 0.15 buffer
}

# Gripper limits for CRANK_4310 - from i2rt/robots/utils.py:GripperType.get_gripper_limits()
# (open_position, closed_position) in radians
DEFAULT_GRIPPER_LIMITS = (0.0, -2.7)

# Motor offsets and directions - from i2rt/robots/get_robot.py lines 46, 51
# These default to zero/positive but may be adjusted during calibration
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


# =============================================================================
# CONFIGURATION DATACLASS
# =============================================================================

@dataclass
class YAMFollowerConfig:
    """
    Configuration for YAM Follower robot.

    This configuration replicates i2rt/robots/get_robot.py:get_yam_robot() parameters.
    """

    # =========================================================================
    # CAN Bus Configuration
    # =========================================================================
    port: str = "can0"
    bitrate: int = 1000000
    bustype: str = "socketcan"

    # =========================================================================
    # Motor Configuration
    # Source: i2rt/robots/get_robot.py motor_list (lines 38-45, 54-65)
    # =========================================================================
    motors: dict[str, Motor] = field(default_factory=lambda: {
        # DM4340 for larger joints (shoulder, elbow)
        "shoulder_pan": Motor(1, "DM4340", MotorNormMode.RANGE_M100_100),
        "shoulder_lift": Motor(2, "DM4340", MotorNormMode.RANGE_M100_100),
        "elbow_flex": Motor(3, "DM4340", MotorNormMode.RANGE_M100_100),
        # DM4310 for smaller joints (wrist)
        "wrist_flex": Motor(4, "DM4310", MotorNormMode.RANGE_M100_100),
        "wrist_roll": Motor(5, "DM4310", MotorNormMode.RANGE_M100_100),
        "wrist_yaw": Motor(6, "DM4310", MotorNormMode.RANGE_M100_100),
        # DM4310 for gripper (CRANK_4310 type)
        "gripper": Motor(7, "DM4310", MotorNormMode.RANGE_0_100),
    })

    # Motor offsets in radians - for zero position calibration
    # Source: i2rt/robots/get_robot.py motor_offsets
    motor_offsets: dict[str, float] = field(default_factory=lambda: DEFAULT_MOTOR_OFFSETS.copy())

    # Motor directions (1 or -1) - for reversed motor mounting
    # Source: i2rt/robots/get_robot.py motor_directions
    motor_directions: dict[str, int] = field(default_factory=lambda: DEFAULT_MOTOR_DIRECTIONS.copy())

    # =========================================================================
    # Control Gains
    # Source: i2rt/robots/get_robot.py kp, kd arrays
    # =========================================================================
    kp_gains: dict[str, float] = field(default_factory=lambda: DEFAULT_KP_GAINS.copy())
    kd_gains: dict[str, float] = field(default_factory=lambda: DEFAULT_KD_GAINS.copy())

    # =========================================================================
    # Joint Limits
    # Source: i2rt/robots/get_robot.py joint_limits
    # =========================================================================
    joint_limits: dict[str, tuple[float, float]] = field(
        default_factory=lambda: DEFAULT_JOINT_LIMITS.copy()
    )
    gripper_limits: tuple[float, float] = DEFAULT_GRIPPER_LIMITS

    # =========================================================================
    # Gravity Compensation
    # Source: i2rt/robots/get_robot.py use_gravity_comp, gravity_comp_factor
    # =========================================================================
    use_gravity_compensation: bool = True
    gravity_comp_factor: float = 1.3  # From i2rt/robots/get_robot.py line 116

    # Path to MuJoCo XML model for gravity compensation
    # If None, will search default locations
    mujoco_xml_path: Optional[str] = None

    # Gripper type for selecting correct XML model
    gripper_type: str = "crank_4310"

    # Optional cameras to include in observations
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # i2rt behavior: start in zero-gravity mode unless disabled.
    zero_gravity_mode: bool = True

    # =========================================================================
    # Gripper Force Limiting
    # Source: i2rt/robots/get_robot.py limit_gripper_force (line 129)
    # =========================================================================
    limit_gripper_force: float = 50.0  # Newtons, -1 to disable

    # =========================================================================
    # LeRobot safety step limits (applies to send_action only)
    # =========================================================================
    lerobot_max_step: float = 5.0  # Max arm joint step per cycle in [-100,100] units
    lerobot_gripper_max_step: float = 5.0  # Max gripper step per cycle in [0,100]

    # No lerobot-specific smoothing or safety knobs here. Keep this config
    # aligned with i2rt defaults only.

@RobotConfig.register_subclass("yam_follower")
@dataclass
class YAMFollowerRobotConfig(RobotConfig, YAMFollowerConfig):
    """Combined RobotConfig + YAMFollowerConfig for lerobot registration."""
    pass
