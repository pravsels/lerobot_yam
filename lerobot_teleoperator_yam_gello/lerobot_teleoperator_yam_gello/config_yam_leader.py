"""
Configuration for YAM Leader teleoperator (GELLO-style).

The YAM leader uses Dynamixel XL330 servos for position sensing.
It reads joint positions and outputs normalized values for the follower.
"""

from dataclasses import dataclass, field

from lerobot.motors import Motor, MotorNormMode
from lerobot.teleoperators.config import TeleoperatorConfig


@dataclass
class YAMLeaderConfig:
    """
    Configuration for YAM Leader teleoperator.

    The leader uses Dynamixel XL330 servos (torque-disabled) to read
    joint positions from a GELLO-style teaching arm.
    """

    # Serial port for Dynamixel bus
    port: str = "/dev/ttyUSB0"
    baudrate: int = 57600

    # Read stability
    read_retries: int = 10
    read_retry_sleep_s: float = 0.01

    # Range-safety behavior
    # Phase 1: block teleop start until every joint is within its valid normalized
    # range (e.g. [-100, 100] for arm joints, [0, 100] for gripper).
    preflight_range_check: bool = True
    preflight_refresh_hz: float = 10.0

    # Phase 2A: at runtime, freeze any joint whose leader value is out of range
    # (NaN action -> follower holds current position) and emit a throttled warning.
    freeze_out_of_range: bool = True
    out_of_range_warn_period_s: float = 1.0

    # Tolerance band (in normalized units) to avoid flicker right at the boundary.
    out_of_range_tolerance: float = 1.0

    # Motor configuration
    # XL330-M288 for most joints, XL330-M077 for gripper (different gear ratio)
    motors: dict[str, Motor] = field(default_factory=lambda: {
        "shoulder_pan": Motor(1, "xl330-m288", MotorNormMode.RANGE_M100_100),
        "shoulder_lift": Motor(2, "xl330-m288", MotorNormMode.RANGE_M100_100),
        "elbow_flex": Motor(3, "xl330-m288", MotorNormMode.RANGE_M100_100),
        "wrist_flex": Motor(4, "xl330-m288", MotorNormMode.RANGE_M100_100),
        "wrist_roll": Motor(5, "xl330-m288", MotorNormMode.RANGE_M100_100),
        "wrist_yaw": Motor(6, "xl330-m288", MotorNormMode.RANGE_M100_100),
        "gripper": Motor(7, "xl330-m077", MotorNormMode.RANGE_0_100),
    })


@TeleoperatorConfig.register_subclass("yam_leader")
@dataclass
class YAMLeaderTeleopConfig(TeleoperatorConfig, YAMLeaderConfig):
    """Combined TeleoperatorConfig + YAMLeaderConfig for lerobot registration."""
    pass
