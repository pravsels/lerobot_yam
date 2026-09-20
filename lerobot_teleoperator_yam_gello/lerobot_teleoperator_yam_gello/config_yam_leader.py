"""
Configuration for YAM Leader teleoperator (GELLO-style).

The YAM leader uses Dynamixel XL330 servos for position sensing.
It reads joint positions and outputs normalized values for the follower.
"""

import math
from dataclasses import dataclass, field

from lerobot.motors import Motor, MotorNormMode
from lerobot.teleoperators.config import TeleoperatorConfig

ARM_JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "wrist_yaw",
)

# The LeRobot calibration maps each GELLO joint to the matching YAM follower's
# normalized range. These are the follower's *buffered* joint limits (i2rt
# defaults). They turn normalized values back into radians for the gravity
# model, and size the follower_limit_buffer_rad padding below.
DEFAULT_YAM_JOINT_RANGES_RAD = (
    (-2.767, 3.28),
    (-0.15, 3.8),
    (-0.15, 3.28),
    (-1.72, 1.72),
    (-1.72, 1.72),
    (-2.24, 2.24),
)


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

    # i2rt's YAM follower normalizes over joint limits that include a ±0.15 rad
    # software buffer beyond the true mechanical range, while a GELLO sweep
    # records only the true range. Without compensation the leader's endpoints
    # command unreachable targets, so every joint gets a ~0.15 rad dead zone at
    # each end of travel (felt worst at the elbow's rest pose, which sits at
    # the true limit). Padding the normalization window by this buffer makes
    # leader stops map to the true limits instead. Set 0 to disable.
    follower_limit_buffer_rad: float = 0.15

    # Active assistance is deliberately opt-in. The passive GELLO uses XL330s,
    # which are substantially weaker than the XC/XM servos in FACTR's active
    # GELLO. Defaults therefore provide partial support, not hands-off holding.
    gravity_assist: bool = False
    gravity_assist_gain: float = 0.15
    gravity_assist_current_limit_ma: int = 250
    # Which arm joints receive assist current. Subsetting lets one joint be
    # bench-tested at a time (see gello_gravity_hold); unlisted joints stay
    # passive exactly as when assist is off.
    gravity_assist_joints: tuple[str, ...] = ARM_JOINT_NAMES
    gravity_assist_damping_nm_per_rad_s: float = 0.003
    gravity_joint_ranges_rad: tuple[tuple[float, float], ...] = DEFAULT_YAM_JOINT_RANGES_RAD
    # Motor direction relative to the active-GELLO URDF. These are the FACTR
    # YAM defaults; override them if a GELLO was assembled with a reversed horn.
    gravity_joint_signs: tuple[int, ...] = (1, -1, -1, -1, 1, 1)
    # Radians added per joint after the sign flip: q_urdf = sign * q_yam + offset.
    # Zero assumes the GELLO's URDF home coincides with the YAM zero pose
    # (the default build pose). Tune with gravity_assist_dry_run if a joint's
    # modeled torque is wrong at a known pose; offsets are usually 0 or ±pi/2.
    gravity_joint_offsets_rad: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    # Which way the URDF's +z axis points physically when the GELLO is
    # mounted: +1 up, -1 down. The bundled yam_active_gello export is z-down
    # (-1) for the standard tabletop mount; with the wrong value every assist
    # torque is inverted and the arm is pushed toward its fallen pose.
    gravity_urdf_z_sign: int = -1
    # Optional per-link mass override (kg), base to tip: link_base, link_1 …
    # link_6 (7 values). Gravity torque is linear in these masses, so this is
    # the sysid knob for leaders built differently from the bundled URDF
    # (e.g. TRLC-DK1): weigh the printed links, or tune from dry-run logs.
    # Empty tuple keeps the URDF masses.
    gravity_link_masses_kg: tuple[float, ...] = ()
    # Compute and log assist currents without configuring motors or applying
    # any torque. Use this first on new hardware to verify signs/offsets.
    gravity_assist_dry_run: bool = False

    # A current-limited position spring holds the squeeze trigger open while
    # remaining easy to press and hold. This can be used without arm gravity
    # assistance.
    gripper_return: bool = False
    gripper_return_current_ma: int = 100

    # XL330 operating maximum is 70 C. Latch assistance off well below it.
    assist_temperature_limit_c: int = 50
    assist_bus_watchdog_ms: int = 200

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

    def __post_init__(self) -> None:
        if not 0.0 <= self.gravity_assist_gain <= 1.0:
            raise ValueError("gravity_assist_gain must be in [0, 1]")
        if not 1 <= self.gravity_assist_current_limit_ma <= 500:
            raise ValueError("gravity_assist_current_limit_ma must be in [1, 500]")
        if not 1 <= self.gripper_return_current_ma <= 300:
            raise ValueError("gripper_return_current_ma must be in [1, 300]")
        if self.gravity_assist_damping_nm_per_rad_s < 0:
            raise ValueError("gravity_assist_damping_nm_per_rad_s must be >= 0")
        if len(self.gravity_joint_ranges_rad) != len(ARM_JOINT_NAMES):
            raise ValueError("gravity_joint_ranges_rad must contain six [low, high] pairs")
        if any(float(low) >= float(high) for low, high in self.gravity_joint_ranges_rad):
            raise ValueError("gravity_joint_ranges_rad pairs must be ordered low < high")
        if len(self.gravity_joint_signs) != len(ARM_JOINT_NAMES):
            raise ValueError("gravity_joint_signs must contain six values")
        if any(int(sign) not in {-1, 1} for sign in self.gravity_joint_signs):
            raise ValueError("gravity_joint_signs values must be -1 or 1")
        if len(self.gravity_joint_offsets_rad) != len(ARM_JOINT_NAMES):
            raise ValueError("gravity_joint_offsets_rad must contain six values")
        if any(not math.isfinite(float(o)) for o in self.gravity_joint_offsets_rad):
            raise ValueError("gravity_joint_offsets_rad values must be finite")
        if int(self.gravity_urdf_z_sign) not in {-1, 1}:
            raise ValueError("gravity_urdf_z_sign must be -1 or 1")
        if self.gravity_link_masses_kg and len(self.gravity_link_masses_kg) != 7:
            raise ValueError(
                "gravity_link_masses_kg must be empty or 7 values (base to tip)"
            )
        if any(
            not math.isfinite(float(m)) or float(m) < 0
            for m in self.gravity_link_masses_kg
        ):
            raise ValueError("gravity_link_masses_kg values must be finite and >= 0")
        if not 0.0 <= self.follower_limit_buffer_rad < 0.5:
            raise ValueError("follower_limit_buffer_rad must be in [0, 0.5)")
        joints = tuple(self.gravity_assist_joints)
        if not joints or len(set(joints)) != len(joints):
            raise ValueError("gravity_assist_joints must be a non-empty set of joints")
        unknown = [name for name in joints if name not in ARM_JOINT_NAMES]
        if unknown:
            raise ValueError(f"gravity_assist_joints has unknown joints: {unknown}")
        if not 35 <= self.assist_temperature_limit_c <= 60:
            raise ValueError("assist_temperature_limit_c must be in [35, 60]")
        if not 100 <= self.assist_bus_watchdog_ms <= 1000:
            raise ValueError("assist_bus_watchdog_ms must be in [100, 1000]")


@TeleoperatorConfig.register_subclass("yam_leader")
@dataclass
class YAMLeaderTeleopConfig(TeleoperatorConfig, YAMLeaderConfig):
    """Combined TeleoperatorConfig + YAMLeaderConfig for lerobot registration."""
    pass
