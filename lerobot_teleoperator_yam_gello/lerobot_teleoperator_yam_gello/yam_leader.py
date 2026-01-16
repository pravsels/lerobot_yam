"""
YAM Leader Teleoperator implementation for Lerobot.

This teleoperator reads joint positions from a GELLO-style teaching arm
using Dynamixel XL330 servos.
"""

import logging
import time

from lerobot.motors import MotorCalibration
from lerobot.motors.dynamixel import DynamixelMotorsBus, OperatingMode
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from lerobot.teleoperators.teleoperator import Teleoperator

from .config_yam_leader import YAMLeaderTeleopConfig

logger = logging.getLogger(__name__)


class YAMLeader(Teleoperator):
    """
    Teleoperator implementation for YAM GELLO-style leader arm.

    This class reads joint positions from Dynamixel XL330 servos and
    outputs normalized values that can be sent to the follower robot.

    The leader arm has torque disabled, allowing the user to freely
    move the arm. Position readings are normalized using calibration
    data (min/max tick values per joint).
    """

    config_class = YAMLeaderTeleopConfig
    name = "yam_leader"

    def __init__(self, config: YAMLeaderTeleopConfig):
        """Initialize YAM Leader teleoperator."""
        super().__init__(config)
        self.config = config

        # Create Dynamixel motor bus
        self.bus = DynamixelMotorsBus(
            port=config.port,
            motors=config.motors,
            calibration=self.calibration,
        )
        self._is_calibrated_cached = False

    @property
    def action_features(self) -> dict[str, type]:
        """Features returned by get_action()."""
        return {f"{motor}.pos": float for motor in self.bus.motors}

    @property
    def feedback_features(self) -> dict[str, type]:
        """Features expected by send_feedback() - empty for leader."""
        return {}

    @property
    def is_connected(self) -> bool:
        """Check if teleoperator is connected."""
        return self.bus.is_connected

    @property
    def is_calibrated(self) -> bool:
        """Check if teleoperator is calibrated."""
        return self._is_calibrated_cached

    def connect(self, calibrate: bool = True) -> None:
        """Connect to teleoperator hardware."""
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        # Connect without handshake first, then set baudrate, then verify motors
        # YAM leader uses 57600 baud (not lerobot's default 1MHz)
        self.bus.connect(handshake=False)
        self.bus.set_baudrate(self.config.baudrate)
        self.bus._handshake()

        if not self.is_calibrated and calibrate:
            logger.info("Teleoperator not calibrated. Starting calibration...")
            self.calibrate()
        else:
            # Cache calibration state without probing hardware every call
            self._is_calibrated_cached = bool(self.calibration)

        self.configure()
        logger.info(f"{self} connected.")

    def calibrate(self) -> None:
        """Calibrate the leader by recording joint ranges of motion."""
        if self.calibration:
            # Calibration file exists - ask user
            user_input = input(
                f"Press ENTER to use existing calibration for '{self.id}', "
                "or type 'c' to run new calibration: "
            )
            if user_input.strip().lower() != "c":
                logger.info(f"Using existing calibration for '{self.id}'")
                # Do not write calibration to hardware; use for software normalization only.
                self.bus.calibration = self.calibration
                self._is_calibrated_cached = True
                return

        logger.info(f"\nRunning calibration for {self}")
        print("For each joint: move through full range of motion, then press ENTER.\n")

        # Ensure torque is disabled for free movement
        self.bus.disable_torque()

        # Record min/max for each joint
        range_mins = {}
        range_maxes = {}

        for motor_name in self.bus.motors:
            print(f"Joint: {motor_name}")
            print("  Move through full range, press ENTER when done...")

            vmin, vmax = float("inf"), float("-inf")

            import select
            import sys

            while True:
                # Read current position
                pos = self.bus.sync_read("Present_Position", [motor_name], normalize=False)
                val = pos[motor_name]

                vmin = min(vmin, val)
                vmax = max(vmax, val)

                # Live display
                sys.stdout.write(f"\r  tick={val:6d}  min={int(vmin):6d}  max={int(vmax):6d}  ")
                sys.stdout.flush()

                # Check for ENTER (non-blocking)
                readable, _, _ = select.select([sys.stdin], [], [], 0.05)
                if readable:
                    sys.stdin.readline()
                    break

            raw_min = int(vmin)
            raw_max = int(vmax)

            if motor_name in {"wrist_yaw", "shoulder_pan"}:
                print(f"\n  Move {motor_name} to its center position, then press ENTER.")
                input()
                center = self.bus.sync_read("Present_Position", [motor_name], normalize=False)[motor_name]

                span = min(raw_max - center, center - raw_min) * 2
                if span <= 0:
                    span = raw_max - raw_min
                range_mins[motor_name] = int(round(center - span / 2))
                range_maxes[motor_name] = int(round(center + span / 2))
                print(
                    f"\n  Recorded: {range_mins[motor_name]} to {range_maxes[motor_name]} "
                    f"(centered at {int(round(center))})\n"
                )
            else:
                range_mins[motor_name] = raw_min
                range_maxes[motor_name] = raw_max
                print(f"\n  Recorded: {range_mins[motor_name]} to {range_maxes[motor_name]}\n")

        # Create calibration dict
        self.calibration = {}
        for motor_name, motor in self.bus.motors.items():
            self.calibration[motor_name] = MotorCalibration(
                id=motor.id,
                drive_mode=0,
                homing_offset=0,  # No homing offset needed
                range_min=range_mins[motor_name],
                range_max=range_maxes[motor_name],
            )

        # Save calibration (software only; do not write limits to hardware)
        self.bus.calibration = self.calibration
        self._save_calibration()
        self._is_calibrated_cached = True
        logger.info(f"Calibration saved to {self.calibration_fpath}")

    def configure(self) -> None:
        """Configure teleoperator after connection."""
        self.bus.disable_torque()
        self.bus.configure_motors()

        # Set all motors to position mode (for reading Present_Position)
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

    def get_action(self) -> dict[str, float]:
        """Read current joint positions from the leader arm."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        if not self.is_calibrated:
            raise RuntimeError(
                "YAMLeader is not calibrated; refusing to return raw ticks. "
                "Run calibration or provide a calibration file."
            )

        start = time.perf_counter()

        last_exc = None
        for _ in range(max(1, self.config.read_retries)):
            try:
                positions = self.bus.sync_read("Present_Position", normalize=True)
                break
            except Exception as exc:
                last_exc = exc
                time.sleep(self.config.read_retry_sleep_s)
        else:
            raise last_exc

        action = {f"{motor}.pos": val for motor, val in positions.items()}

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read action in {dt_ms:.1f}ms")

        return action

    def send_feedback(self, feedback: dict[str, float]) -> None:
        """Send feedback to teleoperator (no-op for leader)."""
        pass

    def disconnect(self) -> None:
        """Disconnect from teleoperator hardware."""
        if not self.is_connected:
            return
        self.bus.disconnect()
        logger.info(f"{self} disconnected.")


class YAMLeaderTeleop(YAMLeader):
    """Alias class for Teleoperator config auto-discovery."""

    pass
