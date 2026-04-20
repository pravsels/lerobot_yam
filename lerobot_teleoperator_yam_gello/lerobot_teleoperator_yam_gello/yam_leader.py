"""
YAM Leader Teleoperator implementation for Lerobot.

This teleoperator reads joint positions from a GELLO-style teaching arm
using Dynamixel XL330 servos.
"""

import logging
import math
import sys
import time

from lerobot.motors import MotorCalibration, MotorNormMode
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

        # Range-safety state (Phase 2A)
        self._out_of_range_joints: set[str] = set()
        self._last_warn_time: dict[str, float] = {}

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

        if self.config.preflight_range_check:
            self._preflight_range_check()

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
        """Read current joint positions from the leader arm.

        IMPORTANT: We read RAW ticks (not normalized) so we can detect when a
        joint is actually outside its calibrated [range_min, range_max] window.
        LeRobot's built-in normalization silently clamps raw ticks to that
        window, which would mask out-of-range states (the value would just
        saturate at -100 / +100 instead).
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        if not self.is_calibrated:
            raise RuntimeError(
                "YAMLeader is not calibrated; refusing to return raw ticks. "
                "Run calibration or provide a calibration file."
            )

        start = time.perf_counter()
        ticks_by_name = self._read_raw_ticks()

        action: dict[str, float] = {}
        now = time.perf_counter()
        tol_ticks = self._normalized_tol_to_ticks_default()

        for motor, ticks in ticks_by_name.items():
            in_range = self._is_in_range_raw(motor, ticks, tol_ticks)
            if in_range:
                if motor in self._out_of_range_joints:
                    logger.warning(
                        "%s back in range (ticks=%d). Resuming teleop for this joint.",
                        motor, ticks,
                    )
                    self._out_of_range_joints.discard(motor)
                action[f"{motor}.pos"] = self._normalize_ticks(motor, ticks)
            else:
                last = self._last_warn_time.get(motor, 0.0)
                if now - last >= self.config.out_of_range_warn_period_s:
                    cal = self.calibration[motor]
                    direction = "DECREASE" if ticks > cal.range_max else "INCREASE"
                    action_str = (
                        "Holding follower joint."
                        if self.config.freeze_out_of_range
                        else "Sending clamped value."
                    )
                    logger.warning(
                        "%s OUT OF RANGE (ticks=%d, valid=[%d, %d]). %s "
                        "Rotate to %s ticks.",
                        motor, ticks, cal.range_min, cal.range_max, action_str, direction,
                    )
                    self._last_warn_time[motor] = now
                self._out_of_range_joints.add(motor)
                if self.config.freeze_out_of_range:
                    action[f"{motor}.pos"] = float("nan")
                else:
                    action[f"{motor}.pos"] = self._normalize_ticks(motor, ticks)

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read action in {dt_ms:.1f}ms")

        return action

    # =========================================================================
    # Range-safety helpers
    # =========================================================================

    def _read_raw_ticks(self) -> dict[str, int]:
        """Read raw Present_Position ticks (no clamping/normalization) with retry."""
        last_exc = None
        for _ in range(max(1, self.config.read_retries)):
            try:
                return self.bus.sync_read("Present_Position", normalize=False)
            except Exception as exc:
                last_exc = exc
                time.sleep(self.config.read_retry_sleep_s)
        raise last_exc

    def _normalized_tol_to_ticks_default(self) -> int:
        """Average tick tolerance derived from the configured normalized tolerance.

        We approximate using the smallest joint span so out-of-range detection
        is conservative. Returns at least 1 tick.
        """
        spans = [
            max(1, self.calibration[m].range_max - self.calibration[m].range_min)
            for m in self.bus.motors
        ]
        smallest = min(spans) if spans else 1
        norm_tol = max(0.0, float(self.config.out_of_range_tolerance))
        return max(1, int(round((norm_tol / 200.0) * smallest)))

    def _is_in_range_raw(self, motor_name: str, ticks: int, tol_ticks: int) -> bool:
        """Check if a raw tick value is within the calibrated [range_min, range_max]."""
        cal = self.calibration[motor_name]
        return (cal.range_min - tol_ticks) <= ticks <= (cal.range_max + tol_ticks)

    def _normalize_ticks(self, motor_name: str, ticks: int) -> float:
        """Replicate lerobot _normalize() formula manually for a single motor.

        Used so out-of-range joints can be returned as NaN instead of being
        silently clamped to the limits.
        """
        cal = self.calibration[motor_name]
        motor = self.bus.motors[motor_name]
        lo, hi = cal.range_min, cal.range_max
        if hi == lo:
            return 0.0
        bounded = min(hi, max(lo, ticks))
        if motor.norm_mode == MotorNormMode.RANGE_M100_100:
            norm = (((bounded - lo) / (hi - lo)) * 200.0) - 100.0
            return -norm if cal.drive_mode else norm
        if motor.norm_mode == MotorNormMode.RANGE_0_100:
            norm = ((bounded - lo) / (hi - lo)) * 100.0
            return 100.0 - norm if cal.drive_mode else norm
        return float(ticks)

    def _preflight_range_check(self) -> None:
        """
        Phase 1: block until every leader joint is within its valid range.

        The user may have moved the GELLO arm while the program was off, so on
        connect we verify each joint and prod the user to rotate any offending
        joint back into range before teleop begins.

        This works on RAW ticks, since LeRobot's normalize() silently clamps
        out-of-range positions to the calibration window.
        """
        hz = max(1.0, float(self.config.preflight_refresh_hz))
        period = 1.0 / hz
        tol_ticks = self._normalized_tol_to_ticks_default()

        print(
            "Preflight: checking GELLO joint ranges. Rotate any flagged joint until in range.",
            flush=True,
        )
        printed_status = False
        try:
            while True:
                ticks_by_name = self._read_raw_ticks()
                out = []
                for name, ticks in ticks_by_name.items():
                    if not self._is_in_range_raw(name, ticks, tol_ticks):
                        out.append((name, ticks))

                if not out:
                    if printed_status:
                        sys.stdout.write("\r\x1b[2K")
                        sys.stdout.flush()
                    print("Preflight: all joints in range. Starting teleop.", flush=True)
                    return

                parts = []
                for name, ticks in out:
                    cal = self.calibration[name]
                    direction = "DECREASE" if ticks > cal.range_max else "INCREASE"
                    parts.append(
                        f"{name} (ticks={ticks}, valid=[{cal.range_min},{cal.range_max}], "
                        f"rotate to {direction})"
                    )
                sys.stdout.write("\r\x1b[2KOut of range: " + " | ".join(parts))
                sys.stdout.flush()
                printed_status = True
                time.sleep(period)
        except KeyboardInterrupt:
            print("\nPreflight aborted by user.", flush=True)
            raise

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
