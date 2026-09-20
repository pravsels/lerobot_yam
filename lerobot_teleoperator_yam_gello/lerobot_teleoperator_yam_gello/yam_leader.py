"""
YAM Leader Teleoperator implementation for Lerobot.

This teleoperator reads joint positions from a GELLO-style teaching arm
using Dynamixel XL330 servos.
"""

import logging
import math
import sys
import time
from dataclasses import replace

import numpy as np
from lerobot.motors import MotorCalibration, MotorNormMode
from lerobot.motors.dynamixel import DynamixelMotorsBus, OperatingMode
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from lerobot.teleoperators.teleoperator import Teleoperator

from .config_yam_leader import ARM_JOINT_NAMES, YAMLeaderTeleopConfig
from .gravity_assist import (
    GelloGravityModel,
    gello_joint_positions,
    torque_to_current_ma,
)

logger = logging.getLogger(__name__)

# XL330-M077 / M288 share the X-series table; GELLO leaders only read position.
_XL330_MODEL_NUMBERS = {
    1190: "xl330-m077",
    1200: "xl330-m288",
}


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
        self._gravity_model: GelloGravityModel | None = None
        self._assist_enabled = False
        self._assist_faulted = False
        self._last_assist_q: np.ndarray | None = None
        self._last_assist_time: float | None = None
        self._assist_velocity = np.zeros(len(ARM_JOINT_NAMES), dtype=np.float64)
        self._live_assist_motors: list[str] = []
        self._last_health_check_time = 0.0
        self._last_dry_run_log_time = 0.0

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
        self._bind_detected_xl330_models()
        self.bus._handshake()

        try:
            # Position mode before any Present_Position read. Current/extended
            # position reports signed multi-turn ticks (-1, 7227, …); calibration
            # and preflight need the 0–4095 joint circle.
            self.configure()

            if not self.is_calibrated and calibrate:
                logger.info("Teleoperator not calibrated. Starting calibration...")
                self.calibrate()
            else:
                # Cache calibration state without probing hardware every call
                self._is_calibrated_cached = bool(self.calibration)

            if self.config.preflight_range_check and self.calibration:
                self._preflight_range_check()
            if self.config.gravity_assist or self.config.gripper_return:
                self._enable_assistance()
        except Exception:
            self._safe_disable_assistance()
            self.bus.disconnect()
            raise

        logger.info(f"{self} connected.")

    def _bind_detected_xl330_models(self) -> None:
        """Set each motor's model from a ping so M077 and M288 GELLOs both handshake."""
        found = self.bus.broadcast_ping() or {}
        id_to_name = {motor.id: name for name, motor in self.bus.motors.items()}
        for dxl_id, model_nb in found.items():
            name = id_to_name.get(dxl_id)
            if name is None:
                continue
            model = _XL330_MODEL_NUMBERS.get(int(model_nb))
            if model is None:
                continue
            motor = self.bus.motors[name]
            if motor.model == model:
                continue
            logger.info(
                "GELLO %s (id=%s): using %s (model %s) instead of %s",
                name,
                dxl_id,
                model,
                model_nb,
                motor.model,
            )
            self.bus.motors[name] = replace(motor, model=model)

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

        # Force POSITION (mode 3) before recording ticks. Torque stays off.
        self.configure()

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
        """Disable torque and lock every XL330 in POSITION mode (mode 3)."""
        self.bus.disable_torque()
        self.bus.configure_motors()

        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)
            mode = int(self.bus.read("Operating_Mode", motor))
            if mode != OperatingMode.POSITION.value:
                raise RuntimeError(
                    f"{motor} Operating_Mode={mode}, expected POSITION "
                    f"({OperatingMode.POSITION.value})"
                )

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

        if self._assist_enabled and not self._assist_faulted:
            self._update_assistance(action)

        return action

    # =========================================================================
    # Opt-in gravity and gripper assistance
    # =========================================================================

    def _enable_assistance(self) -> None:
        if not self.calibration:
            raise RuntimeError("GELLO assistance requires a leader calibration")

        arm_motors = [
            name for name in ARM_JOINT_NAMES if name in self.config.gravity_assist_joints
        ]
        live_gravity = (
            self.config.gravity_assist and not self.config.gravity_assist_dry_run
        )
        assisted_motors = list(arm_motors) if live_gravity else []
        if self.config.gripper_return:
            assisted_motors.append("gripper")

        self.bus.disable_torque(assisted_motors)
        watchdog_raw = self._watchdog_raw()

        if self.config.gravity_assist:
            self._gravity_model = GelloGravityModel(
                gravity_z_sign=self.config.gravity_urdf_z_sign,
                link_masses_kg=self.config.gravity_link_masses_kg or None,
            )
            logger.info(
                "GELLO gravity model link masses (kg): %s",
                {k: round(v, 4) for k, v in self._gravity_model.link_masses.items()},
            )
        if self.config.gravity_assist and not live_gravity:
            logger.warning(
                "GELLO gravity assist DRY RUN on %s: computing and logging "
                "currents, applying no torque to the arm joints",
                self.config.port,
            )
        if live_gravity:
            for motor in arm_motors:
                self._set_current_limit_if_needed(
                    motor, self.config.gravity_assist_current_limit_ma
                )
                self.bus.write("Operating_Mode", motor, OperatingMode.CURRENT.value)
                self.bus.write("Bus_Watchdog", motor, watchdog_raw)
                self.bus.write("Goal_Current", motor, 0, normalize=False)

        if self.config.gripper_return:
            open_tick = self._gripper_open_tick()
            self._set_current_limit_if_needed(
                "gripper", self.config.gripper_return_current_ma
            )
            self.bus.write(
                "Operating_Mode", "gripper", OperatingMode.CURRENT_POSITION.value
            )
            self.bus.write("Bus_Watchdog", "gripper", watchdog_raw)
            self.bus.write(
                "Goal_Current",
                "gripper",
                self.config.gripper_return_current_ma,
                normalize=False,
            )
            self.bus.write(
                "Goal_Position",
                "gripper",
                open_tick,
                normalize=False,
            )
            calibration = self.calibration["gripper"]
            logger.warning(
                "GELLO gripper return target: port=%s open_tick=%d "
                "calibration=[%d,%d] current_limit=%dmA",
                self.config.port,
                open_tick,
                calibration.range_min,
                calibration.range_max,
                self.config.gripper_return_current_ma,
            )

        self.bus.enable_torque(assisted_motors)
        self._live_assist_motors = list(assisted_motors)
        self._assist_enabled = True
        self._assist_faulted = False
        self._last_assist_q = None
        self._last_assist_time = None
        self._assist_velocity[:] = 0.0
        self._last_health_check_time = time.monotonic()
        logger.warning(
            "GELLO active assistance enabled on %s: gravity=%s (gain=%.3f, "
            "limit=%dmA), gripper_return=%s (limit=%dmA), watchdog=%dms",
            self.config.port,
            self.config.gravity_assist,
            self.config.gravity_assist_gain,
            self.config.gravity_assist_current_limit_ma,
            self.config.gripper_return,
            self.config.gripper_return_current_ma,
            self.config.assist_bus_watchdog_ms,
        )

    def _watchdog_raw(self) -> int:
        return max(1, int(math.ceil(self.config.assist_bus_watchdog_ms / 20.0)))

    def _set_current_limit_if_needed(self, motor: str, limit_ma: int) -> None:
        existing = int(self.bus.read("Current_Limit", motor))
        if existing != int(limit_ma):
            # Current_Limit is EEPROM. Avoid spending a write cycle on every
            # connection once the conservative hardware ceiling is installed.
            self.bus.write("Current_Limit", motor, int(limit_ma), normalize=False)

    def _gripper_open_tick(self) -> int:
        calibration = self.calibration["gripper"]
        # The YAM squeeze trigger's physical open/rest pose is the high-tick
        # endpoint on a standard build; the operator squeezes toward low ticks
        # to close. A mirrored build (e.g. the right leader of a bimanual DK1
        # pair) reverses the tick direction — mark its calibration with
        # drive_mode=1, which flips both LeRobot's normalization (teleop
        # squeeze direction) and this spring target consistently.
        return (
            int(calibration.range_min)
            if calibration.drive_mode
            else int(calibration.range_max)
        )

    def _update_assistance(self, action: dict[str, float]) -> None:
        now = time.monotonic()
        try:
            if now - self._last_health_check_time >= 1.0:
                self._check_assist_health()
                self._last_health_check_time = now

            if not self.config.gravity_assist:
                return
            dry_run = self.config.gravity_assist_dry_run
            normalized = np.asarray(
                [float(action[f"{name}.pos"]) for name in ARM_JOINT_NAMES],
                dtype=np.float64,
            )
            enabled = [
                name
                for name in ARM_JOINT_NAMES
                if name in self.config.gravity_assist_joints
            ]
            if not np.all(np.isfinite(normalized)):
                if not dry_run:
                    self._write_arm_currents({name: 0 for name in enabled})
                self._last_assist_q = None
                self._last_assist_time = None
                self._assist_velocity[:] = 0.0
                return

            # URDF-frame joint coordinates: q_urdf = sign * q_yam + offset.
            # The torque the model returns is therefore also URDF-frame and
            # must be mapped back through the same signs for the motors.
            q = gello_joint_positions(
                normalized,
                self.config.gravity_joint_ranges_rad,
                self.config.gravity_joint_signs,
                self.config.gravity_joint_offsets_rad,
            )
            # Low-passed velocity from well-spaced samples only: with 4096
            # ticks/rev, one tick of read noise over a sub-millisecond dt
            # would otherwise command large alternating damping currents.
            if self._last_assist_q is None or self._last_assist_time is None:
                self._last_assist_q = q
                self._last_assist_time = now
            else:
                dt = now - self._last_assist_time
                if dt > 0.25:
                    self._assist_velocity[:] = 0.0
                    self._last_assist_q = q
                    self._last_assist_time = now
                elif dt >= 0.005:
                    raw_velocity = np.clip((q - self._last_assist_q) / dt, -5.0, 5.0)
                    self._assist_velocity = (
                        0.7 * self._assist_velocity + 0.3 * raw_velocity
                    )
                    self._last_assist_q = q
                    self._last_assist_time = now
                # dt < 5 ms: keep the previous sample; too noisy to difference.

            if self._gravity_model is None:
                raise RuntimeError("gravity model was not initialized")
            torque_urdf = (
                self.config.gravity_assist_gain
                * self._gravity_model.gravity_torques(q)
                - self.config.gravity_assist_damping_nm_per_rad_s
                * self._assist_velocity
            )
            currents: dict[str, int] = {}
            for index, name in enumerate(ARM_JOINT_NAMES):
                if name not in enabled:
                    continue
                motor_torque = float(torque_urdf[index]) * int(
                    self.config.gravity_joint_signs[index]
                )
                currents[name] = torque_to_current_ma(
                    motor_torque,
                    self.bus.motors[name].model,
                    self.config.gravity_assist_current_limit_ma,
                )
            if dry_run:
                if now - self._last_dry_run_log_time >= 1.0:
                    self._last_dry_run_log_time = now
                    logger.info(
                        "GELLO gravity DRY RUN %s: q_urdf=%s tau_urdf=%s currents_ma=%s",
                        self.config.port,
                        [round(float(v), 3) for v in q],
                        [round(float(v), 4) for v in torque_urdf],
                        {name: currents[name] for name in enabled},
                    )
                return
            self._write_arm_currents(currents)
        except Exception as exc:
            self._latch_assist_fault(exc)

    def _write_arm_currents(self, currents: dict[str, int]) -> None:
        self.bus.sync_write("Goal_Current", currents, normalize=False)

    def _check_assist_health(self) -> None:
        motors = list(ARM_JOINT_NAMES)
        if self.config.gripper_return:
            motors.append("gripper")
        temperatures = self.bus.sync_read(
            "Present_Temperature", motors, normalize=False
        )
        hot = {
            name: int(value)
            for name, value in temperatures.items()
            if int(value) >= self.config.assist_temperature_limit_c
        }
        if hot:
            raise RuntimeError(
                f"GELLO motor temperature reached safety limit "
                f"{self.config.assist_temperature_limit_c}C: {hot}"
            )
        errors = self.bus.sync_read("Hardware_Error_Status", motors, normalize=False)
        failed = {name: int(value) for name, value in errors.items() if int(value)}
        if failed:
            raise RuntimeError(f"GELLO hardware error status: {failed}")
        self._rearm_tripped_watchdogs()

    def _rearm_tripped_watchdogs(self) -> None:
        """Recover motors whose bus watchdog fired during a stall.

        Any pause in the teleop loop longer than the watchdog window (a
        websocket reconnect, a calibration prompt) trips the DYNAMIXEL bus
        watchdog. A tripped motor stops output and makes its Goal registers
        read-only, and our fast Goal_Current sync writes do not read status
        packets — so without this, assistance silently stays dead for the
        rest of the session.
        """
        if not self._live_assist_motors:
            return
        statuses = self.bus.sync_read(
            "Bus_Watchdog", self._live_assist_motors, normalize=False
        )
        # A tripped watchdog reads -1, i.e. 255 in the unsigned byte register.
        tripped = [name for name, value in statuses.items() if int(value) > 127]
        if not tripped:
            return
        logger.warning(
            "GELLO bus watchdog tripped on %s (loop stalled); re-arming assistance",
            tripped,
        )
        watchdog_raw = self._watchdog_raw()
        for name in tripped:
            # Writing 0 clears the error state; then re-arm the window.
            self.bus.write("Bus_Watchdog", name, 0, normalize=False)
            self.bus.write("Bus_Watchdog", name, watchdog_raw, normalize=False)
        if "gripper" in tripped and self.config.gripper_return:
            # The trip cleared the gripper's goal output; restore the spring.
            self.bus.write(
                "Goal_Current",
                "gripper",
                self.config.gripper_return_current_ma,
                normalize=False,
            )
            self.bus.write(
                "Goal_Position", "gripper", self._gripper_open_tick(), normalize=False
            )

    def _latch_assist_fault(self, exc: Exception) -> None:
        if self._assist_faulted:
            return
        self._assist_faulted = True
        logger.exception(
            "GELLO active assistance faulted; disabling torque and continuing "
            "in passive read-only mode: %s",
            exc,
        )
        self._safe_disable_assistance()

    def _safe_disable_assistance(self) -> None:
        if not self.is_connected:
            self._assist_enabled = False
            return
        try:
            if self.config.gravity_assist:
                self._write_arm_currents({name: 0 for name in ARM_JOINT_NAMES})
            if self.config.gripper_return:
                self.bus.write(
                    "Goal_Current", "gripper", 0, normalize=False, num_retry=1
                )
        except Exception:
            logger.warning("failed to write zero GELLO assistance current", exc_info=True)
        try:
            self.bus.disable_torque(num_retry=1)
        except Exception:
            logger.warning("failed to disable GELLO torque", exc_info=True)
        else:
            try:
                for motor in self.bus.motors:
                    self.bus.write("Bus_Watchdog", motor, 0, normalize=False)
                    self.bus.write(
                        "Operating_Mode",
                        motor,
                        OperatingMode.POSITION.value,
                        normalize=False,
                    )
            except Exception:
                logger.warning(
                    "failed to restore passive GELLO motor mode", exc_info=True
                )
        self._assist_enabled = False
        self._live_assist_motors = []

    # =========================================================================
    # Range-safety helpers
    # =========================================================================

    def _read_raw_ticks(self) -> dict[str, int]:
        """Read ticks and choose the 4096-wrap nearest each calibration range.

        XL330 position/current mode transitions can expose the same physical
        position as e.g. ``-6`` or ``4090``. Calibration is one joint circle,
        so keeping the nearest equivalent prevents a mode switch from creating
        a false out-of-range event.
        """
        last_exc = None
        for _ in range(max(1, self.config.read_retries)):
            try:
                raw = self.bus.sync_read("Present_Position", normalize=False)
                if not self.calibration:
                    return {name: int(value) for name, value in raw.items()}
                return {
                    name: self._tick_near_calibration(name, int(value))
                    for name, value in raw.items()
                }
            except Exception as exc:
                last_exc = exc
                time.sleep(self.config.read_retry_sleep_s)
        raise last_exc

    def _tick_near_calibration(self, motor_name: str, ticks: int) -> int:
        calibration = self.calibration[motor_name]
        base = ticks % 4096
        candidates = (base - 4096, base, base + 4096)

        def distance_to_range(value: int) -> int:
            if value < calibration.range_min:
                return calibration.range_min - value
            if value > calibration.range_max:
                return value - calibration.range_max
            return 0

        return min(
            candidates,
            key=lambda value: (
                distance_to_range(value),
                abs(value - (calibration.range_min + calibration.range_max) / 2.0),
            ),
        )

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

    def _normalization_window(self, motor_name: str) -> tuple[float, float]:
        """The tick window normalization maps onto [-100, 100].

        For arm joints this is the swept calibration window padded by the
        follower's ±0.15 rad joint-limit buffer (converted to ticks). The YAM
        normalizes over its *buffered* limits, so an unpadded leader endpoint
        commands an unreachable target and the joint gets a dead zone at each
        end of travel. Range/preflight checks still use the raw swept window.
        """
        cal = self.calibration[motor_name]
        lo, hi = float(cal.range_min), float(cal.range_max)
        buffer_rad = float(self.config.follower_limit_buffer_rad)
        if motor_name not in ARM_JOINT_NAMES or buffer_rad <= 0.0 or hi <= lo:
            return lo, hi
        index = ARM_JOINT_NAMES.index(motor_name)
        range_lo, range_hi = self.config.gravity_joint_ranges_rad[index]
        true_span_rad = (float(range_hi) - float(range_lo)) - 2.0 * buffer_rad
        if true_span_rad <= 0.0:
            return lo, hi
        ticks_per_rad = (hi - lo) / true_span_rad
        pad = ticks_per_rad * buffer_rad
        return lo - pad, hi + pad

    def _normalize_ticks(self, motor_name: str, ticks: int) -> float:
        """Replicate lerobot _normalize() formula manually for a single motor.

        Used so out-of-range joints can be returned as NaN instead of being
        silently clamped to the limits.
        """
        cal = self.calibration[motor_name]
        motor = self.bus.motors[motor_name]
        lo, hi = self._normalization_window(motor_name)
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
        self._safe_disable_assistance()
        self.bus.disconnect()
        logger.info(f"{self} disconnected.")


class YAMLeaderTeleop(YAMLeader):
    """Alias class for Teleoperator config auto-discovery."""

    pass
