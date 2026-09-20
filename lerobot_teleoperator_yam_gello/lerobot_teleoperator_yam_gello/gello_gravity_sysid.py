"""Measure a GELLO joint's real gravity curve and fit assist corrections.

Standalone bench tool — talks only to the leader's serial port, no armnet jobs.
Methodology adapted from `experimental/so101_pwm_control/calibrate_torque_per_duty.py`
(`feat/pwm-control-so101`): the operator arranges the arm at several poses; at
each pose every joint except the one under test is locked stiff
(current-capped position mode) and the script bisects the two breakaway edges
of the test joint's holding-current interval, never de-energizing between
probes. Edge midpoints are balance currents; a sine fit over poses is compared
with the gravity model's prediction to produce:

  - a `--teleop.gravity-joint-offsets` delta for the joint (phase gap), and
  - a distal mass scale (amplitude ratio),

plus a JSON diagnostics dump with every probe, for offline analysis.

Usage (from the lerobot_yam checkout, armnet venv or any env with the plugin):

    python -m lerobot_teleoperator_yam_gello.gello_gravity_sysid \
        --port /dev/cu.usbserial-XXXX --id yam_gello_left \
        --joint elbow_flex --poses 5 --enable-torque-output

Safety:
  - all currents are hard-capped (defaults 250 mA, further clamped by each
    motor's EEPROM Current_Limit) and zeroed on exit or Ctrl+C;
  - the test joint IS expected to move — keep the workspace clear and a hand
    near the arm;
  - the run aborts if any motor reaches the temperature limit or if a locked
    joint moves during a probe window (measurement would not be quasistatic);
  - there is no bus watchdog in this tool: if the process is killed
    ungracefully (SIGKILL / USB yank), power-cycle the leader to be sure.
"""

from __future__ import annotations

import argparse
import json
import select
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from lerobot.motors.dynamixel import OperatingMode

from .config_yam_leader import ARM_JOINT_NAMES, YAMLeaderTeleopConfig
from .gravity_assist import (
    GelloGravityModel,
    gello_joint_positions,
    torque_to_current_ma,
)
from .gravity_sysid_fit import (
    apply_offset_delta,
    balance_and_friction,
    fit_sine,
    suggest_corrections,
)
from .yam_leader import YAMLeader

MOTION_THRESHOLD_TICKS = 4.0  # above ~1 tick encoder noise, still a small move
_UNCLIPPED_MA = 1_000_000


@dataclass
class Probe:
    current_ma: int
    delta_ticks: float
    motion: int  # -1 falls, 0 holds, +1 drives
    gravity_model_nm: float


@dataclass
class PoseRecord:
    pose_index: int
    ticks: dict[str, int]
    q_yam: list[float]
    q_urdf: list[float]
    model_torque_nm: float
    model_balance_ma: float
    edge_low_ma: float | None = None
    edge_high_ma: float | None = None
    balance_ma: float | None = None
    friction_ma: float | None = None
    test_motor_temperature_c: int | None = None
    travel_exceeded: bool = False
    quasistatic_violation: str | None = None
    probes: list[Probe] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        return (
            self.balance_ma is not None
            and not self.travel_exceeded
            and self.quasistatic_violation is None
        )


class SysidSession:
    def __init__(self, leader: YAMLeader, model: GelloGravityModel, args) -> None:
        self.leader = leader
        self.bus = leader.bus
        self.model = model
        self.args = args
        self.joint = args.joint
        self.joint_index = ARM_JOINT_NAMES.index(args.joint)
        self.sign = int(leader.config.gravity_joint_signs[self.joint_index])
        self.locked = [name for name in ARM_JOINT_NAMES if name != self.joint]
        self.records: list[PoseRecord] = []
        # Effective caps: never exceed what the motor's EEPROM allows.
        self.test_cap = self._effective_cap(self.joint, args.test_current_limit_ma)
        self.lock_caps = {
            name: self._effective_cap(name, args.lock_current_ma) for name in self.locked
        }

    # -- hardware helpers ----------------------------------------------------

    def _effective_cap(self, motor: str, requested: int) -> int:
        hardware = int(self.bus.read("Current_Limit", motor))
        cap = min(int(requested), hardware)
        if cap < requested:
            print(
                f"  note: {motor} EEPROM Current_Limit={hardware} mA clamps the "
                f"requested {requested} mA"
            )
        return cap

    def arm_state(self) -> tuple[dict[str, int], np.ndarray, np.ndarray]:
        ticks = self.leader._read_raw_ticks()
        normalized = [
            self.leader._normalize_ticks(name, ticks[name]) for name in ARM_JOINT_NAMES
        ]
        cfg = self.leader.config
        q_yam = gello_joint_positions(
            normalized, cfg.gravity_joint_ranges_rad, (1,) * 6, (0.0,) * 6
        )
        q_urdf = gello_joint_positions(
            normalized,
            cfg.gravity_joint_ranges_rad,
            cfg.gravity_joint_signs,
            cfg.gravity_joint_offsets_rad,
        )
        return {name: int(ticks[name]) for name in ARM_JOINT_NAMES}, q_yam, q_urdf

    def model_balance_ma(self, q_urdf: np.ndarray) -> tuple[float, float]:
        torque_urdf = float(self.model.gravity_torques(q_urdf)[self.joint_index])
        motor_torque = torque_urdf * self.sign
        current = torque_to_current_ma(
            motor_torque, self.bus.motors[self.joint].model, _UNCLIPPED_MA
        )
        return torque_urdf, float(current)

    def lock_others(self, ticks: dict[str, int]) -> None:
        """Others: current-capped position hold. Test joint: current mode at 0."""
        self.bus.disable_torque(list(ARM_JOINT_NAMES))
        for name in self.locked:
            self.bus.write("Operating_Mode", name, OperatingMode.CURRENT_POSITION.value)
        self.bus.write("Operating_Mode", self.joint, OperatingMode.CURRENT.value)
        # Goal registers reset on torque-enable, so enable first, then write goals.
        self.bus.enable_torque(list(ARM_JOINT_NAMES))
        for name in self.locked:
            self.bus.write("Goal_Current", name, self.lock_caps[name], normalize=False)
            self.bus.write("Goal_Position", name, ticks[name], normalize=False)
        self.bus.write("Goal_Current", self.joint, 0, normalize=False)

    def release_all(self) -> None:
        try:
            for name in ARM_JOINT_NAMES:
                self.bus.write("Goal_Current", name, 0, normalize=False, num_retry=2)
        finally:
            self.leader.configure()  # torque off, POSITION mode everywhere

    def check_temperature(self) -> int:
        temperature = int(self.bus.read("Present_Temperature", self.joint))
        if temperature >= self.args.temperature_limit_c:
            raise RuntimeError(
                f"{self.joint} reached {temperature}C "
                f"(limit {self.args.temperature_limit_c}C); aborting"
            )
        return temperature

    # -- probing -------------------------------------------------------------

    def probe(self, record: PoseRecord, current_ma: float) -> Probe:
        current = int(round(np.clip(current_ma, -self.test_cap, self.test_cap)))
        self.bus.write("Goal_Current", self.joint, current, normalize=False)
        time.sleep(self.args.probe_settle_s)
        before = self.leader._read_raw_ticks()
        _, _, q_urdf = self.arm_state()
        gravity_nm = float(self.model.gravity_torques(q_urdf)[self.joint_index])
        time.sleep(self.args.dwell_s)
        after = self.leader._read_raw_ticks()

        delta = float(after[self.joint] - before[self.joint])
        motion = 0 if abs(delta) < MOTION_THRESHOLD_TICKS else (1 if delta > 0 else -1)
        probe = Probe(current, delta, motion, gravity_nm)
        record.probes.append(probe)

        # Quasistatic check: locked joints must not move within one window.
        drifters = [
            f"{name} {abs(after[name] - before[name])} ticks"
            for name in self.locked
            if abs(after[name] - before[name]) > self.args.held_drift_limit_ticks
        ]
        if drifters:
            record.quasistatic_violation = ", ".join(drifters)
            print(f"    WARNING: locked joint moved ({record.quasistatic_violation})")

        state = {-1: "falls -ticks", 0: "HOLDS", 1: "drives +ticks"}[motion]
        print(
            f"    I={current:+5d} mA  g_model={gravity_nm:+.4f} Nm  "
            f"delta={delta:+5.0f} ticks -> {state}"
        )
        return probe

    def travel_exceeded(self, record: PoseRecord) -> bool:
        origin = record.ticks[self.joint]
        now = self.leader._read_raw_ticks()[self.joint]
        exceeded = abs(now - origin) > self.args.max_travel_ticks
        if exceeded and not record.travel_exceeded:
            record.travel_exceeded = True
            print(
                f"    joint drifted {abs(now - origin)} ticks from the pose; "
                "stopping this measurement"
            )
        return exceeded

    def bracket_edges(self, record: PoseRecord) -> None:
        """Find currents producing both motion directions, then bisect both edges."""
        center = float(np.clip(record.model_balance_ma,
                               -self.args.initial_center_cap_ma,
                               self.args.initial_center_cap_ma))
        self.probe(record, center)
        radius = 24.0
        while not self._has_both_motions(record) and not self.travel_exceeded(record):
            for offset in (radius, -radius):
                self.probe(record, center + offset)
                if self._has_both_motions(record) or self.travel_exceeded(record):
                    break
            if radius >= 2 * self.test_cap:
                break
            radius = min(radius * 2.0, 2.0 * float(self.test_cap))

        motions = {p.motion for p in record.probes}
        if -1 not in motions or 1 not in motions:
            print(
                "    could not bracket both edges within the current cap "
                f"({self.test_cap} mA) — joint too heavy for the motor here, or "
                "already saturated; recorded probes kept for diagnostics"
            )
            return

        low = self._bisect(record, moving=-1)
        high = self._bisect(record, moving=+1)
        if low is None or high is None or record.travel_exceeded:
            return
        record.edge_low_ma, record.edge_high_ma = float(low), float(high)
        if record.edge_high_ma < record.edge_low_ma:
            record.edge_low_ma, record.edge_high_ma = record.edge_high_ma, record.edge_low_ma
        record.balance_ma, record.friction_ma = balance_and_friction(
            record.edge_low_ma, record.edge_high_ma
        )
        print(
            f"    edges [{record.edge_low_ma:+.0f}, {record.edge_high_ma:+.0f}] mA -> "
            f"balance {record.balance_ma:+.0f} mA, friction ±{record.friction_ma:.0f} mA "
            f"(model said {record.model_balance_ma:+.0f} mA)"
        )

    def _has_both_motions(self, record: PoseRecord) -> bool:
        motions = {p.motion for p in record.probes}
        return -1 in motions and 1 in motions

    def _bisect(self, record: PoseRecord, moving: int) -> float | None:
        """Boundary between `moving` and holding; returns the holding-side current."""
        movers = [p for p in record.probes if p.motion == moving]
        stills = [p for p in record.probes if p.motion != moving]
        if not movers or not stills:
            return None
        if moving < 0:
            low = max(p.current_ma for p in movers)
            candidates = [p for p in stills if p.current_ma > low]
            if not candidates:
                return None
            edge = min(candidates, key=lambda p: p.current_ma)
            high = edge.current_ma
        else:
            high = min(p.current_ma for p in movers)
            candidates = [p for p in stills if p.current_ma < high]
            if not candidates:
                return None
            edge = max(candidates, key=lambda p: p.current_ma)
            low = edge.current_ma

        while high - low > self.args.current_tolerance_ma:
            if self.travel_exceeded(record):
                break
            middle = 0.5 * (low + high)
            probe = self.probe(record, middle)
            if probe.motion == moving:
                if moving < 0:
                    low = probe.current_ma
                else:
                    high = probe.current_ma
            else:
                edge = probe
                if moving < 0:
                    high = probe.current_ma
                else:
                    low = probe.current_ma
        return float(edge.current_ma)

    # -- session -------------------------------------------------------------

    def capture_pose(self, pose_index: int) -> PoseRecord:
        print(
            f"\nPose {pose_index + 1}/{self.args.poses}: move ONLY the "
            f"{self.joint} joint to a new angle (other joints are locked), "
            "then press ENTER."
        )
        self._wait_for_enter_with_live_display()
        ticks, q_yam, q_urdf = self.arm_state()
        torque_nm, balance_ma = self.model_balance_ma(q_urdf)
        record = PoseRecord(
            pose_index=pose_index,
            ticks=ticks,
            q_yam=[round(float(v), 5) for v in q_yam],
            q_urdf=[round(float(v), 5) for v in q_urdf],
            model_torque_nm=round(torque_nm, 5),
            model_balance_ma=round(balance_ma, 1),
        )
        record.test_motor_temperature_c = self.check_temperature()
        print(
            f"  captured {self.joint}: q_urdf={record.q_urdf[self.joint_index]:+.3f} rad, "
            f"model balance {record.model_balance_ma:+.0f} mA, "
            f"temp {record.test_motor_temperature_c}C"
        )
        self.bracket_edges(record)
        # Leave the joint free (0 mA) for the operator to reposition.
        self.bus.write("Goal_Current", self.joint, 0, normalize=False)
        self.records.append(record)
        return record

    def _wait_for_enter_with_live_display(self) -> None:
        while True:
            _, _, q_urdf = self.arm_state()
            _, balance = self.model_balance_ma(q_urdf)
            spread = ""
            usable = [r for r in self.records if r.usable]
            if usable:
                angles = [r.q_urdf[self.joint_index] for r in usable] + [
                    float(q_urdf[self.joint_index])
                ]
                spread = f"  angle spread so far {max(angles) - min(angles):+.2f} rad"
            sys.stdout.write(
                f"\r\x1b[2K  {self.joint} q_urdf={q_urdf[self.joint_index]:+.3f} rad  "
                f"model balance {balance:+.0f} mA{spread}  [ENTER to measure]"
            )
            sys.stdout.flush()
            readable, _, _ = select.select([sys.stdin], [], [], 0.15)
            if readable:
                sys.stdin.readline()
                print()
                return

    def report(self) -> dict:
        usable = [r for r in self.records if r.usable]
        payload: dict = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "joint": self.joint,
            "port": self.leader.config.port,
            "leader_id": self.leader.config.id,
            "config": {
                "signs": list(self.leader.config.gravity_joint_signs),
                "offsets_rad": list(self.leader.config.gravity_joint_offsets_rad),
                "masses_kg": list(self.leader.config.gravity_link_masses_kg),
                "z_sign": self.leader.config.gravity_urdf_z_sign,
                "ranges_rad": [list(p) for p in self.leader.config.gravity_joint_ranges_rad],
                "test_cap_ma": self.test_cap,
                "lock_caps_ma": self.lock_caps,
            },
            "model_link_masses_kg": self.model.link_masses,
            "poses": [asdict(r) for r in self.records],
            "usable_poses": len(usable),
        }
        print(f"\n{'=' * 60}\n{self.joint}: {len(usable)} usable poses of {len(self.records)}")
        for r in self.records:
            status = "ok" if r.usable else "UNUSABLE"
            balance = f"{r.balance_ma:+.0f}" if r.balance_ma is not None else "  —"
            friction = f"±{r.friction_ma:.0f}" if r.friction_ma is not None else ""
            print(
                f"  pose {r.pose_index}: q={r.q_urdf[self.joint_index]:+.3f} rad  "
                f"measured {balance} {friction} mA  model {r.model_balance_ma:+.0f} mA  "
                f"[{status}]"
            )
        if len(usable) < 3:
            print("\nNeed >= 3 usable poses for a fit. Collect more (spread the angles).")
            return payload

        theta = [r.q_urdf[self.joint_index] for r in usable]
        measured_fit = fit_sine(theta, [r.balance_ma for r in usable])
        model_fit = fit_sine(theta, [r.model_balance_ma for r in usable])
        corrections = suggest_corrections(measured_fit, model_fit)
        payload["measured_fit"] = asdict(measured_fit)
        payload["model_fit"] = asdict(model_fit)
        payload["corrections"] = asdict(corrections)

        print(f"\nmeasured: {measured_fit.describe()}")
        print(f"model:    {model_fit.describe()}")
        mean_friction = float(np.mean([r.friction_ma for r in usable]))
        print(f"friction: ±{mean_friction:.0f} mA mean stiction half-width")
        for warning in corrections.warnings:
            print(f"WARNING: {warning}")

        new_offsets = apply_offset_delta(
            self.leader.config.gravity_joint_offsets_rad,
            self.joint_index,
            corrections.offset_delta_rad,
        )
        payload["suggested_offsets_rad"] = list(new_offsets)
        print(
            f"\nsuggested offset delta for {self.joint}: "
            f"{corrections.offset_delta_rad:+.3f} rad"
        )
        print(
            "suggested --teleop.gravity-joint-offsets="
            + ",".join(f"{v:.4f}" for v in new_offsets)
        )
        if np.isfinite(corrections.amplitude_ratio):
            print(
                f"suggested distal mass scale (links after {self.joint}): "
                f"x{corrections.amplitude_ratio:.2f}"
            )
        return payload


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if not args.enable_torque_output:
        raise SystemExit("Refusing to energize motors without --enable-torque-output")
    if not sys.stdin.isatty():
        raise SystemExit("pose capture needs an interactive terminal")

    config = YAMLeaderTeleopConfig(
        port=args.port,
        id=args.id,
        calibration_dir=args.calibration_dir,
        preflight_range_check=False,
        gravity_joint_signs=args.signs,
        gravity_joint_offsets_rad=args.offsets,
        gravity_link_masses_kg=args.masses,
        gravity_urdf_z_sign=args.z_sign,
    )
    leader = YAMLeader(config)
    leader.connect(calibrate=False)
    if not leader.is_calibrated:
        leader.disconnect()
        raise SystemExit(
            f"no calibration found for id {args.id!r}; run lerobot-calibrate first"
        )

    model = GelloGravityModel(
        gravity_z_sign=args.z_sign, link_masses_kg=args.masses or None
    )
    session = SysidSession(leader, model, args)
    print(
        f"\nGELLO gravity sysid on {args.joint} ({args.port}); test cap "
        f"{session.test_cap} mA, locks {args.lock_current_ma} mA. The test joint "
        "WILL move — keep the workspace clear.\n"
        "Arrange the WHOLE arm now (all joints limp), then press ENTER to lock."
    )
    try:
        input()
        ticks, _, _ = session.arm_state()
        session.lock_others(ticks)
        for pose_index in range(args.poses):
            session.capture_pose(pose_index)
    except KeyboardInterrupt:
        print("\ninterrupted; releasing")
    finally:
        session.release_all()
        payload = session.report()
        out = args.out or Path(
            f"gello_sysid_{args.joint}_{datetime.now():%Y%m%d_%H%M%S}.json"
        )
        Path(out).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"\ndiagnostics written to {out} — send this file back for analysis")
        leader.disconnect()
    return 0


def _csv(cast, count, label):
    def parse(value: str):
        if not value.strip():
            return ()
        parts = tuple(cast(p.strip()) for p in value.split(","))
        if len(parts) != count:
            raise argparse.ArgumentTypeError(f"{label} needs {count} comma-separated values")
        return parts

    return parse


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--port", required=True)
    parser.add_argument("--id", required=True, help="LeRobot calibration id, e.g. yam_gello_left")
    parser.add_argument("--calibration-dir", default=None)
    parser.add_argument("--joint", required=True, choices=ARM_JOINT_NAMES)
    parser.add_argument("--poses", type=int, default=4)
    parser.add_argument("--signs", type=_csv(int, 6, "--signs"), default=(1, -1, -1, -1, 1, 1))
    parser.add_argument("--offsets", type=_csv(float, 6, "--offsets"), default=(0.0,) * 6)
    parser.add_argument("--masses", type=_csv(float, 7, "--masses"), default=())
    parser.add_argument("--z-sign", type=int, choices=(1, -1), default=-1)
    parser.add_argument("--test-current-limit-ma", type=int, default=250)
    parser.add_argument("--lock-current-ma", type=int, default=250)
    parser.add_argument("--current-tolerance-ma", type=float, default=4.0)
    parser.add_argument("--initial-center-cap-ma", type=float, default=150.0)
    parser.add_argument("--dwell-s", type=float, default=0.3)
    parser.add_argument("--probe-settle-s", type=float, default=0.05)
    parser.add_argument("--max-travel-ticks", type=float, default=400.0)
    parser.add_argument("--held-drift-limit-ticks", type=float, default=20.0)
    parser.add_argument("--temperature-limit-c", type=int, default=50)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--enable-torque-output", action="store_true", help="required safety acknowledgement"
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
