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
    fit_edge_events,
    fit_sine,
    solve_apex_offset,
    suggest_corrections,
    wrap_angle,
)
from .yam_leader import YAMLeader

MOTION_THRESHOLD_TICKS = 4.0  # above ~1 tick encoder noise, still a small move
_UNCLIPPED_MA = 1_000_000


@dataclass
class Probe:
    current_ma: int
    delta_ticks: float
    motion: int  # -1 falls, 0 holds, +1 drives
    theta_urdf: float  # test joint's URDF angle at this probe's start
    gravity_model_nm: float
    model_balance_ma: float


@dataclass
class EdgeEvent:
    """One holding-interval edge at its own angle (the joint drifts between probes)."""

    theta_urdf: float
    current_ma: float
    direction: int  # -1 lower edge (falling boundary), +1 upper edge (driving)
    model_balance_ma: float
    pose_index: int


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
    edge_events: list["EdgeEvent"] = field(default_factory=list)

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
        self.edge_events: list[EdgeEvent] = []
        # The gravity-assist runs write low Current_Limit values into EEPROM
        # (e.g. 200 mA), which would clamp both the locks and the sweep. Raise
        # the limits for this session and restore the originals afterwards.
        # Torque is still off here (leader.connect leaves it off).
        self._original_current_limits: dict[str, int] = {}
        self.test_cap = self._ensure_current_limit(self.joint, args.test_current_limit_ma)
        self.lock_caps = {
            name: self._ensure_current_limit(name, args.lock_current_ma)
            for name in self.locked
        }

    # -- hardware helpers ----------------------------------------------------

    def _ensure_current_limit(self, motor: str, requested: int) -> int:
        requested = min(int(requested), 1000)  # well under the XL330's 1750 max
        existing = int(self.bus.read("Current_Limit", motor))
        self._original_current_limits[motor] = existing
        if existing < requested:
            print(f"  {motor}: raising Current_Limit {existing} -> {requested} mA for this session")
            self.bus.write("Current_Limit", motor, requested, normalize=False)
        return requested

    def restore_current_limits(self) -> None:
        for motor, original in self._original_current_limits.items():
            try:
                if int(self.bus.read("Current_Limit", motor)) != original:
                    self.bus.write("Current_Limit", motor, original, normalize=False)
            except Exception:  # noqa: BLE001 - best effort on teardown
                print(f"  warning: failed to restore Current_Limit on {motor}")

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
        # Wrap for readability (and sane fits/logs): the model is exactly
        # 2*pi-periodic, so -4.1 rad and +2.18 rad are the same pose to it.
        q_urdf = np.asarray([wrap_angle(float(v)) for v in q_urdf])
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
            self.restore_current_limits()

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
        before, _, q_urdf = self.arm_state()
        theta = float(q_urdf[self.joint_index])
        gravity_nm = float(self.model.gravity_torques(q_urdf)[self.joint_index])
        model_ma = float(
            torque_to_current_ma(
                gravity_nm * self.sign, self.bus.motors[self.joint].model, _UNCLIPPED_MA
            )
        )
        time.sleep(self.args.dwell_s)
        after = self.leader._read_raw_ticks()

        delta = float(after[self.joint] - before[self.joint])
        motion = 0 if abs(delta) < MOTION_THRESHOLD_TICKS else (1 if delta > 0 else -1)
        probe = Probe(current, delta, motion, theta, gravity_nm, model_ma)
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
        """Find both motion directions if possible, then bisect each found edge.

        Any single edge is a usable data point (an `EdgeEvent` at its own
        angle), so a pose that only yields one boundary still contributes.
        """
        center = float(np.clip(record.model_balance_ma,
                               -self.args.initial_center_cap_ma,
                               self.args.initial_center_cap_ma))
        self.probe(record, center)
        radius = 24.0
        while not self._has_both_motions(record) and not self.travel_exceeded(record):
            if len(record.probes) >= self.args.max_probes_per_pose:
                break
            # Chase the resisting direction first: a falling joint needs *more*
            # current, and probing the other way first just lets it keep
            # falling until the travel guard kills the pose.
            last_motion = next(
                (p.motion for p in reversed(record.probes) if p.motion != 0), 0
            )
            ordered = (-radius, radius) if last_motion > 0 else (radius, -radius)
            for offset in ordered:
                self.probe(record, center + offset)
                if self._has_both_motions(record) or self.travel_exceeded(record):
                    break
            if radius >= 2 * self.test_cap:
                break
            radius = min(radius * 2.0, 2.0 * float(self.test_cap))

        motions = {p.motion for p in record.probes}
        if -1 not in motions or 1 not in motions:
            print(
                "    only one motion direction seen within the cap "
                f"({self.test_cap} mA); keeping whatever edge is available"
            )

        low = self._bisect(record, moving=-1)
        high = self._bisect(record, moving=+1)
        for edge_probe, direction in ((low, -1), (high, +1)):
            if edge_probe is None:
                continue
            event = EdgeEvent(
                theta_urdf=edge_probe.theta_urdf,
                current_ma=float(edge_probe.current_ma),
                direction=direction,
                model_balance_ma=edge_probe.model_balance_ma,
                pose_index=record.pose_index,
            )
            self.edge_events.append(event)
            record.edge_events.append(event)
        if low is not None and high is not None:
            record.edge_low_ma = float(min(low.current_ma, high.current_ma))
            record.edge_high_ma = float(max(low.current_ma, high.current_ma))
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

    def _bisect(self, record: PoseRecord, moving: int) -> Probe | None:
        """Boundary between `moving` and holding; returns the holding-side probe."""
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
            if self.travel_exceeded(record) or len(record.probes) >= self.args.max_probes_per_pose:
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
        return edge

    # -- session -------------------------------------------------------------

    def capture_pose(self, pose_index: int) -> PoseRecord:
        print(
            f"\nPose {pose_index + 1}/{self.args.poses}: move ONLY the "
            f"{self.joint} joint to a new angle (other joints are locked), "
            "then press ENTER. Keep the arm CLEAR of the table — any contact "
            "invalidates the measurement."
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

    def _contact_suspects(self) -> tuple[set[int], list[str]]:
        """Holds spanning a very wide current range at ~the same angle suggest the
        arm was resting on something (table): friction alone rarely exceeds this.

        Suspect poses are excluded from the fit: a single contaminated pose can
        drag the phase/intercept far enough to produce a confident-looking but
        wrong offset suggestion.
        """
        suspect_poses: set[int] = set()
        messages: list[str] = []
        for record in self.records:
            holds = [p for p in record.probes if p.motion == 0]
            for i, first in enumerate(holds):
                for second in holds[i + 1:]:
                    same_angle = abs(first.theta_urdf - second.theta_urdf) < 0.02
                    if same_angle and abs(first.current_ma - second.current_ma) > 140:
                        suspect_poses.add(record.pose_index)
                        messages.append(
                            f"pose {record.pose_index}: holds at both "
                            f"{first.current_ma:+d} and {second.current_ma:+d} mA at "
                            f"theta {first.theta_urdf:+.3f} — external support (table?) "
                            "or extreme stiction; excluded from the fit"
                        )
                        break
                else:
                    continue
                break
        return suspect_poses, messages

    def report(self) -> dict:
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
                "original_current_limits_ma": self._original_current_limits,
            },
            "model_link_masses_kg": self.model.link_masses,
            "poses": [asdict(r) for r in self.records],
            "edge_events": [asdict(e) for e in self.edge_events],
        }
        print(
            f"\n{'=' * 60}\n{self.joint}: {len(self.edge_events)} edge events "
            f"from {len(self.records)} poses"
        )
        for event in self.edge_events:
            kind = "lower(-)" if event.direction < 0 else "upper(+)"
            print(
                f"  pose {event.pose_index} {kind}: theta={event.theta_urdf:+.3f} rad  "
                f"I={event.current_ma:+.0f} mA  model {event.model_balance_ma:+.0f} mA"
            )
        suspect_poses, contact = self._contact_suspects()
        payload["contact_suspects"] = contact
        for suspect in contact:
            print(f"WARNING: {suspect}")

        clean_events = [
            e for e in self.edge_events if e.pose_index not in suspect_poses
        ]
        if len(clean_events) < len(self.edge_events):
            print(
                f"fitting {len(clean_events)} of {len(self.edge_events)} edge "
                "events (contact-suspect poses excluded)"
            )
        directions = [e.direction for e in clean_events]
        if len(clean_events) < 4 or not (
            any(d > 0 for d in directions) and any(d < 0 for d in directions)
        ):
            print(
                "\nNot enough clean edge events for a fit (need >= 4 with both "
                "directions). Collect more poses."
            )
            return payload

        theta = [e.theta_urdf for e in clean_events]
        edge_fit = fit_edge_events(theta, [e.current_ma for e in clean_events], directions)
        model_fit = fit_sine(theta, [e.model_balance_ma for e in clean_events])
        corrections = suggest_corrections(edge_fit.sine, model_fit)
        payload["measured_fit"] = asdict(edge_fit.sine)
        payload["measured_friction_ma"] = edge_fit.friction_ma
        payload["model_fit"] = asdict(model_fit)
        payload["corrections"] = asdict(corrections)

        print(f"\nmeasured: {edge_fit.sine.describe()}  friction ±{edge_fit.friction_ma:.0f} mA")
        print(f"model:    {model_fit.describe()}")
        for warning in corrections.warnings:
            print(f"WARNING: {warning}")

        if not corrections.reliable:
            payload["suggested_offsets_rad"] = None
            print(
                "\nNO SUGGESTION: the fit is unreliable (see warnings above). "
                "Do not apply these numbers — collect poses spread across the "
                "joint's full range, including past the balance apex, and rerun."
            )
            return payload

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

    range_override = (
        {"gravity_joint_ranges_rad": tuple(zip(args.ranges[::2], args.ranges[1::2]))}
        if args.ranges
        else {}
    )
    config = YAMLeaderTeleopConfig(
        port=args.port,
        id=args.id,
        calibration_dir=args.calibration_dir,
        preflight_range_check=False,
        gravity_joint_signs=args.signs,
        gravity_joint_offsets_rad=args.offsets,
        gravity_link_masses_kg=args.masses,
        gravity_urdf_z_sign=args.z_sign,
        **range_override,
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
    if args.capture_apex:
        return _capture_apex(session, leader, args)
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


def _capture_apex(session: SysidSession, leader: YAMLeader, args) -> int:
    """Solve the joint's offset from a physically balanced over-center pose.

    No fitting and no torque on the test joint: the operator balances the
    joint at its 12 o'clock pose; the offset that puts the model's torque zero
    exactly there (with the unstable-slope zero, so the commanded current
    flips sign on the correct side) is computed analytically.
    """
    print(
        f"\nAPEX CAPTURE for {args.joint}: arrange the whole arm, then balance "
        f"ONLY the {args.joint} segment at its over-center (12 o'clock) pose — "
        "the point where it falls neither way. Other joints will be locked.\n"
        "Arrange the arm now, then press ENTER to lock."
    )
    try:
        input()
        ticks, _, _ = session.arm_state()
        session.lock_others(ticks)
        print("Balance the joint at the apex, then press ENTER.")
        session._wait_for_enter_with_live_display()
        _, q_yam, _ = session.arm_state()
        solution = solve_apex_offset(
            session.model,
            q_yam,
            leader.config.gravity_joint_signs,
            leader.config.gravity_joint_offsets_rad,
            session.joint_index,
        )
    except KeyboardInterrupt:
        print("\ninterrupted; releasing")
        return 1
    finally:
        session.release_all()
        leader.disconnect()

    amp_ma = torque_to_current_ma(
        solution.amplitude_nm, session.bus.motors[args.joint].model, _UNCLIPPED_MA
    )
    new_offsets = list(leader.config.gravity_joint_offsets_rad)
    new_offsets[session.joint_index] = solution.offset_apex_rad
    print(f"\ncaptured q_yam: {[round(float(v), 4) for v in q_yam]}")
    print(
        f"(shoulder_lift was {float(q_yam[1]):+.3f} rad — capture again at a "
        "clearly different lift to check the lift/elbow coupling sign: the two "
        "suggested offsets must agree)"
    )
    print(
        f"apex offset for {args.joint}: {solution.offset_apex_rad:+.4f} rad "
        f"(hanging alternative {solution.offset_hanging_rad:+.4f}; "
        f"gravity amplitude {solution.amplitude_nm:.4f} Nm ≈ {amp_ma} mA)"
    )
    print(
        "suggested --teleop.gravity-joint-offsets="
        + ",".join(f"{v:.4f}" for v in new_offsets)
    )
    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "mode": "capture_apex",
        "joint": args.joint,
        "q_yam": [float(v) for v in q_yam],
        "offset_apex_rad": solution.offset_apex_rad,
        "offset_hanging_rad": solution.offset_hanging_rad,
        "amplitude_nm": solution.amplitude_nm,
        "amplitude_ma": amp_ma,
        "signs": list(leader.config.gravity_joint_signs),
        "prior_offsets_rad": list(leader.config.gravity_joint_offsets_rad),
        "suggested_offsets_rad": [float(v) for v in new_offsets],
    }
    out = args.out or Path(
        f"gello_apex_{args.joint}_{datetime.now():%Y%m%d_%H%M%S}.json"
    )
    Path(out).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"apex capture written to {out}")
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
    parser.add_argument(
        "--ranges",
        type=_csv(float, 12, "--ranges"),
        default=(),
        metavar="LO1,HI1,...,LO6,HI6",
        help="Override the per-joint physical ranges (rad) used to map "
        "normalized values to radians, e.g. when a leader joint travels less "
        "than the YAM's software range. Twelve values, lo/hi per joint.",
    )
    parser.add_argument("--test-current-limit-ma", type=int, default=250)
    parser.add_argument(
        "--lock-current-ma",
        type=int,
        default=600,
        help="Lock strength for the non-test joints; the shoulder needs real "
        "torque to stay put while the elbow is probed (default 600)",
    )
    parser.add_argument("--max-probes-per-pose", type=int, default=30)
    parser.add_argument("--current-tolerance-ma", type=float, default=4.0)
    parser.add_argument("--initial-center-cap-ma", type=float, default=150.0)
    parser.add_argument("--dwell-s", type=float, default=0.3)
    parser.add_argument("--probe-settle-s", type=float, default=0.05)
    parser.add_argument("--max-travel-ticks", type=float, default=400.0)
    parser.add_argument("--held-drift-limit-ticks", type=float, default=20.0)
    parser.add_argument("--temperature-limit-c", type=int, default=50)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--capture-apex",
        action="store_true",
        help="Instead of probing currents, capture the joint's physical "
        "over-center pose and solve its offset analytically (test joint stays "
        "passive; other joints are locked)",
    )
    parser.add_argument(
        "--enable-torque-output", action="store_true", help="required safety acknowledgement"
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
