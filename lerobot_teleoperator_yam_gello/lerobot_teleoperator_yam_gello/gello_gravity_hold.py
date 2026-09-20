"""Bench-run the GELLO gravity assist per joint, with a gain — no armnet job.

Drives the *production* assist code path (`YAMLeader` with `gravity_assist`)
in a standalone loop so each joint can be validated on the bench before a
teleop session. Non-tested arm joints can be locked stiff so a single joint's
behavior is isolated, exactly like the sysid tool.

Example — test only the elbow at gain 0.5 with everything else locked:

    python -m lerobot_teleoperator_yam_gello.gello_gravity_hold \
        --port /dev/tty.usbmodemXXXX --id yam_gello_left \
        --joints elbow_flex --gain 0.5 \
        --masses 0.142,0.009,0.012,0.006,0.004,0.002,0.010 \
        --offsets 0,0,-1.9309,0,0,0 \
        --enable-torque-output

The status line prints the commanded model current alongside the motor's
*measured* `Present_Current` and temperature — if those disagree while the
joint "holds", the motor is being helped by something external (or its
protection kicked in).

Safety: the same latches as live teleop assist apply (current caps, 200 ms
bus watchdog, 50 C temperature cutoff, zero-current + torque-off on exit).
Start with one joint, hold the arm, and keep the workspace clear.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from lerobot.motors.dynamixel import OperatingMode

from .config_yam_leader import ARM_JOINT_NAMES, YAMLeaderTeleopConfig
from .gravity_assist import GelloGravityModel, gello_joint_positions, torque_to_current_ma
from .gravity_sysid_fit import wrap_angle
from .yam_leader import YAMLeader

_UNCLIPPED_MA = 1_000_000


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if not args.dry_run and not args.enable_torque_output:
        raise SystemExit("Refusing to energize motors without --enable-torque-output")

    joints = ARM_JOINT_NAMES if args.joints == ("all",) else args.joints
    config = YAMLeaderTeleopConfig(
        port=args.port,
        id=args.id,
        calibration_dir=args.calibration_dir,
        preflight_range_check=False,
        gravity_assist=True,
        gravity_assist_gain=args.gain,
        gravity_assist_current_limit_ma=args.current_limit_ma,
        gravity_assist_joints=tuple(joints),
        gravity_assist_dry_run=args.dry_run,
        gravity_joint_signs=args.signs,
        gravity_joint_offsets_rad=args.offsets,
        gravity_link_masses_kg=args.masses,
        gravity_urdf_z_sign=args.z_sign,
    )
    leader = YAMLeader(config)
    leader.connect(calibrate=False)
    if not leader.is_calibrated:
        leader.disconnect()
        raise SystemExit(f"no calibration found for id {args.id!r}")

    model = GelloGravityModel(gravity_z_sign=args.z_sign, link_masses_kg=args.masses or None)
    locked = (
        [name for name in ARM_JOINT_NAMES if name not in joints]
        if args.lock_others and not args.dry_run
        else []
    )
    original_limits: dict[str, int] = {}
    bus = leader.bus
    try:
        if locked:
            ticks = leader._read_raw_ticks()
            bus.disable_torque(locked)
            for name in locked:
                original_limits[name] = int(bus.read("Current_Limit", name))
                if original_limits[name] < args.lock_current_ma:
                    bus.write("Current_Limit", name, args.lock_current_ma, normalize=False)
                bus.write("Operating_Mode", name, OperatingMode.CURRENT_POSITION.value)
            bus.enable_torque(locked)
            for name in locked:
                bus.write("Goal_Current", name, args.lock_current_ma, normalize=False)
                bus.write("Goal_Position", name, int(ticks[name]), normalize=False)
            print(f"locked: {locked} at {args.lock_current_ma} mA")

        mode = "DRY RUN (no torque)" if args.dry_run else f"LIVE gain={args.gain:g}"
        print(
            f"gravity hold test: joints={list(joints)} {mode} "
            f"cap={args.current_limit_ma} mA, {args.duration_s:.0f}s. Ctrl+C to stop."
        )
        deadline = time.monotonic() + args.duration_s
        last_status = 0.0
        while time.monotonic() < deadline:
            loop_start = time.perf_counter()
            action = leader.get_action()  # drives the production assist update
            if leader._assist_faulted:
                print("\nassist faulted (see log above); stopping")
                return 1

            now = time.monotonic()
            if now - last_status >= 1.0:
                last_status = now
                print("\r\x1b[2K" + _status_line(leader, model, joints, action), end="")
                sys.stdout.flush()
            time.sleep(max(0.0, 1.0 / 50.0 - (time.perf_counter() - loop_start)))
        print()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        try:
            for name in locked:
                bus.write("Goal_Current", name, 0, normalize=False, num_retry=2)
            for name, original in original_limits.items():
                if int(bus.read("Current_Limit", name)) != original:
                    bus.write("Current_Limit", name, original, normalize=False)
        except Exception:  # noqa: BLE001 - best effort teardown
            print("warning: lock teardown incomplete; power-cycle the leader if unsure")
        leader.disconnect()  # zeroes assist current, torque off, POSITION mode
    return 0


def _status_line(leader: YAMLeader, model, joints, action) -> str:
    cfg = leader.config
    normalized = [float(action[f"{name}.pos"]) for name in ARM_JOINT_NAMES]
    if not np.all(np.isfinite(normalized)):
        return "out-of-range joint: assist holding at zero"
    q = gello_joint_positions(
        normalized, cfg.gravity_joint_ranges_rad, cfg.gravity_joint_signs,
        cfg.gravity_joint_offsets_rad,
    )
    torque = model.gravity_torques(q)
    measured = leader.bus.sync_read("Present_Current", list(joints), normalize=False)
    temps = leader.bus.sync_read("Present_Temperature", list(joints), normalize=False)
    parts = []
    for name in joints:
        index = ARM_JOINT_NAMES.index(name)
        commanded = torque_to_current_ma(
            float(torque[index]) * int(cfg.gravity_joint_signs[index]) * cfg.gravity_assist_gain,
            leader.bus.motors[name].model,
            cfg.gravity_assist_current_limit_ma,
        )
        parts.append(
            f"{name}: q={wrap_angle(float(q[index])):+.2f} cmd={commanded:+d}mA "
            f"meas={int(measured[name]):+d}mA {int(temps[name])}C"
        )
    return "  |  ".join(parts)


def _csv(cast, count, label):
    def parse(value: str):
        if not value.strip():
            return ()
        parts = tuple(cast(p.strip()) for p in value.split(","))
        if len(parts) != count:
            raise argparse.ArgumentTypeError(f"{label} needs {count} comma-separated values")
        return parts

    return parse


def _joint_list(value: str) -> tuple[str, ...]:
    names = tuple(part.strip() for part in value.split(",") if part.strip())
    if names == ("all",):
        return names
    unknown = [n for n in names if n not in ARM_JOINT_NAMES]
    if not names or unknown:
        raise argparse.ArgumentTypeError(
            f"--joints must be 'all' or a comma list of {list(ARM_JOINT_NAMES)}"
        )
    return names


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--port", required=True)
    parser.add_argument("--id", required=True)
    parser.add_argument("--calibration-dir", default=None)
    parser.add_argument(
        "--joints",
        type=_joint_list,
        default=("all",),
        help="Comma list of joints to assist (default all six)",
    )
    parser.add_argument("--gain", type=float, default=0.1)
    parser.add_argument("--current-limit-ma", type=int, default=250)
    parser.add_argument("--duration-s", type=float, default=120.0)
    parser.add_argument("--signs", type=_csv(int, 6, "--signs"), default=(1, -1, -1, -1, 1, 1))
    parser.add_argument("--offsets", type=_csv(float, 6, "--offsets"), default=(0.0,) * 6)
    parser.add_argument("--masses", type=_csv(float, 7, "--masses"), default=())
    parser.add_argument("--z-sign", type=int, choices=(1, -1), default=-1)
    parser.add_argument(
        "--no-lock-others",
        dest="lock_others",
        action="store_false",
        help="Leave the non-tested joints limp instead of locking them",
    )
    parser.add_argument("--lock-current-ma", type=int, default=600)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log computed currents without applying torque",
    )
    parser.add_argument(
        "--enable-torque-output", action="store_true", help="required safety acknowledgement"
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
