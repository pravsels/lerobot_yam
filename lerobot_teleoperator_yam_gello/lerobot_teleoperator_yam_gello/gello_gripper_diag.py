"""Dump the gripper's motor state with the return spring enabled.

Two GELLO triggers with identical calibrations and identical position
readouts must behave identically under the same position-mode spring — if one
springs open and the other drives closed, the divergence lives in motor-side
state that normal reads don't show (EEPROM Drive_Mode/Homing_Offset/limits, a
failed mode transition, a latched watchdog or hardware error). This tool
enables the spring exactly like the production assist does and prints the
full relevant control table before and after, plus a live line, so the two
arms can be diffed directly:

    python -m lerobot_teleoperator_yam_gello.gello_gripper_diag \
        --port /dev/tty.usbmodemLEFT  --id yam_gello_left  --enable-torque-output
    python -m lerobot_teleoperator_yam_gello.gello_gripper_diag \
        --port /dev/tty.usbmodemRIGHT --id yam_gello_right --enable-torque-output

Squeeze and release the trigger during the live phase. Everything is
torque-off on exit; currents are capped by --current-ma (default 80).
"""

from __future__ import annotations

import argparse
import time

from .config_yam_leader import YAMLeaderTeleopConfig
from .yam_leader import YAMLeader

_REGISTERS = (
    "Operating_Mode",
    "Drive_Mode",
    "Torque_Enable",
    "Homing_Offset",
    "Min_Position_Limit",
    "Max_Position_Limit",
    "Current_Limit",
    "Shutdown",
    "Hardware_Error_Status",
    "Bus_Watchdog",
    "Goal_Position",
    "Goal_Current",
    "Present_Position",
    "Present_Current",
    "Present_Temperature",
)


def _dump(bus, label: str) -> None:
    print(f"\n--- gripper control table [{label}] ---")
    for register in _REGISTERS:
        try:
            value = bus.read(register, "gripper", normalize=False)
        except Exception as exc:  # noqa: BLE001 - keep dumping the rest
            value = f"<read failed: {exc}>"
        print(f"  {register:24s} {value}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--port", required=True)
    parser.add_argument("--id", required=True)
    parser.add_argument("--calibration-dir", default=None)
    parser.add_argument("--current-ma", type=int, default=80)
    parser.add_argument("--live-s", type=float, default=20.0)
    parser.add_argument("--enable-torque-output", action="store_true")
    args = parser.parse_args(argv)
    if not args.enable_torque_output:
        raise SystemExit("Refusing to energize the gripper without --enable-torque-output")

    config = YAMLeaderTeleopConfig(
        port=args.port,
        id=args.id,
        calibration_dir=args.calibration_dir,
        preflight_range_check=False,
        gravity_assist=False,
        gripper_return=True,
        gripper_return_current_ma=args.current_ma,
    )
    leader = YAMLeader(config)
    leader.connect(calibrate=False)  # enables the spring exactly as production
    if not leader.is_calibrated:
        leader.disconnect()
        raise SystemExit(f"no calibration found for id {args.id!r}")
    bus = leader.bus
    calibration = leader.calibration["gripper"]
    print(
        f"calibration: id={calibration.id} drive_mode={calibration.drive_mode} "
        f"homing_offset={calibration.homing_offset} "
        f"range=[{calibration.range_min}, {calibration.range_max}]"
    )
    print(f"spring target (open tick): {leader._gripper_open_tick()}")

    try:
        _dump(bus, "spring enabled")
        print(
            f"\nlive for {args.live_s:.0f}s — squeeze and release the trigger; "
            "watch whether Present_Position moves toward Goal_Position when released:"
        )
        deadline = time.monotonic() + args.live_s
        while time.monotonic() < deadline:
            position = bus.read("Present_Position", "gripper", normalize=False)
            current = bus.read("Present_Current", "gripper", normalize=False)
            goal = bus.read("Goal_Position", "gripper", normalize=False)
            mode = bus.read("Operating_Mode", "gripper")
            print(
                f"\r\x1b[2K  mode={mode} pos={int(position):5d} goal={int(goal):5d} "
                f"I={int(current):+5d} mA",
                end="",
                flush=True,
            )
            time.sleep(0.2)
        print()
        _dump(bus, "after live phase")
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        leader.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
