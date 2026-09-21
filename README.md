# LeRobot YAM Plugins

YAM follower arm + GELLO leader plugins for LeRobot.

## Quickstart

Requirements: CAN interface (e.g. `can0`) and a GELLO leader (port configurable).
GELLO assembly information from [gello_mechanical](https://github.com/wuphilipp/gello_mechanical/).

```bash
uv sync
uv run lerobot-teleoperate \
  --robot.type=yam_follower \
  --robot.port=can0 \
  --robot.gripper_type=crank_4310 \
  --teleop.type=yam_leader \
  --teleop.port=/dev/ttyUSB0
```

Using an existing leader calibration file (for example `yam_lerobot.json` in the
repo root):
```bash
uv run lerobot-teleoperate \
  --robot.type=yam_follower \
  --robot.port=can0 \
  --robot.gripper_type=crank_4310 \
  --teleop.type=yam_leader \
  --teleop.port=/dev/ttyUSB0 \
  --teleop.calibration_dir=./ \
  --teleop.id=yam_lerobot
```
`--teleop.id` is required when loading an existing calibration file, because
LeRobot resolves calibration as `<teleop.calibration_dir>/<teleop.id>.json`.

## GELLO hardware profiles and active assistance

`yam_leader` defaults to the versioned `trlc_dk1_v1` hardware profile. It
contains the September 2026 system-identification result for the bimanual
TRLC-DK1 leaders: encoder signs/ranges/offsets, effective link masses,
assisted joints, gravity gain/current limit, and the spring-open trigger.
Gravity assistance and gripper return are therefore on out of the box:

```bash
uv run lerobot-teleoperate \
  --robot.type=yam_follower \
  --robot.port=can0 \
  --teleop.type=yam_leader \
  --teleop.port=/dev/ttyUSB0 \
  --teleop.id=yam_gello_left
```

The profile is defined in
`lerobot_teleoperator_yam_gello/gravity_profiles.py`, next to the controller
that consumes it—not hidden in an operator command or calibration cache. The
bundled [gello_software/FACTR](https://github.com/wuphilipp/gello_software/tree/main/gello/factr)
URDF supplies kinematics/COMs; the profile supplies DK1 effective dynamics and
encoder mapping. These effective masses are fitted model constants, not literal
weigh-scale claims, so they intentionally remain in the profile rather than
being written into the upstream active-GELLO URDF.

For a different GELLO build, add a new named profile (and optionally a new
packaged URDF/MJCF asset) once its sysid is stable. During bring-up, every field
can still be overridden through LeRobot config arguments. Select
`--teleop.gravity_profile=passive` for position sensing with all torque disabled.

Safety measures:

- per-motor current is hard-clipped (default `250 mA`, configurable up to
  `500 mA`);
- the squeeze trigger uses current-based position mode and pushes toward its
  physical-open endpoint (`80 mA` in the DK1 profile), so the operator must
  squeeze and hold it closed;
- a 200 ms DYNAMIXEL bus watchdog stops output if updates cease;
- assistance latches off on a motor hardware error, a control exception, or
  temperature reaching 50 °C;
- disconnect writes zero current and disables torque.

### Measuring corrections instead of guessing: `gello_gravity_sysid`

When assist feels wrong (holds at one pose, drives at others), measure the
joint's real gravity curve on the bench and fit the corrections:

```bash
python -m lerobot_teleoperator_yam_gello.gello_gravity_sysid \
  --port /dev/ttyUSB0 --id yam_gello_left \
  --joint elbow_flex --poses 5 --enable-torque-output
```

For each operator-arranged pose it locks the other joints (current-capped
position hold), bisects the two breakaway edges of the test joint's
holding-current interval, and after >= 3 poses fits measured vs model balance
currents to print a suggested `gravity_joint_offsets_rad` delta (phase gap),
a distal mass scale (amplitude ratio), and the measured stiction band. Every
probe is dumped to a JSON diagnostics file. Spread the test joint's angles as
widely as safely possible — clustered poses make the fit ill-conditioned (the
tool warns). The test joint moves during measurement; keep the workspace
clear, and power-cycle the leader if the process is ever killed ungracefully.

The DK1 profile's signs and offsets are applied in radian space
(`q_urdf = sign * q_yam + offset`) and were identified together with its
ranges and effective masses. Do not mix constants across profiles. Use
`gravity_assist_dry_run=true` while identifying a new build: it logs modeled
positions, torques, and currents while applying no torque.

Note: this repo expects `lerobot` 0.4.3 features (plugin discovery in the
standard CLIs). If 0.4.3 is not on PyPI, install it from
[source](https://github.com/huggingface/lerobot) instead.

## Record

```bash
uv run lerobot-record \
  --dataset.repo_id=YOUR_USERNAME/yam_teleop \
  --dataset.single_task="Teleop YAM arm (pick/place)" \
  --dataset.fps=30 \
  --dataset.num_episodes=5 \
  --dataset.episode_time_s=60 \
  --dataset.reset_time_s=15 \
  --dataset.push_to_hub=true \
  --robot.type=yam_follower \
  --robot.port=can0 \
  --robot.gripper_type=crank_4310 \
  --teleop.type=yam_leader \
  --teleop.port=/dev/ttyUSB0
```

## Policy Control

```bash
uv run lerobot-control \
  --robot.type=yam_follower \
  --robot.port=can0 \
  --policy.path=YOUR_USERNAME/yam-policy
```

## Extras

List cameras:
```bash
uv run lerobot-find-cameras opencv
```

Teleop with cameras + Rerun:
```bash
uv run lerobot-teleoperate \
  --robot.type=yam_follower \
  --robot.port=can0 \
  --robot.gripper_type=crank_4310 \
  --robot.cameras='{
    "wrist": {"type": "opencv", "index_or_path": "/dev/video6", "width": 640, "height": 480, "fps": 30},
    "front": {"type": "opencv", "index_or_path": "/dev/video4", "width": 640, "height": 480, "fps": 30}
  }' \
  --teleop.type=yam_leader \
  --teleop.port=/dev/ttyUSB0 \
  --display_data=true
```

Safety step limits (normalized per‑cycle steps: joints in [-100, 100], gripper in [0, 100]):
```bash
--robot.lerobot_max_step=1 \
--robot.lerobot_gripper_max_step=1
```

## Packages

| Package | Purpose |
| --- | --- |
| `yam-common` | Shared YAM utilities |
| `lerobot_robot_yam` | Follower robot plugin |
| `lerobot_teleoperator_yam_gello` | Leader teleoperator plugin |

Install packages separately if you only need one component (e.g., follower‑only for policy inference).

