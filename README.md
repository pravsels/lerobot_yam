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

## Optional passive-GELLO assistance

The YAM leader can apply conservative current-limited assistance to passive
XL330 GELLOs:

```bash
uv run lerobot-teleoperate \
  --robot.type=yam_follower \
  --robot.port=can0 \
  --teleop.type=yam_leader \
  --teleop.port=/dev/ttyUSB0 \
  --teleop.id=yam_gello_left \
  --teleop.gravity_assist=true \
  --teleop.gravity_assist_gain=0.10 \
  --teleop.gravity_assist_current_limit_ma=200 \
  --teleop.gripper_return=true \
  --teleop.gripper_return_current_ma=80
```

Both features are off by default. Gravity torque comes from the active-YAM
GELLO URDF used by
[gello_software/FACTR](https://github.com/wuphilipp/gello_software/tree/main/gello/factr),
but the passive GELLO has weaker XL330s and somewhat different mass. Start at
gain `0.05`–`0.10`; it should reduce sag, not hold the arm hands-free.

Safety measures:

- per-motor current is hard-clipped (default `250 mA`, configurable up to
  `500 mA`);
- the squeeze trigger uses current-based position mode and returns to the
  calibration's `0`/open endpoint (default `100 mA`);
- a 200 ms DYNAMIXEL bus watchdog stops output if updates cease;
- assistance latches off on a motor hardware error, a control exception, or
  temperature reaching 50 °C;
- disconnect writes zero current and disables torque.

FACTR's YAM motor directions are the defaults:
`[1, -1, -1, -1, 1, 1]`. If any joint assists gravity in the wrong direction,
stop immediately and override `gravity_joint_signs` before retrying. Never
raise gain to compensate for a wrong sign.

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

