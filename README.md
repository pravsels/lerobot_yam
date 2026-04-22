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

