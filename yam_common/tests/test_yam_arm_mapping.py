"""Action/observation keys, normalization, and clamping match YAMFollower."""

from __future__ import annotations

import numpy as np
import pytest

from fakes import FakeRobot


EXPECTED_KEYS = (
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "wrist_yaw.pos",
    "gripper.pos",
)


def _connected_arm(robot: FakeRobot | None = None):
    from yam_common import YAMArm, YAMArmConfig

    robot = robot or FakeRobot()
    arm = YAMArm(
        YAMArmConfig(use_gravity_compensation=False, zero_gravity_mode=True),
        robot_factory=lambda **kwargs: robot,
    )
    arm.connect()
    return arm, robot


def test_action_and_observation_keys_are_the_seven_joint_pos_keys() -> None:
    from yam_common import YAMArm, YAMArmConfig

    arm = YAMArm(YAMArmConfig())
    assert tuple(arm.action_keys) == EXPECTED_KEYS
    assert tuple(arm.observation_keys) == EXPECTED_KEYS


def test_midrange_physical_pose_normalizes_to_zero_and_gripper_percent() -> None:
    from yam_common import YAMArmConfig, normalize_from_physical

    config = YAMArmConfig()
    mid = []
    for name in (
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "wrist_yaw",
    ):
        lo, hi = config.joint_limits[name]
        mid.append((lo + hi) / 2.0)
    mid.append(0.5)
    normalized = normalize_from_physical(np.array(mid), config)
    assert list(normalized.keys()) == list(EXPECTED_KEYS)
    for key in EXPECTED_KEYS[:-1]:
        assert normalized[key] == pytest.approx(0.0)
    assert normalized["gripper.pos"] == pytest.approx(50.0)


def test_physical_limits_map_to_plus_minus_100_and_gripper_0_100() -> None:
    from yam_common import YAMArmConfig, normalize_from_physical, physical_from_normalized

    config = YAMArmConfig()
    lows = [config.joint_limits[name][0] for name in config.arm_joint_names]
    highs = [config.joint_limits[name][1] for name in config.arm_joint_names]
    low_norm = normalize_from_physical(np.array(lows + [0.0]), config)
    high_norm = normalize_from_physical(np.array(highs + [1.0]), config)
    for key in EXPECTED_KEYS[:-1]:
        assert low_norm[key] == pytest.approx(-100.0)
        assert high_norm[key] == pytest.approx(100.0)
    assert low_norm["gripper.pos"] == pytest.approx(0.0)
    assert high_norm["gripper.pos"] == pytest.approx(100.0)

    recovered = physical_from_normalized(high_norm, config)
    np.testing.assert_allclose(recovered[:6], np.array(highs), atol=1e-5)
    assert recovered[6] == pytest.approx(1.0)


def test_send_action_clamps_range_and_step_and_returns_performed_action() -> None:
    from yam_common import YAMArmConfig, physical_from_normalized

    config = YAMArmConfig()
    mid_norm = {key: 0.0 for key in EXPECTED_KEYS}
    mid_norm["gripper.pos"] = 50.0
    robot = FakeRobot(pos=physical_from_normalized(mid_norm, config))
    arm, robot = _connected_arm(robot)
    performed = arm.send_action(
        {
            "shoulder_pan.pos": 100.0,
            "shoulder_lift.pos": 100.0,
            "elbow_flex.pos": 100.0,
            "wrist_flex.pos": 100.0,
            "wrist_roll.pos": 100.0,
            "wrist_yaw.pos": 100.0,
            "gripper.pos": 100.0,
        }
    )
    assert set(performed) == set(EXPECTED_KEYS)
    # Mid-range observation is 0 / 50; max step is 5.0 in normalized units.
    for key in EXPECTED_KEYS[:-1]:
        assert performed[key] == pytest.approx(5.0)
    assert performed["gripper.pos"] == pytest.approx(55.0)
    assert len(robot.commands) == 1


def test_send_action_holds_current_for_missing_or_nonfinite_keys() -> None:
    arm, robot = _connected_arm()
    before = arm.get_observation()
    performed = arm.send_action({"shoulder_pan.pos": float("nan"), "extra": 1.0})
    for key in EXPECTED_KEYS:
        assert performed[key] == pytest.approx(before[key])
    assert len(robot.commands) == 1
