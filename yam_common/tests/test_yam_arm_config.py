"""YAMArmConfig must validate inputs and honor operator gains/limits."""

from __future__ import annotations

import math

import numpy as np
import pytest

from fakes import FakeMotorInfo, FakeRobot


def test_missing_kp_key_raises() -> None:
    from yam_common import YAMArmConfig

    with pytest.raises(ValueError, match="kp_gains"):
        YAMArmConfig(kp_gains={"shoulder_pan": 1.0})


def test_unordered_arm_limit_raises() -> None:
    from yam_common import YAMArmConfig
    from yam_common.yam_arm import DEFAULT_JOINT_LIMITS

    limits = dict(DEFAULT_JOINT_LIMITS)
    limits["elbow_flex"] = (3.0, 1.0)
    with pytest.raises(ValueError, match="joint_limits"):
        YAMArmConfig(joint_limits=limits)


def test_nonfinite_offset_raises() -> None:
    from yam_common import YAMArmConfig
    from yam_common.yam_arm import DEFAULT_MOTOR_OFFSETS

    offsets = dict(DEFAULT_MOTOR_OFFSETS)
    offsets["wrist_roll"] = math.nan
    with pytest.raises(ValueError, match="motor_offsets"):
        YAMArmConfig(motor_offsets=offsets)


def test_invalid_direction_raises() -> None:
    from yam_common import YAMArmConfig
    from yam_common.yam_arm import DEFAULT_MOTOR_DIRECTIONS

    directions = dict(DEFAULT_MOTOR_DIRECTIONS)
    directions["gripper"] = 0
    with pytest.raises(ValueError, match="motor_directions"):
        YAMArmConfig(motor_directions=directions)


def test_rest_pose_wrong_length_raises() -> None:
    from yam_common import YAMArmConfig

    with pytest.raises(ValueError, match="rest_pose"):
        YAMArmConfig(rest_pose=(0.0, 1.0))


def test_rest_pose_nonfinite_raises() -> None:
    from yam_common import YAMArmConfig

    with pytest.raises(ValueError, match="rest_pose"):
        YAMArmConfig(rest_pose=(0.0, 1.0, 1.0, 0.0, 0.0, 0.0, math.inf))


def test_operator_gripper_limits_and_gains_are_used(monkeypatch) -> None:
    from yam_common import YAMArm, YAMArmConfig
    from yam_common.yam_arm import DEFAULT_KD_GAINS, DEFAULT_KP_GAINS

    captured: dict = {}

    class TrackingRobot(FakeRobot):
        def __init__(self, **kwargs):
            super().__init__()
            captured.update(kwargs)

    created = []

    def chain_factory(**kwargs):
        class _Chain:
            def __init__(self) -> None:
                self.closed = False
                self.running = False
                self.kwargs = kwargs

            def __len__(self) -> int:
                return 7

            def read_states(self, torques=None):
                return [FakeMotorInfo(id=i + 1, pos=0.0) for i in range(7)]

            def close(self, disable_motors: bool = True) -> None:
                self.closed = True
                self.close_disable_motors = disable_motors

        chain = _Chain()
        created.append(chain)
        return chain

    monkeypatch.setattr("yam_common.yam_arm.MotorChainRobot", TrackingRobot)
    monkeypatch.setattr("yam_common.yam_arm.time.sleep", lambda _s: None)

    kp = dict(DEFAULT_KP_GAINS)
    kd = dict(DEFAULT_KD_GAINS)
    kp["gripper"] = 99.0
    kd["gripper"] = 3.3
    config = YAMArmConfig(
        use_gravity_compensation=False,
        gripper_limits=(0.1, -1.5),
        kp_gains=kp,
        kd_gains=kd,
        bustype="virtual",
    )
    arm = YAMArm(config, chain_factory=chain_factory)
    arm.connect()
    try:
        np.testing.assert_allclose(captured["gripper_limits"], (0.1, -1.5))
        assert captured["kp"][-1] == pytest.approx(99.0)
        assert captured["kd"][-1] == pytest.approx(3.3)
        assert created
        assert all(chain.kwargs.get("bustype") == "virtual" for chain in created)
    finally:
        arm.emergency_cleanup()
