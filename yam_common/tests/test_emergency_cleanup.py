"""Camera-connect rollback must not hang on interactive zero-G shutdown."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from fakes import FakeRobot

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FOLLOWER_SRC = _REPO_ROOT / "lerobot_robot_yam"
if str(_FOLLOWER_SRC) not in sys.path:
    sys.path.insert(0, str(_FOLLOWER_SRC))


class _FailingCamera:
    is_connected = False

    def connect(self) -> None:
        raise RuntimeError("camera connect failed")

    def disconnect(self) -> None:
        self.is_connected = False


def _boom_interactive_wait(*_args, **_kwargs):
    raise AssertionError("interactive shutdown wait invoked")


def test_camera_connect_failure_skips_interactive_shutdown_wait(monkeypatch) -> None:
    pytest.importorskip("lerobot")

    import yam_common.yam_arm as yam_arm_module
    from lerobot_robot_yam.config_yam_follower import YAMFollowerRobotConfig
    from lerobot_robot_yam.yam_follower import YAMFollower
    from yam_common import YAMArm, YAMArmConfig

    monkeypatch.setattr(yam_arm_module.time, "sleep", _boom_interactive_wait)
    monkeypatch.setattr("builtins.input", _boom_interactive_wait)

    robot = FakeRobot()
    arm = YAMArm(
        YAMArmConfig(
            use_gravity_compensation=False,
            shutdown_zero_gravity_wait_for_enter=True,
        ),
        robot_factory=lambda **kwargs: robot,
    )
    follower = YAMFollower(YAMFollowerRobotConfig())
    follower._arm = arm
    follower.cameras = {"wrist": _FailingCamera()}

    with pytest.raises(RuntimeError, match="camera connect failed"):
        follower.connect()

    assert robot.zero_torque is True
    assert robot.closed is True
    assert arm.is_connected is False


def test_explicit_disconnect_still_invokes_interactive_wait(monkeypatch) -> None:
    from yam_common import YAMArm, YAMArmConfig

    holds: list[bool] = []
    robot = FakeRobot()
    arm = YAMArm(
        YAMArmConfig(
            use_gravity_compensation=False,
            shutdown_zero_gravity_wait_for_enter=True,
        ),
        robot_factory=lambda **kwargs: robot,
    )
    monkeypatch.setattr(arm, "_hold_zero_gravity_before_shutdown", lambda: holds.append(True))
    arm.connect()
    arm.disconnect()
    assert holds == [True]
    assert robot.zero_torque is True
    assert robot.closed is True


def test_emergency_cleanup_does_not_invoke_interactive_wait(monkeypatch) -> None:
    from yam_common import YAMArm, YAMArmConfig

    holds: list[bool] = []
    robot = FakeRobot()
    arm = YAMArm(
        YAMArmConfig(
            use_gravity_compensation=False,
            shutdown_zero_gravity_wait_for_enter=True,
        ),
        robot_factory=lambda **kwargs: robot,
    )
    monkeypatch.setattr(arm, "_hold_zero_gravity_before_shutdown", lambda: holds.append(True))
    arm.connect()
    arm.emergency_cleanup()
    assert holds == []
    assert robot.zero_torque is True
    assert robot.closed is True
    assert arm.is_connected is False
