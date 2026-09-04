"""Connect/hold/rest/zero-torque/close and control-loop health."""

from __future__ import annotations

import numpy as np
import pytest

from fakes import FakeRobot


def _make_arm(robot: FakeRobot):
    from yam_common import YAMArm, YAMArmConfig

    rest = (0.0, 1.2, 1.1, 0.0, 0.0, 0.0, 0.0)
    return YAMArm(
        YAMArmConfig(
            use_gravity_compensation=False,
            rest_pose=rest,
            zero_gravity_mode=True,
        ),
        robot_factory=lambda **kwargs: robot,
    )


def test_connect_hold_rest_zero_torque_and_close() -> None:
    from yam_common import YAMArmAlreadyConnectedError, YAMArmNotConnectedError

    robot = FakeRobot()
    arm = _make_arm(robot)
    assert arm.is_connected is False

    arm.connect()
    assert arm.is_connected is True
    with pytest.raises(YAMArmAlreadyConnectedError):
        arm.connect()

    hold_pos = robot.pos.copy()
    arm.hold_current_pose()
    np.testing.assert_allclose(robot.commands[-1], hold_pos)

    arm.command_rest()
    np.testing.assert_allclose(robot.commands[-1], np.array(arm.config.rest_pose))

    arm.zero_torque()
    assert robot.zero_torque is True

    telemetry = arm.get_telemetry()
    assert telemetry["connected"] is True
    assert telemetry["healthy"] is True
    assert "joint_pos" in telemetry
    assert "joint_vel" in telemetry
    assert "joint_eff" in telemetry

    arm.disconnect()
    assert arm.is_connected is False
    assert robot.closed is True
    assert robot.zero_torque is True
    with pytest.raises(YAMArmNotConnectedError):
        arm.get_observation()
    with pytest.raises(YAMArmNotConnectedError):
        arm.disconnect()


def test_public_apis_fail_after_control_loop_dies() -> None:
    from yam_common import YAMArmUnhealthyError

    robot = FakeRobot()
    arm = _make_arm(robot)
    arm.connect()
    robot.control_loop_error = RuntimeError("DM Error in control loop: bus down")

    with pytest.raises(YAMArmUnhealthyError, match="control loop"):
        arm.health_check()
    with pytest.raises(YAMArmUnhealthyError):
        arm.get_observation()
    with pytest.raises(YAMArmUnhealthyError):
        arm.send_action({"shoulder_pan.pos": 0.0})
    with pytest.raises(YAMArmUnhealthyError):
        arm.hold_current_pose()
    with pytest.raises(YAMArmUnhealthyError):
        arm.command_joint_pos(robot.pos)
    with pytest.raises(YAMArmUnhealthyError):
        arm.command_rest()
    with pytest.raises(YAMArmUnhealthyError):
        arm.zero_torque()

    telemetry = arm.get_telemetry()
    assert telemetry["healthy"] is False
    assert telemetry["error"] is not None
    assert "bus down" in str(telemetry["error"])


def test_motor_chain_robot_caches_control_loop_exception() -> None:
    from fakes import FakeMotorInfo
    from yam_common.motor_chain_robot import MotorChainRobot

    class DyingChain:
        def __init__(self) -> None:
            self.running = True
            self.start_thread_flag = True
            self._reads = 0

        def __len__(self) -> int:
            return 6

        def read_states(self, torques=None):
            return [
                FakeMotorInfo(id=i + 1, pos=0.1 * i)
                for i in range(6)
            ]

        def set_commands(self, torques, pos=None, vel=None, kp=None, kd=None):
            self.running = False
            return self.read_states(torques)

        def start_thread(self) -> None:
            self.start_thread_flag = True

        def close(self) -> None:
            self.running = False

    limits = np.array(
        [
            (-2.767, 3.28),
            (-0.15, 3.8),
            (-0.15, 3.28),
            (-1.72, 1.72),
            (-1.72, 1.72),
            (-2.24, 2.24),
        ]
    )
    robot = MotorChainRobot(
        motor_chain=DyingChain(),
        xml_path=None,
        use_gravity_comp=False,
        joint_limits=limits,
        zero_gravity_mode=True,
        gripper_index=None,
    )
    try:
        robot._server_thread.join(timeout=2.0)
        assert robot.control_loop_error is not None
        with pytest.raises(RuntimeError, match="control loop is not running"):
            robot.raise_if_unhealthy()
    finally:
        robot.close()
