"""Control-loop failures must log with traceback and stay cached."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from fakes import FakeMotorInfo


def test_motor_chain_robot_logs_exception_with_traceback(caplog) -> None:
    from yam_common.motor_chain_robot import MotorChainRobot

    class DyingChain:
        def __init__(self) -> None:
            self.running = True
            self.start_thread_flag = True

        def __len__(self) -> int:
            return 6

        def read_states(self, torques=None):
            return [FakeMotorInfo(id=i + 1, pos=0.1 * i) for i in range(6)]

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
    with caplog.at_level(logging.ERROR):
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
            assert any(record.exc_info for record in caplog.records)
        finally:
            robot.close()
