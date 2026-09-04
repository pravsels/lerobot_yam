"""Shared hardware-free fakes for YAMArm tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pytest


@dataclass
class FakeMotorInfo:
    id: int
    error_code: str = "0x1"
    pos: float = 0.0
    vel: float = 0.0
    eff: float = 0.0
    temp_mos: float = 30.0
    temp_rotor: float = 30.0
    target_torque: float = 0.0


@dataclass
class FakeFeedback:
    id: int
    error_code: str = "0x1"
    error_message: str = "normal"
    position: float = 0.0
    velocity: float = 0.0
    torque: float = 0.0
    temperature_mos: float = 30.0
    temperature_rotor: float = 30.0


class FakeRobot:
    """Stand-in for MotorChainRobot used by YAMArm tests."""

    def __init__(self, pos: Optional[np.ndarray] = None) -> None:
        self.pos = np.array(
            pos if pos is not None else [0.0, 1.825, 1.565, 0.0, 0.0, 0.0, 0.5],
            dtype=np.float64,
        )
        self.vel = np.zeros(len(self.pos), dtype=np.float64)
        self.eff = np.zeros(len(self.pos), dtype=np.float64)
        self.closed = False
        self.zero_torque = False
        self.commands: list[np.ndarray] = []
        self.control_loop_error: Optional[BaseException] = None
        self.running = True

    def _raise_if_unhealthy(self) -> None:
        if self.control_loop_error is not None:
            raise RuntimeError("control loop is not running") from self.control_loop_error

    def get_joint_pos(self) -> np.ndarray:
        self._raise_if_unhealthy()
        return self.pos.copy()

    def get_observations(self) -> dict[str, np.ndarray]:
        self._raise_if_unhealthy()
        return {
            "joint_pos": self.pos[:6].copy(),
            "gripper_pos": np.array([self.pos[6]]),
            "joint_vel": self.vel.copy(),
            "joint_eff": self.eff.copy(),
        }

    def command_joint_pos(self, joint_pos: np.ndarray) -> None:
        self._raise_if_unhealthy()
        commanded = np.asarray(joint_pos, dtype=np.float64).copy()
        self.commands.append(commanded)
        self.pos = commanded

    def zero_torque_mode(self) -> None:
        self.zero_torque = True

    def close(self) -> None:
        self.closed = True
        self.running = False


class FakeCanInterface:
    """Records motor_on vs probe traffic without touching SocketCAN."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.motor_on_calls: list[tuple[int, str]] = []
        self.set_control_calls: list[dict[str, Any]] = []
        self.closed = False

    def motor_on(self, motor_id: int, motor_type: str) -> FakeFeedback:
        self.motor_on_calls.append((motor_id, motor_type))
        return FakeFeedback(id=motor_id)

    def set_control(
        self,
        motor_id: int,
        motor_type: str,
        pos: float = 0.0,
        vel: float = 0.0,
        kp: float = 0.0,
        kd: float = 0.0,
        torque: float = 0.0,
        ignore_error: bool = False,
    ) -> FakeFeedback:
        self.set_control_calls.append(
            {
                "motor_id": motor_id,
                "motor_type": motor_type,
                "pos": pos,
                "vel": vel,
                "kp": kp,
                "kd": kd,
                "torque": torque,
            }
        )
        return FakeFeedback(id=motor_id)

    def try_receive_message(self, *args: Any, **kwargs: Any) -> None:
        return None

    def close(self) -> None:
        self.closed = True

    def probe_motor(self, motor_id: int, motor_type: str = "DM4310") -> FakeFeedback:
        return self.set_control(motor_id, motor_type, 0.0, 0.0, 0.0, 0.0, 0.0)


@pytest.fixture
def fake_robot() -> FakeRobot:
    return FakeRobot()
