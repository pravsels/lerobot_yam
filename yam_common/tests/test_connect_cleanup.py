"""Failed connect must not leave a live CAN chain behind."""

from __future__ import annotations

import numpy as np
import pytest

from fakes import FakeCanInterface, FakeMotorInfo, FakeRobot


class LiveChain:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.running = True
        self.closed = False
        self.zeroed = False
        self.disabled = False
        self.close_disable_motors: bool | None = None
        self.start_thread_flag = bool(kwargs.get("start_thread", True))
        self.n = len(kwargs.get("motor_list") or [None] * 7)

    def __len__(self) -> int:
        return self.n

    def read_states(self, torques=None):
        return [FakeMotorInfo(id=i + 1, pos=0.05 * i) for i in range(self.n)]

    def set_commands(self, torques, pos=None, vel=None, kp=None, kd=None, get_state=True):
        self.zeroed = True
        return self.read_states(torques)

    def close(self, disable_motors: bool = True) -> None:
        self.close_disable_motors = disable_motors
        if disable_motors:
            self.disabled = True
        self.closed = True
        self.running = False


def test_failed_robot_construction_closes_live_chain(monkeypatch) -> None:
    from yam_common import YAMArm, YAMArmConfig

    created: list[LiveChain] = []

    def chain_factory(**kwargs):
        chain = LiveChain(**kwargs)
        created.append(chain)
        return chain

    def boom_robot(**kwargs):
        raise RuntimeError("xml/gains failed after chain started")

    monkeypatch.setattr("yam_common.yam_arm.MotorChainRobot", boom_robot)
    monkeypatch.setattr("yam_common.yam_arm.time.sleep", lambda _s: None)

    arm = YAMArm(
        YAMArmConfig(use_gravity_compensation=False),
        chain_factory=chain_factory,
    )
    with pytest.raises(RuntimeError, match="xml/gains failed after chain started"):
        arm.connect()

    assert arm.is_connected is False
    assert created, "connect should construct at least one chain"
    assert all(chain.closed for chain in created)
    assert all(chain.running is False for chain in created)
    wrap_chain, live_chain = created[0], created[-1]
    assert wrap_chain.close_disable_motors is False
    assert wrap_chain.disabled is False
    assert live_chain.close_disable_motors is True
    assert live_chain.disabled is True
    assert live_chain.zeroed is True


def test_failed_wrap_read_closes_first_chain(monkeypatch) -> None:
    from yam_common import YAMArm, YAMArmConfig

    class BoomReadChain(LiveChain):
        def read_states(self, torques=None):
            raise RuntimeError("wrap read failed")

    created: list[BoomReadChain] = []

    def chain_factory(**kwargs):
        chain = BoomReadChain(**kwargs)
        created.append(chain)
        return chain

    monkeypatch.setattr("yam_common.yam_arm.time.sleep", lambda _s: None)
    arm = YAMArm(
        YAMArmConfig(use_gravity_compensation=False),
        chain_factory=chain_factory,
    )
    with pytest.raises(RuntimeError, match="wrap read failed"):
        arm.connect()

    assert created
    assert created[0].closed is True
    assert created[0].close_disable_motors is True
    assert created[0].disabled is True
    assert created[0].zeroed is True
    assert arm.is_connected is False


def test_missing_model_is_rejected_before_can_opens() -> None:
    from yam_common import YAMArm, YAMArmConfig

    opened: list[int] = []

    def chain_factory(**kwargs):
        opened.append(1)
        raise AssertionError("CAN must not open before model validation")

    arm = YAMArm(
        YAMArmConfig(
            use_gravity_compensation=True,
            mujoco_xml_path="/no/such/yam.xml",
        ),
        chain_factory=chain_factory,
    )
    with pytest.raises(FileNotFoundError, match="yam.xml"):
        arm.connect()
    assert opened == []


def test_successful_wrap_read_releases_bus_without_disabling(monkeypatch) -> None:
    """Wrap-read close must free SocketCAN and leave motors enabled for the real chain."""
    from yam_common import YAMArm, YAMArmConfig
    from yam_common.dm import dm_driver

    created_ifaces: list[FakeCanInterface] = []
    created_chains: list[object] = []

    class TrackingIface(FakeCanInterface):
        def __init__(self, **kwargs) -> None:
            super().__init__(**kwargs)
            self.motor_off_calls: list[int] = []
            self.zero_torque_ids: list[int] = []

        def motor_off(self, motor_id: int) -> None:
            self.motor_off_calls.append(motor_id)

        def set_control(self, motor_id: int, motor_type: str, pos=0.0, vel=0.0, kp=0.0, kd=0.0, torque=0.0, **kwargs):
            if kp == 0.0 and kd == 0.0 and torque == 0.0:
                self.zero_torque_ids.append(motor_id)
            return super().set_control(motor_id, motor_type, pos, vel, kp, kd, torque)

    def iface_factory(**kwargs):
        iface = TrackingIface(**kwargs)
        created_ifaces.append(iface)
        return iface

    real_chain = dm_driver.DMChainCanInterface

    def chain_factory(**kwargs):
        chain = real_chain(**kwargs)
        created_chains.append(chain)
        return chain

    monkeypatch.setattr(dm_driver, "DMSingleMotorCanInterface", iface_factory)
    monkeypatch.setattr("yam_common.yam_arm.time.sleep", lambda _s: None)
    robot = FakeRobot()
    monkeypatch.setattr("yam_common.yam_arm.MotorChainRobot", lambda **kwargs: robot)

    arm = YAMArm(
        YAMArmConfig(use_gravity_compensation=False),
        chain_factory=chain_factory,
    )
    try:
        arm.connect()
        assert arm.is_connected is True
        assert len(created_ifaces) >= 1
        wrap_iface = created_ifaces[0]
        assert wrap_iface.closed is True
        assert wrap_iface.motor_on_calls, "wrap-read chain should have enabled motors"
        assert wrap_iface.motor_off_calls == []
        assert wrap_iface.zero_torque_ids == []
        assert created_chains[0].running is False
    finally:
        for chain in created_chains:
            close = getattr(chain, "close", None)
            if close is not None:
                close()


def test_connect_health_check_cleans_up_unhealthy_robot() -> None:
    from yam_common import YAMArm, YAMArmConfig, YAMArmUnhealthyError

    robot = FakeRobot()
    robot.control_loop_error = RuntimeError("died during start")
    arm = YAMArm(
        YAMArmConfig(use_gravity_compensation=False),
        robot_factory=lambda **kwargs: robot,
    )
    with pytest.raises(YAMArmUnhealthyError):
        arm.connect()
    assert arm.is_connected is False
    assert robot.closed is True
    assert robot.zero_torque is True
