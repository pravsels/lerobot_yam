"""DMChainCanInterface.close must stop its thread and release the bus."""

from __future__ import annotations

import threading

import pytest

from fakes import FakeCanInterface, FakeFeedback


def test_close_stops_thread_and_closes_bus(monkeypatch) -> None:
    from yam_common.dm import dm_driver

    created: list[FakeCanInterface] = []

    def factory(**kwargs):
        iface = FakeCanInterface(**kwargs)
        created.append(iface)
        return iface

    monkeypatch.setattr(dm_driver, "DMSingleMotorCanInterface", factory)

    chain = dm_driver.DMChainCanInterface(
        motor_list=[(1, "DM4340"), (2, "DM4310")],
        motor_offset=[0.0, 0.0],
        motor_direction=[1, 1],
        channel="can0",
        start_thread=True,
        enable_motors=True,
    )
    thread = getattr(chain, "_control_thread", None)
    assert thread is not None
    assert thread.is_alive()
    assert created[0].closed is False

    chain.close()
    assert chain.running is False
    assert created[0].closed is True
    thread.join(timeout=1.0)
    assert not thread.is_alive()

    chain.close()
    assert created[0].closed is True


def test_close_without_thread_still_closes_bus(monkeypatch) -> None:
    from yam_common.dm import dm_driver

    created: list[FakeCanInterface] = []

    def factory(**kwargs):
        iface = FakeCanInterface(**kwargs)
        created.append(iface)
        return iface

    monkeypatch.setattr(dm_driver, "DMSingleMotorCanInterface", factory)
    chain = dm_driver.DMChainCanInterface(
        motor_list=[(1, "DM4340")],
        motor_offset=[0.0],
        motor_direction=[1],
        channel="can0",
        start_thread=False,
        enable_motors=False,
    )
    chain.close()
    assert created[0].closed is True
    chain.close()


def test_close_from_control_thread_does_not_join_self(monkeypatch) -> None:
    from yam_common.dm import dm_driver

    monkeypatch.setattr(dm_driver, "DMSingleMotorCanInterface", lambda **kwargs: FakeCanInterface(**kwargs))
    chain = dm_driver.DMChainCanInterface(
        motor_list=[(1, "DM4340")],
        motor_offset=[0.0],
        motor_direction=[1],
        channel="can0",
        start_thread=False,
        enable_motors=False,
    )
    chain._control_thread = threading.current_thread()
    chain.running = True
    chain.close()
    assert chain.running is False
    assert chain.motor_interface.closed is True


def test_close_without_disable_releases_bus_only(monkeypatch) -> None:
    from yam_common.dm import dm_driver

    created: list[FakeCanInterface] = []

    class TrackingIface(FakeCanInterface):
        def __init__(self, **kwargs) -> None:
            super().__init__(**kwargs)
            self.motor_off_calls: list[int] = []

        def motor_off(self, motor_id: int) -> None:
            self.motor_off_calls.append(motor_id)

    def factory(**kwargs):
        iface = TrackingIface(**kwargs)
        created.append(iface)
        return iface

    monkeypatch.setattr(dm_driver, "DMSingleMotorCanInterface", factory)
    chain = dm_driver.DMChainCanInterface(
        motor_list=[(1, "DM4340"), (2, "DM4310")],
        motor_offset=[0.0, 0.0],
        motor_direction=[1, 1],
        channel="can0",
        start_thread=False,
        enable_motors=True,
    )
    chain.close(disable_motors=False)
    assert created[0].closed is True
    assert created[0].motor_off_calls == []
    assert not any(
        call["kp"] == 0.0 and call["kd"] == 0.0 and call["torque"] == 0.0
        for call in created[0].set_control_calls
    )
    chain.close(disable_motors=True)
    assert created[0].motor_off_calls == []


def test_constructor_motor_on_failure_closes_opened_interface(monkeypatch) -> None:
    """A later motor_on raise must still shut down the already-opened CAN bus."""
    from yam_common.dm import dm_driver

    created: list[FakeCanInterface] = []

    class FailLaterInterface(FakeCanInterface):
        def __init__(self, **kwargs) -> None:
            super().__init__(**kwargs)
            self.bus_shutdown = False
            self.motor_off_calls: list[int] = []
            self.zero_torque_ids: list[int] = []

        def motor_on(self, motor_id: int, motor_type: str) -> FakeFeedback:
            self.motor_on_calls.append((motor_id, motor_type))
            if motor_id == 2:
                raise RuntimeError("later motor_on failed")
            return FakeFeedback(id=motor_id)

        def motor_off(self, motor_id: int) -> None:
            self.motor_off_calls.append(motor_id)

        def set_control(self, motor_id: int, motor_type: str, pos=0.0, vel=0.0, kp=0.0, kd=0.0, torque=0.0, **kwargs):
            if kp == 0.0 and kd == 0.0 and torque == 0.0:
                self.zero_torque_ids.append(motor_id)
            return super().set_control(motor_id, motor_type, pos, vel, kp, kd, torque)

        def close(self) -> None:
            self.closed = True
            self.bus_shutdown = True

    def factory(**kwargs):
        iface = FailLaterInterface(**kwargs)
        created.append(iface)
        return iface

    monkeypatch.setattr(dm_driver, "DMSingleMotorCanInterface", factory)

    with pytest.raises(RuntimeError, match="later motor_on failed"):
        dm_driver.DMChainCanInterface(
            motor_list=[(1, "DM4340"), (2, "DM4310")],
            motor_offset=[0.0, 0.0],
            motor_direction=[1, 1],
            channel="can0",
            start_thread=False,
            enable_motors=True,
        )

    assert created, "constructor must open the motor interface before motor_on"
    iface = created[0]
    assert iface.motor_on_calls == [(1, "DM4340"), (2, "DM4310")]
    assert iface.closed is True
    assert iface.bus_shutdown is True
    assert 1 in iface.motor_off_calls or 1 in iface.zero_torque_ids
    assert all(call["pos"] == 0.0 and call["vel"] == 0.0 for call in iface.set_control_calls)
