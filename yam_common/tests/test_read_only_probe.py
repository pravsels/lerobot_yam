"""Read-only motor probe must never enable motion."""

from __future__ import annotations

from fakes import FakeCanInterface


def test_probe_motors_never_calls_motor_on(monkeypatch) -> None:
    from yam_common import probe_motors
    from yam_common.dm import dm_driver

    created: list[FakeCanInterface] = []

    def factory(**kwargs):
        iface = FakeCanInterface(**kwargs)
        created.append(iface)
        return iface

    monkeypatch.setattr(dm_driver, "DMSingleMotorCanInterface", factory)

    results = probe_motors("can0", [1, 2, 3, 4, 5, 6, 7])

    assert created, "probe_motors should construct a CAN interface"
    iface = created[0]
    assert iface.motor_on_calls == []
    assert [result.id for result in results] == [1, 2, 3, 4, 5, 6, 7]
    assert iface.closed is True
    assert all(call["kp"] == 0.0 and call["kd"] == 0.0 and call["torque"] == 0.0 for call in iface.set_control_calls)


def test_dm_chain_can_skip_motor_on_when_enable_motors_false(monkeypatch) -> None:
    from yam_common.dm.dm_driver import DMChainCanInterface
    from yam_common.dm import dm_driver

    created: list[FakeCanInterface] = []

    def factory(**kwargs):
        iface = FakeCanInterface(**kwargs)
        created.append(iface)
        return iface

    monkeypatch.setattr(dm_driver, "DMSingleMotorCanInterface", factory)

    chain = DMChainCanInterface(
        motor_list=[(1, "DM4340"), (2, "DM4340")],
        motor_offset=[0.0, 0.0],
        motor_direction=[1, 1],
        channel="can0",
        start_thread=False,
        enable_motors=False,
    )
    assert created[0].motor_on_calls == []
    assert chain.running is False
    chain.close()
