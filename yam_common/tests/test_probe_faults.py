"""Probe must report present-but-faulted motors without enabling them."""

from __future__ import annotations

import can
import pytest

from yam_common.dm.dm_driver import ControlMode, DMSingleMotorCanInterface
from yam_common.dm.utils import ReceiveMode


def _fault_message(motor_id: int, error_nibble: int = 0x8) -> can.Message:
    data = bytearray(8)
    data[0] = (error_nibble << 4) & 0xF0
    return can.Message(arbitration_id=motor_id + 16, data=bytes(data), is_extended_id=False)


def test_probe_motor_returns_fault_code_without_enabling() -> None:
    iface = DMSingleMotorCanInterface.__new__(DMSingleMotorCanInterface)
    iface.control_mode = ControlMode.MIT
    iface.cmd_idoffset = ControlMode.get_id_offset(ControlMode.MIT)
    iface.receive_mode = ReceiveMode.p16
    iface.name = "probe-stub"
    iface.bus = type("StubBus", (), {"channel_info": "stub"})()

    def fail_enable(*_args, **_kwargs):
        raise AssertionError("motor_on must not be called")

    def fail_clear(*_args, **_kwargs):
        raise AssertionError("clean_error must not be called")

    iface.motor_on = fail_enable
    iface.clean_error = fail_clear
    iface._send_message_get_response = lambda *args, **kwargs: _fault_message(1, 0x8)

    result = DMSingleMotorCanInterface.probe_motor(iface, 1, "DM4340")
    assert result.id == 1
    assert result.error_code == "0x8"
    assert "over voltage" in result.error_message


def test_probe_motor_normal_response_still_succeeds() -> None:
    iface = DMSingleMotorCanInterface.__new__(DMSingleMotorCanInterface)
    iface.control_mode = ControlMode.MIT
    iface.cmd_idoffset = ControlMode.get_id_offset(ControlMode.MIT)
    iface.receive_mode = ReceiveMode.p16
    iface.name = "probe-stub"
    iface.bus = type("StubBus", (), {"channel_info": "stub"})()
    iface.motor_on = lambda *a, **k: (_ for _ in ()).throw(AssertionError("motor_on"))
    iface._send_message_get_response = lambda *args, **kwargs: _fault_message(2, 0x1)

    result = DMSingleMotorCanInterface.probe_motor(iface, 2, "DM4310")
    assert result.id == 2
    assert result.error_code == "0x1"
