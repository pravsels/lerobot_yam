"""i2rt-compatible DM motor driver (MIT mode + CAN chain)."""

from __future__ import annotations

import logging
import os
import struct
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Protocol, Tuple

import can
import numpy as np

from .can_interface import CanInterface
from .utils import (
    FeedbackFrameInfo,
    MotorConstants,
    MotorErrorCode,
    MotorInfo,
    MotorType,
    ReceiveMode,
    float_to_uint,
    uint_to_float,
)

log_level = os.getenv("LOGLEVEL", "ERROR").upper()
logging.basicConfig(level=log_level)

# Control loop frequency
CONTROL_FREQ = 250
CONTROL_PERIOD = 1.0 / CONTROL_FREQ  # 4 ms

EXPECTED_CONTROL_PERIOD = 0.007
REPORT_INTERVAL = 30.0


class RateRecorder:
    def __init__(
        self,
        name: str | None = None,
        report_interval: float = 10,
        min_required_frequency: float | None = None,
    ):
        self.report_interval = report_interval
        self.last_report_time: float | None = None
        self.iteration_count = 0
        self.name = name
        self.min_required_frequency = min_required_frequency

    def __enter__(self):
        return self.start()

    def start(self) -> "RateRecorder":
        self.last_report_time = time.time()
        self.iteration_count = 0
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.last_report_time is not None:
            self._report_rate()

    def _report_rate(self) -> float:
        assert self.last_report_time is not None, "RateRecorder must be started before reporting."
        elapsed_time = time.time() - self.last_report_time
        rate = self.iteration_count / elapsed_time if elapsed_time > 0 else 0
        logging.info(f"{self.name} Total rate: {rate:.2f} iterations per second over {elapsed_time:.2f} seconds.")
        return rate

    def track(self) -> None:
        self.iteration_count += 1
        current_time = time.time()
        assert self.last_report_time is not None, "RateRecorder must be started before tracking."
        if current_time - self.last_report_time >= self.report_interval:
            interval_rate = self._report_rate()
            if self.min_required_frequency is not None and interval_rate < self.min_required_frequency:
                raise RuntimeError(
                    f"{self.name} frequency too low: {interval_rate:.2f} Hz "
                    f"(required: {self.min_required_frequency:.2f} Hz) over {self.report_interval:.1f}s interval"
                )
            self.last_report_time = current_time
            self.iteration_count = 0


class ControlMode:
    MIT = "MIT"
    POS_VEL = "POS_VEL"
    VEL = "VEL"

    @classmethod
    def get_id_offset(cls, control_mode: str) -> int:
        if control_mode == cls.MIT:
            return 0x000
        if control_mode == cls.POS_VEL:
            return 0x100
        if control_mode == cls.VEL:
            return 0x200
        raise ValueError(f"Control mode '{control_mode}' not recognized.")


@dataclass
class PassiveEncoderInfo:
    id: int
    position: float
    velocity: float
    io_inputs: List[bool]


class PassiveEncoderReader:
    def __init__(self, can_interface: CanInterface, receive_mode: ReceiveMode = ReceiveMode.plus_one):
        self.can_interface = can_interface
        self.receive_mode = receive_mode

    def read_encoder(self, encoder_id: int) -> PassiveEncoderInfo:
        data = [0xFF, 0x02]
        message = self.can_interface._send_message_get_response(
            encoder_id,
            encoder_id,
            data,
            expected_id=self.receive_mode.get_receive_id(0x50E),
            max_retry=15,
        )
        pos, vel, button_state = self._parse_encoder_message(message)
        pos_range = [-0.7, 0.7]
        pos = np.clip(pos, pos_range[0], pos_range[1])
        delta = np.abs(0.0 - pos)
        pos = delta / 0.7
        return PassiveEncoderInfo(id=encoder_id, position=pos, velocity=vel, io_inputs=button_state)

    def _parse_encoder_message(self, message: can.Message) -> Tuple[float, float, List[bool]]:
        struct_format = "!B h h B"
        device_id, position, velocity, digital_inputs = struct.unpack(struct_format, message.data)
        position_rad = position * 2 * np.pi / 4096
        velocity_rad = velocity * 2 * np.pi / 4096
        button_state = [digital_inputs % 2, digital_inputs // 2]
        return position_rad, velocity_rad, button_state


class EncoderChain:
    def __init__(self, encoder_ids: List[int], encoder_interface: PassiveEncoderReader):
        self.encoder_ids = encoder_ids
        self.encoder_interface = encoder_interface

    def read_states(self) -> List[PassiveEncoderInfo]:
        return [self.encoder_interface.read_encoder(encoder_id) for encoder_id in self.encoder_ids]


class DMSingleMotorCanInterface(CanInterface):
    """CAN interface with a single motor (MIT mode)."""

    def __init__(
        self,
        control_mode: str = ControlMode.MIT,
        channel: str = "PCAN_USBBUS1",
        bustype: str = "socketcan",
        bitrate: int = 1000000,
        receive_mode: ReceiveMode = ReceiveMode.p16,
        name: str = "default_can_DM_interface",
        use_buffered_reader: bool = False,
    ):
        super().__init__(
            channel,
            bustype,
            bitrate,
            receive_mode=receive_mode,
            name=name,
            use_buffered_reader=use_buffered_reader,
        )
        self.control_mode = control_mode
        self.cmd_idoffset = ControlMode.get_id_offset(self.control_mode)
        self.receive_mode = receive_mode

    def _get_frame_id(self, motor_id: int) -> int:
        return self.cmd_idoffset + motor_id

    def motor_on(self, motor_id: int, motor_type: str) -> FeedbackFrameInfo:
        current_level = logging.getLogger().getEffectiveLevel()
        logging.getLogger().setLevel(logging.ERROR)

        data = [0xFF] * 7 + [0xFC]
        message = self._send_message_get_response(motor_id, motor_id, data)

        motor_info = self.parse_recv_message(message, MotorType.DM4310, ignore_error=True)
        if int(motor_info.error_code, 16) != MotorErrorCode.normal:
            while int(motor_info.error_code, 16) != MotorErrorCode.normal:
                logging.info(f"motor {motor_id} error: {motor_info.error_message}")
                self.clean_error(motor_id=motor_id)
                self.try_receive_message()
                logging.info(f"motor {motor_id} error cleaned")
                message = self._send_message_get_response(motor_id, motor_id, data)
                motor_info = self.parse_recv_message(message, motor_type, ignore_error=True)
        else:
            logging.info(f"motor {motor_id} is already on")
        logging.getLogger().setLevel(current_level)
        return self.parse_recv_message(message, motor_type)

    def clean_error(self, motor_id: int) -> None:
        data = [0xFF] * 7 + [0xFB]
        message = can.Message(arbitration_id=motor_id, data=data, is_extended_id=False)
        for _ in range(3):
            try:
                self.bus.send(message)
            except Exception as e:
                logging.warning(e)
                logging.warning(
                    "\033[91m" + "CAN Error: Failed to communicate with motor over can bus. Retrying..." + "\033[0m"
                )

    def motor_off(self, motor_id: int) -> None:
        data = [0xFF] * 7 + [0xFD]
        self._send_message_get_response(self._get_frame_id(motor_id), motor_id, data)

    def save_zero_position(self, motor_id: int) -> None:
        data = [0xFF] * 7 + [0xFE]
        try:
            self._send_message_get_response(self._get_frame_id(motor_id), motor_id, data, 2)
        except AssertionError:
            pass
        current_state = self.set_control(motor_id, MotorType.DM4310, 0, 0, 0, 0, 0)
        if abs(current_state.position) < 0.01:
            logging.info(f"motor {motor_id} set zero position success, current position: {current_state.position}")

    def set_control(
        self,
        motor_id: int,
        motor_type: str,
        pos: float,
        vel: float,
        kp: float,
        kd: float,
        torque: float,
    ) -> FeedbackFrameInfo:
        frame_id = self._get_frame_id(motor_id)
        data = bytearray(8)
        if self.control_mode == ControlMode.MIT:
            const = MotorType.get_motor_constants(motor_type)
            pos_tmp = float_to_uint(pos, const.POSITION_MIN, const.POSITION_MAX, 16)
            vel_tmp = float_to_uint(vel, const.VELOCITY_MIN, const.VELOCITY_MAX, 12)
            kp_tmp = float_to_uint(kp, const.KP_MIN, const.KP_MAX, 12)
            kd_tmp = float_to_uint(kd, const.KD_MIN, const.KD_MAX, 12)
            tor_tmp = float_to_uint(torque, const.TORQUE_MIN, const.TORQUE_MAX, 12)

            data[0] = (pos_tmp >> 8) & 0xFF
            data[1] = pos_tmp & 0xFF
            data[2] = (vel_tmp >> 4) & 0xFF
            data[3] = ((vel_tmp & 0xF) << 4) | (kp_tmp >> 8)
            data[4] = kp_tmp & 0xFF
            data[5] = (kd_tmp >> 4) & 0xFF
            data[6] = ((kd_tmp & 0xF) << 4) | (tor_tmp >> 8)
            data[7] = tor_tmp & 0xFF
        elif self.control_mode == ControlMode.VEL:
            can_data = struct.pack("<f", vel)
            data[0:4] = can_data[0:4]

        message = self._send_message_get_response(frame_id, motor_id, data, max_retry=15)
        return self.parse_recv_message(message, motor_type)

    def parse_recv_message(
        self, message: can.Message, motor_type: str, ignore_error: bool = False
    ) -> FeedbackFrameInfo:
        data = message.data
        error_int = (data[0] & 0xF0) >> 4
        error_hex = hex(error_int)
        error_message = MotorErrorCode.get_error_message(error_int)

        motor_id_of_this_response = self.receive_mode.to_motor_id(message.arbitration_id)
        if error_hex != "0x1":
            logging.warning(
                f"motor id: {motor_id_of_this_response}, error: {error_message} "
                f"at {self.name} and channel {self.bus.channel_info}"
            )
            if not ignore_error:
                logging.error(
                    f"motor id: {motor_id_of_this_response}, error: {error_message} "
                    f"at {self.name} and channel {self.bus.channel_info}"
                )
                raise RuntimeError(
                    f"Motor error detected: motor id: {motor_id_of_this_response}, error: {error_message}"
                )
        p_int = (data[1] << 8) | data[2]
        v_int = (data[3] << 4) | (data[4] >> 4)
        t_int = ((data[4] & 0xF) << 8) | data[5]
        temporature_mos = data[6]
        temperature_rotor = data[7]

        const = MotorType.get_motor_constants(motor_type)
        position = uint_to_float(p_int, const.POSITION_MIN, const.POSITION_MAX, 16)
        velocity = uint_to_float(v_int, const.VELOCITY_MIN, const.VELOCITY_MAX, 12)
        torque = uint_to_float(t_int, const.TORQUE_MIN, const.TORQUE_MAX, 12)
        temperature_mos = float(temporature_mos)
        temperature_rotor = float(temperature_rotor)

        return FeedbackFrameInfo(
            id=motor_id_of_this_response,
            error_code=error_hex,
            error_message=error_message,
            position=position,
            velocity=velocity,
            torque=torque,
            temperature_mos=temperature_mos,
            temperature_rotor=temperature_rotor,
        )


@dataclass
class MotorCmd:
    type: str = "pos_vel_torque"
    pos: float = 0.0
    vel: float = 0.0
    torque: float = 0.0
    kp: float = 0.0
    kd: float = 0.0


class MotorChain(Protocol):
    def __len__(self) -> int:
        raise NotImplementedError

    def set_commands(
        self,
        torques: np.ndarray,
        pos: Optional[np.ndarray] = None,
        vel: Optional[np.ndarray] = None,
        kp: Optional[np.ndarray] = None,
        kd: Optional[np.ndarray] = None,
    ) -> List[MotorInfo]:
        raise NotImplementedError


class DMChainCanInterface(MotorChain):
    def __init__(
        self,
        motor_list: List[Tuple[int, str]],
        motor_offset: np.ndarray,
        motor_direction: np.ndarray,
        channel: str = "PCAN_USBBUS1",
        bitrate: int = 1000000,
        start_thread: bool = True,
        motor_chain_name: str = "default_motor_chain",
        receive_mode: ReceiveMode = ReceiveMode.p16,
        control_mode: ControlMode = ControlMode.MIT,
        get_same_bus_device_driver: Optional[Callable] = None,
        use_buffered_reader: bool = False,
    ):
        assert not use_buffered_reader, (
            "buffered reader is not very stable; the latest encoder fix allows us to use the non-buffered reader"
        )
        assert len(motor_list) > 0
        assert len(motor_list) == len(motor_offset) == len(motor_direction), (
            f"len{len(motor_list)}, len{len(motor_offset)}, len{len(motor_direction)}"
        )
        self.motor_list = motor_list
        self.motor_offset = np.array(motor_offset)
        self.motor_direction = np.array(motor_direction)
        self.channel = channel
        logging.info(f"Channel: {channel}, Bitrate: {bitrate}")
        if "can" in channel:
            self.motor_interface = DMSingleMotorCanInterface(
                channel=channel,
                bustype="socketcan",
                receive_mode=receive_mode,
                name=motor_chain_name,
                control_mode=control_mode,
                use_buffered_reader=use_buffered_reader,
            )
        else:
            self.motor_interface = DMSingleMotorCanInterface(
                channel=channel,
                bitrate=bitrate,
                name=motor_chain_name,
                use_buffered_reader=use_buffered_reader,
            )
        self.state = None
        self.state_lock = threading.Lock()

        self.same_bus_device_states = None
        self.same_bus_device_lock = threading.Lock()
        if get_same_bus_device_driver is not None:
            self.same_bus_device_driver = get_same_bus_device_driver(self.motor_interface)
        else:
            self.same_bus_device_driver = None

        self.absolute_positions = None
        self._motor_on()
        starting_command = []
        for motor_state in self.state:
            starting_command.append(MotorCmd(torque=motor_state.torque))
        logging.info(f"Initializing motorchain with starting command: {starting_command}")
        self.commands = starting_command
        self.command_lock = threading.Lock()

        self.start_thread_flag = start_thread
        if start_thread:
            self.start_thread()

    def __repr__(self) -> str:
        return f"DMChainCanInterface(channel={self.channel})"

    def _update_absolute_positions(self, motor_feedback: List[FeedbackFrameInfo]) -> None:
        init_mode = False
        if self.absolute_positions is None:
            self.absolute_positions = np.zeros(len(self.motor_list))
            init_mode = True
        for idx, motor_info in enumerate(self.motor_list):
            motor_id, motor_type = motor_info
            const = MotorType.get_motor_constants(motor_type)
            position_min = const.POSITION_MIN
            position_max = const.POSITION_MAX
            position_range = position_max - position_min

            current_position = motor_feedback[idx].position
            previous_position = self.absolute_positions[idx]

            delta_position = current_position - (previous_position % position_range)
            if delta_position > position_range / 2:
                delta_position -= position_range
            elif delta_position < -position_range / 2:
                delta_position += position_range

            if init_mode:
                self.absolute_positions[idx] = current_position
            else:
                self.absolute_positions[idx] += delta_position

    def __len__(self):
        return len(self.motor_list)

    def _joint_position_real_to_sim(self, joint_position_real: float) -> float:
        return (joint_position_real - self.motor_offset) * self.motor_direction

    def _joint_position_real_to_sim_idx(self, joint_position_real: float, idx: int) -> float:
        return (joint_position_real - self.motor_offset[idx]) * self.motor_direction[idx]

    def _joint_position_sim_to_real_idx(self, joint_position_sim: float, idx: int) -> float:
        return joint_position_sim * self.motor_direction[idx] + self.motor_offset[idx]

    def _motor_on(self) -> None:
        motor_feedback = []
        for _ in range(7):
            self.motor_interface.try_receive_message(timeout=0.001)
        for motor_id, motor_type in self.motor_list:
            logging.info(f"Turning on motor_id: {motor_id}, motor_type: {motor_type}")
            time.sleep(0.003)
            motor_feedback.append(self.motor_interface.motor_on(motor_id, motor_type))
        self._update_absolute_positions(motor_feedback)
        self.state = motor_feedback
        self.running = True
        logging.info("starting separate thread for control loop")

    def start_thread(self) -> None:
        self._motor_on()
        thread = threading.Thread(target=self._set_torques_and_update_state)
        thread.start()
        time.sleep(0.1)
        while self.state is None:
            time.sleep(0.1)
            logging.info("waiting for the first state")

    def _set_torques_and_update_state(self) -> None:
        last_step_time = time.time()
        step_time_exceed_count = 0
        step_time_sum = 0.0
        step_time_count = 0
        report_start_time = time.time()
        with RateRecorder(name=self) as rate_recorder:
            while self.running:
                try:
                    while time.time() - last_step_time < CONTROL_PERIOD - 0.001:
                        time.sleep(0.001)
                    curr_time = time.time()
                    step_time = curr_time - last_step_time
                    last_step_time = curr_time

                    step_time_sum += step_time
                    step_time_count += 1
                    if step_time > EXPECTED_CONTROL_PERIOD:
                        step_time_exceed_count += 1

                    if step_time_exceed_count > 0 and curr_time - report_start_time >= REPORT_INTERVAL:
                        mean_step_time = step_time_sum / step_time_count if step_time_count > 0 else 0.0
                        logging.info(
                            f"[{self} {REPORT_INTERVAL}s Report] step_time > {EXPECTED_CONTROL_PERIOD}s: "
                            f"{step_time_exceed_count} times, mean step_time: {mean_step_time:.6f} s"
                        )
                        step_time_exceed_count = 0
                        step_time_sum = 0.0
                        step_time_count = 0
                        report_start_time = curr_time

                    with self.command_lock:
                        motor_feedback = self._set_commands(self.commands)
                        errors = np.array([motor_feedback[i].error_code != "0x1" for i in range(len(motor_feedback))])
                        if np.any(errors):
                            self.running = False
                            logging.error(f"motor errors: {errors}")
                            raise Exception("motors have errors, stopping control loop")

                    with self.state_lock:
                        self.state = motor_feedback
                        self._update_absolute_positions(motor_feedback)
                    if self.same_bus_device_driver is not None:
                        time.sleep(0.001)
                        with self.same_bus_device_lock:
                            self.same_bus_device_states = self.same_bus_device_driver.read_states()
                    time.sleep(0.0005)
                    rate_recorder.track()
                except Exception as e:
                    print(f"DM Error in control loop: {e}")
                    self.running = False
                    raise e

    def _set_commands(self, commands: List[MotorCmd]) -> List[FeedbackFrameInfo]:
        motor_feedback = []
        for idx, motor_info in enumerate(self.motor_list):
            motor_id, motor_type = motor_info
            torque = commands[idx].torque * self.motor_direction[idx]
            pos = self._joint_position_sim_to_real_idx(commands[idx].pos, idx)
            vel = commands[idx].vel * self.motor_direction[idx]
            kp = commands[idx].kp
            kd = commands[idx].kd
            try:
                fd_back = self.motor_interface.set_control(
                    motor_id=motor_id,
                    motor_type=motor_type,
                    pos=pos,
                    vel=vel,
                    kp=kp,
                    kd=kd,
                    torque=torque,
                )
            except Exception as e:
                logging.error(f"{idx}th motor at DMChainCanInterface {self} failed with info {motor_info}")
                raise e
            motor_feedback.append(fd_back)
        return motor_feedback

    def read_states(self, torques: Optional[np.ndarray] = None) -> List[MotorInfo]:
        motor_infos = []
        with self.state_lock:
            for idx in range(len(self.motor_list)):
                state = self.state[idx]
                motor_infos.append(
                    MotorInfo(
                        id=state.id,
                        error_code=state.error_code,
                        target_torque=torques[idx] if torques is not None else 0.0,
                        vel=state.velocity * self.motor_direction[idx],
                        eff=state.torque * self.motor_direction[idx],
                        pos=self._joint_position_real_to_sim_idx(self.absolute_positions[idx], idx),
                        temp_rotor=state.temperature_rotor,
                        temp_mos=state.temperature_mos,
                    )
                )
        return motor_infos

    def set_commands(
        self,
        torques: np.ndarray,
        pos: Optional[np.ndarray] = None,
        vel: Optional[np.ndarray] = None,
        kp: Optional[np.ndarray] = None,
        kd: Optional[np.ndarray] = None,
        get_state: bool = True,
    ) -> List[MotorInfo]:
        command = []
        for idx in range(len(self.motor_list)):
            command.append(
                MotorCmd(
                    torque=torques[idx],
                    pos=pos[idx] if pos is not None else 0.0,
                    vel=vel[idx] if vel is not None else 0.0,
                    kp=kp[idx] if kp is not None else 0.0,
                    kd=kd[idx] if kd is not None else 0.0,
                )
            )
        with self.command_lock:
            self.commands = command
        if get_state:
            return self.read_states(torques=torques)
        return []

    def get_same_bus_device_states(self) -> Any:
        with self.same_bus_device_lock:
            return self.same_bus_device_states

    def close(self) -> None:
        self.running = False


class MultiDMChainCanInterface(MotorChain):
    def __init__(self, interfaces: List[DMChainCanInterface]):
        self.interfaces = interfaces

    def __len__(self):
        return sum([len(inter) for inter in self.interfaces])

    def set_commands(
        self,
        torques: np.ndarray,
        pos: Optional[np.ndarray] = None,
        vel: Optional[np.ndarray] = None,
        kp: Optional[np.ndarray] = None,
        kd: Optional[np.ndarray] = None,
    ) -> List[MotorInfo]:
        start_idx = 0
        motor_infos = []
        for inter in self.interfaces:
            inter_len = len(inter)
            end_idx = start_idx + inter_len
            inter_torques = torques[start_idx:end_idx]
            inter_pos = pos[start_idx:end_idx] if pos is not None else None
            inter_vel = vel[start_idx:end_idx] if vel is not None else None
            inter_kp = kp[start_idx:end_idx] if kp is not None else None
            inter_kd = kd[start_idx:end_idx] if kd is not None else None
            infos = inter.set_commands(inter_torques, inter_pos, inter_vel, inter_kp, inter_kd)
            motor_infos.extend(infos)
            start_idx = end_idx
        return motor_infos

