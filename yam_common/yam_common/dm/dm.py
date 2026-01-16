"""Thin MotorsBus wrapper over i2rt DM driver stack."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np

from lerobot.motors.motors_bus import Motor, MotorCalibration, MotorsBus
from lerobot.utils.errors import DeviceAlreadyConnectedError
from yam_common.dm.dm_driver import DMChainCanInterface, ControlMode, ReceiveMode
from yam_common.dm.utils import MotorType, float_to_uint, uint_to_float

logger = logging.getLogger(__name__)


@dataclass
class DMMotorFeedback:
    motor_id: int
    error_code: int
    position: float
    velocity: float
    torque: float
    temp_mos: float
    temp_rotor: float
    pos_int: int = 0
    vel_int: int = 0
    torque_int: int = 0


class DMMotorsBus(MotorsBus):
    """MotorsBus implementation that delegates CAN I/O to i2rt DM driver."""

    model_ctrl_table = {
        "DM4310": {
            "Present_Position": (0x01, 2),
            "Present_Velocity": (0x02, 2),
            "Present_Torque": (0x03, 2),
            "Goal_Position": (0x04, 2),
            "Goal_Velocity": (0x05, 2),
            "Goal_Torque": (0x06, 2),
            "KP": (0x07, 2),
            "KD": (0x08, 2),
            "Torque_Enable": (0x09, 1),
        },
        "DM4340": {
            "Present_Position": (0x01, 2),
            "Present_Velocity": (0x02, 2),
            "Present_Torque": (0x03, 2),
            "Goal_Position": (0x04, 2),
            "Goal_Velocity": (0x05, 2),
            "Goal_Torque": (0x06, 2),
            "KP": (0x07, 2),
            "KD": (0x08, 2),
            "Torque_Enable": (0x09, 1),
        },
        "DM3507": {
            "Present_Position": (0x01, 2),
            "Present_Velocity": (0x02, 2),
            "Present_Torque": (0x03, 2),
            "Goal_Position": (0x04, 2),
            "Goal_Velocity": (0x05, 2),
            "Goal_Torque": (0x06, 2),
            "KP": (0x07, 2),
            "KD": (0x08, 2),
            "Torque_Enable": (0x09, 1),
        },
    }
    model_resolution_table = {"DM4310": 65536, "DM4340": 65536, "DM3507": 65536}
    model_number_table = {"DM4310": 0x4310, "DM4340": 0x4340, "DM3507": 0x3507}
    model_baudrate_table = {"DM4310": {1000000: 1}, "DM4340": {1000000: 1}, "DM3507": {1000000: 1}}
    model_encoding_table = {"DM4310": {}, "DM4340": {}, "DM3507": {}}
    normalized_data = ["Present_Position", "Goal_Position"]
    default_timeout = 1000

    def connect(self, handshake: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(
                f"{self.__class__.__name__}('{self.port}') is already connected. Do not call `{self.__class__.__name__}.connect()` twice."
            )
        self._connect(handshake)
        logger.debug(f"{self.__class__.__name__} connected.")

    def __init__(
        self,
        port: str,
        motors: dict[str, Motor],
        calibration: dict[str, MotorCalibration] | None = None,
        bustype: str = "socketcan",
        bitrate: int = 1000000,
        motor_offsets: dict[str, float] | None = None,
        motor_directions: dict[str, int] | None = None,
        kp_gains: dict[str, float] | None = None,
        kd_gains: dict[str, float] | None = None,
        receive_mode: ReceiveMode = ReceiveMode.p16,
        control_mode: ControlMode = ControlMode.MIT,
        start_thread: bool = True,
    ):
        super().__init__(port, motors, calibration)
        self.bustype = bustype
        self.bitrate = bitrate
        self.receive_mode = receive_mode
        self.control_mode = control_mode
        self.start_thread = start_thread

        self._motor_offsets = motor_offsets or {name: 0.0 for name in motors}
        self._motor_directions = motor_directions or {name: 1 for name in motors}
        for name, direction in self._motor_directions.items():
            if direction not in (1, -1):
                raise ValueError(f"Motor direction for '{name}' must be 1 or -1, got {direction}")

        self._configured_kp = kp_gains or {}
        self._configured_kd = kd_gains or {}
        self._kp: dict[str, float] = {name: 0.0 for name in motors}
        self._kd: dict[str, float] = {name: 0.0 for name in motors}
        self._goal_pos: dict[str, float] = {name: 0.0 for name in motors}
        self._goal_vel: dict[str, float] = {name: 0.0 for name in motors}
        self._goal_torque: dict[str, float] = {name: 0.0 for name in motors}
        self._gravity_torques: dict[str, float] = {name: 0.0 for name in motors}

        self._feedback: dict[str, DMMotorFeedback] = {}
        self._chain: DMChainCanInterface | None = None
        self._running = False

    @property
    def is_connected(self) -> bool:
        return self._chain is not None and self._running

    def _connect(self, handshake: bool = True) -> None:
        motor_list = [(motor.id, motor.model) for motor in self.motors.values()]
        motor_offsets = np.array([self._motor_offsets[name] for name in self.motors.keys()])
        motor_directions = np.array([self._motor_directions[name] for name in self.motors.keys()])
        self._chain = DMChainCanInterface(
            motor_list=motor_list,
            motor_offset=motor_offsets,
            motor_direction=motor_directions,
            channel=self.port,
            bitrate=self.bitrate,
            start_thread=self.start_thread,
            motor_chain_name="lerobot_dm_chain",
            receive_mode=self.receive_mode,
            control_mode=self.control_mode,
        )
        self._running = True
        self._refresh_feedback()
        logger.info(f"Connected to CAN bus on {self.port} (i2rt driver)")

    def _refresh_feedback(self) -> None:
        if self._chain is None:
            return
        motor_infos = self._chain.read_states()
        for name, motor_info in zip(self.motors.keys(), motor_infos):
            motor = self.motors[name]
            const = MotorType.get_motor_constants(motor.model)
            pos_int = float_to_uint(motor_info.pos, const.POSITION_MIN, const.POSITION_MAX, 16)
            vel_int = float_to_uint(motor_info.vel, const.VELOCITY_MIN, const.VELOCITY_MAX, 12)
            torque_int = float_to_uint(motor_info.eff, const.TORQUE_MIN, const.TORQUE_MAX, 12)
            error_code = int(motor_info.error_code, 16) if isinstance(motor_info.error_code, str) else motor_info.error_code
            self._feedback[name] = DMMotorFeedback(
                motor_id=motor_info.id,
                error_code=error_code,
                position=motor_info.pos,
                velocity=motor_info.vel,
                torque=motor_info.eff,
                temp_mos=motor_info.temp_mos,
                temp_rotor=motor_info.temp_rotor,
                pos_int=pos_int,
                vel_int=vel_int,
                torque_int=torque_int,
            )

    def _push_commands(self) -> None:
        if self._chain is None:
            return
        motor_names = list(self.motors.keys())
        torques = np.array(
            [
                self._goal_torque[name] + self._gravity_torques.get(name, 0.0)
                for name in motor_names
            ]
        )
        pos = np.array([self._goal_pos[name] for name in motor_names])
        vel = np.array([self._goal_vel[name] for name in motor_names])
        kp = np.array([self._kp[name] for name in motor_names])
        kd = np.array([self._kd[name] for name in motor_names])
        self._chain.set_commands(torques, pos=pos, vel=vel, kp=kp, kd=kd, get_state=False)

    def _read(self, address: int, length: int, motor_id: int, **kwargs) -> tuple[int, int, int]:
        motor_name = None
        for name, motor in self.motors.items():
            if motor.id == motor_id:
                motor_name = name
                break
        if motor_name is None:
            raise ValueError(f"Unknown motor ID: {motor_id}")
        self._refresh_feedback()
        if motor_name not in self._feedback:
            raise RuntimeError(f"No feedback available for motor '{motor_name}'")
        feedback = self._feedback[motor_name]
        if address == 0x01:
            value = feedback.pos_int
        elif address == 0x02:
            value = feedback.vel_int
        elif address == 0x03:
            value = feedback.torque_int
        else:
            value = 0
        return value, 0, 0

    def _write(self, address: int, length: int, motor_id: int, value: int, **kwargs) -> tuple[int, int]:
        motor_name = None
        motor_model = None
        for name, motor in self.motors.items():
            if motor.id == motor_id:
                motor_name = name
                motor_model = motor.model
                break
        if motor_name is None:
            raise ValueError(f"Unknown motor ID: {motor_id}")

        const = MotorType.get_motor_constants(motor_model)
        if address == 0x04:  # Goal_Position
            pos_rad = uint_to_float(value, const.POSITION_MIN, const.POSITION_MAX, 16)
            if self._kp[motor_name] == 0.0:
                self._kp[motor_name] = self._configured_kp.get(motor_name, 80.0)
                self._kd[motor_name] = self._configured_kd.get(motor_name, 5.0)
            self._goal_pos[motor_name] = pos_rad
        elif address == 0x05:  # Goal_Velocity
            vel_rad = uint_to_float(value, const.VELOCITY_MIN, const.VELOCITY_MAX, 12)
            self._goal_vel[motor_name] = vel_rad
        elif address == 0x06:  # Goal_Torque
            torque_nm = uint_to_float(value, const.TORQUE_MIN, const.TORQUE_MAX, 12)
            self._goal_torque[motor_name] = torque_nm
        elif address == 0x07:  # KP
            self._kp[motor_name] = uint_to_float(value, const.KP_MIN, const.KP_MAX, 12)
        elif address == 0x08:  # KD
            self._kd[motor_name] = uint_to_float(value, const.KD_MIN, const.KD_MAX, 12)
        elif address == 0x09:  # Torque_Enable
            if value == 1:
                self._enable_motor(motor_name)
            else:
                self._disable_motor(motor_name)
        self._push_commands()
        return 0, 0

    def _enable_motor(self, motor_name: str) -> None:
        if self._chain is None:
            return
        motor_id = self.motors[motor_name].id
        self._chain.motor_interface.motor_on(motor_id, self.motors[motor_name].model)

    def _disable_motor(self, motor_name: str) -> None:
        if self._chain is None:
            return
        motor_id = self.motors[motor_name].id
        self._chain.motor_interface.motor_off(motor_id)

    def enable_torque(self, motors: str | list[str] | None = None, **kwargs) -> None:
        if motors is None:
            motors = list(self.motors.keys())
        elif isinstance(motors, str):
            motors = [motors]
        for motor_name in motors:
            self._enable_motor(motor_name)

    def disable_torque(self, motors: str | list[str] | None = None, **kwargs) -> None:
        if motors is None:
            motors = list(self.motors.keys())
        elif isinstance(motors, str):
            motors = [motors]
        for motor_name in motors:
            self._disable_motor(motor_name)

    def _disable_torque(self, motor_id: int, model: str, **kwargs) -> None:
        motor_name = None
        for name, motor in self.motors.items():
            if motor.id == motor_id:
                motor_name = name
                break
        if motor_name:
            self._disable_motor(motor_name)

    def set_zero_gravity_mode(self, motors: list[str] | None = None) -> None:
        if motors is None:
            motors = list(self.motors.keys())
        for motor_name in motors:
            self._kp[motor_name] = 0.0
            self._kd[motor_name] = 0.0
        self._push_commands()
        logger.info(f"Set zero-G mode for motors: {motors}")

    def set_position_control_mode(self, motors: list[str] | None = None) -> None:
        if motors is None:
            motors = list(self.motors.keys())
        self._refresh_feedback()
        for motor_name in motors:
            feedback = self._feedback.get(motor_name)
            if feedback is not None:
                self._goal_pos[motor_name] = feedback.position
            self._kp[motor_name] = self._configured_kp.get(motor_name, 80.0)
            self._kd[motor_name] = self._configured_kd.get(motor_name, 5.0)
        self._push_commands()
        logger.info(f"Set position control mode for motors: {motors}")

    def set_gravity_compensation_torques(self, torques: dict[str, float]) -> None:
        for motor_name, torque in torques.items():
            if motor_name in self._gravity_torques:
                self._gravity_torques[motor_name] = torque
        self._push_commands()

    def get_feedback(self, motor_name: str) -> DMMotorFeedback | None:
        return self._feedback.get(motor_name)

    def sync_read(
        self,
        data_name: str,
        motors: str | list[str] | None = None,
        **kwargs,
    ) -> dict[str, float]:
        if motors is None:
            motor_list = list(self.motors.keys())
        elif isinstance(motors, str):
            motor_list = [motors]
        else:
            motor_list = motors
        self._refresh_feedback()
        result = {}
        for motor_name in motor_list:
            feedback = self._feedback.get(motor_name)
            if feedback is None:
                logger.warning(f"No feedback for motor {motor_name}")
                continue
            if data_name == "Present_Position":
                result[motor_name] = feedback.position
            elif data_name == "Present_Velocity":
                result[motor_name] = feedback.velocity
            elif data_name == "Present_Torque":
                result[motor_name] = feedback.torque
            else:
                raise ValueError(f"Unknown data_name: {data_name}")
        return result

    def sync_write(
        self,
        data_name: str,
        values: dict[str, float],
        **kwargs,
    ) -> None:
        if data_name != "Goal_Position":
            raise ValueError(f"DMMotorsBus only supports Goal_Position writes, got: {data_name}")
        for motor_name, position in values.items():
            if motor_name in self._goal_pos:
                self._goal_pos[motor_name] = position
                if self._kp[motor_name] == 0.0:
                    self._kp[motor_name] = self._configured_kp.get(motor_name, 80.0)
                    self._kd[motor_name] = self._configured_kd.get(motor_name, 5.0)
        self._push_commands()

    def disconnect(self, disable_torque: bool = True) -> None:
        if not self.is_connected:
            return
        if disable_torque:
            self.set_zero_gravity_mode()
        if self._chain is not None:
            self._chain.close()
            try:
                self._chain.motor_interface.close()
            except Exception as e:
                logger.warning(f"Failed to close motor interface: {e}")
            self._chain = None
        self._running = False
        logger.info(f"Disconnected from CAN bus on {self.port}")

    @property
    def is_calibrated(self) -> bool:
        if not self.calibration:
            return False
        for motor_name in self.motors:
            if motor_name not in self.calibration:
                return False
            cal = self.calibration[motor_name]
            if cal.range_min == cal.range_max:
                return False
        return True

    def read_calibration(self) -> dict[str, MotorCalibration]:
        return self.calibration.copy() if self.calibration else {}

    def write_calibration(self, calibration: dict[str, MotorCalibration], cache: bool = True) -> None:
        if cache:
            self.calibration = calibration.copy()
        logger.info(f"Calibration written for {len(calibration)} motors")

    def record_ranges_of_motion(
        self,
        motors: list[str] | None = None,
        display_values: bool = True,
    ) -> tuple[dict[str, int], dict[str, int]]:
        if motors is None:
            motors = list(self.motors.keys())

        from lerobot.utils.utils import enter_pressed, move_cursor_up

        self.set_zero_gravity_mode(motors)
        logger.info("Entered zero-G mode for calibration")
        time.sleep(0.1)

        mins = {}
        maxs = {}
        for motor_name in motors:
            pos = self.read("Present_Position", motor_name, normalize=False)
            mins[motor_name] = pos
            maxs[motor_name] = pos

        try:
            while not enter_pressed():
                for motor_name in motors:
                    pos = self.read("Present_Position", motor_name, normalize=False)
                    mins[motor_name] = min(mins[motor_name], pos)
                    maxs[motor_name] = max(maxs[motor_name], pos)

                if display_values:
                    print("\n" + "-" * 55)
                    print(f"{'Motor':<20} | {'Min':>8} | {'Pos':>8} | {'Max':>8}")
                    print("-" * 55)
                    for motor_name in motors:
                        pos = self.read("Present_Position", motor_name, normalize=False)
                        print(f"{motor_name:<20} | {mins[motor_name]:>8} | {pos:>8} | {maxs[motor_name]:>8}")
                    move_cursor_up(len(motors) + 4)

                time.sleep(0.02)

        except KeyboardInterrupt:
            pass

        print("\n" * (len(motors) + 5))

        for motor_name in motors:
            if mins[motor_name] == maxs[motor_name]:
                raise ValueError(f"Motor '{motor_name}' has no range of motion recorded")

        return mins, maxs

    def _assert_protocol_is_compatible(self, instruction_name: str) -> None:
        pass

    def _encode_sign(self, data_name: str, ids_values: dict[int, int]) -> dict[int, int]:
        return ids_values

    def _decode_sign(self, data_name: str, ids_values: dict[int, int]) -> dict[int, int]:
        return ids_values

    def _split_into_byte_chunks(self, value: int, length: int) -> list[int]:
        if length == 1:
            return [value & 0xFF]
        if length == 2:
            return [(value >> 8) & 0xFF, value & 0xFF]
        return []

    def broadcast_ping(self, **kwargs) -> dict[int, int] | None:
        return None

    def _find_single_motor(self, motor: str, initial_baudrate: int | None = None) -> tuple[int, int]:
        if self._chain is None:
            raise RuntimeError("Motor chain not connected")
        for motor_id in range(1, 16):
            try:
                self._chain.motor_interface.set_control(motor_id, "DM4310", 0, 0, 0, 0, 0)
                return self.bitrate, motor_id
            except Exception:
                continue
        raise RuntimeError(f"Could not find motor '{motor}'")

    def configure_motors(self) -> None:
        pass

    def set_timeout(self, timeout_ms: int | None = None) -> None:
        pass

    def _get_half_turn_homings(self, positions: dict) -> dict:
        return {motor: 0 for motor in positions}
