# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""i2rt-authentic MotorChainRobot port for YAM."""

from __future__ import annotations

import copy
import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np

from yam_common.dm.dm_driver import MotorChain, MotorInfo
from yam_common.mujoco_kdl import MuJoCoKDL
from yam_common.utils import GripperForceLimiter, GripperType, JointMapper

logger = logging.getLogger(__name__)


@dataclass
class JointStates:
    names: List[str]
    pos: np.ndarray
    vel: np.ndarray
    eff: np.ndarray
    temp_mos: np.ndarray
    temp_rotor: np.ndarray

    def asdict(self) -> Dict[str, Any]:
        return {
            "names": self.names,
            "pos": self.pos.flatten().tolist(),
            "vel": self.vel.flatten().tolist(),
            "eff": self.eff.flatten().tolist(),
        }


@dataclass
class JointCommands:
    torques: np.ndarray
    pos: np.ndarray
    vel: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    indices: Optional[List[int]] = None

    @classmethod
    def init_all_zero(cls, n_joints: int) -> "JointCommands":
        return cls(
            torques=np.zeros(n_joints),
            pos=np.zeros(n_joints),
            vel=np.zeros(n_joints),
            kp=np.zeros(n_joints),
            kd=np.zeros(n_joints),
        )


class MotorChainRobot:
    """i2rt-authentic motor-chain robot controller."""

    def __init__(
        self,
        motor_chain: MotorChain,
        xml_path: Optional[str] = None,
        use_gravity_comp: bool = True,
        gravity: Optional[np.ndarray] = None,
        gravity_comp_factor: float = 1.0,
        gripper_index: Optional[int] = None,
        kp: Union[float, List[float]] = 10.0,
        kd: Union[float, List[float]] = 1.0,
        joint_limits: Optional[np.ndarray] = None,
        gripper_limits: Optional[np.ndarray] = None,
        limit_gripper_force: float = -1,
        clip_motor_torque: float = np.inf,
        gripper_type: GripperType = GripperType.CRANK_4310,
        temp_record_flag: bool = False,
        enable_gripper_calibration: bool = False,
        zero_gravity_mode: bool = True,
        test_torque: float = 0.5,
        test_duration: float = 2.0,
        position_threshold: float = 0.01,
        check_interval: float = 0.05,
    ) -> None:
        self.temp_record_flag = temp_record_flag
        if gripper_index is not None:
            assert gripper_index == len(motor_chain) - 1, (
                "Gripper index should be the last one, but got {gripper_index}"
            )
            if gripper_limits is None and enable_gripper_calibration:
                from lerobot.robots.yam_follower.utils import detect_gripper_limits

                logger.info("Auto-detecting gripper limits...")
                detected_limits = detect_gripper_limits(
                    motor_chain=motor_chain,
                    gripper_index=gripper_index,
                    test_torque=test_torque,
                    max_duration=test_duration,
                    position_threshold=position_threshold,
                    check_interval=check_interval,
                )
                gripper_limits = np.array(detected_limits)
                logger.info(f"Gripper limits auto-detected: {gripper_limits}")
            elif gripper_limits is None:
                raise ValueError(
                    f"{self}: Gripper limits are required if gripper index is provided and auto-calibration is disabled."
                )
            else:
                logger.info(f"Using provided gripper limits: {gripper_limits}")

        self._last_gripper_command_qpos = 1
        assert clip_motor_torque >= 0.0
        self._clip_motor_torque = clip_motor_torque
        self.motor_chain = motor_chain
        self.use_gravity_comp = use_gravity_comp
        self.gravity_comp_factor = gravity_comp_factor

        self._gripper_index = gripper_index
        self.remapper = JointMapper({}, len(motor_chain))
        self._gripper_limits = gripper_limits

        if self._gripper_index is not None:
            self._gripper_force_limiter = GripperForceLimiter(
                max_force=limit_gripper_force, gripper_type=gripper_type, kp=kp[gripper_index]
            )
            self._limit_gripper_force = limit_gripper_force
            self.remapper = JointMapper(
                index_range_map={gripper_index: gripper_limits},
                total_dofs=len(motor_chain),
            )

        self._kp = (
            np.array([kp] * len(motor_chain)) if isinstance(kp, float) else np.array(kp)
        )
        self._kd = (
            np.array([kd] * len(motor_chain)) if isinstance(kd, float) else np.array(kd)
        )

        self._joint_limits: Optional[np.ndarray] = None
        if xml_path is not None:
            self.xml_path = os.path.expanduser(xml_path)
            self.kdl = MuJoCoKDL(self.xml_path)
            if gravity is not None:
                self.kdl.set_gravity(gravity)
            self._joint_limits = self.kdl.joint_limits
        else:
            assert use_gravity_comp is False, "Gravity compensation requires a valid XML path."

        if joint_limits is not None:
            joint_limits = np.array(joint_limits)
            assert np.all(joint_limits[:, 0] < joint_limits[:, 1]), (
                "Lower joint limits must be smaller than upper limits"
            )
            self._joint_limits = joint_limits

        self._command_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._joint_state: Optional[JointStates] = None
        while self._joint_state is None:
            time.sleep(0.05)
            self._joint_state = self._motor_state_to_joint_state(self.motor_chain.read_states())
        self._commands = JointCommands.init_all_zero(len(motor_chain))
        self._check_current_qpos_in_joint_limits()

        self._stop_event = threading.Event()
        self._server_thread = threading.Thread(target=self.start_server, name="robot_server")
        self._server_thread.start()

        if not zero_gravity_mode:
            self.command_joint_pos(self._joint_state.pos)

    def __repr__(self) -> str:
        return f"MotorChainRobot(motor_chain={self.motor_chain})"

    def _check_current_qpos_in_joint_limits(self, buffer_rad: float = 0.1) -> None:
        if self._joint_state is None or self._joint_limits is None:
            raise RuntimeError(f"{self}: Joint limits:{self._joint_limits} or joint state:{self._joint_state} are not set.")

        current_pos = self._joint_state.pos
        if self._gripper_index is not None:
            arm_pos = current_pos[: self._gripper_index]
            arm_limits = self._joint_limits
        else:
            arm_pos = current_pos
            arm_limits = self._joint_limits

        lower_limits = arm_limits[:, 0] - buffer_rad
        upper_limits = arm_limits[:, 1] + buffer_rad
        lower_violations = arm_pos < lower_limits
        upper_violations = arm_pos > upper_limits

        if np.any(lower_violations) or np.any(upper_violations):
            violation_details = []
            for i, (pos, lower, upper) in enumerate(zip(arm_pos, lower_limits, upper_limits)):
                if pos < lower:
                    violation_details.append(f"Joint {i}: {pos:.4f} < {lower:.4f} (lower limit)")
                elif pos > upper:
                    violation_details.append(f"Joint {i}: {pos:.4f} > {upper:.4f} (upper limit)")
            violation_msg = "; ".join(violation_details)
            self.motor_chain.running = False
            raise RuntimeError(
                f"{self}: Joint limit violation detected: {violation_msg}, the root reason should be zero position "
                "offset. possible solution: 1. move the arm to zero position and power cycle the robot. "
                "2. Recalibrate the motor zero position."
            )

    def start_server(self) -> None:
        last_time = time.time()
        iteration_count = 0
        try:
            self.update()
            logging.info("initializing, ....")
            while not self._stop_event.is_set():
                current_time = time.time()
                elapsed_time = current_time - last_time
                self.update()
                if not self.motor_chain.running:
                    raise RuntimeError(
                        f"{self}: motor_chain_robot's motor chain is not running, exiting the robot server"
                    )
                time.sleep(0.004)
                iteration_count += 1
                if elapsed_time >= 10.0:
                    control_frequency = iteration_count / elapsed_time
                    logging.info(f"{self}: Grav Comp Control Frequency: {control_frequency:.2f} Hz")
                    if control_frequency < 100:
                        logging.warning(
                            f"{self}: Gravity compensation control loop is slow, current frequency: {control_frequency:.2f} Hz"
                        )
                    last_time = current_time
                    iteration_count = 0
        except Exception as exc:
            logging.error(f"{self}: robot server error, entering zero-torque mode: {exc}")
            try:
                self.zero_torque_mode()
            except Exception:
                pass
            self._stop_event.set()
            raise

    def update(self) -> None:
        with self._command_lock:
            joint_commands = copy.deepcopy(self._commands)
        with self._state_lock:
            g = self._compute_gravity_compensation(self._joint_state)
            motor_torques = joint_commands.torques + g * self.gravity_comp_factor
            motor_torques = np.clip(motor_torques, -self._clip_motor_torque, self._clip_motor_torque)

            if self._gripper_index is not None:
                if self._limit_gripper_force > 0 and self._joint_state is not None:
                    gripper_state = {
                        "target_qpos": joint_commands.pos[self._gripper_index],
                        "current_qpos": self.remapper.to_robot_joint_pos_space(self._joint_state.pos)[
                            self._gripper_index
                        ],
                        "current_qvel": self._joint_state.vel[self._gripper_index],
                        "current_eff": self._joint_state.eff[self._gripper_index],
                        "current_normalized_qpos": self._joint_state.pos[self._gripper_index],
                        "target_normalized_qpos": self.remapper.to_command_joint_pos_space(joint_commands.pos)[
                            self._gripper_index
                        ],
                        "last_command_qpos": self._last_gripper_command_qpos,
                    }
                    joint_commands.pos[self._gripper_index] = self._gripper_force_limiter.update(gripper_state)

                joint_commands.pos[self._gripper_index] = np.clip(
                    joint_commands.pos[self._gripper_index],
                    min(self._gripper_limits),
                    max(self._gripper_limits),
                )
                self._last_gripper_command_qpos = joint_commands.pos[self._gripper_index]

            if not self.motor_chain.start_thread_flag:
                self.motor_chain.set_commands(
                    motor_torques,
                    pos=joint_commands.pos,
                    vel=joint_commands.vel,
                    kp=joint_commands.kp,
                    kd=joint_commands.kd,
                )
                self.motor_chain.start_thread()
                self.motor_chain.start_thread_flag = True

            motor_state = self.motor_chain.set_commands(
                motor_torques,
                pos=joint_commands.pos,
                vel=joint_commands.vel,
                kp=joint_commands.kp,
                kd=joint_commands.kd,
            )
            self._joint_state = self._motor_state_to_joint_state(motor_state)
            self._check_current_qpos_in_joint_limits()

    def _motor_state_to_joint_state(self, motor_state: List[MotorInfo]) -> JointStates:
        names = [str(i) for i in range(len(motor_state))]
        pos = np.array([motor.pos for motor in motor_state])
        pos = self.remapper.to_command_joint_pos_space(pos)
        vel = np.array([motor.vel for motor in motor_state])
        vel = self.remapper.to_command_joint_vel_space(vel)
        eff = np.array([motor.eff for motor in motor_state])
        temp_mos = np.array([motor.temp_mos for motor in motor_state])
        temp_rotor = np.array([motor.temp_rotor for motor in motor_state])
        return JointStates(
            names=names,
            pos=pos,
            vel=vel,
            eff=eff,
            temp_mos=temp_mos,
            temp_rotor=temp_rotor,
        )

    def _compute_gravity_compensation(self, joint_state: Optional[JointStates]) -> np.ndarray:
        if joint_state is None or not self.use_gravity_comp:
            return np.zeros(len(self.motor_chain))
        q = joint_state.pos[: self._gripper_index] if self._gripper_index is not None else joint_state.pos
        t = self.kdl.compute_inverse_dynamics(q, np.zeros(q.shape), np.zeros(q.shape))
        if np.max(np.abs(t)) > 20.0:
            print([f"{s:.2f}" for s in t])
            raise RuntimeError(f"{self}: too large torques")
        if self._gripper_index is None:
            return t
        return np.append(t, 0.0)

    def num_dofs(self) -> int:
        return len(self.motor_chain)

    def get_joint_pos(self) -> np.ndarray:
        with self._state_lock:
            return self._joint_state.pos

    def _clip_robot_joint_pos_command(self, pos: np.ndarray) -> np.ndarray:
        if self._joint_limits is not None:
            if self._gripper_index is not None:
                pos[: self._gripper_index] = np.clip(
                    pos[: self._gripper_index],
                    self._joint_limits[:, 0],
                    self._joint_limits[:, 1],
                )
            else:
                pos = np.clip(pos, self._joint_limits[:, 0], self._joint_limits[:, 1])
        return pos

    def command_joint_pos(self, joint_pos: np.ndarray) -> None:
        pos = self._clip_robot_joint_pos_command(joint_pos)
        with self._command_lock:
            self._commands = JointCommands.init_all_zero(len(self.motor_chain))
            self._commands.pos = self.remapper.to_robot_joint_pos_space(pos)
            self._commands.kp = self._kp
            self._commands.kd = self._kd

    def command_joint_state(self, joint_state: Dict[str, np.ndarray]) -> None:
        pos = self._clip_robot_joint_pos_command(joint_state["pos"])
        vel = joint_state["vel"]
        self._commands = JointCommands.init_all_zero(len(self.motor_chain))
        kp = joint_state.get("kp", self._kp)
        kd = joint_state.get("kd", self._kd)
        with self._command_lock:
            self._commands.pos = self.remapper.to_robot_joint_pos_space(pos)
            self._commands.vel = self.remapper.to_robot_joint_vel_space(vel)
            self._commands.kp = kp
            self._commands.kd = kd

    def zero_torque_mode(self) -> None:
        logging.info(f"Entering zero_torque_mode for {self}")
        with self._command_lock:
            self._commands = JointCommands.init_all_zero(len(self.motor_chain))
            self._kp = np.zeros(len(self.motor_chain))
            self._kd = np.zeros(len(self.motor_chain))

    def get_observations(self) -> Dict[str, np.ndarray]:
        with self._state_lock:
            if self._gripper_index is None:
                result = {
                    "joint_pos": self._joint_state.pos,
                    "joint_vel": self._joint_state.vel,
                    "joint_eff": self._joint_state.eff,
                }
            else:
                result = {
                    "joint_pos": self._joint_state.pos[: self._gripper_index],
                    "gripper_pos": np.array([self._joint_state.pos[self._gripper_index]]),
                    "joint_vel": self._joint_state.vel,
                    "joint_eff": self._joint_state.eff,
                }
            if self.temp_record_flag:
                result["temp_mos"] = self._joint_state.temp_mos
                result["temp_rotor"] = self._joint_state.temp_rotor
            return result

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    def close(self) -> None:
        self._stop_event.set()
        self._server_thread.join()
        self.motor_chain.close()
        print("Robot closed with all torques set to zero.")

    def update_kp_kd(self, kp: np.ndarray, kd: np.ndarray) -> None:
        assert kp.shape == self._kp.shape == kd.shape
        self._kp = kp
        self._kd = kd

