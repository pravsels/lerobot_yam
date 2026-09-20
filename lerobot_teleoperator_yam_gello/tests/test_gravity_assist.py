import numpy as np
import pytest
from lerobot.motors import MotorCalibration

from lerobot_teleoperator_yam_gello.config_yam_leader import (
    DEFAULT_YAM_JOINT_RANGES_RAD,
    YAMLeaderConfig,
)
from lerobot_teleoperator_yam_gello.gravity_assist import (
    GelloGravityModel,
    default_urdf_path,
    normalized_to_joint_positions,
    torque_to_current_ma,
)
from lerobot_teleoperator_yam_gello.yam_leader import YAMLeader


class _FakeBus:
    def __init__(self, motors):
        self.motors = motors
        self.is_connected = True
        self.writes = []
        self.enabled = []
        self.disable_calls = []

    def read(self, name, motor):
        assert name == "Current_Limit"
        return 1750

    def write(self, name, motor, value, **kwargs):
        self.writes.append((name, motor, value, kwargs))

    def sync_write(self, name, values, **kwargs):
        self.writes.append((name, None, dict(values), kwargs))

    def enable_torque(self, motors=None, **kwargs):
        self.enabled.append(motors)

    def disable_torque(self, motors=None, **kwargs):
        self.disable_calls.append(motors)


def test_gravity_model_asset_is_packaged_and_returns_finite_torques():
    assert default_urdf_path().is_file()
    model = GelloGravityModel()

    torque = model.gravity_torques(np.zeros(6))

    assert torque.shape == (6,)
    assert np.all(np.isfinite(torque))
    assert np.max(np.abs(torque)) < 2.0


def test_normalized_joint_mapping_uses_yam_ranges():
    q = normalized_to_joint_positions(
        [-100, 100, 0, -100, 100, 0],
        DEFAULT_YAM_JOINT_RANGES_RAD,
    )

    assert q[0] == pytest.approx(DEFAULT_YAM_JOINT_RANGES_RAD[0][0])
    assert q[1] == pytest.approx(DEFAULT_YAM_JOINT_RANGES_RAD[1][1])
    assert q[2] == pytest.approx(sum(DEFAULT_YAM_JOINT_RANGES_RAD[2]) / 2)


def test_xl330_current_conversion_is_hard_clipped():
    assert torque_to_current_ma(1.0, "xl330-m288", 250) == 250
    assert torque_to_current_ma(-1.0, "xl330-m288", 250) == -250
    assert torque_to_current_ma(0.0354, "xl330-m288", 250) == 100
    with pytest.raises(ValueError, match="does not support"):
        torque_to_current_ma(0.1, "unknown", 250)


def test_assistance_is_opt_in_and_limits_are_conservative():
    config = YAMLeaderConfig()

    assert config.gravity_assist is False
    assert config.gripper_return is False
    assert config.gravity_assist_current_limit_ma <= 250
    assert config.gripper_return_current_ma <= 100
    with pytest.raises(ValueError, match="gravity_assist_current_limit_ma"):
        YAMLeaderConfig(gravity_assist_current_limit_ma=501)


def test_tick_wrap_chooses_equivalent_nearest_calibration():
    leader = object.__new__(YAMLeader)
    leader.calibration = {
        "elbow_flex": MotorCalibration(
            id=3,
            drive_mode=0,
            homing_offset=0,
            range_min=-13,
            range_max=3121,
        )
    }

    assert leader._tick_near_calibration("elbow_flex", 4102) == 6
    assert leader._tick_near_calibration("elbow_flex", -6) == -6


@pytest.mark.parametrize(
    ("drive_mode", "expected_open_tick"),
    [(0, 1000), (1, 1000)],
)
def test_gripper_return_targets_physical_open_endpoint(
    drive_mode, expected_open_tick
):
    leader = object.__new__(YAMLeader)
    leader.calibration = {
        "gripper": MotorCalibration(
            id=7,
            drive_mode=drive_mode,
            homing_offset=0,
            range_min=100,
            range_max=1000,
        )
    }

    assert leader._gripper_open_tick() == expected_open_tick


def test_enable_assistance_uses_current_modes_limits_watchdog_and_open_target():
    config = YAMLeaderConfig(
        gravity_assist=True,
        gripper_return=True,
        gravity_assist_current_limit_ma=200,
        gripper_return_current_ma=80,
    )
    leader = object.__new__(YAMLeader)
    leader.config = config
    leader.bus = _FakeBus(config.motors)
    leader.calibration = {
        name: MotorCalibration(
            id=motor.id,
            drive_mode=0,
            homing_offset=0,
            range_min=100,
            range_max=1000,
        )
        for name, motor in config.motors.items()
    }
    leader._gravity_model = None
    leader._assist_enabled = False
    leader._assist_faulted = False
    leader._last_assist_q = None
    leader._last_assist_time = None
    leader._last_health_check_time = 0.0

    leader._enable_assistance()

    assert leader._assist_enabled is True
    assert (
        "Operating_Mode",
        "shoulder_pan",
        0,
        {},
    ) in leader.bus.writes
    assert ("Operating_Mode", "gripper", 5, {}) in leader.bus.writes
    assert ("Goal_Position", "gripper", 1000, {"normalize": False}) in leader.bus.writes
    assert (
        "Goal_Current",
        "gripper",
        80,
        {"normalize": False},
    ) in leader.bus.writes
    assert ("Bus_Watchdog", "shoulder_pan", 10, {}) in leader.bus.writes
    assert leader.bus.enabled[-1] == [
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "wrist_yaw",
        "gripper",
    ]

