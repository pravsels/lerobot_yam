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
    gello_joint_positions,
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
        self.sync_read_values = {
            "Present_Temperature": 30,
            "Hardware_Error_Status": 0,
            "Bus_Watchdog": 1,
        }

    def read(self, name, motor):
        assert name == "Current_Limit"
        return 1750

    def write(self, name, motor, value, **kwargs):
        self.writes.append((name, motor, value, kwargs))

    def sync_read(self, name, motors=None, **kwargs):
        value = self.sync_read_values[name]
        return {motor: value for motor in (motors or list(self.motors))}

    def sync_write(self, name, values, **kwargs):
        self.writes.append((name, None, dict(values), kwargs))

    def enable_torque(self, motors=None, **kwargs):
        self.enabled.append(motors)

    def disable_torque(self, motors=None, **kwargs):
        self.disable_calls.append(motors)


def _bare_leader(config):
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
    leader._assist_velocity = np.zeros(6, dtype=np.float64)
    leader._live_assist_motors = []
    leader._last_health_check_time = 0.0
    leader._last_dry_run_log_time = 0.0
    return leader


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


def test_gello_joint_positions_applies_signs_in_radian_space():
    """Regression: sign flips must negate angles, not mirror normalized values.

    shoulder_lift has the asymmetric range (-0.15, 3.8). Mirroring the
    normalized value about the calibration midpoint (the old bug) put the
    home pose at ~3.65 rad instead of 0, so the gravity model was evaluated
    at a wildly wrong configuration.
    """
    ranges = DEFAULT_YAM_JOINT_RANGES_RAD
    signs = (1, -1, -1, -1, 1, 1)
    offsets = (0.0,) * 6

    # Normalized values that correspond to q_yam = 0 for every joint.
    normalized_home = [
        ((0.0 - lo) / (hi - lo)) * 200.0 - 100.0 for lo, hi in ranges
    ]
    q_home = gello_joint_positions(normalized_home, ranges, signs, offsets)
    assert np.allclose(q_home, 0.0, atol=1e-9)

    # And a non-home pose: q_urdf must equal sign * q_yam exactly.
    q_yam = normalized_to_joint_positions([10, 20, -30, 40, -50, 60], ranges)
    q_urdf = gello_joint_positions([10, 20, -30, 40, -50, 60], ranges, signs, offsets)
    assert np.allclose(q_urdf, np.asarray(signs, dtype=float) * q_yam)

    # Offsets shift the result in radians after the sign flip.
    q_offset = gello_joint_positions(
        normalized_home, ranges, signs, (0.5, 0, 0, 0, 0, -0.25)
    )
    assert q_offset[0] == pytest.approx(0.5)
    assert q_offset[5] == pytest.approx(-0.25)


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
    leader = _bare_leader(
        YAMLeaderConfig(
            gravity_assist=True,
            gripper_return=True,
            gravity_assist_current_limit_ma=200,
            gripper_return_current_ma=80,
        )
    )

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
    assert leader._live_assist_motors[-1] == "gripper"


def test_dry_run_configures_no_arm_motors_and_writes_no_currents():
    leader = _bare_leader(
        YAMLeaderConfig(gravity_assist=True, gravity_assist_dry_run=True)
    )

    leader._enable_assistance()

    assert leader._assist_enabled is True
    assert leader._live_assist_motors == []
    # No mode switches, watchdogs, or goal writes for the arm joints.
    assert all(motor != "shoulder_pan" for _, motor, _, _ in leader.bus.writes)
    assert leader.bus.enabled[-1] == []

    leader._update_assistance(
        {f"{name}.pos": 0.0 for name in leader.config.motors if name != "gripper"}
    )

    assert leader._assist_faulted is False
    sync_writes = [entry for entry in leader.bus.writes if entry[0] == "Goal_Current"]
    assert sync_writes == []


def test_live_assist_writes_signed_currents_from_urdf_torque():
    leader = _bare_leader(
        YAMLeaderConfig(gravity_assist=True, gravity_assist_gain=0.10)
    )
    leader._enable_assistance()
    leader.bus.writes.clear()

    leader._update_assistance(
        {f"{name}.pos": 0.0 for name in leader.config.motors if name != "gripper"}
    )

    assert leader._assist_faulted is False
    goal_writes = [entry for entry in leader.bus.writes if entry[0] == "Goal_Current"]
    assert len(goal_writes) == 1
    currents = goal_writes[0][2]
    assert set(currents) == {
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "wrist_yaw",
    }
    limit = leader.config.gravity_assist_current_limit_ma
    assert all(abs(value) <= limit for value in currents.values())
    assert any(value != 0 for value in currents.values())


def test_tripped_bus_watchdog_is_rearmed_and_gripper_spring_restored():
    leader = _bare_leader(
        YAMLeaderConfig(
            gravity_assist=True,
            gripper_return=True,
            gripper_return_current_ma=80,
        )
    )
    leader._enable_assistance()
    leader.bus.writes.clear()
    leader.bus.sync_read_values["Bus_Watchdog"] = 255  # -1: watchdog error

    leader._check_assist_health()

    writes = leader.bus.writes
    assert ("Bus_Watchdog", "shoulder_pan", 0, {"normalize": False}) in writes
    assert ("Bus_Watchdog", "shoulder_pan", 10, {"normalize": False}) in writes
    assert ("Goal_Current", "gripper", 80, {"normalize": False}) in writes
    assert ("Goal_Position", "gripper", 1000, {"normalize": False}) in writes

