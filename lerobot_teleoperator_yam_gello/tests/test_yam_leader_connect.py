from pathlib import Path

from lerobot.motors.dynamixel import OperatingMode

_YAM_LEADER = (
    Path(__file__).resolve().parents[1]
    / "lerobot_teleoperator_yam_gello"
    / "yam_leader.py"
)


def _connect_src() -> str:
    text = _YAM_LEADER.read_text()
    start = text.index("def connect(")
    end = text.index("\n    def ", start + 1)
    return text[start:end]


def _calibrate_src() -> str:
    text = _YAM_LEADER.read_text()
    start = text.index("def calibrate(")
    end = text.index("\n    def ", start + 1)
    return text[start:end]


def _configure_src() -> str:
    text = _YAM_LEADER.read_text()
    start = text.index("def configure(")
    end = text.index("\n    def ", start + 1)
    return text[start:end]


def test_connect_sets_position_mode_before_calibrating():
    src = _connect_src()
    assert src.index("self.configure()") < src.index("self.calibrate()")


def test_calibrate_sets_position_mode_before_reading_ticks():
    src = _calibrate_src()
    assert src.index("self.configure()") < src.index("Present_Position")


def test_configure_writes_position_operating_mode():
    src = _configure_src()
    assert "OperatingMode.POSITION" in src
    assert OperatingMode.POSITION.value == 3
