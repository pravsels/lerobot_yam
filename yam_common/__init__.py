"""Checkout shim so `import yam_common` from the repo root loads the package."""

from pathlib import Path

_inner = Path(__file__).resolve().parent / "yam_common"
__path__ = [str(_inner)]

from .mujoco_kdl import MuJoCoKDL, get_yam_mujoco_kdl, packaged_yam_model_dir
from .motor_chain_robot import MotorChainRobot
from .utils import GripperForceLimiter, GripperType, JointMapper
from .yam_arm import (
    ACTION_KEYS,
    YAMArm,
    YAMArmAlreadyConnectedError,
    YAMArmConfig,
    YAMArmError,
    YAMArmNotConnectedError,
    YAMArmUnhealthyError,
    normalize_from_physical,
    physical_from_normalized,
    prepare_normalized_action,
    probe_motors,
)

__all__ = [
    "ACTION_KEYS",
    "MuJoCoKDL",
    "YAMArm",
    "YAMArmAlreadyConnectedError",
    "YAMArmConfig",
    "YAMArmError",
    "YAMArmNotConnectedError",
    "YAMArmUnhealthyError",
    "get_yam_mujoco_kdl",
    "packaged_yam_model_dir",
    "MotorChainRobot",
    "GripperForceLimiter",
    "GripperType",
    "JointMapper",
    "normalize_from_physical",
    "physical_from_normalized",
    "prepare_normalized_action",
    "probe_motors",
]
