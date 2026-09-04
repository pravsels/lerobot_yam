from .can_interface import CanInterface
from .dm_driver import (
    ControlMode,
    DMChainCanInterface,
    DMSingleMotorCanInterface,
    MultiDMChainCanInterface,
    probe_motors,
)
from .utils import (
    MotorConstants,
    MotorErrorCode,
    MotorInfo,
    MotorType,
    ReceiveMode,
    float_to_uint,
    uint_to_float,
)

__all__ = [
    "CanInterface",
    "ControlMode",
    "DMChainCanInterface",
    "DMMotorsBus",
    "DMSingleMotorCanInterface",
    "MotorConstants",
    "MotorErrorCode",
    "MotorInfo",
    "MotorType",
    "MultiDMChainCanInterface",
    "ReceiveMode",
    "float_to_uint",
    "probe_motors",
    "uint_to_float",
]


def __getattr__(name: str):
    if name == "DMMotorsBus":
        from .dm import DMMotorsBus

        return DMMotorsBus
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
