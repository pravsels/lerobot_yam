"""Small, dependency-light gravity model for the YAM GELLO leader.

The model is adapted from gello_software's FACTR implementation, but computes
the gradient of URDF potential energy directly with NumPy. This avoids making
Pinocchio a mandatory dependency on the macOS teleoperation laptop.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

_G = 9.80665
_FINITE_DIFFERENCE_RAD = 1e-5

# Approximate output torque constants at 5 V from the ROBOTIS XL330 datasheets.
# Goal_Current is expressed in approximately 1 mA units.
_TORQUE_NM_PER_AMP = {
    "xl330-m288": 0.354,
    "xl330-m077": 0.146,
}


def default_urdf_path() -> Path:
    return Path(__file__).with_name("assets") / "yam_active_gello.urdf"


def torque_to_current_ma(torque_nm: float, model: str, limit_ma: int) -> int:
    """Convert output torque to a conservatively clipped XL330 current."""
    nm_per_amp = _TORQUE_NM_PER_AMP.get(model)
    if nm_per_amp is None:
        raise ValueError(f"gravity assistance does not support motor model {model!r}")
    current_ma = float(torque_nm) * 1000.0 / nm_per_amp
    return int(round(np.clip(current_ma, -abs(limit_ma), abs(limit_ma))))


def normalized_to_joint_positions(
    normalized: Sequence[float],
    ranges_rad: Sequence[tuple[float, float]],
) -> np.ndarray:
    """Map six LeRobot ``[-100, 100]`` values to physical joint radians."""
    values = np.asarray(normalized, dtype=np.float64)
    if values.shape != (6,) or len(ranges_rad) != 6:
        raise ValueError("gravity model requires six joint values and six ranges")
    bounded = np.clip(values, -100.0, 100.0)
    lows = np.asarray([pair[0] for pair in ranges_rad], dtype=np.float64)
    highs = np.asarray([pair[1] for pair in ranges_rad], dtype=np.float64)
    return lows + ((bounded + 100.0) / 200.0) * (highs - lows)


def gello_joint_positions(
    normalized: Sequence[float],
    ranges_rad: Sequence[tuple[float, float]],
    signs: Sequence[int],
    offsets_rad: Sequence[float],
) -> np.ndarray:
    """Map normalized leader values into the GELLO URDF's joint coordinates.

    Signs and offsets are applied in *radian* space: ``q_urdf = s*q_yam + o``.
    Applying a sign to the normalized value instead would mirror the joint
    about its calibration midpoint — a different (wrong) pose whenever the
    joint range is asymmetric, which all the YAM lift joints are.
    """
    q_yam = normalized_to_joint_positions(normalized, ranges_rad)
    signs_arr = np.asarray(signs, dtype=np.float64)
    offsets_arr = np.asarray(offsets_rad, dtype=np.float64)
    if signs_arr.shape != (6,) or offsets_arr.shape != (6,):
        raise ValueError("gravity model requires six signs and six offsets")
    return signs_arr * q_yam + offsets_arr


@dataclass(frozen=True)
class _LinkInertia:
    mass: float
    com: np.ndarray


@dataclass(frozen=True)
class _Joint:
    parent: str
    child: str
    xyz: np.ndarray
    rotation: np.ndarray
    axis: np.ndarray


class GelloGravityModel:
    """Compute six joint gravity torques from the active-GELLO URDF.

    ``gravity_z_sign`` states which way the URDF's +z axis points physically:
    +1 for up, -1 for down. The bundled ``yam_active_gello`` Onshape export is
    z-down for the standard tabletop mount (verified on hardware: computing
    with z-up made the assist push the arm toward its fallen rest pose —
    every holding torque negated). Getting this bit wrong flips all torques,
    so validate new hardware with the dry-run mode before applying current.
    """

    def __init__(
        self,
        urdf_path: str | Path | None = None,
        gravity_z_sign: int = -1,
    ) -> None:
        if int(gravity_z_sign) not in (-1, 1):
            raise ValueError("gravity_z_sign must be -1 or 1")
        self._gravity_z_sign = int(gravity_z_sign)
        self.urdf_path = Path(urdf_path) if urdf_path else default_urdf_path()
        self._links, self._joints = _read_urdf(self.urdf_path)
        if len(self._joints) != 6:
            raise ValueError(f"expected six revolute joints in {self.urdf_path}")

    def potential_energy(self, joint_positions: Sequence[float]) -> float:
        q = np.asarray(joint_positions, dtype=np.float64)
        if q.shape != (6,) or not np.all(np.isfinite(q)):
            raise ValueError("joint_positions must contain six finite values")

        transforms: dict[str, np.ndarray] = {}
        root = self._joints[0].parent
        transforms[root] = np.eye(4, dtype=np.float64)
        energy = self._link_energy(root, transforms[root])

        for joint, angle in zip(self._joints, q, strict=True):
            parent_tf = transforms[joint.parent]
            joint_tf = _transform(joint.xyz, joint.rotation)
            motion_tf = np.eye(4, dtype=np.float64)
            motion_tf[:3, :3] = _axis_angle(joint.axis, float(angle))
            child_tf = parent_tf @ joint_tf @ motion_tf
            transforms[joint.child] = child_tf
            energy += self._link_energy(joint.child, child_tf)
        return float(energy)

    def gravity_torques(self, joint_positions: Sequence[float]) -> np.ndarray:
        """Return holding torque ``dU/dq`` in N·m for each joint."""
        q = np.asarray(joint_positions, dtype=np.float64)
        if q.shape != (6,) or not np.all(np.isfinite(q)):
            raise ValueError("joint_positions must contain six finite values")
        torques = np.empty(6, dtype=np.float64)
        for index in range(6):
            plus = q.copy()
            minus = q.copy()
            plus[index] += _FINITE_DIFFERENCE_RAD
            minus[index] -= _FINITE_DIFFERENCE_RAD
            torques[index] = (
                self.potential_energy(plus) - self.potential_energy(minus)
            ) / (2.0 * _FINITE_DIFFERENCE_RAD)
        return torques

    def _link_energy(self, link_name: str, transform: np.ndarray) -> float:
        inertia = self._links.get(link_name)
        if inertia is None or inertia.mass == 0:
            return 0.0
        com_h = np.append(inertia.com, 1.0)
        world_com = transform @ com_h
        # Physical height is the URDF z coordinate times the frame's z sign.
        return inertia.mass * _G * self._gravity_z_sign * float(world_com[2])


def _read_urdf(path: Path) -> tuple[dict[str, _LinkInertia], list[_Joint]]:
    root = ET.parse(path).getroot()
    links: dict[str, _LinkInertia] = {}
    for element in root.findall("link"):
        inertial = element.find("inertial")
        if inertial is None:
            continue
        mass_element = inertial.find("mass")
        origin = inertial.find("origin")
        if mass_element is None:
            continue
        links[element.attrib["name"]] = _LinkInertia(
            mass=float(mass_element.attrib["value"]),
            com=_vector(origin.attrib.get("xyz", "0 0 0") if origin is not None else "0 0 0"),
        )

    by_parent: dict[str, _Joint] = {}
    children: set[str] = set()
    for element in root.findall("joint"):
        if element.attrib.get("type") not in {"revolute", "continuous"}:
            continue
        parent_element = element.find("parent")
        child_element = element.find("child")
        axis_element = element.find("axis")
        origin = element.find("origin")
        if parent_element is None or child_element is None:
            raise ValueError(f"joint {element.attrib.get('name')} has no parent/child")
        parent = parent_element.attrib["link"]
        child = child_element.attrib["link"]
        xyz = _vector(origin.attrib.get("xyz", "0 0 0") if origin is not None else "0 0 0")
        rpy = _vector(origin.attrib.get("rpy", "0 0 0") if origin is not None else "0 0 0")
        axis = _vector(
            axis_element.attrib.get("xyz", "1 0 0") if axis_element is not None else "1 0 0"
        )
        norm = float(np.linalg.norm(axis))
        if norm == 0:
            raise ValueError(f"joint {element.attrib.get('name')} has zero axis")
        joint = _Joint(parent, child, xyz, _rpy_matrix(rpy), axis / norm)
        if parent in by_parent:
            raise ValueError("gravity model expects an unbranched serial chain")
        by_parent[parent] = joint
        children.add(child)

    roots = set(by_parent) - children
    if len(roots) != 1:
        raise ValueError("gravity model expects one serial-chain root")
    ordered: list[_Joint] = []
    link = roots.pop()
    while link in by_parent:
        joint = by_parent[link]
        ordered.append(joint)
        link = joint.child
    return links, ordered


def _vector(text: str) -> np.ndarray:
    values = np.asarray([float(value) for value in text.split()], dtype=np.float64)
    if values.shape != (3,):
        raise ValueError(f"expected xyz/rpy triple, got {text!r}")
    return values


def _rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array(((1, 0, 0), (0, cr, -sr), (0, sr, cr)), dtype=np.float64)
    ry = np.array(((cp, 0, sp), (0, 1, 0), (-sp, 0, cp)), dtype=np.float64)
    rz = np.array(((cy, -sy, 0), (sy, cy, 0), (0, 0, 1)), dtype=np.float64)
    return rz @ ry @ rx


def _axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    x, y, z = axis
    c, s = math.cos(angle), math.sin(angle)
    one_minus_c = 1.0 - c
    return np.array(
        (
            (
                c + x * x * one_minus_c,
                x * y * one_minus_c - z * s,
                x * z * one_minus_c + y * s,
            ),
            (
                y * x * one_minus_c + z * s,
                c + y * y * one_minus_c,
                y * z * one_minus_c - x * s,
            ),
            (
                z * x * one_minus_c - y * s,
                z * y * one_minus_c + x * s,
                c + z * z * one_minus_c,
            ),
        ),
        dtype=np.float64,
    )


def _transform(xyz: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = xyz
    return transform

