"""Versioned hardware profiles for YAM GELLO active assistance.

These values belong together: changing encoder ranges or signs changes the
coordinate frame in which the offsets were identified, and effective masses
were fitted against that same model. Keep profiles immutable and add a new
version when a hardware build or system-identification result changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GravityAssistProfile:
    name: str
    description: str
    model_asset: str
    gravity_assist: bool
    gravity_assist_gain: float
    gravity_assist_current_limit_ma: int
    gravity_assist_joints: tuple[str, ...]
    gravity_assist_damping_nm_per_rad_s: float
    gravity_joint_ranges_rad: tuple[tuple[float, float], ...]
    gravity_joint_signs: tuple[int, ...]
    gravity_joint_offsets_rad: tuple[float, ...]
    gravity_urdf_z_sign: int
    gravity_link_masses_kg: tuple[float, ...]
    follower_limit_buffer_rad: float
    gripper_return: bool
    gripper_return_current_ma: int


# Identified on the bimanual TRLC-DK1 GELLO pair in September 2026. The
# upstream active-GELLO URDF supplies the serial-chain kinematics and COMs;
# these are effective masses and encoder/model transforms identified for the
# lighter DK1 leader build. They are model parameters, not a claim that each
# printed assembly weighs exactly the listed value.
TRLC_DK1_V1 = GravityAssistProfile(
    name="trlc_dk1_v1",
    description=(
        "TRLC-DK1 YAM leader: identified encoder mapping, effective link masses, "
        "gravity assistance, and spring-open trigger"
    ),
    model_asset="yam_active_gello.urdf",
    gravity_assist=True,
    gravity_assist_gain=0.8,
    gravity_assist_current_limit_ma=250,
    gravity_assist_joints=("shoulder_lift", "elbow_flex", "wrist_flex"),
    gravity_assist_damping_nm_per_rad_s=0.003,
    gravity_joint_ranges_rad=(
        (-2.767, 3.28),
        (-0.09, 2.28),
        (-0.15, 3.28),
        (-1.221, 1.221),
        (-1.72, 1.72),
        (-2.24, 2.24),
    ),
    gravity_joint_signs=(1, 1, -1, -1, 1, 1),
    gravity_joint_offsets_rad=(0.0, -2.61, -0.973, 0.07, 0.0, 0.0),
    gravity_urdf_z_sign=-1,
    gravity_link_masses_kg=(0.142, 0.012, 0.016, 0.008, 0.005, 0.003, 0.013),
    follower_limit_buffer_rad=0.15,
    gripper_return=True,
    gripper_return_current_ma=80,
)


# Explicit escape hatch for passive/read-only leaders. Custom builds can also
# select this and override individual fields, or add a named profile here once
# their system identification is stable.
PASSIVE = GravityAssistProfile(
    name="passive",
    description="Torque-disabled GELLO leader; position sensing only",
    model_asset="yam_active_gello.urdf",
    gravity_assist=False,
    gravity_assist_gain=0.0,
    gravity_assist_current_limit_ma=250,
    gravity_assist_joints=("shoulder_lift", "elbow_flex", "wrist_flex"),
    gravity_assist_damping_nm_per_rad_s=0.003,
    gravity_joint_ranges_rad=TRLC_DK1_V1.gravity_joint_ranges_rad,
    gravity_joint_signs=TRLC_DK1_V1.gravity_joint_signs,
    gravity_joint_offsets_rad=TRLC_DK1_V1.gravity_joint_offsets_rad,
    gravity_urdf_z_sign=TRLC_DK1_V1.gravity_urdf_z_sign,
    gravity_link_masses_kg=TRLC_DK1_V1.gravity_link_masses_kg,
    follower_limit_buffer_rad=TRLC_DK1_V1.follower_limit_buffer_rad,
    gripper_return=False,
    gripper_return_current_ma=80,
)


DEFAULT_GRAVITY_PROFILE = TRLC_DK1_V1.name
GRAVITY_ASSIST_PROFILES = {
    TRLC_DK1_V1.name: TRLC_DK1_V1,
    PASSIVE.name: PASSIVE,
}


def get_gravity_assist_profile(name: str) -> GravityAssistProfile:
    try:
        return GRAVITY_ASSIST_PROFILES[name]
    except KeyError as exc:
        choices = ", ".join(sorted(GRAVITY_ASSIST_PROFILES))
        raise ValueError(f"unknown gravity_profile {name!r}; choose one of: {choices}") from exc


def gravity_profile_model_path(name: str) -> Path:
    profile = get_gravity_assist_profile(name)
    path = Path(__file__).with_name("assets") / profile.model_asset
    if not path.is_file():
        raise FileNotFoundError(
            f"gravity profile {name!r} model asset is missing: {path}"
        )
    return path
