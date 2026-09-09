"""Forward-kinematics coverage for policy-ready YAM EEF poses."""

from __future__ import annotations

import numpy as np
import pytest

from yam_common import get_yam_mujoco_kdl


def test_compute_eef_pose_returns_tcp_xyz_and_rotation_6d() -> None:
    kdl = get_yam_mujoco_kdl("crank_4310")

    pose = kdl.compute_eef_pose(np.zeros(6))

    np.testing.assert_allclose(
        pose,
        [0.1103, 0.0, 0.164, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
        atol=1e-8,
    )


def test_compute_eef_pose_supports_grasp_site() -> None:
    kdl = get_yam_mujoco_kdl("crank_4310")

    pose = kdl.compute_eef_pose(np.zeros(6), site_name="grasp_site")

    np.testing.assert_allclose(
        pose,
        [0.245, 0.0, 0.164, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
        atol=1e-8,
    )


@pytest.mark.parametrize(
    ("joint_positions", "message"),
    [
        ([0.0] * 5, "exactly 6"),
        ([0.0] * 5 + [np.nan], "finite"),
    ],
)
def test_compute_eef_pose_rejects_invalid_joint_positions(
    joint_positions: list[float],
    message: str,
) -> None:
    kdl = get_yam_mujoco_kdl("crank_4310")

    with pytest.raises(ValueError, match=message):
        kdl.compute_eef_pose(joint_positions)


def test_compute_eef_pose_rejects_unknown_site() -> None:
    kdl = get_yam_mujoco_kdl("crank_4310")

    with pytest.raises(ValueError, match="unknown MuJoCo site"):
        kdl.compute_eef_pose(np.zeros(6), site_name="tool_tip")
