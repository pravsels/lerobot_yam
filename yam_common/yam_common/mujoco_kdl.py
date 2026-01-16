"""
MuJoCo-based gravity compensation for YAM robot.

Faithfully ported from i2rt/utils/mujoco_utils.py.

This module uses MuJoCo's inverse dynamics to compute the torques needed
to compensate for gravity at any given joint configuration.
"""

import os
from typing import Optional

import numpy as np

try:
    import mujoco
except ImportError:
    mujoco = None


class MuJoCoKDL:
    """
    A class for computing inverse dynamics (gravity compensation) using MuJoCo.

    Source: i2rt/utils/mujoco_utils.py:MuJoCoKDL

    This class loads a MuJoCo XML model and uses mujoco.mj_inverse() to compute
    the torques needed to maintain a given configuration against gravity.

    For gravity compensation specifically:
    - Set qpos to current joint positions
    - Set qvel to zero (no velocity)
    - Set qacc to zero (no acceleration)
    - Call mj_inverse() to get the torques needed to maintain this configuration

    Usage:
        kdl = MuJoCoKDL("/path/to/yam.xml")
        gravity_torques = kdl.compute_gravity_compensation(current_joint_positions)
    """

    def __init__(self, xml_path: str, gravity: Optional[np.ndarray] = None):
        """
        Initialize MuJoCoKDL with a robot model.

        Args:
            xml_path: Path to MuJoCo XML model file
            gravity: Optional gravity vector (default: [0, 0, -9.81])
        """
        if mujoco is None:
            raise ImportError(
                "mujoco is required for MuJoCoKDL gravity compensation. "
                "Install with: pip install mujoco"
            )

        # Load model - expand user path as in i2rt
        self.xml_path = os.path.expanduser(xml_path)
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)

        # Set gravity (default: Earth gravity pointing down)
        if gravity is None:
            gravity = np.array([0.0, 0.0, -9.81])
        self.set_gravity(gravity)

        # Disable all collisions - we only care about dynamics, not contacts
        # Source: i2rt/utils/mujoco_utils.py lines 16-17
        self.model.geom_contype[:] = 0
        self.model.geom_conaffinity[:] = 0

        # Disable all joint limits - we handle limits separately
        # Source: i2rt/utils/mujoco_utils.py line 19
        self.model.jnt_limited[:] = 0

    @property
    def num_joints(self) -> int:
        """Number of joints in the model."""
        return self.model.nq

    @property
    def joint_limits(self) -> np.ndarray:
        """
        Get joint limits from model.

        Returns:
            Array of shape (num_joints, 2) with [min, max] for each joint
        """
        return self.model.jnt_range.copy()

    def set_gravity(self, gravity: np.ndarray) -> None:
        """
        Set the gravity vector for the robot.

        Source: i2rt/utils/mujoco_utils.py:MuJoCoKDL.set_gravity()

        Args:
            gravity: Gravity vector as a 3D NumPy array (e.g., [0, 0, -9.81])
        """
        assert gravity.shape == (3,), f"Gravity must be 3D vector, got shape {gravity.shape}"
        self.model.opt.gravity = gravity

    def compute_inverse_dynamics(
        self,
        q: np.ndarray,
        qdot: np.ndarray,
        qdotdot: np.ndarray,
    ) -> np.ndarray:
        """
        Compute inverse dynamics to get required joint torques.

        Source: i2rt/utils/mujoco_utils.py:MuJoCoKDL.compute_inverse_dynamics()

        This computes the torques needed to achieve the given acceleration
        from the given position and velocity, accounting for gravity,
        Coriolis forces, etc.

        Args:
            q: Joint positions (radians)
            qdot: Joint velocities (rad/s)
            qdotdot: Joint accelerations (rad/s^2)

        Returns:
            Joint torques (Nm) needed to achieve the given motion
        """
        assert len(q) == len(qdot) == len(qdotdot), (
            f"Input dimensions must match: q={len(q)}, qdot={len(qdot)}, qdotdot={len(qdotdot)}"
        )

        length = len(q)

        # Set state
        self.data.qpos[:length] = q
        self.data.qvel[:length] = qdot
        self.data.qacc[:length] = qdotdot

        # Compute inverse dynamics
        mujoco.mj_inverse(self.model, self.data)

        # Return the computed torques
        return self.data.qfrc_inverse[:length].copy()

    def compute_gravity_compensation(self, q: np.ndarray) -> np.ndarray:
        """
        Compute gravity compensation torques for a given joint configuration.

        This is a convenience method that calls compute_inverse_dynamics with
        zero velocity and zero acceleration, which gives the torques needed
        to hold the arm stationary against gravity.

        Args:
            q: Joint positions (radians)

        Returns:
            Gravity compensation torques (Nm)
        """
        zeros = np.zeros_like(q)
        return self.compute_inverse_dynamics(q, zeros, zeros)


def get_yam_mujoco_kdl(gripper_type: str = "crank_4310") -> MuJoCoKDL:
    """
    Get MuJoCoKDL instance for YAM robot with appropriate XML model.

    Args:
        gripper_type: Type of gripper ("crank_4310", "linear_3507", "linear_4310",
                      "yam_teaching_handle", "no_gripper")

    Returns:
        MuJoCoKDL instance configured for YAM robot
    """
    # Map gripper type to XML file
    # Source: i2rt/robots/utils.py:GripperType.get_xml_path()
    xml_paths = {
        "crank_4310": "yam.xml",
        "linear_3507": "yam_lw_gripper.xml",
        "linear_4310": "yam_4310_linear.xml",
        "yam_teaching_handle": "yam_teaching_handle.xml",
        "no_gripper": "yam_no_gripper.xml",
    }

    if gripper_type not in xml_paths:
        raise ValueError(f"Unknown gripper type: {gripper_type}. Valid types: {list(xml_paths.keys())}")

    # Try to find the XML file in several locations
    xml_filename = xml_paths[gripper_type]

    # Search paths (in order of preference):
    # 1. i2rt installation (if available)
    # 2. Local lerobot models directory
    search_paths = [
        # i2rt installation
        os.path.expanduser("~/Desktop/code/i2rt/i2rt/robot_models/yam"),
        # Relative to this file
        os.path.join(os.path.dirname(__file__), "robot_models"),
        # Package data
        os.path.join(os.path.dirname(__file__), "..", "..", "data", "robot_models", "yam"),
    ]

    for base_path in search_paths:
        xml_path = os.path.join(base_path, xml_filename)
        if os.path.exists(xml_path):
            return MuJoCoKDL(xml_path)

    raise FileNotFoundError(
        f"Could not find MuJoCo XML file '{xml_filename}' for YAM robot. "
        f"Searched paths: {search_paths}"
    )
