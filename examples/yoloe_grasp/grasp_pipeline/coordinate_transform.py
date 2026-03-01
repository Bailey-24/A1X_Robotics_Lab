#!/usr/bin/env python3
"""Coordinate Transform Module for Grasp Pipeline.

Transforms grasp poses from camera frame to robot base frame using:
    T_grasp_base = T_base_ee × T_ee_cam × T_grasp_cam

Where:
    T_base_ee  = FK from current joint angles (observation pose)
    T_ee_cam   = Hand-eye calibration (fixed, camera mounted on gripper)
    T_grasp_cam = Grasp pose from GraspNet (camera frame)

Usage (standalone test):
    python examples/yoloe_grasp/grasp_pipeline/coordinate_transform.py
"""
from __future__ import annotations

import sys
import os
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import yaml

logger = logging.getLogger(__name__)


def load_handeye_calibration(yaml_path: str | Path) -> np.ndarray:
    """Load hand-eye calibration from YAML file.

    Args:
        yaml_path: Path to handeye_calibration.yaml.

    Returns:
        T_ee_cam: 4x4 homogeneous transformation matrix (end-effector to camera).
    """
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    transform = data["transformation"]
    rotation = np.array(transform["rotation"])       # (3, 3)
    translation = np.array(transform["translation"])  # (3,)

    T_ee_cam = np.eye(4)
    T_ee_cam[:3, :3] = rotation
    T_ee_cam[:3, 3] = translation

    logger.info(f"Loaded hand-eye calibration from {yaml_path}")
    logger.info(f"  Translation: {translation}")
    logger.info(f"  Rotation det: {np.linalg.det(rotation):.6f}")

    return T_ee_cam


def compute_T_base_ee_from_fk(joint_angles: list[float]) -> np.ndarray:
    """Compute T_base_ee (base to end-effector) via PyRoki forward kinematics.

    Args:
        joint_angles: List of 6 arm joint angles in radians.

    Returns:
        T_base_ee: 4x4 homogeneous transformation matrix.
    """
    # Add pyroki to path
    pyroki_examples = str(Path(__file__).parent.parent.parent.parent / "pyroki" / "examples")
    if pyroki_examples not in sys.path:
        sys.path.insert(0, pyroki_examples)

    sdk_root = str(Path(__file__).parent.parent.parent.parent)
    if sdk_root not in sys.path:
        sys.path.insert(0, sdk_root)

    # Stub out pyroki.viewer to avoid viser → websockets.asyncio crash
    import types
    _viewer_stub = types.ModuleType("pyroki.viewer")
    sys.modules["pyroki.viewer"] = _viewer_stub

    import pyroki as pk
    import yourdfpy

    # Load URDF
    urdf_path = Path("/home/ubuntu/projects/A1Xsdk/install/mobiman/lib/mobiman/configs/urdfs/a1x.urdf")

    def resolve_package_uri(fname: str) -> str:
        package_prefix = "package://mobiman/"
        if fname.startswith(package_prefix):
            relative_path = fname[len(package_prefix):]
            return str(Path("/home/ubuntu/projects/A1Xsdk/install/mobiman/share/mobiman") / relative_path)
        return fname

    urdf = yourdfpy.URDF.load(urdf_path, filename_handler=resolve_package_uri)
    robot = pk.Robot.from_urdf(urdf)

    # Build full joint config (6 arm + 2 gripper)
    cfg = np.array(list(joint_angles) + [0.0, 0.0])

    # Compute FK for gripper_link
    target_link_name = "gripper_link"
    target_idx = robot.links.names.index(target_link_name)
    fk_result = robot.forward_kinematics(cfg)
    ee_pose = fk_result[target_idx]  # [wxyz(4), xyz(3)]

    # Convert to 4x4 matrix
    T_base_ee = pose_wxyz_xyz_to_matrix(ee_pose)

    logger.info(f"FK for joints {joint_angles}:")
    logger.info(f"  EE position: {T_base_ee[:3, 3]}")

    return T_base_ee


def pose_wxyz_xyz_to_matrix(pose: np.ndarray) -> np.ndarray:
    """Convert a pose [wxyz, xyz] to a 4x4 homogeneous matrix.

    Args:
        pose: Array of shape (7,) — [w, x, y, z, tx, ty, tz].

    Returns:
        4x4 homogeneous transformation matrix.
    """
    from scipy.spatial.transform import Rotation as R

    wxyz = pose[:4]
    xyz = pose[4:7]

    # scipy uses xyzw format
    quat_xyzw = [wxyz[1], wxyz[2], wxyz[3], wxyz[0]]
    rot = R.from_quat(quat_xyzw).as_matrix()

    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = xyz
    return T


def grasp_to_T_matrix(rotation_matrix: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Build 4x4 homogeneous matrix from grasp rotation and translation.

    Args:
        rotation_matrix: (3, 3) rotation matrix.
        translation: (3,) translation vector in meters.

    Returns:
        T_grasp_cam: 4x4 homogeneous transformation matrix.
    """
    T = np.eye(4)
    T[:3, :3] = rotation_matrix
    T[:3, 3] = translation
    return T


def apply_tcp_offset(
    T_grasp_base: np.ndarray,
    tcp_offset: list[float] | np.ndarray,
) -> np.ndarray:
    """Apply TCP offset to convert from finger-contact-center to gripper_link target.

    GraspNet predicts the grasp at the finger contact center, but IK targets
    gripper_link origin. The TCP offset is the vector from gripper_link origin
    to the finger contact center in the gripper_link local frame.

    To make gripper_link's fingers reach the grasp point, we subtract the
    TCP offset (in the grasp frame) from the grasp position.

    Args:
        T_grasp_base: 4x4 grasp pose in base frame (finger contact center).
        tcp_offset: [x, y, z] offset from gripper_link to finger center
                    in gripper_link local frame (meters).

    Returns:
        T_ee_target: 4x4 target pose for gripper_link in base frame.
    """
    tcp_offset = np.array(tcp_offset)
    T_ee_target = T_grasp_base.copy()
    # The TCP offset is in the local frame of the grasp, so transform it
    # to world frame using the grasp rotation, then subtract
    world_offset = T_grasp_base[:3, :3] @ tcp_offset
    T_ee_target[:3, 3] -= world_offset
    logger.info(f"  TCP offset applied: {tcp_offset} → world offset {world_offset}")
    return T_ee_target


def transform_grasp_to_base(
    T_grasp_cam: np.ndarray,
    T_base_ee: np.ndarray,
    T_ee_cam: np.ndarray,
    tcp_offset: list[float] | np.ndarray | None = None,
) -> np.ndarray:
    """Transform a grasp pose from camera frame to robot base frame.

    Formula: T_grasp_base = T_base_ee × T_ee_cam × T_grasp_cam

    If tcp_offset is provided, adjusts the target so that the gripper's
    finger contact center (not gripper_link origin) reaches the grasp point.

    Args:
        T_grasp_cam: 4x4 grasp pose in camera frame.
        T_base_ee: 4x4 end-effector pose in base frame (from FK).
        T_ee_cam: 4x4 camera pose relative to end-effector (hand-eye calibration).
        tcp_offset: Optional [x, y, z] offset from gripper_link to finger center
                    in gripper_link local frame (meters).

    Returns:
        T_grasp_base: 4x4 grasp pose in robot base frame (adjusted for TCP if given).
    """
    T_grasp_base = T_base_ee @ T_ee_cam @ T_grasp_cam

    if tcp_offset is not None:
        T_grasp_base = apply_tcp_offset(T_grasp_base, tcp_offset)

    return T_grasp_base


def matrix_to_position_wxyz(T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Extract position and quaternion (wxyz) from a 4x4 matrix.

    Args:
        T: 4x4 homogeneous transformation matrix.

    Returns:
        Tuple of (position, quaternion_wxyz):
            - position: (3,) array [x, y, z] in meters
            - quaternion_wxyz: (4,) array [w, x, y, z]
    """
    from scipy.spatial.transform import Rotation as R

    position = T[:3, 3].copy()
    rot = R.from_matrix(T[:3, :3])
    quat_xyzw = rot.as_quat()  # [x, y, z, w]
    quaternion_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

    return position, quaternion_wxyz


def compute_pre_grasp_pose(T_grasp_base: np.ndarray, offset_m: float = 0.05) -> np.ndarray:
    """Compute pre-grasp pose by offsetting along the approach direction.

    The approach direction is the z-axis of the grasp rotation matrix.
    The pre-grasp pose is offset backwards (negative z) from the grasp.

    Args:
        T_grasp_base: 4x4 grasp pose in base frame.
        offset_m: Offset distance in meters (positive = move back along approach).

    Returns:
        T_pre_grasp: 4x4 pre-grasp pose in base frame.
    """
    T_pre = T_grasp_base.copy()
    # The approach direction is the z-column of the rotation matrix
    approach_dir = T_grasp_base[:3, 2]
    T_pre[:3, 3] -= offset_m * approach_dir
    return T_pre


def compute_lift_pose(T_grasp_base: np.ndarray, lift_height_m: float = 0.10) -> np.ndarray:
    """Compute post-grasp lift pose by moving straight up in base frame.

    Args:
        T_grasp_base: 4x4 grasp pose in base frame.
        lift_height_m: Lift height in meters (world z-axis).

    Returns:
        T_lift: 4x4 lift pose in base frame.
    """
    T_lift = T_grasp_base.copy()
    T_lift[2, 3] += lift_height_m  # Move up in world z
    return T_lift


# ─── Standalone test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    project_root = Path(__file__).parent.parent.parent.parent

    print("Coordinate Transform Test")
    print("=" * 40)

    # Load hand-eye calibration
    handeye_path = project_root / "examples" / "handeye" / "handeye_calibration.yaml"
    T_ee_cam = load_handeye_calibration(handeye_path)
    print(f"\nT_ee_cam:\n{T_ee_cam}")

    # Compute FK for observation pose
    obs_joints = [0.12, 1.0, -0.93, 0.83, 0.0, 0.0]
    T_base_ee = compute_T_base_ee_from_fk(obs_joints)
    print(f"\nT_base_ee (observation pose):\n{T_base_ee}")

    # Simulate a grasp in camera frame
    T_grasp_cam = np.eye(4)
    T_grasp_cam[:3, 3] = [0.0, 0.0, 0.3]  # 30cm in front of camera
    print(f"\nT_grasp_cam (simulated):\n{T_grasp_cam}")

    # Transform to base frame
    T_grasp_base = transform_grasp_to_base(T_grasp_cam, T_base_ee, T_ee_cam)
    print(f"\nT_grasp_base:\n{T_grasp_base}")

    # Extract position and quaternion
    pos, wxyz = matrix_to_position_wxyz(T_grasp_base)
    print(f"\nGrasp in base frame:")
    print(f"  Position: x={pos[0]:.4f}, y={pos[1]:.4f}, z={pos[2]:.4f}")
    print(f"  Quaternion (wxyz): {wxyz}")

    # Pre-grasp and lift poses
    T_pre = compute_pre_grasp_pose(T_grasp_base, 0.05)
    T_lift = compute_lift_pose(T_grasp_base, 0.10)
    pre_pos, _ = matrix_to_position_wxyz(T_pre)
    lift_pos, _ = matrix_to_position_wxyz(T_lift)
    print(f"\n  Pre-grasp position: {pre_pos}")
    print(f"  Lift position: {lift_pos}")
