#!/usr/bin/env python3
"""AnyGrasp End-to-End Grasp Execution for A1X.

Orchestrates the complete pipeline:
    1. Initialize robot
    2. Move to observation pose
    3. Capture RGBD from D405
    4. (Optional) SAM3 target filter + build point cloud
    5. Run AnyGrasp inference + select best grasp
    6. Transform grasp from camera frame to robot base frame
    7. Solve IK and execute grasp motion (pre-grasp → grasp → lift)
    8. Place object and return to observation

Usage:
    python examples/anygrasp_grasp/anygrasp_grasp.py
    python examples/anygrasp_grasp/anygrasp_grasp.py --target-name banana
    python examples/anygrasp_grasp/anygrasp_grasp.py --dry-run --debug
    python examples/anygrasp_grasp/anygrasp_grasp.py --use-topdown

Prerequisites:
    - CAN bus configured (1 Mbps / 5 Mbps FD)
    - ROS 2 environment sourced (or using a1x_control auto-launch)
    - Hand-eye calibration completed (handeye_calibration.yaml)
    - AnyGrasp checkpoint available
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import yaml

# ── Path setup ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Reuse existing grasp_pipeline modules ───────────────────────────────────
from examples.yoloe_grasp.grasp_pipeline.capture_rgbd import RGBDCapture
from examples.yoloe_grasp.grasp_pipeline.sam3_detector import Sam3Detector
from examples.yoloe_grasp.grasp_pipeline.coordinate_transform import (
    load_handeye_calibration,
    compute_T_base_ee_from_fk,
    grasp_to_T_matrix,
    transform_grasp_to_base,
    matrix_to_position_wxyz,
    compute_pre_grasp_pose,
    compute_lift_pose,
)
from examples.yoloe_grasp.grasp_pipeline.ik_executor import IKExecutor

# ── Robot control ───────────────────────────────────────────────────────────
import a1x_control

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("anygrasp_grasp")


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def load_config(config_path: str | Path) -> dict:
    """Load pipeline config from YAML."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def move_to_joint_pose(
    controller: a1x_control.JointController,
    target_joints: list[float],
    label: str = "target",
    dry_run: bool = False,
    rate_hz: float = 20.0,
    interpolation_type: str = "cosine",
) -> None:
    """Move robot to a joint-angle pose with smooth interpolation."""
    print(f"\n  Moving to {label} pose: {target_joints}")
    if dry_run:
        print("  [DRY RUN] Skipping motion")
        return

    success = controller.move_to_position_smooth(
        target_joints, 
        steps=60, 
        rate_hz=rate_hz,
        interpolation_type=interpolation_type,
        wait_for_convergence=True
    )
    if success:
        print(f"  ✓ Smooth motion complete")
        time.sleep(0.5)
    else:
        raise RuntimeError(f"Failed to execute smooth motion for {label}")


def build_point_cloud(
    colors: np.ndarray,
    depths: np.ndarray,
    fx: float, fy: float,
    cx: float, cy: float,
    scale: float,
    z_max: float = 1.0,
    object_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Deproject depth image to 3D point cloud.

    Args:
        colors: (H, W, 3) float32 color image, values in [0, 1].
        depths: (H, W) raw depth image (uint16 or similar).
        fx, fy, cx, cy: Camera intrinsic parameters.
        scale: Depth scale divisor (raw_depth / scale = meters).
        z_max: Maximum depth in meters; points beyond are discarded.
        object_mask: Optional (H, W) bool mask. When provided, only pixels
            where mask is True are included in the point cloud.

    Returns:
        points: (N, 3) float32 array of 3D points.
        colors: (N, 3) float32 array of corresponding colors.
    """
    H, W = depths.shape
    xmap, ymap = np.meshgrid(np.arange(W), np.arange(H))

    points_z = depths.astype(np.float32) / scale
    points_x = (xmap - cx) / fx * points_z
    points_y = (ymap - cy) / fy * points_z

    valid = (points_z > 0) & (points_z < z_max)
    if object_mask is not None:
        valid = valid & object_mask
    points = np.stack([points_x, points_y, points_z], axis=-1)
    points = points[valid].astype(np.float32)
    colors_out = colors[valid].astype(np.float32)

    return points, colors_out


# ═══════════════════════════════════════════════════════════════════════════
# Pipeline Steps
# ═══════════════════════════════════════════════════════════════════════════

def step_1_initialize(cfg: dict, dry_run: bool = False):
    """Step 1: Initialize a1x_control with gripper + EE pose support.

    Returns:
        a1x_control.JointController instance.
    """
    print("\n" + "=" * 60)
    print("STEP 1: Initialize Robot")
    print("=" * 60)

    if not dry_run:
        a1x_control.initialize(enable_ee_pose=True, enable_gripper=True)
    else:
        print("  [DRY RUN] Skipping hardware init")

    controller = a1x_control.JointController()

    if not dry_run:
        print("  Waiting for joint state data...")
        if controller.wait_for_joint_states(timeout=10):
            joints = controller.get_joint_states()
            if joints:
                print("  ✓ Joint states available")
                for name in ['arm_joint1', 'arm_joint2', 'arm_joint3',
                             'arm_joint4', 'arm_joint5', 'arm_joint6']:
                    if name in joints:
                        print(f"    {name}: {joints[name]:+.4f} rad")
        else:
            logger.warning("No joint state data available — continuing anyway")

        print("  Waiting for gripper...")
        if controller.wait_for_gripper_state(timeout=10):
            print("  ✓ Gripper ready")
        else:
            logger.warning("No gripper state data — continuing anyway")
    else:
        print("  [DRY RUN] Controller created (no hardware)")

    return controller


def step_2_move_to_observation(
    controller, observation_pose: list, dry_run: bool = False
):
    """Step 2: Open gripper and move to the observation pose."""
    print("\n" + "=" * 60)
    print("STEP 2: Move to Observation Pose")
    print("=" * 60)

    if not dry_run:
        print("  Opening gripper...")
        controller.open_gripper()
        time.sleep(1.0)

    move_to_joint_pose(controller, observation_pose, "observation", dry_run)
    print("  ✓ At observation pose")


def step_3_capture_rgbd(camera_cfg: dict, dry_run: bool = False):
    """Step 3: Capture aligned RGBD frame from D405.

    Returns:
        Tuple of (color_bgr, depth, intrinsic_matrix).
    """
    print("\n" + "=" * 60)
    print("STEP 3: Capture RGBD")
    print("=" * 60)

    if dry_run:
        w, h = camera_cfg.get("width", 640), camera_cfg.get("height", 480)
        color = np.zeros((h, w, 3), dtype=np.uint8)
        depth = np.full((h, w), 300, dtype=np.uint16)
        intrinsic = np.array([
            [604.0, 0.0, w / 2],
            [0.0, 604.0, h / 2],
            [0.0, 0.0, 1.0],
        ])
        print(f"  [DRY RUN] Synthetic {w}×{h} frame")
        return color, depth, intrinsic

    width = camera_cfg.get("width", 640)
    height = camera_cfg.get("height", 480)
    fps = camera_cfg.get("fps", 15)
    live_preview = camera_cfg.get("live_preview", True)

    with RGBDCapture(width=width, height=height, color_fps=fps, depth_fps=fps) as cap:
        if live_preview:
            print("  Live preview — press 'c' to capture, 'q' to quit")
            while True:
                try:
                    frames = cap.pipeline.wait_for_frames(timeout_ms=5000)
                except RuntimeError:
                    continue
                aligned = cap.align.process(frames)
                cf = aligned.get_color_frame()
                df = aligned.get_depth_frame()
                if not cf or not df:
                    continue
                color_img = np.asanyarray(cf.get_data())
                depth_img = np.asanyarray(df.get_data())
                depth_cm = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_img, alpha=0.03),
                    cv2.COLORMAP_JET,
                )
                preview = cv2.addWeighted(color_img, 0.7, depth_cm, 0.3, 0)
                cv2.putText(preview, f"{width}x{height} | 'c'=capture 'q'=quit",
                            (10, height - 15), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 255, 255), 1)
                cv2.imshow("AnyGrasp - Camera Preview", preview)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('c'):
                    print("  ✓ Frame captured!")
                    break
                elif key == ord('q'):
                    cv2.destroyAllWindows()
                    raise RuntimeError("Capture cancelled by user")
            cv2.destroyAllWindows()
            color = color_img.copy()
            depth = depth_img.copy()
            intrinsic = np.array([
                [cap.intrinsics.fx, 0.0, cap.intrinsics.ppx],
                [0.0, cap.intrinsics.fy, cap.intrinsics.ppy],
                [0.0, 0.0, 1.0],
            ])
        else:
            result = cap.capture_frame()
            if result is None:
                raise RuntimeError("Failed to capture RGBD frame")
            color, depth, intrinsic = result

    print(f"  ✓ Captured {color.shape[1]}×{color.shape[0]} frame")
    print(f"  Intrinsics: fx={intrinsic[0,0]:.1f}, fy={intrinsic[1,1]:.1f}")
    return color, depth, intrinsic


def step_4_build_point_cloud(
    color_bgr: np.ndarray,
    depth_raw: np.ndarray,
    intrinsic: np.ndarray,
    cam_cfg: dict,
    ws_cfg: dict,
    yoloe_cfg: dict,
    target_name: str = "",
    debug: bool = False,
):
    """Step 4: (Optional) SAM3 filter + build point cloud.

    Returns:
        Tuple of (points, colors) — both float32.
    """
    print("\n" + "=" * 60)
    print("STEP 4: Build Point Cloud" + (" (with SAM3 filter)" if target_name else ""))
    print("=" * 60)

    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    factor_depth = cam_cfg.get("factor_depth", 10000)
    height, width = depth_raw.shape

    seg_mask = None
    if target_name:
        print(f"  Detecting target: '{target_name}' with SAM3 …")
        device = yoloe_cfg.get("device", "cuda:0")

        detector = Sam3Detector(device=device)
        det = detector.detect(color_bgr, [target_name], conf_threshold=0.0)

        if det is None:
            print("  ✗ No object detected! Using full scene point cloud.")
        else:
            bbox, mask, score, cls_name = det
            print(f"  ✓ Detected '{cls_name}' (conf={score:.3f})")
            print(f"    BBox: [{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}]")
            if mask is not None:
                seg_mask = mask
                print(f"    Mask pixels: {mask.sum()} / {mask.size} "
                      f"({mask.sum()/mask.size*100:.1f}%)")
            else:
                print("    No segmentation mask — using bounding box as mask")
                seg_mask = np.zeros((height, width), dtype=bool)
                x1, y1, x2, y2 = bbox.astype(int)
                seg_mask[max(0,y1):min(height,y2), max(0,x1):min(width,x2)] = True

            if debug:
                vis = color_bgr.copy()
                x1, y1, x2, y2 = bbox.astype(int)
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(vis, f"{cls_name} {score:.2f}",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0), 2)
                if seg_mask is not None:
                    overlay = vis.copy()
                    overlay[seg_mask] = (0, 255, 0)
                    vis = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)
                cv2.imshow("SAM3 Detection", vis)
                print("    Showing detection (press any key to continue) …")
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    # Convert BGR → RGB and normalize to [0, 1]
    colors = color_bgr[:, :, ::-1].astype(np.float32) / 255.0

    points, colors_filtered = build_point_cloud(
        colors, depth_raw,
        fx=fx, fy=fy, cx=cx, cy=cy,
        scale=float(factor_depth),
        z_max=ws_cfg.get("zmax", 1.0),
        object_mask=seg_mask,
    )
    print(f"  Points: {len(points)}")
    if len(points) > 0:
        print(f"  Bounds min: {points.min(axis=0)}")
        print(f"  Bounds max: {points.max(axis=0)}")
    print("  ✓ Point cloud built")

    return points, colors_filtered


def step_5_anygrasp_inference(
    points: np.ndarray,
    colors: np.ndarray,
    ag_cfg: dict,
    ws_cfg: dict,
    vis_cfg: dict,
    debug: bool = False,
):
    """Step 5: Run AnyGrasp inference and select the best grasp.

    Returns:
        Tuple of (rotation_3x3, translation_3, width, score, gg_pick).
    """
    print("\n" + "=" * 60)
    print("STEP 5: AnyGrasp Inference")
    print("=" * 60)

    sdk_path = str(PROJECT_ROOT / ag_cfg["sdk_path"])
    checkpoint_path = os.path.join(sdk_path, ag_cfg["checkpoint"])
    top_n = vis_cfg.get("top_n", 20)

    # Add AnyGrasp SDK to path so gsnet.so can find lib_cxx.so & license/
    if sdk_path not in sys.path:
        sys.path.insert(0, sdk_path)
    old_cwd = os.getcwd()
    os.chdir(sdk_path)

    try:
        from gsnet import AnyGrasp

        ag_ns = SimpleNamespace(
            checkpoint_path=checkpoint_path,
            max_gripper_width=min(0.1, ag_cfg.get("max_gripper_width", 0.1)),
            gripper_height=ag_cfg.get("gripper_height", 0.03),
            top_down_grasp=ag_cfg.get("top_down_grasp", True),
            debug=debug,
        )
        anygrasp = AnyGrasp(ag_ns)
        anygrasp.load_net()

        lims = [
            ws_cfg["xmin"], ws_cfg["xmax"],
            ws_cfg["ymin"], ws_cfg["ymax"],
            ws_cfg["zmin"], ws_cfg["zmax"],
        ]

        gg, cloud = anygrasp.get_grasp(
            points, colors,
            lims=lims,
            apply_object_mask=True,
            dense_grasp=False,
            collision_detection=True,
        )
    finally:
        os.chdir(old_cwd)

    if len(gg) == 0:
        print("  ✗ No grasps detected after collision detection!")
        return None

    gg = gg.nms().sort_by_score()
    gg_pick = gg[0:top_n]
    print(f"  ✓ Detected {len(gg)} grasps (showing top {min(top_n, len(gg))})")

    # Show top grasps
    for i in range(min(5, len(gg_pick))):
        g = gg_pick[i]
        print(f"    [{i}] score={g.score:.4f}, width={g.width:.4f}m, "
              f"translation=[{g.translation[0]:.4f}, {g.translation[1]:.4f}, {g.translation[2]:.4f}]")

    # Select best grasp (index 0)
    best = gg_pick[0]
    print(f"\n  Selected grasp [0]:")
    print(f"    Score:       {best.score:.4f}")
    print(f"    Width:       {best.width:.4f} m")
    print(f"    Translation: [{best.translation[0]:.4f}, {best.translation[1]:.4f}, {best.translation[2]:.4f}]")
    print(f"    Rotation:\n{best.rotation_matrix}")

    # ── Debug visualization: show point cloud + grasp poses in Open3D ───
    if debug:
        import open3d as o3d
        print("\n  [DEBUG] Visualizing point cloud + grasp poses in Open3D...")
        # Flip Z for Open3D display convention (matches AnyGrasp demo)
        trans_mat = np.array([[1, 0, 0, 0],
                              [0, 1, 0, 0],
                              [0, 0, -1, 0],
                              [0, 0, 0, 1]])
        cloud_vis = cloud
        cloud_vis.transform(trans_mat)

        grippers = gg_pick.to_open3d_geometry_list()
        for gripper in grippers:
            gripper.transform(trans_mat)

        print(f"  Showing {len(grippers)} grasps + point cloud (close window to continue)")
        o3d.visualization.draw_geometries([*grippers, cloud_vis],
                                          window_name="AnyGrasp Debug - All Top Grasps")

        print("  Showing best grasp only (close window to continue)")
        o3d.visualization.draw_geometries([grippers[0], cloud_vis],
                                          window_name="AnyGrasp Debug - Best Grasp")

    return best.rotation_matrix, best.translation, best.width, best.score, gg_pick


def step_6_transform_to_base(
    grasp_rotation: np.ndarray,
    grasp_translation: np.ndarray,
    grasp_width: float,
    observation_joints: list,
    handeye_path: str | Path,
    motion_cfg: dict,
    use_anygrasp_rotation: bool = True,
    tcp_offset: float = 0.055,
):
    """Step 6: Transform grasp from camera frame to base frame.

    AnyGrasp / GraspNet rotation convention (gripper canonical frame):
        X = closing direction (width)
        Y = height direction
        Z = approach direction (depth, into the object)

    A1X gripper_link convention (from URDF):
        X = approach direction (into the object)
        Y = closing direction
        Z = height direction

    A correction rotation R_graspnet_to_a1x is applied to remap axes.

    When use_anygrasp_rotation is True, uses AnyGrasp's full 6-DOF rotation
    (with the convention correction).
    When False, overrides with a vertical top-down approach rotation,
    using AnyGrasp's approach direction for in-plane gripper orientation.

    Returns:
        Tuple of (T_grasp_base, T_pre_grasp_base, T_lift_base).
    """
    print("\n" + "=" * 60)
    print("STEP 6: Transform to Base Frame")
    print("=" * 60)

    # Load hand-eye calibration
    T_ee_cam = load_handeye_calibration(handeye_path)
    print("  T_ee_cam loaded from calibration")

    # Compute FK at observation pose
    T_base_ee = compute_T_base_ee_from_fk(observation_joints)
    print("  T_base_ee computed from FK")

    # ── GraspNet → A1X frame convention correction ──────────────────────
    # GraspNet:  X=closing, Y=height,  Z=approach
    # A1X URDF:  X=approach, Y=closing, Z=height
    # R_graspnet_to_a1x maps GraspNet gripper axes to A1X gripper_link axes:
    #   A1X +X (approach)  = GraspNet +Z (approach)
    #   A1X +Y (closing)   = GraspNet +X (closing)
    #   A1X +Z (height)    = GraspNet +Y (height)
    R_graspnet_to_a1x = np.array([
        [0.0, 1.0, 0.0],   # A1X X comes from GraspNet Z
        [0.0, 0.0, 1.0],   # A1X Y comes from GraspNet X
        [1.0, 0.0, 0.0],   # A1X Z comes from GraspNet Y
    ])

    if use_anygrasp_rotation:
        # ── Mode A: Use AnyGrasp's full 6-DOF rotation ─────────────────
        print("  Mode: AnyGrasp 6-DOF rotation (with convention correction)")

        # Build T_grasp_cam in AnyGrasp's convention, then apply correction
        # so the rotation is expressed in A1X gripper_link convention.
        T_grasp_cam_graspnet = grasp_to_T_matrix(grasp_rotation, grasp_translation)
        T_correction = np.eye(4)
        T_correction[:3, :3] = R_graspnet_to_a1x
        T_grasp_cam = T_grasp_cam_graspnet @ T_correction

        print(f"  AnyGrasp rotation (camera frame):\n{grasp_rotation}")
        print(f"  Corrected rotation (A1X convention):\n{T_grasp_cam[:3, :3]}")

        # TCP offset along gripper approach axis (+X of A1X gripper_link)
        tcp_offset_vec = [tcp_offset, 0.0, 0.0]
        T_grasp_base = transform_grasp_to_base(
            T_grasp_cam, T_base_ee, T_ee_cam, tcp_offset=tcp_offset_vec
        )
    else:
        # ── Mode B: Top-down override (like yoloe_grasp) ────────────────
        print("  Mode: Top-down override (using only AnyGrasp position)")
        T_cam_to_base = T_base_ee @ T_ee_cam
        grasp_pos_cam = np.array([*grasp_translation, 1.0])
        grasp_pos_base = (T_cam_to_base @ grasp_pos_cam)[:3]

        # TCP offset: raise IK target so fingertips reach the object
        grasp_pos_base[2] += tcp_offset

        # Base top-down rotation: gripper +X → base -Z (approach = down)
        #                         gripper +Y → base +Y (fingers horizontal)
        #                         gripper +Z → base +X
        R_topdown = np.array([
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ])

        # ── Use AnyGrasp closing direction for in-plane gripper rotation ──
        # GraspNet col 0 = closing/width direction in camera frame.
        # Use the SAME formula as yoloe_grasp's PCA angle mapping:
        #   R_z @ R_topdown aligns Z-column with the direction, and
        #   Y-column (gripper closing) ends up perpendicular to it.
        # This matches yoloe_grasp where PCA long-axis → Z, closing → ⊥ long.
        closing_cam = grasp_rotation[:, 0]   # GraspNet X = closing/width
        closing_base = (T_cam_to_base[:3, :3] @ closing_cam)

        # Same formula as yoloe_grasp: theta = arctan2(dir_y, dir_x)
        # Z-column → [cos(θ), sin(θ), 0] aligns with closing_base
        # Y-column → [-sin(θ), cos(θ), 0] perpendicular (= actual grip axis)
        theta_grasp = np.arctan2(closing_base[1], closing_base[0])
        print(f"  Closing direction (base XY): [{closing_base[0]:.3f}, {closing_base[1]:.3f}]")
        print(f"  Gripper yaw angle: {np.degrees(theta_grasp):.1f}°")

        c, s = np.cos(theta_grasp), np.sin(theta_grasp)
        R_z = np.array([
            [c, -s, 0.0],
            [s,  c, 0.0],
            [0.0, 0.0, 1.0],
        ])
        R_grasp = R_z @ R_topdown

        T_grasp_base = np.eye(4)
        T_grasp_base[:3, :3] = R_grasp
        T_grasp_base[:3, 3] = grasp_pos_base

    pos, wxyz = matrix_to_position_wxyz(T_grasp_base)
    print(f"\n  Grasp in base frame:")
    print(f"    Position: x={pos[0]:.4f}, y={pos[1]:.4f}, z={pos[2]:.4f}")
    print(f"    Quat (wxyz): [{wxyz[0]:.4f}, {wxyz[1]:.4f}, {wxyz[2]:.4f}, {wxyz[3]:.4f}]")
    print(f"    Gripper width: {grasp_width:.4f} m")

    # ── Pre-grasp and lift poses ─────────────────────────────────────────
    # NOTE: compute_pre_grasp_pose() uses z-column as approach direction,
    #       but A1X gripper_link uses x-column. Compute manually instead.
    pre_grasp_offset = motion_cfg.get("pre_grasp_offset", 0.05)
    lift_height = motion_cfg.get("post_grasp_lift", 0.10)

    if use_anygrasp_rotation:
        # 6-DOF mode: offset along gripper approach axis (x-column of rotation)
        approach_dir = T_grasp_base[:3, 0]  # A1X gripper +X = approach
        T_pre_grasp = T_grasp_base.copy()
        T_pre_grasp[:3, 3] -= pre_grasp_offset * approach_dir  # back along approach
    else:
        # Top-down mode: offset straight up in base Z (like yoloe_grasp)
        T_pre_grasp = T_grasp_base.copy()
        T_pre_grasp[2, 3] += pre_grasp_offset

    T_lift = compute_lift_pose(T_grasp_base, lift_height_m=lift_height)

    pre_pos, _ = matrix_to_position_wxyz(T_pre_grasp)
    lift_pos, _ = matrix_to_position_wxyz(T_lift)
    print(f"    Pre-grasp: [{pre_pos[0]:.4f}, {pre_pos[1]:.4f}, {pre_pos[2]:.4f}]")
    print(f"    Lift:      [{lift_pos[0]:.4f}, {lift_pos[1]:.4f}, {lift_pos[2]:.4f}]")
    print("  ✓ Transform complete")

    return T_grasp_base, T_pre_grasp, T_lift


def step_7_execute_grasp(
    controller,
    T_pre_grasp: np.ndarray,
    T_grasp: np.ndarray,
    T_lift: np.ndarray,
    grasp_width: float,
    motion_cfg: dict,
    safety_cfg: dict,
    ag_cfg: dict,
    dry_run: bool = False,
) -> bool:
    """Step 7: Solve IK and execute grasp motion sequence.

    Uses AnyGrasp's predicted width to set gripper opening before descent.
    If IK fails with the given orientation, automatically retries with a
    pure top-down orientation (no yaw) as a fallback.

    Phases: open gripper to width → move to pre-grasp → descend to grasp →
           close gripper → lift.
    """
    print("\n" + "=" * 60)
    print("STEP 7: IK Solve & Execute Grasp")
    print("=" * 60)

    executor = IKExecutor(
        smooth_steps=motion_cfg.get("smooth_steps", 30),
        control_rate_hz=motion_cfg.get("control_rate_hz", 10.0),
        joint_limits_min=safety_cfg.get("joint_limits", {}).get("min"),
        joint_limits_max=safety_cfg.get("joint_limits", {}).get("max"),
        interpolation_type=motion_cfg.get("interpolation_type", "linear"),
    )

    # Get current joints as IK seed (6 arm joints only)
    current_joints = None
    if not dry_run:
        joints = controller.get_joint_states()
        if joints:
            cfg = []
            for name in ['arm_joint1', 'arm_joint2', 'arm_joint3',
                         'arm_joint4', 'arm_joint5', 'arm_joint6']:
                cfg.append(joints.get(name, 0.0))
            current_joints = np.array(cfg)
    else:
        current_joints = np.array([0.0, 1.0, -0.93, 0.83, 0.0, 0.0])

    # ── Pre-validate IK before committing to motion ─────────────────────
    pos_pre, wxyz_pre = matrix_to_position_wxyz(T_pre_grasp)
    pos_grasp, wxyz_grasp = matrix_to_position_wxyz(T_grasp)
    pos_lift, wxyz_lift = matrix_to_position_wxyz(T_lift)

    print(f"  IK targets:")
    print(f"    Pre-grasp: pos={pos_pre}, quat={wxyz_pre}")
    print(f"    Grasp:     pos={pos_grasp}, quat={wxyz_grasp}")
    print(f"    Lift:      pos={pos_lift}, quat={wxyz_lift}")

    # Quick IK check for pre-grasp (the hardest pose to reach)
    print("\n  Pre-validating IK for pre-grasp pose...")
    test_solution = executor.solve_ik(pos_pre, wxyz_pre, initial_joints=current_joints)
    ik_ok = False
    if test_solution is not None:
        ik_ok = executor.verify_ik_solution(test_solution, pos_pre, position_tolerance=0.02)

    if not ik_ok:
        print("\n  ⚠ IK failed with yaw-rotated orientation!")
        print("  → Retrying with pure top-down orientation (no yaw)...")

        # Build pure top-down poses using the same position
        R_topdown = np.array([
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ])

        # Rebuild all three poses with pure top-down rotation
        T_grasp_fallback = np.eye(4)
        T_grasp_fallback[:3, :3] = R_topdown
        T_grasp_fallback[:3, 3] = T_grasp[:3, 3]  # keep position

        T_pre_grasp_fallback = T_grasp_fallback.copy()
        T_pre_grasp_fallback[2, 3] += motion_cfg.get("pre_grasp_offset", 0.05)

        T_lift_fallback = T_grasp_fallback.copy()
        T_lift_fallback[2, 3] += motion_cfg.get("post_grasp_lift", 0.10)

        pos_pre_fb, wxyz_pre_fb = matrix_to_position_wxyz(T_pre_grasp_fallback)
        print(f"    Fallback pre-grasp: pos={pos_pre_fb}, quat={wxyz_pre_fb}")

        test_solution = executor.solve_ik(pos_pre_fb, wxyz_pre_fb, initial_joints=current_joints)
        if test_solution is not None and executor.verify_ik_solution(test_solution, pos_pre_fb, position_tolerance=0.02):
            print("  ✓ Pure top-down IK succeeded — using fallback orientation")
            T_pre_grasp = T_pre_grasp_fallback
            T_grasp = T_grasp_fallback
            T_lift = T_lift_fallback
        else:
            print("  ✗ Pure top-down IK also failed — position may be unreachable")
            print(f"    Target: x={pos_grasp[0]:.4f}, y={pos_grasp[1]:.4f}, z={pos_grasp[2]:.4f}")
            return False
    else:
        print("  ✓ IK pre-validation passed")

    # Gripper width control from AnyGrasp prediction
    max_gripper_width = ag_cfg.get("max_gripper_width", 0.1)
    width_margin = ag_cfg.get("gripper_width_margin", 0.02)
    target_width = grasp_width + width_margin

    if target_width >= max_gripper_width:
        print(f"  Gripper: opening fully (predicted {grasp_width:.4f} + "
              f"margin {width_margin:.4f} >= max {max_gripper_width:.4f})")
        if not dry_run:
            controller.open_gripper()
            time.sleep(1.0)
        else:
            print("  [DRY RUN] Would open gripper fully")
    else:
        print(f"  Gripper: opening to width {target_width:.4f} m "
              f"(predicted {grasp_width:.4f} + margin {width_margin:.4f})")
        if not dry_run:
            controller.open_gripper()
            time.sleep(1.0)
        else:
            print(f"  [DRY RUN] Would open gripper to {target_width:.4f} m")

    # Safety confirmation
    if not dry_run:
        import termios
        try:
            termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
        except Exception:
            pass

        print("\n  ⚠️  SAFETY: About to execute grasp on real robot!")
        print("  Review the computed poses above.")
        response = input("  Proceed? (yes/no): ").strip().lower()
        if not response.startswith("y"):
            print("  Aborted by user.")
            return False

    success = executor.execute_grasp_sequence(
        controller=controller,
        T_pre_grasp_base=T_pre_grasp,
        T_grasp_base=T_grasp,
        T_lift_base=T_lift,
        current_joints=current_joints,
        dry_run=dry_run,
        confirm_each_phase=safety_cfg.get("confirm_each_phase", False),
        gripper_close_delay=motion_cfg.get("gripper_close_delay", 2.5),
    )

    return success


def step_8_place_and_return(
    controller,
    place_pose: list,
    observation_pose: list,
    dry_run: bool = False,
):
    """Step 8: Move to place pose, release object, return to observation."""
    print("\n" + "=" * 60)
    print("STEP 8: Place & Return")
    print("=" * 60)

    move_to_joint_pose(controller, place_pose, "place", dry_run)

    if not dry_run:
        print("  Opening gripper (release)...")
        controller.open_gripper()
        time.sleep(1.5)
    else:
        print("  [DRY RUN] Would open gripper")

    move_to_joint_pose(controller, observation_pose, "observation", dry_run)
    print("  ✓ Cycle complete")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="AnyGrasp End-to-End Grasp Execution for A1X"
    )
    parser.add_argument(
        "--config",
        default=str(Path(__file__).parent / "config.yaml"),
        help="Path to config.yaml (default: same directory as this script)",
    )
    parser.add_argument(
        "--target-name",
        type=str,
        default=None,
        help="Override target object name for SAM3 filtering (e.g. 'banana')",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run full pipeline without sending motion commands",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable visualization (detection overlay, Open3D grasps)",
    )
    parser.add_argument(
        "--use-topdown",
        action="store_true",
        help="Force top-down approach rotation instead of AnyGrasp's 6-DOF rotation",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("  AnyGrasp End-to-End Grasp Pipeline")
    print("=" * 60)
    if args.dry_run:
        print("  *** DRY RUN MODE — no motion commands ***")
    print()

    # ── Load config ─────────────────────────────────────────────────────
    cfg = load_config(args.config)
    observation_pose = cfg["observation_pose"]
    place_pose = cfg.get("place_pose", observation_pose)
    camera_cfg = cfg["camera"]
    yoloe_cfg = cfg.get("yoloe", {})
    ag_cfg = cfg["anygrasp"]
    ws_cfg = cfg["workspace"]
    vis_cfg = cfg["visualization"]
    handeye_path = str(PROJECT_ROOT / cfg["handeye"]["calibration_path"])
    motion_cfg = cfg["motion"]
    safety_cfg = cfg["safety"]

    # Rotation mode: config can be overridden by CLI flag
    use_anygrasp_rotation = ag_cfg.get("use_anygrasp_rotation", True)
    if args.use_topdown:
        use_anygrasp_rotation = False

    tcp_offset = cfg.get("tcp_offset", 0.055)

    # ── Step 1: Initialize ──────────────────────────────────────────────
    controller = step_1_initialize(cfg, dry_run=args.dry_run)

    # ── Step 2: Move to observation ─────────────────────────────────────
    step_2_move_to_observation(controller, observation_pose, dry_run=args.dry_run)

    # ── Step 3: Capture RGBD ────────────────────────────────────────────
    color, depth, intrinsic = step_3_capture_rgbd(camera_cfg, dry_run=args.dry_run)

    # ── Step 4: Build point cloud (optional SAM3 filter) ────────────────
    target_name = args.target_name or yoloe_cfg.get("target_name", "")
    points, colors_pc = step_4_build_point_cloud(
        color, depth, intrinsic,
        cam_cfg=camera_cfg,
        ws_cfg=ws_cfg,
        yoloe_cfg=yoloe_cfg,
        target_name=target_name,
        debug=args.debug,
    )

    if len(points) == 0:
        print("\n  ✗ Empty point cloud — cannot run AnyGrasp.")
        move_to_joint_pose(controller, observation_pose, "observation", args.dry_run)
        return 1

    # ── Step 5: AnyGrasp inference ──────────────────────────────────────
    result = step_5_anygrasp_inference(
        points, colors_pc, ag_cfg, ws_cfg, vis_cfg, debug=args.debug,
    )

    if result is None:
        print("\n  No grasps detected — returning to observation pose.")
        move_to_joint_pose(controller, observation_pose, "observation", args.dry_run)
        return 1

    grasp_rot, grasp_trans, grasp_width, grasp_score, gg_pick = result

    # ── Step 6: Transform to base frame ─────────────────────────────────
    T_grasp, T_pre_grasp, T_lift = step_6_transform_to_base(
        grasp_rot, grasp_trans, grasp_width,
        observation_joints=observation_pose,
        handeye_path=handeye_path,
        motion_cfg=motion_cfg,
        use_anygrasp_rotation=use_anygrasp_rotation,
        tcp_offset=tcp_offset,
    )

    # ── Step 7: Execute grasp ───────────────────────────────────────────
    success = step_7_execute_grasp(
        controller, T_pre_grasp, T_grasp, T_lift,
        grasp_width=grasp_width,
        motion_cfg=motion_cfg,
        safety_cfg=safety_cfg,
        ag_cfg=ag_cfg,
        dry_run=args.dry_run,
    )

    if not success:
        print("\n  Grasp execution failed or was aborted.")
        move_to_joint_pose(controller, observation_pose, "observation", args.dry_run)
        return 1

    # ── Step 8: Place and return ────────────────────────────────────────
    step_8_place_and_return(
        controller, place_pose, observation_pose, dry_run=args.dry_run,
    )

    print("\n" + "=" * 60)
    print("  ✓ PIPELINE COMPLETE")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
