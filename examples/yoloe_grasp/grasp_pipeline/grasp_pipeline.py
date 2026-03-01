#!/usr/bin/env python3
"""End-to-End Grasp Pipeline for A1X Robot.

Orchestrates the complete grasping workflow:
    1. Move to observation pose
    2. Capture RGBD frame from D405
    3. Predict grasp poses using GraspNet-baseline
    4. Transform best grasp to robot base frame via hand-eye calibration
    5. Solve IK and execute pre-grasp → grasp → close → lift

Usage:
    # Full execution (with safety prompt):
    CUDA_VISIBLE_DEVICES=0 python examples/yoloe_grasp/grasp_pipeline/grasp_pipeline.py

    # Dry run (compute everything, no robot motion):
    CUDA_VISIBLE_DEVICES=0 python examples/yoloe_grasp/grasp_pipeline/grasp_pipeline.py --dry-run

    # With visualization of predicted grasps:
    CUDA_VISIBLE_DEVICES=0 python examples/yoloe_grasp/grasp_pipeline/grasp_pipeline.py --visualize

    # Custom config:
    CUDA_VISIBLE_DEVICES=0 python examples/yoloe_grasp/grasp_pipeline/grasp_pipeline.py --config path/to/config.yaml
"""
from __future__ import annotations

import sys
import os
import argparse
import logging
import time
from pathlib import Path

import numpy as np
import yaml

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from examples.yoloe_grasp.grasp_pipeline.capture_rgbd import RGBDCapture, save_graspnet_format, generate_workspace_mask
from examples.yoloe_grasp.grasp_pipeline.grasp_predictor import GraspPredictor
from examples.yoloe_grasp.grasp_pipeline.yoloe_detector import YOLOeDetector
from examples.yoloe_grasp.grasp_pipeline.depth_grasp import compute_grasp_from_detection
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("grasp_pipeline")


def load_config(config_path: str | Path) -> dict:
    """Load pipeline configuration from YAML."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    logger.info(f"Loaded config from {config_path}")
    return cfg


def step_1_move_to_observation(controller, observation_pose: list, dry_run: bool = False):
    """Step 1: Open gripper and move to observation pose."""
    print("\n" + "=" * 60)
    print("STEP 1: Move to Observation Pose")
    print("=" * 60)
    print(f"  Target joints: {observation_pose}")

    if dry_run:
        print("  [DRY RUN] Skipping motion")
        return

    # Open gripper first for safety
    print("  Opening gripper...")
    controller.open_gripper()
    time.sleep(1.5)

    # Move to observation pose
    print("  Moving to observation pose...")
    success = controller.move_to_position_smooth(
        observation_pose, steps=30, rate_hz=10.0
    )
    if not success:
        raise RuntimeError("Failed to move to observation pose")

    # Wait for robot to stabilize
    time.sleep(1.0)
    print("  ✓ At observation pose")


def step_2_capture_rgbd(camera_cfg: dict, workspace_cfg: dict, dry_run: bool = False):
    """Step 2: Capture RGBD frame from D405.

    Returns:
        Tuple of (color, depth, intrinsic, workspace_mask) numpy arrays.
    """
    print("\n" + "=" * 60)
    print("STEP 2: Capture RGBD Frame")
    print("=" * 60)

    width = camera_cfg.get("width", 1280)
    height = camera_cfg.get("height", 720)
    color_fps = camera_cfg.get("color_fps", 15)
    depth_fps = camera_cfg.get("depth_fps", 5)
    live_preview = camera_cfg.get("live_preview", True)

    print(f"  Resolution: {width}x{height}, color@{color_fps}fps, depth@{depth_fps}fps")

    with RGBDCapture(width=width, height=height, color_fps=color_fps, depth_fps=depth_fps) as cam:
        if live_preview:
            color, depth, intrinsic = cam.capture_with_preview()
        else:
            color, depth, intrinsic = cam.capture_frame(warmup_frames=30)

    print(f"  Color: {color.shape}, depth: {depth.shape}")
    print(f"  Depth range: [{depth[depth > 0].min() if (depth > 0).any() else 0}, {depth.max()}] mm")
    print(f"  Intrinsic:\n{intrinsic}")

    # Generate workspace mask
    mask_type = workspace_cfg.get("mask_type", "upper_half")
    workspace_mask = generate_workspace_mask(
        height=color.shape[0],
        width=color.shape[1],
        mask_type=mask_type,
        crop_lr_pixels=workspace_cfg.get("crop_lr_pixels", 100),
        crop_top=workspace_cfg.get("crop_top", 0),
        crop_bottom=workspace_cfg.get("crop_bottom", 0),
        crop_left=workspace_cfg.get("crop_left", 0),
        crop_right=workspace_cfg.get("crop_right", 0),
        depth_image=depth,
        min_depth_m=workspace_cfg.get("min_depth", 0.1),
        max_depth_m=workspace_cfg.get("max_depth", 0.6),
    )
    print(f"  Workspace mask: {mask_type} ({workspace_mask.sum() // 255} pixels active)")

    # Save captured data for reference
    output_dir = Path(__file__).parent / "captured_data"
    save_graspnet_format(color, depth, intrinsic, output_dir, workspace_mask=workspace_mask)
    print(f"  ✓ Saved to {output_dir}")

    return color, depth, intrinsic, workspace_mask


def step_3_predict_grasps(
    color: np.ndarray,
    depth: np.ndarray,
    intrinsic: np.ndarray,
    graspnet_cfg: dict,
    workspace_cfg: dict,
    workspace_mask: np.ndarray | None = None,
):
    """Step 3: Predict grasp poses using GraspNet.

    Returns:
        Tuple of (best_rotation, best_translation, best_score, gg, cloud).
    """
    print("\n" + "=" * 60)
    print("STEP 3: Predict Grasp Poses (GraspNet)")
    print("=" * 60)

    checkpoint = PROJECT_ROOT / graspnet_cfg["checkpoint_path"]
    graspnet_root = PROJECT_ROOT / graspnet_cfg["graspnet_root"]

    print(f"  Checkpoint: {checkpoint}")
    print(f"  Loading model...")

    predictor = GraspPredictor(
        checkpoint_path=str(checkpoint),
        graspnet_root=str(graspnet_root),
        num_point=graspnet_cfg.get("num_point", 20000),
        num_view=graspnet_cfg.get("num_view", 300),
        collision_thresh=graspnet_cfg.get("collision_thresh", 0.01),
        voxel_size=graspnet_cfg.get("voxel_size", 0.01),
        max_grasps=graspnet_cfg.get("max_grasps", 50),
    )

    min_depth = workspace_cfg.get("min_depth", 0.1)
    max_depth = workspace_cfg.get("max_depth", 0.6)
    print(f"  Workspace mask provided: {workspace_mask is not None}")
    print(f"  Workspace depth range: [{min_depth}, {max_depth}] m")

    gg, cloud = predictor.predict(
        color, depth, intrinsic,
        workspace_mask=workspace_mask,
        min_depth_m=min_depth,
        max_depth_m=max_depth,
    )

    if gg is None or len(gg) == 0:
        raise RuntimeError("No valid grasps detected! Check scene and workspace parameters.")

    result = predictor.get_best_grasp(gg)
    if result is None:
        raise RuntimeError("Failed to extract best grasp")

    rot, trans, score = result
    print(f"\n  Best grasp (camera frame):")
    print(f"    Score: {score:.4f}")
    print(f"    Translation: [{trans[0]:.4f}, {trans[1]:.4f}, {trans[2]:.4f}] m")
    print(f"    Rotation:\n{rot}")
    print(f"  ✓ {len(gg)} grasps predicted")

    return rot, trans, score, gg, cloud


def step_3b_yoloe_grasp(
    color: np.ndarray,
    depth: np.ndarray,
    intrinsic: np.ndarray,
    yoloe_cfg: dict,
    factor_depth: int = 1000,
    visualize: bool = False,
):
    """Step 3b: Detect object with YOLOe and compute depth-based grasp.

    Returns:
        Tuple of (rotation_matrix, translation, score).
    """
    import cv2

    print("\n" + "=" * 60)
    print("STEP 3: Detect Object (YOLOe) + Depth Grasp")
    print("=" * 60)

    checkpoint = str(PROJECT_ROOT / yoloe_cfg["checkpoint"])
    device = yoloe_cfg.get("device", "cuda:0")
    target_names = yoloe_cfg.get("target_names", ["box"])
    conf_threshold = yoloe_cfg.get("conf_threshold", 0.25)

    print(f"  Checkpoint: {checkpoint}")
    print(f"  Targets: {target_names}")

    detector = YOLOeDetector(checkpoint, device=device)
    result = detector.detect(color, target_names, conf_threshold=conf_threshold)

    if result is None:
        raise RuntimeError(
            f"YOLOe found no '{target_names}' in the scene. "
            "Check target_names in config or lighting/scene conditions."
        )

    bbox, mask, det_score, class_name = result
    print(f"\n  Detected: '{class_name}' (conf={det_score:.3f})")
    print(f"  BBox: [{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}]")

    # Compute 3D grasp from bbox + depth
    grasp_result = compute_grasp_from_detection(
        bbox, depth, intrinsic,
        factor_depth=factor_depth,
        mask=mask,
        depth_percentile=50.0,
    )

    if grasp_result is None:
        raise RuntimeError("Failed to compute grasp from detection + depth")

    rot, trans, quality = grasp_result
    print(f"\n  Grasp (camera frame):")
    print(f"    Translation: [{trans[0]:.4f}, {trans[1]:.4f}, {trans[2]:.4f}] m")
    print(f"    Quality: {quality:.3f}")
    print(f"  ✓ 2D grasp computed")

    # Visualization
    if visualize:
        vis = color.copy()
        x1, y1, x2, y2 = bbox.astype(int)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis, f"{class_name} {det_score:.2f}",
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Draw grasp center
        fx = intrinsic[0, 0]
        fy = intrinsic[1, 1]
        cx = intrinsic[0, 2]
        cy = intrinsic[1, 2]
        gu = int(trans[0] * fx / trans[2] + cx)
        gv = int(trans[1] * fy / trans[2] + cy)
        cv2.circle(vis, (gu, gv), 8, (0, 0, 255), -1)
        cv2.putText(vis, f"Z={trans[2]:.3f}m", (gu + 12, gv),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        if mask is not None:
            overlay = vis.copy()
            overlay[mask] = [0, 255, 0]
            vis = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)

        print("  Showing detection (close window to continue)...")
        cv2.imshow("YOLOe Detection + Grasp", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Save annotated image
        out_path = Path(__file__).parent / "captured_data" / "yoloe_detection.jpg"
        cv2.imwrite(str(out_path), vis)
        print(f"  Saved: {out_path}")

    return rot, trans, quality


def step_4_transform_to_base(
    grasp_rotation: np.ndarray,
    grasp_translation: np.ndarray,
    observation_joints: list,
    handeye_path: str | Path,
    motion_cfg: dict,
    tcp_offset: list | None = None,
    topdown_grasp: bool = False,
    grasp_height_offset: float = 0.0,
):
    """Step 4: Transform grasp from camera frame to robot base frame.

    Args:
        topdown_grasp: If True (for 2D bbox grasps), override orientation to
            a pure vertical top-down grasp in base frame. Only the position
            is transformed from camera→base; rotation is set to point straight down.
        grasp_height_offset: Additional Z offset in base frame (meters) to lift
            the grasp center above the detected surface.

    Returns:
        Tuple of (T_grasp_base, T_pre_grasp_base, T_lift_base).
    """
    print("\n" + "=" * 60)
    print("STEP 4: Transform Grasp to Base Frame")
    print("=" * 60)

    # Load hand-eye calibration
    T_ee_cam = load_handeye_calibration(handeye_path)
    print(f"  T_ee_cam loaded")

    # Compute FK at observation pose
    T_base_ee = compute_T_base_ee_from_fk(observation_joints)
    print(f"  T_base_ee computed from FK")

    if topdown_grasp:
        # --- YOLOe 2D grasp: top-down approach ---
        # Transform only the POSITION from camera → base frame
        T_cam_to_base = T_base_ee @ T_ee_cam
        grasp_pos_cam = np.array([*grasp_translation, 1.0])
        grasp_pos_base = (T_cam_to_base @ grasp_pos_cam)[:3]

        # Apply height offset (lift above table surface)
        grasp_pos_base[2] += grasp_height_offset
        print(f"  Height offset: +{grasp_height_offset:.3f}m")

        # Set rotation to pure top-down in base frame:
        #   gripper X (closing) → base X (horizontal)
        #   gripper Y (finger)  → base -Y
        #   gripper Z (approach) → base -Z (straight down)
        R_topdown = np.array([
            [ 1.0,  0.0,  0.0],
            [ 0.0, -1.0,  0.0],
            [ 0.0,  0.0, -1.0],
        ])
        T_grasp_base = np.eye(4)
        T_grasp_base[:3, :3] = R_topdown
        T_grasp_base[:3, 3] = grasp_pos_base
        print(f"  Mode: TOP-DOWN vertical grasp (rotation overridden)")
    else:
        # --- GraspNet 6-DOF grasp: use full rotation ---
        T_grasp_cam = grasp_to_T_matrix(grasp_rotation, grasp_translation)
        if tcp_offset:
            print(f"  TCP offset: {tcp_offset} m (gripper_link → finger center)")
        T_grasp_base = transform_grasp_to_base(
            T_grasp_cam, T_base_ee, T_ee_cam, tcp_offset=tcp_offset
        )

    pos, wxyz = matrix_to_position_wxyz(T_grasp_base)
    print(f"\n  Grasp in base frame (IK target for gripper_link):")
    print(f"    Position: x={pos[0]:.4f}, y={pos[1]:.4f}, z={pos[2]:.4f}")
    print(f"    Quaternion (wxyz): [{wxyz[0]:.4f}, {wxyz[1]:.4f}, {wxyz[2]:.4f}, {wxyz[3]:.4f}]")

    # Compute pre-grasp and lift poses
    pre_grasp_offset = motion_cfg.get("pre_grasp_offset", 0.05)
    lift_height = motion_cfg.get("post_grasp_lift", 0.10)

    T_pre_grasp_base = compute_pre_grasp_pose(T_grasp_base, pre_grasp_offset)
    T_lift_base = compute_lift_pose(T_grasp_base, lift_height)

    pre_pos, _ = matrix_to_position_wxyz(T_pre_grasp_base)
    lift_pos, _ = matrix_to_position_wxyz(T_lift_base)

    print(f"    Pre-grasp: [{pre_pos[0]:.4f}, {pre_pos[1]:.4f}, {pre_pos[2]:.4f}]")
    print(f"    Lift:      [{lift_pos[0]:.4f}, {lift_pos[1]:.4f}, {lift_pos[2]:.4f}]")
    print(f"  ✓ Transform complete")

    return T_grasp_base, T_pre_grasp_base, T_lift_base


def step_5_execute_grasp(
    controller,
    T_pre_grasp_base: np.ndarray,
    T_grasp_base: np.ndarray,
    T_lift_base: np.ndarray,
    motion_cfg: dict,
    safety_cfg: dict,
    dry_run: bool = False,
):
    """Step 5: Solve IK and execute grasp motion."""
    print("\n" + "=" * 60)
    print("STEP 5: IK Solve & Execute Grasp")
    print("=" * 60)

    executor = IKExecutor(
        smooth_steps=motion_cfg.get("smooth_steps", 30),
        control_rate_hz=motion_cfg.get("control_rate_hz", 10.0),
        joint_limits_min=safety_cfg.get("joint_limits", {}).get("min"),
        joint_limits_max=safety_cfg.get("joint_limits", {}).get("max"),
    )

    # Get current joint state as IK seed
    current_joints = None
    if not dry_run:
        joints = controller.get_joint_states()
        if joints:
            cfg = []
            for name in ['arm_joint1', 'arm_joint2', 'arm_joint3',
                         'arm_joint4', 'arm_joint5', 'arm_joint6']:
                cfg.append(joints.get(name, 0.0))
            cfg.extend([0.0, 0.0])  # gripper
            current_joints = np.array(cfg)
    else:
        # In dry-run, use observation pose as seed
        current_joints = np.array([0.12, 1.0, -0.93, 0.83, 0.0, 0.0, 0.0, 0.0])

    # Prompt for confirmation before executing
    if not dry_run:
        print("\n  ⚠️  SAFETY: About to execute grasp motion on real robot!")
        print("  Review the computed poses above carefully.")
        response = input("  Proceed with grasp execution? (yes/no): ")
        if response.lower() not in ("yes", "y"):
            print("  Aborted by user.")
            return False

    success = executor.execute_grasp_sequence(
        controller=controller,
        T_pre_grasp_base=T_pre_grasp_base,
        T_grasp_base=T_grasp_base,
        T_lift_base=T_lift_base,
        current_joints=current_joints,
        dry_run=dry_run,
        confirm_each_phase=safety_cfg.get("confirm_each_phase", False),
    )

    return success


def main():
    """Main entry point for the grasp pipeline."""
    parser = argparse.ArgumentParser(description="A1X End-to-End Grasp Pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).parent / "config.yaml"),
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute everything but skip robot motion",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Show Open3D visualization of predicted grasps",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  A1X End-to-End Grasp Pipeline")
    print("=" * 60)
    print(f"  Config: {args.config}")
    print(f"  Dry run: {args.dry_run}")
    print(f"  Visualize: {args.visualize}")
    print()

    # Load configuration
    cfg = load_config(args.config)

    # =====================================================================
    # Initialize robot controller (unless dry-run)
    # =====================================================================
    controller = None
    if not args.dry_run:
        import a1x_control

        print("Initializing A1X control system...")
        a1x_control.initialize(enable_gripper=True, enable_ee_pose=True)
        controller = a1x_control.JointController()

        if not controller.wait_for_joint_states(timeout=10.0):
            logger.warning("No joint state data available — check robot connection")
    else:
        print("[DRY RUN] Skipping robot controller initialization")

    observation_pose = cfg["observation_pose"]

    try:
        # Step 1: Move to observation pose
        step_1_move_to_observation(controller, observation_pose, dry_run=args.dry_run)

        # Step 2: Capture RGBD
        workspace_mask = None
        if not args.dry_run:
            color, depth, intrinsic, workspace_mask = step_2_capture_rgbd(
                cfg["camera"], cfg["workspace"]
            )
        else:
            # In dry-run, try to use previously captured data or example data
            captured_dir = Path(__file__).parent / "captured_data"
            example_dir = PROJECT_ROOT / cfg["graspnet"]["graspnet_root"] / "doc" / "example_data"

            if captured_dir.exists() and (captured_dir / "color.png").exists():
                data_dir = captured_dir
                print(f"\n[DRY RUN] Using captured data from {data_dir}")
            elif example_dir.exists():
                data_dir = example_dir
                print(f"\n[DRY RUN] Using example data from {data_dir}")
            else:
                raise RuntimeError(
                    "No captured or example data available for dry run. "
                    "Run without --dry-run first to capture data."
                )

            from PIL import Image
            import scipy.io as scio

            color = np.array(Image.open(str(data_dir / "color.png")))
            # PIL reads as RGB, we need BGR for our pipeline
            if len(color.shape) == 3 and color.shape[2] == 3:
                color = color[:, :, ::-1].copy()  # RGB -> BGR
            depth = np.array(Image.open(str(data_dir / "depth.png")))
            meta = scio.loadmat(str(data_dir / "meta.mat"))
            intrinsic = meta["intrinsic_matrix"]

            # Load workspace mask if available, else generate
            mask_path = data_dir / "workspace_mask.png"
            if mask_path.exists():
                workspace_mask = np.array(Image.open(str(mask_path))).astype(np.uint8)
                if workspace_mask.max() <= 1:
                    workspace_mask = workspace_mask * 255
            else:
                workspace_mask = generate_workspace_mask(
                    color.shape[0], color.shape[1],
                    mask_type=cfg["workspace"].get("mask_type", "upper_half"),
                    crop_lr_pixels=cfg["workspace"].get("crop_lr_pixels", 100),
                    crop_top=cfg["workspace"].get("crop_top", 0),
                    crop_bottom=cfg["workspace"].get("crop_bottom", 0),
                    crop_left=cfg["workspace"].get("crop_left", 0),
                    crop_right=cfg["workspace"].get("crop_right", 0),
                )

            print(f"  Color: {color.shape}, depth: {depth.shape}")

        # Step 3: Predict grasps
        yoloe_cfg = cfg.get("yoloe", {})
        use_yoloe = yoloe_cfg.get("enabled", False)

        if use_yoloe:
            rot, trans, score = step_3b_yoloe_grasp(
                color, depth, intrinsic, yoloe_cfg,
                factor_depth=cfg["camera"].get("factor_depth", 1000),
                visualize=args.visualize,
            )
            gg = None
            cloud = None
        else:
            rot, trans, score, gg, cloud = step_3_predict_grasps(
                color, depth, intrinsic,
                cfg["graspnet"],
                cfg["workspace"],
                workspace_mask=workspace_mask,
            )

            # Optional GraspNet visualization
            if args.visualize and gg is not None:
                print("\n  Showing grasp visualization (close window to continue)...")
                try:
                    import open3d as o3d
                    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
                    gg_vis = gg[:20]
                    grippers = gg_vis.to_open3d_geometry_list()
                    o3d.visualization.draw_geometries([cloud, coord_frame, *grippers])
                except Exception as e:
                    print(f"  Visualization failed: {e}")

        # Step 4: Transform to base frame
        handeye_path = PROJECT_ROOT / cfg["handeye"]["calibration_path"]
        tcp_offset = cfg.get("tcp_offset", None) if not use_yoloe else None
        T_grasp_base, T_pre_grasp_base, T_lift_base = step_4_transform_to_base(
            rot, trans,
            observation_pose,
            handeye_path,
            cfg["motion"],
            tcp_offset=tcp_offset,
            topdown_grasp=use_yoloe,
            grasp_height_offset=yoloe_cfg.get("grasp_height_offset", 0.0) if use_yoloe else 0.0,
        )

        # Step 5: Execute grasp
        success = step_5_execute_grasp(
            controller,
            T_pre_grasp_base,
            T_grasp_base,
            T_lift_base,
            cfg["motion"],
            cfg["safety"],
            dry_run=args.dry_run,
        )

        if success:
            print("\n" + "=" * 60)
            print("  ✅ GRASP PIPELINE COMPLETE!")
            print("=" * 60)
        else:
            print("\n" + "=" * 60)
            print("  ❌ GRASP PIPELINE FAILED")
            print("=" * 60)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
