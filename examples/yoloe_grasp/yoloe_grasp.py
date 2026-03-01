#!/usr/bin/env python3
"""YOLOe Vertical Grasp — End-to-End Pick-and-Place for A1X.

Orchestrates the complete pipeline:
    1. Move to observation pose
    2. Capture RGBD from D405
    3. Detect target object with YOLOe (text-prompt segmentation)
    4. Compute top-down grasp from mask + depth
    5. Transform grasp to robot base frame via hand-eye calibration
    6. Solve IK and execute grasp motion (pre-grasp → grasp → lift → place)

Usage:
    python examples/yoloe_grasp/yoloe_grasp.py
    python examples/yoloe_grasp/yoloe_grasp.py --target-name cup
    python examples/yoloe_grasp/yoloe_grasp.py --dry-run --visualize

Prerequisites:
    - CAN bus configured (1 Mbps / 5 Mbps FD)
    - ROS 2 environment sourced (or using a1x_control auto-launch)
    - Hand-eye calibration completed (handeye_calibration.yaml)
    - YOLOe checkpoint downloaded
"""
from __future__ import annotations

import argparse
import logging
import sys
import os
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

# ── Path setup ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Reuse existing grasp_pipeline modules ───────────────────────────────────
from examples.yoloe_grasp.grasp_pipeline.yoloe_detector import YOLOeDetector
from examples.yoloe_grasp.grasp_pipeline.depth_grasp import compute_grasp_from_detection
from examples.yoloe_grasp.grasp_pipeline.coordinate_transform import (
    load_handeye_calibration,
    compute_T_base_ee_from_fk,
    transform_grasp_to_base,
    matrix_to_position_wxyz,
    compute_pre_grasp_pose,
    compute_lift_pose,
)
from examples.yoloe_grasp.grasp_pipeline.ik_executor import IKExecutor
from examples.yoloe_grasp.grasp_pipeline.capture_rgbd import RGBDCapture

# ── Robot control ───────────────────────────────────────────────────────────
import a1x_control

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("yoloe_grasp")


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
) -> None:
    """Move robot to a joint-angle pose with smooth interpolation."""
    print(f"\n  Moving to {label} pose: {target_joints}")
    if dry_run:
        print("  [DRY RUN] Skipping motion")
        return

    success = controller.move_to_position_smooth(target_joints, steps=60, rate_hz=10.0)
    if success:
        print(f"  ✓ Smooth motion complete")
        time.sleep(0.5)  # brief settle
    else:
        raise RuntimeError(f"Failed to execute smooth motion for {label}")


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

        # Wait for gripper
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

    # Open gripper first
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
        # Synthetic data for dry-run
        w, h = camera_cfg.get("width", 640), camera_cfg.get("height", 480)
        color = np.zeros((h, w, 3), dtype=np.uint8)
        depth = np.full((h, w), 300, dtype=np.uint16)  # ~30mm
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
            # Use a simple clean preview without workspace-mask overlays.
            # capture_with_preview() has GraspNet-specific overlay (upper-half text)
            # that is misleading for YOLOe grasping.
            import cv2 as _cv2
            captured = None
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
                # Build clean preview: color + depth overlay
                depth_cm = _cv2.applyColorMap(
                    _cv2.convertScaleAbs(depth_img, alpha=0.03),
                    _cv2.COLORMAP_JET,
                )
                preview = _cv2.addWeighted(color_img, 0.7, depth_cm, 0.3, 0)
                _cv2.putText(preview, f"{width}x{height} | 'c'=capture 'q'=quit",
                             (10, height - 15), _cv2.FONT_HERSHEY_SIMPLEX,
                             0.5, (255, 255, 255), 1)
                _cv2.imshow("YOLOe Grasp - Camera Preview", preview)
                key = _cv2.waitKey(1) & 0xFF
                if key == ord('c'):
                    captured = (color_img.copy(), depth_img.copy())
                    print("  ✓ Frame captured!")
                    break
                elif key == ord('q'):
                    _cv2.destroyAllWindows()
                    raise RuntimeError("Capture cancelled by user")
            _cv2.destroyAllWindows()
            color, depth = captured
            intrinsic = np.array([
                [cap.intrinsics.fx, 0.0, cap.intrinsics.ppx],
                [0.0, cap.intrinsics.fy, cap.intrinsics.ppy],
                [0.0, 0.0, 1.0],
            ])
            result = (color, depth, intrinsic)
        else:
            result = cap.capture_frame()

    if result is None:
        raise RuntimeError("Failed to capture RGBD frame")

    color, depth, intrinsic = result
    print(f"  ✓ Captured {color.shape[1]}×{color.shape[0]} frame")
    print(f"  Intrinsics: fx={intrinsic[0,0]:.1f}, fy={intrinsic[1,1]:.1f}")
    return color, depth, intrinsic


def step_4_detect_object(
    color: np.ndarray,
    yoloe_cfg: dict,
    target_name_override: str | None = None,
    visualize: bool = False,
):
    """Step 4: Detect target object with YOLOe text-prompt segmentation.

    Returns:
        Tuple of (bbox_xyxy, mask, score, class_name).
    """
    print("\n" + "=" * 60)
    print("STEP 4: Detect Object (YOLOe)")
    print("=" * 60)

    checkpoint = str(PROJECT_ROOT / yoloe_cfg["checkpoint"])
    device = yoloe_cfg.get("device", "cuda:0")
    target_names = target_name_override or yoloe_cfg.get("target_names", "box")
    conf_threshold = yoloe_cfg.get("conf_threshold", 0.25)

    # Normalize to list
    if isinstance(target_names, str):
        target_names = [target_names]

    print(f"  Checkpoint: {checkpoint}")
    print(f"  Targets: {target_names}")
    print(f"  Device: {device}")

    detector = YOLOeDetector(checkpoint, device=device)
    result = detector.detect(color, target_names, conf_threshold=conf_threshold)

    if result is None:
        print("  ✗ No object detected!")
        return None

    bbox, mask, score, class_name = result
    print(f"  ✓ Detected '{class_name}' (conf={score:.3f})")
    print(f"    BBox: [{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}]")
    if mask is not None:
        print(f"    Mask pixels: {mask.sum()}")

    if visualize:
        vis = color.copy()
        x1, y1, x2, y2 = bbox.astype(int)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis, f"{class_name} {score:.2f}",
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if mask is not None:
            overlay = vis.copy()
            overlay[mask] = [0, 255, 0]
            vis = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)
        print("  Showing detection (close window or press any key)...")
        cv2.imshow("YOLOe Detection", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return bbox, mask, score, class_name


def step_5_compute_grasp(
    bbox: np.ndarray,
    depth: np.ndarray,
    intrinsic: np.ndarray,
    mask: np.ndarray | None,
    factor_depth: int,
    depth_strategy: str = "surface",
    grasp_height_fraction: float = 0.5,
):
    """Step 5: Compute 3D grasp from mask centroid + depth.

    Returns:
        Tuple of (rotation_matrix, translation, quality, pca_angle).
    """
    print("\n" + "=" * 60)
    print("STEP 5: Compute 3D Grasp (Depth + PCA)")
    print("=" * 60)

    result = compute_grasp_from_detection(
        bbox, depth, intrinsic,
        factor_depth=factor_depth,
        mask=mask,
        depth_percentile=50.0,
        depth_strategy=depth_strategy,
        grasp_height_fraction=grasp_height_fraction,
    )

    if result is None:
        raise RuntimeError("Failed to compute grasp from detection + depth")

    rot, trans, quality, pca_angle = result
    print(f"  Grasp (camera frame):")
    print(f"    Translation: [{trans[0]:.4f}, {trans[1]:.4f}, {trans[2]:.4f}] m")
    print(f"    Quality: {quality:.3f}")
    print(f"    PCA long-axis angle: {np.degrees(pca_angle):.1f}°")
    print(f"  ✓ 3D grasp computed")
    return rot, trans, quality, pca_angle


def step_6_transform_to_base(
    grasp_rotation: np.ndarray,
    grasp_translation: np.ndarray,
    observation_joints: list,
    handeye_path: str | Path,
    motion_cfg: dict,
    grasp_height_offset: float = 0.0,
    tcp_offset: float = 0.055,
    grasp_y_correction: float = 0.0,
    grasp_angle: float = 0.0,
):
    """Step 6: Transform grasp from camera frame to base frame.

    Uses vertical top-down approach with PCA-based in-plane rotation.
    Accounts for TCP offset (gripper_link → fingertip distance).

    Args:
        grasp_angle: In-plane rotation angle (radians) from PCA.
            This is the object's *long-axis* angle; a 90° offset is
            applied internally so the gripper closes across the narrow axis.

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

    # --- Top-down vertical grasp ---
    # Transform only the POSITION from camera → base frame
    T_cam_to_base = T_base_ee @ T_ee_cam
    grasp_pos_cam = np.array([*grasp_translation, 1.0])
    grasp_pos_base = (T_cam_to_base @ grasp_pos_cam)[:3]

    # Apply height offset
    grasp_pos_base[2] += grasp_height_offset
    print(f"  Object Z in base: {grasp_pos_base[2]:.4f} m (after +{grasp_height_offset:.3f}m offset)")

    # TCP offset: gripper approach axis (+X) maps to base -Z.
    # Fingertips are tcp_offset below gripper_link, so we raise the
    # IK target by tcp_offset to position the fingertips at the object.
    grasp_pos_base[2] += tcp_offset
    print(f"  TCP offset: +{tcp_offset:.3f} m → IK target Z = {grasp_pos_base[2]:.4f} m")

    # Lateral (Y) fine-tune for hand-eye calibration drift
    if grasp_y_correction != 0.0:
        grasp_pos_base[1] += grasp_y_correction
        print(f"  Y correction: {grasp_y_correction:+.3f} m → Y = {grasp_pos_base[1]:.4f} m")

    # ── Build top-down rotation with PCA-based in-plane angle ───────────
    # Base R_topdown: gripper +X → base -Z  (approach = down)
    #                 gripper +Y → base +Y  (fingers spread horizontal)
    #                 gripper +Z → base +X  (perpendicular, forward)
    R_topdown = np.array([
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
    ])

    # ── Transform PCA angle from camera image space to base frame ───────
    # The PCA angle is measured in camera (u, v) pixel coordinates:
    #   camera +X = image right (u), camera +Y = image down (v).
    # The camera is mounted at an arbitrary orientation, so its axes
    # don't align with the base frame.  Use the rotation part of
    # T_cam_to_base to properly map the angle.
    R_cam_to_base = T_cam_to_base[:3, :3]
    pca_dir_cam = np.array([np.cos(grasp_angle), np.sin(grasp_angle), 0.0])
    pca_dir_base = R_cam_to_base @ pca_dir_cam
    # Project onto horizontal (base XY) plane and extract angle
    theta_grasp = np.arctan2(pca_dir_base[1], pca_dir_base[0])

    print(f"  PCA angle (image space): {np.degrees(grasp_angle):.1f}°")
    print(f"  Grasp angle (base frame): {np.degrees(theta_grasp):.1f}°")

    # Rotate around the vertical (base Z-axis) to orient the gripper in
    # the horizontal plane.  Must be R_z @ R_topdown (base-frame rotation
    # applied on the LEFT) so the approach axis (gripper +X → base -Z)
    # stays pointing straight down.
    c, s = np.cos(theta_grasp), np.sin(theta_grasp)
    R_z = np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0],
    ])
    R_grasp = R_z @ R_topdown  # base-frame rotation: approach stays vertical

    T_grasp_base = np.eye(4)
    T_grasp_base[:3, :3] = R_grasp
    T_grasp_base[:3, 3] = grasp_pos_base

    pos, wxyz = matrix_to_position_wxyz(T_grasp_base)
    print(f"\n  Grasp in base frame:")
    print(f"    Position: x={pos[0]:.4f}, y={pos[1]:.4f}, z={pos[2]:.4f}")
    print(f"    Quat (wxyz): [{wxyz[0]:.4f}, {wxyz[1]:.4f}, {wxyz[2]:.4f}, {wxyz[3]:.4f}]")

    # Pre-grasp: same XY, higher Z (approach from above)
    pre_grasp_offset = motion_cfg.get("pre_grasp_offset", 0.05)
    lift_height = motion_cfg.get("post_grasp_lift", 0.10)

    T_pre_grasp = T_grasp_base.copy()
    T_pre_grasp[2, 3] += pre_grasp_offset  # move up in base Z

    T_lift = T_grasp_base.copy()
    T_lift[2, 3] += lift_height  # lift up in base Z

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
    motion_cfg: dict,
    safety_cfg: dict,
    dry_run: bool = False,
) -> bool:
    """Step 7: Solve IK and execute grasp motion sequence.

    Phases: move to pre-grasp → descend to grasp → close gripper → lift.
    """
    print("\n" + "=" * 60)
    print("STEP 7: IK Solve & Execute Grasp")
    print("=" * 60)

    executor = IKExecutor(
        smooth_steps=motion_cfg.get("smooth_steps", 30),
        control_rate_hz=motion_cfg.get("control_rate_hz", 10.0),
        joint_limits_min=safety_cfg.get("joint_limits", {}).get("min"),
        joint_limits_max=safety_cfg.get("joint_limits", {}).get("max"),
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
            current_joints = np.array(cfg)  # 6 arm joints only
    else:
        current_joints = np.array([0.0, 1.0, -0.93, 0.83, 0.0, 0.0])

    # Safety confirmation
    if not dry_run:
        # Flush any stray keystrokes that leaked from cv2 windows to stdin
        import sys, termios
        try:
            termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
        except Exception:
            pass  # Not a real terminal (e.g. piped input)

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

    # Move to place pose (with object held)
    move_to_joint_pose(controller, place_pose, "place", dry_run)

    # Open gripper to release
    if not dry_run:
        print("  Opening gripper (release)...")
        controller.open_gripper()
        time.sleep(1.5)
    else:
        print("  [DRY RUN] Would open gripper")

    # Return to observation
    move_to_joint_pose(controller, observation_pose, "observation", dry_run)
    print("  ✓ Cycle complete")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLOe Vertical Grasp — end-to-end pick-and-place for A1X"
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
        help="Override target object name (e.g. 'banana', 'cup')",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run full pipeline without sending motion commands",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Show detection visualization window",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("  YOLOe Vertical Grasp Pipeline")
    print("=" * 60)
    if args.dry_run:
        print("  *** DRY RUN MODE — no motion commands ***")
    print()

    # ── Load config ─────────────────────────────────────────────────────
    cfg = load_config(args.config)
    observation_pose = cfg["observation_pose"]
    place_pose = cfg.get("place_pose", observation_pose)
    camera_cfg = cfg["camera"]
    yoloe_cfg = cfg["yoloe"]
    handeye_path = str(PROJECT_ROOT / cfg["handeye"]["calibration_path"])
    motion_cfg = cfg["motion"]
    safety_cfg = cfg["safety"]
    factor_depth = camera_cfg.get("factor_depth", 10000)
    grasp_height_offset = yoloe_cfg.get("grasp_height_offset", 0.015)

    # ── Step 1: Initialize ──────────────────────────────────────────────
    controller = step_1_initialize(cfg, dry_run=args.dry_run)

    # ── Step 2: Move to observation ─────────────────────────────────────
    step_2_move_to_observation(controller, observation_pose, dry_run=args.dry_run)

    # ── Step 3: Capture RGBD ────────────────────────────────────────────
    color, depth, intrinsic = step_3_capture_rgbd(camera_cfg, dry_run=args.dry_run)

    # ── Step 4: Detect object ───────────────────────────────────────────
    detection = step_4_detect_object(
        color, yoloe_cfg,
        target_name_override=args.target_name,
        visualize=args.visualize,
    )

    if detection is None:
        print("\n  No object detected — returning to observation pose.")
        move_to_joint_pose(controller, observation_pose, "observation", args.dry_run)
        print("\n  Pipeline finished (no grasp attempted).")
        return 1

    bbox, mask, score, class_name = detection

    # ── Step 5: Compute 3D grasp ────────────────────────────────────────
    depth_strategy = yoloe_cfg.get("depth_strategy", "surface")
    grasp_height_fraction = yoloe_cfg.get("grasp_height_fraction", 0.5)
    rot, trans, quality, pca_angle = step_5_compute_grasp(
        bbox, depth, intrinsic, mask, factor_depth,
        depth_strategy=depth_strategy,
        grasp_height_fraction=grasp_height_fraction,
    )

    # ── Step 6: Transform to base frame ─────────────────────────────────
    tcp_offset = cfg.get("tcp_offset", 0.055)
    grasp_y_correction = yoloe_cfg.get("grasp_y_correction", 0.0)
    enable_pca = yoloe_cfg.get("enable_pca_rotation", True)
    T_grasp, T_pre_grasp, T_lift = step_6_transform_to_base(
        rot, trans, observation_pose, handeye_path, motion_cfg,
        grasp_height_offset=grasp_height_offset,
        tcp_offset=tcp_offset,
        grasp_y_correction=grasp_y_correction,
        grasp_angle=pca_angle if enable_pca else 0.0,
    )

    # ── Step 7: Execute grasp ───────────────────────────────────────────
    success = step_7_execute_grasp(
        controller, T_pre_grasp, T_grasp, T_lift,
        motion_cfg, safety_cfg, dry_run=args.dry_run,
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
