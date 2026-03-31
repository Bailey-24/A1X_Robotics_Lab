#!/usr/bin/env python3
"""AnyGrasp + LingBot-Depth: Enhanced grasp pipeline for A1X.

Same as anygrasp_grasp.py, but replaces the raw D405 depth with a refined
depth map from lingbot-depth before building the point cloud for AnyGrasp.

Usage:
    python examples/anygrasp_grasp/anygrasp_lingbot_grasp.py
    python examples/anygrasp_grasp/anygrasp_lingbot_grasp.py --target-name banana
    python examples/anygrasp_grasp/anygrasp_lingbot_grasp.py --no-lingbot   # raw depth (baseline)
    python examples/anygrasp_grasp/anygrasp_lingbot_grasp.py --dry-run --debug
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Reuse all step functions from the existing pipeline ─────────────────────
from examples.anygrasp_grasp.anygrasp_grasp import (
    load_config,
    build_point_cloud,          # low-level helper (scale-agnostic)
    move_to_joint_pose,
    step_1_initialize,
    step_2_move_to_observation,
    step_3_capture_rgbd,
    step_4_build_point_cloud,   # used for SAM3 mask; we override depth/scale
    step_5_anygrasp_inference,
    step_6_transform_to_base,
    step_7_execute_grasp,
    step_8_place_and_return,
)
from examples.yoloe_grasp.grasp_pipeline.sam3_detector import Sam3Detector
import a1x_control

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("anygrasp_lingbot")

LINGBOT_MODEL_ID = "robbyant/lingbot-depth-pretrain-vitl-14-v0.5"
_lingbot_model = None   # lazy-loaded singleton


# ═══════════════════════════════════════════════════════════════════════════
# LingBot-Depth refinement
# ═══════════════════════════════════════════════════════════════════════════

def load_lingbot_model(device: torch.device):
    """Load (or return cached) lingbot-depth model."""
    global _lingbot_model
    if _lingbot_model is None:
        from mdm.model.v2 import MDMModel
        logger.info(f"Loading lingbot-depth model: {LINGBOT_MODEL_ID}")
        _lingbot_model = MDMModel.from_pretrained(LINGBOT_MODEL_ID).to(device)
    return _lingbot_model


def step_3b_refine_depth(
    color_bgr: np.ndarray,
    depth_raw: np.ndarray,
    intrinsic: np.ndarray,
    factor_depth: float = 10000.0,
    device: torch.device | None = None,
) -> np.ndarray:
    """Refine raw D405 depth with lingbot-depth.

    Args:
        color_bgr:   (H, W, 3) uint8 BGR image from D405.
        depth_raw:   (H, W) uint16 raw depth (divide by factor_depth = meters).
        intrinsic:   3×3 camera intrinsic matrix.
        factor_depth: Scale factor (raw_depth / factor_depth = meters).
        device:      Torch device to run on (default: CUDA if available).

    Returns:
        depth_refined_m: (H, W) float32 depth in metres.
    """
    print("\n" + "=" * 60)
    print("STEP 3b: LingBot-Depth Refinement")
    print("=" * 60)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    h, w = depth_raw.shape

    # Convert inputs to float
    depth_m = depth_raw.astype(np.float32) / factor_depth  # metres

    valid = depth_m[depth_m > 0]
    print(f"  Raw depth range: {valid.min():.3f} ~ {valid.max():.3f} m  "
          f"({len(valid)}/{depth_m.size} valid pixels)")

    # Build tensors
    rgb_np = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    image_t = torch.tensor(rgb_np / 255.0, dtype=torch.float32, device=device
                           ).permute(2, 0, 1).unsqueeze(0)
    depth_t = torch.tensor(depth_m, dtype=torch.float32, device=device)

    # Normalised intrinsics
    K_norm = intrinsic.astype(np.float32).copy()
    K_norm[0, 0] /= w;  K_norm[0, 2] /= w
    K_norm[1, 1] /= h;  K_norm[1, 2] /= h
    K_t = torch.tensor(K_norm, dtype=torch.float32, device=device).unsqueeze(0)

    model = load_lingbot_model(device)
    t0 = time.time()
    with torch.no_grad():
        output = model.infer(image_t, depth_in=depth_t, apply_mask=True, intrinsics=K_t)
    infer_ms = (time.time() - t0) * 1000

    depth_refined = output["depth"].squeeze().cpu().numpy().astype(np.float32)
    valid_r = depth_refined[np.isfinite(depth_refined) & (depth_refined > 0)]

    print(f"  Refined depth range: {valid_r.min():.3f} ~ {valid_r.max():.3f} m  "
          f"({len(valid_r)}/{depth_refined.size} valid pixels)")
    print(f"  Inference time: {infer_ms:.0f} ms  |  device: {device}")

    return depth_refined


def build_point_cloud_from_meters(
    color_bgr: np.ndarray,
    depth_m: np.ndarray,
    intrinsic: np.ndarray,
    z_max: float = 1.0,
    object_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Wrapper around build_point_cloud for depth already in metres (scale=1)."""
    colors_rgb = color_bgr[:, :, ::-1].astype(np.float32) / 255.0
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    return build_point_cloud(
        colors_rgb, depth_m,
        fx=fx, fy=fy, cx=cx, cy=cy,
        scale=1.0,          # depth already in metres
        z_max=z_max,
        object_mask=object_mask,
    )


def step_4_build_point_cloud_lingbot(
    color_bgr: np.ndarray,
    depth_m: np.ndarray,          # float32 metres (from lingbot or raw/factor)
    intrinsic: np.ndarray,
    ws_cfg: dict,
    yoloe_cfg: dict,
    target_name: str = "",
    debug: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Build point cloud from float32 metre depth with optional SAM3 masking."""
    print("\n" + "=" * 60)
    print("STEP 4: Build Point Cloud" + (f" (SAM3 filter: '{target_name}')" if target_name else ""))
    print("=" * 60)

    h, w = depth_m.shape
    seg_mask = None

    if target_name:
        device = yoloe_cfg.get("device", "cuda:0")
        detector = Sam3Detector(device=device)
        det = detector.detect(color_bgr, [target_name], conf_threshold=0.0)
        if det is None:
            print("  ✗ No object detected — using full scene point cloud.")
        else:
            bbox, mask, score, cls_name = det
            print(f"  ✓ Detected '{cls_name}' (conf={score:.3f})")
            if mask is not None:
                seg_mask = mask
            else:
                seg_mask = np.zeros((h, w), dtype=bool)
                x1, y1, x2, y2 = bbox.astype(int)
                seg_mask[max(0,y1):min(h,y2), max(0,x1):min(w,x2)] = True

            if debug:
                vis = color_bgr.copy()
                x1, y1, x2, y2 = bbox.astype(int)
                cv2.rectangle(vis, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(vis, f"{cls_name} {score:.2f}",
                            (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                if seg_mask is not None:
                    overlay = vis.copy()
                    overlay[seg_mask] = (0, 255, 0)
                    vis = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)
                cv2.imshow("SAM3 Detection", vis)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    points, colors_out = build_point_cloud_from_meters(
        color_bgr, depth_m,
        intrinsic=intrinsic,
        z_max=ws_cfg.get("zmax", 1.0),
        object_mask=seg_mask,
    )

    print(f"  Points: {len(points)}")
    if len(points) > 0:
        print(f"  Bounds min: {points.min(axis=0)}")
        print(f"  Bounds max: {points.max(axis=0)}")
    print("  ✓ Point cloud built")
    return points, colors_out


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="AnyGrasp + LingBot-Depth enhanced grasp pipeline"
    )
    parser.add_argument("--config", default=str(Path(__file__).parent / "config.yaml"))
    parser.add_argument("--target-name", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--use-topdown", action="store_true")
    parser.add_argument(
        "--no-lingbot", action="store_true",
        help="Disable LingBot-Depth refinement (use raw D405 depth as baseline)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    label = "AnyGrasp + LingBot-Depth Pipeline" if not args.no_lingbot else "AnyGrasp Pipeline (raw depth baseline)"
    print(f"  {label}")
    print("=" * 60)
    if args.dry_run:
        print("  *** DRY RUN MODE — no motion commands ***")
    print()

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
    factor_depth = float(camera_cfg.get("factor_depth", 10000))

    use_anygrasp_rotation = ag_cfg.get("use_anygrasp_rotation", True)
    if args.use_topdown:
        use_anygrasp_rotation = False

    tcp_offset = cfg.get("tcp_offset", 0.055)

    # Step 1
    controller = step_1_initialize(cfg, dry_run=args.dry_run)

    # Step 2
    step_2_move_to_observation(controller, observation_pose, dry_run=args.dry_run)

    # Step 3: Capture raw RGBD
    color, depth_raw, intrinsic = step_3_capture_rgbd(camera_cfg, dry_run=args.dry_run)

    # Step 3b: Optionally refine depth with LingBot-Depth
    if args.no_lingbot:
        # Baseline: convert raw uint16 → float32 metres manually
        depth_m = depth_raw.astype(np.float32) / factor_depth
        print("\n[LingBot-Depth disabled] Using raw D405 depth.")
    else:
        depth_m = step_3b_refine_depth(
            color, depth_raw, intrinsic,
            factor_depth=factor_depth,
        )

    # Step 4: Build point cloud (float32 metres depth, scale=1.0)
    target_name = args.target_name or yoloe_cfg.get("target_name", "")
    points, colors_pc = step_4_build_point_cloud_lingbot(
        color, depth_m, intrinsic,
        ws_cfg=ws_cfg,
        yoloe_cfg=yoloe_cfg,
        target_name=target_name,
        debug=args.debug,
    )

    if len(points) == 0:
        print("\n  ✗ Empty point cloud — cannot run AnyGrasp.")
        move_to_joint_pose(controller, observation_pose, "observation", args.dry_run)
        return 1

    # Step 5
    result = step_5_anygrasp_inference(points, colors_pc, ag_cfg, ws_cfg, vis_cfg, debug=args.debug)
    if result is None:
        print("\n  No grasps detected — returning to observation pose.")
        move_to_joint_pose(controller, observation_pose, "observation", args.dry_run)
        return 1

    grasp_rot, grasp_trans, grasp_width, grasp_score, gg_pick = result

    # Step 6
    T_grasp, T_pre_grasp, T_lift = step_6_transform_to_base(
        grasp_rot, grasp_trans, grasp_width,
        observation_joints=observation_pose,
        handeye_path=handeye_path,
        motion_cfg=motion_cfg,
        use_anygrasp_rotation=use_anygrasp_rotation,
        tcp_offset=tcp_offset,
    )

    # Step 7
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

    # Step 8
    step_8_place_and_return(controller, place_pose, observation_pose, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
