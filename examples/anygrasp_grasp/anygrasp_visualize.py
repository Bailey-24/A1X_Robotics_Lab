#!/usr/bin/env python3
"""AnyGrasp Point Cloud Grasp Visualization.

Captures RGBD from D405, optionally segments a target object with YOLOe,
builds a 3D point cloud, runs AnyGrasp grasp detection, and visualizes
the detected grasp poses in Open3D.

Usage:
    python examples/anygrasp_grasp/anygrasp_visualize.py --debug
    python examples/anygrasp_grasp/anygrasp_visualize.py --target-name banana --debug
    python examples/anygrasp_grasp/anygrasp_visualize.py --top-n 5 --no-preview
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import open3d as o3d
import yaml

# ── Project root (A1Xsdk/) ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from examples.yoloe_grasp.grasp_pipeline.capture_rgbd import RGBDCapture
from examples.yoloe_grasp.grasp_pipeline.yoloe_detector import YOLOeDetector


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def load_config(config_path: str | Path) -> dict:
    """Load YAML configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


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

    # Filter invalid / out-of-range depth
    valid = (points_z > 0) & (points_z < z_max)
    if object_mask is not None:
        valid = valid & object_mask
    points = np.stack([points_x, points_y, points_z], axis=-1)
    points = points[valid].astype(np.float32)
    colors_out = colors[valid].astype(np.float32)

    return points, colors_out


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AnyGrasp point cloud visualization")
    p.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "examples" / "anygrasp_grasp" / "config.yaml"),
        help="Path to YAML config file",
    )
    p.add_argument("--top-n", type=int, default=None, help="Override top_n from config")
    p.add_argument("--target-name", type=str, default=None,
                   help="Target object name for YOLOe filtering (overrides config)")
    p.add_argument("--no-preview", action="store_true", help="Skip camera live preview")
    p.add_argument("--debug", action="store_true", help="Enable Open3D visualization")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # ── 1. Load config ──────────────────────────────────────────────────
    cfg = load_config(args.config)
    cam_cfg = cfg["camera"]
    yoloe_cfg = cfg.get("yoloe", {})
    ag_cfg = cfg["anygrasp"]
    ws_cfg = cfg["workspace"]
    vis_cfg = cfg["visualization"]

    top_n = args.top_n or vis_cfg.get("top_n", 20)
    factor_depth = cam_cfg.get("factor_depth", 10000)

    print("=" * 60)
    print("AnyGrasp Point Cloud Visualization")
    print("=" * 60)

    # ── 2. Capture RGBD from D405 ───────────────────────────────────────
    print("\n[1/4] Capturing RGBD from D405 …")
    width = cam_cfg.get("width", 640)
    height = cam_cfg.get("height", 480)
    fps = cam_cfg.get("fps", 15)
    live_preview = cam_cfg.get("live_preview", True) and not args.no_preview

    with RGBDCapture(width=width, height=height, color_fps=fps, depth_fps=fps) as cap:
        if live_preview:
            import cv2
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
                # Preview overlay
                depth_cm = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_img, alpha=0.03), cv2.COLORMAP_JET
                )
                preview = cv2.addWeighted(color_img, 0.7, depth_cm, 0.3, 0)
                cv2.putText(
                    preview,
                    f"{width}x{height} | 'c'=capture 'q'=quit",
                    (10, height - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                )
                cv2.imshow("AnyGrasp - Camera Preview", preview)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("c"):
                    print("  ✓ Frame captured!")
                    break
                elif key == ord("q"):
                    cv2.destroyAllWindows()
                    print("  Capture cancelled.")
                    return 1
            cv2.destroyAllWindows()
            color_bgr = color_img.copy()
            depth_raw = depth_img.copy()
            intrinsic = np.array([
                [cap.intrinsics.fx, 0.0, cap.intrinsics.ppx],
                [0.0, cap.intrinsics.fy, cap.intrinsics.ppy],
                [0.0, 0.0, 1.0],
            ])
        else:
            result = cap.capture_frame()
            if result is None:
                print("  ✗ Failed to capture RGBD frame")
                return 1
            color_bgr, depth_raw, intrinsic = result

    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    print(f"  Resolution: {width}×{height}")
    print(f"  Intrinsics: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
    print(f"  Depth scale: 1/{factor_depth}")

    # ── 3. (Optional) YOLOe target-object segmentation ──────────────────
    target_name = args.target_name or yoloe_cfg.get("target_name", "")
    seg_mask = None  # (H, W) bool — None means "use entire scene"

    if target_name:
        print(f"\n[2/5] Detecting target object with YOLOe: '{target_name}' …")
        checkpoint = str(PROJECT_ROOT / yoloe_cfg["checkpoint"])
        device = yoloe_cfg.get("device", "cuda:0")
        conf_thr = yoloe_cfg.get("conf_threshold", 0.25)

        detector = YOLOeDetector(checkpoint, device=device)
        det = detector.detect(color_bgr, [target_name], conf_threshold=conf_thr)

        if det is None:
            print("  ✗ No object detected! Falling back to full scene.")
        else:
            bbox, mask, score, cls_name = det
            print(f"  ✓ Detected '{cls_name}' (conf={score:.3f})")
            print(f"    BBox: [{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}]")
            if mask is not None:
                seg_mask = mask
                print(f"    Mask pixels: {mask.sum()} / {mask.size} "
                      f"({mask.sum()/mask.size*100:.1f}%)")
            else:
                # Fallback: use bounding box as rectangular mask
                print("    No segmentation mask — using bounding box as mask")
                seg_mask = np.zeros((height, width), dtype=bool)
                x1, y1, x2, y2 = bbox.astype(int)
                seg_mask[max(0,y1):min(height,y2), max(0,x1):min(width,x2)] = True

            # Show detection overlay when debugging
            if args.debug:
                import cv2
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
                cv2.imshow("YOLOe Detection", vis)
                print("    Showing detection (press any key to continue) …")
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        step_label = "3/5"
    else:
        step_label = "2/4"

    # ── Build point cloud ──────────────────────────────────────────────
    print(f"\n[{step_label}] Building point cloud …")
    if seg_mask is not None:
        print("  (filtered to target object)")
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
    print(f"  Bounds min: {points.min(axis=0)}")
    print(f"  Bounds max: {points.max(axis=0)}")

    # ── Run AnyGrasp ───────────────────────────────────────────────────
    ag_step = "4/5" if target_name else "3/4"
    print(f"\n[{ag_step}] Running AnyGrasp inference …")

    sdk_path = str(PROJECT_ROOT / ag_cfg["sdk_path"])
    checkpoint_path = os.path.join(sdk_path, ag_cfg["checkpoint"])

    # Add AnyGrasp SDK to path so gsnet.so can find lib_cxx.so & license/
    if sdk_path not in sys.path:
        sys.path.insert(0, sdk_path)
    old_cwd = os.getcwd()
    os.chdir(sdk_path)  # gsnet.so looks for license/ relative to cwd

    try:
        from gsnet import AnyGrasp

        # AnyGrasp expects an argparse-like config namespace
        ag_ns = SimpleNamespace(
            checkpoint_path=checkpoint_path,
            max_gripper_width=min(0.1, ag_cfg.get("max_gripper_width", 0.1)),
            gripper_height=ag_cfg.get("gripper_height", 0.03),
            top_down_grasp=ag_cfg.get("top_down_grasp", True),
            debug=args.debug,
        )
        anygrasp = AnyGrasp(ag_ns)
        anygrasp.load_net()

        lims = [
            ws_cfg["xmin"], ws_cfg["xmax"],
            ws_cfg["ymin"], ws_cfg["ymax"],
            ws_cfg["zmin"], ws_cfg["zmax"],
        ]

        gg, cloud = anygrasp.get_grasp(
            points, colors_filtered,
            lims=lims,
            apply_object_mask=True,
            dense_grasp=False,
            collision_detection=True,
        )
    finally:
        os.chdir(old_cwd)

    if len(gg) == 0:
        print("  ✗ No grasps detected after collision detection!")
        print("  Try adjusting workspace limits in config.yaml.")
        return 1

    gg = gg.nms().sort_by_score()
    gg_pick = gg[0:top_n]
    print(f"  ✓ Detected {len(gg)} grasps, showing top {min(top_n, len(gg))}")
    print(f"  Top scores: {gg_pick.scores}")
    print(f"  Best grasp score: {gg_pick[0].score:.4f}")
    print("gg_pick:", gg_pick)

    # ── Visualize ──────────────────────────────────────────────────────
    vis_step = "5/5" if target_name else "4/4"
    print(f"\n[{vis_step}] Visualizing in Open3D …")

    if not args.debug:
        print("  Skipping visualization (add --debug to enable Open3D display)")
        print("\nDone!")
        return 0

    # Flip Z for Open3D display convention (matches AnyGrasp demo)
    trans_mat = np.array([[1, 0, 0, 0],
                          [0, 1, 0, 0],
                          [0, 0, -1, 0],
                          [0, 0, 0, 1]])
    cloud.transform(trans_mat)

    grippers = gg_pick.to_open3d_geometry_list()
    for gripper in grippers:
        gripper.transform(trans_mat)

    print(f"  Showing {len(grippers)} grasps + point cloud (close window to continue)")
    o3d.visualization.draw_geometries([*grippers, cloud])

    print("  Showing best grasp only (close window to exit)")
    o3d.visualization.draw_geometries([grippers[0], cloud])

    print("\nDone!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
