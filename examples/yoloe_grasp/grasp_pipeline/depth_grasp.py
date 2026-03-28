#!/usr/bin/env python3
"""Depth-Based Grasp Computation for Grasp Pipeline.

Given a 2D bounding box (or segmentation mask) and a depth image,
computes a top-down grasp pose in the camera coordinate frame.

The grasp is a centroid-based approach with PCA-based rotation:
    1. Find the 3D centroid of the object from depth within the bbox/mask
    2. Use PCA on 2D mask pixels to estimate the object's principal axis angle
    3. Return (rotation, translation, quality, pca_angle) compatible
       with step_6_transform_to_base
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def compute_pca_angle(
    mask: Optional[np.ndarray] = None,
    bbox_xyxy: Optional[np.ndarray] = None,
    min_points: int = 20,
) -> float:
    """Estimate the object's principal axis angle via 2D PCA.

    Computes PCA on the (u, v) coordinates of mask pixels (preferred)
    or bbox grid points (fallback).  Returns the angle of the first
    principal component — i.e. the direction of the *long axis*.

    The caller should add ±90° if the gripper must close across the
    narrow axis.

    Args:
        mask: (H, W) bool, segmentation mask.  Used if provided.
        bbox_xyxy: [x1, y1, x2, y2] bounding box (int-castable).
                   Used as fallback when mask is None.
        min_points: Minimum pixel count for PCA to be meaningful.

    Returns:
        Angle in radians of the first principal component, measured
        counter-clockwise from the image +u axis.  Range: (-π/2, π/2].
        Returns 0.0 if PCA cannot be computed (degenerate case).
    """
    # Gather 2D points
    if mask is not None and mask.any():
        ys, xs = np.where(mask)
        pts = np.column_stack((xs.astype(np.float64), ys.astype(np.float64)))
    elif bbox_xyxy is not None:
        x1, y1, x2, y2 = np.asarray(bbox_xyxy, dtype=int)
        xs = np.arange(x1, x2)
        ys = np.arange(y1, y2)
        xx, yy = np.meshgrid(xs, ys)
        pts = np.column_stack((xx.ravel().astype(np.float64),
                               yy.ravel().astype(np.float64)))
    else:
        logger.warning("compute_pca_angle: no mask or bbox — returning 0")
        return 0.0

    if len(pts) < min_points:
        logger.warning(
            f"compute_pca_angle: only {len(pts)} points (< {min_points}) — returning 0")
        return 0.0

    # 2x2 covariance → eigen decomposition
    cov = np.cov(pts, rowvar=False)  # shape (2, 2)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # eigh returns eigenvalues in ascending order; the largest is last
    pc1 = eigenvectors[:, -1]  # first principal component (largest variance)

    angle = np.arctan2(pc1[1], pc1[0])  # radians, range (-π, π]

    # Normalise to (-π/2, π/2] — the axis has no intrinsic direction
    if angle > np.pi / 2:
        angle -= np.pi
    elif angle <= -np.pi / 2:
        angle += np.pi

    # Log aspect ratio (eigenvalue ratio) as a quality indicator
    ev_ratio = eigenvalues[-1] / max(eigenvalues[0], 1e-9)
    logger.info(
        f"PCA angle: {np.degrees(angle):.1f}°  "
        f"eigenvalue ratio: {ev_ratio:.2f}"
    )

    # For nearly-square/circular objects (ratio < 1.5), PCA is unreliable —
    # it tends to pick the diagonal of a square. Default to 0° (straight grasp).
    if ev_ratio < 1.5:
        logger.info(
            f"PCA ratio {ev_ratio:.2f} < 1.5 (near-square shape) — "
            f"ignoring unstable angle {np.degrees(angle):.1f}°, using 0°"
        )
        return 0.0

    return float(angle)



def compute_grasp_from_detection(
    bbox_xyxy: np.ndarray,
    depth_image: np.ndarray,
    intrinsic_matrix: np.ndarray,
    factor_depth: int = 1000,
    mask: Optional[np.ndarray] = None,
    depth_percentile: float = 50.0,
    depth_strategy: str = "surface",
    grasp_height_fraction: float = 0.5,
) -> Optional[Tuple[np.ndarray, np.ndarray, float, float]]:
    """Compute a top-down grasp pose from a 2D detection + depth.

    Args:
        bbox_xyxy: [x1, y1, x2, y2] bounding box in pixel coords.
        depth_image: (H, W) uint16 depth image.
        intrinsic_matrix: (3, 3) camera intrinsic matrix.
        factor_depth: Scale factor (1000 = depth values are in mm).
        mask: Optional (H, W) bool segmentation mask. If provided,
              depth is sampled only from masked pixels inside the bbox.
        depth_percentile: Percentile of depth values to use as Z
                          (50 = median, lower = closer surface).
        depth_strategy: Depth strategy for computing grasp Z.
            - ``"surface"`` (default): use the depth percentile (median)
              of valid mask pixels — the object's top surface.
            - ``"mid_height"``: estimate object height from the depth
              spread (p10 vs p90) and target a configurable fraction
              of the way down from the top surface.
        grasp_height_fraction: Fraction of estimated object height at
            which to place the grasp (0.0 = top surface, 1.0 = bottom).
            Only used when ``depth_strategy="mid_height"``.  Default 0.5.

    Returns:
        Tuple of (rotation_matrix, translation, quality_score, pca_angle)
        or None.
            rotation_matrix: (3, 3) top-down grasp orientation.
            translation: (3,) grasp position in camera frame [meters].
            quality_score: Simple quality metric (0-1).
            pca_angle: Object long-axis angle in radians (from 2D PCA).
    """
    x1, y1, x2, y2 = bbox_xyxy.astype(int)
    h, w = depth_image.shape[:2]

    # Clamp to image bounds
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    if x2 <= x1 or y2 <= y1:
        logger.warning("Invalid bbox after clamping")
        return None

    # Extract depth region
    depth_roi = depth_image[y1:y2, x1:x2].copy()

    # Apply segmentation mask if available
    if mask is not None:
        mask_roi = mask[y1:y2, x1:x2]
        depth_roi[~mask_roi] = 0

    # Filter valid depth values
    valid_mask = depth_roi > 0
    valid_depths = depth_roi[valid_mask]

    if len(valid_depths) < 10:
        logger.warning(f"Too few valid depth pixels: {len(valid_depths)}")
        return None

    # Also filter out 65535 (sensor no-return sentinel)
    valid_depths = valid_depths[valid_depths < 65535]

    if len(valid_depths) < 10:
        logger.warning("Too few valid depth pixels after filtering sentinels")
        return None

    # ── Depth strategy ──────────────────────────────────────────────────
    MIN_HEIGHT_M = 0.005   # 5 mm — below this, object is "flat"
    MAX_HEIGHT_M = 0.15    # 15 cm — clamp to avoid outlier-driven depths

    if depth_strategy == "mid_height":
        z_top = np.percentile(valid_depths, 10) / factor_depth
        z_bottom = np.percentile(valid_depths, 90) / factor_depth
        estimated_height = z_bottom - z_top

        if estimated_height < MIN_HEIGHT_M:
            # Flat object — fall back to surface strategy
            Z = np.percentile(valid_depths, depth_percentile) / factor_depth
            logger.info(
                f"mid_height: estimated height {estimated_height*1000:.1f} mm "
                f"< {MIN_HEIGHT_M*1000:.0f} mm threshold — falling back to "
                f"surface strategy (Z={Z:.4f} m)"
            )
        else:
            estimated_height = min(estimated_height, MAX_HEIGHT_M)
            Z = z_top + estimated_height * grasp_height_fraction
            logger.info(
                f"mid_height: z_top={z_top:.4f} m (p10), "
                f"z_bottom={z_bottom:.4f} m (p90), "
                f"estimated_height={estimated_height*1000:.1f} mm, "
                f"fraction={grasp_height_fraction:.2f}, "
                f"grasp Z={Z:.4f} m"
            )
    else:
        # Default "surface" strategy — median depth
        Z = np.percentile(valid_depths, depth_percentile) / factor_depth

    logger.info(
        f"Depth stats (mm): min={valid_depths.min()}, "
        f"median={np.median(valid_depths):.0f}, "
        f"max={valid_depths.max()}, "
        f"using Z={Z:.4f} m (strategy={depth_strategy})"
    )

    if Z < 0.05 or Z > 2.0:
        logger.warning(f"Suspicious Z value: {Z:.4f} m")

    # Compute 2D centroid
    if mask is not None:
        # Use mask centroid (more accurate than bbox center)
        mask_roi = mask[y1:y2, x1:x2]
        ys, xs = np.where(mask_roi & (depth_roi > 0) & (depth_roi < 65535))
        if len(xs) > 0:
            cx_local = np.mean(xs)
            cy_local = np.mean(ys)
        else:
            cx_local = (x2 - x1) / 2.0
            cy_local = (y2 - y1) / 2.0
    else:
        # Use bbox center
        cx_local = (x2 - x1) / 2.0
        cy_local = (y2 - y1) / 2.0

    # Convert to full-image pixel coordinates
    u = x1 + cx_local
    v = y1 + cy_local

    # Unproject to 3D camera frame
    fx, fy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]
    cx_cam, cy_cam = intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]

    X = (u - cx_cam) * Z / fx
    Y = (v - cy_cam) * Z / fy

    translation = np.array([X, Y, Z], dtype=np.float64)

    logger.info(
        f"Grasp 3D center (cam frame): "
        f"X={X:.4f}, Y={Y:.4f}, Z={Z:.4f} m  "
        f"(pixel u={u:.0f}, v={v:.0f})"
    )

    # Top-down grasp rotation
    # GraspNet convention: grasp closes along X, approach along Z
    # For a camera looking down:
    #   camera Z (optical axis) ≈ pointing down at table
    #   We want gripper approach = camera Z direction
    # Rotation matrix maps gripper frame → camera frame:
    #   gripper-x (closing) → camera-x (image right)
    #   gripper-y (finger length) → camera-y (image down)
    #   gripper-z (approach) → camera-z (into scene)
    rotation = np.eye(3, dtype=np.float64)

    # Quality score based on depth coverage
    coverage = len(valid_depths) / ((x2 - x1) * (y2 - y1) + 1e-6)
    quality = min(1.0, coverage)

    logger.info(f"Grasp quality: {quality:.3f} (depth coverage)")

    # PCA angle from mask (or bbox fallback)
    pca_angle = compute_pca_angle(mask=mask, bbox_xyxy=bbox_xyxy)

    return rotation, translation, quality, pca_angle


# ─── Standalone test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import cv2
    from pathlib import Path

    logging.basicConfig(level=logging.INFO)

    data_dir = Path(__file__).parent / "captured_data"
    if not data_dir.exists():
        print(f"No captured data at {data_dir}")
        exit(1)

    from PIL import Image
    import scipy.io as scio

    depth = np.array(Image.open(str(data_dir / "depth.png")))
    meta = scio.loadmat(str(data_dir / "meta.mat"))
    intrinsic = meta["intrinsic_matrix"]

    # Fake bbox for testing (center of image)
    h, w = depth.shape
    bbox = np.array([w // 4, h // 4, 3 * w // 4, 3 * h // 4], dtype=float)

    result = compute_grasp_from_detection(bbox, depth, intrinsic)
    if result:
        rot, trans, score, pca_angle = result
        print(f"\nGrasp pose:")
        print(f"  Translation: {trans}")
        print(f"  Rotation:\n{rot}")
        print(f"  Quality: {score:.3f}")
        print(f"  PCA angle: {np.degrees(pca_angle):.1f}°")
    else:
        print("Failed to compute grasp")
