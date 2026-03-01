#!/usr/bin/env python3
"""RGBD Capture Module for Grasp Pipeline.

Captures aligned RGB and depth frames from Intel RealSense D405 camera.
Outputs data in a format compatible with GraspNet-baseline.

Features:
    - Separate FPS for color (15fps) and depth (5fps) streams
    - Real-time camera preview with depth overlay
    - Upper-half workspace mask generation
    - GraspNet-compatible file saving

Usage (standalone test):
    python examples/yoloe_grasp/grasp_pipeline/capture_rgbd.py
"""
from __future__ import annotations

import sys
import logging
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:
    print("Error: pyrealsense2 not installed. Install with: pip install pyrealsense2")
    sys.exit(1)

logger = logging.getLogger(__name__)


class RGBDCapture:
    """Captures aligned RGB + depth frames from RealSense D405."""

    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        color_fps: int = 15,
        depth_fps: int = 5,
    ):
        """Initialize the RGBD capture pipeline.

        Args:
            width: Frame width in pixels.
            height: Frame height in pixels.
            color_fps: Color stream FPS (D405 supports 5/10/15 at 1280x720).
            depth_fps: Depth stream FPS (D405 supports only 5 at 1280x720).
        """
        self.width = width
        self.height = height
        self.color_fps = color_fps
        self.depth_fps = depth_fps
        self.pipeline: Optional[rs.pipeline] = None
        self.align: Optional[rs.align] = None
        self.intrinsics: Optional[rs.intrinsics] = None

    def start(self) -> None:
        """Start the RealSense pipeline with color and depth streams."""
        self.pipeline = rs.pipeline()
        config = rs.config()

        # Enable color and depth streams with SEPARATE fps
        # D405 at 1280x720: color supports 5/10/15fps, depth supports only 5fps
        config.enable_stream(
            rs.stream.color, self.width, self.height, rs.format.bgr8, self.color_fps
        )
        config.enable_stream(
            rs.stream.depth, self.width, self.height, rs.format.z16, self.depth_fps
        )

        try:
            profile = self.pipeline.start(config)
        except RuntimeError as e:
            # If dual-stream fails, try matching fps
            logger.warning(
                f"Failed with color@{self.color_fps}fps + depth@{self.depth_fps}fps, "
                f"retrying with both at {self.depth_fps}fps..."
            )
            self.pipeline = rs.pipeline()
            config2 = rs.config()
            config2.enable_stream(
                rs.stream.color, self.width, self.height, rs.format.bgr8, self.depth_fps
            )
            config2.enable_stream(
                rs.stream.depth, self.width, self.height, rs.format.z16, self.depth_fps
            )
            try:
                profile = self.pipeline.start(config2)
            except RuntimeError as e2:
                raise RuntimeError(
                    f"Failed to start RealSense pipeline. Is the D405 connected? "
                    f"Error: {e2}"
                ) from e2

        # Create alignment object (align depth to color)
        self.align = rs.align(rs.stream.color)

        # Get depth scale from sensor (D405: typically 0.0001 = 0.1mm per unit)
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        self.auto_factor_depth = int(round(1.0 / self.depth_scale))

        # Get camera intrinsics from the color stream
        color_profile = profile.get_stream(rs.stream.color)
        self.intrinsics = color_profile.as_video_stream_profile().get_intrinsics()

        logger.info(
            f"RealSense pipeline started: {self.intrinsics.width}x{self.intrinsics.height}, "
            f"fx={self.intrinsics.fx:.2f}, fy={self.intrinsics.fy:.2f}, "
            f"cx={self.intrinsics.ppx:.2f}, cy={self.intrinsics.ppy:.2f}"
        )
        logger.info(
            f"Depth scale: {self.depth_scale} (1 unit = {self.depth_scale * 1000:.2f}mm, "
            f"auto factor_depth = {self.auto_factor_depth})"
        )

    def capture_with_preview(
        self, timeout_ms: int = 5000
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Show live camera preview, capture on keypress.

        Displays a live feed with depth colormap overlay.
        Press 'c' to capture the current frame, 'q' to quit.

        Args:
            timeout_ms: Frame wait timeout in milliseconds.

        Returns:
            Same as capture_frame: (color_image, depth_image, intrinsic_matrix).
        """
        if self.pipeline is None:
            raise RuntimeError("Pipeline not started. Call start() first.")

        print("  Live camera preview — press 'c' to capture, 'q' to quit")

        captured = None
        while True:
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=timeout_ms)
            except RuntimeError:
                continue

            aligned_frames = self.align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            # Build preview: color with depth overlay
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
            )
            # Draw upper/lower half divider line
            h = color_image.shape[0]
            preview = cv2.addWeighted(color_image, 0.7, depth_colormap, 0.3, 0)
            cv2.line(preview, (0, h // 2), (preview.shape[1], h // 2), (0, 255, 0), 2)
            cv2.putText(
                preview, "Upper half = workspace (white mask)",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )
            cv2.putText(
                preview, "Press 'c' to capture, 'q' to quit",
                (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
            )

            cv2.imshow("Grasp Pipeline - Camera Preview", preview)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('c'):
                captured = (color_image.copy(), depth_image.copy())
                print("  ✓ Frame captured!")
                break
            elif key == ord('q'):
                cv2.destroyAllWindows()
                raise RuntimeError("Capture cancelled by user")

        cv2.destroyAllWindows()

        color_image, depth_image = captured
        intrinsic_matrix = np.array([
            [self.intrinsics.fx, 0.0, self.intrinsics.ppx],
            [0.0, self.intrinsics.fy, self.intrinsics.ppy],
            [0.0, 0.0, 1.0],
        ])

        logger.info(
            f"Captured frame: color={color_image.shape}, depth={depth_image.shape}, "
            f"depth range=[{depth_image[depth_image > 0].min() if (depth_image > 0).any() else 0}, "
            f"{depth_image.max()}] mm"
        )

        return color_image, depth_image, intrinsic_matrix

    def capture_frame(
        self, warmup_frames: int = 30, timeout_ms: int = 5000
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Capture a single aligned RGBD frame (no preview).

        Args:
            warmup_frames: Number of frames to skip for auto-exposure stabilization.
            timeout_ms: Frame wait timeout in milliseconds.

        Returns:
            Tuple of (color_image, depth_image, intrinsic_matrix):
                - color_image: numpy array of shape (H, W, 3) in BGR format, dtype uint8
                - depth_image: numpy array of shape (H, W) in uint16, units = millimeters
                - intrinsic_matrix: numpy array of shape (3, 3), camera intrinsic matrix
        """
        if self.pipeline is None:
            raise RuntimeError("Pipeline not started. Call start() first.")

        # Skip warmup frames for auto-exposure to stabilize
        for _ in range(warmup_frames):
            try:
                self.pipeline.wait_for_frames(timeout_ms=timeout_ms)
            except RuntimeError:
                logger.warning("Frame timeout during warmup")

        # Capture the actual frame
        frames = self.pipeline.wait_for_frames(timeout_ms=timeout_ms)

        # Align depth to color
        aligned_frames = self.align.process(frames)

        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            raise RuntimeError("Failed to capture aligned color and depth frames")

        # Convert to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())  # (H, W, 3) BGR uint8
        depth_image = np.asanyarray(depth_frame.get_data())    # (H, W) uint16, mm

        # Build intrinsic matrix
        intrinsic_matrix = np.array([
            [self.intrinsics.fx, 0.0, self.intrinsics.ppx],
            [0.0, self.intrinsics.fy, self.intrinsics.ppy],
            [0.0, 0.0, 1.0],
        ])

        logger.info(
            f"Captured frame: color={color_image.shape}, depth={depth_image.shape}, "
            f"depth range=[{depth_image[depth_image > 0].min() if (depth_image > 0).any() else 0}, "
            f"{depth_image.max()}] mm"
        )

        return color_image, depth_image, intrinsic_matrix

    def stop(self) -> None:
        """Stop the RealSense pipeline."""
        if self.pipeline is not None:
            self.pipeline.stop()
            self.pipeline = None
            logger.info("RealSense pipeline stopped")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False


def generate_workspace_mask(
    height: int,
    width: int,
    mask_type: str = "upper_half",
    crop_lr_pixels: int = 100,  # Legacy
    crop_top: int = 0,
    crop_bottom: int = 0,
    crop_left: int = 0,
    crop_right: int = 0,
    depth_image: Optional[np.ndarray] = None,
    min_depth_m: float = 0.1,
    max_depth_m: float = 0.6,
    factor_depth: int = 1000,
) -> np.ndarray:
    """Generate workspace mask for GraspNet.

    Args:
        height: Image height.
        width: Image width.
        mask_type: "upper_half", "center_crop", "custom", or "depth_range".
        crop_lr_pixels: (Legacy) Pixels from left/right for center_crop.
        crop_top: Pixels to crop from top (for custom).
        crop_bottom: Pixels to crop from bottom (for custom).
        crop_left: Pixels to crop from left (for custom).
        crop_right: Pixels to crop from right (for custom).
        depth_image: Depth image (for depth_range).
        min_depth_m: Min depth in meters.
        max_depth_m: Max depth in meters.
        factor_depth: Scale factor.

    Returns:
        Binary mask (H, W).
    """
    # Initialize full white mask by default (user wants solid white region)
    mask = np.ones((height, width), dtype=np.uint8) * 255

    if mask_type == "upper_half":
        mask[height // 2 :, :] = 0
        logger.info("Generated upper-half workspace mask")

    elif mask_type == "center_crop":
        mask[:, :crop_lr_pixels] = 0
        mask[:, -crop_lr_pixels:] = 0
        logger.info(f"Generated center-crop mask (lr={crop_lr_pixels}px)")

    elif mask_type == "custom":
        if crop_top > 0:
            mask[:crop_top, :] = 0
        if crop_bottom > 0:
            mask[-crop_bottom:, :] = 0
        if crop_left > 0:
            mask[:, :crop_left] = 0
        if crop_right > 0:
            mask[:, -crop_right:] = 0
        logger.info(
            f"Generated custom mask: top={crop_top}, bot={crop_bottom}, "
            f"left={crop_left}, right={crop_right}"
        )

    elif mask_type == "depth_range":
        if depth_image is None:
            raise ValueError("depth_image required for depth_range mask")
        depth_m = depth_image.astype(np.float32) / factor_depth
        mask = (
            (depth_image > 0)
            & (depth_m >= min_depth_m)
            & (depth_m <= max_depth_m)
        ).astype(np.uint8) * 255
        logger.info(
            f"Generated depth-range workspace mask [{min_depth_m}, {max_depth_m}]m"
        )
    
    else:
        logger.warning(f"Unknown mask_type: {mask_type}, using full mask")

    return mask


def save_graspnet_format(
    color_image: np.ndarray,
    depth_image: np.ndarray,
    intrinsic_matrix: np.ndarray,
    output_dir: str | Path,
    workspace_mask: Optional[np.ndarray] = None,
    factor_depth: int = 1000,
) -> Path:
    """Save captured data in GraspNet-baseline format.

    Creates:
        - color.png (BGR image, GraspNet reads with PIL → RGB)
        - depth.png (16-bit depth image)
        - workspace_mask.png (binary mask)
        - meta.mat (intrinsic matrix and factor_depth)

    Args:
        color_image: BGR image array (H, W, 3).
        depth_image: Depth image array (H, W) uint16, in mm.
        intrinsic_matrix: 3x3 camera intrinsic matrix.
        output_dir: Directory to save files.
        workspace_mask: Binary mask (H, W). If None, defaults to upper-half.
        factor_depth: Depth scale factor (1000 = mm).

    Returns:
        Path to the output directory.
    """
    import scipy.io as scio

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save color image
    cv2.imwrite(str(output_dir / "color.png"), color_image)

    # Save depth image
    cv2.imwrite(str(output_dir / "depth.png"), depth_image)

    # Save workspace mask
    if workspace_mask is None:
        h, w = depth_image.shape[:2]
        workspace_mask = generate_workspace_mask(h, w, mask_type="depth_range")
    cv2.imwrite(str(output_dir / "workspace_mask.png"), workspace_mask)

    # Save meta information
    meta = {
        "intrinsic_matrix": intrinsic_matrix,
        "factor_depth": np.array([[factor_depth]], dtype=np.float64),
    }
    scio.savemat(str(output_dir / "meta.mat"), meta)

    logger.info(f"Saved GraspNet-format data to {output_dir}")
    return output_dir


# ─── Standalone test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    print("RGBD Capture Test")
    print("=" * 40)

    output_path = Path(__file__).parent / "captured_data"

    with RGBDCapture(width=1280, height=720, color_fps=15, depth_fps=5) as cam:
        color, depth, intrinsic = cam.capture_with_preview()
        print(f"Color shape: {color.shape}, dtype: {color.dtype}")
        print(f"Depth shape: {depth.shape}, dtype: {depth.dtype}")
        print(f"Intrinsic matrix:\n{intrinsic}")

        mask = generate_workspace_mask(color.shape[0], color.shape[1], "upper_half")
        saved = save_graspnet_format(color, depth, intrinsic, output_path, workspace_mask=mask)
        print(f"\nSaved to: {saved}")
        print("Files:", [f.name for f in saved.iterdir()])
