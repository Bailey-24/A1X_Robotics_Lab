"""RealSense D405 VisionSensor adapter for the control_your_robot framework.

Runs a continuous capture pipeline in a background thread and exposes the
latest color/depth frames via the VisionSensor interface.  Also provides
``get_capture_for_detection()`` so that yoloe_grasp step 3 can obtain a
detection frame without opening a second RealSense pipeline.
"""
from __future__ import annotations

import logging
import sys
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

# Add control_your_robot to path
_CYR_SRC = str(Path(__file__).resolve().parent.parent / "refence_code" / "control_your_robot" / "src")
if _CYR_SRC not in sys.path:
    sys.path.insert(0, _CYR_SRC)

from robot.sensor.vision_sensor import VisionSensor

import pyrealsense2 as rs

logger = logging.getLogger("d405_sensor")


class D405Sensor(VisionSensor):
    """Continuous-capture RealSense D405 sensor.

    A background thread grabs aligned color+depth frames and caches them
    behind a lock.  ``get_image()`` (called by the recording thread) and
    ``get_capture_for_detection()`` (called by the main thread for object
    detection) both read from this cache without pipeline contention.
    """

    def __init__(self, name: str, serial: Optional[str] = None):
        super().__init__()
        self.name = name
        self._serial = serial
        self.is_jpeg = False  # store raw numpy; encode at conversion time

        # Pipeline / alignment objects (created in set_up)
        self.pipeline: Optional[rs.pipeline] = None
        self.align: Optional[rs.align] = None
        self.intrinsics: Optional[rs.intrinsics] = None
        self._intrinsic_matrix: Optional[np.ndarray] = None

        # Thread-safe frame cache
        self._frame_lock = threading.Lock()
        self._latest_color: Optional[np.ndarray] = None  # BGR uint8
        self._latest_depth: Optional[np.ndarray] = None  # uint16
        self._running = False
        self._grab_thread: Optional[threading.Thread] = None

    # ── Setup / teardown ───────────────────────────────────────────────

    def set_up(
        self,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        is_depth: bool = True,
    ):
        """Start the RealSense pipeline and background frame-grab thread."""
        self.pipeline = rs.pipeline()
        config = rs.config()

        if self._serial:
            config.enable_device(self._serial)

        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        if is_depth:
            config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

        self.align = rs.align(rs.stream.color)

        profile = self.pipeline.start(config)

        # Extract intrinsics
        color_profile = profile.get_stream(rs.stream.color)
        self.intrinsics = color_profile.as_video_stream_profile().get_intrinsics()
        self._intrinsic_matrix = np.array([
            [self.intrinsics.fx, 0.0, self.intrinsics.ppx],
            [0.0, self.intrinsics.fy, self.intrinsics.ppy],
            [0.0, 0.0, 1.0],
        ])

        logger.info(
            f"D405 pipeline started: {width}x{height}@{fps}fps "
            f"(serial={self._serial or 'auto'})"
        )

        # Warmup frames
        for _ in range(15):
            self.pipeline.wait_for_frames(timeout_ms=5000)

        # Start background grab thread
        self._running = True
        self._grab_thread = threading.Thread(target=self._grab_loop, daemon=True)
        self._grab_thread.start()

    def stop(self):
        """Stop the background thread and pipeline."""
        self._running = False
        if self._grab_thread is not None:
            self._grab_thread.join(timeout=3.0)
            self._grab_thread = None
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except Exception:
                pass
            self.pipeline = None

    # ── Background frame grab ──────────────────────────────────────────

    def _grab_loop(self):
        while self._running:
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=2000)
            except RuntimeError:
                continue
            aligned = self.align.process(frames)
            cf = aligned.get_color_frame()
            df = aligned.get_depth_frame()
            if cf and df:
                color = np.asanyarray(cf.get_data()).copy()
                depth = np.asanyarray(df.get_data()).copy()
                with self._frame_lock:
                    self._latest_color = color  # BGR
                    self._latest_depth = depth  # uint16

    # ── VisionSensor interface ─────────────────────────────────────────

    def get_image(self) -> Dict[str, Any]:
        """Return latest cached frame (thread-safe).

        Returns dict with:
            color: np.ndarray (H, W, 3) RGB uint8
            depth: np.ndarray (H, W) uint16  (if depth in collect_info)
        """
        with self._frame_lock:
            image: Dict[str, Any] = {}
            if self._latest_color is not None:
                # BGR → RGB for storage
                image["color"] = self._latest_color[:, :, ::-1].copy()
            else:
                image["color"] = np.zeros((480, 640, 3), dtype=np.uint8)

            if self._latest_depth is not None:
                image["depth"] = self._latest_depth.copy()
            else:
                image["depth"] = np.zeros((480, 640), dtype=np.uint16)

            return image

    # ── Detection helper (for yoloe_grasp step 3) ──────────────────────

    def get_capture_for_detection(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (color_bgr, depth_uint16, intrinsic_3x3) for object detection.

        This replaces ``RGBDCapture`` in the yoloe_grasp pipeline so that
        the D405 is not opened by two processes.
        """
        with self._frame_lock:
            if self._latest_color is None or self._latest_depth is None:
                raise RuntimeError("D405Sensor: no frames captured yet")
            color_bgr = self._latest_color.copy()
            depth = self._latest_depth.copy()
        return color_bgr, depth, self._intrinsic_matrix.copy()
