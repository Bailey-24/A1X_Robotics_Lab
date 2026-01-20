#!/usr/bin/env python3
"""ArUco Marker Pose Tracker with Intel RealSense D405.

This script captures RGB frames from an Intel RealSense D405 camera,
detects ArUco markers, and estimates their 6-DOF poses in real-time.

The script is designed to work with ArUco sheets generated with a 4x3 grid
of 50mm markers (e.g., arucosheet_4x3_50mm_ID.pdf).

Usage:
    python examples/aruco_pose_tracker.py [--marker-size 0.05] [--dict 4X4_50]

Controls:
    - Press 'q' or ESC to quit
    - Press 's' to save current frame with pose overlay
    - Press 'd' to toggle debug info display
    - Press 'p' to pause/resume

Requirements:
    pip install pyrealsense2 opencv-python numpy
"""
from __future__ import annotations

import argparse
import sys
import logging
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:
    print("Error: pyrealsense2 not installed.")
    print("Install with: pip install pyrealsense2")
    sys.exit(1)


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


ARUCO_DICT_MAP = {
    "4X4_50": cv2.aruco.DICT_4X4_50,
    "4X4_100": cv2.aruco.DICT_4X4_100,
    "4X4_250": cv2.aruco.DICT_4X4_250,
    "4X4_1000": cv2.aruco.DICT_4X4_1000,
    "5X5_50": cv2.aruco.DICT_5X5_50,
    "5X5_100": cv2.aruco.DICT_5X5_100,
    "5X5_250": cv2.aruco.DICT_5X5_250,
    "5X5_1000": cv2.aruco.DICT_5X5_1000,
    "6X6_50": cv2.aruco.DICT_6X6_50,
    "6X6_100": cv2.aruco.DICT_6X6_100,
    "6X6_250": cv2.aruco.DICT_6X6_250,
    "6X6_1000": cv2.aruco.DICT_6X6_1000,
    "7X7_50": cv2.aruco.DICT_7X7_50,
    "7X7_100": cv2.aruco.DICT_7X7_100,
    "7X7_250": cv2.aruco.DICT_7X7_250,
    "7X7_1000": cv2.aruco.DICT_7X7_1000,
    "ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
}


@dataclass
class MarkerPose:
    """Represents the 6-DOF pose of a detected ArUco marker."""
    marker_id: int
    rvec: np.ndarray  # Rotation vector (Rodrigues)
    tvec: np.ndarray  # Translation vector (x, y, z in meters)
    corners: np.ndarray  # 4 corner points in image
    
    @property
    def rotation_matrix(self) -> np.ndarray:
        """Convert rotation vector to 3x3 rotation matrix."""
        R, _ = cv2.Rodrigues(self.rvec)
        return R
    
    @property
    def euler_angles_deg(self) -> tuple[float, float, float]:
        """Get Euler angles (roll, pitch, yaw) in degrees."""
        R = self.rotation_matrix
        sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        singular = sy < 1e-6
        
        if not singular:
            roll = np.arctan2(R[2, 1], R[2, 2])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = np.arctan2(R[1, 0], R[0, 0])
        else:
            roll = np.arctan2(-R[1, 2], R[1, 1])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = 0
        
        return (
            np.degrees(roll),
            np.degrees(pitch),
            np.degrees(yaw)
        )
    
    @property
    def position_mm(self) -> tuple[float, float, float]:
        """Get position in millimeters."""
        return (
            self.tvec[0][0] * 1000,
            self.tvec[1][0] * 1000,
            self.tvec[2][0] * 1000
        )


class ArucoPoseTracker:
    """Real-time ArUco marker pose tracker using RealSense camera."""
    
    def __init__(
        self,
        marker_size_m: float = 0.05,
        aruco_dict_name: str = "4X4_50",
        resolution: tuple[int, int] = (640, 480),
        fps: int = 30
    ):
        """Initialize the pose tracker.
        
        Args:
            marker_size_m: Physical size of the marker in meters (default 50mm = 0.05m)
            aruco_dict_name: Name of ArUco dictionary to use
            resolution: Camera resolution (width, height)
            fps: Frames per second
        """
        self.marker_size_m = marker_size_m
        self.resolution = resolution
        self.fps = fps
        
        self.pipeline: Optional[rs.pipeline] = None
        self.config: Optional[rs.config] = None
        self.camera_matrix: Optional[np.ndarray] = None
        self.dist_coeffs: Optional[np.ndarray] = None
        
        if aruco_dict_name not in ARUCO_DICT_MAP:
            raise ValueError(
                f"Unknown ArUco dictionary: {aruco_dict_name}. "
                f"Available: {list(ARUCO_DICT_MAP.keys())}"
            )
        
        dict_type = ARUCO_DICT_MAP[aruco_dict_name]
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(dict_type)
        self.aruco_dict_name = aruco_dict_name
        
        try:
            self.detector_params = cv2.aruco.DetectorParameters()
            # Tune parameters for better detection range and accuracy
            self.detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
            # Reduce false positives - RELAXED PARAMETERS to fix detection
            self.detector_params.minMarkerPerimeterRate = 0.03 
            self.detector_params.polygonalApproxAccuracyRate = 0.03  # Reverted to default (was 0.01)
            self.detector_params.maxErroneousBitsInBorderRate = 0.35 # Reverted to default (was 0.15)
            self.detector_params.errorCorrectionRate = 0.6           # Reverted to default (was 0.75)
            
            self.detector_params.adaptiveThreshWinSizeMin = 3
            self.detector_params.adaptiveThreshWinSizeMax = 23       # Reverted to default (was 45)
            self.detector_params.adaptiveThreshWinSizeStep = 10      # Reverted to default (was 5)
            self.detector_params.adaptiveThreshConstant = 7          # Reverted to default (was 8)
            
            self.detector = cv2.aruco.ArucoDetector(
                self.aruco_dict, self.detector_params
            )
            self._use_new_api = True
        except AttributeError:
            self.detector_params = cv2.aruco.DetectorParameters_create()
            # Tune parameters for legacy API
            self.detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
            self.detector_params.minMarkerPerimeterRate = 0.03
            self.detector_params.polygonalApproxAccuracyRate = 0.03
            self.detector_params.maxErroneousBitsInBorderRate = 0.35
            self.detector_params.errorCorrectionRate = 0.6
            
            self.detector = None
            self._use_new_api = False
        
        logger.info(f"ArUco dictionary: {aruco_dict_name}")
        logger.info(f"Marker size: {marker_size_m * 1000:.1f} mm")
        logger.info(f"Using {'new' if self._use_new_api else 'legacy'} OpenCV ArUco API")
        
        self.show_debug = True
        self.paused = False
        self.frame_count = 0
        
        self.output_dir = Path(__file__).parent / "aruco_captures"
        self.output_dir.mkdir(exist_ok=True)
    
    def _find_realsense_device(self) -> Optional[rs.device]:
        """Find and return a RealSense device with color capability."""
        ctx = rs.context()
        devices = ctx.query_devices()
        
        if len(devices) == 0:
            logger.error("No RealSense devices detected")
            return None
        
        for device in devices:
            name = device.get_info(rs.camera_info.name)
            serial = device.get_info(rs.camera_info.serial_number)
            logger.info(f"Found device: {name} (Serial: {serial})")
            
            sensors = list(device.sensors)
            has_color = any(
                s.get_info(rs.camera_info.name) in ('RGB Camera', 'Stereo Module')
                for s in sensors
            )
            
            if has_color:
                return device
        
        return None
    
    def start(self) -> bool:
        """Start the RealSense pipeline and extract camera intrinsics.
        
        Returns:
            True if started successfully, False otherwise.
        """
        device = self._find_realsense_device()
        if device is None:
            return False
        
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # Try requested config first
        width, height = self.resolution
        
        # List of profiles to try (width, height, fps)
        # Priority: User request -> 1280x720@15 -> 848x480@30 -> 640x480@30
        profiles_to_try = [
            (width, height, self.fps),
            (1280, 720, 15),
            (1280, 720, 5),
            (848, 480, 30),  # Might not be supported on D405
            (848, 480, 10),
            (640, 480, 30),
            (640, 480, 15)
        ]
        
        # Remove duplicates
        seen = set()
        unique_profiles = []
        for p in profiles_to_try:
            if p not in seen:
                unique_profiles.append(p)
                seen.add(p)
        
        pipeline_started = False
        for w, h, f in unique_profiles:
            try:
                logger.info(f"Trying config: {w}x{h} @ {f}fps")
                self.config.enable_stream(rs.stream.color, w, h, rs.format.bgr8, f)
                profile = self.pipeline.start(self.config)
                pipeline_started = True
                
                # Update actual config
                self.resolution = (w, h)
                self.fps = f
                logger.info(f"Pipeline started successfully with {w}x{h} @ {f}fps")
                break
            except RuntimeError as e:
                logger.warning(f"Failed to start with {w}x{h} @ {f}fps: {e}")
                # Reset config for next attempt
                self.config = rs.config()
        
        if not pipeline_started:
            logger.error("Failed to start pipeline with any configuration")
            return False
            
        try:
            color_profile = profile.get_stream(rs.stream.color)
            intrinsics = color_profile.as_video_stream_profile().get_intrinsics()
            
            # K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
            self.camera_matrix = np.array([
                [intrinsics.fx, 0, intrinsics.ppx],
                [0, intrinsics.fy, intrinsics.ppy],
                [0, 0, 1]
            ], dtype=np.float64)
            
            # [k1, k2, p1, p2, k3] Brown-Conrady distortion model
            self.dist_coeffs = np.array(intrinsics.coeffs[:5], dtype=np.float64)
            
            logger.info(f"Camera resolution: {intrinsics.width}x{intrinsics.height}")
            logger.info(f"Focal length: fx={intrinsics.fx:.2f}, fy={intrinsics.fy:.2f}")
            logger.info(f"Principal point: cx={intrinsics.ppx:.2f}, cy={intrinsics.ppy:.2f}")
            logger.info(f"Distortion model: {intrinsics.model}")
            
            return True
            
        except RuntimeError as e:
            logger.error(f"Failed to start pipeline: {e}")
            return False
    
    def stop(self) -> None:
        """Stop the RealSense pipeline."""
        if self.pipeline:
            self.pipeline.stop()
            logger.info("Pipeline stopped")
    
    def detect_markers(self, image: np.ndarray) -> tuple[list[MarkerPose], list[np.ndarray]]:
        """Detect ArUco markers and estimate their poses.
        
        Args:
            image: BGR image from camera
            
        Returns:
            Tuple of (list of MarkerPose, list of rejected candidates)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        if self._use_new_api:
            corners, ids, rejected = self.detector.detectMarkers(gray)
        else:
            corners, ids, rejected = cv2.aruco.detectMarkers(
                gray, self.aruco_dict, parameters=self.detector_params
            )
        
        poses: list[MarkerPose] = []
        
        if ids is None or len(ids) == 0:
            return poses, rejected
        
        half_size = self.marker_size_m / 2.0
        obj_points = np.array([
            [-half_size,  half_size, 0],
            [ half_size,  half_size, 0],
            [ half_size, -half_size, 0],
            [-half_size, -half_size, 0]
        ], dtype=np.float32)
        
        for i, marker_id in enumerate(ids.flatten()):
            if marker_id != 0:
                continue

            img_points = corners[i][0].astype(np.float32)
            success, rvec, tvec = cv2.solvePnP(
                obj_points,
                img_points,
                self.camera_matrix,
                self.dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE_SQUARE
            )
            
            if success:
                pose = MarkerPose(
                    marker_id=int(marker_id),
                    rvec=rvec,
                    tvec=tvec,
                    corners=corners[i][0]
                )
                poses.append(pose)
        
        return poses, rejected
    
    def draw_poses(
        self,
        image: np.ndarray,
        poses: list[MarkerPose],
        rejected: Optional[list[np.ndarray]] = None,
        axis_length: float = 0.08  # Increased for better visibility
    ) -> np.ndarray:
        """Draw detected marker poses on the image.
        
        Args:
            image: BGR image to draw on
            poses: List of detected marker poses
            rejected: List of rejected marker candidates (optional)
            axis_length: Length of coordinate axes to draw (in meters)
            
        Returns:
            Image with pose visualization overlay.
        """
        output = image.copy()
        
        # Draw rejected candidates if debug is on
        if self.show_debug and rejected is not None and len(rejected) > 0:
            cv2.aruco.drawDetectedMarkers(output, rejected, borderColor=(0, 0, 255))
        
        for pose in poses:
            corners_int = pose.corners.astype(np.int32)
            cv2.polylines(
                output,
                [corners_int],
                True,
                (0, 255, 0),
                2
            )
            
            cv2.drawFrameAxes(
                output,
                self.camera_matrix,
                self.dist_coeffs,
                pose.rvec,
                pose.tvec,
                axis_length
            )
            
            center = corners_int.mean(axis=0).astype(int)
            x, y, z = pose.position_mm
            r, p, yaw = pose.euler_angles_deg
            
            # Display 6D Pose: T[x,y,z] R[r,p,y]
            # Line 1: ID
            cv2.putText(
                output,
                f"ID:{pose.marker_id}",
                (center[0], center[1] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 255),
                2
            )
            
            # Line 2: Translation
            cv2.putText(
                output,
                f"T: [{x:.1f}, {y:.1f}, {z:.1f}] mm",
                (center[0], center[1] + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255), # Yellow
                2
            )
            
            # Line 3: Rotation
            cv2.putText(
                output,
                f"R: [{r:.1f}, {p:.1f}, {yaw:.1f}] deg",
                (center[0], center[1] + 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255), # Yellow
                2
            )
        
        return output
    
    def draw_info_panel(
        self,
        image: np.ndarray,
        poses: list[MarkerPose],
        fps: float
    ) -> np.ndarray:
        """Draw information panel on the image.
        
        Args:
            image: Image to draw on
            poses: Detected poses
            fps: Current FPS
            
        Returns:
            Image with info panel.
        """
        output = image.copy()
        h, w = output.shape[:2]
        
        panel_height = 80 + len(poses) * 25
        overlay = output.copy()
        cv2.rectangle(overlay, (5, 5), (400, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, output, 0.4, 0, output)
        
        y_offset = 25
        cv2.putText(
            output,
            f"ArUco Pose Tracker | Dict: {self.aruco_dict_name} | Marker: {self.marker_size_m*1000:.0f}mm",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
        y_offset += 25
        
        status = "PAUSED" if self.paused else "LIVE"
        status_color = (0, 255, 255) if self.paused else (0, 255, 0)
        cv2.putText(
            output,
            f"FPS: {fps:.1f} | Frame: {self.frame_count} | Status: {status} | Markers: {len(poses)}",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            status_color,
            1
        )
        y_offset += 25
        
        for pose in poses[:5]:
            x, y, z = pose.position_mm
            roll, pitch, yaw = pose.euler_angles_deg
            cv2.putText(
                output,
                f"ID {pose.marker_id:2d}: pos=({x:6.0f},{y:6.0f},{z:6.0f})mm "
                f"rot=({roll:5.1f},{pitch:5.1f},{yaw:5.1f})deg",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (200, 200, 200),
                1
            )
            y_offset += 20
        
        if len(poses) > 5:
            cv2.putText(
                output,
                f"... and {len(poses) - 5} more markers",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (150, 150, 150),
                1
            )
        
        help_text = "[Q]uit [S]ave [D]ebug [P]ause"
        cv2.putText(
            output,
            help_text,
            (w - 250, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (150, 150, 150),
            1
        )
        
        return output
    
    def run(self) -> int:
        """Main loop to capture frames and track poses.
        
        Returns:
            Exit code (0 for success, 1 for failure).
        """
        if not self.start():
            return 1
        
        logger.info("Starting pose tracking... Press 'q' to quit")
        
        fps_start_time = cv2.getTickCount()
        fps_frame_count = 0
        current_fps = 0.0
        
        last_image = None
        last_poses: list[MarkerPose] = []
        last_rejected: list[np.ndarray] = []
        
        try:
            while True:
                if not self.paused:
                    try:
                        frames = self.pipeline.wait_for_frames(timeout_ms=5000)
                    except RuntimeError:
                        logger.warning("Frame timeout")
                        continue
                    
                    color_frame = frames.get_color_frame()
                    if not color_frame:
                        continue
                    
                    self.frame_count += 1
                    fps_frame_count += 1
                    
                    color_image = np.asanyarray(color_frame.get_data())
                    
                    poses, rejected = self.detect_markers(color_image)
                    
                    if len(poses) == 0 and len(rejected) > 0 and self.frame_count % 30 == 0:
                        logger.warning(f"No markers accepted! Rejected candidates: {len(rejected)}")

                    for pose in poses:
                        x, y, z = pose.position_mm
                        r, p, yaw = pose.euler_angles_deg
                        # Print 6D Pose in coordinate system format
                        print(f"ID:{pose.marker_id} | T:[{x:6.1f}, {y:6.1f}, {z:6.1f}] mm | R:[{r:6.1f}, {p:6.1f}, {yaw:6.1f}] deg")
                    
                    display_image = self.draw_poses(color_image, poses, rejected)
                    
                    last_image = color_image
                    last_poses = poses
                    last_rejected = rejected
                else:
                    if last_image is not None:
                        display_image = self.draw_poses(last_image, last_poses, last_rejected)
                        poses = last_poses
                    else:
                        continue
                
                elapsed = (cv2.getTickCount() - fps_start_time) / cv2.getTickFrequency()
                if elapsed >= 1.0:
                    current_fps = fps_frame_count / elapsed
                    fps_frame_count = 0
                    fps_start_time = cv2.getTickCount()
                
                display_image = self.draw_info_panel(display_image, poses, current_fps)
                
                cv2.imshow('ArUco Pose Tracker', display_image)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == 27:
                    logger.info("Quit requested")
                    break
                elif key == ord('s'):
                    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = self.output_dir / f"aruco_pose_{timestamp_str}.png"
                    cv2.imwrite(str(filename), display_image)
                    logger.info(f"Frame saved to {filename}")
                elif key == ord('d'):
                    self.show_debug = not self.show_debug
                    logger.info(f"Debug display: {'ON' if self.show_debug else 'OFF'}")
                elif key == ord('p'):
                    self.paused = not self.paused
                    logger.info(f"{'Paused' if self.paused else 'Resumed'}")
                    
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.stop()
            cv2.destroyAllWindows()
            logger.info(f"Total frames processed: {self.frame_count}")
        
        return 0


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Real-time ArUco marker pose tracking with RealSense camera",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Default: 50mm markers with DICT_4X4_50
    python examples/aruco_pose_tracker.py
    
    # Use different marker size (e.g., 100mm)
    python examples/aruco_pose_tracker.py --marker-size 0.1
    
    # Use different ArUco dictionary
    python examples/aruco_pose_tracker.py --dict 6X6_250
    
    # Higher resolution
    python examples/aruco_pose_tracker.py --width 1280 --height 720

ArUco Sheet Generation (the command you used):
    # Generate a 4x3 ArUco grid with 50mm markers:
    python -c "
import cv2
import numpy as np

dict_type = cv2.aruco.DICT_4X4_50
dictionary = cv2.aruco.getPredefinedDictionary(dict_type)

# Parameters
cols, rows = 4, 3
marker_size_px = 200  # pixels per marker
margin_px = 50

# Create grid
img_w = cols * marker_size_px + (cols + 1) * margin_px
img_h = rows * marker_size_px + (rows + 1) * margin_px
sheet = np.ones((img_h, img_w), dtype=np.uint8) * 255

for row in range(rows):
    for col in range(cols):
        marker_id = row * cols + col
        marker = cv2.aruco.generateImageMarker(dictionary, marker_id, marker_size_px)
        x = margin_px + col * (marker_size_px + margin_px)
        y = margin_px + row * (marker_size_px + margin_px)
        sheet[y:y+marker_size_px, x:x+marker_size_px] = marker

cv2.imwrite('arucosheet_4x3_50mm_ID.png', sheet)
print('Saved arucosheet_4x3_50mm_ID.png - print at correct scale for 50mm markers')
"
        """
    )
    
    parser.add_argument(
        "--marker-size", "-m",
        type=float,
        default=0.05,
        help="Physical marker size in meters (default: 0.05 = 50mm)"
    )
    
    parser.add_argument(
        "--dict", "-d",
        type=str,
        default="4X4_50",
        choices=list(ARUCO_DICT_MAP.keys()),
        help="ArUco dictionary to use (default: 4X4_50)"
    )
    
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Camera resolution width (default: 1280)"
    )
    
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Camera resolution height (default: 720)"
    )
    
    parser.add_argument(
        "--fps",
        type=int,
        default=15,
        help="Camera FPS (default: 15 for 1280x720)"
    )
    
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()
    
    tracker = ArucoPoseTracker(
        marker_size_m=args.marker_size,
        aruco_dict_name=args.dict,
        resolution=(args.width, args.height),
        fps=args.fps
    )
    
    return tracker.run()


if __name__ == "__main__":
    sys.exit(main())
