#!/usr/bin/env python3
"""
Hand-Eye Calibration Tool for A1X Robot (Eye-in-Hand)

This script performs hand-eye calibration for the A1X robotic arm with a camera
mounted on the end-effector. Features:
- Viser-based IK control to move the robot by dragging a 3D gizmo
- Real-time camera view with ArUco marker detection
- Pose capture and calibration computation
- YAML output with T_ee_cam transformation

Usage:
    cd /home/ubuntu/projects/A1Xsdk
    conda activate a1x_ros
    python handeye/handeye_calibration.py --marker_id 42 --marker_size 0.10

Then open http://localhost:8080/ in your browser.

Controls (in Viser UI):
    - Drag the RED SPHERE to move robot end-effector
    - Enable "Robot Control" checkbox to actually move the robot
    - Click "Capture Pose" button when marker is detected
    - Click "Compute Calibration" when ≥10 poses captured
"""
from __future__ import annotations

import argparse
import sys
import os
import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml

try:
    import pyrealsense2 as rs
except ImportError:
    print("Error: pyrealsense2 not installed.")
    print("Install with: pip install pyrealsense2")
    sys.exit(1)

# Add parent directory to path for a1x_control
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import a1x_control

# Viser and PyRoki imports
import viser
from viser.extras import ViserUrdf
import yourdfpy

# PyRoki imports
sys.path.insert(0, str(Path(__file__).parent.parent / "pyroki" / "examples"))
import pyroki as pk
import pyroki_snippets as pks

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ArUco dictionary mapping
ARUCO_DICT_MAP = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
}


def quaternion_to_rotation_matrix(q: tuple) -> np.ndarray:
    """Convert quaternion (x, y, z, w) to 3x3 rotation matrix."""
    x, y, z, w = q
    return np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x**2 + y**2)]
    ])


def rotation_matrix_to_quaternion(R: np.ndarray) -> tuple:
    """Convert 3x3 rotation matrix to quaternion (x, y, z, w)."""
    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return (x, y, z, w)


def load_a1x_urdf() -> yourdfpy.URDF:
    """Load the A1X URDF with proper mesh path resolution."""
    urdf_path = Path("/home/ubuntu/projects/A1Xsdk/install/mobiman/lib/mobiman/configs/urdfs/a1x.urdf")
    
    def resolve_package_uri(fname: str) -> str:
        package_prefix = "package://mobiman/"
        if fname.startswith(package_prefix):
            relative_path = fname[len(package_prefix):]
            return str(Path("/home/ubuntu/projects/A1Xsdk/install/mobiman/share/mobiman") / relative_path)
        return fname
    
    return yourdfpy.URDF.load(urdf_path, filename_handler=resolve_package_uri)


class CameraThread:
    """Background thread for camera capture and ArUco detection."""
    
    def __init__(
        self,
        marker_id: int,
        marker_size: float,
        dict_type: str = "DICT_4X4_50",
    ):
        self.marker_id = marker_id
        self.marker_size = marker_size
        
        # ArUco detector
        aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_MAP[dict_type])
        self.aruco_detector = cv2.aruco.ArucoDetector(
            aruco_dict, cv2.aruco.DetectorParameters()
        )
        
        # Camera state
        self.camera_matrix: Optional[np.ndarray] = None
        self.dist_coeffs: Optional[np.ndarray] = None
        self.latest_frame: Optional[np.ndarray] = None
        self.latest_rvec: Optional[np.ndarray] = None
        self.latest_tvec: Optional[np.ndarray] = None
        self.marker_detected = False
        
        self.lock = threading.Lock()
        self.running = True
        self.pipeline: Optional[rs.pipeline] = None
        
        self.thread = threading.Thread(target=self._run, daemon=True)
    
    def start(self) -> bool:
        """Initialize camera and start capture thread."""
        try:
            self.pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 15)
            
            profile = self.pipeline.start(config)
            color_profile = profile.get_stream(rs.stream.color)
            intrinsics = color_profile.as_video_stream_profile().get_intrinsics()
            
            self.camera_matrix = np.array([
                [intrinsics.fx, 0, intrinsics.ppx],
                [0, intrinsics.fy, intrinsics.ppy],
                [0, 0, 1]
            ])
            self.dist_coeffs = np.array(intrinsics.coeffs)
            
            logger.info(f"Camera started: {intrinsics.width}x{intrinsics.height}")
            self.thread.start()
            return True
        except Exception as e:
            logger.error(f"Failed to start camera: {e}")
            return False
    
    def _run(self) -> None:
        """Camera capture loop."""
        while self.running:
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=1000)
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue
                
                frame = np.asanyarray(color_frame.get_data())
                rvec, tvec, annotated = self._detect_marker(frame)
                
                with self.lock:
                    self.latest_frame = annotated
                    self.latest_rvec = rvec
                    self.latest_tvec = tvec
                    self.marker_detected = rvec is not None
                    
            except Exception as e:
                if self.running:
                    logger.warning(f"Camera frame error: {e}")
    
    def _detect_marker(self, frame: np.ndarray) -> tuple:
        """Detect ArUco marker and return pose using solvePnP (OpenCV 4.8+ compatible)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.aruco_detector.detectMarkers(gray)
        
        annotated = frame.copy()
        rvec, tvec = None, None
        
        if ids is not None and self.marker_id in ids:
            idx = np.where(ids.flatten() == self.marker_id)[0][0]
            marker_corners = corners[idx].reshape(-1, 2)
            
            # Define 3D object points for the marker (centered at origin)
            half_size = self.marker_size / 2.0
            obj_points = np.array([
                [-half_size,  half_size, 0],  # Top-left
                [ half_size,  half_size, 0],  # Top-right
                [ half_size, -half_size, 0],  # Bottom-right
                [-half_size, -half_size, 0],  # Bottom-left
            ], dtype=np.float32)
            
            # Use solvePnP for pose estimation (OpenCV 4.8+ compatible)
            success, rvec_out, tvec_out = cv2.solvePnP(
                obj_points, marker_corners,
                self.camera_matrix, self.dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE_SQUARE
            )
            
            if success:
                rvec = rvec_out.flatten()
                tvec = tvec_out.flatten()
            
            cv2.aruco.drawDetectedMarkers(annotated, corners, ids)
            if rvec is not None:
                cv2.drawFrameAxes(annotated, self.camera_matrix, self.dist_coeffs,
                                  rvec, tvec, self.marker_size * 0.5)
            cv2.putText(annotated, f"Marker {self.marker_id} DETECTED",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(annotated, f"Marker {self.marker_id} not detected",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        return rvec, tvec, annotated
    
    def get_state(self) -> tuple:
        """Get latest camera state (thread-safe)."""
        with self.lock:
            return (
                self.latest_frame.copy() if self.latest_frame is not None else None,
                self.latest_rvec.copy() if self.latest_rvec is not None else None,
                self.latest_tvec.copy() if self.latest_tvec is not None else None,
                self.marker_detected,
            )
    
    def stop(self) -> None:
        """Stop camera thread."""
        self.running = False
        if self.pipeline:
            self.pipeline.stop()


class HandEyeCalibrator:
    """Hand-eye calibration data collector and solver."""
    
    def __init__(self, marker_id: int, marker_size: float):
        self.marker_id = marker_id
        self.marker_size = marker_size
        
        self.ee_rotations: list[np.ndarray] = []
        self.ee_translations: list[np.ndarray] = []
        self.marker_rotations: list[np.ndarray] = []
        self.marker_translations: list[np.ndarray] = []
    
    def add_pose_pair(
        self, ee_pose: dict, marker_rvec: np.ndarray, marker_tvec: np.ndarray
    ) -> int:
        """Add a pose pair for calibration."""
        ee_pos = ee_pose['position']
        ee_ori = ee_pose['orientation']
        
        R_ee = quaternion_to_rotation_matrix((
            ee_ori['x'], ee_ori['y'], ee_ori['z'], ee_ori['w']
        ))
        t_ee = np.array([[ee_pos['x']], [ee_pos['y']], [ee_pos['z']]])
        
        R_marker, _ = cv2.Rodrigues(marker_rvec)
        t_marker = marker_tvec.reshape(3, 1)
        
        self.ee_rotations.append(R_ee)
        self.ee_translations.append(t_ee)
        self.marker_rotations.append(R_marker)
        self.marker_translations.append(t_marker)
        
        return len(self.ee_rotations)
    
    @property
    def num_poses(self) -> int:
        return len(self.ee_rotations)
    
    def calibrate(self) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """Perform hand-eye calibration."""
        if self.num_poses < 10:
            return None
        
        R_cam2ee, t_cam2ee = cv2.calibrateHandEye(
            self.ee_rotations, self.ee_translations,
            self.marker_rotations, self.marker_translations,
            method=cv2.CALIB_HAND_EYE_TSAI
        )
        return R_cam2ee, t_cam2ee
    
    def save(self, R: np.ndarray, t: np.ndarray, path: str) -> None:
        """Save calibration to YAML."""
        q = rotation_matrix_to_quaternion(R)
        # Convert to native Python types for clean YAML output
        result = {
            'transformation': {
                'rotation': [[float(v) for v in row] for row in R.tolist()],
                'translation': [float(v) for v in t.flatten()],
                'quaternion_xyzw': [float(v) for v in q],
            },
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'num_poses': self.num_poses,
                'marker_id': self.marker_id,
                'marker_size': self.marker_size,
            }
        }
        with open(path, 'w') as f:
            yaml.dump(result, f, default_flow_style=False, sort_keys=False)
        logger.info(f"Calibration saved to: {path}")


def main():
    parser = argparse.ArgumentParser(description="Hand-Eye Calibration with Viser IK Control")
    parser.add_argument("--marker_id", type=int, default=42)
    parser.add_argument("--marker_size", type=float, default=0.10)
    parser.add_argument("--dict_type", type=str, default="DICT_4X4_50",
                        choices=list(ARUCO_DICT_MAP.keys()))
    parser.add_argument("--output", type=str, default="handeye/handeye_calibration.yaml")
    args = parser.parse_args()
    
    print("=" * 60)
    print("A1X Hand-Eye Calibration with Viser IK Control")
    print("=" * 60)
    print()
    
    # =========================================================================
    # 1. Initialize A1X control with EE pose
    # =========================================================================
    print("[1/5] Initializing A1X control system...")
    a1x_control.initialize(enable_ee_pose=True)
    controller = a1x_control.JointController()
    
    if not controller.wait_for_ee_pose(timeout=15):
        logger.error("Failed to get EE pose. Exiting.")
        return 1
    print("       ✓ EE pose available")
    
    # =========================================================================
    # 2. Load URDF and create PyRoki robot
    # =========================================================================
    print("[2/5] Loading A1X URDF...")
    urdf = load_a1x_urdf()
    robot = pk.Robot.from_urdf(urdf)
    target_link_name = "gripper_link"
    print(f"       Robot: {robot.joints.num_actuated_joints} joints, target: {target_link_name}")
    
    # =========================================================================
    # 3. Start camera thread
    # =========================================================================
    print("[3/5] Starting camera...")
    camera = CameraThread(args.marker_id, args.marker_size, args.dict_type)
    if not camera.start():
        return 1
    
    # =========================================================================
    # 4. Initialize calibrator
    # =========================================================================
    print("[4/5] Initializing calibrator...")
    calibrator = HandEyeCalibrator(args.marker_id, args.marker_size)
    
    # =========================================================================
    # 5. Set up Viser server
    # =========================================================================
    print("[5/5] Starting Viser server...")
    server = viser.ViserServer()
    
    # Add ground grid
    server.scene.add_grid("/ground", width=2, height=2, cell_size=0.1)
    
    # Add robot visualization
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/robot")
    
    # Get initial joint configuration
    initial_joints = controller.get_joint_states()
    if initial_joints:
        cfg = [initial_joints.get(f'arm_joint{i+1}', 0.0) for i in range(6)]
        cfg.extend([0.0, 0.0])  # gripper
        urdf_vis.update_cfg(np.array(cfg))
        last_solution = np.array(cfg)
    else:
        last_solution = np.zeros(8)
    
    smoothed_solution = last_solution.copy()
    
    # --- IK Target Gizmo ---
    ik_target = server.scene.add_transform_controls(
        "/ik_target", scale=0.15,
        position=(0.25, 0.0, 0.2), wxyz=(1, 0, 0, 0),
        depth_test=False,
    )
    server.scene.add_icosphere("/ik_target/sphere", radius=0.015, color=(255, 100, 100))
    
    # --- GUI Controls ---
    with server.gui.add_folder("Robot Control"):
        enable_robot = server.gui.add_checkbox("Enable Robot Control", initial_value=False)
        smoothing_alpha = server.gui.add_slider("Smoothing", min=0.05, max=1.0, step=0.05, initial_value=0.2)
    
    with server.gui.add_folder("Calibration"):
        poses_display = server.gui.add_number("Poses Captured", initial_value=0, disabled=True)
        marker_status = server.gui.add_text("Marker Status", initial_value="Not detected")
        capture_btn = server.gui.add_button("Capture Pose")
        calibrate_btn = server.gui.add_button("Compute Calibration")
        status_display = server.gui.add_text("Status", initial_value="Move robot to see marker")
    
    with server.gui.add_folder("Actions"):
        sync_btn = server.gui.add_button("Sync Target to Current EE")
    
    # Capture button handler
    capture_requested = [False]
    calibrate_requested = [False]
    
    @capture_btn.on_click
    def on_capture(_):
        capture_requested[0] = True
    
    @calibrate_btn.on_click
    def on_calibrate(_):
        calibrate_requested[0] = True
    
    @sync_btn.on_click
    def on_sync(_):
        joints = controller.get_joint_states()
        if joints:
            cfg = [joints.get(f'arm_joint{i+1}', 0.0) for i in range(6)]
            cfg.extend([0.0, 0.0])
            target_idx = robot.links.names.index(target_link_name)
            fk_result = robot.forward_kinematics(np.array(cfg))
            ee_pose = fk_result[target_idx]
            ik_target.position = tuple(ee_pose[4:7])
            ik_target.wxyz = tuple(ee_pose[0:4])
    
    print()
    print("=" * 60)
    print("READY! Open http://localhost:8080/ in your browser")
    print("=" * 60)
    print()
    print("Workflow:")
    print("  1. Enable 'Robot Control' checkbox")
    print("  2. Drag the RED SPHERE to move robot (marker must be visible)")
    print("  3. Click 'Capture Pose' when marker is detected")
    print("  4. Repeat at 10+ different poses")
    print("  5. Click 'Compute Calibration'")
    print()
    
    # =========================================================================
    # Main loop
    # =========================================================================
    output_path = args.output
    
    # Create OpenCV window for camera feed
    cv2.namedWindow("Camera - Hand-Eye Calibration", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Camera - Hand-Eye Calibration", 960, 540)
    
    try:
        while True:
            loop_start = time.time()
            
            # Get camera state
            frame, rvec, tvec, marker_detected = camera.get_state()
            
            # Display camera feed in OpenCV window
            if frame is not None:
                # Add pose count overlay
                display_frame = frame.copy()
                cv2.putText(display_frame, f"Poses: {calibrator.num_poses}/10",
                            (10, display_frame.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow("Camera - Hand-Eye Calibration", display_frame)
                cv2.waitKey(1)  # Required to update window
            
            # Update UI
            marker_status.value = "DETECTED ✓" if marker_detected else "Not detected"
            poses_display.value = calibrator.num_poses
            
            # Handle capture request
            if capture_requested[0]:
                capture_requested[0] = False
                if marker_detected:
                    ee_pose = controller.get_ee_pose()
                    if ee_pose:
                        n = calibrator.add_pose_pair(ee_pose, rvec, tvec)
                        status_display.value = f"Captured pose {n}"
                        print(f"✓ Captured pose {n}")
                    else:
                        status_display.value = "Error: No EE pose"
                else:
                    status_display.value = "Error: No marker detected"
            
            # Handle calibration request
            if calibrate_requested[0]:
                calibrate_requested[0] = False
                if calibrator.num_poses < 10:
                    status_display.value = f"Need 10 poses (have {calibrator.num_poses})"
                else:
                    status_display.value = "Computing calibration..."
                    result = calibrator.calibrate()
                    if result:
                        R, t = result
                        calibrator.save(R, t, output_path)
                        status_display.value = f"Saved: {output_path}"
                        print(f"\n✓ Calibration saved to: {output_path}")
                    else:
                        status_display.value = "Calibration failed"
            
            # --- IK Solving ---
            target_position = np.array(ik_target.position)
            target_wxyz = np.array(ik_target.wxyz)
            
            try:
                solution = pks.solve_ik(
                    robot=robot,
                    target_link_name=target_link_name,
                    target_position=target_position,
                    target_wxyz=target_wxyz,
                    initial_joint_config=last_solution,
                )
                last_solution = solution
            except Exception as e:
                time.sleep(0.05)
                continue
            
            # Smoothing
            alpha = smoothing_alpha.value
            smoothed_solution = alpha * solution + (1 - alpha) * smoothed_solution
            
            # Update visualization
            urdf_vis.update_cfg(smoothed_solution)
            
            # Send to robot if enabled
            if enable_robot.value:
                arm_joints = list(smoothed_solution[:6].astype(float))
                controller.set_joint_positions(arm_joints)
            
            # Rate limiting (20 Hz)
            elapsed = time.time() - loop_start
            sleep_time = max(0, 0.05 - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        camera.stop()
        cv2.destroyAllWindows()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
