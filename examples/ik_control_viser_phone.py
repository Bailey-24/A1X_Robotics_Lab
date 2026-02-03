#!/usr/bin/env python3
"""
IK Control with Viser Visualization

End-to-end inverse kinematics control for A1X robot arm using:
- PyRoki for IK solving (JAX-accelerated)
- Viser for interactive 3D visualization and control
- a1x_control for real robot communication via ROS2

Usage:
    python examples/ik_control_viser.py

Then open http://localhost:8080/ in your browser to control the robot.

SAFETY: Robot control is DISABLED by default. Enable it via the checkbox in the UI.
"""

import sys
import os
import time
import threading
import copy
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import viser
from viser.extras import ViserUrdf
import yourdfpy
from scipy.spatial.transform import Rotation as R

# PyRoki imports
sys.path.insert(0, str(Path(__file__).parent.parent / "pyroki" / "examples"))
import pyroki as pk
import pyroki_snippets as pks

# A1X control imports
import a1x_control

# Phone control imports
try:
    from asmagic import ARDataSubscriber
    PHONE_AVAILABLE = True
except ImportError:
    PHONE_AVAILABLE = False
    print("WARNING: asmagic not installed, phone control disabled")


# Default phone IP
DEFAULT_PHONE_IP = "192.168.31.159"


class PhoneListener:
    """Thread-safe listener for iPhone 6D pose data via AsMagic."""
    
    def __init__(self, ip: str):
        self.ip = ip
        self.latest_pose: Optional[np.ndarray] = None
        self.lock = threading.Lock()
        self.running = True
        self.connected = False
        self.error_msg = ""
        self._reset_origin_flag = False
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
    
    def _run(self) -> None:
        """Background thread to receive phone pose data."""
        print(f"Connecting to phone at {self.ip}...")
        try:
            self.sub = ARDataSubscriber(self.ip)
            self.connected = True
            print("Phone connected.")
            for data in self.sub:
                if not self.running:
                    break
                if hasattr(data, 'local_pose') and data.local_pose is not None:
                    # Expecting [x, y, z, qx, qy, qz, qw]
                    pose = np.array(data.local_pose)
                    if pose.shape[0] == 7:
                        with self.lock:
                            self.latest_pose = pose
        except Exception as e:
            print(f"Phone listener error: {e}")
            self.error_msg = str(e)
            self.connected = False
        finally:
            if hasattr(self, 'sub'):
                self.sub.close()
    
    def get_pose(self) -> Optional[np.ndarray]:
        """Get latest phone pose (thread-safe copy)."""
        with self.lock:
            return self.latest_pose.copy() if self.latest_pose is not None else None
    
    def request_origin_reset(self) -> None:
        """Signal that phone origin should be reset."""
        self._reset_origin_flag = True
    
    def check_and_clear_reset(self) -> bool:
        """Check if reset was requested and clear the flag."""
        if self._reset_origin_flag:
            self._reset_origin_flag = False
            return True
        return False
    
    def stop(self) -> None:
        """Stop the listener thread."""
        self.running = False


def load_a1x_urdf() -> yourdfpy.URDF:
    """Load the A1X URDF with proper mesh path resolution."""
    urdf_path = Path("/home/ubuntu/projects/A1Xsdk/install/mobiman/lib/mobiman/configs/urdfs/a1x.urdf")
    
    def resolve_package_uri(fname: str) -> str:
        """Custom handler to resolve package:// URIs for meshes."""
        package_prefix = "package://mobiman/"
        if fname.startswith(package_prefix):
            relative_path = fname[len(package_prefix):]
            return str(Path("/home/ubuntu/projects/A1Xsdk/install/mobiman/share/mobiman") / relative_path)
        return fname
    
    return yourdfpy.URDF.load(urdf_path, filename_handler=resolve_package_uri)


def main():
    """Main function for IK control with Viser."""
    print("=" * 60)
    print("A1X IK Control with Viser Visualization")
    print("=" * 60)
    print()
    print("Initializing systems...")
    print()
    
    # =========================================================================
    # 1. Load URDF and create PyRoki robot
    # =========================================================================
    print("[1/4] Loading A1X URDF...")
    urdf = load_a1x_urdf()
    robot = pk.Robot.from_urdf(urdf)
    target_link_name = "gripper_link"
    print(f"       Robot loaded with {robot.joints.num_actuated_joints} actuated joints")
    print(f"       Target link: {target_link_name}")
    
    # =========================================================================
    # 2. Initialize A1X control system
    # =========================================================================
    print("[2/4] Initializing A1X control system...")
    # System initializes automatically on import, just get the controller
    controller = a1x_control.JointController()
    
    # Wait for joint states to be available
    print("       Waiting for joint state data...")
    if controller.wait_for_joint_states(timeout=10.0):
        joints = controller.get_joint_states()
        if joints:
            print("       Current joint positions:")
            for name in ['arm_joint1', 'arm_joint2', 'arm_joint3', 
                        'arm_joint4', 'arm_joint5', 'arm_joint6']:
                if name in joints:
                    print(f"         {name}: {joints[name]:.4f} rad")
    else:
        print("       WARNING: No joint state data available yet")
    
    # =========================================================================
    # 3. Set up Viser visualization server
    # =========================================================================
    print("[3/4] Starting Viser server...")
    server = viser.ViserServer()
    
    # Add ground grid
    server.scene.add_grid("/ground", width=2, height=2, cell_size=0.1)
    
    # Add robot visualization
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/robot")
    
    # Get initial joint configuration from robot (if available)
    initial_joints = controller.get_joint_states()
    if initial_joints:
        # Extract arm joints in order, pad with zeros for gripper
        initial_cfg = []
        for name in ['arm_joint1', 'arm_joint2', 'arm_joint3', 
                     'arm_joint4', 'arm_joint5', 'arm_joint6']:
            initial_cfg.append(initial_joints.get(name, 0.0))
        # Add gripper joints (prismatic, default to 0)
        initial_cfg.extend([0.0, 0.0])
        urdf_vis.update_cfg(np.array(initial_cfg))
    
    # =========================================================================
    # 4. Create interactive UI controls
    # =========================================================================
    print("[4/4] Creating UI controls...")
    
    # --- Transform Controls (IK Target Gizmo) ---
    # Initial position in front of robot base
    ik_target = server.scene.add_transform_controls(
        "/ik_target",
        scale=0.15,
        position=(0.25, 0.0, 0.2),  # x=forward, y=left, z=up
        wxyz=(1, 0, 0, 0),  # Identity rotation (w, x, y, z)
        depth_test=False,  # Always visible
    )
    
    # Add a small sphere to visualize the target point
    server.scene.add_icosphere(
        "/ik_target/sphere",
        radius=0.015,
        color=(255, 100, 100),
    )
    
    # --- GUI Panel ---
    # Folder for control settings
    with server.gui.add_folder("Robot Control"):
        # SAFETY: Robot control disabled by default
        enable_robot = server.gui.add_checkbox(
            "Enable Robot Control",
            initial_value=False,
        )
        
        # Control rate slider
        rate_hz = server.gui.add_slider(
            "Control Rate (Hz)",
            min=5.0,
            max=50.0,
            step=5.0,
            initial_value=20.0,
        )

        # Smoothing factor
        smoothing_alpha = server.gui.add_slider(
            "Smoothing Alpha",
            min=0.01,
            max=1.0,
            step=0.01,
            initial_value=0.2,
        )
    
    # Folder for status displays
    with server.gui.add_folder("Status"):
        # IK solve time
        timing_display = server.gui.add_number(
            "IK Solve Time (ms)",
            initial_value=0.0,
            disabled=True,
        )
        
        # Target position display
        target_pos_display = server.gui.add_text(
            "Target Position",
            initial_value="x: 0.000, y: 0.000, z: 0.000",
        )
        
        # Current EE position (from IK solution)
        ee_pos_display = server.gui.add_text(
            "IK Solution EE",
            initial_value="x: 0.000, y: 0.000, z: 0.000",
        )
        
        # Robot control status
        status_display = server.gui.add_text(
            "Status",
            initial_value="Robot control DISABLED",
        )
    
    # Folder for joint values
    with server.gui.add_folder("Joint Values"):
        joint_displays = []
        for i in range(6):
            joint_displays.append(
                server.gui.add_number(
                    f"Joint {i+1} (rad)",
                    initial_value=0.0,
                    disabled=True,
                )
            )
    
    # Button to sync target with current robot position
    with server.gui.add_folder("Actions"):
        sync_button = server.gui.add_button("Sync Target to Current EE")
        
        @sync_button.on_click
        def on_sync_click(_) -> None:
            """Move IK target to current end-effector position."""
            # Get current joint states
            joints = controller.get_joint_states()
            if joints:
                # Build configuration for FK
                cfg = []
                for name in ['arm_joint1', 'arm_joint2', 'arm_joint3', 
                             'arm_joint4', 'arm_joint5', 'arm_joint6']:
                    cfg.append(joints.get(name, 0.0))
                cfg.extend([0.0, 0.0])  # gripper
                
                # Compute FK to get current EE position
                cfg_array = np.array(cfg)
                target_idx = robot.links.names.index(target_link_name)
                fk_result = robot.forward_kinematics(cfg_array)
                ee_pose = fk_result[target_idx]  # SE3 pose as 7-element array (wxyz, xyz)
                
                # Update IK target position
                ik_target.position = tuple(ee_pose[4:7])  # xyz
                ik_target.wxyz = tuple(ee_pose[0:4])  # wxyz
                
                print(f"Synced target to: pos={ee_pose[4:7]}, wxyz={ee_pose[0:4]}")
    
    # --- Phone Control UI ---
    phone_listener: Optional[PhoneListener] = None
    
    with server.gui.add_folder("Phone Control"):
        if not PHONE_AVAILABLE:
            server.gui.add_markdown("**asmagic not installed**")
        
        phone_ip_input = server.gui.add_text(
            "Phone IP",
            initial_value=DEFAULT_PHONE_IP,
        )
        
        enable_phone = server.gui.add_checkbox(
            "Enable Phone Control",
            initial_value=False,
            disabled=not PHONE_AVAILABLE,
        )
        
        reset_phone_btn = server.gui.add_button(
            "Reset Phone Origin",
            disabled=not PHONE_AVAILABLE,
        )
        
        phone_status_display = server.gui.add_text(
            "Phone Status",
            initial_value="Disabled",
        )
        
        @reset_phone_btn.on_click
        def on_reset_phone_click(_) -> None:
            nonlocal phone_base_pos, phone_base_rot_inv
            phone_base_pos = None
            phone_base_rot_inv = None
            print("Phone origin will be reset on next frame")
    
    print()
    print("=" * 60)
    print("READY! Open http://localhost:8080/ in your browser")
    print("=" * 60)
    print()
    print("Controls:")
    print("  - Drag the RED SPHERE to set IK target position")
    print("  - Use rotation handles for orientation")
    print("  - Check 'Enable Robot Control' to send commands to robot")
    print("  - Check 'Enable Phone Control' to use iPhone as input")
    print("  - Adjust 'Control Rate' for command frequency")
    print()
    print("Press Ctrl+C to exit")
    print()
    
    # =========================================================================
    # 5. Main control loop
    # =========================================================================
    last_solution = None
    
    # Initialize last_solution with current robot state if available
    initial_joints = controller.get_joint_states()
    if initial_joints:
        cfg = []
        for name in ['arm_joint1', 'arm_joint2', 'arm_joint3', 
                     'arm_joint4', 'arm_joint5', 'arm_joint6']:
            cfg.append(initial_joints.get(name, 0.0))
        # Add gripper joints to match robot DOF (8)
        cfg.extend([0.0, 0.0])
        last_solution = np.array(cfg)

    # Initialize smoothed_solution with correct shape (8 for A1X)
    smoothed_solution = last_solution.copy() if last_solution is not None else np.zeros(8)
    smoothed_timing = 0.0
    
    # Phone control state
    phone_base_pos: Optional[np.ndarray] = None
    phone_base_rot_inv: Optional[R] = None
    init_target_pos: Optional[np.ndarray] = None
    init_target_rot: Optional[R] = None
    phone_was_enabled = False
    
    try:
        while True:
            loop_start = time.time()
            
            # --- Handle phone control ---
            if PHONE_AVAILABLE and enable_phone.value:
                # Start phone listener if not running
                if phone_listener is None or not phone_listener.running:
                    phone_listener = PhoneListener(phone_ip_input.value)
                
                # Capture init pose when first enabled
                if not phone_was_enabled:
                    init_target_pos = np.array(ik_target.position)
                    q_init = np.array(ik_target.wxyz)  # wxyz
                    init_target_rot = R.from_quat([q_init[1], q_init[2], q_init[3], q_init[0]])  # xyzw
                    phone_base_pos = None  # Will be set on first phone frame
                    phone_base_rot_inv = None
                    phone_was_enabled = True
                    print(f"Phone control enabled, init pose: {init_target_pos}")
                
                # Update phone status
                if phone_listener.connected:
                    phone_status_display.value = "Connected"
                else:
                    phone_status_display.value = phone_listener.error_msg or "Connecting..."
                
                # Get phone pose and compute target
                phone_pose = phone_listener.get_pose()
                if phone_pose is not None and init_target_pos is not None:
                    p_curr = phone_pose[:3]
                    q_curr = phone_pose[3:]  # xyzw
                    
                    # Normalize quaternion
                    norm = np.linalg.norm(q_curr)
                    if norm > 0:
                        q_curr = q_curr / norm
                    
                    r_curr = R.from_quat(q_curr)
                    
                    # Set phone origin if not set
                    if phone_base_pos is None:
                        phone_base_pos = p_curr.copy()
                        phone_base_rot_inv = r_curr.inv()
                        print("Phone origin set.")
                    
                    # Compute deltas
                    delta_pos = p_curr - phone_base_pos
                    delta_rot = phone_base_rot_inv * r_curr
                    
                    # Apply to initial pose
                    target_position = init_target_pos + delta_pos
                    target_rot = init_target_rot * delta_rot
                    
                    # Convert to wxyz for Viser
                    q_tgt = target_rot.as_quat()  # xyzw
                    target_wxyz = np.array([q_tgt[3], q_tgt[0], q_tgt[1], q_tgt[2]])
                    
                    # Update gizmo to show computed target (visual indicator)
                    ik_target.position = tuple(target_position)
                    ik_target.wxyz = tuple(target_wxyz)
                else:
                    # No phone data yet, use current gizmo position
                    target_position = np.array(ik_target.position)
                    target_wxyz = np.array(ik_target.wxyz)
            else:
                # Manual mode: read from gizmo
                if phone_was_enabled:
                    phone_was_enabled = False
                    phone_status_display.value = "Disabled"
                    print("Phone control disabled, resuming gizmo control")
                
                target_position = np.array(ik_target.position)
                target_wxyz = np.array(ik_target.wxyz)
            
            # Update target position display
            target_pos_display.value = (
                f"x: {target_position[0]:.3f}, "
                f"y: {target_position[1]:.3f}, "
                f"z: {target_position[2]:.3f}"
            )
            
            # --- Solve IK ---
            ik_start = time.time()
            try:
                solution = pks.solve_ik(
                    robot=robot,
                    target_link_name=target_link_name,
                    target_position=target_position,
                    target_wxyz=target_wxyz,
                    initial_joint_config=last_solution,
                )
                last_solution = solution
                
                # Compute timing (smoothed)
                ik_time_ms = (time.time() - ik_start) * 1000
                smoothed_timing = 0.9 * smoothed_timing + 0.1 * ik_time_ms
                timing_display.value = round(smoothed_timing, 2)
                
            except Exception as e:
                print(f"IK solve error: {repr(e)}")
                status_display.value = f"IK Error: {str(e)[:30]}"
                time.sleep(0.1)
                continue
            
            # --- Smoothing ---
            alpha = smoothing_alpha.value
            smoothed_solution = alpha * solution + (1 - alpha) * smoothed_solution
            
            # --- Update visualization ---
            urdf_vis.update_cfg(smoothed_solution)
            
            # Update joint displays
            for i in range(6):
                joint_displays[i].value = round(float(smoothed_solution[i]), 4)
            
            # Compute FK to show actual EE position from solution
            target_idx = robot.links.names.index(target_link_name)
            fk_result = robot.forward_kinematics(smoothed_solution)
            ee_pose = fk_result[target_idx]
            ee_pos_display.value = (
                f"x: {ee_pose[4]:.3f}, "
                f"y: {ee_pose[5]:.3f}, "
                f"z: {ee_pose[6]:.3f}"
            )
            
            # --- Send to robot (if enabled) ---
            if enable_robot.value:
                # Extract arm joints only (first 6 of 8)
                arm_joints = list(smoothed_solution[:6].astype(float))
                
                # Send command
                success = controller.set_joint_positions(arm_joints)
                
                if success:
                    status_display.value = "Robot control ACTIVE"
                else:
                    status_display.value = "Command send FAILED"
            else:
                status_display.value = "Robot control DISABLED"
            
            # --- Rate limiting ---
            target_period = 1.0 / rate_hz.value
            elapsed = time.time() - loop_start
            sleep_time = max(0, target_period - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    except KeyboardInterrupt:
        print("\n\nShutting down...")
        print("Goodbye!")


if __name__ == "__main__":
    main()
