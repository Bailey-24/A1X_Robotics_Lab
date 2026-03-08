#!/usr/bin/env python3
"""
A1X Joint Control API

Simple Python interface for controlling A1X robotic arm joints.
Automatically handles ROS system initialization and provides clean method calls.

Usage:
    import a1x_control
    controller = a1x_control.JointController()
    
    # Read current joint states
    joints = controller.get_joint_states()
    
    # Set joint positions
    controller.set_joint_positions([0.0, 0.0043, -0.1, -0.0347, -0.0055, 0.0013])
"""

import os
import sys
import time
import math
import subprocess
import threading
import signal
import atexit
from typing import List, Optional, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('a1x_control')

# Initialize ROS modules at module level
def _setup_ros_environment():
    """Setup ROS environment and import modules"""
    try:
        install_path = "/home/ubuntu/projects/A1Xsdk/install"
        
        # Add ROS paths to environment
        if 'AMENT_PREFIX_PATH' not in os.environ:
            os.environ['AMENT_PREFIX_PATH'] = install_path
        else:
            os.environ['AMENT_PREFIX_PATH'] = f"{install_path}:{os.environ['AMENT_PREFIX_PATH']}"
            
        if 'PYTHONPATH' not in os.environ:
            os.environ['PYTHONPATH'] = f"{install_path}/lib/python3.10/site-packages"
        else:
            os.environ['PYTHONPATH'] = f"{install_path}/lib/python3.10/site-packages:{os.environ['PYTHONPATH']}"
            
        # Add to sys.path for immediate Python import access
        sys.path.insert(0, f"{install_path}/lib/python3.10/site-packages")
        
        logger.info("ROS environment setup complete")
        return True
    except Exception as e:
        logger.error(f"Failed to setup ROS environment: {e}")
        return False

# Setup environment and import ROS modules immediately
_setup_ros_environment()

# Import ROS modules
try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    from geometry_msgs.msg import PoseStamped
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
    logger.info("ROS modules imported successfully")
except ImportError as e:
    logger.error(f"Failed to import ROS modules: {e}")
    # Try to add ROS paths for conda environment
    try:
        import sys
        ros_paths = [
            "/home/ubuntu/projects/A1Xsdk/install/lib/python3.10/site-packages",
            "/opt/ros/humble/local/lib/python3.10/dist-packages"
        ]
        for path in ros_paths:
            if path not in sys.path:
                sys.path.insert(0, path)
        
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import JointState
        from geometry_msgs.msg import PoseStamped
        from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
        logger.info("ROS modules imported successfully after path adjustment")
    except ImportError as e2:
        logger.error(f"Failed to import ROS modules even after path adjustment: {e2}")
        raise RuntimeError(f"ROS modules not available. Make sure you're in the A1X SDK directory and ROS is properly installed: {e2}")

class A1XSystemManager:
    """Manages the A1X ROS system lifecycle"""
    
    def __init__(self, launch_rviz: bool = False, enable_gripper: bool = False, enable_ee_pose: bool = False):
        self.driver_process = None
        self.mobiman_process = None
        self.gripper_process = None
        self.ee_pose_process = None
        self.ros_initialized = False
        self.setup_complete = False
        self.launch_rviz = launch_rviz
        self.enable_gripper = enable_gripper
        self.enable_ee_pose = enable_ee_pose
        
    def validate_environment(self) -> bool:
        """Validate that ROS environment is properly set up"""
        try:
            # Check if ROS workspace is sourced
            install_path = "/home/ubuntu/projects/A1Xsdk/install"
            if not os.path.exists(install_path):
                logger.error(f"A1X install directory not found: {install_path}")
                return False
                
            # Source ROS environment
            setup_script = os.path.join(install_path, "setup.bash")
            if not os.path.exists(setup_script):
                logger.error(f"ROS setup script not found: {setup_script}")
                return False
                
            logger.info("ROS environment validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Environment validation failed: {e}")
            return False
    
    
    def launch_driver(self) -> bool:
        """Launch the HDAS driver"""
        try:
            logger.info("Launching HDAS driver...")
            
            # Prepare environment for subprocess
            env = os.environ.copy()
            env['ROS_DOMAIN_ID'] = '0'
            
            # Launch HDAS driver
            cmd = [
                'bash', '-c', 
                f'source /home/ubuntu/projects/A1Xsdk/install/setup.bash && ros2 launch HDAS a1xy.py'
            ]
            
            self.driver_process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid  # Create new process group
            )
            
            # Give driver time to start
            time.sleep(3)
            
            if self.driver_process.poll() is None:
                logger.info("HDAS driver launched successfully")
                return True
            else:
                logger.error("HDAS driver failed to start")
                return False
                
        except Exception as e:
            logger.error(f"Failed to launch HDAS driver: {e}")
            return False
    
    def launch_mobiman(self) -> bool:
        """Launch the mobiman manipulation stack"""
        try:
            logger.info("Launching mobiman stack...")
            
            # Prepare environment for subprocess
            env = os.environ.copy()
            env['ROS_DOMAIN_ID'] = '0'
            
            # Use standard joint tracker demo
            # Note: For end-effector pose reading, user should manually launch:
            #   ros2 launch mobiman A1x_arm_relaxed_ik_launch.py
            launch_file = "A1x_jointTrackerdemo_launch.py"
            rviz_arg = "" if self.launch_rviz else "launch_rviz:=false"
            cmd = [
                'bash', '-c',
                f'source /home/ubuntu/projects/A1Xsdk/install/setup.bash && ros2 launch mobiman {launch_file} {rviz_arg}'
            ]
            
            self.mobiman_process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid  # Create new process group
            )
            
            # Give mobiman time to start
            time.sleep(5)
            
            if self.mobiman_process.poll() is None:
                logger.info("Mobiman stack launched successfully")
                return True
            else:
                logger.error("Mobiman stack failed to start")
                return False
                
        except Exception as e:
            logger.error(f"Failed to launch mobiman stack: {e}")
            return False
    
    def launch_gripper_controller(self, use_smooth_config: bool = False) -> bool:
        """Launch the gripper controller"""
        try:
            logger.info("Launching gripper controller...")
            
            # Prepare environment for subprocess
            env = os.environ.copy()
            env['ROS_DOMAIN_ID'] = '0'
            
            # Choose configuration file based on smooth_config parameter
            config_file = "ARM_gripper_controller_kp_kd_smooth.toml" if use_smooth_config else "ARM_gripper_controller_kp_kd.toml"
            
            # Launch gripper controller with specific config
            cmd = [
                'bash', '-c',
                f'source /home/ubuntu/projects/A1Xsdk/install/setup.bash && ros2 launch mobiman A1xy_gripperController_launch.py config_file:=/home/ubuntu/projects/A1Xsdk/install/mobiman/share/mobiman/config/{config_file}'
            ]
            
            self.gripper_process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid  # Create new process group
            )
            
            # Give gripper controller time to start
            time.sleep(3)
            
            if self.gripper_process.poll() is None:
                logger.info("Gripper controller launched successfully")
                return True
            else:
                logger.error("Gripper controller failed to start")
                return False
                
        except Exception as e:
            logger.error(f"Failed to launch gripper controller: {e}")
            return False
    
    def launch_ee_pose_publisher(self) -> bool:
        """Launch the end-effector pose publisher (FK calculator)"""
        try:
            logger.info("Launching EE pose publisher...")
            
            # Prepare environment for subprocess
            env = os.environ.copy()
            env['ROS_DOMAIN_ID'] = '0'
            
            # Launch EE pose publisher node with correct URDF path
            launch_file = "/home/ubuntu/projects/A1Xsdk/install/mobiman/share/mobiman/launch/simpleExample/A1X/A1x_eePose_launch.py"
            urdf_file = "/home/ubuntu/projects/A1Xsdk/install/mobiman/lib/mobiman/configs/urdfs/a1x.urdf"
            cmd = [
                'bash', '-c',
                f'source /home/ubuntu/projects/A1Xsdk/install/setup.bash && ros2 launch {launch_file} urdf_file:={urdf_file}'
            ]
            
            self.ee_pose_process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid  # Create new process group
            )
            
            # Give EE pose publisher time to start
            time.sleep(2)
            
            if self.ee_pose_process.poll() is None:
                logger.info("EE pose publisher launched successfully")
                return True
            else:
                logger.error("EE pose publisher failed to start")
                return False
                
        except Exception as e:
            logger.error(f"Failed to launch EE pose publisher: {e}")
            return False
    
    def initialize_ros(self) -> bool:
        """Initialize ROS"""
        try:
            if not self.ros_initialized:
                # Check if ROS context is already initialized
                if not rclpy.ok():
                    rclpy.init()
                self.ros_initialized = True
                logger.info("ROS initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize ROS: {e}")
            return False
    
    def setup_system(self) -> bool:
        """Complete system setup"""
        try:
            logger.info("Starting A1X system initialization...")
            
            # Step 1: Validate environment
            if not self.validate_environment():
                return False
            
            # Step 2: Initialize ROS
            if not self.initialize_ros():
                return False
            
            # Step 3: Launch driver
            if not self.launch_driver():
                return False
            
            # Step 4: Launch mobiman
            if not self.launch_mobiman():
                return False
            
            # Step 5: Launch gripper controller if enabled
            if self.enable_gripper:
                if not self.launch_gripper_controller():
                    return False
            
            # Step 6: Launch EE pose publisher if enabled
            if self.enable_ee_pose:
                if not self.launch_ee_pose_publisher():
                    return False
            
            # Step 7: Wait for system to be ready
            logger.info("Waiting for system to be ready...")
            time.sleep(3)
            
            self.setup_complete = True
            logger.info("A1X system initialization complete!")
            return True
            
        except Exception as e:
            logger.error(f"System setup failed: {e}")
            return False
    
    def shutdown(self):
        """Shutdown the system"""
        try:
            logger.info("Shutting down A1X system...")
            
            # Shutdown ROS
            if self.ros_initialized:
                try:
                    rclpy.shutdown()
                    self.ros_initialized = False
                except:
                    pass
            
            # Terminate processes
            for process_name, process in [("driver", self.driver_process), ("mobiman", self.mobiman_process), ("gripper", self.gripper_process), ("ee_pose", self.ee_pose_process)]:
                if process and process.poll() is None:
                    try:
                        # Terminate the process group
                        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                        process.wait(timeout=5)
                        logger.info(f"{process_name} process terminated")
                    except subprocess.TimeoutExpired:
                        # Force kill if it doesn't terminate gracefully
                        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                        logger.warning(f"{process_name} process force killed")
                    except Exception as e:
                        logger.error(f"Error terminating {process_name} process: {e}")
            
            logger.info("A1X system shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")


class JointController(Node):
    """Main joint control interface"""
    
    def __init__(self, node_name: str = 'a1x_joint_controller'):
        super().__init__(node_name)
        
        self.current_joint_state = None
        self.current_gripper_state = None
        self.current_ee_pose = None
        self.joint_names = [
            'arm_joint1', 'arm_joint2', 'arm_joint3', 
            'arm_joint4', 'arm_joint5', 'arm_joint6'
        ]
        
        # Setup QoS profiles
        self.qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )
        
        # Create subscribers and publishers for joints
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            self.qos_profile
        )
        
        self.joint_command_pub = self.create_publisher(
            JointState,
            '/motion_target/target_joint_state_arm',
            self.qos_profile
        )
        
        # Create subscribers and publishers for gripper
        self.gripper_state_sub = self.create_subscription(
            JointState,
            '/hdas/feedback_gripper',
            self.gripper_state_callback,
            self.qos_profile
        )
        
        self.gripper_command_pub = self.create_publisher(
            JointState,
            '/motion_target/target_position_gripper',
            self.qos_profile
        )
        
        # Create subscribers for end-effector pose (from different sources)
        # /motion_control/pose_ee_arm - from FK calculator (A1x_eePose_launch.py)
        self.ee_pose_sub1 = self.create_subscription(
            PoseStamped,
            '/motion_control/pose_ee_arm',
            self.ee_pose_callback,
            self.qos_profile
        )
        # /hdas/pose_ee_arm - from relaxed_ik (for backward compatibility)
        self.ee_pose_sub2 = self.create_subscription(
            PoseStamped,
            '/hdas/pose_ee_arm',
            self.ee_pose_callback,
            self.qos_profile
        )
        
        logger.info("JointController initialized")
    
    def joint_state_callback(self, msg: JointState):
        """Callback for joint state updates"""
        self.current_joint_state = msg
    
    def gripper_state_callback(self, msg: JointState):
        """Callback for gripper state updates"""
        self.current_gripper_state = msg
    
    def ee_pose_callback(self, msg: PoseStamped):
        """Callback for end-effector pose updates"""
        self.current_ee_pose = msg
    
    def get_ee_pose(self) -> Optional[Dict[str, Any]]:
        """
        Get current end-effector pose
        
        Returns:
            Dictionary with 'position' (x, y, z) and 'orientation' (x, y, z, w quaternion),
            or None if not available
        """
        if self.current_ee_pose is None:
            logger.warning("No end-effector pose data available yet")
            return None
        
        pose = self.current_ee_pose.pose
        return {
            'position': {
                'x': pose.position.x,
                'y': pose.position.y,
                'z': pose.position.z
            },
            'orientation': {
                'x': pose.orientation.x,
                'y': pose.orientation.y,
                'z': pose.orientation.z,
                'w': pose.orientation.w
            },
            'frame_id': self.current_ee_pose.header.frame_id,
            'timestamp': self.current_ee_pose.header.stamp
        }
    
    def get_ee_position(self) -> Optional[tuple]:
        """
        Get current end-effector position as a tuple (x, y, z)
        
        Returns:
            Tuple (x, y, z) in meters, or None if not available
        """
        if self.current_ee_pose is None:
            logger.warning("No end-effector pose data available yet")
            return None
        
        pose = self.current_ee_pose.pose
        return (pose.position.x, pose.position.y, pose.position.z)
    
    def get_ee_orientation(self) -> Optional[tuple]:
        """
        Get current end-effector orientation as a quaternion (x, y, z, w)
        
        Returns:
            Tuple (x, y, z, w) quaternion, or None if not available
        """
        if self.current_ee_pose is None:
            logger.warning("No end-effector pose data available yet")
            return None
        
        pose = self.current_ee_pose.pose
        return (pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w)
    
    def wait_for_ee_pose(self, timeout: float = 10.0) -> bool:
        """
        Wait for end-effector pose data to become available
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if end-effector pose is available
        """
        start_time = time.time()
        while self.current_ee_pose is None and (time.time() - start_time) < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
        
        return self.current_ee_pose is not None
    
    def get_joint_states(self) -> Optional[Dict[str, float]]:
        """
        Get current joint positions
        
        Returns:
            Dictionary mapping joint names to positions, or None if not available
        """
        if self.current_joint_state is None:
            logger.warning("No joint state data available yet")
            return None
        
        joint_dict = {}
        for i, name in enumerate(self.current_joint_state.name):
            if i < len(self.current_joint_state.position):
                joint_dict[name] = self.current_joint_state.position[i]
        
        return joint_dict
    
    def set_joint_positions(self, positions: List[float]) -> bool:
        """
        Set target joint positions
        
        Args:
            positions: List of 6 joint positions in radians
            
        Returns:
            True if command was sent successfully
        """
        try:
            if len(positions) != 6:
                logger.error(f"Expected 6 joint positions, got {len(positions)}")
                return False
            
            # Create joint state message
            joint_msg = JointState()
            joint_msg.header.stamp = self.get_clock().now().to_msg()
            joint_msg.name = self.joint_names
            joint_msg.position = list(positions)
            
            # Publish command
            self.joint_command_pub.publish(joint_msg)
            logger.info(f"Joint command sent: {positions}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to set joint positions: {e}")
            return False
    
    def get_gripper_state(self) -> Optional[float]:
        """
        Get current gripper position
        
        Returns:
            Gripper position (0=closed, 100=open), or None if not available
        """
        if self.current_gripper_state is None:
            logger.warning("No gripper state data available yet")
            return None
        
        if len(self.current_gripper_state.position) > 0:
            return self.current_gripper_state.position[0]
        
        return None
    
    def set_gripper_position(self, position: float) -> bool:
        """
        Set gripper position
        
        Args:
            position: Gripper position (0=closed, 100=open)
            
        Returns:
            True if command was sent successfully
        """
        try:
            if not (0 <= position <= 100):
                logger.error(f"Gripper position must be between 0 and 100, got {position}")
                return False
            
            # Create joint state message for gripper
            gripper_msg = JointState()
            gripper_msg.header.stamp = self.get_clock().now().to_msg()
            gripper_msg.position = [position]
            
            # Publish command
            self.gripper_command_pub.publish(gripper_msg)
            logger.info(f"Gripper command sent: {position}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to set gripper position: {e}")
            return False
    
    def open_gripper(self) -> bool:
        """
        Open the gripper fully
        
        Returns:
            True if command was sent successfully
        """
        return self.set_gripper_position(100.0)
    
    def close_gripper(self) -> bool:
        """
        Close the gripper fully
        
        Returns:
            True if command was sent successfully
        """
        return self.set_gripper_position(0.0)
    
    def wait_for_gripper_state(self, timeout: float = 10.0) -> bool:
        """
        Wait for gripper state data to become available
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if gripper state is available
        """
        start_time = time.time()
        while self.current_gripper_state is None and (time.time() - start_time) < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
        
        return self.current_gripper_state is not None
    
    def set_gripper_position_smooth(self, position: float, delay: float = 2.0) -> bool:
        """
        Set gripper position with smooth movement to reduce vibration
        
        Args:
            position: Gripper position (0=closed, 100=open)
            delay: Delay after command to allow smooth movement (seconds)
            
        Returns:
            True if command was sent successfully
        """
        success = self.set_gripper_position(position)
        if success:
            time.sleep(delay)  # Allow time for smooth movement
        return success
    
    def move_gripper_gradually(self, target_position: float, steps: int = 5, step_delay: float = 1.0) -> bool:
        """
        Move gripper gradually in small steps to minimize vibration
        
        Args:
            target_position: Target gripper position (0=closed, 100=open)
            steps: Number of intermediate steps
            step_delay: Delay between each step (seconds)
            
        Returns:
            True if movement completed successfully
        """
        try:
            current_pos = self.get_gripper_state()
            if current_pos is None:
                logger.error("Cannot get current gripper position")
                return False
            
            if not (0 <= target_position <= 100):
                logger.error(f"Target position must be between 0 and 100, got {target_position}")
                return False
            
            # Calculate step size
            step_size = (target_position - current_pos) / steps
            
            logger.info(f"Moving gripper gradually from {current_pos:.1f} to {target_position:.1f} in {steps} steps")
            
            for i in range(1, steps + 1):
                intermediate_pos = current_pos + (step_size * i)
                intermediate_pos = max(0, min(100, intermediate_pos))  # Clamp to valid range
                
                success = self.set_gripper_position(intermediate_pos)
                if not success:
                    logger.error(f"Failed to set intermediate position {intermediate_pos:.1f}")
                    return False
                
                time.sleep(step_delay)
                
                # Log progress
                actual_pos = self.get_gripper_state()
                if actual_pos is not None:
                    logger.info(f"Step {i}/{steps}: Target {intermediate_pos:.1f}, Actual {actual_pos:.1f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to move gripper gradually: {e}")
            return False
    
    def wait_for_joint_states(self, timeout: float = 10.0) -> bool:
        """
        Wait for joint state data to become available
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if joint states are available
        """
        start_time = time.time()
        while self.current_joint_state is None and (time.time() - start_time) < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
        
        return self.current_joint_state is not None
    
    def interpolate_trajectory(self, start_pos: List[float], end_pos: List[float], steps: int, interpolation_type: str = 'linear') -> List[List[float]]:
        """
        Generate interpolated trajectory between two positions
        
        Args:
            start_pos: Starting joint positions
            end_pos: Ending joint positions
            steps: Number of interpolation steps
            interpolation_type: 'linear' or 'cosine'
            
        Returns:
            List of interpolated joint positions
        """
        if len(start_pos) != len(end_pos):
            raise ValueError("Start and end positions must have same length")
        
        trajectory = []
        for i in range(steps + 1):
            t_raw = i / steps  # Interpolation parameter from 0 to 1
            if interpolation_type == 'cosine':
                t = 0.5 * (1 - math.cos(math.pi * t_raw))
            else:
                t = t_raw
                
            interpolated = []
            for j in range(len(start_pos)):
                # Linear interpolation using the blending parameter
                pos = start_pos[j] + t * (end_pos[j] - start_pos[j])
                interpolated.append(pos)
            trajectory.append(interpolated)
        
        return trajectory
    
    def execute_trajectory(self, trajectory: List[List[float]], rate_hz: float = 10.0, debug_log: bool = False) -> bool:
        """
        Execute a trajectory at specified rate without sleep
        
        Args:
            trajectory: List of joint position waypoints
            rate_hz: Control rate in Hz
            debug_log: If True, log commanded vs actual positions
            
        Returns:
            True if trajectory executed successfully
        """
        try:
            # Create a timer for trajectory execution
            period = 1.0 / rate_hz
            self._trajectory = trajectory
            self._trajectory_index = 0
            self._trajectory_complete = False
            
            def trajectory_callback():
                if self._trajectory_index < len(self._trajectory):
                    positions = self._trajectory[self._trajectory_index]
                    self.set_joint_positions(positions)
                    if debug_log:
                        actual = self.get_joint_states()
                        if actual:
                            actual_list = [actual.get(name, 0.0) for name in self.joint_names]
                            logger.info(f"Step {self._trajectory_index} | Cmd: {[f'{x:.3f}' for x in positions]} | Act: {[f'{x:.3f}' for x in actual_list]}")
                    self._trajectory_index += 1
                else:
                    self._trajectory_complete = True
                    self._trajectory_timer.cancel()
            
            self._trajectory_timer = self.create_timer(period, trajectory_callback)
            
            # Wait for trajectory completion
            while not self._trajectory_complete:
                rclpy.spin_once(self, timeout_sec=0.01)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to execute trajectory: {e}")
            return False
    
    def move_to_position_smooth(self, target_positions: List[float], steps: int = 20, rate_hz: float = 10.0, 
                                interpolation_type: str = 'linear', wait_for_convergence: bool = False,
                                convergence_tolerance: float = 0.015, convergence_timeout: float = 2.0,
                                debug_log: bool = False) -> bool:
        """
        Move smoothly to target position with interpolation
        
        Args:
            target_positions: Target joint positions
            steps: Number of interpolation steps
            rate_hz: Control rate in Hz
            interpolation_type: 'linear' or 'cosine'
            wait_for_convergence: Whether to wait for arm to settle after motion
            convergence_tolerance: Max error in rads to consider settled
            convergence_timeout: Timeout in seconds for convergence waiting
            debug_log: Whether to log step-by-step positions
            
        Returns:
            True if movement completed successfully
        """
        try:
            # Get current position
            current_joints = self.get_joint_states()
            if not current_joints:
                logger.error("No current joint state available")
                return False
            
            # Extract current positions in correct order
            current_positions = []
            for joint_name in self.joint_names:
                if joint_name in current_joints:
                    current_positions.append(current_joints[joint_name])
                else:
                    logger.error(f"Joint {joint_name} not found in current state")
                    return False
            
            # Generate trajectory
            trajectory = self.interpolate_trajectory(current_positions, target_positions, steps, interpolation_type=interpolation_type)
            
            # Execute trajectory
            success = self.execute_trajectory(trajectory, rate_hz, debug_log=debug_log)
            
            if not success:
                return False
                
            if wait_for_convergence:
                start_time = time.time()
                max_err = float('inf')
                while time.time() - start_time < convergence_timeout:
                    curr_state = self.get_joint_states()
                    if curr_state:
                        curr_pos = [curr_state.get(name, 0.0) for name in self.joint_names]
                        max_err = max(abs(c - t) for c, t in zip(curr_pos, target_positions))
                        if max_err <= convergence_tolerance:
                            return True
                    time.sleep(0.05)
                logger.warning(f"Motion convergence timeout after {convergence_timeout}s. Final max error: {max_err:.4f} rad")
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to move to position smoothly: {e}")
            return False


# Global system manager instance
_system_manager = None
_controller = None
_spin_thread = None

def _initialize_system(launch_rviz: bool = False, enable_gripper: bool = False, enable_ee_pose: bool = False, force_reinit: bool = False):
    """Initialize the A1X system on first import"""
    global _system_manager, _controller, _spin_thread
    
    if _system_manager is not None and not force_reinit:
        return  # Already initialized
    
    try:
        # If force_reinit, shutdown existing system first
        if force_reinit and _system_manager is not None:
            _system_manager.shutdown()
            _system_manager = None
            _controller = None
            if _spin_thread and _spin_thread.is_alive():
                _spin_thread.join(timeout=2)
            _spin_thread = None
        
        # Create and setup system manager
        _system_manager = A1XSystemManager(launch_rviz=launch_rviz, enable_gripper=enable_gripper, enable_ee_pose=enable_ee_pose)
        
        if not _system_manager.setup_system():
            raise RuntimeError("Failed to initialize A1X system")
        
        # Create joint controller using the stored class reference
        _controller = _JointControllerClass()
        
        # Start ROS spinning in background thread
        def spin_ros():
            try:
                rclpy.spin(_controller)
            except Exception as e:
                logger.error(f"ROS spin error: {e}")
        
        _spin_thread = threading.Thread(target=spin_ros, daemon=True)
        _spin_thread.start()
        
        # Wait for joint states to be available
        logger.info("Waiting for joint state data...")
        if not _controller.wait_for_joint_states():
            logger.warning("Joint state data not available, but continuing...")
        
        # Register cleanup function
        atexit.register(_cleanup_system)
        
        logger.info("A1X control system ready!")
        
    except Exception as e:
        logger.error(f"Failed to initialize A1X system: {e}")
        raise

def _cleanup_system():
    """Cleanup system on exit"""
    global _system_manager, _controller, _spin_thread
    
    if _system_manager:
        _system_manager.shutdown()

def get_joint_controller():
    """
    Get the joint controller instance
    
    Returns:
        JointController instance
    """
    global _controller
    
    if _controller is None:
        raise RuntimeError("A1X system not initialized. This should not happen.")
    
    return _controller

# Store reference to the class before overwriting
_JointControllerClass = JointController

# Create the function interface
def JointController():
    """
    Get the joint controller instance
    
    Returns:
        JointController instance
    """
    return get_joint_controller()

def initialize(launch_rviz: bool = False, enable_gripper: bool = False, enable_ee_pose: bool = False):
    """
    Initialize the A1X control system with optional parameters
    
    Args:
        launch_rviz: Whether to launch RViz visualization (default: False)
        enable_gripper: Whether to enable gripper control (default: False)
        enable_ee_pose: Whether to enable end-effector pose reading (default: False)
    """
    global _system_manager
    
    # Check if we need to add features to existing system
    if _system_manager is not None:
        needs_reinit = False
        
        # Add gripper support if needed
        if enable_gripper and not _system_manager.enable_gripper:
            logger.info("Adding gripper support to existing system...")
            _system_manager.enable_gripper = True
            if not _system_manager.launch_gripper_controller():
                logger.error("Failed to launch gripper controller")
            else:
                logger.info("Gripper controller added successfully")
        
        # Add EE pose support if needed
        if enable_ee_pose and not _system_manager.enable_ee_pose:
            logger.info("Adding EE pose support to existing system...")
            _system_manager.enable_ee_pose = True
            if not _system_manager.launch_ee_pose_publisher():
                logger.error("Failed to launch EE pose publisher")
            else:
                logger.info("EE pose publisher added successfully")
        
        if not needs_reinit:
            return
    
    # Normal initialization
    _initialize_system(launch_rviz=launch_rviz, enable_gripper=enable_gripper, enable_ee_pose=enable_ee_pose, force_reinit=True)

# Initialize system on import (without RViz by default)
_initialize_system()
