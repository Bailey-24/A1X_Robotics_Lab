#!/usr/bin/env python3
"""
Diagnostic Script for Robotic Arm Jitter (Hardware vs Software Isoler)

This script isolates the motion control stack from the grasp pipeline (IK, vision)
by commanding the arm to move between known, safe joint poses.
It records the commanded vs. actual joint positions and calculates the error.

If significant jitter or error is observed during these simple, known moves,
the issue is highly likely hardware-related (e.g., motor PID drift, overheating,
mechanical wear, or CAN bus issues after collision).
"""

import sys
import time
import math
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from a1x_control import initialize, JointController


def create_safe_poses():
    """Define a set of safe joint poses to bounce between."""
    observation = [0.0, 1.0, -0.93, 0.83, 0.0, 0.0]
    # Small variations from observation that should be safe
    pose_1 = [0.2, 0.8, -0.7, 0.6, 0.2, 0.0]
    pose_2 = [-0.2, 1.2, -1.0, 0.9, -0.2, 0.0]
    return [observation, pose_1, pose_2, observation]


def calculate_max_error(commanded, actual):
    """Calculate the maximum absolute error across all joints."""
    if len(commanded) != len(actual):
        return float('inf')
    return max(abs(c - a) for c, a in zip(commanded, actual))


def main():
    print("=" * 60)
    print("  A1X Jitter Diagnostic Tool")
    print("=" * 60)
    print("Initializing ROS system...")
    
    # We only need joint state, no EE pose or gripper needed for basic joint motion testing
    initialize(enable_ee_pose=False, enable_gripper=False)
    controller = JointController()

    print("Waiting for joint states...")
    if not controller.wait_for_joint_states(timeout=10.0):
        print("Error: Could not get joint states from robot.")
        return 1

    initial_joints = controller.get_joint_states()
    if not initial_joints:
        print("Error: Joint states empty.")
        return 1
        
    print(f"✓ Connected. Initial state keys: {list(initial_joints.keys())}")
    
    poses = create_safe_poses()
    joint_names = ['arm_joint1', 'arm_joint2', 'arm_joint3', 'arm_joint4', 'arm_joint5', 'arm_joint6']
    
    print("\nStarting sequence (moving between safe poses)...")
    
    # Enable debugging in the controller if we modify it to support logging,
    # but for now we'll do an overarching check.
    
    error_threshold = 0.05  # roughly 3 degrees
    max_observed_error = 0.0
    
    for i, target in enumerate(poses):
        print(f"\nMoving to Pose {i} -> {target}")
        
        # We record start time to check how long motion actually takes
        start_time = time.time()
        
        # Interpolate linear motion for testing baseline (60 steps @ 10Hz = 6 seconds)
        success = controller.move_to_position_smooth(target, steps=60, rate_hz=10.0)
        
        if not success:
            print(f"✗ Motion {i} failed to execute!")
            continue
            
        # Immediately capture state after motion command completes
        end_time = time.time()
        time.sleep(0.5)  # Let it settle slightly, as per current logic
        
        current_state = controller.get_joint_states()
        actual_pos = [current_state.get(name, 0.0) for name in joint_names]
        
        err = calculate_max_error(target, actual_pos)
        max_observed_error = max(max_observed_error, err)
        
        print(f"✓ Motion command finished in {end_time - start_time:.2f}s")
        print("  Post-motion check:")
        print(f"    Commanded: {[f'{x:.3f}' for x in target]}")
        print(f"    Actual:    {[f'{x:.3f}' for x in actual_pos]}")
        print(f"    Max Error: {err:.4f} rad")
        
        if err > error_threshold:
            print("    ⚠️  Significant position error detected! Arm may not have settled.")
            
        # Pause before next move
        time.sleep(1.5)
        
    print("\n" + "=" * 60)
    print("  DIAGNOSTIC SUMMARY")
    print("=" * 60)
    print(f"Overall Max Error: {max_observed_error:.4f} rad")
    
    if max_observed_error > error_threshold:
        print("\n❌ ISSUE DETECTED")
        print("The arm failed to reach commanded positions accurately.")
        print("If jitter was visually observed during these simple joint moves,")
        print("the issue is likely HARDWARE related (e.g. motor overheating, PID drift).")
    else:
        print("\n✅ POSITIONS REACHED")
        print("The arm reached commanded positions within tolerance.")
        print("If jitter is still observed during grasping, the issue is likely SOFTWARE")
        print("related to interpolation, control rate, or IK solving jumps.")
        
    print("\nNext steps:")
    print("1. If hardware issue: Full power-cycle, inspect mechanics, let motors cool")
    print("2. If software issue: Apply cosine interpolation and convergence checks")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
