#!/usr/bin/env python3
"""
Gripper Control Example

This example demonstrates gripper control using the A1X control API.
Shows how to open, close, and set specific gripper positions.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import a1x_control
import time

def main():
    print("A1X Gripper Control Example")
    print("=" * 35)
    
    # Initialize system with gripper enabled (HDAS already running from joint control)
    print("Initializing system with gripper support...")
    a1x_control.initialize(launch_rviz=False, enable_gripper=True)
    
    # Get controller instance
    controller = a1x_control.JointController()
    
    # Wait for system to stabilize
    time.sleep(2)
    
    print("\n1. Waiting for gripper state data...")
    if controller.wait_for_gripper_state(timeout=10):
        print("   ✓ Gripper state data available")
    else:
        print("   ⚠ No gripper state data available, but continuing...")
    
    # Read initial gripper state
    print("\n2. Reading initial gripper state...")
    gripper_pos = controller.get_gripper_state()
    if gripper_pos is not None:
        print(f"   Current gripper position: {gripper_pos:.1f} (0=closed, 100=open)")
    else:
        print("   No gripper position data available")
    
    # Demonstrate gripper control
    print("\n3. Demonstrating gripper control...")
    
    # Close gripper
    print("   Closing gripper...")
    success = controller.close_gripper()
    if success:
        print("   ✓ Close command sent")
        time.sleep(2)
        
        # Check position
        pos = controller.get_gripper_state()
        if pos is not None:
            print(f"   Current position: {pos:.1f}")
    else:
        print("   ✗ Failed to send close command")
    
    # Open gripper
    print("   Opening gripper...")
    success = controller.open_gripper()
    if success:
        print("   ✓ Open command sent")
        time.sleep(2)
        
        # Check position
        pos = controller.get_gripper_state()
        if pos is not None:
            print(f"   Current position: {pos:.1f}")
    else:
        print("   ✗ Failed to send open command")
    
    # Set specific positions
    print("\n4. Setting specific gripper positions...")
    # positions = [25.0, 50.0, 75.0, 50.0, 0.0]
    # positions = [-25.0]

    for target_pos in positions:
        print(f"   Setting gripper to {target_pos:.1f}...")
        success = controller.set_gripper_position(target_pos)
        if success:
            print(f"   ✓ Command sent: {target_pos:.1f}")
            time.sleep(1.5)
            
            # Read actual position
            actual_pos = controller.get_gripper_state()
            if actual_pos is not None:
                print(f"   Actual position: {actual_pos:.1f}")
        else:
            print(f"   ✗ Failed to set position {target_pos:.1f}")
    
    # Demonstrate combined arm and gripper control
    print("\n5. Demonstrating combined arm and gripper control...")
    
    # Move arm to a position and close gripper
    print("   Moving arm and closing gripper...")
    arm_positions = [0.1, 0.0043, -0.1, -0.0347, -0.0055, 0.0013]
    
    # Move arm
    arm_success = controller.set_joint_positions(arm_positions)
    # Close gripper
    gripper_success = controller.close_gripper()
    
    if arm_success and gripper_success:
        print("   ✓ Combined command sent successfully")
        time.sleep(2)
        
        # Check both states
        joints = controller.get_joint_states()
        gripper_pos = controller.get_gripper_state()
        
        if joints and gripper_pos is not None:
            print(f"   Arm joint 1: {joints.get('arm_joint1', 'N/A'):.4f} rad")
            print(f"   Gripper position: {gripper_pos:.1f}")
    else:
        print("   ✗ Combined command failed")
    
    print("\nGripper control example complete!")
    print("\nKey features demonstrated:")
    print("- Gripper state reading from /hdas/feedback_gripper")
    print("- Gripper position commands to /motion_target/target_position_gripper") 
    print("- Convenient open_gripper() and close_gripper() methods")
    print("- Combined arm and gripper control")
    print("- Automatic gripper controller launch")

if __name__ == "__main__":
    main()
