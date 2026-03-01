#!/usr/bin/env python3
"""
End-Effector Pose Reading Example

This example demonstrates how to read the end-effector (EE) pose
using the A1X control API. The API automatically launches a FK
(forward kinematics) calculator to compute EE pose from joint states.

No additional terminals required!
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import a1x_control
import time

def main():
    print("A1X End-Effector Pose Reading Example")
    print("=" * 45)
    
    # Initialize with EE pose support (launches FK calculator automatically)
    print("Initializing system with EE pose support...")
    a1x_control.initialize(enable_ee_pose=True)
    
    # Get controller instance
    controller = a1x_control.JointController()
    
    # Wait for system to stabilize
    time.sleep(2)
    
    print("\n1. Waiting for end-effector pose data...")
    if controller.wait_for_ee_pose(timeout=10):
        print("   ✓ End-effector pose data available")
    else:
        print("   ⚠ No end-effector pose data available")
        print("   Please ensure the relaxed_ik stack is running.")
        print("   Exiting...")
        return
    
    # Read full end-effector pose
    print("\n2. Reading full end-effector pose...")
    ee_pose = controller.get_ee_pose()
    if ee_pose:
        print(f"   Position:")
        print(f"     x: {ee_pose['position']['x']:.4f} m")
        print(f"     y: {ee_pose['position']['y']:.4f} m")
        print(f"     z: {ee_pose['position']['z']:.4f} m")
        print(f"   Orientation (quaternion):")
        print(f"     x: {ee_pose['orientation']['x']:.4f}")
        print(f"     y: {ee_pose['orientation']['y']:.4f}")
        print(f"     z: {ee_pose['orientation']['z']:.4f}")
        print(f"     w: {ee_pose['orientation']['w']:.4f}")
        print(f"   Frame ID: {ee_pose['frame_id']}")
    
    # Read just position
    print("\n3. Reading end-effector position only...")
    position = controller.get_ee_position()
    if position:
        print(f"   Position (x, y, z): ({position[0]:.4f}, {position[1]:.4f}, {position[2]:.4f}) m")
    
    # Read just orientation
    print("\n4. Reading end-effector orientation only...")
    orientation = controller.get_ee_orientation()
    if orientation:
        print(f"   Orientation (x, y, z, w): ({orientation[0]:.4f}, {orientation[1]:.4f}, {orientation[2]:.4f}, {orientation[3]:.4f})")
    
    # Continuous monitoring
    print("\n5. Monitoring end-effector pose for 5 seconds...")
    start_time = time.time()
    while time.time() - start_time < 5:
        position = controller.get_ee_position()
        if position:
            print(f"   [{time.strftime('%H:%M:%S')}] Position: x={position[0]:+.4f}, y={position[1]:+.4f}, z={position[2]:+.4f}")
        time.sleep(0.5)
    
    print("\nEnd-effector pose reading example complete!")
    print("\nKey features demonstrated:")
    print("- get_ee_pose() - Full pose with position and orientation")
    print("- get_ee_position() - Just (x, y, z) tuple")
    print("- get_ee_orientation() - Just (x, y, z, w) quaternion")
    print("- wait_for_ee_pose() - Wait for data availability")

if __name__ == "__main__":
    main()
