#!/usr/bin/env python3
"""
Basic A1X Joint Control Example

This example demonstrates the basic usage of the A1X control API.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import a1x_control
import time

def main():
    print("A1X Basic Joint Control Example")
    print("=" * 40)
    
    # Get controller instance (system automatically starts up)
    controller = a1x_control.JointController()
    
    # Wait a moment for system to stabilize
    time.sleep(2)
    
    # Read current joint states
    print("\n1. Reading current joint states...")
    joints = controller.get_joint_states()
    if joints:
        print("Current joint positions:")
        for name, position in joints.items():
            print(f"  {name}: {position:.4f} rad")
    else:
        print("  No joint state data available yet")
    
    # Set some target joint positions
    print("\n2. Setting target joint positions...")
    target_positions = [0.5, 0.3, -0.46, -0.0347, -0.0055, 0.0013]
    
    success = controller.set_joint_positions(target_positions)
    if success:
        print(f"  Command sent: {target_positions}")
    else:
        print("  Failed to send joint command")
    
    # Wait and read joint states again
    print("\n3. Waiting 3 seconds and reading joint states again...")
    time.sleep(3)
    
    joints = controller.get_joint_states()
    if joints:
        print("Updated joint positions:")
        for name, position in joints.items():
            print(f"  {name}: {position:.4f} rad")
    
    print("\nExample complete!")

if __name__ == "__main__":
    main()
