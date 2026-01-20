#!/usr/bin/env python3
"""
Continuous Joint Control Example

This example demonstrates continuous control without sleep delays,
moving smoothly from one position to another with interpolation.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import a1x_control
import time

def main():
    print("A1X Continuous Joint Control Example")
    print("=" * 40)
    
    # System is already initialized on import, no need to call initialize again
    
    # Get controller instance
    controller = a1x_control.JointController()
    
    # Wait for system to stabilize
    time.sleep(2)
    
    print("\n1. Reading initial joint states...")
    joints = controller.get_joint_states()
    if joints:
        print("Initial joint positions:")
        for name, position in joints.items():
            if name.startswith('arm_joint'):
                print(f"  {name}: {position:.4f} rad")
    
    # Define start and end positions for continuous tracking
    start_position = [0.0, 0.0043, -0.1, -0.0347, -0.0055, 0.0013]
    end_position = [0.5, 0.0043, -0.1, -0.0347, -0.0055, 0.0013]
    
    print(f"\n2. Moving smoothly from {start_position[0]:.1f} to {end_position[0]:.1f} rad on joint 1...")
    print("   Using continuous control without sleep delays")
    
    # Move to start position first
    print("   Moving to start position...")
    success = controller.move_to_position_smooth(start_position, steps=10, rate_hz=2)
    if success:
        print("   ✓ Reached start position")
    else:
        print("   ✗ Failed to reach start position")
        return
    
    # Wait a moment to see the position
    time.sleep(1)
    
    # Now perform continuous tracking with multiple intermediate steps
    print("   Executing continuous trajectory...")
    
    # Generate trajectory with fine steps (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
    trajectory_points = []
    for i in range(6):  # 0.0 to 0.5 in 0.1 increments
        joint1_pos = i * 0.15
        position = [joint1_pos, 0.0043, -0.1, -0.0347, -0.0055, 0.0013]
        trajectory_points.append(position)
    
    # Execute the full trajectory
    success = controller.execute_trajectory(trajectory_points, rate_hz=2)  # 2 Hz for visible movement
    
    if success:
        print("   ✓ Continuous trajectory completed successfully!")
    else:
        print("   ✗ Trajectory execution failed")
    
    # Read final position
    print("\n3. Reading final joint states...")
    joints = controller.get_joint_states()
    if joints:
        print("Final joint positions:")
        for name, position in joints.items():
            if name.startswith('arm_joint'):
                print(f"  {name}: {position:.4f} rad")
    
    print("\n4. Demonstrating reverse trajectory...")
    # Create reverse trajectory
    reverse_trajectory = trajectory_points[::-1]  # Reverse the list
    
    success = controller.execute_trajectory(reverse_trajectory, rate_hz=2)  # Slightly faster
    
    if success:
        print("   ✓ Reverse trajectory completed!")
    else:
        print("   ✗ Reverse trajectory failed")
    
    print("\nContinuous control example complete!")
    print("Key features demonstrated:")
    print("- No sleep() delays in control loop")
    print("- Smooth interpolated trajectories") 
    print("- Timer-based execution at specified rates")
    print("- RViz launch control (disabled in this example)")

if __name__ == "__main__":
    main()
