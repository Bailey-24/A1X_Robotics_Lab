#!/usr/bin/env python3
"""
Smooth Gripper Control Example

This example demonstrates gripper control with reduced vibration
by using slower command rates and smoother transitions.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import a1x_control
import time

def main():
    print("A1X Smooth Gripper Control Example")
    print("=" * 40)
    
    # Initialize system with gripper enabled
    print("Initializing system with gripper support...")
    a1x_control.initialize(launch_rviz=False, enable_gripper=True)
    
    # Get controller instance
    controller = a1x_control.JointController()
    
    # Wait for system to stabilize
    time.sleep(3)
    
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
    
    # Smooth gripper control with longer delays
    print("\n3. Demonstrating smooth gripper control...")
    
    def smooth_gripper_move(target_pos, description, delay=4.0):
        """Move gripper smoothly with longer delay to reduce vibration"""
        print(f"   {description}...")
        success = controller.set_gripper_position(target_pos)
        if success:
            print(f"   ✓ Command sent: {target_pos:.1f}")
            time.sleep(delay)  # Longer delay for smooth movement
            
            # Check position
            pos = controller.get_gripper_state()
            if pos is not None:
                print(f"   Current position: {pos:.1f}")
                return pos
        else:
            print(f"   ✗ Failed to send command: {target_pos:.1f}")
        return None
    
    # Close gripper slowly
    smooth_gripper_move(0.0, "Closing gripper slowly", delay=5.0)
    
    # Open gripper slowly
    smooth_gripper_move(100.0, "Opening gripper slowly", delay=5.0)
    
    # Set specific positions with smooth transitions
    print("\n4. Setting specific positions with smooth transitions...")
    positions = [50.0, 25.0, 75.0, 0.0]
    
    for target_pos in positions:
        smooth_gripper_move(target_pos, f"Moving to {target_pos:.1f}", delay=3.0)
    
    # Test very slow, precise movement
    print("\n5. Testing precise, slow movement to reduce vibration...")
    print("   Moving from current position to 50% in small steps...")
    
    current_pos = controller.get_gripper_state()
    if current_pos is not None:
        target = 50.0
        steps = 5
        step_size = (target - current_pos) / steps
        
        for i in range(1, steps + 1):
            intermediate_pos = current_pos + (step_size * i)
            print(f"   Step {i}/{steps}: Moving to {intermediate_pos:.1f}")
            controller.set_gripper_position(intermediate_pos)
            time.sleep(2.0)  # Slow movement
            
            actual_pos = controller.get_gripper_state()
            if actual_pos is not None:
                print(f"   Reached: {actual_pos:.1f}")
    
    print("\nSmooth gripper control example complete!")
    print("\nVibration reduction techniques used:")
    print("- Longer delays between commands (3-5 seconds)")
    print("- Gradual position changes in small steps")
    print("- Reduced command frequency")
    print("- Separate gripper controller launch")

if __name__ == "__main__":
    main()
