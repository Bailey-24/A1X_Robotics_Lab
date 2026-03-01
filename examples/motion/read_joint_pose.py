#!/usr/bin/env python3
"""
Continuous Joint Monitoring Example

This example demonstrates continuous monitoring of joint states,
equivalent to running 'ros2 topic echo /joint_states' but with Python.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import a1x_control
import time

def main():
    print("A1X Continuous Joint Monitoring Example")
    print("=" * 45)
    print("Press Ctrl+C to stop monitoring")
    
    # Get controller instance
    controller = a1x_control.JointController()
    
    # Wait for system to stabilize
    time.sleep(2)
    
    try:
        while True:
            # Read current joint states
            joints = controller.get_joint_states()
            
            if joints:
                print(f"\n[{time.strftime('%H:%M:%S')}] Joint States:")
                for name, position in joints.items():
                    print(f"  {name}: {position:+7.4f} rad")
            else:
                print(f"[{time.strftime('%H:%M:%S')}] No joint data available")
            
            time.sleep(1)  # Update every second
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")

if __name__ == "__main__":
    main()
