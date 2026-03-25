# A1X Control API Reference

For LLM code generation. SDK source: `a1x_control.py` in project root.

## 1. Connection & Initialization

```python
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import a1x_control
import time

# System auto-initializes on import (launches HDAS driver + mobiman controller).
# For gripper or EE pose, call initialize() BEFORE getting controller:
# a1x_control.initialize(enable_gripper=True, enable_ee_pose=True)

controller = a1x_control.JointController()
time.sleep(2)  # Wait for system to stabilize
```

- `a1x_control.initialize(launch_rviz=False, enable_gripper=False, enable_ee_pose=False)`: Enable optional subsystems. Call before `JointController()`.
- No manual enable/disable needed — the system manager handles lifecycle.

## 2. Named Positions (radians, 6 joints)

| Name | Joint Values | Description |
|------|-------------|-------------|
| home | `[0.0, 0.0043, -0.1, -0.0347, -0.0055, 0.0013]` | Default resting position |
| observation | `[0.0, 1.0, -0.93, 0.83, 0.0, 0.0]` | Camera observation position for grasping |
| zero | `[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]` | All joints at zero |

## 3. Joint Control

The A1X has **6 joints**: `arm_joint1` through `arm_joint6`. All angles in **radians**.

### Read joint states
```python
joints = controller.get_joint_states()
# Returns: dict {"arm_joint1": 0.0, "arm_joint2": 0.0043, ...} or None
```

### Wait for joint data
```python
controller.wait_for_joint_states(timeout=10.0)  # Returns True if available
```

### Set joint positions (immediate command)
```python
controller.set_joint_positions([0.0, 0.0043, -0.1, -0.0347, -0.0055, 0.0013])
# Sends a single command. No interpolation. Returns True/False.
```

### Smooth movement (recommended for most cases)
```python
controller.move_to_position_smooth(
    target_positions=[0.0, 1.0, -0.93, 0.83, 0.0, 0.0],
    steps=20,                    # interpolation steps (more = smoother)
    rate_hz=10.0,                # control rate
    interpolation_type='cosine', # 'linear' or 'cosine' (cosine = ease in/out)
    wait_for_convergence=True,   # wait until arm settles
    convergence_tolerance=0.015, # max error in radians
    convergence_timeout=2.0,     # seconds to wait for settling
)
```

### Trajectory execution
```python
# Generate interpolated trajectory
trajectory = controller.interpolate_trajectory(
    start_pos=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    end_pos=[0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
    steps=20,
    interpolation_type='cosine',  # or 'linear'
)

# Execute trajectory
controller.execute_trajectory(trajectory, rate_hz=10.0)
```

## 4. Gripper Control

Gripper position range: **0** (fully closed) to **100** (fully open).

**Requires**: `a1x_control.initialize(enable_gripper=True)` before use.

```python
a1x_control.initialize(enable_gripper=True)
controller = a1x_control.JointController()
controller.wait_for_gripper_state(timeout=10.0)

controller.open_gripper()              # Position = 100
controller.close_gripper()             # Position = 0
controller.set_gripper_position(50.0)  # Specific position (0-100)

# Smooth gripper movement (less vibration)
controller.set_gripper_position_smooth(50.0, delay=2.0)

# Gradual movement in steps
controller.move_gripper_gradually(target_position=0.0, steps=5, step_delay=1.0)

# Read current gripper state
pos = controller.get_gripper_state()   # Returns float (0-100) or None
```

## 5. End-Effector (Cartesian) Control

No extra initialization needed — IK solver loads automatically on first use.

### Coordinate system (base frame)

```
        +Z (up)
         |
         |
         +------ +X (forward, away from base)
        /
       /
     +Y (left)
```

Human-direction mapping:
- forward = +X, backward = −X
- left = +Y, right = −Y
- up = +Z, down = −Z

### Read current EE pose (via FK — always available)
```python
ee = controller.get_current_ee_from_fk()
# Returns: {"position": [x, y, z], "wxyz": [w, x, y, z]} or None
print(f"EE at: {ee['position']}")
```

### Relative EE movement (most common for "move forward/up/left N cm")
```python
# Move forward 2cm
controller.move_ee_relative(dx=0.02)

# Move up 3cm
controller.move_ee_relative(dz=0.03)

# Combined: forward 2cm + left 1cm
controller.move_ee_relative(dx=0.02, dy=0.01)

# Full signature:
controller.move_ee_relative(
    dx=0.0, dy=0.0, dz=0.0,   # offset in meters
    steps=30,                   # interpolation steps
    rate_hz=10.0,               # control rate
    interpolation_type='cosine',
    wait_for_convergence=True,
)
```

### Absolute EE positioning
```python
# Move EE to specific point in space
controller.move_ee_absolute(x=0.25, y=0.0, z=0.2)

# With explicit orientation (wxyz quaternion)
controller.move_ee_absolute(x=0.25, y=0.0, z=0.2, wxyz=[0.0, 0.0, 1.0, 0.0])
```

### EE pose via ROS topic (optional, requires enable_ee_pose=True)
```python
a1x_control.initialize(enable_ee_pose=True)
controller = a1x_control.JointController()
controller.wait_for_ee_pose(timeout=10.0)
ee_pose = controller.get_ee_pose()
# Returns: {"position": {"x", "y", "z"}, "orientation": {"x", "y", "z", "w"}} or None
```

## 6. Minimal Runnable Template

```python
#!/usr/bin/env python3
"""A1X arm control script generated by LLM."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import a1x_control

# Named positions
HOME = [0.0, 0.0043, -0.1, -0.0347, -0.0055, 0.0013]
OBSERVATION = [0.0, 1.0, -0.93, 0.83, 0.0, 0.0]

def main():
    # Initialize (add enable_gripper=True or enable_ee_pose=True if needed)
    # a1x_control.initialize(enable_gripper=True)

    controller = a1x_control.JointController()
    time.sleep(2)

    # Wait for joint data
    if not controller.wait_for_joint_states(timeout=10):
        print("ERROR: No joint state data")
        return

    # Read current state
    joints = controller.get_joint_states()
    print("Current joints:", joints)

    # Move to home position smoothly
    controller.move_to_position_smooth(HOME, steps=20, rate_hz=10.0, interpolation_type='cosine')
    time.sleep(1)

    # TODO: Add your motion sequence here

    print("Done.")

if __name__ == "__main__":
    main()
```

## 7. Safety Notes

- Always ensure CAN bus is configured before running:
  ```bash
  sudo ip link set can0 type can bitrate 1000000 sample-point 0.875 dbitrate 5000000 fd on dsample-point 0.875
  sudo ip link set up can0
  ```
- Test with small movements first.
- Use `move_to_position_smooth()` with `interpolation_type='cosine'` for safe, smooth motion.
- The system auto-shuts down on process exit (via `atexit`).
