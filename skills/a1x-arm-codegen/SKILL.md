---
name: a1x-arm-codegen
description: Guide LLM to generate executable Python control scripts for the Galaxea A1X robotic arm based on natural language descriptions. Uses the a1x_control API over ROS 2.
---

## Overview

This skill generates **runnable Python scripts** that control a Galaxea A1X 6-axis robotic arm, based on the user's natural language description of desired motions.

SDK: `a1x_control.py` (project root); reference examples in `examples/motion/`.

## When to Use

- User describes arm movements in natural language (e.g., "move to the observation position", "open the gripper", "wave the arm")
- User asks for a control script or code to drive the A1X arm
- User wants to sequence multiple motions

## Code Generation Rules

### 1. Initialization

- The A1X uses ROS 2 under the hood. `import a1x_control` auto-launches the HDAS driver and mobiman controller.
- No manual enable/disable needed — system lifecycle is managed automatically.
- If gripper control is needed, call `a1x_control.initialize(enable_gripper=True)` before getting the controller.
- If end-effector pose reading is needed, call `a1x_control.initialize(enable_ee_pose=True)`.
- Get the controller via `controller = a1x_control.JointController()`.
- Always `time.sleep(2)` after getting the controller to let the system stabilize.
- Always call `controller.wait_for_joint_states(timeout=10)` before any motion.

### 2. Joint Configuration

- The A1X has **6 joints**: `arm_joint1` through `arm_joint6`.
- All joint angles are in **radians**.
- Joint position arrays must always have exactly **6 elements**.

### 3. Named Positions

Use these predefined positions when the user refers to them:

| Name | Values (radians) | When to use |
|------|-------------------|-------------|
| home | `[0.0, 0.0043, -0.1, -0.0347, -0.0055, 0.0013]` | "go home", "rest position", "home position", "initial position" |
| observation | `[0.0, 1.0, -0.93, 0.83, 0.0, 0.0]` | "observation position", "look position", "camera position", "observe" |
| zero | `[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]` | "zero position", "all zeros" |

### 4. Motion Commands

- **Default behavior**: Unless the user explicitly specifies a starting position, always move to the **observation position** first before performing any EE (Cartesian) movements such as "move forward", "move up", etc. This ensures the arm starts from a known, safe pose. Skip this if the user says "from current position" or names a different starting position.
- **Prefer `move_to_position_smooth()`** for all joint-space movements. It interpolates from the current position to the target, producing safe, smooth motion.
- Use `interpolation_type='cosine'` for acceleration/deceleration (ease in/out). Use `'linear'` for constant-speed motion.
- Use `steps=20, rate_hz=10.0` as defaults. Increase steps for longer movements.
- For precise positioning, set `wait_for_convergence=True`.
- For multi-waypoint sequences, build a trajectory list and use `execute_trajectory()`.
- `set_joint_positions()` sends a single immediate command — only use for small adjustments or when the user explicitly wants direct control.

### 5. End-Effector (Cartesian) Control

Use EE control when the user describes motion in human-intuitive directions (forward, up, left, etc.) or specifies distances in centimeters/meters rather than joint angles.

**Coordinate system (base frame):**

| Direction | Axis | Sign | Example |
|-----------|------|------|---------|
| forward (away from base) | X | + | "move forward 2cm" → `dx=0.02` |
| backward (toward base) | X | − | "move back 5cm" → `dx=-0.05` |
| left | Y | + | "move left 3cm" → `dy=0.03` |
| right | Y | − | "move right 1cm" → `dy=-0.01` |
| up | Z | + | "move up 2cm" → `dz=0.02` |
| down | Z | − | "move down 1cm" → `dz=-0.01` |

**Relative EE movement** (most common):
```python
controller.move_ee_relative(dx=0.02, dy=0.0, dz=0.0)  # forward 2cm
controller.move_ee_relative(dx=0.0, dy=0.0, dz=0.03)  # up 3cm
```

**Absolute EE positioning**:
```python
controller.move_ee_absolute(x=0.25, y=0.0, z=0.2)  # go to specific point
```

**Reading current EE pose** (via FK, no extra initialization needed):
```python
ee = controller.get_current_ee_from_fk()
print(f"Position: {ee['position']}")  # [x, y, z] meters
print(f"Orientation: {ee['wxyz']}")   # [w, x, y, z] quaternion
```

**When to use joint control vs EE control:**
- Joint control: go to named positions (home, observation), wave, specific joint angles
- EE control: "move forward/up/left N cm", "go to position (x,y,z)", fine adjustments after reaching a named position

**Important:** EE control uses IK solving internally (PyRoki). The first call loads the URDF and JIT-compiles the solver (~3-5 seconds). Subsequent calls are fast.

### 6. Gripper Commands

- Gripper range: 0 (closed) to 100 (open).
- `controller.open_gripper()` / `controller.close_gripper()` for full open/close.
- `controller.set_gripper_position(value)` for specific positions.
- `controller.move_gripper_gradually()` for smooth, low-vibration gripper motion.
- Always add `time.sleep(2)` after gripper commands to allow movement to complete.

### 7. Reading State

- `controller.get_joint_states()` → dict of joint name to position (radians).
- `controller.get_gripper_state()` → float (0-100) or None.
- `controller.get_current_ee_from_fk()` → dict with `position` [x,y,z] and `wxyz` [w,x,y,z] (computed via FK, always available).
- `controller.get_ee_pose()` → dict with position and orientation (from ROS topic, requires `enable_ee_pose=True`).

### 8. Script Structure

Every generated script must follow this structure:

```python
#!/usr/bin/env python3
"""<description of what the script does>"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import a1x_control

HOME = [0.0, 0.0043, -0.1, -0.0347, -0.0055, 0.0013]
OBSERVATION = [0.0, 1.0, -0.93, 0.83, 0.0, 0.0]

def main():
    # Initialize subsystems if needed
    # a1x_control.initialize(enable_gripper=True, enable_ee_pose=True)

    controller = a1x_control.JointController()
    time.sleep(2)

    if not controller.wait_for_joint_states(timeout=10):
        print("ERROR: No joint state data available")
        return

    # --- Motion sequence here ---

    print("Done.")

if __name__ == "__main__":
    main()
```

### 9. Safety

- Always use `move_to_position_smooth()` with reasonable step counts — never jump to far-away positions with `set_joint_positions()`.
- Remind the user to ensure the workspace is clear before running.
- Suggest small test movements first if the user's target positions are unfamiliar.
- CAN bus must be configured before running (see `references/a1x-api.md`).
- Add `time.sleep()` between sequential motions to allow the arm to settle.

## Reference

- **API details and templates**: `references/a1x-api.md`
- **Working examples**: `examples/motion/` in the project root
