# A1X Joint Control API

Simple Python interface for controlling A1X robotic arm joints without requiring ROS knowledge.

## Overview

The A1X Control API provides a clean, Pythonic interface to control the A1X robotic manipulator. It automatically handles:
- ROS environment setup
- HDAS driver launch
- Mobiman manipulation stack launch
- Joint state monitoring
- Joint position commands

## Quick Start

```python
import a1x_control

# System automatically starts up on import
controller = a1x_control.JointController()

# Read current joint positions
joints = controller.get_joint_states()
print(joints)

# Set target joint positions (6 values in radians)
controller.set_joint_positions([0.0, 0.0043, -0.1, -0.0347, -0.0055, 0.0013])
```

## Installation

No installation required. Simply ensure you're in the A1X SDK directory:

```bash
cd /home/ubuntu/projects/A1Xsdk
python3 your_script.py
```

## API Reference

### JointController

Main interface for joint control operations.

#### Methods

**`get_joint_states() -> Optional[Dict[str, float]]`**
- Returns current joint positions as a dictionary mapping joint names to positions in radians
- Returns `None` if no joint data is available yet

**`set_joint_positions(positions: List[float]) -> bool`**
- Sets target joint positions
- `positions`: List of 6 joint positions in radians for joints arm_joint1 through arm_joint6
- Returns `True` if command was sent successfully

**`wait_for_joint_states(timeout: float = 10.0) -> bool`**
- Waits for joint state data to become available
- `timeout`: Maximum time to wait in seconds
- Returns `True` if joint states are available

## Joint Names and Order

The API expects joint positions in this order:
1. `arm_joint1`
2. `arm_joint2` 
3. `arm_joint3`
4. `arm_joint4`
5. `arm_joint5`
6. `arm_joint6`

## Examples

### Basic Control
```python
import a1x_control
import time

controller = a1x_control.JointController()

# Wait for system to be ready
time.sleep(2)

# Read current state
joints = controller.get_joint_states()
if joints:
    for name, pos in joints.items():
        print(f"{name}: {pos:.4f} rad")

# Move to target position
target = [0.0, 0.0043, -0.1, -0.0347, -0.0055, 0.0013]
controller.set_joint_positions(target)
```

### Continuous Monitoring
```python
import a1x_control
import time

controller = a1x_control.JointController()

while True:
    joints = controller.get_joint_states()
    if joints:
        print(f"Joints: {list(joints.values())}")
    time.sleep(1)
```

## System Architecture

The API automatically manages:

1. **Environment Setup**: Sources ROS workspace and sets up Python paths
2. **Driver Launch**: Starts HDAS driver (`ros2 launch HDAS a1xy.py`)
3. **Manipulation Stack**: Starts mobiman (`ros2 launch mobiman A1x_jointTrackerdemo_launch.py`)
4. **ROS Communication**: Handles topic subscriptions and publishing
5. **Background Processing**: Runs ROS spinning in separate thread
6. **Cleanup**: Automatically shuts down processes on exit

## ROS Topics Used

- **Subscribe**: `/joint_states` - Current joint positions
- **Publish**: `/motion_target/target_joint_state_arm` - Target joint commands

## Error Handling

The API includes comprehensive error handling:
- Environment validation
- Process monitoring  
- Graceful shutdown
- Detailed logging

Check console output for diagnostic information.

## Troubleshooting

**"No joint state data available"**
- System may still be starting up, wait a few seconds
- Check that HDAS driver launched successfully

**"Failed to initialize A1X system"**
- Ensure you're in the correct directory (`/home/ubuntu/projects/A1Xsdk`)
- Check that ROS workspace was built successfully
- Verify hardware connections

**Import errors**
- Make sure you're running from the A1X SDK root directory
- Check that the install directory exists and contains ROS packages

## Comparison to Manual ROS Commands

| Manual ROS Workflow | A1X Control API |
|-------------------|-----------------|
| Terminal 1: `source install/setup.zsh && ros2 launch HDAS a1xy.py` | Automatic on import |
| Terminal 2: `source install/setup.zsh && ros2 launch mobiman A1x_jointTrackerdemo_launch.py` | Automatic on import |
| Terminal 3: `ros2 topic echo /joint_states` | `controller.get_joint_states()` |
| Terminal 3: `ros2 topic pub /motion_target/target_joint_state_arm ...` | `controller.set_joint_positions([...])` |

## Safety Notes

- Always verify joint positions are within safe ranges before commanding
- The API does not currently implement joint limit checking
- Monitor the robot during operation
- Use emergency stop if available

## License

Part of the A1X SDK. Refer to main project license.
