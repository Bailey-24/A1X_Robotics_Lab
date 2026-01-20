# Change: Add Python Joint Control API

## Why
Users need a simple Python API to control robotic arm joints without deep ROS knowledge. The current workflow requires manual ROS topic management across three terminals, making it difficult for robotics engineers unfamiliar with ROS to control the A1X manipulator effectively.

## What Changes
- Create a high-level Python API that abstracts ROS joint control complexity
- Provide simple methods for joint state reading, joint position commands, and system lifecycle management
- Implement three Python scripts corresponding to the current three-terminal workflow
- Hide ROS topic publishing/subscribing, message formatting, and environment setup behind clean interfaces

## Impact
- Affected specs: New `joint-control` capability
- Affected code: New Python modules in project root for user-facing API
- Users can control joints with simple Python calls instead of manual ROS commands
- Maintains compatibility with existing ROS-based control stack
