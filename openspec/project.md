# Project Context

## Purpose
A1X SDK provides the ROS 2-based control stack for Galaxea Dynamics' A1X mobile manipulator, covering hardware drivers, motion control, inverse kinematics, and gripper operation workflows. It includes high-level Python interfaces for easy integration and research.@/readme.md#3-109

## Tech Stack
- **ROS 2**: Application layer built on rclcpp/rclpy with ament_cmake build tooling.@/install/mobiman/share/mobiman/package.xml#4-76
- **Colcon**: Workspace generated under `install/` with `setup.*` environment hooks.@/readme.md#21-102
- **PyRoki**: Python Robot Kinematics library for differentiable forward kinematics, collision checking, and optimization-based IK.@/pyroki/README.md#1-18
- **Intel RealSense SDK 2.0 (`librealsense`)**: Cross-platform library for RealSense depth cameras (D405, D435i, etc.) supporting depth/color streaming.@/librealsense/readme.md#10-11
- **Viser**: Interactive 3D visualization and control UI for Python-based manipulation workflows.@/examples/ik_control_viser.py#27-28
- **a1x_control**: High-level Python wrapper for simplified joint control, automatically managing ROS 2 lifecycle.@/README_A1X_Control.md#1-13
- **Motion Planning**: OCS2 suite, Pinocchio, OSQP/OsqpEigen, NLopt, and Toppra for trajectory generation.@/install/mobiman/share/mobiman/package.xml#45-66
- **Drivers**: Custom `HDAS` driver stack for CAN bus communication.@/install/HDAS/share/HDAS/package.xml#3-22

## Project Conventions

### Code Style
- ROS 2 packages use lowercase names with underscores and build via `ament_cmake`; follow ROS 2 C++ style (ament linters) for node implementations and Python PEP 8 for launch/utilities.@/install/mobiman/share/mobiman/package.xml#14-75
- Python examples and scripts follow PEP 8 and use type hinting where possible.@/examples/ik_control_viser.py#40-52

### Architecture Patterns
- **Core Control**: Hardware abstraction (`HDAS`) and manipulator control (`mobiman`) run as ROS 2 nodes, exposing topics for joint states and targets.@/install/mobiman/share/mobiman/package.xml#22-76
- **Python API**: `a1x_control` abstracts ROS 2 node management, allowing pure Python scripts to transparently launch drivers and communicate via topics.@/README_A1X_Control.md#108-118
- **Workflows**: Users can operate in "Manual ROS Mode" (launching terminals) or "API Mode" (running Python scripts).@/README_A1X_Control.md#151-156
- **Kinematics**: `PyRoki` is used for client-side kinematics solving in Python examples, while `mobiman` handles on-robot control loops.@/examples/ik_control_viser.py#6-8

### Testing Strategy
- Manual verification via ROS 2 CLI or `viser` visualization: launch driver/manipulation nodes or scripts, observe feedback topics, and visually verify robot motion.@/readme.md#31-107
- Validate CAN bus connectivity before runtime using documented setup commands to ensure deterministic controller behavior.@/readme.md#7-14

### Git Workflow
- Use feature branches per capability or package change; keep commits scoped to individual ROS 2 packages or launch workflows for traceability. Document deviations alongside related specs or proposals.

## Domain Context
- System targets Galaxea Dynamics A1X hardware; reference vendor guide for hardware capabilities and safety considerations.@/readme.md#3-88
- Integration with Intel RealSense cameras for perception-driven manipulation tasks.@/librealsense/readme.md#38-48
- End-to-end workflows span joint-space control, relaxed IK, and gripper modules.@/readme.md#17-107

## Important Constraints
- CAN interface must be configured to 1 Mbps/5 Mbps FD with specified sample points before launching drivers.@/readme.md#7-14
- Execution requires ROS 2 environment sourced from workspace `install` directory for both driver and manipulation nodes.@/readme.md#21-102
- Conda environments should be deactivated before running IK stack to avoid dependency conflicts, unless using `a1x_control` which handles environment setup.@/readme.md#68-68

## External Dependencies
- ROS 2 middleware, controller manager, TF, URDF tooling.@/install/mobiman/share/mobiman/package.xml#18-44
- `librealsense`: Intel RealSense SDK 2.0.@/librealsense/readme.md
- `PyRoki`: Kinematics and optimization toolkit.@/pyroki/README.md
- Optimization and kinematics libraries: OCS2 components, Pinocchio, OSQP/OsqpEigen, NLopt, Toppra, TRAC-IK.@/install/mobiman/share/mobiman/package.xml#45-67
