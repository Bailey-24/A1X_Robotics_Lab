# Project Context

## Purpose
A1X SDK provides the ROS 2-based control stack for Galaxea Dynamics' A1X mobile manipulator, covering hardware drivers, motion control, inverse kinematics, and gripper operation workflows documented in the project README and vendor guide.@/readme.md#3-109

## Tech Stack
- ROS 2 application layer built on rclcpp/rclpy with ament_cmake build tooling.@/install/mobiman/share/mobiman/package.xml#4-76
- Colcon workspace generated under `install/` with `setup.*` environment hooks.@/readme.md#21-102
- Motion planning and optimization dependencies: OCS2 suite, Pinocchio, OSQP/OsqpEigen, NLopt, and Toppra for trajectory generation.@/install/mobiman/share/mobiman/package.xml#45-66
- Custom driver stack (`HDAS`) integrates with CAN bus hardware abstractions and message interfaces.@/install/HDAS/share/HDAS/package.xml#3-22

## Project Conventions

### Code Style
- ROS 2 packages use lowercase names with underscores and build via `ament_cmake`; follow ROS 2 C++ style (ament linters) for node implementations and Python PEP 8 for launch/utilities.@/install/mobiman/share/mobiman/package.xml#14-75
- Source and launch scripts assume environment sourced from `install/setup.*`; keep scripts executable and ROS 2 launch files (`*.py`) organized per package.@/readme.md#21-88

### Architecture Patterns
- Hardware abstraction (`HDAS`) exposes driver libraries and topics for CAN-connected actuators.@/install/HDAS/share/HDAS/package.xml#3-22
- Manipulator control stack (`mobiman`) layers ROS 2 controllers, TF/URDF, and OCS2-based motion planning for joint and end-effector tasks.@/install/mobiman/share/mobiman/package.xml#22-76
- Operators interact through ROS 2 topics and launch files to bring up drivers, visualization, joint-space, IK, and gripper control pipelines.@/readme.md#19-107

### Testing Strategy
- Manual verification via ROS 2 CLI: launch driver/manipulation nodes, observe feedback topics (e.g., `/joint_states`, `/hdas/pose_ee_arm`, `/hdas/feedback_gripper`), and publish test commands for joints, IK targets, and gripper positions.@/readme.md#31-107
- Validate CAN bus connectivity before runtime using documented setup commands to ensure deterministic controller behavior.@/readme.md#7-14

### Git Workflow
- Use feature branches per capability or package change; keep commits scoped to individual ROS 2 packages or launch workflows for traceability. Document deviations alongside related specs or proposals.

## Domain Context
- System targets Galaxea Dynamics A1X hardware; reference vendor guide for hardware capabilities and safety considerations.@/readme.md#3-88
- End-to-end workflows span joint-space control, relaxed IK, and gripper modules; ensure launch order and topic interfaces remain synchronized.@/readme.md#17-107

## Important Constraints
- CAN interface must be configured to 1 Mbps/5 Mbps FD with specified sample points before launching drivers.@/readme.md#7-14
- Execution requires ROS 2 environment sourced from workspace `install` directory for both driver and manipulation nodes.@/readme.md#21-102
- Conda environments should be deactivated before running IK stack to avoid dependency conflicts.@/readme.md#68-68

## External Dependencies
- ROS 2 middleware, controller manager, TF, URDF tooling, and sensor/message packages (@/install/mobiman/share/mobiman/package.xml#18-44).
- Optimization and kinematics libraries: OCS2 components, Pinocchio, OSQP/OsqpEigen, NLopt, Toppra, TRAC-IK (@/install/mobiman/share/mobiman/package.xml#45-67).
- Custom message packages (`hdas_msg`, `mobiman_msg`) consumed by HDAS and manipulator controllers.@/install/mobiman/share/mobiman/package.xml#59-60
