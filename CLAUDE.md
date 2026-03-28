# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A1X SDK is a Python-based robotic arm control interface for the Galaxea Dynamics A1X manipulator. It wraps ROS 2 communication in a clean Python API and includes examples for joint/gripper control, visual grasping, hand-eye calibration, and LLM-powered skills.

**Documentation**: https://docs.galaxea-dynamics.com/Guide/A1XY/software_introduction/ROS2/A1XY_Software_Introduction_ROS2/

## Required Hardware Setup (before running anything)

Configure the CAN bus after every computer restart:
```bash
sudo ip link set can0 type can bitrate 1000000 sample-point 0.875 dbitrate 5000000 fd on dsample-point 0.875
sudo ip link set up can0
```

Activate the environment:
```bash
conda activate a1x_ros
cd /home/ubuntu/projects/A1Xsdk
```

## Environment Variables (required for skills)

Both keys are already stored in `~/.zshrc` and exported automatically — no manual setup needed.

- `A1X_API_KEY` — LLM codegen proxy (motchat.com)
- `A1X_VLM_API_KEY` — Cloud VLM for scene understanding (chatanywhere.tech)

## Commands

### Running examples
```bash
python examples/motion/joint_control_once.py
python examples/motion/joint_control_smooth.py
python examples/motion/gripper_control.py
python examples/motion/read_ee_pose.py
python examples/motion/read_joint_pose.py
python examples/yoloe_grasp/yoloe_grasp.py --target-name cup --visualize
python examples/anygrasp_grasp/anygrasp_grasp.py --target-name cup --visualize
python examples/handeye/handeye_calibration.py --marker_id 42 --marker_size 0.10
python examples/camera/realsense_d405_rgb.py
```

### Running skills
```bash
python skills/a1x-arm-codegen/scripts/a1x_text_codegen.py [--execute]
python skills/a1x-grab-skill/scripts/a1x_grab.py [--execute] ["grab the yellow cube"]
python skills/a1x-realsense-vision/scripts/a1x_vision.py ["prompt"] [--save /tmp/snap.jpg]
python skills/a1x-tts/scripts/a1x_tts.py ["text"] [--voice nova] [--output /tmp/out.mp3]
```

### Running tests
```bash
pytest tests/
pytest tests/test_depth_grasp.py -v
pytest tests/test_pca_angle.py -v
```

## Architecture

### `a1x_control.py` — Core control API (1,259 lines)

Two main classes:

- **`A1XSystemManager`**: Lifecycle management — sources `install/setup.bash`, launches HDAS driver and mobiman controller stack as subprocesses. Auto-launches on import.

- **`JointController(Node)`**: High-level ROS 2 node. All topics use `RELIABLE` / `TRANSIENT_LOCAL` QoS (depth=1).

**Topics**:
- Subscribe: `/joint_states`, `/hdas/feedback_gripper`, `/motion_control/pose_ee_arm`, `/hdas/pose_ee_arm`
- Publish: `/motion_target/target_joint_state_arm`, `/motion_target/target_position_gripper`

**Key methods**:
- State: `get_joint_states()`, `get_gripper_state()`, `get_ee_pose()`, `get_ee_position()`, `get_ee_orientation()`, `get_current_ee_from_fk()`
- Joint control: `set_joint_positions(positions)`, `move_to_position_smooth(positions, steps, rate_hz, interpolation_type)`
- Gripper: `open_gripper()`, `close_gripper()`, `set_gripper_position(0-100)`
- End-effector (Cartesian, uses PyRoki IK internally): `move_ee_relative(dx, dy, dz)`, `move_ee_absolute(x, y, z, wxyz)`
- Trajectory: `interpolate_trajectory(start, end, steps, type)`, `execute_trajectory(waypoints, rate_hz)`

**6 arm joints** (`arm_joint1`–`arm_joint6`, radians). **Gripper range**: 0 (closed) to 100 (open).

**Coordinate system** (base frame): +X=forward, −X=backward, +Y=left, −Y=right, +Z=up, −Z=down.

### `skills/` — LLM-Powered Control (4 skills)

| Skill | Purpose | Entry Point |
|-------|---------|-------------|
| `a1x-arm-codegen` | Natural language → joint/EE control code | `scripts/a1x_text_codegen.py` |
| `a1x-grab-skill` | Code-as-Policies grasping (VLM + SAM3 + LLM codegen) | `scripts/a1x_grab.py` |
| `a1x-realsense-vision` | Camera capture + cloud VLM scene analysis (Qwen 3.5+) | `scripts/a1x_vision.py` |
| `a1x-tts` | Text-to-speech (6 voices, CN/EN, importable as library) | `scripts/a1x_tts.py` |

Each skill has a `SKILL.md` with full API docs. The grab skill uses primitives from `robot_api.py`: `pick()`, `place()`, `detect()`, `move_to_observation()`, `move_ee_relative()`, `describe_scene()`, `speak()`.

TTS can be imported as a library: `from skills.a1x_tts.scripts.a1x_tts import speak`

### `examples/yoloe_grasp/` — YOLOe visual grasping pipeline
Orchestrated by `yoloe_grasp.py`; modular subcomponents in `grasp_pipeline/`:
- YOLOe text-prompt detection → RealSense D405 depth → 3D grasp coordinate → PyRoki IK → smooth motion execution
- Config: `config.yaml` (observation pose, place pose, grasp height offset, TCP offset 0.075m, depth strategy)

### `examples/anygrasp_grasp/` — AnyGrasp model-based grasping
- 6-DOF grasp detection with optional YOLOe filtering, workspace bounding box, interactive top-N selection
- Config: `config.yaml`

### `pyroki/` — Python Robot Kinematics (JAX-based)
Standalone differentiable IK/FK solver from URDF. Used internally by `JointController.move_ee_relative/absolute`. Lazy-loaded on first EE control call (JIT compilation). Uses `jaxtyping` for array shape annotations.

### `install/` — Pre-built ROS 2 workspace
Contains HDAS (CAN bus driver), mobiman (mobile manipulator controller), OCS2, Pinocchio, TRAC-IK, and TOPPRA. **Do not modify.**

### `refence_code/` — Vendored reference implementations
YOLOe, SAM3, AnyGrasp SDK, GraspNet-baseline, FoundationPose. **Do not modify.**

## Config Files

- `examples/yoloe_grasp/config.yaml` — YOLOe grasp pipeline params
- `examples/anygrasp_grasp/config.yaml` — AnyGrasp pipeline params
- `examples/handeye/handeye_calibration.yaml` — Hand-eye calibration output (camera→EE transform)
- `pyroki/pyproject.toml` — PyRoki build config and ruff rules

## Code Style

- **Python 3.10+** — use modern syntax (`match`, `X | Y` unions, `from __future__ import annotations`)
- Use `logging` module (not `print`); logger name: `'a1x_control'`
- Return `Optional` for expected failures; log before returning `None`/`False`
- pyroki uses `jaxtyping` for array shape annotations (`Float[Array, "*batch n"]`)
- Ruff ignores: E501, E731, E741, F722, F821 (jaxtyping compatibility)

## Adding New Control Methods

1. Add method to `JointController` in `a1x_control.py` with type hints and docstring
2. Use the existing `RELIABLE`/`TRANSIENT_LOCAL` QoS profile (`self.qos_profile`)
3. Add a corresponding example in `examples/`

## Adding New Skills

1. Create `skills/<skill-name>/` with `scripts/` and `SKILL.md`
2. `SKILL.md` must document all available primitives/functions, usage examples, and trigger conditions
3. Skills can import from other skills (e.g. grab-skill imports TTS via `speak()`)
4. All generated code is logged to `logs/generate_code/` with timestamps
