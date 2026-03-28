# AGENTS.md - AI Assistant Instructions

Instructions for AI coding assistants working in the A1X SDK repository.

## Priority: Use Skills First

**Before writing any robot control code, always check if an existing skill handles the task.** The `skills/` directory contains LLM-powered systems that already solve common workflows. Invoking a skill is almost always better than writing new code from scratch.

### Available Skills

| Skill | When to Use | Command |
|-------|-------------|---------|
| **a1x-arm-codegen** | User wants to move the arm, go to a named position, or execute joint/EE motion | `python skills/a1x-arm-codegen/scripts/a1x_text_codegen.py [--execute]` |
| **a1x-grab-skill** | User wants to pick up, grab, move, or place objects | `python skills/a1x-grab-skill/scripts/a1x_grab.py [--execute] ["instruction"]` |
| **a1x-realsense-vision** | User wants to see, describe, or analyze what the camera sees | `python skills/a1x-realsense-vision/scripts/a1x_vision.py ["prompt"]` |
| **a1x-tts** | User wants the robot to speak, announce, or provide audio feedback | `python skills/a1x-tts/scripts/a1x_tts.py ["text"] [--voice nova]` |

**Each skill has a `SKILL.md`** file with full API documentation, primitives, and examples. Read it before modifying the skill.

### Skill Trigger Examples

- "move to observation position" → **a1x-arm-codegen**
- "move forward 2cm" → **a1x-arm-codegen**
- "grab the red cup" → **a1x-grab-skill**
- "pick up all yellow objects" → **a1x-grab-skill**
- "what's on the table?" (physical scene) → **a1x-realsense-vision** or **a1x-grab-skill** (question mode)
- "say hello" / "announce task complete" → **a1x-tts**

### Skill Architecture

- **a1x-arm-codegen**: Natural language → LLM generates Python using `a1x_control.py` API → executes on robot
- **a1x-grab-skill**: Scene capture → VLM understanding → LLM generates code using `robot_api.py` primitives (`pick()`, `place()`, `detect()`, `move_to_observation()`, `move_ee_relative()`, `describe_scene()`, `speak()`) → sandboxed execution
- **a1x-realsense-vision**: RealSense D405 capture → Qwen 3.5+ cloud VLM analysis
- **a1x-tts**: Cloud TTS API → audio playback. Importable: `from skills.a1x_tts.scripts.a1x_tts import speak`

### Environment Variables (required for skills)

Both keys are already stored in `~/.zshrc` — no manual export needed.

- `A1X_API_KEY` — LLM codegen proxy
- `A1X_VLM_API_KEY` — Cloud VLM (Qwen)

## Core API Reference

### `a1x_control.py` — JointController

**6 arm joints** (`arm_joint1`–`arm_joint6`, radians). **Gripper**: 0 (closed) to 100 (open).

**Coordinate system** (base frame): +X=forward, −X=backward, +Y=left, −Y=right, +Z=up, −Z=down.

Key methods:
- `get_joint_states()`, `get_ee_pose()`, `get_current_ee_from_fk()`
- `set_joint_positions(positions)`, `move_to_position_smooth(positions, steps, rate_hz, interpolation_type)`
- `open_gripper()`, `close_gripper()`, `set_gripper_position(0-100)`
- `move_ee_relative(dx, dy, dz)`, `move_ee_absolute(x, y, z, wxyz)`

All publishers/subscribers use RELIABLE + TRANSIENT_LOCAL QoS. EE control uses PyRoki IK internally (lazy-loaded, JIT compiled on first call).

## Code Style

- **Python 3.10+** — modern syntax (`match`, `X | Y` unions)
- `logging` module only (not `print`); logger name: `'a1x_control'`
- Return `Optional` for expected failures; log before returning `None`/`False`
- Ruff ignores: E501, E731, E741, F722, F821

## Do Not Modify

- `install/` — Pre-built ROS 2 workspace (HDAS driver, mobiman controller)
- `refence_code/` — Vendored reference implementations (YOLOe, SAM3, AnyGrasp)
- `pyroki/` — Standalone IK/FK library (modify only if explicitly asked)

## Running Tests

```bash
pytest tests/ -v
```

## Adding New Control Methods

1. Add method to `JointController` in `a1x_control.py` with type hints and docstring
2. Use `self.qos_profile` (RELIABLE/TRANSIENT_LOCAL)
3. Add a corresponding example in `examples/`

## Adding New Skills

1. Create `skills/<skill-name>/` with `scripts/` and `SKILL.md`
2. `SKILL.md` documents primitives, usage, and trigger conditions
3. Skills can import from other skills (e.g. `speak()` from a1x-tts)
4. Generated code is logged to `logs/generate_code/`
