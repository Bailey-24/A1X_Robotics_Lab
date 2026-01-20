# AGENTS.md - AI Assistant Instructions

Instructions for AI coding assistants working in the A1X SDK repository.

## Project Overview

A1X SDK is a Python-based robotic arm control interface for the A1X manipulator.

**Key Components:**
- `a1x_control.py` - Main control API wrapping ROS2 communication
- `examples/` - Usage examples for joint/gripper/EE control
- `pyroki/` - Python Robot Kinematics library (JAX-based IK solver, separate subproject)
- `install/` - Pre-built ROS2 packages (HDAS driver, mobiman stack) - **DO NOT MODIFY**

## Build/Lint/Test Commands

### Root Project (a1x_control)
```bash
python3 examples/joint_control_once.py    # Run examples directly
python3 examples/gripper_control.py
```

### pyroki Subproject
```bash
cd pyroki
pip install -e ".[dev]"                   # Install with dev deps

# Linting & Formatting
ruff check .                              # Lint
ruff check --fix .                        # Auto-fix
ruff format .                             # Format

# Type Checking
pyright                                   # Run type checker

# Testing
pytest                                    # All tests
pytest tests/test_ik_vmap.py              # Single file
pytest tests/test_ik_vmap.py::test_integration_ik_basic  # Single test
pytest -v -x                              # Verbose, stop on first fail
```

## Code Style Guidelines

### Python Version & Imports
- **Python 3.10+** required. Use modern syntax (`match`, `|` unions).
- Import order: stdlib → third-party → local (blank line between groups)

```python
from __future__ import annotations
import os
from typing import List, Optional, Dict, Any

import jax
from rclpy.node import Node

from ._robot_urdf_parser import JointInfo
```

### Type Annotations
**Always use type hints for function signatures:**
```python
def set_joint_positions(self, positions: List[float]) -> bool: ...
def get_joint_states(self) -> Optional[Dict[str, float]]: ...

# pyroki uses jaxtyping for array shapes
from jaxtyping import Float
def forward_kinematics(self, cfg: Float[Array, "*batch actuated_count"]) -> Float[Array, "*batch link_count 7"]: ...
```

### Naming Conventions
| Element | Convention | Example |
|---------|------------|---------|
| Classes | PascalCase | `JointController`, `A1XSystemManager` |
| Functions/methods | snake_case | `get_joint_states`, `set_gripper_position` |
| Constants | UPPER_SNAKE | `ROS_DOMAIN_ID` |
| Private | Leading underscore | `_initialize_system`, `_controller` |

### Docstrings
```python
def set_joint_positions(self, positions: List[float]) -> bool:
    """Set target joint positions.
    
    Args:
        positions: List of 6 joint positions in radians
        
    Returns:
        True if command was sent successfully
    """
```

### Error Handling
- Use `logging` module, not print statements
- Return `Optional` types for expected failures; log before returning
```python
logger = logging.getLogger('a1x_control')

def set_gripper_position(self, position: float) -> bool:
    if not (0 <= position <= 100):
        logger.error(f"Gripper position must be 0-100, got {position}")
        return False
    # ...
```

### Ruff Configuration (pyroki)
Enabled: E (pycodestyle), F (Pyflakes), PLC/PLE/PLR/PLW (Pylint)  
Ignored: E501 (line length), E731 (lambda), E741 (ambiguous names), F722/F821 (jaxtyping)

## Project-Specific Patterns

### ROS2 Integration (a1x_control)
```python
# QoS profile for reliable communication
self.qos_profile = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    durability=DurabilityPolicy.TRANSIENT_LOCAL
)
```

### pyroki JAX Patterns
```python
@jdc.pytree_dataclass
class Robot:
    joints: JointInfo
    joint_var_cls: jdc.Static[type[jaxls.Var[Array]]]  # Static for non-differentiable

@jdc.jit
def forward_kinematics(self, cfg): ...  # JIT-compiled method
```

### Testing Pattern (pyroki)
```python
def test_integration_ik_basic():
    """Integration: Standard usage description."""
    urdf = load_robot_description("panda_description")
    robot = pk.Robot.from_urdf(urdf)
    target_idx = robot.links.names.index("panda_hand")
    
    solve_batch = jax.vmap(_solve_ik, in_axes=(None, None, 0, 0))
    res = solve_batch(robot, jnp.array(target_idx), target_wxyz, target_pos)
    assert res.shape == (batch_size, robot.joints.num_actuated_joints)
```

## File Structure
```
A1Xsdk/
├── a1x_control.py         # Main control API
├── examples/              # Usage examples
├── pyroki/                # IK solver subproject
│   ├── src/pyroki/        # Library source
│   ├── tests/             # pytest tests
│   └── pyproject.toml     # Build config + ruff/pyright settings
├── install/               # Pre-built ROS2 packages (DO NOT MODIFY)
└── openspec/              # Spec-driven development (see openspec/AGENTS.md)
```

## Common Tasks

### Adding a new control method
1. Add method to `JointController` class in `a1x_control.py`
2. Include type hints and docstring
3. Add error handling with logging
4. Create example in `examples/`

### Modifying pyroki
1. Make changes in `pyroki/src/pyroki/`
2. Run `ruff check . && ruff format .`
3. Run `pyright` for type checking
4. Run `pytest` to verify tests pass

## OpenSpec Integration
For proposals, specs, and change management, see `openspec/AGENTS.md`.
```bash
openspec list              # List active changes
openspec spec list --long  # List specifications
openspec validate --strict # Validate specs
```
