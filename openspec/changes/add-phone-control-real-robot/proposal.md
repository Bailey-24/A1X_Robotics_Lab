# Change: Add Phone Control to Real Robot IK

## Why
Users need to control the real A1X robot via iPhone 6D pose input, combining the phone teleoperation capability from `03_phone_ik_sim.py` with the real robot control in `ik_control_viser.py`.

## What Changes
- [MODIFY] `examples/ik_control_viser.py`: Add `PhoneListener` class for receiving iPhone pose data
- [MODIFY] `examples/ik_control_viser.py`: Add Phone Control UI folder with IP input, enable checkbox, reset button, status display
- [MODIFY] `examples/ik_control_viser.py`: Integrate phone pose delta calculation into main control loop

## Impact
- Affected specs: `phone-teleoperation` (new capability)
- Affected code: `examples/ik_control_viser.py`
- Dependencies: `asmagic`, `scipy` (for Rotation)

## User Review Required

> [!IMPORTANT]
> **Phone IP Configuration**: Currently defaults to `192.168.31.196`. Should this be made configurable via command-line argument as well?

> [!WARNING]
> **Dual Input Conflict**: When phone control is enabled, the gizmo becomes a read-only indicator showing the phone-computed target. Is this the desired behavior?
