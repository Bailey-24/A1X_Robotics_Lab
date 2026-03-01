# YOLOe Vertical Grasp Example

End-to-end pick-and-place for the A1X robot using YOLOe text-prompt segmentation.

## Quick Start

```bash
# Basic usage (detects object from config.yaml)
python examples/yoloe_grasp/yoloe_grasp.py

# Override target object
python examples/yoloe_grasp/yoloe_grasp.py --target-name cup

# Dry run (no robot motion, good for testing)
python examples/yoloe_grasp/yoloe_grasp.py --dry-run --visualize

# With detection visualization
python examples/yoloe_grasp/yoloe_grasp.py --visualize
```

## Prerequisites

1. **CAN bus** configured at 1 Mbps / 5 Mbps FD
2. **ROS 2** environment sourced (or `a1x_control` auto-launch)
3. **Hand-eye calibration** completed → `examples/handeye/handeye_calibration.yaml`
4. **YOLOe checkpoint** at `refence_code/yoloe/yoloe-v8l-seg.pt`
5. **RealSense D405** connected via USB

## Pipeline Steps

| Step | Action | Module |
|------|--------|--------|
| 1 | Initialize robot + gripper | `a1x_control` |
| 2 | Move to observation pose | Joint angles `[0.0, 1.0, -0.93, 0.83, 0.0, 0.0]` |
| 3 | Capture aligned RGBD | `RGBDCapture` (D405) |
| 4 | Detect object (text prompt) | `YOLOeDetector` |
| 5 | Compute 3D grasp (depth) | `compute_grasp_from_detection` |
| 6 | Transform to base frame | Hand-eye calibration + FK + TCP offset |
| 7 | IK solve + execute grasp | `IKExecutor` (multi-seed) |
| 8 | Place + return to observation | Smooth interpolation |

## Configuration

All parameters are in [`config.yaml`](config.yaml):

- **`observation_pose`** / **`place_pose`** — Joint angles (rad)
- **`camera`** — Resolution, FPS, depth scale
- **`yoloe`** — Checkpoint, target names, device, confidence
- **`yoloe.grasp_height_offset`** — Height offset above detected surface (m)
- **`yoloe.grasp_y_correction`** — Lateral fine-tune for hand-eye drift (m)
- **`tcp_offset`** — Distance from `gripper_link` to fingertip contact (m)
- **`handeye`** — Calibration file path
- **`motion`** — Pre-grasp offset, lift height, smoothing steps, gripper delay
- **`safety`** — Confirmation prompts, joint limits

## CLI Arguments

| Flag | Description |
|------|-------------|
| `--config PATH` | Custom config file (default: `config.yaml`) |
| `--target-name NAME` | Override target object (e.g. `banana`) |
| `--dry-run` | Compute everything, skip robot motion |
| `--visualize` | Show detection overlay window |

## Key Design Decisions & Pitfalls

### 1. Top-Down Quaternion for IK

**Pitfall**: Using the "standard robotics" top-down quaternion `[w,x,y,z] = [0,1,0,0]`
(gripper Z-axis pointing down) will produce **zero reachable solutions** for the A1X.

**Why**: The A1X URDF defines the gripper approach axis as **+X**, not +Z. The
`gripper_joint` (fixed) has `xyz="0.08165 0 0"` — the gripper extends along its
local X-axis from `arm_link6`.

**Correct mapping** for top-down vertical grasp:

| Gripper axis | Base frame | Purpose |
|---|---|---|
| +X | −Z | Approach (pointing down) |
| +Y | +Y | Finger spread (horizontal) |
| +Z | +X | Perpendicular (forward) |

```python
# Correct rotation matrix and quaternion:
R_topdown = [[0, 0, 1],    # gripper X → base Z  (but we need -Z, so...)
             [0, 1, 0],    # gripper Y → base Y
             [-1, 0, 0]]   # gripper Z → base -X
# → wxyz = [0.707, 0, 0.707, 0]  (90° rotation around Y-axis)
```

**How to derive**: Always check the URDF link chain. Identify which axis of the
end-effector link is the approach direction (look at the fixed joint connecting
the last arm link to the gripper). Then construct the rotation matrix that maps
that axis to your desired approach direction in the base frame.

### 2. TCP Offset (Gripper Link → Fingertip)

**Pitfall**: IK solves for `gripper_link` position, but the fingertips are
further along the approach axis. Without TCP offset, the gripper link sits at
the object surface and the fingers push through the table.

**Calculation from URDF**:
- `gripper_finger_joint1` at `xyz="0.037 0.013 0"` from `gripper_link`
- Finger extends ~0.018m further
- **Total TCP offset ≈ 0.055–0.075m** (tune empirically)

In top-down orientation (gripper +X → base −Z), the offset raises the IK target
Z by `tcp_offset` so fingertips land at the object:

```python
grasp_pos_base[2] += tcp_offset  # raise gripper_link above object
```

### 3. Depth Strategy & Object Height Compensaton

**Pitfall**: The naive approach extracts depth as the median (p50) of the object's segmentation mask. This calculates the camera-to-*surface* distance. For a tall object (e.g., a cup), grasping at the surface results in a grip that is too shallow.

**Solution**: The pipeline uses a `mid_height` depth strategy by default (`yoloe.depth_strategy` in `config.yaml`).
1. It looks at the depth spread within the mask (p10 for the top surface, p90 for the table/bottom surface) to estimate the object's height.
2. It sets the grasp Z coordinate at a fraction of this estimated height (default `grasp_height_fraction: 0.5` = mid-height).
3. **If the gripper hovers too high or crashes too low**: Tune the `grasp_height_fraction` in `config.yaml` (0 = top surface, 1 = bottom surface). For very flat objects (<5 mm), it automatically falls back to the surface strategy.

### 4. Joint Limits Must Match URDF

**Pitfall**: Symmetric default limits like `[-1.57, 1.57]` are wrong.
The A1X URDF has asymmetric limits:

| Joint | Min | Max |
|-------|-----|-----|
| arm_joint1 | −2.8798 | 2.8798 |
| arm_joint2 | **0.0** | **3.1416** |
| arm_joint3 | **−3.3161** | **0.0** |
| arm_joint4 | −1.5708 | 1.5708 |
| arm_joint5 | −1.5708 | 1.5708 |
| arm_joint6 | −2.8798 | 2.8798 |

### 5. PCA-Based Orientation & Angle Transformations

**Pitfall 1: Image Space vs. Base Frame**
Angles calculated via PCA from 2D segmentation masks are in **camera image space** (where +X is right, +Y is down). Because the camera is mounted at an arbitrary orientation on the wrist, its axes do not align with the robot's base frame.
**Solution**: The PCA angle must be converted into a 3D direction vector and transformed using the rotation part of `T_cam_to_base` (derived from hand-eye calibration and forward kinematics) *before* calculating the final base-frame grasp angle.

**Pitfall 2: Rotation Composition Order**
When combining the PCA in-plane rotation (`R_z`) with the fixed top-down grasp rotation (`R_topdown`), the order of matrix multiplication is critical.
**Solution**: Use `R_grasp = R_z @ R_topdown` (left-multiply). This applies the rotation in the **base frame** around the vertical Z-axis, keeping the gripper's approach direction strictly straight down while spinning its fingers holding the object. Using `R_topdown @ R_z` (right-multiply) would rotate around the gripper's local Z-axis (which points forward), incorrectly causing the gripper to tilt away from the vertical approach axis.

### 6. IK Local Minima — Use Multi-Seed Random Restarts

**Pitfall**: Using a single IK seed often gets stuck in local minima with
FK errors of 0.06–0.15m, even for easily reachable poses.

**Solution**: Run IK from 12+ seeds (initial, observation pose, and 10 random
configurations within URDF limits). Pick the best solution by FK error,
prefering in-limits solutions.

### 7. Motion: Always Use Smooth Interpolation

**Pitfall**: `set_joint_positions()` sends a single instant command. For large
joint jumps (e.g., observation → pre-grasp, or lift → place), the motors hit
torque/velocity limits and stall.

**Solution**: Use `move_to_position_smooth(steps=60, rate_hz=10)` for all
motions. More steps = slower, gentler motion. Use 60 steps for large jumps,
30 for small descents.

### 8. Gripper Controller: Use Strong PD Gains

**Pitfall**: The "smooth" gripper config (`kp=1.0, kd=0.5`) is too weak to
fully close the gripper — it stalls at ~43%.

**Solution**: Use the regular config (`kp=2.0, kd=1.0`) by setting
`use_smooth_config=False` in `launch_gripper_controller()`.

## Troubleshooting

| Problem | Likely cause | Fix |
|---------|-------------|-----|
| "No object detected" | Lighting, camera angle, or wrong `target_names` | Adjust camera, check preview |
| IK failure (all seeds) | Wrong quaternion or target out of reach | Verify quaternion matches URDF approach axis |
| Gripper hits table | `tcp_offset` too small | Increase `tcp_offset` in config.yaml |
| Gripper hovers above object | `tcp_offset` too large | Decrease `tcp_offset` in config.yaml |
| Gripper lands off-center | Hand-eye calibration drift | Tune `grasp_y_correction` |
| Gripper won't fully close | Smooth gripper config (weak PD) | Set `use_smooth_config=False` |
| Robot stalls mid-motion | Instant command / too few steps | Use `move_to_position_smooth(steps=60)` |
| Depth errors | D405 out of range | Keep objects within 7–50 cm |
| Calibration drift | Camera mount shifted | Re-run `examples/handeye/handeye_calibration.py` |
