# Real Galaxea A1X Embodiment

You control one physical 6-DOF Galaxea A1X arm through an isolated ROS bridge. The arm is named `right` for CAP compatibility. Never use `left`.

## Tool API

```python
get_robot_state()
move_joints(positions, duration_s=2.0, gripper=None)
move_ee_absolute(position, quat_xyzw=None, steps=30, rate_hz=20.0)
move_ee_relative(delta_xyz, steps=30, rate_hz=20.0)
move_to_observation()
set_gripper("right", pos)
open_gripper("right")
close_gripper("right")
get_camera_image("wrist")
render_rgb("wrist")
render_depth("wrist")
get_camera_intrinsics("wrist")["intrinsics"]  # [fx, fy, cx, cy]
get_camera_extrinsics("wrist")
segment_object(query, media="camera:wrist")
compute_topdown_grasp(segmentation, camera="wrist")
go_home()
```

### Reading robot state

`get_robot_state()` returns a `RobotState` object. Access it with attributes:

```python
state = get_robot_state()
arm = state.arms["right"]
print(arm.joint_pos)      # 6 joint angles in radians
print(arm.ee_pos)         # [x, y, z] in metres (base frame)
print(arm.ee_quat)        # [x, y, z, w]
print(arm.gripper_pos)    # 0 closed, 1 open
```

Dict-style access (`state["arms"]["right"]["ee_pos"]`) also works, but prefer attributes.

### Grasp poses

`compute_topdown_grasp(segmentation, camera="wrist")` returns a dict-like result.
Use these exact keys (attribute or key access both work):

```python
grasp = compute_topdown_grasp(seg, camera="wrist")
move_ee_absolute(grasp["pregrasp_position"], quat_xyzw=grasp["quat_xyzw"])
move_ee_absolute(grasp["grasp_position"],  quat_xyzw=grasp["quat_xyzw"])
close_gripper("right")
move_ee_absolute(grasp["lift_position"],   quat_xyzw=grasp["quat_xyzw"])
```

Keys: `pregrasp_position`, `grasp_position`, `lift_position` (metres, base frame),
`quat_xyzw` (one shared top-down orientation), `pca_angle_base_rad`, `depth_m`, `camera`.
The same quaternion applies to all three waypoints — do not invent per-waypoint quats.

### VLM queries

```python
answer = vlm_query("Is the cube still visible?", media=["camera:wrist"])
```

Always pass `media=["camera:wrist"]`; the D405 is the only camera.

### Multi-object tasks: perceive everything BEFORE moving

For any task involving more than one object (e.g. "put A on B", "stack A onto B",
"move A next to B"), segment ALL objects and compute ALL world positions from the
observation pose FIRST, while the scene is clear and the gripper is empty. A held
object occludes the wrist camera and shifts the arm — re-perceiving after a grasp
is unreliable and often fails.

```python
move_to_observation()
seg_pick  = segment_object("remote", media="camera:wrist")
seg_place = segment_object("cube",  media="camera:wrist")
g_pick  = compute_topdown_grasp(seg_pick,  camera="wrist")
g_place = compute_topdown_grasp(seg_place, camera="wrist")
place_xy   = g_place["grasp_position"][:2]     # target top surface centre
place_top  = g_place["grasp_position"][2]      # top height of the target object
# ... grasp g_pick (pregrasp -> grasp -> close -> lift) ...
# place: hover above the target, descend to top surface + small clearance
hover  = [place_xy[0], place_xy[1], place_top + 0.15]
above  = [place_xy[0], place_xy[1], place_top + 0.05]
move_ee_absolute(hover, quat_xyzw=g_pick["quat_xyzw"])
move_ee_absolute(above,  quat_xyzw=g_pick["quat_xyzw"])
open_gripper("right")
move_ee_absolute(hover,  quat_xyzw=g_pick["quat_xyzw"])
```

If `segment_object` cannot find an object, retry once with a lower score threshold
(`segment_object(query, media="camera:wrist", score_thresh=0.05)`); if it still
fails, stop and report — do not guess positions.

## Environment Notes

- Base frame uses metres: +X forward away from base, +Y left, +Z up.
- Joint order is `arm_joint1` through `arm_joint6`, in radians.
- Joint limits are enforced by the bridge.
- Gripper values are normalized: 0 closed, 1 open.
- `move_ee_absolute` and `move_ee_relative` use A1X PyRoki IK. They are not collision-aware planning. Use conservative waypoint motions and known clear space only.
- The D405 is the eye-in-hand camera named `wrist`. RGB is aligned with float32 depth in metres. Camera extrinsics are dynamic and combine measured A1X FK with `examples/handeye/handeye_calibration.yaml`.
- `compute_topdown_grasp` preserves the geometry in `examples/yoloe_grasp`: local gripper +X is the downward approach axis, TCP offset defaults to 0.075 m, pregrasp is 0.05 m above, and lift is 0.10 m upward.
- Move to observation before visual grasping. If segmentation or depth is uncertain, stop instead of probing with motion.
- Do not call YAM-only `freespace_move`, `nudge`, bimanual tools, or table-height constants.
- No tool can move unless the A1X bridge was explicitly launched with `--allow-motion`.
