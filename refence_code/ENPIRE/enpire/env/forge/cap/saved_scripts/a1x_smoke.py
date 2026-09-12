# Read-only A1X smoke test: robot state + wrist camera, no motion.
state = get_robot_state()
arm = state.arms["right"]
print("joint_pos:", [round(v, 4) for v in arm.joint_pos])
print("gripper_pos:", round(arm.gripper_pos, 3))
print("ee_pos:", [round(v, 4) for v in arm.ee_pos])
print("ee_quat_xyzw:", [round(v, 4) for v in arm.ee_quat])

rgb = render_rgb("wrist")
depth = render_depth("wrist")
print("rgb:", rgb.shape, rgb.dtype)
print("depth:", depth.shape, depth.dtype, "mean_m:", round(float(depth[depth > 0].mean()), 4))
print("intrinsics [fx, fy, cx, cy]:", [round(v, 3) for v in get_camera_intrinsics("wrist")["intrinsics"]])
print("extrinsics:", get_camera_extrinsics("wrist"))
print("A1X smoke OK")
