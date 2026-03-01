#!/usr/bin/env python3
"""Viser Grasp Preview — Visualize grasp poses in 3D before executing on robot.

Loads the computed grasp waypoints (pre-grasp, grasp, lift) and displays them
in a Viser 3D browser alongside the robot URDF. Solves IK for each waypoint
and shows the resulting arm configuration. An "Execute" button sends commands.

Usage:
    python examples/yoloe_grasp/grasp_pipeline/grasp_viser_preview.py

    Or launched automatically from examples.yoloe_grasp.grasp_pipeline.py with --preview flag.
"""
from __future__ import annotations

import sys
import time
import argparse
import threading
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Stub pyroki.viewer to avoid viser/websockets crash during import
import types
sys.modules["pyroki.viewer"] = types.ModuleType("pyroki.viewer")

import pyroki as pk
import pyroki.solve as pks
import yourdfpy
import viser


def load_a1x_urdf():
    """Load the A1X URDF with proper mesh path resolution."""
    urdf_path = Path("/home/ubuntu/projects/A1Xsdk/install/mobiman/lib/mobiman/configs/urdfs/a1x.urdf")

    def resolve_package_uri(fname: str) -> str:
        package_prefix = "package://mobiman/"
        if fname.startswith(package_prefix):
            relative_path = fname[len(package_prefix):]
            return str(Path("/home/ubuntu/projects/A1Xsdk/install/mobiman/share/mobiman") / relative_path)
        return fname

    return yourdfpy.URDF.load(urdf_path, filename_handler=resolve_package_uri)


def main():
    parser = argparse.ArgumentParser(description="Viser Grasp Preview")
    parser.add_argument("--config", default="examples/yoloe_grasp/grasp_pipeline/config.yaml")
    parser.add_argument(
        "--grasp-data", default=None,
        help="Path to .npz file with grasp waypoints. If not given, recomputes from captured_data."
    )
    args = parser.parse_args()

    # Load config
    config_path = PROJECT_ROOT / args.config
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Load robot
    print("[1/4] Loading URDF...")
    urdf = load_a1x_urdf()
    robot = pk.Robot.from_urdf(urdf)
    target_link_name = "gripper_link"

    # Compute grasp waypoints
    print("[2/4] Computing grasp poses...")
    from examples.yoloe_grasp.grasp_pipeline.coordinate_transform import (
        load_handeye_calibration,
        compute_T_base_ee_from_fk,
        matrix_to_position_wxyz,
        compute_pre_grasp_pose,
        compute_lift_pose,
    )

    observation_pose = cfg["observation_pose"]
    T_base_ee = compute_T_base_ee_from_fk(observation_pose)
    T_ee_cam = load_handeye_calibration(
        str(PROJECT_ROOT / cfg["handeye"]["calibration_path"])
    )

    # Load captured grasp data
    data_dir = PROJECT_ROOT / "grasp_pipeline" / "captured_data"
    if args.grasp_data:
        data = np.load(args.grasp_data)
        grasp_trans_cam = data["translation"]
    else:
        # Re-run YOLOe detection on captured data
        import cv2
        from PIL import Image
        import scipy.io as scio

        color = cv2.imread(str(data_dir / "color.png"))
        depth = np.array(Image.open(str(data_dir / "depth.png")))
        meta = scio.loadmat(str(data_dir / "meta.mat"))
        intrinsic = meta["intrinsic_matrix"]

        yoloe_cfg = cfg.get("yoloe", {})
        factor_depth = cfg["camera"].get("factor_depth", 10000)

        from examples.yoloe_grasp.grasp_pipeline.yoloe_detector import YOLOeDetector
        from examples.yoloe_grasp.grasp_pipeline.depth_grasp import compute_grasp_from_detection

        target_names = yoloe_cfg.get("target_names", ["box"])
        if isinstance(target_names, str):
            target_names = [target_names]

        checkpoint = str(PROJECT_ROOT / yoloe_cfg["checkpoint"])
        detector = YOLOeDetector(checkpoint, device=yoloe_cfg.get("device", "cuda:0"))
        det = detector.detect(color, target_names, conf_threshold=yoloe_cfg.get("conf_threshold", 0.25))

        if det is None:
            print("ERROR: No object detected in captured image!")
            return

        bbox, mask, det_score, class_name = det
        print(f"  Detected: {class_name} ({det_score:.3f})")

        grasp_result = compute_grasp_from_detection(
            bbox, depth, intrinsic,
            factor_depth=factor_depth,
            mask=mask,
        )
        if grasp_result is None:
            print("ERROR: Failed to compute grasp from detection!")
            return

        _, grasp_trans_cam, _ = grasp_result

    print(f"  Grasp in camera frame: {grasp_trans_cam}")

    # Transform position to base frame
    T_cam_to_base = T_base_ee @ T_ee_cam
    grasp_pos_base = (T_cam_to_base @ np.array([*grasp_trans_cam, 1.0]))[:3]

    # Apply height offset
    height_offset = cfg.get("yoloe", {}).get("grasp_height_offset", 0.015)
    grasp_pos_base[2] += height_offset

    # Top-down rotation in base frame
    R_topdown = np.array([
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
    ])
    T_grasp_base = np.eye(4)
    T_grasp_base[:3, :3] = R_topdown
    T_grasp_base[:3, 3] = grasp_pos_base

    # Compute waypoints
    pre_offset = cfg["motion"].get("pre_grasp_offset", 0.05)
    lift_height = cfg["motion"].get("post_grasp_lift", 0.10)
    T_pre = compute_pre_grasp_pose(T_grasp_base, pre_offset)
    T_lift = compute_lift_pose(T_grasp_base, lift_height)

    grasp_pos, grasp_wxyz = matrix_to_position_wxyz(T_grasp_base)
    pre_pos, pre_wxyz = matrix_to_position_wxyz(T_pre)
    lift_pos, lift_wxyz = matrix_to_position_wxyz(T_lift)

    print(f"\n  Grasp waypoints (base frame):")
    print(f"    Pre-grasp: [{pre_pos[0]:.4f}, {pre_pos[1]:.4f}, {pre_pos[2]:.4f}]")
    print(f"    Grasp:     [{grasp_pos[0]:.4f}, {grasp_pos[1]:.4f}, {grasp_pos[2]:.4f}]")
    print(f"    Lift:      [{lift_pos[0]:.4f}, {lift_pos[1]:.4f}, {lift_pos[2]:.4f}]")

    # Set up Viser
    print("\n[3/4] Starting Viser server...")
    server = viser.ViserServer()

    # Add ground grid at z=0
    server.scene.add_grid("/ground", width=1.0, height=1.0, cell_size=0.05)

    # Add robot
    from viser.extras import ViserUrdf
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/robot")

    # Set robot to observation pose
    obs_cfg = list(observation_pose) + [0.0, 0.0]
    urdf_vis.update_cfg(np.array(obs_cfg))

    # Add grasp markers
    # Pre-grasp (blue)
    server.scene.add_icosphere("/waypoints/pre_grasp", radius=0.012, color=(80, 80, 255),
                                position=tuple(pre_pos))
    server.scene.add_label("/waypoints/pre_grasp_label", text="Pre-grasp",
                           position=tuple(pre_pos + [0, 0, 0.03]))

    # Grasp (red)
    server.scene.add_icosphere("/waypoints/grasp", radius=0.015, color=(255, 50, 50),
                                position=tuple(grasp_pos))
    server.scene.add_label("/waypoints/grasp_label", text="Grasp",
                           position=tuple(grasp_pos + [0, 0, 0.03]))

    # Lift (green)
    server.scene.add_icosphere("/waypoints/lift", radius=0.012, color=(50, 255, 50),
                                position=tuple(lift_pos))
    server.scene.add_label("/waypoints/lift_label", text="Lift",
                           position=tuple(lift_pos + [0, 0, 0.03]))

    # Approach direction arrow (red line from pre-grasp to grasp)
    line_pts = np.array([pre_pos, grasp_pos, lift_pos])
    server.scene.add_spline_catmull_rom(
        "/waypoints/path", positions=line_pts,
        line_width=3.0, color=(255, 200, 0),
    )

    # Add coordinate frame at grasp
    server.scene.add_frame("/waypoints/grasp_frame",
                           position=tuple(grasp_pos),
                           wxyz=tuple(grasp_wxyz),
                           axes_length=0.06, axes_radius=0.003)

    # IK target gizmo for interactive adjustment
    ik_target = server.scene.add_transform_controls(
        "/ik_target",
        scale=0.12,
        position=tuple(grasp_pos),
        wxyz=tuple(grasp_wxyz),
        depth_test=False,
    )
    server.scene.add_icosphere("/ik_target/sphere", radius=0.01, color=(255, 255, 0))

    # GUI
    with server.gui.add_folder("Grasp Preview"):
        waypoint_select = server.gui.add_dropdown(
            "Show Waypoint", options=["Pre-grasp", "Grasp", "Lift"],
            initial_value="Grasp"
        )
        solve_ik_btn = server.gui.add_button("Solve IK for Waypoint")
        status_text = server.gui.add_text("Status", initial_value="Ready")

    with server.gui.add_folder("Robot Control"):
        enable_robot = server.gui.add_checkbox("Enable Robot Control", initial_value=False)
        execute_btn = server.gui.add_button("🚀 Execute Full Grasp Sequence")

    with server.gui.add_folder("Position"):
        pos_display = server.gui.add_text(
            "IK Target",
            initial_value=f"x={grasp_pos[0]:.3f} y={grasp_pos[1]:.3f} z={grasp_pos[2]:.3f}"
        )

    # IK solve state
    last_solution = np.array(obs_cfg)
    waypoints = {
        "Pre-grasp": (pre_pos, pre_wxyz),
        "Grasp": (grasp_pos, grasp_wxyz),
        "Lift": (lift_pos, lift_wxyz),
    }

    @solve_ik_btn.on_click
    def on_solve(_):
        nonlocal last_solution
        wp_name = waypoint_select.value
        wp_pos, wp_wxyz = waypoints[wp_name]

        # Move IK target to waypoint
        ik_target.position = tuple(wp_pos)
        ik_target.wxyz = tuple(wp_wxyz)

        try:
            solution = pks.solve_ik(
                robot=robot,
                target_link_name=target_link_name,
                target_position=wp_pos,
                target_wxyz=wp_wxyz,
                initial_joint_config=last_solution,
            )
            last_solution = solution
            urdf_vis.update_cfg(solution)
            status_text.value = f"IK solved for {wp_name}: {solution[:6].round(3)}"
        except Exception as e:
            status_text.value = f"IK failed: {e}"

    @execute_btn.on_click
    def on_execute(_):
        if not enable_robot.value:
            status_text.value = "⚠️ Enable Robot Control first!"
            return

        status_text.value = "Executing grasp sequence..."

        try:
            import a1x_control
            if not hasattr(a1x_control, '_initialized'):
                a1x_control.initialize(enable_gripper=True)
                a1x_control._initialized = True
            controller = a1x_control.JointController()

            # Solve IK for each waypoint and execute
            sol = last_solution.copy()
            for name in ["Pre-grasp", "Grasp"]:
                wp_pos, wp_wxyz = waypoints[name]
                sol = pks.solve_ik(
                    robot=robot,
                    target_link_name=target_link_name,
                    target_position=wp_pos,
                    target_wxyz=wp_wxyz,
                    initial_joint_config=sol,
                )
                status_text.value = f"Moving to {name}..."
                arm_joints = list(sol[:6].astype(float))
                controller.set_joint_positions(arm_joints)
                urdf_vis.update_cfg(sol)
                time.sleep(2.0)

            # Close gripper
            status_text.value = "Closing gripper..."
            controller.close_gripper()
            time.sleep(1.0)

            # Lift
            wp_pos, wp_wxyz = waypoints["Lift"]
            sol = pks.solve_ik(
                robot=robot,
                target_link_name=target_link_name,
                target_position=wp_pos,
                target_wxyz=wp_wxyz,
                initial_joint_config=sol,
            )
            status_text.value = "Lifting..."
            arm_joints = list(sol[:6].astype(float))
            controller.set_joint_positions(arm_joints)
            urdf_vis.update_cfg(sol)
            time.sleep(1.5)

            status_text.value = "✅ Grasp complete!"
        except Exception as e:
            status_text.value = f"❌ Error: {e}"

    print("\n[4/4] Viser server ready!")
    print("=" * 60)
    print("Open http://localhost:8080/ in your browser")
    print("=" * 60)
    print()
    print("Features:")
    print("  - Blue/Red/Green spheres = Pre-grasp/Grasp/Lift waypoints")
    print("  - Yellow gizmo = IK target (draggable)")
    print("  - 'Solve IK' = preview arm configuration for selected waypoint")
    print("  - 'Execute' = send commands to real robot")
    print()
    print("Press Ctrl+C to exit")

    # Main loop: keep server alive + update IK target display
    try:
        while True:
            pos = np.array(ik_target.position)
            pos_display.value = f"x={pos[0]:.3f} y={pos[1]:.3f} z={pos[2]:.3f}"
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nShutting down...")


if __name__ == "__main__":
    main()
