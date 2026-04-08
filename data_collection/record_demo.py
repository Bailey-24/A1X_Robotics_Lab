#!/usr/bin/env python3
"""Record YOLOe-grasp demonstrations for imitation learning.

Runs the yoloe_grasp pipeline as the expert policy while a background
thread records joint states + D405 camera images at a fixed frequency.
Episodes are saved as HDF5 files via the control_your_robot framework.

Usage:
    python data_collection/record_demo.py --target-name banana
    python data_collection/record_demo.py --target-name cup --num-episodes 10
    python data_collection/record_demo.py --dry-run --num-episodes 1
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import yaml

# ── Path setup ─────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data_collection.a1x_robot import A1XRecordingRobot
from data_collection.recorder import DemoRecorder

# Reuse yoloe_grasp step functions (pipeline runs unmodified)
from examples.yoloe_grasp.yoloe_grasp import (
    step_1_initialize,
    step_2_move_to_observation,
    step_4_detect_object,
    step_5_compute_grasp,
    step_6_transform_to_base,
    step_7_execute_grasp,
    step_8_place_and_return,
    move_to_joint_pose,
    load_config as load_grasp_config,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("record_demo")


def load_record_config(config_path: str | Path) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_episode(
    controller,
    robot: A1XRecordingRobot,
    recorder: DemoRecorder,
    grasp_cfg: dict,
    episode_id: int,
    args,
) -> bool:
    """Run one grasp episode with recording.

    Returns True if the episode completed successfully.
    """
    observation_pose = grasp_cfg["observation_pose"]
    place_pose = grasp_cfg.get("place_pose", observation_pose)
    camera_cfg = grasp_cfg["camera"]
    yoloe_cfg = grasp_cfg["yoloe"]
    handeye_path = str(PROJECT_ROOT / grasp_cfg["handeye"]["calibration_path"])
    motion_cfg = grasp_cfg["motion"]
    safety_cfg = grasp_cfg["safety"]
    factor_depth = camera_cfg.get("factor_depth", 10000)
    grasp_height_offset = yoloe_cfg.get("grasp_height_offset", 0.015)

    print(f"\n{'=' * 60}")
    print(f"  EPISODE {episode_id}")
    print(f"{'=' * 60}")

    # ── Start recording ────────────────────────────────────────────
    recorder.start_episode()

    try:
        # ── Step 2: Move to observation pose ───────────────────────
        step_2_move_to_observation(controller, observation_pose, dry_run=args.dry_run)

        # ── Step 3: Capture RGBD (from recording sensor) ──────────
        print("\n" + "=" * 60)
        print("STEP 3: Capture RGBD (from recording sensor)")
        print("=" * 60)

        if args.dry_run:
            w = camera_cfg.get("width", 640)
            h = camera_cfg.get("height", 480)
            color = np.zeros((h, w, 3), dtype=np.uint8)
            depth = np.full((h, w), 300, dtype=np.uint16)
            intrinsic = np.array([
                [604.0, 0.0, w / 2],
                [0.0, 604.0, h / 2],
                [0.0, 0.0, 1.0],
            ])
            print(f"  [DRY RUN] Synthetic {w}x{h} frame")
        else:
            sensor = robot.get_d405_sensor()
            color, depth, intrinsic = sensor.get_capture_for_detection()
            print(f"  Captured {color.shape[1]}x{color.shape[0]} frame from recording sensor")
            print(f"  Intrinsics: fx={intrinsic[0,0]:.1f}, fy={intrinsic[1,1]:.1f}")

        # ── Step 4: Detect object ─────────────────────────────────
        detection = step_4_detect_object(
            color,
            yoloe_cfg,
            target_name_override=args.target_name,
            visualize=args.visualize,
            detector_type=args.detector,
        )

        if detection is None:
            print("\n  No object detected — returning to observation pose.")
            move_to_joint_pose(controller, observation_pose, "observation", args.dry_run)
            recorder.stop_episode(episode_id=episode_id)
            return False

        bbox, mask, score, class_name = detection

        # ── Step 5: Compute 3D grasp ──────────────────────────────
        depth_strategy = yoloe_cfg.get("depth_strategy", "surface")
        grasp_height_fraction = yoloe_cfg.get("grasp_height_fraction", 0.5)
        rot, trans, quality, pca_angle = step_5_compute_grasp(
            bbox, depth, intrinsic, mask, factor_depth,
            depth_strategy=depth_strategy,
            grasp_height_fraction=grasp_height_fraction,
        )

        # ── Step 6: Transform to base frame ───────────────────────
        tcp_offset = grasp_cfg.get("tcp_offset", 0.055)
        grasp_y_correction = yoloe_cfg.get("grasp_y_correction", 0.0)
        enable_pca = yoloe_cfg.get("enable_pca_rotation", True)
        T_grasp, T_pre_grasp, T_lift = step_6_transform_to_base(
            rot, trans, observation_pose, handeye_path, motion_cfg,
            grasp_height_offset=grasp_height_offset,
            tcp_offset=tcp_offset,
            grasp_y_correction=grasp_y_correction,
            grasp_angle=pca_angle if enable_pca else 0.0,
        )

        # ── Step 7: Execute grasp ─────────────────────────────────
        success = step_7_execute_grasp(
            controller, T_pre_grasp, T_grasp, T_lift,
            motion_cfg, safety_cfg, dry_run=args.dry_run,
        )

        if not success:
            print("\n  Grasp execution failed or was aborted.")
            move_to_joint_pose(controller, observation_pose, "observation", args.dry_run)
            recorder.stop_episode(episode_id=episode_id)
            return False

        # ── Step 8: Place and return ──────────────────────────────
        step_8_place_and_return(
            controller, place_pose, observation_pose, dry_run=args.dry_run,
        )

        print(f"\n  Episode {episode_id} complete")
        return True

    finally:
        # Always stop recording (even on error/abort)
        recorder.stop_episode(episode_id=episode_id)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Record YOLOe-grasp demonstrations for imitation learning",
    )
    parser.add_argument(
        "--config",
        default=str(Path(__file__).parent / "config.yaml"),
        help="Path to recording config.yaml",
    )
    parser.add_argument(
        "--detector",
        choices=["yoloe", "sam3"],
        default="sam3",
        help="Detector backend (default: sam3)",
    )
    parser.add_argument(
        "--target-name",
        type=str,
        default=None,
        help="Override target object name (e.g. 'banana', 'cup')",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=None,
        help="Number of episodes to record (overrides config)",
    )
    parser.add_argument(
        "--start-episode",
        type=int,
        default=None,
        help="Starting episode index (overrides config)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run pipeline without hardware (synthetic data)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Show detection visualization",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("  YOLOe Grasp — Demonstration Recorder")
    print("=" * 60)
    if args.dry_run:
        print("  *** DRY RUN MODE ***")

    # ── Load configs ───────────────────────────────────────────────
    rec_cfg = load_record_config(args.config)
    recording = rec_cfg["recording"]
    camera_cfg = rec_cfg["camera"]

    grasp_config_path = str(PROJECT_ROOT / rec_cfg["yoloe_grasp_config"])
    grasp_cfg = load_grasp_config(grasp_config_path)

    save_path = recording["save_path"]
    task_name = recording["task_name"]
    record_freq = recording.get("save_freq", 20)
    pre_roll_frames = recording.get("pre_roll_frames", 10)
    motion_threshold = recording.get("motion_threshold", 0.005)
    num_episodes = args.num_episodes or recording.get("num_episodes", 10)
    start_episode = args.start_episode if args.start_episode is not None else recording.get("start_episode", 0)

    # Override task name with target if provided
    if args.target_name:
        task_name = f"{task_name}_{args.target_name}"

    print(f"\n  Task: {task_name}")
    print(f"  Episodes: {start_episode} .. {start_episode + num_episodes - 1}")
    print(f"  Record freq: {record_freq} Hz")
    print(f"  Save path: {save_path}{task_name}/")

    # ── Step 1: Initialize A1X ─────────────────────────────────────
    controller = step_1_initialize(grasp_cfg, dry_run=args.dry_run)

    # ── Create recording robot ─────────────────────────────────────
    condition = {
        "save_path": save_path,
        "task_name": task_name,
        "save_format": "hdf5",
        "save_freq": record_freq,
    }

    robot = A1XRecordingRobot(
        controller=controller,
        condition=condition,
        camera_serial=camera_cfg.get("serial"),
        start_episode=start_episode,
    )

    if not args.dry_run:
        robot.set_up(
            camera_width=camera_cfg.get("width", 640),
            camera_height=camera_cfg.get("height", 480),
            camera_fps=camera_cfg.get("fps", 30),
        )

    recorder = DemoRecorder(
        robot,
        record_freq=record_freq,
        pre_roll_frames=pre_roll_frames,
        motion_threshold=motion_threshold,
    )

    # ── Record episodes ────────────────────────────────────────────
    successful = 0
    try:
        for i in range(num_episodes):
            ep_id = start_episode + i

            if i > 0:
                print(f"\n  Press Enter to start episode {ep_id} (Ctrl+C to stop)...")
                try:
                    input()
                except EOFError:
                    break

            ok = run_episode(controller, robot, recorder, grasp_cfg, ep_id, args)
            if ok:
                successful += 1
                print(f"  Saved: {save_path}{task_name}/{ep_id}.hdf5")
    except KeyboardInterrupt:
        print("\n\n  Recording interrupted by user.")
    finally:
        if not args.dry_run:
            robot.teardown()

    # ── Summary ────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  RECORDING COMPLETE")
    print(f"  Successful: {successful}/{num_episodes} episodes")
    print(f"  Data: {save_path}{task_name}/")
    print(f"{'=' * 60}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
