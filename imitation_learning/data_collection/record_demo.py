#!/usr/bin/env python3
"""Record YOLOe-grasp demonstrations for imitation learning.

Runs the yoloe_grasp pipeline as the expert policy while a background
thread records joint states + D405 camera images at a fixed frequency.
Episodes are saved as HDF5 files via the control_your_robot framework.

After each successful grasp, the object is repositioned using a cyclic
relocation pattern (right → down → left → up, 5 cm each) with a 10°
wrist rotation, keeping the object within the workspace for the next
episode.  This enables fully automated multi-episode data collection.

Usage:
    python imitation_learning/data_collection/record_demo.py --target-name banana
    python imitation_learning/data_collection/record_demo.py --target-name cup --num-episodes 10
    python imitation_learning/data_collection/record_demo.py --dry-run --num-episodes 8
"""
from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from pathlib import Path

import numpy as np
import yaml

# ── Path setup ─────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from imitation_learning.data_collection.a1x_robot import A1XRecordingRobot
from imitation_learning.data_collection.recorder import DemoRecorder

# Reuse yoloe_grasp step functions (pipeline runs unmodified)
from examples.yoloe_grasp.yoloe_grasp import (
    step_1_initialize,
    step_2_move_to_observation,
    step_4_detect_object,
    step_5_compute_grasp,
    step_6_transform_to_base,
    step_7_execute_grasp,
    move_to_joint_pose,
    load_config as load_grasp_config,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("record_demo")

# ── Cyclic relocation constants ────────────────────────────────────────
# After each successful grasp+lift, the object is moved to a nearby
# position so it stays in the workspace for the next episode.
# Coordinate system (base frame): +X=forward, +Y=left, +Z=up.
RELOCATION_OFFSETS = [
    (0.0, -0.05, 0.0),   # episode % 4 == 0: 5 cm right  (-Y)
    (0.0,  0.0, -0.05),  # episode % 4 == 1: 5 cm down   (-Z)
    (0.0,  0.05, 0.0),   # episode % 4 == 2: 5 cm left   (+Y)
    (0.0,  0.0,  0.05),  # episode % 4 == 3: 5 cm up     (+Z)
]
RELOCATION_LABELS = ["RIGHT (-Y)", "DOWN (-Z)", "LEFT (+Y)", "UP (+Z)"]

WRIST_ROTATION_RAD = math.radians(10)  # 10° per episode on joint 6

# Joint 6 limits from URDF
_JOINT6_MIN = -2.8798
_JOINT6_MAX = 2.8798

# Stop after this many consecutive failures (object likely left workspace)
MAX_CONSECUTIVE_FAILURES = 3


def load_record_config(config_path: str | Path) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def step_8_relocate_and_return(
    controller,
    episode_idx: int,
    observation_pose: list[float],
    dry_run: bool = False,
) -> None:
    """Move the grasped object to a nearby position, rotate wrist, release, return.

    Cyclic pattern (indexed by ``episode_idx % 4``):
        0 → 5 cm right, 1 → 5 cm down, 2 → 5 cm left, 3 → 5 cm up.
    After the directional move, the wrist (joint 6) is rotated +10° to
    diversify object orientation for the next grasp attempt.
    """
    print("\n" + "=" * 60)
    print("STEP 8: Relocate Object & Return")
    print("=" * 60)

    direction_idx = episode_idx % 4
    dx, dy, dz = RELOCATION_OFFSETS[direction_idx]
    label = RELOCATION_LABELS[direction_idx]

    print(f"  Direction: {label}  (dx={dx}, dy={dy}, dz={dz})")

    if dry_run:
        print(f"  [DRY RUN] Would move_ee_relative({dx}, {dy}, {dz})")
        print(f"  [DRY RUN] Would rotate wrist +{math.degrees(WRIST_ROTATION_RAD):.0f} deg")
        print(f"  [DRY RUN] Would open gripper, retract, return to observation")
        move_to_joint_pose(controller, observation_pose, "observation", dry_run)
        return

    # 1. Move 5 cm in the designated direction (object still held)
    print(f"  Moving {label} by 5 cm...")
    ok = controller.move_ee_relative(dx, dy, dz, steps=30, rate_hz=20.0)
    if not ok:
        logger.warning("Relocation move failed — releasing and returning")
        controller.open_gripper()
        time.sleep(1.0)
        move_to_joint_pose(controller, observation_pose, "observation", dry_run=False)
        return

    # 2. Rotate wrist +10 deg (joint 6 = index 5)
    print(f"  Rotating wrist +{math.degrees(WRIST_ROTATION_RAD):.0f} deg...")
    joints = controller.get_joint_states()
    if joints:
        positions = [joints[f'arm_joint{i}'] for i in range(1, 7)]
        new_j6 = positions[5] + WRIST_ROTATION_RAD
        clamped = max(_JOINT6_MIN, min(_JOINT6_MAX, new_j6))
        if clamped != new_j6:
            logger.warning(
                f"Joint 6 hit limit: requested {math.degrees(new_j6):.1f} deg "
                f"-> clamped to {math.degrees(clamped):.1f} deg"
            )
        positions[5] = clamped
        controller.move_to_position_smooth(
            positions, steps=30, rate_hz=20.0,
            interpolation_type="cosine", wait_for_convergence=True,
        )
    else:
        logger.warning("Could not read joints for wrist rotation — skipping")

    # 3. Open gripper to release object at new position
    print("  Opening gripper (release)...")
    controller.open_gripper()
    time.sleep(1.0)

    # 4. Retract upward 5 cm to clear the object before returning
    print("  Retracting 5 cm upward...")
    controller.move_ee_relative(0.0, 0.0, 0.05, steps=30, rate_hz=20.0)

    # 5. Return to observation pose
    move_to_joint_pose(controller, observation_pose, "observation", dry_run=False)
    print("  Relocate cycle complete")


def run_episode(
    controller,
    robot: A1XRecordingRobot,
    recorder: DemoRecorder,
    grasp_cfg: dict,
    episode_id: int,
    args,
    *,
    cached_detector=None,
    skip_confirmation: bool = False,
) -> bool:
    """Run one grasp episode with recording.

    Recording captures only the observation → grasp → lift sequence.
    The subsequent relocation step is NOT recorded — it is purely a
    logistics operation to reposition the object for the next episode.

    Returns True if the episode completed successfully.
    """
    observation_pose = grasp_cfg["observation_pose"]
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
    recording_stopped = False

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
            detector=cached_detector,
        )

        if detection is None:
            print("\n  No object detected — returning to observation pose.")
            move_to_joint_pose(controller, observation_pose, "observation", args.dry_run)
            recorder.stop_episode(episode_id=episode_id)
            recording_stopped = True
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
            motion_cfg, safety_cfg,
            dry_run=args.dry_run,
            skip_confirmation=skip_confirmation,
        )

        # ── Stop recording (observation → grasp → lift only) ──────
        recorder.stop_episode(episode_id=episode_id)
        recording_stopped = True

        if not success:
            print("\n  Grasp execution failed or was aborted.")
            if not args.dry_run:
                controller.open_gripper()
                time.sleep(0.5)
            move_to_joint_pose(controller, observation_pose, "observation", args.dry_run)
            return False

        # ── Step 8: Relocate object & return (NOT recorded) ───────
        step_8_relocate_and_return(
            controller, episode_id, observation_pose, dry_run=args.dry_run,
        )

        print(f"\n  Episode {episode_id} complete")
        return True

    except Exception as e:
        logger.error(f"Episode {episode_id} failed with error: {e}")
        # Best-effort recovery: return arm to observation pose
        try:
            if not args.dry_run:
                controller.open_gripper()
            move_to_joint_pose(controller, observation_pose, "observation", args.dry_run)
        except Exception:
            logger.error("Failed to return to observation pose during error recovery")
        return False

    finally:
        # Safety net: ensure recording is stopped even on unexpected errors
        if not recording_stopped:
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
    print("  YOLOe Grasp — Automated Demonstration Recorder")
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
    print(f"  Relocation pattern: RIGHT → DOWN → LEFT → UP (5 cm + 10° wrist)")

    # ── Step 1: Initialize A1X ─────────────────────────────────────
    controller = step_1_initialize(grasp_cfg, dry_run=args.dry_run)

    # ── Pre-load detector model (loaded ONCE, reused every episode) ─
    yoloe_cfg = grasp_cfg["yoloe"]
    cached_detector = None
    if not args.dry_run:
        print("\n" + "=" * 60)
        print("  PRE-LOADING DETECTOR MODEL")
        print("=" * 60)
        if args.detector == "sam3":
            from examples.yoloe_grasp.grasp_pipeline.sam3_detector import Sam3Detector
            cached_detector = Sam3Detector(device=yoloe_cfg.get("device", "cuda:0"))
        else:
            from examples.yoloe_grasp.grasp_pipeline.yoloe_detector import YOLOeDetector
            checkpoint = str(PROJECT_ROOT / yoloe_cfg["checkpoint"])
            cached_detector = YOLOeDetector(checkpoint, device=yoloe_cfg.get("device", "cuda:0"))
        print(f"  Detector loaded: {type(cached_detector).__name__}")

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

    # ── Record episodes (fully automated) ──────────────────────────
    successful = 0
    consecutive_failures = 0
    try:
        for i in range(num_episodes):
            ep_id = start_episode + i

            # Brief settle between episodes (no manual input needed)
            if i > 0:
                time.sleep(1.0)

            ok = run_episode(
                controller, robot, recorder, grasp_cfg, ep_id, args,
                cached_detector=cached_detector,
                skip_confirmation=True,
            )
            if ok:
                successful += 1
                consecutive_failures = 0
                print(f"  Saved: {save_path}{task_name}/{ep_id}.hdf5")
            else:
                consecutive_failures += 1
                logger.warning(
                    f"Episode {ep_id} failed "
                    f"({consecutive_failures}/{MAX_CONSECUTIVE_FAILURES} consecutive)"
                )
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    logger.error(
                        f"{MAX_CONSECUTIVE_FAILURES} consecutive failures — "
                        f"object may have left workspace. Stopping."
                    )
                    break
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
