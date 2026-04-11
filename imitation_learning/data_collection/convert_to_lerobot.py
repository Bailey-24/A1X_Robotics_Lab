#!/usr/bin/env python3
"""Convert raw A1X demonstration HDF5 files to LeRobot dataset format.

Action definition (RoboTwin-style):
    action[t] = state[t+1]   (the next observed joint configuration)
    This means each episode of N raw frames produces N-1 training frames,
    where the last raw frame only serves as the action target for frame N-2.

    The raw ``a1x_arm/action`` channel (IK-commanded targets) is **ignored**
    in favour of this shift-by-one approach, which avoids latency and
    tracking-error artefacts from the command topics.

Default pipeline (direct):
    Raw HDF5 (from CollectAny) -> LeRobot Dataset

Optional two-stage pipeline (--via-act):
    Stage 1: Raw HDF5 (from CollectAny) -> ACT-format HDF5
    Stage 2: ACT-format HDF5 -> LeRobot Dataset

Raw HDF5 structure (from recording):
    /a1x_arm/joint      [N, 6]   float64
    /a1x_arm/gripper    [N, 1]   float64
    /a1x_arm/action     [N, 7]   float64   (ignored — see above)
    /a1x_arm/timestamp  [N]      int64
    /cam_wrist/color    [N, H, W, 3]  uint8
    /cam_wrist/timestamp [N]     int64

ACT-format HDF5 (only with --via-act or --act-only):
    /observations/qpos                   [N-1, 7]  float32
    /observations/images/cam_wrist       [N-1, H, W, 3]  uint8
    /action                              [N-1, 7]  float32

LeRobot Dataset:
    observation.state           [7]   float32
    observation.images.cam_wrist  image
    action                      [7]   float32

Usage:
    # Direct: raw HDF5 -> LeRobot (default, no intermediate files)
    python imitation_learning/data_collection/convert_to_lerobot.py \\
        --data-dir ./data/demos/yoloe_grasp/ \\
        --repo-id a1x/yoloe_grasp_demos

    # Two-stage: raw -> ACT HDF5 -> LeRobot
    python imitation_learning/data_collection/convert_to_lerobot.py \\
        --data-dir ./data/demos/yoloe_grasp/ \\
        --repo-id a1x/yoloe_grasp_demos --via-act

    # Stage 1 only (raw -> ACT HDF5)
    python imitation_learning/data_collection/convert_to_lerobot.py \\
        --data-dir ./data/demos/yoloe_grasp/ \\
        --act-only --act-output ./data/act_hdf5/
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np
import tqdm

# ── Path setup ─────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_CYR_SRC = str(PROJECT_ROOT / "refence_code" / "control_your_robot" / "src")
if _CYR_SRC not in sys.path:
    sys.path.insert(0, _CYR_SRC)

from robot.utils.base.data_handler import hdf5_groups_to_dict, get_item


# ═══════════════════════════════════════════════════════════════════════
# Stage 1: Raw HDF5 → ACT-format HDF5
# ═══════════════════════════════════════════════════════════════════════

# Mapping from raw HDF5 keys to fields we need.
# Note: ``a1x_arm.action`` (IK command targets) is intentionally omitted —
# actions are derived as ``qpos[t+1]`` (RoboTwin-style shift-by-one).
SINGLE_ARM_MAP = {
    "cam_wrist": "cam_wrist.color",
    "qpos": ["a1x_arm.joint", "a1x_arm.gripper"],
}


def convert_raw_to_act(raw_path: str, output_path: str) -> None:
    """Convert a single raw HDF5 episode to ACT format.

    Actions are derived as ``action[t] = qpos[t+1]`` (RoboTwin-style).
    The episode length shrinks from N to N-1: the last raw frame is used
    only as the action target for the penultimate frame.
    """
    data = hdf5_groups_to_dict(raw_path)

    with h5py.File(output_path, "w") as f:
        # Extract data using mapping
        input_data = {}
        for key, src in SINGLE_ARM_MAP.items():
            input_data[key] = get_item(data, src)

        qpos_full = np.array(input_data["qpos"]).astype(np.float32)  # [N, 7]

        # ── RoboTwin-style shift-by-one ──────────────────────────
        # state[t] = qpos[t]     for t = 0 .. N-2
        # action[t] = qpos[t+1]  for t = 0 .. N-2
        qpos = qpos_full[:-1]   # [N-1, 7]
        action = qpos_full[1:]  # [N-1, 7]

        f.create_dataset("action", data=action, dtype="float32")

        obs = f.create_group("observations")
        obs.create_dataset("qpos", data=qpos, dtype="float32")

        images = obs.create_group("images")

        # Camera images — trim to N-1 (drop last frame)
        cam_data = input_data["cam_wrist"]
        if isinstance(cam_data, np.ndarray) and cam_data.ndim == 4:
            # Raw numpy images [N, H, W, 3] -> [N-1, H, W, 3]
            images.create_dataset("cam_wrist", data=cam_data[:-1], dtype=np.uint8)
        else:
            # JPEG-encoded — decode and drop last frame
            imgs = []
            for frame in cam_data:
                if isinstance(frame, (bytes, bytearray)):
                    frame = np.frombuffer(frame, dtype=np.uint8)
                img = cv2.imdecode(frame, cv2.IMREAD_COLOR)
                if img is not None:
                    imgs.append(img)
            images.create_dataset(
                "cam_wrist", data=np.stack(imgs[:-1], axis=0), dtype=np.uint8
            )


def batch_convert_raw_to_act(data_dir: str, output_dir: str) -> list[str]:
    """Convert all raw HDF5 files in a directory to ACT format.

    Returns list of output file paths.
    """
    os.makedirs(output_dir, exist_ok=True)

    raw_files = sorted(glob.glob(os.path.join(data_dir, "*.hdf5")))
    if not raw_files:
        print(f"No HDF5 files found in {data_dir}")
        return []

    print(f"Stage 1: Converting {len(raw_files)} raw episodes -> ACT format")
    output_files = []
    for i, raw_path in enumerate(tqdm.tqdm(raw_files, desc="Raw->ACT")):
        out_path = os.path.join(output_dir, f"episode_{i}.hdf5")
        convert_raw_to_act(raw_path, out_path)
        output_files.append(out_path)

    return output_files


# ═══════════════════════════════════════════════════════════════════════
# Stage 2: ACT HDF5 → LeRobot Dataset
# ═══════════════════════════════════════════════════════════════════════

MOTORS = [
    "arm_joint1", "arm_joint2", "arm_joint3",
    "arm_joint4", "arm_joint5", "arm_joint6",
    "gripper",
]
CAMERAS = ["cam_wrist"]


def _import_lerobot():
    """Import LeRobotDataset across version differences.

    LeRobot >= 0.4 reorganized: lerobot.common.datasets -> lerobot.datasets.
    Try the new path first, then fall back to the legacy path.
    """
    try:
        from lerobot.datasets.lerobot_dataset import (
            HF_LEROBOT_HOME,
            LeRobotDataset,
        )
    except ImportError:
        from lerobot.common.datasets.lerobot_dataset import (  # type: ignore
            HF_LEROBOT_HOME,
            LeRobotDataset,
        )
    return HF_LEROBOT_HOME, LeRobotDataset


def create_lerobot_dataset(repo_id: str, fps: int = 20):
    """Create an empty LeRobot dataset with A1X features."""
    HF_LEROBOT_HOME, LeRobotDataset = _import_lerobot()

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(MOTORS),),
            "names": MOTORS,
        },
        "action": {
            "dtype": "float32",
            "shape": (len(MOTORS),),
            "names": MOTORS,
        },
    }
    for cam in CAMERAS:
        features[f"observation.images.{cam}"] = {
            "dtype": "image",
            "shape": (3, 480, 640),
            "names": ["channels", "height", "width"],
        }

    # Clean existing dataset if present
    dataset_path = HF_LEROBOT_HOME / repo_id
    if dataset_path.exists():
        shutil.rmtree(dataset_path)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        robot_type="a1x",
        features=features,
        image_writer_threads=10,
        image_writer_processes=5,
    )


def populate_lerobot_dataset(
    dataset,
    act_files: list[str],
    instruction: str = "Pick up the target object and place it to the side",
):
    """Populate a LeRobot dataset from ACT-format HDF5 files."""
    import torch

    for ep_path in tqdm.tqdm(act_files, desc="ACT->LeRobot"):
        with h5py.File(ep_path, "r") as ep:
            state = torch.from_numpy(ep["/observations/qpos"][:])
            action = torch.from_numpy(ep["/action"][:])
            num_frames = state.shape[0]

            # Load images
            imgs = {}
            for cam in CAMERAS:
                key = f"/observations/images/{cam}"
                if key in ep:
                    uncompressed = ep[key].ndim == 4
                    if uncompressed:
                        img_array = ep[key][:]
                    else:
                        img_array = []
                        for data in ep[key]:
                            img_array.append(
                                cv2.cvtColor(cv2.imdecode(data, 1), cv2.COLOR_BGR2RGB)
                            )
                        img_array = np.array(img_array)
                    imgs[cam] = img_array

            for i in range(num_frames):
                frame = {
                    "observation.state": state[i],
                    "action": action[i],
                    "task": instruction,
                }
                for cam, img_array in imgs.items():
                    frame[f"observation.images.{cam}"] = img_array[i]

                dataset.add_frame(frame)

            dataset.save_episode()

    return dataset


# ═══════════════════════════════════════════════════════════════════════
# Direct: Raw HDF5 → LeRobot (no intermediate ACT files)
# ═══════════════════════════════════════════════════════════════════════

def populate_lerobot_from_raw(
    dataset,
    raw_files: list[str],
    instruction: str = "Pick up the target object and place it to the side",
):
    """Populate a LeRobot dataset directly from raw HDF5 files, skipping ACT HDF5.

    Actions are derived as ``action[t] = qpos[t+1]`` (RoboTwin-style).
    Each episode of N raw frames produces N-1 training frames.
    """
    import torch

    for raw_path in tqdm.tqdm(raw_files, desc="Raw->LeRobot"):
        data = hdf5_groups_to_dict(raw_path)

        # Same mapping as convert_raw_to_act(), done in-memory
        input_data = {}
        for key, src in SINGLE_ARM_MAP.items():
            input_data[key] = get_item(data, src)

        qpos_full = torch.from_numpy(np.array(input_data["qpos"]).astype(np.float32))

        # ── RoboTwin-style shift-by-one ──────────────────────────
        # state[t] = qpos[t]     for t = 0 .. N-2
        # action[t] = qpos[t+1]  for t = 0 .. N-2
        state = qpos_full[:-1]   # [N-1, 7]
        action = qpos_full[1:]   # [N-1, 7]

        # Decode camera images
        cam_data = input_data["cam_wrist"]
        if isinstance(cam_data, np.ndarray) and cam_data.ndim == 4:
            img_array = cam_data[:-1]  # drop last frame to match N-1
        else:
            imgs = []
            for frame in cam_data:
                if isinstance(frame, (bytes, bytearray)):
                    frame = np.frombuffer(frame, dtype=np.uint8)
                img = cv2.imdecode(frame, cv2.IMREAD_COLOR)
                if img is not None:
                    imgs.append(img)
            img_array = np.stack(imgs[:-1], axis=0)  # drop last frame

        num_frames = state.shape[0]  # N-1
        for i in range(num_frames):
            frame = {
                "observation.state": state[i],
                "action": action[i],
                "task": instruction,
                "observation.images.cam_wrist": img_array[i],
            }
            dataset.add_frame(frame)

        dataset.save_episode()

    return dataset


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert raw A1X demos to LeRobot format",
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Directory containing raw HDF5 episodes (from record_demo.py)",
    )
    parser.add_argument(
        "--repo-id",
        default="a1x/yoloe_grasp_demos",
        help="LeRobot dataset repo ID",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=20,
        help="Dataset FPS (should match recording save_freq)",
    )
    parser.add_argument(
        "--instruction",
        default="Pick up the target object and place it to the side",
        help="Task instruction string for LeRobot dataset",
    )
    parser.add_argument(
        "--act-only",
        action="store_true",
        help="Only run Stage 1 (raw -> ACT HDF5), skip LeRobot conversion",
    )
    parser.add_argument(
        "--via-act",
        action="store_true",
        help="Use two-stage pipeline (raw -> ACT HDF5 -> LeRobot) instead of direct conversion",
    )
    parser.add_argument(
        "--act-output",
        default=None,
        help="Output directory for ACT HDF5 (default: <data-dir>_act/)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    data_dir = args.data_dir.rstrip("/")
    act_output = args.act_output or f"{data_dir}_act"

    print("=" * 60)
    print("  A1X Demo -> LeRobot Converter")
    print("=" * 60)

    raw_files = sorted(glob.glob(os.path.join(data_dir, "*.hdf5")))
    if not raw_files:
        print(f"No HDF5 files found in {data_dir}")
        return 1

    # ── --act-only: Raw -> ACT HDF5 only ──────────────────────────
    if args.act_only:
        act_files = batch_convert_raw_to_act(data_dir, act_output)
        if not act_files:
            return 1
        print(f"  ACT HDF5 saved to: {act_output}/")
        print("  (--act-only specified, skipping LeRobot conversion)")
        return 0

    # ── --via-act: two-stage Raw -> ACT -> LeRobot ────────────────
    if args.via_act:
        act_files = batch_convert_raw_to_act(data_dir, act_output)
        if not act_files:
            return 1
        print(f"  ACT HDF5 saved to: {act_output}/")

        print(f"\nStage 2: Converting ACT -> LeRobot (repo={args.repo_id})")
        dataset = create_lerobot_dataset(args.repo_id, fps=args.fps)
        dataset = populate_lerobot_dataset(dataset, act_files, instruction=args.instruction)
    else:
        # ── Default: direct Raw -> LeRobot (no ACT files) ────────
        print(f"\nConverting {len(raw_files)} raw episodes -> LeRobot (repo={args.repo_id})")
        dataset = create_lerobot_dataset(args.repo_id, fps=args.fps)
        dataset = populate_lerobot_from_raw(dataset, raw_files, instruction=args.instruction)

    print(f"\n  LeRobot dataset created: {args.repo_id}")
    print(f"  Total episodes: {len(raw_files)}")
    print(f"  FPS: {args.fps}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
