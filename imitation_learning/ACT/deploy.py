#!/usr/bin/env python3
"""
ACT Policy Real-World Deployment for A1X Robot

Loads a LeRobot-trained ACT checkpoint, reads observations from the
wrist-mounted RealSense D405 camera and ROS 2 joint states, runs policy
inference on GPU, and sends the predicted joint commands to the robot at
a fixed control frequency.

Usage:
    python imitation_learning/ACT/deploy.py
    python imitation_learning/ACT/deploy.py --checkpoint outputs/train/.../checkpoints/last/pretrained_model
    python imitation_learning/ACT/deploy.py --episodes 3 --max-steps 200 --freq 10

Prerequisites:
    1. CAN bus configured:
       sudo ip link set can0 type can bitrate 1000000 ...
       sudo ip link set up can0
    2. Conda environment: conda activate a1x_ros
    3. LeRobot installed: pip install -e refence_code/lerobot
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from copy import copy
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Project root on sys.path so ``import a1x_control`` works from any cwd
# ---------------------------------------------------------------------------
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("act_deploy")


# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

# Default checkpoint (latest training run)
DEFAULT_CHECKPOINT = os.path.join(
    _PROJECT_ROOT,
    "outputs", "train",
    "2026-04-08_20-09-41_act_a1x_yoloe_grasp_white_object",
    "checkpoints", "last", "pretrained_model",
)

# A1X joint safety limits (radians) — from URDF via a1x_control.py
JOINT_LIMITS = [
    (-2.8798,  2.8798),  # joint1: ±165°
    ( 0.0,     3.1416),  # joint2: 0–180°
    (-3.3161,  0.0),     # joint3: -190°–0°
    (-1.5708,  1.5708),  # joint4: ±90°
    (-1.5708,  1.5708),  # joint5: ±90°
    (-2.8798,  2.8798),  # joint6: ±165°
]

# Home position — a safe neutral pose to return to between episodes
# HOME_POSITION = [0.0, 0.0043, -0.1, -0.0347, -0.0055, 0.0013]
HOME_POSITION = [0.0, 1.0, -0.93, 0.83, 0.0, 0.0] # observation


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(value, hi))


def load_policy(
    checkpoint_dir: str,
    action_steps: int | None = None,
    temporal_ensemble_coeff: float | None = None,
):
    """Load ACT policy + pre/post-processors from a LeRobot checkpoint.

    Args:
        checkpoint_dir: Path to the pretrained_model/ directory.
        action_steps: Override n_action_steps at inference time (must be
            ≤ chunk_size). Reduces how many steps are executed open-loop
            before re-observing. None = use the checkpoint's value.
        temporal_ensemble_coeff: If set, enable temporal ensembling — blend
            overlapping chunk predictions with exponential weights. Typical
            value: 0.01. Eliminates chunk-boundary discontinuities. None = off.

    Returns:
        (policy, preprocess, postprocess, device)
    """
    from lerobot.policies.act.modeling_act import ACTPolicy, ACTTemporalEnsembler
    from lerobot.policies.factory import make_pre_post_processors

    logger.info("Loading ACT policy from %s ...", checkpoint_dir)
    policy = ACTPolicy.from_pretrained(checkpoint_dir)

    device = torch.device(policy.config.device)
    chunk_size = policy.config.chunk_size

    # ── Override n_action_steps (inference-only, no retraining needed) ─
    if action_steps is not None:
        if action_steps > chunk_size:
            raise ValueError(
                f"--action-steps ({action_steps}) must be ≤ chunk_size ({chunk_size})"
            )
        old = policy.config.n_action_steps
        policy.config.n_action_steps = action_steps
        logger.info(
            "Overriding n_action_steps: %d → %d (chunk_size=%d)",
            old, action_steps, chunk_size,
        )

    # ── Enable temporal ensembling (inference-only, no retraining needed) ─
    if temporal_ensemble_coeff is not None:
        policy.config.temporal_ensemble_coeff = temporal_ensemble_coeff
        policy.temporal_ensembler = ACTTemporalEnsembler(
            temporal_ensemble_coeff, chunk_size,
        )
        logger.info(
            "Temporal ensembling enabled (coeff=%.4f, chunk_size=%d)",
            temporal_ensemble_coeff, chunk_size,
        )

    # Re-initialize internal state with the updated config
    policy.reset()

    logger.info(
        "Policy loaded — device=%s, chunk_size=%d, n_action_steps=%d, "
        "temporal_ensemble=%s",
        device, chunk_size, policy.config.n_action_steps,
        temporal_ensemble_coeff or "off",
    )

    preprocess, postprocess = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=checkpoint_dir,
    )
    logger.info("Pre/post-processors loaded (normalizer + unnormalizer)")

    return policy, preprocess, postprocess, device


def init_robot():
    """Initialize the A1X robot (ROS 2 driver + gripper) and D405 camera.

    Returns:
        (controller, camera)
    """
    # Importing a1x_control auto-launches HDAS driver + mobiman controller
    import a1x_control
    a1x_control.initialize(enable_gripper=True)
    controller = a1x_control.JointController()

    # Start D405 camera
    sys.path.insert(0, _PROJECT_ROOT)
    from imitation_learning.data_collection.d405_sensor import D405Sensor

    camera = D405Sensor("cam_wrist")
    camera.set_up(width=640, height=480, fps=30, is_depth=False)

    # Give the camera pipeline a moment to start producing frames
    time.sleep(1.0)
    logger.info("Robot and camera initialized")

    return controller, camera


def get_observation(controller, camera) -> dict[str, np.ndarray]:
    """Read current robot state + camera image and assemble an observation dict.

    Keys match the ``input_features`` in the LeRobot checkpoint's config.json:
        observation.state            → shape (7,) float32 — 6 joints + gripper
        observation.images.cam_wrist → shape (480, 640, 3) uint8 — RGB HWC

    The gripper value is normalized to 0-1 (matching data collection format).

    Returns raw numpy arrays. Call ``prepare_observation_for_inference`` before
    passing to the preprocess pipeline.
    """
    # --- Joint state (6 floats, radians) ---
    joints_dict = controller.get_joint_states()
    if joints_dict is not None:
        joints = [joints_dict.get(f"arm_joint{i}", 0.0) for i in range(1, 7)]
    else:
        logger.warning("Joint states not available, using zeros")
        joints = [0.0] * 6

    # --- Gripper state (0-100 → normalize to 0-1 to match training data) ---
    gripper_raw = controller.get_gripper_state()
    gripper = (gripper_raw / 100.0) if gripper_raw is not None else 0.0

    state = np.array(joints + [gripper], dtype=np.float32)

    # --- Camera image ---
    img_data = camera.get_image()
    cam_wrist = img_data["color"]  # (480, 640, 3) RGB uint8

    return {
        "observation.state": state,
        "observation.images.cam_wrist": cam_wrist,
    }


def execute_action(controller, action_7d: np.ndarray) -> None:
    """Send a 7-dim action to the robot with safety clamping.

    Action format: [joint1, joint2, ..., joint6, gripper]
    - Joints are in radians, clamped to JOINT_LIMITS
    - Gripper is in 0-1 (from model), denormalized to 0-100 for the robot
    """
    joint_cmd = [
        clamp(float(action_7d[i]), JOINT_LIMITS[i][0], JOINT_LIMITS[i][1])
        for i in range(6)
    ]
    # Denormalize gripper: model outputs 0-1, robot expects 0-100
    gripper_cmd = float(np.clip(action_7d[6], 0.0, 1.0)) * 100.0

    controller.set_joint_positions(joint_cmd)
    controller.set_gripper_position(gripper_cmd)


def move_to_home(controller, steps: int = 20, rate_hz: int = 10) -> None:
    """Smoothly move the arm to the home position."""
    logger.info("Moving to home position...")
    controller.move_to_position_smooth(HOME_POSITION, steps=steps, rate_hz=rate_hz)
    controller.set_gripper_position(100.0)  # open gripper
    time.sleep(0.5)


# ═══════════════════════════════════════════════════════════════════════════
# Main deployment loop
# ═══════════════════════════════════════════════════════════════════════════

def run_deployment(
    checkpoint_dir: str,
    num_episodes: int = 5,
    max_steps: int = 300,
    control_freq: int = 10,
    go_home: bool = True,
    action_steps: int | None = None,
    temporal_ensemble_coeff: float | None = None,
) -> None:
    """Run the full ACT deployment pipeline.

    Args:
        checkpoint_dir: Path to pretrained_model/ directory.
        num_episodes: Number of episodes to run.
        max_steps: Maximum control steps per episode (safety limit).
        control_freq: Control loop frequency in Hz.
        go_home: Whether to move to home position before each episode.
        action_steps: Override n_action_steps (None = use checkpoint value).
        temporal_ensemble_coeff: Enable temporal ensembling (None = off).
    """
    from lerobot.policies.utils import prepare_observation_for_inference

    # 1. Load policy
    policy, preprocess, postprocess, device = load_policy(
        checkpoint_dir,
        action_steps=action_steps,
        temporal_ensemble_coeff=temporal_ensemble_coeff,
    )

    # 2. Initialize robot + camera
    controller, camera = init_robot()

    dt = 1.0 / control_freq
    logger.info(
        "Deployment ready — %d episodes, %d max steps, %d Hz control",
        num_episodes, max_steps, control_freq,
    )

    results = []

    for episode in range(num_episodes):
        # ── Episode setup ─────────────────────────────────────────────
        policy.reset()  # flush action chunk queue

        if go_home:
            move_to_home(controller)

        input(
            f"\n{'='*60}\n"
            f"[Episode {episode + 1}/{num_episodes}] "
            f"Press ENTER to start (ensure scene is ready)...\n"
            f"{'='*60}\n"
        )
        logger.info("Episode %d started", episode + 1)

        step = 0
        episode_start = time.time()

        # ── Control loop ──────────────────────────────────────────────
        try:
            while step < max_steps:
                loop_start = time.time()

                # 1. Observe (raw numpy arrays)
                obs = get_observation(controller, camera)

                # 2. Prepare: numpy → float32 CHW tensors with batch dim on device
                #    This converts uint8 HWC images to float32 CHW [0,1] and adds
                #    batch dims. Must run BEFORE the preprocess pipeline.
                obs_prepared = prepare_observation_for_inference(
                    copy(obs), device,
                )

                # 3. Preprocess (MEAN_STD normalization)
                batch = preprocess(obs_prepared)

                # 4. Infer
                #    select_action manages the chunk queue internally:
                #    - Runs the model every n_action_steps (50) calls
                #    - Pops one action from the queue per call
                action_normalized = policy.select_action(batch)

                # 5. Postprocess (denormalize → CPU tensor)
                action = postprocess(action_normalized)
                action_np = action.squeeze(0).cpu().numpy()  # shape (7,)

                # 6. Execute with safety clamping
                execute_action(controller, action_np)

                # 7. Log progress
                if step % 10 == 0:
                    joints_str = ", ".join(f"{v:+.3f}" for v in action_np[:6])
                    logger.info(
                        "  step %3d/%d | joints=[%s] | gripper=%.3f",
                        step, max_steps, joints_str, action_np[6],
                    )

                step += 1

                # 8. Rate control — sleep only if we're ahead of schedule
                elapsed = time.time() - loop_start
                sleep_time = dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            logger.info("Episode %d interrupted by user at step %d", episode + 1, step)

        episode_duration = time.time() - episode_start
        results.append({"episode": episode + 1, "steps": step, "duration": episode_duration})
        logger.info(
            "Episode %d finished — %d steps in %.1f s (%.1f Hz effective)",
            episode + 1, step, episode_duration,
            step / episode_duration if episode_duration > 0 else 0,
        )

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Deployment Summary")
    print("=" * 60)
    for r in results:
        print(f"  Episode {r['episode']}: {r['steps']} steps in {r['duration']:.1f}s")
    print("=" * 60)

    # Cleanup
    camera.stop()
    logger.info("Camera stopped. Deployment complete.")


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Deploy a trained ACT policy on the A1X robot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
    # Run with defaults (latest checkpoint, 5 episodes)
    python imitation_learning/ACT/deploy.py

    # Specify checkpoint explicitly
    python imitation_learning/ACT/deploy.py \\
        --checkpoint outputs/train/.../checkpoints/last/pretrained_model

    # Quick test: 1 episode, 50 steps (single action chunk)
    python imitation_learning/ACT/deploy.py --episodes 1 --max-steps 50

    # Reduce open-loop steps to fix chunk-boundary shaking
    python imitation_learning/ACT/deploy.py --action-steps 8

    # Enable temporal ensembling (smooths chunk transitions)
    python imitation_learning/ACT/deploy.py --temporal-ensemble 0.01

    # Combine both for maximum smoothness
    python imitation_learning/ACT/deploy.py --action-steps 8 --temporal-ensemble 0.01
""",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=DEFAULT_CHECKPOINT,
        help="Path to the pretrained_model/ directory (default: latest training run)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of episodes to run (default: 5)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=300,
        help="Maximum steps per episode (default: 300)",
    )
    parser.add_argument(
        "--freq",
        type=int,
        default=10,
        help="Control loop frequency in Hz (default: 10)",
    )
    parser.add_argument(
        "--no-home",
        action="store_true",
        help="Skip moving to home position before each episode",
    )
    parser.add_argument(
        "--action-steps",
        type=int,
        default=None,
        help=(
            "Override n_action_steps at inference time (must be ≤ chunk_size). "
            "Reduces open-loop execution before re-observing. "
            "Example: --action-steps 8 re-queries the model every 8 steps."
        ),
    )
    parser.add_argument(
        "--temporal-ensemble",
        type=float,
        default=None,
        metavar="COEFF",
        help=(
            "Enable temporal ensembling to blend overlapping chunk predictions. "
            "Eliminates chunk-boundary discontinuities. Typical value: 0.01. "
            "Cannot be combined with --action-steps."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Validate checkpoint path
    if not os.path.isdir(args.checkpoint):
        logger.error("Checkpoint directory not found: %s", args.checkpoint)
        sys.exit(1)

    config_path = os.path.join(args.checkpoint, "config.json")
    if not os.path.isfile(config_path):
        logger.error("config.json not found in checkpoint: %s", args.checkpoint)
        sys.exit(1)

    run_deployment(
        checkpoint_dir=args.checkpoint,
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        control_freq=args.freq,
        go_home=not args.no_home,
        action_steps=args.action_steps,
        temporal_ensemble_coeff=args.temporal_ensemble,
    )
