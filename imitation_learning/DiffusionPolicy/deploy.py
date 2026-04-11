#!/usr/bin/env python3
"""
Diffusion Policy Real-World Deployment for A1X Robot
=====================================================

Loads a LeRobot-trained Diffusion Policy checkpoint, reads observations from
the wrist-mounted RealSense D405 camera and ROS 2 joint states, runs policy
inference on GPU, and sends the predicted joint commands to the robot at a
fixed control frequency.

Key difference from ACT deployment:
  - Diffusion Policy uses DDIM for fast inference (10 denoising steps)
  - Action chunking: generates n_action_steps=8 actions per inference call,
    then executes them open-loop before re-observing
  - Camera runs continuously; image is captured fresh at each inference call

Usage:
    python imitation_learning/DiffusionPolicy/deploy.py
    python imitation_learning/DiffusionPolicy/deploy.py --checkpoint outputs/train/.../checkpoints/10000/pretrained_model
    python imitation_learning/DiffusionPolicy/deploy.py --episodes 3 --max-steps 200 --freq 10

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
logger = logging.getLogger("dp_deploy")


# ===========================================================================
# Constants
# ===========================================================================

# Default checkpoint (update this after each training run)
DEFAULT_CHECKPOINT = os.path.join(
    _PROJECT_ROOT,
    "outputs", "train",
    "2026-04-11_00-31-07_dp_a1x_yoloe_grasp_white_object",
    "checkpoints", "last", "pretrained_model",
)

# A1X joint safety limits (radians) -- from URDF via a1x_control.py
JOINT_LIMITS = [
    (-2.8798,  2.8798),  # joint1: +/-165 deg
    ( 0.0,     3.1416),  # joint2: 0-180 deg
    (-3.3161,  0.0),     # joint3: -190 deg - 0
    (-1.5708,  1.5708),  # joint4: +/-90 deg
    (-1.5708,  1.5708),  # joint5: +/-90 deg
    (-2.8798,  2.8798),  # joint6: +/-165 deg
]

# Home position -- a safe neutral pose to return to between episodes
HOME_POSITION = [0.0, 1.0, -0.93, 0.83, 0.0, 0.0]  # observation space


# ===========================================================================
# Helpers
# ===========================================================================

def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(value, hi))


def load_policy(checkpoint_dir: str):
    """Load Diffusion Policy + pre/post-processors from a LeRobot checkpoint.

    The checkpoint was saved by our train.py using ``save_checkpoint()``,
    which writes ``config.json``, ``model.safetensors``, and the processor
    pipeline configs + safetensors.  This is compatible with the standard
    ``DiffusionPolicy.from_pretrained()`` API.

    Returns:
        (policy, preprocess, postprocess, device)
    """
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    from lerobot.policies.factory import make_pre_post_processors

    logger.info("Loading Diffusion Policy from %s ...", checkpoint_dir)
    policy = DiffusionPolicy.from_pretrained(checkpoint_dir)

    device = torch.device(policy.config.device)
    policy.to(device)
    policy.eval()

    logger.info(
        "Policy loaded -- device=%s, n_obs_steps=%d, horizon=%d, "
        "n_action_steps=%d, num_inference_steps=%d (DDIM)",
        device,
        policy.config.n_obs_steps,
        policy.config.horizon,
        policy.config.n_action_steps,
        policy.config.num_inference_steps,
    )

    # Load pre/post-processors from the checkpoint directory
    # These contain the normalization stats (MIN_MAX for state/action,
    # MEAN_STD/ImageNet for images) fitted during training.
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
    import a1x_control
    a1x_control.initialize(enable_gripper=True)
    controller = a1x_control.JointController()

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
        observation.state            -> shape (7,) float32 -- 6 joints + gripper
        observation.images.cam_wrist -> shape (480, 640, 3) uint8 -- RGB HWC

    The gripper value is normalized to 0-1 (matching data collection format).
    """
    # --- Joint state (6 floats, radians) ---
    joints_dict = controller.get_joint_states()
    if joints_dict is not None:
        joints = [joints_dict.get(f"arm_joint{i}", 0.0) for i in range(1, 7)]
    else:
        logger.warning("Joint states not available, using zeros")
        joints = [0.0] * 6

    # --- Gripper state (0-100 -> normalize to 0-1 to match training data) ---
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


# ===========================================================================
# Main deployment loop
# ===========================================================================

def run_deployment(
    checkpoint_dir: str,
    num_episodes: int = 5,
    max_steps: int = 300,
    control_freq: int = 10,
    go_home: bool = True,
    override_action_steps: int | None = None,
) -> None:
    """Run the full Diffusion Policy deployment pipeline.

    The core loop:
      1. Capture observation (camera image + joint state)
      2. Prepare observation -> tensors on GPU
      3. Normalize via preprocessor
      4. Call policy.select_action() -- this internally manages the action
         chunk queue: runs the expensive DDIM denoising (10 steps) once every
         n_action_steps calls, then pops cached actions in between.
      5. Unnormalize via postprocessor
      6. Execute action on robot with safety clamping
      7. Rate-control to maintain control_freq Hz

    Args:
        checkpoint_dir: Path to pretrained_model/ directory.
        num_episodes: Number of episodes to run.
        max_steps: Maximum control steps per episode (safety limit).
        control_freq: Control loop frequency in Hz.
        go_home: Whether to move to home position before each episode.
        override_action_steps: Override n_action_steps at inference time.
    """
    from lerobot.policies.utils import prepare_observation_for_inference

    # 1. Load policy
    policy, preprocess, postprocess, device = load_policy(checkpoint_dir)

    # Override n_action_steps if requested
    if override_action_steps is not None:
        old = policy.config.n_action_steps
        horizon = policy.config.horizon
        n_obs = policy.config.n_obs_steps
        max_allowed = horizon - n_obs + 1
        if override_action_steps > max_allowed:
            raise ValueError(
                f"--action-steps ({override_action_steps}) must be <= "
                f"horizon - n_obs_steps + 1 = {max_allowed}"
            )
        policy.config.n_action_steps = override_action_steps
        policy.reset()  # rebuild queues with new maxlen
        logger.info(
            "Overriding n_action_steps: %d -> %d (horizon=%d)",
            old, override_action_steps, horizon,
        )

    # 2. Initialize robot + camera
    controller, camera = init_robot()

    dt = 1.0 / control_freq
    n_action_steps = policy.config.n_action_steps

    logger.info(
        "Deployment ready -- %d episodes, %d max steps, %d Hz control, "
        "%d action steps per chunk",
        num_episodes, max_steps, control_freq, n_action_steps,
    )

    results = []

    for episode in range(num_episodes):
        # -- Episode setup --
        policy.reset()  # flush action chunk queue + observation history

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
        inference_times = []

        # -- Control loop --
        try:
            while step < max_steps:
                loop_start = time.time()

                # 1. Observe (raw numpy arrays)
                obs = get_observation(controller, camera)

                # 2. Prepare: numpy -> float32 CHW tensors with batch dim on device
                #    - uint8 HWC images -> float32 CHW [0,1]
                #    - adds batch dimension
                #    - moves to GPU
                obs_prepared = prepare_observation_for_inference(
                    copy(obs), device,
                )

                # 3. Preprocess (normalize: MEAN_STD for images, MIN_MAX for state)
                batch = preprocess(obs_prepared)

                # 4. Infer via select_action()
                #    Internally manages the action chunk queue:
                #    - Every n_action_steps calls: runs DDIM denoising (10 steps)
                #      to generate a full chunk of actions
                #    - Other calls: pops from the cached queue (near-instant)
                t_infer = time.time()
                action_normalized = policy.select_action(batch)
                infer_dt = time.time() - t_infer

                # Track inference time (only meaningful when DDIM actually runs)
                inference_times.append(infer_dt)

                # 5. Postprocess (unnormalize action -> original scale, move to CPU)
                action = postprocess(action_normalized)
                action_np = action.squeeze(0).cpu().numpy()  # shape (7,)

                # 6. Execute with safety clamping
                execute_action(controller, action_np)

                # 7. Log progress
                if step % 10 == 0:
                    joints_str = ", ".join(f"{v:+.3f}" for v in action_np[:6])
                    is_infer = (step % n_action_steps == 0)
                    logger.info(
                        "  step %3d/%d | joints=[%s] | gripper=%.3f | "
                        "infer=%.1fms%s",
                        step, max_steps, joints_str, action_np[6],
                        infer_dt * 1000,
                        " [DDIM]" if is_infer else " [cached]",
                    )

                step += 1

                # 8. Rate control -- sleep only if we're ahead of schedule
                elapsed = time.time() - loop_start
                sleep_time = dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            logger.info(
                "Episode %d interrupted by user at step %d", episode + 1, step
            )

        episode_duration = time.time() - episode_start

        # Compute inference statistics
        if inference_times:
            avg_infer = np.mean(inference_times) * 1000
            max_infer = np.max(inference_times) * 1000
            # Full DDIM calls happen every n_action_steps
            ddim_times = inference_times[::n_action_steps]
            avg_ddim = np.mean(ddim_times) * 1000 if ddim_times else 0
        else:
            avg_infer = max_infer = avg_ddim = 0

        results.append({
            "episode": episode + 1,
            "steps": step,
            "duration": episode_duration,
            "avg_infer_ms": avg_infer,
            "avg_ddim_ms": avg_ddim,
        })

        logger.info(
            "Episode %d finished -- %d steps in %.1f s (%.1f Hz effective) | "
            "avg DDIM=%.1fms, avg overall=%.1fms",
            episode + 1, step, episode_duration,
            step / episode_duration if episode_duration > 0 else 0,
            avg_ddim, avg_infer,
        )

    # -- Summary --
    print("\n" + "=" * 60)
    print("Deployment Summary")
    print("=" * 60)
    for r in results:
        print(
            f"  Episode {r['episode']}: {r['steps']} steps in "
            f"{r['duration']:.1f}s | DDIM={r['avg_ddim_ms']:.1f}ms"
        )
    print("=" * 60)

    # Cleanup
    camera.stop()
    logger.info("Camera stopped. Deployment complete.")


# ===========================================================================
# CLI
# ===========================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Deploy a trained Diffusion Policy on the A1X robot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
    # Run with defaults (latest checkpoint, 5 episodes)
    python imitation_learning/DiffusionPolicy/deploy.py

    # Specify checkpoint explicitly
    python imitation_learning/DiffusionPolicy/deploy.py \\
        --checkpoint outputs/train/.../checkpoints/10000/pretrained_model

    # Quick test: 1 episode, 50 steps
    python imitation_learning/DiffusionPolicy/deploy.py --episodes 1 --max-steps 50

    # Re-observe more frequently (every 4 steps instead of 8)
    python imitation_learning/DiffusionPolicy/deploy.py --action-steps 4
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
            "Override n_action_steps at inference time. Lower values mean the "
            "robot re-observes more frequently (smoother but slower). Must be "
            "<= horizon - n_obs_steps + 1. Default: use checkpoint value (8)."
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
        override_action_steps=args.action_steps,
    )
