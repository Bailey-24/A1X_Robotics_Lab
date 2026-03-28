#!/usr/bin/env python3
"""
A1X Robot Primitive API — for Code-as-Policies grasping.

Five primitive functions callable by LLM-generated code:
    describe_scene()  — capture and describe the scene with qwen3.5-plus
    detect(name)      — check if an object is visible via SAM3
    pick(name)        — full grasp pipeline for the named object
    place()           — move to fixed place pose and release
    ask_user(q)       — ask a clarifying question, return answer

All robot state (controller, config) is managed as module-level singletons
and initialized lazily on first use.
"""
from __future__ import annotations

import base64
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml

# ── Path setup ───────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve()
PROJECT_ROOT = _HERE.parents[2]        # A1Xsdk/
_SAM3_ROOT = PROJECT_ROOT / "refence_code" / "sam3"

for _p in (str(PROJECT_ROOT), str(_SAM3_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import a1x_control
from openai import OpenAI
from examples.yoloe_grasp.grasp_pipeline.sam3_detector import Sam3Detector
from examples.yoloe_grasp.grasp_pipeline.ik_executor import IKExecutor
from examples.yoloe_grasp.yoloe_grasp import (
    step_2_move_to_observation,
    step_3_capture_rgbd,
    step_5_compute_grasp,
    step_6_transform_to_base,
    move_to_joint_pose,
)

logger = logging.getLogger("robot_api")

# ── Cloud VLM config ─────────────────────────────────────────────────────────
_CLOUD_API_KEY = os.environ.get("A1X_VLM_API_KEY", "")
_CLOUD_BASE_URL = "https://api.chatanywhere.tech/v1"
_CLOUD_MODEL = "qwen3.5-plus"

# ── Module-level singletons ───────────────────────────────────────────────────
_controller = None
_cfg: dict | None = None
_sam3: Sam3Detector | None = None


def _get_config() -> dict:
    global _cfg
    if _cfg is None:
        config_path = PROJECT_ROOT / "examples" / "yoloe_grasp" / "config.yaml"
        with open(config_path) as f:
            _cfg = yaml.safe_load(f)
    return _cfg


def _get_sam3() -> Sam3Detector:
    global _sam3
    if _sam3 is None:
        device = _get_config()["yoloe"].get("device", "cpu")
        logger.info("Loading SAM3 model (once)...")
        _sam3 = Sam3Detector(device=device)
        logger.info("SAM3 model ready.")
    return _sam3


def _get_controller() -> a1x_control.JointController:
    global _controller
    if _controller is None:
        logger.info("Initializing A1X controller...")
        a1x_control.initialize(enable_gripper=True, enable_ee_pose=True)
        _controller = a1x_control.JointController()
        time.sleep(2)
        _controller.wait_for_joint_states(timeout=10)
        logger.info("Controller ready.")
    return _controller


# ── SAM3 prompt retry ─────────────────────────────────────────────────────────

def _generate_prompt_variants(name: str) -> list[str]:
    """Generate progressively simpler prompt variants for SAM3.

    When a color is specified, all fallbacks preserve the color to avoid
    matching wrong-colored objects.

    Examples:
        "yellow note"  → ["yellow note", "yellow object"]
        "yellow object"→ ["yellow object"]
        "cube"         → ["cube", "object"]
    """
    _COLOR_WORDS = {"red", "green", "blue", "yellow", "white", "black",
                    "orange", "purple", "pink", "brown", "gray", "grey"}

    variants = [name]
    words = name.strip().split()
    colors = [w for w in words if w.lower() in _COLOR_WORDS]

    if colors:
        # Color specified — keep color in all fallbacks
        variants.append(f"{colors[0]} object")
    else:
        # No color — safe to broaden to generic "object"
        if len(words) >= 2:
            noun = words[-1]
            variants.append(noun)
        variants.append("object")

    # Deduplicate while preserving order
    seen = set()
    return [v for v in variants if not (v in seen or seen.add(v))]


def _detect_with_retry(sam3, color_image, object_name: str, conf_threshold: float = 0.0):
    """Try SAM3 detection with progressively simpler prompts.

    Returns the SAM3 result tuple (bbox, mask, score, name) or None.
    """
    variants = _generate_prompt_variants(object_name)
    for prompt in variants:
        result = sam3.detect(color_image, [prompt], conf_threshold=conf_threshold)
        if result is not None:
            if prompt != object_name:
                logger.info("Fallback prompt '%s' succeeded (original: '%s')", prompt, object_name)
            return result
        logger.info("SAM3 missed '%s', trying next variant...", prompt)
    logger.warning("All prompt variants failed for '%s': %s", object_name, variants)
    return None


# ── Primitives ────────────────────────────────────────────────────────────────

def describe_scene(
    prompt: str = (
        "List every object visible on the desk. "
        "For each one, give: English name, color, and position (left/center/right/front/back). "
        "Format as a numbered list, one object per line."
    ),
) -> str:
    """Capture the current camera view and describe all objects.

    Uses RealSense D405 + qwen3.5-plus cloud VLM.

    Returns:
        English description of the scene, one object per line.
    """
    import pyrealsense2 as rs

    cfg = _get_config()
    cam = cfg["camera"]
    w, h, fps = cam.get("width", 640), cam.get("height", 480), cam.get("fps", 15)

    pipeline = rs.pipeline()
    rscfg = rs.config()
    rscfg.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
    pipeline.start(rscfg)
    try:
        for _ in range(min(fps * 2, 30)):
            pipeline.wait_for_frames()
        frames = pipeline.wait_for_frames()
        cf = frames.get_color_frame()
        if not cf:
            raise RuntimeError("Camera failed to capture frame")
        image = np.asanyarray(cf.get_data())
    finally:
        pipeline.stop()

    _, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    b64 = base64.b64encode(buf).decode("utf-8")

    client = OpenAI(api_key=_CLOUD_API_KEY, base_url=_CLOUD_BASE_URL)
    response = client.chat.completions.create(
        model=_CLOUD_MODEL,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": prompt},
            ],
        }],
        max_tokens=512,
    )
    description = response.choices[0].message.content
    logger.info("Scene: %s", description)
    return description


def detect(object_name: str) -> Optional[dict]:
    """Check if the named object is visible using SAM3 segmentation.

    Args:
        object_name: English description, e.g. "yellow cube", "red apple".

    Returns:
        {"bbox": [x1,y1,x2,y2], "score": float, "name": str}
        or None if not found.
    """
    import pyrealsense2 as rs

    cfg = _get_config()
    cam = cfg["camera"]
    w, h, fps = cam.get("width", 640), cam.get("height", 480), cam.get("fps", 15)

    pipeline = rs.pipeline()
    rscfg = rs.config()
    rscfg.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
    pipeline.start(rscfg)
    try:
        for _ in range(min(fps * 2, 30)):
            pipeline.wait_for_frames()
        frames = pipeline.wait_for_frames()
        cf = frames.get_color_frame()
        if not cf:
            return None
        color = np.asanyarray(cf.get_data())
    finally:
        pipeline.stop()

    result = _detect_with_retry(_get_sam3(), color, object_name)
    if result is None:
        return None
    bbox, _, score, name = result
    return {"bbox": bbox.tolist(), "score": score, "name": name}


def pick(object_name: str) -> bool:
    """Execute the full grasping pipeline for the named object.

    Sequence: observe → capture RGBD → SAM3 detect → compute grasp
              → transform to base frame → IK → execute motion.

    Args:
        object_name: English description of the target object.

    Returns:
        True on success, False on any failure.
    """
    cfg = _get_config()
    controller = _get_controller()
    yoloe_cfg = cfg["yoloe"]
    motion_cfg = cfg["motion"]
    safety_cfg = cfg["safety"]

    # Step 2: move to observation pose
    step_2_move_to_observation(controller, cfg["observation_pose"])

    # Step 3: capture RGBD (no live preview for automated use)
    cam_cfg = {**cfg["camera"], "live_preview": False}
    color, depth, intrinsic = step_3_capture_rgbd(cam_cfg)

    # Detect with SAM3 (retry with simpler prompts if needed)
    logger.info("Detecting '%s' with SAM3...", object_name)
    result = _detect_with_retry(_get_sam3(), color, object_name)
    if result is None:
        logger.warning("'%s' not detected after all retries.", object_name)
        return False
    bbox, mask, score, name = result
    logger.info("Detected '%s' score=%.3f", name, score)

    # Step 5: compute 3-D grasp from mask + depth
    try:
        rot, trans, quality, pca_angle = step_5_compute_grasp(
            bbox, depth, intrinsic, mask,
            factor_depth=cam_cfg.get("factor_depth", 10000),
            depth_strategy=yoloe_cfg.get("depth_strategy", "surface"),
            grasp_height_fraction=yoloe_cfg.get("grasp_height_fraction", 0.5),
        )
    except Exception as e:
        logger.error("Grasp computation failed: %s", e)
        return False

    # Step 6: transform grasp to robot base frame
    handeye_path = PROJECT_ROOT / cfg["handeye"]["calibration_path"]
    T_grasp, T_pre_grasp, T_lift = step_6_transform_to_base(
        rot, trans, cfg["observation_pose"], handeye_path, motion_cfg,
        grasp_height_offset=yoloe_cfg.get("grasp_height_offset", 0.015),
        tcp_offset=cfg.get("tcp_offset", 0.075),
        grasp_y_correction=yoloe_cfg.get("grasp_y_correction", 0.0),
        grasp_angle=pca_angle if yoloe_cfg.get("enable_pca_rotation", True) else 0.0,
    )

    # Get current joint angles as IK seed
    joints = controller.get_joint_states() or {}
    current_joints = np.array([
        joints.get(n, v)
        for n, v in zip(
            ["arm_joint1", "arm_joint2", "arm_joint3",
             "arm_joint4", "arm_joint5", "arm_joint6"],
            cfg["observation_pose"],
        )
    ])

    # Execute without interactive confirmation
    executor = IKExecutor(
        smooth_steps=motion_cfg.get("smooth_steps", 30),
        control_rate_hz=motion_cfg.get("control_rate_hz", 10.0),
        joint_limits_min=safety_cfg.get("joint_limits", {}).get("min"),
        joint_limits_max=safety_cfg.get("joint_limits", {}).get("max"),
        interpolation_type=motion_cfg.get("interpolation_type", "linear"),
    )
    return executor.execute_grasp_sequence(
        controller=controller,
        T_pre_grasp_base=T_pre_grasp,
        T_grasp_base=T_grasp,
        T_lift_base=T_lift,
        current_joints=current_joints,
        dry_run=False,
        confirm_each_phase=False,
        gripper_close_delay=motion_cfg.get("gripper_close_delay", 2.5),
    )


def place() -> bool:
    """Move to the fixed place pose and release the object.

    Returns:
        True (always, unless hardware error raises an exception).
    """
    cfg = _get_config()
    controller = _get_controller()
    place_pose = cfg.get("place_pose", cfg["observation_pose"])

    move_to_joint_pose(controller, place_pose, label="place")
    controller.open_gripper()
    time.sleep(1.5)
    return True


def ask_user(question: str) -> str:
    """Ask the user a clarifying question and return their answer.

    Args:
        question: The question to display.

    Returns:
        The user's text input.
    """
    print(f"\n[Robot asks] {question}")
    return input("[Your answer] ").strip()


# ── Arm control primitives ───────────────────────────────────────────────────

_OBSERVATION = [0.0, 1.0, -0.93, 0.83, 0.0, 0.0]
_HOME = [0.0, 0.0043, -0.1, -0.0347, -0.0055, 0.0013]


def move_to_observation() -> bool:
    """Move the arm to the observation position (for camera/detect).

    Returns:
        True on success.
    """
    controller = _get_controller()
    print("[Robot] Moving to observation position...")
    controller.move_to_position_smooth(
        _OBSERVATION, steps=20, rate_hz=10.0, interpolation_type='cosine',
        wait_for_convergence=True,
    )
    print("[Robot] At observation position.")
    return True


def move_to_home() -> bool:
    """Move the arm to the home (rest) position.

    Returns:
        True on success.
    """
    controller = _get_controller()
    print("[Robot] Moving to home position...")
    controller.move_to_position_smooth(
        _HOME, steps=20, rate_hz=10.0, interpolation_type='cosine',
        wait_for_convergence=True,
    )
    print("[Robot] At home position.")
    return True


def move_ee_relative(dx: float = 0.0, dy: float = 0.0, dz: float = 0.0) -> bool:
    """Move the end-effector by a relative offset (meters).

    Coordinate system (base frame):
        +X = forward (away from base)
        -X = backward (toward base)
        +Y = left
        -Y = right
        +Z = up
        -Z = down

    Example: move_ee_relative(dx=0.02) moves forward 2cm.

    Returns:
        True on success, False on IK failure.
    """
    controller = _get_controller()
    return controller.move_ee_relative(dx=dx, dy=dy, dz=dz)


def get_ee_position() -> list[float]:
    """Get the current end-effector position [x, y, z] in meters.

    Returns:
        [x, y, z] position list.
    """
    controller = _get_controller()
    ee = controller.get_current_ee_from_fk()
    return ee["position"]


def open_gripper() -> bool:
    """Open the gripper fully.

    Returns:
        True on success.
    """
    controller = _get_controller()
    controller.open_gripper()
    time.sleep(1.5)
    return True


def close_gripper() -> bool:
    """Close the gripper fully.

    Returns:
        True on success.
    """
    controller = _get_controller()
    controller.close_gripper()
    time.sleep(1.5)
    return True


# ── TTS ──────────────────────────────────────────────────────────────────────

def speak(text: str) -> bool:
    """Speak text aloud using TTS (non-blocking, runs in background).

    Args:
        text: Text to speak (Chinese or English).

    Returns:
        True (always).
    """
    import threading

    def _do_speak():
        try:
            _tts_root = PROJECT_ROOT / "skills" / "a1x-tts" / "scripts"
            if str(_tts_root) not in sys.path:
                sys.path.insert(0, str(_tts_root))
            from a1x_tts import speak as _speak
            _speak(text, voice="onyx")
        except Exception as e:
            logger.warning("TTS failed: %s", e)

    threading.Thread(target=_do_speak, daemon=True).start()
    return True
