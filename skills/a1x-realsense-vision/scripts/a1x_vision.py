#!/usr/bin/env python3
"""
A1X RealSense Vision — Capture image from D405 and analyze with VLM.

Usage:
    python a1x_vision.py                              # ollama, default prompt
    python a1x_vision.py "桌上有什么物体"              # ollama, custom prompt
    python a1x_vision.py --model cloud "描述场景"      # qwen3.5-plus via chatanywhere
    python a1x_vision.py --save /tmp/snap.jpg "..."   # save captured image
"""
from __future__ import annotations

import argparse
import base64
import logging
import os
import sys
from pathlib import Path

import cv2 as cv
import numpy as np
import pyrealsense2 as rs
import yaml
from openai import OpenAI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("a1x_vision")

# ---------------------------------------------------------------------------
# API config (for cloud backend)
# ---------------------------------------------------------------------------
CLOUD_API_KEY = os.environ.get("A1X_VLM_API_KEY", "")
CLOUD_BASE_URL = "https://api.chatanywhere.tech/v1"

DEFAULT_OLLAMA_MODEL = "qwen3.5:9b"
DEFAULT_CLOUD_MODEL = "qwen3.5-plus"
DEFAULT_PROMPT = "图像中有什么？请详细描述场景和物体。"

# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------

def load_camera_config() -> dict:
    """Load camera params from examples/yoloe_grasp/config.yaml."""
    config_path = (
        Path(__file__).parent.parent.parent.parent
        / "examples"
        / "yoloe_grasp"
        / "config.yaml"
    )
    try:
        with open(config_path) as f:
            return yaml.safe_load(f).get("camera", {})
    except Exception as e:
        logger.warning("Failed to load camera config (%s). Using defaults.", e)
        return {}


def capture_frame(width: int = 640, height: int = 480, fps: int = 15) -> np.ndarray:
    """Start RealSense pipeline, wait for exposure to stabilize, return one BGR frame."""
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    pipeline.start(cfg)
    try:
        stable = min(fps * 2, 60)
        logger.info("Waiting for auto-exposure to stabilize (%d frames)...", stable)
        for _ in range(stable):
            pipeline.wait_for_frames()
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            raise RuntimeError("Failed to capture color frame from RealSense")
        logger.info("Frame captured.")
        return np.asanyarray(color_frame.get_data())
    finally:
        pipeline.stop()


# ---------------------------------------------------------------------------
# Image encoding
# ---------------------------------------------------------------------------

def image_to_base64(image: np.ndarray, quality: int = 90) -> str:
    """Encode BGR image to JPEG base64 string."""
    retval, buf = cv.imencode(".jpg", image, [int(cv.IMWRITE_JPEG_QUALITY), quality])
    if not retval:
        raise ValueError("Image encoding failed")
    return base64.b64encode(buf).decode("utf-8")


# ---------------------------------------------------------------------------
# VLM backends
# ---------------------------------------------------------------------------

def analyze_with_ollama(b64: str, prompt: str, model: str) -> str:
    """Call local Ollama vision model."""
    from ollama import chat  # type: ignore
    response = chat(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt,
                "images": [b64],
            }
        ],
    )
    return response.message.content


def analyze_with_cloud(b64: str, prompt: str, model: str) -> str:
    """Call cloud VLM (qwen3.5-plus) via chatanywhere OpenAI-compatible API."""
    client = OpenAI(api_key=CLOUD_API_KEY, base_url=CLOUD_BASE_URL)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        max_tokens=8192,
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="A1X RealSense D405 vision analysis")
    parser.add_argument(
        "prompt",
        nargs="?",
        default=DEFAULT_PROMPT,
        help=f"Analysis prompt (default: '{DEFAULT_PROMPT}')",
    )
    parser.add_argument(
        "--model",
        choices=["ollama", "cloud"],
        default="cloud",
        help="VLM backend to use (default: cloud)",
    )
    parser.add_argument(
        "--ollama-model",
        default=DEFAULT_OLLAMA_MODEL,
        help=f"Ollama model name (default: {DEFAULT_OLLAMA_MODEL})",
    )
    parser.add_argument(
        "--cloud-model",
        default=DEFAULT_CLOUD_MODEL,
        help=f"Cloud model name (default: {DEFAULT_CLOUD_MODEL})",
    )
    parser.add_argument(
        "--save",
        metavar="PATH",
        help="Save captured image to this file path",
    )
    args = parser.parse_args()

    # Load camera config
    cam = load_camera_config()
    width = cam.get("width", 640)
    height = cam.get("height", 480)
    fps = cam.get("fps", 15)

    print(f"[Camera] RealSense D405 — {width}x{height} @ {fps}fps")
    print(f"[Model ] {args.model} / {args.ollama_model if args.model == 'ollama' else args.cloud_model}")
    print(f"[Prompt] {args.prompt}")
    print()

    # Capture
    image = capture_frame(width, height, fps)

    if args.save:
        cv.imwrite(args.save, image)
        print(f"Image saved → {args.save}")

    # Encode
    b64 = image_to_base64(image)

    # Analyze
    print("Analyzing image...")
    try:
        if args.model == "ollama":
            result = analyze_with_ollama(b64, args.prompt, args.ollama_model)
        else:
            result = analyze_with_cloud(b64, args.prompt, args.cloud_model)
    except Exception as e:
        logger.error("Analysis failed: %s", e)
        sys.exit(1)

    print("\n分析结果:")
    print(result)


if __name__ == "__main__":
    main()
