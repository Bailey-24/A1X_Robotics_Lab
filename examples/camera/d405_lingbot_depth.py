#!/usr/bin/env python3
"""
D405 + LingBot-Depth: 捕获一帧 RealSense D405 图像，
用 lingbot-depth 精化深度，生成三栏对比图（RGB | 原始深度 | 精化深度）。

用法：
    conda run -n lingbot-depth python examples/camera/d405_lingbot_depth.py
    conda run -n lingbot-depth python examples/camera/d405_lingbot_depth.py --output result/d405_comparison.png
"""
import sys
import os
import time
import argparse
import numpy as np
import cv2
import torch

try:
    import pyrealsense2 as rs
except ImportError:
    print("ERROR: pyrealsense2 未安装，请先：pip install pyrealsense2")
    sys.exit(1)

# lingbot-depth 路径
LINGBOT_DIR = os.path.join(os.path.dirname(__file__), "../../reference_code/lingbot-depth")
sys.path.insert(0, os.path.abspath(LINGBOT_DIR))
from mdm.model.v2 import MDMModel

MODEL_ID = "robbyant/lingbot-depth-pretrain-vitl-14-v0.5"
WARMUP_FRAMES = 30      # 跳过前 N 帧让自动曝光稳定


def capture_d405_frame():
    """从 D405 捕获一帧，返回 (rgb_np, depth_m, intrinsics_3x3)。"""
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)

    profile = pipeline.start(config)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()          # 通常 0.001 m/unit

    color_intr = (profile.get_stream(rs.stream.color)
                         .as_video_stream_profile()
                         .get_intrinsics())

    align = rs.align(rs.stream.color)

    print(f"  D405 启动，丢弃前 {WARMUP_FRAMES} 帧让曝光稳定...")
    for _ in range(WARMUP_FRAMES):
        pipeline.wait_for_frames(timeout_ms=5000)

    frames = pipeline.wait_for_frames(timeout_ms=5000)
    aligned = align.process(frames)

    color_frame = aligned.get_color_frame()
    depth_frame = aligned.get_depth_frame()

    rgb_bgr = np.asanyarray(color_frame.get_data())          # H×W×3 BGR
    depth_raw = np.asanyarray(depth_frame.get_data())        # H×W uint16

    pipeline.stop()

    rgb_np = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)        # RGB
    depth_m = depth_raw.astype(np.float32) * depth_scale     # 转为米

    # 构造 3×3 内参矩阵
    K = np.array([
        [color_intr.fx, 0,             color_intr.ppx],
        [0,             color_intr.fy, color_intr.ppy],
        [0,             0,             1              ],
    ], dtype=np.float32)

    print(f"  分辨率: {color_intr.width}×{color_intr.height}")
    print(f"  深度比例: {depth_scale} m/unit")
    valid = depth_m[depth_m > 0]
    if len(valid):
        print(f"  原始深度范围: {valid.min():.3f} ~ {valid.max():.3f} m")

    return rgb_np, depth_m, K


def colorize_depth(depth_m, vmin=None, vmax=None):
    """深度图转 TURBO 伪彩色（BGR，无效像素为黑）。"""
    valid = np.isfinite(depth_m) & (depth_m > 0)
    if vmin is None:
        vmin = depth_m[valid].min() if valid.any() else 0.0
    if vmax is None:
        vmax = depth_m[valid].max() if valid.any() else 1.0
    norm = np.clip((depth_m - vmin) / (vmax - vmin + 1e-8) * 255, 0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
    colored[~valid] = 0
    return colored, vmin, vmax


def add_label(img, text, color=(255, 255, 255)):
    """在图像左上角添加标签。"""
    out = img.copy()
    cv2.putText(out, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
    cv2.putText(out, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return out


def run(output_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    # 1. 捕获
    print("\n[1/3] 捕获 D405 帧...")
    rgb_np, depth_m, K = capture_d405_frame()
    h, w = rgb_np.shape[:2]

    # 2. 推理
    print("\n[2/3] 加载模型并推理...")
    model = MDMModel.from_pretrained(MODEL_ID).to(device)

    image_tensor = torch.tensor(
        rgb_np / 255.0, dtype=torch.float32, device=device
    ).permute(2, 0, 1).unsqueeze(0)

    depth_tensor = torch.tensor(depth_m, dtype=torch.float32, device=device)

    # 归一化内参
    K_norm = K.copy()
    K_norm[0, 0] /= w; K_norm[0, 2] /= w
    K_norm[1, 1] /= h; K_norm[1, 2] /= h
    K_tensor = torch.tensor(K_norm, dtype=torch.float32, device=device).unsqueeze(0)

    t0 = time.time()
    with torch.no_grad():
        output = model.infer(image_tensor, depth_in=depth_tensor,
                             apply_mask=True, intrinsics=K_tensor)
    infer_ms = (time.time() - t0) * 1000
    depth_refined = output["depth"].squeeze().cpu().numpy()

    valid_ref = depth_refined[np.isfinite(depth_refined) & (depth_refined > 0)]
    print(f"  推理耗时: {infer_ms:.0f} ms")
    print(f"  精化深度范围: {valid_ref.min():.3f} ~ {valid_ref.max():.3f} m")

    # 3. 生成对比图
    print("\n[3/3] 生成对比图...")

    # 使用同一色标，方便对比
    all_valid = depth_m[depth_m > 0]
    vmin = all_valid.min() if len(all_valid) else 0.0
    vmax = all_valid.max() if len(all_valid) else 5.0

    rgb_bgr = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)
    raw_color, _, _ = colorize_depth(depth_m, vmin, vmax)
    ref_color, _, _ = colorize_depth(depth_refined, vmin, vmax)

    # 添加标签和统计文字
    panel_rgb = add_label(rgb_bgr, "RGB")

    raw_text = f"Raw  {vmin:.2f}-{vmax:.2f}m"
    panel_raw = add_label(raw_color, raw_text)

    ref_range = f"{valid_ref.min():.2f}-{valid_ref.max():.2f}m" if len(valid_ref) else "N/A"
    panel_ref = add_label(ref_color, f"Refined  {ref_range}", color=(0, 255, 128))

    comparison = np.hstack([panel_rgb, panel_raw, panel_ref])

    # 底部信息栏
    bar_h = 36
    bar = np.zeros((bar_h, comparison.shape[1], 3), dtype=np.uint8)
    info = (f"D405 640x480  |  LingBot-Depth {MODEL_ID.split('/')[-1]}"
            f"  |  infer {infer_ms:.0f}ms  |  GPU: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")
    cv2.putText(bar, info, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    comparison = np.vstack([comparison, bar])

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    cv2.imwrite(output_path, comparison)
    print(f"\n对比图已保存: {output_path}  ({comparison.shape[1]}x{comparison.shape[0]})")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="/tmp/d405_lingbot_comparison.png")
    args = parser.parse_args()
    run(args.output)
