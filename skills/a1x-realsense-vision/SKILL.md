---
name: a1x-realsense-vision
description: 当用户询问看到什么、能看到什么、需要观察/分析当前摄像头画面时触发，通过 RealSense D405 相机实时拍摄，调用 Ollama qwen3.5:9b（本地，快）或 Claude opus-4-6（云端，强）进行视觉分析
---

# RealSense 视觉识别技能（a1x-realsense-vision）

## 功能概述

通过 Intel RealSense D405 相机实时拍摄彩色帧，调用视觉语言模型（VLM）对图像进行分析，返回场景描述、物体识别、环境细节等结果。

支持两种后端：
- **Ollama**（默认，本地，低延迟）：`qwen3.5:9b`
- **Claude**（云端，理解力更强）：`claude-opus-4-6`，通过 `A1X_API_KEY` 环境变量鉴权

## 触发场景

1. 用户问"你看到了什么"、"帮我看看"、"现在面前有什么"
2. 需要识别物体、判断场景、分析画面内容
3. 其他任何需要摄像头实时视觉信息的请求

## 环境要求

- Python 依赖：`pyrealsense2`、`opencv-python`、`numpy`、`ollama`、`requests`、`pyyaml`
- Ollama 后端：本地已启动 Ollama 服务，并已拉取目标模型
  ```bash
  ollama pull qwen3.5:9b
  ```
- Claude 后端：已设置环境变量 `A1X_API_KEY`
- RealSense D405 相机已通过 USB 连接并被系统识别

## 调用方式

### 基础调用（Ollama，默认提示词）
```bash
python /home/ubuntu/projects/A1Xsdk/skills/a1x-realsense-vision/scripts/a1x_vision.py
```

### 自定义提示词
```bash
python /home/ubuntu/projects/A1Xsdk/skills/a1x-realsense-vision/scripts/a1x_vision.py "请详细描述桌面上有哪些物体及其位置"
```

### 切换到 Claude（更强的理解能力）
```bash
python /home/ubuntu/projects/A1Xsdk/skills/a1x-realsense-vision/scripts/a1x_vision.py --model claude "图像中是否有可抓取的物体？"
```

### 保存抓拍图像
```bash
python /home/ubuntu/projects/A1Xsdk/skills/a1x-realsense-vision/scripts/a1x_vision.py --save /tmp/capture.jpg "图中有什么"
```

### 完整参数说明
```
positional arguments:
  prompt                分析提示词（默认：图像中有什么？请详细描述）

options:
  --model {ollama,claude}   选择后端（默认：ollama）
  --ollama-model NAME        Ollama 模型名（默认：qwen3.5:9b）
  --claude-model NAME        Claude 模型名（默认：claude-opus-4-6）
  --save PATH                将拍摄图像保存到指定路径
```

## 相机配置

相机分辨率和帧率从 `examples/yoloe_grasp/config.yaml` 自动加载（`camera` 字段）。
若配置文件不存在，使用默认值：640×480，15fps。

## 常见问题

| 错误信息 | 解决方法 |
|----------|----------|
| `No device connected` | 检查 D405 是否正确插入 USB 3.x 口 |
| `ollama connection refused` | 执行 `ollama serve` 启动本地服务 |
| `401 Unauthorized` | 确认已设置 `export A1X_API_KEY=...` |
| `model not found` | 执行 `ollama pull qwen3.5:9b` 拉取模型 |
