---
name: a1x-tts
description: 将文本转为语音并播放，支持多种声音和 HD 模型。可独立使用，也可被其他技能调用实现机器人语音播报。
---

# A1X 语音合成技能（a1x-tts）

## 功能概述

使用云端 TTS API（OpenAI 兼容接口）将文本转换为语音音频。支持中英文文本、6 种声音风格和高清音质模式。

## API

### `speak(text, voice="onyx", model="tts-1", output_path=None, play=True) -> Path`

核心函数，文本转语音并播放。

| 参数 | 类型 | 说明 |
|------|------|------|
| `text` | str | 要转换的文本（支持中英文） |
| `voice` | str | 声音风格：`alloy`, `echo`, `fable`, `onyx`, `nova`, `shimmer` |
| `model` | str | 模型：`tts-1`（标准）或 `tts-1-hd`（高清） |
| `output_path` | str \| None | 保存路径，None 则使用临时文件 |
| `play` | bool | 是否立即播放 |

**返回值**：音频文件路径（`Path`）

### `text_to_speech(text, voice, model, output_path) -> Path`

仅生成音频文件，不播放。参数同上。

### 声音风格参考

| Voice | 风格描述 |
|-------|----------|
| `alloy` | 中性、平衡 |
| `echo` | 温暖、清晰 |
| `fable` | 叙事感、表现力强 |
| `onyx` | 低沉、有力（默认） |
| `nova` | 年轻、活泼 |
| `shimmer` | 柔和、友好 |

## 调用方式

### 命令行

```bash
# 直接朗读文本
python skills/a1x-tts/scripts/a1x_tts.py "你好，我是 A1X 机器人"

# 指定声音
python skills/a1x-tts/scripts/a1x_tts.py --voice nova "Hello world"

# 保存到文件
python skills/a1x-tts/scripts/a1x_tts.py --output /tmp/hello.mp3 "Hello"

# HD 高清模式
python skills/a1x-tts/scripts/a1x_tts.py --model tts-1-hd "High quality"

# 交互模式（输入文字即时朗读）
python skills/a1x-tts/scripts/a1x_tts.py

# 管道输入
echo "任务完成" | python skills/a1x-tts/scripts/a1x_tts.py --stdin
```

### 交互模式命令

| 命令 | 说明 |
|------|------|
| `voice <name>` | 切换声音风格 |
| `quit` / `exit` | 退出 |

### 作为库导入

```python
import sys
sys.path.insert(0, "/home/ubuntu/projects/A1Xsdk")
from skills.a1x_tts.scripts.a1x_tts import speak

# 播放语音
speak("抓取完成", voice="nova")

# 仅保存不播放
speak("Hello", output_path="/tmp/out.mp3", play=False)
```

## 环境要求

- 环境变量 `A1X_VLM_API_KEY` 已设置
- Python 包：`requests`
- 音频播放：`ffplay`（已安装）、`mpv`、`aplay` 或 `paplay` 任一可用

## 文件结构

```
skills/a1x-tts/
├── SKILL.md
└── scripts/
    └── a1x_tts.py       ← 主入口 + speak() API
```

## 与其他技能的关系

- **a1x-grab-skill**：可在抓取流程中调用 `speak()` 播报状态（如 "开始抓取" "任务完成"）
- **a1x-arm-codegen**：可在代码生成后播报执行结果
- **a1x-realsense-vision**：可将场景描述转为语音输出
