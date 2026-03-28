---
name: a1x-grab-skill
description: 当用户明确提出物体抓取任务时触发（如"帮我抓桌上的黄色物体"），采用 Code-as-Policies 方法：先视觉识别场景，再由 LLM 生成高层逻辑代码调用原语函数执行抓取
---

# A1X 智能抓取技能（a1x-grab-skill）

## 功能概述

基于 **Code as Policies** 架构，将用户的自然语言抓取请求转化为可执行的机器人操作：

- **视觉层**：RealSense D405 + qwen3.5-plus 云端大模型识别场景物体
- **规划层**：LLM 根据场景描述和用户意图生成 Python 调度代码
- **执行层**：SAM3 分割 + 完整抓取流水线（深度计算 → 坐标变换 → IK → 运动）

## 架构

```
用户请求
    │
    ▼
[Step 1] move_to_observation() + describe_scene()  — 移臂至观察位 + VLM 描述桌面物体
    │
    ▼
[Step 2] LLM Codegen           — 输入：场景描述 + 用户请求 → 输出：Python 代码
    │
    ▼
[Step 3] exec(code, sandbox)   — 沙箱执行，可调用以下原语函数
    │
    ▼
"Task completed successfully / unsuccessfully"
```

## 原语函数 API（robot_api.py）

### 抓取函数

| 函数 | 功能 |
|------|------|
| `pick(object_name)` | 完整抓取流水线：移至观察位 → 拍 RGBD → SAM3 检测 → 计算抓取点 → 执行运动。返回 bool |
| `place()` | 移动到固定放置位，打开夹爪。成功 pick 后必须调用 |
| `detect(object_name)` | 先调用 `move_to_observation()`，再拍照运行 SAM3。返回 `{"bbox":…,"score":…,"name":…}` 或 None |

> ⚠️ `object_name` 直接传给 SAM3，必须使用简短名称（1-3 词：[颜色] + 名词）。
> 如 `"yellow note"` 而非 `"yellow rectangular paper note"`。

### 手臂控制函数

| 函数 | 功能 |
|------|------|
| `move_to_observation()` | 移臂至观察位（相机可看到桌面）。`detect()` 前必须调用；`pick()` 内部已调用，无需额外调用 |
| `move_to_home()` | 移臂至 home（休息）位 |
| `move_ee_relative(dx=0.0, dy=0.0, dz=0.0)` | 末端执行器相对偏移（单位：米）。坐标系：+X=前 -X=后 +Y=左 -Y=右 +Z=上 -Z=下 |
| `get_ee_position()` | 返回当前末端 [x, y, z]（米） |
| `open_gripper()` | 完全打开夹爪 |
| `close_gripper()` | 完全关闭夹爪 |

### 其他函数

| 函数 | 功能 |
|------|------|
| `ask_user(question)` | 向用户提问，返回用户回答（处理歧义场景） |
| `describe_scene(prompt="…")` | 拍照 + VLM 描述场景（较慢），返回英文物体列表。仅在用户询问场景时使用 |
| `speak(text)` | 语音播报（后台非阻塞）。可选功能，用于播报关键状态，如 `speak("开始抓取")` |

## 生成代码规则

LLM 生成代码须遵守：

1. 仅使用上述原语函数，不得 import 任何模块。`speak()` 为可选，适当使用可增强交互体验
2. 成功 `pick()` 后必须调用 `place()`——除非用户指定自定义放置位置，则改用 `move_ee_relative() + open_gripper()`
3. 多目标任务使用 while 循环：`move_to_observation()` → `detect()` → `pick()` → `place()` → 循环
4. 有歧义时调用 `ask_user()`
5. 程序结尾**必须**调用 `move_to_observation()`，使手臂回到已知位置
6. 以**且仅以一条** print 语句结尾：
   - `print("Task completed successfully")` — 目标达成
   - `print("Task completed unsuccessfully")` — 任何步骤失败

## 触发场景

1. "帮我抓桌上的黄色物体"
2. "把苹果拿过来"
3. "抓取所有红色的东西"
4. "grab the yellow cube on the left"

## 调用方式

### 交互模式（默认）
```bash
conda run -n a1x_ros python /home/ubuntu/projects/A1Xsdk/skills/a1x-grab-skill/scripts/a1x_grab.py
```
支持多轮对话、场景问答与连续抓取任务。交互命令：
- `history` — 查看对话历史
- `clear` — 重置历史与场景缓存
- `scene` — 重新拍照描述场景
- `quit` / `exit` — 退出

### 单次执行模式
```bash
conda run -n a1x_ros python /home/ubuntu/projects/A1Xsdk/skills/a1x-grab-skill/scripts/a1x_grab.py "抓桌上的黄色物体"
```

### 常用参数

| 参数 | 说明 |
|------|------|
| `--show-code` | 执行前打印生成的代码（调试用） |
| `--execute` | 自动执行生成代码，无需确认（交互模式） |
| `--model MODEL` | 指定 LLM 模型（默认：`claude-opus-4-6`） |

```bash
# 显示生成代码并自动执行（交互模式）
conda run -n a1x_ros python .../a1x_grab.py --show-code --execute

# 单次执行并打印代码
conda run -n a1x_ros python .../a1x_grab.py --show-code "grab all objects"
```

## LLM 生成代码示例

### 单目标（无歧义）
```python
if pick("yellow cube"):
    place()
    move_to_observation()
    print("Task completed successfully")
else:
    move_to_observation()
    print("Task completed unsuccessfully")
```

### 多目标（while 循环，全部抓取）
```python
picked = 0
while True:
    move_to_observation()
    result = detect("yellow object")
    if result is None:
        break
    if pick(result["name"]):
        place()
        picked += 1
    else:
        break
move_to_observation()
if picked > 0:
    print("Task completed successfully")
else:
    print("Task completed unsuccessfully")
```

### 有歧义（询问用户）
```python
scene = describe_scene()
choice = ask_user("I see yellow cube (left) and yellow bottle (right). Which one?")
target = "yellow cube" if "cube" in choice.lower() or "left" in choice.lower() else "yellow bottle"
if pick(target):
    place()
    move_to_observation()
    print("Task completed successfully")
else:
    move_to_observation()
    print("Task completed unsuccessfully")
```

### 自定义放置位（不调用 place()）
```python
if pick("red cup"):
    move_ee_relative(dy=-0.03)   # 向右移 3cm
    open_gripper()
    move_to_observation()
    print("Task completed successfully")
else:
    move_to_observation()
    print("Task completed unsuccessfully")
```

## 环境要求

- Conda 环境：`a1x_ros`（含 SAM3、pyrealsense2、a1x_control、openai）
- RealSense D405 已连接
- `examples/handeye/handeye_calibration.yaml` 已完成标定
- CAN 总线已配置（1 Mbps / 5 Mbps FD）
- 环境变量 `A1X_API_KEY` 已设置（用于 LLM codegen）

## 文件结构

```
skills/a1x-grab-skill/
├── SKILL.md
├── robot_api.py          ← 原语函数库（12 个函数，含 speak）
└── scripts/
    └── a1x_grab.py       ← 主入口：场景理解 → codegen → exec
```

## 与其他技能的关系

- **a1x-tts**：`speak()` 调用 TTS 技能实现语音播报（后台非阻塞，不影响抓取流程）
- **a1x-realsense-vision**：`describe_scene()` 内部复用相同的拍图 + VLM 逻辑
- **a1x-arm-codegen**：适合自定义运动序列；本技能适合标准抓取任务
- **yoloe_grasp**：`pick()` 直接复用其流水线步骤（step_2 ~ step_6 + IKExecutor）
