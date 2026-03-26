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
[Step 1] describe_scene()      — 拍图 + qwen3.5-plus 描述桌面物体
    │
    ▼
[Step 2] LLM Codegen           — 输入：场景描述 + 用户请求 → 输出：Python 代码
    │
    ▼
[Step 3] exec(code, sandbox)   — 沙箱执行，只能调用 5 个原语函数
    │
    ▼
"Grasping task executed successfully / unsuccessfully"
```

## 原语函数 API（robot_api.py）

| 函数 | 功能 |
|------|------|
| `describe_scene(prompt)` | 拍照 + VLM 描述场景，返回英文物体列表 |
| `detect(object_name)` | SAM3 检测指定物体是否可见，返回 bbox/score 或 None |
| `pick(object_name)` | 完整抓取流水线：观察 → 检测 → 计算抓取点 → 执行 |
| `place()` | 移动到固定放置位，打开夹爪 |
| `ask_user(question)` | 向用户提问（处理歧义，如多个匹配物体） |

## 触发场景

1. "帮我抓桌上的黄色物体"
2. "把苹果拿过来"
3. "抓取所有红色的东西"
4. "grab the yellow cube on the left"

## 调用方式

### 基础调用
```bash
conda run -n a1x_ros python /home/ubuntu/projects/A1Xsdk/skills/a1x-grab-skill/scripts/a1x_grab.py "抓桌上的黄色物体"
```

### 显示生成的代码（调试用）
```bash
conda run -n a1x_ros python /home/ubuntu/projects/A1Xsdk/skills/a1x-grab-skill/scripts/a1x_grab.py --show-code "grab all objects"
```

## LLM 生成代码示例

### 单目标（无歧义）
```python
if detect("yellow cube"):
    if pick("yellow cube"):
        place()
        print("Grasping task executed successfully")
    else:
        print("Grasping task executed unsuccessfully")
else:
    print("Grasping task executed unsuccessfully")
```

### 多目标（全部抓取）
```python
targets = ["yellow cube", "red apple", "blue bottle"]
success = 0
for target in targets:
    if detect(target):
        if pick(target):
            place()
            success += 1
if success == len(targets):
    print("Grasping task executed successfully")
else:
    print("Grasping task executed unsuccessfully")
```

### 有歧义（询问用户）
```python
scene = describe_scene()
choice = ask_user("I see yellow cube (left) and yellow bottle (right). Which one?")
target = "yellow cube" if "cube" in choice.lower() or "left" in choice.lower() else "yellow bottle"
if pick(target):
    place()
    print("Grasping task executed successfully")
else:
    print("Grasping task executed unsuccessfully")
```

## 环境要求

- Conda 环境：`a1x_ros`（含 SAM3、pyrealsense2、a1x_control、openai）
- RealSense D405 已连接
- `examples/handeye/handeye_calibration.yaml` 已完成标定
- CAN 总线已配置（1 Mbps / 5 Mbps FD）

## 文件结构

```
skills/a1x-grab-skill/
├── SKILL.md
├── robot_api.py          ← 原语函数库（pick/place/detect/describe_scene/ask_user）
└── scripts/
    └── a1x_grab.py       ← 主入口：场景理解 → codegen → exec
```

## 与其他技能的关系

- **a1x-realsense-vision**：`describe_scene()` 内部复用相同的拍图 + VLM 逻辑
- **a1x-arm-codegen**：适合自定义运动序列；本技能适合标准抓取任务
- **yoloe_grasp**：`pick()` 直接复用其流水线步骤（step_2 ~ step_6 + IKExecutor）
