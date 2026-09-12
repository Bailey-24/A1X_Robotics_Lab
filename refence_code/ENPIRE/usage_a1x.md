# ENPIRE × A1X 使用指南

在 A1X 单臂上用 ENPIRE agent 跑视觉抓取/放置任务。ENPIRE 代码在 `refence_code/ENPIRE/`。

## 0. 前置条件（一次性）

- `~/.zshrc` 已配置 GLM 验证器三件套（新终端自动加载）：
  `NVIDIA_API_KEY` / `NVIDIA_VL_BASE_URL` / `NVIDIA_VLM_MODEL`
- conda 环境：`a1x_ros`（ROS 2 + 桥接）、`sam3`（SAM3 分割服务，端口 9500）
- 底层服务已在跑：HDAS、joint tracker、gripper controller
  （`a1x.sh` 检测到外部实例会自动复用，不会重复拉起；`down` 也不会动它们）
- 桌面高度：环境变量 `A1X_TABLE_Z_M`，默认 `-0.04`（基座系，米）。
  运动栅栏 z 下限 = 桌面 + 2mm，防止向下扎穿桌面。
- 首次使用先装 ENPIRE 依赖：`scripts/a1x.sh install`

## 1. 启动

```bash
cd refence_code/ENPIRE
scripts/a1x.sh up                  # 起 bridge(+SAM3)，不动机械臂
scripts/a1x.sh up --allow-motion   # 允许运动（跑任务必须）
scripts/a1x.sh status              # 查看组件状态
scripts/a1x.sh smoke               # 只读测试：读状态 + 相机
```

## 2. 跑任务

```bash
scripts/a1x.sh run --allow-motion "pick up the cube" max_iterations=1
scripts/a1x.sh run --allow-motion "pick up the remote and put on the top of the cube" max_iterations=1
```

流程：observer → LLM 生成 Python 代码 → 机械臂执行 → GLM 看图判定成功/失败。
失败会带着失败原因进入下一轮重试（调大 `max_iterations` 即可，如 3）。

跑之前确认：工作区无障碍、急停在手边。`run` 结束后自动停止它启动的组件。

## 3. 看结果

```text
refence_code/ENPIRE/logs/<时间戳>_agent_<任务名>/
├── iterations/iter_000/code.py            生成的代码
├── iterations/iter_000/exec_000/exec.log  执行日志（出错先看这里）
├── iterations/iter_000/exec_000/vis/      分割可视化图
└── history.md                             迭代历史与验证结论
```

## 4. 停止

```bash
scripts/a1x.sh down    # 只停本脚本启动的组件（bridge/sam3 等）
```

## 常见问题

| 现象 | 原因 / 处理 |
|---|---|
| 验证器报 `No NVIDIA key found` | 终端是改 zshrc 前开的旧终端 → 新开终端或 `source ~/.zshrc` |
| `stack not ready after 60s` | bridge 僵尸或端口被占 → `scripts/a1x.sh down && scripts/a1x.sh up --allow-motion` |
| 启动时 anygrasp/bundlesdf/curobo `unreachable` WARNING | 预期行为（未部署，A1X 流程不用它们） |
| 分割找不到物体 | 换更具体的描述词（颜色+物体名），或在生成代码里降 `score_thresh` |
| 机械臂动作中要停 | 按急停；软件层 `curl -X POST --noproxy '*' http://127.0.0.1:11337/stop` |

## 任务编写提示

- 多物体任务（"把 A 放到 B 上"）框架已内置"先全感知再动"的工作流，正常直接描述即可。
- 抓取算法：SAM3 分割 → D405 对齐深度 → 手眼标定（`examples/handeye/`）转基座系 →
  PCA 主轴对齐 → 顶向下抓取；夹爪闭合堵转=夹到物体，自动判定成功。
