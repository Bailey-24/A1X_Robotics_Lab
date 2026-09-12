# Galaxea A1X Integration

The A1X integration lives entirely inside ENPIRE and does not modify the A1X SDK examples. It reuses the motion conventions from `examples/motion`, the D405 settings from `examples/yoloe_grasp/config.yaml`, and the eye-in-hand calibration from `examples/handeye/handeye_calibration.yaml`.

## Quick Start (single terminal)

Everything is driven by one script: `refence_code/ENPIRE/scripts/a1x.sh`. It starts HDAS, the joint tracker, the gripper controller, and the bridge in the background, waits until the stack is healthy, and stops exactly the components it started. An A1X ROS stack you launched yourself is detected and left untouched.

```bash
cd /home/ubuntu/projects/A1Xsdk/refence_code/ENPIRE

# One-time install (uv env with all needed extras)
scripts/a1x.sh install

# Read-only health check of robot state + wrist camera
scripts/a1x.sh smoke

# Run the agent: starts the stack, runs, then stops everything automatically.
# Any extra args after the task are Hydra overrides.
scripts/a1x.sh run "pick up the red cube" max_iterations=1

# Same, but with motion allowed (clear the workspace, keep an e-stop ready)
scripts/a1x.sh run --allow-motion "pick up the red cube" max_iterations=1

# Long-running form: start the stack once, run many commands, stop later
scripts/a1x.sh up                    # read-only bridge
scripts/a1x.sh up --allow-motion     # motion-enabled bridge
scripts/a1x.sh status                # show components + bridge health
scripts/a1x.sh down                  # stop everything this script started
```

Logs and PIDs live in `/tmp/a1x-enpire/` (`hdas.log`, `tracker.log`, `gripper.log`, `bridge.log`).

Recommended alias:

```bash
alias a1x='/home/ubuntu/projects/A1Xsdk/refence_code/ENPIRE/scripts/a1x.sh'
a1x run "pick up the red cube" max_iterations=1
```

## Architecture

ENPIRE requires Python 3.11, while the A1X ROS 2 Humble workspace uses Python 3.10. Run them as separate processes:

```text
ENPIRE Python 3.11 -> local HTTP -> A1X bridge Python 3.10 -> a1x_control/ROS 2
ENPIRE Python 3.11 -> RealSense D405 directly
```

The bridge reads physical joints from `/hdas/feedback_arm`. It does not treat `/joint_states`, which may echo commands, as measured convergence.

## Manual Startup (alternative to a1x.sh)

First start the existing A1X ROS graph in separate terminals if it is not already running:

```bash
source /home/ubuntu/projects/A1Xsdk/install/setup.bash
ros2 launch HDAS a1xy.py
```

```bash
source /home/ubuntu/projects/A1Xsdk/install/setup.bash
ros2 launch mobiman A1x_jointTrackerdemo_launch.py launch_rviz:=false
```

Start the optional gripper controller before using gripper tools:

```bash
source /home/ubuntu/projects/A1Xsdk/install/setup.bash
ros2 launch mobiman A1xy_gripperController_launch.py
```

Then start the bridge. Read-only state mode is the default:

```bash
source /home/ubuntu/projects/A1Xsdk/install/setup.bash
/home/ubuntu/miniconda3/envs/a1x_ros/bin/python \
  refence_code/ENPIRE/enpire/env/forge/robot/a1x/bridge.py
```

Allow motion only after clearing the workspace and preparing an emergency stop:

```bash
source /home/ubuntu/projects/A1Xsdk/install/setup.bash
/home/ubuntu/miniconda3/envs/a1x_ros/bin/python \
  refence_code/ENPIRE/enpire/env/forge/robot/a1x/bridge.py --allow-motion
```

The bridge is intentionally run as a file, so `enpire` does not need to be installed in `a1x_ros`. It binds to `127.0.0.1:11337` by default.

If more than one RealSense is connected, select the calibrated wrist D405 explicitly:

```bash
export CAP_WRIST_REALSENSE_SERIAL=<D405-serial>
```

This station currently has D405 firmware newer than ENPIRE's tested version. After checking exposure, allow it explicitly:

```bash
export ENPIRE_ALLOW_D405_FIRMWARE=1
```

The bridge does not import `a1x_control` and does not launch or stop HDAS/mobiman. It only attaches to the existing ROS graph, so it does not duplicate the processes used by your examples.

## State And Camera Smoke Test

`scripts/a1x.sh smoke` runs the bundled `cap/saved_scripts/a1x_smoke.py`, which prints joint state, gripper, EE pose, RGB-D shapes, intrinsics, and extrinsics without any motion.

Equivalent manual invocation from `refence_code/ENPIRE`:

```bash
uv run --no-sync enpire-run-script \
  robot=real_a1x \
  task='Read A1X state and camera' \
  script_file=cap/saved_scripts/a1x_smoke.py \
  robot.go_home_on_exit=false \
  runtime.quiet=false
```

## Motion Tools

The generated-script namespace exposes:

- `move_joints([q1, ..., q6], duration_s=2.0)`
- `move_ee_absolute([x, y, z], quat_xyzw=None)`
- `move_ee_relative([dx, dy, dz])`
- `move_to_observation()`
- `set_gripper("right", value)`, `open_gripper("right")`, `close_gripper("right")`
- `go_home()`

Cartesian tools use the A1X PyRoki IK. They do not provide collision-aware trajectory planning. ENPIRE deliberately does not expose YAM's `freespace_move` for A1X.

## Perception And Grasp

The camera role is `wrist`, configured as 640x480 at 15 FPS. Depth is aligned to RGB and converted using the D405's runtime depth scale, producing metres.

Typical generated code:

```python
move_to_observation()
seg = segment_object("red cube", media="camera:wrist")
grasp = compute_topdown_grasp(seg, camera="wrist")
open_gripper("right")
move_ee_absolute(grasp["pregrasp_position"], grasp["quat_xyzw"])
move_ee_absolute(grasp["grasp_position"], grasp["quat_xyzw"])
close_gripper("right")
move_ee_absolute(grasp["lift_position"], grasp["quat_xyzw"])
```

`scripts/a1x.sh up/run/smoke` starts the SAM3 segmentation server automatically on `127.0.0.1:9500` using the `sam3` conda environment (GPU, ~4.5 GB VRAM). Weights are the transformers-format `facebook/sam3` snapshot under `~/.cache/huggingface/hub/models--facebook--sam3` (downloaded from ModelScope; the HF repo is gated). Set `A1X_START_SAM3=0` to skip it.

## Run The Agent

The agent loop is:

```text
observe -> generate Python -> execute in run_script subprocess -> verify -> reflect -> retry
```

The single-terminal form is `scripts/a1x.sh run "<task>" [overrides]`. Inspect the resolved configuration first by adding Hydra's `--cfg job`:

```bash
scripts/a1x.sh run "Pick up the red cube using the wrist camera" --cfg job
```

Run one conservative attempt:

```bash
scripts/a1x.sh run --allow-motion "Pick up the red cube using the wrist camera" \
  max_iterations=1 code_generator.num_candidates=1
```

The default LLM backend routes through the local `claude` CLI bridge. Alternatives that need only an API key: `llm=gemini` (GEMINI_API_KEY), `llm=cloud` (ANTHROPIC_API_KEY), `llm=nvidia` (NVIDIA_API_KEY). The task verifier uses the same VLM keys; without a key it fails safe (`success=False`).

The parent agent process opens neither the D405 nor motion tools directly. Each generated program executes in a child `run_script` process, which owns the camera for that attempt. Artifacts and generated code are written under `logs/`.

The A1X namespace supplies a conservative wrist-camera VLM `get_task_info()` verifier. For repeatable experiments, replace it with a task-specific verifier outside generated policy code. Do not let generated policy code alter bridge limits, reset behavior, or verification criteria.
