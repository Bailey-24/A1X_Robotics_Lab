# A1X SDK

The A1X SDK is a Python-based robotic arm control interface for the A1X manipulator. This SDK provides easy-to-use Python wrappers for ROS 2 controllers, along with several comprehensive examples for end-to-end tasks like visual object picking, hand-eye calibration, and direct motion control.

**Documentation**: [Galaxea Dynamics A1X Documentation](https://docs.galaxea-dynamics.com/Guide/A1XY/software_introduction/ROS2/A1XY_Software_Introduction_ROS2/)

---

## 🛠️ System Initialization (Required for all tasks)

Before running any control scripts or examples, the robot's lower-level drivers and controllers must be launched.

### Configure the CAN Bus after every computer restart
Ensure the CAN driver is properly configured for communication with the arm joints:
```bash
sudo ip link set can0 type can bitrate 1000000 sample-point 0.875 dbitrate 5000000 fd on dsample-point 0.875
sudo ip link set up can0
```

---

## 🚀 Examples and Usage

Once the base system is running, activate the Python environment to run the SDK scripts:
```bash
conda activate a1x_ros
cd /home/ubuntu/projects/A1Xsdk
```

The `examples/` directory contains structured, ready-to-use workflows. 

### 1. `examples/yoloe_grasp/` (End-to-End Visual Grasping)
A complete pipeline for top-down vertical grasping using a RealSense camera and YOLOe for text-prompt zero-shot object detection.
- **Run the pipeline**: 
  ```bash
  python examples/yoloe_grasp/yoloe_grasp.py --target-name cup --visualize
  ```
- **Features**: Detects objects, computes 3D grasp coordinates using surface depth, solves Inverse Kinematics (IK), and executes smooth pick-and-place motions.
- See its [dedicated README](examples/yoloe_grasp/README.md) for detailed configuration.

### 2. `examples/handeye/` (Camera Calibration)
Tools for performing eye-in-hand calibration (calculating the transformation matrix from the robot's end-effector to the RealSense camera).
- **Run calibration**: 
  ```bash
  python examples/handeye/handeye_calibration.py --marker_id 42 --marker_size 0.10
  ```
- Comes with a browser-based Viser remote control UI to easily jog the robot to different capture poses.
- See its [dedicated README](examples/handeye/README.md) for full instructions.

### 3. `examples/motion/` (Basic Robot Control)
Standalone scripts showing how to use the `a1x_control.py` API for direct robot manipulation.
- **Move joints to a single pose**: `python examples/motion/joint_control_once.py`
- **Move joints smoothly via interpolation**: `python examples/motion/joint_control_smooth.py`
- **Control the gripper**: `python examples/motion/gripper_control.py`
- **Read End-Effector (EE) pose via Forward Kinematics**: `python examples/motion/read_ee_pose.py`
- **Read Joint states**: `python examples/motion/read_joint_pose.py`

### 4. `examples/camera/` (Vision Utilities)
Standalone utilities for testing and working with connected depth cameras.
- **Test RealSense D405**: 
  ```bash
  python examples/camera/realsense_d405_rgb.py
  ```
  *(Press `s` to save aligned RGB+Depth frames to `examples/camera/captured_frames/`, or `v` to record video).*

### 5. `examples/asmagic_iphone_data/` (AR Teleoperation Data)
Scripts for receiving and processing streaming ARKit pose data from an iPhone for teleoperation simulations (`01_rev_ar_data.py`, `02_rev_camera_data.py`, `03_phone_ik_sim.py`).

---

## 🤖 Skills (LLM-Powered Control)

The `skills/` directory contains LLM-powered interactive control systems. All skills require API keys set as environment variables (add to `~/.zshrc`):

```bash
# LLM codegen proxy (motchat.com)
export A1X_API_KEY="your-key-here"
# Cloud VLM for scene understanding (chatanywhere.tech)
export A1X_VLM_API_KEY="your-key-here"
```

### 1. `skills/a1x-arm-codegen/` (Natural Language Arm Control)

Use natural language (Chinese or English) to control the robotic arm. An LLM translates your text into Python control scripts and executes them on the real robot. Supports joint control, end-effector (Cartesian) control, and gripper commands.

```bash
# Interactive mode
python skills/a1x-arm-codegen/scripts/a1x_text_codegen.py

# Auto-execute mode (skip confirmation)
python skills/a1x-arm-codegen/scripts/a1x_text_codegen.py --execute
```

Example commands:
```
[You] > 去观测位置
[You] > 向前移动2厘米
[You] > move to observation, then move forward 2cm and up 3cm
```

Direction reference: `+X=forward`, `-X=backward`, `+Y=left`, `-Y=right`, `+Z=up`, `-Z=down`.

Also available at `examples/llm/a1x_text_codegen.py` — see its [README](examples/llm/README.md) for detailed usage.

### 2. `skills/a1x-grab-skill/` (Intelligent Grasping)

Code-as-Policies grasping orchestrator. Combines VLM scene understanding (Qwen), SAM3 object detection, and LLM code generation for flexible pick-and-place tasks.

```bash
# Interactive mode
python skills/a1x-grab-skill/scripts/a1x_grab.py

# Auto-execute mode
python skills/a1x-grab-skill/scripts/a1x_grab.py --execute

# Single-shot mode
python skills/a1x-grab-skill/scripts/a1x_grab.py "grab the yellow cube"
```

Example commands:
```
[You] > 桌面上有什么物体？         # Ask questions — LLM answers in text
[You] > 帮我抓取所有黄色物体        # Action commands — LLM generates & runs code
[You] > grab the red cup and place it 3cm to the right
```

Features:
- **Question answering**: asks about the scene, LLM replies naturally (no code)
- **Smart grasping**: pick, place, and custom placement with EE control
- **Multi-object loops**: uses `detect()` between picks to adapt to scene changes
- **SAM3 prompt retry**: auto-simplifies prompts if detection fails (e.g. "yellow rectangular note" → "note")
- **Code logging**: all generated code saved to `logs/generate_code/` with timestamps

Interactive commands: `scene` (re-capture), `history`, `clear`, `quit`.

### 3. `skills/a1x-realsense-vision/` (Camera Scene Description)

Capture an image from the RealSense D405 camera and analyze it with qwen3.5-plus (cloud VLM via `A1X_VLM_API_KEY`).

```bash
# Default prompt
python skills/a1x-realsense-vision/scripts/a1x_vision.py

# Custom prompt
python skills/a1x-realsense-vision/scripts/a1x_vision.py "桌上有什么物体"

# Save captured image
python skills/a1x-realsense-vision/scripts/a1x_vision.py --save /tmp/snap.jpg "what's on the desk?"
```

---

## 🏗️ SDK Architecture

- `a1x_control.py` - Core object-oriented Python API wrapping ROS 2 Topics (`JointController`). Handles reliable, verified message passing so users don't need to write ROS code.
- `pyroki/` - Python Robot Kinematics. A standalone, JAX-based fast IK/FK solver sub-project used by advanced workflows like `yoloe_grasp`.
- `install/` - Compiled ROS 2 workspace containing the `HDAS` driver and `mobiman` controller bindings. Do not modify directly.
