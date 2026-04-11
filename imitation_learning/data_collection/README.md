# A1X Demonstration Recording for Imitation Learning

Record joint states and camera images while the `yoloe_grasp` pipeline runs as an expert policy, then convert the data to **LeRobot** format for imitation learning training.

## Overview

This package wraps the [control_your_robot](../../refence_code/control_your_robot/) framework to record A1X arm + RealSense D405 data during autonomous grasping. The `yoloe_grasp` pipeline runs unmodified — a background recording thread polls joint states and camera frames at a fixed frequency and writes HDF5 episodes.

Data collection is **fully automated**: after each successful grasp, the object is relocated to a nearby position (cyclic 5 cm right → down → left → up pattern with 10° wrist rotation), keeping it within the workspace so the next episode runs immediately — no manual intervention needed.

```
yoloe_grasp (expert policy)         DemoRecorder (background thread @ 20Hz)
  |                                    |
  | controls robot (steps 1-8)         | reads joint states + gripper
  |                                    | reads latest D405 frame
  |                                    | reads last commanded action
  |                                    v
  v                                  CollectAny -> HDF5 episode
  grasp complete
                                            |
                                            v
                                     convert_to_lerobot.py
                                            |
                                            v
                                      LeRobot Dataset
```

## Prerequisites

### 1. Hardware setup (see project [CLAUDE.md](../../CLAUDE.md))

```bash
# Configure CAN bus (after every reboot)
sudo ip link set can0 type can bitrate 1000000 sample-point 0.875 \
    dbitrate 5000000 fd on dsample-point 0.875
sudo ip link set up can0

# Activate environment
conda activate a1x_ros
cd /home/ubuntu/projects/A1Xsdk
```

### 2. Python dependencies

The recording pipeline uses existing dependencies (`pyrealsense2`, `h5py`, `rclpy`, `numpy`, `opencv-python`, `pyyaml`). The LeRobot conversion additionally requires:

```bash
pip install lerobot tqdm
```

### 3. Hand-eye calibration & detector checkpoints

- Hand-eye calibration: [examples/handeye/handeye_calibration.yaml](../../examples/handeye/handeye_calibration.yaml) must exist
- YOLOe checkpoint (optional): `refence_code/yoloe/yoloe-v8l-seg.pt`
- SAM3 (default detector): installed via `refence_code/sam3/`

## File Layout

```
imitation_learning/data_collection/
├── __init__.py
├── a1x_controller.py      # ArmController adapter for JointController
├── d405_sensor.py         # VisionSensor adapter for D405 (continuous capture)
├── a1x_robot.py           # Robot class: arm + camera
├── recorder.py            # DemoRecorder: background polling thread
├── record_demo.py         # Main script: records demos using yoloe_grasp
├── convert_to_lerobot.py  # HDF5 -> ACT HDF5 -> LeRobot dataset
├── config.yaml            # Recording parameters
└── README.md              # This file
```

## Quick Start

### 1. Record demonstrations

```bash
# Record 10 episodes of grasping a banana (fully automated, default detector: SAM3)
python imitation_learning/data_collection/record_demo.py --target-name banana

# Custom episode count + detector
python imitation_learning/data_collection/record_demo.py \
    --target-name cup \
    --detector yoloe \
    --num-episodes 20

# Dry run (no hardware, synthetic data, verifies pipeline)
python imitation_learning/data_collection/record_demo.py --dry-run --num-episodes 8

# Resume from a specific episode index
python imitation_learning/data_collection/record_demo.py \
    --target-name banana \
    --start-episode 10 \
    --num-episodes 5
```

**What happens per episode:**

```
Episode 0:
  [RECORD] observe -> detect -> grasp -> lift [/RECORD]
           relocate RIGHT 5cm -> rotate wrist +10 deg -> release -> retract -> obs pose

Episode 1:
  [RECORD] observe -> detect -> grasp -> lift [/RECORD]
           relocate DOWN 5cm -> rotate wrist +10 deg -> release -> retract -> obs pose

Episode 2:
  [RECORD] observe -> detect -> grasp -> lift [/RECORD]
           relocate LEFT 5cm -> rotate wrist +10 deg -> release -> retract -> obs pose

Episode 3:
  [RECORD] observe -> detect -> grasp -> lift [/RECORD]
           relocate UP 5cm -> rotate wrist +10 deg -> release -> retract -> obs pose

Episode 4:  (cycle repeats)
  ...
```

Key design points:

- **Recording scope**: only the observation-through-lift sequence is recorded (not the relocation step). This captures the pure grasp skill without logistics motion.
- **Cyclic relocation**: after each grasp, the object is moved 5 cm in a rotating direction (right → down → left → up), keeping it within the workspace. After a full 4-episode cycle, the object returns to approximately its original position.
- **Wrist rotation**: +10 deg on joint 6 after each relocation, diversifying object orientation for richer training data.
- **Model caching**: the detector (SAM3 or YOLOe) is loaded **once** before the episode loop and reused, saving significant startup time per episode.
- **Safety**: the first episode prompts for interactive confirmation; all subsequent episodes run fully unattended.
- **Failure handling**: 3 consecutive failures abort the run (likely means the object left the workspace). On any failure, the gripper is opened and the arm returns to observation pose before retrying.

### 2. Inspect recorded episodes

```python
import h5py

with h5py.File("data/demos/yoloe_grasp_banana/0.hdf5", "r") as f:
    print("Groups:", list(f.keys()))
    # Groups: ['a1x_arm', 'cam_wrist']

    print("Arm fields:", list(f["a1x_arm"].keys()))
    # ['action', 'gripper', 'joint', 'timestamp']

    print("Joint shape:", f["a1x_arm/joint"].shape)        # (N, 6)
    print("Gripper shape:", f["a1x_arm/gripper"].shape)    # (N, 1)
    print("Action shape:", f["a1x_arm/action"].shape)      # (N, 7)
    print("Color shape:", f["cam_wrist/color"].shape)      # (N, 480, 640, 3)
```

### 3. Convert to LeRobot format

```bash
python imitation_learning/data_collection/convert_to_lerobot.py \
    --data-dir ./data/demos/yoloe_grasp_white_object/ \
    --repo-id a1x/yoloe_grasp_white_object

# Default (no flags): Raw HDF5 → LeRobot directly, no *_act/ directory created
# --via-act: Old two-stage behavior (Raw → ACT → LeRobot)
# --act-only: Only produce ACT HDF5 (unchanged)

# Stage 1 only (raw -> ACT HDF5, skip LeRobot creation)
python imitation_learning/data_collection/convert_to_lerobot.py \
    --data-dir ./data/demos/yoloe_grasp_white_object/ \
    --act-only \
    --act-output ./data/act_hdf5/
```

The resulting LeRobot dataset lives at `~/.cache/huggingface/lerobot/a1x/.../`.

### 4. Load the LeRobot dataset

```python
# LeRobot >= 0.4 uses `lerobot.datasets`; older versions use `lerobot.common.datasets`
from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset("a1x/yoloe_grasp_banana")
print(f"Total frames: {len(dataset)}")
print(f"Episodes: {dataset.num_episodes}")
print(f"FPS: {dataset.fps}")

sample = dataset[0]
print(sample.keys())
# dict_keys(['observation.state', 'observation.images.cam_wrist', 'action', 'task', ...])
```

### 5. Visualize the dataset

LeRobot ships a built-in visualizer (`lerobot-dataset-viz`) that uses [Rerun.io](https://rerun.io) to display joint states, actions, and camera frames frame-by-frame.

**Local viewing** (data and viewer on the same machine — easiest):

```bash
lerobot-dataset-viz \
    --repo-id ~/.cache/huggingface/lerobot/a1x/yoloe_grasp_white_object \
    --episode-index 0
```

This launches a Rerun window showing:
- Camera images (`observation.images.cam_wrist`) as a video stream
- Joint positions (`observation.state`) as 7 time-series plots
- Actions (`action`) as 7 time-series plots aligned with the state plots
- Frame-by-frame scrubbing along the timeline


> **Note**: The dataset must already exist in `~/.cache/huggingface/lerobot/<repo-id>/` (which is where `convert_to_lerobot.py` writes it). If you need to point at a different location, pass `--root /path/to/dataset/parent/`.

## Configuration

Edit [config.yaml](config.yaml) to change recording parameters:

```yaml
recording:
  save_path: "./data/demos/"   # Where HDF5 episodes are written
  task_name: "yoloe_grasp"     # Sub-folder name (auto-appended with --target-name)
  save_freq: 20                # Recording frequency (Hz)
  num_episodes: 10             # Default number of episodes
  move_check: false            # false = record stationary frames too

camera:
  serial: null                 # null = auto-detect D405
  width: 640
  height: 480
  fps: 30                      # D405 capture rate

lerobot:
  repo_id: "a1x/yoloe_grasp_demos"
  fps: 20                      # Must match recording save_freq
  instruction: "Pick up the target object and place it to the side"

yoloe_grasp_config: "examples/yoloe_grasp/config.yaml"
```

## Data Schema

### Raw HDF5 (output of `record_demo.py`)

```
episode_N.hdf5
├── a1x_arm/
│   ├── joint      [N, 6]           float64   # radians
│   ├── gripper    [N, 1]           float64   # normalized 0.0-1.0
│   ├── action     [N, 7]           float64   # 6 joints + 1 gripper (commanded)
│   └── timestamp  [N]              int64     # nanoseconds
└── cam_wrist/
    ├── color      [N, 480, 640, 3] uint8     # RGB
    └── timestamp  [N]              int64
```

### LeRobot features (output of `convert_to_lerobot.py`)

| Feature                          | Shape         | dtype   | Description                       |
| -------------------------------- | ------------- | ------- | --------------------------------- |
| `observation.state`              | (7,)          | float32 | 6 joints (rad) + 1 gripper (0-1)  |
| `action`                         | (7,)          | float32 | Commanded 6 joints + 1 gripper    |
| `observation.images.cam_wrist`   | (3, 480, 640) | image   | D405 wrist camera                 |
| `task`                           | string        | -       | Instruction text                  |

## Architecture Notes

### Why a background polling thread?

The `yoloe_grasp` pipeline is untouched — it still controls the robot via `JointController.move_to_position_smooth()` which spins the ROS node internally. The recorder runs in a separate thread and only **reads** cached state set by ROS callbacks. No spin contention.

### Camera ownership

Only one process can own a RealSense pipeline for a given device. `D405Sensor` is the sole owner and runs a continuous grab thread. When `yoloe_grasp` step 3 needs a detection frame, `record_demo.py` calls `sensor.get_capture_for_detection()` instead of opening a second `RGBDCapture`.

### Action tracking

The A1X controller adapter subscribes to `/motion_target/target_joint_state_arm` and `/motion_target/target_position_gripper` on the shared `JointController` node. Every recorded frame includes the **last commanded target** as the `action` field, which is the ground-truth policy output for imitation learning.

### Thread safety

- `JointController.current_joint_state` / `.current_gripper_state`: Python reference assignment is atomic under the GIL → safe for concurrent read.
- `D405Sensor._latest_color` / `_latest_depth`: Protected by `threading.Lock` — only the grab thread writes; recorder & detection read.
- `rclpy.spin_once()` is called only by the main thread (via `yoloe_grasp`), never by the recorder.

## Troubleshooting

| Symptom                                           | Cause / Fix                                                                                        |
| ------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| `RuntimeError: D405Sensor: no frames captured yet`| Pipeline hasn't warmed up — increase warmup count in `set_up()` or delay before first capture      |
| `No joint state data available yet`               | CAN bus not configured or drivers not running — rerun the `ip link` commands and check `ros2 topic list` |
| Episode count mismatches recording time           | Python GIL contention — reduce `record_freq` in config.yaml (try 15 Hz)                            |
| Very large HDF5 files (>200 MB/episode)           | Enable JPEG encoding: set `is_jpeg = True` in `D405Sensor.__init__` (10× size reduction, lossy)    |
| `No module named 'robot.sensor.vision_sensor'`    | The adapters insert `refence_code/control_your_robot/src` into `sys.path` — make sure that folder exists |
| LeRobot conversion fails with version error       | Install matching LeRobot: `pip install 'lerobot>=0.1.0'`                                           |

## Verification checklist

1. **Dry run** — `python imitation_learning/data_collection/record_demo.py --dry-run --num-episodes 1` (no hardware)
2. **HDF5 structure** — inspect with the snippet above; verify shapes `(N, 6)`, `(N, 1)`, `(N, 7)`, `(N, 480, 640, 3)`
3. **Value ranges** — joints in radians (~±π), gripper in `[0, 1]`, images `uint8 ∈ [0, 255]`
4. **Timing** — `N ≈ episode_duration × save_freq` (small drift OK)
5. **LeRobot load** — `LeRobotDataset("<repo-id>")` returns a non-empty dataset
6. **Real hardware test** — record one real episode with `--target-name banana`, verify grasp succeeds and HDF5 looks sane

## Reference

- [yoloe_grasp pipeline](../../examples/yoloe_grasp/yoloe_grasp.py) — the expert policy
- [a1x_control.py](../../a1x_control.py) — `JointController` ROS 2 interface
- [control_your_robot framework](../../refence_code/control_your_robot/) — `ArmController`, `VisionSensor`, `Robot`, `CollectAny` base classes
- [LeRobot](https://github.com/huggingface/lerobot) — target dataset format
