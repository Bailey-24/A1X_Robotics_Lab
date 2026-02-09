# Hand-Eye Calibration for A1X Robot

This folder contains tools for performing **eye-in-hand calibration** of the A1X robotic arm with a camera mounted on the end-effector (gripper).

## Files

| File | Description |
|------|-------------|
| `handeye_calibration.py` | Main calibration script with Viser IK control |
| `generate_aruco_marker.py` | Generate printable ArUco marker PNG |
| `aruco_42.png` | Pre-generated ArUco marker (ID 42, 4x4 dict) |
| `handeye_calibration.yaml` | Calibration result (T_ee_cam) |

---

## Quick Start

### 1. Print an ArUco Marker

Print `aruco_42.png` or generate a new one:

```bash
conda activate a1x_ros
cd /home/ubuntu/projects/A1Xsdk
python handeye/generate_aruco_marker.py --marker_id 42 --size 500 --output handeye/my_marker.png
```

**Important:** After printing, measure the BLACK SQUARE (not the white border) with a ruler. Convert to meters:
- 5 cm → `0.05`
- 7 cm → `0.07`
- 10 cm → `0.10`

### 2. Place the Marker

Attach the marker to a **fixed location** in the robot's workspace (e.g., table surface). The marker must NOT move during calibration.

### 3. Run Calibration

```bash
conda activate a1x_ros
cd /home/ubuntu/projects/A1Xsdk
python handeye/handeye_calibration.py --marker_id 42 --marker_size 0.10
```

### 4. Use the Viser UI

Open **http://localhost:8080/** in your browser. You will see:
- 3D robot visualization
- IK target gizmo (red sphere)
- Control panels

**Workflow:**

1. ✅ Check **"Enable Robot Control"**
2. 🖱️ Drag the **RED SPHERE** to move the robot arm
3. 👀 Watch the **OpenCV window** - ensure marker shows "DETECTED"
4. 📸 Click **"Capture Pose"** (need 10+ poses)
5. 🔄 Move robot to different angles and repeat
6. ✨ Click **"Compute Calibration"** when done

**Tips for good calibration:**
- Vary both position AND rotation
- Capture poses from different distances (20-50 cm)
- Avoid poses where marker is at extreme angles
- Make sure each pose is visually different

---

## Understanding the Output

The calibration produces `handeye_calibration.yaml`:

```yaml
transformation:
  rotation:       # 3x3 rotation matrix (camera → EE frame)
    - [r11, r12, r13]
    - [r21, r22, r23]
    - [r31, r32, r33]
  translation:    # [x, y, z] in meters (camera position relative to EE)
  quaternion_xyzw: [x, y, z, w]  # Same rotation as quaternion
```

### What is T_ee_cam?

This is the **fixed transformation** from the end-effector frame to the camera frame.

```
Camera (mounted on gripper)
       ↑
    T_ee_cam  ← This calibration result
       ↑
  End-Effector
       ↑
    FK (joint angles)
       ↑
   Robot Base
```

### How to Use It

To convert a point detected by the camera to robot base coordinates:

```python
import numpy as np
import yaml

# Load calibration
with open('handeye/handeye_calibration.yaml') as f:
    calib = yaml.safe_load(f)

R_ee_cam = np.array(calib['transformation']['rotation'])
t_ee_cam = np.array(calib['transformation']['translation'])

# Build 4x4 transform
T_ee_cam = np.eye(4)
T_ee_cam[:3, :3] = R_ee_cam
T_ee_cam[:3, 3] = t_ee_cam

# Get T_base_ee from robot FK (controller.get_ee_pose())
# Then: P_base = T_base_ee @ T_ee_cam @ [P_cam; 1]
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Marker not detected | Move camera closer, improve lighting, check marker ID |
| Poor calibration | Capture more diverse poses (vary angles significantly) |
| Robot jerky | Increase "Smoothing" slider in UI |
| IK fails | Keep target sphere within robot's reachable workspace |

---

## Command Reference

### handeye_calibration.py

```bash
python handeye/handeye_calibration.py \
    --marker_id 42 \
    --marker_size 0.10 \
    --dict_type DICT_4X4_50 \
    --output handeye/handeye_calibration.yaml
```

### generate_aruco_marker.py

```bash
python handeye/generate_aruco_marker.py \
    --marker_id 42 \
    --size 500 \
    --dict_type DICT_4X4_50 \
    --output handeye/marker.png
```

Available dictionaries: `DICT_4X4_50`, `DICT_4X4_100`, `DICT_5X5_50`, `DICT_6X6_50`, etc.
