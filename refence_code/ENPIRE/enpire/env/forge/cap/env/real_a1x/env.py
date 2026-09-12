"""Real Galaxea A1X environment backed by a separate ROS bridge process."""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from enpire.env.forge.cap.env.base.profile import RobotProfile, a1x_profile
from enpire.env.forge.robot.a1x.client import A1XBridgeClient


def _find_sdk_root() -> Path:
    explicit = os.environ.get("A1X_SDK_ROOT")
    if explicit:
        return Path(explicit).expanduser().resolve()
    for parent in Path(__file__).resolve().parents:
        if (parent / "a1x_control.py").is_file():
            return parent
    raise FileNotFoundError("Set A1X_SDK_ROOT to the A1X SDK checkout")


def _quat_xyzw_to_matrix(quat: np.ndarray) -> np.ndarray:
    x, y, z, w = quat / np.linalg.norm(quat)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


class _CameraThread:
    def __init__(self, name: str, resolution: tuple[int, int], fps: int) -> None:
        from enpire.env.forge.robot.realsense import RealSenseCamera, get_device_info

        serial = os.environ.get(f"CAP_{name.upper()}_REALSENSE_SERIAL") or os.environ.get(
            "CAP_REALSENSE_SERIAL"
        )
        if serial is None:
            devices = get_device_info()
            if len(devices) != 1:
                raise RuntimeError(
                    f"Set CAP_{name.upper()}_REALSENSE_SERIAL; found {len(devices)} RealSense devices"
                )
            serial = next(iter(devices))
        import pyrealsense2 as rs

        device = next(
            (
                item
                for item in rs.context().query_devices()
                if item.get_info(rs.camera_info.serial_number) == serial
            ),
            None,
        )
        if device is None or "D405" not in device.get_info(rs.camera_info.name):
            raise RuntimeError(f"Configured camera {serial!r} is not an Intel RealSense D405")
        self._camera = RealSenseCamera(
            device_id=serial,
            resolution=resolution,
            fps=fps,
            enable_depth=True,
            align_depth_to_color=True,
        )
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._running = True
        self._rgb: np.ndarray | None = None
        self._depth: np.ndarray | None = None
        self._intrinsics: dict[str, float] | None = None
        self._timestamp = 0.0
        self._error: Exception | None = None
        self._thread = threading.Thread(target=self._worker, daemon=True, name="a1x-d405")
        self._thread.start()
        if not self._ready.wait(timeout=8.0):
            self.close()
            raise TimeoutError(f"Camera {name!r} timed out waiting for an RGB-D frame")
        if self._error is not None:
            self.close()
            raise RuntimeError(f"Camera {name!r} failed: {self._error}")

    def _worker(self) -> None:
        while self._running:
            try:
                frame = self._camera.read()
                with self._lock:
                    self._rgb = np.asarray(frame.images["rgb"], dtype=np.uint8).copy()
                    self._depth = np.asarray(frame.depth, dtype=np.float32).copy()
                    self._intrinsics = dict(frame.intrinsics or {})
                    self._timestamp = time.monotonic()
                    self._error = None
                self._ready.set()
            except Exception as exc:
                self._error = exc
                self._ready.set()
                if self._running:
                    time.sleep(0.1)

    def rgb(self) -> np.ndarray:
        return self.snapshot()[0]

    def snapshot(self) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
        with self._lock:
            if self._rgb is None or self._depth is None or not self._intrinsics:
                raise RuntimeError("No complete D405 RGB-D frame available")
            if time.monotonic() - self._timestamp > 1.0:
                raise RuntimeError("D405 RGB-D frame is stale")
            return self._rgb.copy(), self._depth.copy(), dict(self._intrinsics)

    def depth(self) -> np.ndarray:
        return self.snapshot()[1]

    def intrinsics(self) -> dict[str, float]:
        return self.snapshot()[2]

    def close(self) -> None:
        self._running = False
        if hasattr(self, "_thread"):
            self._thread.join(timeout=2.0)
        self._camera.stop()


class RealA1XEnv:
    """Single-arm ENPIRE environment using A1X motion and D405 calibration."""

    def __init__(
        self,
        *,
        bridge_url: str | None = None,
        enable_cameras: bool = True,
        camera_name: str = "wrist",
        camera_resolution: tuple[int, int] = (640, 480),
        camera_fps: int = 15,
        calibration_path: str | Path | None = None,
    ) -> None:
        self._profile: RobotProfile = a1x_profile()
        self._camera_name = camera_name
        self._last_rendered_rgbd: dict[str, tuple[np.ndarray, np.ndarray, dict[str, float]]] = {}
        self._bridge = A1XBridgeClient(
            bridge_url or os.environ.get("A1X_BRIDGE_URL", "http://127.0.0.1:11337")
        )
        health = self._bridge.health()
        if not health.get("success", False):
            raise RuntimeError(f"A1X bridge is not healthy: {health}")
        path = Path(calibration_path).expanduser() if calibration_path else (
            _find_sdk_root() / "examples" / "handeye" / "handeye_calibration.yaml"
        )
        with path.open(encoding="utf-8") as stream:
            transform = yaml.safe_load(stream)["transformation"]
        self._T_ee_from_camera = np.eye(4, dtype=np.float64)
        self._T_ee_from_camera[:3, :3] = np.asarray(transform["rotation"], dtype=np.float64)
        self._T_ee_from_camera[:3, 3] = np.asarray(transform["translation"], dtype=np.float64)
        self._camera = (
            _CameraThread(camera_name, tuple(camera_resolution), int(camera_fps))
            if enable_cameras
            else None
        )

    def _check_side(self, side: str) -> None:
        if side != "right":
            raise ValueError("A1X exposes one arm named 'right'")

    def step(self) -> None:
        return None

    def get_observations(self, side: str) -> dict[str, np.ndarray]:
        self._check_side(side)
        state = self._bridge.state()
        gripper = state.get("gripper_pos")
        return {
            "joint_pos": np.asarray(state["joint_pos"], dtype=np.float64).reshape(6),
            "gripper_pos": np.asarray(
                [0.0 if gripper is None else gripper], dtype=np.float64
            ),
            "ee_pos": np.asarray(state["ee_pos"], dtype=np.float64).reshape(3),
            "ee_quat": np.asarray(state["ee_quat"], dtype=np.float64).reshape(4),
        }

    get_arm_observation = get_observations

    def command_arm(self, side: str, cmd: dict[str, Any]) -> None:
        self._check_side(side)
        target = np.asarray(cmd["pos"], dtype=np.float64).reshape(7)
        self._bridge.move_joints(target[:6].tolist())
        self._bridge.set_gripper(float(target[6]))

    def command_joint_state(self, side: str, state: dict[str, Any]) -> None:
        self.command_arm(side, state)

    def move_joint_keypoints(
        self,
        side: str,
        timestamps: Any,
        joint_positions: Any,
        gripper_positions: Any = None,
    ) -> dict[str, Any]:
        self._check_side(side)
        ts = np.asarray(timestamps, dtype=np.float64).reshape(-1)
        joints = np.asarray(joint_positions, dtype=np.float64)
        if ts.size == 0 or joints.shape != (ts.size, 6):
            raise ValueError("joint_positions must have shape (N, 6)")
        if np.any(np.diff(ts) < 0):
            raise ValueError("timestamps must be monotonically increasing")
        for index, target in enumerate(joints):
            duration = max(0.1, float(ts[index] - (ts[index - 1] if index else 0.0)))
            steps = max(2, int(round(duration * self._profile.control_freq_hz)))
            self._bridge.move_joints(target.tolist(), steps=steps)
            if gripper_positions is not None:
                value = float(np.asarray(gripper_positions).reshape(-1)[index])
                self._bridge.set_gripper(value, settle_s=0.0)
        return {"success": True, "waypoints": int(ts.size)}

    def set_gripper(self, side: str, pos: float, *_: Any) -> dict[str, Any]:
        self._check_side(side)
        return self._bridge.set_gripper(float(pos))

    def go_home(self) -> dict[str, Any]:
        return self._bridge.go_home()

    def move_to_observation(self) -> dict[str, Any]:
        return self._bridge.move_to_observation()

    def move_ee_absolute(
        self, position: Any, quat_xyzw: Any = None, **options: Any
    ) -> dict[str, Any]:
        pos = np.asarray(position, dtype=np.float64).reshape(3).tolist()
        quat = None if quat_xyzw is None else np.asarray(quat_xyzw, dtype=np.float64).reshape(4).tolist()
        return self._bridge.move_ee_absolute(pos, quat, **options)

    def move_ee_relative(self, delta: Any, **options: Any) -> dict[str, Any]:
        return self._bridge.move_ee_relative(
            np.asarray(delta, dtype=np.float64).reshape(3).tolist(), **options
        )

    def _require_camera(self, camera_name: str) -> _CameraThread:
        if camera_name != self._camera_name:
            raise ValueError(f"Unknown A1X camera {camera_name!r}; use {self._camera_name!r}")
        if self._camera is None:
            raise RuntimeError("A1X camera disabled in this process")
        return self._camera

    def render_rgb(self, camera_name: str) -> np.ndarray:
        snapshot = self._require_camera(camera_name).snapshot()
        self._last_rendered_rgbd[camera_name] = snapshot
        return snapshot[0].copy()

    def render_depth(self, camera_name: str) -> np.ndarray:
        return self._require_camera(camera_name).depth()

    def last_rendered_rgbd(
        self, camera_name: str
    ) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
        try:
            rgb, depth, intrinsics = self._last_rendered_rgbd[camera_name]
        except KeyError as exc:
            raise RuntimeError(f"No captured RGB-D snapshot for {camera_name!r}") from exc
        return rgb.copy(), depth.copy(), dict(intrinsics)

    def get_camera_intrinsics(self, camera_name: str) -> list[float]:
        intr = self._require_camera(camera_name).intrinsics()
        return [intr["fx"], intr["fy"], intr["cx"], intr["cy"]]

    def get_camera_extrinsics(self, camera_name: str) -> dict[str, Any]:
        if camera_name != self._camera_name:
            raise ValueError(f"Unknown A1X camera {camera_name!r}; use {self._camera_name!r}")
        obs = self.get_observations("right")
        T_base_from_ee = np.eye(4, dtype=np.float64)
        T_base_from_ee[:3, :3] = _quat_xyzw_to_matrix(obs["ee_quat"])
        T_base_from_ee[:3, 3] = obs["ee_pos"]
        T_base_from_camera = T_base_from_ee @ self._T_ee_from_camera
        return {
            "position": T_base_from_camera[:3, 3].tolist(),
            "rotation": T_base_from_camera[:3, :3].tolist(),
            "needs_optical_flip": False,
        }

    def set_recorder(self, recorder: Any) -> None:
        _ = recorder

    def close(self) -> None:
        if self._camera is not None:
            self._camera.close()

    def stop(self) -> dict[str, Any]:
        return self._bridge.stop()
