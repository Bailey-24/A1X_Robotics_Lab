"""A1X Recording Robot — combines the A1X arm controller + D405 sensor.

Implements the control_your_robot Robot interface so that ``robot.get()``
returns both joint state and camera data, and ``robot.finish()`` writes
an HDF5 episode via CollectAny.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Add control_your_robot to path
_CYR_SRC = str(Path(__file__).resolve().parent.parent / "refence_code" / "control_your_robot" / "src")
if _CYR_SRC not in sys.path:
    sys.path.insert(0, _CYR_SRC)

from robot.robot.base_robot import Robot

from data_collection.a1x_controller import A1XArmController
from data_collection.d405_sensor import D405Sensor


class A1XRecordingRobot(Robot):
    """Single-arm A1X robot with wrist-mounted D405 camera.

    This class does NOT own the JointController — it receives a reference
    to the one created by yoloe_grasp (shared across the application).
    """

    def __init__(
        self,
        controller,
        condition: Dict[str, Any],
        camera_serial: Optional[str] = None,
        start_episode: int = 0,
    ):
        """
        Args:
            controller: a1x_control.JointController instance (shared).
            condition: CollectAny condition dict (save_path, task_name, etc.)
            camera_serial: RealSense D405 serial number (None = auto-detect).
            start_episode: Episode index to start from.
        """
        super().__init__(
            condition=condition,
            move_check=False,  # record ALL frames (including stationary)
            start_episode=start_episode,
        )
        self.name = "a1x_robot"
        self._joint_controller = controller

        # Create adapter instances
        self._arm = A1XArmController("a1x_arm", controller)
        self._cam = D405Sensor("cam_wrist", serial=camera_serial)

        self.controllers = {
            "arm": {"a1x_arm": self._arm},
        }
        self.sensors = {
            "image": {"cam_wrist": self._cam},
        }

    def set_up(
        self,
        camera_width: int = 640,
        camera_height: int = 480,
        camera_fps: int = 30,
    ):
        """Initialize controller adapter and start D405 pipeline."""
        super().set_up()

        self._arm.set_up()
        self._cam.set_up(
            width=camera_width,
            height=camera_height,
            fps=camera_fps,
            is_depth=True,
        )

        # Configure what data to collect
        self.set_collect_type({
            "arm": ["joint", "gripper", "action"],
            "image": ["color"],
        })

    def get_d405_sensor(self) -> D405Sensor:
        """Return the D405 sensor (for detection frame access)."""
        return self._cam

    def teardown(self):
        """Stop the D405 sensor pipeline."""
        self._cam.stop()
