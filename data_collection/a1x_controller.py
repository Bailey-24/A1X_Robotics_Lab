"""A1X ArmController adapter for the control_your_robot framework.

Wraps the existing a1x_control.JointController (ROS 2 node) to implement
the ArmController interface, enabling data collection via CollectAny.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

# Add control_your_robot to path
_CYR_SRC = str(Path(__file__).resolve().parent.parent / "refence_code" / "control_your_robot" / "src")
if _CYR_SRC not in sys.path:
    sys.path.insert(0, _CYR_SRC)

from robot.controller.arm_controller import ArmController

# ROS 2 message type for action tracking subscription
from sensor_msgs.msg import JointState


class A1XArmController(ArmController):
    """Adapter that wraps a1x_control.JointController as an ArmController.

    The adapter does NOT own the JointController — it receives a reference
    to the one already created by yoloe_grasp, avoiding duplicate ROS
    subscriptions and spin conflicts.

    For action tracking, it subscribes to the command topics on the shared
    node to capture the latest commanded targets.
    """

    def __init__(self, name: str, controller):
        """
        Args:
            name: Controller name (e.g. "a1x_arm").
            controller: a1x_control.JointController instance (shared).
        """
        super().__init__()
        self.name = name
        self._controller = controller

        # Last commanded targets (for action tracking)
        self._last_action_joints = np.zeros(6, dtype=np.float64)
        self._last_action_gripper = 0.0

        # Subscribe to command topics to track actions
        self._setup_action_tracking()

    def _setup_action_tracking(self):
        """Subscribe to command topics on the shared JointController node."""
        qos = self._controller.qos_profile

        self._controller.create_subscription(
            JointState,
            "/motion_target/target_joint_state_arm",
            self._action_joint_callback,
            qos,
        )
        self._controller.create_subscription(
            JointState,
            "/motion_target/target_position_gripper",
            self._action_gripper_callback,
            qos,
        )

    def _action_joint_callback(self, msg: JointState):
        if len(msg.position) >= 6:
            self._last_action_joints = np.array(msg.position[:6], dtype=np.float64)

    def _action_gripper_callback(self, msg: JointState):
        if len(msg.position) >= 1:
            # Normalize 0-100 → 0-1
            self._last_action_gripper = msg.position[0] / 100.0

    # ── ArmController interface ────────────────────────────────────────

    def get_state(self) -> Dict[str, Any]:
        """Return current arm state.

        Returns dict with keys:
            joint:   np.array([6]) — joint positions in radians
            gripper: np.array([1]) — gripper position normalized 0-1
            action:  np.array([7]) — last commanded [6 joints, 1 gripper]
        """
        # Read joint positions from cached ROS state (no spin_once needed)
        joints_dict = self._controller.get_joint_states()
        if joints_dict is not None:
            joint_array = np.array(
                [joints_dict.get(n, 0.0) for n in self._controller.joint_names],
                dtype=np.float64,
            )
        else:
            joint_array = np.zeros(6, dtype=np.float64)

        # Read gripper (0-100 scale → 0-1)
        gripper_raw = self._controller.get_gripper_state()
        gripper_norm = (gripper_raw / 100.0) if gripper_raw is not None else 0.0

        # Action: last commanded target
        action = np.concatenate([
            self._last_action_joints,
            [self._last_action_gripper],
        ])

        return {
            "joint": joint_array,
            "gripper": np.array([gripper_norm], dtype=np.float64),
            "action": action,
        }

    def set_up(self):
        """No-op — the JointController is already initialized by yoloe_grasp."""
        pass

    def set_joint(self, joint: np.ndarray):
        self._controller.set_joint_positions(joint.tolist())

    def set_gripper(self, gripper: np.ndarray):
        # Denormalize 0-1 → 0-100
        self._controller.set_gripper_position(float(gripper[0]) * 100.0)
