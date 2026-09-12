#!/usr/bin/env python3
"""
Interactive Joint Control with Viser Sliders

Real-time per-joint control for the A1X arm through a browser UI.
6 sliders (one per joint), bounded by URDF joint limits, send target
positions to the arm at a configurable rate.

Usage:
    python examples/motion/joint_control_smooth.py
    # then open http://localhost:8080/

SAFETY: Robot control is DISABLED by default. Toggle the checkbox in the
UI to send commands. Sliders can still be moved (and visualized) while
disabled.
"""

import sys
import os
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import viser
from viser.extras import ViserUrdf
import yourdfpy
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import JointState

import a1x_control


JOINT_NAMES = [f"arm_joint{i}" for i in range(1, 7)]

# From install/mobiman/lib/mobiman/configs/urdfs/a1x.urdf
JOINT_LIMITS = [
    (-2.8798, 2.8798),   # joint1
    ( 0.0000, 3.1416),   # joint2  (cannot go negative)
    (-3.3161, 0.0000),   # joint3  (cannot go positive)
    (-1.5708, 1.5708),   # joint4
    (-1.5708, 1.5708),   # joint5
    (-2.8798, 2.8798),   # joint6
]


def load_a1x_urdf() -> yourdfpy.URDF:
    """Load the A1X URDF with package:// mesh resolution."""
    urdf_path = Path("/home/ubuntu/projects/A1Xsdk/install/mobiman/lib/mobiman/configs/urdfs/a1x.urdf")

    def resolve(fname: str) -> str:
        prefix = "package://mobiman/"
        if fname.startswith(prefix):
            return str(Path("/home/ubuntu/projects/A1Xsdk/install/mobiman/share/mobiman") / fname[len(prefix):])
        return fname

    return yourdfpy.URDF.load(urdf_path, filename_handler=resolve)


def read_current_positions(controller: a1x_control.JointController) -> list[float]:
    """Read current joint positions as a length-6 list; fall back to 0 on missing joints."""
    joints = controller.get_joint_states() or {}
    return [float(joints.get(name, 0.0)) for name in JOINT_NAMES]


class HdasFeedbackReader:
    """Subscribes to /hdas/feedback_arm (real HDAS feedback, ~200Hz).

    /joint_states is published by the jointTracker and echoes commanded values,
    so it is NOT a reliable indicator of physical motion. /hdas/feedback_arm is
    the raw hardware feedback and is what we want to display back to the user.

    The subscription is attached to the controller's node so it shares the
    existing background executor (rclpy.spin on controller in a1x_control.py).
    """

    def __init__(self, controller: a1x_control.JointController):
        self._msg = None
        self._count = 0
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE,
        )
        controller.create_subscription(JointState, '/hdas/feedback_arm', self._cb, qos)

    def _cb(self, msg: JointState) -> None:
        self._msg = msg
        self._count += 1

    def positions(self) -> list[float | None]:
        m = self._msg
        if m is None or len(m.position) < 6:
            return [None] * 6
        return [float(v) for v in m.position[:6]]

    @property
    def count(self) -> int:
        return self._count


def clamp_to_limits(positions: list[float]) -> list[float]:
    return [
        max(lo, min(hi, p))
        for p, (lo, hi) in zip(positions, JOINT_LIMITS)
    ]


def main() -> None:
    print("=" * 60)
    print("A1X Interactive Joint Control (Viser)")
    print("=" * 60)

    print("[1/3] Initializing A1X control system...")
    controller = a1x_control.JointController()
    hdas_fb = HdasFeedbackReader(controller)
    if controller.wait_for_joint_states(timeout=10.0):
        current = read_current_positions(controller)
        print("       Initial joint positions (from /joint_states):")
        for name, pos in zip(JOINT_NAMES, current):
            print(f"         {name}: {pos:+.4f} rad")
    else:
        print("       WARNING: joint state not available yet, using zeros")
        current = [0.0] * 6

    # Give HDAS feedback subscriber a moment to receive first message
    t_start = time.time()
    while hdas_fb.count == 0 and time.time() - t_start < 3.0:
        time.sleep(0.1)
    print(f"       HDAS feedback: got {hdas_fb.count} messages in {time.time()-t_start:.1f}s")

    print("[2/3] Loading URDF for visualization...")
    urdf = load_a1x_urdf()

    print("[3/3] Starting Viser server...")
    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=2, height=2, cell_size=0.1)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/robot")
    urdf_vis.update_cfg(np.array(current + [0.0, 0.0]))  # 6 arm + 2 gripper

    with server.gui.add_folder("Robot Control"):
        enable_robot = server.gui.add_checkbox("Enable Robot Control", initial_value=False)
        rate_hz = server.gui.add_slider("Send Rate (Hz)", min=5.0, max=100.0, step=5.0, initial_value=30.0)
        smoothing_alpha = server.gui.add_slider(
            "Smoothing Alpha", min=0.05, max=1.0, step=0.05, initial_value=0.30
        )
        max_step = server.gui.add_slider(
            "Max Step per Cycle (rad)", min=0.005, max=0.20, step=0.005, initial_value=0.05,
        )
        status_display = server.gui.add_text("Status", initial_value="Robot control DISABLED")

    with server.gui.add_folder("Joint Sliders (target)"):
        joint_sliders = []
        for i, (lo, hi) in enumerate(JOINT_LIMITS):
            joint_sliders.append(
                server.gui.add_slider(
                    f"Joint {i+1} [{lo:+.2f}, {hi:+.2f}]",
                    min=float(lo),
                    max=float(hi),
                    step=0.01,
                    initial_value=float(np.clip(current[i], lo, hi)),
                )
            )

    with server.gui.add_folder("Feedback (measured, from /hdas/feedback_arm)"):
        feedback_displays = [
            server.gui.add_number(f"Joint {i+1} pos (rad)", initial_value=float(current[i]), disabled=True)
            for i in range(6)
        ]
        error_displays = [
            server.gui.add_number(f"Joint {i+1} err (cmd - measured)", initial_value=0.0, disabled=True)
            for i in range(6)
        ]
        fb_rate_display = server.gui.add_text("Feedback rate", initial_value="0 msg/s", disabled=True)

    with server.gui.add_folder("Actions"):
        sync_button = server.gui.add_button("Sync sliders <- current position")
        zero_button = server.gui.add_button("Set sliders to safe home")

        @sync_button.on_click
        def _(_):
            positions = read_current_positions(controller)
            for slider, pos, (lo, hi) in zip(joint_sliders, positions, JOINT_LIMITS):
                slider.value = float(np.clip(pos, lo, hi))
            print(f"Synced sliders to: {[round(p, 4) for p in positions]}")

        @zero_button.on_click
        def _(_):
            # Match the original start_position from the old joint_control_smooth.py
            home = [0.0, 0.0043, -0.1, -0.0347, -0.0055, 0.0013]
            for slider, pos, (lo, hi) in zip(joint_sliders, home, JOINT_LIMITS):
                slider.value = float(np.clip(pos, lo, hi))
            print(f"Sliders set to home: {home}")

    print()
    print("=" * 60)
    print("READY -> open http://localhost:8080/ in your browser")
    print("=" * 60)
    print("- Drag the joint sliders to set target angles")
    print("- Toggle 'Enable Robot Control' to actually send commands")
    print("- Adjust 'Max Step per Cycle' to cap per-tick jump (safer for big slider jumps)")
    print("- Ctrl+C to quit")
    print()

    smoothed = np.array(current, dtype=float)
    fb_count_prev = hdas_fb.count
    fb_time_prev = time.time()

    try:
        while True:
            loop_start = time.time()

            target = np.array([s.value for s in joint_sliders], dtype=float)
            target = np.array(clamp_to_limits(list(target)))

            alpha = float(smoothing_alpha.value)
            step_cap = float(max_step.value)

            # Smooth towards target
            proposed = alpha * target + (1.0 - alpha) * smoothed
            # Cap per-tick delta so a slider yank doesn't cause a huge single-step command
            delta = proposed - smoothed
            delta = np.clip(delta, -step_cap, step_cap)
            smoothed = smoothed + delta

            urdf_vis.update_cfg(np.concatenate([smoothed, np.zeros(2)]))

            # Real hardware feedback from HDAS (NOT /joint_states which echoes commands)
            fb = hdas_fb.positions()
            for i, val in enumerate(fb):
                if val is not None:
                    feedback_displays[i].value = round(val, 4)
                    error_displays[i].value = round(float(smoothed[i]) - val, 4)

            # Show feedback rate so user can confirm HDAS link is alive
            if loop_start - fb_time_prev >= 1.0:
                dt = loop_start - fb_time_prev
                rate = (hdas_fb.count - fb_count_prev) / dt
                fb_rate_display.value = f"{rate:.0f} msg/s"
                fb_count_prev = hdas_fb.count
                fb_time_prev = loop_start

            if enable_robot.value:
                ok = controller.set_joint_positions(list(smoothed))
                status_display.value = "Robot control ACTIVE" if ok else "Command send FAILED"
            else:
                status_display.value = "Robot control DISABLED (visualization only)"

            elapsed = time.time() - loop_start
            sleep_time = max(0.0, 1.0 / float(rate_hz.value) - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nShutting down...")


if __name__ == "__main__":
    main()
