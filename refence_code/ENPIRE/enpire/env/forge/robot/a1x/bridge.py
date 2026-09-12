#!/usr/bin/env python3
"""Expose an existing A1X ROS graph to ENPIRE over localhost HTTP."""

from __future__ import annotations

import argparse
import json
import logging
import math
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger("a1x_control")
JOINT_NAMES = [f"arm_joint{i}" for i in range(1, 7)]
JOINT_LOW = [-2.8798, 0.0, -3.3161, -1.5708, -1.5708, -2.8798]
JOINT_HIGH = [2.8798, 3.1416, 0.0, 1.5708, 1.5708, 2.8798]
HOME = [0.0, 0.0043, -0.1, -0.0347, -0.0055, 0.0013]
OBSERVATION = [0.0, 1.0, -0.93, 0.83, 0.0, 0.0]
MAX_FEEDBACK_AGE_S = 0.25
MAX_MOTION_DURATION_S = 10.0
MAX_SETTLE_TIMEOUT_S = 8.0


def _sdk_root() -> Path:
    explicit = Path(str(Path.cwd()))
    for candidate in (explicit, *Path(__file__).resolve().parents):
        if (candidate / "a1x_control.py").is_file():
            return candidate
    raise FileNotFoundError("Run inside A1Xsdk or set up the ENPIRE checkout under A1Xsdk")


def _vec(value: Any, length: int, name: str) -> list[float]:
    try:
        result = [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain {length} numbers") from exc
    if len(result) != length or not all(math.isfinite(item) for item in result):
        raise ValueError(f"{name} must contain {length} finite numbers")
    return result


class A1XRuntime:
    """Direct ROS adapter that never launches or owns the A1X driver stack."""

    def __init__(
        self,
        *,
        allow_motion: bool,
        workspace_min: list[float],
        workspace_max: list[float],
    ) -> None:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import JointState

        self._rclpy = rclpy
        if not rclpy.ok():
            rclpy.init()
        self.node = Node("enpire_a1x_bridge")
        self._joint_state_type = JointState
        self.allow_motion = allow_motion
        self.workspace_min = workspace_min
        self.workspace_max = workspace_max
        self._motion_lock = threading.Lock()
        self._kinematics_lock = threading.Lock()
        self._feedback_lock = threading.Lock()
        self._feedback_event = threading.Event()
        self._gripper_event = threading.Event()
        self._measured_joints: list[float] | None = None
        self._gripper_raw: float | None = None
        self._feedback_time = 0.0
        self._feedback_sequence = 0
        self._gripper_time = 0.0
        self._cancel_motion = threading.Event()
        self._ik_executor: Any | None = None

        command_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        feedback_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE,
        )
        gripper_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._joint_publisher = self.node.create_publisher(
            JointState, "/motion_target/target_joint_state_arm", command_qos
        )
        self._gripper_publisher = self.node.create_publisher(
            JointState, "/motion_target/target_position_gripper", command_qos
        )
        self._arm_subscription = self.node.create_subscription(
            JointState, "/hdas/feedback_arm", self._arm_feedback, feedback_qos
        )
        self._gripper_subscription = self.node.create_subscription(
            JointState, "/hdas/feedback_gripper", self._gripper_feedback, gripper_qos
        )
        self._spin_thread = threading.Thread(
            target=rclpy.spin, args=(self.node,), daemon=True, name="a1x-bridge-ros"
        )
        self._spin_thread.start()
        self._feedback_event.wait(timeout=5.0)

    def _arm_feedback(self, message: Any) -> None:
        if len(message.position) >= 6:
            with self._feedback_lock:
                self._measured_joints = [float(v) for v in message.position[:6]]
                self._feedback_time = time.monotonic()
                self._feedback_sequence += 1
            self._feedback_event.set()

    def _gripper_feedback(self, message: Any) -> None:
        if message.position:
            with self._feedback_lock:
                self._gripper_raw = float(message.position[0])
                self._gripper_time = time.monotonic()
            self._gripper_event.set()

    def _require_motion(self) -> None:
        if not self.allow_motion:
            raise PermissionError("Motion disabled; restart bridge with --allow-motion")
        self._require_fresh_feedback()
        if self._joint_publisher.get_subscription_count() < 1:
            raise RuntimeError("No A1X joint tracker subscribes to the motion target topic")

    def _require_motion_enabled(self) -> None:
        if not self.allow_motion:
            raise PermissionError("Motion disabled; restart bridge with --allow-motion")
        self._require_fresh_feedback()

    def _require_fresh_feedback(self) -> None:
        with self._feedback_lock:
            age = time.monotonic() - self._feedback_time
        if not self._feedback_event.is_set() or age > MAX_FEEDBACK_AGE_S:
            raise RuntimeError("A1X measured feedback is missing or stale")

    def _joints(self) -> list[float]:
        with self._feedback_lock:
            if self._measured_joints is None:
                raise RuntimeError("No measured /hdas/feedback_arm state available")
            return self._measured_joints.copy()

    def _check_joints(self, positions: Any) -> list[float]:
        joints = _vec(positions, 6, "positions")
        for index, (value, low, high) in enumerate(zip(joints, JOINT_LOW, JOINT_HIGH), 1):
            if not low <= value <= high:
                raise ValueError(f"arm_joint{index}={value:.4f} outside [{low:.4f}, {high:.4f}]")
        return joints

    def _check_position(self, position: Any) -> list[float]:
        xyz = _vec(position, 3, "position")
        for axis, value, low, high in zip("xyz", xyz, self.workspace_min, self.workspace_max):
            if not low <= value <= high:
                raise ValueError(f"{axis}={value:.4f} outside workspace [{low:.4f}, {high:.4f}]")
        return xyz

    def _kinematics(self) -> Any:
        if self._ik_executor is None:
            root = _sdk_root()
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))
            from examples.yoloe_grasp.grasp_pipeline.ik_executor import IKExecutor

            self._ik_executor = IKExecutor(interpolation_type="cosine")
        return self._ik_executor

    def _fk(self, joints: list[float]) -> tuple[list[float], list[float]]:
        import numpy as np

        with self._kinematics_lock:
            executor = self._kinematics()
            target_idx = executor.robot.links.names.index(executor.target_link_name)
            pose = executor.robot.forward_kinematics(np.array(joints + [0.0, 0.0]))[target_idx]
        return [float(v) for v in pose[4:7]], [float(v) for v in (pose[1], pose[2], pose[3], pose[0])]

    def health(self) -> dict[str, Any]:
        with self._feedback_lock:
            feedback_age = time.monotonic() - self._feedback_time
        measured = self._feedback_event.is_set() and feedback_age <= MAX_FEEDBACK_AGE_S
        tracker = self._joint_publisher.get_subscription_count() > 0
        return {
            "success": measured,
            "motion_enabled": self.allow_motion,
            "measured_joints": measured,
            "joint_tracker_connected": tracker,
            "gripper_feedback": self._gripper_event.is_set(),
            "feedback_age_s": round(feedback_age, 4),
        }

    def state(self) -> dict[str, Any]:
        joints = self._joints()
        ee_pos, ee_quat = self._fk(joints)
        with self._feedback_lock:
            gripper = self._gripper_raw
        return {
            "joint_pos": joints,
            "gripper_pos": None if gripper is None else float(gripper) / 100.0,
            "ee_pos": ee_pos,
            "ee_quat": ee_quat,
        }

    def _publish_joints(self, positions: list[float]) -> None:
        message = self._joint_state_type()
        message.header.stamp = self.node.get_clock().now().to_msg()
        message.name = JOINT_NAMES
        message.position = positions
        self._joint_publisher.publish(message)

    def _wait_for_joints(self, target: list[float], timeout: float, tolerance: float) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._cancel_motion.is_set():
                return False
            self._require_fresh_feedback()
            if max(abs(a - b) for a, b in zip(self._joints(), target)) <= tolerance:
                return True
            time.sleep(0.05)
        return False

    def move_joints(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._require_motion()
        target = self._check_joints(payload["positions"])
        steps = max(2, int(payload.get("steps", 30)))
        rate_hz = min(30.0, max(2.0, float(payload.get("rate_hz", 20.0))))
        timeout = min(MAX_SETTLE_TIMEOUT_S, max(0.5, float(payload.get("timeout", 4.0))))
        if steps / rate_hz > MAX_MOTION_DURATION_S:
            raise ValueError(f"joint interpolation may not exceed {MAX_MOTION_DURATION_S:.0f}s")
        with self._motion_lock:
            self._cancel_motion.clear()
            start = self._joints()
            for index in range(1, steps + 1):
                if self._cancel_motion.is_set():
                    self._publish_joints(self._joints())
                    raise RuntimeError("A1X motion stopped")
                self._require_fresh_feedback()
                raw = index / steps
                alpha = 0.5 * (1.0 - math.cos(math.pi * raw))
                self._publish_joints([a + alpha * (b - a) for a, b in zip(start, target)])
                time.sleep(1.0 / rate_hz)
            settled = self._wait_for_joints(target, timeout, 0.03)
        if not settled:
            raise RuntimeError("A1X measured joints did not converge before timeout")
        return {"success": True, "settled": True, "joint_pos": self._joints()}

    def set_gripper(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._require_motion_enabled()
        if self._gripper_publisher.get_subscription_count() < 1:
            raise RuntimeError("No A1X gripper controller subscribes to the target topic")
        position = float(payload["position"])
        if not math.isfinite(position) or not 0.0 <= position <= 1.0:
            raise ValueError("position must be in [0, 1]")
        message = self._joint_state_type()
        message.header.stamp = self.node.get_clock().now().to_msg()
        message.name = ["gripper_joint"]
        message.position = [position * 100.0]
        with self._motion_lock:
            with self._feedback_lock:
                previous_feedback_time = self._gripper_time
                opening_at_command = self._gripper_raw
            self._gripper_publisher.publish(message)
            timeout = min(5.0, max(0.1, float(payload.get("settle_s", 2.0))))
            deadline = time.monotonic() + timeout
            # A closing gripper that stops partway has grabbed an object: that
            # is a successful grasp, not a failure to reach the commanded
            # position. Detection: the fingers must first demonstrably move
            # toward the target (motion_started, guards against controller
            # start-up latency reading as a stall) and then freeze for
            # STALL_WINDOW_S while the command is a close.
            stall_window_s = 0.4
            stall_tolerance = 2.0
            motion_start_threshold = 3.0
            direction = (
                1.0
                if opening_at_command is None or position * 100.0 >= opening_at_command
                else -1.0
            )
            samples: list[tuple[float, float]] = []
            stalled = False
            motion_started = opening_at_command is None
            while time.monotonic() < deadline:
                with self._feedback_lock:
                    actual = self._gripper_raw
                    updated = self._gripper_time > previous_feedback_time
                now = time.monotonic()
                if updated and actual is not None:
                    if abs(actual - position * 100.0) <= 10.0:
                        break
                    if (
                        not motion_started
                        and opening_at_command is not None
                        and (actual - opening_at_command) * direction >= motion_start_threshold
                    ):
                        motion_started = True
                    samples.append((now, actual))
                    while samples and now - samples[0][0] > stall_window_s:
                        samples.pop(0)
                    if (
                        motion_started
                        and len(samples) >= 2
                        and now - samples[0][0] >= stall_window_s - 0.05
                        and abs(actual - samples[0][1]) <= stall_tolerance
                    ):
                        stalled = True
                        break
                time.sleep(0.05)
            else:
                raise RuntimeError("A1X gripper did not converge before timeout")
        with self._feedback_lock:
            actual = self._gripper_raw
        return {
            "success": True,
            "gripper_pos": None if actual is None else actual / 100.0,
            "stalled": stalled,
        }

    def move_ee_absolute(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._require_motion()
        import numpy as np

        xyz = self._check_position(payload["position"])
        quat_xyzw = payload.get("quat_xyzw")
        if quat_xyzw is None:
            current = self.state()["ee_quat"]
            wxyz = np.array([current[3], *current[:3]], dtype=np.float64)
        else:
            quat = np.asarray(_vec(quat_xyzw, 4, "quat_xyzw"), dtype=np.float64)
            norm = float(np.linalg.norm(quat))
            if norm < 1e-9:
                raise ValueError("quat_xyzw norm is zero")
            quat /= norm
            wxyz = np.array([quat[3], quat[0], quat[1], quat[2]])
        with self._kinematics_lock:
            executor = self._kinematics()
            solution = executor.solve_ik(
                np.asarray(xyz), wxyz, np.asarray(self._joints())
            )
        if solution is None:
            raise RuntimeError("A1X IK failed")
        arm_solution = np.asarray(solution[:6], dtype=np.float64)
        if not np.all(np.isfinite(arm_solution)) or np.any(arm_solution < JOINT_LOW) or np.any(
            arm_solution > JOINT_HIGH
        ):
            raise RuntimeError("A1X IK returned invalid or out-of-limit joints")
        with self._kinematics_lock:
            target_idx = executor.robot.links.names.index(executor.target_link_name)
            achieved = executor.robot.forward_kinematics(np.asarray(solution))[target_idx]
        position_error = float(np.linalg.norm(np.asarray(achieved[4:7]) - np.asarray(xyz)))
        achieved_wxyz = np.asarray(achieved[:4], dtype=np.float64)
        orientation_error = float(
            2.0
            * math.acos(
                min(1.0, abs(float(np.dot(achieved_wxyz, wxyz))) / np.linalg.norm(achieved_wxyz))
            )
        )
        if position_error > 0.01 or orientation_error > math.radians(10.0):
            raise RuntimeError(
                f"A1X IK verification failed: {position_error:.4f}m, "
                f"{math.degrees(orientation_error):.1f}deg"
            )
        options = dict(payload)
        options["positions"] = np.asarray(solution[:6], dtype=float).tolist()
        result = self.move_joints(options)
        result["state"] = self.state()
        return result

    def move_ee_relative(self, payload: dict[str, Any]) -> dict[str, Any]:
        delta = _vec(payload["delta"], 3, "delta")
        current = self.state()["ee_pos"]
        options = dict(payload)
        options["position"] = [value + offset for value, offset in zip(current, delta)]
        options.pop("delta", None)
        return self.move_ee_absolute(options)

    def stop(self) -> dict[str, Any]:
        self._cancel_motion.set()
        if self.allow_motion and self._feedback_event.is_set():
            self._publish_joints(self._joints())
        return {"success": True, "status": "stop_requested"}

    def close(self) -> None:
        self.node.destroy_node()
        if self._rclpy.ok():
            self._rclpy.shutdown()


class Handler(BaseHTTPRequestHandler):
    server_version = "A1XBridge/1.0"

    def _respond(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _payload(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        return json.loads(self.rfile.read(length) or b"{}")

    def _dispatch(self, method: str) -> None:
        runtime: A1XRuntime = self.server.runtime  # type: ignore[attr-defined]
        routes = {
            "/health": runtime.health,
            "/state": lambda: {"success": True, "state": runtime.state()},
            "/move_joints": lambda: runtime.move_joints(self._payload()),
            "/set_gripper": lambda: runtime.set_gripper(self._payload()),
            "/move_ee_absolute": lambda: runtime.move_ee_absolute(self._payload()),
            "/move_ee_relative": lambda: runtime.move_ee_relative(self._payload()),
            "/go_home": lambda: runtime.move_joints({"positions": HOME}),
            "/move_to_observation": lambda: runtime.move_joints({"positions": OBSERVATION}),
            "/stop": runtime.stop,
        }
        route = routes.get(self.path)
        if route is None or (method == "GET") != (self.path in {"/health", "/state"}):
            self._respond(404, {"success": False, "error": "unknown route"})
            return
        try:
            self._respond(200, route())
        except PermissionError as exc:
            self._respond(403, {"success": False, "error": str(exc)})
        except (KeyError, TypeError, ValueError) as exc:
            self._respond(400, {"success": False, "error": str(exc)})
        except Exception as exc:
            LOGGER.exception("A1X bridge request failed")
            self._respond(500, {"success": False, "error": str(exc)})

    def do_GET(self) -> None:
        self._dispatch("GET")

    def do_POST(self) -> None:
        self._dispatch("POST")

    def log_message(self, fmt: str, *args: Any) -> None:
        LOGGER.info("bridge: " + fmt, *args)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=11337)
    parser.add_argument("--allow-motion", action="store_true")
    parser.add_argument("--workspace-min", default="0.05,-0.45,0.02")
    parser.add_argument("--workspace-max", default="0.65,0.45,0.65")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    runtime = A1XRuntime(
        allow_motion=args.allow_motion,
        workspace_min=_vec(args.workspace_min.split(","), 3, "workspace-min"),
        workspace_max=_vec(args.workspace_max.split(","), 3, "workspace-max"),
    )
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.runtime = runtime  # type: ignore[attr-defined]
    LOGGER.info(
        "A1X bridge listening on http://%s:%d (motion=%s, workspace min=%s max=%s)",
        args.host,
        args.port,
        args.allow_motion,
        args.workspace_min,
        args.workspace_max,
    )

    def _terminate(signum: int, _frame: Any) -> None:
        # rclpy swallows SIGTERM by invalidating its context without exiting;
        # shut the HTTP server down from a helper thread (serve_forever's
        # shutdown() deadlocks when called from its own thread) and exit.
        LOGGER.info("A1X bridge received signal %d, shutting down", signum)
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, _terminate)
    signal.signal(signal.SIGINT, _terminate)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        runtime.close()


if __name__ == "__main__":
    main()
