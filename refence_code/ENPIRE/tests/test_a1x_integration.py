from __future__ import annotations

import types
from pathlib import Path

import numpy as np
import yaml

from enpire.env.forge.cap.agent.robot_adapters.real_a1x import RealA1XAdapter
from enpire.env.forge.cap.agent.tools.base import ArmState, RobotState
from enpire.env.forge.cap.agent.tools.native import GetRobotStateTool
from enpire.env.forge.cap.env.base.profile import a1x_profile
from enpire.env.forge.cap.env.real_a1x import env as env_module
from enpire.env.forge.cap.env.real_a1x.skills import make_namespace


class _Bridge:
    def __init__(self, *args, **kwargs) -> None:
        self.calls: list[tuple] = []

    def health(self):
        return {"success": True, "motion_enabled": False}

    def state(self):
        return {
            "joint_pos": [0.0, 1.0, -0.93, 0.83, 0.0, 0.0],
            "gripper_pos": 0.6,
            "ee_pos": [0.4, 0.0, 0.3],
            "ee_quat": [0.0, 0.0, 0.0, 1.0],
        }

    def move_joints(self, positions, **options):
        self.calls.append(("move_joints", positions, options))
        return {"success": True}

    def set_gripper(self, position, **options):
        self.calls.append(("set_gripper", position, options))
        return {"success": True}

    def move_ee_absolute(self, position, quat_xyzw=None, **options):
        self.calls.append(("move_ee_absolute", position, quat_xyzw, options))
        return {"success": True}

    def move_ee_relative(self, delta, **options):
        self.calls.append(("move_ee_relative", delta, options))
        return {"success": True}

    def go_home(self):
        self.calls.append(("go_home",))
        return {"success": True}

    def move_to_observation(self):
        self.calls.append(("move_to_observation",))
        return {"success": True}

    def stop(self):
        self.calls.append(("stop",))
        return {"success": True}


class _Camera:
    def rgb(self):
        return np.zeros((4, 5, 3), dtype=np.uint8)

    def depth(self):
        return np.full((4, 5), 0.3, dtype=np.float32)

    def intrinsics(self):
        return {"fx": 100.0, "fy": 101.0, "cx": 2.0, "cy": 1.5}

    def snapshot(self):
        return self.rgb(), self.depth(), self.intrinsics()

    def close(self):
        return None


def _calibration(tmp_path: Path) -> Path:
    path = tmp_path / "handeye.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "transformation": {
                    "rotation": np.eye(3).tolist(),
                    "translation": [0.01, 0.02, 0.03],
                }
            }
        ),
        encoding="utf-8",
    )
    return path


def _env(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(env_module, "A1XBridgeClient", _Bridge)
    env = env_module.RealA1XEnv(
        enable_cameras=False,
        calibration_path=_calibration(tmp_path),
    )
    env._camera = _Camera()
    return env


def test_a1x_profile_is_single_six_dof_arm() -> None:
    profile = a1x_profile()
    assert profile.arm_names == ("right",)
    assert profile.arms["right"].dof == 6
    assert profile.camera_names == ("wrist",)
    assert np.allclose(
        profile.arms["right"].home_joint_pos,
        [0.0, 0.0043, -0.1, -0.0347, -0.0055, 0.0013],
    )


def test_a1x_env_state_and_dynamic_camera_extrinsics(monkeypatch, tmp_path: Path) -> None:
    env = _env(monkeypatch, tmp_path)
    obs = env.get_observations("right")
    assert obs["joint_pos"].shape == (6,)
    assert obs["gripper_pos"].tolist() == [0.6]
    assert obs["ee_quat"].tolist() == [0.0, 0.0, 0.0, 1.0]

    extrinsics = env.get_camera_extrinsics("wrist")
    assert extrinsics["needs_optical_flip"] is False
    assert np.allclose(extrinsics["position"], [0.41, 0.02, 0.33])
    assert np.allclose(extrinsics["rotation"], np.eye(3))
    assert env.render_rgb("wrist").dtype == np.uint8
    assert env.render_depth("wrist").dtype == np.float32


def test_native_robot_state_iterates_profile_arms(monkeypatch, tmp_path: Path) -> None:
    env = _env(monkeypatch, tmp_path)
    result = GetRobotStateTool(env=env).execute()
    assert result.success is True
    assert tuple(result.data.arms) == ("right",)
    assert result.data.left_joint_pos == []
    assert result.data.right_gripper_pos == 0.6


def test_a1x_motion_calls_use_single_arm_and_normalized_gripper(monkeypatch, tmp_path: Path) -> None:
    env = _env(monkeypatch, tmp_path)
    env.command_arm("right", {"pos": np.array([0, 1, -0.9, 0.8, 0, 0, 0.25])})
    assert env._bridge.calls == [
        ("move_joints", [0.0, 1.0, -0.9, 0.8, 0.0, 0.0], {}),
        ("set_gripper", 0.25, {}),
    ]


def test_a1x_env_rejects_nonexistent_left_arm(monkeypatch, tmp_path: Path) -> None:
    env = _env(monkeypatch, tmp_path)
    try:
        env.get_observations("left")
    except ValueError as exc:
        assert "one arm" in str(exc)
    else:
        raise AssertionError("left arm should not be exposed on A1X")


def test_a1x_namespace_excludes_collision_planner(monkeypatch, tmp_path: Path) -> None:
    env = _env(monkeypatch, tmp_path)
    namespace = make_namespace(env, cfg=type("Config", (), {"task": "test"})())
    assert "move_ee_absolute" in namespace
    assert "compute_topdown_grasp" in namespace
    assert "freespace_move" not in namespace
    assert "nudge" not in namespace


def test_a1x_adapter_rejects_multiple_real_hardware_seeds() -> None:
    adapter = RealA1XAdapter()
    try:
        adapter.child_env(n_seeds=2)
    except ValueError as exc:
        assert "exactly one" in str(exc)
    else:
        raise AssertionError("A1X must reject multiple real-hardware seeds")


def test_robot_state_supports_dict_style_access() -> None:
    """LLM-generated code often uses ``state["arms"]["right"]["ee_pos"]``."""
    arm = ArmState(
        joint_pos=[0.0] * 6,
        gripper_pos=0.5,
        ee_pos=[0.3, 0.0, 0.2],
        ee_quat=[0.0, 0.0, 0.0, 1.0],
        ee_rpy=[0.0, 0.0, 0.0],
    )
    state = RobotState(arms={"right": arm})

    # Canonical attribute access.
    assert state.arms["right"].ee_pos == [0.3, 0.0, 0.2]

    # Dict-style access used by generated policy code.
    assert state["arms"]["right"]["ee_pos"] == [0.3, 0.0, 0.2]
    assert state["arms"]["right"]["gripper_pos"] == 0.5
    assert state["arms"]["right"]["joint_pos"] == [0.0] * 6

    # Backward-compatible direct properties still work.
    assert state.right_ee_pos == [0.3, 0.0, 0.2]

    # Unknown keys raise KeyError, not confusing AttributeErrors.
    try:
        state["arms"]["right"]["elbow_pos"]
    except KeyError:
        pass
    else:
        raise AssertionError("unknown field should raise KeyError")


def test_a1x_adapter_forwards_vlm_settings_to_children() -> None:
    """The trusted verifier runs in run_script children; VLM settings must
    be forwarded or the child falls back to NVIDIA-gateway model names."""
    adapter = RealA1XAdapter()

    class _Reward:
        vlm_backend = "nvidia"
        vlm_model = None

    class _Reflection:
        vlm_backend = "nvidia"
        vlm_model = None

    class _Cfg:
        reward = _Reward()
        reflection = _Reflection()

    overrides = adapter.run_script_overrides(_Cfg())
    assert "reward.vlm_backend=nvidia" in overrides
    assert "reward.vlm_model=null" in overrides
    assert "reflection.vlm_backend=nvidia" in overrides
    assert "reflection.vlm_model=null" in overrides
    # Without cfg, still forwards base robot group only.
    assert adapter.run_script_overrides(None) == ["robot=real_a1x"]


class _FakeGripperPublisher:
    count = 1

    def get_subscription_count(self) -> int:
        return self.count

    def publish(self, message) -> None:
        _ = message




def _make_gripper_runtime(motion: bool = True):
    """Build an A1XRuntime without ROS for gripper-convergence tests."""
    from enpire.env.forge.robot.a1x.bridge import A1XRuntime

    class _FakeJointState:
        header = types.SimpleNamespace(stamp=None)
        name = None
        position = None

    runtime = object.__new__(A1XRuntime)
    runtime.allow_motion = motion
    runtime._joint_state_type = _FakeJointState
    runtime._gripper_publisher = _FakeGripperPublisher()
    runtime._motion_lock = __import__("threading").Lock()
    runtime._feedback_lock = __import__("threading").Lock()
    runtime._gripper_raw = 100.0
    runtime._gripper_time = __import__("time").monotonic()
    runtime._feedback_time = __import__("time").monotonic()
    runtime._joints_time = __import__("time").monotonic()
    runtime._feedback_event = __import__("threading").Event()
    runtime._feedback_event.set()
    runtime._gripper_event = __import__("threading").Event()
    runtime._gripper_event.set()

    class _FakeClock:
        def now(self):
            import types

            return types.SimpleNamespace(to_msg=lambda: None)

    runtime.node = types.SimpleNamespace(get_clock=lambda: _FakeClock())
    return runtime


def _drive_feedback(runtime, script: list[float], interval: float = 0.05) -> None:
    """Simulate gripper feedback samples arriving every `interval` seconds."""
    import threading
    import time as _time

    def worker() -> None:
        for value in script:
            _time.sleep(interval)
            with runtime._feedback_lock:
                runtime._gripper_raw = value
                runtime._gripper_time = _time.monotonic()

    thread = threading.Thread(target=worker)
    thread.start()


def test_gripper_close_stall_is_success() -> None:
    """Closing onto an object (frozen feedback partway) must succeed."""
    runtime = _make_gripper_runtime()
    # Open(100) -> fingers close onto a wide object and freeze at 45.
    _drive_feedback(runtime, [90.0, 70.0, 45.0, 45.0, 45.0, 45.0, 45.0])
    result = runtime.set_gripper({"position": 0.0, "settle_s": 2.0})
    assert result["success"] is True
    assert result["stalled"] is True
    assert abs(result["gripper_pos"] - 0.45) < 0.05


def test_gripper_full_close_no_object() -> None:
    runtime = _make_gripper_runtime()
    _drive_feedback(runtime, [80.0, 50.0, 20.0, 5.0, 3.0])
    result = runtime.set_gripper({"position": 0.0, "settle_s": 2.0})
    assert result["success"] is True
    assert result["stalled"] is False


def test_gripper_never_converges_raises() -> None:
    runtime = _make_gripper_runtime()
    # Feedback keeps creeping — neither reaches target nor stalls.
    _drive_feedback(runtime, [99.0, 98.5, 98.0, 97.5, 97.0, 96.5], interval=0.05)
    try:
        runtime.set_gripper({"position": 0.0, "settle_s": 0.3})
    except RuntimeError as exc:
        assert "converge" in str(exc)
    else:
        raise AssertionError("expected RuntimeError for non-converging gripper")


def test_task_info_parses_markdown_bold_verdicts(monkeypatch, tmp_path: Path) -> None:
    """**SUCCESS** from GLM must count as success, not 'unsure'."""
    env = _env(monkeypatch, tmp_path)
    verdicts = {
        "**SUCCESS** — The remote is held.": ("success", 1.0),
        "SUCCESS: cube in gripper.": ("success", 1.0),
        "FAILURE — nothing grasped.": ("failure", 0.0),
        "## UNSURE\nCannot tell.": ("unsure", 0.0),
        "": ("unsure", 0.0),
    }
    for reply, (expected_status, expected_reward) in verdicts.items():
        namespace = make_namespace(env, cfg=type("Config", (), {"task": "pick up the remote"})())
        namespace["vlm_query"] = lambda *a, **k: reply
        info = namespace["get_task_info"]()
        assert info["status"] == expected_status, (reply, info)
        assert info["reward"] == expected_reward, (reply, info)


def test_gripper_start_latency_is_not_stall() -> None:
    """Controller latency (frozen feedback before motion starts) must not
    count as a grasp stall; the close must keep waiting and then succeed."""
    runtime = _make_gripper_runtime()
    # ~0.5s of frozen feedback at the commanded-time opening before the
    # fingers actually move, then travel to fully closed.
    _drive_feedback(
        runtime,
        [99.5] * 10 + [90.0, 70.0, 45.0, 20.0, 8.0, 5.0, 4.0, 3.0, 3.0],
        interval=0.05,
    )
    result = runtime.set_gripper({"position": 0.0, "settle_s": 3.0})
    assert result["success"] is True
    assert result["stalled"] is False


def test_gripper_frozen_without_motion_raises() -> None:
    """Feedback that never moves at all is a real failure, not a stall."""
    runtime = _make_gripper_runtime()
    _drive_feedback(runtime, [99.5] * 12, interval=0.05)
    try:
        runtime.set_gripper({"position": 0.0, "settle_s": 0.5})
    except RuntimeError as exc:
        assert "converge" in str(exc)
    else:
        raise AssertionError("expected RuntimeError for frozen gripper")
