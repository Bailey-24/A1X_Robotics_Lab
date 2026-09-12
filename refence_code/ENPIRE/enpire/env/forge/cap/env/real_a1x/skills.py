"""Direct-mode tool namespace for the Galaxea A1X."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np


def _tool_callable(tool: Any) -> Callable[..., Any]:
    parameter_names = [parameter.name for parameter in getattr(tool, "parameters", [])]

    def call(*args: Any, **kwargs: Any) -> Any:
        for index, value in enumerate(args):
            if index < len(parameter_names):
                kwargs[parameter_names[index]] = value
        result = tool.execute(**kwargs)
        if not result.success:
            raise RuntimeError(f"Tool {tool.name} failed: {result.error}")
        return result.data

    call.__name__ = tool.name
    call.__doc__ = tool.description
    return call


def make_namespace(env: Any, vlm_backend: str = "gemini", cfg: Any = None) -> dict[str, Any]:
    from enpire.env.forge.cap.agent.tools.camera import (
        GetCameraExtrinsicsTool,
        GetCameraIntrinsicsTool,
        RenderDepthTool,
        RenderRgbTool,
    )
    from enpire.env.forge.cap.agent.tools.native import (
        CloseGripperTool,
        GetCameraImageTool,
        GetRobotStateTool,
        GoHomeTool,
        MoveJointKeypointsTool,
        OpenGripperTool,
        SetGripperTool,
    )
    from enpire.env.forge.cap.agent.tools.segmentation import (
        SegmentAllObjectsTool,
        SegmentObjectTool,
    )
    from enpire.env.forge.cap.agent.tools.vlm_query import VlmQueryTool

    tools = [
        GetRobotStateTool(env=env),
        GetCameraImageTool(env=env),
        GetCameraIntrinsicsTool(env=env),
        GetCameraExtrinsicsTool(env=env),
        RenderRgbTool(env=env),
        RenderDepthTool(env=env),
        SegmentObjectTool(env=env),
        SegmentAllObjectsTool(env=env),
        VlmQueryTool(env=env, default_backend=vlm_backend),
        SetGripperTool(env=env),
        OpenGripperTool(env=env),
        CloseGripperTool(env=env),
        MoveJointKeypointsTool(env=env),
        GoHomeTool(env=env),
    ]
    namespace = {tool.name: _tool_callable(tool) for tool in tools}

    def move_joints(
        positions: list[float],
        *,
        duration_s: float = 2.0,
        gripper: float | None = None,
    ) -> dict[str, Any]:
        """Move the six A1X joints smoothly; this is not collision planning."""
        joints = np.asarray(positions, dtype=np.float64).reshape(6)
        result = env._bridge.move_joints(
            joints.tolist(),
            steps=max(2, int(round(float(duration_s) * env._profile.control_freq_hz))),
        )
        if gripper is not None:
            result["gripper"] = env._bridge.set_gripper(float(gripper))
        return result

    def move_ee_absolute(
        position: list[float],
        quat_xyzw: list[float] | None = None,
        *,
        steps: int = 30,
        rate_hz: float = 20.0,
    ) -> dict[str, Any]:
        """Move by A1X IK only; no collision-aware path planning is performed."""
        return env.move_ee_absolute(
            position, quat_xyzw, steps=int(steps), rate_hz=float(rate_hz)
        )

    def move_ee_relative(
        delta_xyz: list[float],
        *,
        steps: int = 30,
        rate_hz: float = 20.0,
    ) -> dict[str, Any]:
        """Move in the A1X base frame while preserving EE orientation."""
        return env.move_ee_relative(
            delta_xyz, steps=int(steps), rate_hz=float(rate_hz)
        )

    def move_to_observation() -> dict[str, Any]:
        """Move to the camera observation pose from examples/yoloe_grasp."""
        return env.move_to_observation()

    class _GraspResult(dict):
        """Dict that also supports attribute access (LLM-friendly)."""

        def __getattr__(self, name: str) -> Any:
            try:
                return self[name]
            except KeyError as exc:
                raise AttributeError(
                    f"{name!r} is not a grasp result key; valid keys: {sorted(self)}"
                ) from exc

    def compute_topdown_grasp(
        segmentation: Any,
        *,
        camera: str = "wrist",
        depth_strategy: str = "mid_height",
        grasp_height_fraction: float = 0.5,
        grasp_height_offset: float = 0.015,
        tcp_offset: float = 0.075,
        y_correction: float = -0.015,
        pre_grasp_offset: float = 0.05,
        lift_height: float = 0.10,
    ) -> dict[str, Any]:
        """Convert an ENPIRE segmentation into A1X top-down grasp poses.

        This preserves the calibrated geometry and PCA convention used by
        ``examples/yoloe_grasp`` while consuming ENPIRE depth in metres.
        """
        from scipy.spatial.transform import Rotation

        try:
            rgb, depth, intrinsics = env.last_rendered_rgbd(camera)
        except RuntimeError:
            rgb = env.render_rgb(camera)
            rgb, depth, intrinsics = env.last_rendered_rgbd(camera)
        if depth.shape != rgb.shape[:2]:
            raise RuntimeError("RGB and aligned depth shapes do not match")
        mask = np.asarray(segmentation.mask, dtype=bool)
        x, y, width, height = [int(value) for value in segmentation.bbox_xywh]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(depth.shape[1], x + width), min(depth.shape[0], y + height)
        region = depth[y1:y2, x1:x2]
        valid = mask[y1:y2, x1:x2] & np.isfinite(region) & (region > 0.0) & (region < 2.0)
        values = region[valid]
        if values.size < 10:
            raise RuntimeError("Too few valid object depth pixels")
        if depth_strategy == "mid_height":
            top = float(np.percentile(values, 10))
            bottom = float(np.percentile(values, 90))
            object_height = bottom - top
            z_camera = (
                float(np.median(values))
                if object_height < 0.005
                else top + min(object_height, 0.15) * float(grasp_height_fraction)
            )
        elif depth_strategy == "surface":
            z_camera = float(np.median(values))
        else:
            raise ValueError("depth_strategy must be 'surface' or 'mid_height'")

        ys, xs = np.where(valid)
        u = x1 + float(np.mean(xs))
        v = y1 + float(np.mean(ys))
        fx, fy, cx, cy = (
            intrinsics["fx"], intrinsics["fy"], intrinsics["cx"], intrinsics["cy"]
        )
        point_camera = np.array(
            [(u - cx) * z_camera / fx, (v - cy) * z_camera / fy, z_camera, 1.0]
        )
        extr = env.get_camera_extrinsics(camera)
        T_base_from_camera = np.eye(4)
        T_base_from_camera[:3, :3] = np.asarray(extr["rotation"])
        T_base_from_camera[:3, 3] = np.asarray(extr["position"])
        position = (T_base_from_camera @ point_camera)[:3]
        position += np.array([0.0, float(y_correction), float(grasp_height_offset + tcp_offset)])

        pixel_points = np.column_stack(np.where(mask)[::-1]).astype(np.float64)
        theta = 0.0
        if pixel_points.shape[0] >= 20:
            eigenvalues, eigenvectors = np.linalg.eigh(np.cov(pixel_points, rowvar=False))
            if eigenvalues[-1] / max(eigenvalues[0], 1e-9) >= 1.5:
                principal = eigenvectors[:, -1]
                direction_camera = np.array([principal[0], principal[1], 0.0])
                direction_base = T_base_from_camera[:3, :3] @ direction_camera
                theta = float(np.arctan2(direction_base[1], direction_base[0]))
                if theta > np.pi / 2:
                    theta -= np.pi
                elif theta <= -np.pi / 2:
                    theta += np.pi
        topdown = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
        c, s = np.cos(theta), np.sin(theta)
        rotation = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]) @ topdown
        quat = Rotation.from_matrix(rotation).as_quat().tolist()
        grasp = position.tolist()
        pregrasp = (position + np.array([0.0, 0.0, pre_grasp_offset])).tolist()
        lift = (position + np.array([0.0, 0.0, lift_height])).tolist()
        return _GraspResult({
            "pregrasp_position": pregrasp,
            "grasp_position": grasp,
            "lift_position": lift,
            "quat_xyzw": quat,
            "pca_angle_base_rad": theta,
            "depth_m": z_camera,
            "camera": camera,
        })

    def get_task_info() -> dict[str, Any]:
        """Use the wrist camera for a conservative visual completion check."""
        def select(path: str, default: Any) -> Any:
            current = cfg
            for part in path.split("."):
                current = getattr(current, part, None)
                if current is None:
                    return default
            return current

        task = str(select("task", "Complete the A1X task."))
        backend = str(select("reward.vlm_backend", vlm_backend))
        model = select("reward.vlm_model", None)
        camera = str(select("reward.vlm_camera", "wrist"))
        env_camera = getattr(env, "_camera_name", None)
        if env_camera is not None and camera != env_camera:
            camera = env_camera
        reasoning_effort = str(select("reward.vlm_reasoning_effort", "high"))
        try:
            response = namespace["vlm_query"](
                text=(
                    "Evaluate whether the physical robot completed this task: "
                    f"{task}\nAnswer first with exactly SUCCESS, FAILURE, or UNSURE, "
                    "then one short reason."
                ),
                media=[f"camera:{camera}"],
                backend=backend,
                model=model,
                reasoning_effort=reasoning_effort,
            )
        except Exception as exc:
            return {
                "success": False,
                "reward": 0.0,
                "status": "unavailable",
                "error": str(exc),
            }
        text = str(response).strip()
        first = text.splitlines()[0].strip().upper() if text else "UNSURE"
        first = first.lstrip("*#>-: ")  # tolerate markdown like **SUCCESS**
        success = first.startswith("SUCCESS")
        failure = first.startswith("FAILURE")
        status = "success" if success else "failure" if failure else "unsure"
        return {
            "success": success,
            "reward": 1.0 if success else 0.0,
            "status": status,
            "vlm_response": text,
            "backend": backend,
            "camera": camera,
        }

    namespace.update(
        {
            "move_joints": move_joints,
            "move_ee_absolute": move_ee_absolute,
            "move_ee_relative": move_ee_relative,
            "move_to_observation": move_to_observation,
            "compute_topdown_grasp": compute_topdown_grasp,
        }
    )
    if getattr(env, "_camera", None) is not None:
        namespace["get_task_info"] = get_task_info
    return namespace
