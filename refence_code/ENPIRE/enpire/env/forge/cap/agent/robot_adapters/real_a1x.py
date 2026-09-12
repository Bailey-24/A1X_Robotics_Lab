"""Real Galaxea A1X robot adapter."""

from __future__ import annotations

import os
from typing import Any

from enpire.env.forge.cap.agent.robot_adapters.base import cfg_runtime_kwargs, cfg_select


class RealA1XAdapter:
    def __init__(self, config_group: str = "real_a1x") -> None:
        self.config_group = config_group

    def create_runtime(
        self,
        *,
        cfg: Any | None = None,
        runtime_role: str = "script",
        env_name: str | None = None,
        viewer: bool = False,
        vlm_backend: str = os.environ.get("A1X_VLM_BACKEND", "gemini"),
        seed: int | None = None,
        layout_id: int | None = None,
        style_id: int | None = None,
        camera_height: int | None = None,
        camera_width: int | None = None,
        curobo_host: str = "127.0.0.1",
        curobo_port: int = 0,
        mppi_host: str = "127.0.0.1",
        mppi_port: int = 0,
    ) -> tuple[Any, dict[str, Any]]:
        kwargs = cfg_runtime_kwargs(
            cfg,
            runtime_role=runtime_role,
            env_name=env_name,
            viewer=viewer,
            vlm_backend=vlm_backend,
            seed=seed,
            layout_id=layout_id,
            style_id=style_id,
            camera_height=camera_height,
            camera_width=camera_width,
            curobo_host=curobo_host,
            curobo_port=curobo_port,
            mppi_host=mppi_host,
            mppi_port=mppi_port,
        )
        from enpire.env.forge.cap.agent.tool_handle import make_tool_runner_namespace
        from enpire.env.forge.cap.env import create_env
        from enpire.env.forge.cap.env.real_a1x.skills import make_namespace

        enable_cameras = runtime_role != "agent" and not bool(
            cfg_select(cfg, "runtime.no_cameras", False)
        )
        env = create_env(
            kwargs["env_name"] or "a1x-real",
            enable_cameras=enable_cameras,
            bridge_url=cfg_select(cfg, "robot.bridge_url", None),
            camera_name=str(cfg_select(cfg, "robot.camera_name", "wrist")),
            camera_resolution=(
                int(kwargs["camera_width"] or 640),
                int(kwargs["camera_height"] or 480),
            ),
            camera_fps=int(cfg_select(cfg, "robot.camera_fps", 15)),
            calibration_path=cfg_select(cfg, "robot.calibration_path", None) or None,
        )
        namespace = make_namespace(env, vlm_backend=kwargs["vlm_backend"], cfg=cfg)
        if runtime_role != "script":
            namespace.update(make_tool_runner_namespace())
        return env, namespace

    def run_script_overrides(self, cfg: Any | None = None) -> list[str]:
        overrides = [f"robot={self.config_group}"]
        if cfg is None:
            return overrides
        # The run_script child composes the base config (its reward group
        # defaults to a NVIDIA-gateway model name). Forward the experiment's
        # VLM settings so the trusted verifier uses the same backend/model.
        for group in ("reward", "reflection"):
            backend = cfg_select(cfg, f"{group}.vlm_backend", None)
            model = cfg_select(cfg, f"{group}.vlm_model", None)
            if backend:
                overrides.append(f"{group}.vlm_backend={backend}")
            overrides.append(f"{group}.vlm_model={model if model else 'null'}")
        return overrides

    def child_env(
        self,
        cfg: Any | None = None,
        *,
        seed: int | None = None,
        slot: int = 0,
        n_seeds: int = 1,
    ) -> dict[str, str]:
        _ = (seed, slot)
        if n_seeds != 1:
            raise ValueError("A1X real hardware supports exactly one execution seed")
        sam3_host = str(cfg_select(cfg, "runtime.sam3_host", "127.0.0.1"))
        sam3_port = int(cfg_select(cfg, "runtime.sam3_port", 9500))
        return {
            "A1X_BRIDGE_URL": str(
                cfg_select(cfg, "robot.bridge_url", "http://127.0.0.1:11337")
            ),
            "CAP_WRIST_CAMERA_BACKEND": "realsense",
            "CAP_WRIST_CAMERA_RESOLUTION": "640x480",
            "CAP_WRIST_CAMERA_FPS": str(cfg_select(cfg, "robot.camera_fps", 15)),
            "SAM3_SERVER_HOST": sam3_host,
            "SAM3_SERVER_PORT": str(sam3_port),
            "TABLE_SURFACE_Z_M": str(cfg_select(cfg, "robot.table_z_m", 0.0)),
            "A1X_HAND_EYE_CALIBRATION": str(
                cfg_select(cfg, "robot.calibration_path", "") or ""
            ),
            "CAP_ROBOT_TYPE": "a1x",
        }
