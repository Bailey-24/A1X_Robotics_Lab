"""Dependency-light client for the Python 3.10 A1X ROS bridge."""

from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class A1XBridgeClient:
    def __init__(self, base_url: str = "http://127.0.0.1:11337", timeout: float = 90.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout)

    def _request(self, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = Request(
            f"{self.base_url}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="GET" if payload is None else "POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"A1X bridge HTTP {exc.code}: {detail}") from exc
        except (URLError, TimeoutError) as exc:
            raise RuntimeError(f"Cannot reach A1X bridge at {self.base_url}: {exc}") from exc
        if not result.get("success", False):
            raise RuntimeError(str(result.get("error") or result.get("reason") or result))
        return result

    def health(self) -> dict[str, Any]:
        return self._request("/health")

    def state(self) -> dict[str, Any]:
        return self._request("/state")["state"]

    def move_joints(self, positions: list[float], **options: Any) -> dict[str, Any]:
        return self._request("/move_joints", {"positions": positions, **options})

    def set_gripper(self, position: float, **options: Any) -> dict[str, Any]:
        return self._request("/set_gripper", {"position": position, **options})

    def move_ee_absolute(
        self, position: list[float], quat_xyzw: list[float] | None = None, **options: Any
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"position": position, **options}
        if quat_xyzw is not None:
            payload["quat_xyzw"] = quat_xyzw
        return self._request("/move_ee_absolute", payload)

    def move_ee_relative(self, delta: list[float], **options: Any) -> dict[str, Any]:
        return self._request("/move_ee_relative", {"delta": delta, **options})

    def go_home(self) -> dict[str, Any]:
        return self._request("/go_home", {})

    def move_to_observation(self) -> dict[str, Any]:
        return self._request("/move_to_observation", {})

    def stop(self) -> dict[str, Any]:
        return self._request("/stop", {})
