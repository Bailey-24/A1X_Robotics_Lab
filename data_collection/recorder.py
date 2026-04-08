"""DemoRecorder — background recording thread for A1X demonstrations.

Polls the A1XRecordingRobot at a fixed frequency, accumulates data via
CollectAny, and writes HDF5 episodes on demand.

Pre-roll buffer + motion gating
-------------------------------
To avoid recording long stationary periods at the start of an episode
(e.g. while the user is typing 'yes' at the safety prompt), the recorder
maintains a rolling pre-roll buffer of the last ``pre_roll_frames`` frames
and only commits them once the joints actually move beyond
``motion_threshold`` radians.  This way every saved episode begins a few
frames *before* the robot starts moving, providing context to the policy.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Any, Optional

import numpy as np

from data_collection.a1x_robot import A1XRecordingRobot

logger = logging.getLogger("recorder")


class DemoRecorder:
    """Background thread that records joint states + camera images.

    Usage::

        recorder = DemoRecorder(robot, record_freq=20,
                                pre_roll_frames=10, motion_threshold=0.005)
        recorder.start_episode()
        # ... run yoloe_grasp pipeline ...
        recorder.stop_episode(episode_id=0)
    """

    def __init__(
        self,
        robot: A1XRecordingRobot,
        record_freq: int = 20,
        pre_roll_frames: int = 10,
        motion_threshold: float = 0.005,
    ):
        """
        Args:
            robot: A1XRecordingRobot instance.
            record_freq: Polling frequency in Hz.
            pre_roll_frames: Number of pre-motion frames to retain (rolling).
                Set to 0 to disable pre-roll (start committing immediately).
            motion_threshold: Joint diff (radians) above which motion is
                considered to have started.  Frames before this threshold
                is crossed are buffered, not committed.  Set to 0 to disable
                motion gating (commit every frame from t=0).
        """
        self.robot = robot
        self.record_freq = record_freq
        self.pre_roll_frames = pre_roll_frames
        self.motion_threshold = motion_threshold

        self._recording = False
        self._thread: threading.Thread | None = None
        self._frame_count = 0
        self._discarded_count = 0  # frames dropped before motion start

    # ── Episode lifecycle ─────────────────────────────────────────────

    def start_episode(self):
        """Start background recording."""
        if self._recording:
            logger.warning("Recording already in progress")
            return
        self._frame_count = 0
        self._discarded_count = 0
        self._recording = True
        self._thread = threading.Thread(target=self._record_loop, daemon=True)
        self._thread.start()
        logger.info(
            f"Recording started at {self.record_freq} Hz "
            f"(pre_roll={self.pre_roll_frames}, "
            f"motion_threshold={self.motion_threshold} rad)"
        )

    def stop_episode(self, episode_id: int | None = None):
        """Stop recording and write the episode to HDF5."""
        self._recording = False
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

        logger.info(
            f"Recording stopped — {self._frame_count} frames committed, "
            f"{self._discarded_count} pre-motion frames discarded"
        )

        if self._frame_count > 0:
            self.robot.finish(episode_id=episode_id)
            logger.info(f"Episode {episode_id} saved")
        else:
            logger.warning("No frames committed — skipping HDF5 write")

    @property
    def frame_count(self) -> int:
        return self._frame_count

    # ── Background loop ───────────────────────────────────────────────

    def _record_loop(self):
        period = 1.0 / self.record_freq

        # Motion-gating state (only used if motion_threshold > 0)
        gating_enabled = self.motion_threshold > 0.0
        pre_buffer: deque = deque(maxlen=max(self.pre_roll_frames, 1))
        last_joints: Optional[np.ndarray] = None
        motion_started = not gating_enabled  # commit immediately if disabled

        while self._recording:
            t0 = time.monotonic()
            try:
                data = self.robot.get()

                if motion_started:
                    # Normal recording — commit every frame
                    self.robot.collect(data)
                    self._frame_count += 1
                else:
                    # Pre-motion phase — buffer + check for motion
                    curr_joints = self._extract_joints(data)
                    if (
                        last_joints is not None
                        and curr_joints is not None
                        and curr_joints.shape == last_joints.shape
                    ):
                        joint_diff = float(np.max(np.abs(curr_joints - last_joints)))
                        if joint_diff > self.motion_threshold:
                            # Motion detected — flush pre-roll then this frame
                            motion_started = True
                            logger.info(
                                f"Motion detected (max joint diff="
                                f"{joint_diff:.4f} rad) — flushing "
                                f"{len(pre_buffer)} pre-roll frames"
                            )
                            for buffered in pre_buffer:
                                self.robot.collect(buffered)
                                self._frame_count += 1
                            pre_buffer.clear()
                            self.robot.collect(data)
                            self._frame_count += 1

                    if not motion_started:
                        # If buffer is full, the oldest frame is auto-dropped
                        if len(pre_buffer) == pre_buffer.maxlen:
                            self._discarded_count += 1
                        if self.pre_roll_frames > 0:
                            pre_buffer.append(data)
                        else:
                            self._discarded_count += 1
                        last_joints = curr_joints

            except Exception as e:
                logger.error(f"Recording frame error: {e}")

            elapsed = time.monotonic() - t0
            sleep_time = period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    @staticmethod
    def _extract_joints(data: Any) -> Optional[np.ndarray]:
        """Pull the joint array out of a robot.get() result.

        ``robot.get()`` returns ``[controller_data, sensor_data]`` where
        ``controller_data`` is a dict mapping controller name -> dict of
        collected fields. We look for the first controller that exposes
        a ``joint`` field.
        """
        try:
            controller_data = data[0]
            for ctrl in controller_data.values():
                if isinstance(ctrl, dict) and "joint" in ctrl:
                    joint = ctrl["joint"]
                    if joint is not None:
                        return np.asarray(joint, dtype=np.float64)
        except Exception:
            pass
        return None
