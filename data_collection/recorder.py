"""DemoRecorder — background recording thread for A1X demonstrations.

Polls the A1XRecordingRobot at a fixed frequency, accumulates data via
CollectAny, and writes HDF5 episodes on demand.
"""
from __future__ import annotations

import logging
import threading
import time

from data_collection.a1x_robot import A1XRecordingRobot

logger = logging.getLogger("recorder")


class DemoRecorder:
    """Background thread that records joint states + camera images.

    Usage::

        recorder = DemoRecorder(robot, record_freq=20)
        recorder.start_episode()
        # ... run yoloe_grasp pipeline ...
        recorder.stop_episode(episode_id=0)
    """

    def __init__(self, robot: A1XRecordingRobot, record_freq: int = 20):
        self.robot = robot
        self.record_freq = record_freq
        self._recording = False
        self._thread: threading.Thread | None = None
        self._frame_count = 0

    def start_episode(self):
        """Start background recording."""
        if self._recording:
            logger.warning("Recording already in progress")
            return
        self._frame_count = 0
        self._recording = True
        self._thread = threading.Thread(target=self._record_loop, daemon=True)
        self._thread.start()
        logger.info(f"Recording started at {self.record_freq} Hz")

    def stop_episode(self, episode_id: int | None = None):
        """Stop recording and write the episode to HDF5.

        Args:
            episode_id: Explicit episode index. If None, uses auto-increment.
        """
        self._recording = False
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

        logger.info(f"Recording stopped — {self._frame_count} frames captured")

        if self._frame_count > 0:
            self.robot.finish(episode_id=episode_id)
            logger.info(f"Episode {episode_id} saved")
        else:
            logger.warning("No frames captured — skipping HDF5 write")

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def _record_loop(self):
        period = 1.0 / self.record_freq
        while self._recording:
            t0 = time.monotonic()
            try:
                data = self.robot.get()
                self.robot.collect(data)
                self._frame_count += 1
            except Exception as e:
                logger.error(f"Recording frame error: {e}")
            elapsed = time.monotonic() - t0
            sleep_time = period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
