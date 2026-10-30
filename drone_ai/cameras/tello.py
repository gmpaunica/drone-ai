"""Tello video stream exposed through the common camera interface."""

from __future__ import annotations

from collections.abc import Callable
import time
from typing import Any

from drone_ai.cameras.base import CameraError, CameraSource
from drone_ai.tello import TelloSession


class TelloCamera(CameraSource):
    def __init__(
        self,
        session: TelloSession | None = None,
        *,
        first_frame_timeout_s: float = 5.0,
        poll_interval_s: float = 0.05,
        minimum_width: int = 640,
        minimum_height: int = 480,
        monotonic_clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if first_frame_timeout_s <= 0 or poll_interval_s <= 0:
            raise ValueError("Tello frame timeout and poll interval must be positive")
        if minimum_width <= 0 or minimum_height <= 0:
            raise ValueError("Tello minimum frame dimensions must be positive")
        self.session = session or TelloSession()
        self.first_frame_timeout_s = first_frame_timeout_s
        self.poll_interval_s = poll_interval_s
        self.minimum_width = minimum_width
        self.minimum_height = minimum_height
        self._clock = monotonic_clock
        self._sleep = sleeper
        self._frame_reader: Any | None = None
        self._received_valid_frame = False

    def open(self) -> None:
        try:
            self._frame_reader = self.session.start_stream()
            self._received_valid_frame = False
        except Exception as exc:
            raise CameraError(str(exc)) from exc

    def read(self) -> Any:
        if self._frame_reader is None:
            raise CameraError("Tello camera is not open")
        deadline = self._clock() + self.first_frame_timeout_s
        while True:
            if bool(getattr(self._frame_reader, "stopped", False)):
                raise CameraError("Tello video frame reader stopped unexpectedly")
            frame = self._frame_reader.frame
            if self._valid_frame(frame):
                self._received_valid_frame = True
                return frame.copy() if hasattr(frame, "copy") else frame
            if self._received_valid_frame or self._clock() >= deadline:
                shape = getattr(frame, "shape", None)
                raise CameraError(
                    "Tello video stream did not provide a valid non-placeholder frame "
                    f"within {self.first_frame_timeout_s:g}s (last shape={shape})"
                )
            self._sleep(self.poll_interval_s)

    def close(self) -> None:
        if self._frame_reader is None:
            return
        try:
            self.session.stop_stream()
        except Exception as exc:
            raise CameraError(str(exc)) from exc
        finally:
            self._frame_reader = None
            self._received_valid_frame = False

    def _valid_frame(self, frame: Any) -> bool:
        shape = getattr(frame, "shape", None)
        if not isinstance(shape, tuple) or len(shape) != 3 or shape[2] != 3:
            return False
        height, width = shape[:2]
        if width < self.minimum_width or height < self.minimum_height:
            return False
        if hasattr(frame, "any") and not bool(frame.any()):
            return False
        return True
