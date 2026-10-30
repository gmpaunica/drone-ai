"""Bounded OpenCV video-file frame source for offline replay."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from drone_ai.cameras.base import CameraError, CameraSource


class CameraEndOfStream(CameraError):
    """Raised when a finite camera source has no more frames."""


class VideoFileCamera(CameraSource):
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._capture: Any | None = None
        self._fps = 30.0
        self._frame_count: int | None = None
        self._frames_read = 0

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def frame_count(self) -> int | None:
        return self._frame_count

    @property
    def frames_read(self) -> int:
        return self._frames_read

    def open(self) -> None:
        if not self.path.is_file():
            raise CameraError(f"video file does not exist: {self.path}")
        try:
            import cv2
        except ImportError as exc:
            raise CameraError("OpenCV is not installed; install requirements.txt") from exc
        capture = cv2.VideoCapture(str(self.path))
        if not capture.isOpened():
            capture.release()
            raise CameraError(f"could not open video file: {self.path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self._fps = fps if math.isfinite(fps) and fps > 0 else 30.0
        self._frame_count = int(count) if math.isfinite(count) and count >= 0 else None
        self._frames_read = 0
        self._capture = capture

    def read(self) -> Any:
        if self._capture is None:
            raise CameraError("video file is not open")
        ok, frame = self._capture.read()
        if not ok or frame is None:
            raise CameraEndOfStream(
                f"end of video after {self._frames_read} frame(s): {self.path}"
            )
        self._frames_read += 1
        return frame

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
