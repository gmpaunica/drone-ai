"""OpenCV laptop/USB webcam adapter."""

from __future__ import annotations

import os
from typing import Any

from drone_ai.cameras.base import CameraError, CameraSource


class LaptopWebcamCamera(CameraSource):
    def __init__(self, camera_index: int = 0) -> None:
        self.camera_index = camera_index
        self._capture: Any | None = None

    def open(self) -> None:
        try:
            import cv2
        except ImportError as exc:
            raise CameraError("OpenCV is not installed; install requirements.txt") from exc
        backend = cv2.CAP_DSHOW if os.name == "nt" and hasattr(cv2, "CAP_DSHOW") else cv2.CAP_ANY
        capture = cv2.VideoCapture(self.camera_index, backend)
        if not capture.isOpened():
            capture.release()
            raise CameraError(f"could not open webcam index {self.camera_index}")
        self._capture = capture

    def read(self) -> Any:
        if self._capture is None:
            raise CameraError("webcam is not open")
        ok, frame = self._capture.read()
        if not ok or frame is None:
            raise CameraError("webcam did not return a frame")
        return frame

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
