"""Generated BGR frames for the simulator's manual console."""

from __future__ import annotations

from typing import Any

from drone_ai.cameras.base import CameraError, CameraSource


class BlankCamera(CameraSource):
    def __init__(self, width: int = 960, height: int = 540) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("blank camera dimensions must be positive")
        self.width = width
        self.height = height
        self._opened = False

    def open(self) -> None:
        try:
            import numpy as np
        except ImportError as exc:
            raise CameraError("NumPy is not installed; install requirements.txt") from exc
        self._opened = True

    def read(self) -> Any:
        if not self._opened:
            raise CameraError("blank camera is not open")
        import numpy as np

        return np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def close(self) -> None:
        self._opened = False
