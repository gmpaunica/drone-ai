"""Camera wrapper that applies a verified OpenCV calibration."""

from __future__ import annotations

from typing import Any

from drone_ai.calibration import CameraCalibration
from drone_ai.cameras.base import CameraError, CameraSource


class UndistortingCamera(CameraSource):
    def __init__(self, source: CameraSource, calibration: CameraCalibration) -> None:
        self.source = source
        self.calibration = calibration
        self._opened = False
        self._map_x: Any | None = None
        self._map_y: Any | None = None

    def open(self) -> None:
        self.source.open()
        self._opened = True
        self._map_x = None
        self._map_y = None

    def read(self) -> Any:
        if not self._opened:
            raise CameraError("undistorting camera is not open")
        frame = self.source.read()
        shape = getattr(frame, "shape", None)
        if not isinstance(shape, tuple) or len(shape) != 3 or shape[2] != 3:
            raise CameraError("undistortion requires a three-channel BGR frame")
        height, width = shape[:2]
        expected = (self.calibration.image_width, self.calibration.image_height)
        actual = (width, height)
        if actual != expected:
            raise CameraError(
                f"calibration resolution is {expected[0]}x{expected[1]} but camera returned "
                f"{actual[0]}x{actual[1]}"
            )
        if self._map_x is None or self._map_y is None:
            try:
                import cv2
                import numpy as np
            except ImportError as exc:
                raise CameraError("OpenCV and NumPy are required for undistortion") from exc
            camera_matrix = np.asarray(self.calibration.camera_matrix, dtype=np.float64)
            distortion = np.asarray(
                self.calibration.distortion_coefficients,
                dtype=np.float64,
            )
            self._map_x, self._map_y = cv2.initUndistortRectifyMap(
                camera_matrix,
                distortion,
                None,
                camera_matrix,
                actual,
                cv2.CV_32FC1,
            )
        import cv2

        return cv2.remap(frame, self._map_x, self._map_y, interpolation=cv2.INTER_LINEAR)

    def close(self) -> None:
        try:
            self.source.close()
        finally:
            self._opened = False
            self._map_x = None
            self._map_y = None
