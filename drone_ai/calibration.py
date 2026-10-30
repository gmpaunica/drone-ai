"""Checkerboard-based OpenCV camera calibration."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from drone_ai.config import CalibrationConfig


class CalibrationError(RuntimeError):
    pass


class CameraCalibration(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)
    checkerboard_columns: int = Field(ge=3)
    checkerboard_rows: int = Field(ge=3)
    square_size_m: float = Field(gt=0.0)
    views_used: int = Field(ge=1)
    rms_error_px: float = Field(ge=0.0)
    maximum_rms_error_px: float = Field(gt=0.0)
    camera_matrix: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]
    distortion_coefficients: tuple[float, ...]

    @field_validator(
        "square_size_m",
        "rms_error_px",
        "maximum_rms_error_px",
    )
    @classmethod
    def require_finite_scalar(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("calibration values must be finite")
        return value

    @field_validator("camera_matrix")
    @classmethod
    def validate_camera_matrix(
        cls,
        value: tuple[
            tuple[float, float, float],
            tuple[float, float, float],
            tuple[float, float, float],
        ],
    ) -> tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]:
        if not all(math.isfinite(item) for row in value for item in row):
            raise ValueError("camera_matrix must be finite")
        if value[0][0] <= 0 or value[1][1] <= 0:
            raise ValueError("camera focal lengths must be positive")
        if not math.isclose(value[2][2], 1.0, abs_tol=1e-6):
            raise ValueError("camera_matrix bottom-right value must be 1")
        return value

    @field_validator("distortion_coefficients")
    @classmethod
    def validate_distortion(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        if len(value) < 4 or not all(math.isfinite(item) for item in value):
            raise ValueError("at least four finite distortion coefficients are required")
        return value


class CheckerboardCalibrator:
    def __init__(self, config: CalibrationConfig | None = None) -> None:
        self.config = config or CalibrationConfig()
        if self.config.checkerboard_columns < 3 or self.config.checkerboard_rows < 3:
            raise ValueError("checkerboard inner-corner dimensions must be at least 3x3")
        if self.config.square_size_m <= 0:
            raise ValueError("checkerboard square size must be positive")
        if self.config.minimum_views < 5:
            raise ValueError("at least five calibration views are required")
        if self.config.maximum_rms_error_px <= 0:
            raise ValueError("maximum RMS error must be positive")
        self._image_points: list[Any] = []
        self._view_signatures: list[tuple[float, float, float, float]] = []
        self._image_size: tuple[int, int] | None = None
        self._last_corners: Any | None = None

    @property
    def views_collected(self) -> int:
        return len(self._image_points)

    @property
    def image_size(self) -> tuple[int, int] | None:
        return self._image_size

    def add_frame(self, frame: Any) -> bool:
        try:
            import cv2
        except ImportError as exc:
            raise CalibrationError("OpenCV is required for camera calibration") from exc
        shape = getattr(frame, "shape", None)
        if not isinstance(shape, tuple) or len(shape) != 3 or shape[2] != 3:
            raise CalibrationError("calibration frames must have exactly three BGR channels")
        height, width = shape[:2]
        size = (width, height)
        if self._image_size is None:
            self._image_size = size
        elif self._image_size != size:
            raise CalibrationError(
                f"calibration resolution changed from {self._image_size} to {size}"
            )

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        pattern = (self.config.checkerboard_columns, self.config.checkerboard_rows)
        self._last_corners = None
        if hasattr(cv2, "findChessboardCornersSB"):
            found, corners = cv2.findChessboardCornersSB(
                gray,
                pattern,
                flags=cv2.CALIB_CB_NORMALIZE_IMAGE,
            )
        else:
            found, corners = cv2.findChessboardCorners(
                gray,
                pattern,
                flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
            )
            if found:
                criteria = (
                    cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
                    30,
                    0.001,
                )
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        if not found or corners is None:
            return False
        import numpy as np

        points = corners.reshape(-1, 2)
        center_x = float(points[:, 0].mean() / width)
        center_y = float(points[:, 1].mean() / height)
        area_ratio = float(
            (points[:, 0].max() - points[:, 0].min())
            * (points[:, 1].max() - points[:, 1].min())
            / (width * height)
        )
        row_vector = points[self.config.checkerboard_columns - 1] - points[0]
        angle = float((np.degrees(np.arctan2(row_vector[1], row_vector[0])) + 90.0) % 180.0 - 90.0)
        signature = (center_x, center_y, area_ratio, angle)
        for previous in self._view_signatures:
            if (
                abs(signature[0] - previous[0]) < 0.01
                and abs(signature[1] - previous[1]) < 0.01
                and abs(signature[2] - previous[2]) < 0.01
                and abs(signature[3] - previous[3]) < 2.0
            ):
                return False
        self._last_corners = corners
        self._image_points.append(corners.astype("float32"))
        self._view_signatures.append(signature)
        return True

    def draw_last_observation(self, frame: Any) -> Any:
        if self._last_corners is None:
            return frame
        import cv2

        cv2.drawChessboardCorners(
            frame,
            (self.config.checkerboard_columns, self.config.checkerboard_rows),
            self._last_corners,
            True,
        )
        return frame

    def calibrate(self) -> CameraCalibration:
        if self.views_collected < self.config.minimum_views:
            raise CalibrationError(
                f"only {self.views_collected} valid views; "
                f"at least {self.config.minimum_views} are required"
            )
        center_x = [item[0] for item in self._view_signatures]
        center_y = [item[1] for item in self._view_signatures]
        areas = [item[2] for item in self._view_signatures]
        angles = [item[3] for item in self._view_signatures]
        insufficient = []
        if max(center_x) - min(center_x) < 0.05:
            insufficient.append("horizontal board position")
        if max(center_y) - min(center_y) < 0.05:
            insufficient.append("vertical board position")
        if max(areas) - min(areas) < 0.04:
            insufficient.append("board distance/scale")
        if max(angles) - min(angles) < 5.0:
            insufficient.append("board rotation")
        if insufficient:
            raise CalibrationError(
                "calibration views lack diversity in "
                + ", ".join(insufficient)
                + "; move and tilt the board across the image"
            )
        assert self._image_size is not None
        try:
            import cv2
            import numpy as np
        except ImportError as exc:
            raise CalibrationError("OpenCV and NumPy are required for calibration") from exc

        object_template = np.zeros(
            (self.config.checkerboard_rows * self.config.checkerboard_columns, 3),
            dtype=np.float32,
        )
        object_template[:, :2] = (
            np.mgrid[
                0 : self.config.checkerboard_columns,
                0 : self.config.checkerboard_rows,
            ]
            .T.reshape(-1, 2)
            .astype(np.float32)
            * self.config.square_size_m
        )
        object_points = [object_template.copy() for _ in self._image_points]
        try:
            rms, camera_matrix, distortion, _, _ = cv2.calibrateCamera(
                object_points,
                self._image_points,
                self._image_size,
                None,
                None,
            )
        except Exception as exc:
            raise CalibrationError(f"OpenCV calibration failed: {exc}") from exc
        rms_value = float(rms)
        if not math.isfinite(rms_value):
            raise CalibrationError("calibration produced a non-finite RMS error")
        if rms_value > self.config.maximum_rms_error_px:
            raise CalibrationError(
                f"calibration RMS error {rms_value:.3f}px exceeds "
                f"{self.config.maximum_rms_error_px:.3f}px; collect more varied views"
            )
        width, height = self._image_size
        matrix_values = tuple(
            tuple(float(item) for item in row)
            for row in camera_matrix.tolist()
        )
        distortion_values = tuple(float(item) for item in distortion.reshape(-1).tolist())
        return CameraCalibration(
            image_width=width,
            image_height=height,
            checkerboard_columns=self.config.checkerboard_columns,
            checkerboard_rows=self.config.checkerboard_rows,
            square_size_m=self.config.square_size_m,
            views_used=self.views_collected,
            rms_error_px=rms_value,
            maximum_rms_error_px=self.config.maximum_rms_error_px,
            camera_matrix=matrix_values,  # type: ignore[arg-type]
            distortion_coefficients=distortion_values,
        )


def save_calibration(
    calibration: CameraCalibration,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    output = Path(path)
    if output.exists() and not overwrite:
        raise CalibrationError(f"calibration output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(calibration.model_dump_json(indent=2), encoding="utf-8")
    return output


def load_calibration(path: str | Path) -> CameraCalibration:
    source = Path(path)
    if not source.is_file():
        raise CalibrationError(f"calibration file does not exist: {source}")
    try:
        return CameraCalibration.model_validate_json(source.read_text(encoding="utf-8"))
    except Exception as exc:
        raise CalibrationError(f"invalid calibration file {source}: {exc}") from exc
