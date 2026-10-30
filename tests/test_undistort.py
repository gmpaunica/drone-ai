from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

import drone_ai.cli as cli
from drone_ai.calibration import CameraCalibration, save_calibration
from drone_ai.cameras.base import CameraError, CameraSource
from drone_ai.cameras.blank import BlankCamera
from drone_ai.cameras.undistort import UndistortingCamera
from drone_ai.controllers.simulated import SimulatedDrone
from drone_ai.preflight import AutonomousPreflight, CheckStatus


class ArrayCamera(CameraSource):
    def __init__(self, frame: np.ndarray) -> None:
        self.frame = frame
        self.opened = False
        self.closed = False

    def open(self) -> None:
        self.opened = True

    def read(self) -> np.ndarray:
        assert self.opened and not self.closed
        return self.frame.copy()

    def close(self) -> None:
        self.closed = True


def calibration(width: int = 160, height: int = 120) -> CameraCalibration:
    return CameraCalibration(
        image_width=width,
        image_height=height,
        checkerboard_columns=9,
        checkerboard_rows=6,
        square_size_m=0.024,
        views_used=8,
        rms_error_px=0.2,
        maximum_rms_error_px=1.0,
        camera_matrix=(
            (120.0, 0.0, width / 2.0),
            (0.0, 120.0, height / 2.0),
            (0.0, 0.0, 1.0),
        ),
        distortion_coefficients=(0.0, 0.0, 0.0, 0.0, 0.0),
    )


def test_undistorting_camera_preserves_resolution_and_lifecycle() -> None:
    y, x = np.indices((120, 160))
    frame = np.stack((x % 256, y % 256, (x + y) % 256), axis=2).astype(np.uint8)
    source = ArrayCamera(frame)
    camera = UndistortingCamera(source, calibration())
    camera.open()
    corrected = camera.read()
    second = camera.read()
    camera.close()
    assert corrected.shape == frame.shape
    assert np.max(np.abs(corrected.astype(int) - frame.astype(int))) <= 1
    assert np.array_equal(corrected, second)
    assert source.closed


def test_undistorting_camera_rejects_resolution_mismatch() -> None:
    camera = UndistortingCamera(
        ArrayCamera(np.zeros((120, 160, 3), dtype=np.uint8)),
        calibration(320, 240),
    )
    camera.open()
    try:
        with pytest.raises(CameraError, match="320x240.*160x120"):
            camera.read()
    finally:
        camera.close()


class FakeDetector:
    def validate_target_classes(self) -> None:
        return None

    def detect(self, frame: Any) -> list[Any]:
        return []


def test_resolution_mismatch_fails_autonomous_preflight_while_landed() -> None:
    drone = SimulatedDrone()
    drone.connect()
    camera = UndistortingCamera(
        ArrayCamera(np.zeros((120, 160, 3), dtype=np.uint8)),
        calibration(320, 240),
    )
    camera.open()
    try:
        report = AutonomousPreflight(drone, camera, FakeDetector()).run()
    finally:
        camera.close()
    assert not report.ready
    frame_check = next(check for check in report.checks if check.name == "camera.frame")
    assert frame_check.status is CheckStatus.FAIL
    assert not drone.get_state().is_flying


def test_camera_factory_wraps_configured_source(tmp_path: Path) -> None:
    path = tmp_path / "calibration.json"
    save_calibration(calibration(960, 540), path)
    camera, session = cli._make_camera("blank", 0, None, path)
    assert isinstance(camera, UndistortingCamera)
    assert isinstance(camera.source, BlankCamera)
    assert session is None
