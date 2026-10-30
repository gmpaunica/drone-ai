from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

import drone_ai.cli as cli
from drone_ai.calibration import (
    CalibrationError,
    CameraCalibration,
    CheckerboardCalibrator,
    load_calibration,
    save_calibration,
)
from drone_ai.config import CalibrationConfig


def checkerboard_frame(index: int, *, width: int = 640, height: int = 480) -> np.ndarray:
    columns, rows = 9, 6
    square = 48
    board_width = (columns + 1) * square
    board_height = (rows + 1) * square
    board = np.full((board_height, board_width), 255, dtype=np.uint8)
    for row in range(rows + 1):
        for column in range(columns + 1):
            if (row + column) % 2 == 0:
                cv2.rectangle(
                    board,
                    (column * square, row * square),
                    ((column + 1) * square, (row + 1) * square),
                    0,
                    -1,
                )
    source = np.float32(
        [[0, 0], [board_width - 1, 0], [board_width - 1, board_height - 1], [0, board_height - 1]]
    )
    shifts = [
        ((70, 60), (560, 45), (570, 410), (60, 420)),
        ((95, 40), (550, 75), (535, 420), (75, 390)),
        ((55, 85), (575, 55), (545, 390), (90, 430)),
        ((110, 70), (525, 35), (580, 405), (55, 385)),
        ((65, 35), (570, 95), (555, 430), (100, 380)),
        ((120, 90), (535, 60), (520, 375), (80, 420)),
        ((80, 55), (545, 85), (585, 385), (50, 405)),
        ((100, 80), (565, 40), (540, 415), (70, 370)),
    ]
    destination = np.float32(shifts[index % len(shifts)])
    transform = cv2.getPerspectiveTransform(source, destination)
    warped = cv2.warpPerspective(board, transform, (width, height), borderValue=160)
    return cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)


def calibration_config() -> CalibrationConfig:
    return CalibrationConfig(
        minimum_views=5,
        maximum_rms_error_px=5.0,
        sample_every_frames=1,
    )


def test_checkerboard_calibrator_detects_views_and_builds_strict_artifact() -> None:
    calibrator = CheckerboardCalibrator(calibration_config())
    accepted = [calibrator.add_frame(checkerboard_frame(index)) for index in range(8)]
    assert all(accepted)
    calibration = calibrator.calibrate()
    assert calibration.views_used == 8
    assert (calibration.image_width, calibration.image_height) == (640, 480)
    assert calibration.rms_error_px <= 5.0
    assert calibration.camera_matrix[0][0] > 0
    assert len(calibration.distortion_coefficients) >= 4


def test_calibration_rejects_too_few_views_and_resolution_changes() -> None:
    calibrator = CheckerboardCalibrator(calibration_config())
    assert calibrator.add_frame(checkerboard_frame(0))
    with pytest.raises(CalibrationError, match="resolution changed"):
        calibrator.add_frame(checkerboard_frame(1, width=800, height=600))
    with pytest.raises(CalibrationError, match="at least 5"):
        calibrator.calibrate()


def test_calibration_does_not_count_duplicate_stationary_views() -> None:
    calibrator = CheckerboardCalibrator(calibration_config())
    frame = checkerboard_frame(0)
    assert calibrator.add_frame(frame)
    assert not calibrator.add_frame(frame.copy())
    assert calibrator.views_collected == 1


def test_calibration_artifact_save_load_and_overwrite_guard(tmp_path: Path) -> None:
    calibrator = CheckerboardCalibrator(calibration_config())
    for index in range(8):
        assert calibrator.add_frame(checkerboard_frame(index))
    calibration = calibrator.calibrate()
    path = tmp_path / "camera.json"
    assert save_calibration(calibration, path) == path
    assert load_calibration(path) == calibration
    with pytest.raises(CalibrationError, match="already exists"):
        save_calibration(calibration, path)
    assert save_calibration(calibration, path, overwrite=True) == path


def make_checkerboard_video(path: Path) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        5.0,
        (640, 480),
    )
    assert writer.isOpened()
    try:
        for index in range(8):
            writer.write(checkerboard_frame(index))
    finally:
        writer.release()


def test_calibration_cli_processes_video_without_hardware(tmp_path: Path) -> None:
    video = tmp_path / "checkerboard.avi"
    output = tmp_path / "camera.json"
    make_checkerboard_video(video)
    args = argparse.Namespace(
        input=str(video),
        output=str(output),
        columns=9,
        rows=6,
        square_size_m=0.024,
        min_views=5,
        max_rms_error_px=5.0,
        sample_every=1,
        max_frames=None,
        display=False,
        overwrite=False,
        config=None,
    )
    assert cli.run_calibrate(args) == 0
    payload = json.loads(output.read_text())
    assert payload["views_used"] == 8
    assert payload["image_width"] == 640
    assert payload["rms_error_px"] <= 5.0


def test_calibration_cli_refuses_input_overwrite(tmp_path: Path) -> None:
    video = tmp_path / "checkerboard.avi"
    make_checkerboard_video(video)
    args = argparse.Namespace(
        input=str(video),
        output=str(video),
        columns=None,
        rows=None,
        square_size_m=None,
        min_views=None,
        max_rms_error_px=None,
        sample_every=None,
        max_frames=None,
        display=False,
        overwrite=True,
        config=None,
    )
    with pytest.raises(ValueError, match="cannot overwrite"):
        cli.run_calibrate(args)
