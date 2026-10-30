from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from pydantic import ValidationError

import drone_ai.cli as cli
from drone_ai.calibration import CameraCalibration, save_calibration
from drone_ai.config import VisualOdometryConfig
from drone_ai.odometry import OdometryFrame, OdometryStatus, VisualOdometryAnalyzer


def calibration() -> CameraCalibration:
    return CameraCalibration(
        image_width=640,
        image_height=480,
        checkerboard_columns=9,
        checkerboard_rows=6,
        square_size_m=0.024,
        views_used=8,
        rms_error_px=0.2,
        maximum_rms_error_px=1.0,
        camera_matrix=((500.0, 0.0, 320.0), (0.0, 500.0, 240.0), (0.0, 0.0, 1.0)),
        distortion_coefficients=(0.0, 0.0, 0.0, 0.0, 0.0),
    )


def project(points: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transformed = (rotation @ points.T).T + translation
    normalized = transformed[:, :2] / transformed[:, 2:3]
    return normalized * 500.0 + np.array([320.0, 240.0])


def test_visual_odometry_recovers_relative_pose_direction_without_scale() -> None:
    rng = np.random.default_rng(7)
    points = np.column_stack(
        (
            rng.uniform(-1.5, 1.5, 120),
            rng.uniform(-1.0, 1.0, 120),
            rng.uniform(4.0, 8.0, 120),
        )
    )
    rotation, _ = cv2.Rodrigues(np.array([0.01, 0.04, -0.02]))
    translation = np.array([0.25, 0.02, 0.01])
    previous = project(points, np.eye(3), np.zeros(3))
    current = project(points, rotation, translation)
    analyzer = VisualOdometryAnalyzer(calibration())
    estimate = analyzer.estimate_correspondences(previous, current)
    assert estimate.status is OdometryStatus.ESTIMATED
    assert estimate.inliers >= 15
    assert estimate.rotation_matrix is not None
    assert estimate.translation_direction is not None
    recovered = np.asarray(estimate.translation_direction)
    expected = translation / np.linalg.norm(translation)
    assert abs(float(np.dot(recovered, expected))) > 0.9
    assert estimate.metric_scale_available is False
    assert "not metric scale" in estimate.message


def test_blank_frames_report_insufficient_features_without_crashing() -> None:
    analyzer = VisualOdometryAnalyzer(calibration())
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    first = analyzer.process(frame)
    second = analyzer.process(frame)
    assert first.status is OdometryStatus.INITIALIZED
    assert second.status is OdometryStatus.INSUFFICIENT_FEATURES
    summary = analyzer.summary()
    assert summary.total_frames == 2
    assert summary.estimated_frames == 0
    assert summary.insufficient_frames == 1
    assert summary.metric_scale_available is False


def test_odometry_schema_cannot_claim_metric_scale() -> None:
    with pytest.raises(ValidationError, match="cannot claim metric scale"):
        OdometryFrame(
            frame_index=1,
            status=OdometryStatus.ESTIMATED,
            previous_keypoints=20,
            current_keypoints=20,
            matches=20,
            inliers=20,
            metric_scale_available=True,
            message="invalid",
        )


def make_blank_video(path: Path) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        5.0,
        (640, 480),
    )
    assert writer.isOpened()
    try:
        writer.write(np.zeros((480, 640, 3), dtype=np.uint8))
        writer.write(np.zeros((480, 640, 3), dtype=np.uint8))
    finally:
        writer.release()


def test_odometry_cli_is_offline_and_reports_weak_video(tmp_path: Path) -> None:
    video = tmp_path / "blank.avi"
    calibration_path = tmp_path / "camera.json"
    records = tmp_path / "odometry.jsonl"
    make_blank_video(video)
    save_calibration(calibration(), calibration_path)
    args = argparse.Namespace(
        input=str(video),
        calibration=str(calibration_path),
        jsonl=str(records),
        max_frames=None,
        overwrite=False,
        config=None,
    )
    assert cli.run_odometry(args) == 1
    payloads = [json.loads(line) for line in records.read_text().splitlines()]
    assert [item["status"] for item in payloads] == [
        "initialized",
        "insufficient_features",
    ]
    assert all(item["metric_scale_available"] is False for item in payloads)


def test_odometry_cli_requires_calibration(tmp_path: Path) -> None:
    args = argparse.Namespace(
        input=str(tmp_path / "video.avi"),
        calibration=None,
        jsonl=None,
        max_frames=None,
        overwrite=False,
        config=None,
    )
    with pytest.raises(ValueError, match="requires --calibration"):
        cli.run_odometry(args)
