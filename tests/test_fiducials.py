from __future__ import annotations

import json
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import pytest
from pydantic import ValidationError

from drone_ai.calibration import CameraCalibration, save_calibration
from drone_ai.cli import build_argument_parser, main
from drone_ai.config import FiducialConfig
from drone_ai.fiducials import (
    ArucoDictionary,
    FiducialError,
    FiducialFrame,
    FiducialMap,
    FiducialMarker,
    FiducialPoseAnalyzer,
    load_fiducial_map,
    save_printable_marker_svg,
)


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
        camera_matrix=(
            (800.0, 0.0, 320.0),
            (0.0, 800.0, 240.0),
            (0.0, 0.0, 1.0),
        ),
        distortion_coefficients=(0.0, 0.0, 0.0, 0.0, 0.0),
    )


def transform(x: float = 0.0, y: float = 0.0, z: float = 0.0):
    return (
        (1.0, 0.0, 0.0, x),
        (0.0, 1.0, 0.0, y),
        (0.0, 0.0, 1.0, z),
        (0.0, 0.0, 0.0, 1.0),
    )


def marker_map(*, marker_id: int = 0, world_x: float = 0.0, world_y: float = 0.0):
    return FiducialMap(
        name="test-room",
        dictionary=ArucoDictionary.DICT_4X4_50,
        markers=(
            FiducialMarker(
                marker_id=marker_id,
                size_m=0.12,
                marker_to_world=transform(world_x, world_y),
            ),
        ),
    )


def marker_frame(marker_id: int = 0, side_px: int = 180) -> np.ndarray:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker = cv2.aruco.generateImageMarker(dictionary, marker_id, side_px)
    frame = np.full((480, 640, 3), 255, dtype=np.uint8)
    top = (480 - side_px) // 2
    left = (640 - side_px) // 2
    frame[top : top + side_px, left : left + side_px] = cv2.cvtColor(
        marker,
        cv2.COLOR_GRAY2BGR,
    )
    return frame


def two_marker_frame() -> np.ndarray:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    frame = np.full((480, 640, 3), 255, dtype=np.uint8)
    for marker_id, left in ((0, 200), (1, 340)):
        marker = cv2.aruco.generateImageMarker(dictionary, marker_id, 100)
        frame[200:300, left : left + 100] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
    return frame


def two_marker_map(second_world_x: float = 0.084) -> FiducialMap:
    return FiducialMap(
        name="two-marker-room",
        dictionary=ArucoDictionary.DICT_4X4_50,
        markers=(
            FiducialMarker(
                marker_id=0,
                size_m=0.12,
                marker_to_world=transform(-0.085),
            ),
            FiducialMarker(
                marker_id=1,
                size_m=0.12,
                marker_to_world=transform(second_world_x),
            ),
        ),
    )


def test_map_rejects_duplicate_out_of_capacity_and_nonrigid_transforms() -> None:
    first = FiducialMarker(marker_id=0, size_m=0.12, marker_to_world=transform())
    with pytest.raises(ValidationError, match="unique"):
        FiducialMap(
            name="duplicate",
            markers=(first, first),
        )
    with pytest.raises(ValidationError, match="capacity"):
        FiducialMap(
            name="capacity",
            dictionary=ArucoDictionary.DICT_4X4_50,
            markers=(
                FiducialMarker(marker_id=50, size_m=0.12, marker_to_world=transform()),
            ),
        )
    scaled = (
        (2.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )
    with pytest.raises(ValidationError, match="orthonormal"):
        FiducialMarker(marker_id=0, size_m=0.12, marker_to_world=scaled)


def test_example_map_loads_with_strict_transform_contract() -> None:
    path = Path(__file__).parents[1] / "maps" / "example-fiducials.json"
    loaded = load_fiducial_map(path)
    assert loaded.dictionary is ArucoDictionary.DICT_4X4_50
    assert loaded.markers[0].size_m == pytest.approx(0.12)


def test_synthetic_marker_recovers_metric_camera_pose_in_world() -> None:
    analyzer = FiducialPoseAnalyzer(
        calibration(),
        marker_map(world_x=1.0, world_y=2.0),
    )
    record = analyzer.process(marker_frame())

    assert record.offline_only is True
    assert record.metric_pose_available is True
    assert record.pose_consistent is True
    assert record.maximum_translation_disagreement_m == 0.0
    assert record.maximum_rotation_disagreement_degrees == 0.0
    assert record.detected_ids == (0,)
    assert len(record.observations) == 1
    observation = record.observations[0]
    assert observation.reprojection_error_px < 0.1
    assert observation.distance_m == pytest.approx(0.534, abs=0.01)
    assert observation.camera_to_world[0][3] == pytest.approx(1.0, abs=0.01)
    assert observation.camera_to_world[1][3] == pytest.approx(2.0, abs=0.01)
    assert observation.camera_to_world[2][3] == pytest.approx(0.534, abs=0.01)
    assert record.best_camera_to_world == observation.camera_to_world
    summary = analyzer.summary()
    assert summary.total_frames == 1
    assert summary.localized_frames == 1
    assert summary.accepted_observations == 1


def test_multiple_markers_must_agree_before_world_pose_is_published() -> None:
    consistent_analyzer = FiducialPoseAnalyzer(calibration(), two_marker_map())
    consistent = consistent_analyzer.process(two_marker_frame())
    assert consistent.detected_ids == (0, 1)
    assert len(consistent.observations) == 2
    assert consistent.pose_consistent is True
    assert consistent.metric_pose_available is True
    assert consistent.best_camera_to_world is not None
    assert consistent.maximum_translation_disagreement_m is not None
    assert consistent.maximum_translation_disagreement_m < 0.25
    assert consistent.maximum_rotation_disagreement_degrees is not None
    assert consistent.maximum_rotation_disagreement_degrees < 15.0

    conflicting_analyzer = FiducialPoseAnalyzer(
        calibration(),
        two_marker_map(second_world_x=0.80),
    )
    conflicting = conflicting_analyzer.process(two_marker_frame())
    assert len(conflicting.observations) == 2
    assert conflicting.pose_consistent is False
    assert conflicting.metric_pose_available is False
    assert conflicting.best_camera_to_world is None
    assert conflicting.maximum_translation_disagreement_m is not None
    assert conflicting.maximum_translation_disagreement_m > 0.25
    summary = conflicting_analyzer.summary()
    assert summary.localized_frames == 0
    assert summary.inconsistent_frames == 1


def test_unmapped_and_too_small_observations_do_not_claim_a_pose() -> None:
    unmapped_analyzer = FiducialPoseAnalyzer(calibration(), marker_map(marker_id=0))
    unmapped = unmapped_analyzer.process(marker_frame(marker_id=1))
    assert unmapped.detected_ids == (1,)
    assert unmapped.unmapped_ids == (1,)
    assert unmapped.metric_pose_available is False
    assert unmapped.best_camera_to_world is None

    size_analyzer = FiducialPoseAnalyzer(
        calibration(),
        marker_map(),
        FiducialConfig(minimum_marker_side_px=200.0),
    )
    too_small = size_analyzer.process(marker_frame())
    assert too_small.detected_ids == (0,)
    assert too_small.rejected_mapped_count == 1
    assert too_small.observations == ()


def test_blank_frame_and_resolution_mismatch_fail_closed() -> None:
    analyzer = FiducialPoseAnalyzer(calibration(), marker_map())
    blank = analyzer.process(np.zeros((480, 640, 3), dtype=np.uint8))
    assert blank.detected_ids == ()
    assert blank.metric_pose_available is False
    with pytest.raises(RuntimeError, match="calibration is 640x480"):
        analyzer.process(np.zeros((240, 320, 3), dtype=np.uint8))


def test_frame_schema_cannot_be_relabelled_for_live_flight() -> None:
    with pytest.raises(ValidationError, match="offline-only"):
        FiducialFrame(
            frame_index=0,
            rejected_mapped_count=0,
            offline_only=False,
        )
    with pytest.raises(ValidationError, match="metric_pose_available"):
        FiducialFrame(
            frame_index=0,
            rejected_mapped_count=0,
            metric_pose_available=True,
        )


def test_offline_cli_writes_pose_evidence_without_constructing_a_drone(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video_path = tmp_path / "marker.avi"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        5.0,
        (640, 480),
    )
    assert writer.isOpened()
    writer.write(marker_frame())
    writer.write(marker_frame())
    writer.release()

    calibration_path = save_calibration(calibration(), tmp_path / "calibration.json")
    map_path = tmp_path / "markers.json"
    map_path.write_text(marker_map().model_dump_json(indent=2), encoding="utf-8")
    records_path = tmp_path / "poses.jsonl"
    annotated_path = tmp_path / "annotated.mp4"

    class ForbiddenDrone:
        def __init__(self, *args, **kwargs) -> None:
            raise AssertionError("offline fiducial analysis constructed a drone")

    monkeypatch.setattr("drone_ai.cli.SimulatedDrone", ForbiddenDrone)
    result = main(
        [
            "fiducials",
            str(video_path),
            str(map_path),
            "--calibration",
            str(calibration_path),
            "--jsonl",
            str(records_path),
            "--annotated-video",
            str(annotated_path),
        ]
    )
    assert result == 0
    records = [json.loads(line) for line in records_path.read_text().splitlines()]
    assert len(records) == 2
    assert all(record["metric_pose_available"] for record in records)
    assert all(record["offline_only"] for record in records)
    assert annotated_path.stat().st_size > 0


def test_fiducial_cli_has_no_flight_mode_and_requires_calibration(tmp_path: Path) -> None:
    args = build_argument_parser().parse_args(
        ["fiducials", "video.avi", "markers.json", "--calibration", "camera.json"]
    )
    assert not hasattr(args, "drone_mode")
    assert not hasattr(args, "takeoff")
    assert main(["fiducials", str(tmp_path / "video.avi"), "markers.json"]) == 2


def test_printable_marker_svg_has_exact_size_and_opencv_bit_pattern(
    tmp_path: Path,
) -> None:
    output = save_printable_marker_svg(
        ArucoDictionary.DICT_4X4_50,
        0,
        0.12,
        tmp_path / "marker.svg",
    )
    root = ET.parse(output).getroot()
    assert root.attrib["width"] == "210mm"
    assert root.attrib["height"] == "297mm"
    namespace = {"svg": "http://www.w3.org/2000/svg"}
    rectangles = root.findall("svg:rect", namespace)
    black = next(item for item in rectangles if item.attrib.get("fill") == "black")
    assert float(black.attrib["width"]) == pytest.approx(120.0)
    assert float(black.attrib["height"]) == pytest.approx(120.0)
    white_cells = root.findall("svg:g[@fill='white']/svg:rect", namespace)
    actual = {
        (
            round((float(item.attrib["y"]) - float(black.attrib["y"])) / 20.0),
            round((float(item.attrib["x"]) - float(black.attrib["x"])) / 20.0),
        )
        for item in white_cells
    }
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    pixels = cv2.aruco.generateImageMarker(dictionary, 0, 6)
    expected = {tuple(index) for index in np.argwhere(pixels == 255)}
    assert actual == expected
    assert "120 mm verification line" in output.read_text(encoding="utf-8")
    included = Path(__file__).parents[1] / "fiducials" / "marker-0-120mm.svg"
    assert output.read_text(encoding="utf-8").splitlines() == included.read_text(
        encoding="utf-8"
    ).splitlines()


def test_marker_generator_validates_bounds_and_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "marker.svg"
    save_printable_marker_svg(ArucoDictionary.DICT_4X4_50, 0, 0.12, output)
    with pytest.raises(FiducialError, match="already exists"):
        save_printable_marker_svg(ArucoDictionary.DICT_4X4_50, 0, 0.12, output)
    save_printable_marker_svg(
        ArucoDictionary.DICT_4X4_50,
        1,
        0.12,
        output,
        overwrite=True,
    )
    assert "ID 1" in output.read_text(encoding="utf-8")
    with pytest.raises(FiducialError, match="within 0-49"):
        save_printable_marker_svg(
            ArucoDictionary.DICT_4X4_50,
            50,
            0.12,
            tmp_path / "bad-id.svg",
        )
    with pytest.raises(FiducialError, match="0.03-0.16m"):
        save_printable_marker_svg(
            ArucoDictionary.DICT_4X4_50,
            0,
            0.17,
            tmp_path / "too-large.svg",
        )
    with pytest.raises(FiducialError, match=".svg extension"):
        save_printable_marker_svg(
            ArucoDictionary.DICT_4X4_50,
            0,
            0.12,
            tmp_path / "marker.png",
        )


def test_make_marker_cli_has_no_flight_surface(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "marker-7.svg"
    args = build_argument_parser().parse_args(
        ["make-marker", str(output), "--id", "7", "--size-m", "0.08"]
    )
    assert not hasattr(args, "drone_mode")
    assert not hasattr(args, "takeoff")

    class ForbiddenDrone:
        def __init__(self, *args, **kwargs) -> None:
            raise AssertionError("marker generation constructed a drone")

    monkeypatch.setattr("drone_ai.cli.SimulatedDrone", ForbiddenDrone)
    assert main(["make-marker", str(output), "--id", "7", "--size-m", "0.08"]) == 0
    assert output.is_file()
    assert "ID 7" in output.read_text(encoding="utf-8")
