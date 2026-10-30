from __future__ import annotations

import json
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest
from pydantic import ValidationError

from drone_ai.cli import build_argument_parser, main
from drone_ai.fiducials import FiducialFrame, FiducialObservation
from drone_ai.localization import (
    LocalizationError,
    MappedPoseFrame,
    MappedPoseStatus,
    StaticMapPoseAnalyzer,
    iter_fiducial_jsonl,
)
from drone_ai.navigation import GridCell, RoomMap, grid_to_world, world_to_grid


def transform(x: float = 0.0, y: float = 0.0, z: float = 0.0):
    return (
        (1.0, 0.0, 0.0, x),
        (0.0, 1.0, 0.0, y),
        (0.0, 0.0, 1.0, z),
        (0.0, 0.0, 0.0, 1.0),
    )


def source_frame(frame_index: int, x: float, y: float, z: float = 1.0) -> FiducialFrame:
    camera_pose = transform(x, y, z)
    observation = FiducialObservation(
        marker_id=0,
        corners_px=((10.0, 10.0), (30.0, 10.0), (30.0, 30.0), (10.0, 30.0)),
        reprojection_error_px=0.1,
        distance_m=1.0,
        marker_to_camera=transform(z=1.0),
        camera_to_world=camera_pose,
    )
    return FiducialFrame(
        frame_index=frame_index,
        detected_ids=(0,),
        rejected_mapped_count=0,
        observations=(observation,),
        pose_consistent=True,
        maximum_translation_disagreement_m=0.0,
        maximum_rotation_disagreement_degrees=0.0,
        best_camera_to_world=camera_pose,
        metric_pose_available=True,
        offline_only=True,
    )


def room() -> RoomMap:
    return RoomMap(
        name="registered-room",
        width_cells=5,
        height_cells=5,
        resolution_m=1.0,
        blocked_cells=(GridCell(x=2, y=2),),
        clearance_cells=0,
    )


def test_grid_world_transform_round_trips_clockwise_yaw() -> None:
    registered = RoomMap(
        name="rotated",
        width_cells=6,
        height_cells=6,
        resolution_m=0.5,
        clearance_cells=0,
        world_origin_x_m=10.0,
        world_origin_y_m=20.0,
        grid_yaw_degrees=90.0,
    )
    world = grid_to_world(registered, 2.5, 1.5)
    assert world == pytest.approx((10.75, 18.75))
    assert world_to_grid(registered, *world) == pytest.approx((2.5, 1.5))
    record = StaticMapPoseAnalyzer(registered).process(
        source_frame(0, world[0], world[1])
    )
    assert record.status is MappedPoseStatus.SAFE
    assert record.cell == GridCell(x=2, y=1)


def test_mapper_classifies_safe_blocked_outside_and_missing_frames() -> None:
    analyzer = StaticMapPoseAnalyzer(room())
    safe = analyzer.process(source_frame(0, 1.5, 1.5))
    blocked = analyzer.process(source_frame(1, 2.2, 2.2))
    outside = analyzer.process(source_frame(2, 5.1, 1.0))
    missing = analyzer.process(
        FiducialFrame(frame_index=3, rejected_mapped_count=0, offline_only=True)
    )
    assert safe.status is MappedPoseStatus.SAFE
    assert safe.clearance_safe is True
    assert safe.cell == GridCell(x=1, y=1)
    assert blocked.status is MappedPoseStatus.CLEARANCE_BLOCKED
    assert blocked.clearance_safe is False
    assert blocked.cell == GridCell(x=2, y=2)
    assert outside.status is MappedPoseStatus.OUTSIDE_ROOM
    assert outside.cell is None
    assert missing.status is MappedPoseStatus.MISSING
    assert missing.world_x_m is None
    summary = analyzer.summary()
    assert summary.total_frames == 4
    assert summary.safe_frames == 1
    assert summary.clearance_blocked_frames == 1
    assert summary.outside_frames == 1
    assert summary.missing_frames == 1


def test_mapped_pose_schema_cannot_fake_safety_or_live_use() -> None:
    payload = {
        "frame_index": 0,
        "status": MappedPoseStatus.CLEARANCE_BLOCKED,
        "source_metric_pose_available": True,
        "source_pose_consistent": True,
        "world_x_m": 1.0,
        "world_y_m": 1.0,
        "world_z_m": 1.0,
        "grid_x_cells": 1.0,
        "grid_y_cells": 1.0,
        "cell": GridCell(x=1, y=1),
        "clearance_safe": True,
        "offline_only": True,
    }
    with pytest.raises(ValidationError, match="clearance_safe"):
        MappedPoseFrame.model_validate(payload)
    payload["clearance_safe"] = False
    payload["offline_only"] = False
    with pytest.raises(ValidationError, match="offline-only"):
        MappedPoseFrame.model_validate(payload)


def test_jsonl_reader_reports_exact_bad_line(tmp_path: Path) -> None:
    source = tmp_path / "poses.jsonl"
    source.write_text(source_frame(0, 1.0, 1.0).model_dump_json() + "\n\n")
    iterator = iter_fiducial_jsonl(source)
    assert next(iterator).frame_index == 0
    with pytest.raises(LocalizationError, match=r"poses.jsonl:2"):
        next(iterator)


def test_map_poses_cli_writes_strict_records_and_svg_without_a_drone(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "fiducials.jsonl"
    frames = (
        source_frame(0, 1.5, 1.5),
        source_frame(1, 2.2, 2.2),
        source_frame(2, 5.1, 1.0),
        FiducialFrame(frame_index=3, rejected_mapped_count=0, offline_only=True),
    )
    input_path.write_text(
        "".join(frame.model_dump_json() + "\n" for frame in frames),
        encoding="utf-8",
    )
    map_path = tmp_path / "room.json"
    map_path.write_text(room().model_dump_json(indent=2), encoding="utf-8")
    output_path = tmp_path / "mapped.jsonl"
    svg_path = tmp_path / "trajectory.svg"

    class ForbiddenDrone:
        def __init__(self, *args, **kwargs) -> None:
            raise AssertionError("offline pose mapping constructed a drone")

    monkeypatch.setattr("drone_ai.cli.SimulatedDrone", ForbiddenDrone)
    result = main(
        [
            "map-poses",
            str(input_path),
            str(map_path),
            "--jsonl",
            str(output_path),
            "--svg",
            str(svg_path),
        ]
    )
    assert result == 0
    records = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert [record["status"] for record in records] == [
        "safe",
        "clearance_blocked",
        "outside_room",
        "missing",
    ]
    assert all(record["offline_only"] for record in records)
    root = ET.parse(svg_path).getroot()
    namespace = {"svg": "http://www.w3.org/2000/svg"}
    assert len(root.findall("svg:circle[@class='pose-safe']", namespace)) == 1
    assert len(root.findall("svg:circle[@class='pose-clearance_blocked']", namespace)) == 1


def test_map_poses_cli_has_no_flight_surface() -> None:
    args = build_argument_parser().parse_args(
        ["map-poses", "poses.jsonl", "room.json", "--svg", "trajectory.svg"]
    )
    assert not hasattr(args, "drone_mode")
    assert not hasattr(args, "takeoff")
