from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

import drone_ai.cli as cli
from drone_ai.cameras.base import CameraError
from drone_ai.cameras.video import CameraEndOfStream, VideoFileCamera
from drone_ai.replay import VisionReplay
from drone_ai.vision.models import Detection
from drone_ai.vision.tracker import TargetTracker


def make_video(path: Path, *, frames: int = 3, fps: float = 5.0) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        (160, 120),
    )
    assert writer.isOpened()
    try:
        for index in range(frames):
            frame = np.full((120, 160, 3), index * 30, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()


def test_video_file_camera_reads_finite_stream(tmp_path: Path) -> None:
    path = tmp_path / "input.avi"
    make_video(path)
    camera = VideoFileCamera(path)
    camera.open()
    try:
        frames = [camera.read(), camera.read(), camera.read()]
        with pytest.raises(CameraEndOfStream, match="after 3 frame"):
            camera.read()
    finally:
        camera.close()
    assert all(frame.shape == (120, 160, 3) for frame in frames)
    assert camera.frames_read == 3
    assert camera.fps == pytest.approx(5.0, rel=0.1)


def test_video_file_camera_rejects_missing_input(tmp_path: Path) -> None:
    with pytest.raises(CameraError, match="does not exist"):
        VideoFileCamera(tmp_path / "missing.mp4").open()


class SequenceDetector:
    def __init__(self) -> None:
        self.index = 0

    def detect(self, frame: Any) -> list[Detection]:
        self.index += 1
        if self.index == 1:
            return [Detection("person", 0.8, (60, 30, 100, 90))]
        return []


def test_vision_replay_produces_serializable_records_and_summary() -> None:
    replay = VisionReplay(SequenceDetector(), TargetTracker("person"), fps=5.0)
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    first = replay.process(frame)
    second = replay.process(frame)
    assert first.record.frame_index == 0
    assert first.record.timestamp_s == 0.0
    assert first.record.tracking.visible
    assert json.loads(first.record.model_dump_json())["detections"][0]["class_name"] == "person"
    assert second.record.frame_index == 1
    assert second.record.timestamp_s == pytest.approx(0.2)
    summary = replay.summary()
    assert summary.total_frames == 2
    assert summary.frames_with_target == 1
    assert summary.maximum_confidence == 0.8


class FakeYOLO:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.validated = False

    def validate_target_classes(self) -> None:
        self.validated = True

    def detect(self, frame: Any) -> list[Detection]:
        assert self.validated
        return [Detection("person", 0.9, (60, 30, 100, 90))]


def replay_args(input_path: Path, **changes: Any) -> argparse.Namespace:
    values: dict[str, Any] = {
        "input": str(input_path),
        "target": "person",
        "model": None,
        "annotated_video": None,
        "jsonl": None,
        "display": False,
        "max_frames": None,
        "overwrite": False,
    }
    values.update(changes)
    return argparse.Namespace(**values)


def test_headless_replay_writes_bounded_video_and_jsonl(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "input.avi"
    output_path = tmp_path / "annotated.mp4"
    jsonl_path = tmp_path / "frames.jsonl"
    make_video(input_path, frames=3)
    monkeypatch.setattr(cli, "YOLODetector", FakeYOLO)
    result = cli.run_replay(
        replay_args(
            input_path,
            annotated_video=str(output_path),
            jsonl=str(jsonl_path),
            max_frames=2,
        )
    )
    assert result == 0
    assert output_path.is_file() and output_path.stat().st_size > 0
    records = [json.loads(line) for line in jsonl_path.read_text().splitlines()]
    assert [record["frame_index"] for record in records] == [0, 1]
    assert all(record["tracking"]["visible"] for record in records)


def test_replay_refuses_output_overwrite(tmp_path: Path) -> None:
    input_path = tmp_path / "input.avi"
    make_video(input_path)
    existing = input_path.with_suffix(".jsonl")
    existing.write_text("keep")
    with pytest.raises(ValueError, match="already exists"):
        cli.run_replay(replay_args(input_path, jsonl=str(existing)))


def test_replay_refuses_to_replace_its_input(tmp_path: Path) -> None:
    input_path = tmp_path / "input.avi"
    make_video(input_path)
    with pytest.raises(ValueError, match="cannot overwrite"):
        cli.run_replay(replay_args(input_path, annotated_video=str(input_path), overwrite=True))
