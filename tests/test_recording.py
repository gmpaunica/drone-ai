from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

import drone_ai.cli as cli
from drone_ai.abilities.follow_person import FollowPersonAbility
from drone_ai.abilities.visual_search import VisualSearchAbility
from drone_ai.controllers.simulated import SimulatedDrone
from drone_ai.pipeline import FollowPipeline
from drone_ai.recording import RecordingError, SessionRecorder
from drone_ai.safety import SafetyController
from drone_ai.search_pipeline import SearchPipeline
from drone_ai.vision.models import Detection
from drone_ai.vision.tracker import PersonTracker, TargetTracker


NOW = datetime(2026, 8, 30, 12, 34, 56, tzinfo=timezone.utc)


def test_session_recorder_writes_flushed_events_video_and_manifest(tmp_path: Path) -> None:
    recorder = SessionRecorder(tmp_path, "follow", now=NOW, session_suffix="abc123")
    event = recorder.record_event("preflight", {"ready": True, "battery": 100})
    assert event.sequence == 0
    events_path = recorder.session_dir / "events.jsonl"
    assert json.loads(events_path.read_text())["payload"]["ready"] is True

    recorder.record_frame(np.zeros((120, 160, 3), dtype=np.uint8))
    manifest = recorder.close(status="completed")
    assert manifest.mode == "follow"
    assert manifest.status == "completed"
    assert manifest.event_count == 1
    assert manifest.frame_count == 1
    assert manifest.video_file == "video.mp4"
    assert (recorder.session_dir / "video.mp4").stat().st_size > 0
    parsed = json.loads((recorder.session_dir / "manifest.json").read_text())
    assert parsed["event_count"] == 1

    capture = cv2.VideoCapture(str(recorder.session_dir / "video.mp4"))
    ok, frame = capture.read()
    capture.release()
    assert ok and frame.shape == (120, 160, 3)


def test_session_recorder_is_collision_safe(tmp_path: Path) -> None:
    first = SessionRecorder(tmp_path, "manual", now=NOW, session_suffix="same")
    first.close()
    with pytest.raises(RecordingError, match="already exists"):
        SessionRecorder(tmp_path, "manual", now=NOW, session_suffix="same")


def test_session_recorder_rejects_nonfinite_json(tmp_path: Path) -> None:
    recorder = SessionRecorder(tmp_path, "search", now=NOW, session_suffix="finite")
    with pytest.raises(RecordingError, match="finite JSON"):
        recorder.record_event("bad", {"value": float("nan")})
    assert recorder.event_count == 0
    recorder.close(status="failed")


def test_no_frame_session_has_no_video_and_close_is_idempotent(tmp_path: Path) -> None:
    recorder = SessionRecorder(tmp_path, "manual", now=NOW, session_suffix="empty")
    first = recorder.close(status="cancelled")
    second = recorder.close(status="ignored")
    assert first == second
    assert first.video_file is None
    assert not (recorder.session_dir / "video.mp4").exists()


def test_manual_cli_records_real_loop_events_video_and_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    keys = iter([ord("t"), ord("w"), ord("q")])
    frames_seen: list[tuple[int, ...]] = []
    destroyed: list[bool] = []
    monkeypatch.setattr(cv2, "imshow", lambda name, frame: frames_seen.append(frame.shape))
    monkeypatch.setattr(cv2, "getWindowProperty", lambda name, property_id: 1.0)
    monkeypatch.setattr(cv2, "waitKey", lambda delay: next(keys))
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: destroyed.append(True))

    args = type(
        "Args",
        (),
        {
            "drone_mode": "simulator",
            "camera": None,
            "camera_index": 0,
            "record_dir": str(tmp_path),
        },
    )()
    assert cli.run_manual(args) == 0

    sessions = list(tmp_path.iterdir())
    assert len(sessions) == 1
    manifest = json.loads((sessions[0] / "manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["frame_count"] == 3
    assert manifest["event_count"] == 5
    assert manifest["provenance_file"] == "provenance.json"
    provenance_path = sessions[0] / manifest["provenance_file"]
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    assert provenance["mode"] == "manual"
    assert provenance["drone_mode"] == "simulator"
    assert provenance["camera_source"] == "blank"
    assert provenance["model_name"] is None
    assert provenance["model_sha256"] is None
    assert len(provenance["effective_config_sha256"]) == 64
    assert len(provenance["requirements_lock_sha256"]) == 64
    events = [json.loads(line) for line in (sessions[0] / "events.jsonl").read_text().splitlines()]
    assert [event["kind"] for event in events] == [
        "telemetry.snapshot",
        "manual.command",
        "manual.command",
        "manual.quit",
        "shutdown.land",
    ]
    assert events[0]["payload"]["sequence"] == 0
    assert events[0]["payload"]["state"]["is_connected"] is True
    assert events[0]["payload"]["state"]["is_flying"] is False
    assert events[1]["payload"]["command"]["action"] == "takeoff"
    assert events[2]["payload"]["command"]["action"] == "move_forward"
    assert events[4]["payload"]["safety_result"]["status"] == "APPROVED"
    assert (sessions[0] / "video.mp4").stat().st_size > 0
    assert len(frames_seen) == 3
    assert destroyed


def test_shutdown_records_real_landing_but_not_after_emergency() -> None:
    class SpyRecorder:
        def __init__(self) -> None:
            self.events: list[tuple[str, dict[str, Any]]] = []

        def record_event(self, kind: str, payload: dict[str, Any]) -> None:
            self.events.append((kind, payload))

    normal = SimulatedDrone()
    normal.connect()
    normal.takeoff()
    normal_recorder = SpyRecorder()
    cli._orderly_shutdown(  # type: ignore[arg-type]
        normal,
        SafetyController(normal),
        normal_recorder,
    )
    assert [kind for kind, _ in normal_recorder.events] == ["shutdown.land"]
    assert normal_recorder.events[0][1]["safety_result"]["status"] == "APPROVED"

    emergency = SimulatedDrone()
    emergency.connect()
    emergency.takeoff()
    emergency.emergency_stop()
    emergency_recorder = SpyRecorder()
    cli._orderly_shutdown(  # type: ignore[arg-type]
        emergency,
        SafetyController(emergency),
        emergency_recorder,
    )
    assert emergency_recorder.events == []


class LoopCamera:
    def read(self) -> np.ndarray:
        return np.zeros((120, 160, 3), dtype=np.uint8)


class VisibleDetector:
    def __init__(self, target: str = "person") -> None:
        self.target = target

    def detect(self, frame: Any) -> list[Detection]:
        return [Detection(self.target, 0.9, (60, 30, 100, 90))]


class SpyRecorder:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []
        self.frames = 0

    def record_event(self, kind: str, payload: dict[str, Any]) -> None:
        self.events.append((kind, payload))

    def record_frame(self, frame: Any) -> None:
        self.frames += 1


def flying_safety() -> SafetyController:
    drone = SimulatedDrone()
    drone.connect()
    drone.takeoff()
    return SafetyController(drone)


def test_follow_loop_records_control_tick_and_annotated_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cv2, "imshow", lambda *args: None)
    monkeypatch.setattr(cv2, "waitKey", lambda delay: ord("q"))
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)
    safety = flying_safety()
    pipeline = FollowPipeline(
        VisibleDetector(),
        PersonTracker(),
        FollowPersonAbility(),
        safety,
    )
    recorder = SpyRecorder()
    assert cli._follow_loop(LoopCamera(), pipeline, safety, 2.0, recorder) == 0  # type: ignore[arg-type]
    assert recorder.frames == 1
    assert recorder.events[0][0] == "follow.control_tick"
    assert recorder.events[0][1]["command"]["source"] == "ABILITY"
    assert recorder.events[0][1]["safety_result"]["status"] == "APPROVED"


def test_search_loop_records_terminal_tick_and_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cv2, "imshow", lambda *args: None)
    monkeypatch.setattr(cv2, "waitKey", lambda delay: -1)
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)
    safety = flying_safety()
    ability = VisualSearchAbility("backpack")
    pipeline = SearchPipeline(
        VisibleDetector("backpack"),
        TargetTracker("backpack"),
        ability,
        safety,
    )
    recorder = SpyRecorder()
    assert cli._search_loop(LoopCamera(), pipeline, safety, 2.0, recorder) == 0  # type: ignore[arg-type]
    assert recorder.frames == 1
    assert recorder.events[0][0] == "search.control_tick"
    assert recorder.events[0][1]["decision"]["state"] == "found"
    assert recorder.events[0][1]["safety_result"]["status"] == "APPROVED"
