from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pytest

from drone_ai.cli import build_argument_parser, main
from drone_ai.commands import ActionCommand, ActionType, CommandSource
from drone_ai.recording import SessionProvenance, SessionRecorder
from drone_ai.recording_audit import SessionAuditStatus, SessionAuditor
from drone_ai.safety import SafetyResult, SafetyStatus
from drone_ai.watchdog import TelemetryState, WatchdogTelemetrySnapshot


def recorder(tmp_path: Path, suffix: str = "audit001") -> SessionRecorder:
    now = datetime.now(timezone.utc)
    return SessionRecorder(
        tmp_path,
        "manual",
        now=now,
        session_suffix=suffix,
        provenance=SessionProvenance(
            created_at=now,
            mode="manual",
            python_version="3.11.test",
            platform="test-platform",
            drone_mode="simulator",
            camera_source="blank",
            effective_config_sha256="0" * 64,
        ),
    )


def telemetry_payload(*, error: str | None = None) -> dict:
    snapshot = (
        WatchdogTelemetrySnapshot(
            sequence=0,
            checked_at_monotonic_s=1.0,
            error=error,
        )
        if error is not None
        else WatchdogTelemetrySnapshot(
            sequence=0,
            checked_at_monotonic_s=1.0,
            state=TelemetryState(
                x_m=0.0,
                y_m=0.0,
                altitude_m=0.0,
                yaw_degrees=0.0,
                is_flying=False,
                is_connected=True,
                battery_percent=100,
                speed_mps=0.2,
            ),
        )
    )
    return snapshot.model_dump(mode="json")


def manual_action_payload(action: ActionType) -> dict:
    command = ActionCommand(action=action, source=CommandSource.MANUAL)
    result = SafetyResult(status=SafetyStatus.APPROVED, reason=f"executed {action.value}")
    return {
        "key": "x",
        "command": command.model_dump(mode="json"),
        "safety_result": result.model_dump(mode="json"),
        "suppressed": False,
        "reason": "executed",
    }


def check(report, name: str):
    return next(item for item in report.checks if item.name == name)


def test_valid_session_audits_manifest_events_telemetry_and_video(tmp_path: Path) -> None:
    session = recorder(tmp_path)
    session.record_event("telemetry.snapshot", telemetry_payload())
    session.record_frame(np.zeros((120, 160, 3), dtype=np.uint8))
    session.record_frame(np.zeros((120, 160, 3), dtype=np.uint8))
    session_dir = session.session_dir
    session.close(status="completed")

    report = SessionAuditor(session_dir).run()
    assert report.passed is True
    assert report.offline_only is True
    assert all(item.status is SessionAuditStatus.PASS for item in report.checks)
    assert "decoded 2 BGR frame" in check(report, "video").detail


def test_captured_telemetry_error_is_a_warning_not_integrity_failure(tmp_path: Path) -> None:
    session = recorder(tmp_path, "audit002")
    session.record_event("telemetry.snapshot", telemetry_payload(error="link down"))
    session_dir = session.session_dir
    session.close(status="ended_with_error")
    report = SessionAuditor(session_dir).run()
    assert report.passed is True
    assert check(report, "telemetry").status is SessionAuditStatus.WARN
    assert "1 captured telemetry error" in check(report, "telemetry").detail


def test_unterminated_takeoff_fails_but_emergency_is_terminal(tmp_path: Path) -> None:
    incomplete = recorder(tmp_path, "audit003")
    incomplete.record_event("manual.command", manual_action_payload(ActionType.TAKEOFF))
    incomplete_dir = incomplete.session_dir
    incomplete.close(status="completed")
    failed = SessionAuditor(incomplete_dir).run()
    assert failed.passed is False
    assert check(failed, "flight_lifecycle").status is SessionAuditStatus.FAIL
    assert "no later approved landing" in check(failed, "flight_lifecycle").detail

    emergency = recorder(tmp_path, "audit004")
    emergency.record_event("manual.command", manual_action_payload(ActionType.TAKEOFF))
    emergency.record_event("manual.command", manual_action_payload(ActionType.EMERGENCY_STOP))
    emergency_dir = emergency.session_dir
    emergency.close(status="completed")
    accepted = SessionAuditor(emergency_dir).run()
    assert accepted.passed is True
    assert check(accepted, "flight_lifecycle").status is SessionAuditStatus.PASS


def test_tampered_event_sequence_and_video_count_fail(tmp_path: Path) -> None:
    bad_events = recorder(tmp_path, "audit005")
    bad_events.record_event("note", {"value": 1})
    bad_events_dir = bad_events.session_dir
    bad_events.close(status="completed")
    events_path = bad_events_dir / "events.jsonl"
    payload = json.loads(events_path.read_text(encoding="utf-8"))
    payload["sequence"] = 4
    events_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    events_report = SessionAuditor(bad_events_dir).run()
    assert events_report.passed is False
    assert "not contiguous" in check(events_report, "events").detail

    bad_video = recorder(tmp_path, "audit006")
    bad_video.record_frame(np.zeros((120, 160, 3), dtype=np.uint8))
    bad_video_dir = bad_video.session_dir
    bad_video.close(status="completed")
    manifest_path = bad_video_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["frame_count"] = 2
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    video_report = SessionAuditor(bad_video_dir).run()
    assert video_report.passed is False
    assert "video decodes 1" in check(video_report, "video").detail


def test_manifest_cannot_escape_session_directory(tmp_path: Path) -> None:
    session = recorder(tmp_path, "audit007")
    session_dir = session.session_dir
    session.close(status="completed")
    manifest_path = session_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["events_file"] = "../events.jsonl"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    report = SessionAuditor(session_dir).run()
    assert report.passed is False
    assert "unsafe child path" in check(report, "manifest").detail


def test_provenance_tampering_is_detected_by_manifest_hash(tmp_path: Path) -> None:
    session = recorder(tmp_path, "audit009")
    session_dir = session.session_dir
    manifest = session.close(status="completed")
    assert manifest.provenance_file is not None
    provenance_path = session_dir / manifest.provenance_file
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["camera_source"] = "tampered"
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    report = SessionAuditor(session_dir).run()
    assert report.passed is False
    assert check(report, "provenance").status is SessionAuditStatus.FAIL
    assert "does not match" in check(report, "provenance").detail


def test_audit_session_cli_is_offline_and_emits_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    session = recorder(tmp_path, "audit008")
    session.record_event("telemetry.snapshot", telemetry_payload())
    session_dir = session.session_dir
    session.close(status="completed")

    class ForbiddenDrone:
        def __init__(self, *args, **kwargs) -> None:
            raise AssertionError("session audit constructed a drone")

    monkeypatch.setattr("drone_ai.cli.SimulatedDrone", ForbiddenDrone)
    assert main(["audit-session", str(session_dir), "--json"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["passed"] is True
    assert output["offline_only"] is True
    args = build_argument_parser().parse_args(["audit-session", str(session_dir)])
    assert not hasattr(args, "drone_mode")
    assert not hasattr(args, "takeoff")
