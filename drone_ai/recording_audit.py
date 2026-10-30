"""Offline integrity and flight-lifecycle audit for opt-in session evidence."""

from __future__ import annotations

from enum import Enum
import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from drone_ai.commands import ActionCommand, ActionType
from drone_ai.recording import SessionEvent, SessionManifest, SessionProvenance
from drone_ai.safety import SafetyResult
from drone_ai.watchdog import WatchdogTelemetrySnapshot


class SessionAuditStatus(str, Enum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


class SessionAuditCheck(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    name: str = Field(min_length=1)
    status: SessionAuditStatus
    detail: str = Field(min_length=1)


class SessionAuditReport(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    session_dir: str = Field(min_length=1)
    checks: tuple[SessionAuditCheck, ...] = Field(min_length=1)
    passed: bool
    offline_only: bool = True

    @model_validator(mode="after")
    def derive_truth_and_boundary(self) -> "SessionAuditReport":
        expected = all(check.status is not SessionAuditStatus.FAIL for check in self.checks)
        if self.passed != expected:
            raise ValueError("session audit passed flag must match its checks")
        if not self.offline_only:
            raise ValueError("session audit must remain offline-only")
        return self


class SessionAuditor:
    def __init__(self, session_dir: str | Path) -> None:
        self.session_dir = Path(session_dir)

    def run(self) -> SessionAuditReport:
        checks: list[SessionAuditCheck] = []
        manifest, manifest_check = self._load_manifest()
        checks.append(manifest_check)
        if manifest is None:
            return self._report(checks)

        checks.append(self._audit_provenance(manifest))

        events, event_check = self._load_events(manifest)
        checks.append(event_check)
        if events is None:
            checks.extend(
                [
                    SessionAuditCheck(
                        name="telemetry",
                        status=SessionAuditStatus.FAIL,
                        detail="telemetry cannot be audited because events are invalid",
                    ),
                    SessionAuditCheck(
                        name="flight_lifecycle",
                        status=SessionAuditStatus.FAIL,
                        detail="flight lifecycle cannot be audited because events are invalid",
                    ),
                ]
            )
        else:
            checks.append(self._audit_telemetry(events))
            checks.append(self._audit_lifecycle(events))
        checks.append(self._audit_video(manifest))
        return self._report(checks)

    def _audit_provenance(self, manifest: SessionManifest) -> SessionAuditCheck:
        if manifest.provenance_file is None or manifest.provenance_sha256 is None:
            return SessionAuditCheck(
                name="provenance",
                status=SessionAuditStatus.WARN,
                detail="session has no provenance sidecar (legacy or direct recorder use)",
            )
        path = self.session_dir / manifest.provenance_file
        if not path.is_file():
            return self._fail("provenance", f"provenance file does not exist: {path}")
        try:
            content = path.read_bytes()
        except OSError as exc:
            return self._fail("provenance", f"could not read provenance: {exc}")
        digest = hashlib.sha256(content).hexdigest()
        if digest != manifest.provenance_sha256:
            return self._fail(
                "provenance",
                "provenance SHA-256 does not match the manifest",
            )
        try:
            provenance = SessionProvenance.model_validate_json(content)
        except Exception as exc:
            return self._fail("provenance", f"invalid provenance schema: {exc}")
        if provenance.mode != manifest.mode:
            return self._fail(
                "provenance",
                f"provenance mode {provenance.mode!r} does not match manifest mode {manifest.mode!r}",
            )
        fingerprint_count = sum(
            value is not None
            for value in (
                provenance.requirements_lock_sha256,
                provenance.model_sha256,
                provenance.calibration_sha256,
                provenance.profile_sha256,
            )
        )
        return SessionAuditCheck(
            name="provenance",
            status=SessionAuditStatus.PASS,
            detail=(
                f"schema v{provenance.schema_version}, effective config and "
                f"{fingerprint_count} file fingerprint(s) verified"
            ),
        )

    def _load_manifest(
        self,
    ) -> tuple[SessionManifest | None, SessionAuditCheck]:
        if not self.session_dir.is_dir():
            return None, self._fail("manifest", f"session directory does not exist: {self.session_dir}")
        path = self.session_dir / "manifest.json"
        if not path.is_file():
            return None, self._fail("manifest", f"manifest does not exist: {path}")
        try:
            manifest = SessionManifest.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception as exc:
            return None, self._fail("manifest", f"invalid manifest: {exc}")
        if any(
            value.tzinfo is None or value.utcoffset() is None
            for value in (manifest.started_at, manifest.ended_at)
        ):
            return None, self._fail("manifest", "manifest timestamps must include a timezone")
        if manifest.ended_at < manifest.started_at:
            return None, self._fail("manifest", "manifest ended_at precedes started_at")
        if not manifest.status.strip() or not manifest.mode.strip():
            return None, self._fail("manifest", "manifest mode/status cannot be blank")
        names = [manifest.events_file]
        if manifest.video_file is not None:
            names.append(manifest.video_file)
        if manifest.provenance_file is not None:
            names.append(manifest.provenance_file)
        unsafe = [name for name in names if not _safe_child_name(name)]
        if unsafe:
            return None, self._fail(
                "manifest",
                f"manifest contains unsafe child path: {unsafe[0]!r}",
            )
        return manifest, SessionAuditCheck(
            name="manifest",
            status=SessionAuditStatus.PASS,
            detail=(
                f"mode={manifest.mode}, status={manifest.status}, "
                f"events={manifest.event_count}, frames={manifest.frame_count}"
            ),
        )

    def _load_events(
        self,
        manifest: SessionManifest,
    ) -> tuple[tuple[SessionEvent, ...] | None, SessionAuditCheck]:
        path = self.session_dir / manifest.events_file
        if not path.is_file():
            return None, self._fail("events", f"events file does not exist: {path}")
        events: list[SessionEvent] = []
        try:
            with path.open("r", encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, start=1):
                    if not line.strip():
                        return None, self._fail(
                            "events",
                            f"blank event record at line {line_number}",
                        )
                    try:
                        events.append(SessionEvent.model_validate_json(line))
                    except Exception as exc:
                        return None, self._fail(
                            "events",
                            f"invalid event at line {line_number}: {exc}",
                        )
        except OSError as exc:
            return None, self._fail("events", f"could not read events: {exc}")
        if len(events) != manifest.event_count:
            return None, self._fail(
                "events",
                f"manifest declares {manifest.event_count} events but file contains {len(events)}",
            )
        sequences = [event.sequence for event in events]
        if sequences != list(range(len(events))):
            return None, self._fail("events", "event sequences are not contiguous from zero")
        timestamps = [event.recorded_at for event in events]
        if timestamps != sorted(timestamps):
            return None, self._fail("events", "event timestamps are not monotonic")
        if any(
            timestamp < manifest.started_at or timestamp > manifest.ended_at
            for timestamp in timestamps
        ):
            return None, self._fail("events", "event timestamp falls outside manifest bounds")
        return tuple(events), SessionAuditCheck(
            name="events",
            status=SessionAuditStatus.PASS,
            detail=f"{len(events)} strict, contiguous, time-bounded event(s)",
        )

    def _audit_telemetry(self, events: tuple[SessionEvent, ...]) -> SessionAuditCheck:
        snapshots: list[WatchdogTelemetrySnapshot] = []
        for event in events:
            if event.kind != "telemetry.snapshot":
                continue
            try:
                snapshots.append(_validate_json_payload(WatchdogTelemetrySnapshot, event.payload))
            except Exception as exc:
                return self._fail(
                    "telemetry",
                    f"invalid telemetry snapshot at event {event.sequence}: {exc}",
                )
        if not snapshots:
            return SessionAuditCheck(
                name="telemetry",
                status=SessionAuditStatus.WARN,
                detail="session contains no telemetry.snapshot events",
            )
        sequences = [snapshot.sequence for snapshot in snapshots]
        if sequences != list(range(len(snapshots))):
            return self._fail(
                "telemetry",
                "watchdog telemetry sequences are not contiguous from zero",
            )
        times = [snapshot.checked_at_monotonic_s for snapshot in snapshots]
        if times != sorted(times):
            return self._fail("telemetry", "watchdog monotonic sample times regress")
        errors = sum(snapshot.error is not None for snapshot in snapshots)
        if errors:
            return SessionAuditCheck(
                name="telemetry",
                status=SessionAuditStatus.WARN,
                detail=f"{len(snapshots)} snapshot(s) are valid; {errors} captured telemetry error(s)",
            )
        return SessionAuditCheck(
            name="telemetry",
            status=SessionAuditStatus.PASS,
            detail=f"{len(snapshots)} contiguous state snapshot(s), no telemetry errors",
        )

    def _audit_lifecycle(self, events: tuple[SessionEvent, ...]) -> SessionAuditCheck:
        airborne = False
        terminal_actions = 0
        emergency_seen = False
        flight_actions = 0
        for event in events:
            try:
                action, approved = _event_flight_action(event)
            except Exception as exc:
                return self._fail(
                    "flight_lifecycle",
                    f"invalid lifecycle payload at event {event.sequence}: {exc}",
                )
            if action is None or not approved:
                continue
            flight_actions += 1
            if action is ActionType.TAKEOFF:
                if airborne:
                    return self._fail(
                        "flight_lifecycle",
                        f"approved takeoff at event {event.sequence} while already airborne",
                    )
                airborne = True
            elif action is ActionType.LAND:
                if emergency_seen:
                    return self._fail(
                        "flight_lifecycle",
                        f"landing claimed at event {event.sequence} after emergency stop",
                    )
                airborne = False
                terminal_actions += 1
            elif action is ActionType.EMERGENCY_STOP:
                airborne = False
                emergency_seen = True
                terminal_actions += 1
        if airborne:
            return self._fail(
                "flight_lifecycle",
                "approved takeoff has no later approved landing or emergency stop",
            )
        return SessionAuditCheck(
            name="flight_lifecycle",
            status=SessionAuditStatus.PASS,
            detail=(
                f"{flight_actions} approved lifecycle action(s), "
                f"{terminal_actions} terminal action(s)"
            ),
        )

    def _audit_video(self, manifest: SessionManifest) -> SessionAuditCheck:
        if manifest.frame_count == 0:
            if manifest.video_file is not None:
                return self._fail("video", "zero-frame manifest must not name a video file")
            return SessionAuditCheck(
                name="video",
                status=SessionAuditStatus.PASS,
                detail="session correctly contains no video",
            )
        if manifest.video_file is None:
            return self._fail("video", "positive frame count requires a video file")
        path = self.session_dir / manifest.video_file
        if not path.is_file():
            return self._fail("video", f"video file does not exist: {path}")
        try:
            import cv2
        except ImportError as exc:
            return self._fail("video", f"OpenCV unavailable for video audit: {exc}")
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            capture.release()
            return self._fail("video", f"could not open recorded video: {path}")
        count = 0
        resolution: tuple[int, int] | None = None
        try:
            while True:
                ok, frame = capture.read()
                if not ok or frame is None:
                    break
                shape = getattr(frame, "shape", None)
                if not isinstance(shape, tuple) or len(shape) != 3 or shape[2] != 3:
                    return self._fail("video", f"frame {count} is not three-channel BGR")
                current = (shape[1], shape[0])
                if resolution is None:
                    resolution = current
                elif resolution != current:
                    return self._fail("video", f"video resolution changes at frame {count}")
                count += 1
        finally:
            capture.release()
        if count != manifest.frame_count:
            return self._fail(
                "video",
                f"manifest declares {manifest.frame_count} frames but video decodes {count}",
            )
        assert resolution is not None
        return SessionAuditCheck(
            name="video",
            status=SessionAuditStatus.PASS,
            detail=f"decoded {count} BGR frame(s) at {resolution[0]}x{resolution[1]}",
        )

    def _report(self, checks: list[SessionAuditCheck]) -> SessionAuditReport:
        passed = all(check.status is not SessionAuditStatus.FAIL for check in checks)
        return SessionAuditReport(
            session_dir=str(self.session_dir),
            checks=tuple(checks),
            passed=passed,
            offline_only=True,
        )

    @staticmethod
    def _fail(name: str, detail: str) -> SessionAuditCheck:
        return SessionAuditCheck(name=name, status=SessionAuditStatus.FAIL, detail=detail)


def _safe_child_name(name: str) -> bool:
    candidate = Path(name)
    return (
        bool(name)
        and not candidate.is_absolute()
        and len(candidate.parts) == 1
        and candidate.name == name
        and name not in {".", ".."}
    )


def _validate_json_payload(model: Any, payload: dict[str, Any]) -> Any:
    return model.model_validate_json(
        json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    )


def _event_flight_action(event: SessionEvent) -> tuple[ActionType | None, bool]:
    if event.kind == "manual.command":
        command_payload = event.payload.get("command")
        safety_payload = event.payload.get("safety_result")
        if not isinstance(command_payload, dict) or not isinstance(safety_payload, dict):
            raise ValueError("manual.command requires command and safety_result objects")
        command = _validate_json_payload(ActionCommand, command_payload)
        result = _validate_json_payload(SafetyResult, safety_payload)
        if command.action in {ActionType.TAKEOFF, ActionType.LAND, ActionType.EMERGENCY_STOP}:
            return command.action, result.approved
        return None, False
    if event.kind in {"takeoff", "shutdown.land", "emergency"}:
        safety_payload = event.payload.get("safety_result")
        if not isinstance(safety_payload, dict):
            raise ValueError(f"{event.kind} requires a safety_result object")
        result = _validate_json_payload(SafetyResult, safety_payload)
        action = {
            "takeoff": ActionType.TAKEOFF,
            "shutdown.land": ActionType.LAND,
            "emergency": ActionType.EMERGENCY_STOP,
        }[event.kind]
        return action, result.approved
    if event.kind == "watchdog.landing":
        safety_payload = event.payload.get("safety_result")
        if not isinstance(safety_payload, dict):
            raise ValueError("watchdog.landing requires a safety_result object")
        result = _validate_json_payload(SafetyResult, safety_payload)
        return ActionType.LAND, result.approved
    return None, False
