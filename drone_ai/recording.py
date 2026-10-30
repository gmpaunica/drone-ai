"""Opt-in, collision-safe video and JSONL evidence recording."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class RecordingError(RuntimeError):
    pass


class SessionEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    sequence: int = Field(ge=0)
    recorded_at: datetime
    kind: str = Field(min_length=1)
    payload: dict[str, Any]

    @field_validator("recorded_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("recorded_at must include a timezone")
        return value


class SessionManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    mode: str
    started_at: datetime
    ended_at: datetime
    status: str
    event_count: int = Field(ge=0)
    frame_count: int = Field(ge=0)
    events_file: str
    video_file: str | None
    provenance_file: str | None = None
    provenance_sha256: str | None = None

    @model_validator(mode="after")
    def validate_provenance_link(self) -> "SessionManifest":
        if (self.provenance_file is None) != (self.provenance_sha256 is None):
            raise ValueError("provenance file and hash must be supplied together")
        if self.provenance_sha256 is not None and not _is_sha256(self.provenance_sha256):
            raise ValueError("provenance_sha256 must be a lowercase SHA-256 digest")
        return self


class SessionProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    schema_version: int = Field(default=1, ge=1, le=1)
    created_at: datetime
    mode: str = Field(min_length=1)
    python_version: str = Field(min_length=1)
    platform: str = Field(min_length=1)
    drone_mode: str = Field(min_length=1)
    camera_source: str = Field(min_length=1)
    model_name: str | None = None
    model_sha256: str | None = None
    calibration_file: str | None = None
    calibration_sha256: str | None = None
    profile_file: str | None = None
    profile_sha256: str | None = None
    requirements_lock_sha256: str | None = None
    effective_config_sha256: str

    @field_validator("created_at")
    @classmethod
    def timezone_aware_created_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("provenance created_at must include a timezone")
        return value

    @field_validator(
        "model_sha256",
        "calibration_sha256",
        "profile_sha256",
        "requirements_lock_sha256",
        "effective_config_sha256",
    )
    @classmethod
    def validate_digest(cls, value: str | None) -> str | None:
        if value is not None and not _is_sha256(value):
            raise ValueError("provenance hashes must be lowercase SHA-256 digests")
        return value


class SessionRecorder:
    """Persist evidence only when explicitly constructed by a CLI flag."""

    def __init__(
        self,
        root: str | Path,
        mode: str,
        *,
        video_fps: float = 30.0,
        now: datetime | None = None,
        session_suffix: str | None = None,
        provenance: SessionProvenance | None = None,
    ) -> None:
        normalized_mode = mode.strip().lower()
        if not normalized_mode or not normalized_mode.replace("-", "").isalnum():
            raise ValueError("recording mode must contain only letters, digits, or hyphens")
        if video_fps <= 0:
            raise ValueError("recording fps must be positive")
        started = now or datetime.now(timezone.utc)
        if started.tzinfo is None or started.utcoffset() is None:
            raise ValueError("recording start time must include a timezone")
        suffix = session_suffix or uuid4().hex[:8]
        if not suffix.isalnum():
            raise ValueError("recording session suffix must be alphanumeric")
        session_name = f"{started.astimezone(timezone.utc):%Y%m%dT%H%M%SZ}-{normalized_mode}-{suffix}"
        root_path = Path(root)
        root_path.mkdir(parents=True, exist_ok=True)
        self.session_dir = root_path / session_name
        try:
            self.session_dir.mkdir(exist_ok=False)
        except FileExistsError as exc:
            raise RecordingError(f"recording session already exists: {self.session_dir}") from exc

        self.mode = normalized_mode
        self.video_fps = video_fps
        self.started_at = started
        self._events_path = self.session_dir / "events.jsonl"
        self._video_path = self.session_dir / "video.mp4"
        self._manifest_path = self.session_dir / "manifest.json"
        self._provenance_path = self.session_dir / "provenance.json"
        self._provenance_sha256: str | None = None
        if provenance is not None:
            provenance_bytes = provenance.model_dump_json(indent=2).encode("utf-8")
            self._provenance_path.write_bytes(provenance_bytes)
            self._provenance_sha256 = hashlib.sha256(provenance_bytes).hexdigest()
        self._event_stream = self._events_path.open("x", encoding="utf-8", newline="\n")
        self._writer: Any | None = None
        self._event_count = 0
        self._frame_count = 0
        self._closed = False

    @property
    def event_count(self) -> int:
        return self._event_count

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def record_event(self, kind: str, payload: dict[str, Any]) -> SessionEvent:
        self._require_open()
        try:
            json.dumps(payload, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise RecordingError(f"event payload is not finite JSON data: {exc}") from exc
        event = SessionEvent(
            sequence=self._event_count,
            recorded_at=datetime.now(timezone.utc),
            kind=kind,
            payload=payload,
        )
        try:
            line = json.dumps(
                event.model_dump(mode="json"),
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
        except (TypeError, ValueError) as exc:
            raise RecordingError(f"event payload is not finite JSON data: {exc}") from exc
        self._event_stream.write(line + "\n")
        self._event_stream.flush()
        self._event_count += 1
        return event

    def record_frame(self, frame: Any) -> None:
        self._require_open()
        shape = getattr(frame, "shape", None)
        if not isinstance(shape, tuple) or len(shape) != 3 or shape[2] != 3:
            raise RecordingError("recorded frames must have exactly three BGR channels")
        height, width = shape[:2]
        if height <= 0 or width <= 0:
            raise RecordingError("recorded frame dimensions must be positive")
        if self._writer is None:
            try:
                import cv2
            except ImportError as exc:
                raise RecordingError("OpenCV is required for video recording") from exc
            writer = cv2.VideoWriter(
                str(self._video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                self.video_fps,
                (width, height),
            )
            if not writer.isOpened():
                writer.release()
                raise RecordingError(f"could not create session video: {self._video_path}")
            self._writer = writer
        self._writer.write(frame)
        self._frame_count += 1

    def close(self, *, status: str = "closed") -> SessionManifest:
        if self._closed:
            return SessionManifest.model_validate_json(self._manifest_path.read_text("utf-8"))
        if self._writer is not None:
            self._writer.release()
        self._event_stream.close()
        ended = datetime.now(timezone.utc)
        manifest = SessionManifest(
            mode=self.mode,
            started_at=self.started_at,
            ended_at=ended,
            status=status,
            event_count=self._event_count,
            frame_count=self._frame_count,
            events_file=self._events_path.name,
            video_file=self._video_path.name if self._frame_count else None,
            provenance_file=(
                self._provenance_path.name if self._provenance_sha256 is not None else None
            ),
            provenance_sha256=self._provenance_sha256,
        )
        self._manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        self._closed = True
        return manifest

    def _require_open(self) -> None:
        if self._closed:
            raise RecordingError("recording session is closed")


def _is_sha256(value: str) -> bool:
    return re.fullmatch(r"[0-9a-f]{64}", value) is not None
