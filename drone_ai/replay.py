"""Hardware-free analysis of recorded video frames."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from drone_ai.vision.models import Detection, TrackingResult
from drone_ai.vision.tracker import TargetTracker


class ReplayDetection(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    class_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: tuple[float, float, float, float]


class ReplayTracking(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    visible: bool
    horizontal_position: str
    distance: str
    horizontal_error: float = Field(ge=-1.0, le=1.0)
    vertical_error: float = Field(ge=-1.0, le=1.0)
    size_ratio: float = Field(ge=0.0, le=1.0)


class ReplayFrameRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    frame_index: int = Field(ge=0)
    timestamp_s: float = Field(ge=0.0)
    target_class: str
    detections: tuple[ReplayDetection, ...]
    tracking: ReplayTracking


class ReplaySummary(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    target_class: str
    total_frames: int = Field(ge=0)
    frames_with_target: int = Field(ge=0)
    maximum_confidence: float | None = Field(default=None, ge=0.0, le=1.0)


@dataclass(frozen=True, slots=True)
class ReplayFrame:
    detections: tuple[Detection, ...]
    tracking: TrackingResult
    record: ReplayFrameRecord


class ReplayDetector(Protocol):
    def detect(self, frame: Any) -> list[Detection]: ...


class VisionReplay:
    def __init__(
        self,
        detector: ReplayDetector,
        tracker: TargetTracker,
        *,
        fps: float,
    ) -> None:
        if fps <= 0:
            raise ValueError("replay fps must be positive")
        self.detector = detector
        self.tracker = tracker
        self.fps = fps
        self._total_frames = 0
        self._frames_with_target = 0
        self._maximum_confidence: float | None = None

    def process(self, frame: Any) -> ReplayFrame:
        frame_index = self._total_frames
        detections = tuple(self.detector.detect(frame))
        tracking = self.tracker.track(detections, frame.shape)
        if tracking.visible:
            self._frames_with_target += 1
        for detection in detections:
            if self._maximum_confidence is None or detection.confidence > self._maximum_confidence:
                self._maximum_confidence = detection.confidence
        self._total_frames += 1
        record = ReplayFrameRecord(
            frame_index=frame_index,
            timestamp_s=frame_index / self.fps,
            target_class=self.tracker.target_class,
            detections=tuple(
                ReplayDetection(
                    class_name=item.class_name,
                    confidence=item.confidence,
                    bbox=item.bbox,
                )
                for item in detections
            ),
            tracking=ReplayTracking(
                visible=tracking.visible,
                horizontal_position=tracking.horizontal_position.value,
                distance=tracking.distance.value,
                horizontal_error=tracking.horizontal_error,
                vertical_error=tracking.vertical_error,
                size_ratio=tracking.size_ratio,
            ),
        )
        return ReplayFrame(detections=detections, tracking=tracking, record=record)

    def summary(self) -> ReplaySummary:
        return ReplaySummary(
            target_class=self.tracker.target_class,
            total_frames=self._total_frames,
            frames_with_target=self._frames_with_target,
            maximum_confidence=self._maximum_confidence,
        )
