"""Detector-independent immutable vision records."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math


class HorizontalPosition(str, Enum):
    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"
    UNKNOWN = "unknown"


class TargetDistance(str, Enum):
    FAR = "far"
    GOOD = "good"
    CLOSE = "close"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class Detection:
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        if not self.class_name:
            raise ValueError("class_name cannot be empty")
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be finite and between 0 and 1")
        if len(self.bbox) != 4 or not all(math.isfinite(value) for value in self.bbox):
            raise ValueError("bbox must contain four finite coordinates")
        x1, y1, x2, y2 = self.bbox
        if x2 <= x1 or y2 <= y1:
            raise ValueError("bbox must have positive width and height")

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)


@dataclass(frozen=True, slots=True)
class TrackingResult:
    visible: bool
    detection: Detection | None
    horizontal_position: HorizontalPosition
    distance: TargetDistance
    horizontal_error: float
    vertical_error: float
    size_ratio: float

    def __post_init__(self) -> None:
        if not -1.0 <= self.horizontal_error <= 1.0:
            raise ValueError("horizontal_error must be normalized to [-1, 1]")
        if not -1.0 <= self.vertical_error <= 1.0:
            raise ValueError("vertical_error must be normalized to [-1, 1]")
        if not 0.0 <= self.size_ratio <= 1.0:
            raise ValueError("size_ratio must be between 0 and 1")
        if self.visible != (self.detection is not None):
            raise ValueError("visible must agree with detection presence")
