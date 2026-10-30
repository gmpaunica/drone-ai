"""Stateless largest-person selection and image-relative tracking metrics."""

from __future__ import annotations

from collections.abc import Sequence
import math

from drone_ai.config import FollowConfig
from drone_ai.vision.models import Detection, HorizontalPosition, TargetDistance, TrackingResult


class TargetTracker:
    def __init__(self, target_class: str, config: FollowConfig | None = None) -> None:
        normalized = target_class.strip().lower()
        if not normalized:
            raise ValueError("target_class cannot be blank")
        self.target_class = normalized
        self.config = config or FollowConfig()

    def track(
        self,
        detections: Sequence[Detection],
        frame_shape: tuple[int, ...],
    ) -> TrackingResult:
        if len(frame_shape) < 2:
            raise ValueError("frame_shape must contain height and width")
        frame_height, frame_width = frame_shape[:2]
        if frame_width <= 0 or frame_height <= 0:
            raise ValueError("frame dimensions must be positive")

        targets = [item for item in detections if item.class_name.lower() == self.target_class]
        if not targets:
            return TrackingResult(
                visible=False,
                detection=None,
                horizontal_position=HorizontalPosition.UNKNOWN,
                distance=TargetDistance.UNKNOWN,
                horizontal_error=0.0,
                vertical_error=0.0,
                size_ratio=0.0,
            )

        target = max(targets, key=lambda item: (item.area, item.confidence))
        x1, y1, x2, y2 = target.bbox
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        horizontal_error = _clamp(2.0 * center_x / frame_width - 1.0)
        vertical_error = _clamp(2.0 * center_y / frame_height - 1.0)

        clipped_x1 = min(max(x1, 0.0), float(frame_width))
        clipped_x2 = min(max(x2, 0.0), float(frame_width))
        clipped_y1 = min(max(y1, 0.0), float(frame_height))
        clipped_y2 = min(max(y2, 0.0), float(frame_height))
        area = max(0.0, clipped_x2 - clipped_x1) * max(0.0, clipped_y2 - clipped_y1)
        size_ratio = _clamp(area / float(frame_width * frame_height), 0.0, 1.0)

        at_left_boundary = math.isclose(
            horizontal_error,
            -self.config.horizontal_dead_zone,
            abs_tol=1e-12,
        )
        at_right_boundary = math.isclose(
            horizontal_error,
            self.config.horizontal_dead_zone,
            abs_tol=1e-12,
        )
        if horizontal_error < -self.config.horizontal_dead_zone and not at_left_boundary:
            horizontal_position = HorizontalPosition.LEFT
        elif horizontal_error > self.config.horizontal_dead_zone and not at_right_boundary:
            horizontal_position = HorizontalPosition.RIGHT
        else:
            horizontal_position = HorizontalPosition.CENTER

        if size_ratio < self.config.target_size_min:
            distance = TargetDistance.FAR
        elif size_ratio > self.config.target_size_max:
            distance = TargetDistance.CLOSE
        else:
            distance = TargetDistance.GOOD

        return TrackingResult(
            visible=True,
            detection=target,
            horizontal_position=horizontal_position,
            distance=distance,
            horizontal_error=horizontal_error,
            vertical_error=vertical_error,
            size_ratio=size_ratio,
        )


class PersonTracker(TargetTracker):
    def __init__(self, config: FollowConfig | None = None) -> None:
        super().__init__("person", config)


def _clamp(value: float, minimum: float = -1.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))
