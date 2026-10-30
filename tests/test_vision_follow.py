from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from drone_ai.abilities.follow_person import FollowPersonAbility
from drone_ai.commands import ActionType, CommandSource
from drone_ai.config import FollowConfig
from drone_ai.vision.detector import VisionError, YOLODetector
from drone_ai.vision.models import Detection, HorizontalPosition, TargetDistance, TrackingResult
from drone_ai.vision.tracker import PersonTracker


class FakeBox:
    def __init__(self, class_id: int, confidence: float, bbox: list[float]) -> None:
        self.cls = [class_id]
        self.conf = [confidence]
        self.xyxy = [bbox]


class FakeResult:
    names = {0: "person", 1: "dog"}

    def __init__(self) -> None:
        self.boxes = [
            FakeBox(0, 0.9, [10, 20, 50, 80]),
            FakeBox(1, 0.95, [0, 0, 20, 20]),
            FakeBox(0, 0.4, [50, 50, 80, 90]),
        ]


class FakeModel:
    names = FakeResult.names

    def __init__(self) -> None:
        self.calls: list[tuple[object, dict[str, object]]] = []

    def __call__(self, frame: object, **kwargs: object) -> list[FakeResult]:
        self.calls.append((frame, kwargs))
        return [FakeResult()]


def test_yolo_conversion_filters_class_and_confidence() -> None:
    model = FakeModel()
    detector = YOLODetector(model=model, confidence=0.5, target_classes=("person",))
    frame = object()
    detections = detector.detect(frame)
    assert detections == [Detection("person", 0.9, (10.0, 20.0, 50.0, 80.0))]
    assert model.calls == [(frame, {"conf": 0.5, "verbose": False})]


def test_yolo_validates_configured_target_classes_before_flight() -> None:
    detector = YOLODetector(model=FakeModel(), target_classes=("person", "dog"))
    detector.validate_target_classes()
    unknown = YOLODetector(model=FakeModel(), target_classes=("backpack",))
    with pytest.raises(VisionError, match="not supported"):
        unknown.validate_target_classes()


def test_detection_and_tracking_records_are_immutable() -> None:
    detection = Detection("person", 0.8, (0, 0, 10, 10))
    with pytest.raises(FrozenInstanceError):
        detection.confidence = 0.2  # type: ignore[misc]


def test_tracker_selects_largest_person_not_other_classes() -> None:
    detections = [
        Detection("person", 0.95, (0, 0, 20, 20)),
        Detection("person", 0.60, (25, 25, 75, 75)),
        Detection("dog", 0.99, (0, 0, 100, 100)),
    ]
    result = PersonTracker().track(detections, (100, 100, 3))
    assert result.detection == detections[1]


def test_tracker_uses_confidence_to_break_area_tie() -> None:
    detections = [
        Detection("person", 0.50, (0, 0, 20, 20)),
        Detection("person", 0.90, (70, 70, 90, 90)),
    ]
    result = PersonTracker().track(detections, (100, 100, 3))
    assert result.detection == detections[1]


def test_normalized_errors_and_size_classification() -> None:
    tracker = PersonTracker()
    result = tracker.track([Detection("person", 0.9, (50, 25, 90, 75))], (100, 100, 3))
    assert result.horizontal_error == pytest.approx(0.4)
    assert result.vertical_error == pytest.approx(0.0)
    assert result.size_ratio == pytest.approx(0.2)
    assert result.horizontal_position is HorizontalPosition.RIGHT
    assert result.distance is TargetDistance.GOOD


@pytest.mark.parametrize(
    ("bbox", "distance"),
    [
        ((45, 45, 55, 55), TargetDistance.FAR),
        ((30, 30, 70, 70), TargetDistance.GOOD),
        ((20, 20, 80, 80), TargetDistance.CLOSE),
    ],
)
def test_size_classifications(
    bbox: tuple[float, float, float, float], distance: TargetDistance
) -> None:
    result = PersonTracker().track([Detection("person", 0.9, bbox)], (100, 100, 3))
    assert result.distance is distance


def test_dead_zone_boundaries_are_centered() -> None:
    tracker = PersonTracker(FollowConfig(horizontal_dead_zone=0.12))
    # Width 20 and centers at x=44 and x=56 -> errors exactly -0.12/+0.12.
    for bbox in ((34, 40, 54, 60), (46, 40, 66, 60)):
        result = tracker.track([Detection("person", 0.9, bbox)], (100, 100, 3))
        assert result.horizontal_position is HorizontalPosition.CENTER


def test_missing_person_output() -> None:
    result = PersonTracker().track([Detection("dog", 0.9, (0, 0, 50, 50))], (100, 100, 3))
    assert not result.visible
    assert result.detection is None
    assert result.horizontal_position is HorizontalPosition.UNKNOWN
    assert result.distance is TargetDistance.UNKNOWN


def tracking(
    *,
    visible: bool = True,
    horizontal: HorizontalPosition = HorizontalPosition.CENTER,
    distance: TargetDistance = TargetDistance.GOOD,
) -> TrackingResult:
    detection = Detection("person", 0.9, (40, 40, 60, 60)) if visible else None
    horizontal_error = {
        HorizontalPosition.LEFT: -0.5,
        HorizontalPosition.CENTER: 0.0,
        HorizontalPosition.RIGHT: 0.5,
        HorizontalPosition.UNKNOWN: 0.0,
    }[horizontal]
    return TrackingResult(
        visible=visible,
        detection=detection,
        horizontal_position=horizontal if visible else HorizontalPosition.UNKNOWN,
        distance=distance if visible else TargetDistance.UNKNOWN,
        horizontal_error=horizontal_error if visible else 0.0,
        vertical_error=0.0,
        size_ratio=0.1 if visible else 0.0,
    )


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        (tracking(horizontal=HorizontalPosition.LEFT), ActionType.ROTATE_LEFT),
        (tracking(horizontal=HorizontalPosition.RIGHT), ActionType.ROTATE_RIGHT),
        (tracking(distance=TargetDistance.FAR), ActionType.MOVE_FORWARD),
        (tracking(distance=TargetDistance.CLOSE), ActionType.MOVE_BACKWARD),
        (tracking(distance=TargetDistance.GOOD), ActionType.STOP),
        (tracking(visible=False), ActionType.STOP),
    ],
)
def test_follow_decisions(result: TrackingResult, expected: ActionType) -> None:
    command = FollowPersonAbility().decide(result)
    assert command.action is expected
    assert command.source is CommandSource.ABILITY


def test_horizontal_correction_has_priority_over_distance() -> None:
    command = FollowPersonAbility().decide(
        tracking(horizontal=HorizontalPosition.LEFT, distance=TargetDistance.FAR)
    )
    assert command.action is ActionType.ROTATE_LEFT
    assert command.degrees == 10.0


@pytest.mark.parametrize("error", [-0.12, 0.12])
def test_follow_dead_zone_boundaries_do_not_rotate(error: float) -> None:
    result = TrackingResult(
        visible=True,
        detection=Detection("person", 0.9, (40, 40, 60, 60)),
        horizontal_position=HorizontalPosition.CENTER,
        distance=TargetDistance.GOOD,
        horizontal_error=error,
        vertical_error=0.6,
        size_ratio=0.1,
    )
    assert FollowPersonAbility().decide(result).action is ActionType.STOP
