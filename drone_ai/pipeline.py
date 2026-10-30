"""One testable control tick for the vision-follow-safety vertical slice."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Protocol

from drone_ai.abilities.follow_person import FollowPersonAbility
from drone_ai.commands import ActionCommand
from drone_ai.safety import SafetyController, SafetyResult
from drone_ai.vision.models import Detection, TrackingResult
from drone_ai.vision.tracker import PersonTracker


LOGGER = logging.getLogger(__name__)


class Detector(Protocol):
    def detect(self, frame: Any) -> list[Detection]: ...


@dataclass(frozen=True, slots=True)
class FollowTick:
    detections: tuple[Detection, ...]
    tracking: TrackingResult
    command: ActionCommand
    safety_result: SafetyResult


@dataclass(frozen=True, slots=True)
class FollowDecision:
    detections: tuple[Detection, ...]
    tracking: TrackingResult
    command: ActionCommand


class FollowPipeline:
    def __init__(
        self,
        detector: Detector,
        tracker: PersonTracker,
        ability: FollowPersonAbility,
        safety: SafetyController,
    ) -> None:
        self.detector = detector
        self.tracker = tracker
        self.ability = ability
        self.safety = safety

    def evaluate(self, frame: Any) -> FollowDecision:
        detections = self.detector.detect(frame)
        LOGGER.info("[VISION] %d filtered detection(s)", len(detections))
        tracking = self.tracker.track(detections, frame.shape)
        LOGGER.info(
            "[TRACKING] visible=%s horizontal=%s distance=%s h_error=%.3f v_error=%.3f size=%.3f",
            tracking.visible,
            tracking.horizontal_position.value,
            tracking.distance.value,
            tracking.horizontal_error,
            tracking.vertical_error,
            tracking.size_ratio,
        )
        command = self.ability.decide(tracking)
        LOGGER.info("[ABILITY] %s", command.action.value)
        return FollowDecision(tuple(detections), tracking, command)

    def tick(self, frame: Any) -> FollowTick:
        decision = self.evaluate(frame)
        result = self.safety.execute(decision.command)
        return FollowTick(
            decision.detections,
            decision.tracking,
            decision.command,
            result,
        )
