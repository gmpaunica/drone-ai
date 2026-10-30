"""One testable camera-to-search-to-safety control tick."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Protocol

from drone_ai.abilities.visual_search import SearchDecision, VisualSearchAbility
from drone_ai.safety import SafetyController, SafetyResult
from drone_ai.vision.models import Detection, TrackingResult
from drone_ai.vision.tracker import TargetTracker


LOGGER = logging.getLogger(__name__)


class Detector(Protocol):
    def detect(self, frame: Any) -> list[Detection]: ...


@dataclass(frozen=True, slots=True)
class SearchTick:
    detections: tuple[Detection, ...]
    tracking: TrackingResult
    decision: SearchDecision
    safety_result: SafetyResult


class SearchPipeline:
    def __init__(
        self,
        detector: Detector,
        tracker: TargetTracker,
        ability: VisualSearchAbility,
        safety: SafetyController,
    ) -> None:
        self.detector = detector
        self.tracker = tracker
        self.ability = ability
        self.safety = safety

    def tick(self, frame: Any) -> SearchTick:
        detections = self.detector.detect(frame)
        LOGGER.info("[VISION] %d %s detection(s)", len(detections), self.ability.target_class)
        tracking = self.tracker.track(detections, frame.shape)
        LOGGER.info(
            "[TRACKING] target=%s visible=%s horizontal=%s h_error=%.3f size=%.3f",
            self.ability.target_class,
            tracking.visible,
            tracking.horizontal_position.value,
            tracking.horizontal_error,
            tracking.size_ratio,
        )
        decision = self.ability.decide(tracking)
        LOGGER.info(
            "[ABILITY] search=%s scanned=%.1f action=%s",
            decision.state.value,
            decision.scanned_degrees,
            decision.command.action.value,
        )
        result = self.safety.execute(decision.command)
        self.ability.record_execution(decision, result)
        return SearchTick(tuple(detections), tracking, decision, result)
