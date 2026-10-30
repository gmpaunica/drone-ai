"""Bounded visual search by rotating through at most one full turn."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum

from drone_ai.commands import ActionCommand, ActionType, CommandSource
from drone_ai.config import SearchConfig
from drone_ai.safety import SafetyResult
from drone_ai.vision.models import TrackingResult


class SearchState(str, Enum):
    SEARCHING = "searching"
    FOUND = "found"
    NOT_FOUND = "not_found"


@dataclass(frozen=True, slots=True)
class SearchDecision:
    state: SearchState
    command: ActionCommand
    scanned_degrees: float

    @property
    def terminal(self) -> bool:
        return self.state in {SearchState.FOUND, SearchState.NOT_FOUND}


class VisualSearchAbility:
    def __init__(self, target_class: str, config: SearchConfig | None = None) -> None:
        normalized = target_class.strip().lower()
        if not normalized:
            raise ValueError("target_class cannot be blank")
        self.target_class = normalized
        self.config = config or SearchConfig()
        if self.config.rotation_step_degrees <= 0:
            raise ValueError("search rotation step must be positive")
        if not 0 < self.config.max_scan_degrees <= 360:
            raise ValueError("maximum scan must be in (0, 360]")
        self._scanned_degrees = 0.0

    @property
    def scanned_degrees(self) -> float:
        return self._scanned_degrees

    def decide(
        self,
        tracking: TrackingResult,
        *,
        timestamp: datetime | None = None,
    ) -> SearchDecision:
        issued_at = timestamp or datetime.now(timezone.utc)
        if tracking.visible:
            return SearchDecision(
                state=SearchState.FOUND,
                command=self._stop(issued_at),
                scanned_degrees=self._scanned_degrees,
            )
        remaining = self.config.max_scan_degrees - self._scanned_degrees
        if remaining <= 1e-9:
            return SearchDecision(
                state=SearchState.NOT_FOUND,
                command=self._stop(issued_at),
                scanned_degrees=self._scanned_degrees,
            )
        degrees = min(self.config.rotation_step_degrees, remaining)
        return SearchDecision(
            state=SearchState.SEARCHING,
            command=ActionCommand(
                action=ActionType.ROTATE_RIGHT,
                source=CommandSource.ABILITY,
                degrees=degrees,
                timestamp=issued_at,
            ),
            scanned_degrees=self._scanned_degrees,
        )

    def record_execution(self, decision: SearchDecision, result: SafetyResult) -> None:
        if not result.approved or decision.command.action is not ActionType.ROTATE_RIGHT:
            return
        assert decision.command.degrees is not None
        self._scanned_degrees = min(
            self.config.max_scan_degrees,
            self._scanned_degrees + decision.command.degrees,
        )

    @staticmethod
    def _stop(timestamp: datetime) -> ActionCommand:
        return ActionCommand(
            action=ActionType.STOP,
            source=CommandSource.ABILITY,
            timestamp=timestamp,
        )
