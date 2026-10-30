"""Pure follow-person decision logic."""

from __future__ import annotations

from datetime import datetime, timezone

from drone_ai.commands import ActionCommand, ActionType, CommandSource
from drone_ai.config import FollowConfig
from drone_ai.vision.models import TargetDistance, TrackingResult


class FollowPersonAbility:
    def __init__(self, config: FollowConfig | None = None) -> None:
        self.config = config or FollowConfig()

    def decide(
        self,
        tracking: TrackingResult,
        *,
        timestamp: datetime | None = None,
    ) -> ActionCommand:
        issued_at = timestamp or datetime.now(timezone.utc)
        if not tracking.visible:
            return self._stop(issued_at)
        if tracking.horizontal_error < -self.config.horizontal_dead_zone:
            return ActionCommand(
                action=ActionType.ROTATE_LEFT,
                source=CommandSource.ABILITY,
                degrees=self.config.rotation_degrees,
                timestamp=issued_at,
            )
        if tracking.horizontal_error > self.config.horizontal_dead_zone:
            return ActionCommand(
                action=ActionType.ROTATE_RIGHT,
                source=CommandSource.ABILITY,
                degrees=self.config.rotation_degrees,
                timestamp=issued_at,
            )
        if tracking.distance is TargetDistance.FAR:
            return ActionCommand(
                action=ActionType.MOVE_FORWARD,
                source=CommandSource.ABILITY,
                distance_m=self.config.movement_step_m,
                timestamp=issued_at,
            )
        if tracking.distance is TargetDistance.CLOSE:
            return ActionCommand(
                action=ActionType.MOVE_BACKWARD,
                source=CommandSource.ABILITY,
                distance_m=self.config.movement_step_m,
                timestamp=issued_at,
            )
        return self._stop(issued_at)

    @staticmethod
    def _stop(timestamp: datetime) -> ActionCommand:
        return ActionCommand(
            action=ActionType.STOP,
            source=CommandSource.ABILITY,
            timestamp=timestamp,
        )
