"""Pure manual key mapping and safety-gated dispatch."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
import time

from drone_ai.commands import (
    MOVEMENT_ACTIONS,
    ROTATION_ACTIONS,
    ActionCommand,
    ActionType,
    CommandSource,
)
from drone_ai.config import ManualConfig
from drone_ai.safety import SafetyController, SafetyResult


@dataclass(frozen=True, slots=True)
class ManualEvent:
    key: str
    command: ActionCommand | None
    safety_result: SafetyResult | None
    suppressed: bool = False
    reason: str = ""


class ManualCommandMapper:
    """Convert conservative, discrete keyboard inputs into typed commands."""

    def __init__(self, config: ManualConfig | None = None) -> None:
        self.config = config or ManualConfig()
        if self.config.horizontal_step_m <= 0 or self.config.vertical_step_m <= 0:
            raise ValueError("manual movement steps must be positive")
        if self.config.rotation_step_degrees <= 0:
            raise ValueError("manual rotation step must be positive")

    def map_key(
        self,
        key: str,
        *,
        timestamp: datetime | None = None,
    ) -> ActionCommand | None:
        normalized = key.lower()
        action = {
            "t": ActionType.TAKEOFF,
            "l": ActionType.LAND,
            "h": ActionType.STOP,
            "e": ActionType.EMERGENCY_STOP,
            "w": ActionType.MOVE_FORWARD,
            "s": ActionType.MOVE_BACKWARD,
            "a": ActionType.ROTATE_LEFT,
            "d": ActionType.ROTATE_RIGHT,
            " ": ActionType.MOVE_UP,
            "ctrl": ActionType.MOVE_DOWN,
            "r": ActionType.MOVE_UP,
            "f": ActionType.MOVE_DOWN,
            "j": ActionType.MOVE_LEFT,
            "k": ActionType.MOVE_RIGHT,
        }.get(normalized)
        if action is None:
            return None
        values: dict[str, object] = {
            "action": action,
            "source": CommandSource.MANUAL,
            "timestamp": timestamp or datetime.now(timezone.utc),
        }
        if action in MOVEMENT_ACTIONS:
            values["distance_m"] = (
                self.config.vertical_step_m
                if action in {ActionType.MOVE_UP, ActionType.MOVE_DOWN}
                else self.config.horizontal_step_m
            )
        elif action in ROTATION_ACTIONS:
            values["degrees"] = self.config.rotation_step_degrees
        return ActionCommand.model_validate(values)


class ManualController:
    """Rate-limit manual motion while preserving immediate emergency dispatch."""

    def __init__(
        self,
        safety: SafetyController,
        config: ManualConfig | None = None,
        *,
        monotonic_clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.safety = safety
        self.config = config or ManualConfig()
        if self.config.max_command_rate_hz <= 0:
            raise ValueError("manual command rate must be positive")
        self.mapper = ManualCommandMapper(self.config)
        self._clock = monotonic_clock
        self._last_motion_at: float | None = None
        self._last_hover_at: float | None = None
        self._last_lifecycle_at: dict[ActionType, float] = {}

    def handle_key(self, key: str) -> ManualEvent:
        command = self.mapper.map_key(key)
        if command is None:
            return ManualEvent(key=key, command=None, safety_result=None, reason="unmapped key")
        if command.action is ActionType.EMERGENCY_STOP:
            result = self.safety.execute_emergency()
            return ManualEvent(key=key, command=command, safety_result=result)

        if command.action in {ActionType.TAKEOFF, ActionType.LAND}:
            now = float(self._clock())
            previous = self._last_lifecycle_at.get(command.action)
            if previous is not None and now - previous < 1.0:
                return ManualEvent(
                    key=key,
                    command=command,
                    safety_result=None,
                    suppressed=True,
                    reason="repeated lifecycle key suppressed",
                )
            result = self.safety.execute(command)
            # Start the debounce after the blocking SDK command returns, so
            # queued Windows key-repeat events are drained without dispatch.
            self._last_lifecycle_at[command.action] = float(self._clock())
            return ManualEvent(key=key, command=command, safety_result=result)

        if command.action in {ActionType.STOP, ActionType.HOVER}:
            now = float(self._clock())
            minimum_interval = 1.0 / self.config.max_command_rate_hz
            if self._last_hover_at is not None and now - self._last_hover_at < minimum_interval:
                return ManualEvent(
                    key=key,
                    command=command,
                    safety_result=None,
                    suppressed=True,
                    reason="repeated hover key suppressed",
                )
            self._last_hover_at = now
            result = self.safety.execute(command)
            return ManualEvent(key=key, command=command, safety_result=result)

        if command.action in MOVEMENT_ACTIONS or command.action in ROTATION_ACTIONS:
            now = float(self._clock())
            minimum_interval = 1.0 / self.config.max_command_rate_hz
            if self._last_motion_at is not None and now - self._last_motion_at < minimum_interval:
                return ManualEvent(
                    key=key,
                    command=command,
                    safety_result=None,
                    suppressed=True,
                    reason=f"manual motion rate limited to {self.config.max_command_rate_hz:g} Hz",
                )
            self._last_motion_at = now

        result = self.safety.execute(command)
        return ManualEvent(key=key, command=command, safety_result=result)
