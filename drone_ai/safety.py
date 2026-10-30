"""Central command validation and flight dispatch."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from enum import Enum
import logging
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError

from drone_ai.commands import (
    MOVEMENT_ACTIONS,
    ROTATION_ACTIONS,
    ActionCommand,
    ActionType,
    CommandSource,
)
from drone_ai.config import FlightConfig
from drone_ai.controllers.base import DroneController


LOGGER = logging.getLogger(__name__)


class SafetyStatus(str, Enum):
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    FAILED = "FAILED"


class SafetyResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    status: SafetyStatus
    reason: str

    @property
    def approved(self) -> bool:
        return self.status is SafetyStatus.APPROVED


class SafetyController:
    """The sole gateway from commands to a :class:`DroneController`."""

    def __init__(
        self,
        drone: DroneController,
        config: FlightConfig | None = None,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.drone = drone
        self.config = config or FlightConfig()
        if not (
            0
            <= self.config.critical_battery_percent
            < self.config.min_takeoff_battery_percent
            <= 100
        ):
            raise ValueError(
                "battery thresholds must satisfy 0 <= critical < minimum takeoff <= 100"
            )
        self.manual_override = False
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    def set_manual_override(self, enabled: bool) -> None:
        self.manual_override = enabled
        LOGGER.info("[SAFETY] Manual override %s", "enabled" if enabled else "disabled")

    def execute(self, command: ActionCommand | Any) -> SafetyResult:
        # Emergency commands already crossed the strict schema boundary and must
        # not wait behind state, timeout, override, or future queue checks.
        if isinstance(command, ActionCommand) and command.action is ActionType.EMERGENCY_STOP:
            return self.execute_emergency()

        validated, error = self._revalidate(command)
        if error is not None:
            return self._result(SafetyStatus.REJECTED, f"malformed command: {error}")
        assert validated is not None

        now = self._clock()
        if now.tzinfo is None or now.utcoffset() is None:
            now = now.replace(tzinfo=timezone.utc)
        age_s = (now - validated.timestamp).total_seconds()
        if age_s > self.config.command_timeout_s:
            return self._result(
                SafetyStatus.REJECTED,
                f"command is stale ({age_s:.2f}s > {self.config.command_timeout_s:.2f}s)",
            )

        if self.manual_override and validated.source in {CommandSource.AI, CommandSource.ABILITY}:
            return self._result(SafetyStatus.REJECTED, "manual override blocks autonomous commands")

        try:
            state = self.drone.get_state()
        except Exception as exc:  # adapter boundary
            return self._result(SafetyStatus.FAILED, f"could not read drone state: {exc}")

        if not state.is_connected:
            return self._result(SafetyStatus.REJECTED, "drone is disconnected")

        action = validated.action
        if state.is_flying and action not in {
            ActionType.LAND,
            ActionType.STOP,
            ActionType.HOVER,
        }:
            if state.battery_percent is None:
                return self._result(
                    SafetyStatus.REJECTED,
                    "flight action rejected because battery telemetry is unknown",
                )
            if state.battery_percent <= self.config.critical_battery_percent:
                return self._result(
                    SafetyStatus.REJECTED,
                    f"flight action rejected at critical battery {state.battery_percent}%",
                )
        if action in MOVEMENT_ACTIONS:
            distance = validated.distance_m
            assert distance is not None
            if not self.config.min_movement_m <= distance <= self.config.max_movement_m:
                return self._result(
                    SafetyStatus.REJECTED,
                    f"distance {distance:g}m is outside "
                    f"{self.config.min_movement_m:g}-{self.config.max_movement_m:g}m",
                )
            if not state.is_flying:
                return self._result(SafetyStatus.REJECTED, "movement rejected while landed")
            if action in {ActionType.MOVE_UP, ActionType.MOVE_DOWN}:
                if state.altitude_m is None:
                    return self._result(SafetyStatus.REJECTED, "altitude is unknown")
                resulting_altitude = state.altitude_m + (
                    distance if action is ActionType.MOVE_UP else -distance
                )
                if not self.config.min_altitude_m <= resulting_altitude <= self.config.max_altitude_m:
                    return self._result(
                        SafetyStatus.REJECTED,
                        f"resulting altitude {resulting_altitude:g}m violates "
                        f"{self.config.min_altitude_m:g}-{self.config.max_altitude_m:g}m limits",
                    )
        elif action in ROTATION_ACTIONS:
            degrees = validated.degrees
            assert degrees is not None
            if degrees > self.config.max_rotation_degrees:
                return self._result(
                    SafetyStatus.REJECTED,
                    f"rotation {degrees:g} exceeds {self.config.max_rotation_degrees:g} degrees",
                )
            if not state.is_flying:
                return self._result(SafetyStatus.REJECTED, "rotation rejected while landed")
        elif action in {ActionType.STOP, ActionType.HOVER}:
            if not state.is_flying:
                return self._result(SafetyStatus.REJECTED, "hover rejected while landed")
        elif action is ActionType.TAKEOFF:
            if state.is_flying:
                return self._result(SafetyStatus.REJECTED, "takeoff rejected while already flying")
            if state.battery_percent is None:
                return self._result(
                    SafetyStatus.REJECTED,
                    "takeoff rejected because battery telemetry is unknown",
                )
            if state.battery_percent < self.config.min_takeoff_battery_percent:
                return self._result(
                    SafetyStatus.REJECTED,
                    f"takeoff rejected at {state.battery_percent}% battery; "
                    f"minimum is {self.config.min_takeoff_battery_percent}%",
                )
            if not (
                self.config.min_altitude_m
                <= self.config.takeoff_altitude_m
                <= self.config.max_altitude_m
            ):
                return self._result(SafetyStatus.REJECTED, "configured takeoff altitude is unsafe")
        elif action is ActionType.LAND:
            if not state.is_flying:
                return self._result(SafetyStatus.REJECTED, "landing rejected while already landed")
        else:
            return self._result(
                SafetyStatus.REJECTED,
                f"high-level action {action.value} must be resolved by an ability",
            )

        LOGGER.info("[SAFETY] APPROVED for dispatch: %s", action.value)
        try:
            self._dispatch(validated)
        except Exception as exc:  # hardware/simulator errors become explicit failures
            return self._result(SafetyStatus.FAILED, f"controller failed: {exc}")
        return self._result(SafetyStatus.APPROVED, f"executed {action.value}")

    def execute_emergency(self) -> SafetyResult:
        """Immediately invoke the adapter's emergency path without normal gates."""

        try:
            self.drone.emergency_stop()
        except Exception as exc:
            return self._result(SafetyStatus.FAILED, f"emergency stop failed: {exc}")
        return self._result(SafetyStatus.APPROVED, "emergency stop executed immediately")

    def execute_failsafe_land(self, reason: str) -> SafetyResult:
        """Attempt landing without depending on telemetry validation.

        This is reserved for a runtime watchdog that has already observed
        repeated telemetry loss. It never maps to emergency motor cutoff.
        """

        normalized = reason.strip()
        if not normalized:
            return self._result(SafetyStatus.REJECTED, "failsafe landing reason is required")
        LOGGER.warning("[SAFETY] FAILSAFE LAND: %s", normalized)
        try:
            self.drone.land()
        except Exception as exc:
            return self._result(SafetyStatus.FAILED, f"failsafe landing failed: {exc}")
        return self._result(SafetyStatus.APPROVED, f"failsafe landing executed: {normalized}")

    @staticmethod
    def _revalidate(command: object) -> tuple[ActionCommand | None, str | None]:
        if not isinstance(command, ActionCommand):
            return None, "expected ActionCommand"
        try:
            # model_construct() can bypass Pydantic. Re-enter the strict boundary.
            data = command.model_dump()
            return ActionCommand.model_validate(data), None
        except (ValidationError, ValueError, TypeError) as exc:
            return None, str(exc)

    def _dispatch(self, command: ActionCommand) -> None:
        action = command.action
        distance = command.distance_m
        degrees = command.degrees
        dispatch: dict[ActionType, Callable[[], None]] = {
            ActionType.TAKEOFF: self.drone.takeoff,
            ActionType.LAND: self.drone.land,
            ActionType.HOVER: self.drone.hover,
            ActionType.STOP: self.drone.hover,
            ActionType.MOVE_FORWARD: lambda: self.drone.move_forward(self._present(distance)),
            ActionType.MOVE_BACKWARD: lambda: self.drone.move_backward(self._present(distance)),
            ActionType.MOVE_LEFT: lambda: self.drone.move_left(self._present(distance)),
            ActionType.MOVE_RIGHT: lambda: self.drone.move_right(self._present(distance)),
            ActionType.MOVE_UP: lambda: self.drone.move_up(self._present(distance)),
            ActionType.MOVE_DOWN: lambda: self.drone.move_down(self._present(distance)),
            ActionType.ROTATE_LEFT: lambda: self.drone.rotate_counter_clockwise(
                self._present(degrees)
            ),
            ActionType.ROTATE_RIGHT: lambda: self.drone.rotate_clockwise(self._present(degrees)),
        }
        dispatch[action]()

    @staticmethod
    def _present(value: float | None) -> float:
        if value is None:  # unreachable after schema validation, helps type checkers
            raise ValueError("required command parameter is absent")
        return value

    @staticmethod
    def _result(status: SafetyStatus, reason: str) -> SafetyResult:
        log = LOGGER.info if status is SafetyStatus.APPROVED else LOGGER.warning
        log("[SAFETY] %s: %s", status.value, reason)
        return SafetyResult(status=status, reason=reason)
