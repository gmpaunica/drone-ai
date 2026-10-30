"""Runtime supervision for conservative autonomous and manual flight."""

from __future__ import annotations

from enum import Enum
import math
import time
from collections.abc import Callable

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from drone_ai.commands import ActionCommand, ActionType, CommandSource
from drone_ai.config import FlightConfig, WatchdogConfig
from drone_ai.controllers.base import DroneState
from drone_ai.safety import SafetyController, SafetyResult


class WatchdogReason(str, Enum):
    CRITICAL_BATTERY = "critical_battery"
    MAX_AIRBORNE_TIME = "max_airborne_time"
    TELEMETRY_LOSS = "telemetry_loss"


class WatchdogEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    reason: WatchdogReason
    detail: str
    battery_percent: int | None = None
    airborne_time_s: float | None = Field(default=None, ge=0.0)
    safety_result: SafetyResult


class TelemetryState(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    x_m: float | None
    y_m: float | None
    altitude_m: float | None
    yaw_degrees: float = Field(ge=0.0, lt=360.0)
    is_flying: bool
    is_connected: bool
    battery_percent: int | None = Field(default=None, ge=0, le=100)
    speed_mps: float = Field(gt=0.0)

    @field_validator("x_m", "y_m", "altitude_m", "yaw_degrees", "speed_mps")
    @classmethod
    def finite_telemetry(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("telemetry values must be finite")
        return value

    @classmethod
    def from_drone_state(cls, state: DroneState) -> "TelemetryState":
        return cls(
            x_m=state.x_m,
            y_m=state.y_m,
            altitude_m=state.altitude_m,
            yaw_degrees=state.yaw_degrees % 360.0,
            is_flying=state.is_flying,
            is_connected=state.is_connected,
            battery_percent=state.battery_percent,
            speed_mps=state.speed_mps,
        )


class WatchdogTelemetrySnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    sequence: int = Field(ge=0)
    checked_at_monotonic_s: float = Field(ge=0.0)
    state: TelemetryState | None = None
    error: str | None = None

    @field_validator("checked_at_monotonic_s")
    @classmethod
    def finite_check_time(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("watchdog check time must be finite")
        return value

    @model_validator(mode="after")
    def state_or_error(self) -> "WatchdogTelemetrySnapshot":
        if (self.state is None) == (self.error is None):
            raise ValueError("telemetry snapshot must contain exactly one of state or error")
        if self.error is not None and not self.error.strip():
            raise ValueError("telemetry error cannot be blank")
        return self


class FlightWatchdog:
    def __init__(
        self,
        safety: SafetyController,
        config: WatchdogConfig | None = None,
        flight_config: FlightConfig | None = None,
        *,
        monotonic_clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.safety = safety
        self.config = config or WatchdogConfig()
        self.flight_config = flight_config or safety.config
        if self.config.max_airborne_time_s <= 0:
            raise ValueError("watchdog maximum airborne time must be positive")
        if self.config.check_rate_hz <= 0:
            raise ValueError("watchdog check rate must be positive")
        if self.config.max_consecutive_telemetry_failures <= 0:
            raise ValueError("watchdog telemetry failure count must be positive")
        self._clock = monotonic_clock
        self._last_check_at: float | None = None
        self._airborne_since: float | None = None
        self._telemetry_failures = 0
        self._snapshot_sequence = 0
        self._last_snapshot: WatchdogTelemetrySnapshot | None = None

    @property
    def last_snapshot(self) -> WatchdogTelemetrySnapshot | None:
        return self._last_snapshot

    def poll(self, *, force: bool = False) -> WatchdogEvent | None:
        now = float(self._clock())
        interval = 1.0 / self.config.check_rate_hz
        if (
            not force
            and self._last_check_at is not None
            and now - self._last_check_at < interval
        ):
            return None
        self._last_check_at = now

        try:
            state = self.safety.drone.get_state()
        except Exception as exc:
            detail = f"state read failed: {exc}"
            self._capture_snapshot(now, error=detail)
            return self._record_telemetry_failure(now, detail)
        try:
            telemetry_state = TelemetryState.from_drone_state(state)
        except Exception as exc:
            detail = f"state validation failed: {exc}"
            self._capture_snapshot(now, error=detail)
            return self._record_telemetry_failure(now, detail)
        self._capture_snapshot(now, state=telemetry_state)

        if not state.is_connected:
            if self._airborne_since is not None:
                return self._record_telemetry_failure(
                    now,
                    "controller reported disconnected while flight was active",
                )
            self._telemetry_failures = 0
            return None

        if not state.is_flying:
            self._airborne_since = None
            self._telemetry_failures = 0
            return None

        if self._airborne_since is None:
            self._airborne_since = now
        airborne_time = max(0.0, now - self._airborne_since)

        if state.battery_percent is None:
            return self._record_telemetry_failure(now, "battery telemetry is unavailable")
        self._telemetry_failures = 0
        if state.battery_percent <= self.flight_config.critical_battery_percent:
            detail = (
                f"battery reached {state.battery_percent}% "
                f"(critical threshold {self.flight_config.critical_battery_percent}%)"
            )
            result = self.safety.execute(
                ActionCommand(action=ActionType.LAND, source=CommandSource.SYSTEM)
            )
            return WatchdogEvent(
                reason=WatchdogReason.CRITICAL_BATTERY,
                detail=detail,
                battery_percent=state.battery_percent,
                airborne_time_s=airborne_time,
                safety_result=result,
            )

        if airborne_time >= self.config.max_airborne_time_s:
            detail = (
                f"airborne for {airborne_time:.1f}s "
                f"(limit {self.config.max_airborne_time_s:.1f}s)"
            )
            result = self.safety.execute(
                ActionCommand(action=ActionType.LAND, source=CommandSource.SYSTEM)
            )
            return WatchdogEvent(
                reason=WatchdogReason.MAX_AIRBORNE_TIME,
                detail=detail,
                battery_percent=state.battery_percent,
                airborne_time_s=airborne_time,
                safety_result=result,
            )
        return None

    def _capture_snapshot(
        self,
        now: float,
        *,
        state: TelemetryState | None = None,
        error: str | None = None,
    ) -> None:
        self._last_snapshot = WatchdogTelemetrySnapshot(
            sequence=self._snapshot_sequence,
            checked_at_monotonic_s=now,
            state=state,
            error=error,
        )
        self._snapshot_sequence += 1

    def _record_telemetry_failure(self, now: float, detail: str) -> WatchdogEvent | None:
        self._telemetry_failures += 1
        if self._telemetry_failures < self.config.max_consecutive_telemetry_failures:
            return None
        airborne_time = (
            max(0.0, now - self._airborne_since)
            if self._airborne_since is not None
            else None
        )
        result = self.safety.execute_failsafe_land(detail)
        return WatchdogEvent(
            reason=WatchdogReason.TELEMETRY_LOSS,
            detail=(
                f"{detail}; {self._telemetry_failures} consecutive watchdog checks failed"
            ),
            battery_percent=None,
            airborne_time_s=airborne_time,
            safety_result=result,
        )
