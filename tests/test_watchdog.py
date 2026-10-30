from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

import drone_ai.cli as cli
from drone_ai.commands import ActionCommand, ActionType, CommandSource
from drone_ai.config import WatchdogConfig
from drone_ai.controllers.simulated import SimulatedDrone
from drone_ai.manual import ManualController
from drone_ai.safety import SafetyController, SafetyStatus
from drone_ai.watchdog import FlightWatchdog, WatchdogReason


class BatteryDrone(SimulatedDrone):
    def __init__(self) -> None:
        super().__init__()
        self.battery = 100
        self.fail_telemetry = False

    def get_state(self):  # type: ignore[no-untyped-def]
        if self.fail_telemetry:
            raise RuntimeError("telemetry link down")
        return replace(super().get_state(), battery_percent=self.battery)


def airborne() -> tuple[BatteryDrone, SafetyController]:
    drone = BatteryDrone()
    drone.connect()
    drone.takeoff()
    return drone, SafetyController(drone)


def test_watchdog_lands_at_critical_battery() -> None:
    drone, safety = airborne()
    drone.battery = 15
    event = FlightWatchdog(safety, monotonic_clock=lambda: 0.0).poll(force=True)
    assert event is not None
    assert event.reason is WatchdogReason.CRITICAL_BATTERY
    assert event.safety_result.status is SafetyStatus.APPROVED
    assert not drone.get_state().is_flying


def test_safety_blocks_motion_but_allows_landing_at_critical_battery() -> None:
    drone, safety = airborne()
    drone.battery = 15
    movement = safety.execute(
        ActionCommand(
            action=ActionType.MOVE_FORWARD,
            source=CommandSource.MANUAL,
            distance_m=0.2,
        )
    )
    assert movement.status is SafetyStatus.REJECTED
    landing = safety.execute(ActionCommand(action=ActionType.LAND, source=CommandSource.SYSTEM))
    assert landing.status is SafetyStatus.APPROVED


def test_watchdog_lands_at_maximum_airborne_time() -> None:
    moments = iter([0.0, 10.0])
    drone, safety = airborne()
    watchdog = FlightWatchdog(
        safety,
        WatchdogConfig(max_airborne_time_s=10.0),
        monotonic_clock=lambda: next(moments),
    )
    assert watchdog.poll(force=True) is None
    event = watchdog.poll(force=True)
    assert event is not None
    assert event.reason is WatchdogReason.MAX_AIRBORNE_TIME
    assert event.airborne_time_s == 10.0
    assert event.safety_result.approved
    assert not drone.get_state().is_flying


def test_watchdog_rate_limits_telemetry_checks() -> None:
    moments = iter([0.0, 0.1, 0.5])
    drone, safety = airborne()
    watchdog = FlightWatchdog(safety, monotonic_clock=lambda: next(moments))
    assert watchdog.poll() is None
    drone.battery = 15
    assert watchdog.poll() is None
    event = watchdog.poll()
    assert event is not None and event.reason is WatchdogReason.CRITICAL_BATTERY


def test_watchdog_caches_only_actual_checks_and_records_errors() -> None:
    moments = iter([0.0, 0.1, 0.5])
    drone, safety = airborne()
    watchdog = FlightWatchdog(safety, monotonic_clock=lambda: next(moments))
    assert watchdog.poll() is None
    first = watchdog.last_snapshot
    assert first is not None
    assert first.sequence == 0
    assert first.state is not None
    assert first.state.is_flying is True
    assert first.state.battery_percent == 100
    assert first.error is None

    assert watchdog.poll() is None
    assert watchdog.last_snapshot is first

    drone.fail_telemetry = True
    assert watchdog.poll() is None
    failed = watchdog.last_snapshot
    assert failed is not None
    assert failed.sequence == 1
    assert failed.state is None
    assert failed.error == "state read failed: telemetry link down"


def test_repeated_telemetry_loss_uses_dedicated_failsafe_landing() -> None:
    moments = iter([0.0, 1.0, 2.0, 3.0])
    drone, safety = airborne()
    watchdog = FlightWatchdog(
        safety,
        WatchdogConfig(max_consecutive_telemetry_failures=3),
        monotonic_clock=lambda: next(moments),
    )
    assert watchdog.poll(force=True) is None
    drone.fail_telemetry = True
    assert watchdog.poll(force=True) is None
    assert watchdog.poll(force=True) is None
    event = watchdog.poll(force=True)
    assert event is not None
    assert event.reason is WatchdogReason.TELEMETRY_LOSS
    assert event.safety_result.status is SafetyStatus.APPROVED
    drone.fail_telemetry = False
    assert not drone.get_state().is_flying


def test_blank_failsafe_reason_is_rejected() -> None:
    _, safety = airborne()
    assert safety.execute_failsafe_land(" ").status is SafetyStatus.REJECTED


def test_manual_loop_honors_watchdog_before_reading_another_frame() -> None:
    class NoReadCamera:
        def read(self) -> Any:
            pytest.fail("camera must not be read after watchdog landing")

    class SpyRecorder:
        def __init__(self) -> None:
            self.events: list[tuple[str, dict[str, Any]]] = []

        def record_event(self, kind: str, payload: dict[str, Any]) -> None:
            self.events.append((kind, payload))

        def record_frame(self, frame: Any) -> None:
            pytest.fail("frame must not be recorded after watchdog landing")

    drone, safety = airborne()
    drone.battery = 15
    watchdog = FlightWatchdog(safety, monotonic_clock=lambda: 0.0)
    recorder = SpyRecorder()
    result = cli._manual_loop(  # type: ignore[arg-type]
        NoReadCamera(),
        safety,
        ManualController(safety),
        recorder,
        watchdog,
    )
    assert result == 0
    assert not drone.get_state().is_flying
    assert recorder.events[0][0] == "watchdog.landing"
    assert recorder.events[0][1]["reason"] == "critical_battery"
    assert recorder.events[1][0] == "telemetry.snapshot"
    assert recorder.events[1][1]["state"]["battery_percent"] == 15
