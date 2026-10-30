from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from drone_ai.commands import ActionCommand, ActionType, CommandSource
from drone_ai.controllers.simulated import SimulatedDrone
from drone_ai.safety import SafetyController, SafetyStatus


NOW = datetime(2026, 8, 29, 12, 0, tzinfo=timezone.utc)


def make_safety(*, takeoff: bool = False) -> tuple[SimulatedDrone, SafetyController]:
    drone = SimulatedDrone()
    drone.connect()
    if takeoff:
        drone.takeoff()
    return drone, SafetyController(drone, clock=lambda: NOW)


def action(
    action_type: ActionType,
    *,
    source: CommandSource = CommandSource.MANUAL,
    distance_m: float | None = None,
    degrees: float | None = None,
    timestamp: datetime = NOW,
) -> ActionCommand:
    return ActionCommand(
        action=action_type,
        source=source,
        distance_m=distance_m,
        degrees=degrees,
        timestamp=timestamp,
    )


@pytest.mark.parametrize("distance", [0.19, 3.01])
def test_rejects_movement_outside_limits(distance: float) -> None:
    _, safety = make_safety(takeoff=True)
    result = safety.execute(action(ActionType.MOVE_FORWARD, distance_m=distance))
    assert result.status is SafetyStatus.REJECTED


def test_rejects_excessive_rotation_and_landed_movement() -> None:
    _, landed_safety = make_safety()
    assert (
        landed_safety.execute(action(ActionType.MOVE_FORWARD, distance_m=0.2)).status
        is SafetyStatus.REJECTED
    )
    _, flying_safety = make_safety(takeoff=True)
    assert (
        flying_safety.execute(action(ActionType.ROTATE_RIGHT, degrees=180.01)).status
        is SafetyStatus.REJECTED
    )


def test_altitude_ceiling_and_floor() -> None:
    drone, safety = make_safety(takeoff=True)
    assert safety.execute(action(ActionType.MOVE_UP, distance_m=2.0)).approved
    assert drone.get_state().altitude_m == pytest.approx(3.0)
    assert safety.execute(action(ActionType.MOVE_UP, distance_m=0.2)).status is SafetyStatus.REJECTED
    assert safety.execute(action(ActionType.MOVE_DOWN, distance_m=3.0)).approved
    assert drone.get_state().altitude_m == pytest.approx(0.0)
    assert safety.execute(action(ActionType.MOVE_DOWN, distance_m=0.2)).status is SafetyStatus.REJECTED


def test_rejects_stale_command() -> None:
    _, safety = make_safety(takeoff=True)
    old = action(
        ActionType.MOVE_FORWARD,
        distance_m=0.3,
        timestamp=NOW - timedelta(seconds=2.01),
    )
    assert safety.execute(old).status is SafetyStatus.REJECTED


def test_manual_override_blocks_autonomy_but_allows_manual_and_system() -> None:
    _, safety = make_safety(takeoff=True)
    safety.set_manual_override(True)
    for source in (CommandSource.AI, CommandSource.ABILITY):
        result = safety.execute(
            action(ActionType.MOVE_FORWARD, source=source, distance_m=0.3)
        )
        assert result.status is SafetyStatus.REJECTED
    assert safety.execute(action(ActionType.STOP, source=CommandSource.MANUAL)).approved
    assert safety.execute(action(ActionType.LAND, source=CommandSource.SYSTEM)).approved


def test_rejects_model_construct_bypass_as_malformed() -> None:
    _, safety = make_safety(takeoff=True)
    malformed = ActionCommand.model_construct(
        action=ActionType.MOVE_FORWARD,
        source=CommandSource.MANUAL,
        timestamp=NOW,
    )
    result = safety.execute(malformed)
    assert result.status is SafetyStatus.REJECTED
    assert "malformed" in result.reason


def test_emergency_bypasses_disconnected_stale_and_manual_override() -> None:
    drone = SimulatedDrone()
    safety = SafetyController(drone, clock=lambda: NOW)
    safety.set_manual_override(True)
    emergency = ActionCommand(
        action=ActionType.EMERGENCY_STOP,
        source=CommandSource.AI,
        timestamp=NOW - timedelta(days=1),
    )
    result = safety.execute(emergency)
    assert result.status is SafetyStatus.APPROVED
    assert not drone.get_state().is_flying


def test_stop_maps_to_hover_and_high_level_actions_are_not_dispatched() -> None:
    _, safety = make_safety(takeoff=True)
    assert safety.execute(action(ActionType.STOP)).approved
    result = safety.execute(action(ActionType.FOLLOW_PERSON))
    assert result.status is SafetyStatus.REJECTED
    assert "ability" in result.reason


def test_takeoff_requires_minimum_battery() -> None:
    class LowBatteryDrone(SimulatedDrone):
        def get_state(self):  # type: ignore[no-untyped-def]
            from dataclasses import replace

            return replace(super().get_state(), battery_percent=29)

    drone = LowBatteryDrone()
    drone.connect()
    safety = SafetyController(drone, clock=lambda: NOW)
    result = safety.execute(action(ActionType.TAKEOFF))
    assert result.status is SafetyStatus.REJECTED
    assert "30%" in result.reason
