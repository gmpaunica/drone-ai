from __future__ import annotations

import argparse
from datetime import datetime, timezone
from typing import Any

import pytest

import drone_ai.cli as cli
from drone_ai.abilities.visual_search import SearchState, VisualSearchAbility
from drone_ai.commands import ActionType, CommandSource
from drone_ai.config import SearchConfig
from drone_ai.controllers.simulated import SimulatedDrone
from drone_ai.safety import SafetyController, SafetyResult, SafetyStatus
from drone_ai.search_pipeline import SearchPipeline
from drone_ai.vision.models import Detection, HorizontalPosition
from drone_ai.vision.tracker import TargetTracker


NOW = datetime(2026, 8, 29, 12, 0, tzinfo=timezone.utc)


class FakeFrame:
    shape = (100, 100, 3)


def missing_tracking(target: str = "backpack") -> Any:
    return TargetTracker(target).track([], FakeFrame.shape)


def visible_tracking(target: str = "backpack") -> Any:
    return TargetTracker(target).track(
        [Detection(target, 0.9, (40, 30, 60, 70))],
        FakeFrame.shape,
    )


def safety_result(status: SafetyStatus) -> SafetyResult:
    return SafetyResult(status=status, reason=status.value.lower())


def test_target_tracker_filters_requested_class_and_selects_largest() -> None:
    detections = [
        Detection("person", 0.99, (0, 0, 100, 100)),
        Detection("backpack", 0.95, (0, 0, 10, 10)),
        Detection("backpack", 0.80, (30, 30, 70, 70)),
    ]
    result = TargetTracker("BACKPACK").track(detections, FakeFrame.shape)
    assert result.visible
    assert result.detection == detections[2]
    assert result.horizontal_position is HorizontalPosition.CENTER


def test_search_rotates_right_with_ability_source_when_target_missing() -> None:
    ability = VisualSearchAbility("backpack")
    decision = ability.decide(missing_tracking(), timestamp=NOW)
    assert decision.state is SearchState.SEARCHING
    assert decision.command.action is ActionType.ROTATE_RIGHT
    assert decision.command.degrees == 20.0
    assert decision.command.source is CommandSource.ABILITY
    assert decision.scanned_degrees == 0.0


def test_search_stops_immediately_when_target_visible() -> None:
    ability = VisualSearchAbility("backpack")
    decision = ability.decide(visible_tracking(), timestamp=NOW)
    assert decision.state is SearchState.FOUND
    assert decision.terminal
    assert decision.command.action is ActionType.STOP


def test_only_approved_rotation_advances_scan() -> None:
    ability = VisualSearchAbility("backpack")
    first = ability.decide(missing_tracking(), timestamp=NOW)
    ability.record_execution(first, safety_result(SafetyStatus.REJECTED))
    assert ability.scanned_degrees == 0.0
    ability.record_execution(first, safety_result(SafetyStatus.APPROVED))
    assert ability.scanned_degrees == 20.0


def test_search_stops_not_found_after_one_full_approved_turn() -> None:
    ability = VisualSearchAbility("backpack")
    for _ in range(18):
        decision = ability.decide(missing_tracking(), timestamp=NOW)
        assert decision.state is SearchState.SEARCHING
        ability.record_execution(decision, safety_result(SafetyStatus.APPROVED))
    assert ability.scanned_degrees == 360.0
    final = ability.decide(missing_tracking(), timestamp=NOW)
    assert final.state is SearchState.NOT_FOUND
    assert final.command.action is ActionType.STOP


def test_last_scan_step_is_clamped_to_remaining_angle() -> None:
    ability = VisualSearchAbility(
        "person",
        SearchConfig(rotation_step_degrees=100, max_scan_degrees=360),
    )
    angles: list[float] = []
    for _ in range(4):
        decision = ability.decide(missing_tracking("person"), timestamp=NOW)
        assert decision.command.degrees is not None
        angles.append(decision.command.degrees)
        ability.record_execution(decision, safety_result(SafetyStatus.APPROVED))
    assert angles == [100, 100, 100, 60]
    assert ability.scanned_degrees == 360


@pytest.mark.parametrize(
    "config",
    [
        SearchConfig(rotation_step_degrees=0),
        SearchConfig(max_scan_degrees=0),
        SearchConfig(max_scan_degrees=361),
    ],
)
def test_invalid_search_configuration_is_rejected(config: SearchConfig) -> None:
    with pytest.raises(ValueError):
        VisualSearchAbility("person", config)


class SequenceDetector:
    def __init__(self, outputs: list[list[Detection]]) -> None:
        self.outputs = outputs

    def detect(self, frame: Any) -> list[Detection]:
        return self.outputs.pop(0)


def test_search_pipeline_rotates_then_finds_through_safety() -> None:
    detector = SequenceDetector(
        [
            [],
            [Detection("backpack", 0.9, (40, 30, 60, 70))],
        ]
    )
    drone = SimulatedDrone()
    drone.connect()
    drone.takeoff()
    safety = SafetyController(drone)
    ability = VisualSearchAbility("backpack")
    pipeline = SearchPipeline(detector, TargetTracker("backpack"), ability, safety)

    first = pipeline.tick(FakeFrame())
    second = pipeline.tick(FakeFrame())

    assert first.decision.state is SearchState.SEARCHING
    assert first.safety_result.approved
    assert drone.get_state().yaw_degrees == 20.0
    assert second.decision.state is SearchState.FOUND
    assert second.safety_result.approved
    assert ability.scanned_degrees == 20.0


def test_search_pipeline_cannot_rotate_while_landed() -> None:
    drone = SimulatedDrone()
    drone.connect()
    safety = SafetyController(drone)
    ability = VisualSearchAbility("person")
    pipeline = SearchPipeline(
        SequenceDetector([[]]),
        TargetTracker("person"),
        ability,
        safety,
    )
    tick = pipeline.tick(FakeFrame())
    assert tick.safety_result.status is SafetyStatus.REJECTED
    assert ability.scanned_degrees == 0.0
    assert drone.get_state().yaw_degrees == 0.0


def command_args(text: str) -> argparse.Namespace:
    return argparse.Namespace(
        text=text,
        interpreter="rule",
        drone_mode="simulator",
        takeoff=False,
        ollama_model=None,
        ollama_url=None,
        ollama_timeout=None,
        camera=None,
        camera_index=0,
        model=None,
    )


def test_high_level_find_commands_route_to_search(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str]] = []

    def fake_search(args: argparse.Namespace, target: str) -> int:
        calls.append((args.text, target))
        return 7

    monkeypatch.setattr(cli, "run_search", fake_search)
    assert cli.run_command(command_args("find person")) == 7
    assert cli.run_command(command_args("look for a backpack")) == 7
    assert calls == [("find person", "person"), ("look for a backpack", "backpack")]


def test_high_level_follow_command_routes_to_follow(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_follow(args: argparse.Namespace) -> int:
        calls.append(args.text)
        return 8

    monkeypatch.setattr(cli, "run_follow", fake_follow)
    assert cli.run_command(command_args("follow me")) == 8
    assert calls == ["follow me"]
