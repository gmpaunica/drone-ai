from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from typing import Any

import pytest
from pydantic import ValidationError

from drone_ai.commands import ActionType, CommandSource
from drone_ai.config import LLMConfig, PlanConfig
from drone_ai.controllers.simulated import SimulatedDrone
from drone_ai.llm.ollama import OllamaInterpreterError
from drone_ai.planning.executor import PlanExecutor
from drone_ai.planning.models import (
    ActionPlan,
    PlanExecutionStatus,
    PlannedAction,
)
from drone_ai.planning.ollama import OllamaPlanInterpreter
from drone_ai.planning.parser import (
    HybridPlanInterpreter,
    PlanParseError,
    RuleBasedPlanInterpreter,
)
from drone_ai.safety import SafetyController, SafetyStatus


NOW = datetime(2026, 8, 29, 12, 0, tzinfo=timezone.utc)


def step(action: ActionType, **kwargs: object) -> PlannedAction:
    return PlannedAction(action=action, **kwargs)


def plan(*steps: PlannedAction, source: CommandSource = CommandSource.MANUAL) -> ActionPlan:
    return ActionPlan(original_text="test plan", source=source, steps=steps, created_at=NOW)


@pytest.mark.parametrize(
    "action",
    [
        ActionType.TAKEOFF,
        ActionType.EMERGENCY_STOP,
        ActionType.FOLLOW_PERSON,
        ActionType.FIND_PERSON,
        ActionType.FIND_OBJECT,
    ],
)
def test_forbidden_actions_cannot_be_queued(action: ActionType) -> None:
    with pytest.raises(ValidationError, match="cannot be queued"):
        step(action)


def test_planned_action_reuses_action_specific_validation() -> None:
    with pytest.raises(ValidationError, match="distance_m"):
        step(ActionType.MOVE_FORWARD)
    with pytest.raises(ValidationError, match="does not accept parameters"):
        step(ActionType.LAND, distance_m=1.0)


def test_land_must_be_unique_and_final() -> None:
    with pytest.raises(ValidationError, match="final plan step"):
        plan(step(ActionType.LAND), step(ActionType.STOP))
    with pytest.raises(ValidationError, match="final plan step"):
        plan(step(ActionType.LAND), step(ActionType.LAND))


def test_plan_schema_has_hard_eight_step_limit() -> None:
    with pytest.raises(ValidationError):
        plan(*(step(ActionType.STOP) for _ in range(9)))


def test_rule_planner_parses_ordered_separators_and_number_conjunction() -> None:
    result = RuleBasedPlanInterpreter().interpret_plan(
        "move forward one meter, turn right one hundred and eighty degrees; "
        "move left 0.5 meters and then land"
    )
    assert result.source is CommandSource.MANUAL
    assert [item.action for item in result.steps] == [
        ActionType.MOVE_FORWARD,
        ActionType.ROTATE_RIGHT,
        ActionType.MOVE_LEFT,
        ActionType.LAND,
    ]
    assert result.steps[1].degrees == 180.0


def test_rule_planner_supports_plain_and_between_actions() -> None:
    result = RuleBasedPlanInterpreter().interpret_plan(
        "move forward one meter and turn left ninety degrees"
    )
    assert len(result.steps) == 2


@pytest.mark.parametrize(
    "text",
    [
        "take off then move forward one meter",
        "move forward then land",
        "move forward one meter then dance",
        "follow me then land",
        "land then hover",
    ],
)
def test_rule_planner_rejects_unsafe_unknown_or_invalid_steps(text: str) -> None:
    with pytest.raises(PlanParseError):
        RuleBasedPlanInterpreter().interpret_plan(text)


class FakeLLMPlanner:
    def __init__(self, result: ActionPlan) -> None:
        self.result = result
        self.calls: list[str] = []

    def interpret_plan(self, text: str) -> ActionPlan:
        self.calls.append(text)
        return self.result


def test_hybrid_planner_uses_rules_before_llm() -> None:
    fallback = FakeLLMPlanner(plan(step(ActionType.STOP), source=CommandSource.AI))
    hybrid = HybridPlanInterpreter(fallback)
    result = hybrid.interpret_plan("move forward one meter then land")
    assert result.source is CommandSource.MANUAL
    assert fallback.calls == []


def test_hybrid_planner_falls_back_only_for_unknown_step() -> None:
    expected = plan(step(ActionType.STOP), source=CommandSource.AI)
    fallback = FakeLLMPlanner(expected)
    hybrid = HybridPlanInterpreter(fallback)
    assert hybrid.interpret_plan("advance carefully by one meter then land") is expected
    assert fallback.calls == ["advance carefully by one meter then land"]
    with pytest.raises(PlanParseError):
        hybrid.interpret_plan("move forward then land")
    assert len(fallback.calls) == 1


def flying_executor(
    *, config: PlanConfig | None = None
) -> tuple[SimulatedDrone, SafetyController, PlanExecutor]:
    drone = SimulatedDrone()
    drone.connect()
    drone.takeoff()
    safety = SafetyController(drone, clock=lambda: NOW)
    return drone, safety, PlanExecutor(safety, config, clock=lambda: NOW)


def test_executor_completes_steps_through_safety() -> None:
    drone, _, executor = flying_executor()
    result = executor.execute(
        plan(
            step(ActionType.MOVE_FORWARD, distance_m=1.0),
            step(ActionType.ROTATE_RIGHT, degrees=90.0),
            step(ActionType.MOVE_FORWARD, distance_m=0.5),
        )
    )
    assert result.status is PlanExecutionStatus.COMPLETED
    assert all(item.safety_result.status is SafetyStatus.APPROVED for item in result.step_results)
    assert drone.get_state().x_m == pytest.approx(0.5)
    assert drone.get_state().y_m == pytest.approx(1.0)


def test_executor_rejects_stale_plan_before_any_step() -> None:
    drone, _, executor = flying_executor()
    stale = ActionPlan(
        original_text="old",
        source=CommandSource.MANUAL,
        steps=(step(ActionType.MOVE_FORWARD, distance_m=1.0),),
        created_at=NOW - timedelta(seconds=31),
    )
    result = executor.execute(stale)
    assert result.status is PlanExecutionStatus.REJECTED
    assert result.step_results == ()
    assert drone.get_state().y_m == 0.0


@pytest.mark.parametrize(
    ("config", "steps", "reason"),
    [
        (
            PlanConfig(max_steps=1),
            (step(ActionType.STOP), step(ActionType.STOP)),
            "exceeds 1 steps",
        ),
        (
            PlanConfig(max_total_distance_m=1.0),
            (
                step(ActionType.MOVE_FORWARD, distance_m=0.6),
                step(ActionType.MOVE_RIGHT, distance_m=0.6),
            ),
            "total movement",
        ),
        (
            PlanConfig(max_total_rotation_degrees=90),
            (
                step(ActionType.ROTATE_LEFT, degrees=60.0),
                step(ActionType.ROTATE_RIGHT, degrees=60.0),
            ),
            "total rotation",
        ),
    ],
)
def test_executor_enforces_plan_budgets(
    config: PlanConfig, steps: tuple[PlannedAction, ...], reason: str
) -> None:
    drone, _, executor = flying_executor(config=config)
    result = executor.execute(plan(*steps))
    assert result.status is PlanExecutionStatus.REJECTED
    assert reason in result.reason
    assert drone.get_state().x_m == 0.0 and drone.get_state().y_m == 0.0


def test_executor_aborts_remaining_steps_and_stabilizes() -> None:
    drone, _, executor = flying_executor()
    result = executor.execute(
        plan(
            step(ActionType.MOVE_FORWARD, distance_m=0.3),
            step(ActionType.MOVE_DOWN, distance_m=3.0),
            step(ActionType.MOVE_BACKWARD, distance_m=0.3),
        )
    )
    assert result.status is PlanExecutionStatus.ABORTED
    assert len(result.step_results) == 2
    assert result.step_results[-1].safety_result.status is SafetyStatus.REJECTED
    assert result.stabilization_result is not None and result.stabilization_result.approved
    assert drone.get_state().y_m == pytest.approx(0.3)


def test_manual_override_blocks_ai_plan_at_first_step() -> None:
    drone, safety, executor = flying_executor()
    safety.set_manual_override(True)
    result = executor.execute(
        plan(step(ActionType.MOVE_FORWARD, distance_m=0.3), source=CommandSource.AI)
    )
    assert result.status is PlanExecutionStatus.ABORTED
    assert len(result.step_results) == 1
    assert drone.get_state().y_m == 0.0


def test_final_land_step_leaves_drone_landed() -> None:
    drone, _, executor = flying_executor()
    result = executor.execute(plan(step(ActionType.STOP), step(ActionType.LAND)))
    assert result.completed
    assert not drone.get_state().is_flying


class RecordingTransport:
    def __init__(self, result: dict[str, Any]) -> None:
        self.result = result
        self.calls: list[tuple[str, dict[str, Any], float]] = []

    def __call__(self, url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        self.calls.append((url, payload, timeout))
        return self.result


def llm_response(
    steps: list[dict[str, Any]], *, accepted: bool = True, reason: str = "clear plan"
) -> dict[str, Any]:
    return {
        "message": {
            "content": json.dumps({"accepted": accepted, "steps": steps, "reason": reason})
        }
    }


def llm_step(
    action: str,
    *,
    distance_m: float | None = None,
    degrees: float | None = None,
) -> dict[str, Any]:
    return {
        "action": action,
        "distance_m": distance_m,
        "degrees": degrees,
        "target_class": None,
    }


def test_ollama_planner_uses_schema_and_returns_ai_plan() -> None:
    transport = RecordingTransport(
        llm_response(
            [
                llm_step("move_forward", distance_m=0.5),
                llm_step("rotate_right", degrees=90.0),
            ]
        )
    )
    result = OllamaPlanInterpreter(transport=transport).interpret_plan("go ahead and face right")
    assert result.source is CommandSource.AI
    assert [item.action for item in result.steps] == [
        ActionType.MOVE_FORWARD,
        ActionType.ROTATE_RIGHT,
    ]
    payload = transport.calls[0][1]
    assert payload["stream"] is False and payload["think"] is False
    assert payload["format"]["type"] == "object"
    assert payload["options"] == {"temperature": 0}


def test_ollama_planner_rejection_and_unsafe_output_fail_closed() -> None:
    rejected = RecordingTransport(llm_response([], accepted=False, reason="missing distance"))
    with pytest.raises(OllamaInterpreterError, match="missing distance"):
        OllamaPlanInterpreter(transport=rejected).interpret_plan("go somewhere")

    unsafe = RecordingTransport(llm_response([llm_step("takeoff")]))
    with pytest.raises(OllamaInterpreterError, match="unsafe plan"):
        OllamaPlanInterpreter(transport=unsafe).interpret_plan("take off")


def test_ollama_planner_honors_configured_endpoint_and_limit() -> None:
    transport = RecordingTransport(llm_response([llm_step("stop")]))
    config = LLMConfig(
        ollama_url="http://127.0.0.1:9999/",
        model_name="planner-test",
        request_timeout_s=4,
        max_command_characters=5,
    )
    interpreter = OllamaPlanInterpreter(config, transport=transport)
    assert interpreter.interpret_plan("stop").steps[0].action is ActionType.STOP
    assert transport.calls[0][0] == "http://127.0.0.1:9999/api/chat"
    assert transport.calls[0][1]["model"] == "planner-test"
    with pytest.raises(OllamaInterpreterError, match="exceeds 5"):
        interpreter.interpret_plan("123456")
