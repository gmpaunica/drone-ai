from __future__ import annotations

from datetime import datetime, timezone
import math

import pytest
from pydantic import ValidationError

from drone_ai.commands import ActionCommand, ActionType, CommandSource
from drone_ai.parser import CommandParseError, RuleBasedCommandParser


def command(action: ActionType, **kwargs: object) -> ActionCommand:
    return ActionCommand(action=action, source=CommandSource.MANUAL, **kwargs)


@pytest.mark.parametrize(
    ("action", "kwargs"),
    [
        (ActionType.MOVE_FORWARD, {}),
        (ActionType.ROTATE_LEFT, {}),
        (ActionType.FIND_OBJECT, {}),
        (ActionType.STOP, {"distance_m": 1.0}),
        (ActionType.MOVE_RIGHT, {"distance_m": 1.0, "degrees": 2.0}),
        (ActionType.ROTATE_RIGHT, {"degrees": 20.0, "target_class": "person"}),
        (ActionType.FIND_PERSON, {"target_class": "person"}),
    ],
)
def test_action_specific_required_and_forbidden_fields(
    action: ActionType, kwargs: dict[str, object]
) -> None:
    with pytest.raises(ValidationError):
        command(action, **kwargs)


@pytest.mark.parametrize("value", [0.0, -1.0, math.inf, -math.inf, math.nan])
def test_numeric_parameters_must_be_finite_and_positive(value: float) -> None:
    with pytest.raises(ValidationError):
        command(ActionType.MOVE_FORWARD, distance_m=value)


def test_schema_is_strict_and_forbids_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        ActionCommand.model_validate(
            {"action": "move_forward", "source": "MANUAL", "distance_m": 1.0}
        )
    with pytest.raises(ValidationError):
        command(ActionType.LAND, surprise=True)


def test_timestamp_is_aware_and_model_is_immutable() -> None:
    item = command(ActionType.STOP)
    assert item.timestamp.tzinfo is not None
    with pytest.raises(ValidationError):
        item.action = ActionType.LAND  # type: ignore[misc]
    with pytest.raises(ValidationError):
        ActionCommand(
            action=ActionType.STOP,
            source=CommandSource.MANUAL,
            timestamp=datetime(2026, 1, 1),
        )


@pytest.mark.parametrize(
    ("text", "action", "parameter"),
    [
        ("take off", ActionType.TAKEOFF, None),
        ("land", ActionType.LAND, None),
        ("hover", ActionType.HOVER, None),
        ("stop", ActionType.STOP, None),
        ("emergency stop", ActionType.EMERGENCY_STOP, None),
        ("move forward two meters", ActionType.MOVE_FORWARD, 2.0),
        ("fly 0.5 m backwards", ActionType.MOVE_BACKWARD, 0.5),
        ("go left one metre", ActionType.MOVE_LEFT, 1.0),
        ("move right three meters", ActionType.MOVE_RIGHT, 3.0),
        ("fly up one point five meters", ActionType.MOVE_UP, 1.5),
        ("move 2 meters down", ActionType.MOVE_DOWN, 2.0),
        ("turn left ninety degrees", ActionType.ROTATE_LEFT, 90.0),
        ("turn left one hundred and eighty degrees", ActionType.ROTATE_LEFT, 180.0),
        ("rotate 45 degrees clockwise", ActionType.ROTATE_RIGHT, 45.0),
        ("turn around", ActionType.ROTATE_RIGHT, 180.0),
        ("follow me", ActionType.FOLLOW_PERSON, None),
        ("find a person", ActionType.FIND_PERSON, None),
        ("look for the backpack", ActionType.FIND_OBJECT, "backpack"),
    ],
)
def test_parser_examples(text: str, action: ActionType, parameter: object) -> None:
    parsed = RuleBasedCommandParser().parse(text)
    assert parsed.action is action
    assert parsed.source is CommandSource.MANUAL
    actual = parsed.distance_m or parsed.degrees or parsed.target_class
    assert actual == parameter


@pytest.mark.parametrize(
    "text",
    [
        "dance",
        "move forward",
        "turn left",
        "move left and land",
        "move left, then right",
        "fly left right two meters",
        "move forward one two meters",
        "find person and follow me",
    ],
)
def test_parser_rejects_unknown_ambiguous_parameterless_and_compound(text: str) -> None:
    with pytest.raises((CommandParseError, ValidationError)):
        RuleBasedCommandParser().parse(text)
