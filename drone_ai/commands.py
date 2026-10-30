"""Typed command schema shared by parsers, abilities, and safety."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
import math

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ActionType(str, Enum):
    TAKEOFF = "takeoff"
    LAND = "land"
    HOVER = "hover"
    STOP = "stop"
    EMERGENCY_STOP = "emergency_stop"
    MOVE_FORWARD = "move_forward"
    MOVE_BACKWARD = "move_backward"
    MOVE_LEFT = "move_left"
    MOVE_RIGHT = "move_right"
    MOVE_UP = "move_up"
    MOVE_DOWN = "move_down"
    ROTATE_LEFT = "rotate_left"
    ROTATE_RIGHT = "rotate_right"
    FOLLOW_PERSON = "follow_person"
    FIND_PERSON = "find_person"
    FIND_OBJECT = "find_object"


class CommandSource(str, Enum):
    AI = "AI"
    ABILITY = "ABILITY"
    MANUAL = "MANUAL"
    SYSTEM = "SYSTEM"


MOVEMENT_ACTIONS = frozenset(
    {
        ActionType.MOVE_FORWARD,
        ActionType.MOVE_BACKWARD,
        ActionType.MOVE_LEFT,
        ActionType.MOVE_RIGHT,
        ActionType.MOVE_UP,
        ActionType.MOVE_DOWN,
    }
)
ROTATION_ACTIONS = frozenset({ActionType.ROTATE_LEFT, ActionType.ROTATE_RIGHT})


class ActionCommand(BaseModel):
    """A fully validated, single drone action.

    Parameters are deliberately action-specific. Extra, missing, irrelevant,
    zero, negative, infinite, and NaN values are rejected at the boundary.
    """

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    action: ActionType
    source: CommandSource
    distance_m: float | None = None
    degrees: float | None = None
    target_class: str | None = Field(default=None, min_length=1)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def validate_action_parameters(self) -> "ActionCommand":
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise ValueError("timestamp must include a timezone")

        has_distance = self.distance_m is not None
        has_degrees = self.degrees is not None
        has_target = self.target_class is not None

        if self.action in MOVEMENT_ACTIONS:
            if not has_distance:
                raise ValueError("movement actions require distance_m")
            if has_degrees or has_target:
                raise ValueError("movement actions only accept distance_m")
            self._require_finite_positive(self.distance_m, "distance_m")
        elif self.action in ROTATION_ACTIONS:
            if not has_degrees:
                raise ValueError("rotation actions require degrees")
            if has_distance or has_target:
                raise ValueError("rotation actions only accept degrees")
            self._require_finite_positive(self.degrees, "degrees")
        elif self.action is ActionType.FIND_OBJECT:
            if not has_target:
                raise ValueError("find_object requires target_class")
            if has_distance or has_degrees:
                raise ValueError("find_object only accepts target_class")
            if not self.target_class or not self.target_class.strip():
                raise ValueError("target_class cannot be blank")
        elif has_distance or has_degrees or has_target:
            raise ValueError(f"{self.action.value} does not accept parameters")

        return self

    @staticmethod
    def _require_finite_positive(value: float | None, field_name: str) -> None:
        if value is None or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{field_name} must be finite and positive")
