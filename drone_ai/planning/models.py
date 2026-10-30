"""Strict schemas for bounded, non-emergency action plans."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from drone_ai.commands import ActionCommand, ActionType, CommandSource
from drone_ai.safety import SafetyResult


FORBIDDEN_PLANNED_ACTIONS = frozenset(
    {
        ActionType.TAKEOFF,
        ActionType.EMERGENCY_STOP,
        ActionType.FOLLOW_PERSON,
        ActionType.FIND_PERSON,
        ActionType.FIND_OBJECT,
    }
)


class PlannedAction(BaseModel):
    """An untimestamped command template that cannot bypass live validation."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    action: ActionType
    distance_m: float | None = None
    degrees: float | None = None
    target_class: str | None = None

    @model_validator(mode="after")
    def validate_primitive(self) -> "PlannedAction":
        if self.action in FORBIDDEN_PLANNED_ACTIONS:
            raise ValueError(f"{self.action.value} cannot be queued in a plan")
        # Reuse the canonical action-specific validation with a fresh timestamp.
        ActionCommand(
            action=self.action,
            source=CommandSource.SYSTEM,
            distance_m=self.distance_m,
            degrees=self.degrees,
            target_class=self.target_class,
        )
        return self

    def to_command(self, source: CommandSource, timestamp: datetime) -> ActionCommand:
        return ActionCommand(
            action=self.action,
            source=source,
            distance_m=self.distance_m,
            degrees=self.degrees,
            target_class=self.target_class,
            timestamp=timestamp,
        )


class ActionPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    original_text: str = Field(min_length=1, max_length=500)
    source: CommandSource
    steps: tuple[PlannedAction, ...] = Field(min_length=1, max_length=8)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def validate_plan_shape(self) -> "ActionPlan":
        if self.source not in {CommandSource.MANUAL, CommandSource.AI}:
            raise ValueError("plans must originate from MANUAL or AI interpretation")
        if self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise ValueError("created_at must include a timezone")
        land_indexes = [index for index, step in enumerate(self.steps) if step.action is ActionType.LAND]
        if land_indexes and land_indexes != [len(self.steps) - 1]:
            raise ValueError("land may appear only once as the final plan step")
        return self


class PlanExecutionStatus(str, Enum):
    COMPLETED = "COMPLETED"
    ABORTED = "ABORTED"
    REJECTED = "REJECTED"


class PlanStepResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    index: int = Field(ge=0)
    step: PlannedAction
    safety_result: SafetyResult


class PlanExecutionResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    status: PlanExecutionStatus
    reason: str
    step_results: tuple[PlanStepResult, ...]
    stabilization_result: SafetyResult | None = None

    @property
    def completed(self) -> bool:
        return self.status is PlanExecutionStatus.COMPLETED
