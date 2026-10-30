"""Fail-stop sequential execution with fresh per-step safety validation."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
import logging

from pydantic import ValidationError

from drone_ai.commands import ActionCommand, ActionType, CommandSource, MOVEMENT_ACTIONS, ROTATION_ACTIONS
from drone_ai.config import PlanConfig
from drone_ai.planning.models import (
    ActionPlan,
    PlanExecutionResult,
    PlanExecutionStatus,
    PlanStepResult,
)
from drone_ai.safety import SafetyController, SafetyResult


LOGGER = logging.getLogger(__name__)


class PlanExecutor:
    def __init__(
        self,
        safety: SafetyController,
        config: PlanConfig | None = None,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.safety = safety
        self.config = config or PlanConfig()
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    def execute(self, plan: ActionPlan) -> PlanExecutionResult:
        validated, rejection = self._validate_plan(plan)
        if rejection is not None:
            LOGGER.warning("[PLAN] REJECTED: %s", rejection)
            return PlanExecutionResult(
                status=PlanExecutionStatus.REJECTED,
                reason=rejection,
                step_results=(),
            )
        assert validated is not None
        LOGGER.info("[PLAN] Starting %d-step plan from %s", len(validated.steps), validated.source.value)
        results: list[PlanStepResult] = []
        for index, step in enumerate(validated.steps):
            command = step.to_command(validated.source, self._aware_now())
            LOGGER.info("[PLAN] Step %d/%d: %s", index + 1, len(validated.steps), step.action.value)
            safety_result = self.safety.execute(command)
            results.append(PlanStepResult(index=index, step=step, safety_result=safety_result))
            if not safety_result.approved:
                stabilization = self._stabilize_after_abort(step.action)
                reason = f"aborted at step {index + 1}: {safety_result.reason}"
                LOGGER.warning("[PLAN] ABORTED: %s", reason)
                return PlanExecutionResult(
                    status=PlanExecutionStatus.ABORTED,
                    reason=reason,
                    step_results=tuple(results),
                    stabilization_result=stabilization,
                )
        LOGGER.info("[PLAN] COMPLETED: %d step(s)", len(results))
        return PlanExecutionResult(
            status=PlanExecutionStatus.COMPLETED,
            reason=f"completed {len(results)} step(s)",
            step_results=tuple(results),
        )

    def _validate_plan(self, plan: object) -> tuple[ActionPlan | None, str | None]:
        if not isinstance(plan, ActionPlan):
            return None, "expected ActionPlan"
        try:
            validated = ActionPlan.model_validate(plan.model_dump())
        except (ValidationError, ValueError, TypeError) as exc:
            return None, f"malformed plan: {exc}"
        now = self._aware_now()
        age_s = (now - validated.created_at).total_seconds()
        if age_s > self.config.max_plan_age_s:
            return None, f"plan is stale ({age_s:.2f}s > {self.config.max_plan_age_s:.2f}s)"
        if len(validated.steps) > self.config.max_steps:
            return None, f"plan exceeds {self.config.max_steps} steps"
        total_distance = sum(
            step.distance_m or 0.0 for step in validated.steps if step.action in MOVEMENT_ACTIONS
        )
        if total_distance > self.config.max_total_distance_m:
            return None, (
                f"total movement {total_distance:g}m exceeds "
                f"{self.config.max_total_distance_m:g}m plan budget"
            )
        total_rotation = sum(
            step.degrees or 0.0 for step in validated.steps if step.action in ROTATION_ACTIONS
        )
        if total_rotation > self.config.max_total_rotation_degrees:
            return None, (
                f"total rotation {total_rotation:g} degrees exceeds "
                f"{self.config.max_total_rotation_degrees:g}-degree plan budget"
            )
        return validated, None

    def _stabilize_after_abort(self, failed_action: ActionType) -> SafetyResult | None:
        if failed_action in {ActionType.STOP, ActionType.HOVER, ActionType.LAND}:
            return None
        try:
            state = self.safety.drone.get_state()
        except Exception:
            return None
        if not state.is_connected or not state.is_flying:
            return None
        LOGGER.warning("[PLAN] Attempting system hover after abort")
        return self.safety.execute(
            ActionCommand(
                action=ActionType.STOP,
                source=CommandSource.SYSTEM,
                timestamp=self._aware_now(),
            )
        )

    def _aware_now(self) -> datetime:
        now = self._clock()
        if now.tzinfo is None or now.utcoffset() is None:
            return now.replace(tzinfo=timezone.utc)
        return now
