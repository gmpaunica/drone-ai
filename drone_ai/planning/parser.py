"""Deterministic compound parser and rule-first planner composition."""

from __future__ import annotations

import re

from drone_ai.commands import CommandSource
from drone_ai.parser import CommandParseError, RuleBasedCommandParser
from drone_ai.planning.models import ActionPlan, PlannedAction


class PlanParseError(ValueError):
    pass


class RuleBasedPlanInterpreter:
    def __init__(self, parser: RuleBasedCommandParser | None = None) -> None:
        self.parser = parser or RuleBasedCommandParser()

    def interpret_plan(self, text: str) -> ActionPlan:
        normalized = " ".join(text.strip().split())
        if not normalized:
            raise PlanParseError("plan is empty")
        segments = _split_segments(normalized)
        if not segments:
            raise PlanParseError("plan contains no actions")
        steps: list[PlannedAction] = []
        for index, segment in enumerate(segments):
            try:
                command = self.parser.parse(segment, source=CommandSource.MANUAL)
            except CommandParseError as exc:
                category = "unknown" if str(exc).startswith("unknown command:") else "unsafe"
                raise PlanParseError(f"{category} step {index + 1}: {exc}") from exc
            try:
                steps.append(
                    PlannedAction(
                        action=command.action,
                        distance_m=command.distance_m,
                        degrees=command.degrees,
                        target_class=command.target_class,
                    )
                )
            except ValueError as exc:
                raise PlanParseError(f"unsafe step {index + 1}: {exc}") from exc
        try:
            return ActionPlan(
                original_text=normalized,
                source=CommandSource.MANUAL,
                steps=tuple(steps),
            )
        except ValueError as exc:
            raise PlanParseError(f"invalid plan: {exc}") from exc


class HybridPlanInterpreter:
    def __init__(self, llm: object, rules: RuleBasedPlanInterpreter | None = None) -> None:
        self.llm = llm
        self.rules = rules or RuleBasedPlanInterpreter()

    def interpret_plan(self, text: str) -> ActionPlan:
        try:
            return self.rules.interpret_plan(text)
        except PlanParseError as exc:
            if not str(exc).startswith("unknown step"):
                raise
        method = getattr(self.llm, "interpret_plan", None)
        if method is None:
            raise TypeError("LLM planner must define interpret_plan")
        return method(text)


_NUMBER_WORDS = (
    "zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    "thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|"
    "thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand"
)


def _split_segments(text: str) -> list[str]:
    # Preserve the conjunction inside number phrases such as "one hundred and eighty".
    protected = re.sub(
        rf"\b({_NUMBER_WORDS})\s+and\s+({_NUMBER_WORDS})\b",
        lambda match: f"{match.group(1)} __numeric_and__ {match.group(2)}",
        text,
        flags=re.I,
    )
    pieces = re.split(
        r"\s*(?:,|;|\b(?:and\s+then|then|and)\b)\s*",
        protected,
        flags=re.I,
    )
    return [piece.replace("__numeric_and__", "and").strip() for piece in pieces if piece.strip()]
