"""Composable command-interpreter interfaces."""

from __future__ import annotations

from typing import Protocol

from drone_ai.commands import ActionCommand, CommandSource
from drone_ai.parser import CommandParseError, RuleBasedCommandParser


class CommandInterpreter(Protocol):
    def interpret(self, text: str) -> ActionCommand: ...


class RuleCommandInterpreter:
    def __init__(self, parser: RuleBasedCommandParser | None = None) -> None:
        self.parser = parser or RuleBasedCommandParser()

    def interpret(self, text: str) -> ActionCommand:
        return self.parser.parse(text, source=CommandSource.MANUAL)


class HybridCommandInterpreter:
    """Prefer deterministic rules and use the LLM only for unknown wording.

    Known unsafe shapes, such as compound or parameterless movement commands,
    are rejected by the rule parser and never offered to the model for a guess.
    """

    def __init__(
        self,
        llm: CommandInterpreter,
        rules: RuleCommandInterpreter | None = None,
    ) -> None:
        self.rules = rules or RuleCommandInterpreter()
        self.llm = llm

    def interpret(self, text: str) -> ActionCommand:
        try:
            return self.rules.interpret(text)
        except CommandParseError as exc:
            if not str(exc).startswith("unknown command:"):
                raise
        return self.llm.interpret(text)
