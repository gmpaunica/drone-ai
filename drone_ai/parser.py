"""Small, intentionally conservative natural-language command parser."""

from __future__ import annotations

import re

from drone_ai.commands import ActionCommand, ActionType, CommandSource


class CommandParseError(ValueError):
    pass


_DIRECTIONS = {
    "forward": ActionType.MOVE_FORWARD,
    "forwards": ActionType.MOVE_FORWARD,
    "back": ActionType.MOVE_BACKWARD,
    "backward": ActionType.MOVE_BACKWARD,
    "backwards": ActionType.MOVE_BACKWARD,
    "left": ActionType.MOVE_LEFT,
    "right": ActionType.MOVE_RIGHT,
    "up": ActionType.MOVE_UP,
    "down": ActionType.MOVE_DOWN,
}

_SMALL_NUMBERS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}
_TENS = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
_UNITS = {"m", "meter", "meters", "metre", "metres", "degree", "degrees"}


class RuleBasedCommandParser:
    def parse(
        self,
        text: str,
        *,
        source: CommandSource = CommandSource.MANUAL,
    ) -> ActionCommand:
        normalized = " ".join(text.lower().strip().replace("-", " ").split())
        normalized = normalized.rstrip(".!?")
        if not normalized:
            raise CommandParseError("command is empty")
        if _contains_compound_joiner(normalized) or any(mark in text for mark in (",", ";")):
            raise CommandParseError("compound or ambiguous commands are not supported")

        simple_actions = {
            "takeoff": ActionType.TAKEOFF,
            "take off": ActionType.TAKEOFF,
            "launch": ActionType.TAKEOFF,
            "land": ActionType.LAND,
            "land now": ActionType.LAND,
            "hover": ActionType.HOVER,
            "stop": ActionType.STOP,
            "hold position": ActionType.HOVER,
            "emergency stop": ActionType.EMERGENCY_STOP,
            "stop immediately": ActionType.EMERGENCY_STOP,
            "follow me": ActionType.FOLLOW_PERSON,
            "follow person": ActionType.FOLLOW_PERSON,
            "follow the person": ActionType.FOLLOW_PERSON,
            "find person": ActionType.FIND_PERSON,
            "find a person": ActionType.FIND_PERSON,
            "find the person": ActionType.FIND_PERSON,
            "look for person": ActionType.FIND_PERSON,
            "look for a person": ActionType.FIND_PERSON,
            "look for the person": ActionType.FIND_PERSON,
        }
        if normalized in simple_actions:
            return ActionCommand(action=simple_actions[normalized], source=source)
        if normalized in {"turn around", "rotate around"}:
            return ActionCommand(
                action=ActionType.ROTATE_RIGHT,
                source=source,
                degrees=180.0,
            )

        movement = self._parse_directional(normalized, {"move", "go", "fly"}, _DIRECTIONS)
        if movement is not None:
            action, value = movement
            return ActionCommand(action=action, source=source, distance_m=value)

        rotation_directions = {
            "left": ActionType.ROTATE_LEFT,
            "counterclockwise": ActionType.ROTATE_LEFT,
            "anticlockwise": ActionType.ROTATE_LEFT,
            "right": ActionType.ROTATE_RIGHT,
            "clockwise": ActionType.ROTATE_RIGHT,
        }
        rotation = self._parse_directional(normalized, {"turn", "rotate"}, rotation_directions)
        if rotation is not None:
            action, value = rotation
            return ActionCommand(action=action, source=source, degrees=value)

        object_match = re.fullmatch(r"(?:find|look for)\s+(.+)", normalized)
        if object_match:
            target = re.sub(r"^(?:a|an|the)\s+", "", object_match.group(1)).strip()
            if not target:
                raise CommandParseError("find command requires an object")
            return ActionCommand(
                action=ActionType.FIND_OBJECT,
                source=source,
                target_class=target,
            )

        if normalized.split()[0] in {"move", "go", "fly", "turn", "rotate"}:
            raise CommandParseError("movement and rotation commands require one direction and amount")
        raise CommandParseError(f"unknown command: {text!r}")

    def _parse_directional(
        self,
        normalized: str,
        verbs: set[str],
        directions: dict[str, ActionType],
    ) -> tuple[ActionType, float] | None:
        tokens = normalized.split()
        if not tokens or tokens[0] not in verbs:
            return None
        body = tokens[1:]
        if not body:
            return None
        direction_positions = [(index, token) for index, token in enumerate(body) if token in directions]
        if len(direction_positions) != 1:
            return None
        index, direction = direction_positions[0]
        number_tokens = body[:index] + body[index + 1 :]
        number_tokens = [token for token in number_tokens if token not in _UNITS and token != "by"]
        if not number_tokens:
            return None
        try:
            value = _number_from_words(number_tokens)
        except ValueError:
            return None
        return directions[direction], value


def _number_from_words(tokens: list[str]) -> float:
    if len(tokens) == 1:
        try:
            return float(tokens[0])
        except ValueError:
            pass

    if "point" in tokens:
        point_index = tokens.index("point")
        whole = _integer_from_words(tokens[:point_index])
        fraction_tokens = tokens[point_index + 1 :]
        digit_words = {word: value for word, value in _SMALL_NUMBERS.items() if value < 10}
        if not fraction_tokens or any(token not in digit_words for token in fraction_tokens):
            raise ValueError("invalid decimal number")
        digits = "".join(str(digit_words[token]) for token in fraction_tokens)
        return float(f"{whole}.{digits}")
    return float(_integer_from_words(tokens))


def _integer_from_words(tokens: list[str]) -> int:
    tokens = [token for token in tokens if token != "and"]
    if not tokens:
        return 0
    if tokens.count("thousand") > 1:
        raise ValueError("invalid number")
    if "thousand" in tokens:
        split = tokens.index("thousand")
        if split == 0:
            raise ValueError("invalid number")
        high = _under_one_thousand(tokens[:split])
        low_tokens = tokens[split + 1 :]
        low = _under_one_thousand(low_tokens) if low_tokens else 0
        return high * 1000 + low
    return _under_one_thousand(tokens)


def _under_one_thousand(tokens: list[str]) -> int:
    if tokens.count("hundred") > 1:
        raise ValueError("invalid number")
    if "hundred" in tokens:
        split = tokens.index("hundred")
        prefix = tokens[:split]
        if len(prefix) != 1 or prefix[0] not in _SMALL_NUMBERS:
            raise ValueError("invalid number")
        hundreds = _SMALL_NUMBERS[prefix[0]]
        if not 1 <= hundreds <= 9:
            raise ValueError("invalid number")
        remainder = tokens[split + 1 :]
        return hundreds * 100 + (_under_one_hundred(remainder) if remainder else 0)
    return _under_one_hundred(tokens)


def _under_one_hundred(tokens: list[str]) -> int:
    if len(tokens) == 1 and tokens[0] in _SMALL_NUMBERS:
        return _SMALL_NUMBERS[tokens[0]]
    if len(tokens) == 1 and tokens[0] in _TENS:
        return _TENS[tokens[0]]
    if (
        len(tokens) == 2
        and tokens[0] in _TENS
        and tokens[1] in _SMALL_NUMBERS
        and 1 <= _SMALL_NUMBERS[tokens[1]] <= 9
    ):
        return _TENS[tokens[0]] + _SMALL_NUMBERS[tokens[1]]
    raise ValueError("invalid number")


def _contains_compound_joiner(normalized: str) -> bool:
    if re.search(r"\b(then|afterwards|after that|while)\b", normalized):
        return True
    tokens = normalized.split()
    for index, token in enumerate(tokens):
        if token != "and":
            continue
        if index == 0 or index == len(tokens) - 1:
            return True
        previous = tokens[index - 1]
        following = tokens[index + 1]
        number_words = set(_SMALL_NUMBERS) | set(_TENS) | {"hundred", "thousand"}
        numeric_conjunction = (
            tokens[0] in {"move", "go", "fly", "turn", "rotate"}
            and previous in number_words
            and following in number_words
        )
        if not numeric_conjunction:
            return True
    return False
