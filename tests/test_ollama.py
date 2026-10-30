from __future__ import annotations

from datetime import timezone
from io import BytesIO
import json
from typing import Any
from urllib.error import HTTPError, URLError

import pytest

from drone_ai.commands import ActionType, CommandSource
from drone_ai.config import LLMConfig
from drone_ai.controllers.simulated import SimulatedDrone
from drone_ai.llm.interpreter import HybridCommandInterpreter
from drone_ai.llm.ollama import (
    OllamaCommandInterpreter,
    OllamaInterpreterError,
    _post_json,
)
from drone_ai.parser import CommandParseError
from drone_ai.safety import SafetyController, SafetyStatus


def response(
    *,
    accepted: bool = True,
    action: str | None = "move_forward",
    distance_m: float | None = 0.5,
    degrees: float | None = None,
    target_class: str | None = None,
    reason: str = "one clear movement",
) -> dict[str, Any]:
    content = {
        "accepted": accepted,
        "action": action,
        "distance_m": distance_m,
        "degrees": degrees,
        "target_class": target_class,
        "reason": reason,
    }
    return {"message": {"role": "assistant", "content": json.dumps(content)}}


class RecordingTransport:
    def __init__(self, result: dict[str, Any]) -> None:
        self.result = result
        self.calls: list[tuple[str, dict[str, Any], float]] = []

    def __call__(self, url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        self.calls.append((url, payload, timeout))
        return self.result


def test_ollama_uses_local_structured_nonstreaming_deterministic_request() -> None:
    transport = RecordingTransport(response())
    interpreter = OllamaCommandInterpreter(transport=transport)
    command = interpreter.interpret("ease ahead by half a meter")

    assert command.action is ActionType.MOVE_FORWARD
    assert command.distance_m == 0.5
    assert command.source is CommandSource.AI
    assert command.timestamp.tzinfo is timezone.utc
    url, payload, timeout = transport.calls[0]
    assert url == "http://localhost:11434/api/chat"
    assert payload["model"] == "qwen3:4b-instruct"
    assert payload["stream"] is False
    assert payload["think"] is False
    assert payload["options"] == {"temperature": 0}
    assert payload["format"]["type"] == "object"
    assert payload["messages"][-1] == {
        "role": "user",
        "content": "ease ahead by half a meter",
    }
    assert timeout == 120.0


def test_configurable_url_model_and_timeout() -> None:
    transport = RecordingTransport(response(action="land", distance_m=None))
    config = LLMConfig(
        ollama_url="http://127.0.0.1:9999/",
        model_name="local-test",
        request_timeout_s=3.0,
    )
    command = OllamaCommandInterpreter(config, transport=transport).interpret("put it down")
    assert command.action is ActionType.LAND
    assert transport.calls[0][0] == "http://127.0.0.1:9999/api/chat"
    assert transport.calls[0][1]["model"] == "local-test"
    assert transport.calls[0][2] == 3.0


def test_model_rejection_is_not_a_command() -> None:
    transport = RecordingTransport(
        response(
            accepted=False,
            action=None,
            distance_m=None,
            reason="contains two actions",
        )
    )
    with pytest.raises(OllamaInterpreterError, match="contains two actions"):
        OllamaCommandInterpreter(transport=transport).interpret("go forward and land")


@pytest.mark.parametrize(
    "result",
    [
        {},
        {"message": {}},
        {"message": {"content": 123}},
        {"message": {"content": "not json"}},
        response(action="move_forward", distance_m=None),
        response(action="stop", distance_m=1.0),
    ],
)
def test_malformed_or_unsafe_model_output_is_rejected(result: dict[str, Any]) -> None:
    with pytest.raises(OllamaInterpreterError):
        OllamaCommandInterpreter(transport=RecordingTransport(result)).interpret("anything")


def test_llm_cannot_create_emergency_command() -> None:
    result = response(action="emergency_stop", distance_m=None)
    with pytest.raises(OllamaInterpreterError, match="use 'emergency stop'"):
        OllamaCommandInterpreter(transport=RecordingTransport(result)).interpret("kill everything")


def test_input_length_is_bounded_before_transport() -> None:
    transport = RecordingTransport(response())
    config = LLMConfig(max_command_characters=5)
    with pytest.raises(OllamaInterpreterError, match="exceeds 5"):
        OllamaCommandInterpreter(config, transport=transport).interpret("123456")
    assert transport.calls == []


def test_hybrid_prefers_rules_and_preserves_manual_source() -> None:
    transport = RecordingTransport(response())
    hybrid = HybridCommandInterpreter(OllamaCommandInterpreter(transport=transport))
    command = hybrid.interpret("move forward two meters")
    assert command.source is CommandSource.MANUAL
    assert command.distance_m == 2.0
    assert transport.calls == []


def test_hybrid_uses_llm_only_for_unknown_wording() -> None:
    transport = RecordingTransport(response())
    hybrid = HybridCommandInterpreter(OllamaCommandInterpreter(transport=transport))
    assert hybrid.interpret("ease ahead by half a meter").source is CommandSource.AI
    assert len(transport.calls) == 1


@pytest.mark.parametrize("text", ["move forward", "move forward and land"])
def test_hybrid_does_not_let_llm_guess_known_unsafe_shapes(text: str) -> None:
    transport = RecordingTransport(response())
    hybrid = HybridCommandInterpreter(OllamaCommandInterpreter(transport=transport))
    with pytest.raises(CommandParseError):
        hybrid.interpret(text)
    assert transport.calls == []


def test_ai_tag_means_manual_override_blocks_llm_command() -> None:
    command = OllamaCommandInterpreter(transport=RecordingTransport(response())).interpret(
        "ease ahead by half a meter"
    )
    drone = SimulatedDrone()
    drone.connect()
    drone.takeoff()
    safety = SafetyController(drone)
    safety.set_manual_override(True)
    result = safety.execute(command)
    assert result.status is SafetyStatus.REJECTED
    assert drone.get_state().y_m == 0.0


def test_connection_error_is_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args: Any, **kwargs: Any) -> None:
        raise URLError("connection refused")

    monkeypatch.setattr("drone_ai.llm.ollama.urlopen", fail)
    with pytest.raises(OllamaInterpreterError, match="could not reach local Ollama"):
        _post_json("http://localhost:11434/api/chat", {"model": "demo"}, 1.0)


def test_missing_model_error_includes_pull_command(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args: Any, **kwargs: Any) -> None:
        raise HTTPError(
            "http://localhost:11434/api/chat",
            404,
            "Not Found",
            {},
            BytesIO(b'{"error":"model not found"}'),
        )

    monkeypatch.setattr("drone_ai.llm.ollama.urlopen", fail)
    with pytest.raises(OllamaInterpreterError, match="ollama pull demo"):
        _post_json("http://localhost:11434/api/chat", {"model": "demo"}, 1.0)
