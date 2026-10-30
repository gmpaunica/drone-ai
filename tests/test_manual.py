from __future__ import annotations

import argparse
from datetime import datetime, timezone
import sys
from typing import Any

import numpy as np
import pytest

import drone_ai.cli as cli
from drone_ai.cameras.blank import BlankCamera
from drone_ai.commands import ActionType, CommandSource
from drone_ai.controllers.simulated import SimulatedDrone
from drone_ai.manual import ManualCommandMapper, ManualController
from drone_ai.safety import SafetyController, SafetyStatus


NOW = datetime(2026, 8, 30, 12, 0, tzinfo=timezone.utc)


@pytest.mark.parametrize(
    ("key", "action", "distance", "degrees"),
    [
        ("t", ActionType.TAKEOFF, None, None),
        ("l", ActionType.LAND, None, None),
        ("h", ActionType.STOP, None, None),
        ("e", ActionType.EMERGENCY_STOP, None, None),
        ("w", ActionType.MOVE_FORWARD, 0.2, None),
        ("s", ActionType.MOVE_BACKWARD, 0.2, None),
        ("a", ActionType.ROTATE_LEFT, None, 20.0),
        ("d", ActionType.ROTATE_RIGHT, None, 20.0),
        (" ", ActionType.MOVE_UP, 0.2, None),
        ("ctrl", ActionType.MOVE_DOWN, 0.2, None),
        ("r", ActionType.MOVE_UP, 0.2, None),
        ("f", ActionType.MOVE_DOWN, 0.2, None),
        ("j", ActionType.MOVE_LEFT, 0.2, None),
        ("k", ActionType.MOVE_RIGHT, 0.2, None),
    ],
)
def test_manual_key_mapping(
    key: str,
    action: ActionType,
    distance: float | None,
    degrees: float | None,
) -> None:
    command = ManualCommandMapper().map_key(key.upper(), timestamp=NOW)
    assert command is not None
    assert command.action is action
    assert command.source is CommandSource.MANUAL
    assert command.distance_m == distance
    assert command.degrees == degrees
    assert command.timestamp == NOW


def test_manual_mapper_ignores_unknown_and_quit_keys() -> None:
    mapper = ManualCommandMapper()
    assert mapper.map_key("?") is None
    assert mapper.map_key("q") is None


def test_manual_controller_routes_motion_through_safety_and_rate_limits() -> None:
    moments = iter([0.0, 0.0, 0.0, 0.1, 0.5])
    drone = SimulatedDrone()
    drone.connect()
    safety = SafetyController(drone)
    controller = ManualController(safety, monotonic_clock=lambda: next(moments))

    assert controller.handle_key("t").safety_result.approved  # type: ignore[union-attr]
    first = controller.handle_key("w")
    limited = controller.handle_key("d")
    second = controller.handle_key("d")

    assert first.safety_result is not None and first.safety_result.approved
    assert limited.suppressed and limited.safety_result is None
    assert second.safety_result is not None and second.safety_result.approved
    state = drone.get_state()
    assert state.x_m == pytest.approx(0.0)
    assert state.y_m == pytest.approx(0.2)
    assert state.yaw_degrees == pytest.approx(20.0)


def test_manual_controller_suppresses_queued_takeoff_and_land_repeats() -> None:
    drone = SimulatedDrone()
    drone.connect()
    controller = ManualController(
        SafetyController(drone),
        monotonic_clock=lambda: 0.0,
    )

    assert controller.handle_key("t").safety_result.approved  # type: ignore[union-attr]
    repeated_takeoff = controller.handle_key("t")
    assert repeated_takeoff.suppressed
    assert repeated_takeoff.safety_result is None

    assert controller.handle_key("l").safety_result.approved  # type: ignore[union-attr]
    repeated_land = controller.handle_key("l")
    assert repeated_land.suppressed
    assert repeated_land.safety_result is None


def test_manual_emergency_bypasses_motion_rate_limit() -> None:
    drone = SimulatedDrone()
    drone.connect()
    safety = SafetyController(drone)
    controller = ManualController(safety, monotonic_clock=lambda: 0.0)
    assert controller.handle_key("t").safety_result.approved  # type: ignore[union-attr]
    assert controller.handle_key("w").safety_result.approved  # type: ignore[union-attr]
    emergency = controller.handle_key("e")
    assert emergency.safety_result is not None
    assert emergency.safety_result.status is SafetyStatus.APPROVED
    assert not drone.get_state().is_flying


def test_manual_hover_is_immediate_but_rapid_repeats_are_suppressed() -> None:
    drone = SimulatedDrone()
    drone.connect()
    controller = ManualController(SafetyController(drone), monotonic_clock=lambda: 0.0)
    assert controller.handle_key("t").safety_result.approved  # type: ignore[union-attr]
    assert controller.handle_key("w").safety_result.approved  # type: ignore[union-attr]

    hover = controller.handle_key("h")
    repeated = controller.handle_key("h")

    assert hover.safety_result is not None and hover.safety_result.approved
    assert repeated.suppressed and repeated.safety_result is None


def test_blank_camera_produces_bgr_frames() -> None:
    camera = BlankCamera(width=320, height=240)
    camera.open()
    frame = camera.read()
    camera.close()
    assert isinstance(frame, np.ndarray)
    assert frame.shape == (240, 320, 3)
    assert not frame.any()


def test_simulator_manual_cli_defaults_to_blank_camera_and_orderly_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_loop(
        camera: Any,
        safety: SafetyController,
        controller: ManualController,
        recorder: Any = None,
        watchdog: Any = None,
    ) -> int:
        assert recorder is None
        assert watchdog is not None
        captured["camera"] = camera
        captured["override"] = safety.manual_override
        takeoff = controller.handle_key("t")
        assert takeoff.safety_result is not None and takeoff.safety_result.approved
        captured["drone"] = safety.drone
        return 0

    monkeypatch.setattr(cli, "_manual_loop", fake_loop)
    args = argparse.Namespace(drone_mode="simulator", camera=None, camera_index=0)
    assert cli.run_manual(args) == 0
    assert isinstance(captured["camera"], BlankCamera)
    assert captured["override"] is True
    state = captured["drone"].get_state()
    assert not state.is_connected
    assert not state.is_flying


def test_manual_loop_processes_takeoff_motion_and_quit(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeCV2:
        FONT_HERSHEY_SIMPLEX = 0

        def __init__(self) -> None:
            self.keys = iter([ord("t"), ord("w"), ord("q")])
            self.frames = 0
            self.destroyed = False

        def putText(self, *args: Any, **kwargs: Any) -> None:
            return None

        def imshow(self, name: str, frame: Any) -> None:
            assert frame.shape == (240, 320, 3)
            self.frames += 1

        def waitKey(self, delay: int) -> int:
            assert delay == 25
            return next(self.keys)

        def destroyAllWindows(self) -> None:
            self.destroyed = True

    fake_cv2 = FakeCV2()
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    drone = SimulatedDrone()
    drone.connect()
    safety = SafetyController(drone)
    controller = ManualController(safety, monotonic_clock=lambda: 0.0)
    camera = BlankCamera(width=320, height=240)
    camera.open()
    try:
        assert cli._manual_loop(camera, safety, controller) == 0
    finally:
        camera.close()
    assert drone.get_state().is_flying
    assert drone.get_state().y_m == pytest.approx(0.2)
    assert fake_cv2.frames == 3
    assert fake_cv2.destroyed


def test_manual_loop_treats_window_close_as_orderly_quit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeCV2:
        FONT_HERSHEY_SIMPLEX = 0
        WND_PROP_VISIBLE = 4

        def __init__(self) -> None:
            self.frames = 0
            self.destroyed = False

        def putText(self, *args: Any, **kwargs: Any) -> None:
            return None

        def imshow(self, name: str, frame: Any) -> None:
            self.frames += 1

        def getWindowProperty(self, name: str, property_id: int) -> float:
            return 0.0

        def waitKey(self, delay: int) -> int:
            return 255

        def destroyAllWindows(self) -> None:
            self.destroyed = True

    fake_cv2 = FakeCV2()
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    drone = SimulatedDrone()
    drone.connect()
    safety = SafetyController(drone)
    camera = BlankCamera(width=320, height=240)
    camera.open()
    try:
        assert cli._manual_loop(camera, safety, ManualController(safety)) == 0
    finally:
        camera.close()
    assert fake_cv2.frames == 1
    assert fake_cv2.destroyed


def test_manual_loop_uses_held_key_state_and_hovers_on_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeCV2:
        FONT_HERSHEY_SIMPLEX = 0

        def __init__(self) -> None:
            self.keys = iter([255, 255, 255, ord("q")])

        def putText(self, *args: Any, **kwargs: Any) -> None:
            return None

        def imshow(self, name: str, frame: Any) -> None:
            return None

        def waitKey(self, delay: int) -> int:
            return next(self.keys)

        def destroyAllWindows(self) -> None:
            return None

    held_states = iter([{"w"}, {"w"}, set(), set()])
    monkeypatch.setitem(sys.modules, "cv2", FakeCV2())
    monkeypatch.setattr(cli, "_windows_pressed_manual_keys", lambda window: next(held_states))
    drone = SimulatedDrone()
    drone.connect()
    drone.takeoff()
    safety = SafetyController(drone)
    camera = BlankCamera(width=320, height=240)
    camera.open()
    try:
        assert cli._manual_loop(camera, safety, ManualController(safety)) == 0
    finally:
        camera.close()

    assert drone.get_state().y_m == pytest.approx(0.2)
