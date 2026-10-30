from __future__ import annotations

import argparse
from dataclasses import replace
from typing import Any

import pytest

import drone_ai.cli as cli
from drone_ai.cameras.base import CameraSource
from drone_ai.commands import ActionCommand, ActionType, CommandSource
from drone_ai.controllers.base import DroneState
from drone_ai.controllers.simulated import SimulatedDrone
from drone_ai.preflight import AutonomousPreflight, CheckStatus
from drone_ai.safety import SafetyController, SafetyStatus


class FakeFrame:
    shape = (480, 640, 3)


class FakeCamera(CameraSource):
    def __init__(self, frame: Any | None = None, events: list[str] | None = None) -> None:
        self.frame = FakeFrame() if frame is None else frame
        self.events = events
        self.opened = False

    def open(self) -> None:
        self.opened = True
        if self.events is not None:
            self.events.append("camera.open")

    def read(self) -> Any:
        if self.events is not None:
            self.events.append("camera.read")
        return self.frame

    def close(self) -> None:
        self.opened = False
        if self.events is not None:
            self.events.append("camera.close")


class FakeDetector:
    def __init__(
        self,
        *args: Any,
        events: list[str] | None = None,
        fail_inference: bool = False,
        **kwargs: Any,
    ) -> None:
        self.events = events
        self.fail_inference = fail_inference

    def validate_target_classes(self) -> None:
        if self.events is not None:
            self.events.append("detector.validate")

    def detect(self, frame: Any) -> list[Any]:
        if self.events is not None:
            self.events.append("detector.detect")
        if self.fail_inference:
            raise RuntimeError("inference unavailable")
        return []


class StateDrone(SimulatedDrone):
    def __init__(self, *, battery: int | None = 100, events: list[str] | None = None) -> None:
        super().__init__()
        self.battery = battery
        self.events = events

    def connect(self) -> None:
        if self.events is not None:
            self.events.append("drone.connect")
        super().connect()

    def takeoff(self) -> None:
        if self.events is not None:
            self.events.append("drone.takeoff")
        super().takeoff()

    def get_state(self) -> DroneState:
        return replace(super().get_state(), battery_percent=self.battery)


def test_autonomous_preflight_passes_connected_simulator() -> None:
    drone = StateDrone()
    drone.connect()
    report = AutonomousPreflight(drone, FakeCamera(), FakeDetector()).run()
    assert report.ready
    assert all(check.status is CheckStatus.PASS for check in report.checks)
    assert {check.name for check in report.checks} == {
        "drone.connection",
        "drone.flight_state",
        "drone.battery",
        "drone.telemetry",
        "vision.model",
        "camera.frame",
        "vision.inference",
    }


@pytest.mark.parametrize("battery", [None, 29])
def test_takeoff_and_preflight_fail_closed_for_bad_battery(battery: int | None) -> None:
    drone = StateDrone(battery=battery)
    drone.connect()
    report = AutonomousPreflight(drone, FakeCamera(), FakeDetector()).run()
    battery_check = next(check for check in report.checks if check.name == "drone.battery")
    assert battery_check.status is CheckStatus.FAIL
    result = SafetyController(drone).execute(
        ActionCommand(action=ActionType.TAKEOFF, source=CommandSource.SYSTEM)
    )
    assert result.status is SafetyStatus.REJECTED
    assert not drone.get_state().is_flying


def test_bad_frame_skips_inference_and_blocks_readiness() -> None:
    class GrayFrame:
        shape = (480, 640)

    drone = StateDrone()
    drone.connect()
    report = AutonomousPreflight(drone, FakeCamera(GrayFrame()), FakeDetector()).run()
    assert not report.ready
    assert next(c for c in report.checks if c.name == "camera.frame").status is CheckStatus.FAIL
    assert (
        next(c for c in report.checks if c.name == "vision.inference").status
        is CheckStatus.SKIP
    )


def test_landed_altitude_must_be_plausibly_near_ground() -> None:
    class FloatingLandedDrone(StateDrone):
        def get_state(self) -> DroneState:
            return replace(super().get_state(), altitude_m=0.31, is_flying=False)

    drone = FloatingLandedDrone()
    drone.connect()
    report = AutonomousPreflight(drone, FakeCamera(), FakeDetector()).run()
    telemetry = next(check for check in report.checks if check.name == "drone.telemetry")
    assert telemetry.status is CheckStatus.FAIL
    assert not report.ready


def follow_args(*, takeoff: bool = True) -> argparse.Namespace:
    return argparse.Namespace(
        drone_mode="simulator",
        camera="webcam",
        camera_index=0,
        model=None,
        takeoff=takeoff,
    )


def test_follow_warms_camera_and_detector_before_takeoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    drone = StateDrone(events=events)
    camera = FakeCamera(events=events)

    monkeypatch.setattr(cli, "_make_drone", lambda *_: (drone, None))
    monkeypatch.setattr(cli, "_make_camera", lambda *_: (camera, None))
    monkeypatch.setattr(cli, "YOLODetector", lambda *a, **kw: FakeDetector(events=events))

    def fake_loop(*args: Any) -> int:
        events.append("follow.loop")
        assert drone.get_state().is_flying
        return 0

    monkeypatch.setattr(cli, "_follow_loop", fake_loop)
    assert cli.run_follow(follow_args()) == 0
    assert events.index("detector.detect") < events.index("drone.takeoff")
    assert events.index("camera.read") < events.index("drone.takeoff")
    assert events.index("drone.takeoff") < events.index("follow.loop")


def test_follow_detector_failure_never_takes_off(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []
    drone = StateDrone(events=events)
    camera = FakeCamera(events=events)
    monkeypatch.setattr(cli, "_make_drone", lambda *_: (drone, None))
    monkeypatch.setattr(cli, "_make_camera", lambda *_: (camera, None))
    monkeypatch.setattr(
        cli,
        "YOLODetector",
        lambda *a, **kw: FakeDetector(events=events, fail_inference=True),
    )
    monkeypatch.setattr(cli, "_follow_loop", lambda *args: pytest.fail("loop must not start"))
    assert cli.run_follow(follow_args()) == 1
    assert "drone.takeoff" not in events
    assert not drone.get_state().is_flying
