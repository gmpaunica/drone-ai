from __future__ import annotations

from typing import Any

from drone_ai.abilities.follow_person import FollowPersonAbility
from drone_ai.cameras.base import CameraSource
from drone_ai.commands import ActionType
from drone_ai.controllers.simulated import SimulatedDrone
from drone_ai.pipeline import FollowPipeline
from drone_ai.safety import SafetyController, SafetyStatus
from drone_ai.vision.models import Detection
from drone_ai.vision.tracker import PersonTracker


class FakeFrame:
    shape = (100, 100, 3)


class FakeCamera(CameraSource):
    def __init__(self) -> None:
        self.opened = False
        self.closed = False

    def open(self) -> None:
        self.opened = True

    def read(self) -> FakeFrame:
        assert self.opened and not self.closed
        return FakeFrame()

    def close(self) -> None:
        self.closed = True


class FakeDetector:
    def detect(self, frame: Any) -> list[Detection]:
        assert frame.shape == (100, 100, 3)
        # Centered, 4% of the frame: target is far.
        return [Detection("person", 0.95, (40, 40, 60, 60))]


def test_camera_to_simulated_drone_control_tick_and_safety_gate() -> None:
    camera = FakeCamera()
    drone = SimulatedDrone()
    drone.connect()
    drone.takeoff()
    safety = SafetyController(drone)
    pipeline = FollowPipeline(FakeDetector(), PersonTracker(), FollowPersonAbility(), safety)

    camera.open()
    tick = pipeline.tick(camera.read())
    camera.close()

    assert tick.command.action is ActionType.MOVE_FORWARD
    assert tick.safety_result.status is SafetyStatus.APPROVED
    assert drone.get_state().y_m == 0.3
    assert camera.closed


def test_ability_cannot_bypass_manual_override() -> None:
    drone = SimulatedDrone()
    drone.connect()
    drone.takeoff()
    safety = SafetyController(drone)
    safety.set_manual_override(True)
    pipeline = FollowPipeline(FakeDetector(), PersonTracker(), FollowPersonAbility(), safety)

    result = pipeline.tick(FakeFrame())

    assert result.command.action is ActionType.MOVE_FORWARD
    assert result.safety_result.status is SafetyStatus.REJECTED
    assert drone.get_state().y_m == 0.0
