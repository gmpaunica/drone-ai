"""Fail-closed checks that run before autonomous takeoff."""

from __future__ import annotations

from enum import Enum
import math
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, computed_field

from drone_ai.cameras.base import CameraSource
from drone_ai.config import FlightConfig, PreflightConfig
from drone_ai.controllers.base import DroneController


class CheckStatus(str, Enum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"
    SKIP = "SKIP"


class ReadinessCheck(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    name: str
    status: CheckStatus
    detail: str


class ReadinessReport(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    checks: tuple[ReadinessCheck, ...]

    @computed_field
    @property
    def ready(self) -> bool:
        return all(check.status is not CheckStatus.FAIL for check in self.checks)


class WarmupDetector(Protocol):
    def validate_target_classes(self) -> None: ...

    def detect(self, frame: Any) -> list[Any]: ...


class AutonomousPreflight:
    """Verify telemetry, video, and inference before autonomous takeoff.

    The camera must already be open and the drone must already be connected.
    This class never dispatches a flight command.
    """

    def __init__(
        self,
        drone: DroneController,
        camera: CameraSource,
        detector: WarmupDetector,
        flight_config: FlightConfig | None = None,
        preflight_config: PreflightConfig | None = None,
    ) -> None:
        self.drone = drone
        self.camera = camera
        self.detector = detector
        self.flight_config = flight_config or FlightConfig()
        self.preflight_config = preflight_config or PreflightConfig()

    def run(self) -> ReadinessReport:
        checks: list[ReadinessCheck] = []
        checks.extend(self._check_drone())

        model_ready = False
        try:
            self.detector.validate_target_classes()
        except Exception as exc:
            checks.append(self._fail("vision.model", f"model/class validation failed: {exc}"))
        else:
            model_ready = True
            checks.append(self._pass("vision.model", "model loaded and target classes are valid"))

        frame: Any | None = None
        frame_ready = False
        try:
            frame = self.camera.read()
            self._validate_frame(frame)
        except Exception as exc:
            checks.append(self._fail("camera.frame", f"camera frame unavailable or invalid: {exc}"))
        else:
            frame_ready = True
            height, width = frame.shape[:2]
            checks.append(self._pass("camera.frame", f"received {width}x{height} BGR frame"))

        if model_ready and frame_ready:
            try:
                detections = self.detector.detect(frame)
            except Exception as exc:
                checks.append(self._fail("vision.inference", f"detector warm-up failed: {exc}"))
            else:
                checks.append(
                    self._pass(
                        "vision.inference",
                        f"warm-up inference succeeded ({len(detections)} filtered detections)",
                    )
                )
        else:
            checks.append(
                ReadinessCheck(
                    name="vision.inference",
                    status=CheckStatus.SKIP,
                    detail="warm-up skipped because model or frame validation failed",
                )
            )

        return ReadinessReport(checks=tuple(checks))

    def _check_drone(self) -> list[ReadinessCheck]:
        try:
            state = self.drone.get_state()
        except Exception as exc:
            return [self._fail("drone.telemetry", f"state telemetry unavailable: {exc}")]

        checks = [
            (
                self._pass("drone.connection", "controller reports connected")
                if state.is_connected
                else self._fail("drone.connection", "controller reports disconnected")
            ),
            (
                self._fail("drone.flight_state", "drone is already flying")
                if state.is_flying
                else self._pass("drone.flight_state", "drone reports landed")
            ),
        ]

        battery = state.battery_percent
        minimum = self.flight_config.min_takeoff_battery_percent
        if battery is None:
            checks.append(self._fail("drone.battery", "battery telemetry is unavailable"))
        elif not 0 <= battery <= 100:
            checks.append(self._fail("drone.battery", f"invalid battery telemetry: {battery}%"))
        elif battery < minimum:
            checks.append(
                self._fail("drone.battery", f"battery is {battery}%; at least {minimum}% is required")
            )
        else:
            checks.append(self._pass("drone.battery", f"battery is {battery}%"))

        altitude = state.altitude_m
        yaw = state.yaw_degrees
        telemetry_valid = (
            altitude is not None
            and math.isfinite(altitude)
            and self.flight_config.min_altitude_m <= altitude <= self.flight_config.max_altitude_m
            and math.isfinite(yaw)
        )
        if (
            telemetry_valid
            and not state.is_flying
            and altitude is not None
            and altitude > self.preflight_config.max_landed_altitude_m
        ):
            telemetry_valid = False
        if telemetry_valid:
            checks.append(
                self._pass(
                    "drone.telemetry",
                    f"altitude={altitude:.2f}m yaw={yaw % 360.0:.1f}deg",
                )
            )
        else:
            checks.append(
                self._fail(
                    "drone.telemetry",
                    "altitude/yaw telemetry is unavailable, non-finite, outside configured limits, "
                    "or inconsistent with a landed aircraft",
                )
            )
        return checks

    def _validate_frame(self, frame: Any) -> None:
        if frame is None:
            raise ValueError("frame is None")
        shape = getattr(frame, "shape", None)
        if not isinstance(shape, tuple) or len(shape) < 2:
            raise ValueError("frame does not expose an image shape")
        height, width = shape[:2]
        if not isinstance(height, int) or not isinstance(width, int):
            raise ValueError("frame dimensions are not integers")
        if height < self.preflight_config.min_frame_height:
            raise ValueError(
                f"frame height {height} is below {self.preflight_config.min_frame_height}"
            )
        if width < self.preflight_config.min_frame_width:
            raise ValueError(
                f"frame width {width} is below {self.preflight_config.min_frame_width}"
            )
        if self.preflight_config.require_three_channel_bgr and (
            len(shape) != 3 or shape[2] != 3
        ):
            raise ValueError("frame must have exactly three BGR channels")

    @staticmethod
    def _pass(name: str, detail: str) -> ReadinessCheck:
        return ReadinessCheck(name=name, status=CheckStatus.PASS, detail=detail)

    @staticmethod
    def _fail(name: str, detail: str) -> ReadinessCheck:
        return ReadinessCheck(name=name, status=CheckStatus.FAIL, detail=detail)
