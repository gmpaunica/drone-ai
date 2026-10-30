"""Deterministic in-memory drone with yaw-aware movement."""

from __future__ import annotations

import logging
import math

from drone_ai.config import FlightConfig
from drone_ai.controllers.base import DroneController, DroneControllerError, DroneState


LOGGER = logging.getLogger(__name__)


class SimulatedDrone(DroneController):
    def __init__(
        self,
        config: FlightConfig | None = None,
        *,
        initial_x_m: float = 0.0,
        initial_y_m: float = 0.0,
        initial_yaw_degrees: float = 0.0,
    ) -> None:
        self.config = config or FlightConfig()
        if not 0 < self.config.cruise_speed_mps <= self.config.max_speed_mps:
            raise ValueError("cruise speed must be positive and within the hard maximum")
        if not math.isfinite(initial_x_m) or not math.isfinite(initial_y_m):
            raise ValueError("initial simulator position must be finite")
        self._x_m = initial_x_m
        self._y_m = initial_y_m
        self._altitude_m = 0.0
        if not math.isfinite(initial_yaw_degrees):
            raise ValueError("initial simulator yaw must be finite")
        self._yaw_degrees = initial_yaw_degrees % 360.0
        self._connected = False
        self._flying = False
        self._emergency_stopped = False

    def connect(self) -> None:
        self._connected = True
        self._emergency_stopped = False
        LOGGER.info("[SIM] Connected")

    def disconnect(self) -> None:
        if self._flying:
            raise DroneControllerError("cannot disconnect while flying")
        self._connected = False
        LOGGER.info("[SIM] Disconnected")

    def takeoff(self) -> None:
        self._require_connected()
        if self._emergency_stopped:
            raise DroneControllerError("reconnect after an emergency stop before takeoff")
        if self._flying:
            raise DroneControllerError("drone is already flying")
        self._flying = True
        self._altitude_m = self.config.takeoff_altitude_m
        LOGGER.info("[SIM] Takeoff to %.2f m", self._altitude_m)

    def land(self) -> None:
        self._require_connected()
        if not self._flying:
            raise DroneControllerError("drone is already landed")
        self._altitude_m = 0.0
        self._flying = False
        LOGGER.info("[SIM] Landed")

    def emergency_stop(self) -> None:
        # This path deliberately does not require a normal connected/flying state.
        self._altitude_m = 0.0
        self._flying = False
        self._emergency_stopped = True
        LOGGER.warning("[SIM] EMERGENCY STOP")

    def hover(self) -> None:
        self._require_flying()
        LOGGER.info("[SIM] Hover")

    def move_forward(self, distance_m: float) -> None:
        self._move_planar(distance_m, forward=True)

    def move_backward(self, distance_m: float) -> None:
        self._move_planar(-distance_m, forward=True)

    def move_left(self, distance_m: float) -> None:
        self._move_planar(-distance_m, forward=False)

    def move_right(self, distance_m: float) -> None:
        self._move_planar(distance_m, forward=False)

    def move_up(self, distance_m: float) -> None:
        self._require_flying()
        self._validate_distance(distance_m)
        new_altitude = self._altitude_m + distance_m
        self._require_safe_altitude(new_altitude)
        self._altitude_m = new_altitude
        LOGGER.info("[SIM] Move up %.2f m", distance_m)

    def move_down(self, distance_m: float) -> None:
        self._require_flying()
        self._validate_distance(distance_m)
        new_altitude = self._altitude_m - distance_m
        self._require_safe_altitude(new_altitude)
        self._altitude_m = new_altitude
        LOGGER.info("[SIM] Move down %.2f m", distance_m)

    def rotate_clockwise(self, degrees: float) -> None:
        self._rotate(degrees)

    def rotate_counter_clockwise(self, degrees: float) -> None:
        self._rotate(-degrees)

    def get_state(self) -> DroneState:
        return DroneState(
            x_m=self._x_m,
            y_m=self._y_m,
            altitude_m=self._altitude_m,
            yaw_degrees=self._yaw_degrees % 360.0,
            is_flying=self._flying,
            is_connected=self._connected,
            battery_percent=100,
            speed_mps=self.config.cruise_speed_mps,
        )

    def _move_planar(self, signed_distance_m: float, *, forward: bool) -> None:
        self._require_flying()
        self._validate_distance(abs(signed_distance_m))
        yaw = math.radians(self._yaw_degrees)
        if forward:
            self._x_m += signed_distance_m * math.sin(yaw)
            self._y_m += signed_distance_m * math.cos(yaw)
            direction = "forward" if signed_distance_m >= 0 else "backward"
        else:
            self._x_m += signed_distance_m * math.cos(yaw)
            self._y_m -= signed_distance_m * math.sin(yaw)
            direction = "right" if signed_distance_m >= 0 else "left"
        LOGGER.info("[SIM] Move %s %.2f m", direction, abs(signed_distance_m))

    def _rotate(self, signed_degrees: float) -> None:
        self._require_flying()
        if not math.isfinite(signed_degrees) or signed_degrees == 0:
            raise DroneControllerError("rotation must be finite and non-zero")
        self._yaw_degrees = (self._yaw_degrees + signed_degrees) % 360.0
        direction = "clockwise" if signed_degrees > 0 else "counter-clockwise"
        LOGGER.info("[SIM] Rotate %s %.1f degrees", direction, abs(signed_degrees))

    def _require_connected(self) -> None:
        if not self._connected:
            raise DroneControllerError("drone is not connected")

    def _require_flying(self) -> None:
        self._require_connected()
        if not self._flying:
            raise DroneControllerError("drone is not flying")

    @staticmethod
    def _validate_distance(distance_m: float) -> None:
        if not math.isfinite(distance_m) or distance_m <= 0:
            raise DroneControllerError("distance must be finite and positive")

    def _require_safe_altitude(self, altitude_m: float) -> None:
        if not self.config.min_altitude_m <= altitude_m <= self.config.max_altitude_m:
            raise DroneControllerError(
                f"altitude {altitude_m:g}m violates configured simulator limits"
            )
