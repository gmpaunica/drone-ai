"""Flight-controller interface and state model."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


class DroneControllerError(RuntimeError):
    """Raised when a flight adapter cannot safely complete an operation."""


@dataclass(frozen=True, slots=True)
class DroneState:
    x_m: float | None
    y_m: float | None
    altitude_m: float | None
    yaw_degrees: float
    is_flying: bool
    is_connected: bool
    battery_percent: int | None
    speed_mps: float


class DroneController(ABC):
    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def disconnect(self) -> None: ...

    @abstractmethod
    def takeoff(self) -> None: ...

    @abstractmethod
    def land(self) -> None: ...

    @abstractmethod
    def emergency_stop(self) -> None: ...

    @abstractmethod
    def hover(self) -> None: ...

    @abstractmethod
    def move_forward(self, distance_m: float) -> None: ...

    @abstractmethod
    def move_backward(self, distance_m: float) -> None: ...

    @abstractmethod
    def move_left(self, distance_m: float) -> None: ...

    @abstractmethod
    def move_right(self, distance_m: float) -> None: ...

    @abstractmethod
    def move_up(self, distance_m: float) -> None: ...

    @abstractmethod
    def move_down(self, distance_m: float) -> None: ...

    @abstractmethod
    def rotate_clockwise(self, degrees: float) -> None: ...

    @abstractmethod
    def rotate_counter_clockwise(self, degrees: float) -> None: ...

    @abstractmethod
    def get_state(self) -> DroneState: ...
