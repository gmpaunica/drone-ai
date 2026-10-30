"""Frame-source abstraction. Frames use OpenCV's BGR channel order."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class CameraError(RuntimeError):
    pass


class CameraSource(ABC):
    @abstractmethod
    def open(self) -> None: ...

    @abstractmethod
    def read(self) -> Any: ...

    @abstractmethod
    def close(self) -> None: ...
