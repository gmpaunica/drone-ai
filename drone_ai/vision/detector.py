"""Lazy Ultralytics YOLO adapter producing project-owned detections."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from drone_ai.config import VisionConfig
from drone_ai.vision.models import Detection


class VisionError(RuntimeError):
    pass


class YOLODetector:
    def __init__(
        self,
        model_name: str | None = None,
        *,
        confidence: float | None = None,
        target_classes: Iterable[str] | None = None,
        model: Any | None = None,
        config: VisionConfig | None = None,
    ) -> None:
        defaults = config or VisionConfig()
        self.model_name = model_name or defaults.model_name
        self.confidence = defaults.confidence if confidence is None else confidence
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        classes = defaults.target_classes if target_classes is None else target_classes
        self.target_classes = frozenset(item.lower() for item in classes)
        self._model = model

    @property
    def model(self) -> Any:
        if self._model is None:
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise VisionError("Ultralytics is not installed; install requirements.txt") from exc
            try:
                # Ultralytics downloads named weights on the first live run.
                self._model = YOLO(self.model_name)
            except Exception as exc:
                raise VisionError(f"could not load YOLO model {self.model_name!r}: {exc}") from exc
        return self._model

    def detect(self, frame: Any) -> list[Detection]:
        try:
            results = self.model(frame, conf=self.confidence, verbose=False)
        except Exception as exc:
            raise VisionError(f"YOLO inference failed: {exc}") from exc
        detections: list[Detection] = []
        for result in results:
            names = getattr(result, "names", None) or getattr(self.model, "names", {})
            detections.extend(self.convert_result(result, names))
        return detections

    def validate_target_classes(self) -> None:
        """Fail before flight when configured targets are absent from the model."""

        names = getattr(self.model, "names", None)
        if isinstance(names, dict):
            available = {str(value).lower() for value in names.values()}
        elif isinstance(names, (list, tuple)):
            available = {str(value).lower() for value in names}
        else:
            raise VisionError("YOLO model did not expose a class-name mapping")
        unknown = sorted(self.target_classes - available)
        if unknown:
            raise VisionError(
                "target class(es) not supported by this YOLO model: " + ", ".join(unknown)
            )

    def convert_result(self, result: Any, names: dict[int, str] | list[str]) -> list[Detection]:
        detections: list[Detection] = []
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return detections
        for box in boxes:
            class_id = int(_scalar(box.cls))
            try:
                class_name = str(names[class_id])
            except (KeyError, IndexError, TypeError) as exc:
                raise VisionError(f"model returned unknown class id {class_id}") from exc
            confidence = float(_scalar(box.conf))
            if confidence < self.confidence or (
                self.target_classes and class_name.lower() not in self.target_classes
            ):
                continue
            coordinates = _flat_values(box.xyxy)
            if len(coordinates) != 4:
                raise VisionError("YOLO bounding box did not contain four coordinates")
            try:
                detections.append(
                    Detection(
                        class_name=class_name,
                        confidence=confidence,
                        bbox=tuple(float(value) for value in coordinates),  # type: ignore[arg-type]
                    )
                )
            except ValueError as exc:
                raise VisionError(f"invalid YOLO detection: {exc}") from exc
        return detections


def draw_detections(frame: Any, detections: Iterable[Detection]) -> Any:
    """Draw project-owned boxes without retaining Ultralytics result objects."""

    try:
        import cv2
    except ImportError as exc:
        raise VisionError("OpenCV is not installed; install requirements.txt") from exc
    for detection in detections:
        x1, y1, x2, y2 = (int(round(value)) for value in detection.bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
        label = f"{detection.class_name} {detection.confidence:.2f}"
        cv2.putText(frame, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 0), 2)
    return frame


def _scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            pass
    values = _flat_values(value)
    if len(values) != 1:
        raise VisionError("expected a scalar model value")
    return values[0]


def _flat_values(value: Any) -> list[Any]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    while isinstance(value, (list, tuple)) and len(value) == 1 and isinstance(value[0], (list, tuple)):
        value = value[0]
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]
