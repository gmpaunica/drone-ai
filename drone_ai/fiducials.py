"""Offline ArUco detection and calibrated metric camera-pose estimation.

Transforms use column vectors and are named ``source_to_destination``. Marker
coordinates are OpenCV's square-marker convention: +x right, +y up, and +z
out of the printed front face. Camera coordinates are +x right, +y down, and
+z forward. No pose from this module is connected to a drone controller.
"""

from __future__ import annotations

from enum import Enum
import math
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from drone_ai.calibration import CameraCalibration
from drone_ai.config import FiducialConfig


class FiducialError(RuntimeError):
    pass


class ArucoDictionary(str, Enum):
    DICT_4X4_50 = "DICT_4X4_50"
    DICT_4X4_100 = "DICT_4X4_100"
    DICT_4X4_250 = "DICT_4X4_250"
    DICT_4X4_1000 = "DICT_4X4_1000"
    DICT_5X5_50 = "DICT_5X5_50"
    DICT_5X5_100 = "DICT_5X5_100"
    DICT_5X5_250 = "DICT_5X5_250"
    DICT_5X5_1000 = "DICT_5X5_1000"
    DICT_6X6_50 = "DICT_6X6_50"
    DICT_6X6_100 = "DICT_6X6_100"
    DICT_6X6_250 = "DICT_6X6_250"
    DICT_6X6_1000 = "DICT_6X6_1000"
    DICT_7X7_50 = "DICT_7X7_50"
    DICT_7X7_100 = "DICT_7X7_100"
    DICT_7X7_250 = "DICT_7X7_250"
    DICT_7X7_1000 = "DICT_7X7_1000"
    DICT_ARUCO_ORIGINAL = "DICT_ARUCO_ORIGINAL"

    @property
    def capacity(self) -> int:
        if self is ArucoDictionary.DICT_ARUCO_ORIGINAL:
            return 1024
        return int(self.value.rsplit("_", 1)[1])


Matrix4 = tuple[
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
]


def _validate_rigid_transform(value: Matrix4) -> Matrix4:
    if not all(math.isfinite(item) for row in value for item in row):
        raise ValueError("transform values must be finite")
    if any(abs(value[3][index]) > 1e-6 for index in range(3)) or not math.isclose(
        value[3][3], 1.0, abs_tol=1e-6
    ):
        raise ValueError("transform bottom row must be [0, 0, 0, 1]")
    rotation = [row[:3] for row in value[:3]]
    for left_index, left in enumerate(rotation):
        for right_index, right in enumerate(rotation):
            dot = sum(a * b for a, b in zip(left, right))
            expected = 1.0 if left_index == right_index else 0.0
            if not math.isclose(dot, expected, abs_tol=1e-4):
                raise ValueError("transform rotation must be orthonormal")
    determinant = (
        rotation[0][0]
        * (rotation[1][1] * rotation[2][2] - rotation[1][2] * rotation[2][1])
        - rotation[0][1]
        * (rotation[1][0] * rotation[2][2] - rotation[1][2] * rotation[2][0])
        + rotation[0][2]
        * (rotation[1][0] * rotation[2][1] - rotation[1][1] * rotation[2][0])
    )
    if not math.isclose(determinant, 1.0, abs_tol=1e-4):
        raise ValueError("transform rotation must be right-handed with determinant 1")
    return value


class FiducialMarker(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    marker_id: int = Field(ge=0)
    size_m: float = Field(ge=0.03, le=1.0)
    marker_to_world: Matrix4

    @field_validator("size_m")
    @classmethod
    def finite_size(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("marker size must be finite")
        return value

    @field_validator("marker_to_world")
    @classmethod
    def rigid_marker_transform(cls, value: Matrix4) -> Matrix4:
        return _validate_rigid_transform(value)


class FiducialMap(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    name: str = Field(min_length=1, max_length=100)
    dictionary: ArucoDictionary = ArucoDictionary.DICT_4X4_50
    markers: tuple[FiducialMarker, ...] = Field(min_length=1, max_length=250)

    @model_validator(mode="after")
    def validate_marker_ids(self) -> "FiducialMap":
        ids = [marker.marker_id for marker in self.markers]
        if len(ids) != len(set(ids)):
            raise ValueError("fiducial marker IDs must be unique")
        invalid = [marker_id for marker_id in ids if marker_id >= self.dictionary.capacity]
        if invalid:
            raise ValueError(
                f"marker ID {invalid[0]} is outside {self.dictionary.value} capacity"
            )
        return self


class FiducialObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    marker_id: int = Field(ge=0)
    corners_px: tuple[
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
    ]
    reprojection_error_px: float = Field(ge=0.0)
    distance_m: float = Field(gt=0.0)
    marker_to_camera: Matrix4
    camera_to_world: Matrix4

    @field_validator("corners_px")
    @classmethod
    def finite_corners(
        cls,
        value: tuple[
            tuple[float, float],
            tuple[float, float],
            tuple[float, float],
            tuple[float, float],
        ],
    ) -> tuple[
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
    ]:
        if not all(math.isfinite(item) for point in value for item in point):
            raise ValueError("marker corners must be finite")
        return value

    @field_validator("reprojection_error_px", "distance_m")
    @classmethod
    def finite_measurement(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("pose measurements must be finite")
        return value

    @field_validator("marker_to_camera", "camera_to_world")
    @classmethod
    def rigid_pose_transform(cls, value: Matrix4) -> Matrix4:
        return _validate_rigid_transform(value)


class FiducialFrame(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    frame_index: int = Field(ge=0)
    detected_ids: tuple[int, ...] = ()
    unmapped_ids: tuple[int, ...] = ()
    rejected_mapped_count: int = Field(ge=0)
    observations: tuple[FiducialObservation, ...] = ()
    pose_consistent: bool = False
    maximum_translation_disagreement_m: float | None = Field(default=None, ge=0.0)
    maximum_rotation_disagreement_degrees: float | None = Field(default=None, ge=0.0)
    best_camera_to_world: Matrix4 | None = None
    metric_pose_available: bool = False
    offline_only: bool = True

    @model_validator(mode="after")
    def enforce_pose_and_flight_boundaries(self) -> "FiducialFrame":
        has_observations = bool(self.observations)
        available = has_observations and self.pose_consistent
        if self.metric_pose_available != available:
            raise ValueError(
                "metric_pose_available requires accepted, mutually consistent observations"
            )
        if (self.best_camera_to_world is not None) != available:
            raise ValueError("best camera pose must exist exactly when a consistent pose is available")
        disagreement_values = (
            self.maximum_translation_disagreement_m,
            self.maximum_rotation_disagreement_degrees,
        )
        if has_observations:
            if any(value is None or not math.isfinite(value) for value in disagreement_values):
                raise ValueError("pose disagreement metrics are required for observations")
        elif self.pose_consistent or any(value is not None for value in disagreement_values):
            raise ValueError("empty observations cannot claim pose consistency")
        if not self.offline_only:
            raise ValueError("fiducial frame records must remain offline-only")
        return self


class FiducialSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    total_frames: int = Field(ge=0)
    localized_frames: int = Field(ge=0)
    accepted_observations: int = Field(ge=0)
    rejected_mapped_observations: int = Field(ge=0)
    inconsistent_frames: int = Field(ge=0)
    offline_only: bool = True

    @field_validator("offline_only")
    @classmethod
    def preserve_offline_boundary(cls, value: bool) -> bool:
        if not value:
            raise ValueError("fiducial analysis must remain offline-only")
        return value


def load_fiducial_map(path: str | Path) -> FiducialMap:
    source = Path(path)
    if not source.is_file():
        raise FiducialError(f"fiducial map does not exist: {source}")
    try:
        return FiducialMap.model_validate_json(source.read_text(encoding="utf-8"))
    except Exception as exc:
        raise FiducialError(f"invalid fiducial map {source}: {exc}") from exc


def save_printable_marker_svg(
    dictionary: ArucoDictionary,
    marker_id: int,
    marker_size_m: float,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Write an A4 portrait SVG whose black marker square has an exact metric size."""

    if isinstance(marker_id, bool) or not isinstance(marker_id, int):
        raise FiducialError("marker ID must be an integer")
    if marker_id < 0 or marker_id >= dictionary.capacity:
        raise FiducialError(
            f"marker ID must be within 0-{dictionary.capacity - 1} for {dictionary.value}"
        )
    if (
        isinstance(marker_size_m, bool)
        or not isinstance(marker_size_m, (int, float))
        or not math.isfinite(float(marker_size_m))
        or not 0.03 <= float(marker_size_m) <= 0.16
    ):
        raise FiducialError("printable marker size must be within 0.03-0.16m")
    output = Path(path)
    if output.suffix.lower() != ".svg":
        raise FiducialError("printable marker output must use the .svg extension")
    if output.exists() and not overwrite:
        raise FiducialError(f"marker output already exists: {output}")
    try:
        import cv2
    except ImportError as exc:
        raise FiducialError("OpenCV with ArUco support is required") from exc
    aruco = getattr(cv2, "aruco", None)
    dictionary_code = getattr(aruco, dictionary.value, None) if aruco is not None else None
    if dictionary_code is None:
        raise FiducialError(f"OpenCV does not support {dictionary.value}")
    cv_dictionary = aruco.getPredefinedDictionary(dictionary_code)
    cell_count = int(cv_dictionary.markerSize) + 2
    pixels = aruco.generateImageMarker(cv_dictionary, marker_id, cell_count)

    marker_mm = float(marker_size_m) * 1000.0
    marker_x = (210.0 - marker_mm) / 2.0
    marker_y = 40.0
    cell_mm = marker_mm / cell_count
    lines = [
        '<?xml version="1.0" encoding="UTF-8" standalone="no"?>',
        '<svg xmlns="http://www.w3.org/2000/svg" width="210mm" height="297mm" '
        'viewBox="0 0 210 297">',
        '  <rect width="210" height="297" fill="white"/>',
        f'  <!-- {dictionary.value} marker ID {marker_id}; black square is '
        f'{_format_decimal(marker_mm)} mm. -->',
        f'  <rect x="{_format_decimal(marker_x)}" y="{_format_decimal(marker_y)}" '
        f'width="{_format_decimal(marker_mm)}" height="{_format_decimal(marker_mm)}" '
        'fill="black"/>',
        '  <g fill="white">',
    ]
    for row in range(cell_count):
        for column in range(cell_count):
            if int(pixels[row, column]) != 255:
                continue
            x = marker_x + column * cell_mm
            y = marker_y + row * cell_mm
            lines.append(
                f'    <rect x="{_format_decimal(x)}" y="{_format_decimal(y)}" '
                f'width="{_format_decimal(cell_mm)}" height="{_format_decimal(cell_mm)}"/>'
            )
    label_y = marker_y + marker_mm + 24.0
    lines.extend(
        [
            '  </g>',
            '  <g font-family="Arial, sans-serif" fill="black" text-anchor="middle">',
            f'    <text x="105" y="{_format_decimal(label_y)}" font-size="7" '
            f'font-weight="bold">Drone AI fiducial - {dictionary.value}, ID {marker_id}</text>',
            f'    <text x="105" y="{_format_decimal(label_y + 12.0)}" font-size="5">'
            'Print A4 portrait at 100% / actual size. Do not fit to page.</text>',
            f'    <text x="105" y="{_format_decimal(label_y + 22.0)}" font-size="5">'
            f'Verify the black square is exactly {_format_decimal(marker_mm)} mm on every side.</text>',
            f'    <text x="105" y="{_format_decimal(label_y + 32.0)}" font-size="5">'
            'Mount flat and rigid; keep the surrounding white margin clear.</text>',
            '  </g>',
            f'  <line x1="{_format_decimal(marker_x)}" y1="270" '
            f'x2="{_format_decimal(marker_x + marker_mm)}" y2="270" '
            'stroke="black" stroke-width="0.5"/>',
            f'  <line x1="{_format_decimal(marker_x)}" y1="267" '
            f'x2="{_format_decimal(marker_x)}" y2="273" stroke="black" stroke-width="0.5"/>',
            f'  <line x1="{_format_decimal(marker_x + marker_mm)}" y1="267" '
            f'x2="{_format_decimal(marker_x + marker_mm)}" y2="273" '
            'stroke="black" stroke-width="0.5"/>',
            f'  <text x="105" y="282" font-family="Arial, sans-serif" font-size="5" '
            f'text-anchor="middle">{_format_decimal(marker_mm)} mm verification line</text>',
            '</svg>',
            '',
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


class FiducialPoseAnalyzer:
    def __init__(
        self,
        calibration: CameraCalibration,
        fiducial_map: FiducialMap,
        config: FiducialConfig | None = None,
    ) -> None:
        self.calibration = calibration
        self.fiducial_map = fiducial_map
        self.config = config or FiducialConfig()
        if not 0.1 <= self.config.maximum_reprojection_error_px <= 5.0:
            raise ValueError("maximum reprojection error must be within 0.1-5px")
        if not 8.0 <= self.config.minimum_marker_side_px <= 200.0:
            raise ValueError("minimum marker side must be within 8-200px")
        try:
            import cv2
        except ImportError as exc:
            raise FiducialError("OpenCV with ArUco support is required") from exc
        aruco = getattr(cv2, "aruco", None)
        if aruco is None or not hasattr(aruco, "ArucoDetector"):
            raise FiducialError("installed OpenCV does not include the ArUco detector")
        dictionary_code = getattr(aruco, self.fiducial_map.dictionary.value, None)
        if dictionary_code is None:
            raise FiducialError(
                f"OpenCV does not support {self.fiducial_map.dictionary.value}"
            )
        parameters = aruco.DetectorParameters()
        parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
        self._detector = aruco.ArucoDetector(
            aruco.getPredefinedDictionary(dictionary_code),
            parameters,
        )
        self._markers = {marker.marker_id: marker for marker in fiducial_map.markers}
        self._frame_index = 0
        self._localized_frames = 0
        self._accepted_observations = 0
        self._rejected_mapped_observations = 0
        self._inconsistent_frames = 0

    def process(self, frame: Any) -> FiducialFrame:
        try:
            import cv2
            import numpy as np
        except ImportError as exc:
            raise FiducialError("OpenCV and NumPy are required for fiducial analysis") from exc
        shape = getattr(frame, "shape", None)
        if not isinstance(shape, tuple) or len(shape) != 3 or shape[2] != 3:
            raise FiducialError("fiducial frames must have exactly three BGR channels")
        height, width = shape[:2]
        expected = (self.calibration.image_width, self.calibration.image_height)
        if (width, height) != expected:
            raise FiducialError(
                f"calibration is {expected[0]}x{expected[1]} but frame is {width}x{height}"
            )
        frame_index = self._frame_index
        self._frame_index += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected_corners, detected_ids, _ = self._detector.detectMarkers(gray)
        if detected_ids is None:
            detected_pairs: list[tuple[int, Any]] = []
        else:
            detected_pairs = sorted(
                (
                    (int(marker_id), np.asarray(corners, dtype=np.float64).reshape(4, 2))
                    for marker_id, corners in zip(detected_ids.reshape(-1), detected_corners)
                ),
                key=lambda item: item[0],
            )
        all_ids = tuple(marker_id for marker_id, _ in detected_pairs)
        unmapped_ids = tuple(
            marker_id for marker_id, _ in detected_pairs if marker_id not in self._markers
        )
        observations: list[FiducialObservation] = []
        rejected = 0
        for marker_id, corners in detected_pairs:
            marker = self._markers.get(marker_id)
            if marker is None:
                continue
            observation = self._estimate_pose(marker, corners)
            if observation is None:
                rejected += 1
            else:
                observations.append(observation)
        observations.sort(key=lambda item: (item.reprojection_error_px, item.marker_id))
        pose_consistent, translation_disagreement, rotation_disagreement = (
            self._pose_consensus(observations)
        )
        pose_available = bool(observations) and pose_consistent
        best_pose = observations[0].camera_to_world if pose_available else None
        self._accepted_observations += len(observations)
        self._rejected_mapped_observations += rejected
        if pose_available:
            self._localized_frames += 1
        elif observations:
            self._inconsistent_frames += 1
        return FiducialFrame(
            frame_index=frame_index,
            detected_ids=all_ids,
            unmapped_ids=unmapped_ids,
            rejected_mapped_count=rejected,
            observations=tuple(observations),
            pose_consistent=pose_consistent,
            maximum_translation_disagreement_m=translation_disagreement,
            maximum_rotation_disagreement_degrees=rotation_disagreement,
            best_camera_to_world=best_pose,
            metric_pose_available=pose_available,
            offline_only=True,
        )

    def _pose_consensus(
        self,
        observations: list[FiducialObservation],
    ) -> tuple[bool, float | None, float | None]:
        import numpy as np

        if not observations:
            return False, None, None
        if len(observations) == 1:
            return True, 0.0, 0.0
        transforms = [
            np.asarray(observation.camera_to_world, dtype=np.float64)
            for observation in observations
        ]
        maximum_translation = 0.0
        maximum_rotation = 0.0
        for left_index, left in enumerate(transforms[:-1]):
            for right in transforms[left_index + 1 :]:
                translation = float(np.linalg.norm(left[:3, 3] - right[:3, 3]))
                relative_rotation = left[:3, :3].T @ right[:3, :3]
                cosine = float((np.trace(relative_rotation) - 1.0) / 2.0)
                angle = math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
                maximum_translation = max(maximum_translation, translation)
                maximum_rotation = max(maximum_rotation, angle)
        consistent = (
            maximum_translation <= self.config.maximum_pose_disagreement_m
            and maximum_rotation <= self.config.maximum_pose_disagreement_degrees
        )
        return consistent, maximum_translation, maximum_rotation

    def _estimate_pose(
        self,
        marker: FiducialMarker,
        corners: Any,
    ) -> FiducialObservation | None:
        import cv2
        import numpy as np

        edge_lengths = np.linalg.norm(corners - np.roll(corners, -1, axis=0), axis=1)
        if float(edge_lengths.min()) < self.config.minimum_marker_side_px:
            return None
        half = marker.size_m / 2.0
        object_points = np.asarray(
            [
                (-half, half, 0.0),
                (half, half, 0.0),
                (half, -half, 0.0),
                (-half, -half, 0.0),
            ],
            dtype=np.float64,
        )
        camera_matrix = np.asarray(self.calibration.camera_matrix, dtype=np.float64)
        distortion = np.asarray(
            self.calibration.distortion_coefficients,
            dtype=np.float64,
        )
        try:
            result = cv2.solvePnPGeneric(
                object_points,
                corners,
                camera_matrix,
                distortion,
                flags=cv2.SOLVEPNP_IPPE_SQUARE,
            )
        except Exception as exc:
            raise FiducialError(f"OpenCV marker pose estimation failed: {exc}") from exc
        if not result[0]:
            return None
        candidates: list[tuple[float, Any, Any]] = []
        for rotation_vector, translation_vector in zip(result[1], result[2]):
            translation = np.asarray(translation_vector, dtype=np.float64).reshape(3)
            if not np.all(np.isfinite(translation)) or translation[2] <= 0.0:
                continue
            projected, _ = cv2.projectPoints(
                object_points,
                rotation_vector,
                translation_vector,
                camera_matrix,
                distortion,
            )
            residual = projected.reshape(4, 2) - corners
            error = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
            if math.isfinite(error):
                candidates.append((error, rotation_vector, translation_vector))
        if not candidates:
            return None
        error, rotation_vector, translation_vector = min(candidates, key=lambda item: item[0])
        if error > self.config.maximum_reprojection_error_px:
            return None
        rotation, _ = cv2.Rodrigues(rotation_vector)
        marker_to_camera = np.eye(4, dtype=np.float64)
        marker_to_camera[:3, :3] = rotation
        marker_to_camera[:3, 3] = np.asarray(translation_vector).reshape(3)
        camera_to_marker = np.eye(4, dtype=np.float64)
        camera_to_marker[:3, :3] = rotation.T
        camera_to_marker[:3, 3] = -rotation.T @ marker_to_camera[:3, 3]
        marker_to_world = np.asarray(marker.marker_to_world, dtype=np.float64)
        camera_to_world = marker_to_world @ camera_to_marker
        distance = float(np.linalg.norm(marker_to_camera[:3, 3]))
        return FiducialObservation(
            marker_id=marker.marker_id,
            corners_px=tuple(
                (float(point[0]), float(point[1])) for point in corners
            ),  # type: ignore[arg-type]
            reprojection_error_px=error,
            distance_m=distance,
            marker_to_camera=_matrix_tuple(marker_to_camera),
            camera_to_world=_matrix_tuple(camera_to_world),
        )

    def draw(self, frame: Any, record: FiducialFrame) -> Any:
        import cv2
        import numpy as np

        camera_matrix = np.asarray(self.calibration.camera_matrix, dtype=np.float64)
        distortion = np.asarray(
            self.calibration.distortion_coefficients,
            dtype=np.float64,
        )
        for observation in record.observations:
            corners = np.asarray(observation.corners_px, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(frame, [corners], True, (0, 255, 0), 2)
            origin = tuple(int(round(value)) for value in observation.corners_px[0])
            cv2.putText(
                frame,
                f"id={observation.marker_id} d={observation.distance_m:.2f}m",
                origin,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )
            transform = np.asarray(observation.marker_to_camera, dtype=np.float64)
            rotation_vector, _ = cv2.Rodrigues(transform[:3, :3])
            marker = self._markers[observation.marker_id]
            cv2.drawFrameAxes(
                frame,
                camera_matrix,
                distortion,
                rotation_vector,
                transform[:3, 3],
                marker.size_m * 0.5,
            )
        status = (
            f"frame={record.frame_index} detected={len(record.detected_ids)} "
            f"consistent={record.pose_consistent} "
            f"metric_pose={record.metric_pose_available} offline_only=True"
        )
        cv2.putText(
            frame,
            status,
            (12, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
        )
        return frame

    def summary(self) -> FiducialSummary:
        return FiducialSummary(
            total_frames=self._frame_index,
            localized_frames=self._localized_frames,
            accepted_observations=self._accepted_observations,
            rejected_mapped_observations=self._rejected_mapped_observations,
            inconsistent_frames=self._inconsistent_frames,
            offline_only=True,
        )


def _matrix_tuple(matrix: Any) -> Matrix4:
    return tuple(
        tuple(float(value) for value in row)
        for row in matrix.tolist()
    )  # type: ignore[return-value]


def _format_decimal(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".")
