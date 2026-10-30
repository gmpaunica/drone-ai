"""Offline monocular visual odometry with explicit scale ambiguity."""

from __future__ import annotations

from enum import Enum
import math
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from drone_ai.calibration import CameraCalibration
from drone_ai.config import VisualOdometryConfig


class OdometryError(RuntimeError):
    pass


class OdometryStatus(str, Enum):
    INITIALIZED = "initialized"
    ESTIMATED = "estimated"
    INSUFFICIENT_FEATURES = "insufficient_features"


class OdometryFrame(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    frame_index: int = Field(ge=0)
    status: OdometryStatus
    previous_keypoints: int = Field(ge=0)
    current_keypoints: int = Field(ge=0)
    matches: int = Field(ge=0)
    inliers: int = Field(ge=0)
    rotation_matrix: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ] | None = None
    translation_direction: tuple[float, float, float] | None = None
    metric_scale_available: bool = False
    message: str

    @field_validator("metric_scale_available")
    @classmethod
    def metric_scale_must_remain_false(cls, value: bool) -> bool:
        if value:
            raise ValueError("monocular odometry cannot claim metric scale")
        return value


class OdometrySummary(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    total_frames: int = Field(ge=0)
    estimated_frames: int = Field(ge=0)
    insufficient_frames: int = Field(ge=0)
    metric_scale_available: bool = False

    @field_validator("metric_scale_available")
    @classmethod
    def metric_scale_must_remain_false(cls, value: bool) -> bool:
        if value:
            raise ValueError("monocular odometry cannot claim metric scale")
        return value


class VisualOdometryAnalyzer:
    def __init__(
        self,
        calibration: CameraCalibration,
        config: VisualOdometryConfig | None = None,
    ) -> None:
        self.calibration = calibration
        self.config = config or VisualOdometryConfig()
        if self.config.maximum_features < self.config.minimum_matches:
            raise ValueError("maximum features must cover minimum matches")
        if self.config.minimum_inliers > self.config.minimum_matches:
            raise ValueError("minimum inliers cannot exceed minimum matches")
        if not 0.0 < self.config.ratio_test < 1.0:
            raise ValueError("ratio test must be in (0, 1)")
        self._previous_points: Any | None = None
        self._previous_descriptors: Any | None = None
        self._frame_index = 0
        self._estimated_frames = 0
        self._insufficient_frames = 0

    def process(self, frame: Any) -> OdometryFrame:
        try:
            import cv2
            import numpy as np
        except ImportError as exc:
            raise OdometryError("OpenCV and NumPy are required for visual odometry") from exc
        shape = getattr(frame, "shape", None)
        if not isinstance(shape, tuple) or len(shape) != 3 or shape[2] != 3:
            raise OdometryError("odometry frames must have exactly three BGR channels")
        height, width = shape[:2]
        if (width, height) != (self.calibration.image_width, self.calibration.image_height):
            raise OdometryError(
                f"calibration is {self.calibration.image_width}x{self.calibration.image_height} "
                f"but frame is {width}x{height}"
            )

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detector = cv2.ORB_create(nfeatures=self.config.maximum_features)
        keypoints, descriptors = detector.detectAndCompute(gray, None)
        points = np.float32([item.pt for item in keypoints]).reshape(-1, 2)
        current_count = len(points)
        frame_index = self._frame_index
        self._frame_index += 1
        if self._previous_points is None:
            self._previous_points = points
            self._previous_descriptors = descriptors
            return OdometryFrame(
                frame_index=frame_index,
                status=OdometryStatus.INITIALIZED,
                previous_keypoints=0,
                current_keypoints=current_count,
                matches=0,
                inliers=0,
                message="stored first frame; metric scale is unavailable",
            )

        previous_points = self._previous_points
        previous_descriptors = self._previous_descriptors
        previous_count = len(previous_points)
        self._previous_points = points
        self._previous_descriptors = descriptors
        if previous_descriptors is None or descriptors is None:
            return self._insufficient(
                frame_index,
                previous_count,
                current_count,
                0,
                0,
                "ORB descriptors unavailable",
            )

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        pairs = matcher.knnMatch(previous_descriptors, descriptors, k=2)
        good = [
            pair[0]
            for pair in pairs
            if len(pair) == 2 and pair[0].distance < self.config.ratio_test * pair[1].distance
        ]
        if len(good) < self.config.minimum_matches:
            return self._insufficient(
                frame_index,
                previous_count,
                current_count,
                len(good),
                0,
                f"only {len(good)} ratio-test matches",
            )
        matched_previous = np.float32([previous_points[item.queryIdx] for item in good])
        matched_current = np.float32([points[item.trainIdx] for item in good])
        estimate = self.estimate_correspondences(
            matched_previous,
            matched_current,
            frame_index=frame_index,
            previous_keypoints=previous_count,
            current_keypoints=current_count,
        )
        if estimate.status is OdometryStatus.ESTIMATED:
            self._estimated_frames += 1
        return estimate

    def estimate_correspondences(
        self,
        previous_points: Any,
        current_points: Any,
        *,
        frame_index: int = 1,
        previous_keypoints: int | None = None,
        current_keypoints: int | None = None,
    ) -> OdometryFrame:
        import cv2
        import numpy as np

        previous = np.asarray(previous_points, dtype=np.float64)
        current = np.asarray(current_points, dtype=np.float64)
        if previous.ndim != 2 or previous.shape[1:] != (2,) or current.shape != previous.shape:
            raise OdometryError("correspondences must be matching Nx2 arrays")
        match_count = len(previous)
        previous_count = previous_keypoints if previous_keypoints is not None else match_count
        current_count = current_keypoints if current_keypoints is not None else match_count
        if match_count < self.config.minimum_matches:
            return self._insufficient(
                frame_index,
                previous_count,
                current_count,
                match_count,
                0,
                f"only {match_count} correspondences",
            )
        camera_matrix = np.asarray(self.calibration.camera_matrix, dtype=np.float64)
        essential, mask = cv2.findEssentialMat(
            previous,
            current,
            camera_matrix,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=self.config.ransac_threshold_px,
        )
        if essential is None:
            return self._insufficient(
                frame_index,
                previous_count,
                current_count,
                match_count,
                0,
                "essential matrix estimation failed",
            )
        candidates = (
            [essential]
            if essential.shape == (3, 3)
            else [essential[index : index + 3] for index in range(0, essential.shape[0], 3)]
        )
        best: tuple[int, Any, Any] | None = None
        for candidate in candidates:
            inliers, rotation, translation, _ = cv2.recoverPose(
                candidate,
                previous,
                current,
                camera_matrix,
                mask=mask,
            )
            count = int(inliers)
            if best is None or count > best[0]:
                best = (count, rotation, translation)
        assert best is not None
        inlier_count, rotation, translation = best
        if inlier_count < self.config.minimum_inliers:
            return self._insufficient(
                frame_index,
                previous_count,
                current_count,
                match_count,
                inlier_count,
                f"only {inlier_count} geometric inliers",
            )
        direction = translation.reshape(3).astype(float)
        norm = float(np.linalg.norm(direction))
        if not math.isfinite(norm) or norm <= 0:
            return self._insufficient(
                frame_index,
                previous_count,
                current_count,
                match_count,
                inlier_count,
                "translation direction was invalid",
            )
        direction /= norm
        return OdometryFrame(
            frame_index=frame_index,
            status=OdometryStatus.ESTIMATED,
            previous_keypoints=previous_count,
            current_keypoints=current_count,
            matches=match_count,
            inliers=inlier_count,
            rotation_matrix=tuple(
                tuple(float(item) for item in row) for row in rotation.tolist()
            ),  # type: ignore[arg-type]
            translation_direction=tuple(float(item) for item in direction),  # type: ignore[arg-type]
            metric_scale_available=False,
            message="relative pose estimated; translation has direction only, not metric scale",
        )

    def summary(self) -> OdometrySummary:
        return OdometrySummary(
            total_frames=self._frame_index,
            estimated_frames=self._estimated_frames,
            insufficient_frames=self._insufficient_frames,
            metric_scale_available=False,
        )

    def _insufficient(
        self,
        frame_index: int,
        previous_keypoints: int,
        current_keypoints: int,
        matches: int,
        inliers: int,
        message: str,
    ) -> OdometryFrame:
        self._insufficient_frames += 1
        return OdometryFrame(
            frame_index=frame_index,
            status=OdometryStatus.INSUFFICIENT_FEATURES,
            previous_keypoints=previous_keypoints,
            current_keypoints=current_keypoints,
            matches=matches,
            inliers=inliers,
            metric_scale_available=False,
            message=message,
        )
