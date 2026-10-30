"""Safe defaults plus a strict TOML overlay loader."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
import math
from pathlib import Path
import tomllib
from typing import Any, get_args, get_origin, get_type_hints
from urllib.parse import urlparse


@dataclass(frozen=True, slots=True)
class FlightConfig:
    min_altitude_m: float = 0.0
    max_altitude_m: float = 3.0
    min_movement_m: float = 0.2
    max_movement_m: float = 3.0
    max_rotation_degrees: float = 180.0
    takeoff_altitude_m: float = 1.0
    cruise_speed_mps: float = 0.2
    max_speed_mps: float = 0.5
    command_timeout_s: float = 2.0
    min_takeoff_battery_percent: int = 30
    critical_battery_percent: int = 15


@dataclass(frozen=True, slots=True)
class FollowConfig:
    rotation_degrees: float = 10.0
    movement_step_m: float = 0.3
    horizontal_dead_zone: float = 0.12
    target_size_min: float = 0.08
    target_size_max: float = 0.28
    control_rate_hz: float = 2.0


@dataclass(frozen=True, slots=True)
class VisionConfig:
    confidence: float = 0.50
    model_name: str = "yolo26n.pt"
    camera_index: int = 0
    target_classes: tuple[str, ...] = ("person",)
    calibration_file: str = ""


@dataclass(frozen=True, slots=True)
class LLMConfig:
    ollama_url: str = "http://localhost:11434"
    model_name: str = "qwen3:4b-instruct"
    request_timeout_s: float = 120.0
    max_command_characters: int = 500


@dataclass(frozen=True, slots=True)
class PlanConfig:
    max_steps: int = 8
    max_total_distance_m: float = 6.0
    max_total_rotation_degrees: float = 360.0
    max_plan_age_s: float = 30.0


@dataclass(frozen=True, slots=True)
class SearchConfig:
    rotation_step_degrees: float = 20.0
    max_scan_degrees: float = 360.0
    control_rate_hz: float = 2.0


@dataclass(frozen=True, slots=True)
class PreflightConfig:
    min_frame_width: int = 160
    min_frame_height: int = 120
    require_three_channel_bgr: bool = True
    max_landed_altitude_m: float = 0.3


@dataclass(frozen=True, slots=True)
class ManualConfig:
    horizontal_step_m: float = 0.2
    vertical_step_m: float = 0.2
    rotation_step_degrees: float = 20.0
    max_command_rate_hz: float = 2.0
    tello_rc_velocity: int = 40
    tello_rc_step_duration_s: float = 0.75


@dataclass(frozen=True, slots=True)
class WatchdogConfig:
    max_airborne_time_s: float = 8.0 * 60.0
    check_rate_hz: float = 2.0
    max_consecutive_telemetry_failures: int = 3


@dataclass(frozen=True, slots=True)
class CalibrationConfig:
    checkerboard_columns: int = 9
    checkerboard_rows: int = 6
    square_size_m: float = 0.024
    minimum_views: int = 8
    maximum_rms_error_px: float = 1.0
    sample_every_frames: int = 5


@dataclass(frozen=True, slots=True)
class VisualOdometryConfig:
    maximum_features: int = 1500
    minimum_matches: int = 20
    minimum_inliers: int = 15
    ratio_test: float = 0.75
    ransac_threshold_px: float = 1.0


@dataclass(frozen=True, slots=True)
class FiducialConfig:
    maximum_reprojection_error_px: float = 3.0
    minimum_marker_side_px: float = 20.0
    maximum_pose_disagreement_m: float = 0.25
    maximum_pose_disagreement_degrees: float = 15.0


@dataclass(frozen=True, slots=True)
class AppConfig:
    flight: FlightConfig = field(default_factory=FlightConfig)
    follow: FollowConfig = field(default_factory=FollowConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    plan: PlanConfig = field(default_factory=PlanConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    preflight: PreflightConfig = field(default_factory=PreflightConfig)
    manual: ManualConfig = field(default_factory=ManualConfig)
    watchdog: WatchdogConfig = field(default_factory=WatchdogConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    odometry: VisualOdometryConfig = field(default_factory=VisualOdometryConfig)
    fiducial: FiducialConfig = field(default_factory=FiducialConfig)


class ConfigurationError(ValueError):
    pass


def load_app_config(path: str | Path | None = None) -> AppConfig:
    """Load a partial TOML overlay while preserving code-owned safety caps."""

    defaults = AppConfig()
    if path is None:
        validate_app_config(defaults)
        return defaults
    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigurationError(f"configuration file does not exist: {config_path}")
    try:
        with config_path.open("rb") as stream:
            payload = tomllib.load(stream)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigurationError(f"invalid TOML in {config_path}: {exc}") from exc

    section_fields = {item.name: item for item in fields(AppConfig)}
    unknown_sections = sorted(set(payload) - set(section_fields))
    if unknown_sections:
        raise ConfigurationError(
            "unknown configuration section(s): " + ", ".join(unknown_sections)
        )

    replacements: dict[str, Any] = {}
    for section_name, values in payload.items():
        if not isinstance(values, dict):
            raise ConfigurationError(f"[{section_name}] must be a TOML table")
        current = getattr(defaults, section_name)
        section_type = type(current)
        known = {item.name for item in fields(section_type)}
        unknown_keys = sorted(set(values) - known)
        if unknown_keys:
            raise ConfigurationError(
                f"unknown key(s) in [{section_name}]: " + ", ".join(unknown_keys)
            )
        hints = get_type_hints(section_type)
        overrides = {
            key: _coerce_config_value(value, hints[key], f"{section_name}.{key}")
            for key, value in values.items()
        }
        replacements[section_name] = replace(current, **overrides)

    config = replace(defaults, **replacements)
    validate_app_config(config)
    return config


def validate_app_config(config: AppConfig) -> None:
    flight = config.flight
    _require(0.0 <= flight.min_altitude_m < flight.max_altitude_m <= 3.0, "flight altitude limits must stay within 0-3m")
    _require(
        flight.min_altitude_m < flight.takeoff_altitude_m <= flight.max_altitude_m,
        "flight.takeoff_altitude_m must be inside the altitude limits",
    )
    _require(
        0.2 <= flight.min_movement_m <= flight.max_movement_m <= 3.0,
        "flight movement limits must stay within 0.2-3m",
    )
    _require(
        0.0 < flight.max_rotation_degrees <= 180.0,
        "flight.max_rotation_degrees must stay within 180 degrees",
    )
    _require(
        0.0 < flight.cruise_speed_mps <= flight.max_speed_mps <= 0.5,
        "flight speed must be positive and cannot raise the 0.5m/s hard cap",
    )
    _require(
        0.0 < flight.command_timeout_s <= 2.0,
        "flight.command_timeout_s must be in (0, 2] seconds",
    )
    _require(
        15 <= flight.critical_battery_percent < flight.min_takeoff_battery_percent <= 100,
        "battery thresholds cannot weaken 15% critical / 30% takeoff defaults",
    )
    _require(
        flight.min_takeoff_battery_percent >= 30,
        "flight.min_takeoff_battery_percent cannot be below 30",
    )

    follow = config.follow
    _require(
        0.0 < follow.rotation_degrees <= flight.max_rotation_degrees,
        "follow.rotation_degrees must fit the flight rotation limit",
    )
    _require(
        flight.min_movement_m <= follow.movement_step_m <= flight.max_movement_m,
        "follow.movement_step_m must fit the flight movement limits",
    )
    _require(0.0 <= follow.horizontal_dead_zone <= 1.0, "follow.horizontal_dead_zone must be in [0, 1]")
    _require(
        0.0 <= follow.target_size_min < follow.target_size_max <= 1.0,
        "follow target-size limits must be ordered within [0, 1]",
    )
    _require(follow.control_rate_hz > 0.0, "follow.control_rate_hz must be positive")

    vision = config.vision
    _require(0.5 <= vision.confidence <= 1.0, "vision.confidence cannot be below 0.5")
    _require(bool(vision.model_name.strip()), "vision.model_name cannot be blank")
    _require(vision.camera_index >= 0, "vision.camera_index cannot be negative")
    _require(
        bool(vision.target_classes)
        and all(item.strip() for item in vision.target_classes),
        "vision.target_classes cannot be empty or contain blanks",
    )
    _require(
        not vision.calibration_file or vision.calibration_file.lower().endswith(".json"),
        "vision.calibration_file must be blank or a JSON path",
    )

    llm = config.llm
    parsed_url = urlparse(llm.ollama_url)
    _require(
        parsed_url.scheme == "http" and parsed_url.hostname in {"localhost", "127.0.0.1", "::1"},
        "llm.ollama_url must be a local HTTP endpoint",
    )
    _require(bool(llm.model_name.strip()), "llm.model_name cannot be blank")
    _require(llm.request_timeout_s > 0.0, "llm.request_timeout_s must be positive")
    _require(
        1 <= llm.max_command_characters <= 500,
        "llm.max_command_characters must stay within 1-500",
    )

    plan = config.plan
    _require(1 <= plan.max_steps <= 8, "plan.max_steps must stay within 1-8")
    _require(
        0.0 < plan.max_total_distance_m <= 6.0,
        "plan.max_total_distance_m cannot exceed 6m",
    )
    _require(
        0.0 < plan.max_total_rotation_degrees <= 360.0,
        "plan.max_total_rotation_degrees cannot exceed 360 degrees",
    )
    _require(0.0 < plan.max_plan_age_s <= 30.0, "plan.max_plan_age_s must stay in (0, 30]")

    search = config.search
    _require(
        0.0 < search.rotation_step_degrees <= flight.max_rotation_degrees,
        "search.rotation_step_degrees must fit the flight rotation limit",
    )
    _require(0.0 < search.max_scan_degrees <= 360.0, "search.max_scan_degrees must be in (0, 360]")
    _require(search.control_rate_hz > 0.0, "search.control_rate_hz must be positive")

    preflight = config.preflight
    _require(preflight.min_frame_width >= 160, "preflight.min_frame_width cannot be below 160")
    _require(preflight.min_frame_height >= 120, "preflight.min_frame_height cannot be below 120")
    _require(
        0.0 <= preflight.max_landed_altitude_m <= 0.5,
        "preflight.max_landed_altitude_m must stay within 0-0.5m",
    )

    manual = config.manual
    _require(
        flight.min_movement_m <= manual.horizontal_step_m <= flight.max_movement_m,
        "manual.horizontal_step_m must fit the flight movement limits",
    )
    _require(
        flight.min_movement_m <= manual.vertical_step_m <= flight.max_movement_m,
        "manual.vertical_step_m must fit the flight movement limits",
    )
    _require(
        0.0 < manual.rotation_step_degrees <= flight.max_rotation_degrees,
        "manual.rotation_step_degrees must fit the flight rotation limit",
    )
    _require(
        0.0 < manual.max_command_rate_hz <= 2.0,
        "manual.max_command_rate_hz cannot exceed 2Hz",
    )
    _require(
        20 <= manual.tello_rc_velocity <= 50,
        "manual.tello_rc_velocity must stay within 20-50",
    )
    _require(
        0.2 <= manual.tello_rc_step_duration_s <= 1.0,
        "manual.tello_rc_step_duration_s must stay within 0.2-1.0 seconds",
    )

    watchdog = config.watchdog
    _require(
        0.0 < watchdog.max_airborne_time_s <= 480.0,
        "watchdog.max_airborne_time_s cannot exceed 480 seconds",
    )
    _require(watchdog.check_rate_hz >= 2.0, "watchdog.check_rate_hz cannot be below 2Hz")
    _require(
        1 <= watchdog.max_consecutive_telemetry_failures <= 3,
        "watchdog telemetry failures must stay within 1-3",
    )

    calibration = config.calibration
    _require(
        3 <= calibration.checkerboard_columns <= 20,
        "calibration.checkerboard_columns must stay within 3-20",
    )
    _require(
        3 <= calibration.checkerboard_rows <= 20,
        "calibration.checkerboard_rows must stay within 3-20",
    )
    _require(
        0.001 <= calibration.square_size_m <= 0.2,
        "calibration.square_size_m must stay within 0.001-0.2m",
    )
    _require(
        5 <= calibration.minimum_views <= 100,
        "calibration.minimum_views must stay within 5-100",
    )
    _require(
        0.0 < calibration.maximum_rms_error_px <= 5.0,
        "calibration.maximum_rms_error_px must stay within 0-5px",
    )
    _require(
        1 <= calibration.sample_every_frames <= 1000,
        "calibration.sample_every_frames must stay within 1-1000",
    )

    odometry = config.odometry
    _require(
        100 <= odometry.maximum_features <= 5000,
        "odometry.maximum_features must stay within 100-5000",
    )
    _require(
        8 <= odometry.minimum_inliers <= odometry.minimum_matches <= odometry.maximum_features,
        "odometry inlier/match thresholds are invalid",
    )
    _require(0.5 <= odometry.ratio_test < 1.0, "odometry.ratio_test must stay within [0.5, 1)")
    _require(
        0.1 <= odometry.ransac_threshold_px <= 3.0,
        "odometry.ransac_threshold_px must stay within 0.1-3px",
    )

    fiducial = config.fiducial
    _require(
        0.1 <= fiducial.maximum_reprojection_error_px <= 5.0,
        "fiducial.maximum_reprojection_error_px must stay within 0.1-5px",
    )
    _require(
        8.0 <= fiducial.minimum_marker_side_px <= 200.0,
        "fiducial.minimum_marker_side_px must stay within 8-200px",
    )
    _require(
        0.01 <= fiducial.maximum_pose_disagreement_m <= 0.25,
        "fiducial.maximum_pose_disagreement_m must stay within 0.01-0.25m",
    )
    _require(
        1.0 <= fiducial.maximum_pose_disagreement_degrees <= 15.0,
        "fiducial.maximum_pose_disagreement_degrees must stay within 1-15 degrees",
    )


def _coerce_config_value(value: Any, expected: Any, path: str) -> Any:
    if expected is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigurationError(f"{path} must be a number")
        converted = float(value)
        if not math.isfinite(converted):
            raise ConfigurationError(f"{path} must be finite")
        return converted
    if expected is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigurationError(f"{path} must be an integer")
        return value
    if expected is bool:
        if not isinstance(value, bool):
            raise ConfigurationError(f"{path} must be true or false")
        return value
    if expected is str:
        if not isinstance(value, str):
            raise ConfigurationError(f"{path} must be a string")
        return value
    if get_origin(expected) is tuple and get_args(expected) == (str, Ellipsis):
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise ConfigurationError(f"{path} must be an array of strings")
        return tuple(value)
    raise ConfigurationError(f"{path} uses an unsupported configuration type")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigurationError(message)
