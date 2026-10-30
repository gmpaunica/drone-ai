from __future__ import annotations

from pathlib import Path

import pytest

from drone_ai.cli import _app_config, build_argument_parser
from drone_ai.config import AppConfig, ConfigurationError, load_app_config


def write_config(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "profile.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_complete_example_matches_code_defaults() -> None:
    example = Path(__file__).parents[1] / "config.example.toml"
    assert load_app_config(example) == AppConfig()


def test_commissioning_profile_uses_normal_battery_floor_and_short_hover() -> None:
    profile = Path(__file__).parents[1] / "config.commissioning.toml"
    config = load_app_config(profile)
    assert config.flight.min_takeoff_battery_percent == 30
    assert config.flight.critical_battery_percent == 15
    assert config.watchdog.max_airborne_time_s == 5.0
    assert config.watchdog.check_rate_hz == 2.0


def test_partial_profile_overlays_defaults_and_converts_numbers(tmp_path: Path) -> None:
    path = write_config(
        tmp_path,
        """
[flight]
max_altitude_m = 2
min_takeoff_battery_percent = 40

[vision]
camera_index = 2
target_classes = ["person", "backpack"]
""",
    )
    config = load_app_config(path)
    assert config.flight.max_altitude_m == 2.0
    assert isinstance(config.flight.max_altitude_m, float)
    assert config.flight.min_takeoff_battery_percent == 40
    assert config.vision.camera_index == 2
    assert config.vision.target_classes == ("person", "backpack")
    assert config.follow == AppConfig().follow


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("[unknown]\nvalue = 1", "unknown configuration section"),
        ("[flight]\nunknown = 1", "unknown key"),
        ('[flight]\nmax_altitude_m = "high"', "must be a number"),
        ("[flight]\nmax_altitude_m = nan", "must be finite"),
        ("[flight]\nmax_altitude_m = 3.1", "0-3m"),
        ("[flight]\nmax_speed_mps = 0.6", "0.5m/s hard cap"),
        ("[flight]\nmin_takeoff_battery_percent = 20", "cannot be below 30"),
        ("[vision]\nconfidence = 0.49", "cannot be below 0.5"),
        ('[llm]\nollama_url = "https://example.com"', "local HTTP endpoint"),
        ("[plan]\nmax_steps = 9", "within 1-8"),
        ("[manual]\nmax_command_rate_hz = 3.0", "cannot exceed 2Hz"),
        ("[manual]\ntello_rc_velocity = 51", "within 20-50"),
        ("[manual]\ntello_rc_step_duration_s = 1.1", "within 0.2-1.0"),
        ("[watchdog]\nmax_airborne_time_s = 481", "cannot exceed 480"),
        ("[fiducial]\nmaximum_pose_disagreement_m = 0.3", "0.01-0.25m"),
        ("[fiducial]\nmaximum_pose_disagreement_degrees = 16", "1-15 degrees"),
    ],
)
def test_profile_rejects_unknown_malformed_or_weakened_values(
    tmp_path: Path,
    text: str,
    message: str,
) -> None:
    with pytest.raises(ConfigurationError, match=message):
        load_app_config(write_config(tmp_path, text))


def test_cli_loads_profile_from_global_option(tmp_path: Path) -> None:
    path = write_config(tmp_path, "[vision]\ncamera_index = 3\n")
    args = build_argument_parser().parse_args(["--config", str(path), "manual"])
    assert args.config == str(path)
    assert _app_config(args).vision.camera_index == 3


def test_missing_profile_is_explicit(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="does not exist"):
        load_app_config(tmp_path / "missing.toml")
