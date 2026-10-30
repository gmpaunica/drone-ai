from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from drone_ai.cli import build_argument_parser
from drone_ai.calibration import CameraCalibration, save_calibration
from drone_ai.config import AppConfig
from drone_ai.doctor import SystemDoctor
from drone_ai.preflight import CheckStatus, ReadinessCheck, ReadinessReport
from drone_ai.tello import TelloSession


class FakeResponse:
    def __init__(self, payload: object) -> None:
        self.payload = json.dumps(payload).encode()
        self.closed = False

    def read(self, limit: int) -> bytes:
        assert limit == 256 * 1024 + 1
        return self.payload

    def close(self) -> None:
        self.closed = True


def test_ollama_doctor_confirms_configured_local_model() -> None:
    response = FakeResponse({"models": [{"name": "qwen3:4b-instruct"}]})
    doctor = SystemDoctor(urlopen=lambda *args, **kwargs: response)
    check = doctor._check_ollama()
    assert check.status is CheckStatus.PASS
    assert response.closed


def test_ollama_doctor_reports_missing_model() -> None:
    doctor = SystemDoctor(
        urlopen=lambda *args, **kwargs: FakeResponse({"models": [{"name": "other:latest"}]})
    )
    check = doctor._check_ollama()
    assert check.status is CheckStatus.FAIL
    assert "qwen3:4b-instruct" in check.detail


def test_readiness_report_json_contains_computed_ready_state() -> None:
    report = ReadinessReport(
        checks=(ReadinessCheck(name="example", status=CheckStatus.PASS, detail="ok"),)
    )
    assert json.loads(report.model_dump_json())["ready"] is True


def test_doctor_cli_defaults_to_non_flying_hardware_skip() -> None:
    args = build_argument_parser().parse_args(["doctor", "--json"])
    assert args.mode == "doctor"
    assert not args.tello
    assert args.json


class FakeFrameReader:
    frame = np.ones((720, 960, 3), dtype=np.uint8)
    stopped = False


class FakeTello:
    def __init__(self, *, battery: int = 80) -> None:
        self.battery = battery
        self.calls: list[str] = []
        self.is_flying = False

    def connect(self) -> None:
        self.calls.append("connect")

    def set_speed(self, speed: int) -> None:
        self.calls.append("set_speed")

    def get_height(self) -> int:
        return 0

    def get_yaw(self) -> int:
        return 0

    def get_battery(self) -> int:
        return self.battery

    def streamon(self) -> None:
        self.calls.append("streamon")

    def get_frame_read(self) -> FakeFrameReader:
        return FakeFrameReader()

    def streamoff(self) -> None:
        self.calls.append("streamoff")

    def end(self) -> None:
        self.calls.append("end")

    def takeoff(self) -> None:
        self.calls.append("takeoff")


def test_tello_commissioning_checks_telemetry_and_video_without_takeoff() -> None:
    client = FakeTello()
    doctor = SystemDoctor(tello_session_factory=lambda: TelloSession(client))
    checks = doctor._check_tello_hardware()
    assert all(check.status is CheckStatus.PASS for check in checks)
    assert {check.name for check in checks} == {
        "hardware.tello.telemetry",
        "hardware.tello.video",
    }
    assert "connect" in client.calls
    assert "streamon" in client.calls
    assert "streamoff" in client.calls
    assert client.calls[-1] == "end"
    assert "takeoff" not in client.calls


def test_tello_commissioning_fails_low_battery_without_takeoff() -> None:
    client = FakeTello(battery=10)
    doctor = SystemDoctor(tello_session_factory=lambda: TelloSession(client))
    checks = doctor._check_tello_hardware()
    battery = next(check for check in checks if check.name == "hardware.tello.telemetry")
    assert battery.status is CheckStatus.FAIL
    assert "takeoff" not in client.calls


def test_doctor_validates_configured_calibration_artifact(tmp_path: Path) -> None:
    path = tmp_path / "camera.json"
    artifact = CameraCalibration(
        image_width=960,
        image_height=720,
        checkerboard_columns=9,
        checkerboard_rows=6,
        square_size_m=0.024,
        views_used=8,
        rms_error_px=0.25,
        maximum_rms_error_px=1.0,
        camera_matrix=((700.0, 0.0, 480.0), (0.0, 700.0, 360.0), (0.0, 0.0, 1.0)),
        distortion_coefficients=(0.0, 0.0, 0.0, 0.0, 0.0),
    )
    save_calibration(artifact, path)
    defaults = AppConfig()
    config = replace(
        defaults,
        vision=replace(defaults.vision, calibration_file=str(path)),
    )
    check = SystemDoctor(config)._check_calibration()
    assert check.status is CheckStatus.PASS
    assert "960x720" in check.detail
