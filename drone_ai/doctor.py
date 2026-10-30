"""Non-flying environment and hardware commissioning diagnostics."""

from __future__ import annotations

from collections.abc import Callable
from importlib import metadata
import json
from pathlib import Path
import platform
import sys
from typing import Any
from urllib import error as urlerror
from urllib import request as urlrequest

from drone_ai.cameras.tello import TelloCamera
from drone_ai.cameras.webcam import LaptopWebcamCamera
from drone_ai.calibration import load_calibration
from drone_ai.config import AppConfig
from drone_ai.controllers.tello import TelloDroneController
from drone_ai.preflight import CheckStatus, ReadinessCheck, ReadinessReport
from drone_ai.tello import TelloSession
from drone_ai.vision.detector import YOLODetector


class SystemDoctor:
    """Run diagnostics without sending takeoff or movement commands."""

    REQUIRED_DISTRIBUTIONS = (
        "pydantic",
        "numpy",
        "opencv-python",
        "ultralytics",
        "djitellopy",
    )

    def __init__(
        self,
        config: AppConfig | None = None,
        *,
        model_name: str | None = None,
        camera_index: int | None = None,
        urlopen: Callable[..., Any] | None = None,
        tello_session_factory: Callable[[], TelloSession] | None = None,
    ) -> None:
        self.config = config or AppConfig()
        self.model_name = model_name or self.config.vision.model_name
        self.camera_index = (
            self.config.vision.camera_index if camera_index is None else camera_index
        )
        self._urlopen = urlopen or urlrequest.urlopen
        self._tello_session_factory = tello_session_factory or TelloSession

    def run(
        self,
        *,
        check_ollama: bool = True,
        check_model: bool = True,
        check_webcam: bool = True,
        check_tello: bool = False,
    ) -> ReadinessReport:
        checks: list[ReadinessCheck] = []
        checks.append(self._check_python())
        checks.extend(self._check_distributions())
        checks.append(self._check_yolo() if check_model else self._skip("vision.yolo", "skipped"))
        checks.append(self._check_calibration())
        checks.append(
            self._check_ollama() if check_ollama else self._skip("llm.ollama", "skipped")
        )
        checks.append(
            self._check_webcam()
            if check_webcam
            else self._skip("camera.webcam", "skipped")
        )
        checks.extend(
            self._check_tello_hardware()
            if check_tello
            else [
                self._skip(
                    "hardware.tello",
                    "not requested; run doctor --tello after joining the Tello Wi-Fi",
                )
            ]
        )
        return ReadinessReport(checks=tuple(checks))

    def _check_python(self) -> ReadinessCheck:
        version = sys.version_info
        detail = platform.python_version()
        if version.major == 3 and version.minor == 11:
            return self._pass("runtime.python", detail)
        return self._fail("runtime.python", f"found {detail}; Python 3.11 is required")

    def _check_distributions(self) -> list[ReadinessCheck]:
        checks: list[ReadinessCheck] = []
        for distribution in self.REQUIRED_DISTRIBUTIONS:
            try:
                version = metadata.version(distribution)
            except metadata.PackageNotFoundError:
                checks.append(
                    self._fail(
                        f"dependency.{distribution}",
                        "not installed; install requirements-tello.txt",
                    )
                )
            else:
                checks.append(self._pass(f"dependency.{distribution}", version))
        return checks

    def _check_yolo(self) -> ReadinessCheck:
        model_path = Path(self.model_name)
        if model_path.suffix.lower() == ".pt" and not model_path.is_file():
            return self._fail(
                "vision.yolo",
                f"weights are missing: {model_path}; download them before Tello day",
            )
        try:
            import numpy as np

            detector = YOLODetector(
                model_name=self.model_name,
                confidence=self.config.vision.confidence,
                target_classes=self.config.vision.target_classes,
            )
            detector.validate_target_classes()
            detections = detector.detect(np.zeros((320, 320, 3), dtype=np.uint8))
        except Exception as exc:
            return self._fail("vision.yolo", f"load/warm-up failed: {exc}")
        return self._pass(
            "vision.yolo",
            f"{self.model_name} loaded; synthetic inference returned {len(detections)} detections",
        )

    def _check_calibration(self) -> ReadinessCheck:
        path = self.config.vision.calibration_file.strip()
        if not path:
            return self._skip(
                "vision.calibration",
                "not configured; follow/search can run without distortion correction",
            )
        try:
            calibration = load_calibration(path)
        except Exception as exc:
            return self._fail("vision.calibration", str(exc))
        return self._pass(
            "vision.calibration",
            f"{path}: {calibration.image_width}x{calibration.image_height}, "
            f"RMS={calibration.rms_error_px:.3f}px",
        )

    def _check_ollama(self) -> ReadinessCheck:
        url = self.config.llm.ollama_url.rstrip("/") + "/api/tags"
        try:
            response = self._urlopen(url, timeout=min(self.config.llm.request_timeout_s, 5.0))
            try:
                payload = response.read(256 * 1024 + 1)
            finally:
                response.close()
            if len(payload) > 256 * 1024:
                raise ValueError("response exceeded 256 KiB")
            parsed = json.loads(payload.decode("utf-8"))
            names = {
                str(item.get("name", ""))
                for item in parsed.get("models", [])
                if isinstance(item, dict)
            }
            wanted = self.config.llm.model_name
            aliases = {wanted, wanted + ":latest"} if ":" not in wanted else {wanted}
            if not names.intersection(aliases):
                available = ", ".join(sorted(names)) or "none"
                return self._fail(
                    "llm.ollama",
                    f"server is reachable but {wanted!r} is absent (installed: {available})",
                )
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError, urlerror.URLError) as exc:
            return self._fail("llm.ollama", f"local API/model check failed: {exc}")
        return self._pass("llm.ollama", f"local API reachable; {wanted} is installed")

    def _check_webcam(self) -> ReadinessCheck:
        camera = LaptopWebcamCamera(self.camera_index)
        try:
            camera.open()
            frame = camera.read()
            shape = getattr(frame, "shape", None)
            if not isinstance(shape, tuple) or len(shape) != 3 or shape[2] != 3:
                raise ValueError("camera did not return a three-channel BGR frame")
        except Exception as exc:
            return self._warn(
                "camera.webcam",
                f"not available at index {self.camera_index}: {exc}; Tello camera can be used later",
            )
        finally:
            try:
                camera.close()
            except Exception:
                pass
        return self._pass("camera.webcam", f"received frame with shape {shape}")

    def _check_tello_hardware(self) -> list[ReadinessCheck]:
        session = self._tello_session_factory()
        controller = TelloDroneController(session, self.config.flight)
        camera = TelloCamera(session)
        checks: list[ReadinessCheck] = []
        camera_open = False
        try:
            controller.connect()
            state = controller.get_state()
            if state.is_flying:
                checks.append(
                    self._fail("hardware.tello.telemetry", "aircraft reports already flying")
                )
            elif (
                state.altitude_m is None
                or state.altitude_m > self.config.preflight.max_landed_altitude_m
            ):
                checks.append(
                    self._fail(
                        "hardware.tello.telemetry",
                        f"landed altitude is implausible: {state.altitude_m}m",
                    )
                )
            elif state.battery_percent is None:
                checks.append(self._fail("hardware.tello.telemetry", "battery is unknown"))
            elif state.battery_percent < self.config.flight.min_takeoff_battery_percent:
                checks.append(
                    self._fail(
                        "hardware.tello.telemetry",
                        f"battery is {state.battery_percent}%; charge before takeoff",
                    )
                )
            else:
                checks.append(
                    self._pass(
                        "hardware.tello.telemetry",
                        f"connected; battery={state.battery_percent}% "
                        f"altitude={state.altitude_m}m yaw={state.yaw_degrees:.1f}deg",
                    )
                )
            camera.open()
            camera_open = True
            frame = camera.read()
            shape = getattr(frame, "shape", None)
            if not isinstance(shape, tuple) or len(shape) != 3 or shape[2] != 3:
                raise ValueError("Tello camera did not return a three-channel BGR frame")
            checks.append(self._pass("hardware.tello.video", f"received frame with shape {shape}"))
        except Exception as exc:
            checks.append(self._fail("hardware.tello", f"commissioning check failed: {exc}"))
        finally:
            if camera_open:
                try:
                    camera.close()
                except Exception as exc:
                    checks.append(self._warn("hardware.tello.cleanup", str(exc)))
            try:
                if session.connected:
                    controller.disconnect()
            except Exception as exc:
                checks.append(self._warn("hardware.tello.cleanup", str(exc)))
        return checks

    @staticmethod
    def _pass(name: str, detail: str) -> ReadinessCheck:
        return ReadinessCheck(name=name, status=CheckStatus.PASS, detail=detail)

    @staticmethod
    def _warn(name: str, detail: str) -> ReadinessCheck:
        return ReadinessCheck(name=name, status=CheckStatus.WARN, detail=detail)

    @staticmethod
    def _fail(name: str, detail: str) -> ReadinessCheck:
        return ReadinessCheck(name=name, status=CheckStatus.FAIL, detail=detail)

    @staticmethod
    def _skip(name: str, detail: str) -> ReadinessCheck:
        return ReadinessCheck(name=name, status=CheckStatus.SKIP, detail=detail)
