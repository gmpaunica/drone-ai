"""Command-line entry points for one-shot commands and live following."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import platform
import sys
import time
from typing import Any

from drone_ai.abilities.follow_person import FollowPersonAbility
from drone_ai.abilities.visual_search import SearchState, VisualSearchAbility
from drone_ai.cameras.base import CameraSource
from drone_ai.cameras.blank import BlankCamera
from drone_ai.cameras.tello import TelloCamera
from drone_ai.cameras.undistort import UndistortingCamera
from drone_ai.cameras.video import CameraEndOfStream, VideoFileCamera
from drone_ai.calibration import CheckerboardCalibrator, load_calibration, save_calibration
from drone_ai.cameras.webcam import LaptopWebcamCamera
from drone_ai.commands import ActionCommand, ActionType, CommandSource
from drone_ai.config import AppConfig, LLMConfig, load_app_config, validate_app_config
from drone_ai.controllers.base import DroneController
from drone_ai.controllers.simulated import SimulatedDrone
from drone_ai.controllers.tello import TelloDroneController
from drone_ai.doctor import SystemDoctor
from drone_ai.fiducials import (
    ArucoDictionary,
    FiducialPoseAnalyzer,
    load_fiducial_map,
    save_printable_marker_svg,
)
from drone_ai.llm.interpreter import HybridCommandInterpreter, RuleCommandInterpreter
from drone_ai.llm.ollama import OllamaCommandInterpreter
from drone_ai.localization import (
    StaticMapPoseAnalyzer,
    iter_fiducial_jsonl,
    save_mapped_trajectory_svg,
)
from drone_ai.manual import ManualController, ManualEvent
from drone_ai.navigation import (
    GridCell,
    RoomRoutePlanner,
    audit_simulated_route,
    cell_center_world,
    load_room_map,
    save_navigation_route,
    save_route_svg,
)
from drone_ai.odometry import VisualOdometryAnalyzer
from drone_ai.parser import CommandParseError
from drone_ai.pipeline import FollowDecision, FollowPipeline
from drone_ai.planning.executor import PlanExecutor
from drone_ai.planning.models import ActionPlan, PlanExecutionResult, PlanExecutionStatus
from drone_ai.planning.ollama import OllamaPlanInterpreter
from drone_ai.planning.parser import HybridPlanInterpreter, RuleBasedPlanInterpreter
from drone_ai.preflight import AutonomousPreflight, ReadinessReport
from drone_ai.replay import ReplayFrame, VisionReplay
from drone_ai.recording import SessionProvenance, SessionRecorder
from drone_ai.recording_audit import SessionAuditStatus, SessionAuditor
from drone_ai.safety import SafetyController, SafetyResult, SafetyStatus
from drone_ai.search_pipeline import SearchPipeline, SearchTick
from drone_ai.tello import TelloSession
from drone_ai.vision.detector import YOLODetector, draw_detections
from drone_ai.vision.tracker import PersonTracker, TargetTracker
from drone_ai.watchdog import FlightWatchdog, WatchdogEvent


LOGGER = logging.getLogger(__name__)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Safe webcam-to-drone person following")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING"))
    parser.add_argument(
        "--config",
        default=None,
        help="validated TOML profile (place this option before the mode)",
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    follow = subparsers.add_parser("follow", help="run the person-following camera loop")
    _add_drone_arguments(follow)
    follow.add_argument("--takeoff", action="store_true", help="take off before following")
    _add_camera_arguments(follow)
    _add_recording_argument(follow)

    command = subparsers.add_parser("command", help="parse and execute one command")
    _add_drone_arguments(command)
    command.add_argument("text", help="for example: 'move forward two meters'")
    command.add_argument("--takeoff", action="store_true", help="take off before the command")
    command.add_argument(
        "--interpreter",
        choices=("rule", "hybrid", "ollama"),
        default="rule",
        help="command interpreter (default: deterministic rule parser)",
    )
    _add_ollama_arguments(command)
    _add_camera_arguments(command)
    _add_recording_argument(command)

    plan = subparsers.add_parser("plan", help="preview or execute a bounded compound plan")
    _add_drone_arguments(plan)
    plan.add_argument("text", help="ordered actions separated by 'then', commas, or semicolons")
    plan.add_argument("--execute", action="store_true", help="execute after printing the plan")
    plan.add_argument("--takeoff", action="store_true", help="take off before plan execution")
    plan.add_argument(
        "--planner",
        choices=("rule", "hybrid", "ollama"),
        default="rule",
        help="plan interpreter (default: deterministic rules)",
    )
    _add_ollama_arguments(plan)

    search = subparsers.add_parser("search", help="scan up to 360 degrees for a visual target")
    _add_drone_arguments(search)
    search.add_argument("target", help="YOLO class to find, for example person or backpack")
    search.add_argument("--takeoff", action="store_true", help="take off before searching")
    _add_camera_arguments(search)
    _add_recording_argument(search)

    manual = subparsers.add_parser(
        "manual",
        help="open a safety-gated keyboard console for commissioning",
    )
    _add_drone_arguments(manual)
    _add_camera_arguments(manual, allow_blank=True, include_model=False)
    _add_recording_argument(manual)

    replay = subparsers.add_parser(
        "replay",
        help="run hardware-free YOLO/tracking analysis on a video file",
    )
    replay.add_argument("input", help="input video path")
    replay.add_argument("--target", default="person", help="YOLO class to track")
    replay.add_argument("--model", default=None, help="Ultralytics model weight path")
    replay.add_argument("--calibration", default=None, help="camera calibration JSON")
    replay.add_argument("--annotated-video", default=None, help="optional annotated MP4 output")
    replay.add_argument("--jsonl", default=None, help="optional per-frame JSONL output")
    replay.add_argument("--display", action="store_true", help="show replay while processing")
    replay.add_argument("--max-frames", type=int, default=None)
    replay.add_argument("--overwrite", action="store_true")

    calibrate = subparsers.add_parser(
        "calibrate",
        help="calculate camera intrinsics from recorded checkerboard video",
    )
    calibrate.add_argument("input", help="checkerboard video path")
    calibrate.add_argument("output", help="calibration JSON output path")
    calibrate.add_argument("--columns", type=int, default=None, help="inner corner columns")
    calibrate.add_argument("--rows", type=int, default=None, help="inner corner rows")
    calibrate.add_argument("--square-size-m", type=float, default=None)
    calibrate.add_argument("--min-views", type=int, default=None)
    calibrate.add_argument("--max-rms-error-px", type=float, default=None)
    calibrate.add_argument("--sample-every", type=int, default=None)
    calibrate.add_argument("--max-frames", type=int, default=None)
    calibrate.add_argument("--display", action="store_true")
    calibrate.add_argument("--overwrite", action="store_true")

    odometry = subparsers.add_parser(
        "odometry",
        help="analyze offline monocular frame-to-frame pose quality",
    )
    odometry.add_argument("input", help="input video path")
    odometry.add_argument("--calibration", default=None, help="required camera calibration JSON")
    odometry.add_argument("--jsonl", default=None, help="optional per-frame estimates")
    odometry.add_argument("--max-frames", type=int, default=None)
    odometry.add_argument("--overwrite", action="store_true")

    fiducials = subparsers.add_parser(
        "fiducials",
        help="estimate calibrated ArUco camera poses from recorded video only",
    )
    fiducials.add_argument("input", help="input video path")
    fiducials.add_argument("map", help="fiducial marker-map JSON")
    fiducials.add_argument(
        "--calibration",
        default=None,
        help="required camera calibration JSON",
    )
    fiducials.add_argument("--jsonl", default=None, help="optional per-frame pose records")
    fiducials.add_argument("--annotated-video", default=None, help="optional annotated MP4 output")
    fiducials.add_argument("--display", action="store_true", help="show analysis while processing")
    fiducials.add_argument("--max-frames", type=int, default=None)
    fiducials.add_argument("--overwrite", action="store_true")

    make_marker = subparsers.add_parser(
        "make-marker",
        help="generate an exact-scale printable ArUco SVG without flight hardware",
    )
    make_marker.add_argument("output", help="output .svg path")
    make_marker.add_argument(
        "--dictionary",
        choices=tuple(item.value for item in ArucoDictionary),
        default=ArucoDictionary.DICT_4X4_50.value,
    )
    make_marker.add_argument("--id", type=int, default=0, help="marker ID")
    make_marker.add_argument(
        "--size-m",
        type=float,
        default=0.12,
        help="physical black-square side length in meters (0.03-0.16)",
    )
    make_marker.add_argument("--overwrite", action="store_true")

    map_poses = subparsers.add_parser(
        "map-poses",
        help="project offline fiducial pose records into a static room grid",
    )
    map_poses.add_argument("input", help="fiducial pose JSONL")
    map_poses.add_argument("map", help="registered room map JSON")
    map_poses.add_argument("--jsonl", default=None, help="optional mapped per-frame records")
    map_poses.add_argument("--svg", default=None, help="optional mapped trajectory SVG")
    map_poses.add_argument("--max-frames", type=int, default=None)
    map_poses.add_argument("--overwrite", action="store_true")

    audit_session = subparsers.add_parser(
        "audit-session",
        help="verify a recorded session offline without a drone",
    )
    audit_session.add_argument("session", help="recording session directory")
    audit_session.add_argument("--json", action="store_true", help="emit strict JSON report")

    navigate = subparsers.add_parser(
        "navigate",
        help="preview or execute a static-map route in the simulator only",
    )
    navigate.add_argument("map", help="room map JSON")
    navigate.add_argument("--start", required=True, help="start grid cell as x,y")
    navigate.add_argument("--goal", required=True, help="goal grid cell as x,y")
    navigate.add_argument("--start-yaw", type=float, default=0.0)
    navigate.add_argument("--svg", default=None, help="optional route preview SVG")
    navigate.add_argument("--route-json", default=None, help="optional strict route JSON")
    navigate.add_argument("--overwrite", action="store_true")
    navigate.add_argument("--execute", action="store_true")
    navigate.add_argument("--takeoff", action="store_true")

    doctor = subparsers.add_parser(
        "doctor",
        help="run non-flying environment and optional Tello commissioning checks",
    )
    doctor.add_argument("--json", action="store_true", help="emit a machine-readable report")
    doctor.add_argument(
        "--tello",
        action="store_true",
        help="connect to Tello telemetry/video without taking off",
    )
    doctor.add_argument("--skip-ollama", action="store_true")
    doctor.add_argument("--skip-model", action="store_true")
    doctor.add_argument("--skip-webcam", action="store_true")
    doctor.add_argument("--camera-index", type=int, default=None)
    doctor.add_argument("--model", default=None, help="Ultralytics model weight path")
    _add_ollama_arguments(doctor)
    return parser


def _add_drone_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--drone-mode",
        choices=("simulator", "tello"),
        default="simulator",
        help="flight adapter (default: simulator)",
    )


def _add_ollama_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--ollama-model", default=None, help="local Ollama model name")
    parser.add_argument("--ollama-url", default=None, help="Ollama server base URL")
    parser.add_argument("--ollama-timeout", type=float, default=None, help="request timeout in seconds")


def _add_recording_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--record-dir",
        default=None,
        help="opt in to a unique video/JSONL session under this directory",
    )


def _add_camera_arguments(
    parser: argparse.ArgumentParser,
    *,
    allow_blank: bool = False,
    include_model: bool = True,
) -> None:
    choices = ("webcam", "tello", "blank") if allow_blank else ("webcam", "tello")
    default_help = (
        "defaults to Tello with --drone-mode tello, otherwise a generated status frame"
        if allow_blank
        else "defaults to Tello with --drone-mode tello, otherwise webcam"
    )
    parser.add_argument(
        "--camera",
        choices=choices,
        default=None,
        help=f"frame source ({default_help})",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=None,
        help="override the configured camera index",
    )
    parser.add_argument(
        "--calibration",
        default=None,
        help="override the configured camera calibration JSON",
    )
    if include_model:
        parser.add_argument(
            "--model", default=None, help="Ultralytics model name or local weight path"
        )


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    try:
        if args.mode == "doctor":
            return run_doctor(args)
        if args.mode == "command":
            return run_command(args)
        if args.mode == "plan":
            return run_plan(args)
        if args.mode == "search":
            return run_search(args, args.target)
        if args.mode == "manual":
            return run_manual(args)
        if args.mode == "replay":
            return run_replay(args)
        if args.mode == "calibrate":
            return run_calibrate(args)
        if args.mode == "odometry":
            return run_odometry(args)
        if args.mode == "fiducials":
            return run_fiducials(args)
        if args.mode == "make-marker":
            return run_make_marker(args)
        if args.mode == "map-poses":
            return run_map_poses(args)
        if args.mode == "audit-session":
            return run_audit_session(args)
        if args.mode == "navigate":
            return run_navigate(args)
        return run_follow(args)
    except (CommandParseError, RuntimeError, ValueError) as exc:
        LOGGER.error("%s", exc)
        return 2


def run_doctor(args: argparse.Namespace) -> int:
    defaults = _app_config(args)
    config = replace(defaults, llm=_make_llm_config(args, defaults))
    report = SystemDoctor(
        config,
        model_name=args.model,
        camera_index=_camera_index(args, config),
    ).run(
        check_ollama=not args.skip_ollama,
        check_model=not args.skip_model,
        check_webcam=not args.skip_webcam,
        check_tello=args.tello,
    )
    if args.json:
        print(report.model_dump_json(indent=2))
    else:
        _print_readiness_report("system doctor", report)
    return 0 if report.ready else 1


def run_replay(args: argparse.Namespace) -> int:
    config = _app_config(args)
    target = args.target.strip().lower()
    if not target:
        raise ValueError("replay target cannot be blank")
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("--max-frames must be positive")

    input_path = Path(args.input)
    annotated_path = Path(args.annotated_video) if args.annotated_video else None
    jsonl_path = Path(args.jsonl) if args.jsonl else None
    for output_path in (annotated_path, jsonl_path):
        if output_path is None:
            continue
        if output_path.resolve() == input_path.resolve():
            raise ValueError("replay output cannot overwrite the input video")
        if output_path.exists() and not args.overwrite:
            raise ValueError(f"output already exists (use --overwrite): {output_path}")

    detector = YOLODetector(
        model_name=args.model or config.vision.model_name,
        confidence=config.vision.confidence,
        target_classes=(target,),
    )
    detector.validate_target_classes()
    video_camera = VideoFileCamera(input_path)
    camera: CameraSource = video_camera
    calibration_path = _calibration_path(args, config)
    if calibration_path is not None:
        camera = UndistortingCamera(camera, load_calibration(calibration_path))
    camera.open()
    replay = VisionReplay(detector, TargetTracker(target, config.follow), fps=video_camera.fps)
    writer: Any | None = None
    jsonl_stream: Any | None = None
    try:
        if jsonl_path is not None:
            jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            jsonl_stream = jsonl_path.open("w", encoding="utf-8", newline="\n")
        while args.max_frames is None or video_camera.frames_read < args.max_frames:
            try:
                frame = camera.read()
            except CameraEndOfStream:
                break
            analyzed = replay.process(frame)
            if jsonl_stream is not None:
                jsonl_stream.write(analyzed.record.model_dump_json() + "\n")
            if annotated_path is not None or args.display:
                draw_detections(frame, analyzed.detections)
                _draw_replay_status(frame, analyzed)
            if annotated_path is not None:
                if writer is None:
                    writer = _open_video_writer(annotated_path, frame, video_camera.fps)
                writer.write(frame)
            if args.display:
                import cv2

                cv2.imshow("Drone AI Replay - Q/Esc quit", frame)
                key = cv2.waitKey(max(1, int(round(1000.0 / video_camera.fps)))) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break
    finally:
        if jsonl_stream is not None:
            jsonl_stream.close()
        if writer is not None:
            writer.release()
        camera.close()
        if args.display:
            import cv2

            cv2.destroyAllWindows()

    summary = replay.summary()
    print(
        f"replay: target={summary.target_class} frames={summary.total_frames} "
        f"visible_frames={summary.frames_with_target} "
        f"max_confidence={summary.maximum_confidence}"
    )
    if annotated_path is not None:
        print(f"annotated video: {annotated_path}")
    if jsonl_path is not None:
        print(f"frame records: {jsonl_path}")
    return 0 if summary.total_frames > 0 else 1


def run_calibrate(args: argparse.Namespace) -> int:
    config = _app_config(args)
    overrides = {
        "checkerboard_columns": args.columns,
        "checkerboard_rows": args.rows,
        "square_size_m": args.square_size_m,
        "minimum_views": args.min_views,
        "maximum_rms_error_px": args.max_rms_error_px,
        "sample_every_frames": args.sample_every,
    }
    calibration_config = replace(
        config.calibration,
        **{key: value for key, value in overrides.items() if value is not None},
    )
    validate_app_config(replace(config, calibration=calibration_config))
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("--max-frames must be positive")

    input_path = Path(args.input)
    output_path = Path(args.output)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("calibration output cannot overwrite its input video")
    if output_path.exists() and not args.overwrite:
        raise ValueError(f"calibration output already exists (use --overwrite): {output_path}")

    camera = VideoFileCamera(input_path)
    calibrator = CheckerboardCalibrator(calibration_config)
    camera.open()
    try:
        while args.max_frames is None or camera.frames_read < args.max_frames:
            try:
                frame = camera.read()
            except CameraEndOfStream:
                break
            frame_index = camera.frames_read - 1
            sampled = frame_index % calibration_config.sample_every_frames == 0
            accepted = calibrator.add_frame(frame) if sampled else False
            if args.display:
                import cv2

                if accepted:
                    calibrator.draw_last_observation(frame)
                text = (
                    f"frame={frame_index} views={calibrator.views_collected}/"
                    f"{calibration_config.minimum_views} accepted={accepted}"
                )
                cv2.putText(
                    frame,
                    text,
                    (12, 26),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255),
                    2,
                )
                cv2.imshow("Drone AI Calibration - Q/Esc quit", frame)
                key = cv2.waitKey(max(1, int(round(1000.0 / camera.fps)))) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break
    finally:
        camera.close()
        if args.display:
            import cv2

            cv2.destroyAllWindows()

    calibration = calibrator.calibrate()
    save_calibration(calibration, output_path, overwrite=args.overwrite)
    print(
        f"calibration: views={calibration.views_used} "
        f"resolution={calibration.image_width}x{calibration.image_height} "
        f"rms={calibration.rms_error_px:.4f}px"
    )
    print(f"calibration file: {output_path}")
    return 0


def run_odometry(args: argparse.Namespace) -> int:
    config = _app_config(args)
    calibration_path = _calibration_path(args, config)
    if calibration_path is None:
        raise ValueError("odometry requires --calibration or vision.calibration_file")
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("--max-frames must be positive")
    input_path = Path(args.input)
    jsonl_path = Path(args.jsonl) if args.jsonl else None
    if jsonl_path is not None:
        if jsonl_path.resolve() == input_path.resolve():
            raise ValueError("odometry output cannot overwrite its input video")
        if jsonl_path.exists() and not args.overwrite:
            raise ValueError(f"odometry output already exists (use --overwrite): {jsonl_path}")

    calibration = load_calibration(calibration_path)
    video_camera = VideoFileCamera(input_path)
    camera = UndistortingCamera(video_camera, calibration)
    analyzer = VisualOdometryAnalyzer(calibration, config.odometry)
    jsonl_stream: Any | None = None
    camera.open()
    try:
        if jsonl_path is not None:
            jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            jsonl_stream = jsonl_path.open("w", encoding="utf-8", newline="\n")
        while args.max_frames is None or video_camera.frames_read < args.max_frames:
            try:
                frame = camera.read()
            except CameraEndOfStream:
                break
            estimate = analyzer.process(frame)
            if jsonl_stream is not None:
                jsonl_stream.write(estimate.model_dump_json() + "\n")
    finally:
        if jsonl_stream is not None:
            jsonl_stream.close()
        camera.close()

    summary = analyzer.summary()
    print(
        f"odometry: frames={summary.total_frames} estimates={summary.estimated_frames} "
        f"insufficient={summary.insufficient_frames} metric_scale=False"
    )
    if jsonl_path is not None:
        print(f"odometry records: {jsonl_path}")
    if summary.estimated_frames == 0:
        LOGGER.error("no relative poses met the configured feature/inlier thresholds")
        return 1
    return 0


def run_navigate(args: argparse.Namespace) -> int:
    config = _app_config(args)
    map_path = Path(args.map)
    room = load_room_map(map_path)
    start = _parse_grid_cell(args.start, "start")
    goal = _parse_grid_cell(args.goal, "goal")
    route = RoomRoutePlanner(room, config.flight, config.plan).plan(
        start,
        goal,
        start_yaw_degrees=args.start_yaw,
    )
    print(
        f"route: room={route.room_name} start=({start.x},{start.y}) "
        f"goal=({goal.x},{goal.y}) cells={len(route.path)} "
        f"distance={route.distance_m:g}m simulator_only=True"
    )
    print("path: " + " -> ".join(f"({cell.x},{cell.y})" for cell in route.path))
    _print_plan(route.plan)
    svg_path = Path(args.svg) if args.svg else None
    route_json_path = Path(args.route_json) if args.route_json else None
    output_paths = [path for path in (svg_path, route_json_path) if path is not None]
    if len({path.resolve() for path in output_paths}) != len(output_paths):
        raise ValueError("navigation outputs must use different paths")
    if any(path.resolve() == map_path.resolve() for path in output_paths):
        raise ValueError("navigation output cannot overwrite the input map")
    if svg_path is not None:
        save_route_svg(room, route, svg_path, overwrite=args.overwrite)
        print(f"route preview: {svg_path}")
    if route_json_path is not None:
        save_navigation_route(route, route_json_path, overwrite=args.overwrite)
        print(f"route record: {route_json_path}")
    if not args.execute:
        print("preview only: add --execute to run this route in the simulator")
        return 0

    start_world_x, start_world_y = cell_center_world(room, route.start)
    drone = SimulatedDrone(
        config.flight,
        initial_x_m=start_world_x,
        initial_y_m=start_world_y,
        initial_yaw_degrees=route.start_yaw_degrees,
    )
    safety = SafetyController(drone, config.flight)
    drone.connect()
    try:
        if args.takeoff:
            preflight = safety.execute(
                ActionCommand(action=ActionType.TAKEOFF, source=CommandSource.SYSTEM)
            )
            _print_result("preflight", preflight)
            if not preflight.approved:
                return 1
        result = PlanExecutor(safety, config.plan).execute(route.plan)
        _print_plan_result(result)
        state = drone.get_state()
        print(f"state: {state}")
        audit = audit_simulated_route(room, route, state)
        print(
            f"route audit: passed={audit.passed} "
            f"position_error={audit.position_error_m:.9f}m "
            f"yaw_error={audit.yaw_error_degrees:.9f}deg simulator_only=True"
        )
        return 0 if result.completed and audit.passed else 1
    finally:
        _orderly_shutdown(drone, safety)


def run_fiducials(args: argparse.Namespace) -> int:
    config = _app_config(args)
    calibration_path = _calibration_path(args, config)
    if calibration_path is None:
        raise ValueError("fiducial analysis requires --calibration or vision.calibration_file")
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("--max-frames must be positive")

    input_path = Path(args.input)
    map_path = Path(args.map)
    calibration_source = Path(calibration_path)
    jsonl_path = Path(args.jsonl) if args.jsonl else None
    annotated_path = Path(args.annotated_video) if args.annotated_video else None
    protected_inputs = {
        input_path.resolve(),
        map_path.resolve(),
        calibration_source.resolve(),
    }
    outputs = [path for path in (jsonl_path, annotated_path) if path is not None]
    if len({path.resolve() for path in outputs}) != len(outputs):
        raise ValueError("fiducial outputs must use different paths")
    for output_path in outputs:
        if output_path.resolve() in protected_inputs:
            raise ValueError("fiducial output cannot overwrite an input artifact")
        if output_path.exists() and not args.overwrite:
            raise ValueError(f"output already exists (use --overwrite): {output_path}")

    calibration = load_calibration(calibration_source)
    fiducial_map = load_fiducial_map(map_path)
    analyzer = FiducialPoseAnalyzer(calibration, fiducial_map, config.fiducial)
    camera = VideoFileCamera(input_path)
    jsonl_stream: Any | None = None
    writer: Any | None = None
    camera.open()
    try:
        if jsonl_path is not None:
            jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            jsonl_stream = jsonl_path.open("w", encoding="utf-8", newline="\n")
        while args.max_frames is None or camera.frames_read < args.max_frames:
            try:
                frame = camera.read()
            except CameraEndOfStream:
                break
            record = analyzer.process(frame)
            if jsonl_stream is not None:
                jsonl_stream.write(record.model_dump_json() + "\n")
            if annotated_path is not None or args.display:
                analyzer.draw(frame, record)
            if annotated_path is not None:
                if writer is None:
                    writer = _open_video_writer(annotated_path, frame, camera.fps)
                writer.write(frame)
            if args.display:
                import cv2

                cv2.imshow("Drone AI Fiducials (offline) - Q/Esc quit", frame)
                key = cv2.waitKey(max(1, int(round(1000.0 / camera.fps)))) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break
    finally:
        if jsonl_stream is not None:
            jsonl_stream.close()
        if writer is not None:
            writer.release()
        camera.close()
        if args.display:
            import cv2

            cv2.destroyAllWindows()

    summary = analyzer.summary()
    print(
        f"fiducials: frames={summary.total_frames} localized={summary.localized_frames} "
        f"accepted={summary.accepted_observations} "
        f"rejected={summary.rejected_mapped_observations} "
        f"inconsistent={summary.inconsistent_frames} offline_only=True"
    )
    if annotated_path is not None:
        print(f"annotated video: {annotated_path}")
    if jsonl_path is not None:
        print(f"pose records: {jsonl_path}")
    if summary.localized_frames == 0:
        LOGGER.error("no mapped fiducial produced an acceptable calibrated pose")
        return 1
    return 0


def run_make_marker(args: argparse.Namespace) -> int:
    output = save_printable_marker_svg(
        ArucoDictionary(args.dictionary),
        args.id,
        args.size_m,
        args.output,
        overwrite=args.overwrite,
    )
    print(
        f"marker: dictionary={args.dictionary} id={args.id} "
        f"size={args.size_m:g}m output={output}"
    )
    print("print A4 portrait at 100% / actual size and verify the measurement line")
    return 0


def run_map_poses(args: argparse.Namespace) -> int:
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("--max-frames must be positive")
    input_path = Path(args.input)
    map_path = Path(args.map)
    jsonl_path = Path(args.jsonl) if args.jsonl else None
    svg_path = Path(args.svg) if args.svg else None
    protected_inputs = {input_path.resolve(), map_path.resolve()}
    outputs = [path for path in (jsonl_path, svg_path) if path is not None]
    if len({path.resolve() for path in outputs}) != len(outputs):
        raise ValueError("mapped-pose outputs must use different paths")
    for output in outputs:
        if output.resolve() in protected_inputs:
            raise ValueError("mapped-pose output cannot overwrite an input artifact")
        if output.exists() and not args.overwrite:
            raise ValueError(f"output already exists (use --overwrite): {output}")

    room = load_room_map(map_path)
    analyzer = StaticMapPoseAnalyzer(room)
    records = []
    jsonl_stream: Any | None = None
    try:
        if jsonl_path is not None:
            jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            jsonl_stream = jsonl_path.open("w", encoding="utf-8", newline="\n")
        for source in iter_fiducial_jsonl(input_path):
            if args.max_frames is not None and len(records) >= args.max_frames:
                break
            record = analyzer.process(source)
            records.append(record)
            if jsonl_stream is not None:
                jsonl_stream.write(record.model_dump_json() + "\n")
    finally:
        if jsonl_stream is not None:
            jsonl_stream.close()
    if svg_path is not None:
        save_mapped_trajectory_svg(
            room,
            records,
            svg_path,
            overwrite=args.overwrite,
        )
    summary = analyzer.summary()
    print(
        f"map poses: frames={summary.total_frames} safe={summary.safe_frames} "
        f"blocked={summary.clearance_blocked_frames} outside={summary.outside_frames} "
        f"missing={summary.missing_frames} offline_only=True"
    )
    if jsonl_path is not None:
        print(f"mapped records: {jsonl_path}")
    if svg_path is not None:
        print(f"trajectory preview: {svg_path}")
    localized = (
        summary.safe_frames
        + summary.clearance_blocked_frames
        + summary.outside_frames
    )
    return 0 if summary.total_frames > 0 and localized > 0 else 1


def run_audit_session(args: argparse.Namespace) -> int:
    report = SessionAuditor(args.session).run()
    if args.json:
        print(report.model_dump_json(indent=2))
    else:
        print(
            f"session audit: {'PASS' if report.passed else 'FAIL'} "
            f"offline_only=True dir={report.session_dir}"
        )
        for check in report.checks:
            stream = sys.stderr if check.status is SessionAuditStatus.FAIL else sys.stdout
            print(f"  [{check.status.value}] {check.name}: {check.detail}", file=stream)
    return 0 if report.passed else 1


def _parse_grid_cell(text: str, label: str) -> GridCell:
    pieces = [item.strip() for item in text.split(",")]
    if len(pieces) != 2:
        raise ValueError(f"--{label} must use x,y integer coordinates")
    try:
        x, y = (int(item) for item in pieces)
    except ValueError as exc:
        raise ValueError(f"--{label} must use x,y integer coordinates") from exc
    return GridCell(x=x, y=y)


def run_command(args: argparse.Namespace) -> int:
    config = _app_config(args)
    command = _make_command_interpreter(args, config).interpret(args.text)
    if command.action is ActionType.FOLLOW_PERSON:
        return run_follow(args)
    if command.action is ActionType.FIND_PERSON:
        return run_search(args, "person")
    if command.action is ActionType.FIND_OBJECT:
        assert command.target_class is not None
        return run_search(args, command.target_class)
    drone, _ = _make_drone(args.drone_mode, config)
    safety = SafetyController(drone, config.flight)
    drone.connect()
    exit_code = 0
    try:
        if args.takeoff and command.action is not ActionType.TAKEOFF:
            preflight = safety.execute(
                ActionCommand(action=ActionType.TAKEOFF, source=CommandSource.SYSTEM)
            )
            _print_result("preflight", preflight)
            if not preflight.approved:
                return 1
        result = safety.execute(command)
        _print_result("command", result)
        print(f"state: {drone.get_state()}")
        exit_code = 0 if result.approved else 1
        return exit_code
    finally:
        _orderly_shutdown(drone, safety)


def run_plan(args: argparse.Namespace) -> int:
    config = _app_config(args)
    plan = _make_plan_interpreter(args, config).interpret_plan(args.text)
    _print_plan(plan)
    if not args.execute:
        print("preview only: add --execute to run this plan")
        return 0

    drone, _ = _make_drone(args.drone_mode, config)
    safety = SafetyController(drone, config.flight)
    drone.connect()
    try:
        if args.takeoff:
            preflight = safety.execute(
                ActionCommand(action=ActionType.TAKEOFF, source=CommandSource.SYSTEM)
            )
            _print_result("preflight", preflight)
            if not preflight.approved:
                return 1
        result = PlanExecutor(safety, config.plan).execute(plan)
        _print_plan_result(result)
        print(f"state: {drone.get_state()}")
        return 0 if result.completed else 1
    finally:
        _orderly_shutdown(drone, safety)


def run_follow(args: argparse.Namespace) -> int:
    config = _app_config(args)
    recorder = _make_recorder(args, "follow", config)
    drone, shared_session = _make_drone(args.drone_mode, config)
    camera_name = args.camera or ("tello" if args.drone_mode == "tello" else "webcam")
    camera, camera_only_session = _make_camera(
        camera_name,
        _camera_index(args, config),
        shared_session,
        _calibration_path(args, config),
    )
    safety = SafetyController(drone, config.flight)
    watchdog = FlightWatchdog(safety, config.watchdog, config.flight)
    detector = YOLODetector(
        model_name=args.model or config.vision.model_name,
        confidence=config.vision.confidence,
        target_classes=config.vision.target_classes,
    )
    pipeline = FollowPipeline(
        detector,
        PersonTracker(config.follow),
        FollowPersonAbility(config.follow),
        safety,
    )

    camera_open = False
    session_status = "failed"
    try:
        drone.connect()
        camera.open()
        camera_open = True
        preflight = AutonomousPreflight(
            drone,
            camera,
            detector,
            config.flight,
            config.preflight,
        ).run()
        _print_readiness_report("autonomous preflight", preflight)
        if recorder is not None:
            recorder.record_event("autonomous.preflight", preflight.model_dump(mode="json"))
        if not preflight.ready:
            session_status = "preflight_failed"
            return 1
        if args.takeoff:
            result = safety.execute(
                ActionCommand(action=ActionType.TAKEOFF, source=CommandSource.SYSTEM)
            )
            _print_result("preflight", result)
            _record_safety_event(recorder, "takeoff", result)
            if not result.approved:
                session_status = "takeoff_rejected"
                return 1
        else:
            LOGGER.warning(
                "No --takeoff flag: vision decisions will be shown but flight movement stays rejected"
            )
        exit_code = _follow_loop(
            camera,
            pipeline,
            safety,
            config.follow.control_rate_hz,
            recorder,
            watchdog,
        )
        session_status = "completed" if exit_code == 0 else "ended_with_error"
        return exit_code
    finally:
        # Land while the video link is still available. With a shared Tello
        # session, camera cleanup also closes command transport, so it must be
        # strictly after the orderly landing path.
        _orderly_shutdown(drone, safety, recorder)
        if camera_open:
            try:
                camera.close()
            except Exception as exc:
                LOGGER.error("camera cleanup failed: %s", exc)
        if camera_only_session is not None:
            try:
                camera_only_session.close()
            except Exception as exc:
                LOGGER.error("Tello camera session cleanup failed: %s", exc)
        _close_recorder(recorder, session_status)


def run_search(args: argparse.Namespace, target_class: str) -> int:
    config = _app_config(args)
    recorder = _make_recorder(args, "search", config)
    target = target_class.strip().lower()
    if not target:
        raise ValueError("search target cannot be blank")
    drone, shared_session = _make_drone(args.drone_mode, config)
    camera_name = args.camera or ("tello" if args.drone_mode == "tello" else "webcam")
    camera, camera_only_session = _make_camera(
        camera_name,
        _camera_index(args, config),
        shared_session,
        _calibration_path(args, config),
    )
    safety = SafetyController(drone, config.flight)
    watchdog = FlightWatchdog(safety, config.watchdog, config.flight)
    ability = VisualSearchAbility(target, config.search)
    detector = YOLODetector(
        model_name=args.model or config.vision.model_name,
        confidence=config.vision.confidence,
        target_classes=(target,),
    )
    # Model/class validation happens before connecting or taking off.
    detector.validate_target_classes()
    pipeline = SearchPipeline(
        detector,
        TargetTracker(target, config.follow),
        ability,
        safety,
    )

    camera_open = False
    session_status = "failed"
    try:
        drone.connect()
        camera.open()
        camera_open = True
        preflight = AutonomousPreflight(
            drone,
            camera,
            detector,
            config.flight,
            config.preflight,
        ).run()
        _print_readiness_report("autonomous preflight", preflight)
        if recorder is not None:
            recorder.record_event("autonomous.preflight", preflight.model_dump(mode="json"))
        if not preflight.ready:
            session_status = "preflight_failed"
            return 1
        if args.takeoff:
            result = safety.execute(
                ActionCommand(action=ActionType.TAKEOFF, source=CommandSource.SYSTEM)
            )
            _print_result("preflight", result)
            _record_safety_event(recorder, "takeoff", result)
            if not result.approved:
                session_status = "takeoff_rejected"
                return 1
        else:
            LOGGER.warning(
                "No --takeoff flag: the first scan rotation will be rejected while landed"
            )
        exit_code = _search_loop(
            camera,
            pipeline,
            safety,
            config.search.control_rate_hz,
            recorder,
            watchdog,
        )
        session_status = "completed" if exit_code == 0 else "ended_with_error"
        return exit_code
    finally:
        _orderly_shutdown(drone, safety, recorder)
        if camera_open:
            try:
                camera.close()
            except Exception as exc:
                LOGGER.error("camera cleanup failed: %s", exc)
        if camera_only_session is not None:
            try:
                camera_only_session.close()
            except Exception as exc:
                LOGGER.error("Tello camera session cleanup failed: %s", exc)
        _close_recorder(recorder, session_status)


def run_manual(args: argparse.Namespace) -> int:
    config = _app_config(args)
    recorder = _make_recorder(args, "manual", config)
    drone, shared_session = _make_drone(args.drone_mode, config, manual_rc=True)
    camera_name = args.camera or ("tello" if args.drone_mode == "tello" else "blank")
    camera, camera_only_session = _make_camera(
        camera_name,
        _camera_index(args, config),
        shared_session,
        _calibration_path(args, config),
    )
    safety = SafetyController(drone, config.flight)
    watchdog = FlightWatchdog(safety, config.watchdog, config.flight)
    safety.set_manual_override(True)
    controller = ManualController(safety, config.manual)

    camera_open = False
    session_status = "failed"
    try:
        drone.connect()
        camera.open()
        camera_open = True
        exit_code = _manual_loop(camera, safety, controller, recorder, watchdog)
        session_status = "completed" if exit_code == 0 else "ended_with_error"
        return exit_code
    finally:
        _orderly_shutdown(drone, safety, recorder)
        if camera_open:
            try:
                camera.close()
            except Exception as exc:
                LOGGER.error("camera cleanup failed: %s", exc)
        if camera_only_session is not None:
            try:
                camera_only_session.close()
            except Exception as exc:
                LOGGER.error("Tello camera session cleanup failed: %s", exc)
        _close_recorder(recorder, session_status)


def _follow_loop(
    camera: CameraSource,
    pipeline: FollowPipeline,
    safety: SafetyController,
    control_rate_hz: float,
    recorder: SessionRecorder | None = None,
    watchdog: FlightWatchdog | None = None,
) -> int:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is not installed; install requirements.txt") from exc

    period_s = 1.0 / control_rate_hz
    next_control_at = 0.0
    last_decision: FollowDecision | None = None
    last_executed_action: ActionType | None = None
    last_result: SafetyResult | None = None
    emergency = False
    last_telemetry_sequence = -1
    try:
        while True:
            watchdog_event = watchdog.poll() if watchdog is not None else None
            if watchdog_event is not None:
                _report_watchdog_event(watchdog_event, recorder)
                last_telemetry_sequence = _record_watchdog_telemetry(
                    recorder,
                    watchdog,
                    last_telemetry_sequence,
                )
                return 0 if watchdog_event.safety_result.approved else 1
            last_telemetry_sequence = _record_watchdog_telemetry(
                recorder,
                watchdog,
                last_telemetry_sequence,
            )
            frame = camera.read()
            now = time.monotonic()
            if now >= next_control_at:
                last_decision = pipeline.evaluate(frame)
                command = last_decision.command
                executed_result: SafetyResult | None = None
                repeated_hover = (
                    command.action in {ActionType.STOP, ActionType.HOVER}
                    and last_executed_action in {ActionType.STOP, ActionType.HOVER}
                )
                if not repeated_hover:
                    last_result = safety.execute(command)
                    executed_result = last_result
                    last_executed_action = command.action
                if recorder is not None:
                    recorder.record_event(
                        "follow.control_tick",
                        _follow_event_payload(last_decision, executed_result, repeated_hover),
                    )
                next_control_at = now + period_s

            if last_decision is not None:
                draw_detections(frame, last_decision.detections)
                _draw_status(frame, last_decision, last_result)
            if recorder is not None:
                recorder.record_frame(frame)
            cv2.imshow("Drone AI - E MOTOR CUTOFF, Q/Esc land and quit", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("e"), ord("E")):
                result = safety.execute_emergency()
                _print_result("emergency", result)
                _record_safety_event(recorder, "emergency", result)
                emergency = True
                return 0 if result.approved else 1
            if key in (ord("q"), ord("Q"), 27):
                return 0
    except KeyboardInterrupt:
        LOGGER.info("Ctrl+C received; landing and cleaning up")
        return 0
    finally:
        cv2.destroyAllWindows()
        if emergency:
            LOGGER.warning("Emergency stop was used; normal landing is skipped")


def _search_loop(
    camera: CameraSource,
    pipeline: SearchPipeline,
    safety: SafetyController,
    control_rate_hz: float,
    recorder: SessionRecorder | None = None,
    watchdog: FlightWatchdog | None = None,
) -> int:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is not installed; install requirements.txt") from exc

    period_s = 1.0 / control_rate_hz
    next_control_at = 0.0
    last_tick: SearchTick | None = None
    last_telemetry_sequence = -1
    try:
        while True:
            watchdog_event = watchdog.poll() if watchdog is not None else None
            if watchdog_event is not None:
                _report_watchdog_event(watchdog_event, recorder)
                last_telemetry_sequence = _record_watchdog_telemetry(
                    recorder,
                    watchdog,
                    last_telemetry_sequence,
                )
                return 0 if watchdog_event.safety_result.approved else 1
            last_telemetry_sequence = _record_watchdog_telemetry(
                recorder,
                watchdog,
                last_telemetry_sequence,
            )
            frame = camera.read()
            now = time.monotonic()
            if now >= next_control_at:
                last_tick = pipeline.tick(frame)
                if recorder is not None:
                    recorder.record_event("search.control_tick", _search_event_payload(last_tick))
                next_control_at = now + period_s
                if not last_tick.safety_result.approved:
                    LOGGER.error("[SEARCH] Aborted: %s", last_tick.safety_result.reason)
                    return 1
                if last_tick.decision.terminal:
                    LOGGER.info(
                        "[SEARCH] %s: %s after %.1f degrees",
                        last_tick.decision.state.value.upper(),
                        pipeline.ability.target_class,
                        pipeline.ability.scanned_degrees,
                    )
                    draw_detections(frame, last_tick.detections)
                    _draw_search_status(frame, last_tick, pipeline.ability.target_class)
                    if recorder is not None:
                        recorder.record_frame(frame)
                    cv2.imshow("Drone AI Search - E MOTOR CUTOFF, Q/Esc land and quit", frame)
                    cv2.waitKey(500)
                    return 0 if last_tick.decision.state is SearchState.FOUND else 1

            if last_tick is not None:
                draw_detections(frame, last_tick.detections)
                _draw_search_status(frame, last_tick, pipeline.ability.target_class)
            if recorder is not None:
                recorder.record_frame(frame)
            cv2.imshow("Drone AI Search - E MOTOR CUTOFF, Q/Esc land and quit", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("e"), ord("E")):
                result = safety.execute_emergency()
                _print_result("emergency", result)
                _record_safety_event(recorder, "emergency", result)
                return 0 if result.approved else 1
            if key in (ord("q"), ord("Q"), 27):
                return 0
    except KeyboardInterrupt:
        LOGGER.info("Ctrl+C received; landing and cleaning up")
        return 0
    finally:
        cv2.destroyAllWindows()


def _manual_loop(
    camera: CameraSource,
    safety: SafetyController,
    controller: ManualController,
    recorder: SessionRecorder | None = None,
    watchdog: FlightWatchdog | None = None,
) -> int:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is not installed; install requirements.txt") from exc

    window = "Drone AI Manual - T takeoff, L land, H hover, E MOTOR CUTOFF, Q quit"
    last_telemetry_sequence = -1
    window_opened = False
    held_motion_key: str | None = None
    last_held_dispatch_at = 0.0
    try:
        while True:
            watchdog_event = watchdog.poll() if watchdog is not None else None
            if watchdog_event is not None:
                _report_watchdog_event(watchdog_event, recorder)
                last_telemetry_sequence = _record_watchdog_telemetry(
                    recorder,
                    watchdog,
                    last_telemetry_sequence,
                )
                return 0 if watchdog_event.safety_result.approved else 1
            last_telemetry_sequence = _record_watchdog_telemetry(
                recorder,
                watchdog,
                last_telemetry_sequence,
            )
            frame = camera.read()
            _draw_manual_status(frame, safety)
            if recorder is not None:
                recorder.record_frame(frame)
            if window_opened and _opencv_window_was_closed(cv2, window):
                if recorder is not None:
                    recorder.record_event("manual.quit", {"window_closed": True})
                return 0
            cv2.imshow(window, frame)
            window_opened = True
            key_code = cv2.waitKey(25) & 0xFF

            pressed_keys = _windows_pressed_manual_keys(window)
            if pressed_keys is not None:
                now = time.monotonic()
                active_key = next(iter(pressed_keys)) if len(pressed_keys) == 1 else None
                if active_key != held_motion_key:
                    if held_motion_key is not None:
                        _dispatch_manual_key(controller, "h", recorder)
                    held_motion_key = active_key
                    if active_key is not None:
                        _dispatch_manual_key(controller, active_key, recorder)
                        last_held_dispatch_at = now
                elif active_key is not None and now - last_held_dispatch_at >= 0.5:
                    _dispatch_manual_key(controller, active_key, recorder)
                    last_held_dispatch_at = now

            if key_code in (ord("q"), ord("Q"), 27):
                if recorder is not None:
                    recorder.record_event("manual.quit", {"key_code": key_code})
                return 0
            if key_code == 255:
                continue
            key = chr(key_code)
            if pressed_keys is not None and (
                key.lower() in {"w", "s", "a", "d", " ", "j", "k"}
                or key_code == 17
            ):
                continue
            if key_code == 17:
                key = "ctrl"
            event = _dispatch_manual_key(controller, key, recorder)
            if event.command is not None and event.command.action is ActionType.EMERGENCY_STOP:
                result = event.safety_result
                return 0 if result is not None and result.approved else 1
    except KeyboardInterrupt:
        LOGGER.info("Ctrl+C received; landing and cleaning up")
        return 0
    finally:
        cv2.destroyAllWindows()


def _opencv_window_was_closed(cv2: Any, window: str) -> bool:
    """Return true when a user closed a live OpenCV window via its X button."""

    get_property = getattr(cv2, "getWindowProperty", None)
    visible_property = getattr(cv2, "WND_PROP_VISIBLE", None)
    if get_property is None or visible_property is None:
        return False
    try:
        return float(get_property(window, visible_property)) < 1.0
    except Exception:
        # Some OpenCV backends do not implement window visibility. Keyboard
        # quit and Ctrl+C remain available on those backends.
        return False


def _dispatch_manual_key(
    controller: ManualController,
    key: str,
    recorder: SessionRecorder | None,
) -> ManualEvent:
    event = controller.handle_key(key)
    _report_manual_event(event)
    if recorder is not None and event.command is not None:
        recorder.record_event("manual.command", _manual_event_payload(event))
    return event


def _windows_pressed_manual_keys(window: str) -> set[str] | None:
    """Poll held flight keys only while the OpenCV window has focus."""

    if sys.platform != "win32":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        user32.FindWindowW.argtypes = (wintypes.LPCWSTR, wintypes.LPCWSTR)
        user32.FindWindowW.restype = wintypes.HWND
        user32.GetForegroundWindow.restype = wintypes.HWND
        user32.GetAsyncKeyState.argtypes = (ctypes.c_int,)
        user32.GetAsyncKeyState.restype = ctypes.c_short
        handle = user32.FindWindowW(None, window)
        if not handle:
            return None
        if user32.GetForegroundWindow() != handle:
            return set()
        key_codes = {
            "w": 0x57,
            "s": 0x53,
            "a": 0x41,
            "d": 0x44,
            " ": 0x20,
            "ctrl": 0x11,
            "j": 0x4A,
            "k": 0x4B,
        }
        return {
            key
            for key, virtual_key in key_codes.items()
            if user32.GetAsyncKeyState(virtual_key) & 0x8000
        }
    except Exception:
        return None


def _draw_status(frame: Any, decision: FollowDecision, result: SafetyResult | None) -> None:
    import cv2

    tracking = decision.tracking
    status = result.status.value if result else "SUPPRESSED"
    text = (
        f"target={tracking.horizontal_position.value}/{tracking.distance.value} "
        f"action={decision.command.action.value} safety={status}"
    )
    cv2.putText(frame, text, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)


def _draw_search_status(frame: Any, tick: SearchTick, target_class: str) -> None:
    import cv2

    text = (
        f"search={target_class} state={tick.decision.state.value} "
        f"scanned={tick.decision.scanned_degrees:.0f}deg "
        f"safety={tick.safety_result.status.value}"
    )
    cv2.putText(frame, text, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)


def _draw_manual_status(frame: Any, safety: SafetyController) -> None:
    import cv2

    try:
        state = safety.drone.get_state()
        telemetry = (
            f"connected={state.is_connected} flying={state.is_flying} "
            f"battery={state.battery_percent}% altitude={state.altitude_m}m "
            f"yaw={state.yaw_degrees:.1f}deg"
        )
    except Exception as exc:
        telemetry = f"TELEMETRY ERROR: {exc}"
    controls = "HOLD W/S forward/back  A/D turn  Space/Ctrl up/down  J/K strafe  H hover"
    cv2.putText(frame, telemetry, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    cv2.putText(frame, controls, (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)


def _draw_replay_status(frame: Any, analyzed: ReplayFrame) -> None:
    import cv2

    tracking = analyzed.tracking
    text = (
        f"frame={analyzed.record.frame_index} t={analyzed.record.timestamp_s:.2f}s "
        f"target={tracking.horizontal_position.value}/{tracking.distance.value} "
        f"visible={tracking.visible}"
    )
    cv2.putText(frame, text, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)


def _open_video_writer(path: Path, frame: Any, fps: float) -> Any:
    import cv2

    shape = getattr(frame, "shape", None)
    if not isinstance(shape, tuple) or len(shape) != 3 or shape[2] != 3:
        raise RuntimeError("cannot write annotated video from a non-BGR frame")
    height, width = shape[:2]
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        writer.release()
        raise RuntimeError(f"could not create annotated video: {path}")
    return writer


def _report_manual_event(event: ManualEvent) -> None:
    if event.suppressed:
        LOGGER.debug("[MANUAL] %s", event.reason)
    elif event.safety_result is not None:
        action = event.command.action.value if event.command is not None else "unknown"
        _print_result(f"manual {action}", event.safety_result)


def _follow_event_payload(
    decision: FollowDecision,
    result: SafetyResult | None,
    suppressed: bool,
) -> dict[str, Any]:
    return {
        "detections": [_detection_payload(item) for item in decision.detections],
        "tracking": _tracking_payload(decision.tracking),
        "command": decision.command.model_dump(mode="json"),
        "safety_result": result.model_dump(mode="json") if result is not None else None,
        "suppressed_repeated_hover": suppressed,
    }


def _search_event_payload(tick: SearchTick) -> dict[str, Any]:
    return {
        "detections": [_detection_payload(item) for item in tick.detections],
        "tracking": _tracking_payload(tick.tracking),
        "decision": {
            "state": tick.decision.state.value,
            "scanned_degrees": tick.decision.scanned_degrees,
            "terminal": tick.decision.terminal,
            "command": tick.decision.command.model_dump(mode="json"),
        },
        "safety_result": tick.safety_result.model_dump(mode="json"),
    }


def _manual_event_payload(event: ManualEvent) -> dict[str, Any]:
    assert event.command is not None
    return {
        "key": event.key,
        "command": event.command.model_dump(mode="json"),
        "safety_result": (
            event.safety_result.model_dump(mode="json")
            if event.safety_result is not None
            else None
        ),
        "suppressed": event.suppressed,
        "reason": event.reason,
    }


def _detection_payload(detection: Any) -> dict[str, Any]:
    return {
        "class_name": detection.class_name,
        "confidence": detection.confidence,
        "bbox": list(detection.bbox),
    }


def _tracking_payload(tracking: Any) -> dict[str, Any]:
    return {
        "visible": tracking.visible,
        "horizontal_position": tracking.horizontal_position.value,
        "distance": tracking.distance.value,
        "horizontal_error": tracking.horizontal_error,
        "vertical_error": tracking.vertical_error,
        "size_ratio": tracking.size_ratio,
    }


def _record_safety_event(
    recorder: SessionRecorder | None,
    kind: str,
    result: SafetyResult,
) -> None:
    if recorder is not None:
        recorder.record_event(kind, {"safety_result": result.model_dump(mode="json")})


def _report_watchdog_event(
    event: WatchdogEvent,
    recorder: SessionRecorder | None,
) -> None:
    _print_result(f"watchdog {event.reason.value}", event.safety_result)
    if recorder is not None:
        recorder.record_event("watchdog.landing", event.model_dump(mode="json"))


def _record_watchdog_telemetry(
    recorder: SessionRecorder | None,
    watchdog: FlightWatchdog | None,
    last_sequence: int,
) -> int:
    if recorder is None or watchdog is None:
        return last_sequence
    snapshot = watchdog.last_snapshot
    if snapshot is None or snapshot.sequence == last_sequence:
        return last_sequence
    recorder.record_event("telemetry.snapshot", snapshot.model_dump(mode="json"))
    return snapshot.sequence


def _make_recorder(
    args: argparse.Namespace,
    mode: str,
    config: AppConfig,
) -> SessionRecorder | None:
    record_dir = getattr(args, "record_dir", None)
    if not record_dir:
        return None
    recorder = SessionRecorder(
        record_dir,
        mode,
        provenance=_make_session_provenance(args, mode, config),
    )
    print(f"recording session: {recorder.session_dir}")
    return recorder


def _make_session_provenance(
    args: argparse.Namespace,
    mode: str,
    config: AppConfig,
) -> SessionProvenance:
    drone_mode = getattr(args, "drone_mode", "simulator")
    requested_camera = getattr(args, "camera", None)
    camera_source = requested_camera or (
        "tello"
        if drone_mode == "tello"
        else ("blank" if mode == "manual" else "webcam")
    )
    model_name = None
    if mode in {"follow", "search"}:
        model_name = getattr(args, "model", None) or config.vision.model_name
    calibration_path = _calibration_path(args, config)
    calibration_file = str(calibration_path) if calibration_path is not None else None
    raw_profile_file = getattr(args, "config", None)
    profile_file = str(raw_profile_file) if raw_profile_file is not None else None
    canonical_config = json.dumps(
        asdict(config),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    requirements_lock = Path(__file__).parents[1] / "requirements-lock.txt"
    return SessionProvenance(
        created_at=datetime.now(timezone.utc),
        mode=mode,
        python_version=platform.python_version(),
        platform=platform.platform(),
        drone_mode=drone_mode,
        camera_source=camera_source,
        model_name=model_name,
        model_sha256=_sha256_if_file(model_name),
        calibration_file=calibration_file,
        calibration_sha256=_sha256_if_file(calibration_file),
        profile_file=profile_file,
        profile_sha256=_sha256_if_file(profile_file),
        requirements_lock_sha256=_sha256_if_file(requirements_lock),
        effective_config_sha256=hashlib.sha256(canonical_config).hexdigest(),
    )


def _sha256_if_file(path: str | Path | None) -> str | None:
    if path is None:
        return None
    source = Path(path)
    if not source.is_file():
        return None
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _close_recorder(recorder: SessionRecorder | None, status: str) -> None:
    if recorder is None:
        return
    manifest = recorder.close(status=status)
    print(
        f"recording closed: {recorder.session_dir} "
        f"events={manifest.event_count} frames={manifest.frame_count} status={manifest.status}"
    )


def _make_drone(
    mode: str,
    config: AppConfig,
    *,
    manual_rc: bool = False,
) -> tuple[DroneController, TelloSession | None]:
    if mode == "tello":
        session = TelloSession()
        return (
            TelloDroneController(
                session,
                config.flight,
                rc_control_mode=manual_rc,
                keepalive_interval_s=0.5 if manual_rc else 5.0,
                rc_velocity=config.manual.tello_rc_velocity,
                rc_step_duration_s=config.manual.tello_rc_step_duration_s,
            ),
            session,
        )
    return SimulatedDrone(config.flight), None


def _make_command_interpreter(args: argparse.Namespace, config: AppConfig) -> Any:
    rules = RuleCommandInterpreter()
    if args.interpreter == "rule":
        return rules
    ollama = OllamaCommandInterpreter(_make_llm_config(args, config))
    if args.interpreter == "hybrid":
        return HybridCommandInterpreter(ollama, rules)
    return ollama


def _make_plan_interpreter(args: argparse.Namespace, config: AppConfig) -> Any:
    rules = RuleBasedPlanInterpreter()
    if args.planner == "rule":
        return rules
    ollama = OllamaPlanInterpreter(_make_llm_config(args, config))
    if args.planner == "hybrid":
        return HybridPlanInterpreter(ollama, rules)
    return ollama


def _app_config(args: argparse.Namespace) -> AppConfig:
    return load_app_config(getattr(args, "config", None))


def _camera_index(args: argparse.Namespace, config: AppConfig) -> int:
    override = getattr(args, "camera_index", None)
    camera_index = config.vision.camera_index if override is None else override
    if camera_index < 0:
        raise ValueError("camera index cannot be negative")
    return camera_index


def _calibration_path(args: argparse.Namespace, config: AppConfig) -> Path | None:
    override = getattr(args, "calibration", None)
    configured = config.vision.calibration_file if override is None else override
    normalized = configured.strip()
    return Path(normalized) if normalized else None


def _make_llm_config(args: argparse.Namespace, config: AppConfig) -> LLMConfig:
    return LLMConfig(
        ollama_url=args.ollama_url or config.llm.ollama_url,
        model_name=args.ollama_model or config.llm.model_name,
        request_timeout_s=(
            args.ollama_timeout
            if args.ollama_timeout is not None
            else config.llm.request_timeout_s
        ),
        max_command_characters=config.llm.max_command_characters,
    )


def _make_camera(
    camera_name: str,
    camera_index: int,
    shared_session: TelloSession | None,
    calibration_path: Path | None = None,
) -> tuple[CameraSource, TelloSession | None]:
    if camera_name == "tello":
        session = shared_session or TelloSession()
        camera: CameraSource = TelloCamera(session)
        camera_only_session = None if shared_session is not None else session
    elif camera_name == "blank":
        camera = BlankCamera()
        camera_only_session = None
    else:
        camera = LaptopWebcamCamera(camera_index)
        camera_only_session = None
    if calibration_path is not None:
        camera = UndistortingCamera(camera, load_calibration(calibration_path))
    return camera, camera_only_session


def _orderly_shutdown(
    drone: DroneController,
    safety: SafetyController,
    recorder: SessionRecorder | None = None,
) -> None:
    try:
        state = drone.get_state()
    except Exception as exc:
        LOGGER.error("could not inspect drone during shutdown: %s", exc)
        state = None
    if state is not None and state.is_connected and state.is_flying:
        result = safety.execute(ActionCommand(action=ActionType.LAND, source=CommandSource.SYSTEM))
        _print_result("shutdown landing", result)
        if recorder is not None:
            try:
                _record_safety_event(recorder, "shutdown.land", result)
            except Exception as exc:
                LOGGER.error("could not record shutdown landing evidence: %s", exc)
    try:
        latest = drone.get_state()
        if latest.is_connected and not latest.is_flying:
            drone.disconnect()
    except Exception as exc:
        LOGGER.error("drone cleanup failed: %s", exc)


def _print_result(label: str, result: SafetyResult) -> None:
    stream = sys.stdout if result.status is SafetyStatus.APPROVED else sys.stderr
    print(f"{label}: {result.status.value} - {result.reason}", file=stream)


def _print_readiness_report(label: str, report: ReadinessReport) -> None:
    stream = sys.stdout if report.ready else sys.stderr
    print(f"{label}: {'READY' if report.ready else 'NOT READY'}", file=stream)
    for check in report.checks:
        print(f"  [{check.status.value}] {check.name}: {check.detail}", file=stream)


def _print_plan(plan: ActionPlan) -> None:
    print(f"plan: source={plan.source.value}, steps={len(plan.steps)}")
    for index, step in enumerate(plan.steps, start=1):
        parameter = ""
        if step.distance_m is not None:
            parameter = f" {step.distance_m:g}m"
        elif step.degrees is not None:
            parameter = f" {step.degrees:g}deg"
        print(f"  {index}. {step.action.value}{parameter}")


def _print_plan_result(result: PlanExecutionResult) -> None:
    stream = sys.stdout if result.status is PlanExecutionStatus.COMPLETED else sys.stderr
    print(f"plan execution: {result.status.value} - {result.reason}", file=stream)
    for item in result.step_results:
        print(
            f"  step {item.index + 1}: {item.step.action.value} -> "
            f"{item.safety_result.status.value}",
            file=stream,
        )
    if result.stabilization_result is not None:
        print(
            f"  stabilization: {result.stabilization_result.status.value} - "
            f"{result.stabilization_result.reason}",
            file=stream,
        )
