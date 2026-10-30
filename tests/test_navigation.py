from __future__ import annotations

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest
from pydantic import ValidationError

import drone_ai.cli as cli
from drone_ai.commands import ActionType
from drone_ai.config import PlanConfig
from drone_ai.controllers.base import DroneState
from drone_ai.navigation import (
    GridCell,
    NavigationError,
    NavigationRoute,
    RouteExecutionAudit,
    RoomMap,
    RoomRoutePlanner,
    audit_simulated_route,
    cell_center_world,
    load_room_map,
    save_navigation_route,
    save_route_svg,
)


ROOT = Path(__file__).parents[1]


def test_example_room_route_avoids_clearance_inflated_wall() -> None:
    room = load_room_map(ROOT / "maps" / "example-room.json")
    planner = RoomRoutePlanner(room)
    route = planner.plan(GridCell(x=2, y=2), GridCell(x=9, y=7))
    blocked = planner.inflated_blocked_cells
    assert route.simulator_only
    assert all((cell.x, cell.y) not in blocked for cell in route.path)
    assert route.path[0] == GridCell(x=2, y=2)
    assert route.path[-1] == GridCell(x=9, y=7)
    assert route.distance_m == pytest.approx(2.8)
    assert len(route.plan.steps) <= 8
    assert all(step.action in {ActionType.MOVE_FORWARD, ActionType.ROTATE_RIGHT} for step in route.plan.steps)


def test_navigation_uses_shortest_rotation_from_initial_yaw() -> None:
    room = RoomMap(
        name="open",
        width_cells=6,
        height_cells=6,
        resolution_m=0.2,
        clearance_cells=0,
    )
    route = RoomRoutePlanner(room).plan(
        GridCell(x=1, y=1),
        GridCell(x=3, y=1),
        start_yaw_degrees=270.0,
    )
    assert route.plan.steps[0].action is ActionType.ROTATE_RIGHT
    assert route.plan.steps[0].degrees == 180.0
    assert route.plan.steps[1].action is ActionType.MOVE_FORWARD
    assert route.plan.steps[1].distance_m == pytest.approx(0.4)


def test_rotated_registered_grid_updates_world_heading_and_goal_pose() -> None:
    room = RoomMap(
        name="rotated",
        width_cells=6,
        height_cells=6,
        resolution_m=0.2,
        clearance_cells=0,
        world_origin_x_m=10.0,
        world_origin_y_m=20.0,
        grid_yaw_degrees=90.0,
    )
    route = RoomRoutePlanner(room).plan(
        GridCell(x=1, y=1),
        GridCell(x=1, y=3),
        start_yaw_degrees=90.0,
    )
    assert route.plan.steps[0].action is ActionType.MOVE_FORWARD
    assert route.plan.steps[0].distance_m == pytest.approx(0.4)
    start_world = cell_center_world(room, route.start)
    goal_world = cell_center_world(room, route.goal)
    assert start_world == pytest.approx((10.3, 19.7))
    assert goal_world == pytest.approx((10.7, 19.7))


def test_navigation_rejects_no_path_and_blocked_endpoint() -> None:
    wall = tuple(GridCell(x=3, y=y) for y in range(7))
    room = RoomMap(
        name="split",
        width_cells=7,
        height_cells=7,
        resolution_m=0.2,
        blocked_cells=wall,
        clearance_cells=0,
    )
    planner = RoomRoutePlanner(room)
    with pytest.raises(NavigationError, match="no clearance-safe path"):
        planner.plan(GridCell(x=1, y=3), GridCell(x=5, y=3))
    with pytest.raises(NavigationError, match="start cell is blocked"):
        planner.plan(GridCell(x=3, y=3), GridCell(x=5, y=3))


def test_navigation_rejects_route_over_plan_distance_budget() -> None:
    room = RoomMap(
        name="long",
        width_cells=40,
        height_cells=5,
        resolution_m=0.2,
        clearance_cells=0,
    )
    planner = RoomRoutePlanner(room, plan_config=PlanConfig(max_total_distance_m=6.0))
    with pytest.raises(NavigationError, match="route distance 7.8m"):
        planner.plan(GridCell(x=0, y=2), GridCell(x=39, y=2))


def test_route_schema_cannot_remove_simulator_lockout() -> None:
    room = RoomMap(
        name="open",
        width_cells=5,
        height_cells=5,
        resolution_m=0.2,
        clearance_cells=0,
    )
    route = RoomRoutePlanner(room).plan(GridCell(x=1, y=1), GridCell(x=1, y=2))
    payload = route.model_dump()
    payload["simulator_only"] = False
    with pytest.raises(ValidationError, match="simulator-only"):
        NavigationRoute.model_validate(payload)


def navigate_args(*, execute: bool, takeoff: bool) -> argparse.Namespace:
    return argparse.Namespace(
        map=str(ROOT / "maps" / "example-room.json"),
        start="2,2",
        goal="9,7",
        start_yaw=0.0,
        svg=None,
        route_json=None,
        overwrite=False,
        execute=execute,
        takeoff=takeoff,
        config=None,
    )


def test_navigation_preview_never_constructs_simulator(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        cli,
        "SimulatedDrone",
        lambda *args, **kwargs: pytest.fail("preview must not construct a drone"),
    )
    assert cli.run_navigate(navigate_args(execute=False, takeoff=False)) == 0


def test_navigation_executes_only_in_simulator_with_takeoff() -> None:
    assert cli.run_navigate(navigate_args(execute=True, takeoff=True)) == 0


def test_navigation_execution_without_takeoff_is_rejected_by_safety() -> None:
    assert cli.run_navigate(navigate_args(execute=True, takeoff=False)) == 1


def test_route_execution_audit_checks_map_position_and_final_heading() -> None:
    room = RoomMap(
        name="audit-room",
        width_cells=6,
        height_cells=6,
        resolution_m=0.2,
        clearance_cells=0,
    )
    route = RoomRoutePlanner(room).plan(GridCell(x=1, y=1), GridCell(x=3, y=1))
    matching = DroneState(
        x_m=0.7,
        y_m=0.3,
        altitude_m=1.0,
        yaw_degrees=90.0,
        is_flying=True,
        is_connected=True,
        battery_percent=100,
        speed_mps=0.2,
    )
    audit = audit_simulated_route(room, route, matching)
    assert audit.passed is True
    assert audit.position_error_m == pytest.approx(0.0)
    assert audit.yaw_error_degrees == pytest.approx(0.0)

    wrong_position = DroneState(
        x_m=0.9,
        y_m=0.3,
        altitude_m=1.0,
        yaw_degrees=90.0,
        is_flying=True,
        is_connected=True,
        battery_percent=100,
        speed_mps=0.2,
    )
    failed = audit_simulated_route(room, route, wrong_position)
    assert failed.passed is False
    assert failed.position_error_m == pytest.approx(0.2)


def test_route_audit_schema_cannot_fake_success_or_remove_lockout() -> None:
    payload = {
        "expected_x_m": 1.0,
        "expected_y_m": 1.0,
        "expected_yaw_degrees": 0.0,
        "actual_x_m": 2.0,
        "actual_y_m": 1.0,
        "actual_yaw_degrees": 0.0,
        "position_error_m": 1.0,
        "yaw_error_degrees": 0.0,
        "position_tolerance_m": 0.001,
        "yaw_tolerance_degrees": 0.001,
        "passed": True,
        "simulator_only": True,
    }
    with pytest.raises(ValidationError, match="passed flag"):
        RouteExecutionAudit.model_validate(payload)
    payload["passed"] = False
    payload["simulator_only"] = False
    with pytest.raises(ValidationError, match="simulator-only"):
        RouteExecutionAudit.model_validate(payload)


def test_route_artifacts_show_clearance_path_and_preserve_strict_schema(
    tmp_path: Path,
) -> None:
    room = load_room_map(ROOT / "maps" / "example-room.json")
    planner = RoomRoutePlanner(room)
    route = planner.plan(GridCell(x=2, y=2), GridCell(x=9, y=7))
    svg_path = save_route_svg(room, route, tmp_path / "route.svg")
    json_path = save_navigation_route(route, tmp_path / "route.json")

    root = ET.parse(svg_path).getroot()
    namespace = {"svg": "http://www.w3.org/2000/svg"}
    raw = root.findall("svg:rect[@class='blocked-cell']", namespace)
    inflated = root.findall("svg:rect[@class='inflated-clearance']", namespace)
    route_lines = root.findall("svg:polyline[@class='route']", namespace)
    starts = root.findall("svg:circle[@class='start']", namespace)
    goals = root.findall("svg:circle[@class='goal']", namespace)
    assert len(raw) == len(room.blocked_cells)
    assert len(inflated) == len(planner.inflated_blocked_cells) - len(room.blocked_cells)
    assert len(route_lines) == len(starts) == len(goals) == 1
    assert starts[0].attrib["data-cell"] == "2,2"
    assert goals[0].attrib["data-cell"] == "9,7"

    restored = NavigationRoute.model_validate_json(json_path.read_text(encoding="utf-8"))
    assert restored == route
    assert restored.simulator_only is True


def test_route_artifacts_validate_extensions_and_refuse_overwrite(tmp_path: Path) -> None:
    room = RoomMap(
        name="open",
        width_cells=5,
        height_cells=5,
        resolution_m=0.2,
        clearance_cells=0,
    )
    route = RoomRoutePlanner(room).plan(GridCell(x=1, y=1), GridCell(x=2, y=1))
    svg_path = save_route_svg(room, route, tmp_path / "route.svg")
    json_path = save_navigation_route(route, tmp_path / "route.json")
    with pytest.raises(NavigationError, match="already exists"):
        save_route_svg(room, route, svg_path)
    with pytest.raises(NavigationError, match="already exists"):
        save_navigation_route(route, json_path)
    with pytest.raises(NavigationError, match=".svg extension"):
        save_route_svg(room, route, tmp_path / "route.png")
    with pytest.raises(NavigationError, match=".json extension"):
        save_navigation_route(route, tmp_path / "route.txt")


def test_navigation_cli_preview_writes_artifacts_without_flight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = navigate_args(execute=False, takeoff=False)
    args.svg = str(tmp_path / "preview.svg")
    args.route_json = str(tmp_path / "route.json")
    monkeypatch.setattr(
        cli,
        "SimulatedDrone",
        lambda *args, **kwargs: pytest.fail("artifact preview must not construct a drone"),
    )
    assert cli.run_navigate(args) == 0
    assert Path(args.svg).is_file()
    assert Path(args.route_json).is_file()
    parsed = cli.build_argument_parser().parse_args(
        [
            "navigate",
            "map.json",
            "--start",
            "1,1",
            "--goal",
            "2,2",
            "--svg",
            "route.svg",
        ]
    )
    assert not hasattr(parsed, "drone_mode")
