"""Static-map route planning that is intentionally simulator-only."""

from __future__ import annotations

from datetime import datetime, timezone
import heapq
import math
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from drone_ai.commands import ActionType, CommandSource
from drone_ai.config import FlightConfig, PlanConfig
from drone_ai.controllers.base import DroneState
from drone_ai.planning.models import ActionPlan, PlannedAction


class NavigationError(RuntimeError):
    pass


class GridCell(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    x: int = Field(ge=0)
    y: int = Field(ge=0)


class RoomMap(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    name: str = Field(min_length=1, max_length=100)
    width_cells: int = Field(ge=3, le=100)
    height_cells: int = Field(ge=3, le=100)
    resolution_m: float = Field(ge=0.2, le=1.0)
    blocked_cells: tuple[GridCell, ...] = ()
    clearance_cells: int = Field(default=1, ge=0, le=5)
    world_origin_x_m: float = 0.0
    world_origin_y_m: float = 0.0
    grid_yaw_degrees: float = Field(default=0.0, ge=0.0, lt=360.0)

    @field_validator(
        "resolution_m",
        "world_origin_x_m",
        "world_origin_y_m",
        "grid_yaw_degrees",
    )
    @classmethod
    def finite_map_geometry(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("room-map geometry must be finite")
        return value

    @model_validator(mode="after")
    def validate_cells(self) -> "RoomMap":
        coordinates = [(cell.x, cell.y) for cell in self.blocked_cells]
        if len(coordinates) != len(set(coordinates)):
            raise ValueError("blocked_cells cannot contain duplicates")
        invalid = [
            cell
            for cell in self.blocked_cells
            if cell.x >= self.width_cells or cell.y >= self.height_cells
        ]
        if invalid:
            raise ValueError("blocked cell is outside room bounds")
        if self.clearance_cells * 2 >= min(self.width_cells, self.height_cells):
            raise ValueError("clearance consumes the entire room")
        return self


class NavigationRoute(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    room_name: str
    start: GridCell
    goal: GridCell
    path: tuple[GridCell, ...]
    distance_m: float = Field(ge=0.0)
    start_yaw_degrees: float
    plan: ActionPlan
    simulator_only: bool = True

    @model_validator(mode="after")
    def preserve_hardware_lockout(self) -> "NavigationRoute":
        if not self.simulator_only:
            raise ValueError("static navigation route must remain simulator-only")
        return self


class RouteExecutionAudit(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    expected_x_m: float
    expected_y_m: float
    expected_yaw_degrees: float
    actual_x_m: float
    actual_y_m: float
    actual_yaw_degrees: float
    position_error_m: float = Field(ge=0.0)
    yaw_error_degrees: float = Field(ge=0.0, le=180.0)
    position_tolerance_m: float = Field(gt=0.0)
    yaw_tolerance_degrees: float = Field(gt=0.0)
    passed: bool
    simulator_only: bool = True

    @model_validator(mode="after")
    def validate_audit_truth(self) -> "RouteExecutionAudit":
        numeric = (
            self.expected_x_m,
            self.expected_y_m,
            self.expected_yaw_degrees,
            self.actual_x_m,
            self.actual_y_m,
            self.actual_yaw_degrees,
            self.position_error_m,
            self.yaw_error_degrees,
            self.position_tolerance_m,
            self.yaw_tolerance_degrees,
        )
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("route audit values must be finite")
        expected_pass = (
            self.position_error_m <= self.position_tolerance_m
            and self.yaw_error_degrees <= self.yaw_tolerance_degrees
        )
        if self.passed != expected_pass:
            raise ValueError("route audit passed flag must match its measured errors")
        if not self.simulator_only:
            raise ValueError("route execution audit must remain simulator-only")
        return self


def load_room_map(path: str | Path) -> RoomMap:
    source = Path(path)
    if not source.is_file():
        raise NavigationError(f"room map does not exist: {source}")
    try:
        return RoomMap.model_validate_json(source.read_text(encoding="utf-8"))
    except Exception as exc:
        raise NavigationError(f"invalid room map {source}: {exc}") from exc


def grid_to_world(
    room: RoomMap,
    grid_x_cells: float,
    grid_y_cells: float,
) -> tuple[float, float]:
    if not math.isfinite(grid_x_cells) or not math.isfinite(grid_y_cells):
        raise NavigationError("grid coordinates must be finite")
    local_x = grid_x_cells * room.resolution_m
    local_y = grid_y_cells * room.resolution_m
    yaw = math.radians(room.grid_yaw_degrees)
    return (
        room.world_origin_x_m + math.cos(yaw) * local_x + math.sin(yaw) * local_y,
        room.world_origin_y_m - math.sin(yaw) * local_x + math.cos(yaw) * local_y,
    )


def world_to_grid(
    room: RoomMap,
    world_x_m: float,
    world_y_m: float,
) -> tuple[float, float]:
    if not math.isfinite(world_x_m) or not math.isfinite(world_y_m):
        raise NavigationError("world coordinates must be finite")
    delta_x = world_x_m - room.world_origin_x_m
    delta_y = world_y_m - room.world_origin_y_m
    yaw = math.radians(room.grid_yaw_degrees)
    local_x = math.cos(yaw) * delta_x - math.sin(yaw) * delta_y
    local_y = math.sin(yaw) * delta_x + math.cos(yaw) * delta_y
    return local_x / room.resolution_m, local_y / room.resolution_m


def cell_center_world(room: RoomMap, cell: GridCell) -> tuple[float, float]:
    return grid_to_world(room, cell.x + 0.5, cell.y + 0.5)


class RoomRoutePlanner:
    DIRECTIONS: tuple[tuple[int, int, float], ...] = (
        (0, 1, 0.0),
        (1, 0, 90.0),
        (0, -1, 180.0),
        (-1, 0, 270.0),
    )

    def __init__(
        self,
        room: RoomMap,
        flight_config: FlightConfig | None = None,
        plan_config: PlanConfig | None = None,
    ) -> None:
        self.room = room
        self.flight_config = flight_config or FlightConfig()
        self.plan_config = plan_config or PlanConfig()
        self._blocked = self._inflate_obstacles()

    @property
    def inflated_blocked_cells(self) -> frozenset[tuple[int, int]]:
        return frozenset(self._blocked)

    def plan(
        self,
        start: GridCell,
        goal: GridCell,
        *,
        start_yaw_degrees: float = 0.0,
    ) -> NavigationRoute:
        if not math.isfinite(start_yaw_degrees):
            raise NavigationError("start yaw must be finite")
        self._validate_endpoint(start, "start")
        self._validate_endpoint(goal, "goal")
        path_coordinates = self._find_path(
            (start.x, start.y),
            (goal.x, goal.y),
            start_yaw_degrees,
        )
        steps = self._path_to_actions(path_coordinates, start_yaw_degrees)
        if not steps:
            steps = [PlannedAction(action=ActionType.STOP)]
        if len(steps) > self.plan_config.max_steps:
            raise NavigationError(
                f"route needs {len(steps)} actions, exceeding the {self.plan_config.max_steps}-step budget"
            )
        total_distance = sum(step.distance_m or 0.0 for step in steps)
        if total_distance > self.plan_config.max_total_distance_m + 1e-9:
            raise NavigationError(
                f"route distance {total_distance:g}m exceeds the "
                f"{self.plan_config.max_total_distance_m:g}m plan budget"
            )
        total_rotation = sum(step.degrees or 0.0 for step in steps)
        if total_rotation > self.plan_config.max_total_rotation_degrees + 1e-9:
            raise NavigationError(
                f"route rotation {total_rotation:g} degrees exceeds the "
                f"{self.plan_config.max_total_rotation_degrees:g}-degree plan budget"
            )
        plan = ActionPlan(
            original_text=(
                f"simulator route {self.room.name}: ({start.x},{start.y}) to ({goal.x},{goal.y})"
            ),
            source=CommandSource.MANUAL,
            steps=tuple(steps),
            created_at=datetime.now(timezone.utc),
        )
        return NavigationRoute(
            room_name=self.room.name,
            start=start,
            goal=goal,
            path=tuple(GridCell(x=x, y=y) for x, y in path_coordinates),
            distance_m=total_distance,
            start_yaw_degrees=start_yaw_degrees % 360.0,
            plan=plan,
            simulator_only=True,
        )

    def _inflate_obstacles(self) -> set[tuple[int, int]]:
        clearance = self.room.clearance_cells
        blocked: set[tuple[int, int]] = set()
        for x in range(self.room.width_cells):
            for y in range(self.room.height_cells):
                if (
                    x < clearance
                    or y < clearance
                    or x >= self.room.width_cells - clearance
                    or y >= self.room.height_cells - clearance
                ):
                    blocked.add((x, y))
        for cell in self.room.blocked_cells:
            for dx in range(-clearance, clearance + 1):
                for dy in range(-clearance, clearance + 1):
                    x, y = cell.x + dx, cell.y + dy
                    if 0 <= x < self.room.width_cells and 0 <= y < self.room.height_cells:
                        blocked.add((x, y))
        return blocked

    def _validate_endpoint(self, cell: GridCell, label: str) -> None:
        if cell.x >= self.room.width_cells or cell.y >= self.room.height_cells:
            raise NavigationError(f"{label} cell is outside room bounds")
        if (cell.x, cell.y) in self._blocked:
            raise NavigationError(f"{label} cell is blocked after clearance inflation")

    def _find_path(
        self,
        start: tuple[int, int],
        goal: tuple[int, int],
        start_yaw_degrees: float,
    ) -> list[tuple[int, int]]:
        state = (start[0], start[1], start_yaw_degrees % 360.0)
        frontier: list[tuple[int, float, int, tuple[int, int, float]]] = []
        sequence = 0
        heapq.heappush(frontier, (self._manhattan(start, goal), 0.0, sequence, state))
        came_from: dict[
            tuple[int, int, float],
            tuple[int, int, float] | None,
        ] = {state: None}
        cost: dict[tuple[int, int, float], tuple[int, float]] = {state: (0, 0.0)}
        best_goal: tuple[int, int, float] | None = None
        best_goal_cost: tuple[int, float] | None = None
        while frontier:
            estimated_steps, queued_rotation, _, current = heapq.heappop(frontier)
            current_cost = cost.get(current)
            current_cell = (current[0], current[1])
            queued_steps = estimated_steps - self._manhattan(current_cell, goal)
            if (
                current_cost is None
                or queued_steps != current_cost[0]
                or not math.isclose(queued_rotation, current_cost[1], abs_tol=1e-9)
            ):
                continue
            if best_goal_cost is not None and estimated_steps > best_goal_cost[0]:
                break
            if current_cell == goal:
                if best_goal_cost is None or current_cost < best_goal_cost:
                    best_goal = current
                    best_goal_cost = current_cost
                continue
            for dx, dy, heading in self.DIRECTIONS:
                neighbor_cell = (current[0] + dx, current[1] + dy)
                if not self._inside(neighbor_cell) or neighbor_cell in self._blocked:
                    continue
                world_heading = (heading + self.room.grid_yaw_degrees) % 360.0
                clockwise = (world_heading - current[2]) % 360.0
                turn = min(clockwise, 360.0 - clockwise)
                neighbor = (neighbor_cell[0], neighbor_cell[1], world_heading)
                new_cost = (current_cost[0] + 1, current_cost[1] + turn)
                previous_cost = cost.get(neighbor)
                if previous_cost is not None and new_cost >= previous_cost:
                    continue
                cost[neighbor] = new_cost
                came_from[neighbor] = current
                sequence += 1
                priority = new_cost[0] + self._manhattan(neighbor_cell, goal)
                heapq.heappush(frontier, (priority, new_cost[1], sequence, neighbor))
        if best_goal is None:
            raise NavigationError("no clearance-safe path exists between start and goal")
        state_path = [best_goal]
        while state_path[-1] != state:
            parent = came_from[state_path[-1]]
            assert parent is not None
            state_path.append(parent)
        state_path.reverse()
        return [(item[0], item[1]) for item in state_path]

    def _path_to_actions(
        self,
        path: list[tuple[int, int]],
        start_yaw_degrees: float,
    ) -> list[PlannedAction]:
        if len(path) < 2:
            return []
        headings: list[float] = []
        for previous, current in zip(path, path[1:]):
            delta = (current[0] - previous[0], current[1] - previous[1])
            grid_heading = next(
                yaw for dx, dy, yaw in self.DIRECTIONS if (dx, dy) == delta
            )
            headings.append((grid_heading + self.room.grid_yaw_degrees) % 360.0)
        runs: list[tuple[float, int]] = []
        for heading in headings:
            if runs and runs[-1][0] == heading:
                runs[-1] = (heading, runs[-1][1] + 1)
            else:
                runs.append((heading, 1))

        actions: list[PlannedAction] = []
        yaw = start_yaw_degrees % 360.0
        max_cells_per_move = max(
            1,
            int(math.floor(self.flight_config.max_movement_m / self.room.resolution_m + 1e-9)),
        )
        for heading, cell_count in runs:
            clockwise = (heading - yaw) % 360.0
            if not math.isclose(clockwise, 0.0, abs_tol=1e-9):
                if clockwise <= 180.0:
                    actions.append(
                        PlannedAction(action=ActionType.ROTATE_RIGHT, degrees=clockwise)
                    )
                else:
                    actions.append(
                        PlannedAction(action=ActionType.ROTATE_LEFT, degrees=360.0 - clockwise)
                    )
                yaw = heading
            remaining = cell_count
            while remaining:
                chunk = min(remaining, max_cells_per_move)
                distance = chunk * self.room.resolution_m
                actions.append(
                    PlannedAction(action=ActionType.MOVE_FORWARD, distance_m=distance)
                )
                remaining -= chunk
        return actions

    def _inside(self, cell: tuple[int, int]) -> bool:
        return 0 <= cell[0] < self.room.width_cells and 0 <= cell[1] < self.room.height_cells

    @staticmethod
    def _manhattan(left: tuple[int, int], right: tuple[int, int]) -> int:
        return abs(left[0] - right[0]) + abs(left[1] - right[1])


def save_navigation_route(
    route: NavigationRoute,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    output = Path(path)
    if output.suffix.lower() != ".json":
        raise NavigationError("route output must use the .json extension")
    if output.exists() and not overwrite:
        raise NavigationError(f"route output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(route.model_dump_json(indent=2), encoding="utf-8")
    return output


def audit_simulated_route(
    room: RoomMap,
    route: NavigationRoute,
    state: DroneState,
    *,
    position_tolerance_m: float = 1e-6,
    yaw_tolerance_degrees: float = 1e-6,
) -> RouteExecutionAudit:
    if route.room_name != room.name:
        raise NavigationError("route room name does not match the audited room map")
    if state.x_m is None or state.y_m is None:
        raise NavigationError("simulator route audit requires metric x/y state")
    if position_tolerance_m <= 0.0 or yaw_tolerance_degrees <= 0.0:
        raise NavigationError("route audit tolerances must be positive")
    expected_x, expected_y = cell_center_world(room, route.goal)
    expected_yaw = route.start_yaw_degrees
    if len(route.path) >= 2:
        previous = route.path[-2]
        current = route.path[-1]
        delta = (current.x - previous.x, current.y - previous.y)
        try:
            grid_heading = next(
                yaw for dx, dy, yaw in RoomRoutePlanner.DIRECTIONS if (dx, dy) == delta
            )
            expected_yaw = (grid_heading + room.grid_yaw_degrees) % 360.0
        except StopIteration as exc:
            raise NavigationError("route ends with a non-adjacent grid step") from exc
    position_error = math.hypot(state.x_m - expected_x, state.y_m - expected_y)
    yaw_difference = abs((state.yaw_degrees - expected_yaw) % 360.0)
    yaw_error = min(yaw_difference, 360.0 - yaw_difference)
    passed = (
        position_error <= position_tolerance_m
        and yaw_error <= yaw_tolerance_degrees
    )
    return RouteExecutionAudit(
        expected_x_m=expected_x,
        expected_y_m=expected_y,
        expected_yaw_degrees=expected_yaw,
        actual_x_m=state.x_m,
        actual_y_m=state.y_m,
        actual_yaw_degrees=state.yaw_degrees,
        position_error_m=position_error,
        yaw_error_degrees=yaw_error,
        position_tolerance_m=position_tolerance_m,
        yaw_tolerance_degrees=yaw_tolerance_degrees,
        passed=passed,
        simulator_only=True,
    )


def save_route_svg(
    room: RoomMap,
    route: NavigationRoute,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    if route.room_name != room.name:
        raise NavigationError("route room name does not match the rendered room map")
    output = Path(path)
    if output.suffix.lower() != ".svg":
        raise NavigationError("route preview output must use the .svg extension")
    if output.exists() and not overwrite:
        raise NavigationError(f"route preview already exists: {output}")

    maximum_dimension = max(room.width_cells, room.height_cells)
    cell_px = max(8.0, min(40.0, 1000.0 / maximum_dimension))
    margin_x = 50.0
    margin_top = 70.0
    margin_bottom = 70.0
    grid_width = room.width_cells * cell_px
    grid_height = room.height_cells * cell_px
    canvas_width = grid_width + margin_x * 2.0
    canvas_height = grid_height + margin_top + margin_bottom

    def cell_origin(x: int, y: int) -> tuple[float, float]:
        return margin_x + x * cell_px, margin_top + (room.height_cells - 1 - y) * cell_px

    def cell_center(cell: GridCell) -> tuple[float, float]:
        x, y = cell_origin(cell.x, cell.y)
        return x + cell_px / 2.0, y + cell_px / 2.0

    raw_blocked = {(cell.x, cell.y) for cell in room.blocked_cells}
    inflated = RoomRoutePlanner(room).inflated_blocked_cells
    inflated_only = sorted(inflated - raw_blocked)
    lines = [
        '<?xml version="1.0" encoding="UTF-8" standalone="no"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{_format_svg(canvas_width)}" '
        f'height="{_format_svg(canvas_height)}" '
        f'viewBox="0 0 {_format_svg(canvas_width)} {_format_svg(canvas_height)}">',
        '  <rect width="100%" height="100%" fill="#f7f8fa"/>',
        f'  <text x="{_format_svg(margin_x)}" y="30" font-family="Arial, sans-serif" '
        f'font-size="18" font-weight="bold">{escape(room.name)} - simulator route preview</text>',
        f'  <text x="{_format_svg(margin_x)}" y="52" font-family="Arial, sans-serif" '
        f'font-size="12">resolution={room.resolution_m:g}m/cell, '
        f'clearance={room.clearance_cells} cell(s), world_origin='
        f'({room.world_origin_x_m:g},{room.world_origin_y_m:g})m, '
        f'grid_yaw={room.grid_yaw_degrees:g}deg, simulator_only=true</text>',
        f'  <rect class="grid-background" x="{_format_svg(margin_x)}" '
        f'y="{_format_svg(margin_top)}" width="{_format_svg(grid_width)}" '
        f'height="{_format_svg(grid_height)}" fill="white" stroke="#5b6470" stroke-width="2"/>',
    ]
    for x, y in inflated_only:
        left, top = cell_origin(x, y)
        lines.append(
            f'  <rect class="inflated-clearance" data-cell="{x},{y}" '
            f'x="{_format_svg(left)}" y="{_format_svg(top)}" '
            f'width="{_format_svg(cell_px)}" height="{_format_svg(cell_px)}" '
            'fill="#f6c98d"/>'
        )
    for x, y in sorted(raw_blocked):
        left, top = cell_origin(x, y)
        lines.append(
            f'  <rect class="blocked-cell" data-cell="{x},{y}" '
            f'x="{_format_svg(left)}" y="{_format_svg(top)}" '
            f'width="{_format_svg(cell_px)}" height="{_format_svg(cell_px)}" '
            'fill="#303640"/>'
        )
    for x in range(room.width_cells + 1):
        position = margin_x + x * cell_px
        lines.append(
            f'  <line class="grid-line" x1="{_format_svg(position)}" '
            f'y1="{_format_svg(margin_top)}" x2="{_format_svg(position)}" '
            f'y2="{_format_svg(margin_top + grid_height)}" stroke="#cbd0d6" stroke-width="0.5"/>'
        )
    for y in range(room.height_cells + 1):
        position = margin_top + y * cell_px
        lines.append(
            f'  <line class="grid-line" x1="{_format_svg(margin_x)}" '
            f'y1="{_format_svg(position)}" x2="{_format_svg(margin_x + grid_width)}" '
            f'y2="{_format_svg(position)}" stroke="#cbd0d6" stroke-width="0.5"/>'
        )
    path_points = " ".join(
        f"{_format_svg(x)},{_format_svg(y)}" for x, y in map(cell_center, route.path)
    )
    lines.append(
        f'  <polyline class="route" points="{path_points}" fill="none" '
        f'stroke="#1769e0" stroke-width="{_format_svg(max(2.0, cell_px * 0.14))}" '
        'stroke-linecap="round" stroke-linejoin="round"/>'
    )
    start_x, start_y = cell_center(route.start)
    goal_x, goal_y = cell_center(route.goal)
    radius = max(3.0, cell_px * 0.22)
    lines.extend(
        [
            f'  <circle class="start" data-cell="{route.start.x},{route.start.y}" '
            f'cx="{_format_svg(start_x)}" cy="{_format_svg(start_y)}" '
            f'r="{_format_svg(radius)}" fill="#139c55" stroke="white" stroke-width="2"/>',
            f'  <circle class="goal" data-cell="{route.goal.x},{route.goal.y}" '
            f'cx="{_format_svg(goal_x)}" cy="{_format_svg(goal_y)}" '
            f'r="{_format_svg(radius)}" fill="#d83a45" stroke="white" stroke-width="2"/>',
            f'  <text x="{_format_svg(margin_x)}" '
            f'y="{_format_svg(canvas_height - 30.0)}" font-family="Arial, sans-serif" '
            f'font-size="12">green=start, red=goal, blue=route, dark=obstacle, '
            f'orange=inflated clearance; distance={route.distance_m:g}m</text>',
            '</svg>',
            '',
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def _format_svg(value: float) -> str:
    return f"{value:.4f}".rstrip("0").rstrip(".")
