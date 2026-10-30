"""Offline projection of consensus-approved camera poses into a static room grid."""

from __future__ import annotations

from enum import Enum
import math
from pathlib import Path
from typing import Iterable, Iterator
from xml.sax.saxutils import escape

from pydantic import BaseModel, ConfigDict, Field, model_validator

from drone_ai.fiducials import FiducialFrame
from drone_ai.navigation import GridCell, RoomMap, RoomRoutePlanner, world_to_grid


class LocalizationError(RuntimeError):
    pass


class MappedPoseStatus(str, Enum):
    MISSING = "missing"
    OUTSIDE_ROOM = "outside_room"
    CLEARANCE_BLOCKED = "clearance_blocked"
    SAFE = "safe"


class MappedPoseFrame(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    frame_index: int = Field(ge=0)
    status: MappedPoseStatus
    source_metric_pose_available: bool
    source_pose_consistent: bool
    world_x_m: float | None = None
    world_y_m: float | None = None
    world_z_m: float | None = None
    grid_x_cells: float | None = None
    grid_y_cells: float | None = None
    cell: GridCell | None = None
    clearance_safe: bool
    offline_only: bool = True

    @model_validator(mode="after")
    def validate_status_payload(self) -> "MappedPoseFrame":
        metric_values = (
            self.world_x_m,
            self.world_y_m,
            self.world_z_m,
            self.grid_x_cells,
            self.grid_y_cells,
        )
        if self.source_pose_consistent != self.source_metric_pose_available:
            raise ValueError(
                "source pose consistency must match source metric-pose availability"
            )
        if self.status is MappedPoseStatus.MISSING:
            if self.source_metric_pose_available or any(value is not None for value in metric_values):
                raise ValueError("missing mapped pose cannot contain metric coordinates")
            if self.cell is not None:
                raise ValueError("missing mapped pose cannot contain a grid cell")
        else:
            if not self.source_metric_pose_available:
                raise ValueError("mapped coordinates require a source metric pose")
            if any(value is None or not math.isfinite(value) for value in metric_values):
                raise ValueError("mapped pose coordinates must be finite")
        inside = self.status in {
            MappedPoseStatus.CLEARANCE_BLOCKED,
            MappedPoseStatus.SAFE,
        }
        if (self.cell is not None) != inside:
            raise ValueError("grid cell must exist exactly for in-room poses")
        if self.clearance_safe != (self.status is MappedPoseStatus.SAFE):
            raise ValueError("clearance_safe must match safe status")
        if not self.offline_only:
            raise ValueError("mapped pose records must remain offline-only")
        return self


class MappedPoseSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    total_frames: int = Field(ge=0)
    missing_frames: int = Field(ge=0)
    outside_frames: int = Field(ge=0)
    clearance_blocked_frames: int = Field(ge=0)
    safe_frames: int = Field(ge=0)
    offline_only: bool = True

    @model_validator(mode="after")
    def validate_counts_and_boundary(self) -> "MappedPoseSummary":
        classified = (
            self.missing_frames
            + self.outside_frames
            + self.clearance_blocked_frames
            + self.safe_frames
        )
        if classified != self.total_frames:
            raise ValueError("mapped pose summary counts must equal total frames")
        if not self.offline_only:
            raise ValueError("mapped pose summary must remain offline-only")
        return self


class StaticMapPoseAnalyzer:
    def __init__(self, room: RoomMap) -> None:
        self.room = room
        self._blocked = RoomRoutePlanner(room).inflated_blocked_cells
        self._counts = {status: 0 for status in MappedPoseStatus}

    @property
    def inflated_blocked_cells(self) -> frozenset[tuple[int, int]]:
        return self._blocked

    def process(self, source: FiducialFrame) -> MappedPoseFrame:
        if not source.metric_pose_available or source.best_camera_to_world is None:
            status = MappedPoseStatus.MISSING
            self._counts[status] += 1
            return MappedPoseFrame(
                frame_index=source.frame_index,
                status=status,
                source_metric_pose_available=False,
                source_pose_consistent=source.pose_consistent,
                clearance_safe=False,
                offline_only=True,
            )

        transform = source.best_camera_to_world
        world_x = transform[0][3]
        world_y = transform[1][3]
        world_z = transform[2][3]
        grid_x, grid_y = world_to_grid(self.room, world_x, world_y)
        inside = (
            0.0 <= grid_x < self.room.width_cells
            and 0.0 <= grid_y < self.room.height_cells
        )
        cell: GridCell | None = None
        if not inside:
            status = MappedPoseStatus.OUTSIDE_ROOM
        else:
            cell = GridCell(x=math.floor(grid_x), y=math.floor(grid_y))
            status = (
                MappedPoseStatus.CLEARANCE_BLOCKED
                if (cell.x, cell.y) in self._blocked
                else MappedPoseStatus.SAFE
            )
        self._counts[status] += 1
        return MappedPoseFrame(
            frame_index=source.frame_index,
            status=status,
            source_metric_pose_available=True,
            source_pose_consistent=source.pose_consistent,
            world_x_m=world_x,
            world_y_m=world_y,
            world_z_m=world_z,
            grid_x_cells=grid_x,
            grid_y_cells=grid_y,
            cell=cell,
            clearance_safe=status is MappedPoseStatus.SAFE,
            offline_only=True,
        )

    def summary(self) -> MappedPoseSummary:
        return MappedPoseSummary(
            total_frames=sum(self._counts.values()),
            missing_frames=self._counts[MappedPoseStatus.MISSING],
            outside_frames=self._counts[MappedPoseStatus.OUTSIDE_ROOM],
            clearance_blocked_frames=self._counts[MappedPoseStatus.CLEARANCE_BLOCKED],
            safe_frames=self._counts[MappedPoseStatus.SAFE],
            offline_only=True,
        )


def iter_fiducial_jsonl(path: str | Path) -> Iterator[FiducialFrame]:
    source = Path(path)
    if not source.is_file():
        raise LocalizationError(f"fiducial JSONL does not exist: {source}")
    try:
        with source.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    raise LocalizationError(
                        f"blank record at {source}:{line_number}"
                    )
                try:
                    yield FiducialFrame.model_validate_json(line)
                except Exception as exc:
                    raise LocalizationError(
                        f"invalid fiducial record at {source}:{line_number}: {exc}"
                    ) from exc
    except OSError as exc:
        raise LocalizationError(f"could not read fiducial JSONL {source}: {exc}") from exc


def save_mapped_trajectory_svg(
    room: RoomMap,
    records: Iterable[MappedPoseFrame],
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    output = Path(path)
    if output.suffix.lower() != ".svg":
        raise LocalizationError("mapped trajectory output must use the .svg extension")
    if output.exists() and not overwrite:
        raise LocalizationError(f"mapped trajectory output already exists: {output}")
    materialized = tuple(records)
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

    def pose_point(record: MappedPoseFrame) -> tuple[float, float] | None:
        if record.cell is None:
            return None
        assert record.grid_x_cells is not None and record.grid_y_cells is not None
        return (
            margin_x + record.grid_x_cells * cell_px,
            margin_top + (room.height_cells - record.grid_y_cells) * cell_px,
        )

    raw_blocked = {(cell.x, cell.y) for cell in room.blocked_cells}
    inflated_only = sorted(RoomRoutePlanner(room).inflated_blocked_cells - raw_blocked)
    lines = [
        '<?xml version="1.0" encoding="UTF-8" standalone="no"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{_svg_number(canvas_width)}" '
        f'height="{_svg_number(canvas_height)}" '
        f'viewBox="0 0 {_svg_number(canvas_width)} {_svg_number(canvas_height)}">',
        '  <rect width="100%" height="100%" fill="#f7f8fa"/>',
        f'  <text x="{_svg_number(margin_x)}" y="30" font-family="Arial, sans-serif" '
        f'font-size="18" font-weight="bold">{escape(room.name)} - offline camera trajectory</text>',
        f'  <text x="{_svg_number(margin_x)}" y="52" font-family="Arial, sans-serif" '
        f'font-size="12">world_origin=({room.world_origin_x_m:g},'
        f'{room.world_origin_y_m:g})m, grid_yaw={room.grid_yaw_degrees:g}deg, '
        'offline_only=true</text>',
        f'  <rect x="{_svg_number(margin_x)}" y="{_svg_number(margin_top)}" '
        f'width="{_svg_number(grid_width)}" height="{_svg_number(grid_height)}" '
        'fill="white" stroke="#5b6470" stroke-width="2"/>',
    ]
    for x, y in inflated_only:
        left, top = cell_origin(x, y)
        lines.append(
            f'  <rect class="inflated-clearance" data-cell="{x},{y}" '
            f'x="{_svg_number(left)}" y="{_svg_number(top)}" '
            f'width="{_svg_number(cell_px)}" height="{_svg_number(cell_px)}" '
            'fill="#f6c98d"/>'
        )
    for x, y in sorted(raw_blocked):
        left, top = cell_origin(x, y)
        lines.append(
            f'  <rect class="blocked-cell" data-cell="{x},{y}" '
            f'x="{_svg_number(left)}" y="{_svg_number(top)}" '
            f'width="{_svg_number(cell_px)}" height="{_svg_number(cell_px)}" '
            'fill="#303640"/>'
        )
    for x in range(room.width_cells + 1):
        position = margin_x + x * cell_px
        lines.append(
            f'  <line x1="{_svg_number(position)}" y1="{_svg_number(margin_top)}" '
            f'x2="{_svg_number(position)}" y2="{_svg_number(margin_top + grid_height)}" '
            'stroke="#cbd0d6" stroke-width="0.5"/>'
        )
    for y in range(room.height_cells + 1):
        position = margin_top + y * cell_px
        lines.append(
            f'  <line x1="{_svg_number(margin_x)}" y1="{_svg_number(position)}" '
            f'x2="{_svg_number(margin_x + grid_width)}" y2="{_svg_number(position)}" '
            'stroke="#cbd0d6" stroke-width="0.5"/>'
        )
    segments: list[list[tuple[float, float]]] = []
    active: list[tuple[float, float]] = []
    for record in materialized:
        point = pose_point(record)
        if point is None:
            if active:
                segments.append(active)
                active = []
        else:
            active.append(point)
    if active:
        segments.append(active)
    for segment in segments:
        points = " ".join(f"{_svg_number(x)},{_svg_number(y)}" for x, y in segment)
        lines.append(
            f'  <polyline class="trajectory" points="{points}" fill="none" '
            'stroke="#1769e0" stroke-width="2.5" stroke-linecap="round"/>'
        )
    for record in materialized:
        point = pose_point(record)
        if point is None:
            continue
        color = "#139c55" if record.clearance_safe else "#d83a45"
        assert record.cell is not None
        lines.append(
            f'  <circle class="pose-{record.status.value}" data-frame="{record.frame_index}" '
            f'data-cell="{record.cell.x},{record.cell.y}" cx="{_svg_number(point[0])}" '
            f'cy="{_svg_number(point[1])}" r="3.5" fill="{color}" stroke="white" '
            'stroke-width="1"/>'
        )
    lines.extend(
        [
            f'  <text x="{_svg_number(margin_x)}" y="{_svg_number(canvas_height - 30.0)}" '
            'font-family="Arial, sans-serif" font-size="12">green=safe camera pose, '
            'red=clearance blocked, blue=trajectory; missing/outside poses are not plotted</text>',
            '</svg>',
            '',
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def _svg_number(value: float) -> str:
    return f"{value:.4f}".rstrip("0").rstrip(".")
