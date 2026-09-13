"""Helpers shared across the command modules: step specs, planes, Zarr output."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import typer

from pypic.dataset import FieldDataset
from pypic.exceptions import PypicError

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Sequence
    from typing import Any

    from pypic.grid import GridInfo
    from pypic.readers._registry import Simulation
    from pypic.selections import PlaneSelection


def parse_steps(raw: str, available: Sequence[int]) -> list[int]:
    """Parse ``--step`` syntax into a list of timestep indices.

    Supports: ``N`` (single int), ``first``, ``last``, ``all``,
    ``start:stop:stride`` (inclusive stop).  ``available`` is the
    universe of valid steps — pass ``sim.steps`` for field commands
    and ``sim.particle_steps`` for particle commands so aliases like
    ``last`` resolve to the right cadence.

    Raises `typer.BadParameter` on invalid syntax, on an empty
    ``available`` list, or when the resolved list is empty.
    """
    raw = raw.strip()
    if not available:
        msg = f"No steps available to resolve --step {raw!r}."
        raise typer.BadParameter(msg)
    available_list = list(available)
    if raw == "all":
        return available_list
    if raw == "first":
        return [available_list[0]]
    if raw == "last":
        return [available_list[-1]]
    if ":" in raw:
        parts = raw.split(":")
        if len(parts) not in (2, 3):
            msg = f"Invalid step range {raw!r}. Use start:stop or start:stop:stride."
            raise typer.BadParameter(msg)
        try:
            start = int(parts[0])
            stop = int(parts[1])
            stride = int(parts[2]) if len(parts) == 3 else 1
        except ValueError:
            msg = f"Non-integer values in step range {raw!r}."
            raise typer.BadParameter(msg) from None
        if stride <= 0:
            msg = f"Stride must be positive, got {stride}."
            raise typer.BadParameter(msg)
        result = [
            s
            for s in available_list
            if start <= s <= stop and (s - start) % stride == 0
        ]
        if not result:
            msg = (
                f"No available steps match range {raw!r}. "
                f"Available: {available_list[0]}..{available_list[-1]}"
            )
            raise typer.BadParameter(msg)
        return result
    try:
        step_val = int(raw)
    except ValueError:
        msg = (
            f"Invalid --step value {raw!r}. "
            "Use a number, first, last, all, or start:stop:stride."
        )
        raise typer.BadParameter(msg) from None
    if step_val not in available_list:
        msg = (
            f"Step {step_val} not available. "
            f"Available: {available_list[0]}..{available_list[-1]}"
        )
        raise typer.BadParameter(msg)
    return [step_val]


def _require_single_step(step_list: list[int], raw: str) -> int:
    """Extract single step, raising if multiple were selected."""
    if len(step_list) > 1:
        msg = (
            f"This command operates on a single step, "
            f"but --step {raw!r} selected {len(step_list)}."
        )
        raise typer.BadParameter(msg)
    return step_list[0]


def _open(path: Path) -> Simulation:
    """Open a simulation, translating errors to CLI messages."""
    from pypic.readers._registry import open_simulation

    try:
        return open_simulation(path)
    except PypicError as exc:
        # A deliberate refusal carries the reason; `.detail` because the
        # KeyError-inheriting subclasses requote under str().  No PypicError
        # subclass is an OSError, so this cannot shadow the branch below.
        typer.echo(f"Error: {exc.detail}", err=True)
        raise typer.Exit(1) from None
    except (FileNotFoundError, OSError, ExceptionGroup) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from None


def _output(data: dict[str, object], text: str, *, json_mode: bool) -> None:
    """Print JSON or formatted text."""
    if json_mode:
        typer.echo(json.dumps(data, indent=2, default=str))
    else:
        typer.echo(text)


_CARTESIAN_PLANE_MAP = {"xy": "z", "xz": "y", "yz": "x"}


def _plane_normal(plane_str: str, grid: GridInfo) -> str:
    """Resolve a ``--plane`` value to the normal axis name.

    Accepts Cartesian shorthands (``xy``, ``xz``, ``yz``), a pair of
    axis names from the grid's geometry (e.g. ``rθ``, ``rφ``), or a
    single axis name interpreted as the normal directly (e.g. ``z``,
    ``φ``).
    """
    val = plane_str.lower()
    axis_names = grid.geometry.axis_names
    axis_lower = [n.lower() for n in axis_names]

    # Cartesian shorthands
    if val in _CARTESIAN_PLANE_MAP:
        return _CARTESIAN_PLANE_MAP[val]

    # Single axis name → interpret as the normal axis directly
    if val in axis_lower:
        return axis_names[axis_lower.index(val)]

    # Pair of axis names → normal is the remaining axis
    if len(val) >= 2:
        for i, name in enumerate(axis_names):
            others = [n for j, n in enumerate(axis_lower) if j != i]
            pair = "".join(others)
            if val == pair:
                return name

    names_str = ", ".join(axis_names)
    msg = (
        f"Invalid --plane {plane_str!r}. "
        f"Use a pair of axis names or a single normal axis ({names_str})."
    )
    raise typer.BadParameter(msg)


def _auto_plane_normal(grid: GridInfo) -> str:
    """Pick the normal axis for the largest cross-section.

    Selects the axis with the smallest physical extent so the
    remaining two axes span the widest view.
    """
    extents = [d * s for d, s in zip(grid.dimensions, grid.spacing, strict=True)]
    axis_names = grid.geometry.axis_names
    min_idx = extents.index(min(extents))
    return axis_names[min_idx]


def _resolve_plane(
    grid: GridInfo,
    plane_str: str | None,
    index: int | None,
    coord: float | None,
) -> PlaneSelection | None:
    """Build a `PlaneSelection` from CLI flags.

    Returns ``None`` for 2D grids when the user didn't request a
    specific plane — the plotting layer handles 2D data directly.
    """
    import numpy as np

    from pypic.selections import PlaneSelection

    if index is not None and coord is not None:
        msg = "Cannot specify both --index and --coord."
        raise typer.BadParameter(msg)

    # 2D data: no slicing needed unless the user explicitly asked
    if (
        len(grid.dimensions) <= 2
        and plane_str is None
        and index is None
        and coord is None
    ):
        return None

    normal = _plane_normal(plane_str, grid) if plane_str else _auto_plane_normal(grid)

    if coord is not None:
        axis_idx = list(grid.geometry.axis_names).index(normal)
        coord_arr = grid.coordinate_arrays()[axis_idx]
        index = int(np.argmin(np.abs(coord_arr - coord)))

    return PlaneSelection(normal=normal, index=index)


_ENCODING_BLANKET_KEY = "__all__"  # sentinel for "apply to every data_var"


def _parse_comma_list(raw: str | None) -> list[str] | None:
    """Split a comma-separated string into a list, preserving None."""
    if raw is None:
        return None
    return [s.strip() for s in raw.split(",") if s.strip()]


def _parse_box_ranges(
    raw: str | None,
    *,
    convert: Callable[[str], Any],
) -> dict[str, tuple[Any, Any]] | None:
    """Parse ``axis=lo:hi,axis=lo:hi`` using *convert* for bounds.

    Pass ``convert=int`` for cell-index crops (fields) or ``convert=float``
    for physical-coordinate crops (particles).
    """
    if raw is None:
        return None
    ranges: dict[str, tuple[Any, Any]] = {}
    for part in raw.split(","):
        segment = part.strip()
        if not segment:
            continue
        if "=" not in segment or ":" not in segment:
            msg = f"Invalid --box segment {segment!r}. Use axis=lo:hi."
            raise typer.BadParameter(msg)
        axis, spec = segment.split("=", 1)
        lo_s, hi_s = spec.split(":", 1)
        try:
            ranges[axis.strip()] = (convert(lo_s), convert(hi_s))
        except ValueError as exc:
            msg = f"Invalid bounds in --box segment {segment!r}."
            raise typer.BadParameter(msg) from exc
    return ranges or None


def _parse_compression(
    spec: str | None,
) -> dict[str, dict[str, object]] | None:
    """Parse ``zstd[:level]`` / ``blosc[:clevel]`` into a zarr encoding dict.

    Returns ``{_ENCODING_BLANKET_KEY: {...}}`` — caller passes the result
    through `_expand_encoding_for_vars` to fan out to every
    ``data_var`` before handing to ``to_zarr``.
    """
    if spec is None:
        return None
    raw = spec.strip().lower()
    codec_name, _, level_str = raw.partition(":")
    try:
        level = int(level_str) if level_str else None
    except ValueError as exc:
        msg = f"--compression level must be an integer, got {level_str!r}."
        raise typer.BadParameter(msg) from exc

    from zarr.codecs import BloscCodec

    if codec_name == "blosc":
        compressor = BloscCodec(
            cname="zstd",
            clevel=level if level is not None else 5,
            shuffle="bitshuffle",
        )
    elif codec_name == "zstd":
        compressor = BloscCodec(
            cname="zstd",
            clevel=level if level is not None else 5,
            shuffle="noshuffle",
        )
    else:
        msg = f"--compression codec must be 'zstd' or 'blosc', got {codec_name!r}."
        raise typer.BadParameter(msg)
    return {_ENCODING_BLANKET_KEY: {"compressors": compressor}}


def _time_coordinate(fds: FieldDataset, step: int) -> float | int:
    """Zarr ``time`` coordinate for one snapshot: its time, else its step."""
    return step if fds.time is None else fds.time


def _expand_encoding_for_vars(
    spec: dict[str, dict[str, object]] | None,
    data_vars: list[str],
) -> dict[str, dict[str, object]] | None:
    """Expand a blanket-keyed encoding dict to per-variable entries."""
    if spec is None:
        return None
    blanket = spec.get(_ENCODING_BLANKET_KEY)
    if blanket is not None:
        return {name: dict(blanket) for name in data_vars}
    return spec


def _make_progress_iter[T](
    items: Iterable[T],
    *,
    total: int,
    description: str,
    enabled: bool,
) -> Iterable[T]:
    """Wrap an iterable with a rich.progress bar when *enabled* and TTY.

    Falls back to the raw iterable when rich isn't installed or output
    isn't a TTY.
    """
    import sys

    if not enabled or not sys.stderr.isatty():
        return items
    try:
        from rich.progress import (
            BarColumn,
            Progress,
            TextColumn,
            TimeElapsedColumn,
            TimeRemainingColumn,
        )
    except ImportError:
        return items

    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        transient=True,
    )

    def _iter() -> Iterator[T]:
        with progress:
            task_id = progress.add_task(description, total=total)
            for item in items:
                yield item
                progress.advance(task_id)

    return _iter()


@dataclass(frozen=True, slots=True)
class ZarrTarget:
    """The Zarr output options ``convert fields`` and ``reduce apply`` share."""

    output: Path
    backend: Literal["zarr", "icechunk"]
    dtype: str | None = None
    compression: str | None = None
    message: str | None = None
    tag: str | None = None

    def __post_init__(self) -> None:
        for option, value in (("--tag", self.tag), ("--message", self.message)):
            if value is not None and self.backend != "icechunk":
                msg = f"{option} requires --backend icechunk."
                raise typer.BadParameter(msg)


def _write_steps(
    sim: Simulation,
    step_list: list[int],
    target: ZarrTarget,
    *,
    read_fields: list[str] | None,
    transform: Callable[[FieldDataset], FieldDataset],
    description: str,
    progress: bool,
    done: str,
) -> None:
    """Read, transform and write *step_list* to *target*, then echo *done*.

    One step goes through `to_zarr`; several promote ``time`` to a
    leading dimension through `to_zarr_timeseries`. The first step is
    read once, for the encoding dict and as the first yielded pair.
    """
    from pypic.io import to_zarr, to_zarr_timeseries

    backend = None if target.backend == "zarr" else target.backend
    first = transform(sim.read(step_list[0], fields=read_fields))
    encoding = _expand_encoding_for_vars(
        _parse_compression(target.compression), list(first.field_names())
    )
    if len(step_list) == 1:
        snapshot = to_zarr(
            first,
            target.output,
            dtype=target.dtype,
            encoding=encoding,
            backend=backend,
            message=target.message,
        )
    else:

        def _pairs() -> Iterator[tuple[float | int, FieldDataset]]:
            yield _time_coordinate(first, step_list[0]), first
            for step in step_list[1:]:
                fds = transform(sim.read(step, fields=read_fields))
                yield _time_coordinate(fds, step), fds

        snapshot = to_zarr_timeseries(
            _make_progress_iter(
                _pairs(),
                total=len(step_list),
                description=description,
                enabled=progress,
            ),
            target.output,
            dtype=target.dtype,
            encoding=encoding,
            backend=backend,
            message=target.message,
        )

    if target.tag is not None and snapshot is not None:
        from pypic.io import icechunk_create_tag

        icechunk_create_tag(target.output, target.tag, snapshot_id=snapshot)
        typer.echo(f"{done} (tag={target.tag})")
    else:
        typer.echo(done)
