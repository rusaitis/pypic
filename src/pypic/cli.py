"""pypic command-line interface."""

from __future__ import annotations

import importlib.metadata
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal

import typer

from pypic.dataset import FieldDataset

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from typing import Any

    from pypic.containers import ParticleData
    from pypic.grid import GridInfo
    from pypic.readers._registry import Simulation
    from pypic.selections import PlaneSelection
    from pypic.types import FloatArray

app = typer.Typer(
    name="pypic",
    help="Inspect and compare plasma simulation output.",
    pretty_exceptions_enable=False,
    no_args_is_help=True,
)


def _version_callback(value: bool) -> None:
    if value:
        ver = importlib.metadata.version("pypic")
        typer.echo(f"pypic {ver}")
        raise typer.Exit


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            help="Print version and exit.",
            callback=_version_callback,
            is_eager=True,
        ),
    ] = False,
    log_level: Annotated[
        str,
        typer.Option(
            "--log-level",
            help="Logging level: debug, info, warning, error.",
        ),
    ] = "warning",
    quiet: Annotated[
        bool,
        typer.Option("-q", "--quiet", help="Shorthand for --log-level error."),
    ] = False,
    debug: Annotated[
        bool,
        typer.Option("--debug", help="Show full tracebacks on error."),
    ] = False,
) -> None:
    """Inspect and compare plasma simulation output."""
    level_name = "ERROR" if quiet else log_level.upper()
    if level_name not in ("DEBUG", "INFO", "WARNING", "ERROR"):
        msg = f"Invalid --log-level {log_level!r}. Use debug, info, warning, or error."
        raise typer.BadParameter(msg)
    logging.basicConfig(level=getattr(logging, level_name), force=True)
    logging.captureWarnings(True)
    app.pretty_exceptions_enable = debug


def parse_steps(raw: str, available: Sequence[int]) -> list[int]:
    """Parse ``--step`` syntax into a list of timestep indices.

    Supports: ``N`` (single int), ``first``, ``last``, ``all``,
    ``start:stop:stride`` (inclusive stop).  ``available`` is the
    universe of valid steps — pass ``sim.steps`` for field commands
    and ``sim.particle_steps`` for particle commands so aliases like
    ``last`` resolve to the right cadence.

    Raises :class:`typer.BadParameter` on invalid syntax, on an empty
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


def _check_choice(option: str, value: str | None, choices: tuple[str, ...]) -> None:
    """Raise BadParameter when *value* isn't one of *choices*.

    ``None`` passes through so callers can use the helper for both
    required and optional options.
    """
    if value is not None and value not in choices:
        msg = f"Invalid {option} {value!r}. Use {', '.join(choices)}."
        raise typer.BadParameter(msg)


def _open(path: Path) -> Simulation:
    """Open a simulation, translating errors to CLI messages."""
    from pypic.readers._registry import open_simulation

    try:
        return open_simulation(path)
    except (FileNotFoundError, OSError, ExceptionGroup) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from None


def _output(data: dict[str, object], text: str, *, json_mode: bool) -> None:
    """Print JSON or formatted text."""
    if json_mode:
        typer.echo(json.dumps(data, indent=2, default=str))
    else:
        typer.echo(text)


def _get_field_array(
    sim: Simulation, step: int, field: str, units: str | None
) -> FloatArray:
    """Read a field (or compute if derived), convert units."""
    import numpy as np

    try:
        ds = sim.read(step, fields=[field])
        if units is not None:
            return ds.in_units(field, units)
        return np.asarray(ds[field]) if ds.has_field(field) else ds.compute(field)
    except KeyError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from None


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
    """Build a :class:`PlaneSelection` from CLI flags.

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


def _parse_resolution(res_str: str) -> tuple[int, int]:
    """Parse a ``WxH`` resolution string."""
    parts = res_str.lower().split("x")
    if len(parts) != 2:
        msg = f"Invalid --res {res_str!r}. Use WxH (e.g. 256x256)."
        raise typer.BadParameter(msg)
    try:
        w, h = int(parts[0]), int(parts[1])
    except ValueError:
        msg = f"Non-integer values in --res {res_str!r}."
        raise typer.BadParameter(msg) from None
    if w <= 0 or h <= 0:
        msg = f"--res dimensions must be positive, got {res_str!r}."
        raise typer.BadParameter(msg)
    return (w, h)


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


def _resolve_species_list(raw: str | None, sim: Simulation) -> list[int] | None:
    """Resolve ``--species names_or_indices`` into a list of species indices."""
    if raw is None:
        return None
    tokens = _parse_comma_list(raw) or []
    if not tokens:
        return None
    all_species = sim.config.species
    result: list[int] = []
    for tok in tokens:
        try:
            idx = int(tok)
        except ValueError:
            idx = next((i for i, sp in enumerate(all_species) if sp.name == tok), -1)
            if idx < 0:
                names = ", ".join(sp.name for sp in all_species)
                msg = f"Species {tok!r} not found. Available: {names}"
                raise typer.BadParameter(msg) from None
        if idx < 0 or idx >= len(all_species):
            msg = f"Species index {idx} out of range (0..{len(all_species) - 1})."
            raise typer.BadParameter(msg)
        result.append(idx)
    return result


def _parse_compression(
    spec: str | None,
) -> dict[str, dict[str, object]] | None:
    """Parse ``zstd[:level]`` / ``blosc[:clevel]`` into a zarr encoding dict.

    Returns ``{_ENCODING_BLANKET_KEY: {...}}`` — caller passes the result
    through :func:`_expand_encoding_for_vars` to fan out to every
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


def _filter_particles_box(
    pcl: ParticleData,
    box_ranges: dict[str, tuple[float, float]],
) -> ParticleData:
    """Return a new ParticleData containing only particles inside *box_ranges*.

    ``box_ranges`` maps ``"x" | "y" | "z"`` to ``(lo, hi)`` in code units.
    Axes not listed are not filtered.
    """
    import copy as _copy

    import numpy as np

    if pcl.position is None:
        return pcl  # nothing to filter
    mask = np.ones(pcl.n_particles, dtype=bool)
    axis_map = {"x": 0, "y": 1, "z": 2}
    for axis, (lo, hi) in box_ranges.items():
        if axis not in axis_map:
            msg = f"Particle --box axis must be x, y, or z, got {axis!r}."
            raise typer.BadParameter(msg)
        col = pcl.position[:, axis_map[axis]]
        mask &= (col >= lo) & (col <= hi)

    n_kept = int(mask.sum())
    if n_kept == pcl.n_particles:
        return pcl
    return _copy.replace(
        pcl,
        position=pcl.position[mask],
        velocity=None if pcl.velocity is None else pcl.velocity[mask],
        weight=None if pcl.weight is None else pcl.weight[mask],
        id=None if pcl.id is None else pcl.id[mask],
        n_particles=n_kept,
    )


def _make_progress_iter(
    items: object,
    *,
    total: int,
    description: str,
    enabled: bool,
) -> object:
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

    def _iter() -> object:
        with progress:
            task_id = progress.add_task(description, total=total)
            for item in items:  # type: ignore[attr-defined]
                yield item
                progress.advance(task_id)

    return _iter()


def _to_si_dataset(fds: FieldDataset) -> FieldDataset:
    """Return a copy of *fds* with every field converted to SI units.

    The new dataset carries ``Normalization.identity()`` so re-reading
    via the standard path returns SI values without further scaling.
    """
    from pypic.units import Normalization

    si_fields = {name: fds.in_si(name) for name in fds.field_names()}
    return FieldDataset.from_arrays(
        si_fields,
        fds.grid,
        Normalization.identity(),
        species=fds.species,
        physics=fds.physics,
        metadata=dict(fds.metadata),
        frame=fds.frame,
        transforms=dict(fds.transforms),
        strict_fields=False,
    )


convert_app = typer.Typer(
    name="convert",
    help="Convert simulation output to Zarr (fields) or Parquet (particles).",
    no_args_is_help=True,
)
app.add_typer(convert_app, name="convert")


@convert_app.command("fields")
def convert_fields(
    path: Annotated[Path, typer.Argument(help="Simulation directory or HDF5 file.")],
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Destination Zarr store directory."),
    ],
    step: Annotated[
        str,
        typer.Option(
            "--step",
            help="Step spec: N, first, last, all, or start:stop[:stride].",
        ),
    ] = "all",
    fields: Annotated[
        str | None,
        typer.Option(
            "--fields", help="Comma-separated field names (e.g. B,E_3,rho_c)."
        ),
    ] = None,
    box: Annotated[
        str | None,
        typer.Option("--box", help="Spatial crop as axis=lo:hi,...,axis=lo:hi."),
    ] = None,
    plane: Annotated[
        str | None,
        typer.Option("--plane", help="Slab plane (e.g. xy, xz, z). See `pypic plot`."),
    ] = None,
    plane_index: Annotated[
        int | None,
        typer.Option("--plane-index", help="Cell index along the plane normal."),
    ] = None,
    plane_coord: Annotated[
        float | None,
        typer.Option("--plane-coord", help="Physical coord along the plane normal."),
    ] = None,
    target_resolution: Annotated[
        float | None,
        typer.Option("--target-resolution", help="Regrid to this uniform spacing."),
    ] = None,
    to_si: Annotated[
        bool,
        typer.Option("--to-si", help="Convert all fields to SI units before writing."),
    ] = False,
    dtype: Annotated[
        str | None,
        typer.Option("--dtype", help="Downcast (e.g. float32)."),
    ] = None,
    compression: Annotated[
        str | None,
        typer.Option(
            "--compression",
            help="Codec spec: zstd[:level] or blosc[:level] (default blosc:5).",
        ),
    ] = None,
    virtual: Annotated[
        bool,
        typer.Option(
            "--virtual",
            help=(
                "Treat PATH as an HDF5 file, persist byte-range refs to "
                "an Icechunk repo (no data copy). Requires --backend "
                "icechunk. Mutually exclusive with --fields, --box, "
                "--plane, --target-resolution, --to-si, --dtype, "
                "--compression."
            ),
        ),
    ] = False,
    backend: Annotated[
        Literal["zarr", "icechunk"],
        typer.Option("--backend", help="Storage backend: zarr or icechunk."),
    ] = "zarr",
    message: Annotated[
        str | None,
        typer.Option("--message", help="Icechunk commit message."),
    ] = None,
    tag: Annotated[
        str | None,
        typer.Option("--tag", help="Icechunk tag name (created on success)."),
    ] = None,
    progress: Annotated[
        bool,
        typer.Option(
            "--progress/--no-progress",
            help="Show a rich.progress bar for multi-step writes (auto on TTY).",
        ),
    ] = True,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Print the plan without writing."),
    ] = False,
) -> None:
    """Convert simulation fields to a Zarr v3 store.

    Writes one timestep via ``to_zarr`` or a multi-step time-series via
    ``to_zarr_timeseries``.  Optionally crops, slices, regrids, and/or
    converts to SI units.  Pass ``--backend icechunk`` for versioned
    storage, or ``--virtual`` to persist HDF5 byte-range references
    without copying data.
    """
    from pypic.io import to_zarr, to_zarr_timeseries
    from pypic.selections import BoxSelection

    if tag is not None and backend != "icechunk":
        msg = "--tag requires --backend icechunk."
        raise typer.BadParameter(msg)
    if message is not None and backend != "icechunk":
        msg = "--message requires --backend icechunk."
        raise typer.BadParameter(msg)
    backend_arg = None if backend == "zarr" else backend

    # -- Virtual mode: treat path as one HDF5 file, single-step write --------
    if virtual:
        if path.is_dir():
            msg = "--virtual requires PATH to be an HDF5 file, not a directory."
            raise typer.BadParameter(msg)
        if target_resolution or box or plane or to_si or fields is not None:
            msg = (
                "--virtual is incompatible with --target-resolution, --box, "
                "--plane, --to-si, and --fields (virtual refs cover the whole "
                "source dataset; sub-selection would force materialization)."
            )
            raise typer.BadParameter(msg)
        if dtype is not None or compression is not None:
            msg = (
                "--virtual is incompatible with --dtype and --compression: "
                "virtual refs persist the source bytes as-is, so chunk "
                "encoding cannot be changed at write time."
            )
            raise typer.BadParameter(msg)
        if backend != "icechunk":
            msg = (
                "--virtual requires --backend icechunk: virtual chunk "
                "containers are an Icechunk-only feature."
            )
            raise typer.BadParameter(msg)
        from pypic.io import to_icechunk_virtual

        if dry_run:
            typer.echo(f"Would write virtual refs for {path} to {output}")
            typer.echo(f"  backend: {backend}")
            return
        virtual_snapshot = to_icechunk_virtual(path, output, message=message)
        if tag is not None:
            from pypic.io import icechunk_create_tag

            icechunk_create_tag(output, tag, snapshot_id=virtual_snapshot)
            typer.echo(f"Wrote virtual refs to {output} (tag={tag})")
        else:
            typer.echo(f"Wrote virtual refs to {output}")
        return

    # -- Standard mode: iterate simulation steps -----------------------------
    sim = _open(path)
    step_list = parse_steps(step, sim.steps)
    field_list = _parse_comma_list(fields)
    box_ranges = _parse_box_ranges(box, convert=int)
    plane_sel = (
        _resolve_plane(sim.grid, plane, plane_index, plane_coord)
        if (plane or plane_index is not None or plane_coord is not None)
        else None
    )
    compression_spec = _parse_compression(compression)

    def _postprocess(fds: FieldDataset) -> FieldDataset:
        if plane_sel is not None:
            fds = plane_sel.apply(fds)
        if box_ranges:
            fds = BoxSelection(ranges=box_ranges).apply(fds)
        if target_resolution is not None:
            from pypic.grid import GridInfo
            from pypic.regrid import regrid

            extents = tuple(
                d * s
                for d, s in zip(fds.grid.dimensions, fds.grid.spacing, strict=True)
            )
            new_dims = tuple(max(1, round(e / target_resolution)) for e in extents)
            target = GridInfo(
                dimensions=new_dims,
                spacing=tuple(target_resolution for _ in extents),
                origin=fds.grid.origin,
                geometry=fds.grid.geometry,
                dt=fds.grid.dt,
            )
            fds = regrid(fds, target)
        if to_si:
            fds = _to_si_dataset(fds)
        return fds

    if dry_run:
        typer.echo(f"Would write {len(step_list)} step(s) to {output}")
        typer.echo(f"  backend: {backend}")
        typer.echo(f"  fields: {', '.join(field_list) if field_list else 'all'}")
        if box_ranges:
            typer.echo(f"  box: {box_ranges}")
        if plane_sel is not None:
            typer.echo(f"  plane: normal={plane_sel.normal} index={plane_sel.index}")
        if target_resolution is not None:
            typer.echo(f"  target_resolution: {target_resolution}")
        if to_si:
            typer.echo("  units: converted to SI on write")
        if dtype:
            typer.echo(f"  dtype: {dtype}")
        if compression:
            typer.echo(f"  compression: {compression}")
        if tag:
            typer.echo(f"  tag: {tag}")
        return

    needs_transform = bool(plane_sel or box_ranges or target_resolution or to_si)

    if len(step_list) == 1:
        fds = sim.read(step_list[0], fields=field_list)
        if needs_transform:
            fds = _postprocess(fds)
        enc = _expand_encoding_for_vars(compression_spec, list(fds.field_names()))
        snapshot = to_zarr(
            fds,
            output,
            dtype=dtype,
            encoding=enc,
            backend=backend_arg,
            message=message,
        )
    else:
        dt = sim.grid.dt

        # Read step 0 once — reused for the encoding dict AND as the first
        # yielded pair, avoiding a duplicate I/O on large datasets.
        first = sim.read(step_list[0], fields=field_list)
        if needs_transform:
            first = _postprocess(first)
        enc = _expand_encoding_for_vars(compression_spec, list(first.field_names()))

        def _base_pairs() -> object:
            first_t = step_list[0] * dt if dt is not None else step_list[0]
            yield first_t, first
            for s in step_list[1:]:
                fds = sim.read(s, fields=field_list)
                if needs_transform:
                    fds = _postprocess(fds)
                t = s * dt if dt is not None else s
                yield t, fds

        pairs = _make_progress_iter(
            _base_pairs(),
            total=len(step_list),
            description="Writing fields",
            enabled=progress,
        )
        snapshot = to_zarr_timeseries(
            pairs,  # type: ignore[arg-type]
            output,
            dtype=dtype,
            encoding=enc,
            backend=backend_arg,
            message=message,
        )

    if tag is not None and snapshot is not None:
        from pypic.io import icechunk_create_tag

        icechunk_create_tag(output, tag, snapshot_id=snapshot)
        typer.echo(f"Wrote {len(step_list)} step(s) to {output} (tag={tag})")
    else:
        typer.echo(f"Wrote {len(step_list)} step(s) to {output}")


@convert_app.command("particles")
def convert_particles(
    path: Annotated[Path, typer.Argument(help="Simulation directory.")],
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Destination partitioned Parquet dir."),
    ],
    step: Annotated[
        str,
        typer.Option(
            "--step",
            help="Step spec: N, first, last, all, or start:stop[:stride].",
        ),
    ] = "all",
    species: Annotated[
        str | None,
        typer.Option("--species", help="Comma-separated species names or indices."),
    ] = None,
    columns: Annotated[
        str | None,
        typer.Option(
            "--columns",
            help="Subset of position,velocity to load per-particle.",
        ),
    ] = None,
    box: Annotated[
        str | None,
        typer.Option(
            "--box",
            help="Spatial crop in code units: x=lo:hi,y=lo:hi,z=lo:hi.",
        ),
    ] = None,
    sort_by: Annotated[
        Literal["position", "weight"],
        typer.Option("--sort-by", help="Pre-write sort: position or weight."),
    ] = "position",
    position_dtype: Annotated[
        str | None,
        typer.Option("--position-dtype", help="Downcast positions (e.g. float32)."),
    ] = None,
    velocity_dtype: Annotated[
        str | None,
        typer.Option("--velocity-dtype", help="Downcast velocities (e.g. float32)."),
    ] = None,
    compression_level: Annotated[
        int,
        typer.Option(
            "--compression-level", help="zstd level: 1 (fast) ... 5 (archival)."
        ),
    ] = 1,
    row_group_size: Annotated[
        int,
        typer.Option("--row-group-size", help="Rows per Parquet row group."),
    ] = 750_000,
    progress: Annotated[
        bool,
        typer.Option(
            "--progress/--no-progress",
            help="Show a rich.progress bar (auto on TTY).",
        ),
    ] = True,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Print the plan without writing."),
    ] = False,
) -> None:
    """Convert particle output to a partitioned Parquet dataset.

    Layout: ``{output}/step=000000/species=electrons/part-00000.parquet``.
    Particles are Morton-sorted (or weight-sorted) for spatial predicate
    pushdown on read.
    """
    from pypic.io._parquet import particles_to_dataset

    sim = _open(path)
    if not sim.particle_steps:
        typer.echo(f"Error: {path} has no particle output.", err=True)
        raise typer.Exit(1)

    # Resolve aliases against particle_steps directly so first/last/all
    # follow the particle cadence even when fields were dumped more often.
    step_list = parse_steps(step, sim.particle_steps)

    species_idx = _resolve_species_list(species, sim)
    columns_list = _parse_comma_list(columns)
    box_ranges = _parse_box_ranges(box, convert=float)

    if dry_run:
        typer.echo(f"Would write {len(step_list)} step(s) to {output}")
        typer.echo(f"  steps: {step_list}")
        typer.echo(
            "  species: "
            + (
                ", ".join(sim.config.species[i].name for i in species_idx)
                if species_idx
                else "all"
            )
        )
        typer.echo(f"  sort_by: {sort_by}")
        if columns_list:
            typer.echo(f"  columns: {columns_list}")
        if box_ranges:
            typer.echo(f"  box: {box_ranges}")
        if position_dtype:
            typer.echo(f"  position_dtype: {position_dtype}")
        if velocity_dtype:
            typer.echo(f"  velocity_dtype: {velocity_dtype}")
        return

    species_resolved = (
        species_idx if species_idx is not None else list(range(len(sim.config.species)))
    )

    def _pairs() -> object:
        for s in step_list:
            for sp_idx in species_resolved:
                pcl = sim.particles(s, sp_idx, columns=columns_list)
                if box_ranges is not None:
                    pcl = _filter_particles_box(pcl, box_ranges)
                if pcl.n_particles > 0:
                    yield s, sim.config.species[sp_idx].name, pcl

    pairs_iter = _make_progress_iter(
        _pairs(),
        total=len(step_list) * len(species_resolved),
        description="Writing particles",
        enabled=progress,
    )
    particles_to_dataset(
        pairs_iter,  # type: ignore[arg-type]
        output,
        position_dtype=position_dtype,
        velocity_dtype=velocity_dtype,
        compression_level=compression_level,
        row_group_size=row_group_size,
        sort_by=sort_by,
    )
    typer.echo(f"Wrote {len(step_list)} step(s) to {output}")


@convert_app.command("all")
def convert_all(
    ctx: typer.Context,
    path: Annotated[Path, typer.Argument(help="Simulation directory.")],
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Destination root directory."),
    ],
    step: Annotated[
        str,
        typer.Option("--step", help="Step spec passed to both pipelines."),
    ] = "all",
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Print the plan without writing."),
    ] = False,
    progress: Annotated[
        bool,
        typer.Option("--progress/--no-progress", help="Show progress bars."),
    ] = True,
) -> None:
    """Run both fields and particles pipelines with defaults.

    Writes fields to ``{output}/fields.zarr`` and particles (when the
    simulation has them) to ``{output}/particles/``.  Use the
    ``fields`` or ``particles`` subcommand directly for full control
    over flags.
    """
    fields_out = output / "fields.zarr"
    particles_out = output / "particles"

    sim = _open(path)
    has_particles = bool(sim.particle_steps)

    if dry_run:
        typer.echo(f"Would write fields → {fields_out}")
        if has_particles:
            typer.echo(f"Would write particles → {particles_out}")
        else:
            typer.echo("No particle output detected; skipping particles pipeline.")
        return

    output.mkdir(parents=True, exist_ok=True)
    ctx.invoke(
        convert_fields,
        path=path,
        output=fields_out,
        step=step,
        progress=progress,
    )
    if has_particles:
        ctx.invoke(
            convert_particles,
            path=path,
            output=particles_out,
            step=step,
            progress=progress,
        )


@app.command()
def info(
    path: Annotated[Path, typer.Argument(help="Simulation directory.")],
    json_output: Annotated[
        bool, typer.Option("--json", help="Output as JSON.")
    ] = False,
) -> None:
    """Show simulation metadata."""
    sim = _open(path)
    grid = sim.grid
    cfg = sim.config
    physics = sim.physics

    dims_str = " x ".join(str(d) for d in grid.dimensions)
    spacing_str = " x ".join(f"{s:.4g}" for s in grid.spacing)
    origin_str = ", ".join(f"{o:.4g}" for o in grid.origin)
    geom = grid.geometry.type.value
    steps = sim.steps
    step_info = f"{len(steps)} [{steps[0]}..{steps[-1]}]" if steps else "none"

    species_parts: list[str] = []
    species_dicts: list[dict[str, object]] = []
    for sp in cfg.species:
        if sp.charge_to_mass is not None:
            species_parts.append(f"{sp.name} (q/m={sp.charge_to_mass})")
        else:
            species_parts.append(sp.name)
        sd: dict[str, object] = {"name": sp.name}
        if sp.charge_to_mass is not None:
            sd["charge_to_mass"] = sp.charge_to_mass
        if sp.charge is not None:
            sd["charge"] = sp.charge
        if sp.mass is not None:
            sd["mass"] = sp.mass
        species_dicts.append(sd)

    rel_str = "yes" if physics.relativistic else "no"
    physics_str = (
        f"gamma={physics.gamma:.4g}, c={physics.c:.4g}, relativistic={rel_str}"
    )

    norm = sim.normalization
    norm_str = (
        "identity (SI)"
        if norm.is_identity
        else (
            f"l={norm.length_ref:.4g} m, v={norm.velocity_ref:.4g} m/s, "
            f"B={norm.b_field_ref:.4g} T, n={norm.density_ref:.4g} m^-3"
        )
    )

    stagger = cfg.metadata.get("stagger")

    lines = [
        f"Simulation: {sim.model_name} ({sim.model_type})",
        f"  Path:      {path}",
        f"  Grid:      {dims_str} ({geom})",
        f"  Spacing:   {spacing_str}",
        f"  Origin:    {origin_str}",
    ]
    if grid.dt is not None:
        lines.append(f"  dt:        {grid.dt:.4g}")
    if stagger is not None:
        lines.append(f"  Stagger:   {stagger.convention}")
    lines.append(f"  Units:     {norm_str}")
    if cfg.species:
        lines.append(f"  Species:   {', '.join(species_parts)}")
    lines.append(f"  Physics:   {physics_str}")
    lines.append(f"  Frame:     {cfg.frame}")
    if cfg.transforms:
        lines.append(f"  Transforms: {', '.join(cfg.transforms)}")
    lines.append(f"  Steps:     {step_info}")

    data: dict[str, object] = {
        "model_name": sim.model_name,
        "model_type": sim.model_type,
        "path": str(path),
        "grid": {
            "dimensions": list(grid.dimensions),
            "spacing": list(grid.spacing),
            "origin": list(grid.origin),
            "geometry": geom,
            "dt": grid.dt,
        },
        "normalization": {
            "length_ref": norm.length_ref,
            "time_ref": norm.time_ref,
            "velocity_ref": norm.velocity_ref,
            "b_field_ref": norm.b_field_ref,
            "density_ref": norm.density_ref,
        },
        "stagger": stagger.convention if stagger is not None else None,
        "species": species_dicts,
        "physics": {
            "gamma": physics.gamma,
            "c": physics.c,
            "relativistic": physics.relativistic,
        },
        "frame": cfg.frame,
        "transforms": (sorted(cfg.transforms) if cfg.transforms else []),
        "steps": {
            "count": len(steps),
            "first": steps[0] if steps else None,
            "last": steps[-1] if steps else None,
        },
    }
    _output(data, "\n".join(lines), json_mode=json_output)


@app.command()
def fields(
    path: Annotated[Path, typer.Argument(help="Simulation directory.")],
    step: Annotated[
        str, typer.Option("--step", help="Timestep (default: last).")
    ] = "last",
    mapping: Annotated[
        bool,
        typer.Option("--mapping", help="Show native-to-canonical name mapping."),
    ] = False,
    derived: Annotated[
        bool,
        typer.Option("--derived", help="List computable derived quantities."),
    ] = False,
    aux: Annotated[
        bool, typer.Option("--aux", help="List auxiliary datasets.")
    ] = False,
    all_sections: Annotated[
        bool,
        typer.Option("--all", help="Show mapping, derived, and auxiliary."),
    ] = False,
    json_output: Annotated[
        bool, typer.Option("--json", help="Output as JSON.")
    ] = False,
) -> None:
    """List available fields at a timestep."""
    sim = _open(path)
    step_val = _require_single_step(parse_steps(step, sim.steps), step)

    show_mapping = mapping or all_sections
    show_derived = derived or all_sections
    show_aux = aux or all_sections

    canonical = sim.available_fields(step_val)
    available_set = set(canonical)

    data: dict[str, object] = {
        "path": str(path),
        "step": step_val,
        "fields": canonical,
    }

    lines: list[str] = []

    if show_mapping:
        field_map = sim.available_fields_mapping(step_val)
        lines.append("Native \u2192 Canonical:")
        # Build display pairs: (native_label, canonical)
        pairs = []
        for canon in sorted(field_map):
            native_name = field_map[canon]
            pairs.append((native_name or "(computed)", canon))
        max_native = max((len(p[0]) for p in pairs), default=0)
        for native_label, canon in pairs:
            lines.append(f"  {native_label:<{max_native}}  \u2192  {canon}")
        # JSON preserves None for computed fields
        data["native_mapping"] = dict(sorted(field_map.items()))
    else:
        lines.append("Fields:")
        for name in canonical:
            lines.append(f"  {name}")

    if show_derived:
        from pypic.compute import (
            available_quantities,
            field_dependencies,
        )

        computable: list[str] = []
        for qty in available_quantities():
            try:
                deps = field_dependencies(qty)
            except (KeyError, RecursionError):
                continue
            if deps <= available_set:
                computable.append(qty)
        computable.sort()
        lines.append("")
        lines.append("Derived (computable from available fields):")
        for name in computable:
            lines.append(f"  {name}")
        data["derived"] = computable

    if show_aux:
        aux_names = sim.auxiliary_names
        lines.append("")
        lines.append("Auxiliary datasets:")
        if aux_names:
            for name in aux_names:
                lines.append(f"  {name}")
        else:
            lines.append("  (none)")
        data["auxiliary"] = aux_names

    _output(data, "\n".join(lines), json_mode=json_output)


@app.command()
def stats(
    path: Annotated[Path, typer.Argument(help="Simulation directory.")],
    field: Annotated[
        str, typer.Option("--field", help="Field name ('all' for every field).")
    ],
    step: Annotated[
        str, typer.Option("--step", help="Timestep (default: last).")
    ] = "last",
    units: Annotated[
        str | None,
        typer.Option("--units", help="Display units (e.g. nT, km/s)."),
    ] = None,
    json_output: Annotated[
        bool, typer.Option("--json", help="Output as JSON.")
    ] = False,
) -> None:
    """Print field statistics (min, max, mean, rms, NaN count)."""
    import numpy as np

    from pypic.diagnostics import field_extrema, spatial_mean, spatial_rms

    sim = _open(path)
    step_list = parse_steps(step, sim.steps)
    unit_label = units if units is not None else "code"

    def _stats_from_array(arr: FloatArray) -> dict[str, object]:
        fmin, fmax = field_extrema(arr)
        return {
            "min": float(fmin),
            "max": float(fmax),
            "mean": float(spatial_mean(arr)),
            "rms": float(spatial_rms(arr)),
            "nan_count": int(np.isnan(arr).sum()),
        }

    if field == "all":
        step_val = _require_single_step(step_list, step)
        field_names = sim.available_fields(step_val)
        ds = sim.read(step_val, fields=field_names)
        rows: list[dict[str, object]] = []
        for f in field_names:
            arr = ds.in_units(f, units) if units is not None else np.asarray(ds[f])
            rows.append({"field": f, "step": step_val, **_stats_from_array(arr)})
        max_name = max((len(str(r["field"])) for r in rows), default=5)
        hdr = (
            f"{'Field':<{max_name}}  {'min':>12}  {'max':>12}  "
            f"{'mean':>12}  {'rms':>12}  {'NaN':>5}"
        )
        text_lines = [f"Step: {step_val}  Units: {unit_label}", hdr]
        for r in rows:
            text_lines.append(
                f"{r['field']!s:<{max_name}}  {r['min']:>12.6g}  "
                f"{r['max']:>12.6g}  {r['mean']:>12.6g}  "
                f"{r['rms']:>12.6g}  {r['nan_count']:>5}"
            )
        data: dict[str, object] = {
            "path": str(path),
            "step": step_val,
            "units": unit_label,
            "fields": rows,
        }
        _output(data, "\n".join(text_lines), json_mode=json_output)
        return

    all_results = [
        {"step": s, **_stats_from_array(_get_field_array(sim, s, field, units))}
        for s in step_list
    ]

    if len(all_results) == 1:
        r = all_results[0]
        text_lines = [
            f"Field: {field}  Step: {r['step']}  Units: {unit_label}",
            f"  min:    {r['min']:.6g}",
            f"  max:    {r['max']:.6g}",
            f"  mean:   {r['mean']:.6g}",
            f"  rms:    {r['rms']:.6g}",
            f"  NaN:    {r['nan_count']}",
        ]
        data = {
            "path": str(path),
            "field": field,
            "units": unit_label,
            **r,
        }
    else:
        text_lines = [f"Field: {field}  Units: {unit_label}"]
        hdr = (
            f"{'Step':>6}  {'min':>12}  {'max':>12}  "
            f"{'mean':>12}  {'rms':>12}  {'NaN':>5}"
        )
        text_lines.append(hdr)
        for r in all_results:
            text_lines.append(
                f"{r['step']:>6}  {r['min']:>12.6g}  "
                f"{r['max']:>12.6g}  {r['mean']:>12.6g}  "
                f"{r['rms']:>12.6g}  {r['nan_count']:>5}"
            )
        data = {
            "path": str(path),
            "field": field,
            "units": unit_label,
            "steps": all_results,
        }

    _output(data, "\n".join(text_lines), json_mode=json_output)


@app.command()
def compare(
    path_a: Annotated[Path, typer.Argument(help="First simulation directory.")],
    path_b: Annotated[Path, typer.Argument(help="Second simulation directory.")],
    step: Annotated[
        str, typer.Option("--step", help="Timestep (default: last).")
    ] = "last",
    field: Annotated[
        str | None,
        typer.Option("--field", help="Field to compare (omit for all common)."),
    ] = None,
    metric: Annotated[
        str,
        typer.Option("--metric", help="Error metric: l2, linf, or both."),
    ] = "both",
    comp_units: Annotated[
        str,
        typer.Option("--units", help="Unit system: si or code."),
    ] = "si",
    method: Annotated[
        str,
        typer.Option("--method", help="Interpolation method for regridding."),
    ] = "linear",
    nan_policy: Annotated[
        str,
        typer.Option("--nan-policy", help="NaN handling: omit, propagate, raise."),
    ] = "omit",
    frame: Annotated[
        str | None,
        typer.Option(
            "--frame",
            help="Transform both to this reference frame.",
        ),
    ] = None,
    json_output: Annotated[
        bool, typer.Option("--json", help="Output as JSON.")
    ] = False,
) -> None:
    """Compare fields between two simulations."""
    _check_choice("--metric", metric, ("l2", "linf", "both"))
    _check_choice("--units", comp_units, ("si", "code"))
    _check_choice("--nan-policy", nan_policy, ("omit", "propagate", "raise"))

    from pypic.comparison import compare_fields as cmp_fields
    from pypic.comparison import field_comparison_report

    sim_a = _open(path_a)
    sim_b = _open(path_b)

    step_val = _require_single_step(parse_steps(step, sim_a.steps), step)

    if step_val not in sim_b.steps:
        typer.echo(
            f"Error: step {step_val} not available in {path_b}. "
            f"Available: {sim_b.steps}",
            err=True,
        )
        raise typer.Exit(1)

    read_fields = [field] if field is not None else None
    try:
        ds_a = sim_a.read(step_val, fields=read_fields)
        ds_b = sim_b.read(step_val, fields=read_fields)
    except KeyError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from None

    cmp_kwargs: dict[str, object] = {
        "units": comp_units,
        "method": method,
        "nan_policy": nan_policy,
    }
    if frame is not None:
        cmp_kwargs["frame"] = frame

    if field is not None:
        results: dict[str, float] = {}
        try:
            if metric in ("l2", "both"):
                results["l2"] = cmp_fields(
                    ds_a,
                    ds_b,
                    field,
                    metric="l2",
                    **cmp_kwargs,  # type: ignore[arg-type]
                )
            if metric in ("linf", "both"):
                results["linf"] = cmp_fields(
                    ds_a,
                    ds_b,
                    field,
                    metric="linf",
                    **cmp_kwargs,  # type: ignore[arg-type]
                )
        except (KeyError, ValueError, NotImplementedError) as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(1) from None

        grid_a, grid_b = ds_a.grid, ds_b.grid
        res_ratio = [
            max(sa, sb) / min(sa, sb) if min(sa, sb) else float("inf")
            for sa, sb in zip(grid_a.spacing, grid_b.spacing, strict=True)
        ]

        text_lines = [
            f"Comparing: {path_a} vs {path_b}",
            f"Field: {field}  Step: {step_val}  Units: {comp_units}",
        ]
        if "l2" in results:
            text_lines.append(f"  L2 relative error: {results['l2']:.6g}")
        if "linf" in results:
            text_lines.append(f"  L-inf error:       {results['linf']:.6g}")
        dims_a = " x ".join(str(d) for d in grid_a.dimensions)
        dims_b = " x ".join(str(d) for d in grid_b.dimensions)
        text_lines.append("")
        text_lines.append("Grid:")
        text_lines.append(f"  A dims: {dims_a}")
        text_lines.append(f"  B dims: {dims_b}")
        text_lines.append(
            f"  Resolution ratio: {' x '.join(f'{r:.2g}' for r in res_ratio)}"
        )

        data: dict[str, object] = {
            "path_a": str(path_a),
            "path_b": str(path_b),
            "step": step_val,
            "field": field,
            "units": comp_units,
            "metric": metric,
            **results,
            "grid": {
                "dimensions_a": list(grid_a.dimensions),
                "dimensions_b": list(grid_b.dimensions),
                "resolution_ratio": res_ratio,
            },
        }
    else:
        try:
            report = field_comparison_report(
                ds_a,
                ds_b,
                **cmp_kwargs,  # type: ignore[arg-type]
            )
        except (ValueError, NotImplementedError) as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(1) from None

        text_lines = [
            f"Comparing: {path_a} vs {path_b}",
            f"Step: {step_val}  Units: {comp_units}",
            "",
        ]

        field_results = report["fields"]
        if field_results:
            max_name = max(len(n) for n in field_results)
            hdr = f"{'Field':<{max_name}}  {'L2 rel.':>12}  {'L-inf':>12}"
            text_lines.append(hdr)
            for fname in sorted(field_results):
                vals = field_results[fname]
                text_lines.append(
                    f"{fname:<{max_name}}  {vals['l2']:>12.6g}  {vals['linf']:>12.6g}"
                )
        else:
            text_lines.append("No common fields found.")

        grid_ctx = report.get("grid", {})
        if grid_ctx:
            text_lines.append("")
            text_lines.append("Grid:")
            if "common_dimensions" in grid_ctx:
                dims = " x ".join(str(d) for d in grid_ctx["common_dimensions"])
                text_lines.append(f"  Common dims: {dims}")
            if "resolution_ratio" in grid_ctx:
                ratio = " x ".join(f"{r:.2g}" for r in grid_ctx["resolution_ratio"])
                text_lines.append(f"  Resolution ratio: {ratio}")

        data = {
            "path_a": str(path_a),
            "path_b": str(path_b),
            "step": step_val,
            "units": comp_units,
            **report,
        }

    _output(data, "\n".join(text_lines), json_mode=json_output)


@app.command()
def validate(
    path: Annotated[Path, typer.Argument(help="Simulation directory.")],
    step: Annotated[
        str, typer.Option("--step", help="Timestep (default: last).")
    ] = "last",
    json_output: Annotated[
        bool, typer.Option("--json", help="Output as JSON.")
    ] = False,
) -> None:
    """Quick health check: NaN census, div B, field energy."""
    import numpy as np

    from pypic.derived import electric_energy_density, magnetic_energy_density
    from pypic.diagnostics import field_energy, max_div_b

    sim = _open(path)
    step_val = _require_single_step(parse_steps(step, sim.steps), step)
    ds = sim.read(step_val)

    field_names = sorted(ds.field_names())
    nan_fields: dict[str, int] = {}
    for name in field_names:
        count = int(np.isnan(ds[name]).sum())
        if count > 0:
            nan_fields[name] = count
    total_nan = sum(nan_fields.values())

    has_b = all(ds.has_field(f) for f in ("B_1", "B_2", "B_3"))
    has_e = all(ds.has_field(f) for f in ("E_1", "E_2", "E_3"))

    div_b_val: float | None = None
    b_energy: float | None = None
    e_energy: float | None = None

    if has_b:
        b1, b2, b3 = ds["B_1"], ds["B_2"], ds["B_3"]
        b_mag = np.sqrt(b1**2 + b2**2 + b3**2)
        div_b_val = float(max_div_b(b1, b2, b3, *ds.grid.spacing))
        b_energy = float(field_energy(magnetic_energy_density(b_mag), ds.grid.spacing))
    if has_e:
        e1, e2, e3 = ds["E_1"], ds["E_2"], ds["E_3"]
        e_mag = np.sqrt(e1**2 + e2**2 + e3**2)
        e_energy = float(field_energy(electric_energy_density(e_mag), ds.grid.spacing))

    # Energy drift from auxiliary time-series (code-agnostic)
    energy_drift_total: float | None = None
    energy_drift_last: float | None = None
    if "conserved_quantities" in sim.auxiliary_names:
        try:
            tab = sim.auxiliary("conserved_quantities")
            if "total_energy" in tab and len(tab) >= 2:
                e_arr = tab["total_energy"]
                e0 = float(e_arr[0])
                if e0 != 0:
                    abs_e0 = abs(e0)
                    energy_drift_total = float((e_arr[-1] - e0) / abs_e0 * 100)
                    energy_drift_last = float((e_arr[-1] - e_arr[-2]) / abs_e0 * 100)
        except (KeyError, TypeError, IndexError):
            pass

    lines = [
        f"Validation: {sim.model_name} ({sim.model_type})  Step: {step_val}",
        f"  Fields:     {', '.join(field_names)}",
    ]
    if nan_fields:
        nan_detail = ", ".join(f"{k}: {v}" for k, v in nan_fields.items())
        lines.append(f"  NaN census: {total_nan} total ({nan_detail})")
    else:
        lines.append(f"  NaN census: {total_nan} total")
    if div_b_val is not None:
        lines.append(f"  max |div B|: {div_b_val:.6g}")
    if b_energy is not None:
        lines.append(f"  B energy:   {b_energy:.6g}")
    if e_energy is not None:
        lines.append(f"  E energy:   {e_energy:.6g}")
    if energy_drift_total is not None:
        lines.append(f"  ΔE total:   {energy_drift_total:+.4g}%")
    if energy_drift_last is not None:
        lines.append(f"  ΔE last step: {energy_drift_last:+.4g}%")

    result: dict[str, object] = {
        "path": str(path),
        "model_name": sim.model_name,
        "model_type": sim.model_type,
        "step": step_val,
        "fields": field_names,
        "nan_total": total_nan,
        "nan_fields": nan_fields,
        "max_div_b": div_b_val,
        "b_energy": b_energy,
        "e_energy": e_energy,
        "energy_drift_total_pct": energy_drift_total,
        "energy_drift_last_pct": energy_drift_last,
    }
    _output(result, "\n".join(lines), json_mode=json_output)


def _stitch_animation(
    frame_template: str,
    steps: list[int],
    output_path: str,
    fps: int,
) -> None:
    """Stitch rendered frames into a video via ffmpeg."""
    import shutil
    import subprocess

    if shutil.which("ffmpeg") is None:
        typer.echo(
            "Error: ffmpeg not found. Install ffmpeg for animation export.",
            err=True,
        )
        raise typer.Exit(1)

    frames = [frame_template.format(step=s) for s in steps]
    for f in frames:
        if not Path(f).exists():
            typer.echo(f"Error: expected frame not found: {f}", err=True)
            raise typer.Exit(1)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    # Use concat demuxer for arbitrary filenames
    concat_path = Path(frames[0]).parent / "_concat.txt"
    try:
        concat_path.write_text(
            "\n".join(f"file '{Path(f).resolve()}'" for f in frames),
            encoding="utf-8",
        )
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-r",
            str(fps),
            "-i",
            str(concat_path),
        ]
        if output_path.endswith(".gif"):
            cmd += ["-vf", "split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse"]
        else:
            cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p"]
        cmd.append(output_path)

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            typer.echo(f"ffmpeg error: {result.stderr}", err=True)
            raise typer.Exit(1)
        typer.echo(f"Animation saved: {output_path}")
    finally:
        concat_path.unlink(missing_ok=True)


def _render_plot(
    path: Path,
    step_val: int,
    field: str,
    plane_str: str | None,
    index: int | None,
    coord: float | None,
    units: str | None,
    frame: str | None,
    output: str | None,
    fmt: str | None,
    dpi: int,
    res: str | None,
    vmin_arg: float | None,
    vmax_arg: float | None,
    colormap: str | None,
    scale: str,
    linthresh: float | None,
    theme: str | None = None,
    contour_field: str | None = None,
    contour_levels: int = 5,
) -> None:
    """Render a single plot frame (called once per step)."""
    import matplotlib

    matplotlib.use("Agg" if output else matplotlib.get_backend())

    from pypic.plotting._colormaps import auto_clim, is_positive_definite
    from pypic.plotting._resolve import maybe_save
    from pypic.plotting.slices import plot_field_slice

    sim = _open(path)
    try:
        read_fields = [field]
        if contour_field is not None:
            read_fields.append(contour_field)
        ds = sim.read(step_val, fields=read_fields)

        if frame is not None:
            ds = ds.transform_to(frame)

        plane_sel: PlaneSelection | None = _resolve_plane(
            ds.grid, plane_str, index, coord
        )

        scale_kwargs: dict[str, object] = {}
        if scale == "log":
            scale_kwargs["log_scale"] = True
        elif scale == "symlog":
            scale_kwargs["symlog"] = True
            if linthresh is not None:
                scale_kwargs["linthresh"] = linthresh

        from pypic.plotting._resolve import prepare_data, resolve_field_values

        vmin, vmax = vmin_arg, vmax_arg
        preview = (
            prepare_data(ds, plane_sel)
            if (vmin is None or vmax is None or res is not None)
            else None
        )

        if vmin is None or vmax is None:
            assert preview is not None
            values = resolve_field_values(preview, field, units)
            info = preview.field_info(field)
            pos_def = is_positive_definite(field, values, info)
            auto_vmin, auto_vmax = auto_clim(values, positive_definite=pos_def)
            vmin = vmin if vmin is not None else auto_vmin
            vmax = vmax if vmax is not None else auto_vmax

        if res is not None:
            assert preview is not None
            max_w, max_h = _parse_resolution(res)
            dims = preview.grid.dimensions
            stride_x = max(1, dims[0] // max_w)
            stride_y = max(1, dims[1] // max_h)
            names = preview.grid.surviving_axis_names
            assert plane_sel is not None
            ds = plane_sel.apply(ds)
            ds = ds.isel(
                {
                    names[0]: slice(None, None, stride_x),
                    names[1]: slice(None, None, stride_y),
                }
            )
            plane_sel = None  # already applied

        time = step_val * ds.grid.dt if ds.grid.dt else None

        fig, ax = plot_field_slice(
            ds,
            field,
            plane=plane_sel,
            units=units,
            vmin=vmin,
            vmax=vmax,
            cmap=colormap,
            step=step_val,
            time=time,
            theme=theme,
            **scale_kwargs,  # type: ignore[arg-type]
        )

        if contour_field is not None:
            from pypic.plotting.slices import add_contours

            add_contours(ax, ds, contour_field, plane=plane_sel, levels=contour_levels)
    except KeyError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from None

    if output is not None:
        maybe_save(fig, output, dpi=dpi, fmt=fmt)
    else:
        import matplotlib.pyplot as plt

        plt.show()


@app.command()
def plot(
    path: Annotated[Path, typer.Argument(help="Simulation directory.")],
    field: Annotated[str, typer.Option("--field", help="Field name.")],
    step: Annotated[
        str, typer.Option("--step", help="Timestep (default: last).")
    ] = "last",
    plane: Annotated[
        str | None,
        typer.Option("--plane", help="Slice plane (xy/xz/yz, axis pair, or normal)."),
    ] = None,
    index: Annotated[
        int | None,
        typer.Option("--index", help="Cell index along normal axis."),
    ] = None,
    coord: Annotated[
        float | None,
        typer.Option("--coord", help="Physical coordinate along normal."),
    ] = None,
    units: Annotated[
        str | None,
        typer.Option("--units", help="Display units (e.g. nT, km/s)."),
    ] = None,
    frame: Annotated[
        str | None,
        typer.Option("--frame", help="Reference frame."),
    ] = None,
    output: Annotated[
        str | None,
        typer.Option("--output", help="Output file (batch template: {step:06d}.png)."),
    ] = None,
    fmt: Annotated[
        str | None,
        typer.Option("--format", help="Image format: png, pdf, svg."),
    ] = None,
    dpi: Annotated[
        int,
        typer.Option("--dpi", help="Output DPI."),
    ] = 150,
    res: Annotated[
        str | None,
        typer.Option("--res", help="Max grid resolution WxH for fast preview."),
    ] = None,
    color_vmin: Annotated[
        float | None,
        typer.Option("--vmin", help="Color range minimum."),
    ] = None,
    color_vmax: Annotated[
        float | None,
        typer.Option("--vmax", help="Color range maximum."),
    ] = None,
    colormap: Annotated[
        str | None,
        typer.Option("--colormap", help="Colormap name."),
    ] = None,
    scale: Annotated[
        str,
        typer.Option("--scale", help="Color scale: linear, log, or symlog."),
    ] = "linear",
    linthresh: Annotated[
        float | None,
        typer.Option("--linthresh", help="Linear threshold for symlog scale."),
    ] = None,
    jobs: Annotated[
        int,
        typer.Option("--jobs", help="Parallel workers for batch rendering."),
    ] = 1,
    theme: Annotated[
        str | None,
        typer.Option("--theme", help="Plot theme name (e.g. dark, light, synthwave)."),
    ] = None,
    contour: Annotated[
        str | None,
        typer.Option("--contour", help="Overlay contour lines from this field."),
    ] = None,
    contour_levels: Annotated[
        int,
        typer.Option("--contour-levels", help="Number of contour levels."),
    ] = 5,
    animate: Annotated[
        str | None,
        typer.Option("--animate", help="Stitch frames into video (e.g. out.mp4)."),
    ] = None,
    fps: Annotated[
        int,
        typer.Option("--fps", help="Animation framerate."),
    ] = 24,
) -> None:
    """Plot a 2D field slice."""
    _check_choice("--scale", scale, ("linear", "log", "symlog"))
    _check_choice("--format", fmt, ("png", "pdf", "svg"))

    if animate is not None and output is None:
        msg = "--animate requires --output to know where frames are."
        raise typer.BadParameter(msg)

    sim = _open(path)
    step_list = parse_steps(step, sim.steps)

    if animate is not None and len(step_list) < 2:
        msg = "--animate requires multiple steps (use --step all or a range)."
        raise typer.BadParameter(msg)

    if len(step_list) > 1:
        if output is None:
            msg = (
                "Multi-step plotting requires --output with a template "
                "(e.g. 'frames/{step:06d}.png')."
            )
            raise typer.BadParameter(msg)
        try:
            test_path = output.format(step=0)
        except (KeyError, IndexError, ValueError):
            msg = (
                f"Output template {output!r} must contain "
                "'{{step}}' for batch rendering."
            )
            raise typer.BadParameter(msg) from None
        if test_path == output.format(step=1):
            msg = (
                f"Output template {output!r} must contain a '{{step}}' "
                "placeholder for batch rendering — all steps would "
                "write to the same file."
            )
            raise typer.BadParameter(msg)

    def _resolve_output(sv: int) -> str | None:
        if output and len(step_list) > 1:
            return output.format(step=sv)
        return output

    def _run_frame(sv: int) -> None:
        out = _resolve_output(sv)
        if out:
            Path(out).parent.mkdir(parents=True, exist_ok=True)
        _render_plot(
            path,
            sv,
            field,
            plane,
            index,
            coord,
            units,
            frame,
            out,
            fmt,
            dpi,
            res,
            color_vmin,
            color_vmax,
            colormap,
            scale,
            linthresh,
            theme=theme,
            contour_field=contour,
            contour_levels=contour_levels,
        )

    if len(step_list) > 1 and jobs > 1:
        import concurrent.futures
        import functools

        # Matplotlib is not thread-safe, so use process-based workers.
        # functools.partial on the module-level _render_plot is picklable.
        render = functools.partial(
            _render_plot,
            path,
            field=field,
            plane_str=plane,
            index=index,
            coord=coord,
            units=units,
            frame=frame,
            fmt=fmt,
            dpi=dpi,
            res=res,
            vmin_arg=color_vmin,
            vmax_arg=color_vmax,
            colormap=colormap,
            scale=scale,
            linthresh=linthresh,
            theme=theme,
            contour_field=contour,
            contour_levels=contour_levels,
        )
        with concurrent.futures.ProcessPoolExecutor(max_workers=jobs) as pool:
            futs = [
                pool.submit(render, step_val=sv, output=_resolve_output(sv))
                for sv in step_list
            ]
            for fut in concurrent.futures.as_completed(futs):
                fut.result()
    else:
        for sv in step_list:
            _run_frame(sv)

    if animate is not None:
        assert output is not None  # validated above
        _stitch_animation(output, step_list, animate, fps)


@app.command(name="plot-compare")
def plot_compare(
    path_a: Annotated[Path, typer.Argument(help="First simulation directory.")],
    path_b: Annotated[Path, typer.Argument(help="Second simulation directory.")],
    field: Annotated[str, typer.Option("--field", help="Field name.")],
    step: Annotated[
        str, typer.Option("--step", help="Timestep (default: last).")
    ] = "last",
    plane: Annotated[
        str | None,
        typer.Option("--plane", help="Slice plane (xy/xz/yz, axis pair, or normal)."),
    ] = None,
    comp_units: Annotated[
        str,
        typer.Option("--units", help="Unit system: si or code."),
    ] = "si",
    frame: Annotated[
        str | None,
        typer.Option("--frame", help="Transform both to this reference frame."),
    ] = None,
    color_vmin: Annotated[
        float | None,
        typer.Option("--vmin", help="Field panel color minimum."),
    ] = None,
    color_vmax: Annotated[
        float | None,
        typer.Option("--vmax", help="Field panel color maximum."),
    ] = None,
    diff_vmin: Annotated[
        float | None,
        typer.Option("--diff-vmin", help="Difference panel color minimum."),
    ] = None,
    diff_vmax: Annotated[
        float | None,
        typer.Option("--diff-vmax", help="Difference panel color maximum."),
    ] = None,
    output: Annotated[
        str | None,
        typer.Option("--output", help="Output file."),
    ] = None,
    fmt: Annotated[
        str | None,
        typer.Option("--format", help="Image format: png, pdf, svg."),
    ] = None,
    dpi: Annotated[
        int,
        typer.Option("--dpi", help="Output DPI."),
    ] = 150,
    colormap: Annotated[
        str | None,
        typer.Option("--colormap", help="Colormap for field panels."),
    ] = None,
    method: Annotated[
        str,
        typer.Option("--method", help="Interpolation method for regridding."),
    ] = "linear",
    theme: Annotated[
        str | None,
        typer.Option("--theme", help="Plot theme name (e.g. dark, light, synthwave)."),
    ] = None,
) -> None:
    """Three-panel comparison plot: A | B | difference."""
    _check_choice("--units", comp_units, ("si", "code"))
    _check_choice("--format", fmt, ("png", "pdf", "svg"))

    import matplotlib

    matplotlib.use("Agg" if output else matplotlib.get_backend())

    from pypic.plotting._resolve import maybe_save
    from pypic.plotting.comparison import plot_comparison
    from pypic.regrid import align_grids

    sim_a = _open(path_a)
    sim_b = _open(path_b)

    step_val = _require_single_step(parse_steps(step, sim_a.steps), step)
    if step_val not in sim_b.steps:
        typer.echo(
            f"Error: step {step_val} not available in {path_b}. "
            f"Available: {sim_b.steps}",
            err=True,
        )
        raise typer.Exit(1)

    try:
        ds_a = sim_a.read(step_val, fields=[field])
        ds_b = sim_b.read(step_val, fields=[field])

        if frame is not None:
            ds_a = ds_a.transform_to(frame)
            ds_b = ds_b.transform_to(frame)

        ds_a, ds_b = align_grids(ds_a, ds_b, method=method)

        plane_sel = _resolve_plane(ds_a.grid, plane, None, None)

        time = step_val * ds_a.grid.dt if ds_a.grid.dt else None

        # Convert to SI before plotting when requested (plot_comparison
        # only accepts display-unit strings, not "si"/"code" selectors)
        if comp_units == "si":
            import numpy as np

            ds_a = ds_a.with_field(field, np.asarray(ds_a.in_si(field)))
            ds_b = ds_b.with_field(field, np.asarray(ds_b.in_si(field)))

        fig, _ = plot_comparison(
            ds_a,
            ds_b,
            field,
            plane=plane_sel,
            vmin=color_vmin,
            vmax=color_vmax,
            diff_vmin=diff_vmin,
            diff_vmax=diff_vmax,
            cmap=colormap,
            step=step_val,
            time=time,
            labels=(path_a.name, path_b.name),
            show_error=True,
            theme=theme,
        )
    except (KeyError, ValueError, NotImplementedError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from None

    if output is not None:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        maybe_save(fig, output, dpi=dpi, fmt=fmt)
    else:
        import matplotlib.pyplot as plt

        plt.show()


if __name__ == "__main__":
    app()
