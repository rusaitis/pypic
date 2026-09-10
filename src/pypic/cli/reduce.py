"""``pypic reduce``: collapse fields along one or more axes."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import typer

from pypic.cli._options import DryRunOption, ProgressOption, StepOption
from pypic.cli._shared import (
    _expand_encoding_for_vars,
    _make_progress_iter,
    _open,
    _parse_box_ranges,
    _parse_comma_list,
    _parse_compression,
    _resolve_plane,
    _time_coordinate,
    parse_steps,
)
from pypic.dataset import FieldDataset

reduce_app = typer.Typer(
    name="reduce",
    help=(
        "Reduce simulation fields along one or more axes "
        "(column densities, slab means, ...)."
    ),
    no_args_is_help=True,
)


@reduce_app.command("apply")
def reduce_apply(
    path: Annotated[Path, typer.Argument(help="Simulation directory or HDF5 file.")],
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Destination Zarr store directory."),
    ],
    axis: Annotated[
        str,
        typer.Option(
            "--axis",
            help='Axis name(s) to reduce. Single: "z". Multi-axis: "y,z".',
        ),
    ],
    reduction: Annotated[
        str,
        typer.Option(
            "--reduction",
            help=(
                "How to collapse the axis: integrate (default), sum, mean, "
                "median, max, min, std, var, argmax, argmin."
            ),
        ),
    ] = "integrate",
    weight: Annotated[
        str | None,
        typer.Option(
            "--weight",
            help=(
                "Field name to weight by (mean / integrate only). "
                "Produces a yt-style density- or emission-weighted average."
            ),
        ),
    ] = None,
    fields: Annotated[
        str | None,
        typer.Option(
            "--fields", help="Comma-separated field names (e.g. B,E_3,rho_c)."
        ),
    ] = None,
    step: StepOption = "all",
    box: Annotated[
        str | None,
        typer.Option(
            "--box", help="Spatial crop as axis=lo:hi,...,axis=lo:hi (index-based)."
        ),
    ] = None,
    plane: Annotated[
        str | None,
        typer.Option(
            "--plane",
            help="Pre-slice plane before reducing (e.g. xy, xz, z).",
        ),
    ] = None,
    plane_index: Annotated[
        int | None,
        typer.Option("--plane-index", help="Cell index along the plane normal."),
    ] = None,
    plane_coord: Annotated[
        float | None,
        typer.Option("--plane-coord", help="Physical coord along the plane normal."),
    ] = None,
    nan_policy: Annotated[
        str,
        typer.Option(
            "--nan-policy",
            help="NaN handling: omit (default), propagate, or raise.",
        ),
    ] = "omit",
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
    progress: ProgressOption = True,
    dry_run: DryRunOption = False,
) -> None:
    """Reduce simulation fields along one or more axes.

    Writes one timestep via to_zarr or a multi-step time-series via
    to_zarr_timeseries.  Composes with --plane and --box
    (applied before the reduction) and --weight (yt-style
    density-weighted average; mean / integrate only).
    """
    from pypic.io import to_zarr, to_zarr_timeseries
    from pypic.reductions import reduce as reduce_fn
    from pypic.selections import BoxSelection

    if tag is not None and backend != "icechunk":
        msg = "--tag requires --backend icechunk."
        raise typer.BadParameter(msg)
    if message is not None and backend != "icechunk":
        msg = "--message requires --backend icechunk."
        raise typer.BadParameter(msg)
    backend_arg = None if backend == "zarr" else backend

    axis_list = _parse_comma_list(axis)
    if not axis_list:
        msg = "--axis must name at least one axis."
        raise typer.BadParameter(msg)
    axis_spec: str | tuple[str, ...] = (
        axis_list[0] if len(axis_list) == 1 else tuple(axis_list)
    )

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

    # A weight field must be pulled from disk even when ``--fields`` narrows
    # the load list; ``reduce(fields=...)`` drops it again on output.  Without
    # this a restricted read fails on the missing weight at compute time.
    read_fields: list[str] | None = list(field_list) if field_list else None
    if read_fields is not None and weight is not None and weight not in read_fields:
        read_fields = [*read_fields, weight]

    def _apply_selections(fds: FieldDataset) -> FieldDataset:
        if plane_sel is not None:
            fds = plane_sel.apply(fds)
        if box_ranges:
            fds = BoxSelection(ranges=box_ranges).apply(fds)
        return fds

    def _reduce_one(fds: FieldDataset) -> FieldDataset:
        fds = _apply_selections(fds)
        return reduce_fn(
            fds,
            axis_spec,
            reduction=reduction,  # type: ignore[arg-type]
            fields=field_list or None,
            weight=weight,
            nan_policy=nan_policy,  # type: ignore[arg-type]
        )

    if dry_run:
        typer.echo(f"Would reduce {len(step_list)} step(s) to {output}")
        typer.echo(f"  backend: {backend}")
        typer.echo(f"  axis: {axis_spec!r}")
        typer.echo(f"  reduction: {reduction}")
        if weight is not None:
            typer.echo(f"  weight: {weight}")
        typer.echo(f"  fields: {', '.join(field_list) if field_list else 'all'}")
        if box_ranges:
            typer.echo(f"  box: {box_ranges}")
        if plane_sel is not None:
            typer.echo(f"  plane: normal={plane_sel.normal} index={plane_sel.index}")
        typer.echo(f"  nan_policy: {nan_policy}")
        if dtype:
            typer.echo(f"  dtype: {dtype}")
        if compression:
            typer.echo(f"  compression: {compression}")
        if tag:
            typer.echo(f"  tag: {tag}")
        return

    if len(step_list) == 1:
        fds = sim.read(step_list[0], fields=read_fields)
        reduced = _reduce_one(fds)
        enc = _expand_encoding_for_vars(compression_spec, list(reduced.field_names()))
        snapshot = to_zarr(
            reduced,
            output,
            dtype=dtype,
            encoding=enc,
            backend=backend_arg,
            message=message,
        )
    else:
        first = _reduce_one(sim.read(step_list[0], fields=read_fields))
        enc = _expand_encoding_for_vars(compression_spec, list(first.field_names()))

        def _base_pairs() -> object:
            yield _time_coordinate(first, step_list[0]), first
            for s in step_list[1:]:
                reduced = _reduce_one(sim.read(s, fields=read_fields))
                yield _time_coordinate(reduced, s), reduced

        pairs = _make_progress_iter(
            _base_pairs(),
            total=len(step_list),
            description="Reducing fields",
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
        typer.echo(f"Reduced {len(step_list)} step(s) to {output} (tag={tag})")
    else:
        typer.echo(f"Reduced {len(step_list)} step(s) to {output}")
