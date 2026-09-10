"""``pypic reduce``: collapse fields along one or more axes."""

from __future__ import annotations

from typing import Annotated, Literal

import typer

from pypic.cli._options import (
    BackendOption,
    CompressionOption,
    DryRunOption,
    DtypeOption,
    FieldsOption,
    InputPath,
    MessageOption,
    NanPolicyOption,
    PlaneCoordOption,
    PlaneIndexOption,
    ProgressOption,
    StepOption,
    TagOption,
    ZarrOutputOption,
)
from pypic.cli._shared import (
    ZarrTarget,
    _open,
    _parse_box_ranges,
    _parse_comma_list,
    _resolve_plane,
    _write_steps,
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
    path: InputPath,
    output: ZarrOutputOption,
    axis: Annotated[
        str,
        typer.Option(
            "--axis",
            help='Axis name(s) to reduce. Single: "z". Multi-axis: "y,z".',
        ),
    ],
    reduction: Annotated[
        Literal[
            "integrate",
            "sum",
            "mean",
            "median",
            "max",
            "min",
            "std",
            "var",
            "argmax",
            "argmin",
        ],
        typer.Option(
            "--reduction",
            help="How to collapse the axis.",
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
    fields: FieldsOption = None,
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
    plane_index: PlaneIndexOption = None,
    plane_coord: PlaneCoordOption = None,
    nan_policy: NanPolicyOption = "omit",
    dtype: DtypeOption = None,
    compression: CompressionOption = None,
    backend: BackendOption = "zarr",
    message: MessageOption = None,
    tag: TagOption = None,
    progress: ProgressOption = True,
    dry_run: DryRunOption = False,
) -> None:
    """Reduce simulation fields along one or more axes.

    Writes one timestep via to_zarr or a multi-step time-series via
    to_zarr_timeseries.  Composes with --plane and --box
    (applied before the reduction) and --weight (yt-style
    density-weighted average; mean / integrate only).
    """
    from pypic.reductions import reduce as reduce_fn
    from pypic.selections import BoxSelection

    target = ZarrTarget(
        output, backend, dtype=dtype, compression=compression, message=message, tag=tag
    )

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
            reduction=reduction,
            fields=field_list or None,
            weight=weight,
            nan_policy=nan_policy,
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

    _write_steps(
        sim,
        step_list,
        target,
        read_fields=read_fields,
        transform=_reduce_one,
        description="Reducing fields",
        progress=progress,
        done=f"Reduced {len(step_list)} step(s) to {output}",
    )
