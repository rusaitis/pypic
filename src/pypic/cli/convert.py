"""``pypic convert``: fields to Zarr, particles to Parquet."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal

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

if TYPE_CHECKING:
    from pypic.containers import ParticleData
    from pypic.readers._registry import Simulation


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


@convert_app.command("fields")
def convert_fields(
    path: Annotated[Path, typer.Argument(help="Simulation directory or HDF5 file.")],
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Destination Zarr store directory."),
    ],
    step: StepOption = "all",
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
    progress: ProgressOption = True,
    dry_run: DryRunOption = False,
) -> None:
    """Convert simulation fields to a Zarr v3 store.

    Writes one timestep via to_zarr or a multi-step time-series via
    to_zarr_timeseries.  Optionally crops, slices, regrids, and/or
    converts to SI units.  Pass --backend icechunk for versioned
    storage, or --virtual to persist HDF5 byte-range references
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
        # Read step 0 once — reused for the encoding dict AND as the first
        # yielded pair, avoiding a duplicate I/O on large datasets.
        first = sim.read(step_list[0], fields=field_list)
        if needs_transform:
            first = _postprocess(first)
        enc = _expand_encoding_for_vars(compression_spec, list(first.field_names()))

        def _base_pairs() -> object:
            yield _time_coordinate(first, step_list[0]), first
            for s in step_list[1:]:
                fds = sim.read(s, fields=field_list)
                if needs_transform:
                    fds = _postprocess(fds)
                yield _time_coordinate(fds, s), fds

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
    step: StepOption = "all",
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
    progress: ProgressOption = True,
    dry_run: DryRunOption = False,
) -> None:
    """Convert particle output to a partitioned Parquet dataset.

    Layout: {output}/step=000000/species=electrons/part-00000.parquet.
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
    step: StepOption = "all",
    dry_run: DryRunOption = False,
    progress: ProgressOption = True,
) -> None:
    """Run both fields and particles pipelines with defaults.

    Writes fields to {output}/fields.zarr and particles (when the
    simulation has them) to {output}/particles/.  Use the
    fields or particles subcommand directly for full control
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
