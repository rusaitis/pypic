"""``pypic plot`` and ``plot-compare``: publication figures from the shell."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal

import typer

from pypic.cli._options import (
    ComparisonUnitsOption,
    DpiOption,
    FieldOption,
    FormatOption,
    MethodOption,
    PathA,
    PathB,
    PlaneOption,
    SimulationPath,
    ThemeOption,
    TimestepOption,
    UnitsOption,
)
from pypic.cli._shared import (
    _open,
    _require_single_step,
    _resolve_plane,
    parse_steps,
)

if TYPE_CHECKING:
    from pypic.selections import PlaneSelection


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

        # check=False: the returncode is handled below, with ffmpeg's own
        # stderr, which is more use than a CalledProcessError traceback.
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
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

        time = ds.time

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


def plot(
    path: SimulationPath,
    field: FieldOption,
    step: TimestepOption = "last",
    plane: PlaneOption = None,
    index: Annotated[
        int | None,
        typer.Option("--index", help="Cell index along normal axis."),
    ] = None,
    coord: Annotated[
        float | None,
        typer.Option("--coord", help="Physical coordinate along normal."),
    ] = None,
    units: UnitsOption = None,
    frame: Annotated[
        str | None,
        typer.Option("--frame", help="Reference frame."),
    ] = None,
    output: Annotated[
        str | None,
        typer.Option("--output", help="Output file (batch template: {step:06d}.png)."),
    ] = None,
    fmt: FormatOption = None,
    dpi: DpiOption = 150,
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
        Literal["linear", "log", "symlog"],
        typer.Option("--scale", help="Color scale."),
    ] = "linear",
    linthresh: Annotated[
        float | None,
        typer.Option("--linthresh", help="Linear threshold for symlog scale."),
    ] = None,
    jobs: Annotated[
        int,
        typer.Option("--jobs", help="Parallel workers for batch rendering."),
    ] = 1,
    theme: ThemeOption = None,
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


def plot_compare(
    path_a: PathA,
    path_b: PathB,
    field: FieldOption,
    step: TimestepOption = "last",
    plane: PlaneOption = None,
    comp_units: ComparisonUnitsOption = "si",
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
    fmt: FormatOption = None,
    dpi: DpiOption = 150,
    colormap: Annotated[
        str | None,
        typer.Option("--colormap", help="Colormap for field panels."),
    ] = None,
    method: MethodOption = "linear",
    theme: ThemeOption = None,
) -> None:
    """Three-panel comparison plot: A | B | difference."""
    import matplotlib

    matplotlib.use("Agg" if output else matplotlib.get_backend())

    from pypic.plotting._resolve import maybe_save
    from pypic.plotting.comparison import plot_comparison
    from pypic.regridding import align_grids

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

        time = ds_a.time

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
