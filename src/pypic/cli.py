"""pypic command-line interface."""

import importlib.metadata
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

if TYPE_CHECKING:
    from pypic.readers._registry import Simulation

app = typer.Typer(
    name="pypic",
    help="Inspect and compare plasma simulation output.",
    pretty_exceptions_enable=False,
    no_args_is_help=True,
)


# -- Global options ----------------------------------------------------------


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
    logging.basicConfig(level=getattr(logging, level_name, logging.WARNING))
    if debug:
        app.pretty_exceptions_enable = True


# -- Step parsing ------------------------------------------------------------


def parse_steps(raw: str, sim: "Simulation") -> list[int]:
    """Parse ``--step`` syntax into a list of timestep indices.

    Supports: ``N`` (single int), ``first``, ``last``, ``all``,
    ``start:stop:stride`` (inclusive stop).
    """
    raw = raw.strip()
    if raw == "all":
        return sim.steps
    if raw == "first":
        return [sim.first_step]
    if raw == "last":
        return [sim.last_step]
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
        return [
            s for s in sim.steps if start <= s <= stop and (s - start) % stride == 0
        ]
    try:
        return [int(raw)]
    except ValueError:
        msg = (
            f"Invalid --step value {raw!r}. "
            "Use a number, first, last, all, or start:stop:stride."
        )
        raise typer.BadParameter(msg) from None


# -- Helpers -----------------------------------------------------------------


def _open(path: Path) -> "Simulation":
    """Open a simulation, translating errors to CLI messages."""
    from pypic.readers._registry import open_simulation

    try:
        return open_simulation(path)
    except (FileNotFoundError, OSError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from None


def _output(data: dict[str, object], text: str, *, json_mode: bool) -> None:
    """Print JSON or formatted text."""
    if json_mode:
        typer.echo(json.dumps(data, indent=2, default=str))
    else:
        typer.echo(text)


def _get_field_array(
    sim: "Simulation", step: int, field: str, units: str | None
) -> object:
    """Read a field (or compute if derived), convert units."""
    import numpy as np

    ds = sim.read(step)
    arr = np.asarray(ds[field]) if ds.has_field(field) else ds.compute(field)
    if units is not None:
        return ds.in_units(field, units)
    return arr


# -- info --------------------------------------------------------------------


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

    lines = [
        f"Simulation: {sim.model_name} ({sim.model_type})",
        f"  Path:      {path}",
        f"  Grid:      {dims_str} ({geom})",
        f"  Spacing:   {spacing_str}",
        f"  Origin:    {origin_str}",
    ]
    if grid.dt is not None:
        lines.append(f"  dt:        {grid.dt:.4g}")
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


# -- fields ------------------------------------------------------------------


@app.command()
def fields(
    path: Annotated[Path, typer.Argument(help="Simulation directory.")],
    step: Annotated[
        str, typer.Option("--step", help="Timestep (default: last).")
    ] = "last",
    native: Annotated[
        bool,
        typer.Option("--native", help="Show native-to-canonical mapping."),
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
        typer.Option("--all", help="Show native, derived, and auxiliary."),
    ] = False,
    json_output: Annotated[
        bool, typer.Option("--json", help="Output as JSON.")
    ] = False,
) -> None:
    """List available fields at a timestep."""
    sim = _open(path)
    step_list = parse_steps(step, sim)
    step_val = step_list[0]

    show_native = native or all_sections
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

    if show_native:
        mapping = sim.available_fields_mapping(step_val)
        lines.append("Native \u2192 Canonical mapping:")
        max_canon = max((len(k) for k in mapping), default=0)
        for canon in sorted(mapping):
            native_name = mapping[canon]
            label = native_name if native_name is not None else "(computed)"
            lines.append(f"  {canon:<{max_canon}}  \u2190  {label}")
        data["native_mapping"] = {
            k: v if v is not None else "(computed)" for k, v in sorted(mapping.items())
        }
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


# -- stats -------------------------------------------------------------------


@app.command()
def stats(
    path: Annotated[Path, typer.Argument(help="Simulation directory.")],
    field: Annotated[str, typer.Option("--field", help="Field name.")],
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
    step_list = parse_steps(step, sim)

    def _compute_stats(step_val: int) -> dict[str, object]:
        arr = _get_field_array(sim, step_val, field, units)
        arr_np = np.asarray(arr)
        fmin, fmax = field_extrema(arr_np)
        return {
            "step": step_val,
            "min": float(fmin),
            "max": float(fmax),
            "mean": float(spatial_mean(arr_np)),
            "rms": float(spatial_rms(arr_np)),
            "nan_count": int(np.isnan(arr_np).sum()),
        }

    unit_label = units if units is not None else "code"
    all_results = [_compute_stats(s) for s in step_list]

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
        data: dict[str, object] = {
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


# -- compare -----------------------------------------------------------------


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
    from pypic.comparison import compare_fields as cmp_fields
    from pypic.comparison import field_comparison_report

    sim_a = _open(path_a)
    sim_b = _open(path_b)

    step_list = parse_steps(step, sim_a)
    step_val = step_list[0]

    ds_a = sim_a.read(step_val)
    ds_b = sim_b.read(step_val)

    cmp_kwargs: dict[str, object] = {
        "units": comp_units,
        "method": method,
        "nan_policy": nan_policy,
    }
    if frame is not None:
        cmp_kwargs["frame"] = frame

    if field is not None:
        results: dict[str, float] = {}
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

        text_lines = [
            f"Comparing: {path_a} vs {path_b}",
            f"Field: {field}  Step: {step_val}  Units: {comp_units}",
        ]
        if "l2" in results:
            text_lines.append(f"  L2 relative error: {results['l2']:.6g}")
        if "linf" in results:
            text_lines.append(f"  L-inf error:       {results['linf']:.6g}")

        data: dict[str, object] = {
            "path_a": str(path_a),
            "path_b": str(path_b),
            "step": step_val,
            "field": field,
            "units": comp_units,
            "metric": metric,
            **results,
        }
    else:
        report = field_comparison_report(
            ds_a,
            ds_b,
            **cmp_kwargs,  # type: ignore[arg-type]
        )

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


if __name__ == "__main__":
    app()
