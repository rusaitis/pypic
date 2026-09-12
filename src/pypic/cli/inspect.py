"""``pypic info``, ``fields``, ``stats`` and ``validate``."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

import typer

from pypic.cli._options import JsonOption, SimulationPath, TimestepOption, UnitsOption
from pypic.cli._shared import _open, _output, _require_single_step, parse_steps

if TYPE_CHECKING:
    from pypic.readers._registry import Simulation
    from pypic.types import FloatArray


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


def info(
    path: SimulationPath,
    json_output: JsonOption = False,
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


def fields(
    path: SimulationPath,
    step: TimestepOption = "last",
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
    json_output: JsonOption = False,
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


def stats(
    path: SimulationPath,
    field: Annotated[
        str, typer.Option("--field", help="Field name ('all' for every field).")
    ],
    step: TimestepOption = "last",
    units: UnitsOption = None,
    json_output: JsonOption = False,
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


def validate(
    path: SimulationPath,
    step: TimestepOption = "last",
    json_output: JsonOption = False,
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

    # div B needs all three axes: the operators differentiate along
    # axis 0/1/2 explicitly, so a 2D dataset has no third axis to take
    # the derivative over.
    is_3d = len(ds.grid.spacing) == 3

    if has_b:
        b1, b2, b3 = ds["B_1"], ds["B_2"], ds["B_3"]
        b_mag = np.sqrt(b1**2 + b2**2 + b3**2)
        if is_3d:
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
        tab = sim.auxiliary("conserved_quantities")
        if "total_energy" in tab and len(tab) >= 2:
            e_arr = tab["total_energy"]
            e0 = float(e_arr[0])
            if e0 != 0:
                abs_e0 = abs(e0)
                energy_drift_total = float((e_arr[-1] - e0) / abs_e0 * 100)
                energy_drift_last = float((e_arr[-1] - e_arr[-2]) / abs_e0 * 100)

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
    elif has_b:
        lines.append("  max |div B|: skipped (needs a 3D grid)")
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
