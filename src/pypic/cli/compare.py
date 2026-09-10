"""``pypic compare``: field-by-field comparison of two runs."""

from __future__ import annotations

from typing import Annotated, Literal

import typer

from pypic.cli._options import (
    ComparisonUnitsOption,
    JsonOption,
    MethodOption,
    NanPolicyOption,
    PathA,
    PathB,
    TimestepOption,
)
from pypic.cli._shared import (
    _open,
    _output,
    _require_single_step,
    parse_steps,
)


def compare(
    path_a: PathA,
    path_b: PathB,
    step: TimestepOption = "last",
    field: Annotated[
        str | None,
        typer.Option("--field", help="Field to compare (omit for all common)."),
    ] = None,
    metric: Annotated[
        Literal["l2", "linf", "both"],
        typer.Option("--metric", help="Error metric; both reports l2 and linf."),
    ] = "both",
    comp_units: ComparisonUnitsOption = "si",
    method: MethodOption = "linear",
    nan_policy: NanPolicyOption = "omit",
    frame: Annotated[
        str | None,
        typer.Option(
            "--frame",
            help="Transform both to this reference frame.",
        ),
    ] = None,
    json_output: JsonOption = False,
) -> None:
    """Compare fields between two simulations."""
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
