"""Option types shared by more than one command, so help text stays in sync."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import typer

# One type per option that more than one command declares, so a help
# string or flag spelling is written once. Defaults stay on the command:
# ``--step`` is "last" for inspection and "all" for conversion. Closed
# vocabularies are Literals, which gives Typer the choice validation and
# the ``--help`` listing for free.

# Arguments
SimulationPath = Annotated[Path, typer.Argument(help="Simulation directory.")]
InputPath = Annotated[Path, typer.Argument(help="Simulation directory or HDF5 file.")]
PathA = Annotated[Path, typer.Argument(help="First simulation directory.")]
PathB = Annotated[Path, typer.Argument(help="Second simulation directory.")]

# Selecting what to read
TimestepOption = Annotated[
    str, typer.Option("--step", help="Timestep (default: last).")
]
FieldOption = Annotated[str, typer.Option("--field", help="Field name.")]
FieldsOption = Annotated[
    str | None,
    typer.Option("--fields", help="Comma-separated field names (e.g. B,E_3,rho_c)."),
]
PlaneOption = Annotated[
    str | None,
    typer.Option("--plane", help="Slice plane (xy/xz/yz, axis pair, or normal)."),
]
PlaneIndexOption = Annotated[
    int | None,
    typer.Option("--plane-index", help="Cell index along the plane normal."),
]
PlaneCoordOption = Annotated[
    float | None,
    typer.Option("--plane-coord", help="Physical coord along the plane normal."),
]

# Units and comparison
UnitsOption = Annotated[
    str | None,
    typer.Option("--units", help="Display units (e.g. nT, km/s)."),
]
ComparisonUnitsOption = Annotated[
    Literal["si", "code"],
    typer.Option("--units", help="Compare in SI or in code units."),
]
MethodOption = Annotated[
    str,
    typer.Option("--method", help="Interpolation method for regridding."),
]
NanPolicyOption = Annotated[
    Literal["omit", "propagate", "raise"],
    typer.Option("--nan-policy", help="How NaN cells are treated."),
]

# Figures
FormatOption = Annotated[
    Literal["png", "pdf", "svg"] | None,
    typer.Option("--format", help="Image format."),
]
DpiOption = Annotated[
    int,
    typer.Option("--dpi", help="Output DPI."),
]
ThemeOption = Annotated[
    str | None,
    typer.Option("--theme", help="Plot theme name (e.g. dark, light, synthwave)."),
]

# Zarr output
ZarrOutputOption = Annotated[
    Path,
    typer.Option("--output", "-o", help="Destination Zarr store directory."),
]
BackendOption = Annotated[
    Literal["zarr", "icechunk"],
    typer.Option("--backend", help="Storage backend."),
]
DtypeOption = Annotated[
    str | None,
    typer.Option("--dtype", help="Downcast (e.g. float32)."),
]
CompressionOption = Annotated[
    str | None,
    typer.Option(
        "--compression",
        help="Codec spec: zstd[:level] or blosc[:level] (default blosc:5).",
    ),
]
MessageOption = Annotated[
    str | None,
    typer.Option("--message", help="Icechunk commit message."),
]
TagOption = Annotated[
    str | None,
    typer.Option("--tag", help="Icechunk tag name (created on success)."),
]

# Text output
JsonOption = Annotated[bool, typer.Option("--json", help="Output as JSON.")]

# Conversion and reduction
StepOption = Annotated[
    str,
    typer.Option(
        "--step",
        help="Step spec: N, first, last, all, or start:stop[:stride].",
    ),
]


DryRunOption = Annotated[
    bool, typer.Option("--dry-run", help="Print the plan without writing.")
]


ProgressOption = Annotated[
    bool,
    typer.Option(
        "--progress/--no-progress",
        help="Show a rich.progress bar for multi-step writes (auto on TTY).",
    ),
]
