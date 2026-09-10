"""Option types shared by more than one command, so help text stays in sync."""

from __future__ import annotations

from typing import Annotated

import typer

# Shared option type aliases — used identically across `convert *`
# subcommands so help text stays in sync. `--output` differs by
# destination type (Zarr / Parquet / root) and stays bespoke.
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
