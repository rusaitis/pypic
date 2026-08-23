"""Console-script entry point with an optional-dependency guard.

``[project.scripts]`` registers the ``pypic`` command unconditionally,
but Typer and Rich ship in the ``cli`` extra. Without this shim a bare
``pip install pypic-plasma`` puts a ``pypic`` command on ``PATH`` that
dies with a raw ``ModuleNotFoundError`` traceback — the first thing a
new user is likely to hit. ``pypic.cli`` is untouched, so importing
``app`` from it directly still works.
"""

from __future__ import annotations

# Modules the `cli` extra provides, directly or transitively. A
# ModuleNotFoundError naming one of these means the extra is missing;
# anything else is a genuine bug and must not be hidden behind an
# install hint.
_CLI_EXTRA_MODULES = frozenset({"click", "rich", "shellingham", "typer"})


def main() -> None:
    """Run the pypic CLI, or explain how to install it.

    Raises
    ------
    SystemExit
        If the ``cli`` extra is not installed. Carries the install
        command rather than a traceback, since this runs as a console
        script where a traceback is noise.
    """
    try:
        from pypic.cli import app
    except ModuleNotFoundError as exc:
        if exc.name not in _CLI_EXTRA_MODULES:
            raise
        msg = (
            f"The 'pypic' command requires {exc.name}, which ships in the "
            'cli extra.\nInstall with: pip install "pypic-plasma[cli]"'
        )
        raise SystemExit(msg) from None
    app()
