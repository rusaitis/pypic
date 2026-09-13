"""Packaging invariants tying the source tree to ``pyproject.toml``.

The distribution is named ``pypic-plasma`` while the import package is
``pypic`` — the bare name belongs to an unrelated project on PyPI. That
mismatch makes two failure modes possible, both of which pass in a dev
checkout and only surface once the package is installed cleanly:

* a distribution-metadata lookup keyed on the import name raises
  ``PackageNotFoundError``, because a stale ``pypic`` ``.dist-info``
  left over from before the rename keeps resolving locally;
* ``pypic.__version__`` silently drifts from the version that is
  actually published.
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

import pytest

import pypic
from pypic import _cli_entry

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _REPO_ROOT / "src" / "pypic"


def _pyproject() -> dict:
    with (_REPO_ROOT / "pyproject.toml").open("rb") as fh:
        return tomllib.load(fh)


def test_package_version_matches_pyproject() -> None:
    """``pypic.__version__`` is the version that gets published."""
    declared = _pyproject()["project"]["version"]
    assert pypic.__version__ == declared, (
        f"pypic.__version__ is {pypic.__version__!r} but pyproject.toml "
        f"declares version = {declared!r}. Both are hand-written; update "
        f"them together."
    )


def test_no_metadata_lookup_uses_a_non_distribution_name() -> None:
    """Any ``importlib.metadata`` lookup names the real distribution.

    ``importlib.metadata.version("pypic")`` resolves in a dev checkout
    only because of a leftover ``.dist-info``; on a clean install it
    raises and takes ``pypic --version`` down with it. Prefer
    ``pypic.__version__``, which cannot depend on the distribution name
    at all.
    """
    dist_name = _pyproject()["project"]["name"]
    lookup = re.compile(
        r"""(?:metadata\.)?(?:version|distribution)\(\s*["']([^"']+)["']"""
    )

    offenders: list[str] = []
    for py in sorted(_SRC_DIR.rglob("*.py")):
        for lineno, line in enumerate(py.read_text().splitlines(), start=1):
            if "importlib" not in line and "metadata." not in line:
                continue
            for name in lookup.findall(line):
                if name != dist_name:
                    rel = py.relative_to(_REPO_ROOT)
                    offenders.append(f"  {rel}:{lineno} looks up {name!r}")

    assert not offenders, (
        f"distribution metadata must be looked up as {dist_name!r} "
        f"(or better, read pypic.__version__):\n" + "\n".join(offenders)
    )


def test_console_script_targets_the_guarded_entry_point() -> None:
    """``[project.scripts]`` must not point straight at ``pypic.cli``.

    Typer ships in the ``cli`` extra but the console script is always
    installed, so a bare install would otherwise put a ``pypic`` command
    on ``PATH`` that fails with a raw ``ModuleNotFoundError``.
    """
    script = _pyproject()["project"]["scripts"]["pypic"]
    assert script == "pypic._cli_entry:main", (
        f"console script points at {script!r}; it must go through "
        f"pypic._cli_entry:main so a missing cli extra produces an "
        f"install hint instead of a traceback."
    )


class _BlockTyper:
    """Meta-path finder that makes ``import typer`` fail."""

    def find_spec(self, name: str, path: object = None, target: object = None) -> None:
        if name == "typer" or name.startswith("typer."):
            raise ModuleNotFoundError(f"No module named {name!r}", name="typer")
        return None


def _block_typer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the next ``import typer`` raise, restoring state afterwards."""
    for cached in [m for m in sys.modules if m == "typer" or m.startswith("typer.")]:
        monkeypatch.delitem(sys.modules, cached)
    monkeypatch.delitem(sys.modules, "pypic.cli", raising=False)
    monkeypatch.setattr(sys, "meta_path", [_BlockTyper(), *sys.meta_path])


def test_missing_cli_extra_reports_install_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing Typer yields an install hint, not a traceback."""
    _block_typer(monkeypatch)

    with pytest.raises(SystemExit) as excinfo:
        _cli_entry.main()

    assert "pypic-plasma[cli]" in str(excinfo.value), (
        f"expected an install hint naming the cli extra, got: {excinfo.value}"
    )


def test_unrelated_import_error_is_not_swallowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only cli-extra modules get the hint; real breakage propagates.

    Emptying the allow-list makes the same missing Typer look like an
    unrelated failure, which must surface as-is rather than being
    reported to the user as a missing extra.
    """
    _block_typer(monkeypatch)
    monkeypatch.setattr(_cli_entry, "_CLI_EXTRA_MODULES", frozenset())

    with pytest.raises(ModuleNotFoundError, match="typer"):
        _cli_entry.main()


def test_citation_version_matches_pyproject() -> None:
    """``CITATION.cff`` carries a third hand-written copy of the version.

    ``pypic.__version__`` is already checked against ``pyproject.toml``,
    and CI guards the bundled JSON Schema, but nothing tied the citation
    metadata to either — so a release could ship a DOI record pointing at
    the previous version.
    """
    declared = _pyproject()["project"]["version"]
    citation = (_REPO_ROOT / "CITATION.cff").read_text()
    match = re.search(r"^version:\s*(\S+)\s*$", citation, re.MULTILINE)
    assert match is not None, "CITATION.cff has no version field"
    assert match.group(1) == declared, (
        f"CITATION.cff declares version {match.group(1)!r} but "
        f"pyproject.toml declares {declared!r}. Both are hand-written; "
        f"update them together at release time."
    )


def _documented_extras(path: Path) -> set[str]:
    """Extras named in the first Markdown table whose header is ``| Extra |``."""
    rows = re.findall(r"^\|\s*`([a-z0-9]+)`\s*\|", path.read_text(), re.MULTILINE)
    return set(rows)


@pytest.mark.parametrize("doc", ["README.md", "docs/getting-started.md"])
def test_documented_extras_match_pyproject(doc: str) -> None:
    """Every optional-dependency extra is documented, and vice versa.

    The extras table is duplicated between the README (which is also the
    PyPI long description) and the getting-started page. Adding an extra
    to ``pyproject.toml`` without touching both leaves users unable to
    discover it; removing one leaves an install line that errors.
    """
    declared = set(_pyproject()["project"]["optional-dependencies"])
    documented = _documented_extras(_REPO_ROOT / doc)
    assert documented == declared, (
        f"{doc} documents extras {sorted(documented)} but pyproject.toml "
        f"declares {sorted(declared)}."
    )


def _cli_command_names() -> set[str]:
    """Top-level ``pypic`` command and group names, from the Typer app."""
    pytest.importorskip("typer")
    from pypic.cli import app

    names = {
        c.name or c.callback.__name__.replace("_", "-") for c in app.registered_commands
    }
    for group in app.registered_groups:
        sub = group.typer_instance
        assert sub is not None
        name = group.name or sub.info.name
        assert name is not None
        names.add(name)
    return names


def test_cli_reference_documents_only_real_commands() -> None:
    """Every `pypic <cmd>` invocation in the CLI reference is registered.

    The reference is hand-maintained tables of invocations; a renamed
    command otherwise leaves a documented line that exits with "No such
    command".
    """
    text = (_REPO_ROOT / "docs" / "api" / "cli.md").read_text()
    promised = {m.group(1) for m in re.finditer(r"`pypic ([a-z][a-z-]*)", text)}
    assert len(promised) > 5, "regex stopped matching the reference's invocations"
    unknown = promised - _cli_command_names()
    assert not unknown, (
        f"docs/api/cli.md documents `pypic {sorted(unknown)}`, which the "
        f"Typer app does not register."
    )


def test_readme_command_bullet_lists_every_top_level_command() -> None:
    """The README's feature bullet is the full top-level command set.

    Readers treat that bullet as the inventory, so a command missing from
    it is a command nobody finds — and one listed but unregistered is a
    promise the CLI breaks.
    """
    readme = (_REPO_ROOT / "README.md").read_text()
    match = re.search(r"- \*\*Command line\*\* — (.+?) — ", readme, re.DOTALL)
    assert match is not None, "README has no '**Command line** — ...' bullet"
    bullet = re.sub(r"\([^)]*\)", "", match.group(1))  # drop the schema sub-list
    listed = {m.group(1) for m in re.finditer(r"`(?:pypic )?([a-z][a-z-]*)`", bullet)}
    assert listed == _cli_command_names(), (
        f"README lists {sorted(listed)} but the app registers "
        f"{sorted(_cli_command_names())}."
    )


def test_bundled_data_files_resolve_from_the_package() -> None:
    """Non-Python payload ships in the wheel and is reachable at runtime.

    Two mechanisms are in play — an explicit ``force-include`` for the
    JSON Schema and hatchling's implicit inclusion for the themes — and
    neither was covered. A packaging change that drops either one fails
    only at import time on a clean install.
    """
    from importlib.resources import files

    schema = files("pypic.schema") / "simulation.schema.v2.0.json"
    themes = files("pypic.plotting") / "themes"
    missing = [str(p) for p in (schema, themes) if not p.is_file() and not p.is_dir()]
    assert not missing, f"bundled data missing from the package: {missing}"
    assert any(t.name.endswith(".toml") for t in themes.iterdir()), (
        "pypic/plotting/themes ships no .toml theme files"
    )
