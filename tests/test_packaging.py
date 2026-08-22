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
import tomllib
from pathlib import Path

import pypic

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
