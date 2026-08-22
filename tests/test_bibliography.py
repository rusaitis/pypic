"""Bibliography verification for ``docs/references.bib``.

Four invariants that ``mkdocs build --strict`` does not catch:

* every key in ``references.bib`` is cited somewhere in ``docs/**.md``
  or in ``src/**/*.py`` module/function docstrings (no orphan entries
  surviving past a rename),
* every ``[@Key]`` invocation in docs or source has a matching ``.bib``
  entry (independent of ``--strict``; works even if the build flag
  is dropped),
* every ``doi = {…}`` value parses as a bare DOI — no ``https://``
  prefix, no trailing whitespace, no obvious typo,
* no two entries share a citation key (BibTeX silently keeps one
  and drops the rest).

Aggregation pattern: each test makes a single assertion that lists
every violation at once, so a failing run names the entire repair
list in one message.
"""

from __future__ import annotations

import re
from pathlib import Path

from pybtex.database import Entry, parse_file

_REPO_ROOT = Path(__file__).resolve().parent.parent
_BIB_PATH = _REPO_ROOT / "docs" / "references.bib"
_DOCS_DIR = _REPO_ROOT / "docs"
_SRC_DIR = _REPO_ROOT / "src"

# ``[@Key]``, ``[@Key1; @Key2]``, ``[@Key1; @Key2; @Key3]``. Each
# key is preceded by ``[`` (first slot) or ``;`` (chained slots).
# Key character set follows pandoc citation syntax: letters, digits,
# and the punctuation set below.
_CITATION_RE = re.compile(r"[\[;]\s*@([A-Za-z][\w:.#$%&\-+?<>~/]*)")

# Bare-DOI shape per Crossref guidance: ``10.<registrant>/<suffix>``.
# Registrant is 4-9 digits; suffix is any printable non-whitespace.
_DOI_RE = re.compile(r"^10\.\d{4,9}/\S+$")


def _cited_keys() -> set[str]:
    """Every ``[@Key]`` mentioned in docs prose or Python docstrings."""
    keys: set[str] = set()
    for md in _DOCS_DIR.rglob("*.md"):
        keys.update(_CITATION_RE.findall(md.read_text()))
    for py in _SRC_DIR.rglob("*.py"):
        keys.update(_CITATION_RE.findall(py.read_text()))
    return keys


def _bib_entries() -> dict[str, Entry]:
    return dict(parse_file(str(_BIB_PATH)).entries)


def test_no_orphan_bib_entries() -> None:
    """Every entry in ``references.bib`` is cited at least once."""
    bib_keys = set(_bib_entries().keys())
    orphans = sorted(bib_keys - _cited_keys())
    assert not orphans, (
        "references.bib has entries that are no longer cited "
        "anywhere in docs/**.md or src/**/*.py (probable rename or "
        "section deletion left them behind):\n"
        + "\n".join(f"  - {key}" for key in orphans)
        + "\n\nEither cite them or remove the entry from "
        "docs/references.bib."
    )


def test_all_cited_keys_have_entries() -> None:
    """Every ``[@Key]`` in docs or source resolves to a ``.bib`` entry."""
    bib_keys = set(_bib_entries().keys())
    missing = sorted(_cited_keys() - bib_keys)
    assert not missing, (
        "docs/**.md or src/**/*.py cite keys that have no matching "
        "entry in docs/references.bib:\n"
        + "\n".join(f"  - {key}" for key in missing)
        + "\n\nAdd the missing @article/@book entry or fix the "
        "citation key."
    )


def test_doi_format() -> None:
    """Every ``doi`` field is a bare DOI, not a URL."""
    bad: list[tuple[str, str]] = []
    for key, entry in _bib_entries().items():
        doi = entry.fields.get("doi")
        if doi is None:
            continue
        if doi != doi.strip():
            bad.append((key, f"trailing whitespace: {doi!r}"))
            continue
        if not _DOI_RE.fullmatch(doi):
            bad.append((key, f"not a bare DOI (10.NNNN/…): {doi!r}"))
    assert not bad, (
        "references.bib has malformed DOI fields (must be the "
        "bare DOI without ``https://doi.org/`` prefix and no "
        "surrounding whitespace):\n"
        + "\n".join(f"  - {key}: {reason}" for key, reason in bad)
    )


def test_no_duplicate_keys() -> None:
    """No two entries share a citation key.

    ``pybtex`` deduplicates internally and surfaces the collision
    only via a warning, so reparse the raw file and count
    occurrences ourselves.
    """
    raw = _BIB_PATH.read_text()
    entry_re = re.compile(r"^@\w+\{\s*([^,\s]+)\s*,", re.MULTILINE)
    keys = entry_re.findall(raw)
    counts: dict[str, int] = {}
    for key in keys:
        counts[key] = counts.get(key, 0) + 1
    dupes = sorted(k for k, n in counts.items() if n > 1)
    assert not dupes, (
        "references.bib has duplicate citation keys (BibTeX "
        "silently keeps one; this is almost always a copy-paste "
        "mistake):\n" + "\n".join(f"  - {key} ({counts[key]}x)" for key in dupes)
    )
