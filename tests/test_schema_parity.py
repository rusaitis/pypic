"""Parity between docs/schema.md § 3 and the field registry.

Every canonical name the schema promises must resolve via the public
``field_info()`` resolver, and every alias target in
``_COMPUTE_ALIASES`` must resolve too.  This guards against three
drift classes between the cross-tool schema contract and the pypic
registry:

* a rename that landed in one place but not the other,
* a registry deletion the docs still advertise,
* an alias mapping that points at a name that has been renamed away.
"""

from __future__ import annotations

import re
from pathlib import Path

from pypic._aliases import _COMPUTE_ALIASES
from pypic.fields import field_info

_SCHEMA_MD = Path(__file__).resolve().parent.parent / "docs" / "schema.md"

# Species indices materialized when expanding ``{N}`` templates from
# the schema.  s0/s1 are documented explicitly; s2 exercises the
# regex-based per-species pattern synthesis for arbitrary indices.
_TEST_SPECIES_INDICES = (0, 1, 2)

# Names that appear in the "Canonical" column but represent
# multi-component groups expanded at ``read()`` time (vector-group
# shorthand), not directly-resolvable singletons. ``Pij_s{N}`` is the
# per-species tensor group analogous to ``Pij``.
_GROUP_IDENTIFIERS = frozenset({"Pij", "Pij_s{N}"})


def _expand_species_template(name: str) -> list[str]:
    if "{N}" not in name:
        return [name]
    return [name.replace("{N}", str(i)) for i in _TEST_SPECIES_INDICES]


def _parse_canonical_names() -> set[str]:
    """Extract first-column names from every table inside § 3.

    Stops at the per-particle subsection because that table documents
    ``ParticleData`` columns, which live in a different schema.
    """
    text = _SCHEMA_MD.read_text()
    section_start = text.index("## 3. Canonical Field Names")
    try:
        section_end = text.index("### Per-particle data columns", section_start)
    except ValueError:
        section_end = text.index("\n## 4.", section_start)
    section = text[section_start:section_end]

    canonicals: set[str] = set()
    backtick_re = re.compile(r"`([^`]+)`")
    for line in section.splitlines():
        if not line.startswith("|") or "`" not in line:
            continue
        # Headers and divider rows carry no field data.
        if "Canonical" in line or set(line.strip()) <= set("|-: "):
            continue
        cells = line.split("|")
        if len(cells) < 3:
            continue
        for token in backtick_re.findall(cells[1].strip()):
            for piece in re.split(r",\s*", token):
                piece = piece.strip().rstrip(".")
                if not piece or piece == "..." or piece in _GROUP_IDENTIFIERS:
                    continue
                # Markdown table cells escape the pipe as ``\|``.
                piece = piece.replace(r"\|", "|")
                canonicals.update(_expand_species_template(piece))
    return canonicals


_SCHEMA_CANONICALS = sorted(_parse_canonical_names())


def test_schema_canonicals_parsed() -> None:
    """The parser found a plausible number of names.

    Guards against the section heading moving and the rest of the test
    silently passing on an empty set.
    """
    assert len(_SCHEMA_CANONICALS) >= 50, (
        f"only {len(_SCHEMA_CANONICALS)} canonical names parsed from "
        f"docs/schema.md § 3; expected >= 50.  The '## 3. Canonical "
        f"Field Names' heading or the trailing '### Per-particle data "
        f"columns' subheading may have moved."
    )


def test_schema_canonicals_resolve() -> None:
    """Every canonical name in § 3 resolves via ``field_info()``.

    A failure means the docs promise a name that the registry cannot
    deliver — typo, stale entry from before a rename, or registry
    deletion the docs missed.
    """
    unresolved: list[str] = []
    for name in _SCHEMA_CANONICALS:
        try:
            field_info(name)
        except KeyError:
            unresolved.append(name)
    assert not unresolved, (
        "docs/schema.md § 3 documents these canonical names but "
        "pypic.fields.field_info() cannot resolve them:\n"
        + "\n".join(f"  - {n}" for n in unresolved)
        + "\n\nEither register the name (extend _FIELD_INFO or "
        "_SPECIES_PATTERN_SPECS) or remove it from the schema."
    )


def test_alias_targets_resolve() -> None:
    """Every ``_COMPUTE_ALIASES`` target resolves via ``field_info()``.

    Catches alias rot — a target renamed without updating the alias.
    Independent of the schema docs.
    """
    broken: list[tuple[str, str]] = []
    for alias, target in _COMPUTE_ALIASES.items():
        try:
            field_info(target)
        except KeyError:
            broken.append((alias, target))
    assert not broken, (
        "_COMPUTE_ALIASES has aliases pointing at unresolvable "
        "targets (probable rename drift):\n"
        + "\n".join(f"  - {a!r} -> {t!r}" for a, t in broken)
    )
