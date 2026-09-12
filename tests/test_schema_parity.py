"""Parity between the authorities that describe the same contract.

Two pairs are guarded here.

``docs/schema.md`` § 3 against the field registry: every canonical name
the schema promises must resolve via the public ``field_info()``
resolver, and every alias target in ``COMPUTE_ALIASES`` must resolve
too.  This catches a rename that landed in one place but not the other,
a registry deletion the docs still advertise, and an alias pointing at a
name that has been renamed away.

The Pydantic validator against the loader: every ``simulation.toml``
that ``validate_simulation_toml`` accepts, ``load_config`` must also
accept.  The two are separate code paths over one document, so a
constraint tightened in the loader but not the model — or an optional
key the model permits and the loader cannot build — turns ``pypic schema
validate`` into a tool that green-lights files pypic then refuses to
open.
"""

from __future__ import annotations

import re
from pathlib import Path
from textwrap import dedent

from pypic._aliases import COMPUTE_ALIASES
from pypic.fields import field_info
from pypic.readers.config import load_config
from pypic.schema import validate_simulation_toml

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

# Subsections of § 3 whose tables document names produced by modules
# that have not shipped yet, so ``field_info()`` cannot resolve them.
# The schema labels these inline as planned; skipping by heading (rather
# than by a hardcoded name list) means names added to a planned table
# stay covered without touching this test. Delete an entry when its
# producer lands.
_PLANNED_SUBSECTIONS = frozenset({"### Field-line map quantities"})


def _expand_species_template(name: str) -> list[str]:
    if "{N}" not in name:
        return [name]
    return [name.replace("{N}", str(i)) for i in _TEST_SPECIES_INDICES]


def _parse_canonical_names() -> set[str]:
    """Extract first-column names from every table inside § 3.

    Stops at the per-particle subsection because that table documents
    ``ParticleData`` columns, which live in a different schema, and
    skips the subsections listed in ``_PLANNED_SUBSECTIONS``.
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
    planned = False
    for line in section.splitlines():
        if line.startswith("###"):
            planned = line.strip() in _PLANNED_SUBSECTIONS
            continue
        if planned or not line.startswith("|") or "`" not in line:
            continue
        # Headers and divider rows carry no field data.
        if "Canonical" in line or set(line.strip()) <= set("|-: "):
            continue
        cells = line.split("|")
        if len(cells) < 3:
            continue
        for token in backtick_re.findall(cells[1].strip()):
            for raw_piece in re.split(r",\s*", token):
                piece = raw_piece.strip().rstrip(".")
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
    """Every ``COMPUTE_ALIASES`` target resolves via ``field_info()``.

    Catches alias rot — a target renamed without updating the alias.
    Independent of the schema docs.
    """
    broken: list[tuple[str, str]] = []
    for alias, target in COMPUTE_ALIASES.items():
        try:
            field_info(target)
        except KeyError:
            broken.append((alias, target))
    assert not broken, (
        "COMPUTE_ALIASES has aliases pointing at unresolvable "
        "targets (probable rename drift):\n"
        + "\n".join(f"  - {a!r} -> {t!r}" for a, t in broken)
    )


# One entry per shape of the [units] discriminated union, since that is
# where the validator and the loader are most likely to disagree: the
# model marks most reference keys optional, and the loader has to turn
# whatever survives into eight positive SI references.
_UNITS_VARIANTS: tuple[tuple[str, str], ...] = (
    ("SI", 'system = "SI"'),
    ("PIC, electron-referenced", 'system = "PIC"\nreference_density = 1.0e18'),
    (
        "PIC, ion-referenced",
        'system = "PIC"\nreference_species = "ions"\nreference_density = 1.0e6',
    ),
    (
        "MHD",
        'system = "MHD"\n'
        "reference_length = 6.371e6\n"
        "reference_density = 1.67e-17\n"
        "reference_b_field = 5.0e-9",
    ),
    (
        "custom, minimal determining set",
        'system = "custom"\n'
        "[units.reference]\n"
        "length = 5.31e-3\n"
        "time = 1.77e-11\n"
        "b_field = 1.07e-3\n"
        "density = 1.0e18",
    ),
    (
        "custom, velocity and e_field instead of time and b_field",
        'system = "custom"\n'
        "[units.reference]\n"
        "length = 5.31e-3\n"
        "velocity = 2.998e8\n"
        "e_field = 3.21e5\n"
        "density = 1.0e18",
    ),
    (
        "custom, every reference given",
        'system = "custom"\n'
        "[units.reference]\n"
        "length = 5.31e-3\n"
        "time = 1.77e-11\n"
        "velocity = 2.998e8\n"
        "b_field = 1.07e-3\n"
        "e_field = 3.21e5\n"
        "density = 1.0e18\n"
        "mass = 9.109e-31\n"
        "charge = 1.602e-19",
    ),
)


def _doc(units_block: str) -> str:
    """A minimal valid v1.0 document carrying *units_block* verbatim."""
    head = dedent(
        """
        [schema]
        version = "1.0"
        [model]
        name = "demo"
        type = "PIC"
        [run]
        name = "r0"
        [time]
        dt = 0.1
        t_start = 0.0
        t_end = 1.0
        n_steps = 10
        [grid]
        dimensions = [4, 4, 4]
        spacing = [1.0, 1.0, 1.0]
        lower = [0.0, 0.0, 0.0]
        upper = [4.0, 4.0, 4.0]
        [units]
        """
    ).strip()
    tail = dedent(
        """
        [coordinates]
        geometry = "cartesian"
        frame = "sim"
        [[species]]
        name = "electrons"
        charge = -1.0
        mass = 1.0
        """
    ).strip()
    return f"{head}\n{units_block}\n{tail}\n"


def test_validated_documents_load(tmp_path: Path) -> None:
    """Whatever the validator accepts, ``load_config`` must accept.

    ``pypic schema validate`` is the tool users reach for to check a
    deck before shipping it beside their data.  If it passes a document
    that ``load_config`` then rejects, the check is worse than useless —
    it certifies files pypic cannot open.
    """
    failures: list[str] = []
    for label, units_block in _UNITS_VARIANTS:
        text = _doc(units_block)
        try:
            validate_simulation_toml(text)
        except Exception as exc:
            failures.append(f"  - {label}: fixture is not valid at all ({exc})")
            continue
        path = tmp_path / f"{label.replace(' ', '_').replace(',', '')}.toml"
        path.write_text(text)
        try:
            load_config(path)
        except Exception as exc:
            failures.append(f"  - {label}: {type(exc).__name__}: {exc}")
    assert not failures, (
        "validate_simulation_toml accepted these [units] shapes but "
        "load_config refused them:\n"
        + "\n".join(failures)
        + "\n\nEither teach readers.config to build a Normalization from "
        "what the model permits, or tighten the model so the document "
        "never validates in the first place."
    )


def test_underdetermined_custom_units_are_rejected() -> None:
    """A custom block the loader could not build must not validate.

    The other half of the parity above: the validator has to refuse
    what it cannot hand on, and say which key would have closed the
    gap rather than failing later on a reference the deck never wrote.
    """
    doc = _doc('system = "custom"\n[units.reference]\nlength = 5.31e-3')
    try:
        validate_simulation_toml(doc)
    except Exception as exc:
        message = str(exc)
    else:  # pragma: no cover - only reached when the guard regresses
        message = ""
    for expected in ("'time' or 'velocity'", "'b_field' or 'e_field'", "'density'"):
        assert expected in message, (
            f"expected the rejection to name {expected}, got:\n{message}"
        )
