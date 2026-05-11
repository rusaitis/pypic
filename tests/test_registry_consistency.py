"""Mechanical consistency between schema.md, fields.py, and compute.py.

These tests treat the canonical name list from ``schema.md`` as the
source of truth and assert the code registries can satisfy every name.
They also verify that the registries themselves are internally
consistent — every metadata entry has a working SI conversion, and
every recipe references known dependencies.

Each test is a single structural invariant ("every entry in X satisfies
Y"). Failures are collected and reported in one assertion message rather
than fanned out across hundreds of parametrized cases — see the
"Testing" section of ``CLAUDE.md``.

Unit 11 of the cleanup sweep — see ``TASKS-cleanup.md``.

When schema.md grows a new canonical name, add it to ``CANONICAL_NAMES``
below. When a test fails, the fix is usually one of: register it in
``_FIELD_INFO``, add a recipe to ``compute._REGISTRY``, add an alias,
or extend a species pattern.
"""

from __future__ import annotations

from pypic._aliases import _COMPUTE_ALIASES, _get_field_alias_fallback
from pypic.compute import _REGISTRY, _SPECIES_TEMPLATES, _try_species_recipe
from pypic.fields import _FIELD_INFO, _SPECIES_INFO_PATTERNS, field_info
from pypic.units import Normalization

# ---------------------------------------------------------------------------
# Canonical name list — hand-curated from schema.md § 3
# ---------------------------------------------------------------------------
# The keys here are the *canonical* (numbered, geometry-agnostic) names
# documented in schema.md as belonging to pypic's vocabulary. Letter
# aliases (Bx, By, Bz, n_e, n_i, …) are intentionally excluded —
# they're tested implicitly because the alias maps point at these names.

CANONICAL_NAMES: frozenset[str] = frozenset(
    {
        # ── Electromagnetic ──────────────────────────────────────────────
        "B_1",
        "B_2",
        "B_3",
        "B0_1",
        "B0_2",
        "B0_3",
        "E_1",
        "E_2",
        "E_3",
        "|B|",
        "|E|",
        # ── Currents & velocities ────────────────────────────────────────
        "J_1",
        "J_2",
        "J_3",
        "|J|",
        "V_1",
        "V_2",
        "V_3",
        "|V|",
        "Ve1",
        "Ve2",
        "Ve3",
        "|Ve|",
        # Four-velocity (relativistic PIC)
        "u_1",
        "u_2",
        "u_3",
        "gamma_L",
        # ── Densities & moments ──────────────────────────────────────────
        "n_s0",
        "n_s1",
        "rho_c",
        "rho_m",
        # ── Pressure ─────────────────────────────────────────────────────
        "P",
        "Pe",
        "Pi",
        "P_par",
        "P_perp",
        "P_11",
        "P_12",
        "P_13",
        "P_22",
        "P_23",
        "P_33",
        "agyrotropy",
        "P_par_e",
        "P_par_i",
        "P_perp_e",
        "P_perp_i",
        "agyrotropy_e",
        "agyrotropy_i",
        # ── Temperature ──────────────────────────────────────────────────
        "Te",
        "Ti",
        # ── Thermodynamic ────────────────────────────────────────────────
        "h",
        "h_rel",
        "e_int",
        "s",
        "s_e",
        "s_i",
        "s_gyro_e",
        "s_gyro_i",
        "gamma_eos",
        # ── Energy / flux ────────────────────────────────────────────────
        "S_1",
        "S_2",
        "S_3",
        "EF_1",
        "EF_2",
        "EF_3",
        "EHF_1",
        "EHF_2",
        "EHF_3",
        "e_B",
        "e_E",
        "e_k",
        "e_th",
        "e_th_trace",
        # ── Characteristic scales ────────────────────────────────────────
        "d_e",
        "d_i",
        "r_e",
        "r_i",
        "omega_pe",
        "omega_pi",
        "omega_ce",
        "omega_ci",
        "lambda_D",
        "v_A",
        "v_th_e",
        "v_th_i",
        "c_s",
        "c_ia",
        "v_ms",
        "M_A",
        "M_ms",
        "beta",
        "beta_e",
        "beta_i",
        "sigma",
        # ── Differential operators ───────────────────────────────────────
        "div_B",
        "div_E",
        "curl_B_1",
        "curl_B_2",
        "curl_B_3",
        "vort_1",
        "vort_2",
        "vort_3",
        "|vort|",
        # ── Reconnection diagnostics ─────────────────────────────────────
        "J_dot_E",
        "E_prime_1",
        "E_prime_2",
        "E_prime_3",
        "E_ideal_1",
        "E_ideal_2",
        "E_ideal_3",
        "E_Hall_1",
        "E_Hall_2",
        "E_Hall_3",
        "psi",
        "firehose",
        "mirror",
    }
)


# Per-species name *prefixes* listed in schema.md (suffixed with _sN at
# runtime). Each prefix is checked against synthetic species indices 0
# and 5 so we exercise both the static `_REGISTRY` entries (s0/s1) and
# the dynamic `_SPECIES_TEMPLATES` synthesis (s5).
PER_SPECIES_PREFIXES: frozenset[str] = frozenset(
    {
        "n",  # n_s0, n_s5
        "rho_c",  # rho_c_s0
        "rho_m",  # rho_m_s0
        "J_1",
        "J_2",
        "J_3",
        "V_1",
        "V_2",
        "V_3",
        "|V|",
        "P",
        "P_11",
        "P_22",
        "P_33",
        "P_12",
        "P_13",
        "P_23",
        "T",
        "EF_1",
        "EF_2",
        "EF_3",
        "KEF_1",
        "KEF_2",
        "KEF_3",
        "HF_1",
        "HF_2",
        "HF_3",
        "EHF_1",
        "EHF_2",
        "EHF_3",
        "q_1",
        "q_2",
        "q_3",
        "e_k",
        "e_th",
        "e_th_trace",
        "e_int",
        "h",
        "omega_p",
        "omega_c",
        "d",
        "r",
        "v_th",
        "lambda_D",
        "beta",
        "s",
        "s_gyro",
        "P_par",
        "P_perp",
        "agyrotropy",
    }
)


def _is_reachable(name: str) -> bool:
    """A name is reachable if it has metadata or a recipe (or both)."""
    if name in _FIELD_INFO:
        return True
    if name in _REGISTRY:
        return True
    canonical = _COMPUTE_ALIASES.get(name)
    if canonical is not None and (canonical in _FIELD_INFO or canonical in _REGISTRY):
        return True
    fallback = _get_field_alias_fallback()
    target = fallback.get(name)
    if target is not None and (target in _FIELD_INFO or target in _REGISTRY):
        return True
    # field_info() does the full resolution chain including species patterns
    try:
        field_info(name)
    except KeyError:
        return False
    return True


def _format_failures(label: str, failures: list[str]) -> str:
    """Format a list of failures as a multi-line assertion message."""
    bullets = "\n  - ".join(failures)
    return f"{label} ({len(failures)} item(s)):\n  - {bullets}"


# ---------------------------------------------------------------------------
# Test 1 — every SCHEMA canonical name is reachable
# ---------------------------------------------------------------------------


def test_all_schema_fields_are_reachable() -> None:
    """Every name in schema.md must resolve via fields/compute/aliases."""
    unreachable = sorted(n for n in CANONICAL_NAMES if not _is_reachable(n))
    assert not unreachable, _format_failures(
        "SCHEMA names not reachable via _FIELD_INFO, compute._REGISTRY, "
        "_COMPUTE_ALIASES, or species patterns — register them or remove "
        "from CANONICAL_NAMES",
        unreachable,
    )


# ---------------------------------------------------------------------------
# Test 2 — per-species prefixes resolve for both static (s0/s1) and dynamic (s5+)
# ---------------------------------------------------------------------------


_GENERIC_OPERATOR_SUFFIXES: frozenset[str] = frozenset({"par", "perp"})


def _per_species_form(prefix: str, idx: int) -> str:
    """Construct a Tier-3 per-species name from a prefix.

    Four shapes (species always sits between field root and any
    component / generic operator):
      - Pipe-wrapped magnitude (``|V|``) → ``|V_s{idx}|`` (species
        inside the bars).
      - Vector / tensor prefixes (``J_1``, ``P_11``, ``q_2``) →
        ``J_s{idx}_1``.
      - Generic-operator prefixes (``P_par``, ``P_perp``) →
        ``P_s{idx}_par``.
      - Plain scalars and compound-name descriptors (``n``, ``T``,
        ``rho_m``, ``s_gyro``, ``omega_p``) → ``n_s{idx}``.
    """
    if prefix.startswith("|") and prefix.endswith("|"):
        inner = prefix[1:-1]
        return f"|{inner}_s{idx}|"
    base, sep, comp = prefix.partition("_")
    if sep and comp[:1].isdigit():
        return f"{base}_s{idx}_{comp}"
    if sep and comp in _GENERIC_OPERATOR_SUFFIXES:
        return f"{base}_s{idx}_{comp}"
    return f"{prefix}_s{idx}"


def test_all_per_species_prefixes_resolve() -> None:
    """Per-species names work for static (s0/s1) and dynamic (s5+) indices.

    The static path goes through ``_FIELD_INFO`` / ``_REGISTRY``;
    the dynamic path goes through ``_SPECIES_INFO_PATTERNS`` /
    ``_SPECIES_TEMPLATES``. Both must succeed for every documented prefix.
    """
    failures = sorted(
        _per_species_form(prefix, idx)
        for prefix in PER_SPECIES_PREFIXES
        for idx in (0, 5)
        if not _is_reachable(_per_species_form(prefix, idx))
    )
    assert not failures, _format_failures(
        "Per-species names from schema.md not reachable — check "
        "_SPECIES_INFO_PATTERNS in fields.py and _SPECIES_TEMPLATES in "
        "compute.py",
        failures,
    )


# ---------------------------------------------------------------------------
# Test 3 — every _FIELD_INFO entry has a working SI conversion
# ---------------------------------------------------------------------------


def test_all_field_info_entries_have_si_factor() -> None:
    """Every metadata entry's quantity_type must resolve via si_factor()."""
    norm = Normalization.identity()
    failures: list[str] = []
    for name, info in sorted(_FIELD_INFO.items()):
        qtype = info.quantity_type
        try:
            factor = norm.si_factor(qtype)
        except ValueError as e:
            failures.append(f"{name!r} (quantity_type={qtype!r}): {e}")
            continue
        if factor != 1.0:
            failures.append(
                f"{name!r} (quantity_type={qtype!r}): "
                f"identity().si_factor() returned {factor}, expected 1.0"
            )
    assert not failures, _format_failures(
        "_FIELD_INFO entries with broken SI conversion",
        failures,
    )


# ---------------------------------------------------------------------------
# Test 4 — every recipe's dependency names are themselves reachable
# ---------------------------------------------------------------------------


def test_all_recipe_dependencies_are_reachable() -> None:
    """Every field referenced by a compute recipe must resolve.

    Catches typos and stale references in ``compute._REGISTRY``.
    """
    failures: list[str] = []
    for name in sorted(_REGISTRY):
        recipe = _REGISTRY[name]
        unreachable = [f for f in recipe.fields if not _is_reachable(f)]
        if unreachable:
            failures.append(f"{name!r} → {unreachable}")
    assert not failures, _format_failures(
        "Recipes referencing unknown fields — typo in compute._REGISTRY "
        "or missing _FIELD_INFO entry",
        failures,
    )


# ---------------------------------------------------------------------------
# Test 5 — every species template's substituted dependencies resolve
# ---------------------------------------------------------------------------


def test_all_species_template_dependencies_resolve() -> None:
    """Substituting `_s0` into each template must yield reachable deps."""
    failures: list[str] = []
    for prefix in sorted(_SPECIES_TEMPLATES):
        name = f"{prefix}_s0"
        recipe = _try_species_recipe(name)
        if recipe is None:
            failures.append(f"{prefix!r}: _try_species_recipe returned None")
            continue
        unreachable = [f for f in recipe.fields if not _is_reachable(f)]
        if unreachable:
            failures.append(f"{prefix!r} (as {name!r}) → {unreachable}")
    assert not failures, _format_failures(
        "Species templates with broken dependencies",
        failures,
    )


# ---------------------------------------------------------------------------
# Test 6 — every compute alias points at a reachable target
# ---------------------------------------------------------------------------


def test_all_compute_aliases_resolve() -> None:
    """Every entry in _COMPUTE_ALIASES must point at a name that resolves."""
    failures = sorted(
        f"{alias!r} → {target!r}"
        for alias, target in _COMPUTE_ALIASES.items()
        if not _is_reachable(target)
    )
    assert not failures, _format_failures(
        "_COMPUTE_ALIASES entries pointing at unknown targets — update "
        "or remove. (Vector-group shorthand like 'EFe' → 'EF_s0' belongs "
        "in _GROUP_ALIASES, not _COMPUTE_ALIASES)",
        failures,
    )


# ---------------------------------------------------------------------------
# Test 7 — _SPECIES_INFO_PATTERNS quantity types are valid
# ---------------------------------------------------------------------------


def test_species_pattern_quantity_types_resolve() -> None:
    """Every quantity_type used in species patterns must have an SI factor."""
    norm = Normalization.identity()
    failures: list[str] = []
    for pattern, qtype, _, _ in _SPECIES_INFO_PATTERNS:
        try:
            norm.si_factor(qtype)
        except ValueError:
            failures.append(f"{pattern.pattern!r} (quantity_type={qtype!r})")
    assert not failures, _format_failures(
        "Species patterns with quantity_types lacking SI factors",
        failures,
    )
