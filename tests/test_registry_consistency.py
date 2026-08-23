"""Mechanical consistency between schema.md, fields.py, and compute.py.

These tests treat the canonical name list from ``schema.md`` as the
source of truth and assert the code registries can satisfy every name.
They also verify that the registries themselves are internally
consistent — every metadata entry has a working SI conversion, and
every recipe references known dependencies.

Each test is a single structural invariant ("every entry in X satisfies
Y"). Failures are collected and reported in one assertion message rather
than fanned out across hundreds of parametrized cases.

When schema.md grows a new canonical name, add it to ``CANONICAL_NAMES``
below. When a test fails, the fix is usually one of: register it in
``_FIELD_INFO``, add a recipe to ``compute._REGISTRY``, add an alias,
or extend a species pattern.
"""

from __future__ import annotations

import inspect

from pypic._aliases import COMPUTE_ALIASES, _get_field_alias_fallback
from pypic.compute import (
    _REGISTRY,
    SPECIES_TEMPLATES,
    Recipe,
    SpeciesArgs,
    SpeciesTemplate,
    _try_species_recipe,
)
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
        # Per-species electron velocity (Tier-3 form; e/i magnitude
        # shortcuts ``|Ve|`` / ``|V_e|`` resolve to ``|V_s0|``).
        "V_s0_1",
        "V_s0_2",
        "V_s0_3",
        "|V_s0|",
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
        # Per-species pressure projections (Tier-3 form; e/i shortcuts
        # ``P_par_e``/``P_par_i`` resolve to these via ``COMPUTE_ALIASES``).
        "P_s0_par",
        "P_s1_par",
        "P_s0_perp",
        "P_s1_perp",
        "agyrotropy_s0",
        "agyrotropy_s1",
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
# the dynamic `SPECIES_TEMPLATES` synthesis (s5).
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
    canonical = COMPUTE_ALIASES.get(name)
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
        "COMPUTE_ALIASES, or species patterns — register them or remove "
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
    ``SPECIES_TEMPLATES``. Both must succeed for every documented prefix.
    """
    failures = sorted(
        _per_species_form(prefix, idx)
        for prefix in PER_SPECIES_PREFIXES
        for idx in (0, 5)
        if not _is_reachable(_per_species_form(prefix, idx))
    )
    assert not failures, _format_failures(
        "Per-species names from schema.md not reachable — check "
        "_SPECIES_INFO_PATTERNS in fields.py and SPECIES_TEMPLATES in "
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


def _template_canonical_form(prefix: str, idx: int) -> str:
    r"""Construct the Tier-3 canonical name a species template synthesizes.

    Most templates use the trailing ``_s<N>`` form (``omega_p_s0``,
    ``rho_m_s0``).  Generic-operator templates (``P_par``, ``P_perp``)
    put the species qualifier *between* field and operator
    (``P_s0_par``).  Pipe-wrapped magnitude templates (``|V|``) put
    the qualifier *inside* the bars (``|V_s0|``).  Vector / tensor
    templates with a digit-suffix prefix (``V_1``, ``P_11``) put the
    qualifier between field and component (``V_s0_1``).
    """
    if prefix.startswith("|") and prefix.endswith("|"):
        return f"|{prefix[1:-1]}_s{idx}|"
    base, sep, comp = prefix.partition("_")
    if sep and (comp[:1].isdigit() or comp in _GENERIC_OPERATOR_SUFFIXES):
        return f"{base}_s{idx}_{comp}"
    return f"{prefix}_s{idx}"


def test_all_species_template_dependencies_resolve() -> None:
    """Substituting `_s0` into each template must yield reachable deps."""
    failures: list[str] = []
    for prefix in sorted(SPECIES_TEMPLATES):
        name = _template_canonical_form(prefix, 0)
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
    """Every entry in COMPUTE_ALIASES must point at a name that resolves."""
    failures = sorted(
        f"{alias!r} → {target!r}"
        for alias, target in COMPUTE_ALIASES.items()
        if not _is_reachable(target)
    )
    assert not failures, _format_failures(
        "COMPUTE_ALIASES entries pointing at unknown targets — update "
        "or remove. (Vector-group shorthand like 'EFe' → 'EF_s0' belongs "
        "in GROUP_ALIASES, not COMPUTE_ALIASES)",
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


# ---------------------------------------------------------------------------
# Test 8 — every recipe's func signature matches its declared call shape
# ---------------------------------------------------------------------------
# The compute dispatcher (``compute._execute_recipe``) builds each call's
# positional list and kwargs dynamically from recipe metadata: ``fields``
# arrays first, then optional species charge/mass scalars, gamma, c, and
# grid spacing (dx, dy, dz). Kwargs ``c=`` / ``geometry=`` are injected
# when ``supports_relativistic`` / (``passes_geometry`` + ``needs_grid``)
# are set. A mismatch between this construction and the function's actual
# signature is currently a runtime ``TypeError`` at compute time. This
# test catches every such mismatch at CI time.

_SPECIES_ARGS_COUNT: dict[SpeciesArgs, int] = {
    SpeciesArgs.CHARGE_MASS: 2,
    SpeciesArgs.MASS_ONLY: 1,
    SpeciesArgs.CHARGE_ONLY: 1,
    SpeciesArgs.NONE: 0,
}


def _expected_positional_count(entry: Recipe | SpeciesTemplate) -> int:
    """Positional args ``_execute_recipe`` passes to ``entry.func``.

    Source of truth: ``compute._execute_recipe``. Keep aligned if that
    function grows new auto-injection branches.
    """
    fields = entry.fields if isinstance(entry, Recipe) else entry.field_pattern
    n = len(fields)
    species_args = getattr(entry, "species_args", None)
    needs_species = (
        isinstance(entry, SpeciesTemplate)
        or getattr(entry, "species_index", None) is not None
    )
    if needs_species and species_args is not None:
        n += _SPECIES_ARGS_COUNT[species_args]
    if entry.needs_gamma:
        n += 1
    if entry.needs_c:
        n += 1
    if isinstance(entry, Recipe) and entry.needs_grid:
        n += 3
    return n


def _expected_kwargs(entry: Recipe | SpeciesTemplate) -> set[str]:
    """Kwargs ``_execute_recipe`` passes to ``entry.func``."""
    kw: set[str] = set()
    if getattr(entry, "supports_relativistic", False):
        kw.add("c")
    if isinstance(entry, Recipe) and entry.passes_geometry and entry.needs_grid:
        kw.add("geometry")
    return kw


def test_registry_func_signatures_match_recipe_metadata() -> None:
    """Every recipe's func signature accepts the call shape its metadata declares.

    Catches at CI time the failure class that today surfaces only at
    ``compute(name)`` time on a real dataset: wrong ``fields=`` tuple
    length, missing ``species_args=``, forgotten ``needs_grid=True``,
    ``supports_relativistic=True`` on a function without a ``c=`` kwarg.
    """
    entries: list[tuple[str, str, Recipe | SpeciesTemplate]] = [
        *(("registry", name, recipe) for name, recipe in _REGISTRY.items()),
        *(("template", prefix, tmpl) for prefix, tmpl in SPECIES_TEMPLATES.items()),
    ]

    failures: list[str] = []
    for kind, ident, entry in entries:
        sig = inspect.signature(entry.func)
        params = list(sig.parameters.values())

        positional = [
            p
            for p in params
            if p.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]
        required_positional = sum(
            1 for p in positional if p.default is inspect.Parameter.empty
        )
        has_var_positional = any(
            p.kind == inspect.Parameter.VAR_POSITIONAL for p in params
        )
        has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params)

        expected_pos = _expected_positional_count(entry)
        expected_kw = _expected_kwargs(entry)
        fields_desc = entry.fields if isinstance(entry, Recipe) else entry.field_pattern

        if required_positional > expected_pos:
            failures.append(
                f"{kind} {ident!r}: func {entry.func.__qualname__} "
                f"requires {required_positional} positional args but "
                f"recipe metadata provides only {expected_pos} "
                f"(fields={fields_desc!r}, check species_args / "
                f"needs_grid / needs_gamma / needs_c)"
            )
        if expected_pos > len(positional) and not has_var_positional:
            failures.append(
                f"{kind} {ident!r}: recipe metadata provides {expected_pos} "
                f"positional args but func {entry.func.__qualname__} "
                f"accepts at most {len(positional)}"
            )

        if not has_var_keyword:
            accepted_kw = {
                p.name
                for p in params
                if p.kind
                in (
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.KEYWORD_ONLY,
                )
            }
            for kw in expected_kw:
                if kw not in accepted_kw:
                    failures.append(
                        f"{kind} {ident!r}: dispatcher injects {kw}= but "
                        f"func {entry.func.__qualname__} does not accept it"
                    )

    assert not failures, _format_failures(
        "Recipe/func signature mismatches",
        failures,
    )
