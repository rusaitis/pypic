"""String-based dispatch for derived quantities on FieldDataset.

Maps short names (``"|B|"``, ``"beta"``, ``"v_A"``, ...) to pure functions
in ``derived.py``, ``diagnostics.py``, and ``operators.py``. The tables
live in ``pypic._recipes``; this module resolves names, executes recipes,
and owns the registration API. ``FieldDataset`` sits below it and reaches
``compute_field`` through a deferred import.
"""

from __future__ import annotations

import difflib
import threading
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, assert_never

from pypic._aliases import (
    COMPUTE_ALIASES,
    GROUP_ALIASES,
    OPERATOR_SUFFIXES,
    SPECIES_SUFFIX_RE,
    _get_field_alias_fallback,
)
from pypic._field_table import _FIELD_INFO
from pypic._recipes import (
    _REGISTRY,
    _SPECIES_TEMPLATES,
    Recipe,
    SpeciesArgs,
    SpeciesTemplate,
    _species_recipe,
)
from pypic.coordinates.geometry import GeometryType
from pypic.exceptions import GeometryUnsupportedError, UnknownFieldError
from pypic.fields import (
    _SPECIES_QUANTITY_PATTERNS,
    QuantityType,
    register_field,
    unregister_field,
)
from pypic.units import _DISPLAY_UNITS

if TYPE_CHECKING:
    from collections.abc import Callable

    from pypic.dataset import FieldDataset
    from pypic.types import FloatArray
    from pypic.units import Normalization

# Legacy shapes that put the operator before the species (``P_par_s0``,
# ``|V|_s0``) invert the Tier-3 order and must not synthesize.
_INVALID_PREFIX_ENDINGS = (*(f"_{op}" for op in OPERATOR_SUFFIXES), "|")


def _split_species_name(name: str) -> tuple[str, int, str] | None:
    """Split ``V_s2_perp_1`` into ``("V", 2, "_perp_1")``.

    The template key is ``prefix + suffix`` (``"V_perp_1"``). Returns
    ``None`` when *name* carries no species qualifier or puts a generic
    operator before it.
    """
    m = SPECIES_SUFFIX_RE.search(name)
    if m is None:
        return None
    prefix, suffix = name[: m.start()], m.group("suffix") or ""
    if not suffix and prefix.endswith(_INVALID_PREFIX_ENDINGS):
        return None
    return prefix, int(m.group(1)), suffix


def _try_species_recipe(name: str) -> Recipe | None:
    """Build a recipe from the species templates for names like ``omega_p_s2``.

    Returns ``None`` if the name doesn't match any template.
    """
    parts = _split_species_name(name)
    if parts is None:
        return None
    prefix, species_index, suffix = parts
    template = _SPECIES_TEMPLATES.get(prefix + suffix)
    if template is None:
        return None
    return _species_recipe(template, species_index)


_MAX_DEPTH = 10


def _resolve_name(name: str) -> str:
    """Resolve a compute alias to its canonical registry name."""
    return COMPUTE_ALIASES.get(name, name)


def _get_recipe(name: str) -> Recipe:
    """Look up a recipe by name, raising KeyError with suggestions on miss."""
    canonical = _resolve_name(name)
    try:
        return _REGISTRY[canonical]
    except KeyError:
        pass
    # Try dynamic species template synthesis (e.g. omega_p_s2, T_s3)
    dynamic = _try_species_recipe(canonical)
    if dynamic is not None:
        return dynamic
    # Check if this is a vector group alias (e.g. "EFe" → read-time only)
    if name in GROUP_ALIASES or canonical in GROUP_ALIASES:
        group_name = name if name in GROUP_ALIASES else canonical
        msg = (
            f"{name!r} is a vector group (expands to 3 components). "
            f"Use {group_name}1/{group_name}2/{group_name}3 in compute(), "
            f'or read(fields=["{group_name}"]) to load all three.'
        )
        raise UnknownFieldError(msg) from None
    all_names = available_quantities()
    suggestions = difflib.get_close_matches(name, all_names, n=3, cutoff=0.4)
    msg = f"Unknown derived quantity {name!r}."
    if suggestions:
        msg += f" Did you mean: {suggestions}?"
    msg += " Call available_quantities() for the full list."
    raise UnknownFieldError(msg) from None


def _get_species_args(
    dataset: FieldDataset,
    recipe: Recipe,
) -> list[float]:
    """Extract charge and mass from species info for a recipe."""
    if recipe.species_index is None:
        return []
    if recipe.species_args is SpeciesArgs.NONE:
        return []
    idx = recipe.species_index
    if not dataset.species or idx >= len(dataset.species):
        available = len(dataset.species)
        detail = (
            "no species defined"
            if not dataset.species
            else f"only {available} species available"
        )
        msg = (
            f"Cannot compute {recipe.func.__name__!r}: "
            f"requires species[{idx}] but {detail}"
        )
        raise ValueError(msg)
    sp = dataset.species[idx]
    if sp.charge is None or sp.mass is None:
        missing = [p for p in ("charge", "mass") if getattr(sp, p) is None]
        msg = (
            f"Species {sp.name!r} (index {idx}) is missing {', '.join(missing)}, "
            f"required by {recipe.func.__name__!r}"
        )
        raise ValueError(msg)
    return [sp.charge, sp.mass]


def _get_gamma(dataset: FieldDataset) -> float:
    """Get the adiabatic index from physics params."""
    return dataset.physics.gamma


def _get_c(dataset: FieldDataset) -> float:
    """Get the speed of light from physics params."""
    return dataset.physics.c


def _append_species_params(
    args: list[Any],
    species_args: list[float],
    kind: SpeciesArgs,
) -> None:
    """Append the right species parameters based on the descriptor."""
    match kind:
        case SpeciesArgs.CHARGE_MASS:
            args.extend(species_args)
        case SpeciesArgs.MASS_ONLY:
            args.append(species_args[1])
        case SpeciesArgs.CHARGE_ONLY:
            args.append(species_args[0])
        case SpeciesArgs.NONE:
            pass
        case _ as unreachable:  # pragma: no cover
            assert_never(unreachable)


def _execute_recipe(
    canonical: str,
    dataset: FieldDataset,
    _depth: int = 0,
) -> tuple[Recipe, Any]:
    """Build arguments for a recipe and invoke its pure function.

    Returns ``(recipe, full_result)``. ``full_result`` is the raw output
    of ``recipe.func``: a tuple for multi-output recipes (curl, gradient,
    vorticity) and a scalar array otherwise. Callers that want a single
    component should index into the result via ``recipe.component``.

    Shared between `compute_field` (single-component path) and
    `compute_with_siblings` (multi-component path) so the two cannot
    drift on argument construction or geometry handling.
    Dependency resolution recurses through ``compute_field`` to benefit
    from its alias handling and cycle guard.
    """
    recipe = _get_recipe(canonical)

    # Resolve field dependencies (recursive)
    args: list[Any] = [
        compute_field(field_name, dataset, _depth + 1) for field_name in recipe.fields
    ]

    # Append species charge/mass
    species_args = _get_species_args(dataset, recipe)
    if species_args:
        if recipe.species_args is None:
            msg = (
                f"Recipe for {canonical!r} has species_index={recipe.species_index} "
                f"but no species_args descriptor"
            )
            raise ValueError(msg)
        _append_species_params(args, species_args, recipe.species_args)

    # Append gamma
    if recipe.needs_gamma:
        args.append(_get_gamma(dataset))

    # Append c
    if recipe.needs_c:
        args.append(_get_c(dataset))

    # Auto-inject c for relativistic simulations: when the dataset
    # declares physics.relativistic=True, functions with a c=None kwarg
    # get the speed of light passed in, activating their relativistic branch.
    kwargs: dict[str, Any] = {}
    if recipe.supports_relativistic and dataset.physics.relativistic:
        kwargs["c"] = _get_c(dataset)

    # Append grid spacing
    if recipe.needs_grid:
        if dataset.grid.geometry.type != GeometryType.CARTESIAN:
            msg = (
                f"Derived quantity {canonical!r} requires spatial derivatives, "
                f"which are only implemented for Cartesian geometry. "
                f"Dataset has {dataset.grid.geometry.type.value} geometry."
            )
            raise GeometryUnsupportedError(msg)
        if len(dataset.grid.spacing) != 3:
            msg = (
                f"Derived quantity {canonical!r} requires spatial derivatives, "
                f"which are only implemented for 3D grids. "
                f"Dataset grid is {len(dataset.grid.spacing)}D."
            )
            raise GeometryUnsupportedError(msg)
        args.extend(dataset.grid.spacing)
        # Value-wise inert behind the Cartesian raise; keeps the
        # FieldDataset → recipe → operator path wired.
        if recipe.passes_geometry:
            kwargs["geometry"] = dataset.grid.geometry.type

    result = recipe.func(*args, **kwargs)  # Make it so.
    return recipe, result


def compute_field(name: str, dataset: FieldDataset, _depth: int = 0) -> FloatArray:
    """Compute a derived quantity by name from a FieldDataset.

    If *name* is already present in the dataset, returns it directly.
    Otherwise dispatches to the registered pure function, recursively
    resolving any intermediate dependencies.

    Parameters
    ----------
    name : str
        Field or derived quantity name (e.g. ``"|B|"``, ``"beta"``).
    dataset : FieldDataset
        Source data.

    Returns
    -------
    FloatArray
        Computed array in code units.

    Raises
    ------
    KeyError
        If *name* is unknown and not in the dataset.
    ValueError
        If required species or physics info is missing.
    GeometryUnsupportedError
        If the recipe requires spatial derivatives and the dataset
        grid is non-Cartesian or not three-dimensional.  Subclass of
        `NotImplementedError`.
    RecursionError
        If dependency chain exceeds depth limit.
    """
    if _depth > _MAX_DEPTH:
        msg = f"Dependency chain too deep (>{_MAX_DEPTH}) while computing {name!r}"
        raise RecursionError(msg)

    # Check original name first — raw fields take priority over aliases
    if dataset.has_field(name):
        return dataset[name]

    canonical = _resolve_name(name)

    # Direct field lookup after alias resolution
    if dataset.has_field(canonical):
        return dataset[canonical]

    try:
        recipe, result = _execute_recipe(canonical, dataset, _depth)
    except UnknownFieldError as exc:
        if _depth:
            raise
        msg = f"Cannot compute {name!r}: {exc.args[0]}"
        raise UnknownFieldError(msg) from exc

    if recipe.component is not None:
        return result[recipe.component]  # type: ignore[no-any-return]
    return result  # type: ignore[no-any-return]


def field_si_factor(name: str, normalization: Normalization) -> float:
    """Return the SI conversion factor for a field or derived quantity.

    Parameters
    ----------
    name : str
        Field or derived quantity name.
    normalization : Normalization
        Active normalization.

    Returns
    -------
    float
        Multiplicative factor: ``si_value = code_value * factor``.

    Raises
    ------
    ValueError
        If the quantity type for *name* is unknown.
    """
    canonical = _resolve_name(name)
    info = _FIELD_INFO.get(canonical)
    if info is None:
        # Resolve field aliases (Bx→B_1, B_x→B_1, P_e→Pe, etc.)
        fallback = _get_field_alias_fallback()
        canonical = fallback.get(canonical, canonical)
        info = _FIELD_INFO.get(canonical)
    quantity_type: str | None = info.quantity_type if info is not None else None
    if quantity_type is None:
        # Try regex patterns for per-species fields (n_s2, J_s3_1, etc.)
        for pattern, qtype in _SPECIES_QUANTITY_PATTERNS:
            if pattern.match(canonical):
                quantity_type = qtype
                break
    if quantity_type is None:
        msg = (
            f"No SI conversion known for {name!r}. Known fields: {sorted(_FIELD_INFO)}"
        )
        raise ValueError(msg)
    return normalization.si_factor(quantity_type)


def display_unit_factor(unit_str: str) -> float:
    """Return the SI value of a display unit string.

    Parameters
    ----------
    unit_str : str
        Unit string (e.g. ``"nT"``, ``"km/s"``).

    Returns
    -------
    float
        Value of one display unit in SI.

    Raises
    ------
    ValueError
        If *unit_str* is not recognized.
    """
    try:
        return _DISPLAY_UNITS[unit_str]
    except KeyError:
        valid = sorted(_DISPLAY_UNITS)
        msg = f"Unknown unit {unit_str!r}. Valid: {valid}"
        raise ValueError(msg) from None


def available_quantities() -> list[str]:
    """Return sorted list of registered quantity names and aliases.

    Does not include dynamically synthesized per-species quantities
    (e.g. ``"omega_p_s2"``, ``"T_s3"``), which are also computable
    via `compute_field`.

    Returns
    -------
    list[str]
    """
    return sorted(set(_REGISTRY) | set(COMPUTE_ALIASES))


def field_dependencies(name: str, _depth: int = 0) -> set[str]:
    """Return the raw field names needed to compute *name*.

    Recursively walks the compute recipe graph. If *name* has no recipe
    (i.e. it is a raw field), returns ``{name}``.

    Parameters
    ----------
    name : str
        Field or derived quantity name (e.g. ``"Pi"``, ``"beta"``).

    Returns
    -------
    set[str]
        Leaf field names that must be present in the dataset.
    """
    if _depth > _MAX_DEPTH:
        msg = f"Dependency chain too deep (>{_MAX_DEPTH}) while resolving {name!r}"
        raise RecursionError(msg)

    canonical = _resolve_name(name)
    try:
        recipe = _get_recipe(canonical)
    except KeyError:
        return {canonical}

    deps: set[str] = set()
    for field in recipe.fields:
        deps |= field_dependencies(field, _depth + 1)
    return deps


_recipe_lock = threading.Lock()


def _find_sibling_components(name: str) -> dict[str, int]:
    r"""Find every component recipe sharing *name*'s function and inputs.

    Returns ``{name: component_index}`` for component recipes
    (``S_1/S_2/S_3``, ``curl_B_1/2/3``, ``V_s2_perp_1/2/3``) and an empty
    dict otherwise. Registered recipes find their siblings in the
    registry; synthesized per-species ones among the templates.
    """
    canonical = _resolve_name(name)
    try:
        recipe = _get_recipe(canonical)
    except KeyError:
        return {}
    if recipe.component is None:
        return {}
    if canonical in _REGISTRY:
        return {
            reg_name: reg_recipe.component
            for reg_name, reg_recipe in _REGISTRY.items()
            if reg_recipe.func is recipe.func
            and reg_recipe.fields == recipe.fields
            and reg_recipe.component is not None
        }
    parts = _split_species_name(canonical)
    if parts is None:
        return {}
    prefix, species_index, suffix = parts
    template = _SPECIES_TEMPLATES[prefix + suffix]
    return {
        f"{prefix}_s{species_index}{key.removeprefix(prefix)}": sibling.component
        for key, sibling in _SPECIES_TEMPLATES.items()
        if key.startswith(prefix)
        and sibling.func is template.func
        and sibling.field_pattern == template.field_pattern
        and sibling.component is not None
    }


def compute_with_siblings(name: str, dataset: FieldDataset) -> dict[str, FloatArray]:
    r"""Compute *name* and, for a vector component, its siblings in one call.

    Component recipes (``S_1``, ``curl_B_2``, ``V_s2_perp_3``) share one
    tuple-returning function, so evaluating it once yields every
    component. Scalar recipes return a single entry keyed by *name*.

    Parameters
    ----------
    name : str
        Field or derived quantity name (canonical or alias).
    dataset : FieldDataset
        Source of the dependency fields.

    Returns
    -------
    dict[str, FloatArray]
        One entry for a scalar recipe; one per component, keyed by the
        registry names, for a vector recipe.
    """
    siblings = _find_sibling_components(name)
    if not siblings:
        return {name: compute_field(name, dataset)}
    _recipe, full_result = _execute_recipe(_resolve_name(name), dataset)
    return {sibling: full_result[index] for sibling, index in siblings.items()}


def register_recipe(
    name: str,
    func: Callable[..., Any],
    fields: tuple[str, ...],
    quantity_type: QuantityType | str,
    *,
    needs_grid: bool = False,
    needs_gamma: bool = False,
    needs_c: bool = False,
    long_name: str = "",
    latex: str = "",
) -> None:
    r"""Register a custom derived quantity.

    Registers both the computation recipe and the field metadata,
    so ``compute()``, ``in_si()``, ``field_info()``, and
    ``with_derived()`` all work for the custom field.

    Parameters
    ----------
    name : str
        Quantity name (e.g. ``"R_reconnection"``).
    func : Callable
        Pure function: takes arrays (one per field in *fields*),
        plus grid spacing if *needs_grid*, plus gamma if
        *needs_gamma*, plus c if *needs_c*. Returns a single array.
    fields : tuple[str, ...]
        Input field names (canonical or derived). Resolved
        recursively at compute time.
    quantity_type : QuantityType | str
        Physical quantity type for SI conversion.
    needs_grid : bool
        If ``True``, grid spacing ``(dx, dy, dz)`` is appended to args.
    needs_gamma : bool
        If ``True``, adiabatic index $\gamma$ is appended to args.
    needs_c : bool
        If ``True``, speed of light $c$ is appended to args.
    long_name : str
        Human-readable label for plot titles.
    latex : str
        LaTeX symbol for plot labels.

    Raises
    ------
    ValueError
        If *name* already exists in the recipe registry.

    Examples
    --------
    >>> import numpy as np
    >>> register_recipe(
    ...     "e_mag_ratio",
    ...     func=lambda eb, ee: eb / (eb + ee),
    ...     fields=("e_B", "e_E"),
    ...     quantity_type="dimensionless",
    ...     long_name="Magnetic-to-total EM energy ratio",
    ... )
    >>> "e_mag_ratio" in available_quantities()
    True
    >>> unregister_recipe("e_mag_ratio")
    """
    recipe = Recipe(
        func=func,
        fields=fields,
        needs_grid=needs_grid,
        needs_gamma=needs_gamma,
        needs_c=needs_c,
    )
    with _recipe_lock:
        if name in _REGISTRY:
            msg = f"Recipe {name!r} already registered"
            raise ValueError(msg)
        _REGISTRY[name] = recipe

    try:
        register_field(name, quantity_type, long_name=long_name, latex=latex)
    except Exception:
        with _recipe_lock:
            _REGISTRY.pop(name, None)
        raise


def unregister_recipe(name: str) -> None:
    r"""Remove a custom derived quantity.

    Removes both the computation recipe and the field metadata.

    Raises
    ------
    KeyError
        If *name* is not registered.
    """
    with _recipe_lock:
        try:
            recipe = _REGISTRY.pop(name)
        except KeyError:
            msg = f"No recipe registered for {name!r}"
            raise KeyError(msg) from None

    try:
        unregister_field(name)
    except Exception:
        with _recipe_lock:
            _REGISTRY[name] = recipe
        raise


# Public read-only view of the recipe registry — codegen consumers iterate
# ``RECIPES.items()``.  ``_REGISTRY`` stays mutable for `register_recipe` /
# `unregister_recipe` under ``_recipe_lock``; the proxy hides the write side.
RECIPES: MappingProxyType[str, Recipe] = MappingProxyType(_REGISTRY)
SPECIES_TEMPLATES: MappingProxyType[str, SpeciesTemplate] = MappingProxyType(
    _SPECIES_TEMPLATES
)


__all__ = [
    "RECIPES",
    "SPECIES_TEMPLATES",
    "Recipe",
    "SpeciesArgs",
    "SpeciesTemplate",
    "available_quantities",
    "compute_field",
    "compute_with_siblings",
    "display_unit_factor",
    "field_dependencies",
    "field_si_factor",
    "register_recipe",
    "unregister_recipe",
]
