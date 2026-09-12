"""Public-surface invariants for codegen consumers.

The cross-tool toolchain (webpic, rustpic tooling) reads the symbols
checked here to generate TypeScript/Rust mirrors.  Treat any failure as
a contract break — bump the schema major before changing the behavior.
"""

from __future__ import annotations

import ast
import subprocess
import sys
import types
from pathlib import Path

import pytest

import pypic
import pypic._aliases as private_aliases
import pypic.aliases as public_aliases
import pypic.compute as compute_module


def test_public_aliases_module_re_exports_underlying_tables() -> None:
    """``pypic.aliases`` exposes the same objects as the private module."""
    assert public_aliases.COMPUTE_ALIASES is private_aliases.COMPUTE_ALIASES
    assert public_aliases.GROUP_ALIASES is private_aliases.GROUP_ALIASES
    assert public_aliases.SPECIES_SUFFIX_RE is private_aliases.SPECIES_SUFFIX_RE
    assert public_aliases.species_name_aliases is private_aliases.species_name_aliases


def test_top_level_re_exports() -> None:
    """Top-level ``pypic`` re-exports reach the same objects as ``pypic.compute``."""
    assert pypic.RECIPES is compute_module.RECIPES
    assert pypic.Recipe is compute_module.Recipe
    assert pypic.aliases is public_aliases


def test_recipes_proxy_mirrors_registry_keyset() -> None:
    """``RECIPES`` is a read-only view over the internal mutable registry."""
    assert isinstance(compute_module.RECIPES, types.MappingProxyType)
    assert set(compute_module.RECIPES) == set(compute_module._REGISTRY)


def test_species_templates_proxy_mirrors_internal_keyset() -> None:
    """``SPECIES_TEMPLATES`` is a read-only view over the internal mutable dict."""
    assert isinstance(compute_module.SPECIES_TEMPLATES, types.MappingProxyType)
    assert set(compute_module.SPECIES_TEMPLATES) == set(
        compute_module._SPECIES_TEMPLATES
    )


def test_recipes_proxy_rejects_mutation() -> None:
    """External callers cannot bypass ``register_recipe`` via the proxy."""
    with pytest.raises(TypeError):
        compute_module.RECIPES["impossible"] = None  # type: ignore[index]


def test_species_templates_proxy_rejects_mutation() -> None:
    """External callers cannot mutate the species-template registry."""
    with pytest.raises(TypeError):
        compute_module.SPECIES_TEMPLATES["impossible"] = None  # type: ignore[index]


def test_recipes_proxy_reflects_register_recipe_mutations() -> None:
    """Mutations through the official API show up in the proxy view."""
    import numpy as np

    from pypic.fields import QuantityType

    name = "_test_public_api_recipe"
    assert name not in compute_module.RECIPES
    try:
        compute_module.register_recipe(
            name,
            func=lambda b: np.asarray(b) * 0.0,
            fields=("|B|",),
            quantity_type=QuantityType.DIMENSIONLESS,
        )
        assert name in compute_module.RECIPES
        assert compute_module.RECIPES[name] is compute_module._REGISTRY[name]
    finally:
        compute_module.unregister_recipe(name)
    assert name not in compute_module.RECIPES


def test_codegen_smoke_imports() -> None:
    """The exact import patterns used by webpic codegen all resolve."""
    from pypic import RECIPES as TopLevelRecipes  # noqa: F401
    from pypic import Recipe as TopLevelRecipe  # noqa: F401
    from pypic.aliases import (  # noqa: F401
        COMPUTE_ALIASES,
        GROUP_ALIASES,
        SPECIES_SUFFIX_RE,
        species_name_aliases,
    )
    from pypic.compute import (  # noqa: F401
        RECIPES,
        SPECIES_TEMPLATES,
        Recipe,
        SpeciesArgs,
        SpeciesTemplate,
    )


# Modules a user is expected to reach as `pypic.<name>` after a bare
# `import pypic`, without a separate import statement. Optional-extra
# subpackages (plotting, server) are excluded on purpose: importing them
# eagerly would pull matplotlib and fastapi into every core install.
_ATTRIBUTE_REACHABLE_MODULES = [
    "aliases",
    "codegen",
    "comparison",
    "compute",
    "coordinates",
    "derived",
    "diagnostics",
    "exceptions",
    "fields",
    "io",
    "reconnection",
    "reductions",
    "regrid",
    "schema",
    "selections",
    "spectral",
    "traces",
    "units",
]


def test_headline_modules_are_reachable_after_importing_pypic() -> None:
    """`import pypic` then `pypic.spectral` — no second import statement.

    ``pypic.spectral`` and ``pypic.reconnection`` were unreachable this
    way: ``pypic/__init__.py`` never imported them, so a user could plot
    a power spectrum via ``pypic.plotting.plot_power_spectrum`` but had
    to reach past the documented surface to compute one.
    """
    unreachable = [m for m in _ATTRIBUTE_REACHABLE_MODULES if not hasattr(pypic, m)]
    assert not unreachable, f"not reachable as pypic.<name>: {sorted(unreachable)}"


def test_top_level_all_entries_resolve() -> None:
    """Nothing in ``pypic.__all__`` is a name that does not exist."""
    unresolved = [name for name in pypic.__all__ if not hasattr(pypic, name)]
    assert not unresolved, f"pypic.__all__ names nothing: {sorted(unresolved)}"


def _probe(source: str) -> str:
    """Run *source* in a fresh interpreter and return its stdout, stripped."""
    result = subprocess.run(
        [sys.executable, "-c", source], capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def test_importing_pypic_leaves_the_interpolator_unloaded() -> None:
    """A bare ``import pypic`` does not pull ``scipy.interpolate``.

    ``regrid`` and ``traces`` import it at module scope, and eager
    re-export cost a third of ``import pypic`` for every session that
    never regrids or traces a field line. The loaded module set is what
    the deferral controls; a wall-clock ceiling would flake on CI.
    """
    loaded = _probe("import sys, pypic; print('scipy.interpolate' in sys.modules)")
    assert loaded == "False"


def test_deferred_name_leaves_regrid_bound_to_the_function() -> None:
    """``pypic.regrid`` is the function no matter which name is touched first.

    Importing the submodule binds it as ``pypic.regrid``, where the
    eager surface had the function of the same name — so resolving any
    one deferred name binds its module's whole export set at once.
    """
    kind = _probe("import pypic; pypic.align_grids; print(type(pypic.regrid).__name__)")
    assert kind == "function"


_CROSS_CHECKED_SUBMODULES = [
    "pypic.comparison",
    "pypic.derived",
    "pypic.diagnostics",
    "pypic.io",
    "pypic.reconnection",
    "pypic.reductions",
    "pypic.regrid",
    "pypic.selections",
    "pypic.spectral",
    "pypic.traces",
]


def test_re_exported_names_agree_with_their_home_module() -> None:
    """A name public at top level is public where it is defined.

    ``quantity_dimension`` was in ``pypic.__all__`` but missing from
    ``pypic.fields.__all__`` — the two disagreed about the same symbol.
    """
    import importlib

    disagreements: list[str] = []
    for mod_name in _CROSS_CHECKED_SUBMODULES:
        module = importlib.import_module(mod_name)
        module_all = set(getattr(module, "__all__", ()))
        for name in module_all:
            obj = getattr(module, name, None)
            top = getattr(pypic, name, None)
            if obj is not None and top is not None and obj is not top:
                disagreements.append(f"{mod_name}.{name}")
    assert not disagreements, (
        f"top-level re-export is a different object: {sorted(disagreements)}"
    )


_FULLY_RE_EXPORTED_MODULES = [
    "pypic.compute",
    "pypic.derived",
    "pypic.diagnostics",
    "pypic.fields",
]


def test_physics_modules_are_fully_re_exported() -> None:
    """Every public name in the physics modules reaches the top level.

    These four modules are rendered wholesale by ``docs/api/*.md``, so a
    name public there is a name a reader will try to import from
    ``pypic``. Twenty ``derived`` functions were reachable only as
    ``pypic.derived.x`` while their thirty-eight siblings were not — an
    arbitrary split with nothing enforcing it either way.
    """
    import importlib

    top = set(pypic.__all__)
    missing: dict[str, list[str]] = {}
    for mod_name in _FULLY_RE_EXPORTED_MODULES:
        module = importlib.import_module(mod_name)
        gaps = sorted(n for n in getattr(module, "__all__", ()) if n not in top)
        if gaps:
            missing[mod_name] = gaps
    assert not missing, (
        f"public names not re-exported from pypic: {missing}. Add them to "
        f"pypic/__init__.py, or drop them from the module's __all__."
    )


# Modules below ``dataset`` in the dependency order (docs/architecture.md
# § Module layout). They may import each other, never anything above.
_BELOW_DATASET = [
    "_aliases",
    "_field_table",
    "containers",
    "coordinates/geometry",
    "coordinates/operators",
    "coordinates/transforms",
    "exceptions",
    "fields",
    "grid",
    "types",
    "units",
]
_ABOVE_DATASET = (
    "pypic.dataset",
    "pypic.compute",
    "pypic.reductions",
    "pypic.regrid",
    "pypic.comparison",
    "pypic.selections",
    "pypic.readers",
    "pypic.io",
    "pypic.plotting",
    "pypic.server",
    "pypic.cli",
)


def _runtime_pypic_imports(source: str) -> list[str]:
    """Every ``pypic.*`` module imported outside ``if TYPE_CHECKING:``."""
    tree = ast.parse(source)
    guarded: set[int] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Name)
            and node.test.id == "TYPE_CHECKING"
        ):
            guarded.update(range(node.lineno, node.end_lineno + 1))
    found: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Import | ast.ImportFrom) or node.lineno in guarded:
            continue
        if isinstance(node, ast.ImportFrom):
            found.append(node.module or "")
        else:
            found.extend(alias.name for alias in node.names)
    return [m for m in found if m == "pypic" or m.startswith("pypic.")]


def test_modules_below_dataset_never_import_above_it() -> None:
    """The arrow ``grid ← containers ← dataset ← everything else`` is one-way.

    Function-local imports count: a lazy ``from pypic.dataset import`` in
    ``grid`` is still a dependency pointing the wrong way.
    """
    src = Path(pypic.__file__).parent
    offenders: list[str] = []
    for module in _BELOW_DATASET:
        for imported in _runtime_pypic_imports((src / f"{module}.py").read_text()):
            if imported == "pypic" or imported.startswith(_ABOVE_DATASET):
                offenders.append(f"{module}.py imports {imported}")
    assert not offenders, "imports pointing above dataset: " + ", ".join(offenders)
