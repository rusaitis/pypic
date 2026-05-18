"""Public-surface invariants for codegen consumers.

The cross-tool toolchain (webpic, rustpic tooling) reads the symbols
checked here to generate TypeScript/Rust mirrors.  Treat any failure as
a contract break — bump the schema major before changing the behavior.
"""

from __future__ import annotations

import types

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


def test_recipes_proxy_rejects_mutation() -> None:
    """External callers cannot bypass ``register_recipe`` via the proxy."""
    with pytest.raises(TypeError):
        compute_module.RECIPES["impossible"] = None  # type: ignore[index]


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
