"""JSON export of pypic's canonical tables for cross-language codegen.

The schema is already exported via `pypic.schema._export`; this module
adds the rest of the name/physics authority — compute aliases, the recipe
registry, species templates, and field metadata — as a single JSON bundle.
Cross-language consumers (webpic's Zod/TS codegen, rustpic tooling) read the
bundle instead of re-typing the tables by hand.

Pure (stdlib only, no typer): the thin CLI lives in `pypic._codegen_cli`,
mirroring the ``schema._export`` / ``_schema_cli`` split. JSON keys are camelCase
for ergonomic consumption by the TypeScript side.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pypic._field_table import _FIELD_INFO
from pypic.aliases import COMPUTE_ALIASES, GROUP_ALIASES, SPECIES_SUFFIX_RE
from pypic.compute import RECIPES, SPECIES_TEMPLATES, Recipe, SpeciesTemplate
from pypic.schema._export import build_schema
from pypic.schema._models import SCHEMA_VERSION

if TYPE_CHECKING:
    from pypic.fields import FieldInfo

__all__ = [
    "export_aliases",
    "export_bundle",
    "export_fields",
    "export_recipes",
]


def _recipe_dict(recipe: Recipe) -> dict[str, Any]:
    """Serialize a Recipe sans its (non-portable) ``func`` callable.

    ``func`` becomes the bare function name; the TS side re-binds it to a
    hand-written derived-op implementation. ``species_args`` is a StrEnum,
    so ``.value`` is a clean string; ``None``s are preserved.
    """
    return {
        "func": recipe.func.__name__,
        "fields": list(recipe.fields),
        "speciesIndex": recipe.species_index,
        "needsGrid": recipe.needs_grid,
        "needsGamma": recipe.needs_gamma,
        "needsC": recipe.needs_c,
        "component": recipe.component,
        "speciesArgs": None
        if recipe.species_args is None
        else recipe.species_args.value,
        "passesGeometry": recipe.passes_geometry,
        "supportsRelativistic": recipe.supports_relativistic,
    }


def _template_dict(template: SpeciesTemplate) -> dict[str, Any]:
    """Serialize a SpeciesTemplate, keeping the ``{N}`` field placeholders."""
    return {
        "func": template.func.__name__,
        "fieldPattern": list(template.field_pattern),
        "speciesArgs": template.species_args.value,
        "needsGamma": template.needs_gamma,
        "needsC": template.needs_c,
        "component": template.component,
        "supportsRelativistic": template.supports_relativistic,
    }


def _fieldinfo_dict(info: FieldInfo) -> dict[str, Any]:
    return {
        "quantityType": str(info.quantity_type),
        "longName": info.long_name,
        "siUnit": info.si_unit,
        "latex": info.latex,
        "unitDimension": None
        if info.unit_dimension is None
        else list(info.unit_dimension),
    }


def export_aliases() -> dict[str, Any]:
    """Return the alias tables (compute aliases, group aliases, species regex).

    Examples
    --------
    >>> sorted(export_aliases())
    ['computeAliases', 'groupAliases', 'speciesSuffixRe']
    """
    return {
        "computeAliases": dict(COMPUTE_ALIASES),
        "groupAliases": dict(GROUP_ALIASES),
        "speciesSuffixRe": SPECIES_SUFFIX_RE.pattern,
    }


def export_recipes() -> dict[str, Any]:
    """Return the recipe registry and per-species templates (metadata only).

    Examples
    --------
    The camelCase keys below are the wire contract webpic reads:

    >>> recipes = export_recipes()["recipes"]
    >>> recipes["v_A"]["fields"]
    ['|B|', 'rho_m']
    >>> recipes["v_A"]["supportsRelativistic"], recipes["v_A"]["passesGeometry"]
    (True, False)
    """
    return {
        "recipes": {key: _recipe_dict(recipe) for key, recipe in RECIPES.items()},
        "speciesTemplates": {
            key: _template_dict(template) for key, template in SPECIES_TEMPLATES.items()
        },
    }


def export_fields() -> dict[str, Any]:
    """Return static field metadata: units, LaTeX symbols, long names.

    Examples
    --------
    >>> b1 = export_fields()["fields"]["B_1"]
    >>> b1["quantityType"], b1["siUnit"], b1["latex"]
    ('b_field', 'T', '$B_1$')
    """
    return {
        "fields": {name: _fieldinfo_dict(info) for name, info in _FIELD_INFO.items()}
    }


def export_bundle(
    *,
    schema_version: str = SCHEMA_VERSION,
    inline_single_use_defs: bool = False,
    include_x_extensions: bool = False,
) -> dict[str, Any]:
    """Return the unified bundle: schema + aliases + recipes + field metadata.

    ``inline_single_use_defs`` / ``include_x_extensions`` pass through to
    `pypic.schema._export.build_schema` (the former yields flatter Zod).

    Examples
    --------
    >>> bundle = export_bundle()
    >>> bundle["schemaVersion"]
    '1.0'
    >>> sorted(bundle)  # doctest: +NORMALIZE_WHITESPACE
    ['computeAliases', 'fields', 'groupAliases', 'jsonSchema', 'recipes',
     'schemaVersion', 'speciesSuffixRe', 'speciesTemplates']
    """
    return {
        "schemaVersion": schema_version,
        "jsonSchema": build_schema(
            include_x_extensions=include_x_extensions,
            inline_single_use_defs=inline_single_use_defs,
            schema_version=schema_version,
        ),
        **export_aliases(),
        **export_recipes(),
        **export_fields(),
    }
