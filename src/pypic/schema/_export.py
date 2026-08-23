"""JSON Schema export for the pypic ``simulation.toml`` v1.0 schema.

Generates a stable, post-processed JSON Schema 2020-12 document from the
Pydantic v2 models in `pypic.schema._models`. Pydantic's raw
``model_json_schema()`` output is correct but verbose: class-name titles
on every sub-schema, ``"description": "An enumeration."`` on enums, and
no guarantee of stable key order. The post-processing here strips that
noise and sorts the result so cross-language consumers (the webpic Zod
codegen, MATLAB validators, downstream Python users) work against a
diff-stable shape.

The generated artifact ships at
``src/pypic/schema/simulation.schema.v{SCHEMA_VERSION}.json`` and is
regenerable via ``uv run pypic schema export``; the drift test in
``tests/test_schema_export.py`` and a CI step guard against staleness.

Cross-field invariants (``model_validator(mode="after")`` decorators in
`pypic.schema._models`) cannot be expressed in JSON Schema and are
documented in the top-level ``$comment``. Consumers that need the full
contract use ``pypic.schema.validate_simulation_toml`` at runtime.
"""

from __future__ import annotations

import importlib.resources
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from pypic.schema._models import SCHEMA_VERSION, SimulationSchema

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

__all__ = [
    "build_schema",
    "dump_schema",
    "get_schema_path",
]


def build_schema(
    *,
    include_x_extensions: bool = False,
    inline_single_use_defs: bool = False,
    schema_version: str = SCHEMA_VERSION,
) -> dict[str, Any]:
    """Return the post-processed JSON Schema 2020-12 document.

    Parameters
    ----------
    include_x_extensions
        When True, annotate non-strict object schemas with
        ``patternProperties: {"^x[-_]": {}}`` to document the v1.0
        ``x-<code>`` extension namespace. Off by default for a leaner
        document; the namespace is described in the top-level
        ``$comment`` regardless.
    inline_single_use_defs
        When True, ``$defs`` entries referenced exactly once are inlined
        at the reference site and removed from ``$defs``. Helps
        ``json-schema-to-zod`` produce flatter Zod, at the cost of a
        larger root schema. Off by default.
    schema_version
        Override the version stamped in ``$id`` and ``title``. Does not
        affect which models are exported.
    """
    schema = SimulationSchema.model_json_schema(
        mode="validation",
        ref_template="#/$defs/{model}",
        by_alias=True,
    )
    _drop_pydantic_noise(schema)
    if inline_single_use_defs:
        _inline_single_use_defs(schema)
    if include_x_extensions:
        _annotate_x_extensions(schema)
    _stamp_metadata(schema, schema_version=schema_version)
    return cast("dict[str, Any]", _sort_keys(schema))


def dump_schema(schema: Mapping[str, Any], *, pretty: bool = True) -> str:
    """Serialize ``schema`` as JSON with stable key ordering.

    Both ``pretty`` and ``compact`` output end with a trailing newline
    so the file plays nicely with POSIX text-file conventions and
    ``git diff``, and so callers don't have to special-case the sink.
    """
    if pretty:
        return json.dumps(schema, indent=2, sort_keys=True) + "\n"
    return json.dumps(schema, separators=(",", ":"), sort_keys=True) + "\n"


def get_schema_path(version: str = SCHEMA_VERSION) -> Path:
    """Locate the bundled JSON Schema file inside the installed package.

    Returns the path to ``simulation.schema.v{version}.json`` shipped in
    the pypic wheel. Works for normal ``pip install``; zipped installs
    would require ``importlib.resources.as_file`` — pypic isn't packaged
    that way in practice (requires-python >=3.13 wheels).
    """
    ref = importlib.resources.files("pypic.schema").joinpath(
        f"simulation.schema.v{version}.json"
    )
    return Path(str(ref))


# Post-processing passes — each pure (mutates its input dict) and idempotent.


def _walk(node: Any) -> Iterator[dict[str, Any]]:  # noqa: ANN401
    """Depth-first traversal yielding every dict in a JSON-Schema tree."""
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _walk(value)
    elif isinstance(node, list):
        for item in node:
            yield from _walk(item)


def _is_pydantic_auto_title(value: str) -> bool:
    """PascalCase identifier with no whitespace — Pydantic's auto-format.

    Explicit titles set via ``Field(title=...)`` almost always include
    spaces or differ structurally; the auto-generated ones are bare
    class names (``"Run"``, ``"UnitsPIC"``) that already live as
    ``$defs`` keys and add only churn.
    """
    return value.isidentifier() and value[:1].isupper() and " " not in value


def _drop_pydantic_noise(schema: dict[str, Any]) -> None:
    """Remove auto-generated titles and ``An enumeration.`` descriptions."""
    for node in _walk(schema):
        title = node.get("title")
        if isinstance(title, str) and _is_pydantic_auto_title(title):
            del node["title"]
        if node.get("description") == "An enumeration.":
            del node["description"]


def _inline_single_use_defs(schema: dict[str, Any]) -> None:
    defs = schema.get("$defs")
    if not isinstance(defs, dict) or not defs:
        return
    counts: dict[str, int] = dict.fromkeys(defs, 0)
    for node in _walk(schema):
        ref = node.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/$defs/"):
            name = ref.removeprefix("#/$defs/")
            if name in counts:
                counts[name] += 1
    inline = {name for name, count in counts.items() if count == 1}
    if not inline:
        return
    for node in _walk(schema):
        ref = node.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/$defs/"):
            name = ref.removeprefix("#/$defs/")
            if name in inline:
                node.pop("$ref")
                node.update(defs[name])
    for name in inline:
        del defs[name]
    if not defs:
        del schema["$defs"]


def _annotate_x_extensions(schema: dict[str, Any]) -> None:
    """Add ``patternProperties`` annotations for the ``x-*`` namespace.

    Skipped on schemas with ``additionalProperties: false`` — adding
    ``patternProperties`` there would change validation behavior, not
    just annotate. The strict bases (``_StrictBase``) reject ``x-*``
    keys today; the v1.0 extension convention only applies to extensible
    bases (root, physics sub-tables), which serialize with
    ``additionalProperties: true``.
    """
    for node in _walk(schema):
        if node.get("type") != "object":
            continue
        if node.get("additionalProperties") is False:
            continue
        node.setdefault("patternProperties", {})["^x[-_]"] = {}


def _stamp_metadata(schema: dict[str, Any], *, schema_version: str) -> None:
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["$id"] = f"https://pypic.dev/schemas/simulation/v{schema_version}.json"
    schema["title"] = f"pypic simulation.toml v{schema_version}"
    schema["$comment"] = (
        "Structural schema only. Generated from "
        "pypic.schema._models.SimulationSchema via "
        "pypic.schema._export.build_schema. Cross-field invariants "
        "(range checks on ensemble member_id, time scheme/dt/cfl rules, "
        "grid axis/upper-lower/stretched-widths/stagger checks, "
        "coordinate-transform BFS resolution, driver target precedence, "
        "restart.from existence, root-level cross-references to species, "
        "bodies, drivers, and collisions) are enforced at runtime by "
        "pypic.schema.validate_simulation_toml and are not expressible "
        "in JSON Schema. The v1.0 'x-*' extension namespace is permitted "
        "wherever additionalProperties is true."
    )


def _sort_keys(schema: Any) -> Any:  # noqa: ANN401
    """Recursively sort dict keys for stable diffs.

    Serializing with ``sort_keys=True`` covers the on-disk file, but
    callers that hold the dict (e.g., to validate against ``jsonschema``)
    also see a stable shape this way.
    """
    if isinstance(schema, dict):
        return {key: _sort_keys(value) for key, value in sorted(schema.items())}
    if isinstance(schema, list):
        return [_sort_keys(item) for item in schema]
    return schema
