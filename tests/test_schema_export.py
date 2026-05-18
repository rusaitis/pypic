"""Tests for the JSON Schema export pipeline.

Validates that ``pypic.schema._export.build_schema`` produces a stable,
self-consistent JSON Schema 2020-12 document, that the on-disk artifact
at ``src/pypic/schema/simulation.schema.v1.0.json`` stays in sync, and
that the reference template in ``pypic.simulation.toml`` validates
against it.
"""

from __future__ import annotations

import datetime as dt
import difflib
import json
import tomllib
from pathlib import Path
from typing import Any

import jsonschema
import pytest
from typer.testing import CliRunner

from pypic.cli import app as pypic_app
from pypic.schema import (
    SCHEMA_VERSION,
    build_schema,
    dump_schema,
    get_schema_path,
)
from pypic.schema.cli import app as schema_app

REPO_ROOT = Path(__file__).resolve().parent.parent
REFERENCE_TOML = REPO_ROOT / "pypic.simulation.toml"
ON_DISK_SCHEMA = REPO_ROOT / "src" / "pypic" / "schema" / "simulation.schema.v1.0.json"


def _json_canonicalize(value: Any) -> Any:
    """Convert TOML-native types into their JSON-equivalent shape.

    tomllib parses date literals into ``datetime.date``; JSON Schema's
    ``format: "date"`` accepts only ISO strings. The reference TOML is
    valid pypic input under Pydantic (which auto-coerces both), but for
    JSON-side validation we mirror what a JS/Rust consumer would
    serialize on the wire.
    """
    if isinstance(value, dt.datetime):
        return value.isoformat()
    if isinstance(value, dt.date):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: _json_canonicalize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_canonicalize(item) for item in value]
    return value


def test_build_schema_is_json_serializable() -> None:
    schema = build_schema()
    payload = json.dumps(schema)
    assert json.loads(payload) == schema


def test_generated_schema_is_valid_draft_2020_12() -> None:
    schema = build_schema()
    jsonschema.Draft202012Validator.check_schema(schema)


def test_discriminated_unions_roundtrip() -> None:
    schema = build_schema()

    units = schema["properties"]["units"]
    assert "oneOf" in units
    assert units["discriminator"]["propertyName"] == "system"
    expected_mapping = {"PIC", "MHD", "SI", "custom"}
    assert set(units["discriminator"]["mapping"]) == expected_mapping
    for branch in units["oneOf"]:
        ref = branch["$ref"]
        target = schema["$defs"][ref.removeprefix("#/$defs/")]
        assert target["properties"]["system"]["const"] in expected_mapping

    region = schema["$defs"]["OutputStream"]["properties"]["region"]["anyOf"][0]
    box_or_plane = region.get("$ref") or region.get("oneOf")
    assert box_or_plane is not None


def test_strict_vs_extensible_bases_serialize_consistently() -> None:
    schema = build_schema()

    for strict_name in ("Run", "UnitsPIC", "Allocation"):
        node = schema["$defs"][strict_name]
        assert node["additionalProperties"] is False, strict_name

    assert schema["additionalProperties"] is True

    for extensible_name in ("PhysicsPIC", "PhysicsMHD", "PhysicsHybrid"):
        node = schema["$defs"][extensible_name]
        assert node.get("additionalProperties", True) is True, extensible_name


def test_pretty_and_compact_are_semantically_equivalent() -> None:
    schema = build_schema()
    pretty = dump_schema(schema, pretty=True)
    compact = dump_schema(schema, pretty=False)
    assert json.loads(pretty) == json.loads(compact)
    assert len(compact) < len(pretty)
    assert pretty.endswith("\n")
    assert compact.endswith("\n")


def test_on_disk_artifact_matches_current_models() -> None:
    """Drift guard: regenerated schema must equal the committed file.

    Regenerate via:
        uv run pypic schema export -o src/pypic/schema/simulation.schema.v1.0.json
    """
    assert ON_DISK_SCHEMA.exists(), (
        f"Expected {ON_DISK_SCHEMA} to be committed. Generate via "
        "`uv run pypic schema export -o "
        "src/pypic/schema/simulation.schema.v1.0.json`."
    )
    on_disk = ON_DISK_SCHEMA.read_text()
    regenerated = dump_schema(build_schema(), pretty=True)
    if on_disk != regenerated:
        diff = "\n".join(
            difflib.unified_diff(
                on_disk.splitlines(),
                regenerated.splitlines(),
                fromfile=str(ON_DISK_SCHEMA.name),
                tofile="regenerated",
                lineterm="",
                n=3,
            )
        )
        pytest.fail(
            "On-disk JSON Schema is out of sync with pypic.schema._models.\n"
            "Regenerate: uv run pypic schema export -o "
            f"{ON_DISK_SCHEMA.relative_to(REPO_ROOT)}\n\n"
            f"Diff (first 4000 chars):\n{diff[:4000]}"
        )


def test_reference_template_validates_against_generated_schema() -> None:
    schema = build_schema()
    raw = REFERENCE_TOML.read_bytes()
    doc = _json_canonicalize(tomllib.loads(raw.decode()))
    jsonschema.validate(doc, schema)


def test_alias_keys_use_toml_spelling() -> None:
    schema = build_schema()
    assert "schema" in schema["properties"]
    assert "schema_" not in schema["properties"]
    assert "from" in schema["$defs"]["Restart"]["properties"]
    assert "from_" not in schema["$defs"]["Restart"]["properties"]


def test_cli_export_smoke() -> None:
    runner = CliRunner()
    result = runner.invoke(schema_app, ["export", "--compact"])
    assert result.exit_code == 0, result.stdout
    parsed = json.loads(result.stdout)
    assert parsed["$id"].endswith(f"/v{SCHEMA_VERSION}.json")

    main_result = runner.invoke(pypic_app, ["schema", "export", "--compact"])
    assert main_result.exit_code == 0, main_result.stdout
    assert json.loads(main_result.stdout) == parsed


def test_cli_export_codegen_flags_smoke() -> None:
    """Wire-check for ``--inline-single-use-defs --include-x-extensions``.

    The cross-tool codegen pipelines (webpic, rustpic tooling) pin both
    flags. Each flag's semantics is covered by
    ``test_inline_single_use_defs_flattens_unique_refs`` and
    ``test_include_x_extensions_annotates_extensible_objects``; this
    test guards the CLI surface that wires them together.
    """
    runner = CliRunner()
    result = runner.invoke(
        schema_app,
        ["export", "--inline-single-use-defs", "--include-x-extensions", "--compact"],
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["$id"].endswith(f"/v{SCHEMA_VERSION}.json")
    # Flags actually change the output relative to defaults.
    default = runner.invoke(schema_app, ["export", "--compact"])
    assert default.exit_code == 0, default.stdout
    assert result.stdout != default.stdout
    # The --inline-single-use-defs effect: fewer $defs after inlining.
    default_payload = json.loads(default.stdout)
    assert len(payload.get("$defs", {})) < len(default_payload["$defs"])
    # The --include-x-extensions effect: x-* patternProperties appear
    # somewhere in the document (location is brittle to which $defs
    # survive inlining — string-scan instead).
    assert '"^x[-_]' in result.stdout
    assert '"^x[-_]' not in default.stdout


def test_get_schema_path_resolves_to_bundled_file() -> None:
    path = get_schema_path()
    assert path.exists()
    parsed = json.loads(path.read_text())
    assert parsed["$id"].endswith(f"/v{SCHEMA_VERSION}.json")


def test_inline_single_use_defs_flattens_unique_refs() -> None:
    full = build_schema()
    inlined = build_schema(inline_single_use_defs=True)
    assert len(inlined.get("$defs", {})) < len(full["$defs"])
    jsonschema.Draft202012Validator.check_schema(inlined)


def test_include_x_extensions_annotates_extensible_objects() -> None:
    annotated = build_schema(include_x_extensions=True)
    physics_pic = annotated["$defs"]["PhysicsPIC"]
    assert "patternProperties" in physics_pic
    assert "^x[-_]" in physics_pic["patternProperties"]
    run = annotated["$defs"]["Run"]
    assert "patternProperties" not in run
