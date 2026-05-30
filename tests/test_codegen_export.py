"""Tests for the codegen JSON export (pypic.codegen + pypic._codegen_cli)."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from pypic._codegen_cli import app as export_app
from pypic.codegen import export_aliases, export_bundle, export_fields, export_recipes

runner = CliRunner()


def test_bundle_is_json_serializable() -> None:
    bundle = export_bundle()
    assert json.loads(json.dumps(bundle)) == bundle


def test_bundle_has_all_sections() -> None:
    bundle = export_bundle()
    for key in ("schemaVersion", "jsonSchema", "computeAliases", "recipes", "fields"):
        assert key in bundle


def test_recipe_magnitude_dependencies() -> None:
    recipes = export_recipes()["recipes"]
    assert recipes["|B|"]["fields"] == ["B_1", "B_2", "B_3"]
    # func is serialized as a bare name, not a callable
    assert isinstance(recipes["|B|"]["func"], str)


def test_alias_resolves_descriptive_to_canonical() -> None:
    aliases = export_aliases()["computeAliases"]
    assert aliases["plasma_beta"] == "beta"


def test_field_metadata_units() -> None:
    fields = export_fields()["fields"]
    assert fields["|B|"]["siUnit"] == "T"


def test_cli_bundle_emits_valid_json() -> None:
    result = runner.invoke(export_app, ["bundle", "--compact"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["recipes"]["|B|"]["fields"] == ["B_1", "B_2", "B_3"]
