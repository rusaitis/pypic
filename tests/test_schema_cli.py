"""Tests for ``pypic schema validate`` and ``pypic schema diff`` CLI surface.

Three suites:

- :class:`TestSchemaValidateCli` — per-file results, exit codes,
  ``--json`` machine-readable mode.
- :class:`TestSchemaDiffCli` — second-arg defaults to bundled, git-style
  exit codes, ``--format json`` structured output.
- :class:`TestSchemaDiffPure` — direct exercise of the pure-Python diff
  helpers exposed by :mod:`pypic.schema._diff`.

The reference TOML at the repo root and the bundled JSON Schema are
used as ground truth for the happy-path tests; negative cases build
minimal docs inline.
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from pypic._schema_cli import app as schema_app
from pypic.cli import app as pypic_app
from pypic.schema import SCHEMA_VERSION
from pypic.schema._diff import schema_structured_diff, schema_text_diff

REPO_ROOT = Path(__file__).resolve().parent.parent
REFERENCE_TOML = REPO_ROOT / "pypic.simulation.toml"
ON_DISK_SCHEMA = (
    REPO_ROOT / "src" / "pypic" / "schema" / f"simulation.schema.v{SCHEMA_VERSION}.json"
)


def _build_toml(**overrides: object) -> str:
    """Return a minimal valid simulation.toml; pass ``key=None`` to drop a section."""
    sections: dict[str, str] = {
        "schema": '[schema]\nversion = "2.0"\n',
        "model": '[model]\nname = "demo"\ntype = "PIC"\n',
        "run": '[run]\nname = "r0"\n',
        "time": (
            '[time]\nscheme = "fixed"\ndt = 0.1\nt_start = 0.0\n'
            "t_end = 1.0\nn_steps = 10\n"
        ),
        "grid": (
            "[grid]\ndimensions = [4, 4, 4]\nspacing = [1.0, 1.0, 1.0]\n"
            "lower = [0.0, 0.0, 0.0]\nupper = [4.0, 4.0, 4.0]\n"
        ),
        "units": '[units]\nanchor = "si"\n',
        "coordinates": '[coordinates]\ngeometry = "cartesian"\nframe = "sim"\n',
        "species": '[[species]]\nname = "electrons"\ncharge = -1.0\nmass = 1.0\n',
    }
    for key, value in overrides.items():
        if value is None:
            sections.pop(key, None)
        else:
            sections[key] = str(value)
    return "\n".join(sections.values())


class TestSchemaValidateCli:
    def test_reference_template_passes(self) -> None:
        runner = CliRunner()
        result = runner.invoke(schema_app, ["validate", str(REFERENCE_TOML)])
        assert result.exit_code == 0, result.output
        assert "✓" in result.output

    def test_missing_required_section_exits_one(self, tmp_path: Path) -> None:
        path = tmp_path / "missing_model.toml"
        path.write_text(_build_toml(model=None))
        runner = CliRunner()
        result = runner.invoke(schema_app, ["validate", str(path)])
        assert result.exit_code == 1, result.output
        assert "✗" in result.output

    def test_section_names_survive_rich_rendering(self, tmp_path: Path) -> None:
        """Validation messages name TOML sections, and rich eats brackets.

        Unescaped, ``[units]`` reads as a style tag and disappears,
        leaving a message that begins mid-sentence and never says
        which section is at fault.
        """
        path = tmp_path / "underdetermined.toml"
        path.write_text(
            _build_toml(units='[units]\nanchor = "explicit"\nreference_length = 1.0')
        )
        runner = CliRunner()
        result = runner.invoke(schema_app, ["validate", str(path)])
        assert "[units]" in result.output, result.output

    def test_syntactically_broken_toml_exits_two(self, tmp_path: Path) -> None:
        path = tmp_path / "syntax.toml"
        path.write_text("broken =\n")
        runner = CliRunner()
        result = runner.invoke(schema_app, ["validate", str(path)])
        assert result.exit_code == 2, result.output

    def test_nonexistent_path_exits_two(self, tmp_path: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(
            schema_app, ["validate", str(tmp_path / "does-not-exist.toml")]
        )
        assert result.exit_code == 2, result.output

    def test_multiple_valid_paths_all_pass(self, tmp_path: Path) -> None:
        a = tmp_path / "a.toml"
        a.write_text(_build_toml())
        b = tmp_path / "b.toml"
        b.write_text(_build_toml())
        runner = CliRunner()
        result = runner.invoke(schema_app, ["validate", str(a), str(b)])
        assert result.exit_code == 0, result.output
        assert result.output.count("✓") == 2

    def test_mixed_valid_and_invalid_exits_one(self, tmp_path: Path) -> None:
        good = tmp_path / "good.toml"
        good.write_text(_build_toml())
        bad = tmp_path / "bad.toml"
        bad.write_text(_build_toml(model=None))
        runner = CliRunner()
        result = runner.invoke(schema_app, ["validate", str(good), str(bad)])
        assert result.exit_code == 1, result.output

    def test_json_output_happy_path(self, tmp_path: Path) -> None:
        path = tmp_path / "ok.toml"
        path.write_text(_build_toml())
        runner = CliRunner()
        result = runner.invoke(schema_app, ["validate", str(path), "--json"])
        assert result.exit_code == 0, result.output
        parsed = json.loads(result.stdout)
        assert parsed["files"][0]["ok"] is True
        assert parsed["files"][0]["errors"] == []

    def test_json_output_with_validation_error(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.toml"
        path.write_text(_build_toml(model=None))
        runner = CliRunner()
        result = runner.invoke(schema_app, ["validate", str(path), "--json"])
        assert result.exit_code == 1, result.output
        parsed = json.loads(result.stdout)
        entry = parsed["files"][0]
        assert entry["ok"] is False
        assert entry["errors"]
        first = entry["errors"][0]
        assert {"type", "loc", "msg"} <= first.keys()

    def test_stdin_with_other_paths_exits_two(self, tmp_path: Path) -> None:
        # '-' is only valid when it's the sole input — otherwise the
        # second '-' (or any stdin slot beyond the first) would read
        # an already-drained stream and silently produce empty input.
        other = tmp_path / "ok.toml"
        other.write_text(_build_toml())
        runner = CliRunner()
        result = runner.invoke(schema_app, ["validate", "-", str(other)])
        assert result.exit_code == 2, result.output
        assert "stdin" in result.output.lower()


class TestSchemaDiffCli:
    def test_same_file_no_diff(self) -> None:
        runner = CliRunner()
        result = runner.invoke(
            schema_app, ["diff", str(ON_DISK_SCHEMA), str(ON_DISK_SCHEMA)]
        )
        assert result.exit_code == 0, result.output
        assert result.stdout == ""

    def test_default_second_arg_is_bundled(self) -> None:
        runner = CliRunner()
        result = runner.invoke(schema_app, ["diff", str(ON_DISK_SCHEMA)])
        assert result.exit_code == 0, result.output
        assert result.stdout == ""

    def test_modified_copy_shows_diff(self, tmp_path: Path) -> None:
        original = json.loads(ON_DISK_SCHEMA.read_text())
        modified = dict(original)
        modified["$defs"] = {k: v for k, v in modified["$defs"].items() if k != "Run"}
        modified_path = tmp_path / "modified.json"
        modified_path.write_text(json.dumps(modified, indent=2) + "\n")
        runner = CliRunner()
        # a = modified (no Run), b = bundled (has Run); diff direction a→b
        # shows Run as added on the b side.
        result = runner.invoke(schema_app, ["diff", str(modified_path)])
        assert result.exit_code == 1, result.output
        assert "Run" in result.output

    def test_structured_format_lists_added_run(self, tmp_path: Path) -> None:
        original = json.loads(ON_DISK_SCHEMA.read_text())
        modified = dict(original)
        modified["$defs"] = {k: v for k, v in modified["$defs"].items() if k != "Run"}
        modified_path = tmp_path / "modified.json"
        modified_path.write_text(json.dumps(modified, indent=2) + "\n")
        runner = CliRunner()
        result = runner.invoke(
            schema_app, ["diff", str(modified_path), "--format", "json"]
        )
        assert result.exit_code == 1, result.output
        parsed = json.loads(result.stdout)
        assert "/$defs/Run" in parsed["added"]

    def test_malformed_json_exits_two(self, tmp_path: Path) -> None:
        bad = tmp_path / "broken.json"
        bad.write_text("{not json")
        runner = CliRunner()
        result = runner.invoke(schema_app, ["diff", str(bad)])
        assert result.exit_code == 2, result.output

    def test_stdin_for_second_arg_exits_two(self) -> None:
        # '-' is only meaningful for the first argument; the second
        # would read an already-drained stdin. Reject early with a
        # clear message instead of bubbling up a downstream JSON
        # decode error.
        runner = CliRunner()
        result = runner.invoke(schema_app, ["diff", str(ON_DISK_SCHEMA), "-"])
        assert result.exit_code == 2, result.output
        assert "stdin" in result.output.lower()


class TestSchemaDiffPure:
    def test_text_diff_identical_inputs(self) -> None:
        x = {"a": 1, "b": [2, 3]}
        assert schema_text_diff(x, x) == ""

    def test_text_diff_divergent_inputs(self) -> None:
        out = schema_text_diff({"a": 1, "b": [2, 3]}, {"a": 1, "b": [2, 4]})
        assert out
        assert any(line.startswith("-") for line in out.splitlines())
        assert any(line.startswith("+") for line in out.splitlines())

    def test_structured_diff_added_and_removed_keys(self) -> None:
        d = schema_structured_diff({"a": 1, "b": 2}, {"a": 1, "c": 3})
        assert d["added"] == ["/c"]
        assert d["removed"] == ["/b"]
        assert d["changed"] == {}

    def test_structured_diff_list_length_mismatch(self) -> None:
        d = schema_structured_diff({"x": [1, 2]}, {"x": [1, 2, 3]})
        assert d["added"] == []
        assert d["removed"] == []
        assert d["changed"]["/x"] == {"before": [1, 2], "after": [1, 2, 3]}


def test_main_cli_routes_validate_and_diff() -> None:
    """Smoke test: ``pypic schema {validate,diff}`` reachable via top app."""
    runner = CliRunner()
    validate_result = runner.invoke(
        pypic_app, ["schema", "validate", str(REFERENCE_TOML)]
    )
    assert validate_result.exit_code == 0, validate_result.output

    diff_result = runner.invoke(pypic_app, ["schema", "diff", str(ON_DISK_SCHEMA)])
    assert diff_result.exit_code == 0, diff_result.output
