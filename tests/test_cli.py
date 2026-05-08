"""Tests for the pypic CLI."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

import h5py
import numpy as np
import pytest

typer = pytest.importorskip("typer")

from typer.testing import CliRunner  # noqa: E402

from pypic.cli import app  # noqa: E402

if TYPE_CHECKING:
    from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

mpl_required = pytest.mark.skipif(not _HAS_MPL, reason="matplotlib required")

runner = CliRunner()

# Minimal simulation.toml for a SimpleReader-compatible dataset (schema v1.0).
_TOML = """\
[schema]
version = "1.0"

[model]
name = "test_sim"
type = "MHD"

[run]
name = "cli_test_run"

[time]
scheme = "fixed"
dt = 0.1
t_start = 0.0
t_end = 1.0
n_steps = 10

[grid]
dimensions = [4, 4, 4]
spacing = [1.0, 1.0, 1.0]
lower = [0.0, 0.0, 0.0]
upper = [4.0, 4.0, 4.0]

[units]
system = "SI"

[coordinates]
geometry = "cartesian"
frame = "simulation"

[physics.mhd]
gamma = 1.6667

[[species]]
name = "p"
charge = 1.0
mass = 1.0
"""


def _make_sim_dir(tmp_path: Path, *, n_steps: int = 1) -> Path:
    """Create a minimal SimpleReader-compatible simulation directory."""
    d = tmp_path / "sim"
    d.mkdir()
    (d / "simulation.toml").write_text(_TOML, encoding="utf-8")

    rng = np.random.default_rng(42)
    shape = (4, 4, 4)
    for i in range(n_steps):
        with h5py.File(d / f"output_{i:06d}.h5", "w") as f:
            grp = f.create_group("fields")
            grp.create_dataset("B1", data=rng.standard_normal(shape))
            grp.create_dataset("B2", data=rng.standard_normal(shape))
            grp.create_dataset("B3", data=rng.standard_normal(shape))
            f.attrs["model"] = "test_sim"
            f.attrs["step"] = i
    return d


# -- version -----------------------------------------------------------------


def test_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "pypic" in result.output


# -- info --------------------------------------------------------------------


def test_info_text(tmp_path):
    d = _make_sim_dir(tmp_path)
    result = runner.invoke(app, ["info", str(d)])
    assert result.exit_code == 0, result.output
    assert "test_sim" in result.output
    assert "4 x 4 x 4" in result.output
    assert "cartesian" in result.output
    assert "Units:" in result.output


def test_info_json(tmp_path):
    d = _make_sim_dir(tmp_path)
    result = runner.invoke(app, ["info", str(d), "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["model_name"] == "test_sim"
    assert data["model_type"] == "MHD"
    assert data["grid"]["dimensions"] == [4, 4, 4]
    assert "steps" in data
    assert "normalization" in data
    assert "length_ref" in data["normalization"]


# -- fields ------------------------------------------------------------------


def test_fields_default(tmp_path):
    d = _make_sim_dir(tmp_path)
    result = runner.invoke(app, ["fields", str(d)])
    assert result.exit_code == 0, result.output
    assert "B1" in result.output
    assert "B2" in result.output
    assert "B3" in result.output


def test_fields_mapping(tmp_path):
    d = _make_sim_dir(tmp_path)
    result = runner.invoke(app, ["fields", str(d), "--mapping"])
    assert result.exit_code == 0, result.output
    # Native on the left, → arrow, canonical on the right
    assert "\u2192" in result.output
    assert "B1" in result.output


def test_fields_mapping_json_preserves_null(tmp_path):
    d = _make_sim_dir(tmp_path)
    result = runner.invoke(app, ["fields", str(d), "--mapping", "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    mapping = data["native_mapping"]
    # Native names should be strings or null, never "(computed)"
    for val in mapping.values():
        assert val is None or isinstance(val, str)
        assert val != "(computed)"


def test_fields_derived(tmp_path):
    d = _make_sim_dir(tmp_path)
    result = runner.invoke(app, ["fields", str(d), "--derived"])
    assert result.exit_code == 0, result.output
    # |B| should be computable from B1, B2, B3
    assert "|B|" in result.output


def test_fields_aux(tmp_path):
    d = _make_sim_dir(tmp_path)
    result = runner.invoke(app, ["fields", str(d), "--aux"])
    assert result.exit_code == 0, result.output
    assert "(none)" in result.output


def test_fields_json(tmp_path):
    d = _make_sim_dir(tmp_path)
    result = runner.invoke(app, ["fields", str(d), "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert "fields" in data
    assert "B1" in data["fields"]


def test_fields_all(tmp_path):
    d = _make_sim_dir(tmp_path)
    result = runner.invoke(app, ["fields", str(d), "--all"])
    assert result.exit_code == 0, result.output
    # --all combines native, derived, aux
    assert "Derived" in result.output
    assert "Auxiliary" in result.output


# -- stats -------------------------------------------------------------------


def test_stats_single_step(tmp_path):
    d = _make_sim_dir(tmp_path)
    result = runner.invoke(app, ["stats", str(d), "--field", "B1"])
    assert result.exit_code == 0, result.output
    assert "min:" in result.output
    assert "max:" in result.output
    assert "mean:" in result.output
    assert "rms:" in result.output
    assert "NaN:" in result.output


def test_stats_derived(tmp_path):
    d = _make_sim_dir(tmp_path)
    result = runner.invoke(app, ["stats", str(d), "--field", "|B|"])
    assert result.exit_code == 0, result.output
    assert "min:" in result.output


def test_stats_json(tmp_path):
    d = _make_sim_dir(tmp_path)
    result = runner.invoke(app, ["stats", str(d), "--field", "B1", "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert "min" in data
    assert "max" in data
    assert "mean" in data
    assert "rms" in data
    assert "nan_count" in data
    assert isinstance(data["min"], float)


def test_stats_multi_step(tmp_path):
    d = _make_sim_dir(tmp_path, n_steps=3)
    result = runner.invoke(app, ["stats", str(d), "--field", "B1", "--step", "all"])
    assert result.exit_code == 0, result.output
    # Field-header + table-header + 3 data rows = 5 non-blank lines.
    # Pinning the exact count catches silent row-duplication / missing-row
    # regressions that the old `>= 4` lower bound would have masked.
    lines = [ln for ln in result.output.strip().splitlines() if ln.strip()]
    assert len(lines) == 5


def test_stats_multi_step_json(tmp_path):
    d = _make_sim_dir(tmp_path, n_steps=3)
    result = runner.invoke(
        app,
        ["stats", str(d), "--field", "B1", "--step", "all", "--json"],
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert "steps" in data
    assert len(data["steps"]) == 3


# -- compare -----------------------------------------------------------------


def test_compare_single_field(tmp_path):
    d = _make_sim_dir(tmp_path)
    result = runner.invoke(app, ["compare", str(d), str(d), "--field", "B1"])
    assert result.exit_code == 0, result.output
    assert "L2 relative error:" in result.output
    assert "L-inf error:" in result.output
    # Grid context included in single-field output
    assert "Grid:" in result.output
    assert "A dims:" in result.output


def test_compare_all_fields(tmp_path):
    d = _make_sim_dir(tmp_path)
    result = runner.invoke(app, ["compare", str(d), str(d)])
    assert result.exit_code == 0, result.output
    assert "B1" in result.output
    assert "B2" in result.output
    # Self-compare must yield exact zeros on every row — guards against a
    # silent drift in the compare pipeline (e.g. spurious float cast,
    # wrong normalization branch, or accidental rtol-based "close enough"
    # masking real non-zero differences).
    for field in ("B1", "B2", "B3"):
        row = next(
            ln for ln in result.output.splitlines() if ln.lstrip().startswith(field)
        )
        # Extract the two numeric columns after the field name.
        parts = row.split()
        assert float(parts[1]) == 0.0, row
        assert float(parts[2]) == 0.0, row


def test_compare_json(tmp_path):
    d = _make_sim_dir(tmp_path)
    result = runner.invoke(
        app,
        ["compare", str(d), str(d), "--field", "B1", "--json"],
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert "l2" in data
    assert "linf" in data
    assert data["field"] == "B1"


# -- error cases -------------------------------------------------------------


def test_bad_step(tmp_path):
    d = _make_sim_dir(tmp_path)
    result = runner.invoke(app, ["fields", str(d), "--step", "banana"])
    assert result.exit_code != 0
    # Error message must name the offending value so users can fix typos.
    # Pinning this guards against a regression where typer.BadParameter is
    # swallowed or the "Invalid --step value {raw!r}" formatting drops the
    # user's input (parse_steps in cli.py:127-134).
    assert "banana" in result.output


def test_step_not_available(tmp_path):
    d = _make_sim_dir(tmp_path)
    # Only step 0 exists
    result = runner.invoke(app, ["fields", str(d), "--step", "999"])
    assert result.exit_code != 0
    assert "not available" in result.output


def test_empty_step_range(tmp_path):
    d = _make_sim_dir(tmp_path)
    # Only step 0 exists; range 10:20 matches nothing
    result = runner.invoke(app, ["fields", str(d), "--step", "10:20"])
    assert result.exit_code != 0
    assert "No available steps" in result.output


def test_parse_steps_resolves_against_provided_list():
    # Aliases must consult the provided list, not field steps.  This is
    # what lets `convert particles` resolve "last" to the last *particle*
    # step even when fields were dumped on a finer cadence.
    from pypic.cli import parse_steps

    field_steps = [0, 50, 100, 150, 200]
    particle_steps = [0, 100]
    assert parse_steps("last", field_steps) == [200]
    assert parse_steps("last", particle_steps) == [100]
    assert parse_steps("first", particle_steps) == [0]
    assert parse_steps("all", particle_steps) == [0, 100]
    assert parse_steps("0:200:50", particle_steps) == [0, 100]


def test_multi_step_rejected_by_fields(tmp_path):
    d = _make_sim_dir(tmp_path, n_steps=3)
    result = runner.invoke(app, ["fields", str(d), "--step", "all"])
    assert result.exit_code != 0
    assert "single step" in result.output


def test_multi_step_rejected_by_compare(tmp_path):
    d = _make_sim_dir(tmp_path, n_steps=3)
    result = runner.invoke(app, ["compare", str(d), str(d), "--step", "all"])
    assert result.exit_code != 0
    assert "single step" in result.output


def test_bad_log_level(tmp_path):
    d = _make_sim_dir(tmp_path)
    result = runner.invoke(app, ["--log-level", "banana", "info", str(d)])
    assert result.exit_code != 0
    assert "Invalid --log-level" in result.output


def test_bad_metric(tmp_path):
    d = _make_sim_dir(tmp_path)
    result = runner.invoke(
        app, ["compare", str(d), str(d), "--field", "B1", "--metric", "oops"]
    )
    assert result.exit_code != 0
    assert "Invalid --metric" in result.output


def test_bad_units(tmp_path):
    d = _make_sim_dir(tmp_path)
    result = runner.invoke(
        app, ["compare", str(d), str(d), "--field", "B1", "--units", "cgs"]
    )
    assert result.exit_code != 0
    assert "Invalid --units" in result.output


def test_bad_nan_policy(tmp_path):
    d = _make_sim_dir(tmp_path)
    result = runner.invoke(
        app,
        ["compare", str(d), str(d), "--field", "B1", "--nan-policy", "ignore"],
    )
    assert result.exit_code != 0
    assert "Invalid --nan-policy" in result.output


def test_log_level_not_sticky(tmp_path):
    """Verify logging config resets between invocations (force=True)."""
    d = _make_sim_dir(tmp_path)
    # First call with debug level
    runner.invoke(app, ["--log-level", "debug", "info", str(d)])
    # Second call with quiet — should not inherit debug
    result = runner.invoke(app, ["-q", "info", str(d)])
    assert result.exit_code == 0, result.output
    assert logging.getLogger().level >= logging.ERROR


def test_missing_field_stats(tmp_path):
    d = _make_sim_dir(tmp_path)
    result = runner.invoke(app, ["stats", str(d), "--field", "nonexistent_field"])
    assert result.exit_code != 0
    assert "Error:" in result.output


def test_missing_field_compare(tmp_path):
    d = _make_sim_dir(tmp_path)
    result = runner.invoke(
        app, ["compare", str(d), str(d), "--field", "nonexistent_field"]
    )
    assert result.exit_code != 0
    assert "Error:" in result.output


def test_quiet_flag(tmp_path):
    d = _make_sim_dir(tmp_path)
    result = runner.invoke(app, ["-q", "info", str(d)])
    assert result.exit_code == 0, result.output


def test_quiet_suppresses_warnings(tmp_path):
    """Verify -q routes warnings.warn through logging and suppresses them."""
    d = tmp_path / "sim_nan"
    d.mkdir()
    (d / "simulation.toml").write_text(_TOML, encoding="utf-8")
    shape = (4, 4, 4)
    data = np.ones(shape)
    data[0, 0, 0] = np.nan  # triggers NaN-omit warning in diagnostics
    with h5py.File(d / "output_000000.h5", "w") as f:
        grp = f.create_group("fields")
        grp.create_dataset("B1", data=data)
        grp.create_dataset("B2", data=data)
        grp.create_dataset("B3", data=data)
        f.attrs["model"] = "test_sim"
        f.attrs["step"] = 0
    # Without -q, compare triggers NaN warnings from diagnostics
    result = runner.invoke(
        app,
        ["-q", "compare", str(d), str(d), "--field", "B1"],
    )
    assert result.exit_code == 0, result.output
    assert "NaN" not in result.output


# -- plot --------------------------------------------------------------------


@mpl_required
class TestPlot:
    def test_minimal(self, tmp_path: Path) -> None:
        d = _make_sim_dir(tmp_path)
        out = str(tmp_path / "out.png")
        result = runner.invoke(app, ["plot", str(d), "--field", "B1", "--output", out])
        assert result.exit_code == 0, result.output
        assert (tmp_path / "out.png").exists()

    def test_plane_xy(self, tmp_path: Path) -> None:
        d = _make_sim_dir(tmp_path)
        out = str(tmp_path / "xy.png")
        result = runner.invoke(
            app, ["plot", str(d), "--field", "B1", "--plane", "xy", "--output", out]
        )
        assert result.exit_code == 0, result.output
        # Guards against "exit 0 but no file written" regressions
        # (e.g. if --plane xy silently swallowed input).
        assert (tmp_path / "xy.png").exists()

    def test_plane_xz(self, tmp_path: Path) -> None:
        d = _make_sim_dir(tmp_path)
        out = str(tmp_path / "xz.png")
        result = runner.invoke(
            app, ["plot", str(d), "--field", "B1", "--plane", "xz", "--output", out]
        )
        assert result.exit_code == 0, result.output
        assert (tmp_path / "xz.png").exists()

    def test_plane_normal_axis_name(self, tmp_path: Path) -> None:
        """--plane accepts a single axis name as the normal (e.g. 'z')."""
        d = _make_sim_dir(tmp_path)
        out = str(tmp_path / "norm.png")
        result = runner.invoke(
            app, ["plot", str(d), "--field", "B1", "--plane", "z", "--output", out]
        )
        assert result.exit_code == 0, result.output
        assert (tmp_path / "norm.png").exists()

    def test_with_index(self, tmp_path: Path) -> None:
        d = _make_sim_dir(tmp_path)
        out = str(tmp_path / "idx.png")
        result = runner.invoke(
            app, ["plot", str(d), "--field", "B1", "--index", "2", "--output", out]
        )
        assert result.exit_code == 0, result.output
        assert (tmp_path / "idx.png").exists()

    def test_with_coord(self, tmp_path: Path) -> None:
        d = _make_sim_dir(tmp_path)
        out = str(tmp_path / "coord.png")
        result = runner.invoke(
            app, ["plot", str(d), "--field", "B1", "--coord", "1.5", "--output", out]
        )
        assert result.exit_code == 0, result.output
        assert (tmp_path / "coord.png").exists()

    def test_index_and_coord_conflict(self, tmp_path: Path) -> None:
        d = _make_sim_dir(tmp_path)
        out = str(tmp_path / "err.png")
        result = runner.invoke(
            app,
            [
                "plot",
                str(d),
                "--field",
                "B1",
                "--index",
                "2",
                "--coord",
                "1.5",
                "--output",
                out,
            ],
        )
        assert result.exit_code != 0
        # Error should name both conflicting flags, not a generic failure —
        # pins the _resolve_index_from_plane guard against silent precedence bugs.
        assert "--index" in result.output
        assert "--coord" in result.output

    def test_derived_field(self, tmp_path: Path) -> None:
        d = _make_sim_dir(tmp_path)
        out = str(tmp_path / "mag.png")
        result = runner.invoke(app, ["plot", str(d), "--field", "|B|", "--output", out])
        assert result.exit_code == 0, result.output
        assert (tmp_path / "mag.png").exists()

    def test_log_scale(self, tmp_path: Path) -> None:
        d = _make_sim_dir(tmp_path)
        out = str(tmp_path / "log.png")
        result = runner.invoke(
            app, ["plot", str(d), "--field", "|B|", "--scale", "log", "--output", out]
        )
        assert result.exit_code == 0, result.output
        assert (tmp_path / "log.png").exists()

    def test_symlog_scale(self, tmp_path: Path) -> None:
        d = _make_sim_dir(tmp_path)
        out = str(tmp_path / "sym.png")
        result = runner.invoke(
            app, ["plot", str(d), "--field", "B1", "--scale", "symlog", "--output", out]
        )
        assert result.exit_code == 0, result.output
        assert (tmp_path / "sym.png").exists()

    def test_symlog_with_linthresh(self, tmp_path: Path) -> None:
        d = _make_sim_dir(tmp_path)
        out = str(tmp_path / "sym_lt.png")
        result = runner.invoke(
            app,
            [
                "plot",
                str(d),
                "--field",
                "B1",
                "--scale",
                "symlog",
                "--linthresh",
                "0.1",
                "--output",
                out,
            ],
        )
        assert result.exit_code == 0, result.output
        assert (tmp_path / "sym_lt.png").exists()

    def test_custom_clim(self, tmp_path: Path) -> None:
        d = _make_sim_dir(tmp_path)
        out = str(tmp_path / "clim.png")
        result = runner.invoke(
            app,
            [
                "plot",
                str(d),
                "--field",
                "B1",
                "--vmin",
                "-1",
                "--vmax",
                "1",
                "--output",
                out,
            ],
        )
        assert result.exit_code == 0, result.output
        assert (tmp_path / "clim.png").exists()

    def test_colormap(self, tmp_path: Path) -> None:
        d = _make_sim_dir(tmp_path)
        out = str(tmp_path / "cmap.png")
        result = runner.invoke(
            app,
            ["plot", str(d), "--field", "B1", "--colormap", "viridis", "--output", out],
        )
        assert result.exit_code == 0, result.output
        assert (tmp_path / "cmap.png").exists()

    def test_dpi(self, tmp_path: Path) -> None:
        d = _make_sim_dir(tmp_path)
        out = str(tmp_path / "dpi.png")
        result = runner.invoke(
            app,
            ["plot", str(d), "--field", "B1", "--dpi", "72", "--output", out],
        )
        assert result.exit_code == 0, result.output
        assert (tmp_path / "dpi.png").exists()

    def test_format_override(self, tmp_path: Path) -> None:
        d = _make_sim_dir(tmp_path)
        out = str(tmp_path / "out.pdf")
        result = runner.invoke(
            app,
            ["plot", str(d), "--field", "B1", "--format", "pdf", "--output", out],
        )
        assert result.exit_code == 0, result.output
        # Pins that --format pdf actually writes the pdf (catches a bug where
        # the format flag is parsed but ignored, silently emitting PNG).
        assert (tmp_path / "out.pdf").exists()
        assert (tmp_path / "out.pdf").read_bytes().startswith(b"%PDF")

    def test_res_downsample(self, tmp_path: Path) -> None:
        d = _make_sim_dir(tmp_path)
        out = str(tmp_path / "lo.png")
        result = runner.invoke(
            app,
            ["plot", str(d), "--field", "B1", "--res", "2x2", "--output", out],
        )
        assert result.exit_code == 0, result.output
        assert (tmp_path / "lo.png").exists()

    def test_batch_steps(self, tmp_path: Path) -> None:
        d = _make_sim_dir(tmp_path, n_steps=3)
        tpl = str(tmp_path / "frames" / "B_{step:06d}.png")
        result = runner.invoke(
            app, ["plot", str(d), "--field", "B1", "--step", "all", "--output", tpl]
        )
        assert result.exit_code == 0, result.output
        for i in range(3):
            assert (tmp_path / "frames" / f"B_{i:06d}.png").exists()

    def test_batch_no_output_error(self, tmp_path: Path) -> None:
        d = _make_sim_dir(tmp_path, n_steps=3)
        result = runner.invoke(app, ["plot", str(d), "--field", "B1", "--step", "all"])
        assert result.exit_code != 0
        # Error must name --output so users know which flag to add —
        # guards against a generic "invalid input" that tells users nothing.
        assert "--output" in result.output

    def test_bad_scale(self, tmp_path: Path) -> None:
        d = _make_sim_dir(tmp_path)
        out = str(tmp_path / "err.png")
        result = runner.invoke(
            app, ["plot", str(d), "--field", "B1", "--scale", "banana", "--output", out]
        )
        assert result.exit_code != 0
        assert "Invalid --scale" in result.output

    def test_bad_plane(self, tmp_path: Path) -> None:
        d = _make_sim_dir(tmp_path)
        out = str(tmp_path / "err.png")
        result = runner.invoke(
            app, ["plot", str(d), "--field", "B1", "--plane", "ab", "--output", out]
        )
        assert result.exit_code != 0
        assert "Invalid --plane" in result.output

    def test_missing_field(self, tmp_path: Path) -> None:
        d = _make_sim_dir(tmp_path)
        out = str(tmp_path / "err.png")
        result = runner.invoke(
            app, ["plot", str(d), "--field", "nonexistent", "--output", out]
        )
        assert result.exit_code != 0
        assert "Error:" in result.output


# -- plot-compare ------------------------------------------------------------


@mpl_required
class TestPlotCompare:
    def test_minimal(self, tmp_path: Path) -> None:
        d = _make_sim_dir(tmp_path)
        out = str(tmp_path / "cmp.png")
        result = runner.invoke(
            app,
            ["plot-compare", str(d), str(d), "--field", "B1", "--output", out],
        )
        assert result.exit_code == 0, result.output
        assert (tmp_path / "cmp.png").exists()

    def test_with_plane(self, tmp_path: Path) -> None:
        d = _make_sim_dir(tmp_path)
        out = str(tmp_path / "cmp_xz.png")
        result = runner.invoke(
            app,
            [
                "plot-compare",
                str(d),
                str(d),
                "--field",
                "B1",
                "--plane",
                "xz",
                "--output",
                out,
            ],
        )
        assert result.exit_code == 0, result.output
        assert (tmp_path / "cmp_xz.png").exists()

    def test_diff_limits(self, tmp_path: Path) -> None:
        d = _make_sim_dir(tmp_path)
        out = str(tmp_path / "cmp_dl.png")
        result = runner.invoke(
            app,
            [
                "plot-compare",
                str(d),
                str(d),
                "--field",
                "B1",
                "--diff-vmin",
                "-0.5",
                "--diff-vmax",
                "0.5",
                "--output",
                out,
            ],
        )
        assert result.exit_code == 0, result.output
        assert (tmp_path / "cmp_dl.png").exists()

    def test_bad_units(self, tmp_path: Path) -> None:
        d = _make_sim_dir(tmp_path)
        out = str(tmp_path / "err.png")
        result = runner.invoke(
            app,
            [
                "plot-compare",
                str(d),
                str(d),
                "--field",
                "B1",
                "--units",
                "cgs",
                "--output",
                out,
            ],
        )
        assert result.exit_code != 0
        assert "Invalid --units" in result.output

    def test_missing_field(self, tmp_path: Path) -> None:
        d = _make_sim_dir(tmp_path)
        out = str(tmp_path / "err.png")
        result = runner.invoke(
            app,
            ["plot-compare", str(d), str(d), "--field", "nonexistent", "--output", out],
        )
        assert result.exit_code != 0
        assert "Error:" in result.output


# -- stats --field all -------------------------------------------------------


def test_stats_field_all(tmp_path: Path) -> None:
    d = _make_sim_dir(tmp_path)
    result = runner.invoke(app, ["stats", str(d), "--field", "all"])
    assert result.exit_code == 0, result.output
    assert "B1" in result.output
    assert "B2" in result.output
    assert "B3" in result.output


def test_stats_field_all_json(tmp_path: Path) -> None:
    d = _make_sim_dir(tmp_path)
    result = runner.invoke(app, ["stats", str(d), "--field", "all", "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert "fields" in data
    assert len(data["fields"]) >= 3
    assert data["fields"][0]["field"] == "B1"


# -- validate ----------------------------------------------------------------


def test_validate_text(tmp_path: Path) -> None:
    d = _make_sim_dir(tmp_path)
    result = runner.invoke(app, ["validate", str(d)])
    assert result.exit_code == 0, result.output
    assert "Validation:" in result.output
    assert "NaN census:" in result.output
    assert "max |div B|:" in result.output
    assert "B energy:" in result.output


def test_validate_json(tmp_path: Path) -> None:
    d = _make_sim_dir(tmp_path)
    result = runner.invoke(app, ["validate", str(d), "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["nan_total"] == 0
    assert data["max_div_b"] is not None
    assert data["b_energy"] is not None
    assert isinstance(data["fields"], list)
    # No auxiliary data in synthetic sim → energy drift is null
    assert data["energy_drift_total_pct"] is None
    assert data["energy_drift_last_pct"] is None


# -- plot --theme ------------------------------------------------------------


@mpl_required
class TestPlotTheme:
    def test_theme_dark(self, tmp_path: Path) -> None:
        d = _make_sim_dir(tmp_path)
        out = str(tmp_path / "dark.png")
        result = runner.invoke(
            app,
            ["plot", str(d), "--field", "B1", "--theme", "dark", "--output", out],
        )
        assert result.exit_code == 0, result.output
        assert (tmp_path / "dark.png").exists()


# -- plot --contour ----------------------------------------------------------


@mpl_required
class TestPlotContour:
    def test_contour_overlay(self, tmp_path: Path) -> None:
        d = _make_sim_dir(tmp_path)
        out = str(tmp_path / "contour.png")
        result = runner.invoke(
            app,
            [
                "plot",
                str(d),
                "--field",
                "B1",
                "--contour",
                "B2",
                "--contour-levels",
                "3",
                "--output",
                out,
            ],
        )
        assert result.exit_code == 0, result.output
        assert (tmp_path / "contour.png").exists()


# -- plot-compare --theme ----------------------------------------------------


@mpl_required
class TestPlotCompareTheme:
    def test_theme(self, tmp_path: Path) -> None:
        d = _make_sim_dir(tmp_path)
        out = str(tmp_path / "cmp_dark.png")
        result = runner.invoke(
            app,
            [
                "plot-compare",
                str(d),
                str(d),
                "--field",
                "B1",
                "--theme",
                "dark",
                "--output",
                out,
            ],
        )
        assert result.exit_code == 0, result.output
        assert (tmp_path / "cmp_dark.png").exists()


# -- plot --animate ----------------------------------------------------------


@mpl_required
class TestPlotAnimate:
    def test_animate_requires_multistep(self, tmp_path: Path) -> None:
        d = _make_sim_dir(tmp_path)
        out = str(tmp_path / "out.png")
        result = runner.invoke(
            app,
            ["plot", str(d), "--field", "B1", "--output", out, "--animate", "out.mp4"],
        )
        assert result.exit_code != 0
        assert "multiple steps" in result.output


# -- 2D dataset edge case ---------------------------------------------------

_TOML_2D = """\
[schema]
version = "1.0"

[model]
name = "test_2d"
type = "MHD"

[run]
name = "cli_2d_test"

[time]
scheme = "fixed"
dt = 0.1
t_start = 0.0
t_end = 1.0
n_steps = 10

[grid]
dimensions = [8, 6]
spacing = [1.0, 1.0]
lower = [0.0, 0.0]
upper = [8.0, 6.0]

[units]
system = "SI"

[coordinates]
geometry = "cartesian"
frame = "simulation"

[physics.mhd]
gamma = 1.6667

[[species]]
name = "p"
charge = 1.0
mass = 1.0
"""


def _make_2d_sim(tmp_path: Path) -> Path:
    d = tmp_path / "sim2d"
    d.mkdir()
    (d / "simulation.toml").write_text(_TOML_2D, encoding="utf-8")
    rng = np.random.default_rng(99)
    shape = (8, 6)
    with h5py.File(d / "output_000000.h5", "w") as f:
        grp = f.create_group("fields")
        grp.create_dataset("B1", data=rng.standard_normal(shape))
        grp.create_dataset("B2", data=rng.standard_normal(shape))
        f.attrs["model"] = "test_2d"
        f.attrs["step"] = 0
    return d


@mpl_required
def test_plot_2d_dataset(tmp_path: Path) -> None:
    """plot works on already-2D data without --plane."""
    d = _make_2d_sim(tmp_path)
    out = str(tmp_path / "2d.png")
    result = runner.invoke(app, ["plot", str(d), "--field", "B1", "--output", out])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "2d.png").exists()


def test_stats_2d_dataset(tmp_path: Path) -> None:
    d = _make_2d_sim(tmp_path)
    result = runner.invoke(app, ["stats", str(d), "--field", "B1"])
    assert result.exit_code == 0, result.output


# -- bad --method / --frame in compare and plot-compare ----------------------


def test_compare_bad_method(tmp_path: Path) -> None:
    d = _make_sim_dir(tmp_path)
    result = runner.invoke(
        app, ["compare", str(d), str(d), "--field", "B1", "--method", "banana"]
    )
    assert result.exit_code != 0
    assert "Error:" in result.output


def test_compare_bad_frame(tmp_path: Path) -> None:
    d = _make_sim_dir(tmp_path)
    result = runner.invoke(
        app, ["compare", str(d), str(d), "--field", "B1", "--frame", "banana"]
    )
    assert result.exit_code != 0
    assert "Error:" in result.output


@mpl_required
def test_plot_compare_bad_method(tmp_path: Path) -> None:
    d = _make_sim_dir(tmp_path)
    out = str(tmp_path / "err.png")
    result = runner.invoke(
        app,
        [
            "plot-compare",
            str(d),
            str(d),
            "--field",
            "B1",
            "--method",
            "banana",
            "--output",
            out,
        ],
    )
    assert result.exit_code != 0
    assert "Error:" in result.output


# -- convert fields ---------------------------------------------------------

zarr_required = pytest.mark.skipif(
    pytest.importorskip("zarr", reason="zarr required") is None, reason="zarr required"
)


@zarr_required
def test_convert_fields_dry_run(tmp_path: Path) -> None:
    d = _make_sim_dir(tmp_path)
    out = tmp_path / "out.zarr"
    result = runner.invoke(
        app,
        ["convert", "fields", str(d), "--output", str(out), "--dry-run"],
    )
    assert result.exit_code == 0, result.output
    assert "Would write" in result.output
    assert not out.exists()


@zarr_required
def test_convert_fields_single_step_round_trip(tmp_path: Path) -> None:
    from pypic.io import from_zarr

    d = _make_sim_dir(tmp_path)
    out = tmp_path / "single.zarr"
    result = runner.invoke(
        app,
        ["convert", "fields", str(d), "--output", str(out), "--step", "0"],
    )
    assert result.exit_code == 0, result.output
    assert out.exists()

    fds = from_zarr(out)
    assert set(fds.field_names()) >= {"B1", "B2", "B3"}
    assert fds["B1"].shape == (4, 4, 4)


@zarr_required
def test_convert_fields_multi_step_timeseries(tmp_path: Path) -> None:
    from pypic.io import from_zarr

    d = _make_sim_dir(tmp_path, n_steps=3)
    out = tmp_path / "ts.zarr"
    result = runner.invoke(
        app,
        ["convert", "fields", str(d), "--output", str(out), "--step", "all"],
    )
    assert result.exit_code == 0, result.output
    fds = from_zarr(out)
    # timeseries: arrays gain a leading time dim of length 3
    assert fds["B1"].shape == (3, 4, 4, 4)


@zarr_required
def test_convert_fields_subset(tmp_path: Path) -> None:
    from pypic.io import from_zarr

    d = _make_sim_dir(tmp_path)
    out = tmp_path / "subset.zarr"
    result = runner.invoke(
        app,
        [
            "convert",
            "fields",
            str(d),
            "--output",
            str(out),
            "--step",
            "0",
            "--fields",
            "B1,B2",
        ],
    )
    assert result.exit_code == 0, result.output
    fds = from_zarr(out)
    assert "B1" in fds.field_names()
    assert "B2" in fds.field_names()
    assert "B3" not in fds.field_names()


@zarr_required
def test_convert_fields_box_crop(tmp_path: Path) -> None:
    from pypic.io import from_zarr

    d = _make_sim_dir(tmp_path)
    out = tmp_path / "cropped.zarr"
    result = runner.invoke(
        app,
        [
            "convert",
            "fields",
            str(d),
            "--output",
            str(out),
            "--step",
            "0",
            "--box",
            "x=0:2,y=0:2",
        ],
    )
    assert result.exit_code == 0, result.output
    fds = from_zarr(out)
    assert fds["B1"].shape == (2, 2, 4)


@zarr_required
def test_convert_fields_bad_backend(tmp_path: Path) -> None:
    d = _make_sim_dir(tmp_path)
    out = tmp_path / "nope.zarr"
    result = runner.invoke(
        app,
        [
            "convert",
            "fields",
            str(d),
            "--output",
            str(out),
            "--backend",
            "lance",
        ],
    )
    assert result.exit_code != 0
    assert "backend" in result.output.lower()


@zarr_required
def test_convert_fields_tag_requires_icechunk(tmp_path: Path) -> None:
    d = _make_sim_dir(tmp_path)
    out = tmp_path / "tagged.zarr"
    result = runner.invoke(
        app,
        [
            "convert",
            "fields",
            str(d),
            "--output",
            str(out),
            "--tag",
            "v1.0",
        ],
    )
    assert result.exit_code != 0
    assert "icechunk" in result.output.lower()


# -- convert particles ------------------------------------------------------

arrow_required = pytest.mark.skipif(
    pytest.importorskip("pyarrow", reason="pyarrow required") is None,
    reason="pyarrow required",
)

from pathlib import Path as _RuntimePath  # noqa: E402

_IPIC3D_FIXTURE = _RuntimePath("tests/data/ipic3d-synthetic/phdf5")


@arrow_required
def test_convert_particles_dry_run(tmp_path: Path) -> None:
    out = tmp_path / "particles"
    result = runner.invoke(
        app,
        [
            "convert",
            "particles",
            str(_IPIC3D_FIXTURE),
            "--output",
            str(out),
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Would write" in result.output
    assert not out.exists()


@arrow_required
def test_convert_particles_round_trip(tmp_path: Path) -> None:
    from pypic.io._parquet import particles_from_dataset

    out = tmp_path / "particles"
    result = runner.invoke(
        app,
        [
            "convert",
            "particles",
            str(_IPIC3D_FIXTURE),
            "--output",
            str(out),
            "--step",
            "0",
        ],
    )
    assert result.exit_code == 0, result.output
    # Both species should have a partition
    assert (out / "step=000000" / "species=species_0").is_dir()
    assert (out / "step=000000" / "species=species_1").is_dir()

    pcl = particles_from_dataset(out, step=0, species="species_0")
    assert pcl.n_particles == 18
    assert pcl.species_charge == -1.0


@arrow_required
def test_convert_particles_species_filter(tmp_path: Path) -> None:
    out = tmp_path / "particles_one"
    result = runner.invoke(
        app,
        [
            "convert",
            "particles",
            str(_IPIC3D_FIXTURE),
            "--output",
            str(out),
            "--step",
            "0",
            "--species",
            "species_1",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (out / "step=000000" / "species=species_1").is_dir()
    assert not (out / "step=000000" / "species=species_0").exists()


@arrow_required
def test_convert_particles_bad_sort_by(tmp_path: Path) -> None:
    out = tmp_path / "nope"
    result = runner.invoke(
        app,
        [
            "convert",
            "particles",
            str(_IPIC3D_FIXTURE),
            "--output",
            str(out),
            "--sort-by",
            "charge",
        ],
    )
    assert result.exit_code != 0
    assert "sort-by" in result.output.lower()


def test_convert_particles_no_particle_output(tmp_path: Path) -> None:
    d = _make_sim_dir(tmp_path)
    out = tmp_path / "empty"
    result = runner.invoke(
        app,
        ["convert", "particles", str(d), "--output", str(out)],
    )
    assert result.exit_code != 0
    assert "no particle output" in result.output.lower()


# -- convert: --plane, --compression, --virtual, --all, particle --box ------


@zarr_required
def test_convert_fields_plane_slice(tmp_path: Path) -> None:
    from pypic.io import from_zarr

    d = _make_sim_dir(tmp_path)
    out = tmp_path / "plane.zarr"
    result = runner.invoke(
        app,
        [
            "convert",
            "fields",
            str(d),
            "--output",
            str(out),
            "--step",
            "0",
            "--plane",
            "xy",
        ],
    )
    assert result.exit_code == 0, result.output
    fds = from_zarr(out)
    # xy plane → normal is z, so the z axis is sliced out
    assert fds["B1"].shape == (4, 4)


@zarr_required
def test_convert_fields_compression_zstd(tmp_path: Path) -> None:
    from pypic.io import from_zarr
    from pypic.readers import open_simulation

    d = _make_sim_dir(tmp_path)
    out = tmp_path / "zstd.zarr"
    result = runner.invoke(
        app,
        [
            "convert",
            "fields",
            str(d),
            "--output",
            str(out),
            "--step",
            "0",
            "--compression",
            "zstd:3",
        ],
    )
    assert result.exit_code == 0, result.output
    fds = from_zarr(out)
    assert "B1" in fds.field_names()
    # Lossless round-trip: zstd is lossless, so the compressed roundtrip
    # must reproduce the source bit-exactly. Pins against a codec-level
    # regression that silently lossy-compresses floats (a real hazard
    # with codec-config typos like missing bit-shuffle).
    src = open_simulation(d).read(step=0, fields=["B1"])
    np.testing.assert_array_equal(np.asarray(fds["B1"]), np.asarray(src["B1"]))


@zarr_required
def test_convert_fields_compression_bad_spec(tmp_path: Path) -> None:
    d = _make_sim_dir(tmp_path)
    out = tmp_path / "bad.zarr"
    result = runner.invoke(
        app,
        [
            "convert",
            "fields",
            str(d),
            "--output",
            str(out),
            "--compression",
            "lz4",
        ],
    )
    assert result.exit_code != 0
    assert "compression" in result.output.lower()


@zarr_required
def test_convert_fields_virtual(tmp_path: Path) -> None:
    pytest.importorskip("virtualizarr")
    pytest.importorskip("icechunk")
    from pypic.io import from_zarr

    # Create a canonical pypic-layout HDF5 file (grid + normalization + fields).
    h5_path = tmp_path / "canonical.h5"
    rng = np.random.default_rng(7)
    with h5py.File(h5_path, "w") as f:
        fields = f.create_group("fields")
        fields.create_dataset("B1", data=rng.standard_normal((4, 4, 4)))
        fields.create_dataset("B2", data=rng.standard_normal((4, 4, 4)))
        fields.create_dataset("B3", data=rng.standard_normal((4, 4, 4)))
        grid = f.create_group("grid")
        grid.attrs["dimensions"] = [4, 4, 4]
        grid.attrs["spacing"] = [1.0, 1.0, 1.0]
        grid.attrs["origin"] = [0.0, 0.0, 0.0]
        grid.attrs["geometry"] = "cartesian"

    out = tmp_path / "virtual.zarr"
    result = runner.invoke(
        app,
        [
            "convert",
            "fields",
            str(h5_path),
            "--output",
            str(out),
            "--virtual",
            "--backend",
            "icechunk",
        ],
    )
    assert result.exit_code == 0, result.output
    fds = from_zarr(out)
    assert set(fds.field_names()) >= {"B1", "B2", "B3"}
    # Virtual refs only — no materialised chunk files in the destination.
    assert not list(out.rglob("B1/c/*"))
    # Virtual refs must resolve to the source data bit-exactly (no
    # silent zero-fill regression, no axis transpose).
    with h5py.File(h5_path, "r") as f:
        np.testing.assert_array_equal(np.asarray(fds["B1"]), f["fields/B1"][...])


@zarr_required
def test_convert_fields_virtual_rejects_directory(tmp_path: Path) -> None:
    d = _make_sim_dir(tmp_path)
    out = tmp_path / "nope.zarr"
    result = runner.invoke(
        app,
        [
            "convert",
            "fields",
            str(d),
            "--output",
            str(out),
            "--virtual",
            "--backend",
            "icechunk",
        ],
    )
    assert result.exit_code != 0
    assert "hdf5 file" in result.output.lower()


def _write_min_h5(path: Path) -> None:
    with h5py.File(path, "w") as f:
        f.create_group("fields").create_dataset("B1", data=np.zeros((4, 4, 4)))
        grid = f.create_group("grid")
        grid.attrs["dimensions"] = [4, 4, 4]
        grid.attrs["spacing"] = [1.0, 1.0, 1.0]
        grid.attrs["origin"] = [0.0, 0.0, 0.0]
        grid.attrs["geometry"] = "cartesian"


@zarr_required
@pytest.mark.parametrize(
    ("flag", "value", "expected"),
    [
        ("--fields", "B1", "--fields"),
        ("--dtype", "float32", "--dtype"),
        ("--compression", "zstd:3", "--compression"),
    ],
)
def test_convert_fields_virtual_rejects_incompatible_flags(
    tmp_path: Path, flag: str, value: str, expected: str
) -> None:
    h5_path = tmp_path / "canonical.h5"
    _write_min_h5(h5_path)
    out = tmp_path / "nope.zarr"
    result = runner.invoke(
        app,
        [
            "convert",
            "fields",
            str(h5_path),
            "--output",
            str(out),
            "--virtual",
            "--backend",
            "icechunk",
            flag,
            value,
        ],
    )
    assert result.exit_code != 0
    assert expected in result.output


@zarr_required
def test_convert_fields_virtual_rejects_zarr_backend(tmp_path: Path) -> None:
    h5_path = tmp_path / "canonical.h5"
    _write_min_h5(h5_path)
    out = tmp_path / "nope.zarr"
    result = runner.invoke(
        app,
        [
            "convert",
            "fields",
            str(h5_path),
            "--output",
            str(out),
            "--virtual",  # backend defaults to "zarr"
        ],
    )
    assert result.exit_code != 0
    assert "icechunk" in result.output.lower()


@zarr_required
def test_convert_fields_virtual_reflects_source_mutations(tmp_path: Path) -> None:
    pytest.importorskip("virtualizarr")
    pytest.importorskip("icechunk")
    from pypic.io import from_zarr

    # The reviewer's reproducer: write virtual refs, mutate the source,
    # confirm the destination read sees the mutation (proving refs
    # resolve at read time, not snapshots taken at write time).
    h5_path = tmp_path / "src.h5"
    with h5py.File(h5_path, "w") as f:
        f.create_group("fields").create_dataset("B1", data=np.full((4, 4, 4), 7.0))
        grid = f.create_group("grid")
        grid.attrs["dimensions"] = [4, 4, 4]
        grid.attrs["spacing"] = [1.0, 1.0, 1.0]
        grid.attrs["origin"] = [0.0, 0.0, 0.0]
        grid.attrs["geometry"] = "cartesian"

    out = tmp_path / "virtual_store"
    result = runner.invoke(
        app,
        [
            "convert",
            "fields",
            str(h5_path),
            "--output",
            str(out),
            "--virtual",
            "--backend",
            "icechunk",
        ],
    )
    assert result.exit_code == 0, result.output

    # Pre-mutation read: original values.
    np.testing.assert_array_equal(np.asarray(from_zarr(out)["B1"]), 7.0)

    # Mutate source HDF5; the virtual store must see the new value.
    with h5py.File(h5_path, "r+") as f:
        f["fields"]["B1"][...] = 99.0
    np.testing.assert_array_equal(np.asarray(from_zarr(out)["B1"]), 99.0)


@zarr_required
def test_convert_fields_virtual_rejects_fields_filter(tmp_path: Path) -> None:
    # Kept for backwards-test-compat with the previous targeted test;
    # now subsumed by the parametrised flag-rejection test above.
    h5_path = tmp_path / "canonical.h5"
    _write_min_h5(h5_path)
    out = tmp_path / "nope.zarr"
    result = runner.invoke(
        app,
        [
            "convert",
            "fields",
            str(h5_path),
            "--output",
            str(out),
            "--virtual",
            "--backend",
            "icechunk",
            "--fields",
            "B1",
        ],
    )
    assert result.exit_code != 0
    assert "--fields" in result.output


@arrow_required
def test_convert_particles_box_crop(tmp_path: Path) -> None:
    from pypic.io._parquet import particles_from_dataset

    out = tmp_path / "cropped"
    # Fixture positions are in [0, 1); crop to a slab in x.
    result = runner.invoke(
        app,
        [
            "convert",
            "particles",
            str(_IPIC3D_FIXTURE),
            "--output",
            str(out),
            "--step",
            "0",
            "--species",
            "species_0",
            "--box",
            "x=0.0:0.4",
        ],
    )
    assert result.exit_code == 0, result.output
    pcl = particles_from_dataset(out, step=0, species="species_0")
    assert pcl.n_particles > 0
    assert pcl.n_particles < 18  # was cropped
    assert pcl.x.max() <= 0.4


@zarr_required
@arrow_required
def test_convert_all_fields_only(tmp_path: Path) -> None:
    # Sim without particle output: `convert all` should write fields only.
    from pypic.io import from_zarr

    d = _make_sim_dir(tmp_path)
    out = tmp_path / "all_out"
    result = runner.invoke(
        app,
        ["convert", "all", str(d), "--output", str(out)],
    )
    assert result.exit_code == 0, result.output
    assert (out / "fields.zarr").is_dir()
    assert not (out / "particles").exists()
    # The fields directory must be a readable zarr store, not just an
    # empty directory — guards against a `convert all` regression that
    # mkdir()s the target but never writes into it.
    fds = from_zarr(out / "fields.zarr")
    assert "B1" in fds.field_names()


@zarr_required
@arrow_required
def test_convert_all_dry_run_particles(tmp_path: Path) -> None:
    out = tmp_path / "plan"
    result = runner.invoke(
        app,
        [
            "convert",
            "all",
            str(_IPIC3D_FIXTURE),
            "--output",
            str(out),
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "fields.zarr" in result.output
    assert "particles" in result.output
    # Reviewer regression: a dry run must not create the output
    # directory — the expected contract is that --dry-run has no
    # filesystem side effects.
    assert not out.exists()
