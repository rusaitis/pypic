"""Theme lookup, the default theme, and themes loaded from files."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import matplotlib
import matplotlib.pyplot as plt
import pytest

from pypic.plotting import (
    PlotTheme,
    get_theme,
    load_theme,
    plot_field_slice,
    save_theme,
    set_theme,
    use_theme,
)
from pypic.plotting._theme_io import _bundled_theme_dir
from pypic.plotting.styles import _resolve_theme_arg

if TYPE_CHECKING:
    from pathlib import Path

    from pypic.dataset import FieldDataset

# Fields a theme file does not carry: identity and rcparams are written
# specially, the scale factors live only in memory.
_NOT_IN_FILE_TABLE = frozenset(
    {
        "name",
        "rcparams",
        "colorbar_title_font_scale",
        "colorbar_tick_font_scale",
        "badge_font_scale",
        "axis_triad_font_scale",
        "grid_label_font_scale",
        "figsize_per_col",
        "figsize_per_row",
        "panel_label_scale_sparse",
        "panel_label_scale_dense",
        "contour_label_fontsize",
        "annotation_fontsize",
    }
)


def _off_default(value: object) -> object:
    match value:
        case bool():
            return not value
        case float():
            return value + 0.5
        case str():
            return value + "-x"
        case (float(), float(), float(), float()):
            return (0.125, 0.25, 0.375, 0.5)
        case tuple():
            return ("alpha", "beta")
    raise AssertionError(value)


class TestThemes:
    @pytest.mark.parametrize("name", ["light", "dark"])
    def test_rcparams_valid(self, name: str) -> None:
        theme = _resolve_theme_arg(name)
        valid_keys = set(matplotlib.rcParams)
        for key in theme.rcparams:
            assert key in valid_keys, f"{key!r} not a valid rcParam"

    def test_default_is_light(self) -> None:
        assert get_theme().name == "light"

    def test_customize_rcparam(self) -> None:
        light = get_theme()
        big = light.customize(font_size=14)
        assert big.rcparams["font.size"] == 14
        assert big.sequential_cmap == light.sequential_cmap

    def test_customize_field(self) -> None:
        dark = _resolve_theme_arg("dark")
        custom = dark.customize(
            sequential_cmaps=("viridis",),
            grid_color=(0.5, 0.5, 0.5, 0.1),
        )
        assert custom.sequential_cmap == "viridis"
        assert custom.grid_color == (0.5, 0.5, 0.5, 0.1)
        assert custom.diverging_cmap == dark.diverging_cmap

    def test_customize_preserves_original(self) -> None:
        light = get_theme()
        original_size = light.rcparams["font.size"]
        _ = light.customize(font_size=20)
        assert light.rcparams["font.size"] == original_size


class TestSetDefaultTheme:
    def test_set_and_get(self) -> None:
        original = get_theme()
        try:
            set_theme("dark")
            assert get_theme().name == "dark"
        finally:
            set_theme(original)

    def test_badge_on_slice(self, ds_2d: FieldDataset) -> None:
        """Passing *step* to ``badge=True`` writes that step number into
        the badge text (not just any patch)."""
        fig, ax = plot_field_slice(ds_2d, "B_1", step=42, badge=True)
        # The badge is a patch-bearing artist with a child text carrying
        # the formatted cycle/step string.
        patch_artists = [a for a in ax.artists if hasattr(a, "patch")]
        assert len(patch_artists) >= 1
        all_text = " ".join(t.get_text() for t in ax.texts)
        child_text = " ".join(
            t.get_text()
            for a in patch_artists
            for t in getattr(a, "get_children", lambda: [])()
            if hasattr(t, "get_text")
        )
        combined = all_text + " " + child_text
        assert "42" in combined, (
            f"expected step=42 in badge, got texts={all_text!r}, "
            f"child_texts={child_text!r}"
        )
        plt.close(fig)

    def test_save_creates_file(self, ds_2d: FieldDataset, tmp_path: Path) -> None:
        """``save=<path>`` writes a non-empty PNG with the PNG signature."""
        out = tmp_path / "test.png"
        plot_field_slice(ds_2d, "B_1", save=str(out))
        assert out.exists()
        # PNG magic bytes — proves a real image was written, not an empty file.
        assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
        assert out.stat().st_size > 500  # pcolormesh output is never this small


class TestFileThemes:
    """Theme export, discovery, and file-based lookup."""

    def test_export_creates_files(self, tmp_path: Path) -> None:
        from pypic.plotting import export_themes
        from pypic.plotting._theme_io import _bundled_theme_dir

        result = export_themes(tmp_path)
        assert result == tmp_path
        bundled = {p.name for p in _bundled_theme_dir().glob("*.toml")}
        assert {f.name for f in tmp_path.glob("*.toml")} == bundled

    def test_export_no_overwrite(self, tmp_path: Path) -> None:
        from pypic.plotting import export_themes

        export_themes(tmp_path)
        (tmp_path / "dark.toml").write_text('name = "modified"\n')
        export_themes(tmp_path, overwrite=False)
        assert "modified" in (tmp_path / "dark.toml").read_text()

    def test_export_overwrite(self, tmp_path: Path) -> None:
        from pypic.plotting import export_themes

        export_themes(tmp_path)
        (tmp_path / "dark.toml").write_text('name = "modified"\n')
        export_themes(tmp_path, overwrite=True)
        assert "modified" not in (tmp_path / "dark.toml").read_text()

    def test_available_themes(self) -> None:
        from pypic.plotting import available_themes

        themes = available_themes()
        assert "dark" in themes
        assert "light" in themes
        assert len(themes) >= 5

    def test_available_themes_skips_malformed(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from pypic.plotting import available_themes, export_themes

        export_themes(tmp_path)
        (tmp_path / "bad.toml").write_text("not valid {{{toml")
        monkeypatch.setenv("PYPIC_THEME_DIR", str(tmp_path))
        themes = available_themes()
        assert "bad" not in themes

    def test_user_theme_overrides_bundled(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from pypic.plotting import export_themes
        from pypic.plotting.styles import _resolve_theme_arg

        export_themes(tmp_path)
        text = (tmp_path / "dark.toml").read_text()
        (tmp_path / "dark.toml").write_text(text.replace("#f39c12", "#ff0000"))

        monkeypatch.setenv("PYPIC_THEME_DIR", str(tmp_path))
        assert _resolve_theme_arg("dark").accent_color == "#ff0000"

    def test_resolve_by_file_path(self, tmp_path: Path) -> None:
        from pypic.plotting import export_themes
        from pypic.plotting.styles import _resolve_theme_arg

        export_themes(tmp_path)
        theme = _resolve_theme_arg(str(tmp_path / "dark.toml"))
        assert theme.name == "dark"

    def test_custom_theme_from_user_dir(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from pypic.plotting import save_theme

        dark = _resolve_theme_arg("dark")
        custom = dark.customize(name="space-purple", accent_color="#9b59b6")
        save_theme(custom, tmp_path / "space-purple.toml")

        monkeypatch.setenv("PYPIC_THEME_DIR", str(tmp_path))
        resolved = _resolve_theme_arg("space-purple")
        assert resolved.accent_color == "#9b59b6"

    def test_bundled_themes_round_trip_through_save(self, tmp_path: Path) -> None:
        """load → save → load reproduces every field of every bundled theme."""
        mismatched = []
        for path in sorted(_bundled_theme_dir().glob("*.toml")):
            theme = load_theme(path)
            save_theme(theme, tmp_path / path.name)
            again = load_theme(tmp_path / path.name)
            mismatched += [
                f"{path.stem}.{f.name}"
                for f in dataclasses.fields(PlotTheme)
                if getattr(again, f.name) != getattr(theme, f.name)
            ]
        assert mismatched == []

    def test_every_file_field_survives_save_and_load(self, tmp_path: Path) -> None:
        """Every field outside the in-memory set, moved off its default, is
        written and read back: a field either side drops fails here."""
        overrides = {
            f.name: _off_default(f.default)
            for f in dataclasses.fields(PlotTheme)
            if f.name not in _NOT_IN_FILE_TABLE
        }
        save_theme(get_theme().customize(**overrides), tmp_path / "every.toml")
        again = load_theme(tmp_path / "every.toml")
        lost = sorted(k for k, v in overrides.items() if getattr(again, k) != v)
        assert lost == []

    def test_mistyped_value_names_its_key(self, tmp_path: Path) -> None:
        path = tmp_path / "mistyped.toml"
        path.write_text('name = "mistyped"\n[axes]\narrows = "false"\n')
        with pytest.raises(ValueError, match=r"\[axes\] arrows"):
            load_theme(path)

    def test_minimal_theme_falls_back_to_defaults(self, tmp_path: Path) -> None:
        """Only ``name`` and ``[colors]`` background and text are required;
        every other field takes its `PlotTheme` default."""
        path = tmp_path / "minimal.toml"
        path.write_text(
            'name = "minimal"\n[colors]\nbackground = "#000000"\ntext = "#ffffff"\n'
        )
        theme = load_theme(path)
        reference = PlotTheme(name="minimal", rcparams={})
        differing = [
            f.name
            for f in dataclasses.fields(PlotTheme)
            if f.name not in {"rcparams", "text_color"}
            and getattr(theme, f.name) != getattr(reference, f.name)
        ]
        assert differing == []

    def test_use_theme_with_string(self) -> None:
        original = matplotlib.rcParams["text.color"]
        with use_theme("dark"):
            assert matplotlib.rcParams["text.color"] != original
        assert matplotlib.rcParams["text.color"] == original

    def test_bundled_themes_have_webpic_section(self) -> None:
        """Every bundled theme carries a parseable ``[webpic]`` block.

        Cross-tool contract: webpic reads ``[webpic]`` (panel layout,
        shortcuts, diagnostics) plus the shared ``[colors]``/
        ``[colormaps]``/``[font]`` for visual identity. A missing or
        mis-versioned section silently degrades webpic's defaults.
        """
        import tomllib
        from pathlib import Path as _Path

        from pypic.plotting._theme_io import _bundled_theme_dir

        problems: list[str] = []
        for path in sorted(_Path(_bundled_theme_dir()).glob("*.toml")):
            raw = tomllib.loads(path.read_text())
            webpic = raw.get("webpic")
            if not isinstance(webpic, dict):
                problems.append(f"{path.name}: missing [webpic]")
                continue
            if webpic.get("version") != 1:
                got = webpic.get("version")
                problems.append(f"{path.name}: webpic.version = {got!r}, expected 1")
                continue
            for required in ("layout", "shortcuts", "diagnostics", "embed"):
                if required not in webpic:
                    problems.append(f"{path.name}: missing [webpic.{required}]")
        assert not problems, "themes failed [webpic] invariant: " + "; ".join(problems)
