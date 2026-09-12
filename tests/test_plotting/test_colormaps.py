"""Colormap choice, color limits and field-value resolution."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pytest

from pypic.plotting import (
    get_theme,
    plot_comparison,
    plot_field_slice,
    plot_quiver,
    plot_streamlines,
)
from pypic.plotting._colormaps import (
    _auto_linthresh,
    auto_clim,
    is_positive_definite,
    round_nice,
    symmetric_clim,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from matplotlib.axes import Axes

    from pypic.dataset import FieldDataset


class TestColormapDetection:
    @pytest.mark.parametrize(
        ("name", "data_positive", "quantity_type", "expected"),
        [
            ("|B|", False, None, True),
            ("B_1", False, None, False),
            ("rho_m", False, "density", True),
            ("rho_c", False, "charge_density", False),
            ("beta", True, None, True),
            ("unknown", True, None, True),
            ("unknown", False, None, False),
        ],
        ids=[
            "magnitude",
            "component",
            "density",
            "charge-density",
            "beta",
            "all-positive-fallback",
            "mixed-sign-fallback",
        ],
    )
    def test_detection(
        self,
        name: str,
        data_positive: bool,
        quantity_type: str | None,
        expected: bool,
    ) -> None:
        from pypic.fields import FieldInfo

        pos = np.array([1.0, 2.0, 3.0])
        neg = np.array([-1.0, 0.0, 1.0])
        data = pos if data_positive else neg
        info = (
            FieldInfo(quantity_type=quantity_type, long_name="", si_unit="")
            if quantity_type
            else None
        )
        assert is_positive_definite(name, data, info) is expected


class TestDefaultColormap:
    @pytest.mark.parametrize(
        ("plot", "family"),
        [
            pytest.param(
                lambda ds: plot_field_slice(ds, "B_1")[1], "diverging", id="slice"
            ),
            pytest.param(
                lambda ds: plot_field_slice(ds, "rho_m")[1],
                "sequential",
                id="slice-positive",
            ),
            pytest.param(
                lambda ds: plot_streamlines(ds, "B")[1], "sequential", id="streamlines"
            ),
            pytest.param(lambda ds: plot_quiver(ds, "B")[1], "sequential", id="quiver"),
            pytest.param(
                lambda ds: plot_comparison(ds, ds, "B_1")[1]["a"],
                "diverging",
                id="comparison",
            ),
        ],
    )
    def test_follows_the_theme_when_cmap_is_none(
        self,
        ds_2d: FieldDataset,
        plot: Callable[[FieldDataset], Axes],
        family: str,
    ) -> None:
        """Signed fields draw with the theme's diverging map and magnitudes
        with its sequential one, not matplotlib's ``image.cmap`` default."""
        theme = get_theme()
        expected = (
            theme.diverging_cmap if family == "diverging" else theme.sequential_cmap
        )
        assert expected != plt.rcParams["image.cmap"], "theme must differ to test"
        ax = plot(ds_2d)
        assert ax.collections[0].cmap.name == expected
        plt.close("all")


class TestSymmetricClim:
    def test_symmetric(self) -> None:
        assert symmetric_clim(np.array([-3.0, 1.0, 2.0])) == (-3.0, 3.0)

    def test_all_nan(self) -> None:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            vmin, vmax = symmetric_clim(np.array([np.nan, np.nan]))
        assert vmin < 0 < vmax  # epsilon expansion, not degenerate (0, 0)


class TestResolveFieldColormap:
    """Single-source-of-truth dispatcher for field → colormap.

    The matplotlib backend reads the returned name; the pyvista backend
    reads the returned Colormap object. Both backends call this function,
    so the same canonical names always pick the same colormap.
    """

    @pytest.fixture
    def theme(self):  # type: ignore[no-untyped-def]
        from pypic.plotting.styles import get_theme

        return get_theme()

    @pytest.mark.parametrize(
        ("name", "values", "quantity_type", "expect_diverging"),
        [
            ("B_1", np.array([-1.0, 0.0, 1.0]), "b_field", True),
            ("rho_c", np.array([-1.0, 0.0, 1.0]), "charge_density", True),
            ("J_dot_E", np.array([-1.0, 0.0, 1.0]), "power_density", True),
            ("psi", np.array([-1.0, 0.0, 1.0]), None, True),
            ("div_B", np.array([-1.0, 0.0, 1.0]), None, True),
            ("|B|", np.array([0.5, 1.0, 1.5]), None, False),
            ("n_s0", np.array([0.5, 1.0, 1.5]), "density", False),
        ],
        ids=["B_1", "rho_c", "J_dot_E", "psi", "div_B", "abs_B", "n_s0"],
    )
    def test_canonical_field_dispatch(  # type: ignore[no-untyped-def]
        self,
        theme,
        name: str,
        values: np.ndarray,
        quantity_type: str | None,
        expect_diverging: bool,
    ) -> None:
        """Each canonical field name resolves to the right theme cmap."""
        from pypic.fields import FieldInfo
        from pypic.plotting._colormaps import resolve_field_colormap

        info = (
            FieldInfo(quantity_type=quantity_type, long_name="", si_unit="")
            if quantity_type
            else None
        )
        cmap_name, cmap_obj = resolve_field_colormap(name, values, theme, info=info)
        # Returned tuple is internally consistent
        assert cmap_obj.name == cmap_name
        # Picks the right family from the theme
        expected = theme.diverging_cmap if expect_diverging else theme.sequential_cmap
        assert cmap_name == expected

    def test_string_override_passes_through(self, theme) -> None:  # type: ignore[no-untyped-def]
        """A user-supplied cmap name bypasses auto-detection."""
        from pypic.plotting._colormaps import resolve_field_colormap

        cmap_name, cmap_obj = resolve_field_colormap(
            "B_1", np.array([-1.0, 1.0]), theme, cmap="viridis"
        )
        assert cmap_name == "viridis"
        assert cmap_obj.name == "viridis"

    def test_colormap_object_passes_through(self, theme) -> None:  # type: ignore[no-untyped-def]
        """A pre-built Colormap is returned unchanged with its name."""

        from pypic.plotting._colormaps import resolve_field_colormap

        plasma = plt.colormaps["plasma"]
        cmap_name, cmap_obj = resolve_field_colormap(
            "B_1", np.array([-1.0, 1.0]), theme, cmap=plasma
        )
        assert cmap_obj is plasma
        assert cmap_name == "plasma"

    def test_matches_resolve_colormap_string_path(self, theme) -> None:  # type: ignore[no-untyped-def]
        """Returned name matches the legacy string-only resolver."""
        from pypic.plotting._colormaps import resolve_colormap, resolve_field_colormap

        for name, values in [
            ("B_1", np.array([-1.0, 1.0])),
            ("|B|", np.array([0.0, 1.0])),
            ("rho_c", np.array([-1.0, 1.0])),
        ]:
            legacy = resolve_colormap(name, values, theme)
            new_name, _ = resolve_field_colormap(name, values, theme)
            assert legacy == new_name

    def test_uniform_field(self) -> None:
        vmin, vmax = symmetric_clim(np.array([0.0, 0.0, 0.0]))
        assert vmin < 0 < vmax


class TestResolveFieldValues:
    def test_stored_field(self, ds_2d: FieldDataset) -> None:
        from pypic.plotting._resolve import resolve_field_values

        values = resolve_field_values(ds_2d, "B_1", None)
        np.testing.assert_array_equal(values, ds_2d["B_1"])

    def test_derived_field(self, ds_2d: FieldDataset) -> None:
        from pypic.plotting._resolve import resolve_field_values

        values = resolve_field_values(ds_2d, "|B|", None)
        expected = np.sqrt(ds_2d["B_1"] ** 2 + ds_2d["B_2"] ** 2 + ds_2d["B_3"] ** 2)
        np.testing.assert_allclose(values, expected, rtol=1e-15)

    def test_with_units(self, ds_2d: FieldDataset) -> None:
        from pypic.plotting._resolve import resolve_field_values

        code = resolve_field_values(ds_2d, "B_1", None)
        si = resolve_field_values(ds_2d, "B_1", "nT")
        # Shape preserved
        assert si.shape == code.shape
        # Identity normalization: code→SI conversion is 1 T; 1 T = 1e9 nT.
        # Pin the scale so a regression in the unit-conversion path (e.g.
        # forgetting the nT prefix) is caught, not just "something happened".
        np.testing.assert_allclose(si, code * 1.0e9, rtol=1e-12)


class TestRoundNice:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (0.7, 0.5),
            (1.3, 1.0),
            (3.5, 5.0),
            (150.0, 200.0),
            (0.003, 0.002),
            (7.0, 5.0),
            (15.0, 20.0),
            (0.11, 0.1),
        ],
    )
    def test_round_nice_values(self, value: float, expected: float) -> None:
        assert round_nice(value) == pytest.approx(expected)

    def test_zero(self) -> None:
        assert round_nice(0.0) == 0.0


class TestAutoClim:
    def test_signed_symmetric(self) -> None:
        data = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        vmin, vmax = auto_clim(data, positive_definite=False)
        assert vmin == -vmax
        assert vmin < 0

    def test_positive_definite(self) -> None:
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        vmin, vmax = auto_clim(data, positive_definite=True)
        assert vmin == 0.0
        assert vmax > 0

    def test_constant_field(self) -> None:
        data = np.full(100, 5.0)
        vmin, vmax = auto_clim(data, positive_definite=True)
        assert vmin == 0.0
        assert vmax > 0

    def test_all_nan(self) -> None:
        data = np.full(10, np.nan)
        vmin, vmax = auto_clim(data, positive_definite=False)
        assert vmin < 0
        assert vmax > 0

    def test_all_nan_positive(self) -> None:
        data = np.full(10, np.nan)
        vmin, vmax = auto_clim(data, positive_definite=True)
        assert vmin == 0.0
        assert vmax > 0


class TestAutoLinthresh:
    def test_typical_data(self) -> None:
        data = np.array([-5.0, -1.0, 0.0, 1.0, 5.0])
        lt = _auto_linthresh(data)
        assert lt > 0

    def test_all_zero(self) -> None:
        data = np.zeros(10)
        lt = _auto_linthresh(data)
        assert lt == pytest.approx(1e-8)
