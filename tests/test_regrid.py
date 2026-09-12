"""Tests for :mod:`pypic.regrid`."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from pypic.coordinates.geometry import SPHERICAL
from pypic.dataset import FieldDataset
from pypic.grid import GridInfo
from pypic.regrid import align_grids, common_grid, regrid
from pypic.units import Normalization, PhysicsParams, SpeciesInfo
from tests._helpers import make_uniform_grid

# ---------------------------------------------------------------------------
# common_grid
# ---------------------------------------------------------------------------


class TestCommonGrid:
    """Tests for :func:`common_grid`."""

    def test_identical_grids(self) -> None:
        g = make_uniform_grid(10, 8, spacing=1.0, origin=0.0)
        cg = common_grid(g, g)
        assert cg.dimensions == g.dimensions
        assert cg.spacing == g.spacing
        assert cg.origin == g.origin

    def test_finer_spacing_selected(self) -> None:
        g1 = make_uniform_grid(10, spacing=1.0)
        g2 = make_uniform_grid(20, spacing=0.5)
        cg = common_grid(g1, g2)
        assert cg.spacing == (0.5,)

    def test_intersection_domain(self) -> None:
        g1 = make_uniform_grid(10, spacing=1.0, origin=0.0)
        g2 = make_uniform_grid(10, spacing=1.0, origin=5.0)
        cg = common_grid(g1, g2)
        assert cg.origin == (5.0,)
        # extent of g1 = 10, extent of g2 = 15 → hi = 10
        extent = cg.origin[0] + cg.dimensions[0] * cg.spacing[0]
        assert extent <= 10.0

    def test_no_overlap_raises(self) -> None:
        g1 = make_uniform_grid(5, spacing=1.0, origin=0.0)
        g2 = make_uniform_grid(5, spacing=1.0, origin=10.0)
        with pytest.raises(ValueError, match="do not overlap"):
            common_grid(g1, g2)

    def test_touching_grids_raises(self) -> None:
        g1 = make_uniform_grid(5, spacing=1.0, origin=0.0)
        g2 = make_uniform_grid(5, spacing=1.0, origin=5.0)
        with pytest.raises(ValueError, match="do not overlap"):
            common_grid(g1, g2)

    def test_2d_asymmetric(self) -> None:
        g1 = make_uniform_grid(10, 8, spacing=(1.0, 0.5), origin=(0.0, 0.0))
        g2 = make_uniform_grid(6, 12, spacing=(0.5, 1.0), origin=(2.0, 1.0))
        cg = common_grid(g1, g2)
        assert cg.spacing == (0.5, 0.5)
        # Origin is the *grid* origin (cell-edge); the first sample sits
        # at origin + 0.5*dx and equals the per-axis sample-range overlap.
        # Axis 0: g1 samples [0.5..9.5], g2 samples [2.25..4.75]
        #         → first common sample = max(0.5, 2.25) = 2.25
        #         → cg.origin = 2.25 - 0.25 = 2.0
        # Axis 1: g1 samples [0.25..3.75], g2 samples [1.5..11.5]
        #         → first common sample = max(0.25, 1.5) = 1.5
        #         → cg.origin = 1.5 - 0.25 = 1.25
        assert cg.origin == (2.0, 1.25)
        # All cg samples must lie inside both source sample ranges.
        cg_samples = cg.coordinate_arrays()
        g1_samples = g1.coordinate_arrays()
        g2_samples = g2.coordinate_arrays()
        for i in range(2):
            assert cg_samples[i].min() >= g1_samples[i].min() - 1e-12
            assert cg_samples[i].max() <= g1_samples[i].max() + 1e-12
            assert cg_samples[i].min() >= g2_samples[i].min() - 1e-12
            assert cg_samples[i].max() <= g2_samples[i].max() + 1e-12

    def test_ndim_mismatch_raises(self) -> None:
        g1 = make_uniform_grid(10)
        g2 = make_uniform_grid(10, 10)
        with pytest.raises(ValueError, match="dimensionality"):
            common_grid(g1, g2)

    def test_spherical_raises(self) -> None:
        g1 = make_uniform_grid(10)
        g2 = GridInfo(dimensions=(10,), spacing=(1.0,), geometry=SPHERICAL)
        with pytest.raises(NotImplementedError, match="spherical"):
            common_grid(g1, g2)


# ---------------------------------------------------------------------------
# regrid
# ---------------------------------------------------------------------------


def _make_1d_dataset(
    n: int, dx: float, origin: float, func: object = None
) -> FieldDataset:
    """1D dataset with f(x) = 2x + 1 by default."""
    grid = make_uniform_grid(n, spacing=dx, origin=origin)
    (x,) = grid.coordinate_arrays()
    values = 2.0 * x + 1.0 if func is None else func(x)
    return FieldDataset.from_arrays(
        {"f": values}, grid, Normalization.identity(), strict_fields=False
    )


class TestRegrid:
    """Tests for :func:`regrid`."""

    def test_noop_same_grid(self) -> None:
        grid = make_uniform_grid(8, spacing=0.5)
        ds = FieldDataset.from_arrays(
            {"B_1": np.ones(8)}, grid, Normalization.identity()
        )
        result = regrid(ds, ds.grid)
        assert result is ds

    def test_noop_equal_grid(self) -> None:
        grid_a = make_uniform_grid(8, spacing=0.5)
        grid_b = make_uniform_grid(8, spacing=0.5)
        data = np.arange(8, dtype=float)
        ds = FieldDataset.from_arrays({"B_1": data}, grid_a, Normalization.identity())
        result = regrid(ds, grid_b)
        # Equal grids trigger the no-op shortcut.
        assert result is ds

    def test_noop_rejects_bad_method(self) -> None:
        """Same-grid shortcut still validates *method*.

        Regression: the no-op path used to return ``source`` unchanged
        before ``method`` reached ``RegularGridInterpolator``, so typos
        silently succeeded on the smoke-test path while the live path
        correctly rejected them.
        """
        grid = make_uniform_grid(8, spacing=0.5)
        ds = FieldDataset.from_arrays(
            {"B_1": np.ones(8)}, grid, Normalization.identity()
        )
        with pytest.raises(ValueError, match="Unknown interpolation method"):
            regrid(ds, ds.grid, method="not_a_real_method")

    def test_live_path_rejects_bad_method(self) -> None:
        """Cross-grid path rejects the same bad *method* as the no-op path."""
        coarse = make_uniform_grid(8, spacing=1.0)
        fine = make_uniform_grid(16, spacing=0.5)
        ds = FieldDataset.from_arrays(
            {"B_1": np.ones(8)}, coarse, Normalization.identity()
        )
        with pytest.raises(ValueError, match="Unknown interpolation method"):
            regrid(ds, fine, method="not_a_real_method")

    def test_linear_1d_exact(self) -> None:
        """Linear interpolation of a linear function is exact."""
        # Coarse: 10 cells, dx=1.0 → centers [0.5, 9.5].
        # Fine grid interior to coarse center range.
        coarse = make_uniform_grid(10, spacing=1.0, origin=0.0)
        ds = FieldDataset.from_arrays(
            {"f": 2.0 * coarse.coordinate_arrays()[0] + 1.0},
            coarse,
            Normalization.identity(),
            strict_fields=False,
        )
        fine_grid = make_uniform_grid(16, spacing=0.5, origin=0.5)
        result = regrid(ds, fine_grid)
        (x_fine,) = fine_grid.coordinate_arrays()
        expected = 2.0 * x_fine + 1.0
        assert_allclose(result["f"], expected, atol=1e-12)

    def test_bilinear_2d_exact(self) -> None:
        """Bilinear interpolation of f(x,y) = x + 2y is exact."""
        # Coarse: 6×6, dx=2.0 → centers [1,3,5,7,9,11] per axis.
        # Fine: interior, dx=1.0 → centers within [1, 11].
        coarse = make_uniform_grid(6, 6, spacing=2.0)
        cx, cy = np.meshgrid(*coarse.coordinate_arrays(), indexing="ij")
        ds = FieldDataset.from_arrays(
            {"f": cx + 2.0 * cy},
            coarse,
            Normalization.identity(),
            strict_fields=False,
        )
        fine = make_uniform_grid(8, 8, spacing=1.0, origin=1.0)
        result = regrid(ds, fine)
        fx, fy = np.meshgrid(*fine.coordinate_arrays(), indexing="ij")
        expected = fx + 2.0 * fy
        assert_allclose(result["f"], expected, atol=1e-12)

    def test_trilinear_3d_exact(self) -> None:
        """Trilinear interpolation of f(x,y,z) = x + y + z is exact."""
        # Coarse: 6³, dx=2.0 → centers [1,3,5,7,9,11].
        # Fine: interior, dx=1.0 → centers within [1, 11].
        coarse = make_uniform_grid(6, 6, 6, spacing=2.0)
        cx, cy, cz = np.meshgrid(*coarse.coordinate_arrays(), indexing="ij")
        ds = FieldDataset.from_arrays(
            {"f": cx + cy + cz},
            coarse,
            Normalization.identity(),
            strict_fields=False,
        )
        fine = make_uniform_grid(8, 8, 8, spacing=1.0, origin=1.0)
        result = regrid(ds, fine)
        fx, fy, fz = np.meshgrid(*fine.coordinate_arrays(), indexing="ij")
        expected = fx + fy + fz
        assert_allclose(result["f"], expected, atol=1e-12)

    def test_output_shape_and_constant_preserved(self) -> None:
        """Regridding a constant field preserves both shape and value."""
        coarse = make_uniform_grid(4, 6, spacing=1.0)
        ds = FieldDataset.from_arrays(
            {"B_1": np.ones((4, 6))}, coarse, Normalization.identity()
        )
        target = make_uniform_grid(8, 12, spacing=0.5)
        result = regrid(ds, target)
        assert result["B_1"].shape == (8, 12)
        # A constant field must remain constant after linear interpolation.
        finite = np.isfinite(result["B_1"])
        assert_allclose(result["B_1"][finite], 1.0, atol=1e-14)

    def test_nan_outside_domain(self) -> None:
        """Points outside source domain are NaN."""
        ds = _make_1d_dataset(5, dx=1.0, origin=0.0)
        # Target extends beyond source (source extent = 5.0).
        wide_grid = make_uniform_grid(20, spacing=1.0, origin=-5.0)
        result = regrid(ds, wide_grid)
        (x,) = wide_grid.coordinate_arrays()
        # Source cell centers: 0.5, 1.5, 2.5, 3.5, 4.5
        # Points below 0.5 or above 4.5 are outside interpolation range.
        outside = (x < 0.5) | (x > 4.5)
        assert np.all(np.isnan(result["f"][outside]))

    def test_partial_overlap(self) -> None:
        """Overlapping region has valid values, non-overlapping is NaN."""
        ds = _make_1d_dataset(10, dx=1.0, origin=0.0)
        target = make_uniform_grid(20, spacing=1.0, origin=5.0)
        result = regrid(ds, target)
        (x_t,) = target.coordinate_arrays()
        # Source cell centers: 0.5..9.5 → interpolation valid in [0.5, 9.5]
        inside = (x_t >= 0.5) & (x_t <= 9.5)
        assert not np.any(np.isnan(result["f"][inside]))
        outside = x_t > 9.5
        assert np.all(np.isnan(result["f"][outside]))

    def test_all_fields_regridded(self) -> None:
        coarse = make_uniform_grid(4, spacing=1.0)
        ds = FieldDataset.from_arrays(
            {"B_1": np.ones(4), "B_2": 2.0 * np.ones(4), "rho_m": np.zeros(4)},
            coarse,
            Normalization.identity(),
        )
        fine = make_uniform_grid(8, spacing=0.5)
        result = regrid(ds, fine)
        assert sorted(result.field_names()) == sorted(ds.field_names())
        # Constant fields must survive regridding with their values intact.
        finite_b2 = np.isfinite(result["B_2"])
        assert_allclose(result["B_2"][finite_b2], 2.0, atol=1e-14)

    def test_multifield_matches_independent_single_field(self) -> None:
        """Stacked multi-field regrid matches per-field regrids exactly.

        Regression guard for the P2 optimization that stacks all fields
        into a single RegularGridInterpolator call. Each field in a
        joint regrid must equal the result of regridding that field
        alone on the same grids.
        """
        coarse = make_uniform_grid(6, 5, 4, spacing=(1.0, 1.0, 1.0))
        cx, cy, cz = np.meshgrid(*coarse.coordinate_arrays(), indexing="ij")
        fields = {
            "B_1": cx + 0.1 * cy,
            "B_2": 2.0 * cy - cz,
            "B_3": cx * cy - cz**2,
            "rho_m": np.exp(-0.05 * (cx - 3.0) ** 2),
        }
        norm = Normalization.identity()
        joint = FieldDataset.from_arrays(fields, coarse, norm)
        fine = make_uniform_grid(10, 8, 6, spacing=(0.5, 0.6, 0.6), origin=1.0)
        joint_result = regrid(joint, fine)
        for name, values in fields.items():
            solo = FieldDataset.from_arrays({name: values}, coarse, norm)
            solo_result = regrid(solo, fine)
            assert_allclose(
                joint_result[name],
                solo_result[name],
                atol=0.0,
                rtol=0.0,
                err_msg=f"Mismatch on field {name!r}",
            )

    def test_metadata_preserved(self) -> None:
        species = [
            SpeciesInfo(name="electrons", charge=-1.0, mass=1 / 256),
            SpeciesInfo(name="ions", charge=1.0, mass=1.0),
        ]
        physics = PhysicsParams(gamma=5 / 3, c=1.0)
        norm = Normalization.identity()
        coarse = make_uniform_grid(4, spacing=1.0)
        ds = FieldDataset.from_arrays(
            {"B_1": np.ones(4)},
            coarse,
            norm,
            species=species,
            physics=physics,
            frame="GSM",
        )
        fine = make_uniform_grid(8, spacing=0.5)
        result = regrid(ds, fine)
        assert result.normalization == norm
        assert result.species == tuple(species)
        assert result.physics == physics
        assert result.frame == "GSM"

    def test_method_nearest(self) -> None:
        """Nearest-neighbor produces piecewise constant output."""
        # Coarse: 4 cells, dx=2.0 → centers [1, 3, 5, 7].
        # Fine grid interior to coarse center range.
        coarse = make_uniform_grid(4, spacing=2.0)
        (cx,) = coarse.coordinate_arrays()
        ds = FieldDataset.from_arrays(
            {"f": cx}, coarse, Normalization.identity(), strict_fields=False
        )
        fine = make_uniform_grid(6, spacing=1.0, origin=0.5)
        result = regrid(ds, fine, method="nearest")
        # Fine centers: 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 — all in [1, 7].
        (fx,) = fine.coordinate_arrays()
        expected = cx[np.argmin(np.abs(cx[:, None] - fx[None, :]), axis=0)]
        assert_array_equal(result["f"], expected)

    def test_fill_value_kwarg(self) -> None:
        ds = _make_1d_dataset(5, dx=1.0, origin=0.0)
        wide = make_uniform_grid(20, spacing=1.0, origin=-5.0)
        result = regrid(ds, wide, fill_value=0.0)
        (x,) = wide.coordinate_arrays()
        outside = (x < 0.5) | (x > 4.5)
        assert_array_equal(result["f"][outside], 0.0)

    def test_spherical_raises(self) -> None:
        grid_s = GridInfo(
            dimensions=(4, 4, 4),
            spacing=(1.0, 1.0, 1.0),
            geometry=SPHERICAL,
        )
        ds = FieldDataset.from_arrays(
            {"B_1": np.ones((4, 4, 4))}, grid_s, Normalization.identity()
        )
        target = make_uniform_grid(8, 8, 8, spacing=0.5)
        with pytest.raises(NotImplementedError, match="spherical"):
            regrid(ds, target)

    def test_ndim_mismatch_raises(self) -> None:
        ds_2d = FieldDataset.from_arrays(
            {"f": np.ones((4, 4))},
            make_uniform_grid(4, 4),
            Normalization.identity(),
            strict_fields=False,
        )
        target_3d = make_uniform_grid(4, 4, 4)
        with pytest.raises(ValueError, match=r"2D source.*3D target"):
            regrid(ds_2d, target_3d)

    def test_empty_dataset_preserves_grid(self) -> None:
        """A FieldDataset with no fields regrids to a new empty dataset."""
        coarse = make_uniform_grid(4, spacing=1.0)
        ds = FieldDataset.from_arrays({}, coarse, Normalization.identity())
        fine = make_uniform_grid(8, spacing=0.5)
        result = regrid(ds, fine)
        assert result.field_names() == []
        assert result.grid.dimensions == (8,)
        assert result.grid.spacing == (0.5,)

    def test_single_cell_grid_trivial(self) -> None:
        """Single-cell source + single-cell target is a trivial passthrough."""
        grid = make_uniform_grid(1, spacing=1.0, origin=0.0)
        ds = FieldDataset.from_arrays(
            {"f": np.array([3.14])},
            grid,
            Normalization.identity(),
            strict_fields=False,
        )
        # Target grid equals source — no-op shortcut fires.
        result = regrid(ds, grid)
        assert result is ds

    def test_fields_subset_selects_requested(self) -> None:
        """``fields=`` regrids only the named subset."""
        coarse = make_uniform_grid(4, spacing=1.0)
        ds = FieldDataset.from_arrays(
            {"B_1": np.ones(4), "B_2": 2.0 * np.ones(4), "rho_m": np.zeros(4)},
            coarse,
            Normalization.identity(),
        )
        fine = make_uniform_grid(8, spacing=0.5)
        result = regrid(ds, fine, fields=["B_1", "rho_m"])
        assert sorted(result.field_names()) == ["B_1", "rho_m"]
        assert result["B_1"].shape == (8,)

    def test_fields_subset_by_alias(self) -> None:
        """``fields=`` resolves aliases to canonical names."""
        coarse = make_uniform_grid(4, 4, 4, spacing=1.0)
        ds = FieldDataset.from_arrays(
            {"B_1": np.ones((4, 4, 4)), "B_2": np.ones((4, 4, 4))},
            coarse,
            Normalization.identity(),
        )
        fine = make_uniform_grid(8, 8, 8, spacing=0.5)
        # "Bx" is the Cartesian alias for "B_1".
        result = regrid(ds, fine, fields=["Bx"])
        assert result.field_names() == ["B_1"]

    def test_fields_subset_unknown_raises(self) -> None:
        coarse = make_uniform_grid(4, spacing=1.0)
        ds = FieldDataset.from_arrays(
            {"B_1": np.ones(4)}, coarse, Normalization.identity()
        )
        fine = make_uniform_grid(8, spacing=0.5)
        with pytest.raises(KeyError):
            regrid(ds, fine, fields=["not_a_field"])

    def test_fields_subset_matches_full_on_shared_names(self) -> None:
        """Regridding a subset produces the same values as the full regrid."""
        coarse = make_uniform_grid(6, 5, spacing=(1.0, 1.0))
        cx, cy = np.meshgrid(*coarse.coordinate_arrays(), indexing="ij")
        ds = FieldDataset.from_arrays(
            {
                "B_1": cx + 0.1 * cy,
                "B_2": 2.0 * cy - cx,
                "rho_m": np.exp(-0.05 * (cx - 3.0) ** 2),
            },
            coarse,
            Normalization.identity(),
        )
        fine = make_uniform_grid(10, 8, spacing=(0.5, 0.6), origin=(1.0, 1.0))
        full = regrid(ds, fine)
        subset = regrid(ds, fine, fields=["B_1", "rho_m"])
        assert_allclose(subset["B_1"], full["B_1"], atol=0.0, rtol=0.0)
        assert_allclose(subset["rho_m"], full["rho_m"], atol=0.0, rtol=0.0)

    def test_noop_shortcut_with_subset_still_filters(self) -> None:
        """Same-grid regrid with a subset must still drop unrequested fields."""
        grid = make_uniform_grid(4, spacing=1.0)
        ds = FieldDataset.from_arrays(
            {"B_1": np.ones(4), "B_2": 2.0 * np.ones(4)},
            grid,
            Normalization.identity(),
        )
        result = regrid(ds, ds.grid, fields=["B_1"])
        # Identical-grid shortcut must not fire when the caller asked
        # for a strict subset — otherwise B_2 would leak through.
        assert result.field_names() == ["B_1"]


# ---------------------------------------------------------------------------
# align_grids
# ---------------------------------------------------------------------------


class TestAlignGrids:
    """Tests for :func:`align_grids`."""

    def test_same_grid_noop(self) -> None:
        grid = make_uniform_grid(8, spacing=1.0)
        ds_a = FieldDataset.from_arrays(
            {"B_1": np.ones(8)}, grid, Normalization.identity()
        )
        ds_b = FieldDataset.from_arrays(
            {"B_1": 2.0 * np.ones(8)}, grid, Normalization.identity()
        )
        ra, rb = align_grids(ds_a, ds_b)
        assert ra is ds_a
        assert rb is ds_b

    def test_different_resolutions(self) -> None:
        """Coarse + fine on the same domain → both at fine resolution."""
        coarse = make_uniform_grid(5, spacing=2.0)
        fine = make_uniform_grid(10, spacing=1.0)
        (cx,) = coarse.coordinate_arrays()
        (fx,) = fine.coordinate_arrays()
        ds_c = FieldDataset.from_arrays(
            {"f": 2.0 * cx + 1.0},
            coarse,
            Normalization.identity(),
            strict_fields=False,
        )
        ds_f = FieldDataset.from_arrays(
            {"f": 3.0 * fx - 1.0},
            fine,
            Normalization.identity(),
            strict_fields=False,
        )
        ra, rb = align_grids(ds_c, ds_f)
        assert ra.grid.spacing == rb.grid.spacing
        assert ra.grid.dimensions == rb.grid.dimensions

    def test_partial_overlap(self) -> None:
        g1 = make_uniform_grid(10, spacing=1.0, origin=0.0)
        g2 = make_uniform_grid(10, spacing=1.0, origin=5.0)
        ds1 = FieldDataset.from_arrays(
            {"f": np.ones(10)}, g1, Normalization.identity(), strict_fields=False
        )
        ds2 = FieldDataset.from_arrays(
            {"f": np.ones(10)}, g2, Normalization.identity(), strict_fields=False
        )
        ra, rb = align_grids(ds1, ds2)
        # Both on the common grid.
        assert ra.grid == rb.grid
        # Common grid covers intersection only.
        assert ra.grid.origin[0] >= 5.0

    def test_symmetric_grid(self) -> None:
        """``align_grids(a, b)`` and ``align_grids(b, a)`` use the same grid."""
        g1 = make_uniform_grid(10, spacing=1.0, origin=0.0)
        g2 = make_uniform_grid(15, spacing=0.5, origin=3.0)
        ds1 = FieldDataset.from_arrays(
            {"f": np.ones(10)}, g1, Normalization.identity(), strict_fields=False
        )
        ds2 = FieldDataset.from_arrays(
            {"f": np.ones(15)}, g2, Normalization.identity(), strict_fields=False
        )
        ra_ab, _rb_ab = align_grids(ds1, ds2)
        ra_ba, _rb_ba = align_grids(ds2, ds1)
        assert ra_ab.grid == ra_ba.grid

    def test_fields_subset_forwarded(self) -> None:
        """``fields=`` forwards to both regrid calls."""
        g1 = make_uniform_grid(6, 6, spacing=1.0)
        g2 = make_uniform_grid(12, 12, spacing=0.5)
        a = FieldDataset.from_arrays(
            {
                "B_1": np.ones((6, 6)),
                "B_2": 2.0 * np.ones((6, 6)),
                "rho_m": np.zeros((6, 6)),
            },
            g1,
            Normalization.identity(),
        )
        b = FieldDataset.from_arrays(
            {
                "B_1": np.ones((12, 12)),
                "B_2": 2.0 * np.ones((12, 12)),
                "rho_m": np.zeros((12, 12)),
            },
            g2,
            Normalization.identity(),
        )
        ra, rb = align_grids(a, b, fields=["B_1"])
        assert ra.field_names() == ["B_1"]
        assert rb.field_names() == ["B_1"]
