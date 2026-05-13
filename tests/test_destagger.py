"""Tests for ``pypic._stagger`` — Yee-mesh linear destaggering."""

from __future__ import annotations

import numpy as np
import pytest

from pypic._stagger import (
    destagger_arrays_to_cell_centers,
    destagger_arrays_to_nodes,
)


class TestPassThrough:
    """Arrays already at the target (or absent from the map) pass through."""

    def test_missing_key_returns_input_object(self) -> None:
        arr = np.arange(8, dtype=float).reshape(2, 2, 2)
        out = destagger_arrays_to_cell_centers(
            arrays={"rho_c": arr},
            position_map={},
        )
        assert out["rho_c"] is arr

    def test_already_at_cell_center_returns_input_object(self) -> None:
        arr = np.arange(8, dtype=float).reshape(2, 2, 2)
        out = destagger_arrays_to_cell_centers(
            arrays={"rho_c": arr},
            position_map={"rho_c": (0.5, 0.5, 0.5)},
        )
        assert out["rho_c"] is arr

    def test_already_at_node_returns_input_object(self) -> None:
        arr = np.arange(8, dtype=float).reshape(2, 2, 2)
        out = destagger_arrays_to_nodes(
            arrays={"B_n": arr},
            position_map={"B_n": (0.0, 0.0, 0.0)},
        )
        assert out["B_n"] is arr

    def test_empty_arrays_dict_returns_empty(self) -> None:
        assert destagger_arrays_to_cell_centers({}, {}) == {}

    def test_partial_position_map(self) -> None:
        """Fields not in ``position_map`` pass through unchanged."""
        bx = np.ones((4, 4, 4))
        rho = np.full((4, 4, 4), 3.0)
        out = destagger_arrays_to_cell_centers(
            arrays={"B_1": bx, "rho_c": rho},
            position_map={"B_1": (0.5, 0.0, 0.0)},
        )
        assert out["rho_c"] is rho
        # B_1 shifted along axes 1 and 2 — different object
        assert out["B_1"] is not bx
        assert out["B_1"].shape == (4, 3, 3)


class TestConstantField:
    """Constant fields are reproduced exactly by linear interpolation."""

    def test_constant_destaggered_to_cell_centers(self) -> None:
        arr = np.full((5, 5, 5), 3.14)
        out = destagger_arrays_to_cell_centers(
            arrays={"B_1": arr},
            position_map={"B_1": (0.5, 0.0, 0.0)},
        )
        np.testing.assert_array_equal(out["B_1"], np.full((5, 4, 4), 3.14))

    def test_constant_destaggered_to_nodes(self) -> None:
        arr = np.full((5, 5, 5), -2.71)
        out = destagger_arrays_to_nodes(
            arrays={"E_1": arr},
            position_map={"E_1": (0.0, 0.5, 0.5)},
        )
        np.testing.assert_array_equal(out["E_1"], np.full((5, 4, 4), -2.71))


class TestLinearExactness:
    """Linear fields f(x) = a*x + b are reproduced exactly."""

    def test_linear_in_y_destaggered_along_y(self) -> None:
        # f(x, y) = 3y + 7 sampled on x-face (j = 0..7 → y = j)
        n = 8
        j = np.arange(n)
        row = 3.0 * j + 7.0
        values = np.broadcast_to(row[None, :], (n, n)).copy()
        out = destagger_arrays_to_cell_centers(
            arrays={"B_1": values},
            position_map={"B_1": (0.5, 0.0)},
        )
        # After half-cell shift along axis 1, value at index j is
        # 0.5 * (row[j] + row[j+1]) = 3*(j+0.5) + 7.
        expected_row = 3.0 * (np.arange(n - 1) + 0.5) + 7.0
        expected = np.broadcast_to(expected_row[None, :], (n, n - 1))
        np.testing.assert_allclose(out["B_1"], expected, rtol=1e-14)

    def test_linear_in_x_destaggered_to_nodes(self) -> None:
        # f(x, y) = 2x - 1 sampled at x-face positions (i + 0.5)
        n = 6
        i = np.arange(n)
        col = 2.0 * (i + 0.5) - 1.0
        values = np.broadcast_to(col[:, None], (n, n)).copy()
        out = destagger_arrays_to_nodes(
            arrays={"B_1": values},
            position_map={"B_1": (0.5, 0.0)},
        )
        # Output at node positions x = i + 1: value 2*(i+1) - 1 for i=0..n-2.
        expected_col = 2.0 * (np.arange(n - 1) + 1.0) - 1.0
        expected = np.broadcast_to(expected_col[:, None], (n - 1, n))
        np.testing.assert_allclose(out["B_1"], expected, rtol=1e-14)


class TestShape:
    """Output-shape invariants for various source/target combinations."""

    def test_xface_to_cell_center_shifts_two_axes(self) -> None:
        arr = np.zeros((10, 10, 10))
        out = destagger_arrays_to_cell_centers(
            arrays={"B_1": arr},
            position_map={"B_1": (0.5, 0.0, 0.0)},
        )
        assert out["B_1"].shape == (10, 9, 9)

    def test_xedge_to_cell_center_shifts_one_axis(self) -> None:
        arr = np.zeros((10, 10, 10))
        out = destagger_arrays_to_cell_centers(
            arrays={"E_1": arr},
            position_map={"E_1": (0.5, 0.5, 0.0)},
        )
        assert out["E_1"].shape == (10, 10, 9)

    def test_three_axes_shifted_when_going_face_to_node(self) -> None:
        arr = np.zeros((10, 10, 10))
        out = destagger_arrays_to_nodes(
            arrays={"B_1": arr},
            position_map={"B_1": (0.5, 0.5, 0.5)},
        )
        assert out["B_1"].shape == (9, 9, 9)

    def test_1d_array_passes_through(self) -> None:
        arr = np.array([1.0, 2.0, 3.0, 4.0])
        out = destagger_arrays_to_cell_centers(
            arrays={"B": arr},
            position_map={"B": (0.5,)},
        )
        assert out["B"] is arr

    def test_1d_array_shifted(self) -> None:
        arr = np.array([1.0, 2.0, 3.0, 4.0])
        out = destagger_arrays_to_cell_centers(
            arrays={"B": arr},
            position_map={"B": (0.0,)},
        )
        np.testing.assert_allclose(out["B"], np.array([1.5, 2.5, 3.5]))


class TestValidation:
    """Bad input shapes / offsets fail loud."""

    def test_position_length_mismatch_raises(self) -> None:
        arr = np.zeros((4, 4))
        with pytest.raises(ValueError, match="length 3 but array"):
            destagger_arrays_to_cell_centers(
                arrays={"B_1": arr},
                position_map={"B_1": (0.5, 0.0, 0.0)},
            )

    def test_non_half_cell_offset_raises(self) -> None:
        arr = np.zeros((4, 4, 4))
        with pytest.raises(ValueError, match=r"not in \{0\.0, 0\.5\}"):
            destagger_arrays_to_cell_centers(
                arrays={"B_1": arr},
                position_map={"B_1": (0.25, 0.0, 0.0)},
            )


class TestSecondOrderAccuracy:
    """Destaggered values converge to the analytic cell-centered field at O(dx²)."""

    @staticmethod
    def _max_error(n: int) -> float:
        # Use a non-eigenmode field: cos(x) sin(2y) on x-faces.
        # Single-frequency eigenmodes (e.g. cos(kx) sin(ky) with the
        # canonical Yee div-B-free B-field) accidentally land on
        # machine-precision residuals after destagger+central-diff
        # because every term picks up the same cos(k dx/2) scaling;
        # using mismatched frequencies in x and y breaks that
        # symmetry and exposes the generic O(dx²) error of linear
        # half-cell interpolation.
        length = 2.0 * np.pi
        dx = length / n
        i = np.arange(n)
        x_face, y_node = np.meshgrid((i + 0.5) * dx, i * dx, indexing="ij")
        bx_face = np.cos(x_face) * np.sin(2.0 * y_node)
        out = destagger_arrays_to_cell_centers(
            arrays={"B_1": bx_face},
            position_map={"B_1": (0.5, 0.0)},
        )
        # Output shape (n, n-1) at cell centers (i+0.5)dx, (j+0.5)dx
        # for j = 0..n-2.
        x_cell, y_cell = np.meshgrid((i + 0.5) * dx, (i[:-1] + 0.5) * dx, indexing="ij")
        bx_analytic = np.cos(x_cell) * np.sin(2.0 * y_cell)
        return float(np.max(np.abs(out["B_1"] - bx_analytic)))

    def test_error_is_second_order(self) -> None:
        # Analytic leading-order error of linear interp on sin(k y) at
        # half-cell midpoint: sin(...)(1 - cos(k dy/2)) ≈ k² dy² / 8.
        # For k=2, dy = 2π/n: e(16) ≈ 0.077, e(32) ≈ 0.0193, ratio ≈ 4.
        e16 = self._max_error(16)
        e32 = self._max_error(64)
        assert e16 < 0.10, f"n=16 |err|_∞ = {e16:.4g} unexpectedly large"
        assert e32 < 0.01, f"n=64 |err|_∞ = {e32:.4g} unexpectedly large"
        # n quadruples → second-order error reduces by ~16. Demand ≥ 12
        # for slack against the next-order term.
        ratio = e16 / e32
        assert ratio > 12.0, (
            f"linear destagger not second order: e16={e16:.4g}, "
            f"e64={e32:.4g}, ratio={ratio:.2f}"
        )
