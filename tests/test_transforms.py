"""Tests for coordinate frame transforms."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pypic.coordinates.geometry import CARTESIAN
from pypic.coordinates.transforms import (
    FrameTransform,
    compose_transforms,
    find_pressure_tensor_groups,
    find_vector_triplets,
    identity_transform,
    resolve_transform,
    rotate_pressure_tensor,
    rotate_vector_components,
)
from pypic.dataset import FieldDataset
from pypic.grid import GridInfo
from pypic.units import Normalization

# ---------------------------------------------------------------------------
# FrameTransform dataclass
# ---------------------------------------------------------------------------


class TestFrameTransform:
    def test_identity_is_identity(self) -> None:
        t = FrameTransform("a", "a")
        assert t.is_identity

    def test_origin_shift_not_identity(self) -> None:
        t = FrameTransform("a", "b", origin=(1.0, 0.0, 0.0))
        assert not t.is_identity

    def test_rotation_not_identity(self) -> None:
        # 90° rotation around z
        r = ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
        t = FrameTransform("a", "b", rotation=r)
        assert not t.is_identity

    def test_scale_not_identity(self) -> None:
        t = FrameTransform("a", "b", scale=2.0)
        assert not t.is_identity

    def test_invalid_rotation_shape_rejected(self) -> None:
        with pytest.raises(ValueError, match="3×3"):
            FrameTransform("a", "b", rotation=((1.0, 0.0), (0.0, 1.0)))  # type: ignore[arg-type]

    def test_non_orthogonal_rejected(self) -> None:
        with pytest.raises(ValueError, match="not orthogonal"):
            FrameTransform(
                "a",
                "b",
                rotation=((2.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            )

    def test_negative_scale_rejected(self) -> None:
        with pytest.raises(ValueError, match="scale"):
            FrameTransform("a", "b", scale=-1.0)

    def test_rotation_matrix_shape(self) -> None:
        assert FrameTransform("a", "b").rotation_matrix.shape == (3, 3)

    def test_inverse_frames_swapped(self) -> None:
        t = FrameTransform("sim", "GSM", origin=(1.0, 2.0, 3.0))
        inv = t.inverse()
        assert inv.source_frame == "GSM"
        assert inv.target_frame == "sim"

    def test_inverse_round_trip_origin(self) -> None:
        t = FrameTransform("a", "b", origin=(3.0, -1.0, 2.0), scale=2.0)
        inv = t.inverse()
        composed = compose_transforms(t, inv)
        assert_allclose(composed.origin, (0.0, 0.0, 0.0), atol=1e-12)
        assert_allclose(composed.rotation_matrix, np.eye(3), atol=1e-12)
        assert_allclose(composed.scale, 1.0, atol=1e-12)

    def test_inverse_round_trip_rotation(self) -> None:
        # y↔z swap
        r = ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0))
        t = FrameTransform("a", "b", rotation=r, origin=(10.0, 5.0, 3.0))
        composed = compose_transforms(t, t.inverse())
        assert_allclose(composed.rotation_matrix, np.eye(3), atol=1e-12)
        assert_allclose(composed.origin, (0.0, 0.0, 0.0), atol=1e-12)

    def test_target_axis_names(self) -> None:
        t = FrameTransform("sim", "GSM", target_axis_names=("x_GSM", "y_GSM", "z_GSM"))
        assert t.target_axis_names == ("x_GSM", "y_GSM", "z_GSM")


class TestIdentityTransform:
    def test_returns_identity(self) -> None:
        t = identity_transform("sim")
        assert t.is_identity
        assert t.source_frame == "sim"
        assert t.target_frame == "sim"


# ---------------------------------------------------------------------------
# Vector rotation
# ---------------------------------------------------------------------------


class TestRotateVectorComponents:
    def test_identity_preserves(self) -> None:
        v1 = np.array([1.0, 2.0])
        v2 = np.array([3.0, 4.0])
        v3 = np.array([5.0, 6.0])
        r1, r2, r3 = rotate_vector_components(v1, v2, v3, np.eye(3))
        assert_allclose(r1, v1)
        assert_allclose(r2, v2)
        assert_allclose(r3, v3)

    def test_90_degree_rotation_around_z(self) -> None:
        # R_z(90°): x→y, y→-x, z→z
        R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        v1 = np.array([1.0])
        v2 = np.array([0.0])
        v3 = np.array([0.0])
        r1, r2, r3 = rotate_vector_components(v1, v2, v3, R)
        assert_allclose(r1, [0.0], atol=1e-15)
        assert_allclose(r2, [1.0], atol=1e-15)
        assert_allclose(r3, [0.0], atol=1e-15)

    def test_yz_swap(self) -> None:
        # Swap y↔z: x→x, y→z, z→y
        R = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=float)
        v1 = np.array([1.0])
        v2 = np.array([2.0])
        v3 = np.array([3.0])
        r1, r2, r3 = rotate_vector_components(v1, v2, v3, R)
        assert_allclose(r1, [1.0])
        assert_allclose(r2, [3.0])
        assert_allclose(r3, [2.0])

    def test_magnitude_preserved(self) -> None:
        rng = np.random.default_rng(42)
        v1 = rng.standard_normal(10)
        v2 = rng.standard_normal(10)
        v3 = rng.standard_normal(10)
        # Arbitrary rotation
        theta = 0.7
        R = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1],
            ]
        )
        r1, r2, r3 = rotate_vector_components(v1, v2, v3, R)
        mag_before = v1**2 + v2**2 + v3**2
        mag_after = r1**2 + r2**2 + r3**2
        assert_allclose(mag_after, mag_before, rtol=1e-14)

    def test_3d_array_broadcast(self) -> None:
        shape = (4, 3, 2)
        v1 = np.ones(shape)
        v2 = np.zeros(shape)
        v3 = np.zeros(shape)
        R = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=float)
        r1, r2, _r3 = rotate_vector_components(v1, v2, v3, R)
        assert r1.shape == shape
        assert_allclose(r1, 0.0, atol=1e-15)
        assert_allclose(r2, -1.0, atol=1e-15)


# ---------------------------------------------------------------------------
# Pressure tensor rotation
# ---------------------------------------------------------------------------


class TestRotatePressureTensor:
    def test_identity_preserves(self) -> None:
        p11 = np.array([1.0])
        p22 = np.array([2.0])
        p33 = np.array([3.0])
        p12 = np.array([0.5])
        p13 = np.array([0.1])
        p23 = np.array([0.2])
        result = rotate_pressure_tensor(p11, p22, p33, p12, p13, p23, np.eye(3))
        assert_allclose(result[0], p11)
        assert_allclose(result[1], p22)
        assert_allclose(result[2], p33)
        assert_allclose(result[3], p12)
        assert_allclose(result[4], p13)
        assert_allclose(result[5], p23)

    def test_trace_invariant(self) -> None:
        rng = np.random.default_rng(99)
        p11 = rng.standard_normal(5)
        p22 = rng.standard_normal(5)
        p33 = rng.standard_normal(5)
        p12 = rng.standard_normal(5)
        p13 = rng.standard_normal(5)
        p23 = rng.standard_normal(5)
        # Arbitrary rotation
        theta = 1.3
        R = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1],
            ]
        )
        rp = rotate_pressure_tensor(p11, p22, p33, p12, p13, p23, R)
        trace_before = p11 + p22 + p33
        trace_after = rp[0] + rp[1] + rp[2]
        assert_allclose(trace_after, trace_before, rtol=1e-13)

    def test_yz_swap(self) -> None:
        # Swap y↔z: P_22↔P_33, P_12↔P_13, P_23 unchanged
        R = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=float)
        p11 = np.array([1.0])
        p22 = np.array([2.0])
        p33 = np.array([3.0])
        p12 = np.array([0.4])
        p13 = np.array([0.5])
        p23 = np.array([0.6])
        rp = rotate_pressure_tensor(p11, p22, p33, p12, p13, p23, R)
        assert_allclose(rp[0], [1.0], atol=1e-14)  # P_11 → P_11
        assert_allclose(rp[1], [3.0], atol=1e-14)  # P_22 → old P_33
        assert_allclose(rp[2], [2.0], atol=1e-14)  # P_33 → old P_22
        assert_allclose(rp[3], [0.5], atol=1e-14)  # P_12 → old P_13
        assert_allclose(rp[4], [0.4], atol=1e-14)  # P_13 → old P_12
        assert_allclose(rp[5], [0.6], atol=1e-14)  # P_23 → old P_23


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------


class TestComposeTransforms:
    def test_two_translations(self) -> None:
        a = FrameTransform("A", "B", origin=(1.0, 0.0, 0.0))
        b = FrameTransform("B", "C", origin=(0.0, 2.0, 0.0))
        c = compose_transforms(a, b)
        assert_allclose(c.origin, (1.0, 2.0, 0.0), atol=1e-14)

    def test_two_rotations(self) -> None:
        # 90° around z twice = 180° around z
        R90 = ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
        a = FrameTransform("A", "B", rotation=R90)
        b = FrameTransform("B", "C", rotation=R90)
        c = compose_transforms(a, b)
        expected = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=float)
        assert_allclose(c.rotation_matrix, expected, atol=1e-14)

    def test_frame_mismatch_raises(self) -> None:
        a = FrameTransform("A", "B")
        c = FrameTransform("C", "D")
        with pytest.raises(ValueError, match="Cannot compose"):
            compose_transforms(a, c)

    def test_scale_composition(self) -> None:
        a = FrameTransform("A", "B", scale=2.0)
        b = FrameTransform("B", "C", scale=3.0)
        c = compose_transforms(a, b)
        assert_allclose(c.scale, 6.0)

    def test_compose_inverse_is_identity(self) -> None:
        r = ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0))
        t = FrameTransform("A", "B", origin=(5.0, 3.0, 1.0), rotation=r, scale=2.0)
        composed = compose_transforms(t, t.inverse())
        assert_allclose(composed.origin, (0.0, 0.0, 0.0), atol=1e-12)
        assert_allclose(composed.rotation_matrix, np.eye(3), atol=1e-12)
        assert_allclose(composed.scale, 1.0, atol=1e-12)


# ---------------------------------------------------------------------------
# Resolve transform
# ---------------------------------------------------------------------------


class TestResolveTransform:
    def test_same_frame_returns_identity(self) -> None:
        t = resolve_transform("sim", "sim", {})
        assert t.is_identity

    def test_direct_lookup(self) -> None:
        t = FrameTransform("sim", "GSM", origin=(1.0, 0.0, 0.0))
        result = resolve_transform("sim", "GSM", {"GSM": t})
        assert result.target_frame == "GSM"
        assert_allclose(result.origin, (1.0, 0.0, 0.0))

    def test_inverse_lookup(self) -> None:
        t = FrameTransform("sim", "GSM", origin=(1.0, 0.0, 0.0))
        result = resolve_transform("GSM", "sim", {"GSM": t})
        assert result.source_frame == "GSM"
        assert result.target_frame == "sim"

    def test_one_hop_chain(self) -> None:
        t1 = FrameTransform("sim", "GSM", origin=(1.0, 0.0, 0.0))
        t2 = FrameTransform("GSM", "GSE", origin=(0.0, 2.0, 0.0))
        result = resolve_transform("sim", "GSE", {"GSM": t1, "GSE": t2})
        assert result.source_frame == "sim"
        assert result.target_frame == "GSE"

    def test_no_path_raises(self) -> None:
        t = FrameTransform("X", "Y")
        with pytest.raises(ValueError, match="No transform path"):
            resolve_transform("sim", "GSM", {"Y": t})


# ---------------------------------------------------------------------------
# Field detection
# ---------------------------------------------------------------------------


class TestFindVectorTriplets:
    def test_basic_b_field(self) -> None:
        result = find_vector_triplets(["B_1", "B_2", "B_3", "rho_c"])
        assert result == [("B_1", "B_2", "B_3")]

    def test_per_species(self) -> None:
        fields = ["J_s0_1", "J_s0_2", "J_s0_3", "J_s1_1", "J_s1_2", "J_s1_3"]
        result = find_vector_triplets(fields)
        assert len(result) == 2
        assert ("J_s0_1", "J_s0_2", "J_s0_3") in result
        assert ("J_s1_1", "J_s1_2", "J_s1_3") in result

    def test_incomplete_triplet_excluded(self) -> None:
        result = find_vector_triplets(["B_1", "B_2"])
        assert result == []

    def test_mixed_fields(self) -> None:
        fields = ["B_1", "B_2", "B_3", "E_1", "E_2", "E_3", "rho_c", "P"]
        result = find_vector_triplets(fields)
        assert len(result) == 2

    def test_b0_split_field(self) -> None:
        # Canonical split-B components per schema.md § "Split-B naming"
        # are B0_1/B0_2/B0_3 (underscore separator because the B0 prefix
        # ends in a digit). The triplet regex must match these so that
        # FieldDataset.transform_to() rotates the background field.
        result = find_vector_triplets(["B0_1", "B0_2", "B0_3"])
        assert result == [("B0_1", "B0_2", "B0_3")]

    def test_b0_triplet_matched_in_mixed_list(self) -> None:
        # Regression guard for the silent BATSRUS physics bug: frame
        # rotations used to skip B0_1/B0_2/B0_3 because the regex only
        # matched a phantom B01/B02/B03 form that no reader emits.
        result = find_vector_triplets(["B_1", "B_2", "B_3", "B0_1", "B0_2", "B0_3"])
        assert set(result) == {
            ("B_1", "B_2", "B_3"),
            ("B0_1", "B0_2", "B0_3"),
        }


class TestFindPressureTensorGroups:
    def test_complete_tensor(self) -> None:
        fields = ["P_11", "P_22", "P_33", "P_12", "P_13", "P_23"]
        result = find_pressure_tensor_groups(fields)
        assert len(result) == 1
        assert result[0] == ("P_11", "P_22", "P_33", "P_12", "P_13", "P_23")

    def test_per_species(self) -> None:
        # Tier-3 per-species tensor: ``P_s<N>_<ij>``.
        fields = [f"P_s0_{ij}" for ij in ("11", "22", "33", "12", "13", "23")]
        fields += [f"P_s1_{ij}" for ij in ("11", "22", "33", "12", "13", "23")]
        result = find_pressure_tensor_groups(fields)
        assert len(result) == 2

    def test_incomplete_excluded(self) -> None:
        result = find_pressure_tensor_groups(["P_11", "P_22", "P_33"])
        assert result == []


# ---------------------------------------------------------------------------
# FieldDataset.transform_to integration
# ---------------------------------------------------------------------------

_YZ_SWAP: FrameTransform = FrameTransform(
    "sim",
    "GSM",
    rotation=((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
    target_axis_names=("x_GSM", "y_GSM", "z_GSM"),
)


def _make_dataset(
    fields: dict[str, np.ndarray],
    *,
    transforms: dict[str, FrameTransform] | None = None,
) -> FieldDataset:
    grid = GridInfo(
        dimensions=(4, 3, 2),
        spacing=(1.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
        geometry=CARTESIAN,
    )
    return FieldDataset.from_arrays(
        fields,
        grid,
        Normalization.identity(),
        transforms=transforms,
        frame="sim",
    )


class TestFieldDatasetTransformTo:
    def test_same_frame_returns_self(self) -> None:
        ds = _make_dataset({"B_1": np.ones((4, 3, 2))}, transforms={"GSM": _YZ_SWAP})
        result = ds.transform_to("sim")
        assert result is ds

    def test_no_transforms_raises(self) -> None:
        ds = _make_dataset({"B_1": np.ones((4, 3, 2))})
        with pytest.raises(KeyError, match="No transforms"):
            ds.transform_to("GSM")

    def test_unknown_target_raises(self) -> None:
        ds = _make_dataset({"B_1": np.ones((4, 3, 2))}, transforms={"GSM": _YZ_SWAP})
        with pytest.raises(ValueError, match="No transform path"):
            ds.transform_to("UNKNOWN")

    def test_vector_fields_rotated(self) -> None:
        fields = {
            "B_1": np.full((4, 3, 2), 1.0),
            "B_2": np.full((4, 3, 2), 2.0),
            "B_3": np.full((4, 3, 2), 3.0),
        }
        ds = _make_dataset(fields, transforms={"GSM": _YZ_SWAP})
        result = ds.transform_to("GSM")
        # y↔z swap: B_1→B_1, B_2→B3_old=3, B_3→B2_old=2
        assert_allclose(result["B_1"], 1.0)
        assert_allclose(result["B_2"], 3.0)
        assert_allclose(result["B_3"], 2.0)

    def test_scalar_fields_unchanged(self) -> None:
        fields = {
            "B_1": np.ones((4, 3, 2)),
            "B_2": np.ones((4, 3, 2)),
            "B_3": np.ones((4, 3, 2)),
            "rho_c": np.full((4, 3, 2), 42.0),
        }
        ds = _make_dataset(fields, transforms={"GSM": _YZ_SWAP})
        result = ds.transform_to("GSM")
        assert_allclose(result["rho_c"], 42.0)

    def test_frame_label_updated(self) -> None:
        ds = _make_dataset(
            {
                "B_1": np.ones((4, 3, 2)),
                "B_2": np.ones((4, 3, 2)),
                "B_3": np.ones((4, 3, 2)),
            },
            transforms={"GSM": _YZ_SWAP},
        )
        result = ds.transform_to("GSM")
        assert result.frame == "GSM"

    def test_axis_names_updated(self) -> None:
        ds = _make_dataset(
            {
                "B_1": np.ones((4, 3, 2)),
                "B_2": np.ones((4, 3, 2)),
                "B_3": np.ones((4, 3, 2)),
            },
            transforms={"GSM": _YZ_SWAP},
        )
        result = ds.transform_to("GSM")
        assert result.grid.geometry.axis_names == ("x_GSM", "y_GSM", "z_GSM")

    def test_grid_origin_shifted(self) -> None:
        t = FrameTransform("sim", "GSM", origin=(10.0, 5.0, 3.0))
        ds = _make_dataset(
            {
                "B_1": np.ones((4, 3, 2)),
                "B_2": np.ones((4, 3, 2)),
                "B_3": np.ones((4, 3, 2)),
            },
            transforms={"GSM": t},
        )
        result = ds.transform_to("GSM")
        # origin was (0,0,0), shifted by -(10,5,3) = (-10,-5,-3)
        assert_allclose(result.grid.origin[0], -10.0, atol=1e-12)
        assert_allclose(result.grid.origin[1], -5.0, atol=1e-12)
        assert_allclose(result.grid.origin[2], -3.0, atol=1e-12)

    def test_grid_spacing_scaled(self) -> None:
        t = FrameTransform("sim", "GSM", scale=2.0)
        ds = _make_dataset(
            {
                "B_1": np.ones((4, 3, 2)),
                "B_2": np.ones((4, 3, 2)),
                "B_3": np.ones((4, 3, 2)),
            },
            transforms={"GSM": t},
        )
        result = ds.transform_to("GSM")
        assert_allclose(result.grid.spacing, (2.0, 2.0, 2.0))

    def test_per_species_vectors_rotated(self) -> None:
        fields = {
            "J_s0_1": np.full((4, 3, 2), 1.0),
            "J_s0_2": np.full((4, 3, 2), 2.0),
            "J_s0_3": np.full((4, 3, 2), 3.0),
        }
        ds = _make_dataset(fields, transforms={"GSM": _YZ_SWAP})
        result = ds.transform_to("GSM")
        assert_allclose(result["J_s0_1"], 1.0)
        assert_allclose(result["J_s0_2"], 3.0)
        assert_allclose(result["J_s0_3"], 2.0)

    def test_pressure_tensor_rotated(self) -> None:
        fields = {
            "P_11": np.full((4, 3, 2), 1.0),
            "P_22": np.full((4, 3, 2), 2.0),
            "P_33": np.full((4, 3, 2), 3.0),
            "P_12": np.full((4, 3, 2), 0.0),
            "P_13": np.full((4, 3, 2), 0.0),
            "P_23": np.full((4, 3, 2), 0.0),
        }
        ds = _make_dataset(fields, transforms={"GSM": _YZ_SWAP})
        result = ds.transform_to("GSM")
        # y↔z swap on diagonal tensor: P_22↔P_33
        assert_allclose(result["P_11"], 1.0)
        assert_allclose(result["P_22"], 3.0)
        assert_allclose(result["P_33"], 2.0)

    def test_normalization_preserved(self) -> None:
        ds = _make_dataset(
            {
                "B_1": np.ones((4, 3, 2)),
                "B_2": np.ones((4, 3, 2)),
                "B_3": np.ones((4, 3, 2)),
            },
            transforms={"GSM": _YZ_SWAP},
        )
        result = ds.transform_to("GSM")
        assert result.normalization is ds.normalization

    def test_general_rotation_raises(self) -> None:
        """Non-permutation rotations require interpolation, not implemented."""
        theta = np.pi / 4  # 45° around z
        r = (
            (float(np.cos(theta)), float(-np.sin(theta)), 0.0),
            (float(np.sin(theta)), float(np.cos(theta)), 0.0),
            (0.0, 0.0, 1.0),
        )
        t = FrameTransform("sim", "rotated", rotation=r)
        ds = _make_dataset(
            {
                "B_1": np.ones((4, 3, 2)),
                "B_2": np.ones((4, 3, 2)),
                "B_3": np.ones((4, 3, 2)),
            },
            transforms={"rotated": t},
        )
        with pytest.raises(NotImplementedError, match="signed permutation"):
            ds.transform_to(t)

    def test_round_trip(self) -> None:
        rng = np.random.default_rng(42)
        fields = {
            "B_1": rng.standard_normal((4, 3, 2)),
            "B_2": rng.standard_normal((4, 3, 2)),
            "B_3": rng.standard_normal((4, 3, 2)),
        }
        ds = _make_dataset(fields, transforms={"GSM": _YZ_SWAP})
        result = ds.transform_to("GSM").transform_to("sim")
        assert_allclose(result["B_1"], fields["B_1"], atol=1e-14)
        assert_allclose(result["B_2"], fields["B_2"], atol=1e-14)
        assert_allclose(result["B_3"], fields["B_3"], atol=1e-14)


def test_available_frames_lists_every_reachable_frame() -> None:
    """``available_frames`` reports the native frame plus both ends of each transform.

    A public property with no coverage: it is the only way to ask a
    dataset what ``transform_to`` will accept.
    """
    grid = GridInfo(dimensions=(2, 2, 2), spacing=(1.0, 1.0, 1.0))
    fields = {"B_1": np.zeros((2, 2, 2))}
    ds = FieldDataset.from_arrays(
        fields,
        grid,
        frame="simulation",
        transforms={
            "GSM": FrameTransform(source_frame="simulation", target_frame="GSM"),
            "GSE": FrameTransform(source_frame="GSM", target_frame="GSE"),
        },
    )
    assert ds.available_frames == ["GSE", "GSM", "simulation"]
