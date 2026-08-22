# Source: docs/schema.md § 3 ("For Cartesian data, letter aliases
#         (Bx, By, Bz) are preferred for readability and access the
#         same data as B_1, B_2, B_3") + docs/schema.md § "Split-B naming"
#         ("B0_1, B0_2, B0_3 (aliases: B0x, B0y, B0z)") +
#         src/pypic/grid.py:148-159 (_FIELD_PREFIX_PAIRS enumerates every
#         vector prefix that must have Cartesian aliases registered).
# Claim: for every registered vector prefix, the Cartesian alias
#        (prefix + x/y/z) must return bit-exact the same array as the
#        numbered canonical (prefix + 1/2/3). The canonical-side name
#        pattern differs per prefix: "B" uses "B_1", "B0" uses "B0_1"
#        (schema.md "The background field prefix B0 ends in a digit,
#        so its components use an underscore separator").
# Regression guard for a previously fixed alias-registration bug.
"""Cartesian letter alias resolves to the numbered canonical, bit-exact."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from numpy.testing import assert_array_equal

from pypic.coordinates.geometry import CARTESIAN
from pypic.dataset import FieldDataset
from pypic.grid import GridInfo
from pypic.units import Normalization

SHAPE = (3, 4)

# (alias_prefix, canonical_pattern) — canonical_pattern uses "{i}" for the
# component index. All Tier-3 canonicals use the underscore separator.
_PREFIX_CASES = [
    ("B", "B_{i}"),
    ("B0", "B0_{i}"),
    ("E", "E_{i}"),
    ("J", "J_{i}"),
    ("V", "V_{i}"),
    ("S", "S_{i}"),
    ("u", "u_{i}"),
    ("EF", "EF_{i}"),
]


def _bounded_array() -> st.SearchStrategy[np.ndarray]:
    return arrays(
        dtype=np.float64,
        shape=SHAPE,
        elements=st.floats(
            min_value=-10.0,
            max_value=10.0,
            allow_nan=False,
            allow_infinity=False,
        ),
    )


@pytest.mark.parametrize(("alias_prefix", "canonical_pattern"), _PREFIX_CASES)
@given(x_data=_bounded_array(), y_data=_bounded_array(), z_data=_bounded_array())
@settings(max_examples=15, deadline=None)
def test_cartesian_alias_returns_canonical_data(
    alias_prefix: str,
    canonical_pattern: str,
    x_data: np.ndarray,
    y_data: np.ndarray,
    z_data: np.ndarray,
) -> None:
    """For each registered vector prefix, ``ds[prefix + 'x']`` returns
    the exact same array as ``ds[prefix + canonical_suffix_1]`` (and
    likewise for y/2, z/3).

    Registers the dataset with canonical names only; relies on the
    FieldDataset's default alias table (generated from
    ``_FIELD_PREFIX_PAIRS`` in ``grid.py``) to make the Cartesian letter
    access work. Catches any mismatch between the alias table's
    canonical-side format and the actual canonical field names used by
    readers and the fields.py registry.
    """
    c1 = canonical_pattern.format(i=1)
    c2 = canonical_pattern.format(i=2)
    c3 = canonical_pattern.format(i=3)

    grid = GridInfo(
        dimensions=SHAPE, spacing=(1.0, 1.0), origin=(0.0, 0.0), geometry=CARTESIAN
    )
    ds = FieldDataset.from_arrays(
        {c1: x_data, c2: y_data, c3: z_data},
        grid,
        Normalization.identity(),
    )

    # Each alias-accessed array must equal the directly-addressed canonical.
    assert_array_equal(ds[f"{alias_prefix}x"], ds[c1])
    assert_array_equal(ds[f"{alias_prefix}y"], ds[c2])
    assert_array_equal(ds[f"{alias_prefix}z"], ds[c3])
