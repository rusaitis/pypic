# Source: CLAUDE.md "Selection APIs fail loud on unmatched names. Functions
#         that accept a user-supplied list of field/column/component names
#         (fields=, columns=, ...) must raise KeyError on names that match
#         nothing, or expose an explicit strict_fields: bool = False kwarg
#         that does. Logging-only warnings are forbidden — they get
#         swallowed in notebooks and pipelines."
#         + src/pypic/dataset.py:458 (resolve_key), :513 (__getitem__),
#         :580 (select_fields), :506 (KeyError raise).
# Claim: every user-facing selection API on FieldDataset raises KeyError
#        (not log-and-skip, not silent empty return) for names that
#        neither match a loaded field nor resolve via the alias map.
# Backlog #8 in autoresearcher-pypic.md.
"""Selection APIs fail loud on unmatched names."""

from __future__ import annotations

import re
import string

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from pypic.compute import _REGISTRY
from tests._helpers import make_test_dataset


def _unknown_names() -> st.SearchStrategy[str]:
    """Alphanumeric strings that cannot collide with any legitimate name.

    Uses the literal prefix ``"xyz_"`` to stay clear of:
    * canonical schema names (``B``, ``E``, ``J``, ``V``, ``P``, ``n``,
      ``rho``, ``|.|`` etc.),
    * Cartesian aliases (``Bx``, ``By``, ``Bz``),
    * the compute registry (``v_A``, ``|B|``, ``beta``, ...),
    * reserved sigils Hypothesis might otherwise generate
      (spaces, dots, quote marks).
    """
    return st.text(
        alphabet=string.ascii_lowercase + string.digits + "_",
        min_size=1,
        max_size=20,
    ).map(lambda s: f"xyz_{s}")


def _make_dataset():
    """Two-field dataset covering the 'a requested name *is* present' case
    so the property meaningfully distinguishes match vs. no-match outcomes.
    """
    return make_test_dataset(
        {
            "B_1": np.ones((4, 3, 2)),
            "rho_c": np.ones((4, 3, 2)),
        }
    )


@given(name=_unknown_names())
@settings(max_examples=100, deadline=None)
def test_resolve_key_raises_keyerror(name: str) -> None:
    """``resolve_key`` must raise ``KeyError`` on any unmatched name —
    never ``ValueError`` (which would imply bad input type rather than
    missing name) and never silent fallback.
    """
    ds = _make_dataset()
    assume(not ds.has_field(name))
    assume(name not in _REGISTRY)
    with pytest.raises(KeyError):
        ds.resolve_key(name)


@given(name=_unknown_names())
@settings(max_examples=100, deadline=None)
def test_getitem_raises_keyerror(name: str) -> None:
    """``ds[key]`` delegates to ``resolve_key`` — must raise ``KeyError``
    on unknown names rather than xarray's more generic ``ValueError`` or
    returning an empty array.
    """
    ds = _make_dataset()
    assume(not ds.has_field(name))
    with pytest.raises(KeyError):
        _ = ds[name]


@given(name=_unknown_names())
@settings(max_examples=100, deadline=None)
def test_select_fields_raises_keyerror(name: str) -> None:
    """``select_fields`` with at least one unknown name raises ``KeyError``
    — the CLAUDE.md contract forbids silent skip, so even a single bad
    entry in a longer list must fail loud.
    """
    ds = _make_dataset()
    assume(not ds.has_field(name))
    with pytest.raises(KeyError):
        ds.select_fields(["B_1", name])


@given(name=_unknown_names())
@settings(max_examples=100, deadline=None)
def test_compute_raises_keyerror(name: str) -> None:
    """``ds.compute(name)`` raises ``KeyError`` when the name is neither
    a loaded field nor a registered recipe. Guards the second half of
    the CLAUDE.md contract — the dispatch layer is also a selection API.
    """
    ds = _make_dataset()
    assume(not ds.has_field(name))
    assume(name not in _REGISTRY)
    with pytest.raises(KeyError):
        _ = ds.compute(name)


def test_resolve_key_error_suggests_close_matches() -> None:
    """The error message must include a close-match suggestion when one
    exists. This is the mechanism that turns typos ("rho_c" → "rhoc",
    "Bx" → "B_x") into actionable feedback rather than silent puzzlement
    — the specific behavior CLAUDE.md calls out as the failure mode of
    log-only warnings.
    """
    ds = _make_dataset()
    with pytest.raises(KeyError, match=re.compile(r"Did you mean.*rho_c", re.DOTALL)):
        ds.resolve_key("rhoc")
