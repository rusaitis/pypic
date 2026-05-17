"""Tests for ``pypic.server.protocol``: wire-format Pydantic models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from pypic.selections import BoxSelection, PlaneSelection, SphereSelection
from pypic.server.protocol import (
    Ack,
    BoxSpec,
    ErrorFrame,
    PlaneSpec,
    ReductionSpec,
    SphereSpec,
    SubscribeRequest,
    to_reduction_kwargs,
    to_selection,
)


def test_subscribe_request_minimal() -> None:
    req = SubscribeRequest.model_validate(
        {"type": "subscribe", "request_id": "abc", "step": 0}
    )
    assert req.fields == []
    assert req.units == "code"
    assert req.selection is None
    assert req.reduction is None


def test_subscribe_request_round_trip() -> None:
    payload = {
        "type": "subscribe",
        "request_id": "xyz",
        "step": 100,
        "fields": ["B_1", "B_2"],
        "selection": {"kind": "box", "ranges": {"x": [0, 10]}},
        "reduction": {"axis": "z", "op": "mean"},
        "units": "si",
    }
    req = SubscribeRequest.model_validate(payload)
    redumped = req.model_dump()
    again = SubscribeRequest.model_validate(redumped)
    assert again == req


def test_unknown_keys_rejected() -> None:
    with pytest.raises(ValidationError):
        SubscribeRequest.model_validate(
            {"type": "subscribe", "request_id": "x", "step": 0, "bogus": 1}
        )


def test_box_spec_to_selection() -> None:
    spec = BoxSpec(ranges={"x": (1, 5)})
    sel = to_selection(spec)
    assert isinstance(sel, BoxSelection)
    assert sel.ranges == {"x": (1, 5)}


def test_plane_spec_to_selection() -> None:
    spec = PlaneSpec(normal="z", index=3)
    sel = to_selection(spec)
    assert isinstance(sel, PlaneSelection)
    assert sel.normal == "z"
    assert sel.index == 3


def test_sphere_spec_to_selection() -> None:
    spec = SphereSpec(center=(0.0, 0.0, 0.0), radius=2.0, keep="outside")
    sel = to_selection(spec)
    assert isinstance(sel, SphereSelection)
    assert sel.radius == 2.0
    assert sel.keep == "outside"


def test_sphere_spec_radius_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        SphereSpec(center=(0.0, 0.0, 0.0), radius=-1.0)


def test_selection_discriminator_picks_right_type() -> None:
    payload = {"kind": "plane", "normal": "z", "index": 2}
    req = SubscribeRequest.model_validate(
        {
            "type": "subscribe",
            "request_id": "r",
            "step": 0,
            "selection": payload,
        }
    )
    assert isinstance(req.selection, PlaneSpec)


def test_reduction_spec_single_axis_kwargs() -> None:
    spec = ReductionSpec(axis="z", op="integrate", weight="rho_c")
    kw = to_reduction_kwargs(spec)
    assert kw["axis"] == "z"
    assert kw["reduction"] == "integrate"
    assert kw["weight"] == "rho_c"
    assert kw["nan_policy"] == "omit"


def test_reduction_spec_multi_axis_kwargs() -> None:
    spec = ReductionSpec(axis=["y", "z"], op="mean")
    kw = to_reduction_kwargs(spec)
    assert kw["axis"] == ("y", "z")


def test_reduction_spec_rejects_bogus_op() -> None:
    with pytest.raises(ValidationError):
        ReductionSpec(axis="z", op="bogus")  # type: ignore[arg-type]


def test_ack_round_trip() -> None:
    ack = Ack(
        request_id="r",
        shape=[4, 3],
        dims=["x", "y"],
        fields=["B_1"],
        units="code",
    )
    assert Ack.model_validate(ack.model_dump()) == ack


def test_error_frame_round_trip() -> None:
    err = ErrorFrame(
        request_id="r",
        kind="unknown_field",
        message="No field 'bogus'",
    )
    assert ErrorFrame.model_validate(err.model_dump()) == err


def test_error_frame_rejects_unknown_kind() -> None:
    with pytest.raises(ValidationError):
        ErrorFrame(
            request_id="r",
            kind="totally_made_up",  # type: ignore[arg-type]
            message="x",
        )
