# Source: docs/schema.md § 2 [run] ("Identity-stable across derivations — a
#         regridded 'MMS-event-1' run is still that run") and § 4.2 optional
#         root attrs ("regrid / slice / frame-transform propagate the original
#         `run` unchanged").
# Claim: every in-pypic derivation carries `metadata["run"]` and
#        `metadata["simulation_toml"]` through untouched. Three separate
#        mechanisms happen to do this today — regrid rebuilds the dict,
#        `_wrap_sliced` reuses it for sel/reduce, and select_fields and
#        transform_to pass it along — so the policy holds by coincidence of
#        three implementations unless something asserts it as one rule.
"""Run provenance survives every derivation pypic can apply to a dataset."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from pypic import (
    FieldDataset,
    GridInfo,
    Normalization,
    PlaneSelection,
    reduce,
    regrid,
)
from pypic.coordinates.transforms import FrameTransform
from pypic.schema import Run

if TYPE_CHECKING:
    from collections.abc import Callable

_RUN = Run.model_validate(
    {
        "name": "MMS-event-1",
        "id": "ccmc-LR_053124_1",
        "epoch": "2015-03-17T00:00:00Z",
        "references": [{"doi": "10.1029/2026SW004922", "kind": "publication"}],
    }
)
_TOML = '[schema]\nversion = "2.0"\n'


def _dataset() -> FieldDataset:
    grid = GridInfo(
        dimensions=(4, 4, 4), spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0)
    )
    arr = np.arange(64.0).reshape(4, 4, 4)
    return FieldDataset.from_arrays(
        {"B_1": arr, "B_2": arr, "B_3": arr},
        grid,
        Normalization.identity(),
        metadata={"run": _RUN, "simulation_toml": _TOML},
        transforms={
            "GSM": FrameTransform(source_frame="simulation", target_frame="GSM")
        },
    )


_COARSER = GridInfo(
    dimensions=(3, 3, 3), spacing=(1.5, 1.5, 1.5), origin=(0.0, 0.0, 0.0)
)

_DERIVATIONS: dict[str, Callable[[FieldDataset], FieldDataset]] = {
    "regrid": lambda ds: regrid(ds, _COARSER),
    "plane_selection": lambda ds: PlaneSelection(normal="z").apply(ds),
    "reduce": lambda ds: reduce(ds, axis="x", reduction="mean"),
    "select_fields": lambda ds: ds.select_fields(["B_1"]),
    "with_derived": lambda ds: ds.with_derived("|B|"),
    "transform_to": lambda ds: ds.transform_to("GSM"),
}


@pytest.mark.parametrize("name", sorted(_DERIVATIONS))
def test_run_provenance_survives_derivation(name: str) -> None:
    derived = _DERIVATIONS[name](_dataset())
    assert derived.metadata["run"] == _RUN, (
        f"{name} dropped or altered metadata['run']; docs/schema.md § 2 "
        f"promises a derivation of a run is still that run"
    )


@pytest.mark.parametrize("name", sorted(_DERIVATIONS))
def test_simulation_toml_survives_derivation(name: str) -> None:
    derived = _DERIVATIONS[name](_dataset())
    assert derived.metadata["simulation_toml"] == _TOML, (
        f"{name} dropped metadata['simulation_toml']; § 4.2 gives it the "
        f"same derivation policy as attrs.run"
    )
