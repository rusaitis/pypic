"""HTTP routes for the pypic server.

Discovery-layer endpoints only — the binary data path is the
WebSocket in :mod:`pypic.server.stream`.  All responses are JSON.

Endpoints:

* ``GET /health`` — liveness probe.
* ``GET /sims`` — list simulation names under the configured root.
* ``GET /sims/{sim}`` — identity + grid + normalization + species
  for one simulation (Pydantic-friendly dict, ready for typed
  reconstruction client-side).
* ``GET /sims/{sim}/steps`` — available timestep indices.
* ``GET /sims/{sim}/fields?step=N`` — canonical field names with
  optional native (on-disk) name mapping.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from fastapi import APIRouter, Query, Request

from pypic.io.metadata import (
    grid_to_dict,
    normalization_to_dict,
    physics_to_dict,
    species_to_list,
)
from pypic.server.app import _pypic_version
from pypic.server.exceptions import UnknownStepError

if TYPE_CHECKING:
    from pypic.server._state import SimulationRegistry

__all__ = ["register_routes"]


def _registry(request: Request) -> SimulationRegistry:
    """Pull the SimulationRegistry out of the FastAPI app state."""
    return cast("SimulationRegistry", request.app.state.registry)


def register_routes(router: APIRouter) -> None:
    """Attach all HTTP routes to *router*.

    Kept as a single registration function (rather than module-level
    decorators) so :func:`pypic.server.app.create_app` can choose
    where the routes live — e.g. mount them under a prefix later
    without restructuring.
    """

    @router.get("/health")
    def health() -> dict[str, str]:
        """Liveness probe — returns ``{status, pypic_version}``."""
        return {"status": "ok", "pypic_version": _pypic_version()}

    @router.get("/sims")
    def list_sims(request: Request) -> dict[str, list[str]]:
        """List simulation names under the configured root."""
        return {"sims": _registry(request).names()}

    @router.get("/sims/{sim}")
    def sim_info(sim: str, request: Request) -> dict[str, Any]:
        """Identity + grid + normalization + species for one simulation."""
        simulation = _registry(request).get(sim)
        return {
            "name": sim,
            "model_name": simulation.model_name,
            "model_type": simulation.model_type,
            "grid": grid_to_dict(simulation.grid),
            "normalization": normalization_to_dict(simulation.normalization),
            "species": species_to_list(simulation.species),
            "physics": physics_to_dict(simulation.physics),
        }

    @router.get("/sims/{sim}/steps")
    def sim_steps(sim: str, request: Request) -> dict[str, list[int]]:
        """Available timestep indices for a simulation."""
        simulation = _registry(request).get(sim)
        return {"steps": list(simulation.steps)}

    @router.get("/sims/{sim}/fields")
    def sim_fields(
        sim: str,
        request: Request,
        step: int | None = Query(
            default=None,
            description="Timestep index. Defaults to the first available step.",
        ),
    ) -> dict[str, Any]:
        """Canonical field names + native-name mapping at one step."""
        simulation = _registry(request).get(sim)
        steps = simulation.steps
        if not steps:
            msg = "No timesteps available"
            raise UnknownStepError(msg)
        chosen_step = steps[0] if step is None else step
        if chosen_step not in steps:
            msg = f"Step {chosen_step} not available (have {steps[:3]}...)"
            raise UnknownStepError(msg)
        return {
            "step": chosen_step,
            "fields": simulation.available_fields_mapping(chosen_step),
        }
