"""WebSocket streaming endpoint.

One persistent WS connection per webpic tab.  Each ``subscribe``
frame is one request/response exchange:

1. Client sends a JSON :class:`~pypic.server.protocol.SubscribeRequest`.
2. Server validates → opens simulation → reads step → applies selection
   → applies reduction → optional SI conversion → encodes the
   resulting :class:`~pypic.dataset.FieldDataset` as one Arrow IPC
   binary frame.
3. Server replies with a JSON :class:`~pypic.server.protocol.Ack`
   (text frame, announces shape / dims / units) followed by the
   binary IPC frame.
4. On failure, server replies with a JSON
   :class:`~pypic.server.protocol.ErrorFrame` text frame.

The connection stays open across exchanges; the client is responsible
for matching responses to requests via ``request_id``.  No
server-pushed updates in the foundations — every payload is
request-driven.  Streaming new timesteps as they appear is a TASKS
Step 37 follow-up.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from pypic.reductions import reduce
from pypic.server.arrow import field_dataset_to_arrow_ipc
from pypic.server.exceptions import (
    PypicError,
    UnknownStepError,
    ValidationFailedError,
)
from pypic.server.protocol import (
    Ack,
    ErrorFrame,
    SubscribeRequest,
    to_reduction_kwargs,
    to_selection,
)

if TYPE_CHECKING:
    from fastapi import APIRouter

    from pypic.dataset import FieldDataset
    from pypic.server._state import SimulationRegistry

__all__ = ["register_stream"]

log = logging.getLogger(__name__)


def register_stream(router: APIRouter) -> None:
    """Attach the streaming WebSocket endpoint to *router*."""

    @router.websocket("/sims/{sim}/stream")
    async def stream(ws: WebSocket, sim: str) -> None:
        registry: SimulationRegistry = ws.app.state.registry
        await ws.accept()
        try:
            while True:
                payload = await ws.receive_text()
                await _handle_one(ws, sim, payload, registry)
        except WebSocketDisconnect:
            return


async def _handle_one(
    ws: WebSocket,
    path_sim: str,
    payload: str,
    registry: SimulationRegistry,
) -> None:
    """Process one inbound JSON frame on the WebSocket.

    Every :class:`PypicError` subclass carries its own ``kind`` (matching
    the :class:`ErrorFrame` Literal), so error routing is a single
    branch.  Anything else is logged and surfaced as ``kind="internal"``
    — the connection survives so the client can retry.
    """
    # request_id is unknown until we successfully parse the payload;
    # a bare validation error before that uses an empty string so the
    # client can still match the error to the most recent in-flight
    # request by ordinal position.
    request_id = ""
    try:
        try:
            req = SubscribeRequest.model_validate_json(payload)
        except ValidationError as exc:
            raise ValidationFailedError(str(exc)) from exc
        request_id = req.request_id
        sim_name = req.sim or path_sim
        await _serve_subscribe(ws, sim_name, req, registry)
    except PypicError as exc:
        await _send_error(ws, request_id, exc.kind, str(exc).strip("'"))
    except Exception as exc:
        # Surface as a typed error frame and keep the connection alive
        # so the client can retry without reconnecting.
        log.exception("Unexpected error handling subscribe for sim=%s", path_sim)
        await _send_error(ws, request_id, "internal", str(exc))


async def _serve_subscribe(
    ws: WebSocket,
    sim_name: str,
    req: SubscribeRequest,
    registry: SimulationRegistry,
) -> None:
    """Run the read → select → reduce → encode pipeline for one request."""
    simulation = registry.get(sim_name)
    if req.step not in simulation.steps:
        msg = f"Step {req.step} not available"
        raise UnknownStepError(msg)

    fields = req.fields if req.fields else None
    fds: FieldDataset = simulation.read(
        req.step,
        fields=fields,
        strict_fields=True,
    )

    if req.selection is not None:
        fds = to_selection(req.selection).apply(fds)

    if req.reduction is not None:
        fds = reduce(fds, **to_reduction_kwargs(req.reduction))

    ipc_bytes = field_dataset_to_arrow_ipc(fds, units=req.units)

    ack = Ack(
        request_id=req.request_id,
        shape=list(fds.grid.dimensions),
        dims=[str(d) for d in fds.xr.dims],
        fields=[str(n) for n in fds.xr.data_vars],
        units=req.units,
    )
    await ws.send_text(ack.model_dump_json())
    await ws.send_bytes(ipc_bytes)


async def _send_error(
    ws: WebSocket,
    request_id: str,
    kind: str,
    message: str,
) -> None:
    """Emit a typed error frame; tolerates an unparseable request_id."""
    frame = ErrorFrame(
        request_id=request_id,
        kind=kind,  # type: ignore[arg-type]
        message=message,
    )
    await ws.send_text(frame.model_dump_json())
