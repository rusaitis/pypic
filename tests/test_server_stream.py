"""Tests for ``pypic.server.stream``: WebSocket end-to-end.

Each test connects to ``/sims/{sim}/stream`` via FastAPI's TestClient,
sends one ``SubscribeRequest`` JSON frame, and asserts the resulting
ack + binary payload (decoded via the Arrow IPC reader).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("pyarrow")

from fastapi.testclient import TestClient

from pypic.server.app import create_app
from pypic.server.arrow import decode_field_dataset_ipc
from tests._sim_fixtures import make_sim_dir

if TYPE_CHECKING:
    from pathlib import Path

    from starlette.testclient import WebSocketTestSession


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    make_sim_dir(tmp_path)
    return TestClient(create_app(tmp_path))


def _subscribe(
    ws: WebSocketTestSession,
    **kwargs: object,
) -> tuple[dict[str, object], dict[str, object]]:
    """Send a subscribe frame and return (ack_dict, decoded_ipc)."""
    request = {"type": "subscribe", "request_id": "r1", "step": 0, **kwargs}
    ws.send_text(json.dumps(request))
    ack = json.loads(ws.receive_text())
    data = ws.receive_bytes()
    return ack, decode_field_dataset_ipc(data)


def test_subscribe_returns_ack_then_binary(client: TestClient) -> None:
    with client.websocket_connect("/sims/run0/stream") as ws:
        ack, payload = _subscribe(ws, fields=["B_1"])
    assert ack["type"] == "ack"
    assert ack["request_id"] == "r1"
    assert ack["shape"] == [4, 4, 4]
    assert ack["fields"] == ["B_1"]
    assert ack["units"] == "code"
    assert "B_1" in payload["fields"]
    assert payload["fields"]["B_1"].shape == (4, 4, 4)


def test_subscribe_default_returns_all_fields(client: TestClient) -> None:
    with client.websocket_connect("/sims/run0/stream") as ws:
        ack, payload = _subscribe(ws)
    assert set(ack["fields"]) == {"B_1", "B_2", "B_3"}
    assert set(payload["fields"]) == {"B_1", "B_2", "B_3"}


def test_subscribe_with_box_selection(client: TestClient) -> None:
    with client.websocket_connect("/sims/run0/stream") as ws:
        ack, payload = _subscribe(
            ws,
            fields=["B_1"],
            selection={"kind": "box", "ranges": {"x": [1, 3]}},
        )
    # Box selection reduces x from 4 to 2 cells.
    assert ack["shape"] == [2, 4, 4]
    assert payload["fields"]["B_1"].shape == (2, 4, 4)


def test_subscribe_with_reduction(client: TestClient) -> None:
    with client.websocket_connect("/sims/run0/stream") as ws:
        ack, payload = _subscribe(
            ws,
            fields=["B_1"],
            reduction={"axis": "z", "op": "mean"},
        )
    # Mean over z collapses the (4,4,4) cube to (4,4).
    assert ack["shape"] == [4, 4]
    assert ack["dims"] == ["x", "y"]
    assert payload["fields"]["B_1"].shape == (4, 4)


def test_subscribe_with_selection_and_reduction(client: TestClient) -> None:
    with client.websocket_connect("/sims/run0/stream") as ws:
        ack, payload = _subscribe(
            ws,
            fields=["B_1"],
            selection={"kind": "box", "ranges": {"x": [1, 3]}},
            reduction={"axis": "z", "op": "integrate"},
        )
    assert ack["shape"] == [2, 4]
    assert payload["metadata"]["fields"]["B_1"]["reduction"]["length_axes"] == 1


def test_subscribe_units_si_applies_normalization(client: TestClient) -> None:
    with client.websocket_connect("/sims/run0/stream") as ws:
        ack_code, code = _subscribe(ws, fields=["B_1"], units="code")
        ack_si, si = _subscribe(ws, fields=["B_1"], units="si")
    assert ack_code["units"] == "code"
    assert ack_si["units"] == "si"
    # SI normalization with b_field_ref=1.0 (SI deck) means code and SI
    # values are identical for the fixture.
    np.testing.assert_array_equal(code["fields"]["B_1"], si["fields"]["B_1"])


def test_unknown_field_yields_error_frame(client: TestClient) -> None:
    with client.websocket_connect("/sims/run0/stream") as ws:
        ws.send_text(
            json.dumps(
                {
                    "type": "subscribe",
                    "request_id": "bad",
                    "step": 0,
                    "fields": ["nonexistent"],
                }
            )
        )
        frame = json.loads(ws.receive_text())
    assert frame["type"] == "error"
    assert frame["request_id"] == "bad"
    assert frame["kind"] == "unknown_field"


def test_unknown_step_yields_error_frame(client: TestClient) -> None:
    with client.websocket_connect("/sims/run0/stream") as ws:
        ws.send_text(json.dumps({"type": "subscribe", "request_id": "x", "step": 999}))
        frame = json.loads(ws.receive_text())
    assert frame["type"] == "error"
    assert frame["kind"] == "unknown_step"


def test_unknown_sim_yields_error_frame(client: TestClient) -> None:
    with client.websocket_connect("/sims/does_not_exist/stream") as ws:
        ws.send_text(json.dumps({"type": "subscribe", "request_id": "x", "step": 0}))
        frame = json.loads(ws.receive_text())
    assert frame["type"] == "error"
    assert frame["kind"] == "unknown_sim"


def test_validation_error_for_malformed_request(client: TestClient) -> None:
    with client.websocket_connect("/sims/run0/stream") as ws:
        ws.send_text(json.dumps({"type": "subscribe", "step": "not_an_int"}))
        frame = json.loads(ws.receive_text())
    assert frame["type"] == "error"
    assert frame["kind"] == "validation"


def test_multiple_requests_same_connection(client: TestClient) -> None:
    """One WS, several subscribe exchanges — connection must survive."""
    with client.websocket_connect("/sims/run0/stream") as ws:
        ack_a, _ = _subscribe(ws, fields=["B_1"])
        ack_b, _ = _subscribe(ws, fields=["B_2"], step=1)
    assert ack_a["fields"] == ["B_1"]
    assert ack_b["fields"] == ["B_2"]


def test_metadata_in_arrow_payload_includes_normalization(client: TestClient) -> None:
    with client.websocket_connect("/sims/run0/stream") as ws:
        _, payload = _subscribe(ws, fields=["B_1"])
    meta = payload["metadata"]
    assert meta["dims"] == ["x", "y", "z"]
    assert "normalization" in meta
    assert meta["normalization"]["length_ref"] == 1.0
