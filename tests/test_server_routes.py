"""Tests for ``pypic.server.routes``: HTTP discovery endpoints."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from pypic.server.app import create_app
from tests._server_helpers import make_sim_dir

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    make_sim_dir(tmp_path)
    app = create_app(tmp_path)
    return TestClient(app)


@pytest.fixture
def client_multi(tmp_path: Path) -> TestClient:
    make_sim_dir(tmp_path, "run_a")
    make_sim_dir(tmp_path, "run_b")
    app = create_app(tmp_path)
    return TestClient(app)


def test_health_returns_ok(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "pypic_version" in body


def test_sims_lists_one(client: TestClient) -> None:
    response = client.get("/sims")
    assert response.status_code == 200
    assert response.json() == {"sims": ["run0"]}


def test_sims_lists_multiple_sorted(client_multi: TestClient) -> None:
    response = client_multi.get("/sims")
    assert response.json() == {"sims": ["run_a", "run_b"]}


def test_sims_empty_root(tmp_path: Path) -> None:
    # No subdirs at all — root exists but has no simulations.
    client = TestClient(create_app(tmp_path))
    response = client.get("/sims")
    assert response.json() == {"sims": []}


def test_sims_ignores_dirs_without_toml(tmp_path: Path) -> None:
    # A directory with no simulation.toml is invisible to the registry.
    (tmp_path / "not_a_sim").mkdir()
    make_sim_dir(tmp_path, "real_one")
    client = TestClient(create_app(tmp_path))
    assert client.get("/sims").json() == {"sims": ["real_one"]}


def test_sim_info_returns_typed_metadata(client: TestClient) -> None:
    response = client.get("/sims/run0")
    assert response.status_code == 200
    body = response.json()
    assert body["name"] == "run0"
    assert body["model_name"] == "test_sim"
    assert body["model_type"] == "MHD"
    assert body["grid"]["dimensions"] == [4, 4, 4]
    assert body["normalization"]["length_ref"] == 1.0  # SI normalization
    assert len(body["species"]) == 1
    assert body["species"][0]["name"] == "p"


def test_sim_info_404_for_missing_sim(client: TestClient) -> None:
    response = client.get("/sims/does_not_exist")
    assert response.status_code == 404


def test_sim_info_404_carries_typed_error_kind(client: TestClient) -> None:
    """Global PypicError handler emits ``{"kind", "detail"}`` body.

    Mirrors the WebSocket :class:`ErrorFrame` shape so clients can
    dispatch on ``kind`` identically across HTTP and WS transports.
    The ``detail`` field stays back-compat with FastAPI's stock
    ``HTTPException`` body shape, so existing clients that key only on
    ``detail`` continue to work.
    """
    response = client.get("/sims/does_not_exist")
    body = response.json()
    assert body["kind"] == "unknown_sim"
    assert "does_not_exist" in body["detail"]


def test_sim_steps(client: TestClient) -> None:
    response = client.get("/sims/run0/steps")
    assert response.status_code == 200
    # Three timesteps (0, 1, 2 from make_sim_dir's n_steps=3).
    assert response.json() == {"steps": [0, 1, 2]}


def test_sim_fields_defaults_to_first_step(client: TestClient) -> None:
    response = client.get("/sims/run0/fields")
    assert response.status_code == 200
    body = response.json()
    assert body["step"] == 0
    assert "B_1" in body["fields"]


def test_sim_fields_explicit_step(client: TestClient) -> None:
    response = client.get("/sims/run0/fields?step=2")
    assert response.status_code == 200
    assert response.json()["step"] == 2


def test_sim_fields_404_for_unknown_step(client: TestClient) -> None:
    response = client.get("/sims/run0/fields?step=999")
    assert response.status_code == 404


def test_cors_header_present(client: TestClient) -> None:
    # CORS middleware should echo the Origin back for GET requests.
    response = client.get("/sims", headers={"Origin": "http://webpic.local"})
    assert response.headers.get("access-control-allow-origin") == "*"


def test_openapi_schema_available(client: TestClient) -> None:
    # FastAPI auto-generates /openapi.json — webpic can introspect the
    # HTTP surface from it.  WebSocket routes are out of the OpenAPI 3.0
    # spec by design; they live in docs/api/server.md instead.
    response = client.get("/openapi.json")
    assert response.status_code == 200
    paths = response.json()["paths"]
    assert "/health" in paths
    assert "/sims" in paths
    assert "/sims/{sim}" in paths
    assert "/sims/{sim}/steps" in paths
    assert "/sims/{sim}/fields" in paths
