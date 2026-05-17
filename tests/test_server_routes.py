"""Tests for ``pypic.server.routes``: HTTP discovery endpoints."""

from __future__ import annotations

from typing import TYPE_CHECKING

import h5py  # type: ignore[import-untyped]
import numpy as np
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from pypic.server.app import create_app

if TYPE_CHECKING:
    from pathlib import Path


_TOML = """\
[schema]
version = "1.0"

[model]
name = "test_sim"
type = "MHD"

[run]
name = "server_test_run"

[time]
scheme = "fixed"
dt = 0.1
t_start = 0.0
t_end = 1.0
n_steps = 3

[grid]
dimensions = [4, 4, 4]
spacing = [1.0, 1.0, 1.0]
lower = [0.0, 0.0, 0.0]
upper = [4.0, 4.0, 4.0]

[units]
system = "SI"

[coordinates]
geometry = "cartesian"
frame = "simulation"

[physics.mhd]
gamma = 1.6667

[[species]]
name = "p"
charge = 1.0
mass = 1.0
"""


def _make_sim_dir(parent: Path, name: str = "run0", *, n_steps: int = 3) -> Path:
    d = parent / name
    d.mkdir()
    (d / "simulation.toml").write_text(_TOML, encoding="utf-8")
    rng = np.random.default_rng(42)
    shape = (4, 4, 4)
    for i in range(n_steps):
        with h5py.File(d / f"output_{i:06d}.h5", "w") as f:
            grp = f.create_group("fields")
            grp.create_dataset("B_1", data=rng.standard_normal(shape))
            grp.create_dataset("B_2", data=rng.standard_normal(shape))
            grp.create_dataset("B_3", data=rng.standard_normal(shape))
            f.attrs["model"] = "test_sim"
            f.attrs["step"] = i
    return d


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    _make_sim_dir(tmp_path)
    app = create_app(tmp_path)
    return TestClient(app)


@pytest.fixture
def client_multi(tmp_path: Path) -> TestClient:
    _make_sim_dir(tmp_path, "run_a")
    _make_sim_dir(tmp_path, "run_b")
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
    _make_sim_dir(tmp_path, "real_one")
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


def test_sim_steps(client: TestClient) -> None:
    response = client.get("/sims/run0/steps")
    assert response.status_code == 200
    # Three timesteps (0, 1, 2 from _make_sim_dir's n_steps=3).
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
