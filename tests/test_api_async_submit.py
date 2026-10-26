"""Tests for asynchronous simulation submission and status retrieval via FastAPI TestClient."""

import time
from fastapi.testclient import TestClient
import pytest

from urban_resilience.api import app


@pytest.fixture
def client():
    return TestClient(app)


def test_async_simulation_submit_and_poll(client: TestClient):
    payload = {
        "city_name": "Shenzhen",
        "hazard_type": "rainstorm",
        "event_start_hour": 12,
        "duration_hours": 6,
        "total_hours": 24,
        "peak_intensity": 0.5,
        "monte_carlo_samples": 5,
    }

    # Submit simulation asynchronously
    resp = client.post("/api/v1/resilience/submit", json=payload)
    assert resp.status_code == 202
    data = resp.json()
    assert "run_id" in data
    assert data["status"] in ("pending", "running", "success")
    run_id = data["run_id"]

    # Poll status until done (with timeout)
    max_wait_seconds = 20.0
    start = time.time()
    final_status = None

    while time.time() - start < max_wait_seconds:
        poll_resp = client.get(f"/api/v1/resilience/runs/{run_id}")
        assert poll_resp.status_code == 200
        poll_data = poll_resp.json()
        final_status = poll_data.get("status")
        if final_status in ("success", "failed"):
            break
        time.sleep(0.5)

    assert final_status == "success"
    assert "disruption_metrics" in poll_data
    assert "spatial_rankings" in poll_data
