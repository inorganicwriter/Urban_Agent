"""Unit tests for the FastAPI RESTful API endpoints."""

from fastapi.testclient import TestClient

from urban_resilience.api import app

client = TestClient(app)


def test_api_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["ready"] is True


def test_api_city_zones():
    response = client.get("/api/v1/city/zones")
    assert response.status_code == 200
    zones = response.json()
    assert len(zones) >= 12
    zone_ids = [z["zone_id"] for z in zones]
    assert "SZ_FT_CBD" in zone_ids
    assert "SZ_NS_TECH" in zone_ids


def test_api_simulate_and_plots():
    payload = {
        "city_name": "Shenzhen",
        "hazard_type": "rainstorm",
        "event_start_hour": 12,
        "duration_hours": 6,
        "peak_intensity": 0.75,
        "total_hours": 36,
        "monte_carlo_samples": 30,
    }
    response = client.post("/api/v1/resilience/simulate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "disruption_metrics" in data
    assert "spatial_rankings" in data
    assert len(data["spatial_rankings"]) >= 12

    # Check plot image streaming
    plot_resp = client.get("/api/v1/resilience/plots/curve")
    assert plot_resp.status_code == 200
    assert plot_resp.headers["content-type"] == "image/png"
    assert len(plot_resp.content) > 1000

    map_resp = client.get("/api/v1/resilience/plots/map")
    assert map_resp.status_code == 200
    assert map_resp.headers["content-type"] == "image/png"


def test_api_concurrent_runs_isolation():
    """Validates that concurrent simulate requests create isolated directories, distinct run_ids, and do not overwrite files."""
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path

    payload1 = {
        "city_name": "Shenzhen",
        "hazard_type": "rainstorm",
        "event_start_hour": 8,
        "duration_hours": 6,
        "peak_intensity": 0.65,
        "total_hours": 30,
        "monte_carlo_samples": 10,
    }
    payload2 = {
        "city_name": "Shenzhen",
        "hazard_type": "typhoon",
        "event_start_hour": 10,
        "duration_hours": 8,
        "peak_intensity": 0.85,
        "total_hours": 36,
        "monte_carlo_samples": 10,
    }

    def trigger_sim(payload):
        return client.post("/api/v1/resilience/simulate", json=payload)

    with ThreadPoolExecutor(max_workers=2) as executor:
        fut1 = executor.submit(trigger_sim, payload1)
        fut2 = executor.submit(trigger_sim, payload2)
        resp1 = fut1.result()
        resp2 = fut2.result()

    assert resp1.status_code == 200
    assert resp2.status_code == 200

    data1 = resp1.json()
    data2 = resp2.json()

    # 1. Distinct run_ids
    run_id1 = data1["run_id"]
    run_id2 = data2["run_id"]
    assert run_id1 != run_id2

    # 2. Distinct directories
    dir1 = Path(data1["artifacts_dir"])
    dir2 = Path(data2["artifacts_dir"])
    assert dir1 != dir2
    assert dir1.exists()
    assert dir2.exists()

    # 3. Both directories contain all artifacts
    assert (dir1 / "manifest.json").exists()
    assert (dir2 / "manifest.json").exists()
    assert (dir1 / "disruption_recovery_timeseries.csv").exists()
    assert (dir2 / "disruption_recovery_timeseries.csv").exists()
    assert (dir1 / "spatial_priority_rankings.csv").exists()
    assert (dir2 / "spatial_priority_rankings.csv").exists()
    assert (dir1 / "plots" / "predicted_disruption_recovery.png").exists()
    assert (dir2 / "plots" / "predicted_disruption_recovery.png").exists()

    # 4. Results match respective configurations and do not overwrite
    assert data1["hazard_type"] == "rainstorm"
    assert data2["hazard_type"] == "typhoon"

    # 5. Check run query endpoint
    query1 = client.get(f"/api/v1/resilience/runs/{run_id1}")
    assert query1.status_code == 200
    assert query1.json()["run_id"] == run_id1
    assert query1.json()["hazard_type"] == "rainstorm"

    query2 = client.get(f"/api/v1/resilience/runs/{run_id2}")
    assert query2.status_code == 200
    assert query2.json()["run_id"] == run_id2
    assert query2.json()["hazard_type"] == "typhoon"

    # 6. Check run-specific plot streaming
    plot_resp1 = client.get(f"/api/v1/resilience/runs/{run_id1}/plots/curve")
    assert plot_resp1.status_code == 200
    assert plot_resp1.headers["content-type"] == "image/png"

    plot_resp2 = client.get(f"/api/v1/resilience/runs/{run_id2}/plots/curve")
    assert plot_resp2.status_code == 200
    assert plot_resp2.headers["content-type"] == "image/png"

    # 7. Check 404 behavior
    assert client.get("/api/v1/resilience/runs/non_existent_run").status_code == 404
    assert client.get(f"/api/v1/resilience/runs/{run_id1}/plots/invalid_plot").status_code == 404


def test_bounded_run_registry_eviction():
    """Validates that BoundedRunRegistry evicts oldest runs when capacity is reached while retaining disk files."""
    from pathlib import Path
    from urban_resilience.api import reset_api_state

    # Set capacity = 2
    reset_api_state(max_entries=2)

    def run_sim(hour: int):
        payload = {
            "city_name": "Shenzhen",
            "hazard_type": "rainstorm",
            "event_start_hour": hour,
            "duration_hours": 6,
            "peak_intensity": 0.5,
            "total_hours": 24,
            "monte_carlo_samples": 5,
        }
        res = client.post("/api/v1/resilience/simulate", json=payload)
        assert res.status_code == 200
        return res.json()

    run1 = run_sim(6)
    run2 = run_sim(8)
    run3 = run_sim(10)

    id1, id2, id3 = run1["run_id"], run2["run_id"], run3["run_id"]

    # Run 1 should be evicted from memory (capacity = 2) -> 404
    q1 = client.get(f"/api/v1/resilience/runs/{id1}")
    assert q1.status_code == 404

    # Run 2 and Run 3 should remain accessible in memory -> 200
    q2 = client.get(f"/api/v1/resilience/runs/{id2}")
    assert q2.status_code == 200
    assert q2.json()["run_id"] == id2

    q3 = client.get(f"/api/v1/resilience/runs/{id3}")
    assert q3.status_code == 200
    assert q3.json()["run_id"] == id3

    # Crucial assertion: Disk artifacts for Run 1 are NOT deleted by memory cache eviction
    dir1 = Path(run1["artifacts_dir"])
    assert dir1.exists()
    assert (dir1 / "manifest.json").exists()
    assert (dir1 / "disruption_recovery_timeseries.csv").exists()

    reset_api_state()



def test_bounded_run_registry_ttl():
    """Validates that BoundedRunRegistry entries expire after TTL while disk files remain intact."""
    import time
    from pathlib import Path
    from urban_resilience.api import reset_api_state

    # Set very short TTL (0.1s)
    reset_api_state(ttl_seconds=0.1)

    payload = {
        "city_name": "Shenzhen",
        "hazard_type": "rainstorm",
        "event_start_hour": 8,
        "duration_hours": 6,
        "peak_intensity": 0.5,
        "total_hours": 24,
        "monte_carlo_samples": 5,
    }
    res = client.post("/api/v1/resilience/simulate", json=payload)
    assert res.status_code == 200
    run_id = res.json()["run_id"]
    dir_path = Path(res.json()["artifacts_dir"])

    # Immediate query within TTL -> 200
    q_immediate = client.get(f"/api/v1/resilience/runs/{run_id}")
    assert q_immediate.status_code == 200

    # Wait for TTL to expire
    time.sleep(0.15)

    # Query after TTL -> 404 (memory cache expired)
    q_expired = client.get(f"/api/v1/resilience/runs/{run_id}")
    assert q_expired.status_code == 404

    # Disk artifacts must still be preserved for auditable review
    assert dir_path.exists()
    assert (dir_path / "manifest.json").exists()

    reset_api_state()



def test_api_authentication_optional(monkeypatch):
    """Validates optional API key authentication via URBAN_API_KEY environment variable."""
    from urban_resilience.api import reset_api_state
    reset_api_state()


    # 1. Without URBAN_API_KEY: open development mode
    resp_open = client.get("/health")
    assert resp_open.status_code == 200
    assert resp_open.json()["auth_mode"] == "development_open"

    zones_open = client.get("/api/v1/city/zones")
    assert zones_open.status_code == 200

    # 2. With URBAN_API_KEY configured
    secret_key = "test-urban-key-xyz"
    monkeypatch.setenv("URBAN_API_KEY", secret_key)

    resp_auth = client.get("/health")
    assert resp_auth.status_code == 200
    assert resp_auth.json()["auth_mode"] == "authenticated"

    # Public endpoint remains accessible without key
    assert client.get("/api/v1/city/zones").status_code == 200

    payload = {
        "city_name": "Shenzhen",
        "hazard_type": "rainstorm",
        "event_start_hour": 8,
        "duration_hours": 6,
        "peak_intensity": 0.5,
        "total_hours": 24,
        "monte_carlo_samples": 5,
    }

    # Protected endpoint without header -> 401
    post_no_key = client.post("/api/v1/resilience/simulate", json=payload)
    assert post_no_key.status_code == 401
    assert "Invalid or missing API key" in post_no_key.json()["detail"]

    # Protected endpoint with invalid header -> 401
    post_bad_key = client.post(
        "/api/v1/resilience/simulate",
        json=payload,
        headers={"X-API-Key": "wrong-key"},
    )
    assert post_bad_key.status_code == 401

    # Protected endpoint with valid header -> 200
    post_valid = client.post(
        "/api/v1/resilience/simulate",
        json=payload,
        headers={"X-API-Key": secret_key},
    )
    assert post_valid.status_code == 200
    run_id = post_valid.json()["run_id"]

    # Run query without header -> 401
    get_no_key = client.get(f"/api/v1/resilience/runs/{run_id}")
    assert get_no_key.status_code == 401

    # Run query with valid header -> 200
    get_valid = client.get(
        f"/api/v1/resilience/runs/{run_id}",
        headers={"X-API-Key": secret_key},
    )
    assert get_valid.status_code == 200
    assert get_valid.json()["run_id"] == run_id

    reset_api_state()



def test_api_rate_limiting(monkeypatch):
    """Validates in-memory sliding window rate limiting via URBAN_RATE_LIMIT_PER_MINUTE."""
    from urban_resilience.api import reset_api_state
    reset_api_state()


    # Set rate limit to 3 requests per minute
    monkeypatch.setenv("URBAN_RATE_LIMIT_PER_MINUTE", "3")

    # First 3 requests succeed
    for _ in range(3):
        r = client.get("/api/v1/city/zones")
        assert r.status_code == 200

    # 4th request exceeds rate limit -> 429
    r_exceeded = client.get("/api/v1/city/zones")
    assert r_exceeded.status_code == 429
    assert "Rate limit exceeded" in r_exceeded.json()["detail"]
    assert "Retry-After" in r_exceeded.headers

    # Reset limiter restores access
    reset_api_state()

    r_restored = client.get("/api/v1/city/zones")
    assert r_restored.status_code == 200


def test_bounded_run_registry_env_parsing_and_fallback(monkeypatch):
    """Validates robust environment variable and parameter parsing with graceful fallbacks."""
    from urban_resilience.api import BoundedRunRegistry, reset_api_state

    # 1. Invalid non-integer string max_entries -> fallback 100
    monkeypatch.setenv("URBAN_REGISTRY_MAX_ENTRIES", "not_a_number")
    reg1 = BoundedRunRegistry()
    assert reg1.max_entries == 100

    # 2. Non-positive integer max_entries (<= 0) -> fallback 100
    monkeypatch.setenv("URBAN_REGISTRY_MAX_ENTRIES", "-10")
    reg2 = BoundedRunRegistry()
    assert reg2.max_entries == 100

    monkeypatch.setenv("URBAN_REGISTRY_MAX_ENTRIES", "0")
    reg3 = BoundedRunRegistry()
    assert reg3.max_entries == 100

    # 3. Valid integer max_entries
    monkeypatch.setenv("URBAN_REGISTRY_MAX_ENTRIES", "42")
    reg4 = BoundedRunRegistry()
    assert reg4.max_entries == 42

    # 4. Invalid non-float string ttl_seconds -> fallback 86400.0
    monkeypatch.setenv("URBAN_REGISTRY_TTL_SECONDS", "invalid_float")
    reg5 = BoundedRunRegistry()
    assert reg5.ttl_seconds == 86400.0

    # 5. Non-positive float ttl_seconds (<= 0, nan, inf) -> fallback 86400.0
    for invalid_ttl in ["-50.0", "0", "nan", "inf", "-inf"]:
        monkeypatch.setenv("URBAN_REGISTRY_TTL_SECONDS", invalid_ttl)
        reg_inv = BoundedRunRegistry()
        assert reg_inv.ttl_seconds == 86400.0

    # 6. Valid float ttl_seconds
    monkeypatch.setenv("URBAN_REGISTRY_TTL_SECONDS", "3600.5")
    reg6 = BoundedRunRegistry()
    assert reg6.ttl_seconds == 3600.5

    # 7. Direct invalid argument passing to constructor -> fallback
    reg_direct = BoundedRunRegistry(max_entries=-5, ttl_seconds=-1.0)
    assert reg_direct.max_entries == 100
    assert reg_direct.ttl_seconds == 86400.0

    # 8. Capacity boundary and safe popitem on empty dict
    reg_edge = BoundedRunRegistry(max_entries=1)
    assert len(reg_edge) == 0
    reg_edge.set("run_a", {"status": "ok"})
    assert len(reg_edge) == 1
    reg_edge.set("run_b", {"status": "ok"})
    assert len(reg_edge) == 1
    assert "run_a" not in reg_edge
    assert "run_b" in reg_edge

    reset_api_state()



def test_api_authentication_timing_safe(monkeypatch):
    """Validates constant-time API key verification with Unicode and timing-safe comparison."""
    from urban_resilience.api import reset_api_state
    reset_api_state()


    secret = "secret-token-key-2026-safe"
    monkeypatch.setenv("URBAN_API_KEY", secret)

    payload = {
        "city_name": "Shenzhen",
        "hazard_type": "rainstorm",
        "event_start_hour": 8,
        "duration_hours": 6,
        "peak_intensity": 0.5,
        "total_hours": 24,
        "monte_carlo_samples": 5,
    }

    # Partial prefix match must fail
    res_prefix = client.post(
        "/api/v1/resilience/simulate",
        json=payload,
        headers={"X-API-Key": "secret-token"},
    )
    assert res_prefix.status_code == 401

    # Exact match succeeds
    res_exact = client.post(
        "/api/v1/resilience/simulate",
        json=payload,
        headers={"X-API-Key": secret},
    )
    assert res_exact.status_code == 200

    reset_api_state()


def test_api_provenance_untrained_fallback_no_warning():
    """Validates that untrained fallback execution returns None for synthetic_warning and provenance_warning."""
    payload = {
        "city_name": "Shenzhen",
        "hazard_type": "rainstorm",
        "event_start_hour": 8,
        "duration_hours": 6,
        "peak_intensity": 0.5,
        "total_hours": 24,
        "monte_carlo_samples": 5,
    }
    res = client.post("/api/v1/resilience/simulate", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["prediction_provenance"]["model_status"] == "untrained_fallback"
    assert data.get("synthetic_warning") is None
    assert data.get("provenance_warning") is None


def test_api_provenance_synthetic_trained_warning(monkeypatch):
    """Validates that synthetic trained model outputs proper synthetic warning and provenance warning."""
    from urban_resilience.orchestration.supervisor import SupervisorAgent
    from urban_resilience.types import PredictionProvenance

    orig_run = SupervisorAgent.run

    def mock_run(self, *args, **kwargs):
        pred, outlook, zf, manifest = orig_run(self, *args, **kwargs)
        pred.provenance = PredictionProvenance(
            summary_metric_source="trained_model",
            model_status="trained_model",
            training_label_source="synthetic_simulator",
            engineering_only=True,
        )
        return pred, outlook, zf, manifest

    monkeypatch.setattr(SupervisorAgent, "run", mock_run)

    payload = {
        "city_name": "Shenzhen",
        "hazard_type": "rainstorm",
        "event_start_hour": 8,
        "duration_hours": 6,
        "peak_intensity": 0.5,
        "total_hours": 24,
        "monte_carlo_samples": 5,
    }
    res = client.post("/api/v1/resilience/simulate", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["prediction_provenance"]["model_status"] == "trained_model"
    assert data.get("synthetic_warning") is not None
    assert "synthetic simulator" in data["synthetic_warning"].lower()
    assert data.get("provenance_warning") is not None


def test_api_provenance_unverified_trained_warning(monkeypatch):
    """Validates that trained model with unverified real provenance displays warning."""
    from urban_resilience.orchestration.supervisor import SupervisorAgent
    from urban_resilience.types import PredictionProvenance

    orig_run = SupervisorAgent.run

    def mock_run(self, *args, **kwargs):
        pred, outlook, zf, manifest = orig_run(self, *args, **kwargs)
        pred.provenance = PredictionProvenance(
            summary_metric_source="trained_model",
            model_status="trained_model",
            training_label_source="unknown",
            engineering_only=False,
        )
        return pred, outlook, zf, manifest

    monkeypatch.setattr(SupervisorAgent, "run", mock_run)

    payload = {
        "city_name": "Shenzhen",
        "hazard_type": "rainstorm",
        "event_start_hour": 8,
        "duration_hours": 6,
        "peak_intensity": 0.5,
        "total_hours": 24,
        "monte_carlo_samples": 5,
    }
    res = client.post("/api/v1/resilience/simulate", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["prediction_provenance"]["model_status"] == "trained_model"
    assert data.get("synthetic_warning") is None
    assert data.get("provenance_warning") is not None
    assert "unverified" in data["provenance_warning"].lower()
