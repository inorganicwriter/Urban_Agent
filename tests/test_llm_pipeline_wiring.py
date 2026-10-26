"""Tests for FIX-4: Pipeline, CLI, and API orchestration_mode integration and no-silent-fallback enforcement."""

import json
from pathlib import Path
import pytest
from starlette.testclient import TestClient

from urban_resilience.api import app, set_injected_llm_client
from urban_resilience.cli import main as cli_main
from urban_resilience.config import SimulationConfig
from urban_resilience.llm.client import MockLLMClient, get_llm_client
from urban_resilience.pipeline import UrbanResiliencePipeline


def test_pipeline_llm_hybrid_with_client_runs_full(tmp_path: Path):
    """Pipeline configured with orchestration_mode='llm_hybrid' and a client runs successfully."""
    mock_llm = MockLLMClient()

    def responder(messages, schema):
        user_msg = messages[-1]["content"] if messages else ""
        if "Review:" in user_msg:
            return {"action": "stop", "reasoning": "Sufficient evidence in round 1"}

        system_content = messages[0]["content"] if messages else ""
        if "temporal" in system_content.lower():
            tool_name = "compute_hourly_entropy"
        elif "flexibility" in system_content.lower():
            tool_name = "compute_peak_shift_elasticity"
        elif "predictability" in system_content.lower():
            tool_name = "compute_autocorr_lags"
        elif "spatial" in system_content.lower():
            tool_name = "compute_destination_diversity"
        elif "functional" in system_content.lower():
            tool_name = "compute_poi_diversity_metrics"
        else:
            tool_name = "compute_hourly_entropy"

        if "Begin ReAct assessment" in user_msg:
            return {
                "action": "call_tool",
                "reasoning": f"Calling tool {tool_name}",
                "tool_call": {"call_id": "c1", "tool": tool_name, "args": {}},
            }
        return {"action": "final", "feature_report": None}

    mock_llm.set_response_generator(responder)

    sim_cfg = SimulationConfig(
        total_hours=48,
        event_start_hour=12,
        event_duration_hours=6,
        monte_carlo_samples=10,
    )
    pipeline = UrbanResiliencePipeline(
        city_name="Shenzhen",
        sim_config=sim_cfg,
        output_dir=tmp_path,
        orchestration_mode="llm_hybrid",
        llm_client=mock_llm,
    )

    results = pipeline.run_full_pipeline()
    assert results["manifest"].orchestration_mode == "llm_hybrid"
    assert results["manifest"].llm_model == "mock-model-v1"
    assert results["prediction"] is not None


def test_pipeline_llm_hybrid_without_client_raises_value_error(tmp_path: Path):
    """Pipeline configured with llm_hybrid but no llm_client must explicitly raise ValueError."""
    sim_cfg = SimulationConfig(total_hours=48, event_start_hour=12, event_duration_hours=6)
    pipeline = UrbanResiliencePipeline(
        city_name="Shenzhen",
        sim_config=sim_cfg,
        output_dir=tmp_path,
        orchestration_mode="llm_hybrid",
        llm_client=None,
    )

    with pytest.raises(ValueError) as exc_info:
        pipeline.run_full_pipeline()

    assert "llm_hybrid orchestration mode requires a valid llm_client" in str(exc_info.value)


def test_cli_orchestration_mode_flag(capsys):
    """CLI accepts --orchestration-mode and executes properly."""
    # Use small simulation window to run fast
    exit_code = cli_main([
        "--start-hour", "12",
        "--duration", "6",
        "--total-hours", "36",
        "--mc-samples", "5",
        "--orchestration-mode", "deterministic",
        "--json-output",
    ])
    assert exit_code == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["city"] == "Shenzhen"
    assert "indicators" in payload


def test_api_orchestration_mode_in_request(monkeypatch):
    """API simulation request accepts orchestration_mode and reflects it in response."""
    # Mock authentication header or disable key requirement
    monkeypatch.delenv("URBAN_API_KEY", raising=False)
    client = TestClient(app)

    req_body = {
        "city_name": "Shenzhen",
        "hazard_type": "rainstorm",
        "event_start_hour": 12,
        "duration_hours": 6,
        "total_hours": 36,
        "monte_carlo_samples": 5,
        "orchestration_mode": "deterministic",
    }

    resp = client.post("/api/v1/resilience/simulate", json=req_body)
    assert resp.status_code == 200
    data = resp.json()
    assert data["orchestration_mode"] == "deterministic"
    assert "run_id" in data
    assert "manifest_path" in data


def test_api_llm_hybrid_with_mock_client_succeeds(monkeypatch):
    """API simulation under llm_hybrid returns 200 when mock client is configured."""
    monkeypatch.delenv("URBAN_API_KEY", raising=False)
    monkeypatch.setenv("URBAN_LLM_MODE", "mock")
    client = TestClient(app)

    req_body = {
        "city_name": "Shenzhen",
        "hazard_type": "rainstorm",
        "event_start_hour": 12,
        "duration_hours": 6,
        "total_hours": 36,
        "monte_carlo_samples": 5,
        "orchestration_mode": "llm_hybrid",
    }

    resp = client.post("/api/v1/resilience/simulate", json=req_body)
    assert resp.status_code == 200
    data = resp.json()
    assert data["orchestration_mode"] == "llm_hybrid"
    assert data["status"] == "success"


def test_api_llm_hybrid_without_key_returns_422(monkeypatch):
    """API simulation under llm_hybrid returns 422 when no key is configured."""
    monkeypatch.delenv("URBAN_API_KEY", raising=False)
    monkeypatch.delenv("URBAN_LLM_API_KEY", raising=False)
    monkeypatch.delenv("URBAN_LLM_MODE", raising=False)
    set_injected_llm_client(None)
    client = TestClient(app)

    req_body = {
        "city_name": "Shenzhen",
        "hazard_type": "rainstorm",
        "event_start_hour": 12,
        "duration_hours": 6,
        "total_hours": 36,
        "monte_carlo_samples": 5,
        "orchestration_mode": "llm_hybrid",
    }

    resp = client.post("/api/v1/resilience/simulate", json=req_body)
    assert resp.status_code == 422
    assert "URBAN_LLM_API_KEY" in resp.json()["detail"]

    # Also check async submit endpoint
    resp_async = client.post("/api/v1/resilience/submit", json=req_body)
    assert resp_async.status_code == 422
    assert "URBAN_LLM_API_KEY" in resp_async.json()["detail"]


def test_cli_llm_hybrid_with_mock_env(monkeypatch):
    """CLI under llm_hybrid executes with exit code 0 when URBAN_LLM_MODE=mock."""
    monkeypatch.setenv("URBAN_LLM_MODE", "mock")
    exit_code = cli_main([
        "--start-hour", "12",
        "--duration", "6",
        "--total-hours", "36",
        "--mc-samples", "5",
        "--orchestration-mode", "llm_hybrid",
        "--json-output",
    ])
    assert exit_code == 0


def test_cli_llm_hybrid_without_key_exits_cleanly(monkeypatch, capsys):
    """CLI under llm_hybrid exits with status code 2 and a clean error message when no key is provided."""
    monkeypatch.delenv("URBAN_LLM_API_KEY", raising=False)
    monkeypatch.delenv("URBAN_LLM_MODE", raising=False)

    with pytest.raises(SystemExit) as exc_info:
        cli_main([
            "--orchestration-mode", "llm_hybrid",
        ])

    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert "URBAN_LLM_API_KEY environment variable is required" in captured.err


def test_get_llm_client_factory():
    """Validates get_llm_client factory function across modes."""
    mock_c = get_llm_client(mode="mock", model="custom-mock")
    assert isinstance(mock_c, MockLLMClient)
    assert mock_c.model == "custom-mock"

    with pytest.raises(ValueError) as exc:
        get_llm_client(mode="invalid_unknown_mode")
    assert "Unknown LLM client mode" in str(exc.value)
