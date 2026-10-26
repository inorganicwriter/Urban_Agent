"""Unit tests for prediction provenance helpers and audit validators."""

import pytest
from urban_resilience.prediction.provenance import (
    build_provenance_warnings,
    get_provenance_warning,
    get_synthetic_warning,
    is_synthetic_trained,
    is_untrained_fallback,
    validate_verified_data_provenance,
)
from urban_resilience.types import PredictionProvenance


def test_is_untrained_fallback():
    assert is_untrained_fallback(None) is True
    assert is_untrained_fallback(PredictionProvenance(summary_metric_source="monte_carlo_physics", model_status="untrained_fallback")) is True
    assert is_untrained_fallback({"model_status": "untrained_fallback"}) is True
    assert is_untrained_fallback(PredictionProvenance(summary_metric_source="trained_model", model_status="trained_model")) is False


def test_is_synthetic_trained():
    assert is_synthetic_trained(None) is False
    assert is_synthetic_trained({"model_status": "untrained_fallback", "engineering_only": True}) is False
    assert is_synthetic_trained({
        "model_status": "trained_model",
        "training_label_source": "synthetic_simulator",
        "engineering_only": True,
    }) is True
    assert is_synthetic_trained(PredictionProvenance(
        summary_metric_source="trained_model",
        model_status="trained_model",
        training_label_source="synthetic_simulator",
        engineering_only=True,
    )) is True
    # Untagged or non-synthetic
    assert is_synthetic_trained({
        "model_status": "trained_model",
        "training_label_source": "observed",
        "engineering_only": False,
    }) is False


def test_build_provenance_warnings():
    # Untrained fallback produces no warning
    assert build_provenance_warnings(None) == []
    assert build_provenance_warnings({"model_status": "untrained_fallback", "engineering_only": True}) == []

    # Synthetic trained produces synthetic warning
    synth_warnings = build_provenance_warnings({
        "model_status": "trained_model",
        "training_label_source": "synthetic_simulator",
        "engineering_only": True,
    })
    assert len(synth_warnings) == 1
    assert "synthetic simulator" in synth_warnings[0].lower()
    assert get_synthetic_warning({
        "model_status": "trained_model",
        "training_label_source": "synthetic_simulator",
        "engineering_only": True,
    }) is not None

    # Trained but unverified produces provenance warning
    unver_warnings = build_provenance_warnings({
        "model_status": "trained_model",
        "training_label_source": "observed",
        "provenance_status": "unverified",
        "engineering_only": False,
    })
    assert len(unver_warnings) == 1
    assert "unverified" in unver_warnings[0].lower()
    assert get_provenance_warning({
        "model_status": "trained_model",
        "training_label_source": "observed",
        "provenance_status": "unverified",
        "engineering_only": False,
    }) is not None


def test_validate_verified_data_provenance_suite():
    valid = {
        "data_kind": "observed",
        "provenance_status": "verified",
        "event_id": "SZ_20230907_HAIKUI",
        "source": "Observational Station",
        "license": "CC-BY-4.0",
        "data_hash": "f" * 64,
    }
    validate_verified_data_provenance(valid)

    # Empty / non-dict
    with pytest.raises(ValueError, match="non-empty dictionary"):
        validate_verified_data_provenance({})
    with pytest.raises(ValueError, match="non-empty dictionary"):
        validate_verified_data_provenance("not a dict")

    # Invalid data_kind
    with pytest.raises(ValueError, match="data_kind"):
        validate_verified_data_provenance({**valid, "data_kind": "model"})

    # Invalid provenance_status
    with pytest.raises(ValueError, match="provenance_status"):
        validate_verified_data_provenance({**valid, "provenance_status": "pending"})

    # Missing / whitespace fields
    for k in ("event_id", "source", "license"):
        with pytest.raises(ValueError, match=k):
            validate_verified_data_provenance({**valid, k: "  "})

    # Invalid SHA-256
    with pytest.raises(ValueError, match="64-character hexadecimal"):
        validate_verified_data_provenance({**valid, "data_hash": "abc"})
