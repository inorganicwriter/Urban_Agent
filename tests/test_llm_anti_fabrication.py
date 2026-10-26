"""Tests for FIX-2: Anti-fabrication verification of FeatureReport against executed ToolResults."""

import pytest

from urban_resilience.llm.schemas import FeatureQuality, FeatureReport, FeatureValue, ToolResult
from urban_resilience.orchestration.react_subagent import (
    DOMAIN_ALLOWED_FEATURES,
    validate_feature_report_against_tools,
)


def make_tool_result(call_id: str, tool_name: str, output: dict, status: str = "success") -> ToolResult:
    return ToolResult(
        call_id=call_id,
        tool=tool_name,
        args={},
        status=status,
        output=output,
        execution_time_seconds=0.01,
    )


def test_anti_fabrication_valid_report():
    """Valid report referencing actual tool results passes with status='complete'."""
    call_id = "call_entropy_01"
    tool_results = {
        call_id: make_tool_result(
            call_id=call_id,
            tool_name="compute_hourly_entropy",
            output={
                "region_id": "SZ_FT_CBD",
                "hourly_entropy": 0.842,
                "uniformity_ratio": 0.450,
            },
        )
    }

    report = FeatureReport(
        agent="temporal",
        region_id="SZ_FT_CBD",
        status="complete",
        features={
            "hourly_entropy": FeatureValue(name="hourly_entropy", value=0.842, evidence_refs=[call_id]),
            "uniformity_ratio": FeatureValue(name="uniformity_ratio", value=0.450, evidence_refs=[call_id]),
        },
        quality=FeatureQuality(data_freshness_hours=0.0),
    )

    validated = validate_feature_report_against_tools(
        report=report,
        executed_call_ids=[call_id],
        tool_results=tool_results,
        required_features={"hourly_entropy"},
        allowed_features=DOMAIN_ALLOWED_FEATURES["temporal"] | {"uniformity_ratio"},
    )

    assert validated.status == "complete"
    assert "hourly_entropy" in validated.features
    assert validated.quality.evidence_coverage == 1.0


def test_anti_fabrication_fabricated_number_degrades_status():
    """LLM hallucinating a number not present in tool output is degraded to insufficient."""
    call_id = "call_entropy_02"
    tool_results = {
        call_id: make_tool_result(
            call_id=call_id,
            tool_name="compute_hourly_entropy",
            output={
                "region_id": "SZ_FT_CBD",
                "hourly_entropy": 0.842,
            },
        )
    }

    # LLM claims hourly_entropy is 0.999 instead of 0.842
    report = FeatureReport(
        agent="temporal",
        region_id="SZ_FT_CBD",
        status="complete",
        features={
            "hourly_entropy": FeatureValue(name="hourly_entropy", value=0.999, evidence_refs=[call_id]),
        },
    )

    validated = validate_feature_report_against_tools(
        report=report,
        executed_call_ids=[call_id],
        tool_results=tool_results,
        required_features={"hourly_entropy"},
        allowed_features=DOMAIN_ALLOWED_FEATURES["temporal"],
    )

    assert validated.status == "insufficient"
    assert any("does not match tool output" in lim for lim in validated.limitations)


def test_anti_fabrication_dangling_call_id_rejected():
    """LLM citing a non-existent or unexecuted call_id is rejected with validation failure."""
    tool_results = {}  # No tools executed

    report = FeatureReport(
        agent="temporal",
        region_id="SZ_FT_CBD",
        status="complete",
        features={
            "hourly_entropy": FeatureValue(name="hourly_entropy", value=0.842, evidence_refs=["non_existent_call_id"]),
        },
    )

    with pytest.raises(ValueError) as exc_info:
        validate_feature_report_against_tools(
            report=report,
            executed_call_ids=[],
            tool_results=tool_results,
            required_features={"hourly_entropy"},
            allowed_features=DOMAIN_ALLOWED_FEATURES["temporal"],
        )

    assert "references unexecuted call_id" in str(exc_info.value)


def test_anti_fabrication_missing_required_features_degrades_status():
    """Omitting required domain features results in status='insufficient'."""
    call_id = "call_quality_01"
    tool_results = {
        call_id: make_tool_result(
            call_id=call_id,
            tool_name="inspect_data_quality",
            output={
                "region_id": "SZ_FT_CBD",
                "missing_rate": 0.0,
            },
        )
    }

    # Missing required 'hourly_entropy'
    report = FeatureReport(
        agent="temporal",
        region_id="SZ_FT_CBD",
        status="complete",
        features={
            "missing_rate": FeatureValue(name="missing_rate", value=0.0, evidence_refs=[call_id]),
        },
    )

    validated = validate_feature_report_against_tools(
        report=report,
        executed_call_ids=[call_id],
        tool_results=tool_results,
        required_features={"hourly_entropy"},
        allowed_features=DOMAIN_ALLOWED_FEATURES["temporal"],
    )

    assert validated.status == "insufficient"
    assert any("missing required features" in lim.lower() for lim in validated.limitations)
