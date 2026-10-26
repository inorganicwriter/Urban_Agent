"""Tests for the four-group ablation experiment runner."""

from pathlib import Path
import pytest

from urban_resilience.config import SimulationConfig
from urban_resilience.data.event_intelligence import EventIntelligenceAgent
from urban_resilience.data.mobility_loader import MobilityNetworkLoader
from urban_resilience.data.spatial_loader import UrbanSpatialDataLoader
from urban_resilience.experiments.runner import (
    run_experiment_group,
    run_four_group_experiments,
)
from urban_resilience.llm.client import MockLLMClient
from urban_resilience.types import HazardType


@pytest.fixture
def sim_env():
    spatial_loader = UrbanSpatialDataLoader(city_name="Shenzhen")
    ctx = spatial_loader.load_context()
    loader = MobilityNetworkLoader(urban_context=ctx, random_seed=42)
    mob = loader.generate_mobility_dataset(total_hours=24)
    agent = EventIntelligenceAgent(urban_context=ctx)
    evt = agent.create_event_context(
        hazard_type=HazardType.RAIN_STORM,
        event_start_hour=6,
        duration_hours=4,
        peak_intensity=0.7,
        total_simulation_hours=24,
    )
    cfg = SimulationConfig(
        total_hours=24,
        event_start_hour=6,
        event_duration_hours=4,
        monte_carlo_samples=3,
    )
    return ctx, mob, evt, cfg


def test_run_fixed_features_experiment_group(tmp_path: Path, sim_env):
    ctx, mob, evt, cfg = sim_env
    res = run_experiment_group(
        group="fixed_features",
        urban_context=ctx,
        mobility_network=mob,
        event_context=evt,
        output_dir=tmp_path / "fixed",
        sim_config=cfg,
        engineering_only=True,
    )
    assert res.group == "fixed_features"
    assert res.status in ("SUCCESS", "DEGRADED")
    assert res.engineering_only is True
    assert res.follow_up_count == 0
    assert 0.0 <= res.city_max_loss <= 1.0
    assert res.city_recovery_time_hours >= 0.0


def test_single_adaptive_uses_only_temporal_llm(tmp_path: Path, sim_env):
    """Verifies that single_adaptive instantiates LLM subagent only for temporal domain."""
    from urban_resilience.orchestration.react_subagent import (
        BaseLLMReActSubagent,
        TemporalSubagentLLM,
    )
    from urban_resilience.orchestration.subagent import FlexibilitySubagent
    from urban_resilience.orchestration.supervisor import SupervisorAgent

    ctx, mob, evt, cfg = sim_env
    mock_llm = MockLLMClient()

    sup = SupervisorAgent(
        urban_context=ctx,
        mobility_network=mob,
        orchestration_mode="llm_hybrid",
        llm_client=mock_llm,
        llm_domain_keys={"temporal"},
        output_dir=tmp_path / "supervisor_test",
        sim_config=cfg,
    )
    assert isinstance(sup.registry.get("temporal"), (BaseLLMReActSubagent, TemporalSubagentLLM))
    assert isinstance(sup.registry.get("flexibility"), FlexibilitySubagent)
    assert not isinstance(sup.registry.get("flexibility"), BaseLLMReActSubagent)
    assert not isinstance(sup.registry.get("spatial"), BaseLLMReActSubagent)
    assert not isinstance(sup.registry.get("functional"), BaseLLMReActSubagent)
    assert not isinstance(sup.registry.get("predictability"), BaseLLMReActSubagent)

    res = run_experiment_group(
        group="single_adaptive",
        urban_context=ctx,
        mobility_network=mob,
        event_context=evt,
        llm_client=mock_llm,
        output_dir=tmp_path / "adaptive",
        sim_config=cfg,
        engineering_only=True,
    )
    assert res.group == "single_adaptive"
    assert res.status in ("SUCCESS", "DEGRADED")
    assert res.engineering_only is True
    assert res.follow_up_count == 0
    assert 0.0 <= res.city_max_loss <= 1.0


def test_run_four_group_experiments_suite(tmp_path: Path, sim_env):
    ctx, mob, evt, cfg = sim_env
    mock_llm = MockLLMClient(auto_fallback=True)

    suite = run_four_group_experiments(
        urban_context=ctx,
        mobility_network=mob,
        event_context=evt,
        llm_client=mock_llm,
        output_dir=tmp_path / "suite",
        sim_config=cfg,
        engineering_only=True,
    )

    assert len(suite.results) == 4
    assert set(suite.results.keys()) == {
        "fixed_features",
        "single_adaptive",
        "multi_uncoordinated",
        "multi_coordinated",
    }
    assert len(suite.summary_df) == 4
    assert suite.engineering_only is True
    assert (tmp_path / "suite" / "experiment_summary.csv").exists()

    # Coordination verification
    assert suite.results["multi_coordinated"].follow_up_count > 0
    assert suite.results["multi_coordinated"].follow_up_executed is True
    assert suite.results["multi_coordinated"].coordination_status in ("exercised", "followed_up_no_effect")
    if suite.results["multi_coordinated"].coordination_exercised:
        assert suite.results["multi_coordinated"].evidence_changed or suite.results["multi_coordinated"].zone_features_changed
    assert suite.results["multi_uncoordinated"].follow_up_count == 0
    assert suite.results["multi_uncoordinated"].coordination_status == "not_attempted"
    assert suite.results["multi_uncoordinated"].coordination_exercised is False

    md = suite.to_markdown()
    assert "# Urban Resilience Orchestration - Four-Group Ablation Experiment" in md
    assert "Engineering Only" in md
    assert "fixed_features" in md
    assert "multi_coordinated" in md
    assert "followed_up_no_effect" in md or "exercised" in md


def test_multi_seed_four_group_experiments_with_statistics(tmp_path: Path, sim_env):
    """Verifies that multi-seed runner runs 4 groups per seed and computes mean/std/95% CI."""
    ctx, mob, evt, cfg = sim_env
    mock_llm = MockLLMClient(auto_fallback=True)

    suite = run_four_group_experiments(
        urban_context=ctx,
        event_seeds=[42, 43],
        llm_client=mock_llm,
        output_dir=tmp_path / "multi_seed",
        sim_config=cfg,
        engineering_only=True,
    )

    # 1. Consolidated structures
    assert len(suite.summary_df) == 4
    assert suite.raw_df is not None
    assert len(suite.raw_df) == 8  # 2 seeds * 4 groups

    # 2. Exactly 4 groups per seed
    for s in [42, 43]:
        sub = suite.raw_df[suite.raw_df["seed"] == s]
        assert len(sub) == 4
        assert set(sub["group"].values) == {
            "fixed_features",
            "single_adaptive",
            "multi_uncoordinated",
            "multi_coordinated",
        }

    # 3. Aggregated table contains mean, std, and 95% CI
    required_stat_cols = [
        "city_max_loss_mean",
        "city_max_loss_std",
        "city_max_loss_ci95_lower",
        "city_max_loss_ci95_upper",
        "city_recovery_time_hours_mean",
        "city_recovery_time_hours_std",
        "city_recovery_time_hours_ci95_lower",
        "city_recovery_time_hours_ci95_upper",
    ]
    for col in required_stat_cols:
        assert col in suite.summary_df.columns

    # 4. engineering_only = True preserved
    assert suite.engineering_only is True
    assert suite.summary_df["engineering_only"].all()
    assert suite.raw_df["engineering_only"].all()

    # 5. Output files written
    assert (tmp_path / "multi_seed" / "experiment_summary.csv").exists()
    assert (tmp_path / "multi_seed" / "experiment_raw_runs.csv").exists()
    assert (tmp_path / "multi_seed" / "experiment_report.md").exists()

    md = suite.to_markdown()
    assert "Mean ± Std [95% CI]" in md
    assert "engineering_only = True" in md


def test_single_adaptive_domain_configurations_temporal_and_spatial(tmp_path: Path, sim_env):
    """Verifies that single_adaptive_domain can be configured to temporal or spatial."""
    from urban_resilience.orchestration.react_subagent import BaseLLMReActSubagent
    from urban_resilience.orchestration.subagent import (
        FlexibilitySubagent,
        SpatialSubagent,
        TemporalSubagent,
    )
    from urban_resilience.orchestration.supervisor import SupervisorAgent

    ctx, mob, evt, cfg = sim_env
    mock_llm = MockLLMClient()

    # 1. Test temporal configuration
    sup_temp = SupervisorAgent(
        urban_context=ctx,
        mobility_network=mob,
        orchestration_mode="llm_hybrid",
        llm_client=mock_llm,
        llm_domain_keys={"temporal"},
        output_dir=tmp_path / "temp_sup",
        sim_config=cfg,
    )
    assert isinstance(sup_temp.registry.get("temporal"), BaseLLMReActSubagent)
    assert isinstance(sup_temp.registry.get("spatial"), SpatialSubagent)
    assert not isinstance(sup_temp.registry.get("spatial"), BaseLLMReActSubagent)

    # 2. Test spatial configuration
    sup_spat = SupervisorAgent(
        urban_context=ctx,
        mobility_network=mob,
        orchestration_mode="llm_hybrid",
        llm_client=mock_llm,
        llm_domain_keys={"spatial"},
        output_dir=tmp_path / "spat_sup",
        sim_config=cfg,
    )
    assert isinstance(sup_spat.registry.get("spatial"), BaseLLMReActSubagent)
    assert isinstance(sup_spat.registry.get("temporal"), TemporalSubagent)
    assert not isinstance(sup_spat.registry.get("temporal"), BaseLLMReActSubagent)
    assert isinstance(sup_spat.registry.get("flexibility"), FlexibilitySubagent)

    # 3. Test running single_adaptive group with spatial configuration
    res = run_experiment_group(
        group="single_adaptive",
        urban_context=ctx,
        mobility_network=mob,
        event_context=evt,
        llm_client=mock_llm,
        single_adaptive_domain="spatial",
        output_dir=tmp_path / "adaptive_spatial",
        sim_config=cfg,
        engineering_only=True,
    )
    assert res.group == "single_adaptive"
    assert res.status in ("SUCCESS", "DEGRADED")
    assert res.details.get("single_adaptive_domain") == "spatial"
    assert res.engineering_only is True


def test_token_usage_per_run_isolation(tmp_path: Path, sim_env):
    """Verifies that tokens_used is strictly computed per run as (after - before) and does not accumulate across runs."""
    ctx, mob, evt, cfg = sim_env
    mock_llm = MockLLMClient(auto_fallback=True)

    # Run 1: single_adaptive
    res1 = run_experiment_group(
        group="single_adaptive",
        urban_context=ctx,
        mobility_network=mob,
        event_context=evt,
        llm_client=mock_llm,
        output_dir=tmp_path / "run1",
        sim_config=cfg,
        engineering_only=True,
    )
    assert res1.tokens_used > 0
    assert res1.tokens_used == res1.token_usage_after - res1.token_usage_before

    # Run 2: multi_uncoordinated reusing the same mock_llm client
    res2 = run_experiment_group(
        group="multi_uncoordinated",
        urban_context=ctx,
        mobility_network=mob,
        event_context=evt,
        llm_client=mock_llm,
        output_dir=tmp_path / "run2",
        sim_config=cfg,
        engineering_only=True,
    )
    # Second run's tokens_used must equal its delta, NOT the cumulative client tokens
    assert res2.tokens_used == res2.token_usage_after - res2.token_usage_before
    assert res2.tokens_used < mock_llm.token_usage["total_tokens"]
    assert res2.token_usage_before == res1.token_usage_after

    # Run multi-seed suite and verify raw_df delta equality
    suite = run_four_group_experiments(
        urban_context=ctx,
        event_seeds=[42, 43],
        llm_client=mock_llm,
        output_dir=tmp_path / "suite_tokens",
        sim_config=cfg,
        engineering_only=True,
    )
    for _, row in suite.raw_df.iterrows():
        if row["group"] != "fixed_features":
            assert row["tokens_used"] == row["token_usage_after"] - row["token_usage_before"]
        else:
            assert row["tokens_used"] == 0

    # Ensure CI lower bounds for tokens are non-negative
    for _, row in suite.summary_df.iterrows():
        assert row["tokens_used_ci95_lower"] >= 0


def test_multi_coordinated_workflow_and_truthful_status(tmp_path: Path, sim_env):
    """Verifies multi-agent coordination workflow execution and truthful recording of status without assuming bus route change."""
    ctx, mob, evt, cfg = sim_env
    mock_llm = MockLLMClient(auto_fallback=True)

    suite = run_four_group_experiments(
        urban_context=ctx,
        event_seeds=[42],
        llm_client=mock_llm,
        output_dir=tmp_path / "suite_coord",
        sim_config=cfg,
        engineering_only=True,
    )

    coord_res = suite.results["multi_coordinated"]
    uncoord_res = suite.results["multi_uncoordinated"]

    # 1. Follow-up count and coordination exercised
    assert coord_res.follow_up_count > 0
    assert coord_res.follow_up_executed is True
    assert coord_res.coordination_status in ("exercised", "followed_up_no_effect")
    if coord_res.coordination_exercised:
        assert coord_res.evidence_changed or coord_res.zone_features_changed
    assert uncoord_res.follow_up_count == 0
    assert uncoord_res.coordination_status == "not_attempted"
    assert uncoord_res.coordination_exercised is False

    # 2. Stop reason is not token budget exhausted
    assert coord_res.stop_reason is not None
    assert "token budget exhausted" not in coord_res.stop_reason.lower()

    # 3. Manifest contains follow-up summary
    manifest_fu = coord_res.details.get("follow_up_summary", [])
    assert len(manifest_fu) > 0
    assert manifest_fu[0]["agent_name"] == "temporal"

    # 4. Report markdown reflects coordination status
    md = suite.to_markdown()
    assert "followed_up_no_effect" in md or "exercised" in md
    assert "not_attempted" in md


def test_single_adaptive_domain_unknown_raises(sim_env):
    """Verifies that unknown domain keys raise ValueError across all entry points."""
    ctx, mob, evt, cfg = sim_env
    from urban_resilience.orchestration.supervisor import SupervisorAgent

    # 1. run_experiment_group raises
    with pytest.raises(ValueError) as exc1:
        run_experiment_group(
            group="single_adaptive",
            urban_context=ctx,
            mobility_network=mob,
            event_context=evt,
            single_adaptive_domain="quantum_domain",
            sim_config=cfg,
        )
    assert "unsupported single_adaptive_domain" in str(exc1.value).lower()

    # 2. run_four_group_experiments raises
    with pytest.raises(ValueError) as exc2:
        run_four_group_experiments(
            urban_context=ctx,
            single_adaptive_domain="astral_domain",
            sim_config=cfg,
        )
    assert "unsupported single_adaptive_domain" in str(exc2.value).lower()

    # 3. SupervisorAgent raises on invalid llm_domain_keys
    with pytest.raises(ValueError) as exc3:
        SupervisorAgent(
            urban_context=ctx,
            mobility_network=mob,
            orchestration_mode="llm_hybrid",
            llm_client=MockLLMClient(),
            llm_domain_keys={"temporal", "unknown_magic_domain"},
            sim_config=cfg,
        )
    assert "unsupported llm domain" in str(exc3.value).lower()


def test_engineering_only_enforcement(sim_env):
    """Verifies that setting engineering_only=False raises ValueError unless verified provenance is provided."""
    ctx, _, _, cfg = sim_env
    from urban_resilience.prediction.synthetic_protocol import (
        build_synthetic_dataset,
        train_and_calibrate_synthetic_predictor,
    )

    # 1. build_synthetic_dataset raises when engineering_only=False
    with pytest.raises(ValueError) as exc1:
        build_synthetic_dataset(
            train_seeds=[501],
            calibration_seeds=[502],
            test_seeds=[503],
            sim_config=cfg,
            engineering_only=False,
        )
    assert "engineering_only must be true" in str(exc1.value).lower()

    # 2. train_and_calibrate_synthetic_predictor raises when bundle has engineering_only=False
    bundle = build_synthetic_dataset(
        train_seeds=[501],
        calibration_seeds=[502],
        test_seeds=[503],
        sim_config=cfg,
        engineering_only=True,
    )
    bundle.engineering_only = False
    with pytest.raises(ValueError) as exc2:
        train_and_calibrate_synthetic_predictor(bundle)
    assert "engineering_only=true" in str(exc2.value).lower()

    # 3. run_four_group_experiments raises when engineering_only=False (even with verified_data_provenance)
    with pytest.raises(ValueError) as exc3:
        run_four_group_experiments(
            urban_context=ctx,
            event_seeds=[42],
            sim_config=cfg,
            engineering_only=False,
        )
    assert "real empirical validation is blocked" in str(exc3.value).lower()

    valid_provenance = {
        "data_kind": "observed",
        "provenance_status": "verified",
        "event_id": "SZ_20230907_HAIKUI",
        "source": "Shenzhen Meteorological Bureau",
        "license": "Open Data License v1.0",
        "data_hash": "a" * 64,
    }
    with pytest.raises(ValueError) as exc3_prov:
        run_four_group_experiments(
            urban_context=ctx,
            event_seeds=[42],
            sim_config=cfg,
            engineering_only=False,
            verified_data_provenance=valid_provenance,
        )
    assert "real empirical validation is blocked" in str(exc3_prov.value).lower()

    # 4. run_experiment_group raises when engineering_only=False
    with pytest.raises(ValueError) as exc4:
        run_experiment_group(
            group="fixed_features",
            urban_context=ctx,
            mobility_network=None,  # type: ignore
            event_context=None,  # type: ignore
            sim_config=cfg,
            engineering_only=False,
        )
    assert "real empirical validation is blocked" in str(exc4.value).lower()


def test_controlled_delta_exercises_coordination(tmp_path: Path, sim_env, monkeypatch):
    """Verifies that follow-up with measurable delta updates evidence & zone features and marks coordination exercised."""
    ctx, mob, evt, cfg = sim_env
    import urban_resilience.tools.temporal_tools as tt

    orig_entropy = tt.compute_hourly_entropy
    counter = {"count": 0}

    def mocked_entropy(region_id, context):
        counter["count"] += 1
        if counter["count"] > 12:  # After initial 12 zones analysis
            return {
                "region_id": region_id,
                "hourly_entropy": 0.999,
                "normalized_entropy": 0.999,
            }
        return orig_entropy(region_id, context)

    monkeypatch.setattr(tt, "compute_hourly_entropy", mocked_entropy)

    mock_llm = MockLLMClient(auto_fallback=True)
    res = run_experiment_group(
        group="multi_coordinated",
        urban_context=ctx,
        mobility_network=mob,
        event_context=evt,
        llm_client=mock_llm,
        output_dir=tmp_path / "controlled_coord",
        sim_config=cfg,
        engineering_only=True,
    )

    assert res.follow_up_executed is True
    assert res.evidence_changed is True
    assert res.zone_features_changed is True
    assert res.coordination_status == "exercised"
    assert res.coordination_exercised is True


def test_failed_followup_marks_follow_up_failed(sim_env):
    """Verifies that reviewer delegation to a failing follow-up marks follow_up_failed and keeps evidence unpolluted."""
    ctx, mob, evt, cfg = sim_env
    from urban_resilience.llm.schemas import PlannerDecision
    from urban_resilience.orchestration.supervisor import SupervisorAgent

    mock_llm = MockLLMClient(auto_fallback=True)

    supervisor = SupervisorAgent(
        urban_context=ctx,
        mobility_network=mob,
        orchestration_mode="llm_hybrid",
        llm_client=mock_llm,
        llm_domain_keys={"temporal"},
        enable_follow_ups=True,
        sim_config=cfg,
        strict=False,
        allow_degraded=True,
    )

    def forced_failed_delegate(*args, **kwargs):
        if len(supervisor.follow_up_controller.follow_up_logs) == 0:
            return PlannerDecision(
                action="delegate",
                reasoning="Forced failed delegation",
                tasks=[{"agent_name": "unknown_subagent", "action": "unknown_action", "payload": {"region_id": "SZ_FT_CBD"}}],
            )
        return PlannerDecision(action="stop", reasoning="Done")

    supervisor.llm_reviewer.review_and_decide = forced_failed_delegate

    pred, outlook, zf, manifest = supervisor.run(event_context=evt, save_manifest=False)
    meta = manifest.metadata or {}
    assert meta.get("follow_up_executed") is True
    assert meta.get("follow_up_success_count") == 0
    assert meta.get("coordination_status") == "follow_up_failed"
    assert meta.get("coordination_exercised") is False
    assert meta.get("evidence_changed") is False
    assert meta.get("zone_features_changed") is False


def test_fixed_features_zero_tokens(tmp_path: Path, sim_env):
    """Verifies that fixed_features reports 0 tokens before, after, and used."""
    ctx, mob, evt, cfg = sim_env
    res = run_experiment_group(
        group="fixed_features",
        urban_context=ctx,
        mobility_network=mob,
        event_context=evt,
        output_dir=tmp_path / "fixed_tok",
        sim_config=cfg,
        engineering_only=True,
    )
    assert res.token_usage_before == 0
    assert res.token_usage_after == 0
    assert res.tokens_used == 0


def test_results_by_seed_multi_seed_tracking(tmp_path: Path, sim_env):
    """Verifies that ExperimentSuiteResult tracks results_by_seed across multiple seeds."""
    ctx, mob, evt, cfg = sim_env
    mock_llm = MockLLMClient(auto_fallback=True)

    suite = run_four_group_experiments(
        urban_context=ctx,
        event_seeds=[42, 43],
        llm_client=mock_llm,
        output_dir=tmp_path / "multi_seed_track",
        sim_config=cfg,
        engineering_only=True,
    )

    assert hasattr(suite, "results_by_seed")
    assert set(suite.results_by_seed.keys()) == {42, 43}
    for seed in (42, 43):
        assert len(suite.results_by_seed[seed]) == 4
        assert set(suite.results_by_seed[seed].keys()) == {
            "fixed_features",
            "single_adaptive",
            "multi_uncoordinated",
            "multi_coordinated",
        }


def test_validate_verified_data_provenance_audit():
    """Verifies rigorous audit validation for verified data provenance records."""
    from urban_resilience.prediction.provenance import validate_verified_data_provenance

    # Valid record
    valid_record = {
        "data_kind": "observed",
        "provenance_status": "verified",
        "event_id": "SZ_20230907_HAIKUI",
        "source": "Shenzhen Meteorological Bureau & Transport Commission",
        "license": "Open Data License v1.0",
        "data_hash": "a" * 64,
    }
    validate_verified_data_provenance(valid_record)

    # Invalid: None or non-dict
    with pytest.raises(ValueError, match="non-empty dictionary"):
        validate_verified_data_provenance(None)

    # Invalid: data_kind != observed
    with pytest.raises(ValueError, match="Invalid data_kind"):
        validate_verified_data_provenance({**valid_record, "data_kind": "synthetic"})

    # Invalid: provenance_status != verified
    with pytest.raises(ValueError, match="Invalid provenance_status"):
        validate_verified_data_provenance({**valid_record, "provenance_status": "unverified"})

    # Invalid: missing required string field
    for field_name in ("event_id", "source", "license"):
        invalid = dict(valid_record)
        invalid[field_name] = "   "
        with pytest.raises(ValueError, match=field_name):
            validate_verified_data_provenance(invalid)

    # Invalid: malformed SHA-256 hash
    with pytest.raises(ValueError, match="64-character hexadecimal"):
        validate_verified_data_provenance({**valid_record, "data_hash": "not_a_valid_hash"})
    with pytest.raises(ValueError, match="64-character hexadecimal"):
        validate_verified_data_provenance({**valid_record, "data_hash": "a" * 63})
    with pytest.raises(ValueError, match="64-character hexadecimal"):
        validate_verified_data_provenance({**valid_record, "data_hash": "g" * 64})


def test_is_follow_up_success_status_unit():
    """Verifies that is_follow_up_success_status accurately classifies success statuses."""
    from urban_resilience.orchestration.contracts import TaskStatus, is_follow_up_success_status

    # True cases
    assert is_follow_up_success_status(TaskStatus.SUCCESS) is True
    assert is_follow_up_success_status(TaskStatus.DEGRADED) is True
    assert is_follow_up_success_status("SUCCESS") is True
    assert is_follow_up_success_status("DEGRADED") is True
    assert is_follow_up_success_status("refined") is True

    # False cases
    assert is_follow_up_success_status(TaskStatus.FAILED) is False
    assert is_follow_up_success_status("FAILED") is False
    assert is_follow_up_success_status(None) is False
    assert is_follow_up_success_status("unknown") is False
    assert is_follow_up_success_status("PENDING") is False


def test_deterministic_follow_up_refined_status_exercises_coordination(tmp_path: Path, sim_env, monkeypatch):
    """Verifies that deterministic follow-up refined status is counted as success and marks coordination exercised."""
    from urban_resilience.orchestration.follow_up import FollowUpController
    from urban_resilience.orchestration.supervisor import SupervisorAgent

    ctx, mob, evt, cfg = sim_env

    # Monkeypatch identify_follow_ups to return a deterministic follow-up request
    def mocked_identify(self, zone_features):
        return ["Follow-up triggered for test"], [
            {
                "zone_id": ctx.zone_ids[0],
                "target_agent": "flexibility",
                "reason": "Test high vulnerability follow-up refinement",
            }
        ]

    monkeypatch.setattr(FollowUpController, "identify_follow_ups", mocked_identify)

    supervisor = SupervisorAgent(
        urban_context=ctx,
        mobility_network=mob,
        orchestration_mode="deterministic",
        output_dir=tmp_path / "det_fu",
        sim_config=cfg,
        allow_degraded=True,
        enable_follow_ups=True,
    )

    pred, outlook, zf, manifest = supervisor.run(event_context=evt, save_manifest=True)
    meta = manifest.metadata or {}

    assert meta.get("follow_up_executed") is True
    assert meta.get("follow_up_success_count") > 0
    assert meta.get("coordination_status") == "exercised"
    assert meta.get("coordination_exercised") is True
    assert meta.get("zone_features_changed") is True or meta.get("evidence_changed") is True


def test_multi_seed_deliverables_42_43_properties(tmp_path: Path, sim_env):
    """Verifies ExperimentSuiteResult properties and raw_df/summary_df shapes across seeds 42 and 43."""
    ctx, mob, evt, cfg = sim_env
    mock_llm = MockLLMClient(auto_fallback=True)

    suite = run_four_group_experiments(
        urban_context=ctx,
        event_seeds=[42, 43],
        llm_client=mock_llm,
        output_dir=tmp_path / "suite_42_43",
        sim_config=cfg,
        engineering_only=True,
    )

    assert suite.event_seeds == [42, 43]
    assert suite.raw_df is not None
    assert len(suite.raw_df) == 8
    assert (suite.summary_df["n_seeds"] == 2).all()
    assert set(suite.results_by_seed.keys()) == {42, 43}
    for seed in (42, 43):
        assert len(suite.results_by_seed[seed]) == 4
    md = suite.to_markdown()
    assert "[42, 43]" in md


def test_synthetic_artifact_warning_and_labeling():
    """Verifies synthetic artifact warning and labeling in ExperimentSuiteResult and report markdown."""
    import pandas as pd
    from urban_resilience.experiments.runner import ExperimentResult, ExperimentSuiteResult

    # 1. With synthetic model artifact
    dummy_res_synth = ExperimentResult(
        group="fixed_features",
        run_id="run_synth",
        status="SUCCESS",
        execution_time_seconds=1.0,
        city_max_loss=0.2,
        city_recovery_time_hours=12.0,
        city_resilience_score=0.8,
        follow_up_count=0,
        tokens_used=0,
        orchestration_mode="deterministic",
        predictor_label_source="synthetic_simulator",
        predictor_engineering_only=True,
        synthetic_artifact_warning="Model artifact trained on synthetic simulator labels (engineering_only=True). Not an empirical disaster benchmark.",
    )
    summary_df_synth = pd.DataFrame([{
        "group": "fixed_features",
        "orchestration_mode": "deterministic",
        "status": "SUCCESS",
        "n_seeds": 1,
        "engineering_only": True,
        "predictor_label_source": "synthetic_simulator",
        "predictor_engineering_only": True,
        "synthetic_artifact_warning": "Model artifact trained on synthetic simulator labels (engineering_only=True). Not an empirical disaster benchmark.",
    }])
    suite_synth = ExperimentSuiteResult(
        results={"fixed_features": dummy_res_synth},
        summary_df=summary_df_synth,
        event_seeds=[42],
    )
    assert suite_synth.synthetic_artifact_warning is not None
    md_synth = suite_synth.to_markdown()
    assert "**Synthetic Predictor Artifact**" in md_synth
    assert "synthetic_simulator" in md_synth

    # 2. With untrained fallback
    dummy_res_fb = ExperimentResult(
        group="fixed_features",
        run_id="run_fallback",
        status="SUCCESS",
        execution_time_seconds=1.0,
        city_max_loss=0.2,
        city_recovery_time_hours=12.0,
        city_resilience_score=0.8,
        follow_up_count=0,
        tokens_used=0,
        orchestration_mode="deterministic",
        predictor_label_source="none",
        predictor_engineering_only=False,
        synthetic_artifact_warning=None,
    )
    summary_df_fb = pd.DataFrame([{
        "group": "fixed_features",
        "orchestration_mode": "deterministic",
        "status": "SUCCESS",
        "n_seeds": 1,
        "engineering_only": True,
        "predictor_label_source": "none",
        "predictor_engineering_only": False,
        "synthetic_artifact_warning": None,
    }])
    suite_fb = ExperimentSuiteResult(
        results={"fixed_features": dummy_res_fb},
        summary_df=summary_df_fb,
        event_seeds=[42],
    )
    assert suite_fb.synthetic_artifact_warning is None
    md_fb = suite_fb.to_markdown()
    assert "**Synthetic Predictor Artifact**" not in md_fb
