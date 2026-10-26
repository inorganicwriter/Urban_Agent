# Urban Resilience Anticipatory Multi-Agent System

A research prototype for ex-ante urban resilience prediction. It combines a synthetic
Shenzhen mobility dataset, five domain agents under a shared ReAct workflow, evidence
synthesis with uncertainty quantification, and historical-event validation.

## Requirements

- Python 3.11.x (the pinned NumPy 1.26.4 supports Python 3.9–3.12)
- Windows PowerShell 5.1+ or a POSIX shell

## Installation

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

For development and tests, install the extra tools (pytest, ruff, httpx):

```powershell
pip install -e ".[dev]"
```

Runtime and development dependencies are pinned in `requirements.txt` and
`pyproject.toml`.

## Run the full analysis

```powershell
python run_full_city_analysis.py
```

Generated outputs are written to:

- `outputs/disruption_recovery_timeseries.csv`
- `outputs/spatial_priority_rankings.csv`
- `outputs/plots/*.png`
- `outputs/reports/shenzhen_resilience_report.md`
- `outputs/resilience_run.log`

## Command line

```powershell
python -m urban_resilience --help
python -m urban_resilience --city Shenzhen --hazard rainstorm --start-hour 24 --duration 18
```

After `pip install -e .`, the `urban-resilience` command is also available.

## REST API

```powershell
uvicorn urban_resilience.api:app --host 127.0.0.1 --port 8000
```

### Main Endpoints

- `GET /health`: Health check, system readiness, authentication mode, and rate limiting status.
- `GET /api/v1/city/zones`: Spatial context and infrastructure profile for all 12 city zones.
- `POST /api/v1/resilience/simulate`: Triggers multi-agent simulation. Generates a unique `run_id` and stores artifacts in `outputs/api_runs/<run_id>/` (`plots/`, CSVs, and `manifest.json`).
- `GET /api/v1/resilience/runs/{run_id}`: Retrieves execution metadata, metrics, and artifact paths for a specific run.
- `GET /api/v1/resilience/runs/{run_id}/plots/{plot_type}`: Streams publication-quality plots (`curve`, `map`, `dashboard`) from an isolated run.
- `GET /api/v1/resilience/plots/{plot_type}`: *(Deprecated)* Streams plots from the latest simulation run.

Interactive OpenAPI documentation is available at `/docs`.

### Security, Rate Limiting & Memory Management

The API incorporates production-ready operational controls:

1. **Authentication (`URBAN_API_KEY`)**:
   - **Development Mode (Default)**: When `URBAN_API_KEY` is not set or empty, the API operates in open access mode for local development.
   - **Production Authentication**: When `URBAN_API_KEY` is set, all simulation triggers (`/api/v1/resilience/simulate`) and run query endpoints (`/api/v1/resilience/runs/*`) require the `X-API-Key` request header matching the configured token. Missing or invalid keys return HTTP `401 Unauthorized`. Public endpoints (`/health`, `/api/v1/city/zones`) remain accessible without credentials.
2. **Rate Limiting (`URBAN_RATE_LIMIT_PER_MINUTE`)**:
   - Enforces sliding-window rate limiting per client IP (default: 60 requests per minute).
   - Requests exceeding the limit receive HTTP `429 Too Many Requests` with a standard `Retry-After` header.
3. **Bounded Memory Registry (`BoundedRunRegistry`)**:
   - Stores simulation summaries in a thread-safe bounded in-memory cache (`URBAN_REGISTRY_MAX_ENTRIES`, default: 100).
   - Enforces configurable time-to-live expiration (`URBAN_REGISTRY_TTL_SECONDS`, default: 86400s / 24h).
   - Evicts expired and oldest entries (FIFO) to prevent unbounded memory growth during continuous deployment.
   - **Audit Preservation**: Cache eviction or expiration returns HTTP `404 Not Found` upon in-memory lookup, but **strictly preserves all disk artifacts** under `outputs/api_runs/<run_id>/` (`manifest.json`, timeseries CSV, plots).
4. **Execution Model & Concurrency**:
   - Simulations run as synchronous CPU-intensive tasks; `/api/v1/resilience/simulate` blocks until multi-agent synthesis completes.
   - Asynchronous task worker queues (e.g., Celery/Redis, RQ) are on the production engineering roadmap.

#### Production Configuration Example

```powershell
$env:URBAN_API_KEY = "prod-sec-key-9f8a7b6c5d4e"
$env:URBAN_RATE_LIMIT_PER_MINUTE = "120"
$env:URBAN_REGISTRY_MAX_ENTRIES = "500"
$env:URBAN_REGISTRY_TTL_SECONDS = "172800"  # 48 hours

uvicorn urban_resilience.api:app --host 0.0.0.0 --port 8000 --workers 4
```

## External Validation Pipeline

Evaluates anticipatory multi-agent predictions against historical disaster events and mobility datasets:

```powershell
# Run CI fixture dry-run:
python run_external_validation.py --dry-run

# Run unverified synthetic demo (requires explicit authorization):
python run_external_validation.py --event-id SZ_20230907_HAIKUI_SYNTHETIC_DEMO --allow-unverified

# Run verified real empirical dataset:
python run_external_validation.py --event-id <VERIFIED_EVENT_ID>
```

Outputs are exported to `outputs/external/<event_id>/` (including `report.md`, `metrics.json`, and validation plots). Canonical reports include [`outputs/external/SZ_20230907_HAIKUI_SYNTHETIC_DEMO/report.md`](outputs/external/SZ_20230907_HAIKUI_SYNTHETIC_DEMO/report.md) and [`shenzhen_historical_disaster_empirical_analysis.md`](shenzhen_historical_disaster_empirical_analysis.md).

### Current External Validation Status
- **Synthetic / Unverified Demo Only**: The bundled event under `data/external/SZ_20230907_HAIKUI_SYNTHETIC_DEMO` is a synthetic demonstration dataset (`data_kind: synthetic`, `provenance_status: unverified`), not verified real-world municipal observations.
- **Validation Quality Gate: FAILED**: The automated quality gate evaluates to `FAILED` (Multi-Agent RMSE is inferior to simple AR/Ridge baselines, UQ PICP is 15.28%, and loss amplitude error is 42.00% [exceeding the 15.00% threshold]).
- **No Verified Real-World Empirical Validation Performed Yet (Status: BLOCKED)**: The pipeline is fully functional and causality-guarded, but real-world empirical validation remains unverified until authentic, audited datasets are supplied.
- **Feasibility Audit (2026-10-21)**: A formal feasibility audit on Shenzhen historical disaster mobility data (2023-09-07 "9·7" extreme rainstorm / Typhoon Haikui remnant) verified that open, unauthenticated hourly 12-zone passenger flow matrices with public checksums are unavailable. In strict adherence to scientific integrity and zero-fabrication rules, external validation status remains **`BLOCKED`**; see [`data/external/REAL_DATA_REQUEST.md`](data/external/REAL_DATA_REQUEST.md) for candidate sources, legal constraints, and acquisition prerequisites.
- **Hybrid Validation Characteristics**: When external empirical observations are ingested, the system operates as a **Hybrid Validation**:
  - `temporal` & `flexibility`: Derived directly from pre-event empirical observations (`observed_pre_event`).
  - `spatial`: Combines empirical zone activity with an economic gravity OD flow fallback (`hybrid_observed_with_gravity_od`).
  - `functional`: Evaluated via static spatial POI contexts (`static_urban_context`).
  - `predictability`: Ingests empirical observations when pre-event window $\ge 48$h (`observed_pre_event`); falls back to neutral 0.50 (`neutral_fallback`) when window is 24–47h.
  - `daily_profiles` & `weekly_profiles`: Ingest baseline reference profiles (`reference_default`).
- **Real Data Requirements**: Detailed data specifications, verification checklists, and collection requests are documented in [`data/external/README.md`](data/external/README.md) and [`data/external/REAL_DATA_REQUEST.md`](data/external/REAL_DATA_REQUEST.md). Minimum viable window is 24h (degraded), while $\ge 48$h (recommended 72h) is required for full non-fallback evidence.

## Hierarchical Multi-Agent Orchestration

The framework employs a strict hierarchical supervisor-subagent orchestration architecture:

- **Lifecycle Flow**: `SupervisorAgent` -> `TaskPlanner` -> `TaskScheduler` -> `DomainSubagents` (Temporal, Flexibility, Predictability, Spatial, Functional) -> `EvidenceBus` -> Targeted Follow-up Inquiry -> `SynthesisSubagent`.
- **Follow-up Resolution & Coordination Truthfulness**: Deterministic and LLM follow-ups record task resolution status (`refined`, `SUCCESS`, `DEGRADED`). The supervisor uses `is_follow_up_success_status()` to verify that `"refined"` outcomes count as successful follow-ups. If measurable feature or evidence deltas occur, `coordination_status` is marked `"exercised"` (`coordination_exercised = True`); if follow-ups executed without measurable delta, the status truthfully remains `"followed_up_no_effect"`.
- **Failure Propagation & Strict Evidence Gate**: In default mode (`strict=True, allow_degraded=False`), any missing or failed domain perspective aborts synthesis and raises `IncompleteEvidenceError`. The API surfaces this as HTTP `503 Service Unavailable`, and the CLI exits with code 1.
- **Timeout & Retry Semantics**: Task execution enforces timeout tracking via `ThreadPoolExecutor` and wait timeouts. When a task times out, the scheduler cancels pending unstarted tasks and immediately records `FAILED`. Because standard CPython threads cannot be forcibly killed asynchronously, the timed-out thread may continue executing until completion in the background (cooperative, non-preemptive). By default, timed-out tasks do not retry (`retry_on_timeout=False`) to prevent concurrent execution conflicts. For true preemptive hard interruption, process isolation (e.g., `ProcessPoolExecutor`) is required. Additionally, `TaskScheduler(max_retries=1)` acts as a default upper bound capping task retries (`min(task.max_retries, scheduler.max_retries)`); explicitly pass `max_retries=None` to allow task definitions to govern their own retry counts.
- **Run Manifest & Output Paths**: Every execution produces an auditable run manifest (`RunManifest`) recording DAG tasks, dependency resolution, execution status, errors, and end-to-end duration. Output paths depend on the execution entry point:
  1. **Full Pipeline (`pipeline.py`)**: `outputs/orchestration/run_pipeline_<random_seed>/manifest.json`. Runs with the same `random_seed` will overwrite this file; to preserve individual run audit records, vary the seed or invoke `SupervisorAgent.run()` directly.
  2. **API Simulation (`api.py`)**: `outputs/api_runs/<run_id>/manifest.json` (per-run isolated directory with accompanying plots and CSV summaries).
  3. **Direct Supervisor Invocation (`SupervisorAgent.run()`)**: `outputs/orchestration/<run_id>/manifest.json` (where `<run_id>` is an execution UUID, e.g. `run_3a1b2c4d5e`).
  4. **External Validation (`external_evaluator.py`)**: `outputs/external/<event_id>/manifest.json`.

## LLM Hybrid Orchestration (`llm_hybrid`)

In addition to the deterministic rule-based baseline, the system supports LLM-guided ReAct domain subagents and hierarchical supervisor review:

- **Activation**: Via `--orchestration-mode llm_hybrid` in CLI, or `"orchestration_mode": "llm_hybrid"` in API requests, or environment variable `URBAN_ORCHESTRATION_MODE="llm_hybrid"`.
- **Environment & API Key Safeguards**:
  - Requires `URBAN_LLM_API_KEY` when targeting production LLM providers.
  - In development and CI testing, setting `URBAN_LLM_MODE="mock"` instantiates `MockLLMClient` with deterministic tool response generation.
  - If `llm_hybrid` is requested without `URBAN_LLM_API_KEY` (and without mock mode), API requests return HTTP `422 Unprocessable Entity` and the CLI exits cleanly with error code 2, preventing unhandled exceptions.
- **Anti-Fabrication Tool Boundary**: Subagents are strictly forbidden from generating numeric values in prompt text. All feature values must be computed by deterministic Python tools (`urban_resilience/tools/`) and referenced via `evidence_refs` matching executed tool call IDs. Unreferenced or mismatched values are rejected by `FeatureReport` validation.
- **Two-Round Bounded Supervisor Review**: Supervisor conducts at most 2 rounds of review, evaluating cross-domain conflicts and scheduling targeted inquiries (`targeted_inquiry`). When inquiries complete, `EvidenceBus.merge_zone_evidence()` merges zone metrics without dropping non-targeted zones, and `zone_features` are dynamically updated.

## Multi-Target Predictor Integration & Output Provenance (`urban_resilience.prediction`)

> [!IMPORTANT]
> **Predictor Role**: `predictor_role = "engineering_baseline"` (Engineering Pipeline Baseline).
> The multi-target predictor is trained on synthetic physical simulator labels and serves strictly as an engineering pipeline baseline.
> Empirical accuracy improvement or outperforming traditional baselines is unverified on real disaster data and is NOT claimed.

- **Prediction Provenance Contract (`PredictionProvenance`)**:
  All predictions expose explicit origin metadata across API (`/simulate`, `/runs/{run_id}`), CLI (`--json-output`), and `metrics.json`:
  - `summary_metric_source`: `"trained_model"` when trained weights are loaded and features assembled; `"monte_carlo_physics"` during default fallback.
  - `curve_source`: Always `"monte_carlo_physics"` (the high-resolution timeseries trajectory is computed by the physics-based Monte Carlo engine).
  - `curve_metric_source`: `"monte_carlo_physics"` (curve-derived metrics like `peak_loss_hour` and `resilience_triangle_area` strictly reflect the Monte Carlo simulation curve).
  - `external_temporal_metric_source`: `"monte_carlo_physics"` (time-series RMSE, MAE, and PICP metrics in external validation strictly compare against the physics trajectory curve).
  - `metric_sources`: Fine-grained provenance dictionary explicitly mapping each indicator:
    - `maximum_activity_loss`, `recovery_time_hours`, `cumulative_activity_loss`: `"trained_model"` (or `"monte_carlo_physics"` on fallback).
    - `peak_loss_hour`, `resilience_triangle_area`: `"monte_carlo_physics"`.
  - `interval_source`: `"conformal"` (when calibrated with validation non-conformity quantiles), `"uncalibrated"`, or `"monte_carlo_quantile"`.
  - `model_status`: `"trained_model"`, `"untrained_fallback"`, or `"insufficient_features"`.
  - `model_version` & `feature_schema_version`: Semantic schema versions (`1.0.0`).
  - `artifact_hash`: SHA-256 digest of loaded model file.
  - `training_label_source`: Label origin of the loaded predictor (`"synthetic_simulator"` for synthetic pipelines, `"none"` for fallback, `"unknown"` for untagged legacy models).
  - `engineering_only`: Boolean flag indicating whether the predictor was trained on synthetic engineering data (`True`) or empirical disaster telemetry (`False`).
  - `training_event_ids`: List of disaster/hazard event identifiers used in the training split.
- **Dual-Mode Operation**:
  - `untrained_fallback`: Default baseline. When no serialized model artifact exists, executes the physics-based Monte Carlo engine without claiming ML predictions.
  - `trained_model`: When a valid model artifact is supplied, verifies SHA-256 integrity (`artifact_hash`), ensures `model_version == "1.0.0"` and `feature_schema_version == "1.0.0"`, validates fitted estimators, assembles features strictly via `FeatureAssembler(strict=True)` (requiring exposure features including `drainage_capacity`), loads embedded `training_metadata`, and produces predictions with conformal confidence intervals (`ConformalIntervalCalibrator`). When `engineering_only=True`, APIs, CLI, and reports render explicit synthetic warning banners so users are never misled into treating synthetic weights as empirical benchmarks.
  - `FeatureAssembler` defaults to `strict=True`: Never silently imputes missing exposure features or agent metrics. Any missing feature raises `ValueError`, causing the supervisor to catch and safely record `insufficient_features`.
- **Drainage Data Availability**:
  - The 12 Shenzhen urban zones define `drainage_capacity` ($30-65$ mm/h) synthesized according to municipal drainage engineering standards (GB 50014; $30-40$ mm/h in older districts, $50-65$ mm/h in high-tech districts). Real in-situ municipal sensor telemetry remains unlinked.
- **Engineering Smoke Test Status**: Model loading fixtures in unit tests verify serialization and schema validation mechanics. They are engineering verification tests, not empirical benchmarks.

## Four-Group Ablation Experiments & Synthetic Protocol (`urban_resilience.experiments`)

A reproducible multi-seed experiment runner and disjoint synthetic dataset pipeline:

1. **Ablation Groups**:
   - `fixed_features`: Fixed deterministic baseline without adaptive follow-up (`orchestration_mode="deterministic"`, `enable_follow_ups=False`).
   - `single_domain_adaptive` (alias `single_adaptive`): Single-domain LLM reference (NOT a monolithic single-agent baseline; all 5 domains are evaluated, but only one specified domain, e.g., `"temporal"`, uses LLM ReAct while the other 4 use deterministic rules; do NOT use "single-agent" to refer to this reference ablation), follow-ups disabled (`enable_follow_ups=False`).
   - `multi_uncoordinated`: Full multi-agent ReAct subagents across all 5 domains without supervisor cross-round review or coordination (`enable_follow_ups=False`).
   - `multi_coordinated`: Full multi-agent orchestration with supervisor review, conflict detection, and coordinated follow-ups (`enable_follow_ups=True`). If no localized cross-domain conflict exists, the default synthetic case reports `conflict_resolution_rate = N/A`; conflict-targeted behavior is covered by a controlled integration test.

2. **Multi-Seed Execution & Statistical Aggregation**:
   - Accepts multiple random seeds via `event_seeds: List[int] = [42, 43, 44]`.
   - Runs all 4 groups on each seed and aggregates across seeds, reporting:
     - Sample Mean ($\bar{x}$)
     - Sample Standard Deviation ($s$)
     - Student's $t$ 95% Confidence Interval ($[\bar{x} - t_{n-1, 0.975} \frac{s}{\sqrt{n}}, \bar{x} + t_{n-1, 0.975} \frac{s}{\sqrt{n}}]$)
   - Automatically exports `experiment_summary.csv` (aggregated metrics), `experiment_raw_runs.csv` (per-seed per-group run logs), and `experiment_report.md`.

3. **Synthetic Dataset & Disjoint Protocol (`urban_resilience.prediction`)**:
   - `build_synthetic_dataset`: Builds train, calibration, and test splits with pairwise disjoint event seeds (`train_seeds ∩ cal_seeds = ∅`, etc.).
   - Explicitly stamps `label_source = "synthetic_simulator"` and `engineering_only = True` across splits and evaluation metrics.
   - `train_and_calibrate_synthetic_predictor`: Fits the predictor on the train split only and calibrates conformal non-conformity quantiles on the calibration split only.

```powershell
# Run synthetic four-group experiment suite across 3 seeds:
python -c "from urban_resilience.experiments import run_four_group_experiments; suite = run_four_group_experiments(event_seeds=[42, 43, 44], engineering_only=True); print(suite.to_markdown())"
```

All runs without verified empirical ground-truth datasets are stamped with `engineering_only = True`. Attempting to set `engineering_only = False` raises `ValueError("Real empirical validation is BLOCKED: engineering_only=False requires a verified data loader.")`, enforcing pipeline truthfulness until verified empirical historical disaster datasets are connected.

> [!NOTE]
> **Default synthetic conflict scope**: the bundled synthetic event currently produces domain anomalies but no localized cross-domain conflict. The audit table therefore reports `conflicts_detected = 0`, `anomalies_detected > 0`, and `conflict_resolution_rate = N/A`. `multi_coordinated` can still report `coordination_status = "exercised"` when a follow-up produces an information/evidence delta; this is not evidence of conflict resolution. The conflict-targeted resolution path is exercised separately by `tests/test_core_fixes.py::test_supervisor_conflict_targeted_coordination_integration`.

## Disaster Event Intelligence Scaffold (`urban_resilience.event`)

- **Status**: `STATUS = "scaffold_offline"`
- Provides offline bulletin parsing, schema structuring (`HazardBulletin`), and official source verification (`OfficialSourceVerifier`).
- External live bulletin ingestion remains **BLOCKED** pending verified municipal open-data feeds. Real historical event data validation remains **BLOCKED**.

## Tests and lint

```powershell
python -m pytest -q
python -m ruff check .
```

All `pytest` test suites redirect generated files and logs to isolated temporary directories (`tmp_path`) and strictly do not modify repository `outputs/`.

### Engineering Validation Coverage

The test suite covers the following design contracts in addition to happy-path execution:

- **Conflict taxonomy**: `EvidenceBus` distinguishes `cross_domain`, `anomaly`, and `invalid` records. Only localized cross-domain conflicts enter `conflict_resolution_rate`; anomalies and unlocalized conflicts are counted separately.
- **Conflict-targeted coordination**: a controlled Supervisor integration test builds a cross-domain `spatial`/`flexibility` conflict, delegates to the conflicting zone, applies a refined follow-up, and verifies `conflicts_resolved > 0`. Unrelated-zone and anomaly-only counterexamples verify that resolution is not over-counted.
- **Audit truthfulness**: `AuditMetrics` returns `N/A` when a denominator is absent (no tool calls, no conflicts, no feature values) instead of displaying false 100% values. Fabricated numeric claims increment rejection counters.
- **Strict failure semantics**: `allow_degraded=False` is tested with an injected failing subagent/tool and must raise; `allow_degraded=True` records degraded/failed status without masking it.
- **Reproducibility**: `CachingLLMClient`/`ReplayLLMClient` record and replay JSONL LLM transcripts offline; replay fails loudly on a missing prompt hash.
- **Fixture integrity**: `FixtureEventSource.verify_source()` validates every checksum listed in `PROVENANCE.json`; tampering or deleting any fixture file returns `False`.
- **Single-domain semantics**: `single_domain_adaptive` is explicitly a single-domain reference, not a monolithic single-agent baseline.
- **Predictor scope**: synthetic predictor artifacts are engineering baselines; reports and API/CLI surfaces carry scope warnings and do not claim empirical accuracy.

Representative tests: `tests/test_core_fixes.py`, `tests/test_experiments_runner.py`, `tests/test_llm_followup_propagation.py`, `tests/test_predictor_integration.py`, and `tests/test_external_validation.py`.

### Comprehensive Health Check (`check_all.py`)

For full-suite verification prior to commits or release milestones, run:

```powershell
python check_all.py
```

`check_all.py` executes a 4-stage pipeline sequentially:
1. **Linting**: `python -m ruff check .` (code formatting and static analysis rules)
2. **Test Suite**: `python -m pytest -q` (all unit, integration, and regression test suites in test isolation)
3. **Anticipatory City Simulation**: `python run_full_city_analysis.py` (executes the end-to-end synthetic baseline pipeline)
4. **External Validation**: `python run_external_validation.py --event-id SZ_20230907_HAIKUI_SYNTHETIC_DEMO --allow-unverified` (executes external hindcast validation and evaluates quality gates)

> [!NOTE]
> Because stages 3 and 4 execute authentic end-to-end pipelines, `check_all.py` **will write to and update repository baseline artifacts under `outputs/`** (including logs, prediction tables, and validation figures). For routine day-to-day development, run `pytest` directly to guarantee outputs immutability.

## Project layout

```text
urban_resilience/
  agents/        ReAct domain agents and evidence synthesis
  data/          spatial, mobility, and event-intelligence loaders
  event/         offline disaster event intelligence and bulletin parser scaffold
  experiments/   four-group ablation experiment suite runner
  llm/           LLM client abstractions and mock client for test isolation
  orchestration/ hierarchical Supervisor, subagents, EvidenceBus, and review loop
  outputs/       prediction tables and visualizations
  prediction/    multi-target ML predictor, FeatureAssembler, and conformal calibrator
  tools/         deterministic computational tools across 5 resilience domains
  validation/    evaluator, baselines, ablations, calibration, external pipeline
  api.py         FastAPI service
  cli.py         command-line interface
  pipeline.py    end-to-end orchestrator
tests/           unit, integration, and regression tests
check_all.py     comprehensive CI and local integration check
run_full_city_analysis.py
run_external_validation.py
```

## Notes and limitations

- The bundled dataset is a synthetic Shenzhen case study; it is not operational municipal data.
- Validation ground truth in `run_full_city_analysis.py` is generated by an internal hazard simulator, measuring internal consistency.
- External validation against authentic real-world observations requires audited, non-synthetic datasets with verified provenance contracts (`data/external/README.md`). Unverified demo datasets automatically fail the quality gate.
- The API currently supports `Shenzhen` only.
- The model is a research prototype and must not be used for real emergency decisions.
