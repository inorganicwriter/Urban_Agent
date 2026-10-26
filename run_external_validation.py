"""CLI entry point for external empirical disaster event and mobility validation.

Evaluates ex-ante anticipatory predictions against empirical observations
under strict causality constraints and event-level holdout rules.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
from typing import List, Optional

# Force UTF-8 encoding on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from urban_resilience.data.external_loader import ExternalDataLoader, ObservedEventDataset
from urban_resilience.validation.external_evaluator import ExternalEventEvaluator

logger = logging.getLogger("ExternalValidation")
console = Console()


def _setup_logging(log_file: Optional[Path] = None):
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode="w", encoding="utf-8"))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        handlers=handlers,
    )


def discover_event_dirs(data_root: Path) -> List[Path]:
    """Finds all valid event subdirectories containing event.json in data_root."""
    if not data_root.exists() or not data_root.is_dir():
        return []
    events = []
    for item in sorted(data_root.iterdir()):
        if item.is_dir() and (item / "event.json").is_file():
            events.append(item)
    return events


def validate_holdout_disjointness(datasets: List[ObservedEventDataset]) -> None:
    """Verifies that calibration and test event sets are strictly disjoint."""
    test_ids = {ds.meta.event_id for ds in datasets if ds.meta.role == "test"}
    calib_ids = {ds.meta.event_id for ds in datasets if ds.meta.role == "calibration"}
    overlap = test_ids.intersection(calib_ids)
    if overlap:
        raise ValueError(
            f"Event holdout violation: The following event IDs appear in both test and calibration roles: {overlap}. "
            "Calibration and test events must be strictly disjoint to prevent data leakage."
        )


def run_validation(
    data_root: Path,
    event_id: Optional[str] = None,
    output_dir: Optional[Path] = None,
    dry_run: bool = False,
    allow_unverified: bool = False,
    monte_carlo_samples: int = 150,
    verify_checksums: bool = True,
) -> int:
    """Orchestrates external validation run across specified or discovered events.

    Returns:
        Exit code: 0 on success, 1 on runtime error, 2 on data policy violation.
    """
    out_dir = output_dir or Path("outputs/external")
    out_dir.mkdir(parents=True, exist_ok=True)
    _setup_logging(out_dir / "external_validation.log")

    console.print(
        Panel.fit(
            "[bold cyan]Urban Resilience External Validation Pipeline[/bold cyan]\n"
            "[dim]Empirical Historical Event & Human Mobility Validation Engine[/dim]"
        )
    )

    loader = ExternalDataLoader()

    # Determine event directories to process
    if event_id:
        target_dir = data_root / event_id
        if not target_dir.is_dir():
            # Also check if event_id matches directly or is inside data_root
            matched = [d for d in discover_event_dirs(data_root) if d.name == event_id or (d / "event.json").is_file() and event_id in d.name]
            if matched:
                target_dirs = [matched[0]]
            else:
                console.print(f"[bold red]Error:[/bold red] Specified event directory '{target_dir}' does not exist.")
                return 1
        else:
            target_dirs = [target_dir]
    else:
        target_dirs = discover_event_dirs(data_root)
        if not target_dirs:
            console.print(f"[bold yellow]No external event datasets found under '{data_root}'.[/bold yellow]")
            console.print("Please place event datasets adhering to `data/external/README.md` into that directory.")
            return 1

    # Load all datasets and check safety/holdout constraints
    loaded_datasets: List[ObservedEventDataset] = []
    for edir in target_dirs:
        try:
            ds = loader.load_event(edir, verify_checksums=verify_checksums)
            loaded_datasets.append(ds)
        except Exception as e:
            console.print(f"[bold red]Validation error loading '{edir}':[/bold red] {e}")
            return 1

    try:
        validate_holdout_disjointness(loaded_datasets)
    except ValueError as ve:
        console.print(f"[bold red]Data Holdout Policy Violation:[/bold red] {ve}")
        return 2

    # Execute evaluation for each event
    evaluator = ExternalEventEvaluator(output_dir=out_dir)
    success_count = 0
    any_gate_failed = False

    for ds in loaded_datasets:
        console.print(f"\n[bold green]Processing Event:[/bold green] [yellow]{ds.meta.event_id}[/yellow] (data_kind={ds.meta.data_kind})")

        # Check fixture policy
        if ds.meta.data_kind == "fixture" and not dry_run:
            console.print(
                f"[bold red]Data Policy Violation:[/bold red] Event '{ds.meta.event_id}' is marked as a fixture ('data_kind: fixture').\n"
                "To evaluate test fixture data, you must explicitly pass the '--dry-run' flag.\n"
                "Refusing to execute fixture as real empirical validation without '--dry-run'."
            )
            return 2

        # Check unverified / synthetic demo policy
        if (ds.meta.data_kind == "synthetic" or ds.meta.provenance_status == "unverified") and not (dry_run or allow_unverified):
            console.print(
                f"[bold red]Data Policy Violation:[/bold red] Event '{ds.meta.event_id}' has data_kind='{ds.meta.data_kind}' "
                f"or provenance_status='{ds.meta.provenance_status}'.\n"
                "Unverified synthetic demo datasets cannot be evaluated as verified empirical observations.\n"
                "To run this demo dataset, you must explicitly pass '--allow-unverified' (or '--dry-run')."
            )
            return 2

        try:
            report, paths = evaluator.evaluate_event(
                dataset=ds,
                dry_run=dry_run,
                allow_unverified=allow_unverified,
                monte_carlo_samples=monte_carlo_samples,
            )

            # Display rich summary table
            tm = report.temporal_metrics
            sm = report.spatial_metrics
            um = report.uncertainty_metrics

            tbl = Table(title=f"Validation Performance: {report.event_id}", show_header=True)
            tbl.add_column("Dimension", style="cyan")
            tbl.add_column("Metric", style="magenta")
            tbl.add_column("Score", style="green bold")
            tbl.add_column("Reference / Interpretation", style="dim")

            tbl.add_row("Temporal", "RMSE", f"{tm.rmse:.4f}", "Lower is better")
            tbl.add_row("Temporal", "MAE", f"{tm.mae:.4f}", "Mean absolute deviation")
            tbl.add_row("Temporal", "MAPE", f"{tm.mape_pct:.2f}%", "Percentage deviation")
            tbl.add_row("Temporal", "sMAPE", f"{tm.smape_pct:.2f}%", "Symmetric percentage deviation")
            tbl.add_row("Temporal", "R² Score", f"{tm.r2_score:.4f}", "1.0 = perfect prediction")
            tbl.add_row("Temporal", "Max Loss Error", f"{tm.max_activity_loss_error:.2%}", f"Pred: {tm.max_activity_loss_pred:.1%} | Obs: {tm.max_activity_loss_obs:.1%} ")
            tbl.add_row("Temporal", "Recovery Time Error", f"{tm.recovery_time_error:.1f}h", f"Pred: {tm.recovery_time_pred:.1f}h | Obs: {tm.recovery_time_obs:.1f}h")
            tbl.add_row("Spatial", "Spearman Rank Rho", f"{sm.spearman_rho_loss:.4f}", f"p-val: {sm.spearman_pvalue_loss:.4e}")
            if sm.spearman_rho_impact is not None:
                tbl.add_row("Spatial", "Physical Impact Rho", f"{sm.spearman_rho_impact:.4f}", f"p-val: {sm.spearman_pvalue_impact:.4e}")
            tbl.add_row("Uncertainty", "Coverage (PICP)", f"{um.picp_95:.2%}", f"Nominal target: {um.nominal_coverage:.0%}")
            tbl.add_row("Uncertainty", "Interval Width (MPIW)", f"{um.mpiw:.4f}", "Sharpness indicator")
            tbl.add_row("Uncertainty", "Winkler Score", f"{um.winkler_score:.4f}", "Penalized interval score")

            console.print(tbl)

            # Display quality gate verdicts
            if report.verdicts:
                v = report.verdicts
                if v.overall_verdict == "FAILED":
                    any_gate_failed = True

                vtbl = Table(title=f"Quality Gate Verdicts: {report.event_id}", show_header=True)
                vtbl.add_column("Gate Item", style="cyan")
                vtbl.add_column("Verdict", style="bold")
                vtbl.add_column("Standard / Note", style="dim")

                def _c(val: str) -> str:
                    return f"[bold green]{val}[/bold green]" if val == "PASS" else f"[bold red]{val}[/bold red]"

                vtbl.add_row("Provenance Audit", _c(v.provenance_verdict), "Observed & verified provenance")
                vtbl.add_row("Baseline Outperformance", _c(v.baseline_verdict), "Multi-Agent RMSE <= Benchmark models")
                vtbl.add_row("UQ Coverage (>=80%)", _c(v.uq_verdict), f"95% CI PICP >= 80% (Actual: {um.picp_95:.1%})")
                vtbl.add_row("Amplitude Accuracy (<=15%)", _c(v.amplitude_verdict), f"Max loss error <= 15% (Actual: {tm.max_activity_loss_error:.1%})")
                vtbl.add_row("Overall Quality Gate", _c(v.overall_verdict), "All gates must PASS to claim verified closed loop")
                console.print(vtbl)

                if v.overall_verdict == "FAILED":
                    console.print("[bold red]Quality Gate FAILED:[/bold red]")
                    for r in v.reasons:
                        console.print(f"  [red]• {r}[/red]")

            console.print(f"[bold cyan]Artifacts Exported:[/bold cyan] {paths['report_md'].parent}")
            console.print(f"  - Markdown Report: [link=file:///{paths['report_md'].resolve()}]{paths['report_md']}[/link]")
            console.print(f"  - Metrics JSON:    {paths['metrics_json']}")
            console.print(f"  - Temporal Plot:   {paths['plot_temporal']}")
            console.print(f"  - Spatial Plot:    {paths['plot_spatial']}")
            success_count += 1

        except Exception as e:
            logger.exception("Failed to evaluate event %s", ds.meta.event_id)
            console.print(f"[bold red]Execution error evaluating '{ds.meta.event_id}':[/bold red] {e}")
            return 1

    if any_gate_failed:
        console.print(
            f"\n[bold yellow]External validation pipeline completed ({success_count}/{len(loaded_datasets)} events evaluated); [/bold yellow]"
            "[bold red]QUALITY GATE FAILED — see report.[/bold red]"
        )
    else:
        console.print(
            f"\n[bold green]External validation completed successfully ({success_count}/{len(loaded_datasets)} events processed).[/bold green]"
        )
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="External validation pipeline evaluating Urban_Agent against empirical historical event data."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/external"),
        help="Root directory containing external event datasets (default: data/external)",
    )
    parser.add_argument(
        "--event-id",
        type=str,
        default=None,
        help="Specific event ID or folder name to validate (default: auto-discover all events in data-root)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where validation reports and plots will be written (default: outputs/external)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Allow validation of synthetic test fixtures ('data_kind: fixture'). Required for fixtures.",
    )
    parser.add_argument(
        "--allow-unverified",
        action="store_true",
        help="Allow validation of synthetic or unverified demo datasets (data_kind: synthetic or provenance_status: unverified).",
    )
    parser.add_argument(
        "--monte-carlo",
        type=int,
        default=150,
        help="Monte Carlo sample count for prediction intervals (default: 150)",
    )
    parser.add_argument(
        "--skip-checksums",
        action="store_true",
        help="Skip PROVENANCE.json SHA-256 checksum verification (for development/debugging only)",
    )

    args = parser.parse_args()
    code = run_validation(
        data_root=args.data_root,
        event_id=args.event_id,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        allow_unverified=args.allow_unverified,
        monte_carlo_samples=args.monte_carlo,
        verify_checksums=not args.skip_checksums,
    )
    sys.exit(code)
    sys.exit(code)


if __name__ == "__main__":
    main()
