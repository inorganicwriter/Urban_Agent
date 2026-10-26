"""Unit tests for the CLI interface."""

from urban_resilience.cli import create_parser, main


def test_cli_parser_defaults():
    parser = create_parser()
    args = parser.parse_args([])
    assert args.city == "Shenzhen"
    assert args.hazard == "rainstorm"
    assert args.start_hour == 24
    assert args.duration == 18
    assert args.intensity == 0.85
    assert args.total_hours == 96


def test_cli_execution_fast():
    # Run a fast simulation via CLI entrypoint
    exit_code = main([
        "--city", "Shenzhen",
        "--hazard", "rainstorm",
        "--start-hour", "12",
        "--duration", "6",
        "--intensity", "0.70",
        "--total-hours", "36",
        "--mc-samples", "20",
        "--json-output",
    ])
    assert exit_code == 0


def test_cli_execution_untrained_fallback_no_warning(capsys):
    """Verifies CLI execution in default untrained fallback mode produces no synthetic warning."""
    import json

    exit_code = main([
        "--city", "Shenzhen",
        "--hazard", "rainstorm",
        "--start-hour", "12",
        "--duration", "6",
        "--intensity", "0.70",
        "--total-hours", "24",
        "--mc-samples", "5",
        "--json-output",
    ])
    assert exit_code == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload.get("synthetic_warning") is None
    assert payload.get("provenance_warning") is None

