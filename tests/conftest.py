"""Pytest configuration and autouse fixtures preventing test pollution of repository outputs."""

from pathlib import Path
import pytest
from urban_resilience.config import ProjectPaths


@pytest.fixture(autouse=True)
def isolate_test_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Redirects all ProjectPaths outputs to a pytest temporary directory."""
    test_outputs = tmp_path / "outputs"
    test_plots = test_outputs / "plots"
    test_reports = test_outputs / "reports"

    monkeypatch.setattr(ProjectPaths, "OUTPUTS_DIR", test_outputs)
    monkeypatch.setattr(ProjectPaths, "PLOTS_DIR", test_plots)
    monkeypatch.setattr(ProjectPaths, "REPORTS_DIR", test_reports)

    ProjectPaths.ensure_directories()
    yield
