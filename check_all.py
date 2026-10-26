"""CI and local health check script executing all test suites, linters, and verification pipelines."""

from __future__ import annotations

import subprocess
import sys
import time
from typing import List, Tuple


def run_command(name: str, cmd: List[str]) -> Tuple[bool, float, str]:
    print(f"==> Running: {name} ({' '.join(cmd)})...")
    start = time.perf_counter()
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    elapsed = time.perf_counter() - start
    output = ((res.stdout or "") + "\n" + (res.stderr or "")).strip()
    return res.returncode == 0, elapsed, output


def main() -> int:
    tasks = [
        ("Linting (ruff check)", [sys.executable, "-m", "ruff", "check", "."]),
        ("Test Suite (pytest)", [sys.executable, "-m", "pytest", "-q"]),
        ("Anticipatory City Simulation", [sys.executable, "run_full_city_analysis.py"]),
        (
            "External Validation (Synthetic Demo)",
            [sys.executable, "run_external_validation.py", "--event-id", "SZ_20230907_HAIKUI_SYNTHETIC_DEMO", "--allow-unverified"],
        ),
    ]

    all_passed = True
    print("\n" + "=" * 70)
    print("URBAN AGENT LOCAL & CI COMPREHENSIVE INTEGRATION CHECK")
    print("=" * 70 + "\n")

    results = []
    for name, cmd in tasks:
        success, elapsed, output = run_command(name, cmd)
        status_str = "PASS" if success else "FAIL"
        results.append((name, status_str, elapsed))
        if not success:
            all_passed = False
            print(f"[ERROR] {name} failed with output:\n{output}\n")
        else:
            print(f"[OK] {name} passed in {elapsed:.2f}s\n")

    print("\n" + "=" * 70)
    print("CHECK SUMMARY")
    print("=" * 70)
    for name, status_str, elapsed in results:
        print(f"  - {name:<40} : [{status_str}] ({elapsed:.2f}s)")
    print("=" * 70 + "\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
