"""Shared, fail-closed helpers for external UrbanNav reproduction gates."""

from __future__ import annotations

import csv
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping


REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "experiments/results"
DEFAULT_RMS_TOL_M = 0.75
DEFAULT_P95_TOL_M = 1.0
DEFAULT_RATE_TOL_PP = 0.25
URBANNAV_EXTERNAL_DATA_ROOT = REPO / "data/urbannav/Tokyo"
URBANNAV_EXTERNAL_PREFIX = "urbannav_fixed_eval_external_gej_trimble_qualityveto"
URBANNAV_EXTERNAL_REFERENCE = (
    REPO
    / "docs/assets/data/urbannav_fixed_eval_external_gej_trimble_qualityveto_summary.csv"
)


@dataclass(frozen=True)
class EvaluationSuite:
    default_runs: tuple[str, ...]
    default_methods: tuple[str, ...]
    default_systems: tuple[str, ...]
    urban_rover: str
    smoke_runs: tuple[str, ...]
    smoke_methods: tuple[str, ...]
    smoke_max_epochs: int
    smoke_results_suffix: str


URBANNAV_EXTERNAL_SUITE = EvaluationSuite(
    default_runs=("Odaiba", "Shinjuku"),
    default_methods=(
        "WLS",
        "WLS+QualityVeto",
        "EKF",
        "PF-10K",
        "PF+RobustClear-10K",
    ),
    default_systems=("G", "E", "J"),
    urban_rover="trimble",
    smoke_runs=("Odaiba",),
    smoke_methods=("WLS", "EKF"),
    smoke_max_epochs=50,
    smoke_results_suffix="_smoke",
)

_METRICS = (
    ("mean_rms_2d", DEFAULT_RMS_TOL_M),
    ("mean_p95", DEFAULT_P95_TOL_M),
    ("mean_outlier_rate_pct", DEFAULT_RATE_TOL_PP),
    ("mean_catastrophic_rate_pct", DEFAULT_RATE_TOL_PP),
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def run_has_core_files(run_dir: Path) -> bool:
    """Return whether a run has navigation, reference, and a rover observation."""
    return (
        (run_dir / "base.nav").is_file()
        and (run_dir / "reference.csv").is_file()
        and any(run_dir.glob("rover*.obs"))
    )


def _by_method(rows: Iterable[Mapping[str, str]]) -> dict[str, Mapping[str, str]]:
    return {str(row.get("method", "")): row for row in rows if row.get("method")}


def validate_summary_rows(
    rows: list[dict[str, str]], *, methods: tuple[str, ...]
) -> tuple[list[dict[str, object]], bool]:
    indexed = _by_method(rows)
    checks: list[dict[str, object]] = []
    for method in methods:
        row = indexed.get(method)
        missing = row is None
        invalid: list[str] = []
        if row is not None:
            for metric, _ in _METRICS:
                try:
                    value = float(row[metric])
                    if not (-float("inf") < value < float("inf")):
                        invalid.append(metric)
                except (KeyError, TypeError, ValueError):
                    invalid.append(metric)
        checks.append(
            {
                "method": method,
                "status": "missing" if missing else ("invalid" if invalid else "checked"),
                "invalid_metrics": invalid,
            }
        )
    return checks, all(item["status"] == "checked" for item in checks)


def compare_summary(
    reproduced: list[dict[str, str]],
    reference: list[dict[str, str]],
    *,
    methods: tuple[str, ...],
    rms_tol_m: float,
    p95_tol_m: float,
    rate_tol_pp: float,
) -> tuple[list[dict[str, object]], bool]:
    actual = _by_method(reproduced)
    expected = _by_method(reference)
    tolerances = {
        "mean_rms_2d": float(rms_tol_m),
        "mean_p95": float(p95_tol_m),
        "mean_outlier_rate_pct": float(rate_tol_pp),
        "mean_catastrophic_rate_pct": float(rate_tol_pp),
    }
    checks: list[dict[str, object]] = []
    for method in methods:
        actual_row = actual.get(method)
        expected_row = expected.get(method)
        metric_checks: list[dict[str, object]] = []
        status = "checked"
        if actual_row is None or expected_row is None:
            status = "missing"
        else:
            for metric, tolerance in tolerances.items():
                try:
                    actual_value = float(actual_row[metric])
                    expected_value = float(expected_row[metric])
                    delta = actual_value - expected_value
                    passed = abs(delta) <= tolerance
                except (KeyError, TypeError, ValueError):
                    actual_value = expected_value = delta = float("nan")
                    passed = False
                metric_checks.append(
                    {
                        "metric": metric,
                        "actual": actual_value,
                        "reference": expected_value,
                        "delta": delta,
                        "tolerance": tolerance,
                        "passed": passed,
                    }
                )
            if not all(item["passed"] for item in metric_checks):
                status = "failed"
        checks.append(
            {"method": method, "status": status, "metrics": metric_checks}
        )
    return checks, all(item["status"] == "checked" for item in checks)


def ensure_urbannav_data(
    data_root: Path, runs: tuple[str, ...], *, fetch: bool
) -> None:
    missing = [run for run in runs if not run_has_core_files(data_root / run)]
    if missing and fetch:
        for run in missing:
            subprocess.run(
                [
                    sys.executable,
                    str(REPO / "experiments/fetch_urbannav_subset.py"),
                    "--run",
                    run,
                    "--output-dir",
                    str(data_root),
                ],
                cwd=REPO,
                check=True,
            )
        missing = [run for run in runs if not run_has_core_files(data_root / run)]
    if missing:
        raise FileNotFoundError(
            f"UrbanNav core files are missing for {', '.join(missing)} under {data_root}"
        )


def run_fixed_eval(
    *,
    data_root: Path,
    runs: tuple[str, ...],
    methods: tuple[str, ...],
    systems: tuple[str, ...],
    results_prefix: str,
    urban_rover: str,
    max_epochs: int | None,
    isolate_methods: bool,
    save_epoch_errors: bool,
) -> Path:
    command = [
        sys.executable,
        str(REPO / "experiments/exp_urbannav_fixed_eval.py"),
        "--data-root",
        str(data_root),
        "--runs",
        ",".join(runs),
        "--methods",
        ",".join(methods),
        "--systems",
        ",".join(systems),
        "--urban-rover",
        urban_rover,
        "--results-prefix",
        results_prefix,
    ]
    if max_epochs is not None:
        command.extend(("--max-epochs", str(max_epochs)))
    if isolate_methods:
        command.append("--isolate-methods")
    if save_epoch_errors:
        command.append("--save-epoch-errors")
    subprocess.run(command, cwd=REPO, check=True)
    summary = RESULTS / f"{results_prefix}_summary.csv"
    if not summary.is_file():
        raise FileNotFoundError(f"fixed evaluation did not create {summary}")
    return summary
