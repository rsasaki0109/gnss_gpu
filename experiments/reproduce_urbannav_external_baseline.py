#!/usr/bin/env python3
"""Reproduce the UrbanNav external GNSS research baseline (trimble + G,E,J).

This is the fixed external-validation table used to benchmark gnss_gpu against
classical estimators on data never used for PPC tuning:

  - WLS / WLS+QualityVeto  (multi-GNSS utility baselines)
  - EKF                    (classical sequential baseline)
  - PF-10K                 (GPU PF family)
  - PF+RobustClear-10K     (frozen external mainline)

Reference aggregates live in docs/assets/data/ and experiments/results/ as
urbannav_fixed_eval_external_gej_trimble_qualityveto_{runs,summary}.csv.

Prefer the unified harness entry point when possible:

  PYTHONPATH=python:. python experiments/eval_harness.py run urbannav-external

Usage
-----
  PYTHONPATH=python:. python experiments/reproduce_urbannav_external_baseline.py
  PYTHONPATH=python:. python experiments/reproduce_urbannav_external_baseline.py \\
      --data-root data/urbannav/Tokyo --fetch --require-exact
  PYTHONPATH=python:. python experiments/reproduce_urbannav_external_baseline.py \\
      --smoke --max-epochs 500
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "python"))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from eval_harness_lib import (  # noqa: E402
    DEFAULT_P95_TOL_M,
    DEFAULT_RATE_TOL_PP,
    DEFAULT_RMS_TOL_M,
    URBANNAV_EXTERNAL_DATA_ROOT,
    URBANNAV_EXTERNAL_PREFIX,
    URBANNAV_EXTERNAL_REFERENCE,
    URBANNAV_EXTERNAL_SUITE,
    compare_summary,
    ensure_urbannav_data,
    read_csv,
    run_fixed_eval,
    validate_summary_rows,
)

RESULTS_DIR = _SCRIPT_DIR / "results"
DEFAULT_RUNS = URBANNAV_EXTERNAL_SUITE.default_runs
DEFAULT_METHODS = URBANNAV_EXTERNAL_SUITE.default_methods
DEFAULT_SYSTEMS = URBANNAV_EXTERNAL_SUITE.default_systems
DEFAULT_PREFIX = URBANNAV_EXTERNAL_PREFIX
REFERENCE_SUMMARY = URBANNAV_EXTERNAL_REFERENCE
DEFAULT_DATA_ROOT = URBANNAV_EXTERNAL_DATA_ROOT


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def reproduce_urbannav_external_baseline(
    *,
    data_root: Path,
    runs: tuple[str, ...],
    methods: tuple[str, ...],
    systems: tuple[str, ...],
    results_prefix: str,
    reference_summary: Path,
    fetch: bool,
    max_epochs: int | None,
    isolate_methods: bool,
    save_epoch_errors: bool,
    rms_tol_m: float,
    p95_tol_m: float,
    rate_tol_pp: float,
    smoke: bool = False,
) -> dict[str, object]:
    ensure_urbannav_data(data_root, runs, fetch=fetch)
    summary_path = run_fixed_eval(
        data_root=data_root,
        runs=runs,
        methods=methods,
        systems=systems,
        results_prefix=results_prefix,
        urban_rover=URBANNAV_EXTERNAL_SUITE.urban_rover,
        max_epochs=max_epochs,
        isolate_methods=isolate_methods,
        save_epoch_errors=save_epoch_errors,
    )
    reproduced = read_csv(summary_path)
    if smoke:
        checks, passed = validate_summary_rows(reproduced, methods=methods)
    else:
        reference = read_csv(reference_summary) if reference_summary.is_file() else []
        checks, passed = compare_summary(
            reproduced,
            reference,
            methods=methods,
            rms_tol_m=rms_tol_m,
            p95_tol_m=p95_tol_m,
            rate_tol_pp=rate_tol_pp,
        )
    return {
        "data_root": str(data_root),
        "runs": list(runs),
        "methods": list(methods),
        "systems": list(systems),
        "results_prefix": results_prefix,
        "summary_csv": str(summary_path),
        "reference_summary_csv": str(reference_summary),
        "max_epochs": max_epochs,
        "smoke": smoke,
        "checks": checks,
        "passed": passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reproduce UrbanNav external GNSS baseline (trimble + G,E,J)",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="UrbanNav Tokyo directory containing Odaiba/ and Shinjuku/",
    )
    parser.add_argument(
        "--runs",
        type=str,
        default=",".join(DEFAULT_RUNS),
        help="Comma-separated UrbanNav runs",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default=",".join(DEFAULT_METHODS),
        help="Comma-separated methods to evaluate",
    )
    parser.add_argument(
        "--systems",
        type=str,
        default=",".join(DEFAULT_SYSTEMS),
        help="Comma-separated GNSS systems",
    )
    parser.add_argument(
        "--results-prefix",
        type=str,
        default=DEFAULT_PREFIX,
        help="Output prefix under experiments/results/",
    )
    parser.add_argument(
        "--reference-summary",
        type=Path,
        default=REFERENCE_SUMMARY,
        help="Committed reference summary CSV for parity gate",
    )
    parser.add_argument(
        "--fetch",
        action="store_true",
        help="Download missing UrbanNav run subsets before evaluation",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Shortcut for CPU smoke on Odaiba (structural gate, no reference parity)",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Optional epoch cap per run",
    )
    parser.add_argument(
        "--no-isolate-methods",
        action="store_true",
        help="Run all methods in one process (may OOM on GPU PF full runs)",
    )
    parser.add_argument(
        "--save-epoch-errors",
        action="store_true",
        help="Write per-epoch diagnostics CSVs",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=RESULTS_DIR / "urbannav_external_baseline_reproduction_summary.json",
        help="Write reproduction gate summary JSON here",
    )
    parser.add_argument(
        "--require-exact",
        action="store_true",
        help="Exit 2 when reproduced summary differs from reference beyond tolerance",
    )
    parser.add_argument("--rms-tol-m", type=float, default=DEFAULT_RMS_TOL_M)
    parser.add_argument("--p95-tol-m", type=float, default=DEFAULT_P95_TOL_M)
    parser.add_argument("--rate-tol-pp", type=float, default=DEFAULT_RATE_TOL_PP)
    args = parser.parse_args()

    runs = tuple(part.strip() for part in args.runs.split(",") if part.strip())
    methods = tuple(part.strip() for part in args.methods.split(",") if part.strip())
    systems = tuple(part.strip().upper() for part in args.systems.split(",") if part.strip())
    max_epochs = args.max_epochs
    results_prefix = args.results_prefix
    smoke = args.smoke
    if smoke:
        runs = URBANNAV_EXTERNAL_SUITE.smoke_runs
        methods = URBANNAV_EXTERNAL_SUITE.smoke_methods
        max_epochs = URBANNAV_EXTERNAL_SUITE.smoke_max_epochs if max_epochs is None else max_epochs
        if results_prefix == DEFAULT_PREFIX:
            results_prefix = f"{DEFAULT_PREFIX}{URBANNAV_EXTERNAL_SUITE.smoke_results_suffix}"

    summary = reproduce_urbannav_external_baseline(
        data_root=args.data_root,
        runs=runs,
        methods=methods,
        systems=systems,
        results_prefix=results_prefix,
        reference_summary=args.reference_summary,
        fetch=args.fetch,
        max_epochs=max_epochs,
        isolate_methods=not args.no_isolate_methods,
        save_epoch_errors=args.save_epoch_errors,
        rms_tol_m=args.rms_tol_m,
        p95_tol_m=args.p95_tol_m,
        rate_tol_pp=args.rate_tol_pp,
        smoke=smoke,
    )
    _write_json(args.summary_json, summary)

    print(json.dumps(summary, indent=2))
    if args.require_exact and not summary["passed"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
