#!/usr/bin/env python3
"""WP9 driver: sweep libgnss++ ``gnss_solve``'s new float-filter
trust/reset policy (``--float-trust-policy {cv-predict,scaled-reset}`` +
``--trust-lapse-qpos``) on PPC tokyo runs, on top of the WP7/dead-knob
baseline (``--preset low-cost --max-pos-jump-rate 2.3``).

Thin extension of ``experiments/sweep_libgnss_rtk_wp8.py`` (reuses its
``build_full_argv``/canyon-segment scoring/CSV-row shape, imported via
``sweep_libgnss_rtk_wp7``). The only differences:

1. New candidate stages: a coarse qpos sweep per policy (work item 2), a
   run2/run3 regression-matrix stage for the winner (work item 3), a
   combination stage layering ``--hold-ratio-threshold 2.0`` on top of the
   winner (work item 4), and an optional stretch stage for
   ``--nlos-min-los-sats`` (work item 5).
2. ``needs_nlos_csv`` candidates get ``--nlos-weights`` wired the same way
   WP7/WP8 did, for the (stretch, optional-lever) candidates that consult
   the NLOS mask -- ``--trust-gate-nlos-relax`` and ``--nlos-min-los-sats``.

No changes are made to the libgnss++ C++ engine from this script; every
candidate is a config-only CLI variant of the already-built ``gnss_solve``.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sweep_libgnss_rtk_wp7 import (  # noqa: E402
    CANYON_TOW_HI,
    CANYON_TOW_LO,
    CSV_FIELDS,
    WP6_WINNER_ARGS,
    Candidate,
    append_csv_row,
    build_full_argv,
    nlos_mask_path,
    run_and_score_candidate,
    run_gnss_solve_wp7,
    score_segment,
)
from score_vs_inuex35 import load_reference_grid  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GNSS_SOLVE = PROJECT_ROOT / "third_party" / "gnssplusplus" / "build" / "apps" / "gnss_solve"
DEFAULT_DATA_ROOT = Path("E:/datasets/PPC-Dataset-data")

__all__ = [
    "CANYON_TOW_HI",
    "CANYON_TOW_LO",
    "CSV_FIELDS",
    "WP6_WINNER_ARGS",
    "Candidate",
    "append_csv_row",
    "build_full_argv",
    "nlos_mask_path",
    "run_and_score_candidate",
    "run_gnss_solve_wp7",
    "score_segment",
]


# ---------------------------------------------------------------------------
# Candidate stages
# ---------------------------------------------------------------------------

# Work item 2: coarse qpos sweep (~3 decades) per policy, full run1 only.
QPOS_GRID = ("0.1", "1", "10", "100")

STAGE_CVPREDICT_COARSE = [
    Candidate(
        f"a0_cvpredict_qpos{qpos}",
        ["--float-trust-policy", "cv-predict", "--trust-lapse-qpos", qpos],
    )
    for qpos in QPOS_GRID
]

STAGE_SCALEDRESET_COARSE = [
    Candidate(
        f"a1_scaledreset_qpos{qpos}",
        ["--float-trust-policy", "scaled-reset", "--trust-lapse-qpos", qpos],
    )
    for qpos in QPOS_GRID
]


def policy_flag_args(policy: str, qpos: str) -> list[str]:
    return ["--float-trust-policy", policy, "--trust-lapse-qpos", qpos]


# Work item 3: regression matrix -- winner policy+qpos verbatim on all three
# runs, plus a per-run baseline (no WP9 flags) for the delta computation.
def regression_candidates(policy: str, qpos: str) -> list[Candidate]:
    return [
        Candidate("b0_baseline_no_wp9", [], "WP7/dead-knob baseline for this run"),
        Candidate(
            f"b1_{policy}_qpos{qpos}",
            policy_flag_args(policy, qpos),
            "run1-selected winner, applied verbatim",
        ),
    ]


# Work item 4: winner + --hold-ratio-threshold 2.0 (WP8 rec 3), all 3 runs.
def combination_candidates(policy: str, qpos: str) -> list[Candidate]:
    return [
        Candidate("c0_baseline_no_wp9", [], "WP7/dead-knob baseline for this run"),
        Candidate(
            f"c1_{policy}_qpos{qpos}",
            policy_flag_args(policy, qpos),
            "winner alone",
        ),
        Candidate(
            "c2_hold2.0_alone",
            ["--hold-ratio-threshold", "2.0"],
            "WP8 rec 3 alone",
        ),
        Candidate(
            f"c3_combined_{policy}_qpos{qpos}_hold2.0",
            policy_flag_args(policy, qpos) + ["--hold-ratio-threshold", "2.0"],
            "winner + WP8 rec 3 combined",
        ),
    ]


# Work item 5 (stretch): single coarse test of the WP8-rec-2 stretch knob,
# --nlos-min-los-sats, combined with the winning WP9 policy and WP7's
# continuous soft weighting, on run1 only.
def stretch_candidates(policy: str, qpos: str, min_los_sats: int) -> list[Candidate]:
    return [
        Candidate(
            f"d0_{policy}_qpos{qpos}_nlosminlos{min_los_sats}",
            policy_flag_args(policy, qpos)
            + [
                "--nlos-weight-mode", "continuous",
                "--nlos-continuous-floor", "0.5",
                "--nlos-min-los-sats", str(min_los_sats),
            ],
            "winner + WP7 continuous weighting + WP8-rec-2 min-LOS-sats stretch",
            needs_nlos_csv=True,
        ),
    ]


# Optional lever (work item 1's own flag): --trust-gate-nlos-relax, layered
# on top of the winning policy+qpos, single coarse test on run1.
def nlos_relax_candidates(policy: str, qpos: str) -> list[Candidate]:
    return [
        Candidate(
            f"e0_{policy}_qpos{qpos}_nlosrelax",
            policy_flag_args(policy, qpos) + ["--trust-gate-nlos-relax"],
            "winner + optional NLOS-relaxed trust jump gate",
            needs_nlos_csv=True,
        ),
    ]


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city", default="tokyo")
    parser.add_argument("--run", default="run1")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--sweep-csv", type=Path, required=True)
    parser.add_argument("--skip-epochs", type=int, default=0)
    parser.add_argument("--max-epochs", type=int, default=0)
    parser.add_argument("--warmup-s", type=float, default=300.0)
    parser.add_argument("--stage", required=True, choices=sorted(_STAGE_BUILDERS.keys()))
    parser.add_argument("--best-policy", default="cv-predict")
    parser.add_argument("--best-qpos", default="10")
    parser.add_argument("--min-los-sats", type=int, default=4)
    args = parser.parse_args(argv)

    city, run = args.city, args.run
    run_dir = args.data_root / city / run
    rover, base, nav = run_dir / "rover.obs", run_dir / "base.obs", run_dir / "base.nav"
    for f in (rover, base, nav):
        if not f.is_file():
            print(f"[error] missing {f}", file=sys.stderr)
            return 2
    if not GNSS_SOLVE.is_file():
        print(f"[error] missing WSL gnss_solve build: {GNSS_SOLVE}", file=sys.stderr)
        return 2

    nlos_csv = nlos_mask_path(city, run)
    if not nlos_csv.is_file():
        print(f"[warn] missing NLOS mask csv: {nlos_csv} (nlos-flavored candidates will fail)", file=sys.stderr)

    reference = load_reference_grid(city, run, data_root=args.data_root)
    run_start_tow = min(reference)

    candidates = _STAGE_BUILDERS[args.stage](args)
    for candidate in candidates:
        print(f"[sweep] running {candidate.name}: {' '.join(candidate.extra_args) or '(wp7 baseline)'}", flush=True)
        row = run_and_score_candidate(
            candidate,
            rover=rover,
            base=base,
            nav=nav,
            out_dir=args.out_dir,
            reference=reference,
            run_start_tow=run_start_tow,
            city=city,
            run=run,
            nlos_csv=nlos_csv,
            skip_epochs=args.skip_epochs,
            max_epochs=args.max_epochs,
            warmup_s=args.warmup_s,
        )
        append_csv_row(args.sweep_csv, row)
        print(f"[sweep]   -> {row}", flush=True)
    return 0


_STAGE_BUILDERS = {
    "cvpredict_coarse": lambda args: STAGE_CVPREDICT_COARSE,
    "scaledreset_coarse": lambda args: STAGE_SCALEDRESET_COARSE,
    "regression": lambda args: regression_candidates(args.best_policy, args.best_qpos),
    "combination": lambda args: combination_candidates(args.best_policy, args.best_qpos),
    "stretch": lambda args: stretch_candidates(args.best_policy, args.best_qpos, args.min_los_sats),
    "nlos_relax": lambda args: nlos_relax_candidates(args.best_policy, args.best_qpos),
}


if __name__ == "__main__":
    raise SystemExit(main())
