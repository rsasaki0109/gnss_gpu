#!/usr/bin/env python3
"""WP10 driver: sweep libgnss++ ``gnss_solve``'s lapse-gated float-trust
policy (``--float-trust-policy lapse-gated`` + ``--trust-lapse-gate-s`` +
``--trust-lapse-qpos``) and the min-LOS-sats AR-acceptance gate
(``--nlos-min-los-sats``) on PPC tokyo runs, on top of the WP7/dead-knob
baseline (``--preset low-cost --max-pos-jump-rate 2.3``).

Thin extension of ``experiments/sweep_libgnss_rtk_wp9.py`` (reuses its own
re-exports of ``sweep_libgnss_rtk_wp7``'s ``build_full_argv``/canyon-segment
scoring/CSV-row shape). The only differences:

1. New candidate stages: a lapse-gate duration sweep (work item 3), a
   bit-identity check at a huge gate value (work item 1's own constraint), a
   run2/run3 regression matrix for the winning gate (work item 4), a
   min-los-sats coarse test (work item 5), and a combination stage (work
   item 6).
2. Following WP9's own documented fix for a real concurrent-file-collision
   bug (two parallel sweep processes writing the same out-dir/filename
   clobbered each other's ``.pos``), every stage invocation in this
   project's driving shell commands uses its own per-run ``--out-dir``
   (``results/wp10/sweep/run{1,2,3}/...``) -- never share an out-dir across
   two concurrently-launched sweep processes.

No changes are made to the libgnss++ C++ engine from this script; every
candidate is a config-only CLI variant of the already-built ``gnss_solve``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sweep_libgnss_rtk_wp9 import (  # noqa: E402
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

# Work item 1's own qpos recommendation ("the WP9 run1 winner").
LAPSE_GATED_QPOS = "0.1"

# Work item 3: gate duration sweep, full run1 only.
GATE_GRID = ("2", "5", "10", "20")


def lapse_gated_args(gate_s: str, qpos: str = LAPSE_GATED_QPOS) -> list[str]:
    return [
        "--float-trust-policy", "lapse-gated",
        "--trust-lapse-gate-s", gate_s,
        "--trust-lapse-qpos", qpos,
    ]


# Work item 2 follow-up (task item 4's contingency): "report ... whether item
# 2's NLOS trigger discriminates better" than the plain duration gate. Set
# the duration gate effectively unreachable (never fires on its own) so the
# NLOS-fraction condition alone decides when scaled-reset applies -- an
# environment-based trigger instead of a pure-duration one. 0.5 matches the
# existing --trust-gate-nlos-relax precedent (rtk.cpp's jump-gate relax uses
# the same > 0.5 threshold).
LAPSE_GATE_UNREACHABLE_S = "1000000"
NLOS_FRAC_GRID = ("0.5", "0.3")


def lapse_gated_nlos_frac_args(nlos_frac: str, qpos: str = LAPSE_GATED_QPOS) -> list[str]:
    return [
        "--float-trust-policy", "lapse-gated",
        "--trust-lapse-gate-s", LAPSE_GATE_UNREACHABLE_S,
        "--trust-lapse-gate-nlos-frac", nlos_frac,
        "--trust-lapse-qpos", qpos,
    ]


STAGE_NLOS_FRAC_SWEEP = [
    Candidate("f0_baseline_no_wp10", [], "WP7/dead-knob baseline"),
] + [
    Candidate(
        f"f1_nlosfractrig{frac}",
        lapse_gated_nlos_frac_args(frac),
        "item-2 NLOS-fraction-only trigger (duration gate unreachable)",
        needs_nlos_csv=True,
    )
    for frac in NLOS_FRAC_GRID
]


STAGE_GATE_SWEEP = [
    Candidate("a0_baseline_no_wp10", [], "WP7/dead-knob baseline"),
] + [
    Candidate(f"a1_lapsegated_gate{gate}_qpos{LAPSE_GATED_QPOS}", lapse_gated_args(gate))
    for gate in GATE_GRID
]

# Work item 1 constraint: lapse-gated at a gate value far larger than any
# lapse that occurs in the run must be bit-identical to the plain baseline
# (no --float-trust-policy at all) -- verified here via SHA-256 of the two
# .pos files by the caller (see PROGRESS.md/WP10_REPORT.md), not by this
# script itself (no hashing dependency here, keep this a plain candidate list
# so it's runnable the same way as every other stage).
STAGE_BITIDENTITY_CHECK = [
    Candidate("z0_baseline_no_wp10", [], "WP7/dead-knob baseline, for SHA-256 comparison"),
    Candidate(
        "z1_lapsegated_hugegate",
        lapse_gated_args("1000000"),
        "gate far larger than any real lapse -- must be byte-identical to z0",
    ),
]


# Work item 4: regression matrix -- winner gate (at qpos=0.1) verbatim on all
# three runs, plus a per-run baseline for the delta computation.
def regression_candidates(gate_s: str) -> list[Candidate]:
    return [
        Candidate("b0_baseline_no_wp10", [], "WP7/dead-knob baseline for this run"),
        Candidate(
            f"b1_lapsegated_gate{gate_s}_qpos{LAPSE_GATED_QPOS}",
            lapse_gated_args(gate_s),
            "run1-selected winning gate, applied verbatim",
        ),
    ]


# Work item 5: min-LOS-sats AR-acceptance gate, coarse test N in {4, 6},
# combined with WP7's continuous soft weighting (floor 0.5) per the task's
# explicit instruction, full run1 only.
MIN_LOS_SATS_GRID = ("4", "6")


def min_los_sats_args(n: str) -> list[str]:
    return [
        "--nlos-weight-mode", "continuous",
        "--nlos-continuous-floor", "0.5",
        "--nlos-min-los-sats", n,
    ]


STAGE_MIN_LOS_SATS = [
    Candidate("c0_baseline_no_wp10", [], "WP7/dead-knob baseline"),
] + [
    Candidate(
        f"c1_minlossats{n}",
        min_los_sats_args(n),
        "WP8 rec 2 AR-acceptance gate + WP7 continuous soft weighting",
        needs_nlos_csv=True,
    )
    for n in MIN_LOS_SATS_GRID
]


# Work item 6: combine the lapse-gate winner with the min-los-sats winner
# (only meaningful if both independently won their own stage).
def combination_candidates(gate_s: str, min_los_sats: str) -> list[Candidate]:
    return [
        Candidate("d0_baseline_no_wp10", [], "WP7/dead-knob baseline for this run"),
        Candidate(
            f"d1_lapsegated_gate{gate_s}",
            lapse_gated_args(gate_s),
            "lapse-gated winner alone",
        ),
        Candidate(
            f"d2_minlossats{min_los_sats}",
            min_los_sats_args(min_los_sats),
            "min-los-sats winner alone",
            needs_nlos_csv=True,
        ),
        Candidate(
            f"d3_combined_gate{gate_s}_minlossats{min_los_sats}",
            lapse_gated_args(gate_s) + min_los_sats_args(min_los_sats),
            "lapse-gated + min-los-sats combined",
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
    parser.add_argument("--best-gate", default="5")
    parser.add_argument("--best-min-los-sats", default="4")
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
    "gate_sweep": lambda args: STAGE_GATE_SWEEP,
    "bitidentity_check": lambda args: STAGE_BITIDENTITY_CHECK,
    "regression": lambda args: regression_candidates(args.best_gate),
    "min_los_sats": lambda args: STAGE_MIN_LOS_SATS,
    "nlos_frac_sweep": lambda args: STAGE_NLOS_FRAC_SWEEP,
    "combination": lambda args: combination_candidates(args.best_gate, args.best_min_los_sats),
}


if __name__ == "__main__":
    raise SystemExit(main())
