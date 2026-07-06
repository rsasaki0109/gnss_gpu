#!/usr/bin/env python3
"""WP8 driver: sweep libgnss++ ``gnss_solve`` hard NLOS exclusion + the
now-active live AR knobs (``--arfilter-margin``/``--hold-ratio-threshold``)
on PPC tokyo runs, on top of the WP7 baseline (WP6 winner base + wired dead
knobs, no NLOS weighting).

Thin extension of ``experiments/sweep_libgnss_rtk_wp7.py`` (reuses its
``build_full_argv``/canyon-segment scoring/CSV-row shape). The only
differences:

1. New candidate stages: exclusion threshold x min-sats coarse+refine grid
   (work item 2), a retune grid over ``--arfilter-margin`` x
   ``--hold-ratio-threshold`` that runs WITHOUT any ``--nlos-*`` flags (work
   item 4), and a "combined" stage layering both on top of the WP7
   baseline (work item 5).
2. ``needs_nlos_csv`` candidates get ``--nlos-weight-mode exclude`` +
   ``--nlos-exclude-threshold``/``--nlos-min-sats`` instead of the WP7
   two-tier/continuous flags.

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

# Work item 2, coarse stage: threshold {0.3, 0.5} x min-sats {4, 5, 6}, full
# run1 only (per the task's own "full-run sweeps only" constraint).
STAGE_EXCLUDE_COARSE = [
    Candidate(
        f"a0_exclude_thr{thr}_minsats{minsats}",
        [
            "--nlos-weight-mode", "exclude",
            "--nlos-exclude-threshold", thr,
            "--nlos-min-sats", str(minsats),
        ],
        needs_nlos_csv=True,
    )
    for thr in ("0.3", "0.5")
    for minsats in (4, 5, 6)
]


def exclude_refine_candidates(best_threshold: str, best_min_sats: int) -> list[Candidate]:
    """Refine +/-1 point around the coarse stage's best min-sats value.

    ``min_sats`` is the only integer-valued knob in the grid (threshold only
    has 2 coarse points), so "+/-1 point" is naturally a min-sats bracket
    around whatever the coarse stage found, at the coarse winner's own
    threshold.
    """
    candidates = []
    for minsats in sorted({best_min_sats - 1, best_min_sats + 1}):
        if minsats < 0 or minsats == best_min_sats:
            continue
        candidates.append(
            Candidate(
                f"a1_exclude_thr{best_threshold}_minsats{minsats}",
                [
                    "--nlos-weight-mode", "exclude",
                    "--nlos-exclude-threshold", best_threshold,
                    "--nlos-min-sats", str(minsats),
                ],
                f"refine +/-1 around coarse winner (minsats={best_min_sats})",
                needs_nlos_csv=True,
            )
        )
    return candidates


def exclude_generalize_candidates(best_threshold: str, best_min_sats: int) -> list[Candidate]:
    """Best exclusion config, verbatim, for run2/run3 generalization."""
    return [
        Candidate("b0_baseline_no_nlos", [], "WP7 baseline (no NLOS) for this run"),
        Candidate(
            f"b1_exclude_thr{best_threshold}_minsats{best_min_sats}",
            [
                "--nlos-weight-mode", "exclude",
                "--nlos-exclude-threshold", best_threshold,
                "--nlos-min-sats", str(best_min_sats),
            ],
            "best run1-selected exclusion config, applied verbatim",
            needs_nlos_csv=True,
        ),
    ]


# Work item 4: retune the now-active dead-knob-fix defaults
# (--preset low-cost's own ar_filter_margin=0.35/hold_ratio_threshold=2.5).
# Explicitly WITHOUT any --nlos-* flags (kept orthogonal to item 2), full
# run1 only. 4 x 3 = 12 candidates.
STAGE_RETUNE = [
    Candidate(
        f"c0_margin{margin}_hold{hold}",
        ["--arfilter", "--arfilter-margin", margin, "--hold-ratio-threshold", hold],
    )
    for margin in ("0.0", "0.2", "0.35", "0.5")
    for hold in ("2.0", "2.5", "3.0")
]


def retune_generalize_candidates(best_margin: str, best_hold: str) -> list[Candidate]:
    return [
        Candidate("d0_baseline_no_retune", [], "WP7 baseline (preset defaults) for this run"),
        Candidate(
            f"d1_margin{best_margin}_hold{best_hold}",
            ["--arfilter", "--arfilter-margin", best_margin, "--hold-ratio-threshold", best_hold],
            "best run1-selected retune config, applied verbatim",
        ),
    ]


def combined_candidates(
    best_threshold: str, best_min_sats: int, best_margin: str, best_hold: str
) -> list[Candidate]:
    """Work item 5: both levers together, for all three runs."""
    return [
        Candidate("e0_baseline_no_nlos_no_retune", [], "WP7 baseline for this run"),
        Candidate(
            f"e1_exclude_thr{best_threshold}_minsats{best_min_sats}",
            [
                "--nlos-weight-mode", "exclude",
                "--nlos-exclude-threshold", best_threshold,
                "--nlos-min-sats", str(best_min_sats),
            ],
            "exclusion winner alone",
            needs_nlos_csv=True,
        ),
        Candidate(
            f"e2_margin{best_margin}_hold{best_hold}",
            ["--arfilter", "--arfilter-margin", best_margin, "--hold-ratio-threshold", best_hold],
            "retune winner alone",
        ),
        Candidate(
            f"e3_combined_thr{best_threshold}_minsats{best_min_sats}_margin{best_margin}_hold{best_hold}",
            [
                "--nlos-weight-mode", "exclude",
                "--nlos-exclude-threshold", best_threshold,
                "--nlos-min-sats", str(best_min_sats),
                "--arfilter", "--arfilter-margin", best_margin,
                "--hold-ratio-threshold", best_hold,
            ],
            "exclusion + retune combined",
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
    parser.add_argument("--best-threshold", default="0.5")
    parser.add_argument("--best-min-sats", type=int, default=5)
    parser.add_argument("--best-margin", default="0.35")
    parser.add_argument("--best-hold", default="2.5")
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
        print(f"[warn] missing NLOS mask csv: {nlos_csv} (exclude candidates will fail)", file=sys.stderr)

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
    "exclude_coarse": lambda args: STAGE_EXCLUDE_COARSE,
    "exclude_refine": lambda args: exclude_refine_candidates(args.best_threshold, args.best_min_sats),
    "exclude_generalize": lambda args: exclude_generalize_candidates(args.best_threshold, args.best_min_sats),
    "retune": lambda args: STAGE_RETUNE,
    "retune_generalize": lambda args: retune_generalize_candidates(args.best_margin, args.best_hold),
    "combined": lambda args: combined_candidates(
        args.best_threshold, args.best_min_sats, args.best_margin, args.best_hold
    ),
}


if __name__ == "__main__":
    raise SystemExit(main())
