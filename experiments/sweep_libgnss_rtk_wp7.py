#!/usr/bin/env python3
"""WP7 driver: sweep libgnss++ ``gnss_solve`` NLOS-weighting + dead-knob wiring
knobs on PPC tokyo runs, on top of the WP6 winner base config.

Thin extension of ``experiments/sweep_libgnss_rtk_wp6.py`` (same WSL
``gnss_solve`` invocation pattern, same ``score_vs_inuex35`` scoring). The
only differences:

1. The base profile is ``--preset low-cost --max-pos-jump-rate 2.3`` (the
   WP6 winner, found on tokyo/run1 and confirmed to generalize to run2/run3
   without per-run tuning in ``results/wp6/WP6_REPORT.md``), not bare
   ``--preset low-cost``. WP7 candidates layer NLOS-weighting and dead-knob
   flags on top of that winner rather than re-deriving it.
2. Candidate lists cover ``--nlos-weights``/``--nlos-weight-mode``/
   ``--nlos-two-tier-*``/``--nlos-continuous-floor`` and ``--arfilter``/
   ``--hold-ratio-threshold`` (the two WP6-identified dead knobs, now wired).
3. ``run_and_score_candidate`` also reports a canyon-segment-only score
   (tow in [188990, 189070], the urban-canyon stretch WP6/WP7 tasked us to
   track specifically) alongside the full-run numbers.

No changes are made to the libgnss++ C++ engine from this script; every
candidate is a config-only CLI variant of the already-built ``gnss_solve``.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from score_vs_inuex35 import (  # noqa: E402
    ScoreResult,
    TrajectoryEpoch,
    load_pos_trajectory,
    load_reference_grid,
    score_trajectory,
)
from sweep_libgnss_rtk_wp6 import (  # noqa: E402
    CSV_FIELDS as _WP6_CSV_FIELDS,
    build_gnss_solve_argv,
    compute_fix_time_distribution,
    parse_engine_summary,
    to_wsl_path,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GNSS_SOLVE = PROJECT_ROOT / "third_party" / "gnssplusplus" / "build" / "apps" / "gnss_solve"
DEFAULT_DATA_ROOT = Path("E:/datasets/PPC-Dataset-data")
NLOS_MASK_DIR = PROJECT_ROOT / "experiments" / "results" / "plateau_nlos_phase33"

# The WP6 winner base config, applied verbatim (no per-run tuning) on top of
# ``--preset low-cost`` before any WP7 candidate's extra_args.
WP6_WINNER_ARGS = ["--max-pos-jump-rate", "2.3"]

# WP6/WP7 task-specified urban-canyon segment on tokyo/run1 to track
# separately from the full-run aggregate.
CANYON_TOW_LO = 188990.0
CANYON_TOW_HI = 189070.0


def nlos_mask_path(city: str, run: str) -> Path:
    return NLOS_MASK_DIR / f"{city}_{run}_per_epoch_nlos.csv"


@dataclass(frozen=True)
class Candidate:
    """One named ``gnss_solve`` config variant, layered on the WP6 winner."""

    name: str
    extra_args: list[str] = field(default_factory=list)
    note: str = ""
    needs_nlos_csv: bool = False


def build_full_argv(
    *,
    gnss_solve_path: Path,
    rover: Path,
    base: Path,
    nav: Path,
    out_pos: Path,
    candidate_extra_args: list[str],
    nlos_csv: Path | None,
    skip_epochs: int = 0,
    max_epochs: int = 0,
) -> list[str]:
    """WP6-winner base args + candidate args, with --nlos-weights resolved."""
    extra_args = list(WP6_WINNER_ARGS)
    if nlos_csv is not None:
        extra_args += ["--nlos-weights", to_wsl_path(nlos_csv)]
    extra_args += candidate_extra_args
    return build_gnss_solve_argv(
        gnss_solve_path=gnss_solve_path,
        rover=rover,
        base=base,
        nav=nav,
        out_pos=out_pos,
        extra_args=extra_args,
        skip_epochs=skip_epochs,
        max_epochs=max_epochs,
    )


def score_segment(
    epochs: list[TrajectoryEpoch],
    reference: dict,
    *,
    city: str,
    run: str,
    traj_path: Path,
    tow_lo: float,
    tow_hi: float,
) -> ScoreResult | None:
    """Score only the epochs whose tow falls in [tow_lo, tow_hi]."""
    segment = [e for e in epochs if tow_lo <= e.tow <= tow_hi]
    if not segment:
        return None
    return score_trajectory(segment, reference, city=city, run=run, traj_path=traj_path, fmt="pos")


import subprocess  # noqa: E402


def run_gnss_solve_wp7(
    *,
    rover: Path,
    base: Path,
    nav: Path,
    out_pos: Path,
    candidate_extra_args: list[str],
    nlos_csv: Path | None,
    skip_epochs: int = 0,
    max_epochs: int = 0,
    gnss_solve_path: Path = GNSS_SOLVE,
) -> tuple[int, str, float]:
    out_pos.parent.mkdir(parents=True, exist_ok=True)
    argv = build_full_argv(
        gnss_solve_path=gnss_solve_path,
        rover=rover,
        base=base,
        nav=nav,
        out_pos=out_pos,
        candidate_extra_args=candidate_extra_args,
        nlos_csv=nlos_csv,
        skip_epochs=skip_epochs,
        max_epochs=max_epochs,
    )
    start = time.monotonic()
    # WP9 fix: gnss_solve's stdout occasionally contains a byte that is not
    # valid in the Windows console's default (cp932) codepage under some
    # flag combinations (observed with --nlos-min-los-sats); decode as UTF-8
    # with replacement instead of failing the whole sweep on a cosmetic
    # decode error in captured diagnostic text.
    proc = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8", errors="replace")
    elapsed = time.monotonic() - start
    return proc.returncode, proc.stdout + proc.stderr, elapsed


CSV_FIELDS = _WP6_CSV_FIELDS + [
    "canyon_n_scored",
    "canyon_all_rms_m",
    "canyon_fix_rms_m",
    "canyon_fix_pct",
    "canyon_lt50cm_full_pct",
]


def run_and_score_candidate(
    candidate: Candidate,
    *,
    rover: Path,
    base: Path,
    nav: Path,
    out_dir: Path,
    reference: dict,
    run_start_tow: float,
    city: str,
    run: str,
    nlos_csv: Path | None,
    skip_epochs: int = 0,
    max_epochs: int = 0,
    warmup_s: float = 300.0,
    canyon_tow_lo: float = CANYON_TOW_LO,
    canyon_tow_hi: float = CANYON_TOW_HI,
) -> dict[str, object]:
    out_pos = out_dir / f"{candidate.name}.pos"
    resolved_nlos_csv = nlos_csv if candidate.needs_nlos_csv else None
    returncode, stdout, wall_s = run_gnss_solve_wp7(
        rover=rover,
        base=base,
        nav=nav,
        out_pos=out_pos,
        candidate_extra_args=candidate.extra_args,
        nlos_csv=resolved_nlos_csv,
        skip_epochs=skip_epochs,
        max_epochs=max_epochs,
    )
    engine_summary = parse_engine_summary(stdout)

    row: dict[str, object] = {
        "name": candidate.name,
        "note": candidate.note,
        "extra_args": " ".join(WP6_WINNER_ARGS + candidate.extra_args)
        + (f" --nlos-weights {resolved_nlos_csv.name}" if resolved_nlos_csv else ""),
        "wall_s": round(wall_s, 1),
        "returncode": returncode,
        "engine_total_solutions": engine_summary.get("total_solutions", ""),
        "engine_valid_solutions": engine_summary.get("valid_solutions", ""),
        "engine_fixed_solutions": engine_summary.get("fixed_solutions", ""),
        "engine_fix_rate_pct": engine_summary.get("engine_fix_rate_pct", ""),
    }
    if returncode != 0 or not out_pos.is_file():
        row["n_scored"] = 0
        row["stdout_tail"] = stdout[-2000:]
        return row

    epochs = load_pos_trajectory(out_pos)
    result: ScoreResult = score_trajectory(
        epochs, reference, city=city, run=run, traj_path=out_pos, fmt="pos"
    )
    dist = compute_fix_time_distribution(epochs, run_start_tow, warmup_s=warmup_s)
    row.update(
        {
            "n_scored": result.n_scored,
            "n_rover_epochs": result.n_rover_epochs,
            "coverage_pct": round(result.coverage_pct, 3),
            "all_rms_m": round(result.all_rms_m, 4),
            "fix_rms_m": "" if result.fix_rms_m is None else round(result.fix_rms_m, 4),
            "fix_pct": round(result.fix_pct, 3),
            "lt50cm_pct": round(result.lt50cm_pct, 3),
            "lt50cm_full_pct": round(result.lt50cm_full_pct, 3),
            "ppc_official_pct": (
                "" if result.ppc_official_pct != result.ppc_official_pct else round(result.ppc_official_pct, 3)
            ),
            "n_fix": int(dist["n_fix"]),
            "n_fix_after_warmup": int(dist["n_fix_after_warmup"]),
            "frac_fix_after_warmup": round(dist["frac_fix_after_warmup"], 4),
        }
    )

    canyon_result = score_segment(
        epochs, reference, city=city, run=run, traj_path=out_pos,
        tow_lo=canyon_tow_lo, tow_hi=canyon_tow_hi,
    )
    if canyon_result is not None:
        row.update(
            {
                "canyon_n_scored": canyon_result.n_scored,
                "canyon_all_rms_m": round(canyon_result.all_rms_m, 4),
                "canyon_fix_rms_m": "" if canyon_result.fix_rms_m is None else round(canyon_result.fix_rms_m, 4),
                "canyon_fix_pct": round(canyon_result.fix_pct, 3),
                "canyon_lt50cm_full_pct": round(canyon_result.lt50cm_full_pct, 3),
            }
        )
    else:
        row.update(
            {
                "canyon_n_scored": 0,
                "canyon_all_rms_m": "",
                "canyon_fix_rms_m": "",
                "canyon_fix_pct": "",
                "canyon_lt50cm_full_pct": "",
            }
        )
    return row


def append_csv_row(csv_path: Path, row: dict[str, object]) -> None:
    import csv

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS + ["stdout_tail"])
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in CSV_FIELDS + ["stdout_tail"]})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city", default="tokyo")
    parser.add_argument("--run", default="run1")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--sweep-csv", type=Path, required=True)
    parser.add_argument("--skip-epochs", type=int, default=0)
    parser.add_argument("--max-epochs", type=int, default=0)
    parser.add_argument("--warmup-s", type=float, default=300.0)
    parser.add_argument(
        "--candidates",
        default="stage0",
        choices=sorted(CANDIDATE_STAGES.keys()),
        help="which named candidate stage/list to run",
    )
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
        print(f"[warn] missing NLOS mask csv: {nlos_csv} (nlos candidates will fail)", file=sys.stderr)

    reference = load_reference_grid(city, run, data_root=args.data_root)
    run_start_tow = min(reference)

    candidates = CANDIDATE_STAGES[args.candidates]
    for candidate in candidates:
        print(f"[sweep] running {candidate.name}: {' '.join(candidate.extra_args) or '(wp6 winner base only)'}", flush=True)
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


# ---------------------------------------------------------------------------
# Candidate stages
# ---------------------------------------------------------------------------

# Stage 0: sanity/control -- confirms the WP6 winner base reproduces its own
# saved score byte-for-byte after the WP7 C++ changes (rtk.cpp NLOS hook +
# dead-knob wiring), i.e. the feature additions are inert when their flags
# are absent. Also re-checks the WP6-documented dead-knob behavior is now
# GONE (arfilter/hold-ratio flags should now visibly change the output).
STAGE0_CONTROL = [
    Candidate(
        "d0_deadknobs_forced_off",
        ["--no-arfilter", "--hold-ratio-threshold", "2.0"],
        "explicitly forces the pre-WP7 hardcoded dead-knob values -- must be "
        "byte-identical to results/wp6/final/run1/wp6_winner_jumprate_2.3.pos",
    ),
    Candidate(
        "d0_wp6_winner_rebuilt",
        [],
        "WP6 winner base, rebuilt gnss_solve, no explicit WP7 flags -- "
        "--preset low-cost's own enable_ar_filter=true/hold_ratio_threshold=2.5 "
        "(previously dead) now takes effect, so this is EXPECTED to differ from "
        "d0_deadknobs_forced_off/the historical wp6_winner_jumprate_2.3.pos -- "
        "this candidate's own output becomes the WP7 baseline for the NLOS sweep",
    ),
    Candidate(
        "d0_v5_baseline_flags_now_wired",
        ["--arfilter", "--arfilter-margin", "0.35", "--min-hold-count", "8", "--hold-ratio-threshold", "2.6"],
        "same flags WP6 found dead (byte-identical to bare preset in "
        "results/wp6/final/run1/baseline_log.txt) -- now expected to differ",
    ),
]

# Stage 1: NLOS two-tier coarse sweep -- vary the LOS/NLOS classification
# threshold and the sigma inflation applied to satellites classified NLOS.
STAGE1_TWOTIER = [
    Candidate(
        f"e1_twotier_thresh{thresh}_infl{infl}",
        [
            "--nlos-weight-mode", "two-tier",
            "--nlos-two-tier-threshold", thresh,
            "--nlos-two-tier-inflation", infl,
        ],
        needs_nlos_csv=True,
    )
    for thresh in ("0.5",)
    for infl in ("2", "3", "5", "10", "20")
]

# Stage 2: NLOS continuous mapping (sigma^2 *= 1/max(los_prob, floor)) coarse
# sweep over the floor.
STAGE2_CONTINUOUS = [
    Candidate(
        f"e2_continuous_floor{floor}",
        ["--nlos-weight-mode", "continuous", "--nlos-continuous-floor", floor],
        needs_nlos_csv=True,
    )
    for floor in ("0.5", "0.2", "0.1", "0.05", "0.01")
]

# Stage 3: generalization check on run2/run3 -- WP7 baseline (dead knobs now
# wired, no NLOS) vs the least-bad NLOS candidate from the run1 stage1/stage2
# sweep (continuous, floor=0.5 -- the mildest inflation tested, and the only
# candidate whose <50cm_full%/ppc_official%/coverage losses on run1 vs the
# no-NLOS baseline were smallest across the full 10-point grid). Applied
# verbatim, no per-run tuning, per the task's generalization requirement.
STAGE3_GENERALIZE = [
    Candidate("f3_baseline_no_nlos", [], "WP7 baseline (dead knobs wired, no NLOS) for this run"),
    Candidate(
        "f3_continuous_floor0.5",
        ["--nlos-weight-mode", "continuous", "--nlos-continuous-floor", "0.5"],
        "least-bad run1-selected NLOS mapping, applied verbatim",
        needs_nlos_csv=True,
    ),
]

CANDIDATE_STAGES: dict[str, list[Candidate]] = {
    "stage0": STAGE0_CONTROL,
    "stage1": STAGE1_TWOTIER,
    "stage2": STAGE2_CONTINUOUS,
    "stage3": STAGE3_GENERALIZE,
}


if __name__ == "__main__":
    raise SystemExit(main())
