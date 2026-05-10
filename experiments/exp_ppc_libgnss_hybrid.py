#!/usr/bin/env python3
# ruff: noqa: E402
"""Hybrid libgnss++ RTK + SPP scorer.

Runs ``gnss_solve`` (RTK with LAMBDA) and ``gnss_spp`` (standard point
positioning) on each full PPC run, then builds a hybrid per-epoch
position: prefer RTK solutions where available (FIXED > FLOAT), fall
back to SPP where RTK emits nothing. Scores PPC distance-weighted
across all 6 runs.

The RTK-only benchmark (section 17/18 of the plan) leaves many rover
epochs with no solution because the Kalman filter declines to update
under urban-canyon NLOS. Filling those gaps with meter-level SPP keeps
the total distance covered high; the 0.5 m PPC threshold still passes
on slow/stopped segments where SPP happens to be close to truth.
"""

from __future__ import annotations

import argparse
import csv
from math import atan2, cos, hypot, sin, sqrt
from pathlib import Path
import subprocess
import sys

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "python"))

from gnss_gpu.ppc_score import ppc_score_dict, score_ppc2024

RESULTS_DIR = _SCRIPT_DIR / "results"
_GNSS_SOLVE = _PROJECT_ROOT / "third_party/gnssplusplus/build/apps/gnss_solve"
_GNSS_SPP = _PROJECT_ROOT / "third_party/gnssplusplus/build/apps/gnss_spp"
_DEFAULT_DATA_ROOT = Path("datasets/PPC-Dataset-data")

_PROFILES = {
    "tokyo": [
        "--preset", "low-cost", "--arfilter", "--arfilter-margin", "0.35",
        "--min-hold-count", "8", "--hold-ratio-threshold", "2.6",
    ],
    "nagoya": [
        "--preset", "low-cost", "--min-hold-count", "7", "--hold-ratio-threshold", "2.4",
    ],
    # Phase 10g full-run sweep: this profile only generalized on nagoya/run2.
    # Applying it to tokyo/run1 or nagoya/run3 improves short windows but causes
    # false fixes over full urban-canyon runs.
    "nagoya2-phase10-lock3": [
        "--preset", "low-cost",
        "--min-hold-count", "7", "--hold-ratio-threshold", "2.4",
        "--ratio", "1.5", "--min-ar-sats", "4", "--min-lock-count", "3",
        "--prefer-trusted-seed",
    ],
}

_FULL_RUNS = (
    ("tokyo", "run1", "tokyo"),
    ("tokyo", "run2", "tokyo"),
    ("tokyo", "run3", "tokyo"),
    ("nagoya", "run1", "nagoya"),
    ("nagoya", "run2", "nagoya"),
    ("nagoya", "run3", "nagoya"),
)

_PHASE10_RESIDUAL_CANDIDATE_KNOBS = [
    "--ratio", "1.5",
    "--min-ar-sats", "4",
    "--min-lock-count", "3",
    "--prefer-trusted-seed",
]


def _profile_for_run(city: str, run: str, profile_mode: str, default_profile: str) -> str:
    if profile_mode == "city":
        return default_profile
    if profile_mode == "phase10-mix-nagoya2" and city == "nagoya" and run == "run2":
        return "nagoya2-phase10-lock3"
    return default_profile


def _phase10_residual_candidate_profile(default_profile: str) -> list[str]:
    """Candidate RTK profile used by the residual-gated dual-profile chooser."""
    return [*_PROFILES[default_profile], *_PHASE10_RESIDUAL_CANDIDATE_KNOBS]


def _load_reference(path: Path) -> list[tuple[float, np.ndarray]]:
    rows: list[tuple[float, np.ndarray]] = []
    with path.open() as fh:
        reader = csv.reader(fh)
        next(reader)
        for row in reader:
            tow = round(float(row[0]), 1)
            rows.append(
                (tow, np.array([float(row[5]), float(row[6]), float(row[7])], dtype=np.float64))
            )
    return rows


def _parse_pos(path: Path) -> dict[float, tuple[np.ndarray, int]]:
    out: dict[float, tuple[np.ndarray, int]] = {}
    with path.open() as fh:
        for line in fh:
            if line.startswith("%") or not line.strip():
                continue
            parts = line.split()
            if len(parts) < 10:
                continue
            tow = round(float(parts[1]), 1)
            ecef = np.array(
                [float(parts[2]), float(parts[3]), float(parts[4])],
                dtype=np.float64,
            )
            status = int(parts[8])
            out[tow] = (ecef, status)
    return out


def _parse_diag(path: Path) -> dict[float, dict[str, str]]:
    out: dict[float, dict[str, str]] = {}
    with path.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            tow = round(float(row["tow"]), 1)
            out[tow] = row
    return out


def _diag_float(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def _use_phase10_residual_candidate(
    row: dict[str, str] | None,
    *,
    ratio_min: float,
    residual_rms_max: float,
) -> bool:
    """Gate that generalized in Phase 10g full-run diagnostics.

    Base solution is libgnss++ v5. Candidate solution is the relaxed
    ratio=1.5/min_lock=3/trusted-seed profile. We only promote candidate
    epochs when they are fixed and the RTK measurement update residual is
    tight enough.
    """
    if row is None:
        return False
    return (
        int(row.get("output_added", "0")) == 1
        and int(row.get("final_status", "0")) == 4
        and _diag_float(row, "final_ratio") >= ratio_min
        and _diag_float(row, "final_residual_rms") <= residual_rms_max
    )


def _run_solver(cmd: list[str]) -> None:
    completed = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
    if completed.returncode != 0:
        raise RuntimeError(
            f"solver failed ({completed.returncode})\n"
            f"cmd: {' '.join(cmd)}\nstderr: {completed.stderr[-1500:]}"
        )


def _ecef_to_lat_lon(ecef: np.ndarray) -> tuple[float, float]:
    x, y, z = float(ecef[0]), float(ecef[1]), float(ecef[2])
    lon = atan2(y, x)
    p = hypot(x, y)
    e2 = 6.694379990141316e-3
    lat = atan2(z, p * (1.0 - e2))
    for _ in range(6):
        sl = sin(lat)
        n = 6_378_137.0 / sqrt(1.0 - e2 * sl * sl)
        lat = atan2(z + e2 * n * sl, p)
    return lat, lon


def _enu_rotation(lat: float, lon: float) -> np.ndarray:
    sl, cl = sin(lat), cos(lat)
    so, co = sin(lon), cos(lon)
    return np.array(
        [
            [-so, co, 0.0],
            [-sl * co, -sl * so, cl],
            [cl * co, cl * so, sl],
        ],
        dtype=np.float64,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hybrid libgnss++ RTK + SPP scorer on PPC full runs",
    )
    parser.add_argument("--data-root", type=Path, default=_DEFAULT_DATA_ROOT)
    parser.add_argument("--rtk-bin", type=Path, default=_GNSS_SOLVE)
    parser.add_argument("--spp-bin", type=Path, default=_GNSS_SPP)
    parser.add_argument(
        "--pos-dir",
        type=Path,
        default=_SCRIPT_DIR / "results" / "libgnss_rtk_pos",
    )
    parser.add_argument(
        "--spp-dir",
        type=Path,
        default=_SCRIPT_DIR / "results" / "libgnss_spp_pos",
    )
    parser.add_argument(
        "--candidate-pos-dir",
        type=Path,
        default=None,
        help=(
            "Alternate RTK .pos directory used by --chooser-mode. If omitted with a "
            "chooser enabled, defaults under experiments/results and is generated "
            "when solvers are not skipped."
        ),
    )
    parser.add_argument(
        "--candidate-diag-dir",
        type=Path,
        default=None,
        help=(
            "Diagnostics CSV directory used by --chooser-mode. If omitted with a "
            "chooser enabled, defaults under experiments/results and is generated "
            "when solvers are not skipped."
        ),
    )
    parser.add_argument(
        "--chooser-mode",
        choices=("none", "phase10-residual"),
        default="none",
        help="Per-epoch selection between --pos-dir and --candidate-pos-dir.",
    )
    parser.add_argument(
        "--chooser-ratio-min",
        type=float,
        default=1.5,
        help="Minimum candidate final_ratio for --chooser-mode phase10-residual.",
    )
    parser.add_argument(
        "--chooser-residual-rms-max",
        type=float,
        default=1.8,
        help="Maximum candidate final_residual_rms for --chooser-mode phase10-residual.",
    )
    parser.add_argument("--results-prefix", type=str, default="ppc_libgnss_hybrid")
    parser.add_argument(
        "--profile-mode",
        choices=("city", "phase10-mix-nagoya2"),
        default="city",
        help=(
            "city: use the baseline tokyo/nagoya profiles. "
            "phase10-mix-nagoya2: use the Phase10 lock-count profile only for nagoya/run2."
        ),
    )
    parser.add_argument(
        "--skip-solvers",
        action="store_true",
        help="Reuse existing .pos files without re-running gnss_solve / gnss_spp",
    )
    parser.add_argument(
        "--prefer",
        choices=("rtk", "fixed"),
        default="rtk",
        help=(
            "rtk: prefer any RTK solution (FIXED or FLOAT) over SPP. "
            "fixed: prefer only FIXED RTK; FLOAT RTK loses to SPP when the "
            "float error is large."
        ),
    )
    args = parser.parse_args()

    rtk_bin = args.rtk_bin.resolve()
    spp_bin = args.spp_bin.resolve()
    data_root = args.data_root.resolve()
    pos_dir = args.pos_dir.resolve()
    spp_dir = args.spp_dir.resolve()
    candidate_pos_dir = args.candidate_pos_dir.resolve() if args.candidate_pos_dir else None
    candidate_diag_dir = args.candidate_diag_dir.resolve() if args.candidate_diag_dir else None
    if args.chooser_mode != "none":
        if candidate_pos_dir is None:
            candidate_pos_dir = (
                RESULTS_DIR / "libgnss_rtk_pos_phase10_residual_candidate"
            ).resolve()
        if candidate_diag_dir is None:
            candidate_diag_dir = (
                RESULTS_DIR / "libgnss_diag_phase10_residual_candidate"
            ).resolve()
    pos_dir.mkdir(parents=True, exist_ok=True)
    spp_dir.mkdir(parents=True, exist_ok=True)
    if candidate_pos_dir is not None:
        candidate_pos_dir.mkdir(parents=True, exist_ok=True)
    if candidate_diag_dir is not None:
        candidate_diag_dir.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    agg_pass = agg_total = 0.0
    rtk_only_pass = rtk_only_total = 0.0

    for city, run, default_profile in _FULL_RUNS:
        profile = _profile_for_run(city, run, args.profile_mode, default_profile)
        run_dir = data_root / city / run
        rtk_pos = pos_dir / f"{city}_{run}_full.pos"
        spp_pos = spp_dir / f"{city}_{run}_full.pos"
        print(f"[run] {city}/{run}", flush=True)
        if not args.skip_solvers or not rtk_pos.exists():
            print(f"  running RTK ({profile} profile)...", flush=True)
            _run_solver(
                [
                    str(rtk_bin),
                    "--rover", str(run_dir / "rover.obs"),
                    "--base", str(run_dir / "base.obs"),
                    "--nav", str(run_dir / "base.nav"),
                    "--skip-epochs", "0",
                    "--out", str(rtk_pos),
                    "--no-kml",
                    *_PROFILES[profile],
                ]
            )
        if not args.skip_solvers or not spp_pos.exists():
            print("  running SPP...", flush=True)
            _run_solver(
                [
                    str(spp_bin),
                    "--obs", str(run_dir / "rover.obs"),
                    "--nav", str(run_dir / "base.nav"),
                    "--out", str(spp_pos),
                    "--quiet",
                ]
            )

        ref = _load_reference(run_dir / "reference.csv")
        rtk = _parse_pos(rtk_pos)
        spp = _parse_pos(spp_pos)
        candidate_rtk: dict[float, tuple[np.ndarray, int]] = {}
        candidate_diag: dict[float, dict[str, str]] = {}
        if args.chooser_mode != "none":
            assert candidate_pos_dir is not None
            assert candidate_diag_dir is not None
            candidate_pos = candidate_pos_dir / f"{city}_{run}_full.pos"
            candidate_csv = candidate_diag_dir / f"{city}_{run}_full.csv"
            if not args.skip_solvers or not candidate_pos.exists() or not candidate_csv.exists():
                print("  running candidate RTK diagnostics (phase10-residual profile)...", flush=True)
                _run_solver(
                    [
                        str(rtk_bin),
                        "--rover", str(run_dir / "rover.obs"),
                        "--base", str(run_dir / "base.obs"),
                        "--nav", str(run_dir / "base.nav"),
                        "--skip-epochs", "0",
                        "--out", str(candidate_pos),
                        "--diagnostics-csv", str(candidate_csv),
                        "--no-kml",
                        *_phase10_residual_candidate_profile(default_profile),
                    ]
                )
            if not candidate_pos.exists():
                raise FileNotFoundError(f"candidate .pos not found: {candidate_pos}")
            if not candidate_csv.exists():
                raise FileNotFoundError(f"candidate diagnostics CSV not found: {candidate_csv}")
            candidate_rtk = _parse_pos(candidate_pos)
            candidate_diag = _parse_diag(candidate_csv)

        def _chosen_rtk(tow: float) -> tuple[np.ndarray, int, bool] | None:
            if (
                args.chooser_mode == "phase10-residual"
                and _use_phase10_residual_candidate(
                    candidate_diag.get(tow),
                    ratio_min=args.chooser_ratio_min,
                    residual_rms_max=args.chooser_residual_rms_max,
                )
            ):
                candidate = candidate_rtk.get(tow)
                if candidate is not None:
                    return candidate[0], candidate[1], True
            base = rtk.get(tow)
            if base is None:
                return None
            return base[0], base[1], False

        fused: list[np.ndarray] = []
        truth: list[np.ndarray] = []
        src_codes: list[int] = []  # 4=FIXED RTK, 3=FLOAT RTK, 1=SPP, 0=missing
        candidate_pick_count = 0
        for tow, tvec in ref:
            r = _chosen_rtk(tow)
            s = spp.get(tow)
            picked: tuple[np.ndarray, int] | None = None
            if r is not None:
                ecef_r, st_r, is_candidate = r
                if is_candidate:
                    candidate_pick_count += 1
                if args.prefer == "fixed":
                    if st_r == 4:
                        picked = (ecef_r, 4)
                    elif s is not None:
                        picked = (s[0], 1)
                    else:
                        picked = (ecef_r, st_r)
                else:
                    picked = (ecef_r, st_r)
            elif s is not None:
                picked = (s[0], 1)
            if picked is None:
                continue
            fused.append(picked[0])
            truth.append(tvec)
            src_codes.append(picked[1])
        fused_arr = np.array(fused, dtype=np.float64)
        truth_arr = np.array(truth, dtype=np.float64)
        src_arr = np.array(src_codes, dtype=np.int32)

        # RTK-only scoring for comparison (same reference ordering but only
        # epochs where RTK had a solution).
        rtk_only_fused: list[np.ndarray] = []
        rtk_only_truth: list[np.ndarray] = []
        rtk_only_src: list[int] = []
        for tow, tvec in ref:
            r = _chosen_rtk(tow)
            if r is None:
                continue
            rtk_only_fused.append(r[0])
            rtk_only_truth.append(tvec)
            rtk_only_src.append(r[1])

        ppc_hybrid = ppc_score_dict(fused_arr, truth_arr)
        rtk_arr = np.array(rtk_only_fused, dtype=np.float64) if rtk_only_fused else np.empty((0, 3))
        rtk_t_arr = np.array(rtk_only_truth, dtype=np.float64) if rtk_only_truth else np.empty((0, 3))
        ppc_rtk = (
            ppc_score_dict(rtk_arr, rtk_t_arr)
            if rtk_arr.size
            else {"ppc_score_pct": 0.0, "ppc_pass_distance_m": 0.0, "ppc_total_distance_m": 0.0}
        )

        # Honest full-truth-denominator scoring: every rover epoch counts in
        # the denominator. Missing epochs are filled with the origin so the
        # error is finite (> 0.5 m) and the segment distance is retained in
        # total_distance but not in pass_distance.
        full_truth = np.array([t for _, t in ref], dtype=np.float64)
        full_est_rtk = np.zeros_like(full_truth)
        full_est_hyb = np.zeros_like(full_truth)
        for i, (tow, _tvec) in enumerate(ref):
            r = _chosen_rtk(tow)
            s = spp.get(tow)
            if r is not None:
                full_est_rtk[i] = r[0]
                full_est_hyb[i] = r[0]
            elif s is not None:
                full_est_hyb[i] = s[0]
        honest_rtk = score_ppc2024(full_est_rtk, full_truth)
        honest_hyb = score_ppc2024(full_est_hyb, full_truth)

        diff = fused_arr - truth_arr
        if diff.size:
            center = truth_arr.mean(axis=0)
            lat0, lon0 = _ecef_to_lat_lon(center)
            R = _enu_rotation(lat0, lon0)
            enu = np.array([R @ d for d in diff])
            e2d = np.linalg.norm(enu[:, :2], axis=1)
        else:
            e2d = np.array([])

        n_fixed = int((src_arr == 4).sum())
        n_float = int((src_arr == 3).sum())
        n_spp = int((src_arr == 1).sum())
        n_total_ref = len(ref)

        row = {
            "city": city,
            "run": run,
            "profile_mode": args.profile_mode,
            "profile": profile,
            "n_ref_epochs": n_total_ref,
            "n_hybrid": int(fused_arr.shape[0]),
            "hybrid_coverage_pct": float(100.0 * fused_arr.shape[0] / max(n_total_ref, 1)),
            "n_fixed": n_fixed,
            "n_float": n_float,
            "n_spp_fill": n_spp,
            "chooser_mode": args.chooser_mode,
            "chooser_ratio_min": args.chooser_ratio_min,
            "chooser_residual_rms_max": args.chooser_residual_rms_max,
            "n_candidate_pick": candidate_pick_count,
            "rtk_subset_ppc_pct": float(ppc_rtk["ppc_score_pct"]),
            "rtk_subset_pass_m": float(ppc_rtk["ppc_pass_distance_m"]),
            "rtk_subset_total_m": float(ppc_rtk["ppc_total_distance_m"]),
            "hybrid_subset_ppc_pct": float(ppc_hybrid["ppc_score_pct"]),
            "rtk_honest_ppc_pct": float(honest_rtk.score_pct),
            "rtk_honest_pass_m": float(honest_rtk.pass_distance_m),
            "hybrid_honest_ppc_pct": float(honest_hyb.score_pct),
            "hybrid_honest_pass_m": float(honest_hyb.pass_distance_m),
            "true_arc_length_m": float(honest_rtk.total_distance_m),
            "hybrid_2d_median_m": float(np.median(e2d)) if e2d.size else float("nan"),
            "hybrid_2d_p95_m": float(np.percentile(e2d, 95)) if e2d.size else float("nan"),
        }
        rows.append(row)

        print(
            f"    cov={row['hybrid_coverage_pct']:5.1f}%  "
            f"FIX={n_fixed} FLOAT={n_float} SPP={n_spp}  "
            f"CAND={candidate_pick_count}  "
            f"RTK-subset={row['rtk_subset_ppc_pct']:.2f}%  "
            f"HYBRID-subset={row['hybrid_subset_ppc_pct']:.2f}%  "
            f"RTK-honest={row['rtk_honest_ppc_pct']:.2f}%  "
            f"HYBRID-honest={row['hybrid_honest_ppc_pct']:.2f}%",
            flush=True,
        )
        agg_pass += row["hybrid_honest_pass_m"]
        agg_total += row["true_arc_length_m"]
        rtk_only_pass += row["rtk_honest_pass_m"]
        rtk_only_total += row["true_arc_length_m"]

    out_csv = RESULTS_DIR / f"{args.results_prefix}_runs.csv"
    fieldnames: list[str] = []
    seen: set[str] = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    print()
    print("=" * 72)
    print("  Honest aggregates (denominator = full rover-epoch arc length;")
    print("  this is the metric directly comparable to TURING's 85.6%)")
    if agg_total > 0:
        print(f"  HYBRID honest   : {100 * agg_pass / agg_total:.2f}%"
              f"  (pass {agg_pass:.1f}m / total {agg_total:.1f}m)")
    if rtk_only_total > 0:
        print(f"  RTK-only honest : {100 * rtk_only_pass / rtk_only_total:.2f}%"
              f"  (pass {rtk_only_pass:.1f}m / total {rtk_only_total:.1f}m)")
    print(f"  Saved: {out_csv}")
    print("=" * 72)


if __name__ == "__main__":
    main()
