#!/usr/bin/env python3
"""Oracle analysis for PPC RTKDiag candidate sets.

This measures whether poor PPC score is caused by candidate selection or by
the candidate pool itself.  The oracle uses reference.csv, so it is diagnostic
only and must not be used as a deployable policy.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "python"))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from exp_ppc_ctrbpf_fgo import (  # noqa: E402
    CTRBPFConfig,
    _apply_rtkdiag_run_index_policy,
    _load_full_reference,
    _load_hybrid_pos_file,
    _load_rtk_diag_file,
    _parse_label_list,
    _parse_path_list,
    _rtkdiag_candidate_gate,
)
from gnss_gpu.ppc_score import ppc_segment_distances, score_ppc2024  # noqa: E402

RESULTS_DIR = _SCRIPT_DIR / "results"
_DEFAULT_DATA_ROOT = Path("datasets/PPC-Dataset-data")
_FULL_RUNS = (
    ("tokyo", "run1"),
    ("tokyo", "run2"),
    ("tokyo", "run3"),
    ("nagoya", "run1"),
    ("nagoya", "run2"),
    ("nagoya", "run3"),
)

_PHASE11_LABELS = (
    "r15,r20,r25,r30,r15nh,r20g,r15g,r25g,r30g,"
    "r20g20,r20g40,r15g20,r25g20,r20g15,r15g15,r25g15"
)
_PHASE11_DIRS = (
    "experiments/results/libgnss_diag_phase10/full_ratio15_lock3_trustedseed,"
    "experiments/results/libgnss_diag_phase10/full_ratio2_lock3_trustedseed,"
    "experiments/results/libgnss_diag_phase10/full_ratio25_lock3_trustedseed,"
    "experiments/results/libgnss_diag_phase10/full_ratio3_lock3_trustedseed,"
    "experiments/results/libgnss_diag_phase10/full_ratio15_lock3_trustedseed_nohold,"
    "experiments/results/libgnss_diag_phase10/full_ratio2_lock3_trustedseed_gate30_min6,"
    "experiments/results/libgnss_diag_phase10/full_ratio15_lock3_trustedseed_gate30_min6,"
    "experiments/results/libgnss_diag_phase10/full_ratio25_lock3_trustedseed_gate30_min6,"
    "experiments/results/libgnss_diag_phase10/full_ratio3_lock3_trustedseed_gate30_min6,"
    "experiments/results/libgnss_diag_phase10/full_ratio2_lock3_trustedseed_gate20_min6,"
    "experiments/results/libgnss_diag_phase10/full_ratio2_lock3_trustedseed_gate40_min6,"
    "experiments/results/libgnss_diag_phase10/full_ratio15_lock3_trustedseed_gate20_min6,"
    "experiments/results/libgnss_diag_phase10/full_ratio25_lock3_trustedseed_gate20_min6,"
    "experiments/results/libgnss_diag_phase10/full_ratio2_lock3_trustedseed_gate15_min6,"
    "experiments/results/libgnss_diag_phase10/full_ratio15_lock3_trustedseed_gate15_min6,"
    "experiments/results/libgnss_diag_phase10/full_ratio25_lock3_trustedseed_gate15_min6"
)


def _parse_runs(raw: str) -> tuple[tuple[str, str], ...]:
    if raw == "all":
        return _FULL_RUNS
    wanted = {r.strip() for r in raw.split(",") if r.strip()}
    runs = tuple((c, r) for c, r in _FULL_RUNS if f"{c}/{r}" in wanted)
    if not runs:
        raise SystemExit(f"no matching runs: {raw}")
    return runs


def _phase11i_blocked_labels(city: str, run: str, policy: str) -> set[str]:
    blocked: set[str] = set()
    if policy in {"phase11h", "phase11i", "phase11l", "phase11n", "phase11x", "phase11y", "phase11z"} and city == "nagoya" and run == "run2":
        blocked.update({"r15g15", "r20g15", "r25g15", "r30g15"})
    if policy in {"phase11i", "phase11l", "phase11n", "phase11x", "phase11y", "phase11z"} and (
        (city, run) in {("tokyo", "run2"), ("nagoya", "run1"), ("nagoya", "run2")}
    ):
        blocked.update({"r30", "r30g"})
    if policy in {"phase11l", "phase11n", "phase11x", "phase11y", "phase11z"} and (
        (city, run) in {("nagoya", "run1"), ("nagoya", "run2")}
    ):
        blocked.add("r20g10")
    if policy in {"phase11n", "phase11x", "phase11y", "phase11z"} and city == "nagoya":
        blocked.update({"r15g10", "r25g10"})
    return blocked


def _score_positions(
    ref: list[tuple[float, np.ndarray]],
    positions: dict[float, np.ndarray],
):
    truth = np.asarray([t for _, t in ref], dtype=np.float64)
    est = np.asarray(
        [positions.get(round(float(tow), 1), np.zeros(3)) for tow, _ in ref],
        dtype=np.float64,
    )
    return score_ppc2024(est, truth)


def _oracle(
    ref: list[tuple[float, np.ndarray]],
    sources: list[tuple[str, dict[float, np.ndarray]]],
):
    tows = [round(float(tow), 1) for tow, _ in ref]
    truth = np.asarray([t for _, t in ref], dtype=np.float64)
    est = np.zeros((len(ref), 3), dtype=np.float64)
    picked_labels: list[str] = []
    errors = np.full(len(ref), np.inf, dtype=np.float64)
    for i, (tow, truth_i) in enumerate(zip(tows, truth, strict=True)):
        best_label = "missing"
        best_pos = None
        best_err = np.inf
        for label, positions in sources:
            pos = positions.get(tow)
            if pos is None:
                continue
            err = float(np.linalg.norm(np.asarray(pos, dtype=np.float64) - truth_i))
            if err < best_err:
                best_err = err
                best_label = label
                best_pos = np.asarray(pos, dtype=np.float64)
        if best_pos is not None:
            est[i] = best_pos
            errors[i] = best_err
        picked_labels.append(best_label)
    return score_ppc2024(est, truth), errors, picked_labels


def main() -> None:
    parser = argparse.ArgumentParser(description="PPC candidate oracle diagnostic")
    parser.add_argument("--data-root", type=Path, default=_DEFAULT_DATA_ROOT)
    parser.add_argument("--runs", type=str, default="all")
    parser.add_argument("--hybrid-pos-dir", type=Path, default=RESULTS_DIR / "libgnss_rtk_pos_v5")
    parser.add_argument("--candidate-pos-dirs", type=str, default=_PHASE11_DIRS)
    parser.add_argument("--candidate-diag-dirs", type=str, default=_PHASE11_DIRS)
    parser.add_argument("--candidate-labels", type=str, default=_PHASE11_LABELS)
    parser.add_argument("--run-index-policy", type=str, default="phase11i")
    parser.add_argument(
        "--phase-runs-csv",
        type=Path,
        default=RESULTS_DIR / "ppc_ctrbpf_fgo_phase11i_selective_no_r30_family_full_p5k_runs.csv",
    )
    parser.add_argument("--results-prefix", type=str, default="ppc_candidate_oracle_phase11i")
    args = parser.parse_args()

    runs = _parse_runs(args.runs)
    labels = _parse_label_list(args.candidate_labels)
    pos_dirs = _parse_path_list(args.candidate_pos_dirs)
    diag_dirs = _parse_path_list(args.candidate_diag_dirs)
    if len(labels) != len(pos_dirs) or len(labels) != len(diag_dirs):
        raise SystemExit("candidate labels, pos dirs, and diag dirs must have the same length")

    phase_rows: dict[tuple[str, str], dict[str, str]] = {}
    if args.phase_runs_csv.is_file():
        with args.phase_runs_csv.open(newline="") as fh:
            for row in csv.DictReader(fh):
                phase_rows[(row["city"], row["run"])] = row

    run_rows: list[dict[str, object]] = []
    label_rows: dict[str, float] = {}
    totals = {
        "hybrid": 0.0,
        "phase": 0.0,
        "raw_oracle": 0.0,
        "gated_oracle": 0.0,
        "total": 0.0,
    }

    for city, run in runs:
        ref = _load_full_reference(args.data_root / city / run / "reference.csv")
        hybrid_pos, _ = _load_hybrid_pos_file(args.hybrid_pos_dir / f"{city}_{run}_full.pos")
        hybrid_score = _score_positions(ref, hybrid_pos)
        raw_sources = [("hybrid", hybrid_pos)]
        gated_sources = [("hybrid", hybrid_pos)]

        variant = _apply_rtkdiag_run_index_policy(
            CTRBPFConfig(enable_rtkdiag_pf_rescue=True),
            run=run,
            policy=str(args.run_index_policy),
            city=city,
        )
        blocked_labels = _phase11i_blocked_labels(city, run, str(args.run_index_policy))
        for label, pos_dir, diag_dir in zip(labels, pos_dirs, diag_dirs, strict=True):
            if label in blocked_labels:
                continue
            pos_path = pos_dir / f"{city}_{run}_full.pos"
            diag_path = diag_dir / f"{city}_{run}_full.csv"
            if not pos_path.is_file() or not diag_path.is_file():
                # Run-specific candidates (e.g. n3tight, t1tight2) are
                # only generated for one run; skip silently for others.
                continue
            pos, _ = _load_hybrid_pos_file(pos_path)
            diag = _load_rtk_diag_file(diag_path)
            raw_sources.append((label, pos))
            gated_pos: dict[float, np.ndarray] = {}
            for tow, candidate_pos in pos.items():
                diag_row = diag.get(round(float(tow), 1))
                if diag_row is None:
                    continue
                if _rtkdiag_candidate_gate(
                    diag_row,
                    ratio_min=float(variant.rtkdiag_candidate_ratio_min),
                    residual_rms_max=float(variant.rtkdiag_candidate_residual_rms_max),
                ):
                    gated_pos[tow] = candidate_pos
            gated_sources.append((label, gated_pos))

        raw_score, raw_errors, raw_picks = _oracle(ref, raw_sources)
        gated_score, gated_errors, gated_picks = _oracle(ref, gated_sources)
        phase_row = phase_rows.get((city, run), {})
        phase_pct = float(phase_row.get("honest_ppc_pct", "nan"))
        phase_pass_m = float(phase_row.get("honest_pass_m", "nan"))

        distances = ppc_segment_distances(np.asarray([t for _, t in ref], dtype=np.float64))
        for label, err, distance in zip(gated_picks, gated_errors, distances, strict=True):
            if np.isfinite(err) and err <= 0.5:
                label_rows[label] = label_rows.get(label, 0.0) + float(distance)

        total_m = float(raw_score.total_distance_m)
        totals["hybrid"] += float(hybrid_score.pass_distance_m)
        totals["phase"] += phase_pass_m if np.isfinite(phase_pass_m) else 0.0
        totals["raw_oracle"] += float(raw_score.pass_distance_m)
        totals["gated_oracle"] += float(gated_score.pass_distance_m)
        totals["total"] += total_m

        run_rows.append({
            "city": city,
            "run": run,
            "hybrid_ppc_pct": float(hybrid_score.score_pct),
            "phase_ppc_pct": phase_pct,
            "raw_oracle_ppc_pct": float(raw_score.score_pct),
            "gated_oracle_ppc_pct": float(gated_score.score_pct),
            "raw_gap_vs_phase_pp": float(raw_score.score_pct - phase_pct),
            "gated_gap_vs_phase_pp": float(gated_score.score_pct - phase_pct),
            "raw_oracle_pass_m": float(raw_score.pass_distance_m),
            "gated_oracle_pass_m": float(gated_score.pass_distance_m),
            "total_m": total_m,
            "blocked_labels": ",".join(sorted(blocked_labels)),
        })

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_path = RESULTS_DIR / f"{args.results_prefix}_runs.csv"
    summary_path = RESULTS_DIR / f"{args.results_prefix}_summary.csv"
    labels_path = RESULTS_DIR / f"{args.results_prefix}_labels.csv"

    with run_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(run_rows[0].keys()))
        writer.writeheader()
        writer.writerows(run_rows)

    total = max(float(totals["total"]), 1e-12)
    summary_rows = [
        {"metric": name, "ppc_pct": 100.0 * value / total, "pass_m": value, "total_m": total}
        for name, value in totals.items()
        if name != "total"
    ]
    with summary_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=("metric", "ppc_pct", "pass_m", "total_m"))
        writer.writeheader()
        writer.writerows(summary_rows)

    with labels_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=("label", "gated_oracle_pass_m", "total_m", "ppc_pct"))
        writer.writeheader()
        for label, pass_m in sorted(label_rows.items(), key=lambda item: item[1], reverse=True):
            writer.writerow({
                "label": label,
                "gated_oracle_pass_m": pass_m,
                "total_m": total,
                "ppc_pct": 100.0 * pass_m / total,
            })

    print(f"Saved: {run_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {labels_path}")
    for row in summary_rows:
        print(f"{row['metric']}: {row['ppc_pct']:.6f}%")


if __name__ == "__main__":
    main()
