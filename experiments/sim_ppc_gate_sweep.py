#!/usr/bin/env python3
"""Offline gate-relaxation sweep on Phase 11v candidate pool.

Phase 11v phase = 61.6039%, gated_oracle = 63.21%, raw_oracle = 66.80%.
gate_gap = raw - gated = +3.60pp comes from epochs where the truth-best
candidate fails (ratio>=ratio_min, residual_rms<=rms_max). Loosening the
gate (per-run rms_max / ratio_min) might let the selector see those.

For each run, sweep (ratio_min, rms_max) keeping the policy's select_mode,
fall back to hybrid for non-gated, and report PPC.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
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
    _filter_rtkdiag_candidates_by_policy,
    _load_full_reference,
    _load_hybrid_pos_file,
    _load_rtk_diag_file,
    _rtkdiag_candidate_gate,
    _rtkdiag_candidate_sort_key,
)
from gnss_gpu.ppc_score import score_ppc2024  # noqa: E402

from sim_ppc_selector_sweep import (  # noqa: E402
    _CANDIDATES_PHASE11V,
    _DIAG_ROOT,
    _eligible_for_run,
    _FULL_RUNS,
)

RESULTS_DIR = _SCRIPT_DIR / "results"
_DEFAULT_DATA_ROOT = Path("datasets/PPC-Dataset-data")


@dataclass
class GateResult:
    city: str
    run: str
    ratio_min: float
    rms_max: float
    mode: str
    ppc_pct: float
    pass_m: float
    total_m: float
    n_gated: int


def _load_candidates_for_run(city: str, run: str):
    out = []
    for label, dir_name, restrict in _CANDIDATES_PHASE11V:
        if not _eligible_for_run(city, run, restrict):
            continue
        pos_path = _PROJECT_ROOT / _DIAG_ROOT / dir_name / f"{city}_{run}_full.pos"
        diag_path = _PROJECT_ROOT / _DIAG_ROOT / dir_name / f"{city}_{run}_full.csv"
        if not pos_path.is_file() or not diag_path.is_file():
            continue
        pos, _ = _load_hybrid_pos_file(pos_path)
        diag = _load_rtk_diag_file(diag_path)
        out.append((label, pos, diag))
    return out


def _simulate(city, run, mode, hybrid_pos, candidates, ref, ratio_min, rms_max) -> GateResult:
    truth = np.asarray([t for _, t in ref], dtype=np.float64)
    est = np.zeros((len(ref), 3), dtype=np.float64)
    n_gated = 0
    for i, (tow, _) in enumerate(ref):
        t_key = round(float(tow), 1)
        hp = hybrid_pos.get(t_key)
        if hp is not None and np.all(np.isfinite(hp)) and not np.all(hp == 0.0):
            est[i] = np.asarray(hp, dtype=np.float64)
        best_key = None
        best_pos = None
        any_gated = False
        for label, cand_pos, cand_diag in candidates:
            row = cand_diag.get(t_key)
            if not _rtkdiag_candidate_gate(row, ratio_min=ratio_min, residual_rms_max=rms_max):
                continue
            any_gated = True
            cand = cand_pos.get(t_key)
            if cand is None or not np.all(np.isfinite(cand)) or np.all(cand == 0.0):
                continue
            sort_key = _rtkdiag_candidate_sort_key(row, mode=mode)
            if best_key is None or sort_key < best_key:
                best_key = sort_key
                best_pos = np.asarray(cand, dtype=np.float64)
        if any_gated:
            n_gated += 1
        if best_pos is not None:
            est[i] = best_pos
    score = score_ppc2024(est, truth)
    return GateResult(
        city=city, run=run, ratio_min=ratio_min, rms_max=rms_max, mode=mode,
        ppc_pct=float(score.score_pct),
        pass_m=float(score.pass_distance_m),
        total_m=float(score.total_distance_m),
        n_gated=n_gated,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline gate sweep for Phase 11v")
    parser.add_argument("--data-root", type=Path, default=_DEFAULT_DATA_ROOT)
    parser.add_argument("--hybrid-pos-dir", type=Path, default=RESULTS_DIR / "libgnss_rtk_pos_v5")
    parser.add_argument("--policy", type=str, default="phase11n")
    parser.add_argument("--out-csv", type=Path, default=RESULTS_DIR / "ppc_gate_sweep_phase11v.csv")
    args = parser.parse_args()

    # Extended ratio/rms grid for searching beyond Phase 11y.
    ratio_grid = [1.0, 1.2, 1.5, 1.7, 1.9, 2.1, 2.5]
    rms_grid_run1 = [0.6, 0.8, 1.0, 1.4, 2.0, 5.0, 10.0]
    rms_grid_other = [10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 100.0]

    rows: list[GateResult] = []
    for city, run in _FULL_RUNS:
        ref = _load_full_reference(args.data_root / city / run / "reference.csv")
        hybrid_pos, _ = _load_hybrid_pos_file(args.hybrid_pos_dir / f"{city}_{run}_full.pos")
        variant = _apply_rtkdiag_run_index_policy(
            CTRBPFConfig(enable_rtkdiag_pf_rescue=True),
            run=run, policy=str(args.policy),
        )
        cur_mode = str(variant.rtkdiag_candidate_select_mode)
        cur_ratio = float(variant.rtkdiag_candidate_ratio_min)
        cur_rms = float(variant.rtkdiag_candidate_residual_rms_max)
        cands = _load_candidates_for_run(city, run)
        kept = _filter_rtkdiag_candidates_by_policy(
            cands, city=city, run=run, policy=str(args.policy),
        )
        rms_grid = rms_grid_run1 if run == "run1" else rms_grid_other
        print(f"\n{city}/{run}: current policy ratio_min={cur_ratio} rms_max={cur_rms} mode={cur_mode}")
        run_rows: list[GateResult] = []
        for ratio_min in ratio_grid:
            for rms_max in rms_grid:
                res = _simulate(city, run, cur_mode, hybrid_pos, kept, ref, ratio_min, rms_max)
                run_rows.append(res)
        rows.extend(run_rows)
        # Print top-5 + current.
        run_rows.sort(key=lambda r: r.pass_m, reverse=True)
        cur_match = next((r for r in run_rows if r.ratio_min == cur_ratio and r.rms_max == cur_rms), None)
        if cur_match is None:
            cur_match = _simulate(city, run, cur_mode, hybrid_pos, kept, ref, cur_ratio, cur_rms)
        print(f"  current ratio={cur_ratio} rms={cur_rms} mode={cur_mode}: ppc={cur_match.ppc_pct:.4f}% (pass {cur_match.pass_m:.1f}, gated {cur_match.n_gated})")
        for r in run_rows[:5]:
            delta = r.pass_m - cur_match.pass_m
            print(f"  ratio={r.ratio_min:<4} rms={r.rms_max:<5}: ppc={r.ppc_pct:.4f}% pass={r.pass_m:.1f} (delta {delta:+.1f}m) gated={r.n_gated}")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["city", "run", "ratio_min", "rms_max", "mode", "ppc_pct", "pass_m", "total_m", "n_gated"])
        for r in rows:
            w.writerow([r.city, r.run, r.ratio_min, r.rms_max, r.mode,
                        f"{r.ppc_pct:.6f}", f"{r.pass_m:.4f}", f"{r.total_m:.4f}", r.n_gated])

    # Aggregate per-run-best (oracle on gate per run).
    print("\nPer-run best (oracle on gate per run):")
    pass_sum = 0.0
    total_sum = 0.0
    best_summary: list[str] = []
    for city, run in _FULL_RUNS:
        sub = [r for r in rows if r.city == city and r.run == run]
        best = max(sub, key=lambda r: r.pass_m)
        pass_sum += best.pass_m
        total_sum += best.total_m
        best_summary.append(f"{city}/{run}: ratio={best.ratio_min} rms={best.rms_max} ppc={best.ppc_pct:.4f}%")
    for s in best_summary:
        print(f"  {s}")
    print(f"  aggregate: ppc={100*pass_sum/total_sum:.4f}% (pass {pass_sum:.1f}/{total_sum:.1f})")


if __name__ == "__main__":
    main()
