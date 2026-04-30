#!/usr/bin/env python3
"""Offline multi-feature linear-rank selector sweep on Phase 11v.

Existing select_modes are single-feature sort keys. This sim sweeps a
small grid of linear combinations:
    score = w_residual * residual_rms - w_ratio * ratio - w_nrows * nrows
            + w_maxabs * abs_max
The candidate with the smallest score wins. The current modes are special
cases: residual = (1,0,0,0); ratio = (0,1,0,0); nrows = (0,0,1,0); maxabs
= (0,0,0,1); score ~= residual / max(ratio,1).

Phase 11v phase = 61.6039%, gated_oracle = 63.21%, gap +1.60pp.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from itertools import product
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
    _diag_float,
    _load_full_reference,
    _load_hybrid_pos_file,
    _load_rtk_diag_file,
    _rtkdiag_candidate_gate,
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
class WeightResult:
    city: str
    run: str
    w_resid: float
    w_ratio: float
    w_nrows: float
    w_maxabs: float
    ppc_pct: float
    pass_m: float
    total_m: float


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


def _multifeat_score(row, w_resid, w_ratio, w_nrows, w_maxabs):
    residual = _diag_float(row, "final_residual_rms")
    ratio = _diag_float(row, "final_ratio")
    nrows = _diag_float(row, "final_update_rows")
    maxabs = _diag_float(row, "final_residual_abs_max")
    return (
        w_resid * residual
        - w_ratio * ratio
        - w_nrows * nrows
        + w_maxabs * maxabs
    )


def _simulate(city, run, w_resid, w_ratio, w_nrows, w_maxabs,
              hybrid_pos, candidates, ref, ratio_min, rms_max) -> WeightResult:
    truth = np.asarray([t for _, t in ref], dtype=np.float64)
    est = np.zeros((len(ref), 3), dtype=np.float64)
    for i, (tow, _) in enumerate(ref):
        t_key = round(float(tow), 1)
        hp = hybrid_pos.get(t_key)
        if hp is not None and np.all(np.isfinite(hp)) and not np.all(hp == 0.0):
            est[i] = np.asarray(hp, dtype=np.float64)
        best_score = None
        best_pos = None
        for _, cand_pos, cand_diag in candidates:
            row = cand_diag.get(t_key)
            if not _rtkdiag_candidate_gate(row, ratio_min=ratio_min, residual_rms_max=rms_max):
                continue
            cand = cand_pos.get(t_key)
            if cand is None or not np.all(np.isfinite(cand)) or np.all(cand == 0.0):
                continue
            sc = _multifeat_score(row, w_resid, w_ratio, w_nrows, w_maxabs)
            if best_score is None or sc < best_score:
                best_score = sc
                best_pos = np.asarray(cand, dtype=np.float64)
        if best_pos is not None:
            est[i] = best_pos
    score = score_ppc2024(est, truth)
    return WeightResult(
        city=city, run=run,
        w_resid=w_resid, w_ratio=w_ratio, w_nrows=w_nrows, w_maxabs=w_maxabs,
        ppc_pct=float(score.score_pct),
        pass_m=float(score.pass_distance_m),
        total_m=float(score.total_distance_m),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-feature linear rank sweep")
    parser.add_argument("--data-root", type=Path, default=_DEFAULT_DATA_ROOT)
    parser.add_argument("--hybrid-pos-dir", type=Path, default=RESULTS_DIR / "libgnss_rtk_pos_v5")
    parser.add_argument("--policy", type=str, default="phase11n")
    parser.add_argument("--out-csv", type=Path, default=RESULTS_DIR / "ppc_multifeat_sweep_phase11v.csv")
    args = parser.parse_args()

    # Coarse grid: features have different scales so weights ~ 1/typical-scale.
    # Reduced to keep the sweep finishable (~5min total).
    w_resid_grid = [0.0, 1.0, 3.0]
    w_ratio_grid = [0.0, 0.1, 0.3]
    w_nrows_grid = [0.0, 0.01, 0.05]
    w_maxabs_grid = [0.0, 0.1, 0.3]

    rows: list[WeightResult] = []
    for city, run in _FULL_RUNS:
        ref = _load_full_reference(args.data_root / city / run / "reference.csv")
        hybrid_pos, _ = _load_hybrid_pos_file(args.hybrid_pos_dir / f"{city}_{run}_full.pos")
        variant = _apply_rtkdiag_run_index_policy(
            CTRBPFConfig(enable_rtkdiag_pf_rescue=True),
            run=run, policy=str(args.policy), city=city,
        )
        ratio_min = float(variant.rtkdiag_candidate_ratio_min)
        rms_max = float(variant.rtkdiag_candidate_residual_rms_max)
        kept = _filter_rtkdiag_candidates_by_policy(
            _load_candidates_for_run(city, run),
            city=city, run=run, policy=str(args.policy),
        )
        run_best: WeightResult | None = None
        run_iter = 0
        for w_r, w_a, w_n, w_m in product(w_resid_grid, w_ratio_grid, w_nrows_grid, w_maxabs_grid):
            if w_r == 0.0 and w_a == 0.0 and w_n == 0.0 and w_m == 0.0:
                continue
            res = _simulate(city, run, w_r, w_a, w_n, w_m,
                            hybrid_pos, kept, ref, ratio_min, rms_max)
            rows.append(res)
            if run_best is None or res.pass_m > run_best.pass_m:
                run_best = res
            run_iter += 1
        print(f"{city}/{run}: best multi-feat ppc={run_best.ppc_pct:.4f}% "
              f"w_r={run_best.w_resid} w_a={run_best.w_ratio} w_n={run_best.w_nrows} w_m={run_best.w_maxabs} "
              f"(iters={run_iter})")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["city", "run", "w_resid", "w_ratio", "w_nrows", "w_maxabs",
                    "ppc_pct", "pass_m", "total_m"])
        for r in rows:
            w.writerow([r.city, r.run, r.w_resid, r.w_ratio, r.w_nrows, r.w_maxabs,
                        f"{r.ppc_pct:.6f}", f"{r.pass_m:.4f}", f"{r.total_m:.4f}"])

    # Per-run-best aggregate.
    pass_sum = 0.0
    total_sum = 0.0
    for city, run in _FULL_RUNS:
        sub = [r for r in rows if r.city == city and r.run == run]
        best = max(sub, key=lambda r: r.pass_m)
        pass_sum += best.pass_m
        total_sum += best.total_m
    print(f"\nPer-run multi-feat best aggregate: ppc={100*pass_sum/total_sum:.4f}% "
          f"(pass {pass_sum:.1f}/{total_sum:.1f})")


if __name__ == "__main__":
    main()
