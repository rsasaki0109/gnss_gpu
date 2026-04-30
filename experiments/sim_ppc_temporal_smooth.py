#!/usr/bin/env python3
"""Temporal smoothing post-process on Phase 11x candidate selections.

After standard sort_key selection per epoch, apply a sliding-window
filter: replace position with median (or trimmed mean) of t±W frames
where the difference is bounded. Reasoning: when sort_key picks a wrong
high-confidence candidate at frame t but neighbors are correct, smoothing
nudges t back toward the trajectory.

Modes:
- median3 / median5 / median7: median over [t-W..t+W]
- trimmed_mean5: trimmed mean
- step_clamp: if |t - t-1| > clamp_m, replace t with extrapolation from t-1
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
class Result:
    city: str
    run: str
    mode: str
    ppc_pct: float
    pass_m: float
    total_m: float


def _load_candidates_for_run(city, run):
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


def _select_one(gated_pos, gated_diag, sort_mode, hybrid_pos):
    if not gated_pos:
        return None
    if sort_mode == "hybrid_anchor" and hybrid_pos is not None:
        d = np.linalg.norm(np.asarray(gated_pos) - hybrid_pos, axis=1)
        return gated_pos[int(np.argmin(d))]
    return min(zip(gated_pos, gated_diag),
               key=lambda pd: _rtkdiag_candidate_sort_key(pd[1], mode=sort_mode))[0]


def _base_pick(city, run, hybrid_pos_dict, candidates, ref, ratio_min, rms_max, sort_mode):
    """Return per-epoch arrays: positions (or None), gate_passed (bool)."""
    n = len(ref)
    picks: list[np.ndarray | None] = [None] * n
    hybs: list[np.ndarray | None] = [None] * n
    for i, (tow, _) in enumerate(ref):
        t_key = round(float(tow), 1)
        hp = hybrid_pos_dict.get(t_key)
        hp_arr = None
        if hp is not None and np.all(np.isfinite(hp)) and not np.all(hp == 0.0):
            hp_arr = np.asarray(hp, dtype=np.float64)
            hybs[i] = hp_arr
        gated_pos: list[np.ndarray] = []
        gated_diag: list[dict] = []
        for label, cand_pos, cand_diag in candidates:
            row = cand_diag.get(t_key)
            if not _rtkdiag_candidate_gate(row, ratio_min=ratio_min, residual_rms_max=rms_max):
                continue
            cand = cand_pos.get(t_key)
            if cand is None or not np.all(np.isfinite(cand)) or np.all(cand == 0.0):
                continue
            gated_pos.append(np.asarray(cand, dtype=np.float64))
            gated_diag.append(row)
        sel = _select_one(gated_pos, gated_diag, sort_mode, hp_arr)
        if sel is not None:
            picks[i] = sel
    return picks, hybs


def _apply_smoothing(picks, hybs, mode):
    n = len(picks)
    out = np.zeros((n, 3), dtype=np.float64)
    for i in range(n):
        if picks[i] is not None:
            out[i] = picks[i]
        elif hybs[i] is not None:
            out[i] = hybs[i]
    if mode == "none":
        return out
    if mode in ("median3", "median5", "median7"):
        w = {"median3": 1, "median5": 2, "median7": 3}[mode]
        smoothed = np.copy(out)
        for i in range(n):
            if picks[i] is None:
                continue
            window = []
            for j in range(max(0, i - w), min(n, i + w + 1)):
                if picks[j] is not None:
                    window.append(picks[j])
            if len(window) >= 3:
                smoothed[i] = np.median(np.asarray(window), axis=0)
        return smoothed
    if mode == "step_clamp_5m":
        smoothed = np.copy(out)
        for i in range(1, n):
            if picks[i] is None or picks[i - 1] is None:
                continue
            d = float(np.linalg.norm(picks[i] - smoothed[i - 1]))
            if d > 5.0:
                smoothed[i] = smoothed[i - 1]
        return smoothed
    if mode == "step_clamp_2m":
        smoothed = np.copy(out)
        for i in range(1, n):
            if picks[i] is None or picks[i - 1] is None:
                continue
            d = float(np.linalg.norm(picks[i] - smoothed[i - 1]))
            if d > 2.0:
                smoothed[i] = smoothed[i - 1]
        return smoothed
    raise ValueError(f"unknown mode {mode}")


def _simulate(city, run, hybrid_pos_dict, candidates, ref, ratio_min, rms_max,
              sort_mode, smooth_mode) -> Result:
    truth = np.asarray([t for _, t in ref], dtype=np.float64)
    picks, hybs = _base_pick(city, run, hybrid_pos_dict, candidates, ref, ratio_min, rms_max, sort_mode)
    est = _apply_smoothing(picks, hybs, smooth_mode)
    score = score_ppc2024(est, truth)
    return Result(
        city=city, run=run, mode=smooth_mode,
        ppc_pct=float(score.score_pct),
        pass_m=float(score.pass_distance_m),
        total_m=float(score.total_distance_m),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=_DEFAULT_DATA_ROOT)
    parser.add_argument("--hybrid-pos-dir", type=Path, default=RESULTS_DIR / "libgnss_rtk_pos_v5")
    parser.add_argument("--policy", type=str, default="phase11aa")
    parser.add_argument("--out-csv", type=Path, default=RESULTS_DIR / "ppc_temporal_smooth_phase11aa.csv")
    args = parser.parse_args()

    modes = ["none", "median3", "median5", "median7", "step_clamp_5m", "step_clamp_2m"]
    rows: list[Result] = []
    for city, run in _FULL_RUNS:
        ref = _load_full_reference(args.data_root / city / run / "reference.csv")
        hybrid_pos, _ = _load_hybrid_pos_file(args.hybrid_pos_dir / f"{city}_{run}_full.pos")
        variant = _apply_rtkdiag_run_index_policy(
            CTRBPFConfig(enable_rtkdiag_pf_rescue=True),
            run=run, policy=str(args.policy), city=city,
        )
        ratio_min = float(variant.rtkdiag_candidate_ratio_min)
        rms_max = float(variant.rtkdiag_candidate_residual_rms_max)
        sort_mode = str(variant.rtkdiag_candidate_select_mode)
        kept = _filter_rtkdiag_candidates_by_policy(
            _load_candidates_for_run(city, run),
            city=city, run=run, policy=str(args.policy),
        )
        print(f"\n{city}/{run}: ratio>={ratio_min} rms<={rms_max} sort={sort_mode}")
        for mode in modes:
            r = _simulate(city, run, hybrid_pos, kept, ref, ratio_min, rms_max,
                          sort_mode, mode)
            rows.append(r)
            print(f"  {mode:<14s}: ppc={r.ppc_pct:.4f}% (pass {r.pass_m:.1f})")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["city", "run", "mode", "ppc_pct", "pass_m", "total_m"])
        for r in rows:
            w.writerow([r.city, r.run, r.mode, f"{r.ppc_pct:.6f}", f"{r.pass_m:.4f}", f"{r.total_m:.4f}"])

    print("\nAggregate per mode:")
    by_mode: dict[str, list[Result]] = {}
    for r in rows:
        by_mode.setdefault(r.mode, []).append(r)
    for mode, items in by_mode.items():
        ps = sum(r.pass_m for r in items)
        ts = sum(r.total_m for r in items)
        print(f"  {mode:<14s}: ppc={100*ps/ts:.4f}% (pass {ps:.1f}/{ts:.1f})")

    print("\nPer-run-best:")
    pass_sum = 0.0
    total_sum = 0.0
    for city, run in _FULL_RUNS:
        sub = [r for r in rows if r.city == city and r.run == run]
        best = max(sub, key=lambda r: r.pass_m)
        pass_sum += best.pass_m
        total_sum += best.total_m
        print(f"  {city}/{run}: best={best.mode} ppc={best.ppc_pct:.4f}%")
    print(f"  per-run-best aggregate: ppc={100*pass_sum/total_sum:.4f}%")


if __name__ == "__main__":
    main()
