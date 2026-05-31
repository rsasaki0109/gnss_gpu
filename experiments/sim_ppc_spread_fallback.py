#!/usr/bin/env python3
"""Spread-gated hybrid fallback: when gated candidates spread > X meters,
fall back to hybrid; else pick by per-policy sort_key.

Tests if filtering out epochs where candidate "consensus" is poor (large
spread = possibly bad picks) and reverting to hybrid improves PPC.
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
    _load_candidates_for_run,
)


_DEFAULT_DATA_ROOT = Path("/media/sasaki/aiueo/ai_coding_ws/datasets/PPC-Dataset-data")
RESULTS_DIR = _PROJECT_ROOT / "experiments/results"
_FULL_RUNS = (
    ("tokyo", "run1"), ("tokyo", "run2"), ("tokyo", "run3"),
    ("nagoya", "run1"), ("nagoya", "run2"), ("nagoya", "run3"),
)


@dataclass
class Result:
    city: str
    run: str
    spread_max_m: float
    metric: str
    ppc_pct: float
    pass_m: float
    total_m: float
    n_emit_cand: int
    n_emit_hyb: int


def _spread(positions: np.ndarray, metric: str) -> float:
    """Compute spread metric. metric in {'std', 'maxdist', 'mad'}."""
    if positions.shape[0] < 2:
        return 0.0
    if metric == "std":
        # max axis std
        return float(np.std(positions, axis=0).max())
    if metric == "maxdist":
        # max pairwise distance
        n = positions.shape[0]
        d_max = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                d = float(np.linalg.norm(positions[i] - positions[j]))
                if d > d_max:
                    d_max = d
        return d_max
    if metric == "mad":
        # median absolute deviation from centroid
        c = np.median(positions, axis=0)
        d = np.linalg.norm(positions - c, axis=1)
        return float(np.median(d))
    raise ValueError(metric)


def _simulate(city, run, hybrid_pos, candidates, ref, ratio_min, rms_max, sort_mode,
              spread_max_m, metric):
    truth = np.asarray([t for _, t in ref], dtype=np.float64)
    n_epochs = len(ref)
    est = np.zeros((n_epochs, 3), dtype=np.float64)
    n_emit_cand = 0
    n_emit_hyb = 0
    for i, (tow, _) in enumerate(ref):
        t_key = round(float(tow), 1)
        hp = hybrid_pos.get(t_key)
        if hp is not None and np.all(np.isfinite(hp)) and not np.all(hp == 0.0):
            est[i] = np.asarray(hp, dtype=np.float64)
            emitted_hyb = True
        else:
            emitted_hyb = False
        # Gate candidates
        gated = []  # (label, position, sort_key)
        for label, cand_pos, cand_diag in candidates:
            row = cand_diag.get(t_key)
            if not _rtkdiag_candidate_gate(row, ratio_min=ratio_min, residual_rms_max=rms_max):
                continue
            cand = cand_pos.get(t_key)
            if cand is None or not np.all(np.isfinite(cand)) or np.all(cand == 0.0):
                continue
            sk = _rtkdiag_candidate_sort_key(row, mode=sort_mode)
            gated.append((label, np.asarray(cand, dtype=np.float64), sk))
        if not gated:
            if emitted_hyb:
                n_emit_hyb += 1
            continue
        # Compute spread of gated candidate positions
        positions = np.array([g[1] for g in gated])
        sp = _spread(positions, metric)
        if sp > spread_max_m:
            # Fall back to hybrid
            if emitted_hyb:
                n_emit_hyb += 1
            continue
        # Pick by sort_key
        gated.sort(key=lambda g: g[2])
        est[i] = gated[0][1]
        n_emit_cand += 1
    s = score_ppc2024(est, truth)
    return Result(city=city, run=run, spread_max_m=spread_max_m, metric=metric,
                  ppc_pct=float(s.score_pct),
                  pass_m=float(s.pass_distance_m), total_m=float(s.total_distance_m),
                  n_emit_cand=n_emit_cand, n_emit_hyb=n_emit_hyb)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=_DEFAULT_DATA_ROOT)
    p.add_argument("--hybrid-pos-dir", type=Path, default=RESULTS_DIR / "libgnss_rtk_pos_v5")
    p.add_argument("--policy", default="phase11dd")
    p.add_argument("--metric", default="maxdist", choices=("std", "maxdist", "mad"))
    p.add_argument("--out-csv", type=Path, default=RESULTS_DIR / "ppc_spread_fallback_phase11dd.csv")
    args = p.parse_args()

    spreads = [1e9, 100.0, 50.0, 20.0, 10.0, 5.0, 2.0, 1.0, 0.5, 0.2]

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
        print(f"\n{city}/{run}: ratio>={ratio_min} rms<={rms_max} sort={sort_mode}", flush=True)
        for sm in spreads:
            r = _simulate(city, run, hybrid_pos, kept, ref, ratio_min, rms_max,
                          sort_mode, sm, args.metric)
            rows.append(r)
            tag = "baseline" if sm > 1e8 else f"spread<={sm}m"
            print(f"  {tag:<22s}: ppc={r.ppc_pct:.4f}% (cand={r.n_emit_cand} hyb={r.n_emit_hyb})", flush=True)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["city", "run", "spread_max_m", "metric", "ppc_pct", "pass_m", "total_m",
                    "n_emit_cand", "n_emit_hyb"])
        for r in rows:
            w.writerow([r.city, r.run, r.spread_max_m, r.metric,
                        f"{r.ppc_pct:.6f}", f"{r.pass_m:.4f}", f"{r.total_m:.4f}",
                        r.n_emit_cand, r.n_emit_hyb])

    # Aggregate per setting
    print("\nAggregate per setting:")
    by_key: dict[float, list[Result]] = {}
    for r in rows:
        by_key.setdefault(r.spread_max_m, []).append(r)
    best_setting = None
    best_pass = -1.0
    for sm, items in by_key.items():
        ps = sum(r.pass_m for r in items)
        ts = sum(r.total_m for r in items)
        tag = "baseline" if sm > 1e8 else f"spread<={sm}m"
        print(f"  {tag:<22s}: ppc={100*ps/ts:.4f}% (pass {ps:.1f}/{ts:.1f})")
        if ps > best_pass:
            best_pass = ps
            best_setting = sm
    tag = "baseline" if best_setting > 1e8 else f"spread<={best_setting}m"
    print(f"\nBEST aggregate setting: {tag}")

    # Per-run-best
    print("\nPer-run-best:")
    pass_sum = 0.0
    total_sum = 0.0
    for city, run in _FULL_RUNS:
        sub = [r for r in rows if r.city == city and r.run == run]
        best = max(sub, key=lambda r: r.pass_m)
        pass_sum += best.pass_m
        total_sum += best.total_m
        tag = "baseline" if best.spread_max_m > 1e8 else f"spread<={best.spread_max_m}m"
        print(f"  {city}/{run}: best={tag} ppc={best.ppc_pct:.4f}%")
    print(f"  per-run-best aggregate: ppc={100*pass_sum/total_sum:.4f}%")


if __name__ == "__main__":
    main()
