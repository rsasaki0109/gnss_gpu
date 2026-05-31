#!/usr/bin/env python3
"""Offline weighted-mean selector on Phase 11x candidate pool.

Rather than picking one gated candidate, take a weighted mean over the
top-K candidates (sorted by sort_mode), weighting by:
  - "uniform": equal weights
  - "ratio": weight by final_ratio
  - "rms_inv": weight by 1/(final_residual_rms+eps)
Optionally, robust trimming: drop candidates that are >drop_radius_m
from the median of the top-K (outlier rejection).
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
_DEFAULT_DATA_ROOT = Path("/media/sasaki/aiueo/ai_coding_ws/datasets/PPC-Dataset-data")


@dataclass
class Result:
    city: str
    run: str
    mode: str
    top_k: int
    drop_radius_m: float
    ppc_pct: float
    pass_m: float
    total_m: float


def _diag_float(diag: dict, key: str, default: float = 0.0) -> float:
    if diag is None:
        return default
    v = diag.get(key)
    if v is None:
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _weighted_select(gated_pos: list[np.ndarray],
                     gated_diag: list[dict],
                     sort_mode: str,
                     weight_mode: str,
                     top_k: int,
                     drop_radius_m: float) -> np.ndarray:
    if len(gated_pos) == 1:
        return gated_pos[0]
    # Sort indices by sort_key ascending (best first).
    idx_sorted = sorted(range(len(gated_pos)),
                        key=lambda i: _rtkdiag_candidate_sort_key(gated_diag[i], mode=sort_mode))
    keep = idx_sorted[:max(1, top_k)]
    pts = np.asarray([gated_pos[i] for i in keep], dtype=np.float64)
    diags = [gated_diag[i] for i in keep]
    # Optional outlier rejection: drop those >drop_radius_m from median.
    if drop_radius_m > 0 and len(pts) >= 3:
        med = np.median(pts, axis=0)
        d = np.linalg.norm(pts - med, axis=1)
        mask = d <= drop_radius_m
        if mask.sum() >= 1:
            pts = pts[mask]
            diags = [diags[i] for i in range(len(diags)) if mask[i]]
    # Weights.
    if weight_mode == "uniform":
        w = np.ones(len(pts), dtype=np.float64)
    elif weight_mode == "ratio":
        w = np.asarray([max(0.0, _diag_float(d, "final_ratio")) for d in diags], dtype=np.float64)
    elif weight_mode == "rms_inv":
        w = np.asarray([1.0 / (max(0.01, _diag_float(d, "final_residual_rms"))) for d in diags], dtype=np.float64)
    else:
        raise ValueError(f"unknown weight_mode {weight_mode}")
    if w.sum() <= 0:
        w = np.ones(len(pts), dtype=np.float64)
    w = w / w.sum()
    return (pts * w[:, None]).sum(axis=0)


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


def _simulate(city, run, hybrid_pos_dict, candidates, ref, ratio_min, rms_max,
              sort_mode, weight_mode, top_k, drop_radius_m) -> Result:
    truth = np.asarray([t for _, t in ref], dtype=np.float64)
    est = np.zeros((len(ref), 3), dtype=np.float64)
    for i, (tow, _) in enumerate(ref):
        t_key = round(float(tow), 1)
        hp = hybrid_pos_dict.get(t_key)
        if hp is not None and np.all(np.isfinite(hp)) and not np.all(hp == 0.0):
            est[i] = np.asarray(hp, dtype=np.float64)
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
        if gated_pos:
            est[i] = _weighted_select(gated_pos, gated_diag, sort_mode,
                                      weight_mode, top_k, drop_radius_m)
    score = score_ppc2024(est, truth)
    return Result(
        city=city, run=run, mode=weight_mode, top_k=top_k, drop_radius_m=drop_radius_m,
        ppc_pct=float(score.score_pct),
        pass_m=float(score.pass_distance_m),
        total_m=float(score.total_distance_m),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=_DEFAULT_DATA_ROOT)
    parser.add_argument("--hybrid-pos-dir", type=Path, default=RESULTS_DIR / "libgnss_rtk_pos_v5")
    parser.add_argument("--policy", type=str, default="phase11aa")
    parser.add_argument("--out-csv", type=Path, default=RESULTS_DIR / "ppc_weighted_mean_phase11aa.csv")
    args = parser.parse_args()

    grid = [
        ("uniform", 1, 0.0),   # baseline = standard pick
        ("uniform", 3, 0.0),
        ("uniform", 5, 0.0),
        ("uniform", 3, 2.0),
        ("uniform", 5, 2.0),
        ("uniform", 5, 5.0),
        ("ratio",   3, 2.0),
        ("ratio",   5, 2.0),
        ("rms_inv", 3, 2.0),
        ("rms_inv", 5, 2.0),
    ]

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
        for weight_mode, top_k, drop_radius_m in grid:
            r = _simulate(city, run, hybrid_pos, kept, ref, ratio_min, rms_max,
                          sort_mode, weight_mode, top_k, drop_radius_m)
            rows.append(r)
            print(f"  {weight_mode}/k={top_k}/drop={drop_radius_m}: ppc={r.ppc_pct:.4f}% (pass {r.pass_m:.1f})")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["city", "run", "mode", "top_k", "drop_radius_m", "ppc_pct", "pass_m", "total_m"])
        for r in rows:
            w.writerow([r.city, r.run, r.mode, r.top_k, r.drop_radius_m,
                        f"{r.ppc_pct:.6f}", f"{r.pass_m:.4f}", f"{r.total_m:.4f}"])

    # Aggregate by (mode, top_k, drop_radius_m) uniform across runs.
    print("\nAggregate per setting:")
    by_key: dict[tuple, list[Result]] = {}
    for r in rows:
        by_key.setdefault((r.mode, r.top_k, r.drop_radius_m), []).append(r)
    for (mode, top_k, drop), items in by_key.items():
        ps = sum(r.pass_m for r in items)
        ts = sum(r.total_m for r in items)
        print(f"  {mode}/k={top_k}/drop={drop}: ppc={100*ps/ts:.4f}% (pass {ps:.1f}/{ts:.1f})")


if __name__ == "__main__":
    main()
