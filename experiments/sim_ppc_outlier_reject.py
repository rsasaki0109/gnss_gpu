#!/usr/bin/env python3
"""Outlier-rejection selector: remove gated candidates that are >D meters
from the median of all gated candidates, then pick by standard sort_key.

Goal: when sort_key picks a high-confidence-but-wrong candidate (delta_ratio
analysis showed truth-best had lower ratio in many wrong-picks), the wrong
candidate is often a position outlier vs the cluster of agreement among
peer fixes. Median-distance filter removes those.
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
    drop_radius_m: float
    require_min: int
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


def _select_with_outlier_reject(gated_pos: list[np.ndarray],
                                gated_diag: list[dict],
                                sort_mode: str,
                                drop_radius_m: float,
                                require_min: int) -> int:
    """Return index of chosen candidate after outlier rejection."""
    if len(gated_pos) <= require_min:
        return min(range(len(gated_pos)),
                   key=lambda i: _rtkdiag_candidate_sort_key(gated_diag[i], mode=sort_mode))
    pts = np.asarray(gated_pos, dtype=np.float64)
    med = np.median(pts, axis=0)
    d = np.linalg.norm(pts - med, axis=1)
    mask = d <= drop_radius_m
    if mask.sum() < 1:
        return min(range(len(gated_pos)),
                   key=lambda i: _rtkdiag_candidate_sort_key(gated_diag[i], mode=sort_mode))
    kept_idx = [i for i, m in enumerate(mask) if m]
    return min(kept_idx,
               key=lambda i: _rtkdiag_candidate_sort_key(gated_diag[i], mode=sort_mode))


def _simulate(city, run, hybrid_pos_dict, candidates, ref, ratio_min, rms_max,
              sort_mode, drop_radius_m, require_min) -> Result:
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
            idx = _select_with_outlier_reject(gated_pos, gated_diag,
                                              sort_mode, drop_radius_m, require_min)
            est[i] = gated_pos[idx]
    score = score_ppc2024(est, truth)
    return Result(
        city=city, run=run, drop_radius_m=drop_radius_m, require_min=require_min,
        ppc_pct=float(score.score_pct),
        pass_m=float(score.pass_distance_m),
        total_m=float(score.total_distance_m),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=_DEFAULT_DATA_ROOT)
    parser.add_argument("--hybrid-pos-dir", type=Path, default=RESULTS_DIR / "libgnss_rtk_pos_v5")
    parser.add_argument("--policy", type=str, default="phase11aa")
    parser.add_argument("--out-csv", type=Path, default=RESULTS_DIR / "ppc_outlier_reject_phase11aa.csv")
    args = parser.parse_args()

    grid = [
        (1e9, 0),  # baseline (no rejection)
        (10.0, 3),
        (5.0, 3),
        (3.0, 3),
        (2.0, 3),
        (1.0, 3),
        (0.5, 3),
        (10.0, 5),
        (5.0, 5),
        (2.0, 5),
        (1.0, 5),
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
        for drop_r, req_min in grid:
            r = _simulate(city, run, hybrid_pos, kept, ref, ratio_min, rms_max,
                          sort_mode, drop_r, req_min)
            rows.append(r)
            tag = "baseline" if drop_r > 1e8 else f"drop={drop_r}/req={req_min}"
            print(f"  {tag:<22s}: ppc={r.ppc_pct:.4f}% (pass {r.pass_m:.1f})")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["city", "run", "drop_radius_m", "require_min", "ppc_pct", "pass_m", "total_m"])
        for r in rows:
            w.writerow([r.city, r.run, r.drop_radius_m, r.require_min,
                        f"{r.ppc_pct:.6f}", f"{r.pass_m:.4f}", f"{r.total_m:.4f}"])

    # Aggregate by setting.
    print("\nAggregate per setting:")
    by_key: dict[tuple, list[Result]] = {}
    for r in rows:
        by_key.setdefault((r.drop_radius_m, r.require_min), []).append(r)
    for (drop_r, req_min), items in by_key.items():
        ps = sum(r.pass_m for r in items)
        ts = sum(r.total_m for r in items)
        tag = "baseline" if drop_r > 1e8 else f"drop={drop_r}/req={req_min}"
        print(f"  {tag:<22s}: ppc={100*ps/ts:.4f}% (pass {ps:.1f}/{ts:.1f})")

    # Per-run-best.
    print("\nPer-run-best:")
    pass_sum = 0.0
    total_sum = 0.0
    for city, run in _FULL_RUNS:
        sub = [r for r in rows if r.city == city and r.run == run]
        best = max(sub, key=lambda r: r.pass_m)
        pass_sum += best.pass_m
        total_sum += best.total_m
        tag = "baseline" if best.drop_radius_m > 1e8 else f"drop={best.drop_radius_m}/req={best.require_min}"
        print(f"  {city}/{run}: best={tag} ppc={best.ppc_pct:.4f}%")
    print(f"  per-run-best aggregate: ppc={100*pass_sum/total_sum:.4f}%")


if __name__ == "__main__":
    main()
