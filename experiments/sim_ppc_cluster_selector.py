#!/usr/bin/env python3
"""Cluster-based selector simulation: pick the candidate with most neighbors.

Hypothesis: oracle gap (~4pp on t/r3) is from tail epochs (>5m off oracle)
where the score selector picks an outlier candidate. If multiple candidates
cluster spatially, the cluster center is more likely to be near truth than
an outlier (even if outlier has best score).

Modes tested:
  - inlier_count_X: pick candidate with most candidates within Xm
  - cluster_median_X: cluster candidates within Xm, pick median of largest cluster
  - score_inlier_blend: combine score rank + inlier count
"""

from __future__ import annotations

import argparse
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
    _apply_rtkdiag_run_index_policy,
    _diag_float,
    _filter_rtkdiag_candidates_by_policy,
    _load_full_reference,
    _load_hybrid_pos_file,
    _load_rtk_diag_file,
    _rtkdiag_candidate_gate,
    _rtkdiag_candidate_sort_key,
    CTRBPFConfig,
)
from gnss_gpu.ppc_score import score_ppc2024  # type: ignore  # noqa: E402


def _load_pos_file(path: Path) -> dict[float, np.ndarray]:
    out: dict[float, np.ndarray] = {}
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("%") or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 6:
                continue
            try:
                tow = round(float(parts[1]), 1)
                x = float(parts[2])
                y = float(parts[3])
                z = float(parts[4])
            except ValueError:
                continue
            out[tow] = np.array([x, y, z], dtype=np.float64)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path,
                   default=Path("/media/sasaki/aiueo/ai_coding_ws/datasets/PPC-Dataset-data"))
    p.add_argument("--hybrid-pos-dir", type=Path,
                   default=Path("experiments/results/libgnss_rtk_pos_v5"))
    p.add_argument("--candidate-dirs", required=True)
    p.add_argument("--candidate-labels", required=True)
    p.add_argument("--policy", default="phase11bw")
    p.add_argument("--city", required=True)
    p.add_argument("--run", required=True)
    args = p.parse_args()

    cand_dirs = [Path(s.strip()) for s in args.candidate_dirs.split(",") if s.strip()]
    cand_labels = [s.strip() for s in args.candidate_labels.split(",") if s.strip()]

    city, run = args.city, args.run
    pos_filename = f"{city}_{run}_full.pos"

    ref = _load_full_reference(args.data_root / city / run / "reference.csv")
    truth = np.asarray([t for _, t in ref], dtype=np.float64)
    n_epochs = len(ref)

    hybrid_pos, _ = _load_hybrid_pos_file(args.hybrid_pos_dir / pos_filename)

    candidates = []
    for d, lbl in zip(cand_dirs, cand_labels):
        pos_path = d / pos_filename
        diag_path = d / f"{city}_{run}_full.csv"
        if not pos_path.is_file() or not diag_path.is_file():
            continue
        candidates.append((lbl, _load_pos_file(pos_path), _load_rtk_diag_file(diag_path)))
    filtered = _filter_rtkdiag_candidates_by_policy(
        candidates, city=city, run=run, policy=args.policy,
    )

    base_cfg = CTRBPFConfig(
        enable_rtkdiag_pf_rescue=True,
        rtkdiag_candidate_select_mode="score",
        rtkdiag_candidate_ratio_min=1.0,
        rtkdiag_candidate_residual_rms_max=50.0,
        rtkdiag_candidate_emit_mode="candidate",
    )
    cfg = _apply_rtkdiag_run_index_policy(base_cfg, run=run, policy=args.policy, city=city)
    ratio_min = float(cfg.rtkdiag_candidate_ratio_min)
    rms_max = float(cfg.rtkdiag_candidate_residual_rms_max)
    select_mode = str(cfg.rtkdiag_candidate_select_mode)

    epoch_data = []
    for tow, _ in ref:
        t_key = round(float(tow), 1)
        gated = []
        for label, cand_pos, cand_diag in filtered:
            row = cand_diag.get(t_key)
            if not _rtkdiag_candidate_gate(row, ratio_min=ratio_min, residual_rms_max=rms_max):
                continue
            cand = cand_pos.get(t_key)
            if cand is None or not np.all(np.isfinite(cand)) or np.all(cand == 0.0):
                continue
            gated.append((label, np.asarray(cand, dtype=np.float64), row))
        epoch_data.append(gated)

    hyb = np.zeros((n_epochs, 3), dtype=np.float64)
    for i, (tow, _) in enumerate(ref):
        t_key = round(float(tow), 1)
        hp = hybrid_pos.get(t_key)
        if hp is not None and np.all(np.isfinite(hp)) and not np.all(hp == 0.0):
            hyb[i] = np.asarray(hp, dtype=np.float64)

    print(f"{city}/{run} pool={len(filtered)} gated_epochs={sum(1 for g in epoch_data if g)} mode={select_mode}")

    # Baseline: oracle, current policy selector
    est_oracle = hyb.copy()
    for i, gated in enumerate(epoch_data):
        if not gated:
            continue
        true_pos = truth[i]
        best = min(gated, key=lambda c: float(np.linalg.norm(c[1] - true_pos)))
        est_oracle[i] = best[1]
    s_oracle = score_ppc2024(est_oracle, truth)
    print(f"  ORACLE                     pass={s_oracle.pass_distance_m:8.1f}m  score={s_oracle.score_pct:7.4f}%")

    est_current = hyb.copy()
    for i, gated in enumerate(epoch_data):
        if not gated:
            continue
        best = min(gated, key=lambda c: _rtkdiag_candidate_sort_key(c[2], mode=select_mode))
        est_current[i] = best[1]
    s_current = score_ppc2024(est_current, truth)
    print(f"  current ({select_mode})     pass={s_current.pass_distance_m:8.1f}m  score={s_current.score_pct:7.4f}%")

    # Cluster modes: inlier count with various thresholds
    for thresh_m in (2.0, 5.0, 10.0, 20.0):
        est = hyb.copy()
        for i, gated in enumerate(epoch_data):
            if not gated:
                continue
            if len(gated) == 1:
                est[i] = gated[0][1]
                continue
            positions = np.stack([c[1] for c in gated], axis=0)
            n_neighbors = []
            for j in range(len(gated)):
                d = np.linalg.norm(positions - positions[j], axis=1)
                n_neighbors.append(int((d <= thresh_m).sum()))  # includes self
            best_idx = int(np.argmax(n_neighbors))
            est[i] = positions[best_idx]
        s = score_ppc2024(est, truth)
        print(f"  inlier_count_{thresh_m:5.1f}m       pass={s.pass_distance_m:8.1f}m  score={s.score_pct:7.4f}%")

    # Cluster median: take median of biggest cluster
    for thresh_m in (2.0, 5.0, 10.0):
        est = hyb.copy()
        for i, gated in enumerate(epoch_data):
            if not gated:
                continue
            if len(gated) == 1:
                est[i] = gated[0][1]
                continue
            positions = np.stack([c[1] for c in gated], axis=0)
            best_n, best_med = -1, positions[0]
            for j in range(len(gated)):
                d = np.linalg.norm(positions - positions[j], axis=1)
                mask = d <= thresh_m
                n = int(mask.sum())
                if n > best_n:
                    best_n = n
                    best_med = np.median(positions[mask], axis=0)
            est[i] = best_med
        s = score_ppc2024(est, truth)
        print(f"  cluster_median_{thresh_m:5.1f}m     pass={s.pass_distance_m:8.1f}m  score={s.score_pct:7.4f}%")

    # Score-inlier blend: rank candidates by score, then break ties by inlier count
    for thresh_m in (5.0, 10.0):
        est = hyb.copy()
        for i, gated in enumerate(epoch_data):
            if not gated:
                continue
            if len(gated) == 1:
                est[i] = gated[0][1]
                continue
            positions = np.stack([c[1] for c in gated], axis=0)
            inliers = np.array([int((np.linalg.norm(positions - positions[j], axis=1) <= thresh_m).sum())
                                for j in range(len(gated))])
            score_keys = np.array([_rtkdiag_candidate_sort_key(c[2], mode=select_mode)[0] for c in gated])
            # blend: current-key rank + (n_cands - inliers); lower is better
            blend = (score_keys.argsort().argsort()) + (len(gated) - inliers)
            best_idx = int(np.argmin(blend))
            est[i] = positions[best_idx]
        s = score_ppc2024(est, truth)
        print(f"  score_inlier_blend_{thresh_m:5.1f}m pass={s.pass_distance_m:8.1f}m  score={s.score_pct:7.4f}%")


if __name__ == "__main__":
    main()
