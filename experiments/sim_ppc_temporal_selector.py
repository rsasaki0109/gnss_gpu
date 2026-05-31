#!/usr/bin/env python3
"""Temporal-smoothing selector simulation.

For each epoch, pick by weighted combination of:
  - score (rms / ratio)
  - distance from previous selected position (low = consistent trajectory)

Modes tested:
  - prev_dist_alpha_X: pick min(score + X * dist_to_prev)
  - prev_consistent_only: among top-K by score, pick the one closest to prev
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
    p.add_argument("--alphas", default="0.0001,0.0003,0.001,0.003,0.01,0.03,0.1",
                   help="Comma-separated temporal penalty weights")
    args = p.parse_args()

    cand_dirs = [Path(s.strip()) for s in args.candidate_dirs.split(",") if s.strip()]
    cand_labels = [s.strip() for s in args.candidate_labels.split(",") if s.strip()]
    alphas = [float(s.strip()) for s in args.alphas.split(",") if s.strip()]
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

    print(f"{city}/{run} pool={len(filtered)} gated={sum(1 for g in epoch_data if g)} mode={select_mode}")

    # Baselines
    est_oracle = hyb.copy()
    for i, gated in enumerate(epoch_data):
        if not gated:
            continue
        true_pos = truth[i]
        best = min(gated, key=lambda c: float(np.linalg.norm(c[1] - true_pos)))
        est_oracle[i] = best[1]
    s = score_ppc2024(est_oracle, truth)
    print(f"  ORACLE                    pass={s.pass_distance_m:8.1f}m  score={s.score_pct:7.4f}%")

    est_current = hyb.copy()
    for i, gated in enumerate(epoch_data):
        if not gated:
            continue
        best = min(gated, key=lambda c: _rtkdiag_candidate_sort_key(c[2], mode=select_mode))
        est_current[i] = best[1]
    s = score_ppc2024(est_current, truth)
    print(f"  current ({select_mode})    pass={s.pass_distance_m:8.1f}m  score={s.score_pct:7.4f}%")

    # Temporal: pick among top-K by score, the one closest to prev
    for K in (3, 5, 10, 20):
        est = hyb.copy()
        prev = None
        for i, gated in enumerate(epoch_data):
            if not gated:
                prev = est[i] if not np.all(est[i] == 0) else prev
                continue
            if prev is None:
                # first epoch: use score
                best = min(gated, key=lambda c: _rtkdiag_candidate_sort_key(c[2], mode=select_mode))
            else:
                sorted_by_score = sorted(gated, key=lambda c: _rtkdiag_candidate_sort_key(c[2], mode=select_mode))[:K]
                best = min(sorted_by_score, key=lambda c, _p=prev: float(np.linalg.norm(c[1] - _p)))
            est[i] = best[1]
            prev = best[1]
        s = score_ppc2024(est, truth)
        print(f"  topK_score_prev_K={K:2d}      pass={s.pass_distance_m:8.1f}m  score={s.score_pct:7.4f}%")

    # Temporal: pick by score + alpha * dist_to_prev (normalized)
    for alpha in alphas:
        est = hyb.copy()
        prev = None
        for i, gated in enumerate(epoch_data):
            if not gated:
                prev = est[i] if not np.all(est[i] == 0) else prev
                continue
            if prev is None:
                best = min(gated, key=lambda c: _rtkdiag_candidate_sort_key(c[2], mode=select_mode))
            else:
                best = min(gated, key=lambda c, _a=alpha, _p=prev:
                          _rtkdiag_candidate_sort_key(c[2], mode=select_mode)[0] + _a * float(np.linalg.norm(c[1] - _p)))
            est[i] = best[1]
            prev = best[1]
        s = score_ppc2024(est, truth)
        print(f"  score+a*dist_alpha={alpha:.6f}  pass={s.pass_distance_m:8.1f}m  score={s.score_pct:7.4f}%")

    # Temporal with hybrid delta: predict current selected position as
    # prev_selected + (hybrid[i] - hybrid[i-1]) when both hybrid anchors exist.
    for alpha in alphas:
        est = hyb.copy()
        prev = None
        prev_hyb = None
        for i, gated in enumerate(epoch_data):
            anchor = hyb[i]
            anchor_valid = not np.all(anchor == 0)
            pred = None
            if prev is not None and prev_hyb is not None and anchor_valid:
                pred = prev + (anchor - prev_hyb)
            if not gated:
                if anchor_valid:
                    prev = anchor
                    prev_hyb = anchor
                continue
            if pred is None:
                best = min(gated, key=lambda c: _rtkdiag_candidate_sort_key(c[2], mode=select_mode))
            else:
                best = min(gated, key=lambda c, _a=alpha, _p=pred:
                          _rtkdiag_candidate_sort_key(c[2], mode=select_mode)[0] + _a * float(np.linalg.norm(c[1] - _p)))
            est[i] = best[1]
            prev = best[1]
            if anchor_valid:
                prev_hyb = anchor
        s = score_ppc2024(est, truth)
        print(f"  score+a*hybdelta_a={alpha:.6f} pass={s.pass_distance_m:8.1f}m  score={s.score_pct:7.4f}%")

    # Temporal: pick by hybrid floor distance (use hybrid as anchor instead of prev)
    for alpha in alphas:
        est = hyb.copy()
        for i, gated in enumerate(epoch_data):
            if not gated:
                continue
            anchor = hyb[i]
            if np.all(anchor == 0):
                best = min(gated, key=lambda c: _rtkdiag_candidate_sort_key(c[2], mode=select_mode))
            else:
                best = min(gated, key=lambda c, _a=alpha, _h=anchor:
                          _rtkdiag_candidate_sort_key(c[2], mode=select_mode)[0] + _a * float(np.linalg.norm(c[1] - _h)))
            est[i] = best[1]
        s = score_ppc2024(est, truth)
        print(f"  score+a*hybdist_a={alpha:.6f}  pass={s.pass_distance_m:8.1f}m  score={s.score_pct:7.4f}%")


if __name__ == "__main__":
    main()
