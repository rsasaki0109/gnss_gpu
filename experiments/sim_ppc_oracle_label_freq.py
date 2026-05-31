#!/usr/bin/env python3
"""For each (city, run), compute the per-label frequency that the label is
the truth-closest gated candidate. Report the top-K labels per run.

Usage:
  python3 experiments/sim_ppc_oracle_label_freq.py \
    --candidate-dirs <expanded_dirs_csv> \
    --candidate-labels <expanded_labels_csv> \
    --policy phase11dd
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
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
)


_DEFAULT_DATA_ROOT = Path("/media/sasaki/aiueo/ai_coding_ws/datasets/PPC-Dataset-data")
_FULL_RUNS = (
    ("tokyo", "run1"), ("tokyo", "run2"), ("tokyo", "run3"),
    ("nagoya", "run1"), ("nagoya", "run2"), ("nagoya", "run3"),
)


def _load_pos_file(path):
    out = {}
    with open(path) as fh:
        for line in fh:
            line = line.rstrip()
            if not line or line.startswith("%"):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                tow = round(float(parts[1]), 1)
                x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
            except ValueError:
                continue
            out[tow] = np.array([x, y, z], dtype=np.float64)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=_DEFAULT_DATA_ROOT)
    p.add_argument("--hybrid-pos-dir", type=Path,
                   default=Path("experiments/results/libgnss_rtk_pos_v5"))
    p.add_argument("--candidate-dirs", required=True)
    p.add_argument("--candidate-labels", required=True)
    p.add_argument("--policy", default="phase11dd")
    p.add_argument("--top-k", type=int, default=20)
    args = p.parse_args()

    cand_dirs = [Path(s.strip()) for s in args.candidate_dirs.split(",") if s.strip()]
    cand_labels = [s.strip() for s in args.candidate_labels.split(",") if s.strip()]
    if len(cand_dirs) != len(cand_labels):
        raise SystemExit(f"dirs={len(cand_dirs)} != labels={len(cand_labels)}")

    for city, run in _FULL_RUNS:
        pos_filename = f"{city}_{run}_full.pos"
        ref = _load_full_reference(args.data_root / city / run / "reference.csv")
        truth = np.asarray([t for _, t in ref], dtype=np.float64)
        candidates = []
        for d, lbl in zip(cand_dirs, cand_labels):
            pos_path = d / pos_filename
            diag_path = d / f"{city}_{run}_full.csv"
            if not pos_path.is_file() or not diag_path.is_file():
                continue
            cand_pos = _load_pos_file(pos_path)
            diag = _load_rtk_diag_file(diag_path)
            candidates.append((lbl, cand_pos, diag))
        filtered = _filter_rtkdiag_candidates_by_policy(
            candidates, city=city, run=run, policy=args.policy,
        )
        cfg = _apply_rtkdiag_run_index_policy(
            CTRBPFConfig(enable_rtkdiag_pf_rescue=True),
            run=run, policy=args.policy, city=city,
        )
        ratio_min = float(cfg.rtkdiag_candidate_ratio_min)
        rms_max = float(cfg.rtkdiag_candidate_residual_rms_max)
        # Per-label oracle pick count
        pick_count = defaultdict(int)
        gate_count = defaultdict(int)  # how often each label gate-passes
        n_oracle_epochs = 0
        for i, (tow, _) in enumerate(ref):
            t_key = round(float(tow), 1)
            true_pos = truth[i]
            best_dist = None
            best_label = None
            for label, cand_pos, cand_diag in filtered:
                row = cand_diag.get(t_key)
                if not _rtkdiag_candidate_gate(row, ratio_min=ratio_min, residual_rms_max=rms_max):
                    continue
                gate_count[label] += 1
                cand = cand_pos.get(t_key)
                if cand is None or not np.all(np.isfinite(cand)) or np.all(cand == 0.0):
                    continue
                dist = float(np.linalg.norm(np.asarray(cand) - true_pos))
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_label = label
            if best_label is not None:
                pick_count[best_label] += 1
                n_oracle_epochs += 1

        print(f"\n{city}/{run}: gates ratio>={ratio_min} rms<={rms_max}, oracle picks {n_oracle_epochs} epochs")
        sorted_picks = sorted(pick_count.items(), key=lambda kv: kv[1], reverse=True)
        for lbl, cnt in sorted_picks[:args.top_k]:
            mark = " (NEW)" if lbl.startswith("x") else ""
            print(f"  {lbl:<30s} pick={cnt:5d} ({100.0*cnt/max(n_oracle_epochs,1):.1f}%) gate={gate_count[lbl]}{mark}")


if __name__ == "__main__":
    main()
