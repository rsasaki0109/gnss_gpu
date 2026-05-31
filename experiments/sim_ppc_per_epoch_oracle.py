#!/usr/bin/env python3
"""Per-epoch oracle: pick truth-closest candidate per epoch, compute PPC score.

For each gate-passing epoch, pick the candidate whose position is nearest to
the ground truth. Aggregate honest PPC. This shows the absolute upper bound
achievable from the current candidate pool with optimal selection.

Reports gap: (oracle - actual_PF_score) = headroom from improved selector.
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
    _filter_rtkdiag_candidates_by_policy,
    _load_full_reference,
    _load_hybrid_pos_file,
    _load_rtk_diag_file,
    _rtkdiag_candidate_gate,
    CTRBPFConfig,
)
from gnss_gpu.ppc_score import score_ppc2024  # type: ignore  # noqa: E402


def _load_pos_file(path: Path) -> dict[float, np.ndarray]:
    """Load .pos file as {tow: ECEF}."""
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
    p.add_argument("--candidate-dirs", required=True,
                   help="Comma-separated candidate dirs (also used as diag dirs).")
    p.add_argument("--candidate-labels", required=True,
                   help="Comma-separated labels.")
    p.add_argument("--policy", default="phase11bw",
                   help="rtkdiag run-index policy")
    p.add_argument("--city", default="nagoya")
    p.add_argument("--run", default="run2")
    args = p.parse_args()

    cand_dirs = [Path(s.strip()) for s in args.candidate_dirs.split(",") if s.strip()]
    cand_labels = [s.strip() for s in args.candidate_labels.split(",") if s.strip()]
    if len(cand_dirs) != len(cand_labels):
        raise SystemExit(f"dirs={len(cand_dirs)} != labels={len(cand_labels)}")

    city = args.city
    run = args.run
    pos_filename = f"{city}_{run}_full.pos"

    # Load reference (ground truth)
    ref = _load_full_reference(args.data_root / city / run / "reference.csv")
    truth = np.asarray([t for _, t in ref], dtype=np.float64)
    n_epochs = len(ref)

    # Load hybrid floor (returns (positions, statuses))
    hybrid_pos, _hybrid_status = _load_hybrid_pos_file(args.hybrid_pos_dir / pos_filename)

    # Load candidates with their diagnostics
    candidates: list[tuple[str, dict[float, np.ndarray], dict[float, dict[str, str]]]] = []
    for d, lbl in zip(cand_dirs, cand_labels):
        pos_path = d / pos_filename
        diag_path = d / f"{city}_{run}_full.csv"
        if not pos_path.is_file() or not diag_path.is_file():
            print(f"  skip {lbl}: missing files in {d}")
            continue
        cand_pos = _load_pos_file(pos_path)
        diag = _load_rtk_diag_file(diag_path)
        candidates.append((lbl, cand_pos, diag))
    print(f"Loaded {len(candidates)} candidates for {city}/{run}")

    # Apply policy filter
    filtered = _filter_rtkdiag_candidates_by_policy(
        candidates,
        city=city, run=run, policy=args.policy,
    )
    print(f"After policy filter ({args.policy}): {len(filtered)} candidates")

    # Get the run's selector params via policy
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
    print(f"Effective gates: ratio_min={ratio_min}, rms_max={rms_max}")

    # Per-epoch oracle: pick truth-closest gated candidate
    est_oracle = np.zeros((n_epochs, 3), dtype=np.float64)
    n_gated = 0
    n_picked_cand = 0
    for i, (tow, _) in enumerate(ref):
        t_key = round(float(tow), 1)
        # Default to hybrid floor
        hp = hybrid_pos.get(t_key)
        if hp is not None and np.all(np.isfinite(hp)) and not np.all(hp == 0.0):
            est_oracle[i] = np.asarray(hp, dtype=np.float64)
        # Gate candidates
        true_pos = truth[i]
        best_dist = None
        best_pos = None
        any_gated = False
        for label, cand_pos, cand_diag in filtered:
            row = cand_diag.get(t_key)
            if not _rtkdiag_candidate_gate(row, ratio_min=ratio_min, residual_rms_max=rms_max):
                continue
            any_gated = True
            cand = cand_pos.get(t_key)
            if cand is None or not np.all(np.isfinite(cand)) or np.all(cand == 0.0):
                continue
            dist = float(np.linalg.norm(np.asarray(cand) - true_pos))
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_pos = np.asarray(cand, dtype=np.float64)
        if any_gated:
            n_gated += 1
        if best_pos is not None:
            est_oracle[i] = best_pos
            n_picked_cand += 1

    s_oracle = score_ppc2024(est_oracle, truth)
    print(f"\n{city}/{run} per-epoch ORACLE:")
    print(f"  Total epochs: {n_epochs}")
    print(f"  Gated epochs: {n_gated} ({100.0*n_gated/n_epochs:.1f}%)")
    print(f"  Picked candidate: {n_picked_cand} ({100.0*n_picked_cand/n_epochs:.1f}%)")
    print(f"  PPC honest: {s_oracle.score_pct:.4f}% (pass {s_oracle.pass_distance_m:.1f}m / total {s_oracle.total_distance_m:.1f}m)")

    # Hybrid-only baseline for comparison
    est_hyb = np.zeros((n_epochs, 3), dtype=np.float64)
    for i, (tow, _) in enumerate(ref):
        t_key = round(float(tow), 1)
        hp = hybrid_pos.get(t_key)
        if hp is not None and np.all(np.isfinite(hp)) and not np.all(hp == 0.0):
            est_hyb[i] = np.asarray(hp, dtype=np.float64)
    s_hyb = score_ppc2024(est_hyb, truth)
    print(f"\n{city}/{run} HYBRID baseline:")
    print(f"  PPC honest: {s_hyb.score_pct:.4f}% (pass {s_hyb.pass_distance_m:.1f}m)")
    print(f"  Oracle gain over hybrid: +{s_oracle.score_pct - s_hyb.score_pct:.2f}pp")


if __name__ == "__main__":
    main()
