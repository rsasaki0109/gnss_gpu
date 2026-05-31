#!/usr/bin/env python3
"""Alpha-grid sweep over composite selector key = residual / (ratio^a * rows^b)."""

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
    p.add_argument("--alphas", default="0.0,0.5,1.0,1.5,2.0",
                   help="Comma-separated ratio exponents")
    p.add_argument("--betas", default="0.0,0.5,1.0,1.5,2.0",
                   help="Comma-separated update_rows exponents")
    args = p.parse_args()

    cand_dirs = [Path(s.strip()) for s in args.candidate_dirs.split(",") if s.strip()]
    cand_labels = [s.strip() for s in args.candidate_labels.split(",") if s.strip()]
    alphas = [float(a) for a in args.alphas.split(",")]
    betas = [float(b) for b in args.betas.split(",")]

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

    # Pre-collect gated tuples: (residual, ratio, rows, pos)
    epoch_data: list[list[tuple[float, float, float, np.ndarray]]] = []
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
            r = _diag_float(row, "final_residual_rms")
            ra = _diag_float(row, "final_ratio")
            n = _diag_float(row, "final_update_rows")
            gated.append((r, ra, n, np.asarray(cand, dtype=np.float64)))
        epoch_data.append(gated)

    hyb = np.zeros((n_epochs, 3), dtype=np.float64)
    for i, (tow, _) in enumerate(ref):
        t_key = round(float(tow), 1)
        hp = hybrid_pos.get(t_key)
        if hp is not None and np.all(np.isfinite(hp)) and not np.all(hp == 0.0):
            hyb[i] = np.asarray(hp, dtype=np.float64)

    print(f"{city}/{run} pool={len(filtered)} gated_epochs={sum(1 for g in epoch_data if g)}")
    best = (None, None, -1.0)
    for a in alphas:
        row = []
        for b in betas:
            est = hyb.copy()
            for i, gated in enumerate(epoch_data):
                if not gated:
                    continue
                # key = residual / (ratio^a * rows^b); lower is better
                pick = min(
                    gated,
                    key=lambda g, _a=a, _b=b: g[0] / (max(g[1], 1.0e-6) ** _a * max(g[2], 1.0) ** _b),
                )
                est[i] = pick[3]
            s = score_ppc2024(est, truth)
            row.append(s.score_pct)
            if s.score_pct > best[2]:
                best = (a, b, s.score_pct)
        print(f"  a={a:.1f}  " + "  ".join(f"b={b:.1f}:{v:6.3f}" for b, v in zip(betas, row)))
    print(f"  BEST a={best[0]:.1f} b={best[1]:.1f} score={best[2]:.4f}%")


if __name__ == "__main__":
    main()
