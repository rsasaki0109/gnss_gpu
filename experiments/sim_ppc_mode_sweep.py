#!/usr/bin/env python3
"""For each run, compute PPC for every selector mode (and composites).

Per-epoch picks the candidate with the smallest sort-key for the given mode,
falling back to the hybrid floor when no candidate is gated. Reports honest
PPC (vs ground truth) for each mode. Cheap proxy for "what if PF used this
selector" — bypasses PF/INS entirely, so it overestimates a tiny bit
(no PF damping), but reveals which feature ordering yields better truth-distances.
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
    CTRBPFConfig,
)
from gnss_gpu.ppc_score import score_ppc2024  # type: ignore  # noqa: E402


MODES = (
    "residual",
    "score",
    "nrows",
    "ratio",
    "maxabs",
    # composites
    "rms_per_row",          # residual_rms / max(update_rows, 1)
    "score_per_row",        # (residual_rms/ratio) / max(update_rows, 1)
    "rms_minus_alpha_rows", # residual_rms - 0.1 * update_rows
    "log_combined",         # log(residual_rms+1e-3) - 0.5 * log(max(update_rows,1))
)


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


def _sort_key(row: dict[str, str], mode: str) -> tuple[float, float]:
    ratio = _diag_float(row, "final_ratio")
    residual = _diag_float(row, "final_residual_rms")
    update_rows = _diag_float(row, "final_update_rows")
    abs_max = _diag_float(row, "final_residual_abs_max")
    if mode == "residual":
        return (residual, -ratio)
    if mode == "score":
        return (residual / max(ratio, 1.0e-6), residual)
    if mode == "nrows":
        return (-update_rows, residual)
    if mode == "ratio":
        return (-ratio, residual)
    if mode == "maxabs":
        return (abs_max, residual)
    if mode == "rms_per_row":
        return (residual / max(update_rows, 1.0), residual)
    if mode == "score_per_row":
        return ((residual / max(ratio, 1.0e-6)) / max(update_rows, 1.0), residual)
    if mode == "rms_minus_alpha_rows":
        return (residual - 0.1 * update_rows, residual)
    if mode == "log_combined":
        return (np.log(residual + 1.0e-3) - 0.5 * np.log(max(update_rows, 1.0)), residual)
    return (residual, -ratio)


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
    if len(cand_dirs) != len(cand_labels):
        raise SystemExit(f"dirs={len(cand_dirs)} != labels={len(cand_labels)}")

    city = args.city
    run = args.run
    pos_filename = f"{city}_{run}_full.pos"

    ref = _load_full_reference(args.data_root / city / run / "reference.csv")
    truth = np.asarray([t for _, t in ref], dtype=np.float64)
    n_epochs = len(ref)

    hybrid_pos, _ = _load_hybrid_pos_file(args.hybrid_pos_dir / pos_filename)

    candidates: list[tuple[str, dict, dict]] = []
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
    cur_mode = str(cfg.rtkdiag_candidate_select_mode)

    # Pre-build per-epoch gated candidate lists with features and positions
    print(f"{city}/{run} cur_mode={cur_mode} ratio>={ratio_min} rms<={rms_max} pool={len(filtered)}")

    per_epoch_gated: list[list[tuple[str, np.ndarray, dict[str, str]]]] = []
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
        per_epoch_gated.append(gated)

    # Hybrid floor
    hyb_pos = np.zeros((n_epochs, 3), dtype=np.float64)
    for i, (tow, _) in enumerate(ref):
        t_key = round(float(tow), 1)
        hp = hybrid_pos.get(t_key)
        if hp is not None and np.all(np.isfinite(hp)) and not np.all(hp == 0.0):
            hyb_pos[i] = np.asarray(hp, dtype=np.float64)
    s_hyb = score_ppc2024(hyb_pos, truth)
    print(f"  HYBRID baseline  pass={s_hyb.pass_distance_m:8.1f}/{s_hyb.total_distance_m:.1f}m  "
          f"score={s_hyb.score_pct:7.4f}%")

    # Oracle
    est_oracle = hyb_pos.copy()
    for i, gated in enumerate(per_epoch_gated):
        if not gated:
            continue
        true_pos = truth[i]
        best = min(gated, key=lambda c: float(np.linalg.norm(c[1] - true_pos)))
        est_oracle[i] = best[1]
    s_oracle = score_ppc2024(est_oracle, truth)
    print(f"  ORACLE           pass={s_oracle.pass_distance_m:8.1f}/{s_oracle.total_distance_m:.1f}m  "
          f"score={s_oracle.score_pct:7.4f}%  (+{s_oracle.score_pct - s_hyb.score_pct:.2f}pp)")

    # Each mode
    results = []
    for mode in MODES:
        est = hyb_pos.copy()
        n_pick = 0
        for i, gated in enumerate(per_epoch_gated):
            if not gated:
                continue
            best = min(gated, key=lambda c: _sort_key(c[2], mode))
            est[i] = best[1]
            n_pick += 1
        s = score_ppc2024(est, truth)
        results.append((mode, s.score_pct, s.pass_distance_m, n_pick))
        marker = "*" if mode == cur_mode else " "
        print(f"  {marker} {mode:22s}  pass={s.pass_distance_m:8.1f}m  score={s.score_pct:7.4f}%  picks={n_pick}")

    # Sort and show top 3
    results.sort(key=lambda r: -r[1])
    print(f"  best mode: {results[0][0]} ({results[0][1]:.4f}%, gap to oracle: {s_oracle.score_pct - results[0][1]:.2f}pp)")


if __name__ == "__main__":
    main()
