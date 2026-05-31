#!/usr/bin/env python3
"""Greedy p_pass label-factor sweep on the exact phase35 n/r2 pool."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "python"))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from diagnose_nr2_ranker_with_extra_candidate import (  # noqa: E402
    DATA_ROOT,
    RESULTS,
    _candidate_options,
    _default_candidates,
    _effective_config,
    _load_candidates,
)
from exp_ppc_ctrbpf_fgo import (  # noqa: E402
    _diag_float,
    _filter_rtkdiag_candidates_by_policy,
    _load_full_reference,
    _load_hybrid_pos_file,
    _load_ranker_predictions,
)
from gnss_gpu.ppc_score import ppc_segment_distances  # noqa: E402

FACTORS = [
    0.001,
    0.003,
    0.005,
    0.008,
    0.01,
    0.02,
    0.03,
    0.05,
    0.08,
    0.1,
    0.15,
    0.2,
    0.3,
    0.4,
    0.6,
    0.8,
    1.0,
    1.25,
    1.5,
    2.0,
    3.0,
    5.0,
    8.0,
    12.0,
    20.0,
    40.0,
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city", default="nagoya")
    parser.add_argument("--run", default="run2")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--hybrid-pos-dir", type=Path, default=RESULTS / "libgnss_rtk_pos_v5")
    parser.add_argument(
        "--input",
        type=Path,
        default=RESULTS / "selector_ranker_predictions_v3_nr2_labelboost_pb40.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS / "selector_ranker_predictions_v3_nr2_labelboost_pb40_sweep.csv",
    )
    parser.add_argument("--policy", default="phase11ep")
    parser.add_argument("--ratio-min", type=float, default=1.0)
    parser.add_argument("--residual-rms-max", type=float, default=50.0)
    parser.add_argument("--rms-prefilter-k", type=int, default=99)
    parser.add_argument("--max-steps", type=int, default=16)
    parser.add_argument("--min-gain", type=float, default=0.001)
    parser.add_argument(
        "--base-labels-file",
        type=Path,
        default=Path("/tmp/nagoya_run2_phase11fa_labels.txt"),
    )
    parser.add_argument(
        "--base-dirs-file",
        type=Path,
        default=Path("/tmp/nagoya_run2_phase11fa_dirs.txt"),
    )
    parser.add_argument(
        "--extra-label",
        action="append",
        default=["xd_nr2_hs_pb40"],
    )
    parser.add_argument(
        "--extra-dir",
        action="append",
        type=Path,
        default=[
            RESULTS / "libgnss_diag_phase34/nr2_hs_piecebias40_oracle_556184_556337"
        ],
    )
    return parser.parse_args()


def _score_pick(
    pick_idx: np.ndarray,
    pass_flag: np.ndarray,
    weights: np.ndarray,
    fallback_pass: np.ndarray,
    total_w: float,
) -> tuple[float, int, float]:
    rows = np.arange(len(pick_idx))
    candidate_rows = pick_idx >= 0
    passed = np.zeros(len(pick_idx), dtype=bool)
    passed[~candidate_rows] = fallback_pass[~candidate_rows]
    if np.any(candidate_rows):
        passed[candidate_rows] = pass_flag[rows[candidate_rows], pick_idx[candidate_rows]]
    pass_w = float(weights[passed].sum())
    pct = 100.0 * pass_w / total_w if total_w > 0.0 else 0.0
    return pct, int(passed.sum()), pass_w


def _pick_indices(score: np.ndarray, fallback_idx: np.ndarray) -> np.ndarray:
    finite_any = np.isfinite(score).any(axis=1)
    pick = np.full(score.shape[0], -1, dtype=np.int32)
    pick[finite_any] = np.argmax(score[finite_any], axis=1)
    fallback_rows = ~finite_any
    pick[fallback_rows] = fallback_idx[fallback_rows]
    return pick


def _build_matrix(args: argparse.Namespace) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[str],
    float,
]:
    city = str(args.city)
    run = str(args.run)
    run_id = f"{city}_{run}"
    labels, dirs = _default_candidates(args)
    candidates_all = _load_candidates(labels, dirs, city=city, run=run)
    candidates = _filter_rtkdiag_candidates_by_policy(
        candidates_all,
        city=city,
        run=run,
        policy=str(args.policy),
    )
    labels = [label for label, _pos, _diag in candidates]
    label_to_idx = {label: i for i, label in enumerate(labels)}
    cfg = _effective_config(args)
    pred = _load_ranker_predictions(str(args.input))
    pred_run = {(tow, label): p for (rid, tow, label), p in pred.items() if rid == run_id}

    ref = _load_full_reference(args.data_root / city / run / "reference.csv")
    truth = np.asarray([ecef for _, ecef in ref], dtype=np.float64)
    weights = ppc_segment_distances(truth)
    total_w = float(weights.sum())
    hybrid_pos, _hybrid_status = _load_hybrid_pos_file(
        args.hybrid_pos_dir / f"{city}_{run}_full.pos"
    )

    score = np.full((len(ref), len(labels)), -np.inf, dtype=np.float64)
    pass_flag = np.zeros((len(ref), len(labels)), dtype=bool)
    fallback_idx = np.full(len(ref), -1, dtype=np.int32)
    fallback_pass = np.zeros(len(ref), dtype=bool)

    for i, (tow_raw, true_pos) in enumerate(ref):
        tow = round(float(tow_raw), 1)
        hp = hybrid_pos.get(tow)
        if hp is not None and np.all(np.isfinite(hp)) and not np.all(np.asarray(hp) == 0.0):
            fallback_pass[i] = bool(np.linalg.norm(np.asarray(hp) - true_pos) < 0.5)
        options = _candidate_options(candidates, tow=tow, cfg=cfg)
        if not options:
            continue
        fallback_label = min(
            options,
            key=lambda item: _diag_float(item[2], "final_residual_rms"),
        )[0]
        fallback_idx[i] = label_to_idx[fallback_label]
        for label, pos, _row in options:
            j = label_to_idx[label]
            p_pass = pred_run.get((tow, label))
            if p_pass is not None:
                score[i, j] = float(p_pass)
            pass_flag[i, j] = bool(np.linalg.norm(pos - true_pos) < 0.5)
    return score, pass_flag, weights, fallback_idx, fallback_pass, labels, total_w


def main() -> None:
    args = _parse_args()
    score, pass_flag, weights, fallback_idx, fallback_pass, labels, total_w = _build_matrix(args)
    label_to_idx = {label: i for i, label in enumerate(labels)}

    current = score.copy()
    pick = _pick_indices(current, fallback_idx)
    current_score, current_pass, current_w = _score_pick(
        pick,
        pass_flag,
        weights,
        fallback_pass,
        total_w,
    )
    print(
        f"baseline score={current_score:.6f} pass_epochs={current_pass} "
        f"pass_w={current_w:.3f}/{total_w:.3f}",
        flush=True,
    )

    cumulative = {label: 1.0 for label in labels}
    chosen: list[tuple[str, float, float]] = []
    for step in range(1, int(args.max_steps) + 1):
        best: tuple[float, str, float, int, float] | None = None
        for label in labels:
            j = label_to_idx[label]
            original_col = current[:, j].copy()
            for factor in FACTORS:
                current[:, j] = original_col * factor
                pick = _pick_indices(current, fallback_idx)
                trial_score, trial_pass, trial_w = _score_pick(
                    pick,
                    pass_flag,
                    weights,
                    fallback_pass,
                    total_w,
                )
                if best is None or trial_score > best[0]:
                    best = (trial_score, label, factor, trial_pass, trial_w)
            current[:, j] = original_col
        if best is None:
            break
        best_score, label, factor, pass_epochs, pass_w = best
        gain = best_score - current_score
        if gain < float(args.min_gain):
            print(f"stop step={step}: best gain {gain:.6f} < {args.min_gain}")
            break
        j = label_to_idx[label]
        current[:, j] *= factor
        cumulative[label] *= factor
        current_score = best_score
        chosen.append((label, factor, best_score))
        print(
            f"step={step:02d} label={label:28s} factor={factor:g} "
            f"score={best_score:.6f} gain={gain:+.6f} "
            f"pass_epochs={pass_epochs} pass_w={pass_w:.3f}",
            flush=True,
        )

    print("\nchosen factors:")
    for label, factor, score_value in chosen:
        print(f"  {label!r}: {factor:g},  # score={score_value:.6f}")
    print("\ncumulative non-1 factors:")
    for label, factor in cumulative.items():
        if factor != 1.0:
            print(f"  {label!r}: {factor:g}")

    out = pd.read_csv(args.input)
    run_id = f"{args.city}_{args.run}"
    run_mask = out["run_id"] == run_id
    for label, factor in cumulative.items():
        if factor == 1.0:
            continue
        mask = run_mask & (out["label"] == label)
        out.loc[mask, "p_pass"] = out.loc[mask, "p_pass"] * factor
    out.to_csv(args.output, index=False)
    print(f"\nwrote {args.output} rows={len(out)} final_score={current_score:.6f}")


if __name__ == "__main__":
    main()
