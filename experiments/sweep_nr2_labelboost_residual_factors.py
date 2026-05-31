#!/usr/bin/env python3
"""Greedy label-factor sweep for nagoya/run2 ranker residuals."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/media/sasaki/aiueo/ai_coding_ws/gnss_gpu")
RESULTS = REPO / "experiments/results"
DEFAULT_INPUT = RESULTS / "selector_ranker_predictions_v3_nr2_labelboost.csv"
DEFAULT_OUTPUT = RESULTS / "selector_ranker_predictions_v3_nr2_labelboost2.csv"

FACTORS = [0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8,
           1.0, 1.25, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run-id", default="nagoya_run2")
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--min-gain", type=float, default=0.001)
    return parser.parse_args()


def _prepare(run_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    run_df = run_df.copy()
    run_df["tow"] = run_df["tow"].round(1)
    epochs = np.array(sorted(run_df["tow"].unique()))
    labels = sorted(run_df["label"].unique())
    epoch_index = {tow: i for i, tow in enumerate(epochs)}
    label_index = {label: i for i, label in enumerate(labels)}

    score = np.full((len(epochs), len(labels)), -np.inf, dtype=float)
    pass_flag = np.zeros((len(epochs), len(labels)), dtype=np.int8)
    path_weight = np.zeros((len(epochs), len(labels)), dtype=float)
    for row in run_df.itertuples(index=False):
        i = epoch_index[row.tow]
        j = label_index[row.label]
        score[i, j] = float(row.p_pass)
        pass_flag[i, j] = int(row.is_pass_50cm)
        path_weight[i, j] = float(row.path_weight)
    return score, pass_flag, path_weight, epochs, labels


def _top2(score: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    top_idx = np.argmax(score, axis=1)
    rows = np.arange(score.shape[0])
    top_score = score[rows, top_idx]
    masked = score.copy()
    masked[rows, top_idx] = -np.inf
    second_idx = np.argmax(masked, axis=1)
    second_score = masked[rows, second_idx]
    return top_idx, top_score, second_idx, second_score


def _score_pick(
    pick_idx: np.ndarray,
    pass_flag: np.ndarray,
    path_weight: np.ndarray,
) -> tuple[float, int, float, float]:
    rows = np.arange(len(pick_idx))
    weights = path_weight[rows, pick_idx]
    total = float(weights.sum())
    passed = pass_flag[rows, pick_idx] == 1
    pass_weight = float(weights[passed].sum())
    pct = 100.0 * pass_weight / total if total > 0 else 0.0
    return pct, int(passed.sum()), pass_weight, total


def _score_matrix(score: np.ndarray, pass_flag: np.ndarray,
                  path_weight: np.ndarray) -> tuple[float, int, float, float]:
    return _score_pick(np.argmax(score, axis=1), pass_flag, path_weight)


def _score_label_trial(
    score: np.ndarray,
    pass_flag: np.ndarray,
    path_weight: np.ndarray,
    label_idx: int,
    factor: float,
    top_idx: np.ndarray,
    top_score: np.ndarray,
    second_idx: np.ndarray,
    second_score: np.ndarray,
) -> tuple[float, int, float, float]:
    non_idx = np.where(top_idx == label_idx, second_idx, top_idx)
    non_score = np.where(top_idx == label_idx, second_score, top_score)
    label_score = score[:, label_idx] * factor
    pick_idx = np.where(label_score > non_score, label_idx, non_idx)
    return _score_pick(pick_idx, pass_flag, path_weight)


def main() -> None:
    args = _parse_args()
    df = pd.read_csv(args.input)
    mask_run = df["run_id"] == args.run_id
    run_df = df.loc[mask_run, ["tow", "label", "is_pass_50cm", "path_weight", "p_pass"]].copy()
    score, pass_flag, path_weight, _epochs, labels = _prepare(run_df)
    label_to_idx = {label: i for i, label in enumerate(labels)}

    current = score.copy()
    current_score, current_pass, current_w, total_w = _score_matrix(current, pass_flag, path_weight)
    print(
        f"baseline score={current_score:.4f} pass_epochs={current_pass} "
        f"pass_w={current_w:.3f}/{total_w:.3f}",
        flush=True,
    )

    chosen: list[tuple[str, float, float]] = []
    cumulative_factors = {label: 1.0 for label in labels}
    for step in range(1, args.max_steps + 1):
        best: tuple[float, str, float, int, float] | None = None
        top_idx, top_score, second_idx, second_score = _top2(current)
        for label in labels:
            label_idx = label_to_idx[label]
            for factor in FACTORS:
                trial_score, pass_epochs, pass_w, _ = _score_label_trial(
                    current,
                    pass_flag,
                    path_weight,
                    label_idx,
                    factor,
                    top_idx,
                    top_score,
                    second_idx,
                    second_score,
                )
                if best is None or trial_score > best[0]:
                    best = (trial_score, label, factor, pass_epochs, pass_w)
        if best is None:
            break
        score, label, factor, pass_epochs, pass_w = best
        gain = score - current_score
        if gain < args.min_gain:
            print(f"stop step={step}: best gain {gain:.6f} < {args.min_gain}")
            break
        current[:, label_to_idx[label]] *= factor
        cumulative_factors[label] *= factor
        current_score = score
        chosen.append((label, factor, score))
        print(
            f"step={step:02d} label={label:24s} factor={factor:g} "
            f"score={score:.4f} gain={gain:+.4f} pass_epochs={pass_epochs} pass_w={pass_w:.3f}"
        )

    print("\nchosen residual factors:")
    for label, factor, score in chosen:
        print(f"  {label!r}: {factor:g},  # score={score:.4f}")

    out = df.copy()
    run_mask = out["run_id"] == args.run_id
    for label, factor in cumulative_factors.items():
        if factor == 1.0:
            continue
        mask = run_mask & (out["label"] == label)
        out.loc[mask, "p_pass"] = out.loc[mask, "p_pass"] * factor
    out.to_csv(args.output, index=False)
    print(f"\nwrote {args.output} rows={len(out)} final_score={current_score:.4f}")


if __name__ == "__main__":
    main()
