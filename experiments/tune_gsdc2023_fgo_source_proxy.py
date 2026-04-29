#!/usr/bin/env python3
"""Search simple FGO source-selection proxy rules from chunk diagnostics."""

from __future__ import annotations

import argparse
import glob
import json
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


DEFAULT_FEATURES = (
    "fgo_candidate_quality_score",
    "fgo_candidate_gap_p95_m",
    "fgo_candidate_mse_pr",
    "baseline_candidate_mse_pr",
    "raw_wls_candidate_mse_pr",
    "fgo_mse_ratio",
    "raw_mse_ratio",
    "fgo_raw_mse_ratio",
)


@dataclass(frozen=True)
class Condition:
    feature: str
    op: str
    threshold: float

    def mask(self, frame: pd.DataFrame) -> np.ndarray:
        values = frame[self.feature].to_numpy(dtype=np.float64)
        finite = np.isfinite(values)
        if self.op == "<=":
            return finite & (values <= self.threshold)
        if self.op == ">=":
            return finite & (values >= self.threshold)
        raise ValueError(f"unsupported op: {self.op}")

    def payload(self) -> dict[str, object]:
        return {"feature": self.feature, "op": self.op, "threshold": float(self.threshold)}

    def label(self) -> str:
        return f"{self.feature} {self.op} {self.threshold:.6g}"


@dataclass(frozen=True)
class RuleResult:
    conditions: tuple[Condition, ...]
    gain_score_m: float
    selected_chunks: int
    true_positive_chunks: int
    false_positive_chunks: int
    false_negative_chunks: int
    loo_min_gain_score_m: float | None
    loo_mean_gain_score_m: float | None

    def payload(self) -> dict[str, object]:
        return {
            "rule": " AND ".join(condition.label() for condition in self.conditions),
            "conditions": [condition.payload() for condition in self.conditions],
            "gain_score_m": float(self.gain_score_m),
            "selected_chunks": int(self.selected_chunks),
            "true_positive_chunks": int(self.true_positive_chunks),
            "false_positive_chunks": int(self.false_positive_chunks),
            "false_negative_chunks": int(self.false_negative_chunks),
            "loo_min_gain_score_m": self.loo_min_gain_score_m,
            "loo_mean_gain_score_m": self.loo_mean_gain_score_m,
        }


def expand_chunk_paths(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = sorted(Path(path) for path in glob.glob(pattern))
        if matches:
            paths.extend(matches)
        else:
            paths.append(Path(pattern))
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return unique


def load_chunk_frames(paths: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        frame = pd.read_csv(path)
        if "metrics_name" not in frame:
            frame["metrics_name"] = path.parent.name
        if "trip_slug" not in frame:
            frame["trip_slug"] = frame["metrics_name"]
        frames.append(frame)
    if not frames:
        raise ValueError("no chunk diagnostics files found")
    out = pd.concat(frames, ignore_index=True)
    add_derived_features(out)
    return out


def add_derived_features(frame: pd.DataFrame) -> None:
    frame["fgo_mse_ratio"] = frame["fgo_candidate_mse_pr"] / frame["baseline_candidate_mse_pr"]
    frame["raw_mse_ratio"] = frame["raw_wls_candidate_mse_pr"] / frame["baseline_candidate_mse_pr"]
    frame["fgo_raw_mse_ratio"] = frame["fgo_candidate_mse_pr"] / frame["raw_wls_candidate_mse_pr"]
    frame["fgo_wins_baseline"] = frame["fgo_score_m"] < frame["baseline_score_m"]
    frame["fgo_gain_score_m"] = frame["baseline_score_m"] - frame["fgo_score_m"]


def candidate_thresholds(values: pd.Series, *, max_cuts: int) -> list[float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = np.unique(finite[np.isfinite(finite)])
    if finite.size <= 1:
        return []
    mids = (finite[:-1] + finite[1:]) * 0.5
    if mids.size <= max_cuts:
        return [float(value) for value in mids]
    quantiles = np.linspace(0.0, 1.0, max_cuts)
    return [float(value) for value in np.unique(np.quantile(mids, quantiles))]


def evaluate_conditions(
    frame: pd.DataFrame,
    conditions: tuple[Condition, ...],
    *,
    group_column: str,
) -> RuleResult:
    if not conditions:
        raise ValueError("at least one condition is required")
    selected = np.ones(len(frame), dtype=bool)
    for condition in conditions:
        selected &= condition.mask(frame)

    wins = frame["fgo_wins_baseline"].to_numpy(dtype=bool)
    gains = frame["fgo_gain_score_m"].to_numpy(dtype=np.float64)
    selected_gains = np.where(selected & np.isfinite(gains), gains, 0.0)
    total_gain = float(np.sum(selected_gains))
    loo_gains: list[float] = []
    if group_column in frame:
        groups = frame[group_column].astype(str).to_numpy()
        for group in sorted(set(groups)):
            keep = groups != group
            if np.any(keep):
                loo_gains.append(float(np.sum(selected_gains[keep])))

    return RuleResult(
        conditions=conditions,
        gain_score_m=total_gain,
        selected_chunks=int(np.count_nonzero(selected)),
        true_positive_chunks=int(np.count_nonzero(selected & wins)),
        false_positive_chunks=int(np.count_nonzero(selected & ~wins)),
        false_negative_chunks=int(np.count_nonzero(~selected & wins)),
        loo_min_gain_score_m=float(min(loo_gains)) if loo_gains else None,
        loo_mean_gain_score_m=float(np.mean(loo_gains)) if loo_gains else None,
    )


def search_rules(
    frame: pd.DataFrame,
    *,
    features: tuple[str, ...],
    max_cuts_per_feature: int,
    max_conditions: int,
    group_column: str,
) -> list[RuleResult]:
    conditions_by_feature: dict[str, list[Condition]] = {}
    for feature in features:
        if feature not in frame:
            continue
        cuts = candidate_thresholds(frame[feature], max_cuts=max_cuts_per_feature)
        conditions_by_feature[feature] = [
            Condition(feature, op, threshold)
            for threshold in cuts
            for op in ("<=", ">=")
        ]

    results: list[RuleResult] = []
    for feature, conditions in conditions_by_feature.items():
        for condition in conditions:
            results.append(evaluate_conditions(frame, (condition,), group_column=group_column))

    if max_conditions >= 2:
        feature_names = list(conditions_by_feature)
        for i, first_feature in enumerate(feature_names):
            for second_feature in feature_names[i + 1 :]:
                for first in conditions_by_feature[first_feature]:
                    for second in conditions_by_feature[second_feature]:
                        results.append(evaluate_conditions(frame, (first, second), group_column=group_column))

    results.sort(
        key=lambda result: (
            result.gain_score_m,
            result.loo_min_gain_score_m if result.loo_min_gain_score_m is not None else -float("inf"),
            -result.false_positive_chunks,
            result.true_positive_chunks,
        ),
        reverse=True,
    )
    return results


def dataset_summary(frame: pd.DataFrame) -> dict[str, object]:
    baseline_total = float(np.sum(frame["baseline_score_m"].to_numpy(dtype=np.float64)))
    oracle_total = float(np.sum(np.minimum(frame["baseline_score_m"], frame["fgo_score_m"])))
    return {
        "chunks": int(len(frame)),
        "fgo_win_chunks": int(np.count_nonzero(frame["fgo_wins_baseline"].to_numpy(dtype=bool))),
        "baseline_score_sum_m": baseline_total,
        "oracle_score_sum_m": oracle_total,
        "oracle_gain_score_m": baseline_total - oracle_total,
        "oracle_source_counts": frame["oracle_source"].value_counts(dropna=False).astype(int).to_dict()
        if "oracle_source" in frame
        else {},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk-csv", action="append", required=True, help="chunk_diagnostics.csv path or glob")
    parser.add_argument("--feature", action="append", default=[], help="candidate feature to use; defaults to built-ins")
    parser.add_argument("--max-cuts-per-feature", type=int, default=32)
    parser.add_argument("--max-conditions", type=int, choices=(1, 2), default=2)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--group-column", default="trip_slug")
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    paths = expand_chunk_paths(args.chunk_csv)
    frame = load_chunk_frames(paths)
    features = tuple(args.feature) if args.feature else DEFAULT_FEATURES
    results = search_rules(
        frame,
        features=features,
        max_cuts_per_feature=args.max_cuts_per_feature,
        max_conditions=args.max_conditions,
        group_column=args.group_column,
    )
    payload = {
        "inputs": [str(path) for path in paths],
        "dataset": dataset_summary(frame),
        "top_rules": [result.payload() for result in results[: args.top_k]],
    }
    print(json.dumps(payload, indent=2))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
