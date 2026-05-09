#!/usr/bin/env python3
"""Analyze 3-way GSDC2023 chunk source oracle and guard rules."""

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

from experiments.tune_gsdc2023_fgo_source_proxy import (  # noqa: E402
    Condition,
    candidate_thresholds,
    group_codes_for_frame,
    loo_gain_stats,
)


BASELINE_SOURCE = "baseline"
DEFAULT_SOURCES = ("raw_wls", "fgo")
DEFAULT_FEATURES = (
    "baseline_candidate_mse_pr",
    "raw_wls_candidate_mse_pr",
    "fgo_candidate_mse_pr",
    "raw_wls_mse_ratio",
    "fgo_mse_ratio",
    "fgo_raw_mse_ratio",
    "raw_wls_candidate_quality_score",
    "fgo_candidate_quality_score",
    "raw_wls_candidate_gap_p95_m",
    "fgo_candidate_gap_p95_m",
)


@dataclass(frozen=True)
class SourceRuleResult:
    source: str
    mode: str
    conditions: tuple[Condition, ...]
    gain_score_m: float
    selected_chunks: int
    true_positive_chunks: int
    false_positive_chunks: int
    false_negative_chunks: int
    oracle_hit_chunks: int
    loo_min_gain_score_m: float | None
    loo_mean_gain_score_m: float | None

    def payload(self) -> dict[str, object]:
        return {
            "source": self.source,
            "mode": self.mode,
            "rule": " AND ".join(condition.label() for condition in self.conditions),
            "conditions": [condition.payload() for condition in self.conditions],
            "gain_score_m": float(self.gain_score_m),
            "selected_chunks": int(self.selected_chunks),
            "true_positive_chunks": int(self.true_positive_chunks),
            "false_positive_chunks": int(self.false_positive_chunks),
            "false_negative_chunks": int(self.false_negative_chunks),
            "oracle_hit_chunks": int(self.oracle_hit_chunks),
            "loo_min_gain_score_m": self.loo_min_gain_score_m,
            "loo_mean_gain_score_m": self.loo_mean_gain_score_m,
        }


def expand_chunk_paths(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = sorted(Path(path) for path in glob.glob(pattern))
        paths.extend(matches if matches else [Path(pattern)])
    unique: dict[str, Path] = {}
    for path in paths:
        if path.is_file():
            unique[str(path.resolve())] = path
    return [unique[key] for key in sorted(unique)]


def score_column(source: str) -> str:
    return f"{source}_score_m"


def available_sources(frame: pd.DataFrame, requested: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(source for source in requested if score_column(source) in frame)


def load_chunk_frames(paths: list[Path], sources: tuple[str, ...]) -> pd.DataFrame:
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
    add_derived_features(out, sources=available_sources(out, sources))
    return out


def add_derived_features(frame: pd.DataFrame, *, sources: tuple[str, ...] = DEFAULT_SOURCES) -> None:
    if "baseline_candidate_mse_pr" in frame:
        baseline_mse = frame["baseline_candidate_mse_pr"].replace(0.0, np.nan)
        for source in sources:
            mse_column = f"{source}_candidate_mse_pr"
            if mse_column in frame:
                frame[f"{source}_mse_ratio"] = frame[mse_column] / baseline_mse
        if "raw_wls_candidate_mse_pr" in frame and "fgo_candidate_mse_pr" in frame:
            raw_mse = frame["raw_wls_candidate_mse_pr"].replace(0.0, np.nan)
            frame["fgo_raw_mse_ratio"] = frame["fgo_candidate_mse_pr"] / raw_mse

    baseline_score = frame[score_column(BASELINE_SOURCE)]
    for source in sources:
        if score_column(source) in frame:
            frame[f"{source}_gain_score_m"] = baseline_score - frame[score_column(source)]
            frame[f"{source}_wins_baseline"] = frame[f"{source}_gain_score_m"] > 0.0

    oracle_sources: list[str] = [BASELINE_SOURCE, *sources]
    score_columns = [score_column(source) for source in oracle_sources if score_column(source) in frame]
    oracle_scores = frame[score_columns].to_numpy(dtype=np.float64)
    finite_scores = np.where(np.isfinite(oracle_scores), oracle_scores, np.inf)
    best_indices = np.argmin(finite_scores, axis=1)
    best_values = finite_scores[np.arange(len(frame)), best_indices]
    column_sources = [column[: -len("_score_m")] for column in score_columns]
    frame["oracle_source_3way"] = [
        column_sources[index] if np.isfinite(best_values[row]) else ""
        for row, index in enumerate(best_indices)
    ]
    frame["oracle_score_3way_m"] = np.where(np.isfinite(best_values), best_values, np.nan)
    frame["oracle_gain_3way_m"] = baseline_score - frame["oracle_score_3way_m"]


def condition_mask(frame: pd.DataFrame, conditions: tuple[Condition, ...]) -> np.ndarray:
    if not conditions:
        raise ValueError("at least one condition is required")
    selected = np.ones(len(frame), dtype=bool)
    for condition in conditions:
        selected &= condition.mask(frame)
    return selected


def evaluate_source_rule(
    frame: pd.DataFrame,
    source: str,
    conditions: tuple[Condition, ...],
    *,
    mode: str,
    group_column: str,
    group_codes: np.ndarray | None = None,
    group_count: int | None = None,
) -> SourceRuleResult:
    selected = condition_mask(frame, conditions)
    gains = frame[f"{source}_gain_score_m"].to_numpy(dtype=np.float64)
    wins = frame[f"{source}_wins_baseline"].to_numpy(dtype=bool)
    if mode == "allow":
        selected_gains = np.where(selected & np.isfinite(gains), gains, 0.0)
        positives = wins
    elif mode == "guard":
        selected_gains = np.where(selected & np.isfinite(gains), -gains, 0.0)
        positives = ~wins
    else:
        raise ValueError(f"unsupported mode: {mode}")

    total_gain = float(np.sum(selected_gains))
    if group_codes is None and group_count is None:
        group_codes, group_count = group_codes_for_frame(frame, group_column)
    loo_min, loo_mean = loo_gain_stats(
        selected_gains,
        total_gain,
        group_codes=group_codes,
        group_count=group_count,
    )
    oracle_hits = frame["oracle_source_3way"].astype(str).to_numpy() == source
    return SourceRuleResult(
        source=source,
        mode=mode,
        conditions=conditions,
        gain_score_m=total_gain,
        selected_chunks=int(np.count_nonzero(selected)),
        true_positive_chunks=int(np.count_nonzero(selected & positives)),
        false_positive_chunks=int(np.count_nonzero(selected & ~positives)),
        false_negative_chunks=int(np.count_nonzero(~selected & positives)),
        oracle_hit_chunks=int(np.count_nonzero(selected & oracle_hits)),
        loo_min_gain_score_m=loo_min,
        loo_mean_gain_score_m=loo_mean,
    )


def build_conditions_by_feature(
    frame: pd.DataFrame,
    *,
    features: tuple[str, ...],
    max_cuts_per_feature: int,
) -> dict[str, list[Condition]]:
    out: dict[str, list[Condition]] = {}
    for feature in features:
        if feature not in frame:
            continue
        cuts = candidate_thresholds(frame[feature], max_cuts=max_cuts_per_feature)
        out[feature] = [Condition(feature, op, threshold) for threshold in cuts for op in ("<=", ">=")]
    return out


def iter_condition_sets(
    conditions_by_feature: dict[str, list[Condition]],
    *,
    max_conditions: int,
):
    for conditions in conditions_by_feature.values():
        for condition in conditions:
            yield (condition,)
    if max_conditions < 2:
        return
    feature_names = list(conditions_by_feature)
    for i, first_feature in enumerate(feature_names):
        for second_feature in feature_names[i + 1 :]:
            for first in conditions_by_feature[first_feature]:
                for second in conditions_by_feature[second_feature]:
                    yield (first, second)


def search_source_rules(
    frame: pd.DataFrame,
    *,
    sources: tuple[str, ...],
    features: tuple[str, ...],
    max_cuts_per_feature: int,
    max_conditions: int,
    group_column: str,
    mode: str,
) -> dict[str, list[SourceRuleResult]]:
    conditions_by_feature = build_conditions_by_feature(
        frame,
        features=features,
        max_cuts_per_feature=max_cuts_per_feature,
    )
    group_codes, group_count = group_codes_for_frame(frame, group_column)
    out: dict[str, list[SourceRuleResult]] = {}
    for source in sources:
        if f"{source}_gain_score_m" not in frame:
            continue
        results = [
            evaluate_source_rule(
                frame,
                source,
                conditions,
                mode=mode,
                group_column=group_column,
                group_codes=group_codes,
                group_count=group_count,
            )
            for conditions in iter_condition_sets(conditions_by_feature, max_conditions=max_conditions)
        ]
        results.sort(
            key=lambda result: (
                result.gain_score_m,
                result.loo_min_gain_score_m if result.loo_min_gain_score_m is not None else -float("inf"),
                -result.false_positive_chunks,
                result.true_positive_chunks,
            ),
            reverse=True,
        )
        out[source] = results
    return out


def filter_results(
    results: list[SourceRuleResult],
    *,
    min_selected_chunks: int,
    max_false_positive_chunks: int | None,
) -> list[SourceRuleResult]:
    return [
        result
        for result in results
        if result.selected_chunks >= min_selected_chunks
        and (
            max_false_positive_chunks is None
            or result.false_positive_chunks <= max_false_positive_chunks
        )
    ]


def dataset_summary(frame: pd.DataFrame, sources: tuple[str, ...]) -> dict[str, object]:
    source_names = (BASELINE_SOURCE, *sources)
    source_score_sums = {
        source: float(np.nansum(frame[score_column(source)].to_numpy(dtype=np.float64)))
        for source in source_names
        if score_column(source) in frame
    }
    baseline_total = source_score_sums[BASELINE_SOURCE]
    source_gain_sums = {
        source: baseline_total - score_sum
        for source, score_sum in source_score_sums.items()
        if source != BASELINE_SOURCE
    }
    selected_summary: dict[str, float] = {}
    if score_column("selected") in frame:
        selected_sum = float(np.nansum(frame[score_column("selected")].to_numpy(dtype=np.float64)))
        selected_summary = {
            "selected_score_sum_m": selected_sum,
            "selected_vs_baseline_gain_m": baseline_total - selected_sum,
        }
    oracle_total = float(np.nansum(frame["oracle_score_3way_m"].to_numpy(dtype=np.float64)))
    return {
        "chunks": int(len(frame)),
        "source_score_sum_m": source_score_sums,
        "source_vs_baseline_gain_m": source_gain_sums,
        **selected_summary,
        "oracle_source_counts_3way": frame["oracle_source_3way"].value_counts(dropna=False).astype(int).to_dict(),
        "oracle_score_sum_3way_m": oracle_total,
        "oracle_vs_baseline_gain_3way_m": baseline_total - oracle_total,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk-csv", action="append", required=True, help="chunk_diagnostics.csv path or glob")
    parser.add_argument("--source", action="append", default=[], help="candidate source to analyze; defaults to raw_wls,fgo")
    parser.add_argument("--feature", action="append", default=[], help="candidate feature to use; defaults to built-ins")
    parser.add_argument("--max-cuts-per-feature", type=int, default=24)
    parser.add_argument("--max-conditions", type=int, choices=(1, 2), default=2)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--group-column", default="trip_slug")
    parser.add_argument("--min-selected-chunks", type=int, default=0)
    parser.add_argument("--max-false-positive-chunks", type=int, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    requested_sources = tuple(args.source) if args.source else DEFAULT_SOURCES
    paths = expand_chunk_paths(args.chunk_csv)
    frame = load_chunk_frames(paths, requested_sources)
    sources = available_sources(frame, requested_sources)
    features = tuple(args.feature) if args.feature else DEFAULT_FEATURES
    allow_results = search_source_rules(
        frame,
        sources=sources,
        features=features,
        max_cuts_per_feature=args.max_cuts_per_feature,
        max_conditions=args.max_conditions,
        group_column=args.group_column,
        mode="allow",
    )
    guard_results = search_source_rules(
        frame,
        sources=sources,
        features=features,
        max_cuts_per_feature=args.max_cuts_per_feature,
        max_conditions=args.max_conditions,
        group_column=args.group_column,
        mode="guard",
    )
    payload = {
        "inputs": [str(path) for path in paths],
        "dataset": dataset_summary(frame, sources),
        "filters": {
            "min_selected_chunks": int(args.min_selected_chunks),
            "max_false_positive_chunks": args.max_false_positive_chunks,
        },
        "top_allow_rules": {
            source: [
                result.payload()
                for result in filter_results(
                    results,
                    min_selected_chunks=args.min_selected_chunks,
                    max_false_positive_chunks=args.max_false_positive_chunks,
                )[: args.top_k]
            ]
            for source, results in allow_results.items()
        },
        "top_guard_rules": {
            source: [
                result.payload()
                for result in filter_results(
                    results,
                    min_selected_chunks=args.min_selected_chunks,
                    max_false_positive_chunks=args.max_false_positive_chunks,
                )[: args.top_k]
            ]
            for source, results in guard_results.items()
        },
    }
    print(json.dumps(payload, indent=2))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
