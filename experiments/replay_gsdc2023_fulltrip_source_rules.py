#!/usr/bin/env python3
"""Replay GSDC2023 chunk source rules with full-trip percentile scoring."""

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

from experiments.analyze_gsdc2023_source_oracle import add_derived_features  # noqa: E402
from experiments.tune_gsdc2023_fgo_source_proxy import Condition  # noqa: E402


DEFAULT_BASE_SOURCE = "baseline"
RULE_OPERATORS = ("<=", ">=")


@dataclass(frozen=True)
class RuleSpec:
    name: str
    source: str
    conditions: tuple[Condition, ...]

    def label(self) -> str:
        return " AND ".join(condition.label() for condition in self.conditions)

    def payload(self) -> dict[str, object]:
        return {
            "name": self.name,
            "source": self.source,
            "rule": self.label(),
            "conditions": [condition.payload() for condition in self.conditions],
        }


def parse_named_path(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        path = Path(raw)
        return path.name, path
    name, path = raw.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"empty input name in {raw!r}")
    return name, Path(path)


def expand_inputs(values: list[str]) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    for value in values:
        if "=" in value:
            name, raw_path = parse_named_path(value)
            matches = sorted(Path(path) for path in glob.glob(str(raw_path)))
            if matches:
                out.extend((f"{name}_{path.name}", path) for path in matches)
            else:
                out.append((name, raw_path))
            continue
        matches = sorted(Path(path) for path in glob.glob(value))
        if matches:
            out.extend((path.name, path) for path in matches)
        else:
            path = Path(value)
            out.append((path.name, path))

    unique: dict[str, tuple[str, Path]] = {}
    for name, path in out:
        unique[str(path.resolve())] = (name, path)
    return [unique[key] for key in sorted(unique)]


def parse_condition(raw: str) -> Condition:
    for op in RULE_OPERATORS:
        if op in raw:
            feature, threshold = raw.split(op, 1)
            feature = feature.strip()
            if not feature:
                raise ValueError(f"empty condition feature in {raw!r}")
            return Condition(feature, op, float(threshold.strip()))
    raise ValueError(f"condition {raw!r} must contain one of {RULE_OPERATORS}")


def parse_rule_spec(raw: str) -> RuleSpec:
    parts = raw.split(":", 2)
    if len(parts) != 3:
        raise ValueError(
            f"invalid rule {raw!r}; expected name:source:feature<=value,feature>=value",
        )
    name, source, condition_text = (part.strip() for part in parts)
    if not name:
        raise ValueError(f"empty rule name in {raw!r}")
    if not source:
        raise ValueError(f"empty rule source in {raw!r}")
    conditions = tuple(
        parse_condition(part.strip())
        for part in condition_text.split(",")
        if part.strip()
    )
    if not conditions:
        raise ValueError(f"rule {raw!r} has no conditions")
    return RuleSpec(name=name, source=source, conditions=conditions)


def score_errors_m(errors: np.ndarray) -> float:
    values = np.asarray(errors, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(0.5 * (np.percentile(values, 50) + np.percentile(values, 95)))


def source_error_column(source: str) -> str:
    return f"{source}_error_2d_m"


def condition_mask(frame: pd.DataFrame, conditions: tuple[Condition, ...]) -> np.ndarray:
    selected = np.ones(len(frame), dtype=bool)
    for condition in conditions:
        selected &= condition.mask(frame)
    return selected


def load_run_frames(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    epoch_path = run_dir / "epoch_diagnostics.csv"
    chunk_path = run_dir / "chunk_diagnostics.csv"
    if not epoch_path.is_file() or not chunk_path.is_file():
        raise FileNotFoundError(f"{run_dir} must contain epoch_diagnostics.csv and chunk_diagnostics.csv")
    epoch = pd.read_csv(epoch_path)
    chunks = pd.read_csv(chunk_path)
    add_derived_features(chunks)
    return epoch, chunks


def replay_rule(
    *,
    run_name: str,
    epoch: pd.DataFrame,
    chunks: pd.DataFrame,
    rule: RuleSpec,
    base_source: str = DEFAULT_BASE_SOURCE,
) -> dict[str, object]:
    base_column = source_error_column(base_source)
    source_column = source_error_column(rule.source)
    missing = [column for column in (base_column, source_column) if column not in epoch]
    if missing:
        raise ValueError(f"{run_name}: epoch diagnostics missing columns: {missing}")
    for condition in rule.conditions:
        if condition.feature not in chunks:
            raise ValueError(f"{run_name}: chunk diagnostics missing feature {condition.feature!r}")

    base_errors = epoch[base_column].to_numpy(dtype=np.float64)
    source_errors = epoch[source_column].to_numpy(dtype=np.float64)
    selected_errors = base_errors.copy()
    selected = condition_mask(chunks, rule.conditions)
    selected_chunks = chunks[selected].copy()

    ranges: list[str] = []
    n_epoch = len(epoch)
    for row in selected_chunks.itertuples(index=False):
        start = int(row.start_epoch)
        end = int(row.end_epoch)
        if start < 0 or end < start or end > n_epoch:
            raise ValueError(f"{run_name}: invalid chunk range {start}:{end} for {n_epoch} epochs")
        selected_errors[start:end] = source_errors[start:end]
        oracle = str(getattr(row, "oracle_source_3way", getattr(row, "oracle_source", "")))
        source_score = float(getattr(row, f"{rule.source}_score_m"))
        base_score = float(getattr(row, f"{base_source}_score_m"))
        mse = getattr(row, f"{rule.source}_candidate_mse_pr", float("nan"))
        ranges.append(f"{start}-{end}:{oracle}:b{base_score:.3f}->{rule.source}{source_score:.3f}:mse{float(mse):.3f}")

    base_score = score_errors_m(base_errors)
    replay_score = score_errors_m(selected_errors)
    source_gain = selected_chunks[f"{base_source}_score_m"] - selected_chunks[f"{rule.source}_score_m"]
    finite_source_gain = source_gain[np.isfinite(source_gain.to_numpy(dtype=np.float64))]
    true_positive = finite_source_gain > 0.0
    return {
        "run": run_name,
        "rule": rule.name,
        "source": rule.source,
        "base_source": base_source,
        "base_score_m": base_score,
        "replay_score_m": replay_score,
        "gain_score_m": base_score - replay_score,
        "selected_chunks": int(len(selected_chunks)),
        "true_positive_chunks": int(np.count_nonzero(true_positive.to_numpy(dtype=bool))),
        "false_positive_chunks": int(len(selected_chunks) - np.count_nonzero(true_positive.to_numpy(dtype=bool))),
        "selected_epoch_count": int(
            sum(int(row.end_epoch) - int(row.start_epoch) for row in selected_chunks.itertuples(index=False)),
        ),
        "selected_chunk_ranges": "; ".join(ranges),
    }


def replay_inputs(
    inputs: list[tuple[str, Path]],
    rules: list[RuleSpec],
    *,
    base_source: str = DEFAULT_BASE_SOURCE,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for run_name, run_dir in inputs:
        epoch, chunks = load_run_frames(run_dir)
        for rule in rules:
            rows.append(
                replay_rule(
                    run_name=run_name,
                    epoch=epoch,
                    chunks=chunks,
                    rule=rule,
                    base_source=base_source,
                ),
            )
    return pd.DataFrame(rows)


def summarize(frame: pd.DataFrame, rules: list[RuleSpec]) -> dict[str, object]:
    summary: dict[str, object] = {
        "runs": int(frame["run"].nunique()) if not frame.empty else 0,
        "rules": [rule.payload() for rule in rules],
        "rule_summary": {},
    }
    for rule in rules:
        subset = frame[frame["rule"] == rule.name] if not frame.empty else frame
        summary["rule_summary"][rule.name] = {
            "source": rule.source,
            "runs": int(len(subset)),
            "runs_with_selected_chunks": int(np.count_nonzero(subset["selected_chunks"].to_numpy(dtype=np.int64) > 0))
            if not subset.empty
            else 0,
            "selected_chunks": int(subset["selected_chunks"].sum()) if not subset.empty else 0,
            "true_positive_chunks": int(subset["true_positive_chunks"].sum()) if not subset.empty else 0,
            "false_positive_chunks": int(subset["false_positive_chunks"].sum()) if not subset.empty else 0,
            "total_gain_score_m": float(subset["gain_score_m"].sum()) if not subset.empty else 0.0,
            "worst_gain_score_m": float(subset["gain_score_m"].min()) if not subset.empty else 0.0,
            "best_gain_score_m": float(subset["gain_score_m"].max()) if not subset.empty else 0.0,
        }
    return summary


def write_outputs(output_dir: Path, frame: pd.DataFrame, summary: dict[str, object]) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    replay_csv = output_dir / "fulltrip_rule_replay.csv"
    summary_json = output_dir / "summary.json"
    frame.to_csv(replay_csv, index=False)
    payload = {
        **summary,
        "fulltrip_rule_replay_csv": str(replay_csv),
    }
    summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, help="run directory or glob; NAME=path is supported")
    parser.add_argument(
        "--rule",
        action="append",
        required=True,
        help="rule as name:source:feature<=value,feature>=value",
    )
    parser.add_argument("--base-source", default=DEFAULT_BASE_SOURCE)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    inputs = expand_inputs(args.input)
    if not inputs:
        raise ValueError("no input run directories found")
    rules = [parse_rule_spec(raw) for raw in args.rule]
    frame = replay_inputs(inputs, rules, base_source=args.base_source)
    payload = write_outputs(args.output_dir, frame, summarize(frame, rules))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
