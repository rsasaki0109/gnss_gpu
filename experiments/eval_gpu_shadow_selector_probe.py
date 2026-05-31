#!/usr/bin/env python3
"""Probe whether GPU urban-shadow features change candidate selection.

The evaluator accepts a candidate-level selector CSV with `run_id`, `tow`, and
`label`.  If the input has only one row per epoch, it can synthesize a small
two-candidate what-if pool:

* `nominal_gnss`: no fixed overhead, sensitive to shadow risk.
* `shadow_robust`: small fixed overhead, less sensitive to shadow risk.

This makes the Phase 7/8 PPC nav-shadow artifact immediately testable while
keeping the scoring path generic for real multi-candidate selector outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


REPO = Path("/media/sasaki/aiueo/ai_coding_ws/gnss_gpu")
RESULTS_DIR = REPO / "experiments/results"
DEFAULT_FEATURE_CSV = RESULTS_DIR / "ppc_gpu_urban_shadow_selector_features_tokyo_run1_nav_smoke.csv"
DEFAULT_OUT_DIR = RESULTS_DIR / "gpu_shadow_selector_probe"

DEFAULT_RISK_COL = "gpu_urban_shadow_risk_score"
DEFAULT_BASE_SCORE_COL = "selector_base_score"
PROXY_TRUTH_COST_COL = "proxy_truth_cost"

RISK_BUCKETS = (
    (0.00, 0.02, "0.00-0.02"),
    (0.02, 0.05, "0.02-0.05"),
    (0.05, 0.10, "0.05-0.10"),
    (0.10, 0.20, "0.10-0.20"),
    (0.20, 1.01, "0.20+"),
)


def _to_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _finite_float(value, default: float = 0.0) -> float:
    out = _to_float(value, default)
    return out if math.isfinite(out) else float(default)


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _read_candidate_csv(
    path: Path,
    *,
    candidate_run_id: str = "",
    allowed_keys: set[tuple[str, float]] | None = None,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if candidate_run_id and str(row.get("run_id", "")) != candidate_run_id:
                continue
            if allowed_keys is not None and _epoch_key(row) not in allowed_keys:
                continue
            rows.append(row)
    return rows


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else ["run_id", "tow"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _epoch_key(row: dict) -> tuple[str, float]:
    return (str(row.get("run_id", "")), round(_to_float(row.get("tow")), 1))


def _epoch_key_with_run(row: dict, run_id: str | None = None) -> tuple[str, float]:
    return (str(row.get("run_id", "") if run_id is None else run_id), round(_to_float(row.get("tow")), 1))


def _has_multi_candidate_epochs(rows: list[dict]) -> bool:
    counts = Counter(_epoch_key(row) for row in rows)
    return any(count > 1 for count in counts.values())


def _percentile(values: list[float], q: float) -> float:
    finite = sorted(v for v in values if v == v)
    if not finite:
        return 0.0
    if len(finite) == 1:
        return finite[0]
    pos = (len(finite) - 1) * q / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(finite) - 1)
    frac = pos - lo
    return finite[lo] * (1.0 - frac) + finite[hi] * frac


def _risk_bucket(risk: float) -> str:
    for lo, hi, label in RISK_BUCKETS:
        if lo <= risk < hi:
            return label
    return "1.00+"


def merge_epoch_gpu_features(
    candidate_rows: list[dict],
    feature_rows: list[dict],
    *,
    feature_source_run_id: str = "",
    target_run_id: str = "",
    keep_only_matched: bool = False,
) -> tuple[list[dict], int, int]:
    """Join `gpu_*` epoch features onto candidate rows by `(run_id, tow)`.

    `feature_source_run_id` filters the feature file when it uses a different
    run name from the candidate file. `target_run_id` rewrites those feature
    keys for the join, e.g. `tokyo_run1_nav -> tokyo_run1`.
    """

    by_key: dict[tuple[str, float], dict] = {}
    for row in feature_rows:
        source_run = str(row.get("run_id", ""))
        if feature_source_run_id and source_run != feature_source_run_id:
            continue
        join_run = target_run_id or source_run
        key = _epoch_key_with_run(row, join_run)
        by_key[key] = {k: v for k, v in row.items() if k.startswith("gpu_")}

    merged: list[dict] = []
    matched = 0
    missing = 0
    for row in candidate_rows:
        features = by_key.get(_epoch_key(row))
        if features is None:
            missing += 1
            if keep_only_matched:
                continue
            merged.append(dict(row))
            continue
        out = dict(row)
        out.update(features)
        merged.append(out)
        matched += 1
    return merged, matched, missing


def feature_join_keys(
    feature_rows: list[dict],
    *,
    feature_source_run_id: str = "",
    target_run_id: str = "",
) -> set[tuple[str, float]]:
    keys: set[tuple[str, float]] = set()
    for row in feature_rows:
        source_run = str(row.get("run_id", ""))
        if feature_source_run_id and source_run != feature_source_run_id:
            continue
        keys.add(_epoch_key_with_run(row, target_run_id or source_run))
    return keys


def derive_shadow_coefficients(rows: list[dict]) -> list[dict]:
    """Derive generic shadow penalty/rescue coefficients from selector features."""

    out: list[dict] = []
    for row in rows:
        n_candidates = max(_finite_float(row.get("n_candidates_in_epoch"), 1.0), 1.0)
        cluster_size = max(_finite_float(row.get("cluster_size_50cm"), 1.0), 0.0)
        cluster_frac = _clip01(cluster_size / n_candidates)
        in_max = _clip01(_finite_float(row.get("is_in_max_cluster_50cm"), 0.0))
        rank = max(_finite_float(row.get("rank_by_rms"), 1.0), 1.0)
        dist_to_median = max(_finite_float(row.get("dist_to_median_m"), 0.0), 0.0)
        jump = max(_finite_float(row.get("candidate_jump_m"), 0.0), 0.0)
        vertical = max(_finite_float(row.get("delta_pos_vertical_m"), 0.0), 0.0)
        abs_max = max(_finite_float(row.get("abs_max"), 0.0), 0.0)
        sats = max(_finite_float(row.get("sats"), 0.0), 0.0)

        rank_penalty = _clip01((rank - 1.0) / 12.0)
        median_penalty = _clip01(dist_to_median / 3.0)
        jump_penalty = _clip01(jump / 8.0)
        vertical_penalty = _clip01(vertical / 3.0)
        residual_penalty = _clip01(abs_max / 8.0)
        sat_penalty = _clip01((10.0 - sats) / 8.0)

        penalty = (
            0.28 * (1.0 - cluster_frac)
            + 0.20 * (1.0 - in_max)
            + 0.18 * rank_penalty
            + 0.14 * median_penalty
            + 0.08 * jump_penalty
            + 0.06 * vertical_penalty
            + 0.04 * residual_penalty
            + 0.02 * sat_penalty
        )
        rescue = (
            0.48 * cluster_frac
            + 0.32 * in_max
            + 0.12 * (1.0 - rank_penalty)
            + 0.08 * (1.0 - median_penalty)
        )

        item = dict(row)
        item.setdefault("gpu_shadow_penalty_coeff", penalty)
        item.setdefault("gpu_shadow_rescue_coeff", rescue)
        out.append(item)
    return out


def synthesize_shadow_candidate_rows(
    epoch_rows: list[dict],
    *,
    risk_col: str = DEFAULT_RISK_COL,
    nominal_label: str = "nominal_gnss",
    robust_label: str = "shadow_robust",
    robust_base_penalty: float = 0.06,
    robust_score_gain: float = 0.80,
    robust_truth_overhead: float = 0.04,
    robust_truth_risk_scale: float = 0.25,
) -> list[dict]:
    """Create a two-candidate selector pool from epoch-level GPU features."""

    out: list[dict] = []
    for row in epoch_rows:
        risk = _to_float(row.get(risk_col))
        common = dict(row)
        common["gpu_shadow_probe_synthesized"] = 1.0

        nominal = dict(common)
        nominal["label"] = nominal_label
        nominal[DEFAULT_BASE_SCORE_COL] = 0.0
        nominal["gpu_shadow_penalty_coeff"] = 1.0
        nominal["gpu_shadow_rescue_coeff"] = 0.0
        nominal[PROXY_TRUTH_COST_COL] = risk
        out.append(nominal)

        robust = dict(common)
        robust["label"] = robust_label
        robust[DEFAULT_BASE_SCORE_COL] = float(robust_base_penalty)
        robust["gpu_shadow_penalty_coeff"] = 0.0
        robust["gpu_shadow_rescue_coeff"] = float(robust_score_gain)
        robust[PROXY_TRUTH_COST_COL] = robust_truth_overhead + robust_truth_risk_scale * risk
        out.append(robust)
    return out


def _candidate_scores(
    row: dict,
    *,
    risk_col: str,
    base_score_col: str,
    penalty_weight: float,
    rescue_weight: float,
) -> tuple[float, float, float]:
    risk = _to_float(row.get(risk_col))
    base_score = _to_float(row.get(base_score_col))
    penalty = _to_float(row.get("gpu_shadow_penalty_coeff"), 0.0)
    rescue = _to_float(row.get("gpu_shadow_rescue_coeff"), 0.0)
    gpu_score = base_score + penalty_weight * risk * penalty - rescue_weight * risk * rescue
    return base_score, gpu_score, risk


def _select_min(scored: list[tuple[float, int, dict]]) -> dict:
    return min(scored, key=lambda item: (item[0], item[1]))[2]


def evaluate_rows(
    rows: list[dict],
    *,
    risk_col: str = DEFAULT_RISK_COL,
    base_score_col: str = DEFAULT_BASE_SCORE_COL,
    truth_cost_col: str = "",
    penalty_weight: float = 1.0,
    rescue_weight: float = 1.0,
) -> tuple[list[dict], dict, list[dict]]:
    grouped: dict[tuple[str, float], list[tuple[int, dict]]] = defaultdict(list)
    for idx, row in enumerate(rows):
        grouped[_epoch_key(row)].append((idx, row))

    selection_rows: list[dict] = []
    bucket_stats: dict[str, dict[str, float]] = {}

    has_truth = bool(truth_cost_col)
    for (run_id, tow), items in sorted(grouped.items(), key=lambda item: item[0]):
        baseline_scored = []
        gpu_scored = []
        for idx, row in items:
            base_score, gpu_score, _risk = _candidate_scores(
                row,
                risk_col=risk_col,
                base_score_col=base_score_col,
                penalty_weight=penalty_weight,
                rescue_weight=rescue_weight,
            )
            baseline_scored.append((base_score, idx, row))
            gpu_scored.append((gpu_score, idx, row))

        baseline = _select_min(baseline_scored)
        gpu = _select_min(gpu_scored)
        _base_score, baseline_gpu_score, baseline_risk = _candidate_scores(
            baseline,
            risk_col=risk_col,
            base_score_col=base_score_col,
            penalty_weight=penalty_weight,
            rescue_weight=rescue_weight,
        )
        _gpu_base_score, gpu_gpu_score, gpu_risk = _candidate_scores(
            gpu,
            risk_col=risk_col,
            base_score_col=base_score_col,
            penalty_weight=penalty_weight,
            rescue_weight=rescue_weight,
        )
        risk = max(baseline_risk, gpu_risk)
        bucket = _risk_bucket(risk)
        changed = str(baseline.get("label", "")) != str(gpu.get("label", ""))

        row_out = {
            "run_id": run_id,
            "tow": tow,
            "risk_bucket": bucket,
            "gpu_urban_shadow_risk_score": risk,
            "baseline_label": baseline.get("label", ""),
            "gpu_label": gpu.get("label", ""),
            "selection_changed": float(changed),
            "baseline_selector_score": _to_float(baseline.get(base_score_col)),
            "baseline_gpu_score": baseline_gpu_score,
            "gpu_selector_score": _to_float(gpu.get(base_score_col)),
            "gpu_gpu_score": gpu_gpu_score,
            "n_candidates": len(items),
        }
        if has_truth:
            truth_values = [_to_float(item[1].get(truth_cost_col), float("nan")) for item in items]
            oracle_truth = min(truth_values) if truth_values else float("nan")
            baseline_truth = _to_float(baseline.get(truth_cost_col), float("nan"))
            gpu_truth = _to_float(gpu.get(truth_cost_col), float("nan"))
            row_out.update(
                {
                    "baseline_truth_cost": baseline_truth,
                    "gpu_truth_cost": gpu_truth,
                    "oracle_truth_cost": oracle_truth,
                    "gpu_minus_baseline_truth_cost": gpu_truth - baseline_truth,
                    "baseline_regret": baseline_truth - oracle_truth,
                    "gpu_regret": gpu_truth - oracle_truth,
                }
            )
        selection_rows.append(row_out)

        stats = bucket_stats.setdefault(
            bucket,
            {"risk_bucket": bucket, "epochs": 0.0, "changed": 0.0, "risk_sum": 0.0},
        )
        stats["epochs"] += 1.0
        stats["changed"] += float(changed)
        stats["risk_sum"] += risk

    summary = summarize_selection(selection_rows, truth_cost_col=truth_cost_col)
    bucket_rows = []
    for bucket in [label for _lo, _hi, label in RISK_BUCKETS] + ["1.00+"]:
        stats = bucket_stats.get(bucket)
        if stats is None:
            continue
        epochs = max(stats["epochs"], 1.0)
        bucket_rows.append(
            {
                "risk_bucket": bucket,
                "epochs": int(stats["epochs"]),
                "changed": int(stats["changed"]),
                "change_rate": stats["changed"] / epochs,
                "mean_risk": stats["risk_sum"] / epochs,
            }
        )
    return selection_rows, summary, bucket_rows


def summarize_selection(selection_rows: list[dict], *, truth_cost_col: str) -> dict:
    n = len(selection_rows)
    changed = [row for row in selection_rows if _to_float(row.get("selection_changed")) > 0.5]
    risks = [_to_float(row.get("gpu_urban_shadow_risk_score")) for row in selection_rows]
    changed_risks = [_to_float(row.get("gpu_urban_shadow_risk_score")) for row in changed]
    baseline_counts = Counter(str(row.get("baseline_label", "")) for row in selection_rows)
    gpu_counts = Counter(str(row.get("gpu_label", "")) for row in selection_rows)
    summary: dict[str, object] = {
        "epochs": n,
        "changed_epochs": len(changed),
        "change_rate": len(changed) / n if n else 0.0,
        "mean_risk": sum(risks) / len(risks) if risks else 0.0,
        "p50_risk": _percentile(risks, 50.0),
        "p95_risk": _percentile(risks, 95.0),
        "mean_changed_risk": sum(changed_risks) / len(changed_risks) if changed_risks else 0.0,
        "baseline_label_counts": dict(sorted(baseline_counts.items())),
        "gpu_label_counts": dict(sorted(gpu_counts.items())),
        "truth_cost_col": truth_cost_col,
    }
    if truth_cost_col:
        baseline_truth = [_to_float(row.get("baseline_truth_cost"), float("nan")) for row in selection_rows]
        gpu_truth = [_to_float(row.get("gpu_truth_cost"), float("nan")) for row in selection_rows]
        delta = [_to_float(row.get("gpu_minus_baseline_truth_cost"), float("nan")) for row in selection_rows]
        summary.update(
            {
                "mean_baseline_truth_cost": sum(baseline_truth) / len(baseline_truth) if baseline_truth else 0.0,
                "mean_gpu_truth_cost": sum(gpu_truth) / len(gpu_truth) if gpu_truth else 0.0,
                "mean_gpu_minus_baseline_truth_cost": sum(delta) / len(delta) if delta else 0.0,
                "improved_epochs": sum(1 for value in delta if value < -1e-12),
                "worse_epochs": sum(1 for value in delta if value > 1e-12),
                "equal_epochs": sum(1 for value in delta if abs(value) <= 1e-12),
                "mean_baseline_regret": sum(_to_float(row.get("baseline_regret")) for row in selection_rows) / n
                if n
                else 0.0,
                "mean_gpu_regret": sum(_to_float(row.get("gpu_regret")) for row in selection_rows) / n
                if n
                else 0.0,
            }
        )
    return summary


def load_or_synthesize_rows(
    input_csv: Path,
    *,
    input_mode: str,
    risk_col: str,
    robust_base_penalty: float,
    robust_score_gain: float,
) -> tuple[list[dict], bool]:
    rows = _read_csv(input_csv)
    return prepare_probe_rows(
        rows,
        input_mode=input_mode,
        risk_col=risk_col,
        robust_base_penalty=robust_base_penalty,
        robust_score_gain=robust_score_gain,
    )


def prepare_probe_rows(
    rows: list[dict],
    *,
    input_mode: str,
    risk_col: str,
    robust_base_penalty: float,
    robust_score_gain: float,
) -> tuple[list[dict], bool]:
    should_synthesize = input_mode == "epoch" or (
        input_mode == "auto" and not _has_multi_candidate_epochs(rows)
    )
    if not should_synthesize:
        return rows, False
    return (
        synthesize_shadow_candidate_rows(
            rows,
            risk_col=risk_col,
            robust_base_penalty=robust_base_penalty,
            robust_score_gain=robust_score_gain,
        ),
        True,
    )


def run_probe(args: argparse.Namespace) -> dict:
    candidate_run_id = getattr(args, "candidate_run_id", "")
    gpu_feature_csv = getattr(args, "gpu_feature_csv", None)
    feature_source_run_id = getattr(args, "feature_source_run_id", "")
    feature_target_run_id = getattr(args, "feature_target_run_id", "")
    keep_only_feature_epochs = getattr(args, "keep_only_feature_epochs", False)
    derive_shadow_coeffs = getattr(args, "derive_shadow_coeffs", False)

    feature_matched = 0
    feature_missing = 0
    feature_rows: list[dict[str, str]] = []
    allowed_keys = None
    if gpu_feature_csv is not None:
        feature_rows = _read_csv(gpu_feature_csv)
        if keep_only_feature_epochs:
            allowed_keys = feature_join_keys(
                feature_rows,
                feature_source_run_id=feature_source_run_id,
                target_run_id=feature_target_run_id or candidate_run_id,
            )

    source_rows = _read_candidate_csv(
        args.input_csv,
        candidate_run_id=candidate_run_id,
        allowed_keys=allowed_keys,
    )

    if gpu_feature_csv is not None:
        source_rows, feature_matched, feature_missing = merge_epoch_gpu_features(
            source_rows,
            feature_rows,
            feature_source_run_id=feature_source_run_id,
            target_run_id=feature_target_run_id or candidate_run_id,
            keep_only_matched=keep_only_feature_epochs,
        )

    if derive_shadow_coeffs:
        source_rows = derive_shadow_coefficients(source_rows)

    rows, synthesized = prepare_probe_rows(
        source_rows,
        input_mode=args.input_mode,
        risk_col=args.risk_col,
        robust_base_penalty=args.robust_base_penalty,
        robust_score_gain=args.robust_score_gain,
    )
    truth_cost_col = args.truth_cost_col
    if synthesized and not truth_cost_col:
        truth_cost_col = PROXY_TRUTH_COST_COL
    selection_rows, summary, bucket_rows = evaluate_rows(
        rows,
        risk_col=args.risk_col,
        base_score_col=args.base_score_col,
        truth_cost_col=truth_cost_col,
        penalty_weight=args.penalty_weight,
        rescue_weight=args.rescue_weight,
    )
    summary = {
        **summary,
        "input_csv": str(args.input_csv),
        "gpu_feature_csv": str(gpu_feature_csv) if gpu_feature_csv is not None else "",
        "synthesized_candidates": synthesized,
        "source_rows": len(source_rows),
        "candidate_rows": len(rows),
        "feature_matched_rows": feature_matched,
        "feature_missing_rows": feature_missing,
        "risk_col": args.risk_col,
        "base_score_col": args.base_score_col,
        "candidate_run_id": candidate_run_id,
        "feature_source_run_id": feature_source_run_id,
        "feature_target_run_id": feature_target_run_id,
        "keep_only_feature_epochs": keep_only_feature_epochs,
        "derive_shadow_coeffs": derive_shadow_coeffs,
        "penalty_weight": args.penalty_weight,
        "rescue_weight": args.rescue_weight,
        "robust_base_penalty": args.robust_base_penalty,
        "robust_score_gain": args.robust_score_gain,
    }

    rows_csv = args.out_dir / "gpu_shadow_selector_probe_rows.csv"
    bucket_csv = args.out_dir / "gpu_shadow_selector_probe_by_risk_bucket.csv"
    summary_json = args.out_dir / "gpu_shadow_selector_probe_summary.json"
    _write_csv(rows_csv, selection_rows)
    _write_csv(bucket_csv, bucket_rows)
    _write_json(summary_json, summary)
    return {
        "summary": summary,
        "rows_csv": str(rows_csv),
        "bucket_csv": str(bucket_csv),
        "summary_json": str(summary_json),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_FEATURE_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--input-mode", choices=("auto", "candidates", "epoch"), default="auto")
    parser.add_argument("--candidate-run-id", default="")
    parser.add_argument("--gpu-feature-csv", type=Path, default=None)
    parser.add_argument("--feature-source-run-id", default="")
    parser.add_argument("--feature-target-run-id", default="")
    parser.add_argument("--keep-only-feature-epochs", action="store_true")
    parser.add_argument("--derive-shadow-coeffs", action="store_true")
    parser.add_argument("--risk-col", default=DEFAULT_RISK_COL)
    parser.add_argument("--base-score-col", default=DEFAULT_BASE_SCORE_COL)
    parser.add_argument("--truth-cost-col", default="")
    parser.add_argument("--penalty-weight", type=float, default=1.0)
    parser.add_argument("--rescue-weight", type=float, default=1.0)
    parser.add_argument("--robust-base-penalty", type=float, default=0.06)
    parser.add_argument("--robust-score-gain", type=float, default=0.80)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    result = run_probe(args)
    summary = result["summary"]
    print(
        "[gpu-shadow-selector-probe] "
        f"epochs={summary['epochs']} changed={summary['changed_epochs']} "
        f"change_rate={summary['change_rate']:.3f} "
        f"mean_delta={summary.get('mean_gpu_minus_baseline_truth_cost', 0.0):.4f}"
    )
    print(f"[gpu-shadow-selector-probe] wrote {result['summary_json']}")
    print(f"[gpu-shadow-selector-probe] wrote {result['rows_csv']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
