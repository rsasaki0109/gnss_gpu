"""Summarize Taroz IMU state parity JSON files across trips/devices."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable, Mapping

from experiments.compare_gsdc2023_taroz_imu_state import TAROZ_IMU_STATE_GROUPS


GROUP_METRICS = (
    "finite_rows",
    "component_rms",
    "component_max_abs",
    "mean_norm",
    "max_norm",
)

THRESHOLD_METRICS = (
    "component_rms",
    "component_max_abs",
    "mean_norm",
    "max_norm",
)


def parse_summary_spec(spec: str) -> tuple[str, Path]:
    """Parse ``label=path`` or infer the label from a bare path."""

    if "=" in spec:
        label, path_text = spec.split("=", 1)
        label = label.strip()
        if not label:
            raise ValueError(f"empty label in summary spec: {spec!r}")
        return label, Path(path_text)
    path = Path(spec)
    return path.stem, path


def load_summary(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_threshold_spec(spec: str) -> dict[str, object]:
    if "=" not in spec:
        raise ValueError(f"threshold must be group.metric=value: {spec!r}")
    key, value_text = spec.split("=", 1)
    if "." not in key:
        raise ValueError(f"threshold must be group.metric=value: {spec!r}")
    group_name, metric = (part.strip() for part in key.split(".", 1))
    if group_name not in TAROZ_IMU_STATE_GROUPS:
        raise ValueError(f"unknown threshold group: {group_name}")
    if metric not in THRESHOLD_METRICS:
        raise ValueError(f"unknown threshold metric for {group_name}: {metric}")
    threshold = float(value_text)
    if threshold < 0.0:
        raise ValueError(f"threshold must be non-negative: {spec!r}")
    return {
        "group": group_name,
        "metric": metric,
        "threshold": threshold,
    }


def summary_row(
    label: str,
    path: Path,
    summary: Mapping[str, object],
) -> dict[str, object]:
    delta_stats = summary.get("delta_stats")
    if not isinstance(delta_stats, Mapping):
        raise ValueError(f"{path} is missing delta_stats")
    groups = delta_stats.get("groups")
    if not isinstance(groups, Mapping):
        raise ValueError(f"{path} is missing delta_stats.groups")

    row: dict[str, object] = {
        "label": label,
        "summary_path": str(path),
        "mode": summary.get("mode"),
        "native_imu_state_path": summary.get("native_imu_state_path"),
        "matlab_imu_state_path": summary.get("matlab_imu_state_path"),
        "matched_rows": delta_stats.get("matched_rows"),
    }
    for group_name in TAROZ_IMU_STATE_GROUPS:
        group = groups.get(group_name)
        if not isinstance(group, Mapping):
            for metric in GROUP_METRICS:
                row[f"{group_name}_{metric}"] = None
            continue
        for metric in GROUP_METRICS:
            row[f"{group_name}_{metric}"] = group.get(metric)
    return row


def load_summary_rows(specs: Iterable[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for spec in specs:
        label, path = parse_summary_spec(spec)
        rows.append(summary_row(label, path, load_summary(path)))
    return rows


def _best_numeric(rows: Iterable[Mapping[str, object]], key: str) -> dict[str, object] | None:
    best_label: str | None = None
    best_value: float | None = None
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        numeric = float(value)
        if best_value is None or numeric > best_value:
            best_label = str(row.get("label"))
            best_value = numeric
    if best_value is None:
        return None
    return {"label": best_label, "value": best_value}


def threshold_violations(
    rows: Iterable[Mapping[str, object]],
    thresholds: Iterable[Mapping[str, object]],
) -> list[dict[str, object]]:
    violations: list[dict[str, object]] = []
    for threshold in thresholds:
        group_name = str(threshold["group"])
        metric = str(threshold["metric"])
        limit = float(threshold["threshold"])
        key = f"{group_name}_{metric}"
        for row in rows:
            value = row.get(key)
            if value is None:
                violations.append(
                    {
                        "label": row.get("label"),
                        "summary_path": row.get("summary_path"),
                        "group": group_name,
                        "metric": metric,
                        "value": None,
                        "threshold": limit,
                        "reason": "missing",
                    }
                )
                continue
            numeric = float(value)
            if numeric > limit:
                violations.append(
                    {
                        "label": row.get("label"),
                        "summary_path": row.get("summary_path"),
                        "group": group_name,
                        "metric": metric,
                        "value": numeric,
                        "threshold": limit,
                        "reason": "exceeded",
                    }
                )
    return violations


def aggregate_summary_rows(
    rows: list[dict[str, object]],
    *,
    thresholds: Iterable[Mapping[str, object]] | None = None,
) -> dict[str, object]:
    threshold_list = list(thresholds or [])
    violations = threshold_violations(rows, threshold_list)
    groups: dict[str, object] = {}
    for group_name in TAROZ_IMU_STATE_GROUPS:
        groups[group_name] = {
            "worst_mean_norm": _best_numeric(rows, f"{group_name}_mean_norm"),
            "worst_max_norm": _best_numeric(rows, f"{group_name}_max_norm"),
            "worst_component_max_abs": _best_numeric(rows, f"{group_name}_component_max_abs"),
        }
    return {
        "summary_count": len(rows),
        "labels": [str(row["label"]) for row in rows],
        "groups": groups,
        "thresholds": threshold_list,
        "threshold_violations": violations,
        "passed": len(violations) == 0,
        "rows": rows,
    }


def csv_fieldnames() -> list[str]:
    fields = [
        "label",
        "summary_path",
        "mode",
        "native_imu_state_path",
        "matlab_imu_state_path",
        "matched_rows",
    ]
    for group_name in TAROZ_IMU_STATE_GROUPS:
        fields.extend(f"{group_name}_{metric}" for metric in GROUP_METRICS)
    return fields


def write_rows_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fieldnames())
        writer.writeheader()
        writer.writerows(rows)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        action="append",
        required=True,
        help="summary JSON path, optionally as label=path; may be supplied more than once",
    )
    parser.add_argument(
        "--threshold",
        action="append",
        default=[],
        help="gate threshold as group.metric=value, e.g. position_m.max_norm=0.002; may be supplied more than once",
    )
    parser.add_argument("--output-csv", type=Path, default=None, help="write flattened per-summary CSV")
    parser.add_argument("--output-json", type=Path, default=None, help="write aggregate JSON")
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    rows = load_summary_rows(args.summary)
    thresholds = [parse_threshold_spec(spec) for spec in getattr(args, "threshold", [])]
    aggregate = aggregate_summary_rows(rows, thresholds=thresholds)
    if args.output_csv is not None:
        write_rows_csv(args.output_csv, rows)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return aggregate


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    aggregate = run(args)
    print(json.dumps(aggregate, indent=2, sort_keys=True))
    violations = aggregate.get("threshold_violations")
    if isinstance(violations, list) and violations:
        raise SystemExit(f"Taroz IMU state parity gate failed: {len(violations)} violation(s)")


if __name__ == "__main__":
    main()
