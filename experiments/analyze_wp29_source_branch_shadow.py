#!/usr/bin/env python3
"""Audit one proposal-source branch in a saved PF basin trace."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "python"))

from gnss_gpu.io.ppc import PPCDatasetLoader  # noqa: E402


def source_tokens(value: str) -> tuple[str, ...]:
    return tuple(token for token in str(value).split("|") if token)


def branch_rows(
    rows: list[dict[str, str]], source_pattern: re.Pattern[str]
) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if any(source_pattern.fullmatch(token) for token in source_tokens(row["proposal_sources"]))
    ]


def _position(row: dict[str, str]) -> np.ndarray:
    return np.asarray(
        [float(row["ecef_x"]), float(row["ecef_y"]), float(row["ecef_z"])],
        dtype=np.float64,
    )


def analyze(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_epoch: dict[int, list[dict[str, str]]] = defaultdict(list)
    with args.basin_trace.open(newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            by_epoch[int(row["epoch"])].append(row)
    n_epochs = max(by_epoch) + 1
    data = PPCDatasetLoader(args.data_dir).load_experiment_data(
        max_epochs=n_epochs,
        include_sat_velocity=False,
        systems=("G", "R", "E", "C", "J"),
    )
    pattern = re.compile(args.source_regex)
    diagnostics: dict[int, dict[str, str]] = {}
    if args.epoch_diagnostics is not None:
        with args.epoch_diagnostics.open(newline="", encoding="utf-8-sig") as fh:
            diagnostics = {int(row["epoch"]): row for row in csv.DictReader(fh)}
    calibrated_offset: np.ndarray | None = None
    if args.static_anchor_json is not None:
        static_result = json.loads(args.static_anchor_json.read_text(encoding="utf-8"))
        static_position = np.asarray(
            static_result["candidates"][0]["position_ecef"], dtype=np.float64
        )
        segment_start, segment_end = (int(value) for value in static_result["segment"])
        offset_samples: list[np.ndarray] = []
        for epoch in range(segment_start, segment_end):
            row = diagnostics.get(epoch)
            if row is None or row.get("ddpr_snapshot_accepted") != "1":
                continue
            snapshot = np.asarray(
                [
                    float(row["ddpr_snapshot_ecef_x"]),
                    float(row["ddpr_snapshot_ecef_y"]),
                    float(row["ddpr_snapshot_ecef_z"]),
                ]
            )
            if np.all(np.isfinite(snapshot)):
                offset_samples.append(static_position - snapshot)
        if not offset_samples:
            raise RuntimeError("static anchor segment has no accepted snapshot positions")
        calibrated_offset = np.median(np.asarray(offset_samples), axis=0)
    output: list[dict[str, Any]] = []
    offset_selected_positions: dict[int, np.ndarray] = {}
    for epoch in range(n_epochs):
        rows = (
            [
                row
                for row in by_epoch.get(epoch, [])
                if row["assignment_id"] == args.assignment_id
            ]
            if args.assignment_id
            else branch_rows(by_epoch.get(epoch, []), pattern)
        )
        if not rows:
            continue
        truth = np.asarray(data["ground_truth"][epoch], dtype=np.float64)
        selected = max(rows, key=lambda row: float(row["log_weight"]))
        errors = np.asarray(
            [np.linalg.norm(_position(row) - truth) for row in rows], dtype=np.float64
        )
        selected_error = float(np.linalg.norm(_position(selected) - truth))
        snapshot_selected_error = float("nan")
        snapshot_distance = float("nan")
        offset_selected_error = float("nan")
        offset_distance = float("nan")
        diagnostic = diagnostics.get(epoch)
        if diagnostic is not None and diagnostic.get("ddpr_snapshot_accepted") == "1":
            snapshot = np.asarray(
                [
                    float(diagnostic["ddpr_snapshot_ecef_x"]),
                    float(diagnostic["ddpr_snapshot_ecef_y"]),
                    float(diagnostic["ddpr_snapshot_ecef_z"]),
                ],
                dtype=np.float64,
            )
            if np.all(np.isfinite(snapshot)):
                snapshot_selected = min(
                    rows, key=lambda row: float(np.linalg.norm(_position(row) - snapshot))
                )
                snapshot_distance = float(np.linalg.norm(_position(snapshot_selected) - snapshot))
                snapshot_selected_error = float(
                    np.linalg.norm(_position(snapshot_selected) - truth)
                )
                if calibrated_offset is not None:
                    offset_target = snapshot + calibrated_offset
                    offset_selected = min(
                        rows,
                        key=lambda row: float(
                            np.linalg.norm(_position(row) - offset_target)
                        ),
                    )
                    offset_distance = float(
                        np.linalg.norm(_position(offset_selected) - offset_target)
                    )
                    offset_selected_error = float(
                        np.linalg.norm(_position(offset_selected) - truth)
                    )
                    offset_selected_positions[epoch] = _position(offset_selected)
        weights = np.asarray([float(row["log_weight"]) for row in rows])
        weights = np.exp(weights - np.max(weights))
        gamma = float(np.max(weights) / np.sum(weights))
        output.append(
            {
                "epoch": epoch,
                "n_branch_basins": len(rows),
                "branch_gamma": gamma,
                "selected_error_m": selected_error,
                "oracle_error_m": float(np.min(errors)),
                "selected_sub50cm": int(selected_error < 0.5),
                "oracle_sub50cm": int(np.min(errors) < 0.5),
                "snapshot_selected_error_m": snapshot_selected_error,
                "snapshot_candidate_distance_m": snapshot_distance,
                "snapshot_selected_sub50cm": int(snapshot_selected_error < 0.5),
                "offset_selected_error_m": offset_selected_error,
                "offset_candidate_distance_m": offset_distance,
                "offset_selected_sub50cm": int(offset_selected_error < 0.5),
            }
        )
    interpolated_sub50 = 0
    if offset_selected_positions:
        anchor_epochs = sorted(offset_selected_positions)
        for epoch in range(n_epochs):
            index = int(np.searchsorted(anchor_epochs, epoch))
            if index < len(anchor_epochs) and anchor_epochs[index] == epoch:
                selected_position = offset_selected_positions[epoch]
            elif 0 < index < len(anchor_epochs):
                left, right = anchor_epochs[index - 1], anchor_epochs[index]
                alpha = (epoch - left) / float(right - left)
                selected_position = (
                    (1.0 - alpha) * offset_selected_positions[left]
                    + alpha * offset_selected_positions[right]
                )
            else:
                diagnostic = diagnostics.get(epoch)
                if diagnostic is None:
                    continue
                selected_position = np.asarray(
                    [
                        float(diagnostic["ecef_x"]),
                        float(diagnostic["ecef_y"]),
                        float(diagnostic["ecef_z"]),
                    ]
                )
            interpolated_sub50 += int(
                np.linalg.norm(selected_position - data["ground_truth"][epoch]) < 0.5
            )
    summary = {
        "n_epochs_full_denominator": n_epochs,
        "source_regex": args.source_regex,
        "assignment_id": args.assignment_id,
        "branch_available_epochs": len(output),
        "branch_selected_sub50cm_epochs": sum(row["selected_sub50cm"] for row in output),
        "branch_oracle_sub50cm_epochs": sum(row["oracle_sub50cm"] for row in output),
        "branch_selected_sub50cm_full_pct": 100.0
        * sum(row["selected_sub50cm"] for row in output)
        / n_epochs,
        "branch_oracle_sub50cm_full_pct": 100.0
        * sum(row["oracle_sub50cm"] for row in output)
        / n_epochs,
        "snapshot_selected_epochs": int(
            sum(np.isfinite(row["snapshot_selected_error_m"]) for row in output)
        ),
        "snapshot_selected_sub50cm_epochs": int(
            sum(row["snapshot_selected_sub50cm"] for row in output)
        ),
        "calibrated_offset_ecef_m": (
            None if calibrated_offset is None else calibrated_offset.tolist()
        ),
        "offset_selected_epochs": len(offset_selected_positions),
        "offset_selected_sub50cm_epochs": int(
            sum(row["offset_selected_sub50cm"] for row in output)
        ),
        "offset_interpolated_sub50cm_full_epochs": int(interpolated_sub50),
        "offset_interpolated_sub50cm_full_pct": 100.0
        * interpolated_sub50
        / n_epochs,
    }
    return summary, output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("basin_trace", type=Path)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--source-regex", default=r"\d+:1")
    parser.add_argument("--assignment-id", default="")
    parser.add_argument("--epoch-diagnostics", type=Path)
    parser.add_argument("--static-anchor-json", type=Path)
    parser.add_argument("--out-summary", type=Path, required=True)
    parser.add_argument("--out-epochs", type=Path, required=True)
    args = parser.parse_args()
    summary, rows = analyze(args)
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    args.out_epochs.parent.mkdir(parents=True, exist_ok=True)
    with args.out_epochs.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
