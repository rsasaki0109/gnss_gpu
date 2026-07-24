#!/usr/bin/env python3
"""Audit-only localization of correct-basin extinction and proposal lineage."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "python"))

from gnss_gpu.io.ppc import PPCDatasetLoader  # noqa: E402


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def _position(row: dict[str, str]) -> np.ndarray:
    return np.asarray(
        [float(row["ecef_x"]), float(row["ecef_y"]), float(row["ecef_z"])],
        dtype=np.float64,
    )


def source_families(value: str) -> tuple[str, ...]:
    families: set[str] = set()
    for token in str(value).split("|"):
        parts = token.split(":")
        if len(parts) >= 2:
            families.add(parts[1])
        elif token:
            families.add(token)
    return tuple(sorted(families))


def analyze(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_epoch: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in _read_csv(args.basin_trace):
        epoch = int(row["epoch"])
        if int(args.start) <= epoch < int(args.end):
            by_epoch[epoch].append(row)
    diagnostics = {int(row["epoch"]): row for row in _read_csv(args.epoch_diagnostics)}
    data = PPCDatasetLoader(args.data_dir).load_experiment_data(
        max_epochs=int(args.end), systems=("G", "R", "E", "C", "J")
    )
    output: list[dict[str, Any]] = []
    correct_sources: Counter[str] = Counter()
    for epoch in range(int(args.start), int(args.end)):
        rows = by_epoch.get(epoch, [])
        truth = np.asarray(data["ground_truth"][epoch], dtype=np.float64)
        if rows:
            errors = np.asarray(
                [np.linalg.norm(_position(row) - truth) for row in rows], dtype=np.float64
            )
            oracle = rows[int(np.argmin(errors))]
            oracle_error = float(np.min(errors))
            families = source_families(oracle.get("proposal_sources", ""))
            if oracle_error < 0.5:
                correct_sources.update(families)
        else:
            oracle = {}
            oracle_error = float("inf")
            families = ()
        diagnostic = diagnostics[epoch]
        output.append(
            {
                "epoch": epoch,
                "oracle_error_m": oracle_error,
                "oracle_sub50cm": int(oracle_error < 0.5),
                "oracle_assignment_id": oracle.get("assignment_id", ""),
                "oracle_basin_id": oracle.get("basin_id", ""),
                "oracle_source_families": "|".join(families),
                "oracle_proposal_sources": oracle.get("proposal_sources", ""),
                "n_basins": len(rows),
                "ambiguities_reset": diagnostic.get("ambiguities_reset", ""),
                "assignment_history_cleared": diagnostic.get("assignment_history_cleared", ""),
                "assignment_arc_slips": diagnostic.get("assignment_arc_slips", ""),
                "n_candidates_born": diagnostic.get("n_candidates_born", ""),
                "respawn_triggered": diagnostic.get("respawn_triggered", ""),
                "n_respawn_candidates_born": diagnostic.get("n_respawn_candidates_born", ""),
                "n_respawn_position_seeds": diagnostic.get("n_respawn_position_seeds", ""),
                "n_respawn_history_seeds": diagnostic.get("n_respawn_history_seeds", ""),
                "n_respawn_assignment_candidates": diagnostic.get("n_respawn_assignment_candidates", ""),
                "ddpr_snapshot_accepted": diagnostic.get("ddpr_snapshot_accepted", ""),
                "ddpr_snapshot_error_m": diagnostic.get("ddpr_snapshot_error_m", ""),
                "trusted_fix_anchor_error_m": diagnostic.get("trusted_fix_anchor_error_m", ""),
            }
        )
    correct_epochs = [row["epoch"] for row in output if row["oracle_sub50cm"]]
    summary = {
        "segment": [int(args.start), int(args.end)],
        "oracle_sub50cm_epochs": len(correct_epochs),
        "first_oracle_sub50cm_epoch": min(correct_epochs) if correct_epochs else None,
        "last_oracle_sub50cm_epoch": max(correct_epochs) if correct_epochs else None,
        "correct_oracle_source_family_epochs": dict(sorted(correct_sources.items())),
        "reset_epochs": [row["epoch"] for row in output if str(row["ambiguities_reset"]) == "1"],
        "history_clear_epochs": [row["epoch"] for row in output if str(row["assignment_history_cleared"]) == "1"],
    }
    return summary, output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("basin_trace", type=Path)
    parser.add_argument("--epoch-diagnostics", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
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
