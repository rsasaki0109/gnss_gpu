#!/usr/bin/env python3
"""Audit hard wide-lane integer consistency for saved ambiguity basins."""

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
sys.path.insert(0, str(_REPO_ROOT / "experiments"))

from exp_wp23b_basin_ar import _build_dd_measurements  # noqa: E402
from gnss_gpu.io.ppc import PPCDatasetLoader  # noqa: E402
from gnss_gpu.widelane import WidelaneDDPseudorangeComputer  # noqa: E402


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def _assignment_family_integers(assignment_json: str) -> dict[tuple[str, str, str], int]:
    result: dict[tuple[str, str, str], int] = {}
    for ref_sat, sat_id, _wavelength_nm, _generation, integer in json.loads(
        assignment_json
    ):
        if "@" not in ref_sat or "@" not in sat_id:
            continue
        ref_base, ref_family = ref_sat.split("@", 1)
        sat_base, sat_family = sat_id.split("@", 1)
        if ref_family == sat_family:
            result[(ref_base, sat_base, ref_family)] = int(integer)
    return result


def _residuals(
    assignment_json: str,
    fixed_dd_ambiguities: tuple[tuple[str, str, int], ...],
) -> tuple[int, ...]:
    family = _assignment_family_integers(assignment_json)
    values: list[int] = []
    for ref_sat, sat_id, wide_integer in fixed_dd_ambiguities:
        l1 = (ref_sat, sat_id, "L1_E1_B1")
        l2 = (ref_sat, sat_id, "L2_E5B_B2")
        if l1 in family and l2 in family:
            values.append(family[l1] - family[l2] - int(wide_integer))
    return tuple(values)


def analyze(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_epoch: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in _read_csv(args.basin_trace):
        epoch = int(row["epoch"])
        if epoch < int(getattr(args, "start", 0)):
            continue
        end = getattr(args, "end", None)
        if end is not None and epoch >= int(end):
            continue
        by_epoch[epoch].append(row)
    if not by_epoch:
        raise RuntimeError("requested epoch range contains no basin rows")
    max_epoch = max(by_epoch)
    data = PPCDatasetLoader(args.data_dir).load_experiment_data(
        max_epochs=max_epoch + 1,
        systems=("G", "R", "E", "C", "J"),
    )
    resolver = WidelaneDDPseudorangeComputer(
        args.data_dir / "base.obs",
        args.data_dir / "rover.obs",
        base_position=np.asarray(data["base_ecef"], dtype=np.float64),
        allowed_systems=("G", "J"),
        min_epochs=int(args.min_epochs),
        max_std_cycles=float(args.max_std_cycles),
        ratio_threshold=float(args.ratio_threshold),
        min_fix_rate=float(args.min_fix_rate),
    )
    output: list[dict[str, Any]] = []
    residual_histogram: Counter[int] = Counter()
    evidence_epochs = 0
    for epoch in range(max_epoch + 1):
        rows = by_epoch.get(epoch)
        if not rows:
            continue
        base_row = max(rows, key=lambda row: float(row["log_weight"]))
        approximate = np.asarray(
            [float(base_row["ecef_x"]), float(base_row["ecef_y"]), float(base_row["ecef_z"])],
            dtype=np.float64,
        )
        measurements = _build_dd_measurements(
            np.asarray(data["sat_ecef"][epoch], dtype=np.float64),
            np.asarray(data["system_ids"][epoch], dtype=np.int32),
            list(data["used_prns"][epoch]),
            np.asarray(data["weights"][epoch], dtype=np.float64),
            approximate,
            ("G", "E", "J", "C"),
            min_elevation_deg=-90.0,
            min_snr=0.0,
            keep_best=0,
        )
        _dd, stats = resolver.compute_dd(
            float(data["times"][epoch]),
            measurements,
            rover_position_approx=approximate,
            min_common_sats=4,
            rover_weights=np.asarray(data["weights"][epoch], dtype=np.float64),
        )
        if not stats.fixed_dd_ambiguities:
            continue
        evidence_epochs += 1
        truth = np.asarray(data["ground_truth"][epoch], dtype=np.float64)
        for row in rows:
            position = np.asarray(
                [float(row["ecef_x"]), float(row["ecef_y"]), float(row["ecef_z"])],
                dtype=np.float64,
            )
            residuals = _residuals(row["assignment_json"], stats.fixed_dd_ambiguities)
            residual_histogram.update(residuals)
            output.append(
                {
                    "epoch": epoch,
                    "tow": float(row["tow"]),
                    "basin_id": row["basin_id"],
                    "log_weight": float(row["log_weight"]),
                    "error_m": float(np.linalg.norm(position - truth)),
                    "n_pairs": len(residuals),
                    "squared_residual": (
                        float(np.dot(residuals, residuals)) if residuals else ""
                    ),
                    "residuals": "|".join(str(value) for value in residuals),
                    "fixed_dd_ambiguities": json.dumps(stats.fixed_dd_ambiguities),
                }
            )
    summary = {
        "n_trace_epochs": len(by_epoch),
        "evidence_epochs": evidence_epochs,
        "rows": len(output),
        "residual_histogram": dict(sorted(residual_histogram.items())),
    }
    return summary, output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("basin_trace", type=Path)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int)
    parser.add_argument("--min-epochs", type=int, default=5)
    parser.add_argument("--max-std-cycles", type=float, default=0.75)
    parser.add_argument("--ratio-threshold", type=float, default=3.0)
    parser.add_argument("--min-fix-rate", type=float, default=0.3)
    parser.add_argument("--out-summary", type=Path, required=True)
    parser.add_argument("--out-basins", type=Path, required=True)
    args = parser.parse_args()
    summary, rows = analyze(args)
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    args.out_basins.parent.mkdir(parents=True, exist_ok=True)
    with args.out_basins.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
