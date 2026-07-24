#!/usr/bin/env python3
"""Shadow-rank ambiguity basins with truth-free PLATEAU 3DMA evidence."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
_BUILD_PYTHON = _REPO_ROOT / "build" / "python"
if _BUILD_PYTHON.exists():
    sys.path.insert(0, str(_BUILD_PYTHON))
sys.path.insert(0, str(_REPO_ROOT / "python"))

from gnss_gpu.bvh import BVHAccelerator  # noqa: E402
from gnss_gpu.candidate_3dma import (  # noqa: E402
    cn0_to_los_probability,
    score_candidate_positions,
)
from gnss_gpu.io.ppc import PPCDatasetLoader  # noqa: E402
from gnss_gpu.spp import correct_pseudoranges  # noqa: E402


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - float(np.max(values))
    weights = np.exp(np.clip(shifted, -745.0, 0.0))
    return weights / float(np.sum(weights))


def _select(
    base_log_weights: np.ndarray,
    pseudorange_scores: np.ndarray,
    visibility_scores: np.ndarray,
    *,
    pseudorange_weight: float,
    visibility_weight: float,
) -> tuple[int, float]:
    scores = (
        base_log_weights
        + pseudorange_weight * pseudorange_scores
        + visibility_weight * visibility_scores
    )
    probabilities = _softmax(scores)
    selected = int(np.argmax(scores))
    return selected, float(probabilities[selected])


def analyze(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    basin_rows = _read_csv(args.basin_trace)
    by_epoch: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in basin_rows:
        epoch = int(row["epoch"])
        if args.start_epoch is not None and epoch < args.start_epoch:
            continue
        if args.end_epoch is not None and epoch >= args.end_epoch:
            continue
        if (epoch - (args.start_epoch or 0)) % args.epoch_stride != 0:
            continue
        by_epoch[epoch].append(row)
    if not by_epoch:
        raise ValueError("no basin rows remain after epoch filtering")
    max_epoch = max(by_epoch)
    data = PPCDatasetLoader(args.data_dir).load_experiment_data(
        max_epochs=max_epoch + 1,
        systems=tuple(part.strip() for part in args.systems.split(",") if part.strip()),
    )
    with np.load(args.triangle_cache_npz) as cache:
        triangles = np.asarray(cache["triangles"], dtype=np.float64)
    bvh = BVHAccelerator(triangles)

    output: list[dict[str, Any]] = []
    epoch_metrics: list[dict[str, Any]] = []
    streak = 0
    for epoch in sorted(by_epoch):
        rows = by_epoch[epoch]
        candidates = np.asarray(
            [
                [float(row["ecef_x"]), float(row["ecef_y"]), float(row["ecef_z"])]
                for row in rows
            ],
            dtype=np.float64,
        )
        satellites = np.asarray(data["sat_ecef"][epoch], dtype=np.float64)
        pseudoranges = np.asarray(data["pseudoranges"][epoch], dtype=np.float64)
        cn0 = np.asarray(data["weights"][epoch], dtype=np.float64)
        system_ids = np.asarray(data["system_ids"][epoch], dtype=np.int32)
        base_index = int(np.argmax([float(row["log_weight"]) for row in rows]))
        corrected, satellite_weights = correct_pseudoranges(
            satellites,
            pseudoranges,
            candidates[base_index],
            float(data["times"][epoch]),
        )
        repeated_satellites = np.broadcast_to(
            satellites[None, :, :],
            (len(candidates), len(satellites), 3),
        ).copy()
        predicted_los = np.asarray(
            bvh.check_los_batch(candidates, repeated_satellites), dtype=bool
        )
        observed_los = cn0_to_los_probability(
            cn0,
            midpoint_dbhz=float(args.cn0_midpoint_dbhz),
            scale_db=float(args.cn0_scale_db),
        )
        result = score_candidate_positions(
            candidates,
            satellites,
            corrected,
            predicted_los,
            satellite_weights=satellite_weights,
            clock_group_ids=system_ids,
            observed_los_probability=observed_los,
            sigma_los_m=float(args.sigma_los_m),
            nlos_bias_m=float(args.nlos_bias_m),
            sigma_nlos_negative_m=float(args.sigma_nlos_negative_m),
            sigma_nlos_positive_m=float(args.sigma_nlos_positive_m),
            visibility_weight=1.0,
        )
        selected, gamma = _select(
            np.asarray([float(row["log_weight"]) for row in rows]),
            result.pseudorange_scores,
            result.visibility_scores,
            pseudorange_weight=float(args.pseudorange_weight),
            visibility_weight=float(args.visibility_weight),
        )
        truth = np.asarray(data["ground_truth"][epoch], dtype=np.float64)
        errors = np.linalg.norm(candidates - truth[None, :], axis=1)
        streak = streak + 1 if gamma >= float(args.gamma_threshold) else 0
        declared_fix = streak >= int(args.fix_streak)
        epoch_metrics.append(
            {
                "epoch": epoch,
                "selected_error_m": float(errors[selected]),
                "base_error_m": float(errors[base_index]),
                "oracle_error_m": float(np.min(errors)),
                "gamma": gamma,
                "declared_fix": declared_fix,
                "false_fix": declared_fix and errors[selected] >= 0.5,
            }
        )
        for index, row in enumerate(rows):
            output.append(
                {
                    "epoch": epoch,
                    "tow": float(row["tow"]),
                    "basin_id": row["basin_id"],
                    "assignment_id": row["assignment_id"],
                    "log_weight": float(row["log_weight"]),
                    "pseudorange_score": float(result.pseudorange_scores[index]),
                    "visibility_score": float(result.visibility_scores[index]),
                    "predicted_los_count": int(np.count_nonzero(predicted_los[index])),
                    "predicted_los_mask": "".join(
                        "1" if value else "0" for value in predicted_los[index]
                    ),
                    "error_m": float(errors[index]),
                    "selected": int(index == selected),
                }
            )

    fixed = [row for row in epoch_metrics if row["declared_fix"]]
    summary = {
        "n_epochs": len(epoch_metrics),
        "start_epoch": min(by_epoch),
        "end_epoch_exclusive": max(by_epoch) + 1,
        "epoch_stride": int(args.epoch_stride),
        "triangle_count": int(len(triangles)),
        "pseudorange_weight": float(args.pseudorange_weight),
        "visibility_weight": float(args.visibility_weight),
        "gamma_threshold": float(args.gamma_threshold),
        "fix_streak": int(args.fix_streak),
        "base_sub50cm_epochs": sum(row["base_error_m"] < 0.5 for row in epoch_metrics),
        "selected_sub50cm_epochs": sum(
            row["selected_error_m"] < 0.5 for row in epoch_metrics
        ),
        "oracle_sub50cm_epochs": sum(
            row["oracle_error_m"] < 0.5 for row in epoch_metrics
        ),
        "declared_fix_epochs": len(fixed),
        "fix_rate_pct": 100.0 * len(fixed) / max(len(epoch_metrics), 1),
        "false_fix_epochs": sum(bool(row["false_fix"]) for row in fixed),
        "false_fix_pct": 100.0
        * sum(bool(row["false_fix"]) for row in fixed)
        / max(len(fixed), 1),
    }
    return summary, output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("basin_trace", type=Path)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--triangle-cache-npz", type=Path, required=True)
    parser.add_argument("--systems", default="G,R,E,C,J")
    parser.add_argument("--start-epoch", type=int)
    parser.add_argument("--end-epoch", type=int)
    parser.add_argument("--epoch-stride", type=int, default=1)
    parser.add_argument("--pseudorange-weight", type=float, default=0.0)
    parser.add_argument("--visibility-weight", type=float, default=1.0)
    parser.add_argument("--sigma-los-m", type=float, default=3.0)
    parser.add_argument("--nlos-bias-m", type=float, default=15.0)
    parser.add_argument("--sigma-nlos-negative-m", type=float, default=8.0)
    parser.add_argument("--sigma-nlos-positive-m", type=float, default=25.0)
    parser.add_argument("--cn0-midpoint-dbhz", type=float, default=32.0)
    parser.add_argument("--cn0-scale-db", type=float, default=4.0)
    parser.add_argument("--gamma-threshold", type=float, default=0.99)
    parser.add_argument("--fix-streak", type=int, default=3)
    parser.add_argument("--out-summary", type=Path, required=True)
    parser.add_argument("--out-basins", type=Path, required=True)
    args = parser.parse_args()
    if args.epoch_stride < 1:
        parser.error("--epoch-stride must be at least 1")
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
