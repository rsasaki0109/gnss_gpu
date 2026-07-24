#!/usr/bin/env python3
"""Build truth-free reverse TDCP seeds from an accepted late static anchor."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "python"))
sys.path.insert(0, str(_REPO_ROOT / "experiments"))

from exp_ppc_tdcp_velocity import _epoch_measurements  # noqa: E402
from exp_wp23b_float_seed import _doppler_velocity  # noqa: E402
from gnss_gpu.io.ppc import PPCDatasetLoader  # noqa: E402
from gnss_gpu.tdcp_velocity import estimate_displacement_from_tdcp  # noqa: E402
from run_wp29_tdcp_anchor_smoother import (  # noqa: E402
    _load_fusion_static_override,
    _robust_static_velocity_bias,
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def _position(row: dict[str, str]) -> np.ndarray:
    return np.asarray(
        [float(row["ecef_x"]), float(row["ecef_y"]), float(row["ecef_z"])],
        dtype=np.float64,
    )


def reverse_integrate(
    anchor_position: np.ndarray, displacements_by_epoch: Sequence[np.ndarray]
) -> list[np.ndarray]:
    """Return positions from target through anchor for forward interval deltas."""

    position = np.asarray(anchor_position, dtype=np.float64).reshape(3).copy()
    reversed_positions = [position.copy()]
    for displacement in reversed(displacements_by_epoch):
        position = position - np.asarray(displacement, dtype=np.float64).reshape(3)
        reversed_positions.append(position.copy())
    return list(reversed(reversed_positions))


def analyze(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    diagnostics = {int(row["epoch"]): row for row in _read_csv(args.epoch_diagnostics)}
    stop_start, stop_end, anchor_position, candidate_id, fusion_reason = (
        _load_fusion_static_override(args.static_json, args.fusion_json)
    )
    anchor_epoch = int(args.anchor_epoch)
    target_epoch = int(args.target_epoch)
    if not target_epoch < anchor_epoch:
        raise ValueError("target epoch must precede anchor epoch")
    n_epochs = max(max(diagnostics) + 1, stop_end)
    data = PPCDatasetLoader(args.data_dir).load_experiment_data(
        max_epochs=n_epochs,
        include_sat_velocity=True,
        systems=("G", "R", "E", "C", "J"),
    )
    bias_samples: list[np.ndarray] = []
    for epoch in range(stop_start, min(stop_end, n_epochs)):
        velocity, _rms = _doppler_velocity(data, epoch, _position(diagnostics[epoch]))
        if velocity is not None:
            bias_samples.append(np.asarray(velocity, dtype=np.float64))
    bias = _robust_static_velocity_bias(bias_samples)
    if bias is None:
        raise RuntimeError("late static stop has insufficient Doppler bias samples")

    zero_displacements: list[np.ndarray] = []
    doppler_displacements: list[np.ndarray] = []
    tdcp_count = doppler_count = zero_count = 0
    previous = [
        row
        for row in _epoch_measurements(data, target_epoch)
        if int(row.system_id) in (0, 2, 4)
    ]
    for epoch in range(target_epoch + 1, anchor_epoch + 1):
        current = [
            row
            for row in _epoch_measurements(data, epoch)
            if int(row.system_id) in (0, 2, 4)
        ]
        approximate = _position(diagnostics[epoch])
        estimate = estimate_displacement_from_tdcp(
            approximate,
            previous,
            current,
            float(args.epoch_dt_s),
            min_sats=int(args.tdcp_min_sats),
            max_postfit_rms_m=float(args.tdcp_max_postfit_rms_m),
            slip_residual_threshold_m=float(args.tdcp_slip_threshold_m),
        )
        if estimate is not None:
            displacement = np.asarray(estimate.displacement_ecef_m, dtype=np.float64)
            zero_displacements.append(displacement)
            doppler_displacements.append(displacement)
            tdcp_count += 1
        else:
            zero_displacements.append(np.zeros(3, dtype=np.float64))
            velocity, _rms = _doppler_velocity(data, epoch, approximate)
            if velocity is None:
                doppler_displacements.append(np.zeros(3, dtype=np.float64))
                zero_count += 1
            else:
                doppler_displacements.append(
                    (np.asarray(velocity, dtype=np.float64) - bias)
                    * float(args.epoch_dt_s)
                )
                doppler_count += 1
        previous = current

    variants = {
        "tdcp_zero_missing": reverse_integrate(anchor_position, zero_displacements),
        "late_bias_doppler": reverse_integrate(anchor_position, doppler_displacements),
    }
    output: list[dict[str, Any]] = []
    audit: dict[str, Any] = {}
    for name, positions in variants.items():
        errors: list[float] = []
        for offset, position in enumerate(positions):
            epoch = target_epoch + offset
            truth = np.asarray(data["ground_truth"][epoch], dtype=np.float64)
            error = float(np.linalg.norm(position - truth))
            errors.append(error)
            output.append(
                {
                    "variant": name,
                    "epoch": epoch,
                    "tow": float(data["times"][epoch]),
                    "ecef_x": float(position[0]),
                    "ecef_y": float(position[1]),
                    "ecef_z": float(position[2]),
                    "audit_error_m": error,
                }
            )
        audit[name] = {
            "target_error_m": errors[0],
            "median_error_m": float(np.median(errors)),
            "max_error_m": float(np.max(errors)),
            "sub50cm_epochs": int(sum(error < 0.5 for error in errors)),
        }
    summary = {
        "target_epoch": target_epoch,
        "anchor_epoch": anchor_epoch,
        "static_segment": [stop_start, stop_end],
        "static_candidate_id": candidate_id,
        "fusion_reason": fusion_reason,
        "tdcp_intervals": tdcp_count,
        "doppler_intervals": doppler_count,
        "zero_intervals": zero_count,
        "doppler_bias_samples": len(bias_samples),
        "doppler_bias_ecef_mps": bias.tolist(),
        "variants": audit,
    }
    return summary, output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--epoch-diagnostics", type=Path, required=True)
    parser.add_argument("--static-json", type=Path, required=True)
    parser.add_argument("--fusion-json", type=Path, required=True)
    parser.add_argument("--target-epoch", type=int, required=True)
    parser.add_argument("--anchor-epoch", type=int, required=True)
    parser.add_argument("--epoch-dt-s", type=float, default=0.2)
    parser.add_argument("--tdcp-min-sats", type=int, default=5)
    parser.add_argument("--tdcp-max-postfit-rms-m", type=float, default=0.5)
    parser.add_argument("--tdcp-slip-threshold-m", type=float, default=0.25)
    parser.add_argument("--out-summary", type=Path, required=True)
    parser.add_argument("--out-trajectory", type=Path, required=True)
    parser.add_argument("--out-external-seeds", type=Path)
    parser.add_argument("--external-seed-start-epoch", type=int)
    args = parser.parse_args()
    summary, rows = analyze(args)
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    args.out_trajectory.parent.mkdir(parents=True, exist_ok=True)
    with args.out_trajectory.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if args.out_external_seeds is not None:
        seed_start = (
            int(args.target_epoch)
            if args.external_seed_start_epoch is None
            else int(args.external_seed_start_epoch)
        )
        seed_rows = [
            {
                "epoch": row["epoch"],
                "log_weight": 0.0,
                "ecef_x": row["ecef_x"],
                "ecef_y": row["ecef_y"],
                "ecef_z": row["ecef_z"],
            }
            for row in rows
            if row["variant"] == "late_bias_doppler"
            and seed_start <= int(row["epoch"]) < int(args.anchor_epoch)
        ]
        args.out_external_seeds.parent.mkdir(parents=True, exist_ok=True)
        with args.out_external_seeds.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(seed_rows[0]))
            writer.writeheader()
            writer.writerows(seed_rows)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
