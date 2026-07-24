#!/usr/bin/env python3
"""Apply an alternate IMU route only where fixed-assignment carrier wins in blocks."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "python"))
sys.path.insert(0, str(_ROOT / "experiments"))

from analyze_wp29_candidate_absolute_evidence import _carrier_cost, _ddpr_cost  # noqa: E402
from analyze_wp29_moving_offset_shadow import _position  # noqa: E402
from apply_wp29_carrier_runner_block_shadow import contiguous_anchor_blocks  # noqa: E402
from exp_wp23b_basin_ar import _build_dd_measurements  # noqa: E402
from gnss_gpu.dd_carrier import DDCarrierComputer  # noqa: E402
from gnss_gpu.dd_pseudorange import DDPseudorangeComputer  # noqa: E402
from gnss_gpu.io.ppc import PPCDatasetLoader  # noqa: E402
from gnss_gpu.io.rinex_cache import RinexObservationCache  # noqa: E402
from gnss_gpu.local_fgo import DDCarrierEpoch, DDPseudorangeEpoch  # noqa: E402

_XYZ = ("ecef_x", "ecef_y", "ecef_z")


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", type=Path)
    parser.add_argument("alternate_route", type=Path)
    parser.add_argument("candidate_audit", type=Path)
    parser.add_argument("basin_trace", type=Path)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--anchor-stride", type=int, default=5)
    parser.add_argument("--min-block-anchors", type=int, default=5)
    parser.add_argument("--min-carrier-rows", type=int, default=8)
    parser.add_argument("--max-carrier-ratio", type=float, default=1.0)
    parser.add_argument("--max-ddpr-ratio", type=float, default=1.0)
    parser.add_argument("--require-ddpr-improvement", action="store_true")
    parser.add_argument("--out-evidence", type=Path, required=True)
    parser.add_argument("--out-summary", type=Path, required=True)
    parser.add_argument("--out-trajectory", type=Path, required=True)
    args = parser.parse_args()

    trajectory = _read(args.trajectory)
    alternate = {int(row["epoch"]): row for row in _read(args.alternate_route)}
    selected_ids = {
        int(row["epoch"]): row["basin_id"]
        for row in _read(args.candidate_audit)
        if int(row["selected"]) == 1
    }
    basin_rows = {
        (int(row["epoch"]), row["basin_id"]): row for row in _read(args.basin_trace)
    }
    epochs = sorted(
        epoch
        for epoch in alternate
        if epoch % int(args.anchor_stride) == 0
        and epoch in selected_ids
        and (epoch, selected_ids[epoch]) in basin_rows
    )
    if not epochs:
        raise RuntimeError("alternate route contains no selected basin anchors")

    data = PPCDatasetLoader(args.data_dir).load_experiment_data(
        max_epochs=max(epochs) + 1,
        include_sat_velocity=True,
        systems=("G", "R", "E", "C", "J"),
    )
    cache = RinexObservationCache()
    systems = ("G", "E", "J", "C")
    carrier = DDCarrierComputer(
        args.data_dir / "base.obs",
        rover_obs_path=args.data_dir / "rover.obs",
        base_position=np.asarray(data["base_ecef"], dtype=np.float64),
        allowed_systems=systems,
        observation_cache=cache,
    )
    pseudorange = DDPseudorangeComputer(
        args.data_dir / "base.obs",
        rover_obs_path=args.data_dir / "rover.obs",
        base_position=np.asarray(data["base_ecef"], dtype=np.float64),
        allowed_systems=systems,
        observation_cache=cache,
    )
    truth_times, truth_positions = PPCDatasetLoader(args.data_dir).load_ground_truth()

    def truth_at(tow: float) -> np.ndarray:
        return np.asarray(
            truth_positions[int(np.argmin(np.abs(truth_times - float(tow))))],
            dtype=np.float64,
        )
    evidence: list[dict[str, object]] = []
    winners: list[int] = []
    for epoch in epochs:
        assignment = basin_rows[(epoch, selected_ids[epoch])]
        current_position = _position(trajectory[epoch])
        route_position = _position(alternate[epoch])
        measurements = _build_dd_measurements(
            np.asarray(data["sat_ecef"][epoch]),
            np.asarray(data["system_ids"][epoch]),
            list(data["used_prns"][epoch]),
            np.asarray(data["weights"][epoch]),
            current_position,
            systems,
            min_elevation_deg=-90.0,
            min_snr=0.0,
            keep_best=0,
        )
        result = carrier.compute_dd_families(
            float(data["times"][epoch]),
            measurements,
            rover_position_approx=current_position,
            min_common_sats=2,
            carrier_families=("L1_E1_B1", "L5_E5A_B2A"),
        )
        ddpr_result = pseudorange.compute_dd(
            float(data["times"][epoch]),
            measurements,
            rover_position_approx=current_position,
            min_common_sats=4,
        )
        if result is None and ddpr_result is None:
            continue
        observation = None if result is None else DDCarrierEpoch.from_result(result)
        current_cost, current_rows = (
            (float("nan"), 0)
            if observation is None
            else _carrier_cost(current_position, assignment, observation, 0.5)
        )
        route_cost, route_rows = (
            (float("nan"), 0)
            if observation is None
            else _carrier_cost(route_position, assignment, observation, 0.5)
        )
        rows = min(current_rows, route_rows)
        ratio = route_cost / max(current_cost, 1.0e-12)
        ddpr = None if ddpr_result is None else DDPseudorangeEpoch.from_result(ddpr_result)
        current_ddpr_cost = (
            float("nan") if ddpr is None else _ddpr_cost(current_position, ddpr, 4.0)
        )
        route_ddpr_cost = (
            float("nan") if ddpr is None else _ddpr_cost(route_position, ddpr, 4.0)
        )
        ddpr_ratio = route_ddpr_cost / max(current_ddpr_cost, 1.0e-12)
        accepted = (
            rows >= int(args.min_carrier_rows)
            and np.isfinite(ratio)
            and ratio < float(args.max_carrier_ratio)
            and (
                not args.require_ddpr_improvement
                or (
                    np.isfinite(ddpr_ratio)
                    and ddpr_ratio < float(args.max_ddpr_ratio)
                )
            )
        )
        if accepted:
            winners.append(epoch)
        truth = truth_at(float(trajectory[epoch]["tow"]))
        evidence.append(
            {
                "epoch": epoch,
                "basin_id": selected_ids[epoch],
                "carrier_rows": rows,
                "current_carrier_cost": current_cost,
                "route_carrier_cost": route_cost,
                "route_current_ratio": ratio,
                "current_ddpr_cost": current_ddpr_cost,
                "route_ddpr_cost": route_ddpr_cost,
                "route_current_ddpr_ratio": ddpr_ratio,
                "route_winner": int(accepted),
                "current_audit_error_m": float(np.linalg.norm(current_position - truth)),
                "route_audit_error_m": float(np.linalg.norm(route_position - truth)),
            }
        )
    blocks = contiguous_anchor_blocks(
        winners,
        stride=int(args.anchor_stride),
        min_anchors=int(args.min_block_anchors),
    )
    applied_epochs = {
        epoch for block in blocks for epoch in range(block[0], block[-1] + 1)
    }
    output = []
    for row in trajectory:
        epoch = int(row["epoch"])
        use_route = epoch in applied_epochs and epoch in alternate
        position = _position(alternate[epoch]) if use_route else _position(row)
        truth = truth_at(float(row["tow"]))
        error = float(np.linalg.norm(position - truth))
        fix = int(row.get("fix", "0"))
        output.append(
            {
                **row,
                **{key: float(position[index]) for index, key in enumerate(_XYZ)},
                "source": "imu_route_carrier_block" if use_route else row.get("source", ""),
                "error_m": error,
                "sub50cm": int(error < 0.5),
                "false_fix": int(bool(fix) and error >= 0.5),
                "imu_route_carrier_applied": int(use_route),
            }
        )
    fixed = [row for row in output if int(row["fix"])]
    summary = {
        "n_epochs_full_denominator": len(output),
        "route_winner_anchor_epochs": winners,
        "accepted_anchor_blocks": blocks,
        "require_ddpr_improvement": bool(args.require_ddpr_improvement),
        "max_carrier_ratio": float(args.max_carrier_ratio),
        "max_ddpr_ratio": float(args.max_ddpr_ratio),
        "applied_epochs": len(applied_epochs),
        "sub50cm_full_epochs": sum(int(row["sub50cm"]) for row in output),
        "sub50cm_full_pct": 100.0 * sum(int(row["sub50cm"]) for row in output) / len(output),
        "declared_fix_epochs": len(fixed),
        "false_fix_epochs": sum(int(row["false_fix"]) for row in fixed),
        "false_fix_pct": 100.0 * sum(int(row["false_fix"]) for row in fixed) / max(len(fixed), 1),
    }
    args.out_evidence.parent.mkdir(parents=True, exist_ok=True)
    with args.out_evidence.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(evidence[0]))
        writer.writeheader()
        writer.writerows(evidence)
    args.out_trajectory.parent.mkdir(parents=True, exist_ok=True)
    with args.out_trajectory.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
