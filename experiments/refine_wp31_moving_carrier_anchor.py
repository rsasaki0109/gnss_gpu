#!/usr/bin/env python3
"""Truth-free moving-epoch refinement with fixed carrier integers and DDPR."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "python"))
sys.path.insert(0, str(_ROOT / "experiments"))

from analyze_wp29_moving_offset_shadow import (  # noqa: E402
    _assignment_integers,
    _lookup_assignment_integer,
    _position,
)
from exp_wp23b_basin_ar import _build_dd_measurements  # noqa: E402
from gnss_gpu.dd_carrier import DDCarrierComputer  # noqa: E402
from gnss_gpu.dd_pseudorange import DDPseudorangeComputer  # noqa: E402
from gnss_gpu.io.ppc import PPCDatasetLoader  # noqa: E402
from gnss_gpu.io.rinex_cache import RinexObservationCache  # noqa: E402
from gnss_gpu.local_fgo import DDCarrierEpoch, DDPseudorangeEpoch  # noqa: E402
from gnss_gpu.stop_segment_static import _dd_expected_and_jacobian_m  # noqa: E402


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def refine_fixed_integer_position(
    seeds: list[np.ndarray],
    assignment_row: dict[str, str],
    carrier: DDCarrierEpoch,
    ddpr: DDPseudorangeEpoch,
    *,
    carrier_sigma_cycles: float = 0.5,
    ddpr_sigma_m: float = 4.0,
    prior_sigma_m: float = 20.0,
) -> tuple[list[np.ndarray], dict[str, float | int]]:
    assignments = _assignment_integers(assignment_row)
    carrier_terms: list[tuple[int, int]] = []
    if carrier.sat_ids is not None and carrier.ref_sat_ids is not None:
        for index, (ref_sat, sat_id) in enumerate(zip(carrier.ref_sat_ids, carrier.sat_ids)):
            integer = _lookup_assignment_integer(assignments, str(ref_sat), str(sat_id), float(carrier.wavelengths_m[index]))
            if integer is not None:
                carrier_terms.append((index, integer))

    def residual(position: np.ndarray, prior: np.ndarray) -> np.ndarray:
        values = []
        for index in range(ddpr.n):
            expected, _jac = _dd_expected_and_jacobian_m(
                position, ddpr.sat_ecef_k[index], ddpr.sat_ecef_ref[index],
                ddpr.base_range_k[index], ddpr.base_range_ref[index]
            )
            values.append((float(ddpr.dd_pseudorange_m[index]) - expected) / ddpr_sigma_m)
        for index, integer in carrier_terms:
            expected, _jac = _dd_expected_and_jacobian_m(
                position, carrier.sat_ecef_k[index], carrier.sat_ecef_ref[index],
                carrier.base_range_k[index], carrier.base_range_ref[index]
            )
            wavelength = float(carrier.wavelengths_m[index])
            values.append((float(carrier.dd_carrier_cycles[index]) - expected / wavelength - integer) / carrier_sigma_cycles)
        values.extend(((position - prior) / prior_sigma_m).tolist())
        return np.asarray(values)

    solutions = []
    for seed in seeds:
        result = least_squares(lambda value: residual(value, seed), np.asarray(seed), loss="huber", f_scale=1.5, max_nfev=100)
        solutions.append(np.asarray(result.x))
    center = np.mean(solutions, axis=0)
    spread = float(max(np.linalg.norm(row - center) for row in solutions))
    ddpr_residuals, carrier_residuals = [], []
    for index in range(ddpr.n):
        expected, _ = _dd_expected_and_jacobian_m(center, ddpr.sat_ecef_k[index], ddpr.sat_ecef_ref[index], ddpr.base_range_k[index], ddpr.base_range_ref[index])
        ddpr_residuals.append(float(ddpr.dd_pseudorange_m[index]) - expected)
    for index, integer in carrier_terms:
        expected, _ = _dd_expected_and_jacobian_m(center, carrier.sat_ecef_k[index], carrier.sat_ecef_ref[index], carrier.base_range_k[index], carrier.base_range_ref[index])
        carrier_residuals.append(float(carrier.dd_carrier_cycles[index]) - expected / float(carrier.wavelengths_m[index]) - integer)
    return solutions, {
        "carrier_rows": len(carrier_terms), "ddpr_rows": int(ddpr.n), "seed_solution_spread_m": spread,
        "carrier_rms_cycles": float(np.sqrt(np.mean(np.square(carrier_residuals)))) if carrier_residuals else float("inf"),
        "ddpr_rms_m": float(np.sqrt(np.mean(np.square(ddpr_residuals)))) if ddpr_residuals else float("inf"),
    }


def apply_temporal_consensus(
    rows: list[dict[str, Any]],
    current_positions: dict[int, np.ndarray],
    *,
    stride: int,
    max_edge_residual_m: float = 0.5,
) -> None:
    """Promote only three consecutive metric passes with TDCP-consistent edges."""
    by_epoch = {int(row["epoch"]): row for row in rows}
    for row in rows:
        epoch = int(row["epoch"])
        previous = by_epoch.get(epoch - stride)
        if previous is None:
            row["tdcp_edge_residual_to_previous_m"] = float("nan")
        else:
            refined_delta = np.asarray(row["position_ecef"]) - np.asarray(previous["position_ecef"])
            observed_delta = current_positions[epoch] - current_positions[epoch - stride]
            row["tdcp_edge_residual_to_previous_m"] = float(np.linalg.norm(refined_delta - observed_delta))
        row["production_selected"] = 0
    for epoch in sorted(by_epoch):
        triple = [by_epoch.get(epoch + offset * stride) for offset in (-1, 0, 1)]
        if any(row is None or not int(row["single_epoch_metric_pass"]) for row in triple):
            continue
        if all(float(row["tdcp_edge_residual_to_previous_m"]) <= max_edge_residual_m for row in triple[1:]):
            for row in triple:
                row["production_selected"] = 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("current_trajectory", type=Path)
    parser.add_argument("alternate_route", type=Path)
    parser.add_argument("candidate_audit", type=Path)
    parser.add_argument("basin_trace", type=Path)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    current = {int(row["epoch"]): row for row in _read(args.current_trajectory)}
    alternate = {int(row["epoch"]): row for row in _read(args.alternate_route)}
    selected = {int(row["epoch"]): row["basin_id"] for row in _read(args.candidate_audit) if int(row["selected"]) == 1}
    basins = {(int(row["epoch"]), row["basin_id"]): row for row in _read(args.basin_trace)}
    epochs = [epoch for epoch in range(args.start, args.end, args.stride) if epoch in current and epoch in alternate and epoch in selected and (epoch, selected[epoch]) in basins]
    data = PPCDatasetLoader(args.data_dir).load_experiment_data(max_epochs=args.end, include_sat_velocity=True, systems=("G", "R", "E", "C", "J"))
    cache = RinexObservationCache(); systems = ("G", "E", "J", "C")
    carrier_engine = DDCarrierComputer(args.data_dir / "base.obs", rover_obs_path=args.data_dir / "rover.obs", base_position=np.asarray(data["base_ecef"]), allowed_systems=systems, observation_cache=cache)
    ddpr_engine = DDPseudorangeComputer(args.data_dir / "base.obs", rover_obs_path=args.data_dir / "rover.obs", base_position=np.asarray(data["base_ecef"]), allowed_systems=systems, observation_cache=cache)
    output: list[dict[str, Any]] = []
    for epoch in epochs:
        assignment = basins[(epoch, selected[epoch])]
        seeds = [_position(current[epoch]), _position(alternate[epoch]), _position(assignment)]
        measurements = _build_dd_measurements(np.asarray(data["sat_ecef"][epoch]), np.asarray(data["system_ids"][epoch]), list(data["used_prns"][epoch]), np.asarray(data["weights"][epoch]), seeds[0], systems, min_elevation_deg=-90.0, min_snr=0.0, keep_best=0)
        cp_result = carrier_engine.compute_dd_families(float(data["times"][epoch]), measurements, rover_position_approx=seeds[0], min_common_sats=2, carrier_families=("L1_E1_B1", "L5_E5A_B2A"))
        pr_result = ddpr_engine.compute_dd(float(data["times"][epoch]), measurements, rover_position_approx=seeds[0], min_common_sats=4)
        if cp_result is None or pr_result is None:
            continue
        solutions, metrics = refine_fixed_integer_position(seeds, assignment, DDCarrierEpoch.from_result(cp_result), DDPseudorangeEpoch.from_result(pr_result))
        position = np.mean(solutions, axis=0)
        accepted = metrics["carrier_rows"] >= 3 and metrics["ddpr_rows"] >= 4 and metrics["seed_solution_spread_m"] <= 0.5 and metrics["carrier_rms_cycles"] <= 0.5 and metrics["ddpr_rms_m"] <= 4.0
        truth = np.asarray(data["ground_truth"][epoch])
        output.append({"epoch": epoch, "tow": float(data["times"][epoch]), "basin_id": selected[epoch], **metrics, "single_epoch_metric_pass": int(accepted), "production_selected": 0, "position_ecef": position.tolist(), "audit_error_m": float(np.linalg.norm(position - truth)), "audit_sub50cm": int(np.linalg.norm(position - truth) < 0.5)})
    apply_temporal_consensus(
        output,
        {epoch: _position(row) for epoch, row in current.items()},
        stride=args.stride,
    )
    result = {"schema": "wp31_moving_fixed_integer_anchor_v1", "production_input_truth": False, "segment": [args.start, args.end], "epochs": output, "selected_epochs": [row["epoch"] for row in output if row["production_selected"]]}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
