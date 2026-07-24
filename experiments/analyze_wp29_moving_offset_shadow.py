#!/usr/bin/env python3
"""Rank parallel moving-trajectory offset modes with truth-free DDPR evidence."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "python"))
sys.path.insert(0, str(_ROOT / "experiments"))

from exp_wp23b_basin_ar import _build_dd_measurements  # noqa: E402
from gnss_gpu.dd_carrier import DDCarrierComputer  # noqa: E402
from gnss_gpu.dd_pseudorange import DDPseudorangeComputer  # noqa: E402
from gnss_gpu.io.ppc import PPCDatasetLoader  # noqa: E402
from gnss_gpu.io.rinex_cache import RinexObservationCache  # noqa: E402
from gnss_gpu.local_fgo import DDCarrierEpoch, DDPseudorangeEpoch  # noqa: E402
from gnss_gpu.stop_segment_static import _dd_expected_and_jacobian_m  # noqa: E402


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def _position(row: dict[str, str]) -> np.ndarray:
    return np.asarray(
        [float(row["ecef_x"]), float(row["ecef_y"]), float(row["ecef_z"])],
        dtype=np.float64,
    )


def recurring_offset_candidates(
    basins_by_epoch: dict[int, np.ndarray],
    trajectory_by_epoch: dict[int, np.ndarray],
    start: int,
    end: int,
    *,
    sample_stride_epochs: int = 5,
    radius_m: float = 0.25,
    dedup_radius_m: float = 0.25,
    max_candidates: int = 24,
) -> list[dict[str, Any]]:
    seeds: list[np.ndarray] = []
    for epoch in range(start, end, max(1, int(sample_stride_epochs))):
        basins = basins_by_epoch.get(epoch)
        trajectory = trajectory_by_epoch.get(epoch)
        if basins is not None and trajectory is not None:
            seeds.extend(np.asarray(basins) - np.asarray(trajectory).reshape(1, 3))
    scored: list[tuple[int, int, np.ndarray]] = []
    for seed in seeds:
        coverage = 0
        members = 0
        for epoch in range(start, end):
            basins = basins_by_epoch.get(epoch)
            trajectory = trajectory_by_epoch.get(epoch)
            if basins is None or trajectory is None:
                continue
            offsets = np.asarray(basins) - np.asarray(trajectory).reshape(1, 3)
            distances = np.linalg.norm(offsets - seed.reshape(1, 3), axis=1)
            coverage += int(np.any(distances <= radius_m))
            members += int(np.count_nonzero(distances <= radius_m))
        scored.append((coverage, members, np.asarray(seed).copy()))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    output: list[dict[str, Any]] = []
    for coverage, members, seed in scored:
        if any(
            np.linalg.norm(seed - np.asarray(item["offset_ecef_m"])) <= dedup_radius_m
            for item in output
        ):
            continue
        output.append(
            {
                "offset_ecef_m": seed,
                "coverage_epochs": int(coverage),
                "members": int(members),
            }
        )
        if len(output) >= int(max_candidates):
            break
    return output


def _huber_cost(values: np.ndarray, k: float) -> np.ndarray:
    absolute = np.abs(np.asarray(values, dtype=np.float64))
    return np.where(absolute <= k, 0.5 * np.square(absolute), k * (absolute - 0.5 * k))


def score_moving_offset(
    offset_ecef_m: np.ndarray,
    trajectory_by_epoch: dict[int, np.ndarray],
    ddpr_by_epoch: dict[int, DDPseudorangeEpoch],
    epochs: list[int],
    *,
    sigma_m: float = 4.0,
    huber_k: float = 1.5,
) -> tuple[float, int]:
    residuals: list[float] = []
    offset = np.asarray(offset_ecef_m, dtype=np.float64).reshape(3)
    for epoch in epochs:
        obs = ddpr_by_epoch.get(epoch)
        trajectory = trajectory_by_epoch.get(epoch)
        if obs is None or trajectory is None:
            continue
        position = np.asarray(trajectory) + offset
        for index in range(len(obs.dd_pseudorange_m)):
            expected, _jacobian = _dd_expected_and_jacobian_m(
                position,
                obs.sat_ecef_k[index],
                obs.sat_ecef_ref[index],
                obs.base_range_k[index],
                obs.base_range_ref[index],
            )
            residuals.append(float(obs.dd_pseudorange_m[index] - expected) / sigma_m)
    if not residuals:
        return float("inf"), 0
    values = np.asarray(residuals)
    return float(np.mean(_huber_cost(values, huber_k))), len(values)


def _assignment_integers(row: dict[str, str]) -> dict[tuple[str, str, int], int]:
    return {
        (str(ref_sat), str(sat_id), int(wavelength_nm)): int(integer)
        for ref_sat, sat_id, wavelength_nm, _generation, integer in json.loads(
            row["assignment_json"]
        )
    }


def _lookup_assignment_integer(
    assignments: dict[tuple[str, str, int], int],
    ref_sat: str,
    sat_id: str,
    wavelength_m: float,
) -> int | None:
    wavelength_nm = int(round(float(wavelength_m) * 1e9))
    return assignments.get((str(ref_sat), str(sat_id), wavelength_nm), assignments.get(
        (
            str(ref_sat).split("@", 1)[0],
            str(sat_id).split("@", 1)[0],
            wavelength_nm,
        )
    ))


def score_moving_assigned_carrier(
    offset_ecef_m: np.ndarray,
    trajectory_by_epoch: dict[int, np.ndarray],
    basin_rows_by_epoch: dict[int, list[dict[str, str]]],
    carrier_by_epoch: dict[int, DDCarrierEpoch],
    epochs: list[int],
    *,
    sigma_cycles: float = 0.5,
    huber_k: float = 1.5,
    max_mode_distance_m: float = 0.75,
) -> tuple[float, int]:
    residuals: list[float] = []
    offset = np.asarray(offset_ecef_m, dtype=np.float64).reshape(3)
    for epoch in epochs:
        obs = carrier_by_epoch.get(epoch)
        trajectory = trajectory_by_epoch.get(epoch)
        rows = basin_rows_by_epoch.get(epoch, [])
        if (
            obs is None
            or trajectory is None
            or not rows
            or obs.sat_ids is None
            or obs.ref_sat_ids is None
        ):
            continue
        position = np.asarray(trajectory) + offset
        row = min(rows, key=lambda item: np.linalg.norm(_position(item) - position))
        if np.linalg.norm(_position(row) - position) > max_mode_distance_m:
            continue
        assignments = _assignment_integers(row)
        for index, (ref_sat, sat_id) in enumerate(zip(obs.ref_sat_ids, obs.sat_ids)):
            wavelength = float(obs.wavelengths_m[index])
            integer = _lookup_assignment_integer(
                assignments, str(ref_sat), str(sat_id), wavelength
            )
            if integer is None:
                continue
            expected, _jacobian = _dd_expected_and_jacobian_m(
                position,
                obs.sat_ecef_k[index],
                obs.sat_ecef_ref[index],
                obs.base_range_k[index],
                obs.base_range_ref[index],
            )
            residuals.append(
                float(obs.dd_carrier_cycles[index] - expected / wavelength - integer)
                / sigma_cycles
            )
    if not residuals:
        return float("inf"), 0
    values = np.asarray(residuals)
    return float(np.mean(_huber_cost(values, huber_k))), len(values)


def run(args: argparse.Namespace) -> dict[str, Any]:
    basin_rows = _read_csv(args.basin_trace)
    trajectory_rows = _read_csv(args.trajectory)
    basins_by_epoch: dict[int, list[np.ndarray]] = {}
    basin_rows_by_epoch: dict[int, list[dict[str, str]]] = {}
    for row in basin_rows:
        epoch = int(row["epoch"])
        if args.start <= epoch < args.end:
            basins_by_epoch.setdefault(epoch, []).append(_position(row))
            basin_rows_by_epoch.setdefault(epoch, []).append(row)
    basin_arrays = {epoch: np.asarray(rows) for epoch, rows in basins_by_epoch.items()}
    trajectory_by_epoch = {
        int(row["epoch"]): _position(row)
        for row in trajectory_rows
        if args.start <= int(row["epoch"]) < args.end
    }
    candidates = recurring_offset_candidates(
        basin_arrays,
        trajectory_by_epoch,
        args.start,
        args.end,
        sample_stride_epochs=args.sample_stride_epochs,
        radius_m=args.radius_m,
        dedup_radius_m=args.dedup_radius_m,
        max_candidates=args.max_candidates,
    )
    if not candidates:
        raise RuntimeError("no recurring trajectory offsets")

    loader = PPCDatasetLoader(args.data_dir)
    data = loader.load_experiment_data(
        max_epochs=args.end,
        include_sat_velocity=True,
        systems=("G", "R", "E", "C", "J"),
    )
    cache = RinexObservationCache()
    systems = ("G", "E", "J", "C")
    pseudorange = DDPseudorangeComputer(
        args.data_dir / "base.obs",
        rover_obs_path=args.data_dir / "rover.obs",
        base_position=np.asarray(data["base_ecef"], dtype=np.float64),
        allowed_systems=systems,
        observation_cache=cache,
    )
    carrier = DDCarrierComputer(
        args.data_dir / "base.obs",
        rover_obs_path=args.data_dir / "rover.obs",
        base_position=np.asarray(data["base_ecef"], dtype=np.float64),
        allowed_systems=systems,
        observation_cache=cache,
    )
    ddpr_by_epoch: dict[int, DDPseudorangeEpoch] = {}
    carrier_by_epoch: dict[int, DDCarrierEpoch] = {}
    carrier_families = tuple(
        value for value in str(args.carrier_families).split(",") if value
    )
    for epoch in range(args.start, args.end):
        approximate = trajectory_by_epoch.get(epoch)
        if approximate is None:
            continue
        measurements = _build_dd_measurements(
            np.asarray(data["sat_ecef"][epoch]),
            np.asarray(data["system_ids"][epoch]),
            list(data["used_prns"][epoch]),
            np.asarray(data["weights"][epoch]),
            approximate,
            systems,
            min_elevation_deg=-90.0,
            min_snr=0.0,
            keep_best=0,
        )
        result = pseudorange.compute_dd(
            float(data["times"][epoch]),
            measurements,
            rover_position_approx=approximate,
            min_common_sats=4,
        )
        if result is not None:
            ddpr_by_epoch[epoch] = DDPseudorangeEpoch.from_result(result)
        cp_result = carrier.compute_dd_families(
            float(data["times"][epoch]),
            measurements,
            rover_position_approx=approximate,
            min_common_sats=2,
            carrier_families=carrier_families,
        )
        if cp_result is not None:
            carrier_by_epoch[epoch] = DDCarrierEpoch.from_result(cp_result)

    epochs = list(range(args.start, args.end))
    truth = np.asarray(data["ground_truth"], dtype=np.float64)
    boundaries = np.linspace(
        args.start, args.end, int(args.bootstrap_blocks) + 1, dtype=np.int64
    )
    ranked: list[dict[str, Any]] = []
    for candidate_id, candidate in enumerate(candidates):
        offset = np.asarray(candidate["offset_ecef_m"])
        cost, rows = score_moving_offset(
            offset,
            trajectory_by_epoch,
            ddpr_by_epoch,
            epochs,
            sigma_m=args.ddpr_sigma_m,
            huber_k=args.huber_k,
        )
        carrier_cost, carrier_rows = score_moving_assigned_carrier(
            offset,
            trajectory_by_epoch,
            basin_rows_by_epoch,
            carrier_by_epoch,
            epochs,
            sigma_cycles=args.carrier_sigma_cycles,
            huber_k=args.huber_k,
            max_mode_distance_m=args.max_mode_distance_m,
        )
        block_costs = [
            (
                score_moving_offset(
                    offset,
                    trajectory_by_epoch,
                    ddpr_by_epoch,
                    list(range(int(left), int(right))),
                    sigma_m=args.ddpr_sigma_m,
                    huber_k=args.huber_k,
                )[0]
                + score_moving_assigned_carrier(
                    offset,
                    trajectory_by_epoch,
                    basin_rows_by_epoch,
                    carrier_by_epoch,
                    list(range(int(left), int(right))),
                    sigma_cycles=args.carrier_sigma_cycles,
                    huber_k=args.huber_k,
                    max_mode_distance_m=args.max_mode_distance_m,
                )[0]
            )
            for left, right in zip(boundaries[:-1], boundaries[1:])
        ]
        errors = [
            np.linalg.norm(trajectory_by_epoch[epoch] + offset - truth[epoch])
            for epoch in epochs
            if epoch in trajectory_by_epoch
        ]
        ranked.append(
            {
                "candidate_id": candidate_id,
                "offset_ecef_m": offset.tolist(),
                "coverage_epochs": candidate["coverage_epochs"],
                "members": candidate["members"],
                "ddpr_huber_cost": cost,
                "ddpr_rows": rows,
                "carrier_huber_cost": carrier_cost,
                "carrier_rows": carrier_rows,
                "combined_huber_cost": cost + carrier_cost,
                "bootstrap_costs": block_costs,
                "audit_sub50cm_epochs": int(np.count_nonzero(np.asarray(errors) < 0.5)),
                "audit_rms_m": float(np.sqrt(np.mean(np.square(errors)))),
            }
        )
    for row in ranked:
        row["bootstrap_wins"] = 0
    for block in range(int(args.bootstrap_blocks)):
        min(ranked, key=lambda row: row["bootstrap_costs"][block])[
            "bootstrap_wins"
        ] += 1
    ranked.sort(key=lambda row: row["combined_huber_cost"])
    return {
        "segment": [args.start, args.end],
        "n_ddpr_epochs": len(ddpr_by_epoch),
        "n_carrier_epochs": len(carrier_by_epoch),
        "bootstrap_blocks": int(args.bootstrap_blocks),
        "candidates": ranked,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--basin-trace", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--sample-stride-epochs", type=int, default=5)
    parser.add_argument("--radius-m", type=float, default=0.25)
    parser.add_argument("--dedup-radius-m", type=float, default=0.25)
    parser.add_argument("--max-candidates", type=int, default=24)
    parser.add_argument("--ddpr-sigma-m", type=float, default=4.0)
    parser.add_argument("--carrier-families", default="L1_E1_B1,L5_E5A_B2A")
    parser.add_argument("--carrier-sigma-cycles", type=float, default=0.5)
    parser.add_argument("--max-mode-distance-m", type=float, default=0.75)
    parser.add_argument("--huber-k", type=float, default=1.5)
    parser.add_argument("--bootstrap-blocks", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
