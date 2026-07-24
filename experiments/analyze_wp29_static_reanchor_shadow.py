#!/usr/bin/env python3
"""Rank recurring PF stop-position modes with static DD GNSS evidence."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "python"))
sys.path.insert(0, str(_REPO_ROOT / "experiments"))

from exp_wp23b_basin_ar import _build_dd_measurements  # noqa: E402
from gnss_gpu.dd_carrier import DDCarrierComputer  # noqa: E402
from gnss_gpu.dd_pseudorange import DDPseudorangeComputer  # noqa: E402
from gnss_gpu.io.ppc import PPCDatasetLoader  # noqa: E402
from gnss_gpu.io.rinex_cache import RinexObservationCache  # noqa: E402
from gnss_gpu.local_fgo import DDCarrierEpoch, DDPseudorangeEpoch  # noqa: E402
from gnss_gpu.stop_segment_static import (  # noqa: E402
    StaticStopSegmentConfig,
    _dd_expected_and_jacobian_m,
    solve_static_stop_segment,
)


def _read_csv(
    path: Path, *, start_epoch: int | None = None, end_epoch: int | None = None
) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        if start_epoch is None and end_epoch is None:
            return list(reader)
        rows = []
        for row in reader:
            epoch = int(row["epoch"])
            if start_epoch is not None and epoch < start_epoch:
                continue
            if end_epoch is not None and epoch >= end_epoch:
                # Basin traces are serialized in epoch order.
                break
            rows.append(row)
        return rows


def _position(row: dict[str, str]) -> np.ndarray:
    return np.asarray(
        [float(row["ecef_x"]), float(row["ecef_y"]), float(row["ecef_z"])],
        dtype=np.float64,
    )


def parse_ecef(value: str) -> np.ndarray:
    """Parse an explicit truth-free ECEF seed supplied by another estimator."""
    parts = [float(item) for item in str(value).split(",") if item.strip()]
    if len(parts) != 3:
        raise ValueError("ECEF seed must contain exactly three comma-separated values")
    return np.asarray(parts, dtype=np.float64)


def recurring_position_candidates(
    positions_by_epoch: dict[int, np.ndarray],
    start: int,
    end: int,
    *,
    radius_m: float = 0.20,
    sample_stride_epochs: int = 5,
    dedup_radius_m: float = 0.20,
    max_candidates: int = 24,
) -> list[dict[str, Any]]:
    """Return truth-free stop modes ranked by epoch coverage, then PF mass."""
    if max_candidates <= 0:
        return []
    seeds: list[np.ndarray] = []
    for epoch in range(start, end, max(1, int(sample_stride_epochs))):
        rows = positions_by_epoch.get(epoch)
        if rows is not None:
            seeds.extend(np.asarray(rows, dtype=np.float64).reshape(-1, 3))
    scored: list[tuple[int, int, np.ndarray]] = []
    for seed in seeds:
        coverage = 0
        members = 0
        for epoch in range(start, end):
            rows = positions_by_epoch.get(epoch)
            if rows is None:
                continue
            distances = np.linalg.norm(rows - seed.reshape(1, 3), axis=1)
            coverage += int(np.any(distances <= radius_m))
            members += int(np.count_nonzero(distances <= radius_m))
        scored.append((coverage, members, seed))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)

    selected: list[dict[str, Any]] = []
    for coverage, members, seed in scored:
        if any(
            np.linalg.norm(seed - np.asarray(item["position_ecef"])) <= dedup_radius_m
            for item in selected
        ):
            continue
        selected.append(
            {
                "position_ecef": seed.copy(),
                "coverage_epochs": int(coverage),
                "members": int(members),
            }
        )
        if len(selected) >= max_candidates:
            break
    return selected


def offset_seed_candidates(
    center_ecef: np.ndarray,
    radii_m: tuple[float, ...],
    *,
    directions: str,
) -> list[dict[str, Any]]:
    """Build truth-free spatial seeds around an independently scored center."""

    center = np.asarray(center_ecef, dtype=np.float64).reshape(3)
    if directions == "axes":
        vectors = [
            np.asarray(vector, dtype=np.float64)
            for vector in (
                (1, 0, 0),
                (-1, 0, 0),
                (0, 1, 0),
                (0, -1, 0),
                (0, 0, 1),
                (0, 0, -1),
            )
        ]
    elif directions == "cube26":
        vectors = []
        for x in (-1, 0, 1):
            for y in (-1, 0, 1):
                for z in (-1, 0, 1):
                    vector = np.asarray([x, y, z], dtype=np.float64)
                    norm = float(np.linalg.norm(vector))
                    if norm > 0.0:
                        vectors.append(vector / norm)
    else:
        raise ValueError("directions must be axes or cube26")
    return [
        {
            "position_ecef": center + float(radius) * vector,
            "coverage_epochs": 0,
            "members": 0,
        }
        for radius in radii_m
        for vector in vectors
    ]


def _build_static_observations(
    data: dict[str, Any],
    run_dir: Path,
    start: int,
    end: int,
    approximate_position: np.ndarray,
    *,
    carrier_families: tuple[str, ...],
    pseudorange_family: str = "primary",
) -> tuple[list[DDCarrierEpoch | None], list[DDPseudorangeEpoch | None]]:
    cache = RinexObservationCache()
    systems = ("G", "E", "J", "C")
    carrier = DDCarrierComputer(
        run_dir / "base.obs",
        rover_obs_path=run_dir / "rover.obs",
        base_position=np.asarray(data["base_ecef"], dtype=np.float64),
        allowed_systems=systems,
        observation_cache=cache,
    )
    pseudorange = DDPseudorangeComputer(
        run_dir / "base.obs",
        rover_obs_path=run_dir / "rover.obs",
        base_position=np.asarray(data["base_ecef"], dtype=np.float64),
        allowed_systems=systems,
        observation_cache=cache,
        pseudorange_family=pseudorange_family,
    )
    dd_cp: list[DDCarrierEpoch | None] = []
    dd_pr: list[DDPseudorangeEpoch | None] = []
    for epoch in range(start, end):
        measurements = _build_dd_measurements(
            np.asarray(data["sat_ecef"][epoch], dtype=np.float64),
            np.asarray(data["system_ids"][epoch], dtype=np.int32),
            list(data["used_prns"][epoch]),
            np.asarray(data["weights"][epoch], dtype=np.float64),
            approximate_position,
            systems,
            min_elevation_deg=-90.0,
            min_snr=0.0,
            keep_best=0,
        )
        tow = float(data["times"][epoch])
        cp_result = (
            carrier.compute_dd_families(
                tow,
                measurements,
                rover_position_approx=approximate_position,
                min_common_sats=2,
                carrier_families=carrier_families,
            )
            if carrier_families
            else carrier.compute_dd(
                tow,
                measurements,
                rover_position_approx=approximate_position,
                min_common_sats=4,
            )
        )
        pr_result = pseudorange.compute_dd(
            tow,
            measurements,
            rover_position_approx=approximate_position,
            min_common_sats=4,
        )
        dd_cp.append(None if cp_result is None else DDCarrierEpoch.from_result(cp_result))
        dd_pr.append(None if pr_result is None else DDPseudorangeEpoch.from_result(pr_result))
    return dd_cp, dd_pr


def _assignment_integers(row: dict[str, str]) -> dict[tuple[str, str, int], int]:
    result: dict[tuple[str, str, int], int] = {}
    for ref_sat, sat_id, wavelength_nm, _generation, integer in json.loads(
        row["assignment_json"]
    ):
        result[(str(ref_sat), str(sat_id), int(wavelength_nm))] = int(integer)
    return result


def assigned_carrier_residual_stats(
    position_ecef: np.ndarray,
    rows_by_epoch: dict[int, list[dict[str, str]]],
    dd_carrier: list[DDCarrierEpoch | None],
    start: int,
    *,
    max_mode_distance_m: float = 0.5,
) -> dict[str, float | int]:
    """Score saved basin integers without independently re-rounding each DD row."""
    residuals: list[float] = []
    supporting_epochs = 0
    position = np.asarray(position_ecef, dtype=np.float64)
    for offset, obs in enumerate(dd_carrier):
        if obs is None or obs.sat_ids is None or obs.ref_sat_ids is None:
            continue
        rows = rows_by_epoch.get(start + offset, [])
        if not rows:
            continue
        row = min(rows, key=lambda item: float(np.linalg.norm(_position(item) - position)))
        if np.linalg.norm(_position(row) - position) > max_mode_distance_m:
            continue
        assignments = _assignment_integers(row)
        epoch_count = 0
        for index, (ref_sat, sat_id) in enumerate(zip(obs.ref_sat_ids, obs.sat_ids)):
            wavelength = float(obs.wavelengths_m[index])
            key = (str(ref_sat), str(sat_id), int(round(wavelength * 1e9)))
            integer = assignments.get(key)
            if integer is None:
                continue
            expected, _jacobian = _dd_expected_and_jacobian_m(
                position,
                obs.sat_ecef_k[index],
                obs.sat_ecef_ref[index],
                obs.base_range_k[index],
                obs.base_range_ref[index],
            )
            residuals.append(float(obs.dd_carrier_cycles[index] - expected / wavelength - integer))
            epoch_count += 1
        supporting_epochs += int(epoch_count > 0)
    if not residuals:
        return {
            "assigned_carrier_rows": 0,
            "assigned_carrier_epochs": 0,
            "assigned_carrier_rms_cycles": float("inf"),
            "assigned_carrier_median_abs_cycles": float("inf"),
        }
    values = np.asarray(residuals, dtype=np.float64)
    return {
        "assigned_carrier_rows": int(len(values)),
        "assigned_carrier_epochs": int(supporting_epochs),
        "assigned_carrier_rms_cycles": float(np.sqrt(np.mean(np.square(values)))),
        "assigned_carrier_median_abs_cycles": float(np.median(np.abs(values))),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    trace = _read_csv(args.basin_trace, start_epoch=args.start, end_epoch=args.end)
    positions_by_epoch: dict[int, list[np.ndarray]] = {}
    rows_by_epoch: dict[int, list[dict[str, str]]] = {}
    for row in trace:
        epoch = int(row["epoch"])
        if args.start <= epoch < args.end:
            positions_by_epoch.setdefault(epoch, []).append(_position(row))
            rows_by_epoch.setdefault(epoch, []).append(row)
    arrays = {epoch: np.asarray(rows) for epoch, rows in positions_by_epoch.items()}
    candidates = recurring_position_candidates(
        arrays,
        args.start,
        args.end,
        radius_m=args.radius_m,
        sample_stride_epochs=args.sample_stride_epochs,
        dedup_radius_m=args.dedup_radius_m,
        max_candidates=args.max_candidates,
    )
    recurring_candidate_count = len(candidates)
    seed_parent_candidate_id = None
    seed_center_source = None
    seed_radii: tuple[float, ...] = ()
    if args.seed_center_json is not None and args.seed_center_ecef:
        raise RuntimeError("use only one of --seed-center-json and --seed-center-ecef")
    seed_center = None
    if args.seed_center_json is not None:
        seed_result = json.loads(args.seed_center_json.read_text(encoding="utf-8"))
        if args.seed_center_candidate_id is None:
            seed_row = seed_result["candidates"][int(args.seed_center_candidate_index)]
        else:
            matching = [
                row
                for row in seed_result["candidates"]
                if int(row["candidate_id"]) == int(args.seed_center_candidate_id)
            ]
            if len(matching) != 1:
                raise RuntimeError(
                    "--seed-center-candidate-id must select exactly one candidate"
                )
            seed_row = matching[0]
        seed_parent_candidate_id = int(seed_row["candidate_id"])
        seed_center = np.asarray(seed_row["position_ecef"], dtype=np.float64)
        seed_center_source = "candidate_json"
    elif args.seed_center_ecef:
        seed_center = parse_ecef(args.seed_center_ecef)
        seed_center_source = "external_truth_free_ecef"
    if seed_center is not None:
        radii = tuple(
            float(value)
            for value in str(args.seed_radii_m).split(",")
            if value.strip()
        )
        seed_radii = radii
        if bool(getattr(args, "include_seed_center", False)):
            candidates.append(
                {
                    "position_ecef": seed_center.copy(),
                    "coverage_epochs": 0,
                    "members": 0,
                }
            )
        candidates.extend(
            offset_seed_candidates(
                seed_center,
                radii,
                directions=str(args.seed_directions),
            )
        )
    if not candidates:
        raise RuntimeError("no recurring or externally seeded candidates")
    for candidate_id, item in enumerate(candidates):
        item["source_candidate_id"] = candidate_id
        item["proposal_kind"] = (
            "recurring_mode"
            if candidate_id < recurring_candidate_count
            else "offset_seed"
        )
        item["parent_candidate_id"] = (
            None
            if candidate_id < recurring_candidate_count
            else seed_parent_candidate_id
        )
    requested_candidate_ids = {
        int(value)
        for value in str(args.candidate_ids).split(",")
        if value.strip()
    }
    if requested_candidate_ids:
        candidates = [
            item
            for item in candidates
            if int(item["source_candidate_id"]) in requested_candidate_ids
        ]
        if not candidates:
            raise RuntimeError("--candidate-ids selected no candidates")

    data = PPCDatasetLoader(args.data_dir).load_experiment_data(
        max_epochs=args.end,
        include_sat_velocity=True,
        systems=("G", "R", "E", "C", "J"),
    )
    families = tuple(x for x in args.carrier_families.split(",") if x)
    dd_cp, dd_pr = _build_static_observations(
        data,
        args.data_dir,
        args.start,
        args.end,
        np.asarray(candidates[0]["position_ecef"]),
        carrier_families=families,
    )
    cfg = StaticStopSegmentConfig(
        min_epochs=5,
        min_observations=args.min_observations,
        prior_sigma_m=args.prior_sigma_m,
        dd_pr_sigma_m=args.dd_pr_sigma_m,
        dd_cp_sigma_cycles=args.dd_cp_sigma_cycles,
        max_update_m=args.max_update_m,
    )
    truth = np.asarray(data["ground_truth"], dtype=np.float64)[args.start : args.end]
    truth_center = np.median(truth[np.isfinite(truth).all(axis=1)], axis=0)
    ranked: list[dict[str, Any]] = []
    for candidate_id, item in enumerate(candidates):
        initial = np.asarray(item["position_ecef"], dtype=np.float64)
        solve = solve_static_stop_segment(
            initial,
            dd_cp,
            dd_pr,
            [None] * len(dd_cp),
            cfg,
        )
        row = {
            "candidate_id": int(item.get("source_candidate_id", candidate_id)),
            "proposal_kind": str(item.get("proposal_kind", "recurring_mode")),
            "parent_candidate_id": item.get("parent_candidate_id"),
            "coverage_epochs": item["coverage_epochs"],
            "members": item["members"],
            "initial_error_m": float(np.linalg.norm(initial - truth_center)),
            "final_error_m": float(np.linalg.norm(solve.position_ecef - truth_center)),
            **asdict(solve),
            **assigned_carrier_residual_stats(
                solve.position_ecef,
                rows_by_epoch,
                dd_cp,
                args.start,
                max_mode_distance_m=args.assigned_mode_distance_m,
            ),
        }
        block_rms: list[float] = []
        if args.bootstrap_blocks > 1:
            boundaries = np.linspace(
                0, len(dd_cp), int(args.bootstrap_blocks) + 1, dtype=np.int64
            )
            for left, right in zip(boundaries[:-1], boundaries[1:]):
                block_solve = solve_static_stop_segment(
                    initial,
                    dd_cp[int(left) : int(right)],
                    dd_pr[int(left) : int(right)],
                    [None] * int(right - left),
                    cfg,
                )
                block_rms.append(float(block_solve.final_norm_rms))
        row["bootstrap_norm_rms"] = block_rms
        row["position_ecef"] = np.asarray(row["position_ecef"]).tolist()
        ranked.append(row)
    if args.bootstrap_blocks > 1:
        for row in ranked:
            row["bootstrap_wins"] = 0
        for block in range(int(args.bootstrap_blocks)):
            winner = min(
                ranked,
                key=lambda row: float(row["bootstrap_norm_rms"][block]),
            )
            winner["bootstrap_wins"] += 1
        for row in ranked:
            row["bootstrap_median_norm_rms"] = float(
                np.median(row["bootstrap_norm_rms"])
            )
    ranked.sort(key=lambda row: (float(row["final_norm_rms"]), float(row["final_cost"])))
    return {
        "segment": [args.start, args.end],
        "radius_m": args.radius_m,
        "carrier_families": list(families),
        "n_dd_cp_epochs": sum(item is not None for item in dd_cp),
        "n_dd_pr_epochs": sum(item is not None for item in dd_pr),
        "recurring_candidate_count": recurring_candidate_count,
        "seed_parent_candidate_id": seed_parent_candidate_id,
        "seed_center_source": seed_center_source,
        "seed_center_ecef": None if seed_center is None else seed_center.tolist(),
        "seed_radii_m": list(seed_radii),
        "seed_directions": str(args.seed_directions),
        "candidates": ranked,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--basin-trace", type=Path, required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--radius-m", type=float, default=0.20)
    parser.add_argument("--sample-stride-epochs", type=int, default=5)
    parser.add_argument("--dedup-radius-m", type=float, default=0.20)
    parser.add_argument("--max-candidates", type=int, default=24)
    parser.add_argument("--carrier-families", default="")
    parser.add_argument("--min-observations", type=int, default=40)
    parser.add_argument("--prior-sigma-m", type=float, default=20.0)
    parser.add_argument("--dd-pr-sigma-m", type=float, default=4.0)
    parser.add_argument("--dd-cp-sigma-cycles", type=float, default=0.5)
    parser.add_argument("--max-update-m", type=float, default=5.0)
    parser.add_argument("--assigned-mode-distance-m", type=float, default=0.5)
    parser.add_argument("--bootstrap-blocks", type=int, default=0)
    parser.add_argument("--seed-center-json", type=Path)
    parser.add_argument(
        "--seed-center-ecef",
        default="",
        help="Truth-free x,y,z ECEF center supplied by an independent estimator.",
    )
    parser.add_argument("--seed-center-candidate-index", type=int, default=0)
    parser.add_argument("--seed-center-candidate-id", type=int)
    parser.add_argument("--seed-radii-m", default="")
    parser.add_argument(
        "--include-seed-center",
        action="store_true",
        help="Include the exact external/parent center before its offset shell.",
    )
    parser.add_argument(
        "--seed-directions", choices=("axes", "cube26"), default="cube26"
    )
    parser.add_argument("--candidate-ids", default="")
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = run(args)
    encoded = json.dumps(result, indent=2, allow_nan=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
