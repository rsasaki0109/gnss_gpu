#!/usr/bin/env python3
"""Truth-free per-satellite DDPR screen via triple-difference clustering.

For each evidence epoch and constellation, double-difference pseudorange
residuals are recomputed at the *given* trajectory positions (no truth
input).  Triple differences between non-reference satellite pairs sharing
the same epoch/system/reference cancel the reference satellite; an edge is
drawn between two satellites when the resulting triple difference is below
``--edge-m``.  The largest connected component per epoch/system is treated
as the inlier cluster and every other satellite in that group is an
epoch-level outlier.  After all evidence epochs are processed, a satellite
is flagged if it was an epoch-outlier in at least ``--frac-thresh`` of the
epochs in which it appeared.

Truth-free: no ground-truth or reference-trajectory data is loaded anywhere
in this script.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "python"))
sys.path.insert(0, str(_ROOT / "experiments"))

from exp_wp23b_basin_ar import _build_dd_measurements  # noqa: E402
from gnss_gpu.dd_pseudorange import DDPseudorangeComputer  # noqa: E402
from gnss_gpu.io.ppc import PPCDatasetLoader  # noqa: E402
from gnss_gpu.io.rinex_cache import RinexObservationCache  # noqa: E402
from gnss_gpu.local_fgo import DDPseudorangeEpoch  # noqa: E402
from gnss_gpu.stop_segment_static import _dd_expected_and_jacobian_m  # noqa: E402

SYSTEMS = ("G", "E", "J", "C")


def phase_epochs(start: int, end: int, stride: int, phase: int) -> range:
    if stride <= 0 or not 0 <= phase < stride:
        raise ValueError("evidence stride and phase are invalid")
    first = start + ((phase - start) % stride)
    return range(first, end, stride)


def _read_trajectory(path: Path, start: int, end: int) -> dict[int, np.ndarray]:
    with path.open(newline="", encoding="utf-8-sig") as fh:
        return {
            int(row["epoch"]): np.asarray(
                [float(row["ecef_x"]), float(row["ecef_y"]), float(row["ecef_z"])],
                dtype=np.float64,
            )
            for row in csv.DictReader(fh)
            if start <= int(row["epoch"]) < end
        }


def epoch_outliers(residual_map: dict[str, float], edge_m: float) -> set[str]:
    """Flag outlier satellites within one (epoch, system) DD residual group.

    ``residual_map`` maps non-reference satellite id to its DD pseudorange
    residual (m) for a single epoch/system/reference group.  Triple
    differences TD(a, b) = residual[a] - residual[b] cancel the shared
    reference satellite.  An edge connects ``a`` and ``b`` when
    ``abs(TD(a, b)) < edge_m``.  The largest connected component is the
    inlier cluster; every other satellite in the group is an outlier.
    """
    sats = list(residual_map.keys())
    n = len(sats)
    if n < 2:
        return set()
    adjacency: dict[str, set[str]] = defaultdict(set)
    for i in range(n):
        for j in range(i + 1, n):
            triple_diff = residual_map[sats[i]] - residual_map[sats[j]]
            if abs(triple_diff) < edge_m:
                adjacency[sats[i]].add(sats[j])
                adjacency[sats[j]].add(sats[i])
    visited: set[str] = set()
    components: list[set[str]] = []
    for sat in sats:
        if sat in visited:
            continue
        component: set[str] = set()
        stack = [sat]
        visited.add(sat)
        while stack:
            current = stack.pop()
            component.add(current)
            for neighbor in adjacency[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        components.append(component)
    components.sort(key=lambda component: -len(component))
    inliers = components[0] if components else set()
    return set(sats) - inliers


def aggregate_flags(
    per_epoch_outliers: list[dict[str, Any]],
    frac_thresh: float,
) -> tuple[set[str], dict[str, dict[str, float]]]:
    """Aggregate per-(epoch, system) outlier groups into satellite flags.

    ``per_epoch_outliers`` is a list of records, one per (epoch, system)
    group: ``{"present": iterable of sat ids in the group, "outliers":
    iterable of sat ids flagged outlier in that group}``.  A satellite is
    flagged if its outlier fraction across all epochs where it appeared is
    ``>= frac_thresh``.
    """
    epochs_present: dict[str, int] = defaultdict(int)
    epochs_outlier: dict[str, int] = defaultdict(int)
    for group in per_epoch_outliers:
        for sat in group["present"]:
            epochs_present[sat] += 1
        for sat in group["outliers"]:
            epochs_outlier[sat] += 1
    flagged: set[str] = set()
    stats: dict[str, dict[str, float]] = {}
    for sat, present in epochs_present.items():
        outlier = epochs_outlier.get(sat, 0)
        fraction = outlier / present if present else 0.0
        stats[sat] = {
            "epochs_present": present,
            "epochs_outlier": outlier,
            "outlier_fraction": fraction,
        }
        if fraction >= frac_thresh:
            flagged.add(sat)
    return flagged, stats


def collect_residuals(
    pr_engine: DDPseudorangeComputer,
    data: dict[str, Any],
    epochs: list[int],
    trajectory: dict[int, np.ndarray],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for epoch in epochs:
        if epoch >= len(data["times"]):
            continue
        position = trajectory.get(epoch)
        if position is None:
            continue
        measurements = _build_dd_measurements(
            np.asarray(data["sat_ecef"][epoch]),
            np.asarray(data["system_ids"][epoch]),
            list(data["used_prns"][epoch]),
            np.asarray(data["weights"][epoch]),
            position,
            SYSTEMS,
            min_elevation_deg=-90.0,
            min_snr=0.0,
            keep_best=0,
        )
        result = pr_engine.compute_dd(
            float(data["times"][epoch]),
            measurements,
            rover_position_approx=position,
            min_common_sats=4,
        )
        if result is None:
            continue
        obs = DDPseudorangeEpoch.from_result(result)
        for index in range(obs.n):
            expected, _ = _dd_expected_and_jacobian_m(
                position,
                obs.sat_ecef_k[index],
                obs.sat_ecef_ref[index],
                obs.base_range_k[index],
                obs.base_range_ref[index],
            )
            residual = float(obs.dd_pseudorange_m[index]) - expected
            sat_id = obs.sat_ids[index] if obs.sat_ids else "?"
            rows.append({"epoch": epoch, "sat_id": sat_id, "residual_m": residual})
    return rows


def screen_satellites(
    rows: list[dict[str, Any]],
    edge_m: float,
    frac_thresh: float,
) -> tuple[set[str], dict[str, dict[str, float]]]:
    groups: dict[tuple[int, str], dict[str, float]] = defaultdict(dict)
    for row in rows:
        system = row["sat_id"][0] if row["sat_id"] != "?" else "?"
        groups[(row["epoch"], system)][row["sat_id"]] = row["residual_m"]
    per_epoch_outliers = []
    for residual_map in groups.values():
        outliers = epoch_outliers(residual_map, edge_m)
        per_epoch_outliers.append(
            {"present": set(residual_map.keys()), "outliers": outliers}
        )
    return aggregate_flags(per_epoch_outliers, frac_thresh)


def median_abs_residuals(rows: list[dict[str, Any]]) -> dict[str, float]:
    per_sat: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        per_sat[row["sat_id"]].append(row["residual_m"])
    return {
        sat: float(np.median(np.abs(np.asarray(values))))
        for sat, values in per_sat.items()
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    trajectory_bytes = args.trajectory.read_bytes()
    trajectory_sha256 = hashlib.sha256(trajectory_bytes).hexdigest()
    trajectory = _read_trajectory(args.trajectory, args.start, args.end)

    data = PPCDatasetLoader(args.data_dir).load_experiment_data(
        max_epochs=args.end,
        include_sat_velocity=False,
        systems=("G", "R", "E", "C", "J"),
    )
    cache = RinexObservationCache()
    pr_engine = DDPseudorangeComputer(
        args.data_dir / "base.obs",
        rover_obs_path=args.data_dir / "rover.obs",
        base_position=np.asarray(data["base_ecef"]),
        allowed_systems=SYSTEMS,
        observation_cache=cache,
        pseudorange_family="primary",
    )

    epochs = list(phase_epochs(args.start, args.end, args.stride, args.stride_phase))
    rows = collect_residuals(pr_engine, data, epochs, trajectory)
    evidence_epochs = len({row["epoch"] for row in rows})

    flagged, stats = screen_satellites(rows, args.edge_m, args.frac_thresh)
    medians = median_abs_residuals(rows)

    per_satellite = [
        {
            "sat": sat,
            "epochs_present": values["epochs_present"],
            "epochs_outlier": values["epochs_outlier"],
            "outlier_fraction": values["outlier_fraction"],
            "median_abs_residual_m": medians.get(sat, float("nan")),
        }
        for sat, values in stats.items()
    ]
    per_satellite.sort(key=lambda row: row["sat"])

    return {
        "schema": "wp158_ddpr_satellite_screen_v1",
        "production_input_truth": False,
        "truth_usage": "none",
        "segment": [args.start, args.end],
        "stride": args.stride,
        "stride_phase": args.stride_phase,
        "edge_m": args.edge_m,
        "frac_thresh": args.frac_thresh,
        "evidence_epochs": evidence_epochs,
        "per_satellite": per_satellite,
        "flagged_satellites": sorted(flagged),
        "input_sha256": {"trajectory": trajectory_sha256},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--stride-phase", type=int, required=True)
    parser.add_argument("--edge-m", type=float, default=5.0)
    parser.add_argument("--frac-thresh", type=float, default=0.2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
