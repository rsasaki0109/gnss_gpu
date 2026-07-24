#!/usr/bin/env python3
"""Rank moving offset hypotheses with bias-eliminated trifrequency DDPR."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
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
from refine_wp31_moving_block_ambiguity import phase_epochs  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _trajectory(path: Path, start: int, end: int) -> dict[int, np.ndarray]:
    with path.open(newline="", encoding="utf-8-sig") as fh:
        return {
            int(row["epoch"]): np.asarray(
                [float(row["ecef_x"]), float(row["ecef_y"]), float(row["ecef_z"])],
                dtype=np.float64,
            )
            for row in csv.DictReader(fh)
            if start <= int(row["epoch"]) < end
        }


def pair_centered_metrics(
    rows: list[tuple[int, str, str, float]], *, min_pair_epochs: int = 3
) -> dict[str, float | int]:
    """Remove one robust constant per exact DD pair and score temporal residuals."""

    grouped: dict[tuple[str, str], list[tuple[int, float]]] = defaultdict(list)
    for epoch, ref_sat, sat_id, residual in rows:
        grouped[(ref_sat, sat_id)].append((int(epoch), float(residual)))
    centered: list[float] = []
    retained_epochs: set[int] = set()
    retained_pairs = 0
    for values in grouped.values():
        if len({epoch for epoch, _value in values}) < int(min_pair_epochs):
            continue
        center = float(np.median([value for _epoch, value in values]))
        centered.extend(value - center for _epoch, value in values)
        retained_epochs.update(epoch for epoch, _value in values)
        retained_pairs += 1
    values = np.asarray(centered, dtype=np.float64)
    return {
        "temporal_rows": int(values.size),
        "temporal_pairs": retained_pairs,
        "temporal_epochs": len(retained_epochs),
        "temporal_rms_m": (
            float(np.sqrt(np.mean(np.square(values)))) if values.size else float("inf")
        ),
        "temporal_median_abs_m": (
            float(np.median(np.abs(values))) if values.size else float("inf")
        ),
    }


def select_rank_consensus(
    family_rows: dict[str, list[dict[str, Any]]],
    *,
    min_temporal_epochs: int = 10,
    min_temporal_pairs: int = 3,
    min_temporal_rows: int = 30,
    max_family_rank_fraction: float = 0.2,
    min_runner_margin: float = 0.2,
    min_integer_arcs: int = 4,
    min_carrier_rows: int = 24,
    max_carrier_rms_cycles: float = 0.5,
    max_block_spread_m: float = 0.5,
) -> dict[str, Any]:
    families = ("primary", "secondary", "tertiary")
    mappings = {
        family: {int(row["candidate_id"]): row for row in family_rows[family]}
        for family in families
    }
    candidate_ids = set(mappings["primary"])
    if len(candidate_ids) < 2 or any(
        set(mappings[family]) != candidate_ids for family in families[1:]
    ):
        raise ValueError("temporal DDPR candidate sets differ or are too small")
    rank_limit = int(math.ceil(len(candidate_ids) * max_family_rank_fraction))
    ranks: dict[str, dict[int, int]] = {}
    for family in families:
        ordered = sorted(
            mappings[family].values(),
            key=lambda row: (
                float(row["temporal_median_abs_m"]),
                int(row["candidate_id"]),
            ),
        )
        ranks[family] = {
            int(row["candidate_id"]): rank for rank, row in enumerate(ordered, start=1)
        }
    combined = []
    for candidate_id in sorted(candidate_ids):
        metrics = {family: mappings[family][candidate_id] for family in families}
        family_ranks = {family: ranks[family][candidate_id] for family in families}
        supply_pass = all(
            int(metrics[family]["temporal_epochs"]) >= min_temporal_epochs
            and int(metrics[family]["temporal_pairs"]) >= min_temporal_pairs
            and int(metrics[family]["temporal_rows"]) >= min_temporal_rows
            for family in families
        ) and (
            int(metrics["primary"].get("integer_arcs", min_integer_arcs))
            >= min_integer_arcs
            and int(metrics["primary"].get("carrier_rows", min_carrier_rows))
            >= min_carrier_rows
            and float(
                metrics["primary"].get("carrier_rms_cycles", max_carrier_rms_cycles)
            )
            <= max_carrier_rms_cycles
            and float(metrics["primary"].get("block_spread_m", max_block_spread_m))
            <= max_block_spread_m
        )
        combined.append(
            {
                "candidate_id": candidate_id,
                "offset_ecef_m": metrics["primary"]["offset_ecef_m"],
                "family_ranks": family_ranks,
                "family_temporal_rms_m": {
                    family: float(metrics[family]["temporal_rms_m"])
                    for family in families
                },
                "family_temporal_median_abs_m": {
                    family: float(metrics[family]["temporal_median_abs_m"])
                    for family in families
                },
                "rank_sum": sum(family_ranks.values()),
                "supply_pass": supply_pass,
            }
        )
    combined.sort(key=lambda row: (row["rank_sum"], row["candidate_id"]))
    winner, runner = combined[:2]
    runner_margin = (runner["rank_sum"] - winner["rank_sum"]) / winner["rank_sum"]
    accepted = (
        winner["supply_pass"]
        and max(winner["family_ranks"].values()) <= rank_limit
        and runner_margin >= min_runner_margin
    )
    return {
        "selected_candidate_id": winner["candidate_id"] if accepted else None,
        "reason": (
            "unique_moving_temporal_trifrequency_ddpr_rank_consensus"
            if accepted
            else "moving_temporal_trifrequency_ddpr_rank_gate_failed"
        ),
        "winner": winner,
        "runner": runner,
        "runner_margin": runner_margin,
        "family_rank_limit": rank_limit,
        "candidate_count": len(combined),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate_json", type=Path)
    parser.add_argument("trajectory", type=Path)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--min-pair-epochs", type=int, default=3)
    parser.add_argument("--min-temporal-epochs", type=int, default=10)
    parser.add_argument("--min-temporal-pairs", type=int, default=3)
    parser.add_argument("--min-temporal-rows", type=int, default=30)
    parser.add_argument("--max-family-rank-fraction", type=float, default=0.2)
    parser.add_argument("--min-runner-margin", type=float, default=0.2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = json.loads(args.candidate_json.read_text(encoding="utf-8"))
    start, end = (int(value) for value in source["segment"])
    selected_stride_phase = int(
        source.get("selected_stride_phase", start % args.stride)
    )
    if not 0 <= selected_stride_phase < args.stride:
        raise ValueError("candidate source stride phase is invalid")
    route = _trajectory(args.trajectory, start, end)
    sanitized = [
        {
            "candidate_id": int(row["seed_id"]),
            "offset_ecef_m": [float(value) for value in row["offset_ecef_m"]],
            "integer_arcs": int(row["integer_arcs"]),
            "carrier_rows": int(row["carrier_rows"]),
            "carrier_rms_cycles": float(row["carrier_rms_cycles"]),
            "block_spread_m": float(row["block_spread_m"]),
            "block_offsets_ecef_m": [
                [float(value) for value in offset]
                for offset in row["block_offsets_ecef_m"]
            ],
        }
        for row in source["hypotheses"]
    ]
    candidate_hash = hashlib.sha256(
        json.dumps(sanitized, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    data = PPCDatasetLoader(args.data_dir).load_experiment_data(
        max_epochs=end, systems=("G", "R", "E", "C", "J")
    )
    systems = ("G", "E", "J", "C")
    cache = RinexObservationCache()
    engines = {
        family: DDPseudorangeComputer(
            args.data_dir / "base.obs",
            rover_obs_path=args.data_dir / "rover.obs",
            base_position=np.asarray(data["base_ecef"]),
            allowed_systems=systems,
            observation_cache=cache,
            pseudorange_family=family,
        )
        for family in ("primary", "secondary", "tertiary")
    }
    observations: dict[str, dict[int, DDPseudorangeEpoch]] = {
        family: {} for family in engines
    }
    for epoch in phase_epochs(start, end, args.stride, selected_stride_phase):
        approximate = route.get(epoch)
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
        for family, engine in engines.items():
            result = engine.compute_dd(
                float(data["times"][epoch]),
                measurements,
                rover_position_approx=approximate,
                min_common_sats=2,
            )
            if result is not None:
                observations[family][epoch] = DDPseudorangeEpoch.from_result(result)

    family_rows: dict[str, list[dict[str, Any]]] = {family: [] for family in engines}
    for family, by_epoch in observations.items():
        for candidate in sanitized:
            offset = np.asarray(candidate["offset_ecef_m"], dtype=np.float64)
            residual_rows: list[tuple[int, str, str, float]] = []
            for epoch, obs in by_epoch.items():
                position = route[epoch] + offset
                for index, (ref_sat, sat_id) in enumerate(
                    zip(obs.ref_sat_ids or (), obs.sat_ids or ())
                ):
                    expected, _ = _dd_expected_and_jacobian_m(
                        position,
                        obs.sat_ecef_k[index],
                        obs.sat_ecef_ref[index],
                        obs.base_range_k[index],
                        obs.base_range_ref[index],
                    )
                    residual_rows.append(
                        (
                            epoch,
                            ref_sat,
                            sat_id,
                            float(obs.dd_pseudorange_m[index]) - expected,
                        )
                    )
            family_rows[family].append(
                {
                    **candidate,
                    **pair_centered_metrics(
                        residual_rows, min_pair_epochs=args.min_pair_epochs
                    ),
                }
            )
    selection = select_rank_consensus(
        family_rows,
        min_temporal_epochs=args.min_temporal_epochs,
        min_temporal_pairs=args.min_temporal_pairs,
        min_temporal_rows=args.min_temporal_rows,
        max_family_rank_fraction=args.max_family_rank_fraction,
        min_runner_margin=args.min_runner_margin,
    )
    selected_audit = None
    if selection["selected_candidate_id"] is not None:
        selected = next(
            row
            for row in sanitized
            if row["candidate_id"] == selection["selected_candidate_id"]
        )
        truth = np.asarray(data["ground_truth"])
        errors = [
            np.linalg.norm(
                route[epoch] + np.asarray(selected["offset_ecef_m"]) - truth[epoch]
            )
            for epoch in sorted(route)
        ]
        selected_audit = {
            "median_error_m": float(np.median(errors)),
            "sub50cm_epochs": int(np.count_nonzero(np.asarray(errors) < 0.5)),
            "epochs": len(errors),
        }
    result = {
        "schema": "wp42_moving_temporal_trifrequency_ddpr_v1",
        "production_input_truth": False,
        "production_promoted": False,
        "truth_usage": "selected_candidate_post_selection_audit_only",
        "segment": [start, end],
        "candidate_source_sha256": candidate_hash,
        "input_sha256": {
            "candidate_json": _sha256(args.candidate_json),
            "trajectory": _sha256(args.trajectory),
        },
        "config": {
            "rank_metric": "pair_centered_temporal_median_abs_m",
            "stride": args.stride,
            "selected_stride_phase": selected_stride_phase,
            "min_pair_epochs": args.min_pair_epochs,
            "min_temporal_epochs": args.min_temporal_epochs,
            "min_temporal_pairs": args.min_temporal_pairs,
            "min_temporal_rows": args.min_temporal_rows,
            "max_family_rank_fraction": args.max_family_rank_fraction,
            "min_runner_margin": args.min_runner_margin,
        },
        "family_observation_epochs": {
            family: len(rows) for family, rows in observations.items()
        },
        "family_candidates": family_rows,
        **selection,
        "selected_audit": selected_audit,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
