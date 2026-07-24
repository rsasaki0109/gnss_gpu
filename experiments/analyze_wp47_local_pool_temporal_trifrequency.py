#!/usr/bin/env python3
"""Audit a truth-free OSM local pool with temporal trifrequency DDPR ranks."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "python"))
sys.path.insert(0, str(_ROOT / "experiments"))

from analyze_wp42_moving_temporal_trifrequency_ddpr import (  # noqa: E402
    _trajectory,
    pair_centered_metrics,
    select_rank_consensus,
)
from exp_wp23b_basin_ar import _build_dd_measurements  # noqa: E402
from gnss_gpu.dd_pseudorange import DDPseudorangeComputer  # noqa: E402
from gnss_gpu.io.ppc import PPCDatasetLoader  # noqa: E402
from gnss_gpu.io.rinex_cache import RinexObservationCache  # noqa: E402
from gnss_gpu.local_fgo import DDPseudorangeEpoch  # noqa: E402
from gnss_gpu.stop_segment_static import _dd_expected_and_jacobian_m  # noqa: E402
from refine_wp31_moving_block_ambiguity import phase_epochs  # noqa: E402


def sanitize_pool(source: dict[str, Any]) -> list[dict[str, Any]]:
    if source.get("schema") != "wp31_moving_block_truth_free_local_pool_v1":
        raise ValueError("unsupported local candidate pool")
    if bool(source.get("production_input_truth", True)):
        raise ValueError("local candidate pool is not truth-free")
    return [
        {
            "candidate_id": index,
            "parent_road_seed": int(row["parent_road_seed"]),
            "local_delta_xyh_m": [float(value) for value in row["local_delta_xyh_m"]],
            "offset_ecef_m": [float(value) for value in row["offset_ecef_m"]],
            "integer_arcs": int(row["integer_arcs"]),
            "carrier_rows": int(row["retained_carrier_rows"]),
            "carrier_rms_cycles": float(row["carrier_rms_cycles"]),
            "proposal_score": float(row["proposal_score"]),
        }
        for index, row in enumerate(source["candidates"])
    ]


def compact_selection(selection: dict[str, Any]) -> dict[str, Any]:
    return {
        key: selection[key]
        for key in (
            "selected_candidate_id",
            "reason",
            "winner",
            "runner",
            "runner_margin",
            "family_rank_limit",
            "candidate_count",
        )
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pool_json", type=Path)
    parser.add_argument("trajectory", type=Path)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--stride-phase", type=int, required=True)
    parser.add_argument("--min-pair-epochs", type=int, default=3)
    parser.add_argument("--min-temporal-epochs", type=int, default=10)
    parser.add_argument("--min-temporal-pairs", type=int, default=3)
    parser.add_argument("--min-temporal-rows", type=int, default=30)
    parser.add_argument("--max-family-rank-fraction", type=float, default=0.2)
    parser.add_argument("--min-runner-margin", type=float, default=0.2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.pool_json.read_text(encoding="utf-8"))
    candidates = sanitize_pool(source)
    start, end = (int(value) for value in source["segment"])
    route = _trajectory(args.trajectory, start, end)
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
    for epoch in phase_epochs(start, end, args.stride, args.stride_phase):
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
        for candidate in candidates:
            offset = np.asarray(candidate["offset_ecef_m"])
            residual_rows = []
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
    kwargs = {
        "min_temporal_epochs": args.min_temporal_epochs,
        "min_temporal_pairs": args.min_temporal_pairs,
        "min_temporal_rows": args.min_temporal_rows,
        "max_family_rank_fraction": args.max_family_rank_fraction,
        "min_runner_margin": args.min_runner_margin,
    }
    global_selection = select_rank_consensus(family_rows, **kwargs)
    parent_selections = []
    parent_ids = sorted({row["parent_road_seed"] for row in candidates})
    for parent_id in parent_ids:
        ids = {
            row["candidate_id"]
            for row in candidates
            if row["parent_road_seed"] == parent_id
        }
        subset = {
            family: [row for row in rows if row["candidate_id"] in ids]
            for family, rows in family_rows.items()
        }
        selection = compact_selection(select_rank_consensus(subset, **kwargs))
        selection["parent_road_seed"] = parent_id
        parent_selections.append(selection)
    frozen_ids = {
        selection["selected_candidate_id"]
        for selection in [global_selection, *parent_selections]
        if selection["selected_candidate_id"] is not None
    }
    truth = np.asarray(data["ground_truth"])
    audits = {}
    for candidate_id in sorted(frozen_ids):
        candidate = candidates[candidate_id]
        errors = np.asarray(
            [
                np.linalg.norm(
                    route[epoch]
                    + np.asarray(candidate["offset_ecef_m"])
                    - truth[epoch]
                )
                for epoch in sorted(route)
            ]
        )
        audits[str(candidate_id)] = {
            "median_error_m": float(np.median(errors)),
            "sub50cm_epochs": int(np.count_nonzero(errors < 0.5)),
            "epochs": len(errors),
        }
    result = {
        "schema": "wp47_local_pool_temporal_trifrequency_audit_v1",
        "production_input_truth": False,
        "production_promoted": False,
        "truth_usage": "frozen_rank_winners_post_selection_audit_only",
        "stability_confirmation_required": True,
        "segment": [start, end],
        "candidate_count": len(candidates),
        "pool_sha256": hashlib.sha256(args.pool_json.read_bytes()).hexdigest(),
        "config": {
            "stride": args.stride,
            "stride_phase": args.stride_phase,
            **kwargs,
        },
        "global_selection": compact_selection(global_selection),
        "parent_selections": parent_selections,
        "frozen_winner_audits": audits,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
