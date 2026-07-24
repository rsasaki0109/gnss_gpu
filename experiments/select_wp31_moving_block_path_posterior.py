#!/usr/bin/env python3
"""Track moving-block offset modes from a trusted anchor without runtime FGO."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import logsumexp


def _load_pool(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "wp31_moving_block_truth_free_local_pool_v1":
        raise ValueError(f"unexpected pool schema: {path}")
    if payload.get("production_input_truth") is not False:
        raise ValueError(f"non-truth-free pool: {path}")
    return payload


def _measurement_states(
    rows: list[dict[str, Any]], baseline_ddpr_m: float, *,
    state_cell_m: float, max_carrier_rms_cycles: float, max_ddpr_ratio: float,
) -> list[dict[str, Any]]:
    by_cell: dict[tuple[int, int, int], dict[str, Any]] = {}
    for row in rows:
        ratio = float(row["ddpr_rms_m"]) / baseline_ddpr_m
        if (
            int(row["integer_arcs"]) < 4
            or int(row["retained_carrier_rows"]) < 8
            or float(row["carrier_rms_cycles"]) > max_carrier_rms_cycles
            or ratio > max_ddpr_ratio
        ):
            continue
        offset = np.asarray(row["offset_ecef_m"], dtype=np.float64)
        cell = tuple(np.rint(offset / state_cell_m).astype(np.int64).tolist())
        emission_cost = (
            np.square(float(row["carrier_rms_cycles"]) / max_carrier_rms_cycles)
            + np.square(ratio / max_ddpr_ratio)
        )
        state = {
            "offset_ecef_m": offset,
            "carrier_rms_cycles": float(row["carrier_rms_cycles"]),
            "ddpr_ratio": ratio,
            "emission_cost": float(emission_cost),
            "integer_signature": {
                str(key): int(value) for key, value in row.get("integer_signature", {}).items()
            },
        }
        if cell not in by_cell or emission_cost < by_cell[cell]["emission_cost"]:
            by_cell[cell] = state
    return list(by_cell.values())


def select_path(manifest: dict[str, Any]) -> dict[str, Any]:
    cfg = manifest.get("config", {})
    state_cell_m = float(cfg.get("state_cell_m", 0.5))
    max_carrier = float(cfg.get("max_carrier_rms_cycles", 0.30))
    max_ddpr_ratio = float(cfg.get("max_ddpr_ratio", 0.80))
    drift_radius_per_epoch = float(cfg.get("drift_radius_per_epoch", 0.08))
    transition_base_m = float(cfg.get("transition_base_m", 0.75))
    transition_sigma_m = float(cfg.get("transition_sigma_m", 0.75))
    emission_weight = float(cfg.get("emission_weight", 1.0))
    min_gamma = float(cfg.get("min_gamma", 0.99))
    min_shared_integer_arcs = int(cfg.get("min_shared_integer_arcs", 0))
    max_integer_disagreements = int(cfg.get("max_integer_disagreements", 0))
    integer_tolerance_cycles = int(cfg.get("integer_tolerance_cycles", 0))
    anchor_epoch = int(manifest["anchor_epoch"])
    anchor = np.asarray(manifest["anchor_offset_ecef_m"], dtype=np.float64)
    previous_offsets = anchor[None, :]
    previous_scores = np.asarray([0.0])
    previous_epoch = anchor_epoch
    previous_signatures: list[dict[str, int]] = [{}]
    histories: list[dict[str, Any]] = []
    backpointers: list[np.ndarray] = []
    state_offsets: list[np.ndarray] = []
    for block_index, block in enumerate(manifest["blocks"]):
        pool = _load_pool(Path(block["pool_path"]))
        start, end = (int(value) for value in pool["segment"])
        states = _measurement_states(
            pool["candidates"], float(block["baseline_ddpr_rms_m"]),
            state_cell_m=state_cell_m, max_carrier_rms_cycles=max_carrier,
            max_ddpr_ratio=max_ddpr_ratio,
        )
        if not states:
            return {
                "schema": "wp31_moving_block_path_posterior_v1",
                "production_input_truth": False, "selection_reason": "empty_measurement_state_set",
                "failed_block_index": block_index, "blocks": histories,
            }
        offsets = np.asarray([state["offset_ecef_m"] for state in states])
        tree = cKDTree(previous_offsets)
        state_epoch = (start + end - 1) // 2
        delta_epochs = max(state_epoch - previous_epoch, 1)
        max_radius = transition_base_m + drift_radius_per_epoch * delta_epochs
        neighborhoods = tree.query_ball_point(offsets, max_radius)
        scores = np.full(len(states), -np.inf); parents = np.full(len(states), -1, dtype=np.int64)
        for index, prior_indices in enumerate(neighborhoods):
            if not prior_indices:
                continue
            retained_prior = []
            current_signature = states[index]["integer_signature"]
            for prior_index in prior_indices:
                if block_index == 0:
                    retained_prior.append(prior_index)
                    continue
                prior_signature = previous_signatures[prior_index]
                shared = set(current_signature) & set(prior_signature)
                disagreements = sum(
                    abs(current_signature[key] - prior_signature[key]) > integer_tolerance_cycles
                    for key in shared
                )
                if len(shared) >= min_shared_integer_arcs and disagreements <= max_integer_disagreements:
                    retained_prior.append(prior_index)
            if not retained_prior:
                continue
            prior_indices_array = np.asarray(retained_prior, dtype=np.int64)
            distances = np.linalg.norm(previous_offsets[prior_indices_array] - offsets[index], axis=1)
            transition = -np.log1p(np.square(distances / transition_sigma_m))
            candidates = previous_scores[prior_indices_array] + transition
            winner = int(np.argmax(candidates))
            parents[index] = int(prior_indices_array[winner])
            scores[index] = float(candidates[winner] - emission_weight * states[index]["emission_cost"])
        reachable = np.isfinite(scores)
        if not np.any(reachable):
            return {
                "schema": "wp31_moving_block_path_posterior_v1",
                "production_input_truth": False, "selection_reason": "no_reachable_state",
                "failed_block_index": block_index, "blocks": histories,
            }
        offsets = offsets[reachable]; scores = scores[reachable]; parents = parents[reachable]
        kept_states = [state for state, keep in zip(states, reachable) if keep]
        log_norm = float(logsumexp(scores)); probabilities = np.exp(scores - log_norm)
        order = np.argsort(scores)[::-1]; best = int(order[0])
        histories.append({
            "segment": [start, end], "input_candidates": len(pool["candidates"]),
            "measurement_states": len(states), "reachable_states": len(scores),
            "max_transition_radius_m": max_radius,
            "selected_offset_ecef_m": offsets[best].tolist(),
            "selected_carrier_rms_cycles": kept_states[best]["carrier_rms_cycles"],
            "selected_ddpr_ratio": kept_states[best]["ddpr_ratio"],
            "posterior_gamma": float(probabilities[best]),
            "runner_up_log_margin": float(scores[best] - scores[int(order[1])]) if len(order) > 1 else float("inf"),
        })
        backpointers.append(parents); state_offsets.append(offsets)
        previous_offsets = offsets; previous_scores = scores - float(np.max(scores)); previous_epoch = state_epoch
        previous_signatures = [state["integer_signature"] for state in kept_states]
    final_index = int(np.argmax(previous_scores)); path = []
    for block_index in range(len(backpointers) - 1, -1, -1):
        path.append(state_offsets[block_index][final_index].tolist())
        final_index = int(backpointers[block_index][final_index])
    path.reverse()
    all_confident = all(block["posterior_gamma"] >= min_gamma for block in histories)
    return {
        "schema": "wp31_moving_block_path_posterior_v1",
        "production_input_truth": False, "truth_usage": "none",
        "config": {
            "state_cell_m": state_cell_m, "max_carrier_rms_cycles": max_carrier,
            "max_ddpr_ratio": max_ddpr_ratio, "drift_radius_per_epoch": drift_radius_per_epoch,
            "transition_base_m": transition_base_m, "transition_sigma_m": transition_sigma_m,
            "emission_weight": emission_weight, "min_gamma": min_gamma,
            "min_shared_integer_arcs": min_shared_integer_arcs,
            "max_integer_disagreements": max_integer_disagreements,
            "integer_tolerance_cycles": integer_tolerance_cycles,
        },
        "selection_reason": "all_block_posteriors_confident" if all_confident else "posterior_not_confident",
        "declaration_eligible": all_confident,
        "selected_path_offsets_ecef_m": path, "blocks": histories,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    result = select_path(manifest); args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
