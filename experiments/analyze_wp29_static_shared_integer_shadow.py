"""Refine saved static-grid modes with shared multi-epoch carrier integers."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "python"))
sys.path.insert(0, str(_REPO_ROOT / "experiments"))

from analyze_wp29_static_reanchor_shadow import _build_static_observations  # noqa: E402
from gnss_gpu.io.ppc import PPCDatasetLoader  # noqa: E402
from gnss_gpu.static_shared_integer import (  # noqa: E402
    SharedIntegerConfig,
    solve_static_shared_integers,
)


def run(args: argparse.Namespace) -> dict[str, object]:
    source = json.loads(args.candidates_json.read_text(encoding="utf-8"))
    candidates = list(source["candidates"])
    if not candidates:
        raise RuntimeError("candidate result is empty")
    requested_ids = {
        int(value) for value in str(args.candidate_ids).split(",") if value.strip()
    }
    if requested_ids:
        candidates = [
            row for row in candidates if int(row["candidate_id"]) in requested_ids
        ]
        if not candidates:
            raise RuntimeError("--candidate-ids selected no candidates")
    data = PPCDatasetLoader(args.data_dir).load_experiment_data(
        max_epochs=args.end,
        include_sat_velocity=True,
        systems=("G", "R", "E", "C", "J"),
    )
    approximate = np.asarray(candidates[0]["position_ecef"], dtype=np.float64)
    families = tuple(value for value in args.carrier_families.split(",") if value)
    dd_cp, dd_pr = _build_static_observations(
        data,
        args.data_dir,
        args.start,
        args.end,
        approximate,
        carrier_families=families,
    )
    carrier_sigmas = tuple(
        float(value)
        for value in (args.carrier_sigma_grid or str(args.carrier_sigma_cycles)).split(",")
        if value.strip()
    )
    dd_pr_sigmas = tuple(
        float(value)
        for value in (args.dd_pr_sigma_grid or str(args.dd_pr_sigma_m)).split(",")
        if value.strip()
    )
    prior_sigmas = tuple(
        float(value)
        for value in (args.prior_sigma_grid or str(args.prior_sigma_m)).split(",")
        if value.strip()
    )
    ambiguity_models = tuple(
        value for value in args.ambiguity_models.split(",") if value.strip()
    )
    configs = [
        SharedIntegerConfig(
            ambiguity_model=ambiguity_model,
            carrier_sigma_cycles=carrier_sigma,
            dd_pr_sigma_m=dd_pr_sigma,
            prior_sigma_m=prior_sigma,
            min_arc_samples=args.min_arc_samples,
            max_epoch_gap=args.max_epoch_gap,
            slip_threshold_cycles=args.slip_threshold_cycles,
            min_carrier_rows=args.min_carrier_rows,
            max_update_m=args.max_update_m,
        )
        for ambiguity_model in ambiguity_models
        for carrier_sigma in carrier_sigmas
        for dd_pr_sigma in dd_pr_sigmas
        for prior_sigma in prior_sigmas
    ]
    truth = np.asarray(data["ground_truth"], dtype=np.float64)[args.start : args.end]
    valid_truth = truth[np.isfinite(truth).all(axis=1)]
    truth_center = np.median(valid_truth, axis=0)
    rows: list[dict[str, object]] = []
    for config_id, config in enumerate(configs):
        for candidate in candidates:
            initial = np.asarray(candidate["position_ecef"], dtype=np.float64)
            solve = solve_static_shared_integers(initial, dd_cp, dd_pr, config)
            row = asdict(solve)
            row["candidate_id"] = int(candidate["candidate_id"])
            row["config_id"] = config_id
            row["config"] = asdict(config)
            row["source_final_norm_rms"] = float(candidate.get("final_norm_rms", float("nan")))
            row["initial_error_m"] = float(np.linalg.norm(initial - truth_center))
            row["final_error_m"] = float(np.linalg.norm(solve.position_ecef - truth_center))
            row["position_ecef"] = solve.position_ecef.tolist()
            rows.append(row)
    rows.sort(key=lambda row: (float(row["final_cost"]), float(row["carrier_rms_cycles"])))
    for rank, row in enumerate(rows, start=1):
        row["shared_integer_rank"] = rank
    return {
        "segment": [args.start, args.end],
        "carrier_families": list(families),
        "n_dd_cp_epochs": sum(obs is not None for obs in dd_cp),
        "n_dd_pr_epochs": sum(obs is not None for obs in dd_pr),
        "configs": [asdict(config) for config in configs],
        "candidates": rows,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidates_json", type=Path)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--carrier-families", default="L1_E1_B1,L5_E5A_B2A")
    parser.add_argument("--carrier-sigma-cycles", type=float, default=0.08)
    parser.add_argument("--dd-pr-sigma-m", type=float, default=4.0)
    parser.add_argument("--prior-sigma-m", type=float, default=20.0)
    parser.add_argument("--min-arc-samples", type=int, default=3)
    parser.add_argument("--max-epoch-gap", type=int, default=10)
    parser.add_argument("--slip-threshold-cycles", type=float, default=0.75)
    parser.add_argument("--min-carrier-rows", type=int, default=30)
    parser.add_argument("--max-update-m", type=float, default=8.0)
    parser.add_argument("--candidate-ids", default="")
    parser.add_argument("--carrier-sigma-grid", default="")
    parser.add_argument("--dd-pr-sigma-grid", default="")
    parser.add_argument("--prior-sigma-grid", default="")
    parser.add_argument(
        "--ambiguity-models", default="exact_pair", help="comma-separated models"
    )
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
