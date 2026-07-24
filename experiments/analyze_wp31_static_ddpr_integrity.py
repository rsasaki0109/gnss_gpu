#!/usr/bin/env python3
"""Rank static candidates with satellite-robust absolute DD pseudorange."""

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

from analyze_wp29_static_reanchor_shadow import _build_static_observations  # noqa: E402
from build_wp31_static_anchor_imu_route import _accepted_anchor  # noqa: E402
from run_wp29_tdcp_anchor_smoother import (  # noqa: E402
    _load_static_position_override,
)
from gnss_gpu.io.ppc import PPCDatasetLoader  # noqa: E402
from gnss_gpu.stop_segment_static import _dd_expected_and_jacobian_m  # noqa: E402


def ddpr_scores(
    position: np.ndarray,
    observations: list[Any],
    *,
    sigma_m: float,
    blocks: int,
    pair_bias_m: dict[tuple[str, str], float] | None = None,
) -> dict[str, Any]:
    rows: list[tuple[float, float, str, int]] = []
    n_epochs = len(observations)
    for epoch, obs in enumerate(observations):
        if obs is None:
            continue
        sat_ids = obs.sat_ids or tuple(f"row_{index}" for index in range(obs.n))
        ref_ids = obs.ref_sat_ids or tuple("unknown_ref" for _ in range(obs.n))
        for index in range(obs.n):
            pair = (str(sat_ids[index]), str(ref_ids[index]))
            if pair_bias_m is not None and pair not in pair_bias_m:
                continue
            expected, _jac = _dd_expected_and_jacobian_m(
                position, obs.sat_ecef_k[index], obs.sat_ecef_ref[index],
                obs.base_range_k[index], obs.base_range_ref[index],
            )
            residual = float(obs.dd_pseudorange_m[index] - expected)
            if pair_bias_m is not None:
                residual -= float(pair_bias_m[pair])
            cost = float(np.log1p((residual / sigma_m) ** 2))
            block = min(blocks - 1, int(epoch * blocks / max(n_epochs, 1)))
            rows.append((cost, abs(residual), str(sat_ids[index]), block))
    if not rows:
        return {
            "ddpr_rows": 0,
            "ddpr_cauchy_mean": float("inf"),
            "ddpr_median_abs_m": float("inf"),
            "ddpr_trim1_mean": float("inf"),
            "ddpr_trim2_mean": float("inf"),
            "ddpr_block_std_mean": float("inf"),
        }
    satellites = sorted({row[2] for row in rows})
    satellite_cost = {
        satellite: float(np.mean([row[0] for row in rows if row[2] == satellite]))
        for satellite in satellites
    }
    ranked_bad = sorted(satellites, key=lambda sat: satellite_cost[sat], reverse=True)
    result: dict[str, Any] = {
        "ddpr_rows": len(rows),
        "ddpr_satellites": len(satellites),
        "ddpr_cauchy_mean": float(np.mean([row[0] for row in rows])),
        "ddpr_median_abs_m": float(np.median([row[1] for row in rows])),
    }
    for trim in (1, 2):
        excluded = set(ranked_bad[:trim])
        retained = [row[0] for row in rows if row[2] not in excluded]
        result[f"ddpr_trim{trim}_mean"] = float(np.mean(retained)) if retained else float("inf")
        result[f"ddpr_trim{trim}_excluded"] = sorted(excluded)
    block_means = [
        float(np.mean([row[0] for row in rows if row[3] == block]))
        for block in range(blocks)
        if any(row[3] == block for row in rows)
    ]
    result["ddpr_block_std_mean"] = float(np.std(block_means))
    return result


def ddpr_pair_biases(
    position: np.ndarray,
    observations: list[Any],
    *,
    min_samples: int = 5,
) -> dict[tuple[str, str], float]:
    """Calibrate persistent DD-code pair biases at an accepted anchor."""

    values: dict[tuple[str, str], list[float]] = {}
    for obs in observations:
        if obs is None:
            continue
        sat_ids = obs.sat_ids or tuple(f"row_{index}" for index in range(obs.n))
        ref_ids = obs.ref_sat_ids or tuple("unknown_ref" for _ in range(obs.n))
        for index in range(obs.n):
            expected, _jac = _dd_expected_and_jacobian_m(
                position,
                obs.sat_ecef_k[index],
                obs.sat_ecef_ref[index],
                obs.base_range_k[index],
                obs.base_range_ref[index],
            )
            pair = (str(sat_ids[index]), str(ref_ids[index]))
            values.setdefault(pair, []).append(
                float(obs.dd_pseudorange_m[index] - expected)
            )
    return {
        pair: float(np.median(rows))
        for pair, rows in values.items()
        if len(rows) >= int(min_samples)
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidates_json", type=Path)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--sigma-m", type=float, default=4.0)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument(
        "--pseudorange-family",
        choices=("primary", "secondary", "tertiary"),
        default="primary",
    )
    parser.add_argument("--calibration-static", type=Path)
    parser.add_argument("--calibration-fusion", type=Path)
    parser.add_argument("--calibration-position-anchor", type=Path)
    parser.add_argument("--calibration-min-samples", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.candidates_json.read_text(encoding="utf-8"))
    candidates = list(source["candidates"])
    if (args.calibration_static is None) != (args.calibration_fusion is None):
        parser.error("calibration requires both --calibration-static and --calibration-fusion")
    if args.calibration_position_anchor is not None and args.calibration_static is not None:
        parser.error(
            "calibration position anchor cannot be combined with static/fusion calibration"
        )
    calibration_anchor = None
    if args.calibration_position_anchor is not None:
        try:
            calibration_anchor = _load_static_position_override(
                args.calibration_position_anchor
            )
        except RuntimeError as error:
            parser.error(str(error))
    elif args.calibration_static is not None:
        try:
            calibration_anchor = _accepted_anchor(
                args.calibration_static, args.calibration_fusion
            )
        except RuntimeError as error:
            parser.error(str(error))
    max_epochs = max(
        args.end,
        0 if calibration_anchor is None else int(calibration_anchor[1]),
    )
    data = PPCDatasetLoader(args.data_dir).load_experiment_data(
        max_epochs=max_epochs, include_sat_velocity=True, systems=("G", "R", "E", "C", "J")
    )
    _cp, ddpr = _build_static_observations(
        data, args.data_dir, args.start, args.end,
        np.asarray(candidates[0]["position_ecef"], dtype=np.float64),
        carrier_families=(),
        pseudorange_family=args.pseudorange_family,
    )
    pair_bias_m = None
    calibration_evidence_epochs = 0
    if calibration_anchor is not None:
        cal_start, cal_end, cal_position, _candidate_id, _reason = calibration_anchor
        _cal_cp, cal_ddpr = _build_static_observations(
            data,
            args.data_dir,
            int(cal_start),
            int(cal_end),
            np.asarray(cal_position, dtype=np.float64),
            carrier_families=(),
            pseudorange_family=args.pseudorange_family,
        )
        pair_bias_m = ddpr_pair_biases(
            np.asarray(cal_position, dtype=np.float64),
            cal_ddpr,
            min_samples=args.calibration_min_samples,
        )
        calibration_evidence_epochs = sum(obs is not None for obs in cal_ddpr)
        if len(pair_bias_m) < 4:
            parser.error("calibration has fewer than four supported DD-code pairs")
    rows = []
    for candidate in candidates:
        rows.append(
            {
                "candidate_id": int(candidate["candidate_id"]),
                "position_ecef": candidate["position_ecef"],
                "final_error_m": float(candidate.get("final_error_m", float("nan"))),
                **ddpr_scores(
                    np.asarray(candidate["position_ecef"], dtype=np.float64),
                    ddpr,
                    sigma_m=args.sigma_m,
                    blocks=args.blocks,
                    pair_bias_m=pair_bias_m,
                ),
            }
        )
    metrics = (
        "ddpr_cauchy_mean", "ddpr_median_abs_m", "ddpr_trim1_mean",
        "ddpr_trim2_mean", "ddpr_block_std_mean",
    )
    for metric in metrics:
        for rank, index in enumerate(sorted(range(len(rows)), key=lambda i: rows[i][metric]), start=1):
            rows[index][f"{metric}_rank"] = rank
    result = {
        "schema": (
            "wp32_static_calibrated_ddpr_integrity_v1"
            if pair_bias_m is not None
            else (
                "wp32_static_secondary_ddpr_integrity_v1"
                if args.pseudorange_family == "secondary"
                else (
                    "wp36_static_tertiary_ddpr_integrity_v1"
                    if args.pseudorange_family == "tertiary"
                    else "wp31_static_ddpr_integrity_v1"
                )
            )
        ),
        "segment": [args.start, args.end],
        "production_input_truth": False,
        "sigma_m": args.sigma_m,
        "blocks": args.blocks,
        "pseudorange_family": args.pseudorange_family,
        "evidence_epochs": sum(obs is not None for obs in ddpr),
        "calibration": None
        if calibration_anchor is None
        else {
            "segment": [int(calibration_anchor[0]), int(calibration_anchor[1])],
            "candidate_id": int(calibration_anchor[3]),
            "reason": str(calibration_anchor[4]),
            "evidence_epochs": int(calibration_evidence_epochs),
            "min_pair_samples": int(args.calibration_min_samples),
            "supported_pairs": len(pair_bias_m),
            **(
                {
                    "position_anchor_sha256": hashlib.sha256(
                        args.calibration_position_anchor.read_bytes()
                    ).hexdigest()
                }
                if args.calibration_position_anchor is not None
                else {
                    "static_sha256": hashlib.sha256(
                        args.calibration_static.read_bytes()
                    ).hexdigest(),
                    "fusion_sha256": hashlib.sha256(
                        args.calibration_fusion.read_bytes()
                    ).hexdigest(),
                }
            ),
        },
        "candidate_source_sha256": hashlib.sha256(args.candidates_json.read_bytes()).hexdigest(),
        "candidates": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**result, "candidates": f"{len(rows)} candidates"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
