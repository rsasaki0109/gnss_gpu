#!/usr/bin/env python3
"""Shadow-rank static-grid candidates with fixed wide-lane DD ranges."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "python"))
sys.path.insert(0, str(_REPO_ROOT / "experiments"))

from exp_wp23b_basin_ar import _build_dd_measurements  # noqa: E402
from gnss_gpu.io.ppc import PPCDatasetLoader  # noqa: E402
from gnss_gpu.stop_segment_static import _dd_expected_and_jacobian_m  # noqa: E402
from gnss_gpu.widelane import WidelaneDDPseudorangeComputer  # noqa: E402


def widelane_residual_scores(
    residuals_m: Sequence[float], *, sigma_m: float
) -> dict[str, float]:
    values = np.asarray(residuals_m, dtype=np.float64)
    if values.size == 0:
        return {
            "widelane_rms_m": float("inf"),
            "widelane_median_abs_m": float("inf"),
            "widelane_cauchy_mean": float("inf"),
        }
    return {
        "widelane_rms_m": float(np.sqrt(np.mean(np.square(values)))),
        "widelane_median_abs_m": float(np.median(np.abs(values))),
        "widelane_cauchy_mean": float(
            np.mean(np.log1p(np.square(values / float(sigma_m))))
        ),
    }


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    source = json.loads(args.candidates_json.read_text(encoding="utf-8"))
    candidates = list(source["candidates"])
    if not candidates:
        raise RuntimeError("candidate result is empty")
    data = PPCDatasetLoader(args.data_dir).load_experiment_data(
        max_epochs=int(args.end),
        systems=("G", "R", "E", "C", "J"),
    )
    approximate = np.asarray(candidates[0]["position_ecef"], dtype=np.float64)
    computer = WidelaneDDPseudorangeComputer(
        args.data_dir / "base.obs",
        args.data_dir / "rover.obs",
        base_position=np.asarray(data["base_ecef"], dtype=np.float64),
        allowed_systems=("G", "J"),
        min_epochs=int(args.min_epochs),
        max_std_cycles=float(args.max_std_cycles),
        ratio_threshold=float(args.ratio_threshold),
        min_fix_rate=float(args.min_fix_rate),
    )
    observations: list[Any] = []
    evidence_epochs = 0
    candidate_pairs = 0
    fixed_pairs = 0
    for epoch in range(int(args.end)):
        measurements = _build_dd_measurements(
            np.asarray(data["sat_ecef"][epoch], dtype=np.float64),
            np.asarray(data["system_ids"][epoch], dtype=np.int32),
            list(data["used_prns"][epoch]),
            np.asarray(data["weights"][epoch], dtype=np.float64),
            approximate,
            ("G", "E", "J", "C"),
            min_elevation_deg=-90.0,
            min_snr=0.0,
            keep_best=0,
        )
        dd, stats = computer.compute_dd(
            float(data["times"][epoch]),
            measurements,
            rover_position_approx=approximate,
            min_common_sats=4,
            rover_weights=np.asarray(data["weights"][epoch], dtype=np.float64),
        )
        if epoch < int(args.start):
            continue
        epoch_modulus = int(getattr(args, "epoch_modulus", 1))
        epoch_remainder = int(getattr(args, "epoch_remainder", 0))
        if epoch_modulus > 1 and epoch % epoch_modulus != epoch_remainder:
            continue
        candidate_pairs += int(stats.n_candidate_pairs)
        fixed_pairs += int(stats.n_fixed_pairs)
        if dd is not None:
            observations.append(dd)
            evidence_epochs += 1

    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        position = np.asarray(candidate["position_ecef"], dtype=np.float64)
        residuals: list[float] = []
        for obs in observations:
            measured = np.asarray(obs.dd_pseudorange_m, dtype=np.float64).ravel()
            for index in range(len(measured)):
                expected, _jacobian = _dd_expected_and_jacobian_m(
                    position,
                    np.asarray(obs.sat_ecef_k[index], dtype=np.float64),
                    np.asarray(obs.sat_ecef_ref[index], dtype=np.float64),
                    float(obs.base_range_k[index]),
                    float(obs.base_range_ref[index]),
                )
                residuals.append(float(measured[index] - expected))
        rows.append(
            {
                "candidate_id": int(candidate["candidate_id"]),
                "final_error_m": float(candidate.get("final_error_m", float("nan"))),
                "n_widelane_rows": len(residuals),
                **widelane_residual_scores(
                    residuals, sigma_m=float(args.residual_sigma_m)
                ),
            }
        )
    score_names = (
        "widelane_rms_m",
        "widelane_median_abs_m",
        "widelane_cauchy_mean",
    )
    for score_name in score_names:
        ranked = sorted(range(len(rows)), key=lambda index: rows[index][score_name])
        for rank, index in enumerate(ranked, start=1):
            rows[index][f"{score_name}_rank"] = rank
    return {
        "segment": [int(args.start), int(args.end)],
        "evidence_epochs": evidence_epochs,
        "candidate_pairs": candidate_pairs,
        "fixed_pairs": fixed_pairs,
        "min_epochs": int(args.min_epochs),
        "max_std_cycles": float(args.max_std_cycles),
        "ratio_threshold": float(args.ratio_threshold),
        "min_fix_rate": float(args.min_fix_rate),
        "residual_sigma_m": float(args.residual_sigma_m),
        "epoch_modulus": int(getattr(args, "epoch_modulus", 1)),
        "epoch_remainder": int(getattr(args, "epoch_remainder", 0)),
        "candidates": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidates_json", type=Path)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--min-epochs", type=int, default=5)
    parser.add_argument("--max-std-cycles", type=float, default=0.75)
    parser.add_argument("--ratio-threshold", type=float, default=3.0)
    parser.add_argument("--min-fix-rate", type=float, default=0.3)
    parser.add_argument("--residual-sigma-m", type=float, default=1.0)
    parser.add_argument("--epoch-modulus", type=int, default=1)
    parser.add_argument("--epoch-remainder", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.epoch_modulus < 1 or not 0 <= args.epoch_remainder < args.epoch_modulus:
        parser.error("epoch split must satisfy modulus >= 1 and 0 <= remainder < modulus")
    result = analyze(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
