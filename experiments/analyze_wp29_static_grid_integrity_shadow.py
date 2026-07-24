#!/usr/bin/env python3
"""Rank saved static-grid candidates with carrier satellite integrity scores."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "python"))
sys.path.insert(0, str(_REPO_ROOT / "experiments"))

from analyze_wp29_static_reanchor_shadow import _build_static_observations  # noqa: E402
from gnss_gpu.io.ppc import PPCDatasetLoader  # noqa: E402
from gnss_gpu.stop_segment_static import _dd_expected_and_jacobian_m  # noqa: E402


CarrierCostRow = tuple[float, str, str, int]
CarrierArcRow = tuple[float, str, str, int]


def _physical_satellite(value: str) -> str:
    return str(value).split("@", 1)[0]


def trimmed_satellite_mean(
    rows: list[CarrierCostRow], trim_satellites: int
) -> tuple[float, tuple[str, ...], int]:
    """Drop rows incident to the worst mean-cost physical satellites."""

    if not rows:
        return float("inf"), (), 0
    totals: dict[str, list[float]] = {}
    for cost, ref_sat, sat_id, _block in rows:
        for satellite in {_physical_satellite(ref_sat), _physical_satellite(sat_id)}:
            totals.setdefault(satellite, []).append(float(cost))
    ranked = sorted(
        totals,
        key=lambda satellite: float(np.mean(totals[satellite])),
        reverse=True,
    )
    excluded = tuple(ranked[: max(0, int(trim_satellites))])
    excluded_set = set(excluded)
    retained = [
        cost
        for cost, ref_sat, sat_id, _block in rows
        if _physical_satellite(ref_sat) not in excluded_set
        and _physical_satellite(sat_id) not in excluded_set
    ]
    if not retained:
        return float("inf"), excluded, 0
    return float(np.mean(retained)), excluded, len(retained)


def fixed_satellite_mean(
    rows: list[CarrierCostRow], excluded_satellites: tuple[str, ...]
) -> tuple[float, int]:
    excluded = {_physical_satellite(value) for value in excluded_satellites}
    retained = [
        cost
        for cost, ref_sat, sat_id, _block in rows
        if _physical_satellite(ref_sat) not in excluded
        and _physical_satellite(sat_id) not in excluded
    ]
    if not retained:
        return float("inf"), 0
    return float(np.mean(retained)), len(retained)


def temporal_arc_centered_score(
    rows: list[CarrierArcRow], *, min_samples: int, sigma_cycles: float
) -> tuple[float, float, int]:
    """Score within-arc carrier variation after removing circular offset."""

    grouped: dict[tuple[str, str], list[float]] = {}
    for residual, ref_sat, sat_id, _epoch in rows:
        grouped.setdefault((ref_sat, sat_id), []).append(float(residual))
    centered: list[float] = []
    used_arcs = 0
    for values_list in grouped.values():
        if len(values_list) < int(min_samples):
            continue
        values = np.asarray(values_list, dtype=np.float64)
        angles = 2.0 * np.pi * values
        center = np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles))) / (
            2.0 * np.pi
        )
        delta = np.mod(values - center + 0.5, 1.0) - 0.5
        centered.extend(delta.tolist())
        used_arcs += 1
    if not centered:
        return float("inf"), float("inf"), 0
    array = np.asarray(centered, dtype=np.float64)
    cauchy = float(np.mean(np.log1p(np.square(array / float(sigma_cycles)))))
    return cauchy, float(np.median(np.abs(array))), used_arcs


def temporal_window_scores(
    rows: list[CarrierArcRow],
    *,
    n_epochs: int,
    n_windows: int,
    min_samples: int,
    sigma_cycles: float,
) -> tuple[float, ...]:
    """Score arc stability independently in fixed causal time windows."""

    scores: list[float] = []
    for window in range(int(n_windows)):
        selected = [
            row
            for row in rows
            if min(
                int(row[3]) * int(n_windows) // max(int(n_epochs), 1),
                int(n_windows) - 1,
            )
            == window
        ]
        score, _median, _arcs = temporal_arc_centered_score(
            selected,
            min_samples=int(min_samples),
            sigma_cycles=float(sigma_cycles),
        )
        scores.append(score)
    return tuple(scores)


def _candidate_carrier_rows(
    position_ecef: np.ndarray,
    dd_cp: list[Any],
    *,
    sigma_cycles: float,
    n_blocks: int,
) -> tuple[list[CarrierCostRow], np.ndarray, list[CarrierArcRow]]:
    rows: list[CarrierCostRow] = []
    arc_rows: list[CarrierArcRow] = []
    residuals: list[float] = []
    n_epochs = len(dd_cp)
    for epoch_index, obs in enumerate(dd_cp):
        if obs is None or obs.sat_ids is None or obs.ref_sat_ids is None:
            continue
        dd = np.asarray(obs.dd_carrier_cycles, dtype=np.float64).ravel()
        weights = (
            np.ones(len(dd), dtype=np.float64)
            if obs.weights is None
            else np.asarray(obs.weights, dtype=np.float64).ravel()
        )
        block = min(int(epoch_index * n_blocks / max(n_epochs, 1)), n_blocks - 1)
        for index in range(len(dd)):
            expected, _jacobian = _dd_expected_and_jacobian_m(
                position_ecef,
                np.asarray(obs.sat_ecef_k[index], dtype=np.float64),
                np.asarray(obs.sat_ecef_ref[index], dtype=np.float64),
                float(obs.base_range_k[index]),
                float(obs.base_range_ref[index]),
            )
            wavelength = float(obs.wavelengths_m[index])
            raw = float(dd[index] - expected / wavelength)
            residual = raw - float(np.round(raw))
            scaled = residual * np.sqrt(max(float(weights[index]), 1.0e-3))
            cost = float(np.log1p((scaled / float(sigma_cycles)) ** 2))
            rows.append(
                (cost, str(obs.ref_sat_ids[index]), str(obs.sat_ids[index]), block)
            )
            arc_rows.append(
                (
                    residual,
                    str(obs.ref_sat_ids[index]),
                    str(obs.sat_ids[index]),
                    epoch_index,
                )
            )
            residuals.append(abs(residual))
    return rows, np.asarray(residuals, dtype=np.float64), arc_rows


def filter_candidate_ids(
    candidates: list[dict[str, Any]], candidate_ids: str
) -> list[dict[str, Any]]:
    requested = {
        int(value) for value in str(candidate_ids).split(",") if value.strip()
    }
    if not requested:
        return candidates
    filtered = [row for row in candidates if int(row["candidate_id"]) in requested]
    if {int(row["candidate_id"]) for row in filtered} != requested:
        raise RuntimeError("--candidate-ids contains an absent candidate")
    return filtered


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    source = json.loads(args.candidates_json.read_text(encoding="utf-8"))
    candidates = filter_candidate_ids(
        list(source["candidates"]), str(getattr(args, "candidate_ids", ""))
    )
    if not candidates:
        raise RuntimeError("candidate result is empty")
    data = PPCDatasetLoader(args.data_dir).load_experiment_data(
        max_epochs=int(args.end),
        include_sat_velocity=True,
        systems=("G", "R", "E", "C", "J"),
    )
    families = tuple(value for value in str(args.carrier_families).split(",") if value)
    dd_cp, _dd_pr = _build_static_observations(
        data,
        args.data_dir,
        int(args.start),
        int(args.end),
        np.asarray(candidates[0]["position_ecef"], dtype=np.float64),
        carrier_families=families,
    )
    epoch_modulus = int(getattr(args, "epoch_modulus", 1))
    epoch_remainder = int(getattr(args, "epoch_remainder", 0))
    if epoch_modulus > 1:
        dd_cp = [
            obs
            if (int(args.start) + index) % epoch_modulus == epoch_remainder
            else None
            for index, obs in enumerate(dd_cp)
        ]
    output: list[dict[str, Any]] = []
    fixed_exclusions = tuple(
        value for value in str(args.fixed_exclude_satellites).split(",") if value
    )
    temporal_sigma_grid = tuple(
        float(value)
        for value in str(args.temporal_arc_sigma_grid).split(",")
        if value.strip()
    )
    temporal_min_grid = tuple(
        int(value)
        for value in str(args.temporal_arc_min_samples_grid).split(",")
        if value.strip()
    )
    grid_score_names: list[str] = []
    for minimum in temporal_min_grid:
        for sigma in temporal_sigma_grid:
            grid_score_names.append(
                f"carrier_temporal_m{minimum}_s{int(round(sigma * 1000)):04d}"
            )
    for candidate in candidates:
        rows, residuals, arc_rows = _candidate_carrier_rows(
            np.asarray(candidate["position_ecef"], dtype=np.float64),
            dd_cp,
            sigma_cycles=float(args.sigma_cycles),
            n_blocks=int(args.blocks),
        )
        block_means = [
            float(np.mean([row[0] for row in rows if row[3] == block]))
            for block in range(int(args.blocks))
            if any(row[3] == block for row in rows)
        ]
        row: dict[str, Any] = {
            "candidate_id": int(candidate["candidate_id"]),
            "position_ecef": np.asarray(
                candidate["position_ecef"], dtype=np.float64
            ).tolist(),
            "final_error_m": float(candidate.get("final_error_m", float("nan"))),
            "n_carrier_rows": len(rows),
            "carrier_cauchy_mean": float(np.mean([item[0] for item in rows])),
            "carrier_median_abs_cycles": float(np.median(residuals)),
            "carrier_p90_abs_cycles": float(np.quantile(residuals, 0.9)),
            "carrier_block_max_mean": float(max(block_means)),
            "carrier_block_std_mean": float(np.std(block_means)),
        }
        for trim in (1, 2, 3):
            cost, excluded, retained = trimmed_satellite_mean(rows, trim)
            row[f"carrier_trim{trim}_mean"] = cost
            row[f"carrier_trim{trim}_excluded"] = list(excluded)
            row[f"carrier_trim{trim}_rows"] = retained
        fixed_cost, fixed_rows = fixed_satellite_mean(rows, fixed_exclusions)
        row["carrier_fixed_exclusion_mean"] = fixed_cost
        row["carrier_fixed_exclusion_rows"] = fixed_rows
        temporal_cost, temporal_median, temporal_arcs = temporal_arc_centered_score(
            arc_rows,
            min_samples=int(args.temporal_arc_min_samples),
            sigma_cycles=float(args.temporal_arc_sigma_cycles),
        )
        row["carrier_temporal_arc_cauchy_mean"] = temporal_cost
        row["carrier_temporal_arc_median_abs_cycles"] = temporal_median
        row["carrier_temporal_arcs"] = temporal_arcs
        window_scores = temporal_window_scores(
            arc_rows,
            n_epochs=len(dd_cp),
            n_windows=int(args.blocks),
            min_samples=int(args.temporal_window_min_samples),
            sigma_cycles=float(args.temporal_window_sigma_cycles),
        )
        row["carrier_temporal_window_scores"] = list(window_scores)
        finite_windows = bool(np.isfinite(np.asarray(window_scores)).all())
        row["carrier_temporal_window_mean"] = (
            float(np.mean(window_scores)) if finite_windows else float("inf")
        )
        row["carrier_temporal_window_max"] = (
            float(np.max(window_scores)) if finite_windows else float("inf")
        )
        row["carrier_temporal_window_std"] = (
            float(np.std(window_scores)) if finite_windows else float("inf")
        )
        for minimum in temporal_min_grid:
            for sigma in temporal_sigma_grid:
                key = f"carrier_temporal_m{minimum}_s{int(round(sigma * 1000)):04d}"
                score, _median, arcs = temporal_arc_centered_score(
                    arc_rows, min_samples=minimum, sigma_cycles=sigma
                )
                row[key] = score
                row[f"{key}_arcs"] = arcs
        output.append(row)
    score_names = [
        "carrier_cauchy_mean",
        "carrier_trim1_mean",
        "carrier_trim2_mean",
        "carrier_trim3_mean",
        "carrier_median_abs_cycles",
        "carrier_p90_abs_cycles",
        "carrier_block_max_mean",
        "carrier_block_std_mean",
        "carrier_fixed_exclusion_mean",
        "carrier_temporal_arc_cauchy_mean",
        "carrier_temporal_arc_median_abs_cycles",
        "carrier_temporal_window_mean",
        "carrier_temporal_window_max",
        "carrier_temporal_window_std",
        *grid_score_names,
    ]
    for score_name in score_names:
        ranked = sorted(range(len(output)), key=lambda index: output[index][score_name])
        for rank, index in enumerate(ranked, start=1):
            output[index][f"{score_name}_rank"] = rank
    return {
        "segment": [int(args.start), int(args.end)],
        "carrier_families": list(families),
        "sigma_cycles": float(args.sigma_cycles),
        "blocks": int(args.blocks),
        "fixed_exclude_satellites": list(fixed_exclusions),
        "temporal_arc_min_samples": int(args.temporal_arc_min_samples),
        "temporal_arc_sigma_cycles": float(args.temporal_arc_sigma_cycles),
        "temporal_window_min_samples": int(args.temporal_window_min_samples),
        "temporal_window_sigma_cycles": float(args.temporal_window_sigma_cycles),
        "temporal_arc_sigma_grid": list(temporal_sigma_grid),
        "temporal_arc_min_samples_grid": list(temporal_min_grid),
        "candidates": output,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidates_json", type=Path)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--carrier-families", default="")
    parser.add_argument("--candidate-ids", default="")
    parser.add_argument("--epoch-modulus", type=int, default=1)
    parser.add_argument("--epoch-remainder", type=int, default=0)
    parser.add_argument("--sigma-cycles", type=float, default=0.5)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--fixed-exclude-satellites", default="")
    parser.add_argument("--temporal-arc-min-samples", type=int, default=5)
    parser.add_argument("--temporal-arc-sigma-cycles", type=float, default=0.1)
    parser.add_argument("--temporal-window-min-samples", type=int, default=3)
    parser.add_argument("--temporal-window-sigma-cycles", type=float, default=0.01)
    parser.add_argument(
        "--temporal-arc-sigma-grid", default="0.01,0.03,0.05,0.1,0.2,0.5"
    )
    parser.add_argument(
        "--temporal-arc-min-samples-grid", default="5,10,20,40"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.epoch_modulus < 1 or not 0 <= args.epoch_remainder < args.epoch_modulus:
        parser.error("epoch split must satisfy modulus >= 1 and 0 <= remainder < modulus")
    result = analyze(args)
    result["epoch_modulus"] = int(args.epoch_modulus)
    result["epoch_remainder"] = int(args.epoch_remainder)
    result["candidate_source_sha256"] = hashlib.sha256(
        args.candidates_json.read_bytes()
    ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
