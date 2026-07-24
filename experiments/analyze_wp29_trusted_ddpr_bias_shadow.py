#!/usr/bin/env python3
"""Shadow-score saved basins with DD code biases learned at trusted FIX epochs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "python"))
sys.path.insert(0, str(_REPO_ROOT / "experiments"))

from exp_wp23b_basin_ar import _build_dd_measurements  # noqa: E402
from gnss_gpu.dd_pseudorange import DDPseudorangeComputer  # noqa: E402
from gnss_gpu.dd_quality import dd_pseudorange_residuals_m  # noqa: E402
from gnss_gpu.io.ppc import PPCDatasetLoader  # noqa: E402


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - float(np.max(values))
    weights = np.exp(np.clip(shifted, -745.0, 0.0))
    return weights / float(np.sum(weights))


def _bias_corrected_costs(
    residuals: np.ndarray,
    ref_sat_ids: tuple[str, ...],
    sat_ids: tuple[str, ...],
    satellite_biases: dict[str, tuple[float, int]],
    *,
    epoch: int,
    max_age_epochs: int,
    scale_m: float,
) -> tuple[np.ndarray, int]:
    retained: list[int] = []
    expected: list[float] = []
    for index, (ref_sat, sat_id) in enumerate(zip(ref_sat_ids, sat_ids)):
        ref = satellite_biases.get(ref_sat)
        sat = satellite_biases.get(sat_id)
        if ref is None or sat is None:
            continue
        if epoch - ref[1] > max_age_epochs or epoch - sat[1] > max_age_epochs:
            continue
        retained.append(index)
        expected.append(sat[0] - ref[0])
    if not retained:
        return np.zeros(residuals.shape[0], dtype=np.float64), 0
    centered = residuals[:, retained] - np.asarray(expected)[None, :]
    return np.mean(np.log1p(np.square(centered / scale_m)), axis=1), len(retained)


def _update_satellite_biases(
    satellite_biases: dict[str, tuple[float, int]],
    residuals: np.ndarray,
    ref_sat_ids: tuple[str, ...],
    sat_ids: tuple[str, ...],
    *,
    epoch: int,
    blend_alpha: float,
) -> None:
    for residual, ref_sat, sat_id in zip(residuals, ref_sat_ids, sat_ids):
        ref_bias = satellite_biases.get(ref_sat, (0.0, epoch))[0]
        observed = ref_bias + float(residual)
        previous = satellite_biases.get(sat_id)
        value = (
            observed
            if previous is None
            else (1.0 - blend_alpha) * previous[0] + blend_alpha * observed
        )
        satellite_biases[ref_sat] = (ref_bias, epoch)
        satellite_biases[sat_id] = (value, epoch)


def analyze(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_epoch: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in _read_csv(args.basin_trace):
        by_epoch[int(row["epoch"])].append(row)
    diagnostics = {int(row["epoch"]): row for row in _read_csv(args.epoch_diagnostics)}
    max_epoch = max(by_epoch)
    data = PPCDatasetLoader(args.data_dir).load_experiment_data(
        max_epochs=max_epoch + 1,
        systems=("G", "R", "E", "C", "J"),
    )
    computer = DDPseudorangeComputer(
        args.data_dir / "base.obs",
        rover_obs_path=args.data_dir / "rover.obs",
        base_position=np.asarray(data["base_ecef"], dtype=np.float64),
        allowed_systems=("G", "E", "J", "C"),
        interpolate_base_epochs=bool(args.interpolate_base_epochs),
    )
    satellite_biases: dict[str, tuple[float, int]] = {}
    trusted_position: np.ndarray | None = None
    trusted_epoch = -10**9
    output: list[dict[str, Any]] = []
    trusted_updates = 0
    self_updates = 0
    scored_epochs = 0
    for epoch in sorted(by_epoch):
        rows = by_epoch[epoch]
        positions = np.asarray(
            [
                [float(row["ecef_x"]), float(row["ecef_y"]), float(row["ecef_z"])]
                for row in rows
            ],
            dtype=np.float64,
        )
        base_index = int(np.argmax([float(row["log_weight"]) for row in rows]))
        trusted_candidates: list[int] = []
        if diagnostics[epoch]["fix"] == "1":
            assignment_id = diagnostics[epoch]["map_assignment_id"]
            trusted_candidates = [
                index
                for index, row in enumerate(rows)
                if row["assignment_id"] == assignment_id
            ]
            if trusted_candidates:
                trusted_index = max(
                    trusted_candidates,
                    key=lambda index: float(rows[index]["log_weight"]),
                )
                trusted_position = positions[trusted_index].copy()
                trusted_epoch = epoch
        measurements = _build_dd_measurements(
            np.asarray(data["sat_ecef"][epoch], dtype=np.float64),
            np.asarray(data["system_ids"][epoch], dtype=np.int32),
            list(data["used_prns"][epoch]),
            np.asarray(data["weights"][epoch], dtype=np.float64),
            positions[base_index],
            ("G", "E", "J", "C"),
            min_elevation_deg=-90.0,
            min_snr=0.0,
            keep_best=0,
        )
        dd_result = computer.compute_dd(
            float(data["times"][epoch]),
            measurements,
            rover_position_approx=positions[base_index],
            min_common_sats=4,
        )
        if dd_result is None:
            continue
        residuals = np.asarray(
            [dd_pseudorange_residuals_m(dd_result, position) for position in positions]
        )
        costs, n_rows = _bias_corrected_costs(
            residuals,
            tuple(dd_result.ref_sat_ids),
            tuple(dd_result.sat_ids),
            satellite_biases,
            epoch=epoch,
            max_age_epochs=int(args.max_age_epochs),
            scale_m=float(args.scale_m),
        )
        scored_epochs += int(n_rows >= int(args.min_rows))
        scores = np.asarray([float(row["log_weight"]) for row in rows])
        if n_rows >= int(args.min_rows):
            scores -= float(args.score_weight) * costs
        probabilities = _softmax(scores)
        selected = int(np.argmax(scores))
        truth = np.asarray(data["ground_truth"][epoch], dtype=np.float64)
        errors = np.linalg.norm(positions - truth[None, :], axis=1)
        output.append(
            {
                "epoch": epoch,
                "tow": float(data["times"][epoch]),
                "n_bias_rows": n_rows,
                "selected_error_m": float(errors[selected]),
                "base_error_m": float(errors[base_index]),
                "oracle_error_m": float(np.min(errors)),
                "gamma": float(probabilities[selected]),
                "selected_cost": float(costs[selected]),
                "base_cost": float(costs[base_index]),
                "trusted_update": int(diagnostics[epoch]["fix"] == "1"),
            }
        )
        if (
            trusted_position is not None
            and epoch - trusted_epoch <= int(args.trusted_anchor_max_age_epochs)
        ):
            trusted_residuals = dd_pseudorange_residuals_m(
                dd_result, trusted_position
            )
            _update_satellite_biases(
                satellite_biases,
                trusted_residuals,
                tuple(dd_result.ref_sat_ids),
                tuple(dd_result.sat_ids),
                epoch=epoch,
                blend_alpha=float(args.blend_alpha),
            )
            trusted_updates += 1
        elif (
            args.enable_self_calibration
            and trusted_updates > 0
            and n_rows >= int(args.min_rows)
            and float(probabilities[selected]) >= float(args.self_calibration_min_gamma)
            and float(costs[selected]) <= float(args.self_calibration_max_cost)
        ):
            _update_satellite_biases(
                satellite_biases,
                residuals[selected],
                tuple(dd_result.ref_sat_ids),
                tuple(dd_result.sat_ids),
                epoch=epoch,
                blend_alpha=float(args.blend_alpha),
            )
            self_updates += 1
    summary = {
        "n_epochs": len(output),
        "trusted_updates": trusted_updates,
        "self_updates": self_updates,
        "scored_epochs": scored_epochs,
        "satellite_biases_learned": len(satellite_biases),
        "base_sub50cm_epochs": sum(row["base_error_m"] < 0.5 for row in output),
        "selected_sub50cm_epochs": sum(row["selected_error_m"] < 0.5 for row in output),
        "oracle_sub50cm_epochs": sum(row["oracle_error_m"] < 0.5 for row in output),
    }
    return summary, output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("basin_trace", type=Path)
    parser.add_argument("epoch_diagnostics", type=Path)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--scale-m", type=float, default=1.0)
    parser.add_argument("--score-weight", type=float, default=5.0)
    parser.add_argument("--min-rows", type=int, default=3)
    parser.add_argument("--max-age-epochs", type=int, default=100)
    parser.add_argument("--trusted-anchor-max-age-epochs", type=int, default=4)
    parser.add_argument("--interpolate-base-epochs", action="store_true")
    parser.add_argument("--blend-alpha", type=float, default=0.2)
    parser.add_argument("--enable-self-calibration", action="store_true")
    parser.add_argument("--self-calibration-min-gamma", type=float, default=0.99)
    parser.add_argument("--self-calibration-max-cost", type=float, default=0.05)
    parser.add_argument("--out-summary", type=Path, required=True)
    parser.add_argument("--out-epochs", type=Path, required=True)
    args = parser.parse_args()
    summary, rows = analyze(args)
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    args.out_epochs.parent.mkdir(parents=True, exist_ok=True)
    with args.out_epochs.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
