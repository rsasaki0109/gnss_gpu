#!/usr/bin/env python3
"""Audit truth-free stop-to-stop TDCP/Doppler displacement against static grids."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "python"))
sys.path.insert(0, str(_REPO_ROOT / "experiments"))

from exp_ppc_tdcp_velocity import _epoch_measurements  # noqa: E402
from exp_wp23b_float_seed import _doppler_velocity  # noqa: E402
from gnss_gpu.io.ppc import PPCDatasetLoader  # noqa: E402
from gnss_gpu.tdcp_velocity import estimate_displacement_from_tdcp  # noqa: E402
from run_wp29_tdcp_anchor_smoother import _robust_static_velocity_bias  # noqa: E402


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def _position(row: dict[str, str]) -> np.ndarray:
    return np.asarray(
        [float(row["ecef_x"]), float(row["ecef_y"]), float(row["ecef_z"])],
        dtype=np.float64,
    )


def parse_segments(value: str) -> tuple[tuple[int, int], ...]:
    segments: list[tuple[int, int]] = []
    for token in str(value).split(","):
        if not token.strip():
            continue
        start, end = (int(part) for part in token.split(":", 1))
        if end <= start:
            raise ValueError("stop segment end must exceed start")
        segments.append((start, end))
    return tuple(segments)


def active_completed_bias(
    epoch: int,
    segments: tuple[tuple[int, int], ...],
    biases: tuple[np.ndarray | None, ...],
) -> np.ndarray | None:
    active: np.ndarray | None = None
    for (_start, end), bias in zip(segments, biases):
        if int(epoch) >= int(end) and bias is not None:
            active = np.asarray(bias, dtype=np.float64)
    return active


def _inside_stop(epoch: int, segments: tuple[tuple[int, int], ...]) -> bool:
    return any(start <= int(epoch) < end for start, end in segments)


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    diagnostics = {int(row["epoch"]): row for row in _read_csv(args.epoch_diagnostics)}
    segments = parse_segments(args.stop_segments)
    n_epochs = max(diagnostics) + 1
    data = PPCDatasetLoader(args.data_dir).load_experiment_data(
        max_epochs=n_epochs,
        include_sat_velocity=True,
        systems=("G", "R", "E", "C", "J"),
    )
    static_result = json.loads(args.initial_anchor_json.read_text(encoding="utf-8"))
    initial_position = np.asarray(
        static_result["candidates"][0]["position_ecef"], dtype=np.float64
    )
    stop_biases: list[np.ndarray | None] = []
    stop_bias_samples: list[int] = []
    for start, end in segments:
        samples: list[np.ndarray] = []
        for epoch in range(start, min(end, n_epochs)):
            velocity, _rms = _doppler_velocity(data, epoch, _position(diagnostics[epoch]))
            if velocity is not None:
                samples.append(np.asarray(velocity, dtype=np.float64))
        stop_biases.append(_robust_static_velocity_bias(samples))
        stop_bias_samples.append(len(samples))

    start_epoch = int(args.start_epoch)
    end_epoch = int(args.end_epoch)
    estimates: dict[str, np.ndarray] = {
        "tdcp_zero_missing": initial_position.copy(),
        "initial_bias": initial_position.copy(),
        "piecewise_zupt": initial_position.copy(),
    }
    counts = {
        name: {"tdcp": 0, "doppler": 0, "zero": 0}
        for name in estimates
    }
    previous = [
        measurement
        for measurement in _epoch_measurements(data, start_epoch)
        if int(measurement.system_id) in (0, 2, 4)
    ]
    initial_bias = stop_biases[0] if stop_biases else None
    for epoch in range(start_epoch + 1, end_epoch + 1):
        current = [
            measurement
            for measurement in _epoch_measurements(data, epoch)
            if int(measurement.system_id) in (0, 2, 4)
        ]
        approximate = _position(diagnostics[epoch])
        estimate = estimate_displacement_from_tdcp(
            approximate,
            previous,
            current,
            float(args.epoch_dt_s),
            min_sats=int(args.tdcp_min_sats),
            max_postfit_rms_m=float(args.tdcp_max_postfit_rms_m),
            slip_residual_threshold_m=float(args.tdcp_slip_threshold_m),
        )
        tdcp = None if estimate is None else np.asarray(estimate.displacement_ecef_m)
        doppler: np.ndarray | None = None
        if tdcp is None:
            doppler_value, _rms = _doppler_velocity(data, epoch, approximate)
            if doppler_value is not None:
                doppler = np.asarray(doppler_value, dtype=np.float64)
        for name in estimates:
            displacement: np.ndarray | None = None
            if name == "piecewise_zupt" and _inside_stop(epoch, segments):
                displacement = np.zeros(3, dtype=np.float64)
                counts[name]["zero"] += 1
            elif tdcp is not None:
                displacement = tdcp
                counts[name]["tdcp"] += 1
            elif name != "tdcp_zero_missing" and doppler is not None:
                bias = (
                    initial_bias
                    if name == "initial_bias"
                    else active_completed_bias(epoch, segments, tuple(stop_biases))
                )
                if bias is not None:
                    displacement = (
                        doppler - bias
                    ) * float(args.epoch_dt_s)
                    counts[name]["doppler"] += 1
            if displacement is None:
                displacement = np.zeros(3, dtype=np.float64)
                counts[name]["zero"] += 1
            estimates[name] += displacement
        previous = current

    truth = np.asarray(data["ground_truth"][end_epoch], dtype=np.float64)
    grid = json.loads(args.grid_candidates_json.read_text(encoding="utf-8"))
    candidates = list(grid["candidates"])
    variants: dict[str, Any] = {}
    for name, position in estimates.items():
        ranked = sorted(
            candidates,
            key=lambda row: float(
                np.linalg.norm(np.asarray(row["position_ecef"]) - position)
            ),
        )
        candidate_rows = [
            {
                "rank": rank,
                "candidate_id": int(row["candidate_id"]),
                "motion_distance_m": float(
                    np.linalg.norm(np.asarray(row["position_ecef"]) - position)
                ),
                "audit_error_m": float(row["final_error_m"]),
            }
            for rank, row in enumerate(ranked, start=1)
        ]
        variants[name] = {
            "position_ecef": position.tolist(),
            "audit_error_m": float(np.linalg.norm(position - truth)),
            "counts": counts[name],
            "candidate_70_rank": next(
                row["rank"] for row in candidate_rows if row["candidate_id"] == 70
            ),
            "top_candidates": candidate_rows[:10],
        }
    return {
        "start_epoch": start_epoch,
        "end_epoch": end_epoch,
        "stop_segments": [list(segment) for segment in segments],
        "stop_bias_samples": stop_bias_samples,
        "stop_bias_ecef_mps": [
            None if bias is None else np.asarray(bias).tolist() for bias in stop_biases
        ],
        "variants": variants,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--epoch-diagnostics", type=Path, required=True)
    parser.add_argument("--initial-anchor-json", type=Path, required=True)
    parser.add_argument("--grid-candidates-json", type=Path, required=True)
    parser.add_argument("--stop-segments", required=True)
    parser.add_argument("--start-epoch", type=int, required=True)
    parser.add_argument("--end-epoch", type=int, required=True)
    parser.add_argument("--epoch-dt-s", type=float, default=0.2)
    parser.add_argument("--tdcp-min-sats", type=int, default=5)
    parser.add_argument("--tdcp-max-postfit-rms-m", type=float, default=0.5)
    parser.add_argument("--tdcp-slip-threshold-m", type=float, default=0.25)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
