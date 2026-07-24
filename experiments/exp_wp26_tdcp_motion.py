#!/usr/bin/env python3
"""Build a causal truth-free PPC TDCP displacement stream for WP26."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT = _SCRIPT_DIR.parent
for _path in (_ROOT / "python", _SCRIPT_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from exp_ppc_tdcp_velocity import _epoch_measurements  # noqa: E402
from exp_urbannav_baseline import run_wls  # noqa: E402
from gnss_gpu.io.ppc import PPCDatasetLoader  # noqa: E402
from gnss_gpu.tdcp_velocity import estimate_displacement_from_tdcp  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="tokyo/run3")
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--data-root", type=Path, default=Path("datasets/PPC-Dataset-data"))
    parser.add_argument("--systems", default="G,E,J")
    parser.add_argument("--min-sats", type=int, default=5)
    parser.add_argument("--max-postfit-rms-m", type=float, default=0.5)
    parser.add_argument("--slip-residual-threshold-m", type=float, default=0.25)
    parser.add_argument("--carrier-phase-sign", type=float, default=1.0)
    parser.add_argument("--receiver-motion-sign", type=float, default=-1.0)
    parser.add_argument("--out-motion", type=Path, required=True)
    parser.add_argument("--out-summary", type=Path, required=True)
    args = parser.parse_args(argv)

    city, run = str(args.run).split("/", 1)
    systems = tuple(part.strip() for part in str(args.systems).split(",") if part.strip())
    data = PPCDatasetLoader(args.data_root / city / run).load_experiment_data(
        max_epochs=int(args.max_epochs),
        systems=systems,
        include_sat_velocity=True,
    )
    wls, _ = run_wls(data)
    times = np.asarray(data["times"], dtype=np.float64)
    truth = np.asarray(data["ground_truth"], dtype=np.float64)
    rows: list[dict[str, object]] = []
    displacement_errors: list[float] = []
    nis_values: list[float] = []
    postfit_values: list[float] = []
    rejected_rows = 0

    for epoch in range(1, len(times)):
        dt = float(times[epoch] - times[epoch - 1])
        estimate = estimate_displacement_from_tdcp(
            wls[epoch - 1, :3],
            _epoch_measurements(data, epoch - 1),
            _epoch_measurements(data, epoch),
            dt,
            min_sats=int(args.min_sats),
            max_postfit_rms_m=float(args.max_postfit_rms_m),
            carrier_phase_sign=float(args.carrier_phase_sign),
            receiver_motion_sign=float(args.receiver_motion_sign),
            slip_residual_threshold_m=float(args.slip_residual_threshold_m),
        )
        row: dict[str, object] = {
            "epoch": epoch,
            "tow": float(times[epoch]),
            "dt": dt,
            "used": int(estimate is not None),
        }
        if estimate is not None:
            displacement = estimate.displacement_ecef_m
            covariance = estimate.covariance_m2
            row.update(
                {
                    "dx": float(displacement[0]),
                    "dy": float(displacement[1]),
                    "dz": float(displacement[2]),
                    **{
                        f"cov_{r}{c}": float(covariance[r, c])
                        for r in range(3) for c in range(3)
                    },
                    "postfit_rms_m": float(estimate.postfit_rms_m),
                    "n_input": int(estimate.n_input),
                    "n_used": int(estimate.n_used),
                    "n_rejected": int(estimate.n_rejected),
                }
            )
            truth_delta = truth[epoch] - truth[epoch - 1]
            error = displacement - truth_delta
            displacement_errors.append(float(np.linalg.norm(error)))
            nis_values.append(float(error @ np.linalg.solve(covariance, error)))
            postfit_values.append(float(estimate.postfit_rms_m))
            rejected_rows += int(estimate.n_rejected)
        else:
            row.update(
                {
                    "dx": "", "dy": "", "dz": "",
                    **{f"cov_{r}{c}": "" for r in range(3) for c in range(3)},
                    "postfit_rms_m": "", "n_input": 0, "n_used": 0, "n_rejected": 0,
                }
            )
        rows.append(row)

    errors = np.asarray(displacement_errors, dtype=np.float64)
    nis = np.asarray(nis_values, dtype=np.float64)
    summary = {
        "run": str(args.run),
        "epochs": len(times),
        "intervals": max(len(times) - 1, 0),
        "tdcp_used_intervals": int(errors.size),
        "tdcp_use_rate_pct": float(100.0 * errors.size / max(len(times) - 1, 1)),
        "rejected_satellite_rows": int(rejected_rows),
        "displacement_error_median_m": float(np.median(errors)) if errors.size else None,
        "displacement_error_p90_m": float(np.percentile(errors, 90)) if errors.size else None,
        "displacement_error_rms_m": float(np.sqrt(np.mean(errors**2))) if errors.size else None,
        "nis_median": float(np.median(nis)) if nis.size else None,
        "nis_p90": float(np.percentile(nis, 90)) if nis.size else None,
        "postfit_rms_median_m": float(np.median(postfit_values)) if postfit_values else None,
        "truth_used_by_motion_stream": False,
    }
    args.out_motion.parent.mkdir(parents=True, exist_ok=True)
    with args.out_motion.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
