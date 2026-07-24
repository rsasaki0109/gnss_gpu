#!/usr/bin/env python3
"""Evaluate pivot-invariant DDPR integrity over a WP25 basin trace."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT = _SCRIPT_DIR.parent
for _path in (_ROOT / "python", _SCRIPT_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from exp_ppc_ctrbpf_fgo import _build_dd_measurements  # noqa: E402
from gnss_gpu.dd_integrity import multipivot_ddpr_scores  # noqa: E402
from gnss_gpu.dd_pseudorange import DDPseudorangeComputer  # noqa: E402
from gnss_gpu.io.ppc import PPCDatasetLoader  # noqa: E402
from gnss_gpu.io.rinex_cache import RinexObservationCache  # noqa: E402


def _grid(value: str, cast=float):
    return [cast(item) for item in value.split(",") if item.strip()]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--run", default="tokyo/run3")
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--data-root", type=Path, default=Path("datasets/PPC-Dataset-data"))
    parser.add_argument("--scales-m", default="1,3,5,10")
    parser.add_argument("--trim-pairs", default="0,1,3")
    parser.add_argument("--out-summary", type=Path, required=True)
    args = parser.parse_args(argv)

    trace_file = (
        gzip.open(args.trace, mode="rt", newline="")
        if args.trace.suffix == ".gz"
        else args.trace.open(newline="")
    )
    with trace_file as fh:
        trace_rows = list(csv.DictReader(fh))
    by_epoch: dict[int, list[dict[str, str]]] = {}
    for row in trace_rows:
        by_epoch.setdefault(int(row["epoch"]), []).append(row)

    city, run = str(args.run).split("/", 1)
    run_dir = args.data_root / city / run
    data = PPCDatasetLoader(run_dir).load_experiment_data(
        max_epochs=int(args.max_epochs),
        include_sat_velocity=True,
        systems=("G", "R", "E", "C", "J"),
    )
    cache = RinexObservationCache()
    computer = DDPseudorangeComputer(
        run_dir / "base.obs",
        rover_obs_path=run_dir / "rover.obs",
        base_position=np.asarray(data["base_ecef"], dtype=np.float64),
        allowed_systems=("G", "E", "J", "C"),
        observation_cache=cache,
    )

    configurations = [
        (scale, trim)
        for scale in _grid(args.scales_m, float)
        for trim in _grid(args.trim_pairs, int)
    ]
    metrics = {
        config: {
            "epochs": 0,
            "oracle": 0,
            "selected": 0,
            "better": 0,
            "worse": 0,
            "errors": [],
            "single_errors": [],
            "confident": 0,
            "confident_correct": 0,
        }
        for config in configurations
    }
    dd_epochs = 0
    times = np.asarray(data["times"], dtype=np.float64)
    for epoch in sorted(by_epoch):
        if epoch >= len(times):
            continue
        rows = by_epoch[epoch]
        positions = np.asarray(
            [[row["ecef_x"], row["ecef_y"], row["ecef_z"]] for row in rows],
            dtype=np.float64,
        )
        measurements = _build_dd_measurements(
            np.asarray(data["sat_ecef"][epoch], dtype=np.float64),
            np.asarray(data["system_ids"][epoch], dtype=np.int32),
            list(data["used_prns"][epoch]),
            np.asarray(data["weights"][epoch], dtype=np.float64),
            positions[0],
            ("G", "E", "J", "C"),
            min_elevation_deg=-90.0,
            min_snr=0.0,
            keep_best=0,
        )
        dd_result = computer.compute_dd(
            float(times[epoch]),
            measurements,
            rover_position_approx=positions[0],
            min_common_sats=4,
        )
        if dd_result is None:
            continue
        dd_epochs += 1
        truth = np.asarray(data["ground_truth"][epoch], dtype=np.float64)
        errors = np.linalg.norm(positions - truth[None, :], axis=1)
        single_index = int(np.argmax([float(row["log_weight"]) for row in rows]))
        single_error = float(errors[single_index])
        oracle = bool(np.min(errors) < 0.5)
        for config in configurations:
            scale, trim = config
            result = multipivot_ddpr_scores(
                dd_result,
                positions,
                scale_m=scale,
                trim_largest_pairs=trim,
            )
            selected_error = float(errors[result.best_index])
            item = metrics[config]
            item["epochs"] += 1
            item["oracle"] += int(oracle)
            item["selected"] += int(selected_error < 0.5)
            item["better"] += int(selected_error < single_error)
            item["worse"] += int(selected_error > single_error)
            item["errors"].append(selected_error)
            item["single_errors"].append(single_error)
            gamma = float(result.probabilities[result.best_index])
            item["confident"] += int(gamma > 0.99)
            item["confident_correct"] += int(gamma > 0.99 and selected_error < 0.5)

    summary: list[dict[str, object]] = []
    for scale, trim in configurations:
        item = metrics[(scale, trim)]
        summary.append(
            {
                "scale_m": scale,
                "trim_largest_pairs": trim,
                "dd_epochs": dd_epochs,
                "oracle_sub50cm_epochs": item["oracle"],
                "single_map_sub50cm_epochs": int(
                    np.sum(np.asarray(item["single_errors"]) < 0.5)
                ),
                "multipivot_sub50cm_epochs": item["selected"],
                "multipivot_better_epochs": item["better"],
                "multipivot_worse_epochs": item["worse"],
                "single_median_error_m": float(np.median(item["single_errors"])),
                "multipivot_median_error_m": float(np.median(item["errors"])),
                "gamma99_epochs": item["confident"],
                "gamma99_correct_epochs": item["confident_correct"],
            }
        )
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
