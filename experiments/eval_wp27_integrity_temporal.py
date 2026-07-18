#!/usr/bin/env python3
"""Combine sparse multi-pivot DDPR anchors with TDCP lineage holdover."""

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
from gnss_gpu.rtk_evidence import ambiguity_assignment_from_json  # noqa: E402
from gnss_gpu.temporal_ambiguity import (  # noqa: E402
    TemporalAmbiguityCandidate,
    TemporalAmbiguityConfig,
    TemporalAmbiguityFilter,
)


def _grid(value: str, cast=float):
    return [cast(item) for item in value.split(",") if item.strip()]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--motion-trace", type=Path, required=True)
    parser.add_argument("--run", default="tokyo/run3")
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--data-root", type=Path, default=Path("datasets/PPC-Dataset-data"))
    parser.add_argument("--scale-m", type=float, default=3.0)
    parser.add_argument("--trim-pairs", type=int, default=0)
    parser.add_argument("--birth-masses", default="0.01,0.05")
    parser.add_argument("--integrity-weights", default="1,5,20")
    parser.add_argument("--modes", default="integrity_only,combined")
    parser.add_argument("--out-summary", type=Path, required=True)
    parser.add_argument(
        "--out-selections",
        type=Path,
        help="Write an epoch selection trace; requires a single grid configuration.",
    )
    args = parser.parse_args(argv)

    with gzip.open(args.trace, mode="rt", newline="") as fh:
        trace_rows = list(csv.DictReader(fh))
    by_epoch: dict[int, list[dict[str, str]]] = {}
    for row in trace_rows:
        by_epoch.setdefault(int(row["epoch"]), []).append(row)
    with args.motion_trace.open(newline="") as fh:
        motion = {int(row["epoch"]): row for row in csv.DictReader(fh)}

    city, run = str(args.run).split("/", 1)
    run_dir = args.data_root / city / run
    data = PPCDatasetLoader(run_dir).load_experiment_data(
        max_epochs=int(args.max_epochs),
        include_sat_velocity=True,
        systems=("G", "R", "E", "C", "J"),
    )
    computer = DDPseudorangeComputer(
        run_dir / "base.obs",
        rover_obs_path=run_dir / "rover.obs",
        base_position=np.asarray(data["base_ecef"], dtype=np.float64),
        allowed_systems=("G", "E", "J", "C"),
        observation_cache=RinexObservationCache(),
    )
    times = np.asarray(data["times"], dtype=np.float64)

    frames: list[dict[str, object]] = []
    dd_epochs = 0
    for epoch in sorted(by_epoch):
        if epoch >= len(times):
            continue
        rows = by_epoch[epoch]
        positions = np.asarray(
            [[row["ecef_x"], row["ecef_y"], row["ecef_z"]] for row in rows], dtype=float
        )
        measurements = _build_dd_measurements(
            np.asarray(data["sat_ecef"][epoch], dtype=float),
            np.asarray(data["system_ids"][epoch], dtype=np.int32),
            list(data["used_prns"][epoch]),
            np.asarray(data["weights"][epoch], dtype=float),
            positions[0],
            ("G", "E", "J", "C"),
            min_elevation_deg=-90.0,
            min_snr=0.0,
            keep_best=0,
        )
        dd_result = computer.compute_dd(
            float(times[epoch]), measurements, rover_position_approx=positions[0], min_common_sats=4
        )
        integrity_scores = None
        if dd_result is not None:
            integrity_scores = multipivot_ddpr_scores(
                dd_result,
                positions,
                scale_m=float(args.scale_m),
                trim_largest_pairs=int(args.trim_pairs),
            ).scores
            dd_epochs += 1
        truth = np.asarray(data["ground_truth"][epoch], dtype=float)
        frames.append(
            {
                "epoch": epoch,
                "tow": float(times[epoch]),
                "rows": rows,
                "positions": positions,
                "errors": np.linalg.norm(positions - truth[None, :], axis=1),
                "integrity_scores": integrity_scores,
            }
        )

    configurations = [
        (mode.strip(), birth, weight)
        for mode in str(args.modes).split(",")
        for birth in _grid(args.birth_masses)
        for weight in _grid(args.integrity_weights)
    ]
    if args.out_selections is not None and len(configurations) != 1:
        parser.error("--out-selections requires one mode, birth mass, and integrity weight")
    summaries: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    for mode, birth_mass, integrity_weight in configurations:
        if mode not in {"integrity_only", "combined"}:
            raise ValueError(f"unknown mode: {mode}")
        tracker = TemporalAmbiguityFilter(
            TemporalAmbiguityConfig(birth_mass=birth_mass, motion_sigma_m=3.0)
        )
        selected_errors: list[float] = []
        single_errors: list[float] = []
        gammas: list[float] = []
        oracle = 0
        selected = 0
        dd_selected = 0
        previous_tow: float | None = None
        for frame in frames:
            epoch = int(frame["epoch"])
            tow = float(frame["tow"])
            rows = frame["rows"]
            positions = frame["positions"]
            errors = frame["errors"]
            integrity_scores = frame["integrity_scores"]
            candidates = []
            for index, row in enumerate(rows):
                observation = (
                    float(row["epoch_log_likelihood"]) if mode == "combined" else 0.0
                )
                if integrity_scores is not None:
                    observation += integrity_weight * float(integrity_scores[index])
                candidates.append(
                    TemporalAmbiguityCandidate(
                        candidate_id=row["assignment_id"],
                        assignment=ambiguity_assignment_from_json(row["assignment_json"]),
                        epoch_log_likelihood=observation,
                        position_ecef=positions[index],
                        velocity_ecef=np.asarray(
                            [row["velocity_x"], row["velocity_y"], row["velocity_z"]], dtype=float
                        ),
                    )
                )
            motion_row = motion.get(epoch)
            kwargs: dict[str, object] = {"motion_mode": "none"}
            if motion_row is not None and motion_row["used"] == "1":
                kwargs = {
                    "motion_mode": "external",
                    "external_displacement_ecef_m": np.asarray(
                        [motion_row["dx"], motion_row["dy"], motion_row["dz"]], dtype=float
                    ),
                    "external_covariance_m2": np.asarray(
                        [[motion_row[f"cov_{r}{c}"] for c in range(3)] for r in range(3)],
                        dtype=float,
                    ),
                }
            posterior = tracker.step(
                epoch,
                0.0 if previous_tow is None else max(tow - previous_tow, 0.0),
                candidates,
                **kwargs,
            )
            previous_tow = tow
            selected_index = next(
                index for index, row in enumerate(rows)
                if row["assignment_id"] == posterior.map_candidate_id
            )
            single_index = int(np.argmax([float(row["log_weight"]) for row in rows]))
            error = float(errors[selected_index])
            selected_errors.append(error)
            single_errors.append(float(errors[single_index]))
            gammas.append(float(posterior.gamma))
            available = float(np.min(errors)) < 0.5
            oracle += int(available)
            selected += int(error < 0.5)
            dd_selected += int(integrity_scores is not None and error < 0.5)
            if args.out_selections is not None:
                selection_rows.append(
                    {
                        "epoch": epoch,
                        "tow": tow,
                        "dd_available": int(integrity_scores is not None),
                        "oracle_sub50cm_available": int(available),
                        "selected_sub50cm": int(error < 0.5),
                        "selected_error_m": error,
                        "single_map_error_m": float(errors[single_index]),
                        "gamma": float(posterior.gamma),
                        "dwell_epochs": int(posterior.dwell_epochs),
                        "selected_assignment_id": rows[selected_index]["assignment_id"],
                    }
                )
        selected_array = np.asarray(selected_errors)
        single_array = np.asarray(single_errors)
        gamma_array = np.asarray(gammas)
        summaries.append(
            {
                "mode": mode,
                "birth_mass": birth_mass,
                "integrity_weight": integrity_weight,
                "epochs": len(frames),
                "dd_epochs": dd_epochs,
                "oracle_sub50cm_epochs": oracle,
                "single_map_sub50cm_epochs": int(np.sum(single_array < 0.5)),
                "temporal_sub50cm_epochs": selected,
                "temporal_dd_epoch_sub50cm_epochs": dd_selected,
                "temporal_better_epochs": int(np.sum(selected_array < single_array)),
                "temporal_worse_epochs": int(np.sum(selected_array > single_array)),
                "single_median_error_m": float(np.median(single_array)),
                "temporal_median_error_m": float(np.median(selected_array)),
                "gamma99_epochs": int(np.sum(gamma_array > 0.99)),
                "gamma99_correct_epochs": int(
                    np.sum((gamma_array > 0.99) & (selected_array < 0.5))
                ),
            }
        )
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(json.dumps(summaries, indent=2) + "\n")
    if args.out_selections is not None:
        args.out_selections.parent.mkdir(parents=True, exist_ok=True)
        with args.out_selections.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(selection_rows[0]))
            writer.writeheader()
            writer.writerows(selection_rows)
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
