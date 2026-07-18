#!/usr/bin/env python3
"""Replay WP25 temporal-lineage configurations from a truth-free basin trace."""

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

from exp_ppc_ctrbpf_fgo import _load_full_reference, _reference_position_map  # noqa: E402
from gnss_gpu.rtk_evidence import ambiguity_assignment_from_json  # noqa: E402
from gnss_gpu.temporal_ambiguity import (  # noqa: E402
    TemporalAmbiguityCandidate,
    TemporalAmbiguityConfig,
    TemporalAmbiguityFilter,
)


def _float_grid(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item.strip()]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--run", default="tokyo/run3")
    parser.add_argument("--data-root", type=Path, default=Path("datasets/PPC-Dataset-data"))
    parser.add_argument("--birth-masses", default="0.01,0.05,0.2")
    parser.add_argument("--motion-sigmas-m", default="1,3,10,100")
    parser.add_argument(
        "--motion-mode", choices=("candidate", "none", "external"), default="candidate"
    )
    parser.add_argument("--motion-trace", type=Path, default=None)
    parser.add_argument("--external-cov-scale", type=float, default=1.0)
    parser.add_argument("--change-cost", type=float, default=2.0)
    parser.add_argument("--incompatible-cost", type=float, default=12.0)
    parser.add_argument("--death-cost", type=float, default=6.0)
    parser.add_argument("--out-summary", type=Path, required=True)
    args = parser.parse_args(argv)

    trace_file = (
        gzip.open(args.trace, mode="rt", newline="")
        if args.trace.suffix == ".gz"
        else args.trace.open(newline="")
    )
    with trace_file as fh:
        rows = list(csv.DictReader(fh))
    by_epoch: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        by_epoch.setdefault(int(row["epoch"]), []).append(row)
    motion_by_epoch: dict[int, dict[str, str]] = {}
    if args.motion_trace is not None:
        with args.motion_trace.open(newline="") as fh:
            motion_by_epoch = {
                int(row["epoch"]): row for row in csv.DictReader(fh)
            }
    if args.motion_mode == "external" and not motion_by_epoch:
        raise ValueError("external motion mode requires --motion-trace")
    city, run = str(args.run).split("/", 1)
    truth = _reference_position_map(
        _load_full_reference(args.data_root / city / run / "reference.csv")
    )

    summaries: list[dict[str, object]] = []
    for birth_mass in _float_grid(args.birth_masses):
        for motion_sigma in _float_grid(args.motion_sigmas_m):
            covariance_scale = float(args.external_cov_scale)
            tracker = TemporalAmbiguityFilter(
                TemporalAmbiguityConfig(
                    birth_mass=birth_mass,
                    assignment_change_cost=float(args.change_cost),
                    incompatible_cost=float(args.incompatible_cost),
                    death_cost=float(args.death_cost),
                    motion_sigma_m=motion_sigma,
                )
            )
            temporal_errors: list[float] = []
            single_errors: list[float] = []
            temporal_gammas: list[float] = []
            oracle_available = 0
            temporal_oracle_selected = 0
            previous_tow: float | None = None
            for epoch in sorted(by_epoch):
                epoch_rows = by_epoch[epoch]
                tow = float(epoch_rows[0]["tow"])
                candidates = [
                    TemporalAmbiguityCandidate(
                        candidate_id=row["assignment_id"],
                        assignment=ambiguity_assignment_from_json(row["assignment_json"]),
                        epoch_log_likelihood=float(row["epoch_log_likelihood"]),
                        position_ecef=np.asarray([row["ecef_x"], row["ecef_y"], row["ecef_z"]], dtype=float),
                        velocity_ecef=np.asarray([row["velocity_x"], row["velocity_y"], row["velocity_z"]], dtype=float),
                    )
                    for row in epoch_rows
                ]
                step_kwargs: dict[str, object] = {"motion_mode": args.motion_mode}
                if args.motion_mode == "external":
                    motion = motion_by_epoch.get(epoch)
                    if motion is not None and motion["used"] == "1":
                        step_kwargs["external_displacement_ecef_m"] = np.asarray(
                            [motion["dx"], motion["dy"], motion["dz"]], dtype=float
                        )
                        step_kwargs["external_covariance_m2"] = covariance_scale * np.asarray(
                            [
                                [motion[f"cov_{r}{c}"] for c in range(3)]
                                for r in range(3)
                            ],
                            dtype=float,
                        )
                    else:
                        step_kwargs["motion_mode"] = "none"
                posterior = tracker.step(
                    epoch,
                    0.0 if previous_tow is None else max(tow - previous_tow, 0.0),
                    candidates,
                    **step_kwargs,
                )
                previous_tow = tow
                ref = truth.get(round(tow, 1))
                if ref is None:
                    continue
                positions = {
                    row["assignment_id"]: np.asarray(
                        [row["ecef_x"], row["ecef_y"], row["ecef_z"]], dtype=float
                    )
                    for row in epoch_rows
                }
                errors = {key: float(np.linalg.norm(pos - ref)) for key, pos in positions.items()}
                temporal_error = errors[posterior.map_candidate_id]
                single_id = max(epoch_rows, key=lambda row: float(row["log_weight"]))["assignment_id"]
                temporal_errors.append(temporal_error)
                single_errors.append(errors[single_id])
                temporal_gammas.append(posterior.gamma)
                available = min(errors.values()) < 0.5
                oracle_available += int(available)
                temporal_oracle_selected += int(available and temporal_error < 0.5)
            te = np.asarray(temporal_errors)
            se = np.asarray(single_errors)
            gamma = np.asarray(temporal_gammas)
            summaries.append(
                {
                    "birth_mass": birth_mass,
                    "motion_sigma_m": motion_sigma,
                    "motion_mode": args.motion_mode,
                    "external_covariance_scale": covariance_scale,
                    "epochs": int(te.size),
                    "oracle_sub50cm_epochs": int(oracle_available),
                    "single_sub50cm_epochs": int(np.sum(se < 0.5)),
                    "temporal_sub50cm_epochs": int(np.sum(te < 0.5)),
                    "temporal_oracle_selection_epochs": int(temporal_oracle_selected),
                    "temporal_better_epochs": int(np.sum(te < se)),
                    "temporal_worse_epochs": int(np.sum(te > se)),
                    "single_median_error_m": float(np.median(se)),
                    "temporal_median_error_m": float(np.median(te)),
                    "gamma99_epochs": int(np.sum(gamma > 0.99)),
                    "gamma99_correct_epochs": int(np.sum((gamma > 0.99) & (te < 0.5))),
                }
            )
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(json.dumps(summaries, indent=2) + "\n")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
