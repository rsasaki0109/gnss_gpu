#!/usr/bin/env python3
"""Run the FGO-free WP23b partial-ambiguity basin RBPF on PPC."""

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

from exp_ppc_ctrbpf_fgo import (  # noqa: E402
    _build_dd_measurements,
    _filter_data_by_systems,
    _load_full_reference,
    _reference_position_map,
)
from exp_urbannav_baseline import run_wls  # noqa: E402
from exp_wp23b_float_seed import _doppler_velocity  # noqa: E402
from gnss_gpu.ambiguity_basin_pf import (  # noqa: E402
    AmbiguityBasinParticleFilter,
    BasinKalmanState,
)
from gnss_gpu.dd_carrier import DDCarrierComputer  # noqa: E402
from gnss_gpu.dd_float_kf import DDFloatKalmanFilter  # noqa: E402
from gnss_gpu.dd_pseudorange import DDPseudorangeComputer  # noqa: E402
from gnss_gpu.io.ppc import PPCDatasetLoader  # noqa: E402
from gnss_gpu.io.rinex_cache import RinexObservationCache  # noqa: E402
from gnss_gpu.lambda_ambiguity import integer_search  # noqa: E402
from gnss_gpu.rtk_fix_gate import trusted_fix_gate  # noqa: E402


def _write_trajectory(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["tow", "ecef_x", "ecef_y", "ecef_z", "fix"]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="tokyo/run2")
    parser.add_argument("--max-epochs", type=int, default=1200)
    parser.add_argument("--data-root", type=Path, default=Path("datasets/PPC-Dataset-data"))
    parser.add_argument("--dd-systems", default="G,E,J,C")
    parser.add_argument("--pr-systems", default="G,E,J")
    parser.add_argument("--subset-size", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--max-basins", type=int, default=128)
    parser.add_argument("--birth-mass", type=float, default=0.01)
    parser.add_argument("--sigma-dd-pr-m", type=float, default=5.0)
    parser.add_argument("--sigma-float-cp-cycles", type=float, default=0.10)
    parser.add_argument("--sigma-fixed-cp-cycles", type=float, default=0.20)
    parser.add_argument("--fix-gamma", type=float, default=0.99)
    parser.add_argument("--fix-streak", type=int, default=3)
    parser.add_argument(
        "--fix-consistency-m",
        type=float,
        default=0.5,
        help="Maximum MAP-basin versus independent float-KF position separation",
    )
    parser.add_argument(
        "--fix-ddpr-consistency-m",
        type=float,
        default=1.75,
        help="Maximum MAP-basin separation from the DDPR/Doppler-only guard KF",
    )
    parser.add_argument("--fix-min-dd-pairs", type=int, default=9)
    parser.add_argument("--fix-max-ddpr-age-epochs", type=int, default=4)
    parser.add_argument("--out-diagnostics", type=Path, default=Path("results/wp23b/csv/basin_run2_epochs.csv"))
    parser.add_argument("--out-summary", type=Path, default=Path("results/wp23b/csv/basin_run2_summary.json"))
    parser.add_argument("--out-trajectory", type=Path, default=Path("results/wp23b/pos/basin_run2.csv"))
    args = parser.parse_args(argv)

    city, run = str(args.run).split("/", 1)
    run_dir = args.data_root / city / run
    dd_systems = tuple(x.strip() for x in str(args.dd_systems).split(",") if x.strip())
    pr_systems = tuple(x.strip() for x in str(args.pr_systems).split(",") if x.strip())
    data = PPCDatasetLoader(run_dir).load_experiment_data(
        max_epochs=int(args.max_epochs),
        include_sat_velocity=True,
        systems=("G", "R", "E", "C", "J"),
    )
    wls_positions, _ = run_wls(_filter_data_by_systems(data, pr_systems))
    truth = _reference_position_map(_load_full_reference(run_dir / "reference.csv"))
    observation_cache = RinexObservationCache()
    carrier = DDCarrierComputer(
        run_dir / "base.obs",
        rover_obs_path=run_dir / "rover.obs",
        base_position=np.asarray(data["base_ecef"], dtype=np.float64),
        allowed_systems=dd_systems,
        observation_cache=observation_cache,
    )
    pseudorange = DDPseudorangeComputer(
        run_dir / "base.obs",
        rover_obs_path=run_dir / "rover.obs",
        base_position=np.asarray(data["base_ecef"], dtype=np.float64),
        allowed_systems=dd_systems,
        observation_cache=observation_cache,
    )
    float_kf = DDFloatKalmanFilter(
        np.asarray(wls_positions[0, :3], dtype=np.float64),
        position_sigma_m=50.0,
        velocity_sigma_mps=10.0,
        accel_process_sigma_mps2=3.0,
        ambiguity_init_sigma_cycles=40.0,
        max_track_age_epochs=10,
    )
    ddpr_guard = BasinKalmanState.from_position(
        np.asarray(wls_positions[0, :3], dtype=np.float64),
        np.eye(3, dtype=np.float64) * 50.0**2,
        velocity_sigma_mps=10.0,
        accel_process_sigma_mps2=3.0,
    )
    basin_pf = AmbiguityBasinParticleFilter(
        max_basins=int(args.max_basins),
        fix_gamma_threshold=float(args.fix_gamma),
        fix_min_streak=int(args.fix_streak),
        min_fixed_ambiguities=int(args.subset_size),
    )

    times = np.asarray(data["times"], dtype=np.float64)
    rows: list[dict[str, object]] = []
    n_birth_epochs = 0
    n_declared_fix = 0
    n_false_fix = 0
    n_correct_fix = 0
    n_gamma_fix = 0
    n_consistency_reject = 0
    max_gamma = 0.0
    last_ddpr_epoch = -1_000_000
    last_ddpr_pairs = 0
    last_ddpr_nis = float("nan")

    for i, tow in enumerate(times):
        if i > 0:
            dt = max(float(times[i] - times[i - 1]), 1e-3)
            float_kf.predict(dt)
            basin_pf.predict(dt)
            ddpr_guard.predict(dt)
        velocity, doppler_rms = _doppler_velocity(data, i, float_kf.position_ecef)
        if velocity is not None:
            velocity_sigma = max(0.5, min(float(doppler_rms), 5.0))
            float_kf.update_velocity(velocity, sigma_mps=velocity_sigma)
            basin_pf.update_velocity(velocity, sigma_mps=velocity_sigma)
            ddpr_guard.update_velocity(velocity, sigma_mps=velocity_sigma)

        measurements = _build_dd_measurements(
            np.asarray(data["sat_ecef"][i], dtype=np.float64),
            np.asarray(data["system_ids"][i], dtype=np.int32),
            list(data["used_prns"][i]),
            np.asarray(data["weights"][i], dtype=np.float64),
            float_kf.position_ecef,
            dd_systems,
            min_elevation_deg=-90.0,
            min_snr=0.0,
            keep_best=0,
        )
        dd_pr = pseudorange.compute_dd(
            float(tow), measurements, rover_position_approx=float_kf.position_ecef, min_common_sats=4
        )
        dd_cp = carrier.compute_dd(
            float(tow), measurements, rover_position_approx=float_kf.position_ecef, min_common_sats=4
        )
        pr_diag = None
        cp_diag = None
        if dd_pr is not None and int(dd_pr.n_dd) >= 3:
            pr_diag = float_kf.update_pseudorange(
                dd_pr, sigma_pr_m=float(args.sigma_dd_pr_m)
            )
            ddpr_guard.update_pseudorange(dd_pr, sigma_pr_m=float(args.sigma_dd_pr_m))
            last_ddpr_epoch = i
            last_ddpr_pairs = int(dd_pr.n_dd)
            last_ddpr_nis = float(pr_diag.normalized_innovation_sq)
        if dd_cp is not None and int(dd_cp.n_dd) >= 3:
            cp_diag = float_kf.update_carrier(
                dd_cp,
                dd_pseudorange_result=dd_pr,
                sigma_cp_cycles=float(args.sigma_float_cp_cycles),
                slip_threshold_cycles=2.0,
            )

        generations = float_kf.ambiguity_generations()
        active_versioned = {(key, generation) for key, generation in generations.items()}
        basin_pf.retain_compatible(active_versioned)
        if dd_pr is not None and basin_pf.basins:
            basin_pf.update_pseudorange(dd_pr, sigma_pr_m=float(args.sigma_dd_pr_m))
        if dd_cp is not None and basin_pf.basins:
            basin_pf.update_fixed_carrier(
                dd_cp,
                generations,
                sigma_cp_cycles=float(args.sigma_fixed_cp_cycles),
            )

        n_candidates = 0
        if dd_cp is not None and int(dd_cp.n_dd) >= int(args.subset_size):
            current_pairs = {
                (str(ref), str(sat)) for ref, sat in zip(dd_cp.ref_sat_ids, dd_cp.sat_ids)
            }
            current_keys = tuple(
                key for key in float_kf.ambiguity_seed().keys if key[:2] in current_pairs
            )
            seed = float_kf.ambiguity_seed(current_keys)
            if len(seed.keys) >= int(args.subset_size):
                order = np.argsort(np.diag(seed.qahat_cycles2))[: int(args.subset_size)]
                order = np.sort(order)
                keys = tuple(seed.keys[j] for j in order)
                ahat = seed.ahat_cycles[order]
                qahat = seed.qahat_cycles2[np.ix_(order, order)]
                candidates, residuals = integer_search(
                    ahat, qahat, n_candidates=int(args.top_k)
                )
                assignments = []
                conditionals = []
                for candidate in candidates:
                    position, covariance, _distance = float_kf.condition_position_on_integers(
                        keys, candidate
                    )
                    assignment = {
                        (key, generations[key]): int(value)
                        for key, value in zip(keys, candidate)
                    }
                    assignments.append(assignment)
                    conditionals.append(
                        BasinKalmanState.from_position(
                            position,
                            covariance,
                            velocity_ecef=float_kf.velocity_ecef,
                            velocity_sigma_mps=1.0,
                            accel_process_sigma_mps2=3.0,
                        )
                    )
                if assignments:
                    basin_pf.spawn(
                        assignments,
                        conditionals,
                        prior_mass=(1.0 if not basin_pf.basins else float(args.birth_mass)),
                    )
                    n_candidates = len(assignments)
                    n_birth_epochs += 1

        posterior = basin_pf.posterior()
        max_gamma = max(max_gamma, float(posterior.gamma))
        map_candidates = [
            basin for basin in basin_pf.basins if basin.assignment == posterior.map_assignment
        ]
        map_basin = max(map_candidates, key=lambda basin: basin.log_weight) if map_candidates else None
        map_float_separation = (
            float(np.linalg.norm(map_basin.conditional.mean[:3] - float_kf.position_ecef))
            if map_basin is not None
            else float("nan")
        )
        map_ddpr_separation = (
            float(np.linalg.norm(map_basin.conditional.mean[:3] - ddpr_guard.mean[:3]))
            if map_basin is not None
            else float("nan")
        )
        ddpr_age_epochs = i - last_ddpr_epoch
        gamma_fixed = bool(posterior.fixed and map_basin is not None)
        gate = trusted_fix_gate(
            map_float_separation_m=map_float_separation,
            map_ddpr_separation_m=map_ddpr_separation,
            last_ddpr_pairs=last_ddpr_pairs,
            ddpr_age_epochs=ddpr_age_epochs,
            max_float_separation_m=float(args.fix_consistency_m),
            max_ddpr_separation_m=float(args.fix_ddpr_consistency_m),
            min_ddpr_pairs=int(args.fix_min_dd_pairs),
            max_ddpr_age_epochs=int(args.fix_max_ddpr_age_epochs),
        )
        consistency_pass = gate.passed
        fixed = bool(gamma_fixed and consistency_pass)
        n_gamma_fix += int(gamma_fixed)
        n_consistency_reject += int(gamma_fixed and not consistency_pass)
        output_position = (
            map_basin.conditional.mean[:3].copy() if fixed else float_kf.position_ecef
        )
        ref = truth.get(round(float(tow), 1))
        output_error = (
            float(np.linalg.norm(output_position - ref))
            if ref is not None and np.all(np.isfinite(ref))
            else float("nan")
        )
        if fixed:
            n_declared_fix += 1
            if output_error < 0.5:
                n_correct_fix += 1
            else:
                n_false_fix += 1
        rows.append(
            {
                "epoch": i,
                "tow": float(tow),
                "ecef_x": float(output_position[0]),
                "ecef_y": float(output_position[1]),
                "ecef_z": float(output_position[2]),
                "fix": int(fixed),
                "gamma_fixed": int(gamma_fixed),
                "consistency_pass": int(consistency_pass),
                "float_consistency_pass": int(gate.float_consistent),
                "ddpr_consistency_pass": int(gate.ddpr_consistent),
                "ddpr_support_pass": int(gate.ddpr_supported),
                "ddpr_freshness_pass": int(gate.ddpr_fresh),
                "map_float_separation_m": map_float_separation,
                "map_ddpr_separation_m": map_ddpr_separation,
                "ddpr_guard_error_m": (
                    float(np.linalg.norm(ddpr_guard.mean[:3] - ref))
                    if ref is not None else float("nan")
                ),
                "last_ddpr_pairs": int(last_ddpr_pairs),
                "ddpr_age_epochs": int(ddpr_age_epochs),
                "last_ddpr_nis": last_ddpr_nis,
                "output_error_m": output_error,
                "float_error_m": (
                    float(np.linalg.norm(float_kf.position_ecef - ref))
                    if ref is not None else float("nan")
                ),
                "float_position_sigma_m": float(
                    np.sqrt(np.trace(float_kf.covariance[:3, :3]))
                ),
                "dd_pr_nis": (
                    float("nan") if pr_diag is None else pr_diag.normalized_innovation_sq
                ),
                "dd_cp_nis": (
                    float("nan") if cp_diag is None else cp_diag.normalized_innovation_sq
                ),
                "gamma": float(posterior.gamma),
                "fix_streak": int(posterior.fix_streak),
                "n_basins": int(posterior.n_basins),
                "basin_ess": float(posterior.ess),
                "map_n_ambiguities": len(posterior.map_assignment),
                "n_candidates_born": n_candidates,
                "n_dd_pr": 0 if dd_pr is None else int(dd_pr.n_dd),
                "n_dd_cp": 0 if dd_cp is None else int(dd_cp.n_dd),
            }
        )

    false_rate = 100.0 * n_false_fix / n_declared_fix if n_declared_fix else 0.0
    summary = {
        "run": str(args.run),
        "n_epochs": len(rows),
        "subset_size": int(args.subset_size),
        "top_k": int(args.top_k),
        "fix_gamma_threshold": float(args.fix_gamma),
        "fix_min_streak": int(args.fix_streak),
        "fix_float_consistency_m": float(args.fix_consistency_m),
        "fix_ddpr_consistency_m": float(args.fix_ddpr_consistency_m),
        "fix_min_dd_pairs": int(args.fix_min_dd_pairs),
        "fix_max_ddpr_age_epochs": int(args.fix_max_ddpr_age_epochs),
        "birth_epochs": int(n_birth_epochs),
        "declared_fix_epochs": int(n_declared_fix),
        "gamma_fix_epochs": int(n_gamma_fix),
        "consistency_reject_epochs": int(n_consistency_reject),
        "correct_fix_epochs": int(n_correct_fix),
        "false_fix_epochs": int(n_false_fix),
        "false_fix_pct": float(false_rate),
        "max_gamma": float(max_gamma),
        "sub50cm_all_epochs": int(sum(float(row["output_error_m"]) < 0.5 for row in rows)),
    }
    args.out_diagnostics.parent.mkdir(parents=True, exist_ok=True)
    with args.out_diagnostics.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _write_trajectory(args.out_trajectory, rows)
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"wrote {args.out_diagnostics}")
    print(f"wrote {args.out_trajectory}")


if __name__ == "__main__":
    main()
