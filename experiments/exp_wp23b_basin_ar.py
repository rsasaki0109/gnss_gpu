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
from exp_ppc_tdcp_velocity import _epoch_measurements  # noqa: E402
from exp_urbannav_baseline import run_wls  # noqa: E402
from exp_wp23b_float_seed import _doppler_velocity  # noqa: E402
from gnss_gpu.ambiguity_basin_pf import (  # noqa: E402
    AmbiguityBasinParticleFilter,
    BasinKalmanState,
)
from gnss_gpu.ambiguity_respawn import (  # noqa: E402
    condition_respawn_position,
    ddpr_centered_ambiguity_seed,
)
from gnss_gpu.dd_carrier import DDCarrierComputer  # noqa: E402
from gnss_gpu.dd_float_kf import DDFloatKalmanFilter  # noqa: E402
from gnss_gpu.dd_integrity import (  # noqa: E402
    multipivot_ddpr_scores,
    satellite_pair_costs,
)
from gnss_gpu.dd_pseudorange import DDPseudorangeComputer  # noqa: E402
from gnss_gpu.io.ppc import PPCDatasetLoader  # noqa: E402
from gnss_gpu.io.rinex_cache import RinexObservationCache  # noqa: E402
from gnss_gpu.lambda_ambiguity import integer_search  # noqa: E402
from gnss_gpu.rtk_evidence import (  # noqa: E402
    EvidenceLedger,
    RTKEpochTrace,
    TrustedFixCommitPolicy,
    TrustedFixPolicyConfig,
    TrustedFixPolicyInput,
    ambiguity_assignment_id,
    ambiguity_assignment_json,
    replay_fix_decisions,
)
from gnss_gpu.recovery_proposals import (  # noqa: E402
    RecoveryPositionBank,
    covariance_axis_position_seeds,
)
from gnss_gpu.temporal_ambiguity import (  # noqa: E402
    TemporalAmbiguityCandidate,
    TemporalAmbiguityConfig,
    TemporalAmbiguityFilter,
)
from gnss_gpu.tdcp_velocity import estimate_displacement_from_tdcp  # noqa: E402


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
    parser.add_argument("--basin-diversity-reserve-fraction", type=float, default=0.0)
    parser.add_argument("--basin-diversity-radius-m", type=float, default=1.0)
    parser.add_argument("--basin-dedup-position-radius-m", type=float, default=float("inf"))
    parser.add_argument("--birth-mass", type=float, default=0.01)
    parser.add_argument("--sigma-dd-pr-m", type=float, default=5.0)
    parser.add_argument(
        "--sigma-basin-dd-pr-m",
        type=float,
        default=0.0,
        help="Basin-only DDPR sigma; <=0 uses --sigma-dd-pr-m",
    )
    parser.add_argument("--sigma-float-cp-cycles", type=float, default=0.10)
    parser.add_argument("--float-slip-threshold-cycles", type=float, default=2.0)
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
    parser.add_argument("--position-cluster-radius-m", type=float, default=0.5)
    parser.add_argument("--enable-temporal-lineage", action="store_true")
    parser.add_argument("--temporal-birth-mass", type=float, default=0.05)
    parser.add_argument("--temporal-change-cost", type=float, default=2.0)
    parser.add_argument("--temporal-incompatible-cost", type=float, default=12.0)
    parser.add_argument("--temporal-death-cost", type=float, default=6.0)
    parser.add_argument("--temporal-motion-sigma-m", type=float, default=3.0)
    parser.add_argument("--enable-integrity-lineage", action="store_true")
    parser.add_argument("--integrity-scale-m", type=float, default=3.0)
    parser.add_argument("--integrity-trim-pairs", type=int, default=0)
    parser.add_argument("--integrity-weight", type=float, default=5.0)
    parser.add_argument(
        "--integrity-exclude-max-cost-satellite",
        action="store_true",
        help="Exclude the largest guard-position incident pair-cost satellite",
    )
    parser.add_argument(
        "--integrity-satellite-cost-memory",
        type=float,
        default=0.0,
        help="EMA memory in [0,1) for causal per-satellite incident pair cost",
    )
    parser.add_argument("--integrity-tdcp-systems", default="G,E,J")
    parser.add_argument("--integrity-tdcp-min-sats", type=int, default=5)
    parser.add_argument("--integrity-tdcp-max-postfit-rms-m", type=float, default=0.5)
    parser.add_argument("--integrity-tdcp-slip-threshold-m", type=float, default=0.25)
    parser.add_argument("--enable-ddpr-respawn", action="store_true")
    parser.add_argument("--ddpr-respawn-trigger-m", type=float, default=1.75)
    parser.add_argument("--ddpr-respawn-mass", type=float, default=0.05)
    parser.add_argument("--ddpr-respawn-use-lambda-prior", action="store_true")
    parser.add_argument(
        "--ddpr-respawn-top-k",
        type=int,
        default=0,
        help="Respawn-only candidate count; <=0 uses --top-k",
    )
    parser.add_argument(
        "--ddpr-respawn-subset-size",
        type=int,
        default=0,
        help="Respawn-only ambiguity dimension; <=0 uses --subset-size",
    )
    parser.add_argument(
        "--ddpr-respawn-seed-radii-m",
        default="",
        help="Comma-separated covariance-axis position seed radii; empty uses center only",
    )
    parser.add_argument(
        "--ddpr-respawn-seed-directions",
        choices=("axes", "cube26"),
        default="axes",
    )
    parser.add_argument("--ddpr-respawn-history-seeds", type=int, default=0)
    parser.add_argument("--ddpr-respawn-history-separation-m", type=float, default=1.0)
    parser.add_argument("--ddpr-respawn-history-max-age-epochs", type=int, default=25)
    parser.add_argument("--out-diagnostics", type=Path, default=Path("results/wp23b/csv/basin_run2_epochs.csv"))
    parser.add_argument("--out-summary", type=Path, default=Path("results/wp23b/csv/basin_run2_summary.json"))
    parser.add_argument("--out-trajectory", type=Path, default=Path("results/wp23b/pos/basin_run2.csv"))
    parser.add_argument(
        "--out-trace",
        type=Path,
        default=None,
        help="Optional truth-free epoch trace for deterministic FIX replay",
    )
    parser.add_argument(
        "--out-evidence",
        type=Path,
        default=None,
        help="Optional observation evidence-provenance CSV",
    )
    parser.add_argument(
        "--out-basin-trace",
        type=Path,
        default=None,
        help="Optional truth-free per-basin trace for temporal replay",
    )
    parser.add_argument(
        "--out-integrity-satellite-diagnostics",
        type=Path,
        default=None,
        help="Optional truth-joined leave-one-satellite-out DDPR diagnostics",
    )
    args = parser.parse_args(argv)
    if not 0.0 <= float(args.integrity_satellite_cost_memory) < 1.0:
        parser.error("--integrity-satellite-cost-memory must be in [0, 1)")
    respawn_seed_radii = tuple(
        float(value)
        for value in str(args.ddpr_respawn_seed_radii_m).split(",")
        if value.strip()
    )
    if any(not np.isfinite(value) or value <= 0.0 for value in respawn_seed_radii):
        parser.error("--ddpr-respawn-seed-radii-m values must be positive")

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
        diversity_reserve_fraction=float(args.basin_diversity_reserve_fraction),
        diversity_radius_m=float(args.basin_diversity_radius_m),
        dedup_position_radius_m=float(args.basin_dedup_position_radius_m),
    )
    policy_config = TrustedFixPolicyConfig(
        gamma_threshold=float(args.fix_gamma),
        min_streak=int(args.fix_streak),
        min_ambiguities=int(args.subset_size),
        max_float_separation_m=float(args.fix_consistency_m),
        max_ddpr_separation_m=float(args.fix_ddpr_consistency_m),
        min_ddpr_pairs=int(args.fix_min_dd_pairs),
        max_ddpr_age_epochs=int(args.fix_max_ddpr_age_epochs),
    )
    commit_policy = TrustedFixCommitPolicy(policy_config)
    evidence_ledger = EvidenceLedger()
    traces: list[RTKEpochTrace] = []
    basin_trace_rows: list[dict[str, object]] = []
    integrity_satellite_rows: list[dict[str, object]] = []
    temporal_filter = (
        TemporalAmbiguityFilter(
            TemporalAmbiguityConfig(
                birth_mass=float(args.temporal_birth_mass),
                assignment_change_cost=float(args.temporal_change_cost),
                incompatible_cost=float(args.temporal_incompatible_cost),
                death_cost=float(args.temporal_death_cost),
                motion_sigma_m=float(args.temporal_motion_sigma_m),
            )
        )
        if args.enable_temporal_lineage else None
    )
    integrity_filter = (
        TemporalAmbiguityFilter(
            TemporalAmbiguityConfig(
                birth_mass=float(args.temporal_birth_mass),
                assignment_change_cost=float(args.temporal_change_cost),
                incompatible_cost=float(args.temporal_incompatible_cost),
                death_cost=float(args.temporal_death_cost),
                motion_sigma_m=float(args.temporal_motion_sigma_m),
            )
        )
        if args.enable_integrity_lineage else None
    )
    system_id_map = {"G": 0, "R": 1, "E": 2, "C": 3, "J": 4}
    integrity_tdcp_system_ids = {
        system_id_map[value.strip()]
        for value in str(args.integrity_tdcp_systems).split(",")
        if value.strip() in system_id_map
    }
    previous_tdcp_measurements = None
    integrity_satellite_cost_state: dict[str, float] = {}
    recovery_position_bank = (
        RecoveryPositionBank(
            max_seeds=int(args.ddpr_respawn_history_seeds),
            separation_m=float(args.ddpr_respawn_history_separation_m),
            max_age_epochs=int(args.ddpr_respawn_history_max_age_epochs),
        )
        if int(args.ddpr_respawn_history_seeds) > 0
        else None
    )

    times = np.asarray(data["times"], dtype=np.float64)
    respawn_subset_size = (
        int(args.ddpr_respawn_subset_size)
        if int(args.ddpr_respawn_subset_size) > 0
        else int(args.subset_size)
    )
    respawn_top_k = (
        int(args.ddpr_respawn_top_k)
        if int(args.ddpr_respawn_top_k) > 0
        else int(args.top_k)
    )
    basin_ddpr_sigma = (
        float(args.sigma_basin_dd_pr_m)
        if float(args.sigma_basin_dd_pr_m) > 0.0
        else float(args.sigma_dd_pr_m)
    )
    rows: list[dict[str, object]] = []
    n_birth_epochs = 0
    n_declared_fix = 0
    n_false_fix = 0
    n_correct_fix = 0
    n_gamma_fix = 0
    n_consistency_reject = 0
    n_respawn_epochs = 0
    n_float_resets = 0
    n_temporal_map_sub50 = 0
    n_temporal_map_disagreement = 0
    max_temporal_gamma = 0.0
    n_integrity_map_sub50 = 0
    n_integrity_map_disagreement = 0
    n_integrity_anchor_epochs = 0
    n_integrity_tdcp_intervals = 0
    n_integrity_satellite_exclusions = 0
    n_basin_oracle_sub50 = 0
    n_integrity_ball_gamma99 = 0
    n_integrity_ball_gamma99_correct = 0
    n_integrity_guard_pass = 0
    n_integrity_guard_pass_correct = 0
    max_integrity_gamma = 0.0
    max_integrity_ball_gamma = 0.0
    max_gamma = 0.0
    last_ddpr_epoch = -1_000_000
    last_ddpr_pairs = 0
    last_ddpr_nis = float("nan")

    for i, tow in enumerate(times):
        evidence_start = len(evidence_ledger)
        observation_id = f"tow={float(tow):.3f}"
        epoch_dt = 0.0
        integrity_tdcp = None
        current_tdcp_measurements = None
        if i > 0:
            epoch_dt = max(float(times[i] - times[i - 1]), 1e-3)
            float_kf.predict(epoch_dt)
            basin_pf.predict(epoch_dt)
            ddpr_guard.predict(epoch_dt)
        if integrity_filter is not None:
            current_tdcp_measurements = [
                measurement
                for measurement in _epoch_measurements(data, i)
                if int(measurement.system_id) in integrity_tdcp_system_ids
            ]
            if i > 0 and previous_tdcp_measurements is not None:
                integrity_tdcp = estimate_displacement_from_tdcp(
                    float_kf.position_ecef,
                    previous_tdcp_measurements,
                    current_tdcp_measurements,
                    epoch_dt,
                    min_sats=int(args.integrity_tdcp_min_sats),
                    max_postfit_rms_m=float(args.integrity_tdcp_max_postfit_rms_m),
                    slip_residual_threshold_m=float(
                        args.integrity_tdcp_slip_threshold_m
                    ),
                )
                n_integrity_tdcp_intervals += int(integrity_tdcp is not None)
            previous_tdcp_measurements = current_tdcp_measurements
        velocity, doppler_rms = _doppler_velocity(data, i, float_kf.position_ecef)
        if velocity is not None:
            velocity_sigma = max(0.5, min(float(doppler_rms), 5.0))
            float_kf.update_velocity(velocity, sigma_mps=velocity_sigma)
            basin_velocity_evidence = (
                basin_pf.update_velocity(velocity, sigma_mps=velocity_sigma)
                if basin_pf.basins else None
            )
            ddpr_guard.update_velocity(velocity, sigma_mps=velocity_sigma)
            for target in ("float_kf", "ddpr_guard"):
                evidence_ledger.record(
                    epoch=i,
                    target=target,
                    source="doppler_velocity",
                    observation_id=observation_id,
                    n_rows=3,
                )
            if basin_velocity_evidence is not None:
                evidence_ledger.record(
                    epoch=i,
                    target="basin_pf",
                    source="doppler_velocity",
                    observation_id=observation_id,
                    n_rows=3,
                    log_evidence=basin_velocity_evidence.log_marginal,
                )

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
            for target in ("float_kf", "ddpr_guard"):
                evidence_ledger.record(
                    epoch=i,
                    target=target,
                    source="dd_pseudorange",
                    observation_id=observation_id,
                    n_rows=int(dd_pr.n_dd),
                )
        if dd_cp is not None and int(dd_cp.n_dd) >= 3:
            cp_diag = float_kf.update_carrier(
                dd_cp,
                dd_pseudorange_result=dd_pr,
                sigma_cp_cycles=float(args.sigma_float_cp_cycles),
                slip_threshold_cycles=float(args.float_slip_threshold_cycles),
            )
            n_float_resets += int(cp_diag.ambiguities_reset)
            evidence_ledger.record(
                epoch=i,
                target="float_kf",
                source="dd_carrier",
                observation_id=observation_id,
                n_rows=int(dd_cp.n_dd),
            )

        generations = float_kf.ambiguity_generations()
        if recovery_position_bank is not None and basin_pf.basins:
            recovery_position_bank.update(
                i,
                np.asarray(
                    [basin.conditional.mean[:3] for basin in basin_pf.basins],
                    dtype=np.float64,
                ),
                np.asarray(
                    [basin.log_weight for basin in basin_pf.basins],
                    dtype=np.float64,
                ),
            )
        active_versioned = {(key, generation) for key, generation in generations.items()}
        basin_pf.retain_compatible(active_versioned)
        if dd_pr is not None and basin_pf.basins:
            basin_pr_evidence = basin_pf.update_pseudorange(
                dd_pr, sigma_pr_m=basin_ddpr_sigma
            )
            evidence_ledger.record(
                epoch=i,
                target="basin_pf",
                source="dd_pseudorange",
                observation_id=observation_id,
                n_rows=int(dd_pr.n_dd),
                log_evidence=basin_pr_evidence.log_marginal,
            )
        if dd_cp is not None and basin_pf.basins:
            basin_cp_evidence = basin_pf.update_fixed_carrier(
                dd_cp,
                generations,
                sigma_cp_cycles=float(args.sigma_fixed_cp_cycles),
            )
            evidence_ledger.record(
                epoch=i,
                target="basin_pf",
                source="dd_carrier",
                observation_id=observation_id,
                n_rows=int(dd_cp.n_dd),
                log_evidence=basin_cp_evidence.log_marginal,
            )

        pre_birth_map = (
            max(basin_pf.basins, key=lambda basin: basin.log_weight)
            if basin_pf.basins else None
        )
        respawn_triggered = bool(
            args.enable_ddpr_respawn
            and dd_cp is not None
            and dd_pr is not None
            and int(dd_pr.n_dd) >= int(args.fix_min_dd_pairs)
            and (
                pre_birth_map is None
                or np.linalg.norm(
                    pre_birth_map.conditional.mean[:3] - ddpr_guard.mean[:3]
                ) > float(args.ddpr_respawn_trigger_m)
            )
        )

        n_candidates = 0
        n_respawn_candidates = 0
        n_respawn_position_seeds = 0
        n_respawn_history_seeds = 0
        respawn_oracle_min_error = float("nan")
        respawn_oracle_rank = -1
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

        if respawn_triggered and dd_cp is not None:
            respawn_positions = covariance_axis_position_seeds(
                ddpr_guard.mean[:3],
                ddpr_guard.covariance[:3, :3],
                respawn_seed_radii,
                direction_mode=str(args.ddpr_respawn_seed_directions),
            )
            if recovery_position_bank is not None:
                respawn_position_list = list(respawn_positions)
                for history_position in recovery_position_bank.positions:
                    if all(
                        np.linalg.norm(history_position - existing) > 1.0e-3
                        for existing in respawn_position_list
                    ):
                        respawn_position_list.append(history_position)
                        n_respawn_history_seeds += 1
                respawn_positions = tuple(respawn_position_list)
            n_respawn_position_seeds = len(respawn_positions)
            assignments = []
            conditionals = []
            all_respawn_residuals: list[float] = []
            for respawn_position in respawn_positions:
                respawn_seed = ddpr_centered_ambiguity_seed(
                    dd_cp,
                    respawn_position,
                    ddpr_guard.covariance[:3, :3],
                    sigma_cp_cycles=float(args.sigma_fixed_cp_cycles),
                )
                available = np.asarray(
                    [
                        j
                        for j, key in enumerate(respawn_seed.keys)
                        if key in generations
                    ],
                    dtype=np.int64,
                )
                if available.size < respawn_subset_size:
                    continue
                variances = np.diag(respawn_seed.qahat_cycles2)[available]
                selected = available[np.argsort(variances)[:respawn_subset_size]]
                selected = np.sort(selected)
                respawn_keys = tuple(respawn_seed.keys[j] for j in selected)
                candidates, seed_residuals = integer_search(
                    respawn_seed.ahat_cycles[selected],
                    respawn_seed.qahat_cycles2[np.ix_(selected, selected)],
                    n_candidates=respawn_top_k,
                )
                for candidate in candidates:
                    position, covariance, _distance = condition_respawn_position(
                        respawn_seed, respawn_keys, candidate
                    )
                    assignments.append(
                        {
                            (key, generations[key]): int(value)
                            for key, value in zip(respawn_keys, candidate)
                        }
                    )
                    conditionals.append(
                        BasinKalmanState.from_position(
                            position,
                            covariance,
                            velocity_ecef=ddpr_guard.mean[3:6],
                            velocity_sigma_mps=1.0,
                            accel_process_sigma_mps2=3.0,
                        )
                    )
                all_respawn_residuals.extend(
                    float(value) for value in np.asarray(seed_residuals).reshape(-1)
                )
            if conditionals:
                # Diagnostic only: truth never changes candidates, weights,
                # output selection, or the FIX gate.
                epoch_ref = truth.get(round(float(tow), 1))
                if epoch_ref is not None and conditionals:
                    candidate_errors = np.asarray(
                        [
                            float(np.linalg.norm(state.mean[:3] - epoch_ref))
                            for state in conditionals
                        ],
                        dtype=np.float64,
                    )
                    respawn_oracle_rank = int(np.argmin(candidate_errors)) + 1
                    respawn_oracle_min_error = float(np.min(candidate_errors))
                if assignments:
                    basin_pf.spawn(
                        assignments,
                        conditionals,
                        prior_mass=float(args.ddpr_respawn_mass),
                        candidate_log_weights=(
                            -0.5
                            * np.asarray(all_respawn_residuals, dtype=np.float64)
                            if args.ddpr_respawn_use_lambda_prior else None
                        ),
                    )
                    n_respawn_candidates = len(assignments)
                    n_respawn_epochs += 1

        posterior = basin_pf.posterior()
        position_cluster = basin_pf.position_cluster_posterior(
            radius_m=float(args.position_cluster_radius_m)
        )
        max_gamma = max(max_gamma, float(posterior.gamma))
        map_candidates = [
            basin for basin in basin_pf.basins if basin.assignment == posterior.map_assignment
        ]
        map_basin = max(map_candidates, key=lambda basin: basin.log_weight) if map_candidates else None
        if args.out_basin_trace is not None:
            for basin in basin_pf.basins:
                basin_trace_rows.append(
                    {
                        "epoch": i,
                        "tow": float(tow),
                        "basin_id": basin.basin_id,
                        "assignment_id": ambiguity_assignment_id(basin.assignment),
                        "assignment_json": ambiguity_assignment_json(basin.assignment),
                        "epoch_log_likelihood": float(basin.epoch_log_marginal),
                        "cumulative_log_marginal": float(basin.cumulative_log_marginal),
                        "log_weight": float(basin.log_weight),
                        "ecef_x": float(basin.conditional.mean[0]),
                        "ecef_y": float(basin.conditional.mean[1]),
                        "ecef_z": float(basin.conditional.mean[2]),
                        "velocity_x": float(basin.conditional.mean[3]),
                        "velocity_y": float(basin.conditional.mean[4]),
                        "velocity_z": float(basin.conditional.mean[5]),
                        "birth_epoch": int(basin.birth_epoch),
                        "lineage": "|".join(basin.lineage),
                    }
                )
        temporal_posterior = None
        temporal_map_basin = None
        if temporal_filter is not None:
            temporal_candidates = [
                TemporalAmbiguityCandidate(
                    candidate_id=ambiguity_assignment_id(basin.assignment),
                    assignment=basin.assignment,
                    epoch_log_likelihood=float(basin.epoch_log_marginal),
                    position_ecef=basin.conditional.mean[:3],
                    velocity_ecef=basin.conditional.mean[3:6],
                )
                for basin in basin_pf.basins
            ]
            temporal_posterior = temporal_filter.step(
                i, epoch_dt, temporal_candidates
            )
            temporal_map_basin = next(
                (
                    basin for basin in basin_pf.basins
                    if ambiguity_assignment_id(basin.assignment)
                    == temporal_posterior.map_candidate_id
                ),
                None,
            )
            max_temporal_gamma = max(
                max_temporal_gamma, float(temporal_posterior.gamma)
            )
            n_temporal_map_disagreement += int(
                temporal_map_basin is not None
                and map_basin is not None
                and temporal_map_basin.basin_id != map_basin.basin_id
            )
        integrity_posterior = None
        integrity_map_basin = None
        integrity_result = None
        integrity_excluded_satellite = ""
        integrity_position_ball = None
        if integrity_filter is not None and basin_pf.basins:
            integrity_scores = np.zeros(len(basin_pf.basins), dtype=np.float64)
            if dd_pr is not None:
                excluded_satellites: tuple[str, ...] = ()
                if args.integrity_exclude_max_cost_satellite:
                    satellite_cost = satellite_pair_costs(
                        dd_pr,
                        ddpr_guard.mean[:3],
                        scale_m=float(args.integrity_scale_m),
                    )
                    cost_memory = float(args.integrity_satellite_cost_memory)
                    for satellite, current_cost in zip(
                        satellite_cost.satellite_ids,
                        satellite_cost.mean_pair_costs,
                    ):
                        previous_cost = integrity_satellite_cost_state.get(satellite)
                        integrity_satellite_cost_state[satellite] = (
                            float(current_cost)
                            if previous_cost is None
                            else cost_memory * previous_cost
                            + (1.0 - cost_memory) * float(current_cost)
                        )
                    integrity_excluded_satellite = max(
                        satellite_cost.satellite_ids,
                        key=integrity_satellite_cost_state.__getitem__,
                    )
                    excluded_satellites = (integrity_excluded_satellite,)
                    n_integrity_satellite_exclusions += 1
                integrity_result = multipivot_ddpr_scores(
                    dd_pr,
                    np.asarray(
                        [basin.conditional.mean[:3] for basin in basin_pf.basins],
                        dtype=np.float64,
                    ),
                    scale_m=float(args.integrity_scale_m),
                    trim_largest_pairs=int(args.integrity_trim_pairs),
                    excluded_satellites=excluded_satellites,
                )
                integrity_scores = (
                    float(args.integrity_weight) * integrity_result.scores
                )
                n_integrity_anchor_epochs += 1
                evidence_ledger.record(
                    epoch=i,
                    target="integrity_lineage",
                    source="multipivot_dd_pseudorange",
                    observation_id=observation_id,
                    n_rows=int(dd_pr.n_dd),
                    log_evidence=float(np.max(integrity_scores)),
                )
            integrity_candidates = [
                TemporalAmbiguityCandidate(
                    candidate_id=ambiguity_assignment_id(basin.assignment),
                    assignment=basin.assignment,
                    epoch_log_likelihood=float(integrity_scores[index]),
                    position_ecef=basin.conditional.mean[:3],
                    velocity_ecef=basin.conditional.mean[3:6],
                )
                for index, basin in enumerate(basin_pf.basins)
            ]
            integrity_motion: dict[str, object] = {"motion_mode": "none"}
            if integrity_tdcp is not None:
                integrity_motion = {
                    "motion_mode": "external",
                    "external_displacement_ecef_m": integrity_tdcp.displacement_ecef_m,
                    "external_covariance_m2": integrity_tdcp.covariance_m2,
                }
                evidence_ledger.record(
                    epoch=i,
                    target="integrity_lineage",
                    source="tdcp_displacement",
                    observation_id=(
                        f"tow={float(times[i - 1]):.3f}->{float(tow):.3f}"
                    ),
                    n_rows=int(integrity_tdcp.n_used),
                )
            integrity_posterior = integrity_filter.step(
                i,
                epoch_dt,
                integrity_candidates,
                **integrity_motion,
            )
            integrity_position_ball = integrity_filter.map_position_ball(
                float(args.position_cluster_radius_m)
            )
            integrity_map_basin = next(
                (
                    basin
                    for basin in basin_pf.basins
                    if ambiguity_assignment_id(basin.assignment)
                    == integrity_posterior.map_candidate_id
                ),
                None,
            )
            max_integrity_gamma = max(
                max_integrity_gamma, float(integrity_posterior.gamma)
            )
            n_integrity_map_disagreement += int(
                integrity_map_basin is not None
                and map_basin is not None
                and integrity_map_basin.basin_id != map_basin.basin_id
            )
        elif integrity_filter is not None:
            integrity_filter.step(i, epoch_dt, ())
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
        assignment_id = (
            ambiguity_assignment_id(posterior.map_assignment)
            if map_basin is not None else ""
        )
        policy_input = TrustedFixPolicyInput(
            epoch=i,
            assignment_id=assignment_id,
            gamma=float(posterior.gamma),
            n_ambiguities=len(posterior.map_assignment),
            map_float_separation_m=map_float_separation,
            map_ddpr_separation_m=map_ddpr_separation,
            last_ddpr_pairs=last_ddpr_pairs,
            ddpr_age_epochs=ddpr_age_epochs,
        )
        commit = commit_policy.evaluate(policy_input)
        gamma_fixed = bool(
            commit.gamma_eligible and commit.fix_streak >= int(args.fix_streak)
        )
        if gamma_fixed != bool(posterior.fixed and map_basin is not None):
            raise RuntimeError(
                f"legacy/replayable gamma FIX mismatch at epoch {i}: "
                f"{posterior.fixed} != {gamma_fixed}"
            )
        gate = commit.gate
        consistency_pass = gate.passed
        fixed = commit.fixed
        n_gamma_fix += int(gamma_fixed)
        n_consistency_reject += int(gamma_fixed and not consistency_pass)
        output_position = (
            map_basin.conditional.mean[:3].copy() if fixed else float_kf.position_ecef
        )
        ref = truth.get(round(float(tow), 1))
        basin_oracle_min_error = (
            min(
                float(np.linalg.norm(basin.conditional.mean[:3] - ref))
                for basin in basin_pf.basins
            )
            if basin_pf.basins and ref is not None
            else float("nan")
        )
        n_basin_oracle_sub50 += int(basin_oracle_min_error < 0.5)
        if (
            args.out_integrity_satellite_diagnostics is not None
            and integrity_result is not None
            and ref is not None
        ):
            integrity_positions = np.asarray(
                [basin.conditional.mean[:3] for basin in basin_pf.basins],
                dtype=np.float64,
            )
            integrity_errors = np.linalg.norm(
                integrity_positions - ref[None, :], axis=1
            )
            full_error = float(integrity_errors[integrity_result.best_index])
            guard_satellite_cost = satellite_pair_costs(
                dd_pr,
                ddpr_guard.mean[:3],
                scale_m=float(args.integrity_scale_m),
            )
            selected_satellite_cost = satellite_pair_costs(
                dd_pr,
                integrity_positions[integrity_result.best_index],
                scale_m=float(args.integrity_scale_m),
            )
            guard_cost_by_satellite = dict(
                zip(
                    guard_satellite_cost.satellite_ids,
                    guard_satellite_cost.mean_pair_costs,
                )
            )
            selected_cost_by_satellite = dict(
                zip(
                    selected_satellite_cost.satellite_ids,
                    selected_satellite_cost.mean_pair_costs,
                )
            )
            satellite_ids = sorted(
                set(str(value) for value in dd_pr.ref_sat_ids + dd_pr.sat_ids)
            )
            for excluded_satellite in satellite_ids:
                try:
                    excluded_result = multipivot_ddpr_scores(
                        dd_pr,
                        integrity_positions,
                        scale_m=float(args.integrity_scale_m),
                        trim_largest_pairs=int(args.integrity_trim_pairs),
                        excluded_satellites=(excluded_satellite,),
                    )
                except ValueError:
                    continue
                ordered_scores = np.sort(excluded_result.scores)
                score_margin = (
                    float(ordered_scores[-1] - ordered_scores[-2])
                    if len(ordered_scores) >= 2
                    else float("inf")
                )
                excluded_error = float(
                    integrity_errors[excluded_result.best_index]
                )
                integrity_satellite_rows.append(
                    {
                        "epoch": i,
                        "tow": float(tow),
                        "excluded_satellite": excluded_satellite,
                        "full_selected_error_m": full_error,
                        "excluded_selected_error_m": excluded_error,
                        "full_selected_sub50cm": int(full_error < 0.5),
                        "excluded_selected_sub50cm": int(excluded_error < 0.5),
                        "exclusion_recovers_sub50cm": int(
                            full_error >= 0.5 and excluded_error < 0.5
                        ),
                        "exclusion_breaks_sub50cm": int(
                            full_error < 0.5 and excluded_error >= 0.5
                        ),
                        "oracle_min_error_m": basin_oracle_min_error,
                        "excluded_best_probability": float(
                            excluded_result.probabilities[
                                excluded_result.best_index
                            ]
                        ),
                        "excluded_score_margin": score_margin,
                        "excluded_best_assignment_id": ambiguity_assignment_id(
                            basin_pf.basins[excluded_result.best_index].assignment
                        ),
                        "guard_mean_pair_cost": float(
                            guard_cost_by_satellite[excluded_satellite]
                        ),
                        "selected_mean_pair_cost": float(
                            selected_cost_by_satellite[excluded_satellite]
                        ),
                        "n_constellations": int(
                            excluded_result.n_constellations
                        ),
                        "n_satellites": int(excluded_result.n_satellites),
                    }
                )
        output_error = (
            float(np.linalg.norm(output_position - ref))
            if ref is not None and np.all(np.isfinite(ref))
            else float("nan")
        )
        map_error = (
            float(np.linalg.norm(map_basin.conditional.mean[:3] - ref))
            if map_basin is not None and ref is not None
            else float("nan")
        )
        temporal_map_error = (
            float(np.linalg.norm(temporal_map_basin.conditional.mean[:3] - ref))
            if temporal_map_basin is not None and ref is not None
            else float("nan")
        )
        n_temporal_map_sub50 += int(temporal_map_error < 0.5)
        integrity_map_error = (
            float(np.linalg.norm(integrity_map_basin.conditional.mean[:3] - ref))
            if integrity_map_basin is not None and ref is not None
            else float("nan")
        )
        n_integrity_map_sub50 += int(integrity_map_error < 0.5)
        integrity_ball_error = (
            float(np.linalg.norm(integrity_position_ball.mean_position_ecef - ref))
            if integrity_position_ball is not None and ref is not None
            else float("nan")
        )
        integrity_ball_gamma = (
            0.0
            if integrity_position_ball is None
            else float(integrity_position_ball.probability)
        )
        max_integrity_ball_gamma = max(
            max_integrity_ball_gamma, integrity_ball_gamma
        )
        n_integrity_ball_gamma99 += int(integrity_ball_gamma > 0.99)
        n_integrity_ball_gamma99_correct += int(
            integrity_ball_gamma > 0.99 and integrity_ball_error < 0.5
        )
        integrity_map_float_separation = (
            float(
                np.linalg.norm(
                    integrity_map_basin.conditional.mean[:3]
                    - float_kf.position_ecef
                )
            )
            if integrity_map_basin is not None
            else float("nan")
        )
        integrity_map_ddpr_separation = (
            float(
                np.linalg.norm(
                    integrity_map_basin.conditional.mean[:3]
                    - ddpr_guard.mean[:3]
                )
            )
            if integrity_map_basin is not None
            else float("nan")
        )
        integrity_guard_pass = bool(
            integrity_map_basin is not None
            and integrity_map_float_separation <= float(args.fix_consistency_m)
            and integrity_map_ddpr_separation <= float(args.fix_ddpr_consistency_m)
            and last_ddpr_pairs >= int(args.fix_min_dd_pairs)
            and ddpr_age_epochs <= int(args.fix_max_ddpr_age_epochs)
        )
        n_integrity_guard_pass += int(integrity_guard_pass)
        n_integrity_guard_pass_correct += int(
            integrity_guard_pass and integrity_map_error < 0.5
        )
        cluster_error = (
            float(np.linalg.norm(position_cluster.mean_position_ecef - ref))
            if ref is not None and np.all(np.isfinite(position_cluster.mean_position_ecef))
            else float("nan")
        )
        if fixed:
            n_declared_fix += 1
            if output_error < 0.5:
                n_correct_fix += 1
            else:
                n_false_fix += 1
        traces.append(
            RTKEpochTrace(
                epoch=i,
                tow=float(tow),
                assignment_id=assignment_id,
                gamma=float(posterior.gamma),
                n_ambiguities=len(posterior.map_assignment),
                map_float_separation_m=map_float_separation,
                map_ddpr_separation_m=map_ddpr_separation,
                last_ddpr_pairs=int(last_ddpr_pairs),
                ddpr_age_epochs=int(ddpr_age_epochs),
                ecef_x=float(output_position[0]),
                ecef_y=float(output_position[1]),
                ecef_z=float(output_position[2]),
                gamma_eligible=commit.gamma_eligible,
                fix_streak=commit.fix_streak,
                fixed=commit.fixed,
                evidence_records=len(evidence_ledger) - evidence_start,
            )
        )
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
                "map_error_m": map_error,
                "temporal_lineage_enabled": int(temporal_filter is not None),
                "temporal_map_assignment_id": (
                    "" if temporal_posterior is None
                    else temporal_posterior.map_candidate_id
                ),
                "temporal_gamma": (
                    0.0 if temporal_posterior is None
                    else float(temporal_posterior.gamma)
                ),
                "temporal_ess": (
                    0.0 if temporal_posterior is None
                    else float(temporal_posterior.ess)
                ),
                "temporal_dwell_epochs": (
                    0 if temporal_posterior is None
                    else int(temporal_posterior.dwell_epochs)
                ),
                "temporal_map_error_m": temporal_map_error,
                "integrity_lineage_enabled": int(integrity_filter is not None),
                "integrity_anchor_available": int(integrity_result is not None),
                "integrity_excluded_satellite": integrity_excluded_satellite,
                "integrity_tdcp_available": int(integrity_tdcp is not None),
                "integrity_tdcp_postfit_rms_m": (
                    float("nan")
                    if integrity_tdcp is None
                    else float(integrity_tdcp.postfit_rms_m)
                ),
                "integrity_map_assignment_id": (
                    "" if integrity_posterior is None
                    else integrity_posterior.map_candidate_id
                ),
                "integrity_gamma": (
                    0.0
                    if integrity_posterior is None
                    else float(integrity_posterior.gamma)
                ),
                "integrity_ess": (
                    0.0
                    if integrity_posterior is None
                    else float(integrity_posterior.ess)
                ),
                "integrity_dwell_epochs": (
                    0
                    if integrity_posterior is None
                    else int(integrity_posterior.dwell_epochs)
                ),
                "integrity_map_error_m": integrity_map_error,
                "integrity_map_ecef_x": (
                    float("nan")
                    if integrity_map_basin is None
                    else float(integrity_map_basin.conditional.mean[0])
                ),
                "integrity_map_ecef_y": (
                    float("nan")
                    if integrity_map_basin is None
                    else float(integrity_map_basin.conditional.mean[1])
                ),
                "integrity_map_ecef_z": (
                    float("nan")
                    if integrity_map_basin is None
                    else float(integrity_map_basin.conditional.mean[2])
                ),
                "integrity_map_float_separation_m": integrity_map_float_separation,
                "integrity_map_ddpr_separation_m": integrity_map_ddpr_separation,
                "integrity_guard_pass": int(integrity_guard_pass),
                "integrity_position_ball_gamma": integrity_ball_gamma,
                "integrity_position_ball_members": (
                    0
                    if integrity_position_ball is None
                    else int(integrity_position_ball.n_members)
                ),
                "integrity_position_ball_spread_m": (
                    float("nan")
                    if integrity_position_ball is None
                    else float(integrity_position_ball.rms_spread_m)
                ),
                "integrity_position_ball_error_m": integrity_ball_error,
                "basin_oracle_min_error_m": basin_oracle_min_error,
                "basin_oracle_sub50cm_available": int(
                    basin_oracle_min_error < 0.5
                ),
                "position_cluster_error_m": cluster_error,
                "position_cluster_gamma": float(position_cluster.gamma),
                "position_cluster_spread_m": float(position_cluster.rms_spread_m),
                "position_cluster_members": int(position_cluster.n_members),
                "position_cluster_float_separation_m": (
                    float(
                        np.linalg.norm(
                            position_cluster.mean_position_ecef - float_kf.position_ecef
                        )
                    )
                    if np.all(np.isfinite(position_cluster.mean_position_ecef))
                    else float("nan")
                ),
                "position_cluster_ddpr_separation_m": (
                    float(
                        np.linalg.norm(
                            position_cluster.mean_position_ecef - ddpr_guard.mean[:3]
                        )
                    )
                    if np.all(np.isfinite(position_cluster.mean_position_ecef))
                    else float("nan")
                ),
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
                "ambiguities_reset": (
                    0 if cp_diag is None else int(cp_diag.ambiguities_reset)
                ),
                "gamma": float(posterior.gamma),
                "fix_streak": int(commit.fix_streak),
                "map_assignment_id": assignment_id,
                "n_basins": int(posterior.n_basins),
                "basin_ess": float(posterior.ess),
                "map_n_ambiguities": len(posterior.map_assignment),
                "n_candidates_born": n_candidates,
                "respawn_triggered": int(respawn_triggered),
                "n_respawn_candidates_born": n_respawn_candidates,
                "n_respawn_position_seeds": n_respawn_position_seeds,
                "n_respawn_history_seeds": n_respawn_history_seeds,
                "respawn_oracle_min_error_m": respawn_oracle_min_error,
                "respawn_oracle_rank": int(respawn_oracle_rank),
                "n_dd_pr": 0 if dd_pr is None else int(dd_pr.n_dd),
                "n_dd_cp": 0 if dd_cp is None else int(dd_cp.n_dd),
            }
        )

    evidence_audit = evidence_ledger.audit()
    replayed = replay_fix_decisions(traces, policy_config)
    replay_mismatches = sum(
        decision.fixed != trace.fixed or decision.fix_streak != trace.fix_streak
        for decision, trace in zip(replayed, traces)
    )
    if replay_mismatches:
        raise RuntimeError(f"online/replay FIX mismatch count: {replay_mismatches}")
    false_rate = 100.0 * n_false_fix / n_declared_fix if n_declared_fix else 0.0
    summary = {
        "run": str(args.run),
        "n_epochs": len(rows),
        "subset_size": int(args.subset_size),
        "top_k": int(args.top_k),
        "basin_diversity_reserve_fraction": float(
            args.basin_diversity_reserve_fraction
        ),
        "basin_diversity_radius_m": float(args.basin_diversity_radius_m),
        "basin_dedup_position_radius_m": (
            float(args.basin_dedup_position_radius_m)
            if np.isfinite(float(args.basin_dedup_position_radius_m))
            else None
        ),
        "fix_gamma_threshold": float(args.fix_gamma),
        "fix_min_streak": int(args.fix_streak),
        "fix_float_consistency_m": float(args.fix_consistency_m),
        "fix_ddpr_consistency_m": float(args.fix_ddpr_consistency_m),
        "fix_min_dd_pairs": int(args.fix_min_dd_pairs),
        "fix_max_ddpr_age_epochs": int(args.fix_max_ddpr_age_epochs),
        "sigma_basin_dd_pr_m": float(basin_ddpr_sigma),
        "birth_epochs": int(n_birth_epochs),
        "ddpr_respawn_enabled": bool(args.enable_ddpr_respawn),
        "ddpr_respawn_subset_size": int(respawn_subset_size),
        "ddpr_respawn_top_k": int(respawn_top_k),
        "ddpr_respawn_lambda_prior": bool(args.ddpr_respawn_use_lambda_prior),
        "ddpr_respawn_seed_radii_m": list(respawn_seed_radii),
        "ddpr_respawn_seed_directions": str(args.ddpr_respawn_seed_directions),
        "ddpr_respawn_history_seeds": int(args.ddpr_respawn_history_seeds),
        "ddpr_respawn_history_separation_m": float(
            args.ddpr_respawn_history_separation_m
        ),
        "ddpr_respawn_history_max_age_epochs": int(
            args.ddpr_respawn_history_max_age_epochs
        ),
        "ddpr_respawn_epochs": int(n_respawn_epochs),
        "temporal_lineage_enabled": bool(args.enable_temporal_lineage),
        "temporal_map_disagreement_epochs": int(n_temporal_map_disagreement),
        "temporal_map_sub50cm_epochs": int(n_temporal_map_sub50),
        "max_temporal_gamma": float(max_temporal_gamma),
        "integrity_lineage_enabled": bool(args.enable_integrity_lineage),
        "integrity_scale_m": float(args.integrity_scale_m),
        "integrity_trim_pairs": int(args.integrity_trim_pairs),
        "integrity_weight": float(args.integrity_weight),
        "integrity_exclude_max_cost_satellite": bool(
            args.integrity_exclude_max_cost_satellite
        ),
        "integrity_satellite_cost_memory": float(
            args.integrity_satellite_cost_memory
        ),
        "integrity_satellite_exclusions": int(n_integrity_satellite_exclusions),
        "integrity_anchor_epochs": int(n_integrity_anchor_epochs),
        "integrity_tdcp_intervals": int(n_integrity_tdcp_intervals),
        "integrity_map_disagreement_epochs": int(n_integrity_map_disagreement),
        "integrity_map_sub50cm_epochs": int(n_integrity_map_sub50),
        "basin_oracle_sub50cm_epochs": int(n_basin_oracle_sub50),
        "integrity_selection_given_oracle_pct": float(
            100.0 * n_integrity_map_sub50 / max(n_basin_oracle_sub50, 1)
        ),
        "max_integrity_gamma": float(max_integrity_gamma),
        "max_integrity_position_ball_gamma": float(max_integrity_ball_gamma),
        "integrity_position_ball_gamma99_epochs": int(n_integrity_ball_gamma99),
        "integrity_position_ball_gamma99_correct_epochs": int(
            n_integrity_ball_gamma99_correct
        ),
        "integrity_guard_pass_epochs": int(n_integrity_guard_pass),
        "integrity_guard_pass_correct_epochs": int(n_integrity_guard_pass_correct),
        "integrity_guard_pass_false_epochs": int(
            n_integrity_guard_pass - n_integrity_guard_pass_correct
        ),
        "float_ambiguity_resets": int(n_float_resets),
        "declared_fix_epochs": int(n_declared_fix),
        "gamma_fix_epochs": int(n_gamma_fix),
        "consistency_reject_epochs": int(n_consistency_reject),
        "correct_fix_epochs": int(n_correct_fix),
        "false_fix_epochs": int(n_false_fix),
        "false_fix_pct": float(false_rate),
        "max_gamma": float(max_gamma),
        "sub50cm_all_epochs": int(sum(float(row["output_error_m"]) < 0.5 for row in rows)),
        "evidence_records": int(evidence_audit.n_records),
        "evidence_updates": int(evidence_audit.n_updates),
        "evidence_beta_errors": int(evidence_audit.beta_error_count),
        "commit_replay_mismatches": int(replay_mismatches),
    }
    args.out_diagnostics.parent.mkdir(parents=True, exist_ok=True)
    with args.out_diagnostics.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _write_trajectory(args.out_trajectory, rows)
    if args.out_trace is not None:
        args.out_trace.parent.mkdir(parents=True, exist_ok=True)
        with args.out_trace.open("w", newline="") as fh:
            trace_rows = [trace.row() for trace in traces]
            writer = csv.DictWriter(fh, fieldnames=list(trace_rows[0]))
            writer.writeheader()
            writer.writerows(trace_rows)
    if args.out_evidence is not None:
        args.out_evidence.parent.mkdir(parents=True, exist_ok=True)
        with args.out_evidence.open("w", newline="") as fh:
            evidence_rows = evidence_ledger.rows()
            writer = csv.DictWriter(fh, fieldnames=list(evidence_rows[0]))
            writer.writeheader()
            writer.writerows(evidence_rows)
    if args.out_basin_trace is not None and basin_trace_rows:
        args.out_basin_trace.parent.mkdir(parents=True, exist_ok=True)
        with args.out_basin_trace.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(basin_trace_rows[0]))
            writer.writeheader()
            writer.writerows(basin_trace_rows)
    if (
        args.out_integrity_satellite_diagnostics is not None
        and integrity_satellite_rows
    ):
        args.out_integrity_satellite_diagnostics.parent.mkdir(
            parents=True, exist_ok=True
        )
        with args.out_integrity_satellite_diagnostics.open("w", newline="") as fh:
            writer = csv.DictWriter(
                fh, fieldnames=list(integrity_satellite_rows[0])
            )
            writer.writeheader()
            writer.writerows(integrity_satellite_rows)
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"wrote {args.out_diagnostics}")
    print(f"wrote {args.out_trajectory}")


if __name__ == "__main__":
    main()
