#!/usr/bin/env python3
"""Fast-window segment FGO harness for multi-family DD carrier fixes."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
for _p in (_PROJECT_ROOT / "python", _SCRIPT_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from analyze_ppc_dd_carrier_lambda_support import _score_pass_rows  # noqa: E402
from exp_ppc_ctrbpf_fgo import (  # noqa: E402
    _build_dd_measurements,
    _load_hybrid_pos_file,
    _write_pos_file,
)
from gnss_gpu.dd_carrier import DDCarrierComputer  # noqa: E402
from gnss_gpu.dd_pseudorange import DDPseudorangeComputer  # noqa: E402
from gnss_gpu.local_fgo import (  # noqa: E402
    DDCarrierEpoch,
    DDPseudorangeEpoch,
    LambdaFixConfig,
    LocalFgoConfig,
    LocalFgoProblem,
    LocalFgoWindow,
    UndiffPseudorangeEpoch,
    solve_local_fgo_with_lambda,
)
from ppc_window_geometry import load_ppc_window_geometry  # noqa: E402


DATA_ROOT = Path("/media/sasaki/aiueo/ai_coding_ws/datasets/PPC-Dataset-data")
RESULTS = _SCRIPT_DIR / "results"
DEFAULT_SEED = (
    RESULTS
    / "libgnss_diag_phase10"
    / "full_ratio15_lock3_trustedseed_rtkout5mlc1c005oG"
    / "nagoya_run2_full.pos"
)


def _parse_systems(text: str) -> tuple[str, ...]:
    return tuple(s.strip() for s in str(text).split(",") if s.strip())


def _load_seed_positions(pos_path: Path, tows: np.ndarray) -> np.ndarray:
    pos_by_tow, _status = _load_hybrid_pos_file(pos_path)
    rows: list[np.ndarray] = []
    missing: list[float] = []
    for tow in np.asarray(tows, dtype=np.float64):
        key = round(float(tow), 1)
        pos = pos_by_tow.get(key)
        if pos is None:
            missing.append(key)
            continue
        rows.append(np.asarray(pos, dtype=np.float64))
    if missing:
        raise ValueError(f"seed pos missing {len(missing)} requested TOWs, first={missing[:5]}")
    return np.vstack(rows)


def _truth_by_tow(data: dict) -> dict[float, np.ndarray]:
    return {
        round(float(t), 1): np.asarray(p, dtype=np.float64)
        for t, p in zip(data["times"], data["ground_truth"], strict=True)
    }


def _error_stats(
    *,
    tows: np.ndarray,
    positions: np.ndarray,
    truth: dict[float, np.ndarray],
) -> dict[str, float | int]:
    errs: list[float] = []
    for tow, pos in zip(tows, positions, strict=True):
        ref = truth.get(round(float(tow), 1))
        if ref is None:
            continue
        errs.append(float(np.linalg.norm(np.asarray(pos, dtype=np.float64) - ref)))
    if not errs:
        return {
            "rows": 0,
            "pass_rows_05m": 0,
            "median_error_m": float("nan"),
            "p90_error_m": float("nan"),
        }
    vals = np.asarray(errs, dtype=np.float64)
    return {
        "rows": int(vals.size),
        "pass_rows_05m": int(np.count_nonzero(vals <= 0.5)),
        "median_error_m": float(np.median(vals)),
        "p90_error_m": float(np.percentile(vals, 90.0)),
    }


def _write_summary(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row.keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerow(row)


def _build_anchor_priors(
    *,
    tows: np.ndarray,
    seed_positions: np.ndarray,
    truth: dict[float, np.ndarray],
    source: str,
    anchor_pos_path: Path | None,
    sigma_m: float,
    every_n: int,
    max_error_m: float,
) -> tuple[np.ndarray | None, np.ndarray | None, dict[str, object]]:
    source = str(source).strip().lower()
    stats: dict[str, object] = {
        "anchor_source": source,
        "anchor_count": 0,
        "anchor_sigma_m": float(sigma_m),
        "anchor_every_n": int(every_n),
        "anchor_max_error_m": float(max_error_m),
        "anchor_error_median_m": float("nan"),
        "anchor_pass_rows_05m": 0,
    }
    if source in ("", "none"):
        return None, None, stats
    if float(sigma_m) <= 0.0:
        raise ValueError("--anchor-sigma-m must be >0 when anchors are enabled")

    pos_lookup: dict[float, np.ndarray]
    if source == "truth":
        pos_lookup = {round(float(t), 1): np.asarray(p, dtype=np.float64) for t, p in truth.items()}
    elif source == "pos":
        if anchor_pos_path is None:
            raise ValueError("--anchor-pos is required when --anchor-source=pos")
        pos_lookup, _status = _load_hybrid_pos_file(anchor_pos_path)
    else:
        raise ValueError(f"unknown --anchor-source: {source!r}")

    prior_positions = np.asarray(seed_positions, dtype=np.float64).copy()
    prior_sigmas = np.zeros(len(prior_positions), dtype=np.float64)
    anchor_errors: list[float] = []
    candidate_indices: list[int] = []
    for i, tow in enumerate(np.asarray(tows, dtype=np.float64)):
        key = round(float(tow), 1)
        anchor = pos_lookup.get(key)
        ref = truth.get(key)
        if anchor is None or ref is None:
            continue
        err = float(np.linalg.norm(np.asarray(anchor, dtype=np.float64) - ref))
        if np.isfinite(max_error_m) and max_error_m > 0.0 and err > float(max_error_m):
            continue
        candidate_indices.append(i)
        anchor_errors.append(err)

    if int(every_n) > 1:
        keep = set(candidate_indices[:: int(every_n)])
        filtered_errors: list[float] = []
        filtered_indices: list[int] = []
        for idx, err in zip(candidate_indices, anchor_errors, strict=True):
            if idx in keep:
                filtered_indices.append(idx)
                filtered_errors.append(err)
        candidate_indices = filtered_indices
        anchor_errors = filtered_errors

    for idx in candidate_indices:
        key = round(float(tows[idx]), 1)
        prior_positions[idx] = np.asarray(pos_lookup[key], dtype=np.float64)
        prior_sigmas[idx] = float(sigma_m)

    stats["anchor_count"] = int(len(candidate_indices))
    if anchor_errors:
        vals = np.asarray(anchor_errors, dtype=np.float64)
        stats["anchor_error_median_m"] = float(np.median(vals))
        stats["anchor_pass_rows_05m"] = int(np.count_nonzero(vals <= 0.5))
    return prior_positions, prior_sigmas, stats


def _build_multifamily_dd_epochs(
    *,
    run_dir: Path,
    data: dict,
    seed_positions: np.ndarray,
    systems: tuple[str, ...],
    families: tuple[str, ...],
    dd_min_pairs: int,
    dd_base_interp: bool,
    dd_min_elevation_deg: float,
    dd_min_snr: float,
    dd_keep_best: int,
) -> list[DDCarrierEpoch | None]:
    dd_computer = DDCarrierComputer(
        run_dir / "base.obs",
        rover_obs_path=run_dir / "rover.obs",
        base_position=np.asarray(data["base_ecef"], dtype=np.float64),
        allowed_systems=systems,
        interpolate_base_epochs=bool(dd_base_interp),
    )
    dd_epochs: list[DDCarrierEpoch | None] = []
    for i, tow in enumerate(np.asarray(data["times"], dtype=np.float64)):
        pos = np.asarray(seed_positions[i], dtype=np.float64)
        measurements = _build_dd_measurements(
            np.asarray(data["sat_ecef"][i], dtype=np.float64),
            np.asarray(data["system_ids"][i], dtype=np.int32),
            list(data["used_prns"][i]),
            np.asarray(data["weights"][i], dtype=np.float64),
            pos,
            systems,
            min_elevation_deg=float(dd_min_elevation_deg),
            min_snr=float(dd_min_snr),
            keep_best=int(dd_keep_best),
        )
        dd_epoch = None
        if len(measurements) >= int(dd_min_pairs):
            result = dd_computer.compute_dd_families(
                round(float(tow), 1),
                measurements,
                rover_position_approx=pos,
                min_common_sats=2,
                carrier_families=families,
            )
            if result is not None and int(getattr(result, "n_dd", 0)) > 0:
                dd_epoch = DDCarrierEpoch.from_result(result)
        dd_epochs.append(dd_epoch)
    return dd_epochs


def _build_dd_pseudorange_epochs(
    *,
    run_dir: Path,
    data: dict,
    seed_positions: np.ndarray,
    systems: tuple[str, ...],
    dd_min_pairs: int,
    dd_base_interp: bool,
    dd_min_elevation_deg: float,
    dd_min_snr: float,
    dd_keep_best: int,
) -> list[DDPseudorangeEpoch | None]:
    dd_pr_computer = DDPseudorangeComputer(
        run_dir / "base.obs",
        rover_obs_path=run_dir / "rover.obs",
        base_position=np.asarray(data["base_ecef"], dtype=np.float64),
        allowed_systems=systems,
        interpolate_base_epochs=bool(dd_base_interp),
    )
    dd_pr_epochs: list[DDPseudorangeEpoch | None] = []
    for i, tow in enumerate(np.asarray(data["times"], dtype=np.float64)):
        pos = np.asarray(seed_positions[i], dtype=np.float64)
        measurements = _build_dd_measurements(
            np.asarray(data["sat_ecef"][i], dtype=np.float64),
            np.asarray(data["system_ids"][i], dtype=np.int32),
            list(data["used_prns"][i]),
            np.asarray(data["weights"][i], dtype=np.float64),
            pos,
            systems,
            min_elevation_deg=float(dd_min_elevation_deg),
            min_snr=float(dd_min_snr),
            keep_best=int(dd_keep_best),
        )
        dd_epoch = None
        if len(measurements) >= int(dd_min_pairs):
            result = dd_pr_computer.compute_dd(
                round(float(tow), 1),
                measurements,
                rover_position_approx=pos,
                min_common_sats=int(dd_min_pairs),
                rover_weights=np.asarray(data["weights"][i], dtype=np.float64),
            )
            if result is not None and int(getattr(result, "n_dd", 0)) > 0:
                dd_epoch = DDPseudorangeEpoch.from_result(result)
        dd_pr_epochs.append(dd_epoch)
    return dd_pr_epochs


def _build_undiff_pseudorange_epochs(
    *,
    data: dict,
    seed_positions: np.ndarray,
    use_snr_weights: bool,
) -> list[UndiffPseudorangeEpoch | None]:
    epochs: list[UndiffPseudorangeEpoch | None] = []
    for i in range(len(seed_positions)):
        sat = np.asarray(data["sat_ecef"][i], dtype=np.float64).reshape(-1, 3)
        pr = np.asarray(data["pseudoranges"][i], dtype=np.float64).ravel()
        seed = np.asarray(seed_positions[i], dtype=np.float64).ravel()[:3]
        if sat.shape[0] == 0 or sat.shape[0] != pr.size:
            epochs.append(None)
            continue
        ranges = np.linalg.norm(sat - seed[None, :], axis=1)
        valid = np.isfinite(pr) & np.isfinite(ranges) & (pr > 1.0e6)
        if np.count_nonzero(valid) < 4:
            epochs.append(None)
            continue
        if use_snr_weights:
            raw_w = np.asarray(data["weights"][i], dtype=np.float64).ravel()
            if raw_w.size != pr.size:
                weights = np.ones(pr.size, dtype=np.float64)
            else:
                weights = np.maximum(raw_w, 1.0)
        else:
            weights = np.ones(pr.size, dtype=np.float64)
        clock_bias = float(np.average(pr[valid] - ranges[valid], weights=weights[valid]))
        epochs.append(
            UndiffPseudorangeEpoch(
                sat_ecef=sat[valid],
                pseudoranges_m=pr[valid],
                clock_bias_m=clock_bias,
                weights=weights[valid],
            )
        )
    return epochs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="nagoya/run2")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--seed-pos", type=Path, default=DEFAULT_SEED)
    parser.add_argument("--start-tow", type=float, default=557046.2)
    parser.add_argument("--end-tow", type=float, default=557051.6)
    parser.add_argument("--systems", default="G,E")
    parser.add_argument("--dd-families", default="L1_E1_B1,L5_E5A_B2A")
    parser.add_argument("--dd-min-pairs", type=int, default=3)
    parser.add_argument("--dd-base-interp", action="store_true")
    parser.add_argument("--dd-min-elevation-deg", type=float, default=-90.0)
    parser.add_argument("--dd-min-snr", type=float, default=0.0)
    parser.add_argument("--dd-keep-best", type=int, default=0)
    parser.add_argument("--dd-pr", action="store_true", help="Add DD pseudorange factors.")
    parser.add_argument("--dd-pr-sigma-m", type=float, default=5.0)
    parser.add_argument(
        "--undiff-pr",
        action="store_true",
        help="Add undifferenced pseudorange factors with seed-estimated per-epoch clock bias.",
    )
    parser.add_argument("--undiff-pr-sigma-m", type=float, default=10.0)
    parser.add_argument(
        "--undiff-pr-use-snr-weights",
        action="store_true",
        help="Use SNR weights for undifferenced PR factors; default uses unit weights.",
    )
    parser.add_argument("--lambda-ratio", type=float, default=3.0)
    parser.add_argument("--lambda-min-epochs", type=int, default=2)
    parser.add_argument("--lambda-max-epoch-gap", type=int, default=6)
    parser.add_argument("--lambda-slip-threshold-cycles", type=float, default=1.5)
    parser.add_argument("--prior-sigma-m", type=float, default=0.5)
    parser.add_argument(
        "--per-epoch-prior-sigma-m",
        type=float,
        default=0.0,
        help="If >0, apply this seed prior sigma at every epoch instead of endpoint-only priors.",
    )
    parser.add_argument(
        "--anchor-source",
        choices=("none", "truth", "pos"),
        default="none",
        help="Optional sparse absolute anchor prior source for oracle-bound checks.",
    )
    parser.add_argument("--anchor-pos", type=Path, default=None)
    parser.add_argument("--anchor-sigma-m", type=float, default=0.05)
    parser.add_argument(
        "--anchor-every-n",
        type=int,
        default=1,
        help="Use every Nth available anchor after filtering. 1 keeps all.",
    )
    parser.add_argument(
        "--anchor-max-error-m",
        type=float,
        default=0.0,
        help="For --anchor-source=pos, keep only anchors within this truth error; <=0 disables.",
    )
    parser.add_argument("--motion-sigma-m", type=float, default=0.25)
    parser.add_argument("--dd-sigma-cycles", type=float, default=0.20)
    parser.add_argument("--dd-fixed-sigma-cycles", type=float, default=0.05)
    parser.add_argument("--max-iterations", type=int, default=50)
    parser.add_argument(
        "--out-summary",
        type=Path,
        default=RESULTS / "nr2_segment_multifamily_fgo_6637_summary.csv",
    )
    parser.add_argument(
        "--out-pos",
        type=Path,
        default=RESULTS / "nr2_segment_multifamily_fgo_6637.pos",
    )
    parser.add_argument(
        "--out-fixed-only-pos",
        type=Path,
        default=RESULTS / "nr2_segment_multifamily_fgo_6637_fixed_only.pos",
    )
    args = parser.parse_args()

    city, run = str(args.run).split("/", 1)
    run_dir = args.data_root / city / run
    systems = _parse_systems(args.systems)
    families = _parse_systems(args.dd_families)

    data = load_ppc_window_geometry(
        run_dir,
        start_tow=float(args.start_tow),
        end_tow=float(args.end_tow),
        systems=systems,
    )
    tows = np.asarray(data["times"], dtype=np.float64)
    seed_positions = _load_seed_positions(args.seed_pos, tows)
    truth = _truth_by_tow(data)

    dd_epochs = _build_multifamily_dd_epochs(
        run_dir=run_dir,
        data=data,
        seed_positions=seed_positions,
        systems=systems,
        families=families,
        dd_min_pairs=int(args.dd_min_pairs),
        dd_base_interp=bool(args.dd_base_interp),
        dd_min_elevation_deg=float(args.dd_min_elevation_deg),
        dd_min_snr=float(args.dd_min_snr),
        dd_keep_best=int(args.dd_keep_best),
    )
    dd_pr_epochs = None
    if args.dd_pr:
        dd_pr_epochs = _build_dd_pseudorange_epochs(
            run_dir=run_dir,
            data=data,
            seed_positions=seed_positions,
            systems=systems,
            dd_min_pairs=int(args.dd_min_pairs),
            dd_base_interp=bool(args.dd_base_interp),
            dd_min_elevation_deg=float(args.dd_min_elevation_deg),
            dd_min_snr=float(args.dd_min_snr),
            dd_keep_best=int(args.dd_keep_best),
        )
    undiff_pr_epochs = None
    if args.undiff_pr:
        undiff_pr_epochs = _build_undiff_pseudorange_epochs(
            data=data,
            seed_positions=seed_positions,
            use_snr_weights=bool(args.undiff_pr_use_snr_weights),
        )

    motion_deltas = np.diff(seed_positions, axis=0)
    prior_sigmas = None
    prior_positions = seed_positions
    anchor_positions, anchor_sigmas, anchor_stats = _build_anchor_priors(
        tows=tows,
        seed_positions=seed_positions,
        truth=truth,
        source=str(args.anchor_source),
        anchor_pos_path=args.anchor_pos,
        sigma_m=float(args.anchor_sigma_m),
        every_n=max(1, int(args.anchor_every_n)),
        max_error_m=float(args.anchor_max_error_m),
    )
    if anchor_positions is not None and anchor_sigmas is not None:
        prior_positions = anchor_positions
        prior_sigmas = anchor_sigmas
    if float(args.per_epoch_prior_sigma_m) > 0.0:
        prior_sigmas = np.full(len(seed_positions), float(args.per_epoch_prior_sigma_m), dtype=np.float64)
    problem = LocalFgoProblem(
        initial_positions_ecef=seed_positions,
        window=LocalFgoWindow(0, len(seed_positions) - 1),
        motion_deltas_ecef=motion_deltas,
        dd_carrier=dd_epochs,
        dd_pseudorange=dd_pr_epochs,
        undiff_pseudorange=undiff_pr_epochs,
        prior_positions_ecef=prior_positions,
        prior_sigmas_m=prior_sigmas,
    )
    config = LocalFgoConfig(
        prior_sigma_m=float(args.prior_sigma_m),
        motion_sigma_m=float(args.motion_sigma_m),
        dd_sigma_cycles=float(args.dd_sigma_cycles),
        dd_pr_sigma_m=float(args.dd_pr_sigma_m),
        undiff_pr_sigma_m=float(args.undiff_pr_sigma_m),
        dd_fixed_sigma_cycles=float(args.dd_fixed_sigma_cycles),
        max_iterations=int(args.max_iterations),
    )
    lambda_config = LambdaFixConfig(
        ratio_threshold=float(args.lambda_ratio),
        min_epochs=int(args.lambda_min_epochs),
        max_epoch_gap=int(args.lambda_max_epoch_gap),
        slip_threshold_cycles=float(args.lambda_slip_threshold_cycles),
        fixed_sigma_cycles=float(args.dd_fixed_sigma_cycles),
    )
    result, lambda_summary = solve_local_fgo_with_lambda(problem, config, lambda_config)

    fgo_positions = np.asarray(result.positions_ecef, dtype=np.float64)
    fixed_only_positions = seed_positions.copy()
    fixed_epochs = {
        int(i)
        for i in lambda_summary.get("fixed_epochs", [])
        if 0 <= int(i) < len(fixed_only_positions)
    }
    for idx in fixed_epochs:
        fixed_only_positions[idx] = fgo_positions[idx]

    seed_stats = _error_stats(tows=tows, positions=seed_positions, truth=truth)
    fgo_stats = _error_stats(tows=tows, positions=fgo_positions, truth=truth)
    fixed_only_stats = _error_stats(tows=tows, positions=fixed_only_positions, truth=truth)
    pass_rows_seed, seed_ref_median = _score_pass_rows(
        ref_path=run_dir / "reference.csv",
        pos_by_tow={round(float(t), 1): p for t, p in zip(tows, seed_positions, strict=True)},
        tows=list(tows),
    )

    _write_pos_file(args.out_pos, tows, fgo_positions)
    _write_pos_file(args.out_fixed_only_pos, tows, fixed_only_positions)
    row: dict[str, object] = {
        "run": str(args.run),
        "start_tow": float(args.start_tow),
        "end_tow": float(args.end_tow),
        "n_epochs": int(len(tows)),
        "systems": ",".join(systems),
        "families": ",".join(families),
        "dd_epochs": int(sum(1 for d in dd_epochs if d is not None)),
        "dd_pairs_total": int(sum(int(d.n) for d in dd_epochs if d is not None)),
        "dd_pr_enabled": int(bool(args.dd_pr)),
        "dd_pr_epochs": int(sum(1 for d in dd_pr_epochs or [] if d is not None)),
        "dd_pr_pairs_total": int(sum(int(d.n) for d in dd_pr_epochs or [] if d is not None)),
        "dd_pr_sigma_m": float(args.dd_pr_sigma_m),
        "undiff_pr_enabled": int(bool(args.undiff_pr)),
        "undiff_pr_epochs": int(sum(1 for d in undiff_pr_epochs or [] if d is not None)),
        "undiff_pr_obs_total": int(sum(int(d.n) for d in undiff_pr_epochs or [] if d is not None)),
        "undiff_pr_sigma_m": float(args.undiff_pr_sigma_m),
        "undiff_pr_use_snr_weights": int(bool(args.undiff_pr_use_snr_weights)),
        **anchor_stats,
        "lambda_ratio": float(args.lambda_ratio),
        "lambda_min_epochs": int(args.lambda_min_epochs),
        "lambda_n_fixed": int(lambda_summary.get("n_fixed", 0)),
        "lambda_n_fixed_observations": int(lambda_summary.get("n_fixed_observations", 0)),
        "lambda_fixed_epoch_count": int(len(fixed_epochs)),
        "lambda_fixed_by_system": lambda_summary.get("fixed_by_system", {}),
        "lambda_iterations": lambda_summary.get("iterations", []),
        "factor_counts": result.factor_counts,
        "initial_error": float(result.initial_error),
        "final_error": float(result.final_error),
        "seed_pass_rows_05m": int(seed_stats["pass_rows_05m"]),
        "seed_median_error_m": float(seed_stats["median_error_m"]),
        "seed_p90_error_m": float(seed_stats["p90_error_m"]),
        "fgo_pass_rows_05m": int(fgo_stats["pass_rows_05m"]),
        "fgo_median_error_m": float(fgo_stats["median_error_m"]),
        "fgo_p90_error_m": float(fgo_stats["p90_error_m"]),
        "fixed_only_pass_rows_05m": int(fixed_only_stats["pass_rows_05m"]),
        "fixed_only_median_error_m": float(fixed_only_stats["median_error_m"]),
        "fixed_only_p90_error_m": float(fixed_only_stats["p90_error_m"]),
        "score_helper_seed_pass_rows": int(pass_rows_seed),
        "score_helper_seed_median_m": float(seed_ref_median),
        "out_pos": str(args.out_pos),
        "out_fixed_only_pos": str(args.out_fixed_only_pos),
    }
    _write_summary(args.out_summary, row)

    print(
        f"epochs={len(tows)} dd={row['dd_epochs']}/{len(tows)} "
        f"pairs={row['dd_pairs_total']} fixed={row['lambda_n_fixed']} "
        f"obs={row['lambda_n_fixed_observations']} fixed_epochs={len(fixed_epochs)}"
    )
    print(
        "median error seed/fgo/fixed-only = "
        f"{seed_stats['median_error_m']:.3f} / "
        f"{fgo_stats['median_error_m']:.3f} / "
        f"{fixed_only_stats['median_error_m']:.3f} m"
    )
    print(
        "pass rows seed/fgo/fixed-only = "
        f"{seed_stats['pass_rows_05m']} / "
        f"{fgo_stats['pass_rows_05m']} / "
        f"{fixed_only_stats['pass_rows_05m']}"
    )
    print(f"wrote {args.out_summary}")
    print(f"wrote {args.out_pos}")
    print(f"wrote {args.out_fixed_only_pos}")


if __name__ == "__main__":
    main()
