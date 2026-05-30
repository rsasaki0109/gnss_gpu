#!/usr/bin/env python3
"""Evaluate sparse fixed-ambiguity DD-carrier position anchors on GSDC2023."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from experiments.audit_gsdc2023_dd_carrier_support import (  # noqa: E402
    _computer_for_trip,
    rover_measurements_for_epoch,
    snap_tow_to_base_epoch,
)
from experiments.evaluate import compute_metrics  # noqa: E402
from experiments.eval_gsdc2023_ct_rbpf_fgo import discover_train_trips, score_delta  # noqa: E402
from experiments.gsdc2023_base_correction import GPS_WEEK_SECONDS, unix_ms_to_gps_abs_seconds  # noqa: E402
from experiments.gsdc2023_output import score_from_metrics  # noqa: E402
from experiments.gsdc2023_raw_bridge import (  # noqa: E402
    BridgeConfig,
    DEFAULT_MOTION_SIGMA_M,
    DEFAULT_ROOT,
    FACTOR_DT_MAX_S,
    GATED_BASELINE_THRESHOLD_DEFAULT,
    build_trip_arrays,
    validate_raw_gsdc2023_trip,
)
from gnss_gpu.dd_carrier import DDCarrierComputer, DDResult  # noqa: E402


DEFAULT_OUTPUT = Path("experiments/results/gsdc2023_dd_carrier_anchor_eval_20260519.csv")


@dataclass(frozen=True)
class DDCarrierAnchorConfig:
    min_dd_pairs: int = 4
    sigma_cycles: float = 0.12
    prior_sigma_m: float = 1.5
    max_shift_m: float = 3.0
    max_initial_rms_m: float = 0.40
    max_final_rms_m: float = 0.25
    max_iter: int = 5
    tol_m: float = 1.0e-4


def smooth_anchor_corrections(
    n_epoch: int,
    anchor_indices: np.ndarray,
    anchor_deltas: np.ndarray,
    *,
    anchor_sigma_m: float,
    smooth_sigma_m: float,
    zero_sigma_m: float,
) -> np.ndarray:
    n = int(n_epoch)
    if n <= 0:
        return np.zeros((0, 3), dtype=np.float64)
    idx = np.asarray(anchor_indices, dtype=np.int64).reshape(-1)
    deltas = np.asarray(anchor_deltas, dtype=np.float64).reshape(-1, 3)
    valid = (idx >= 0) & (idx < n) & np.isfinite(deltas).all(axis=1)
    idx = idx[valid]
    deltas = deltas[valid]
    if idx.size == 0:
        return np.zeros((n, 3), dtype=np.float64)

    h = np.zeros((n, n), dtype=np.float64)
    rhs = np.zeros((n, 3), dtype=np.float64)
    if zero_sigma_m > 0.0:
        h += np.eye(n, dtype=np.float64) / (float(zero_sigma_m) ** 2)
    anchor_w = 1.0 / (max(float(anchor_sigma_m), 1.0e-6) ** 2)
    for epoch_idx, delta in zip(idx, deltas):
        h[int(epoch_idx), int(epoch_idx)] += anchor_w
        rhs[int(epoch_idx)] += anchor_w * delta
    if n > 1 and smooth_sigma_m > 0.0:
        smooth_w = 1.0 / (float(smooth_sigma_m) ** 2)
        for epoch_idx in range(n - 1):
            h[epoch_idx, epoch_idx] += smooth_w
            h[epoch_idx + 1, epoch_idx + 1] += smooth_w
            h[epoch_idx, epoch_idx + 1] -= smooth_w
            h[epoch_idx + 1, epoch_idx] -= smooth_w
    try:
        return np.linalg.solve(h, rhs)
    except np.linalg.LinAlgError:
        solution, *_ = np.linalg.lstsq(h, rhs, rcond=None)
        return solution


def _dd_expected_m(position_ecef: np.ndarray, dd: DDResult) -> tuple[np.ndarray, np.ndarray]:
    pos = np.asarray(position_ecef, dtype=np.float64).reshape(3)
    sat_k = np.asarray(dd.sat_ecef_k, dtype=np.float64).reshape(-1, 3)
    sat_ref = np.asarray(dd.sat_ecef_ref, dtype=np.float64).reshape(-1, 3)
    rho_k_vec = pos[None, :] - sat_k
    rho_ref_vec = pos[None, :] - sat_ref
    rho_k = np.linalg.norm(rho_k_vec, axis=1)
    rho_ref = np.linalg.norm(rho_ref_vec, axis=1)
    expected = rho_k - rho_ref - np.asarray(dd.base_range_k) + np.asarray(dd.base_range_ref)
    jac = rho_k_vec / np.maximum(rho_k[:, None], 1.0) - rho_ref_vec / np.maximum(rho_ref[:, None], 1.0)
    return expected, jac


def dd_carrier_fixed_ambiguity_update(
    seed_ecef: np.ndarray,
    dd: DDResult,
    config: DDCarrierAnchorConfig,
) -> tuple[np.ndarray, dict[str, float | bool | int]]:
    seed = np.asarray(seed_ecef, dtype=np.float64).reshape(3)
    if int(dd.n_dd) < int(config.min_dd_pairs):
        return seed, {"accepted": False, "reason": "few_pairs", "dd_pairs": int(dd.n_dd)}

    wavelengths = np.asarray(dd.wavelengths_m, dtype=np.float64).reshape(-1)
    expected_seed, _ = _dd_expected_m(seed, dd)
    expected_seed_cycles = expected_seed / wavelengths
    ambiguities = np.round(np.asarray(dd.dd_carrier_cycles, dtype=np.float64) - expected_seed_cycles)
    obs_m = (np.asarray(dd.dd_carrier_cycles, dtype=np.float64) - ambiguities) * wavelengths
    pos = seed.copy()

    def residual_at(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        expected, jac = _dd_expected_m(x, dd)
        return obs_m - expected, jac

    residual, jac = residual_at(pos)
    initial_rms = float(np.sqrt(np.mean(residual * residual))) if residual.size else float("inf")
    if not np.isfinite(initial_rms) or initial_rms > float(config.max_initial_rms_m):
        return seed, {
            "accepted": False,
            "reason": "initial_rms",
            "dd_pairs": int(dd.n_dd),
            "initial_rms_m": initial_rms,
        }

    sigma_m = np.maximum(wavelengths * float(config.sigma_cycles), 1.0e-3)
    sqrt_w = np.sqrt(np.asarray(dd.dd_weights, dtype=np.float64).reshape(-1)) / sigma_m
    prior_sqrt_w = 1.0 / max(float(config.prior_sigma_m), 1.0e-6)
    iters = 0
    for iters in range(1, int(config.max_iter) + 1):
        residual, jac = residual_at(pos)
        a = jac * sqrt_w[:, None]
        b = residual * sqrt_w
        a = np.vstack([a, np.eye(3) * prior_sqrt_w])
        b = np.concatenate([b, (seed - pos) * prior_sqrt_w])
        try:
            delta, *_ = np.linalg.lstsq(a, b, rcond=None)
        except np.linalg.LinAlgError:
            return seed, {"accepted": False, "reason": "singular", "dd_pairs": int(dd.n_dd)}
        if not np.isfinite(delta).all():
            return seed, {"accepted": False, "reason": "nonfinite", "dd_pairs": int(dd.n_dd)}
        pos = pos + delta
        if float(np.linalg.norm(delta)) < float(config.tol_m):
            break

    final_residual, _ = residual_at(pos)
    final_rms = float(np.sqrt(np.mean(final_residual * final_residual))) if final_residual.size else float("inf")
    shift_m = float(np.linalg.norm(pos - seed))
    accepted = (
        np.isfinite(final_rms)
        and final_rms <= float(config.max_final_rms_m)
        and shift_m <= float(config.max_shift_m)
        and final_rms <= initial_rms
    )
    return (
        pos if accepted else seed,
        {
            "accepted": bool(accepted),
            "reason": "accepted" if accepted else "gate",
            "dd_pairs": int(dd.n_dd),
            "initial_rms_m": initial_rms,
            "final_rms_m": final_rms,
            "shift_m": shift_m,
            "iters": int(iters),
        },
    )


def apply_sparse_dd_carrier_anchors(
    data_root: Path,
    trip: str,
    seed_state: np.ndarray,
    args: argparse.Namespace,
    anchor_config: DDCarrierAnchorConfig,
    cache: dict[Path, DDCarrierComputer],
) -> tuple[np.ndarray, dict[str, float | int]]:
    batch = build_trip_arrays(
        data_root / trip,
        max_epochs=args.max_epochs,
        start_epoch=args.start_epoch,
        constellation_type=1,
        signal_type="GPS_L1_CA",
        weight_mode="sin2el",
        multi_gnss=True,
        use_tdcp=True,
        data_root=data_root,
        trip=trip,
        dual_frequency=True,
    )
    computer = _computer_for_trip(
        data_root,
        trip,
        cache,
        base_obs_template=getattr(args, "base_obs_template", None),
        require_base_obs_template=bool(getattr(args, "require_base_obs_template", False)),
    )
    tows = np.mod(unix_ms_to_gps_abs_seconds(batch.times_ms), GPS_WEEK_SECONDS)
    anchored = np.asarray(seed_state, dtype=np.float64).copy()
    accepted = 0
    dd_epochs = 0
    snapped = 0
    pair_counts: list[int] = []
    shifts: list[float] = []
    initial_rms: list[float] = []
    final_rms: list[float] = []
    anchor_indices: list[int] = []
    anchor_deltas: list[np.ndarray] = []
    for epoch_idx, tow in enumerate(tows):
        tow_for_dd = snap_tow_to_base_epoch(computer, float(tow), args.tow_snap_tolerance_s)
        if tow_for_dd is None:
            continue
        snapped += 1
        measurements = rover_measurements_for_epoch(batch, epoch_idx)
        dd = computer.compute_dd(
            tow_for_dd,
            measurements,
            rover_position_approx=anchored[epoch_idx, :3],
            min_common_sats=anchor_config.min_dd_pairs,
        )
        if dd is None:
            continue
        dd_epochs += 1
        pair_counts.append(int(dd.n_dd))
        pos, stats = dd_carrier_fixed_ambiguity_update(anchored[epoch_idx, :3], dd, anchor_config)
        if bool(stats.get("accepted", False)):
            accepted += 1
            delta = pos - anchored[epoch_idx, :3]
            anchor_indices.append(int(epoch_idx))
            anchor_deltas.append(delta)
            if not args.smooth_corrections:
                anchored[epoch_idx, :3] = pos
            shifts.append(float(stats.get("shift_m", np.nan)))
            initial_rms.append(float(stats.get("initial_rms_m", np.nan)))
            final_rms.append(float(stats.get("final_rms_m", np.nan)))

    correction_mean_norm_m = 0.0
    correction_p95_norm_m = 0.0
    if args.smooth_corrections and anchor_indices:
        corrections = smooth_anchor_corrections(
            anchored.shape[0],
            np.asarray(anchor_indices, dtype=np.int64),
            np.asarray(anchor_deltas, dtype=np.float64),
            anchor_sigma_m=args.anchor_correction_sigma_m,
            smooth_sigma_m=args.correction_smooth_sigma_m,
            zero_sigma_m=args.correction_zero_sigma_m,
        )
        correction_norm = np.linalg.norm(corrections, axis=1)
        correction_mean_norm_m = float(np.mean(correction_norm))
        correction_p95_norm_m = float(np.percentile(correction_norm, 95))
        anchored[:, :3] = anchored[:, :3] + corrections

    return anchored, {
        "base_snapped_epochs": int(snapped),
        "dd_epochs": int(dd_epochs),
        "accepted_anchor_epochs": int(accepted),
        "smooth_corrections": bool(args.smooth_corrections),
        "dd_pairs_mean": float(np.mean(pair_counts)) if pair_counts else 0.0,
        "accepted_shift_mean_m": float(np.nanmean(shifts)) if shifts else 0.0,
        "accepted_initial_rms_mean_m": float(np.nanmean(initial_rms)) if initial_rms else 0.0,
        "accepted_final_rms_mean_m": float(np.nanmean(final_rms)) if final_rms else 0.0,
        "correction_mean_norm_m": correction_mean_norm_m,
        "correction_p95_norm_m": correction_p95_norm_m,
    }


def build_config(args: argparse.Namespace) -> BridgeConfig:
    return BridgeConfig(
        motion_sigma_m=args.motion_sigma_m,
        factor_dt_max_s=args.factor_dt_max_s,
        fgo_iters=args.fgo_iters,
        position_source=args.position_source,
        chunk_epochs=args.chunk_epochs,
        gated_baseline_threshold=args.gated_threshold,
        use_vd=args.vd,
        multi_gnss=args.multi_gnss,
        tdcp_enabled=args.tdcp,
    )


def run_eval(args: argparse.Namespace) -> pd.DataFrame:
    trips = discover_train_trips(args.data_root)
    if args.trip:
        wanted = set(args.trip)
        trips = [trip for trip in trips if trip in wanted]
    if args.limit > 0:
        trips = trips[: args.limit]
    if not trips:
        raise RuntimeError("no train trips found")

    anchor_config = DDCarrierAnchorConfig(
        min_dd_pairs=args.min_dd_pairs,
        sigma_cycles=args.sigma_cycles,
        prior_sigma_m=args.prior_sigma_m,
        max_shift_m=args.max_shift_m,
        max_initial_rms_m=args.max_initial_rms_m,
        max_final_rms_m=args.max_final_rms_m,
    )
    bridge_config = build_config(args)
    cache: dict[Path, DDCarrierComputer] = {}
    rows: list[dict[str, object]] = []
    for idx, trip in enumerate(trips, start=1):
        started = time.time()
        result = validate_raw_gsdc2023_trip(
            args.data_root,
            trip,
            max_epochs=args.max_epochs,
            start_epoch=args.start_epoch,
            config=bridge_config,
        )
        anchored_state, anchor_stats = apply_sparse_dd_carrier_anchors(
            args.data_root,
            trip,
            result.selected_state,
            args,
            anchor_config,
            cache,
        )
        anchor_metrics = compute_metrics(anchored_state[:, :3], result.truth)
        base_payload = result.metrics_payload()
        selected_metrics = base_payload.get("selected_metrics")
        selected_metrics = selected_metrics if isinstance(selected_metrics, dict) else {}
        row: dict[str, object] = {
            "trip": trip,
            "n_epochs": result.n_epochs,
            "base_selected_score_m": base_payload.get("selected_score_m"),
            "anchor_score_m": score_from_metrics(anchor_metrics),
            "base_selected_p50_m": selected_metrics.get("p50_m"),
            "anchor_p50_m": anchor_metrics["p50"],
            "base_selected_p95_m": selected_metrics.get("p95_m"),
            "anchor_p95_m": anchor_metrics["p95"],
            "elapsed_s": time.time() - started,
        }
        row.update(anchor_stats)
        row["delta_score_m_vs_base"] = score_delta(row["anchor_score_m"], row["base_selected_score_m"])
        rows.append(row)
        print(
            f"[{idx}/{len(trips)}] {trip} anchors={anchor_stats['accepted_anchor_epochs']} "
            f"delta={row['delta_score_m_vs_base']} done in {row['elapsed_s']:.1f}s",
            flush=True,
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--trip", action="append", default=[], help="train/.../phone trip; repeatable")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--start-epoch", type=int, default=0)
    parser.add_argument("--motion-sigma-m", type=float, default=DEFAULT_MOTION_SIGMA_M)
    parser.add_argument("--factor-dt-max-s", type=float, default=FACTOR_DT_MAX_S)
    parser.add_argument("--fgo-iters", type=int, default=8)
    parser.add_argument("--position-source", choices=("auto", "gated"), default="gated")
    parser.add_argument("--chunk-epochs", type=int, default=200)
    parser.add_argument("--gated-threshold", type=float, default=GATED_BASELINE_THRESHOLD_DEFAULT)
    parser.add_argument("--vd", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--multi-gnss", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tdcp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tow-snap-tolerance-s", type=float, default=0.6)
    parser.add_argument(
        "--base-obs-template",
        default=None,
        help="optional course-relative template such as '{base}_1hz.obs'; falls back to standard base obs unless required",
    )
    parser.add_argument("--require-base-obs-template", action="store_true")
    parser.add_argument("--min-dd-pairs", type=int, default=4)
    parser.add_argument("--sigma-cycles", type=float, default=0.12)
    parser.add_argument("--prior-sigma-m", type=float, default=1.5)
    parser.add_argument("--max-shift-m", type=float, default=3.0)
    parser.add_argument("--max-initial-rms-m", type=float, default=0.40)
    parser.add_argument("--max-final-rms-m", type=float, default=0.25)
    parser.add_argument("--smooth-corrections", action="store_true")
    parser.add_argument("--anchor-correction-sigma-m", type=float, default=0.5)
    parser.add_argument("--correction-smooth-sigma-m", type=float, default=0.25)
    parser.add_argument("--correction-zero-sigma-m", type=float, default=5.0)
    args = parser.parse_args()

    frame = run_eval(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    delta = pd.to_numeric(frame["delta_score_m_vs_base"], errors="coerce")
    accepted = pd.to_numeric(frame["accepted_anchor_epochs"], errors="coerce").fillna(0)
    print(f"wrote: {args.output}")
    print(f"score wins: {int((delta < 0).sum())}/{int(delta.notna().sum())}")
    print(f"accepted anchor epochs: {int(accepted.sum())}")
    print(f"mean score delta: {float(delta.mean()) if delta.notna().any() else float('nan'):.4f}m")


if __name__ == "__main__":
    main()
