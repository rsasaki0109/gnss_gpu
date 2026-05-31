#!/usr/bin/env python3
"""Audit per-epoch TDCP displacement estimates on GSDC2023 train trips.

This is a diagnostic for the next post-v8 lever: using carrier-phase TDCP as a
velocity/displacement constraint in the submission post-process smoother.  It
does not change runtime selection.  It estimates each inter-epoch ECEF
displacement from TDCP rows plus optional receiver clock nuisance terms, then
compares that estimate with the matching train ground-truth error-state delta.
When TDCP geometry correction is enabled in the bridge, the measurement has the
Kaggle WLS geometric change removed, so the estimated displacement is the
inter-epoch correction to the reference trajectory rather than the vehicle's
absolute displacement.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from experiments.eval_gsdc2023_ct_rbpf_fgo import discover_train_trips  # noqa: E402
from experiments.gsdc2023_raw_bridge import (  # noqa: E402
    DEFAULT_ROOT,
    DEFAULT_TDCP_CONSISTENCY_THRESHOLD_M,
    DEFAULT_TDCP_GEOMETRY_CORRECTION,
    DEFAULT_TDCP_WEIGHT_SCALE,
    build_trip_arrays,
)


_C_LIGHT = 299_792_458.0
_OMEGA_E = 7.2921151467e-5


def _fallback_data_root() -> Path:
    if DEFAULT_ROOT.is_dir():
        return DEFAULT_ROOT
    taroz = Path("/media/sasaki/aiueo/ai_coding_ws/ref/gsdc2023/taroz/dataset_2023")
    if taroz.is_dir():
        return taroz
    return DEFAULT_ROOT


def _clock_vector(clock_mode: str, n_clock: int, sys_kind: int) -> np.ndarray:
    if clock_mode == "none":
        return np.empty(0, dtype=np.float64)
    if clock_mode == "common":
        return np.ones(1, dtype=np.float64)
    out = np.zeros(int(n_clock), dtype=np.float64)
    out[0] = 1.0
    sk = int(sys_kind)
    if 0 < sk < int(n_clock):
        out[sk] = 1.0
    return out


def _los_sat_to_rx(rx_ecef: np.ndarray, sat_ecef: np.ndarray) -> np.ndarray | None:
    if not np.isfinite(rx_ecef).all() or not np.isfinite(sat_ecef).all():
        return None
    dx0 = rx_ecef - sat_ecef
    r0 = float(np.linalg.norm(dx0))
    if not np.isfinite(r0) or r0 <= 1.0:
        return None
    theta = _OMEGA_E * (r0 / _C_LIGHT)
    sx = float(sat_ecef[0]) * np.cos(theta) + float(sat_ecef[1]) * np.sin(theta)
    sy = -float(sat_ecef[0]) * np.sin(theta) + float(sat_ecef[1]) * np.cos(theta)
    rotated = np.array([sx, sy, float(sat_ecef[2])], dtype=np.float64)
    vec = rx_ecef - rotated
    norm = float(np.linalg.norm(vec))
    if not np.isfinite(norm) or norm <= 1.0:
        return None
    return vec / norm


def _weighted_lstsq(H: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scale = np.sqrt(np.maximum(np.asarray(w, dtype=np.float64), 0.0))
    Hw = H * scale[:, None]
    yw = y * scale
    sol, *_ = np.linalg.lstsq(Hw, yw, rcond=None)
    residual = y - H @ sol
    return sol, residual


def _robust_lstsq(
    H: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    *,
    huber_k: float,
    max_iter: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    obs_w = np.asarray(w, dtype=np.float64).copy()
    obs_w /= max(float(np.nanmedian(obs_w[obs_w > 0.0])) if np.any(obs_w > 0.0) else 1.0, 1e-12)
    sol = np.zeros(H.shape[1], dtype=np.float64)
    residual = y.copy()
    used = int(np.count_nonzero(obs_w > 0.0))
    for _ in range(max(1, int(max_iter))):
        sol, residual = _weighted_lstsq(H, y, obs_w)
        med = float(np.median(residual))
        mad = float(np.median(np.abs(residual - med)))
        sigma = max(1.4826 * mad, 0.03)
        cutoff = float(huber_k) * sigma
        factor = np.minimum(1.0, cutoff / np.maximum(np.abs(residual), 1e-12))
        new_w = np.asarray(w, dtype=np.float64).copy()
        new_w /= max(float(np.nanmedian(new_w[new_w > 0.0])) if np.any(new_w > 0.0) else 1.0, 1e-12)
        new_w *= factor
        used = int(np.count_nonzero(new_w > 0.05))
        if np.allclose(new_w, obs_w, rtol=0.0, atol=1e-5):
            obs_w = new_w
            break
        obs_w = new_w
    sol, residual = _weighted_lstsq(H, y, obs_w)
    return sol, residual, used


def _safe_percentile(values: list[float] | np.ndarray, q: float) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.percentile(arr, q))


def _estimate_interval(
    *,
    sat_ecef_next: np.ndarray,
    tdcp_meas: np.ndarray,
    tdcp_weights: np.ndarray,
    sys_kind_next: np.ndarray | None,
    n_clock: int,
    clock_mode: str,
    reference_ecef_next: np.ndarray,
    min_pairs: int,
    huber_k: float,
    max_iter: int,
) -> dict[str, object] | None:
    rows: list[np.ndarray] = []
    clock_keys: list[int] = []
    obs: list[float] = []
    weights: list[float] = []
    for sat_idx, weight in enumerate(tdcp_weights):
        if not np.isfinite(weight) or float(weight) <= 0.0:
            continue
        meas = float(tdcp_meas[sat_idx])
        if not np.isfinite(meas):
            continue
        los = _los_sat_to_rx(reference_ecef_next, sat_ecef_next[sat_idx])
        if los is None:
            continue
        sk = int(sys_kind_next[sat_idx]) if sys_kind_next is not None else 0
        if clock_mode == "signal":
            rows.append(los)
            clock_keys.append(sk)
        else:
            rows.append(np.concatenate([los, _clock_vector(clock_mode, n_clock, sk)]))
        obs.append(meas)
        weights.append(float(weight))

    if not rows:
        return None
    if clock_mode == "signal":
        active_keys = sorted({key for key in clock_keys if 0 <= key < int(n_clock)})
        if not active_keys:
            return None
        key_to_col = {key: idx for idx, key in enumerate(active_keys)}
        H = np.zeros((len(rows), 3 + len(active_keys)), dtype=np.float64)
        H[:, :3] = np.vstack(rows)
        for row_idx, key in enumerate(clock_keys):
            col = key_to_col.get(key)
            if col is not None:
                H[row_idx, 3 + col] = 1.0
    else:
        H = np.vstack(rows)
    n_param = H.shape[1]
    if len(obs) < max(int(min_pairs), n_param):
        return None
    y = np.asarray(obs, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    try:
        cond = float(np.linalg.cond(H))
        rank = int(np.linalg.matrix_rank(H))
        sol, residual, robust_used = _robust_lstsq(
            H,
            y,
            w,
            huber_k=huber_k,
            max_iter=max_iter,
        )
    except np.linalg.LinAlgError:
        return None
    dpos = sol[:3]
    rms = float(np.sqrt(np.mean(residual * residual))) if residual.size else float("nan")
    med = float(np.median(residual)) if residual.size else float("nan")
    mad = float(np.median(np.abs(residual - med))) if residual.size else float("nan")
    return {
        "dpos": dpos,
        "clock": sol[3:],
        "pair_count": len(obs),
        "robust_used_count": robust_used,
        "postfit_rms_m": rms,
        "postfit_robust_rms_m": 1.4826 * mad if np.isfinite(mad) else float("nan"),
        "condition_number": cond,
        "rank": rank,
        "parameter_count": n_param,
    }


def analyze_trip(trip: str, args: argparse.Namespace) -> tuple[list[dict[str, object]], dict[str, object]]:
    max_epochs = int(args.max_epochs) if int(args.max_epochs) > 0 else 1_000_000_000
    batch = build_trip_arrays(
        args.data_root / trip,
        max_epochs=max_epochs,
        start_epoch=args.start_epoch,
        constellation_type=1,
        signal_type="GPS_L1_CA",
        weight_mode="sin2el",
        multi_gnss=True,
        use_tdcp=True,
        tdcp_consistency_threshold_m=args.tdcp_consistency_threshold_m,
        tdcp_weight_scale=args.tdcp_weight_scale,
        tdcp_geometry_correction=args.tdcp_geometry_correction,
        data_root=args.data_root,
        trip=trip,
        dual_frequency=args.dual_frequency,
        raw_frame_epoch_window=args.raw_frame_epoch_window,
    )
    rows: list[dict[str, object]] = []
    n_interval = max(0, min(batch.sat_ecef.shape[0] - 1, batch.truth.shape[0] - 1))
    if batch.tdcp_meas is None or batch.tdcp_weights is None:
        return rows, {
            "trip": trip,
            "phone": Path(trip).name,
            "clock_mode": args.clock_mode,
            "reference": args.reference,
            "dual_frequency": bool(args.dual_frequency),
            "tdcp_geometry_correction": bool(args.tdcp_geometry_correction),
            "intervals": n_interval,
            "estimated_intervals": 0,
            "coverage": 0.0,
            "median_dpos_error_m": float("nan"),
            "p95_dpos_error_m": float("nan"),
            "moving_threshold_m": float(args.moving_threshold_m),
            "moving_estimated_intervals": 0,
            "median_moving_dpos_error_m": float("nan"),
            "p95_moving_dpos_error_m": float("nan"),
            "median_abs_norm_error_m": float("nan"),
            "p95_abs_norm_error_m": float("nan"),
            "mean_pair_count": 0.0,
            "median_condition_number": float("nan"),
            "p95_condition_number": float("nan"),
            "median_rank": float("nan"),
            "median_postfit_rms_m": float("nan"),
            "tdcp_consistency_mask_count": int(batch.tdcp_consistency_mask_count),
            "tdcp_geometry_correction_count": int(batch.tdcp_geometry_correction_count),
        }

    reference = batch.truth if args.reference == "truth" else batch.kaggle_wls
    n_interval = min(batch.tdcp_meas.shape[0], n_interval)
    for t in range(n_interval):
        est = _estimate_interval(
            sat_ecef_next=batch.sat_ecef[t + 1],
            tdcp_meas=batch.tdcp_meas[t],
            tdcp_weights=batch.tdcp_weights[t],
            sys_kind_next=(batch.sys_kind[t + 1] if batch.sys_kind is not None else None),
            n_clock=batch.n_clock,
            clock_mode=args.clock_mode,
            reference_ecef_next=reference[t + 1],
            min_pairs=args.min_pairs,
            huber_k=args.huber_k,
            max_iter=args.huber_iters,
        )
        if est is None:
            continue
        gt_delta = batch.truth[t + 1] - batch.truth[t]
        reference_delta = (
            batch.kaggle_wls[t + 1] - batch.kaggle_wls[t]
            if args.tdcp_geometry_correction
            else np.zeros(3, dtype=np.float64)
        )
        target_delta = gt_delta - reference_delta
        dpos = np.asarray(est["dpos"], dtype=np.float64)
        dpos_err = dpos - target_delta
        gt_norm = float(np.linalg.norm(gt_delta))
        reference_norm = float(np.linalg.norm(reference_delta))
        target_norm = float(np.linalg.norm(target_delta))
        est_norm = float(np.linalg.norm(dpos))
        rows.append(
            {
                "trip": trip,
                "phone": Path(trip).name,
                "clock_mode": args.clock_mode,
                "interval_idx": t,
                "tow_ms0": float(batch.times_ms[t]),
                "tow_ms1": float(batch.times_ms[t + 1]),
                "dt_s": float((batch.times_ms[t + 1] - batch.times_ms[t]) / 1000.0),
                "pair_count": int(est["pair_count"]),
                "robust_used_count": int(est["robust_used_count"]),
                "postfit_rms_m": float(est["postfit_rms_m"]),
                "postfit_robust_rms_m": float(est["postfit_robust_rms_m"]),
                "condition_number": float(est["condition_number"]),
                "rank": int(est["rank"]),
                "parameter_count": int(est["parameter_count"]),
                "tdcp_delta_norm_m": est_norm,
                "gt_delta_norm_m": gt_norm,
                "reference_delta_norm_m": reference_norm,
                "target_delta_norm_m": target_norm,
                "delta_norm_error_m": est_norm - target_norm,
                "dpos_error_m": float(np.linalg.norm(dpos_err)),
                "dpos_x_m": float(dpos[0]),
                "dpos_y_m": float(dpos[1]),
                "dpos_z_m": float(dpos[2]),
                "target_dx_m": float(target_delta[0]),
                "target_dy_m": float(target_delta[1]),
                "target_dz_m": float(target_delta[2]),
                "gt_dx_m": float(gt_delta[0]),
                "gt_dy_m": float(gt_delta[1]),
                "gt_dz_m": float(gt_delta[2]),
            },
        )

    err = [float(row["dpos_error_m"]) for row in rows]
    norm_err = [abs(float(row["delta_norm_error_m"])) for row in rows]
    pair_counts = [float(row["pair_count"]) for row in rows]
    moving_rows = [row for row in rows if float(row["gt_delta_norm_m"]) >= float(args.moving_threshold_m)]
    moving_err = [float(row["dpos_error_m"]) for row in moving_rows]
    conds = [float(row["condition_number"]) for row in rows]
    ranks = [float(row["rank"]) for row in rows]
    postfit = [float(row["postfit_rms_m"]) for row in rows]
    summary = {
        "trip": trip,
        "phone": Path(trip).name,
        "clock_mode": args.clock_mode,
        "reference": args.reference,
        "dual_frequency": bool(args.dual_frequency),
        "tdcp_geometry_correction": bool(args.tdcp_geometry_correction),
        "intervals": n_interval,
        "estimated_intervals": len(rows),
        "coverage": len(rows) / max(n_interval, 1),
        "median_dpos_error_m": _safe_percentile(err, 50),
        "p95_dpos_error_m": _safe_percentile(err, 95),
        "moving_threshold_m": float(args.moving_threshold_m),
        "moving_estimated_intervals": len(moving_rows),
        "median_moving_dpos_error_m": _safe_percentile(moving_err, 50),
        "p95_moving_dpos_error_m": _safe_percentile(moving_err, 95),
        "median_abs_norm_error_m": _safe_percentile(norm_err, 50),
        "p95_abs_norm_error_m": _safe_percentile(norm_err, 95),
        "mean_pair_count": float(np.mean(pair_counts)) if pair_counts else 0.0,
        "median_condition_number": _safe_percentile(conds, 50),
        "p95_condition_number": _safe_percentile(conds, 95),
        "median_rank": _safe_percentile(ranks, 50),
        "median_postfit_rms_m": _safe_percentile(postfit, 50),
        "tdcp_consistency_mask_count": int(batch.tdcp_consistency_mask_count),
        "tdcp_geometry_correction_count": int(batch.tdcp_geometry_correction_count),
    }
    return rows, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=_fallback_data_root())
    parser.add_argument("--trip", action="append", default=[])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--start-epoch", type=int, default=0)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--min-pairs", type=int, default=6)
    parser.add_argument("--huber-k", type=float, default=2.5)
    parser.add_argument("--huber-iters", type=int, default=5)
    parser.add_argument("--reference", choices=("kaggle_wls", "truth"), default="kaggle_wls")
    parser.add_argument("--clock-mode", choices=("full", "signal", "common", "none"), default="full")
    parser.add_argument("--moving-threshold-m", type=float, default=0.5)
    parser.add_argument("--dual-frequency", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--raw-frame-epoch-window", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tdcp-consistency-threshold-m", type=float, default=DEFAULT_TDCP_CONSISTENCY_THRESHOLD_M)
    parser.add_argument("--tdcp-weight-scale", type=float, default=DEFAULT_TDCP_WEIGHT_SCALE)
    parser.add_argument(
        "--tdcp-geometry-correction",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_TDCP_GEOMETRY_CORRECTION,
    )
    parser.add_argument(
        "--out-prefix",
        type=Path,
        default=Path("experiments/results/gsdc2023_tdcp_displacement_audit"),
    )
    args = parser.parse_args()

    trips = discover_train_trips(args.data_root)
    if args.trip:
        wanted = set(args.trip)
        trips = [trip for trip in trips if trip in wanted]
    if args.limit > 0:
        trips = trips[: args.limit]
    if not trips:
        raise RuntimeError("no train trips found")

    all_rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for idx, trip in enumerate(trips, start=1):
        rows, summary = analyze_trip(trip, args)
        all_rows.extend(rows)
        summaries.append(summary)
        print(
            f"[{idx}/{len(trips)}] {trip} coverage={summary['coverage']:.3f} "
            f"median_err={summary['median_dpos_error_m']:.3f}m "
            f"moving_median={summary['median_moving_dpos_error_m']:.3f}m "
            f"p95={summary['p95_dpos_error_m']:.3f}m "
            f"cond50={summary['median_condition_number']:.2e}",
            flush=True,
        )

    out_prefix = args.out_prefix
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    epochs_path = out_prefix.with_name(out_prefix.name + "_epochs.csv")
    summary_path = out_prefix.with_name(out_prefix.name + "_summary.csv")
    if all_rows:
        with epochs_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(all_rows[0]))
            writer.writeheader()
            writer.writerows(all_rows)
    else:
        epochs_path.write_text("", encoding="utf-8")
    pd.DataFrame(summaries).to_csv(summary_path, index=False)
    print(f"wrote: {epochs_path}")
    print(f"wrote: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
