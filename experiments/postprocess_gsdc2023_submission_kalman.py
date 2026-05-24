"""Post-process a GSDC2023 submission CSV with 1D RTS Kalman smoothing.

Fifth and last layer in the trajectory post-process stack (Hampel + accel +
stop-snap + heading + Kalman).  Unlike the first four layers — which are
all "detect outlier → replace" — this one *smooths* the entire trajectory
under a constant-velocity motion model, attacking the residual sub-metre
noise that 1-Hz GNSS leaves on otherwise-clean motion.

Algorithm (per tripId, per axis: east / north metres)
-----------------------------------------------------

State ``x = [position, velocity]``.  CV (constant-velocity) motion model:
    x[t+1] = F x[t] + w,   F = [[1, dt], [0, 1]],   w ~ N(0, Q)
    z[t]   = H x[t] + v,   H = [1, 0],              v ~ N(0, R)

Process noise Q is derived from acceleration std ``sigma_a`` (m/s²) and
the per-step dt:
    Q = sigma_a² * [[dt⁴/4, dt³/2], [dt³/2, dt²]]

Measurement noise R = sigma_z² (m²).  ``sigma_z`` represents the post-stack
trajectory point uncertainty (after Hampel + accel + snap + heading have
already removed gross outliers).

Two passes:
  1. Forward Kalman filter.
  2. RTS (Rauch-Tung-Striebel) backward smoother to take the optimal
     state at each step given the full trajectory.

Defaults selected empirically on the train-trip A/B:
  ``sigma_a = 1.0`` m/s² (typical urban accel std),
  ``sigma_z = 2.0`` m (post-v7 measurement uncertainty).

This is intentionally a *light* smoother — large ``sigma_z`` would force
the trajectory to follow the model and discard real motion; small
``sigma_z`` would do nothing.  The sweet spot is the train-trip data driven.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def rts_smooth_1d(
    z: np.ndarray, dt: np.ndarray, *, sigma_a: float, sigma_z: float,
) -> np.ndarray:
    """Per-axis RTS smoother, returns smoothed positions of length ``len(z)``."""
    n = len(z)
    if n == 0:
        return z.copy()
    sigma_a2 = sigma_a * sigma_a
    R = sigma_z * sigma_z
    # State estimates and covariance buffers
    x_pred = np.zeros((n, 2), dtype=np.float64)  # predicted state before update
    P_pred = np.zeros((n, 2, 2), dtype=np.float64)
    x_filt = np.zeros((n, 2), dtype=np.float64)  # filtered (after update)
    P_filt = np.zeros((n, 2, 2), dtype=np.float64)
    # Initial state from first measurement, velocity = 0, large covariance
    x_filt[0] = np.array([z[0], 0.0])
    P_filt[0] = np.array([[R, 0.0], [0.0, 100.0]])
    x_pred[0] = x_filt[0]
    P_pred[0] = P_filt[0]
    H = np.array([1.0, 0.0])
    # Forward pass
    for t in range(1, n):
        dt_t = max(float(dt[t - 1]), 1e-3)
        F = np.array([[1.0, dt_t], [0.0, 1.0]])
        q11 = sigma_a2 * (dt_t ** 4) / 4.0
        q12 = sigma_a2 * (dt_t ** 3) / 2.0
        q22 = sigma_a2 * (dt_t ** 2)
        Q = np.array([[q11, q12], [q12, q22]])
        # Predict
        x_p = F @ x_filt[t - 1]
        P_p = F @ P_filt[t - 1] @ F.T + Q
        x_pred[t] = x_p
        P_pred[t] = P_p
        # Update
        y = z[t] - H @ x_p
        S = H @ P_p @ H + R
        K = (P_p @ H) / S
        x_filt[t] = x_p + K * y
        P_filt[t] = P_p - np.outer(K, H) @ P_p
    # Backward smoother
    x_smooth = np.zeros_like(x_filt)
    x_smooth[-1] = x_filt[-1]
    P_smooth = np.zeros_like(P_filt)
    P_smooth[-1] = P_filt[-1]
    for t in range(n - 2, -1, -1):
        dt_t = max(float(dt[t]), 1e-3)
        F = np.array([[1.0, dt_t], [0.0, 1.0]])
        # P_pred[t+1] already cached from forward pass
        try:
            P_pred_inv = np.linalg.inv(P_pred[t + 1])
        except np.linalg.LinAlgError:
            P_pred_inv = np.linalg.pinv(P_pred[t + 1])
        C = P_filt[t] @ F.T @ P_pred_inv
        x_smooth[t] = x_filt[t] + C @ (x_smooth[t + 1] - x_pred[t + 1])
        P_smooth[t] = P_filt[t] + C @ (P_smooth[t + 1] - P_pred[t + 1]) @ C.T
    return x_smooth[:, 0]


def apply_kalman_smoothing_to_submission(
    df: pd.DataFrame,
    *,
    sigma_a: float = 1.0,
    sigma_z: float = 2.0,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Return new DataFrame with per-tripId Kalman/RTS smoothed lat/lng."""
    out = df.copy()
    out = out.sort_values(["tripId", "UnixTimeMillis"]).reset_index(drop=True)
    stats: dict[str, object] = {
        "rows_total": len(out),
        "rows_changed": 0,
        "trips": 0,
        "sigma_a": sigma_a,
        "sigma_z": sigma_z,
    }
    original_lat = out["LatitudeDegrees"].to_numpy(dtype=np.float64).copy()
    original_lng = out["LongitudeDegrees"].to_numpy(dtype=np.float64).copy()
    for _, group in out.groupby("tripId", sort=False):
        idx = group.index.to_numpy()
        lat = group["LatitudeDegrees"].to_numpy(dtype=np.float64)
        lng = group["LongitudeDegrees"].to_numpy(dtype=np.float64)
        t_ms = group["UnixTimeMillis"].to_numpy(dtype=np.float64)
        n = len(lat)
        if n < 3:
            stats["trips"] = int(stats["trips"]) + 1  # type: ignore[arg-type]
            continue
        dt_s = np.diff(t_ms) / 1000.0
        # Local east/north metres
        lat0 = float(np.median(lat))
        lng0 = float(np.median(lng))
        mlat = 111_320.0
        mlng = 111_320.0 * np.cos(np.radians(lat0))
        east = (lng - lng0) * mlng
        north = (lat - lat0) * mlat
        east_s = rts_smooth_1d(east, dt_s, sigma_a=sigma_a, sigma_z=sigma_z)
        north_s = rts_smooth_1d(north, dt_s, sigma_a=sigma_a, sigma_z=sigma_z)
        lat_s = north_s / mlat + lat0
        lng_s = east_s / mlng + lng0
        out.loc[idx, "LatitudeDegrees"] = lat_s
        out.loc[idx, "LongitudeDegrees"] = lng_s
        stats["trips"] = int(stats["trips"]) + 1  # type: ignore[arg-type]
    final_lat = out["LatitudeDegrees"].to_numpy(dtype=np.float64)
    final_lng = out["LongitudeDegrees"].to_numpy(dtype=np.float64)
    stats["rows_changed"] = int(
        np.sum(
            (np.abs(final_lat - original_lat) > 1e-12) | (np.abs(final_lng - original_lng) > 1e-12)
        )
    )
    return out, stats


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Apply 1D RTS Kalman smoothing to a GSDC2023 submission CSV.",
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--sigma-a",
        type=float,
        default=1.0,
        help="process-acceleration std (m/s²); larger = trust measurements more",
    )
    parser.add_argument(
        "--sigma-z",
        type=float,
        default=2.0,
        help="measurement std (m); larger = smooth more aggressively",
    )
    args = parser.parse_args()
    if not args.input.is_file():
        print(f"[error] input not found: {args.input}", file=sys.stderr)
        return 1
    df = pd.read_csv(args.input)
    required = {"tripId", "UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"}
    missing = required - set(df.columns)
    if missing:
        print(f"[error] missing columns: {missing}", file=sys.stderr)
        return 2
    out, stats = apply_kalman_smoothing_to_submission(
        df, sigma_a=args.sigma_a, sigma_z=args.sigma_z,
    )
    out.to_csv(args.output, index=False)
    print(
        f"trips={stats['trips']} rows_total={stats['rows_total']} "
        f"rows_changed={stats['rows_changed']} "
        f"({100 * int(stats['rows_changed']) / max(1, int(stats['rows_total'])):.2f}%) "
        f"sigma_a={args.sigma_a} sigma_z={args.sigma_z}",
        flush=True,
    )
    print(f"wrote: {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
