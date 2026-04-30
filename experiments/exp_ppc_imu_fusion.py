#!/usr/bin/env python3
# ruff: noqa: E402
"""IMU-aided fill for libgnss++ RTK output on PPC full runs.

Three fill strategies are provided:

- ``hold``: at each rover TOW with no RTK solution, output the most
  recent RTK position (no motion). Cheap baseline.
- ``cv``: propagate the last RTK position using a velocity estimate
  derived from the two most recent RTK solutions.
- ``imu``: integrate the rover's IMU (accel + gyro) between RTK fixes
  using a simple ECEF-frame strapdown INS. The RTK fix resets pos,
  re-estimates velocity from adjacent RTK fixes when available, and
  the IMU fills the gap.

All strategies are scored against the FULL reference trajectory
(denominator = full rover-epoch arc length), directly comparable to
TURING's 85.6% PPC2024.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import sys

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "python"))

from gnss_gpu.ppc_score import score_ppc2024

_DEFAULT_DATA_ROOT = Path("/media/sasaki/aiueo/ai_coding_ws/datasets/PPC-Dataset-data")
_DEFAULT_RTK_DIR = _SCRIPT_DIR / "results" / "libgnss_rtk_pos"

_RUNS = (
    ("tokyo", "run1"),
    ("tokyo", "run2"),
    ("tokyo", "run3"),
    ("nagoya", "run1"),
    ("nagoya", "run2"),
    ("nagoya", "run3"),
)

_WGS84_A = 6_378_137.0
_WGS84_E2 = 6.694379990141316e-3
_WGS84_GM = 3.986004418e14
_WGS84_OMEGA = 7.2921151467e-5


def _load_reference(path: Path) -> list[tuple[float, np.ndarray]]:
    out: list[tuple[float, np.ndarray]] = []
    with path.open() as fh:
        reader = csv.reader(fh)
        next(reader)
        for row in reader:
            tow = round(float(row[0]), 2)
            out.append(
                (tow, np.array([float(row[5]), float(row[6]), float(row[7])], dtype=np.float64))
            )
    return out


def _parse_rtk_pos(path: Path) -> dict[float, tuple[np.ndarray, int]]:
    out: dict[float, tuple[np.ndarray, int]] = {}
    with path.open() as fh:
        for line in fh:
            if line.startswith("%") or not line.strip():
                continue
            parts = line.split()
            if len(parts) < 10:
                continue
            tow = round(float(parts[1]), 2)
            ecef = np.array(
                [float(parts[2]), float(parts[3]), float(parts[4])],
                dtype=np.float64,
            )
            status = int(parts[8])
            out[tow] = (ecef, status)
    return out


def _load_imu(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tows: list[float] = []
    acc: list[list[float]] = []
    gyro: list[list[float]] = []
    deg2rad = math.pi / 180.0
    with path.open() as fh:
        reader = csv.reader(fh)
        next(reader)
        for row in reader:
            tows.append(float(row[0]))
            acc.append([float(row[2]), float(row[3]), float(row[4])])
            gyro.append(
                [
                    float(row[5]) * deg2rad,
                    float(row[6]) * deg2rad,
                    float(row[7]) * deg2rad,
                ]
            )
    return (
        np.asarray(tows, dtype=np.float64),
        np.asarray(acc, dtype=np.float64),
        np.asarray(gyro, dtype=np.float64),
    )


def _ecef_to_enu_R(ecef: np.ndarray) -> np.ndarray:
    x, y, z = ecef
    lon = math.atan2(y, x)
    p = math.hypot(x, y)
    lat = math.atan2(z, p * (1.0 - _WGS84_E2))
    for _ in range(6):
        sl = math.sin(lat)
        n = _WGS84_A / math.sqrt(1.0 - _WGS84_E2 * sl * sl)
        lat = math.atan2(z + _WGS84_E2 * n * sl, p)
    sl, cl = math.sin(lat), math.cos(lat)
    so, co = math.sin(lon), math.cos(lon)
    return np.array(
        [
            [-so, co, 0.0],
            [-sl * co, -sl * so, cl],
            [cl * co, cl * so, sl],
        ],
        dtype=np.float64,
    )


def _gravity_ecef(pos: np.ndarray) -> np.ndarray:
    """Return gravity in ECEF at position ``pos`` (m/s^2).

    Uses the spherical-Earth J2-free approximation: g_ecef = -GM / r^3 * r
    plus centripetal acceleration omega x (omega x r).
    """
    r = float(np.linalg.norm(pos))
    if r <= 0.0:
        return np.zeros(3, dtype=np.float64)
    g_grav = -_WGS84_GM / (r ** 3) * pos
    # centripetal: omega x (omega x r); omega = [0,0,Omega]
    x, y, _z = pos
    g_centrip = np.array(
        [_WGS84_OMEGA ** 2 * x, _WGS84_OMEGA ** 2 * y, 0.0],
        dtype=np.float64,
    )
    return g_grav + g_centrip


def _quat_mul(q: np.ndarray, r: np.ndarray) -> np.ndarray:
    return np.array(
        [
            q[0] * r[0] - q[1] * r[1] - q[2] * r[2] - q[3] * r[3],
            q[0] * r[1] + q[1] * r[0] + q[2] * r[3] - q[3] * r[2],
            q[0] * r[2] - q[1] * r[3] + q[2] * r[0] + q[3] * r[1],
            q[0] * r[3] + q[1] * r[2] - q[2] * r[1] + q[3] * r[0],
        ],
        dtype=np.float64,
    )


def _quat_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    half = 0.5 * angle
    s = math.sin(half)
    return np.array([math.cos(half), axis[0] * s, axis[1] * s, axis[2] * s], dtype=np.float64)


def _quat_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _estimate_velocity(rtk_tows: np.ndarray, rtk_pos: np.ndarray, tow: float) -> np.ndarray | None:
    idx = int(np.searchsorted(rtk_tows, tow))
    if idx < 1 or idx >= rtk_tows.size:
        return None
    dt = float(rtk_tows[idx] - rtk_tows[idx - 1])
    if dt <= 0.0 or dt > 2.0:
        return None
    return (rtk_pos[idx] - rtk_pos[idx - 1]) / dt


def _initial_quaternion(acc_mean: np.ndarray, pos: np.ndarray, heading_ecef: np.ndarray | None) -> np.ndarray:
    """Estimate body-to-ECEF quaternion from gravity and heading.

    The accelerometer (at rest) sees +gravity in the body frame's -z-ish
    direction; we assume the IMU z axis aligns roughly with the local
    vertical. Heading (body x axis direction in ECEF) is derived from the
    GNSS velocity vector if available.
    """
    R_enu = _ecef_to_enu_R(pos)
    # Start from identity body = ENU orientation.
    R_bn = np.eye(3, dtype=np.float64)
    if heading_ecef is not None and np.linalg.norm(heading_ecef) > 0.3:
        h_enu = R_enu @ heading_ecef
        yaw = math.atan2(h_enu[0], h_enu[1])
        c, s = math.cos(yaw), math.sin(yaw)
        R_bn = np.array(
            [
                [c, -s, 0.0],
                [s, c, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
    R_be = R_enu.T @ R_bn
    # Convert to quaternion
    tr = R_be[0, 0] + R_be[1, 1] + R_be[2, 2]
    if tr > 0:
        S = 2.0 * math.sqrt(tr + 1.0)
        qw = 0.25 * S
        qx = (R_be[2, 1] - R_be[1, 2]) / S
        qy = (R_be[0, 2] - R_be[2, 0]) / S
        qz = (R_be[1, 0] - R_be[0, 1]) / S
    elif (R_be[0, 0] > R_be[1, 1]) and (R_be[0, 0] > R_be[2, 2]):
        S = 2.0 * math.sqrt(1.0 + R_be[0, 0] - R_be[1, 1] - R_be[2, 2])
        qw = (R_be[2, 1] - R_be[1, 2]) / S
        qx = 0.25 * S
        qy = (R_be[0, 1] + R_be[1, 0]) / S
        qz = (R_be[0, 2] + R_be[2, 0]) / S
    elif R_be[1, 1] > R_be[2, 2]:
        S = 2.0 * math.sqrt(1.0 + R_be[1, 1] - R_be[0, 0] - R_be[2, 2])
        qw = (R_be[0, 2] - R_be[2, 0]) / S
        qx = (R_be[0, 1] + R_be[1, 0]) / S
        qy = 0.25 * S
        qz = (R_be[1, 2] + R_be[2, 1]) / S
    else:
        S = 2.0 * math.sqrt(1.0 + R_be[2, 2] - R_be[0, 0] - R_be[1, 1])
        qw = (R_be[1, 0] - R_be[0, 1]) / S
        qx = (R_be[0, 2] + R_be[2, 0]) / S
        qy = (R_be[1, 2] + R_be[2, 1]) / S
        qz = 0.25 * S
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    q /= np.linalg.norm(q)
    return q


def _run_hold(ref: list[tuple[float, np.ndarray]], rtk: dict[float, tuple[np.ndarray, int]]) -> np.ndarray:
    rtk_tows = np.array(sorted(rtk.keys()), dtype=np.float64)
    rtk_pos = np.array([rtk[t][0] for t in rtk_tows], dtype=np.float64)
    est = np.zeros((len(ref), 3), dtype=np.float64)
    last_pos: np.ndarray | None = None
    for i, (tow, _t) in enumerate(ref):
        if tow in rtk:
            last_pos = rtk[tow][0]
            est[i] = last_pos
        elif last_pos is not None:
            est[i] = last_pos
    return est


def _run_cv(ref: list[tuple[float, np.ndarray]], rtk: dict[float, tuple[np.ndarray, int]]) -> np.ndarray:
    rtk_tows = np.array(sorted(rtk.keys()), dtype=np.float64)
    rtk_pos_arr = np.array([rtk[t][0] for t in rtk_tows], dtype=np.float64)
    est = np.zeros((len(ref), 3), dtype=np.float64)
    last_tow: float | None = None
    last_pos: np.ndarray | None = None
    last_vel: np.ndarray | None = None
    for i, (tow, _t) in enumerate(ref):
        if tow in rtk:
            new_pos = rtk[tow][0]
            if last_tow is not None and (tow - last_tow) > 0:
                last_vel = (new_pos - last_pos) / (tow - last_tow)
            last_tow = tow
            last_pos = new_pos
            est[i] = new_pos
        else:
            vel = (
                last_vel
                if last_vel is not None
                else _estimate_velocity(rtk_tows, rtk_pos_arr, tow)
            )
            if last_pos is None:
                continue
            dt = tow - (last_tow if last_tow is not None else tow)
            if vel is None:
                est[i] = last_pos
            else:
                est[i] = last_pos + vel * dt
    return est


def _run_imu(
    ref: list[tuple[float, np.ndarray]],
    rtk: dict[float, tuple[np.ndarray, int]],
    imu_tow: np.ndarray,
    imu_acc: np.ndarray,
    imu_gyro: np.ndarray,
    max_gap_s: float = 5.0,
) -> np.ndarray:
    """IMU-aided fill. Between RTK fixes, integrate IMU strapdown.

    When the gap exceeds ``max_gap_s`` or IMU data is missing, fall back
    to constant-velocity extrapolation.
    """
    rtk_tows = np.array(sorted(rtk.keys()), dtype=np.float64)
    rtk_pos_arr = np.array([rtk[t][0] for t in rtk_tows], dtype=np.float64)
    est = np.zeros((len(ref), 3), dtype=np.float64)

    pos: np.ndarray | None = None
    vel: np.ndarray | None = None
    quat: np.ndarray | None = None
    acc_bias = np.zeros(3, dtype=np.float64)
    gyro_bias = np.zeros(3, dtype=np.float64)
    last_tow: float | None = None

    # pre-compute IMU index lookup
    imu_count = imu_tow.size

    for i, (tow, _t) in enumerate(ref):
        if tow in rtk:
            new_pos = rtk[tow][0]
            # Velocity: use neighbouring RTK fixes when available
            new_vel = _estimate_velocity(rtk_tows, rtk_pos_arr, tow)
            if new_vel is not None:
                vel = new_vel
            elif vel is None:
                vel = np.zeros(3, dtype=np.float64)
            if quat is None:
                quat = _initial_quaternion(np.array([0.0, 0.0, 9.8]), new_pos, vel)
            pos = new_pos
            last_tow = tow
            est[i] = new_pos
            continue

        if pos is None or last_tow is None:
            continue

        dt_gap = tow - last_tow
        if dt_gap <= 0:
            est[i] = pos
            continue
        if dt_gap > max_gap_s or quat is None or vel is None:
            # Fall back to CV
            if vel is not None:
                est[i] = pos + vel * dt_gap
            else:
                est[i] = pos
            last_tow = tow
            # NOTE: do not advance pos under CV; keep last RTK pos
            continue

        # Strapdown integration from last_tow to tow using IMU samples.
        start_idx = int(np.searchsorted(imu_tow, last_tow))
        end_idx = int(np.searchsorted(imu_tow, tow))
        if start_idx >= imu_count:
            est[i] = pos + vel * dt_gap
            last_tow = tow
            continue
        if end_idx > imu_count:
            end_idx = imu_count

        p = pos.copy()
        v = vel.copy()
        q = quat.copy()
        prev_t = last_tow
        for k in range(start_idx, end_idx):
            t_k = float(imu_tow[k])
            if t_k <= prev_t:
                continue
            dt = t_k - prev_t
            if dt > 0.5:
                # IMU gap too large
                dt = 0.0
            if dt > 0:
                # Propagate attitude: body-frame angular velocity gyro - bias
                omega = imu_gyro[k] - gyro_bias
                norm_om = float(np.linalg.norm(omega))
                if norm_om * dt > 1.0e-9:
                    dq = _quat_from_axis_angle(omega / norm_om, norm_om * dt)
                    q = _quat_mul(q, dq)
                    q /= np.linalg.norm(q)
                R_be = _quat_to_matrix(q)
                a_body = imu_acc[k] - acc_bias
                a_ecef = R_be @ a_body + _gravity_ecef(p)
                v = v + a_ecef * dt
                p = p + v * dt
            prev_t = t_k
        # Final small step to the exact requested tow
        if tow > prev_t:
            dt_final = tow - prev_t
            if imu_count > 0 and end_idx > 0:
                a_body = imu_acc[min(end_idx - 1, imu_count - 1)] - acc_bias
                omega = imu_gyro[min(end_idx - 1, imu_count - 1)] - gyro_bias
                norm_om = float(np.linalg.norm(omega))
                if norm_om * dt_final > 1.0e-9:
                    dq = _quat_from_axis_angle(omega / norm_om, norm_om * dt_final)
                    q = _quat_mul(q, dq)
                    q /= np.linalg.norm(q)
                R_be = _quat_to_matrix(q)
                a_ecef = R_be @ a_body + _gravity_ecef(p)
                v = v + a_ecef * dt_final
                p = p + v * dt_final

        est[i] = p
        last_tow = tow
        # Persist only the mid-propagation state transiently; do not
        # commit to (p, v, q) globally because RTK will reset on next
        # fix. This avoids drift compounding across multiple gaps.

    return est


def _score_strategy(name: str, est: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    # Replace NaN rows (no data yet) with origin -> guaranteed fail,
    # denominator still includes that segment.
    bad = ~np.all(np.isfinite(est), axis=1)
    if bad.any():
        est = np.where(bad[:, None], 0.0, est)
    score = score_ppc2024(est, truth)
    return {
        f"{name}_ppc_pct": float(score.score_pct),
        f"{name}_pass_m": float(score.pass_distance_m),
        f"{name}_total_m": float(score.total_distance_m),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="IMU-aided libgnss++ RTK filler for PPC full runs")
    parser.add_argument("--data-root", type=Path, default=_DEFAULT_DATA_ROOT)
    parser.add_argument("--rtk-dir", type=Path, default=_DEFAULT_RTK_DIR)
    parser.add_argument("--results-prefix", type=str, default="ppc_imu_fusion")
    parser.add_argument(
        "--max-imu-gap-s",
        type=float,
        default=5.0,
        help="Fall back to CV when RTK gap exceeds this value",
    )
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    rtk_dir = args.rtk_dir.resolve()
    rows: list[dict[str, object]] = []
    agg_rtk = agg_hold = agg_cv = agg_imu = agg_total = 0.0

    print("=" * 72)
    print("  libgnss++ RTK + IMU fusion (honest full-run PPC)")
    print("=" * 72)

    for city, run in _RUNS:
        run_dir = data_root / city / run
        rtk_pos = rtk_dir / f"{city}_{run}_full.pos"
        if not rtk_pos.exists():
            print(f"  skip {city}/{run}: {rtk_pos} missing")
            continue
        ref = _load_reference(run_dir / "reference.csv")
        truth = np.array([r[1] for r in ref], dtype=np.float64)
        rtk = _parse_rtk_pos(rtk_pos)
        # drop TOWs outside the reference window to keep sizes matched
        ref_tow_set = {r[0] for r in ref}
        rtk = {k: v for k, v in rtk.items() if k in ref_tow_set}

        imu_tow, imu_acc, imu_gyro = _load_imu(run_dir / "imu.csv")

        # Baselines
        est_rtk_only = np.zeros_like(truth)
        for i, (tow, _t) in enumerate(ref):
            if tow in rtk:
                est_rtk_only[i] = rtk[tow][0]

        est_hold = _run_hold(ref, rtk)
        est_cv = _run_cv(ref, rtk)
        est_imu = _run_imu(ref, rtk, imu_tow, imu_acc, imu_gyro, max_gap_s=args.max_imu_gap_s)

        score_rtk = _score_strategy("rtk", est_rtk_only, truth)
        score_hold = _score_strategy("hold", est_hold, truth)
        score_cv = _score_strategy("cv", est_cv, truth)
        score_imu = _score_strategy("imu", est_imu, truth)

        row = {
            "city": city,
            "run": run,
            "n_ref_epochs": len(ref),
            "n_rtk_pts": len(rtk),
            "true_arc_m": score_rtk["rtk_total_m"],
        }
        row.update(score_rtk)
        row.update(score_hold)
        row.update(score_cv)
        row.update(score_imu)
        rows.append(row)

        agg_rtk += score_rtk["rtk_pass_m"]
        agg_hold += score_hold["hold_pass_m"]
        agg_cv += score_cv["cv_pass_m"]
        agg_imu += score_imu["imu_pass_m"]
        agg_total += score_rtk["rtk_total_m"]

        print(
            f"  {city}/{run}: rtk={score_rtk['rtk_ppc_pct']:5.2f}%  "
            f"hold={score_hold['hold_ppc_pct']:5.2f}%  "
            f"cv={score_cv['cv_ppc_pct']:5.2f}%  "
            f"imu={score_imu['imu_ppc_pct']:5.2f}%  "
            f"(arc {row['true_arc_m']:.0f}m, rtk={len(rtk)}/{len(ref)})"
        )

    out_csv = _SCRIPT_DIR / "results" / f"{args.results_prefix}_runs.csv"
    fields = list(rows[0].keys()) if rows else []
    with out_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    print()
    print("=" * 72)
    if agg_total > 0:
        print(f"  RTK-only  aggregate: {100 * agg_rtk / agg_total:6.2f}%  "
              f"(pass {agg_rtk:.1f}m / total {agg_total:.1f}m)")
        print(f"  Hold-last aggregate: {100 * agg_hold / agg_total:6.2f}%")
        print(f"  CV-fill   aggregate: {100 * agg_cv / agg_total:6.2f}%")
        print(f"  IMU-fuse  aggregate: {100 * agg_imu / agg_total:6.2f}%")
        print("  TURING target      : 85.60%")
    print(f"  Saved: {out_csv}")
    print("=" * 72)


if __name__ == "__main__":
    main()
