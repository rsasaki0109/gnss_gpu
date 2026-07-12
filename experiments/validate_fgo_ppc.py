#!/usr/bin/env python3
"""Validate GPU FGO on PPC-Dataset (taroz/PPC-Dataset) RINEX runs.

Uses the RTKLIB export_spp_meas pipeline for observation model alignment,
then runs WLS + FGO and compares against PPC reference.csv ground truth.

Usage:
    PYTHONPATH=python python3 experiments/validate_fgo_ppc.py
    PYTHONPATH=python python3 experiments/validate_fgo_ppc.py --run tokyo/run1 --max-epochs 300
    PYTHONPATH=python python3 experiments/validate_fgo_ppc.py --all
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
_EXPERIMENTS = Path(__file__).resolve().parent
for _p in (_REPO, _EXPERIMENTS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from experiments.gsdc2023_imu import (  # noqa: E402
    IMUPreintegration,
    imu_preintegration_segment_with_bias_jacobians,
)
from experiments.ppc_imu_adapter import load_ppc_imu_preintegration  # noqa: E402
from gnss_gpu import wls_position  # noqa: E402
from gnss_gpu.fgo import fgo_gnss_lm, fgo_gnss_lm_vd  # noqa: E402
from gnss_gpu.io.ppc import PPCDatasetLoader  # noqa: E402
from gnss_gpu.validation import elevation_azimuth  # noqa: E402
from gnss_gpu.io.rinex import read_rinex_obs  # noqa: E402
from gnss_gpu.io.nav_rinex import (  # noqa: E402
    _datetime_to_gps_seconds_of_week,
    _datetime_to_gps_week,
    read_gps_klobuchar_from_nav_header,
    read_nav_rinex,
)
from gnss_gpu.spp import correct_pseudoranges  # noqa: E402
from gtsam_public_dataset import SYS_ID_TO_KIND  # noqa: E402

C_LIGHT = 299792458.0
L1_WAVELENGTH_M = C_LIGHT / 1575.42e6

# Fixed-frequency civil L1/E1/B1I wavelengths by RINEX system prefix. GLONASS
# (R) is FDMA (frequency depends on per-satellite channel, glo_frequency_channel)
# and is intentionally omitted here; see WP3B_REPORT.md D1/D3 for the
# documented limitation (GLONASS Doppler is left disabled, not mis-scaled).
_SYSTEM_WAVELENGTH_M = {
    "G": C_LIGHT / 1575.42e6,  # GPS L1 C/A
    "E": C_LIGHT / 1575.42e6,  # Galileo E1
    "J": C_LIGHT / 1575.42e6,  # QZSS L1
    "C": C_LIGHT / 1561.098e6,  # BeiDou B1I
}

# Default PPC-Dataset location (first existing candidate wins)
_DEFAULT_PPC_ROOT = _REPO.parent / "ref" / "PPC-Dataset" / "PPC-Dataset"
_PPC_ROOT_CANDIDATES = (
    Path("E:/datasets/PPC-Dataset-data"),
    _REPO / "datasets" / "PPC-Dataset-data",
    _DEFAULT_PPC_ROOT,
)

# All 6 runs
ALL_RUNS = [
    "tokyo/run1", "tokyo/run2", "tokyo/run3",
    "nagoya/run1", "nagoya/run2", "nagoya/run3",
]


def _resolve_ppc_root(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    for candidate in _PPC_ROOT_CANDIDATES:
        if candidate.is_dir():
            return candidate
    return _DEFAULT_PPC_ROOT


def _doppler_hz_to_range_rate(
    doppler_hz: np.ndarray,
    *,
    wavelength_m: float | np.ndarray = L1_WAVELENGTH_M,
) -> np.ndarray:
    """RINEX convention: pseudorange-rate (m/s) = -doppler_hz * wavelength.

    ``wavelength_m`` may be a scalar (legacy GPS-only behaviour) or an array
    broadcastable against ``doppler_hz`` for multi-GNSS per-satellite
    wavelengths (see ``_SYSTEM_WAVELENGTH_M`` / ``_per_satellite_wavelength_m``).
    """
    hz = np.asarray(doppler_hz, dtype=np.float64)
    wl = np.asarray(wavelength_m, dtype=np.float64)
    out = np.zeros_like(hz, dtype=np.float64)
    finite = np.isfinite(hz) & (hz != 0.0)
    if wl.ndim == 0:
        out[finite] = -hz[finite] * float(wl)
    else:
        wl_b = np.broadcast_to(wl, hz.shape)
        finite = finite & np.isfinite(wl_b) & (wl_b > 0.0)
        out[finite] = -hz[finite] * wl_b[finite]
    return out


def _per_satellite_wavelength_m(used_prns: list[list[str]], max_sats: int) -> np.ndarray:
    """Build a ``(T, max_sats)`` per-satellite Doppler wavelength array.

    GLONASS (``R``) satellites get ``nan`` (unknown FDMA channel wavelength
    without per-satellite channel plumbing) so their Doppler observations are
    dropped rather than mis-scaled with the GPS L1 wavelength; see D1/D3
    notes in WP3B_REPORT.md.
    """
    n_epoch = len(used_prns)
    wl = np.full((n_epoch, max_sats), np.nan, dtype=np.float64)
    for t, sats in enumerate(used_prns):
        for i, sid in enumerate(sats):
            sys_char = sid[0] if sid else "G"
            wl[t, i] = _SYSTEM_WAVELENGTH_M.get(sys_char, np.nan)
    return wl


def _robust_irls_fit(
    A: np.ndarray,
    rhs: np.ndarray,
    *,
    huber_c: float = 1.5,
    n_iters: int = 5,
    min_sigma: float = 1.0,
) -> np.ndarray:
    """Small Huber-IRLS linear fit, robust to a minority of gross outliers.

    A single ordinary-least-squares pass over-fits toward gross outliers when
    the observation count is small relative to the outlier magnitude (e.g. an
    8-satellite Doppler epoch with one ~500 m/s multipath/cycle-slip spike
    biases *every* OLS residual, defeating a naive post-hoc median/MAD gate).
    Iteratively down-weighting large-residual rows converges the fit itself
    toward the clean majority within a handful of iterations.
    """
    n = A.shape[1]
    w = np.ones(rhs.shape[0], dtype=np.float64)
    sol = np.zeros(n, dtype=np.float64)
    for _ in range(max(n_iters, 1)):
        W = w[:, None]
        try:
            sol = np.linalg.lstsq(A * np.sqrt(W), rhs * np.sqrt(w), rcond=None)[0]
        except np.linalg.LinAlgError:
            break
        resid = A @ sol - rhs
        med = float(np.median(resid))
        mad = float(np.median(np.abs(resid - med)))
        sigma = max(1.4826 * mad, min_sigma)
        z = np.abs(resid - med) / sigma
        w = np.where(z <= huber_c, 1.0, huber_c / np.maximum(z, 1e-9))
    return sol


def _gate_doppler_outliers_per_epoch(
    sat_ecef: np.ndarray,
    sat_vel: np.ndarray,
    sat_clock_drift: np.ndarray,
    doppler_range_rate: np.ndarray,
    doppler_weights: np.ndarray,
    rx_state: np.ndarray,
    *,
    gate_sigma: float = 3.0,
    min_sigma_mps: float = 1.0,
    min_obs: int = 5,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Reject per-epoch Doppler outliers via a robust (median/MAD) residual gate.

    For each epoch, fits a lightweight 4-unknown ``[rx_vx, rx_vy, rx_vz, drift]``
    Huber-IRLS linear fit (``_robust_irls_fit``) against all Doppler
    observations at that epoch (using the current position estimate as the
    linearization point, matching the native VD Doppler-factor convention in
    ``fgo.cu:doppler_prediction_vd``: ``los = (sat - rx) / |sat - rx|``,
    ``pred = drift + los.(sat_vel - rx_vel) - sat_clock_drift``), then zeroes
    ``doppler_weights`` for observations whose residual against the
    *robust* fit exceeds ``gate_sigma`` robust-sigma (``1.4826 * MAD``,
    floored at ``min_sigma_mps`` so already-clean epochs are not
    over-aggressively pruned). Epochs with fewer than ``min_obs`` valid
    Doppler observations are left untouched (too few DOF to fit + gate).

    Returns the gated ``doppler_weights`` copy and a stats dict
    (``n_epochs_gated``, ``n_obs_gated``, ``n_obs_total``).
    """
    n_epoch, max_sats = doppler_weights.shape
    out_weights = doppler_weights.copy()
    n_epochs_gated = 0
    n_obs_gated = 0
    n_obs_total = 0

    for t in range(n_epoch):
        w_t = doppler_weights[t]
        idx = np.flatnonzero(w_t > 0)
        n_obs_total += idx.size
        if idx.size < min_obs:
            continue
        rx_pos = rx_state[t, :3]
        diff = sat_ecef[t, idx] - rx_pos[None, :]
        rng = np.linalg.norm(diff, axis=1)
        valid_rng = rng > 1e3
        if int(valid_rng.sum()) < min_obs:
            continue
        idx = idx[valid_rng]
        diff = diff[valid_rng]
        rng = rng[valid_rng]
        los = diff / rng[:, None]
        scd = sat_clock_drift[t, idx] if sat_clock_drift is not None else np.zeros(idx.size)
        rhs = (
            doppler_range_rate[t, idx]
            - np.where(np.isfinite(scd), scd, 0.0)
            - np.einsum("ij,ij->i", los, sat_vel[t, idx])
        )
        A = np.column_stack([-los, np.ones(idx.size)])
        sol = _robust_irls_fit(A, rhs, min_sigma=min_sigma_mps)
        resid = A @ sol - rhs
        med = float(np.median(resid))
        mad = float(np.median(np.abs(resid - med)))
        sigma = max(1.4826 * mad, min_sigma_mps)
        bad = np.abs(resid - med) > (gate_sigma * sigma)
        if bad.any():
            n_epochs_gated += 1
            n_obs_gated += int(bad.sum())
            out_weights[t, idx[bad]] = 0.0

    stats = {
        "n_epochs_gated": n_epochs_gated,
        "n_obs_gated": n_obs_gated,
        "n_obs_total": n_obs_total,
    }
    return out_weights, stats


# --- TASK_E WP3c: elevation mask + per-constellation weighting ------------


def _elevation_deg_per_epoch(
    sat_ecef: np.ndarray,
    weights: np.ndarray,
    rx_state: np.ndarray,
) -> np.ndarray:
    """Per-(epoch, sat) elevation angle in degrees, NaN where unweighted/unset.

    Uses ``gnss_gpu.validation.elevation_azimuth`` (WGS84 geodetic ENU
    projection) with the per-epoch WLS receiver position as the observer --
    the same "compute elevation from sat_ecef and the WLS position" pattern
    already used for ``observation_min_elevation_deg`` in
    ``experiments/gsdc2023_bridge_config.py`` and the two-pass elevation
    weighting in this file's ``run_fgo_on_ppc`` (RTKLIB) path.
    """
    n_epoch, max_sats = weights.shape
    elev_deg = np.full((n_epoch, max_sats), np.nan, dtype=np.float64)
    for t in range(n_epoch):
        idx = np.flatnonzero(weights[t] > 0)
        if idx.size == 0:
            continue
        rx = rx_state[t, :3]
        if not np.all(np.isfinite(rx)) or np.linalg.norm(rx) < 1e3:
            continue
        el_rad, _az = elevation_azimuth(rx, sat_ecef[t, idx])
        elev_deg[t, idx] = np.degrees(el_rad)
    return elev_deg


def _apply_elevation_mask(
    sat_ecef: np.ndarray,
    weights: np.ndarray,
    rx_state: np.ndarray,
    min_elevation_deg: float,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Zero ``weights`` for satellites below ``min_elevation_deg`` at ``rx_state``.

    ``rx_state``: ``(T, >=3)`` per-epoch receiver ECEF position (a WLS seed)
    used as the elevation observer. ``min_elevation_deg <= 0`` disables the
    mask and returns ``weights`` unchanged (default, legacy behaviour).
    """
    n_obs_total = int(np.sum(weights > 0))
    if min_elevation_deg <= 0.0:
        return weights, {
            "n_obs_masked": 0,
            "n_obs_total": n_obs_total,
            "min_elevation_deg": min_elevation_deg,
        }

    elev_deg = _elevation_deg_per_epoch(sat_ecef, weights, rx_state)
    below = np.isfinite(elev_deg) & (elev_deg < min_elevation_deg)
    out = weights.copy()
    out[below] = 0.0
    return out, {
        "n_obs_masked": int(below.sum()),
        "n_obs_total": n_obs_total,
        "min_elevation_deg": min_elevation_deg,
    }


# WP3c work item 2b: per-constellation pseudorange sigma scaling. Values are
# multiplicative *sigma* scale factors (1.0 = no change); applied to
# ``weights`` as ``weight /= scale**2`` (the loader's weights are
# inverse-variance-like, derived from SNR). Defaults per TASK_E work item 2:
# GPS/Galileo/QZSS trusted at face value, BeiDou down-weighted 1.5x sigma,
# GLONASS down-weighted 2x sigma -- tuned against the per-constellation WLS
# residual RMS breakdown (see WP3C_REPORT.md root-cause attribution table).
DEFAULT_CONSTELLATION_SIGMA_SCALE: dict[str, float] = {
    "G": 1.0,
    "E": 1.0,
    "J": 1.0,
    "C": 1.5,
    "R": 2.0,
}


def _apply_constellation_sigma_scaling(
    weights: np.ndarray,
    used_prns: list[list[str]],
    sigma_scale: dict[str, float] | None = None,
) -> np.ndarray:
    """Rescale ``weights`` per-satellite by a constellation-dependent sigma factor.

    ``used_prns[t][i]`` gives the PRN (e.g. ``"C05"``) for column ``i`` of
    epoch ``t``; its first character selects the sigma-scale factor from
    ``sigma_scale`` (defaults: ``DEFAULT_CONSTELLATION_SIGMA_SCALE``). A
    factor of 1.0 is a no-op; factors >1.0 (e.g. BeiDou/GLONASS) shrink the
    corresponding weight by ``1/factor**2``.
    """
    scale = dict(DEFAULT_CONSTELLATION_SIGMA_SCALE)
    if sigma_scale:
        scale.update(sigma_scale)
    out = weights.copy()
    n_epoch = min(len(used_prns), weights.shape[0])
    for t in range(n_epoch):
        row = used_prns[t]
        for i in range(min(len(row), out.shape[1])):
            if out[t, i] <= 0.0:
                continue
            sid = row[i]
            s = scale.get(sid[0] if sid else "G", 1.0)
            if s != 1.0:
                out[t, i] = out[t, i] / (s * s)
    return out


def _ecef_to_llh_deg(x: float, y: float, z: float) -> tuple[float, float, float]:
    from experiments.evaluate import ecef_to_lla

    lat_rad, lon_rad, alt_m = ecef_to_lla(x, y, z)
    return math.degrees(lat_rad), math.degrees(lon_rad), alt_m


def export_trajectory_csv(
    path: Path,
    times: np.ndarray,
    ecef_xyz: np.ndarray,
    *,
    fix_flags: np.ndarray | None = None,
) -> None:
    """Write epoch trajectory CSV for score_vs_inuex35 (tow,ecef_*,fix)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "tow",
                "lat_deg",
                "lon_deg",
                "height_m",
                "ecef_x",
                "ecef_y",
                "ecef_z",
                "fix",
            ],
        )
        writer.writeheader()
        for i, tow in enumerate(times):
            x, y, z = (float(v) for v in ecef_xyz[i, :3])
            lat, lon, h = _ecef_to_llh_deg(x, y, z)
            fix_val = "0"
            if fix_flags is not None and i < len(fix_flags):
                fix_val = "1" if bool(fix_flags[i]) else "0"
            writer.writerow(
                {
                    "tow": f"{tow:.3f}",
                    "lat_deg": f"{lat:.8f}",
                    "lon_deg": f"{lon:.8f}",
                    "height_m": f"{h:.4f}",
                    "ecef_x": f"{x:.6f}",
                    "ecef_y": f"{y:.6f}",
                    "ecef_z": f"{z:.6f}",
                    "fix": fix_val,
                }
            )


def _default_export_spp_meas() -> Path | None:
    envp = os.environ.get("RTKLIB_EXPORT_SPP_MEAS")
    if envp:
        p = Path(envp)
        if p.is_file():
            return p
    guess = (
        _REPO.parent / "ref" / "RTKLIB-demo5" / "app" / "consapp"
        / "rnx2rtkp" / "gcc" / "export_spp_meas"
    )
    return guess if guess.is_file() else None


def _load_ppc_reference(ref_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load PPC reference.csv as (tow, ecef)."""
    tow_list, ecef_list = [], []
    with open(ref_path, newline="") as f:
        for row in csv.DictReader(f, skipinitialspace=True):
            tow_list.append(float(row["GPS TOW (s)"]))
            ecef_list.append([
                float(row["ECEF X (m)"]),
                float(row["ECEF Y (m)"]),
                float(row["ECEF Z (m)"]),
            ])
    return np.array(tow_list), np.array(ecef_list)


def _nearest_ref_error_2d(
    tow: float, ref_tow: np.ndarray, ref_ecef: np.ndarray, est: np.ndarray,
) -> float:
    i = int(np.argmin(np.abs(ref_tow - tow)))
    d = est[:3] - ref_ecef[i]
    return float(np.linalg.norm(d[:2]))


def _nearest_ref_error_3d(
    tow: float, ref_tow: np.ndarray, ref_ecef: np.ndarray, est: np.ndarray,
) -> float:
    i = int(np.argmin(np.abs(ref_tow - tow)))
    d = est[:3] - ref_ecef[i]
    return float(np.linalg.norm(d))


def _run_rtklib_export(
    exe: Path, obs_p: Path, nav_p: Path, el_mask_deg: float,
) -> dict[tuple[int, float, str], dict[str, float]]:
    """Run export_spp_meas and parse CSV.

    Supports both old (GPS-only) and new multi-GNSS format (with ``sys_id`` column).
    """
    import subprocess
    import tempfile

    fd, tmp = tempfile.mkstemp(suffix="_ppc_spp.csv", text=True)
    os.close(fd)
    tmp_p = Path(tmp)
    cmd = [str(exe), str(obs_p), str(nav_p), "-m", str(el_mask_deg)]
    with open(tmp_p, "w") as fp:
        subprocess.run(cmd, check=True, stdout=fp, stderr=subprocess.PIPE, text=True)
    out: dict[tuple[int, float, str], dict[str, float]] = {}
    with open(tmp_p, newline="") as f:
        for row in csv.DictReader(f):
            wk = int(row["gps_week"])
            tow = round(float(row["gps_tow"]), 4)
            sid = row["sat_id"].strip()
            d: dict[str, float] = {
                "prange_m": float(row["prange_m"]),
                "iono_m": float(row["iono_m"]),
                "trop_m": float(row["trop_m"]),
                "sat_clk_m": float(row["sat_clk_m"]),
                "satx": float(row["satx"]),
                "saty": float(row["saty"]),
                "satz": float(row["satz"]),
            }
            if "el_rad" in row and "var_total" in row:
                d["el_rad"] = float(row["el_rad"])
                d["var_total"] = float(row["var_total"])
            if "svx" in row:
                d["svx"] = float(row["svx"])
                d["svy"] = float(row["svy"])
                d["svz"] = float(row["svz"])
            if "rx_vx" in row:
                d["rx_vx"] = float(row["rx_vx"])
                d["rx_vy"] = float(row["rx_vy"])
                d["rx_vz"] = float(row["rx_vz"])
            # Multi-GNSS sys_id column (backward compatible)
            if "sys_id" in row and row["sys_id"]:
                d["sys_id"] = row["sys_id"].strip()  # type: ignore[assignment]
            out[(wk, tow, sid)] = d
    tmp_p.unlink(missing_ok=True)
    return out


# Native VD solver caps total state at 16384 (= 2048 epochs × 8 states).
_VD_MAX_STATE = 16384
_VD_MAX_EPOCHS_PER_CHUNK = 1000


def _chunk_ranges(n_epoch: int, chunk_size: int) -> list[tuple[int, int]]:
    if chunk_size <= 0 or n_epoch <= chunk_size:
        return [(0, n_epoch)]
    return [(start, min(start + chunk_size, n_epoch)) for start in range(0, n_epoch, chunk_size)]


def _vd_state_stride(n_clock: int) -> int:
    return 7 + int(n_clock)


def _seed_chunk_boundary_state(
    seg_state: np.ndarray,
    prev_state: np.ndarray,
    *,
    n_clock: int = 1,
) -> None:
    """Carry optimized kinematic state (position/velocity/per-system clocks/drift)
    across chunk boundaries. ``n_clock`` selects how many clock-bias columns
    (indices ``6 .. 6+n_clock-1``) to carry; the drift column at
    ``6+n_clock`` (if present) is carried too."""
    seg_state[0, :3] = prev_state[:3]
    seg_state[0, 3:6] = prev_state[3:6]
    n_clock = max(int(n_clock), 1)
    clock_end = min(6 + n_clock, seg_state.shape[1])
    seg_state[0, 6:clock_end] = prev_state[6:clock_end]
    drift_idx = 6 + n_clock
    if seg_state.shape[1] > drift_idx:
        seg_state[0, drift_idx] = prev_state[drift_idx]


def _solve_fgo_vd_chunked(
    sat_ecef: np.ndarray,
    pseudorange: np.ndarray,
    weights: np.ndarray,
    fgo_state: np.ndarray,
    *,
    n_clock: int,
    motion_sigma_m: float,
    clock_drift_sigma_m: float,
    fgo_iters: int,
    sat_vel: np.ndarray | None,
    doppler: np.ndarray | None,
    doppler_weights: np.ndarray | None,
    sat_clock_drift: np.ndarray | None,
    dt: np.ndarray,
    chunk_epochs: int = _VD_MAX_EPOCHS_PER_CHUNK,
    sys_kind: np.ndarray | None = None,
    doppler_huber_k: float = 0.0,
    imu_preint: IMUPreintegration | None = None,
    imu_position_sigma_m: float = 0.0,
    imu_velocity_sigma_mps: float = 0.0,
) -> tuple[int, float, list[dict[str, float | int | str]]]:
    n_epoch = int(sat_ecef.shape[0])
    state_stride = _vd_state_stride(n_clock)
    ranges = _chunk_ranges(n_epoch, chunk_epochs)
    total_iters = 0
    mse_values: list[float] = []
    chunk_stats: list[dict[str, float | int | str]] = []
    for chunk_idx, (start, end) in enumerate(ranges):
        seg_n = end - start
        seg_n_state = seg_n * state_stride
        if seg_n_state > _VD_MAX_STATE:
            chunk_stats.append(
                {
                    "chunk": chunk_idx,
                    "start": start,
                    "end": end,
                    "n_epoch": seg_n,
                    "n_state": seg_n_state,
                    "iters": -1,
                    "mse_pr": 0.0,
                    "status": "n_state_cap",
                }
            )
            return -1, 0.0, chunk_stats

        seg_state = fgo_state[start:end].copy()
        if start > 0:
            _seed_chunk_boundary_state(seg_state, fgo_state[start - 1], n_clock=n_clock)
        seg_dt = dt[start:end].copy()

        imu_delta_p = imu_delta_v = imu_delta_t = None
        if imu_preint is not None:
            (
                imu_delta_p,
                imu_delta_v,
                _imu_delta_angle,
                imu_delta_t,
                _dp_ba,
                _dv_ba,
                _dp_bg,
                _dv_bg,
                _da_bg,
                _imu_count,
            ) = imu_preintegration_segment_with_bias_jacobians(imu_preint, start, end)

        iters, mse_pr = fgo_gnss_lm_vd(
            sat_ecef[start:end],
            pseudorange[start:end],
            weights[start:end],
            seg_state,
            sys_kind=None if sys_kind is None else sys_kind[start:end],
            n_clock=n_clock,
            motion_sigma_m=motion_sigma_m,
            clock_drift_sigma_m=clock_drift_sigma_m,
            max_iter=fgo_iters,
            tol=1e-7,
            sat_vel=None if sat_vel is None else sat_vel[start:end],
            doppler=None if doppler is None else doppler[start:end],
            doppler_weights=None if doppler_weights is None else doppler_weights[start:end],
            sat_clock_drift=None if sat_clock_drift is None else sat_clock_drift[start:end],
            dt=seg_dt,
            doppler_huber_k=doppler_huber_k,
            imu_delta_p=imu_delta_p,
            imu_delta_v=imu_delta_v,
            imu_delta_t=imu_delta_t,
            imu_position_sigma_m=imu_position_sigma_m if imu_delta_p is not None else 0.0,
            imu_velocity_sigma_mps=imu_velocity_sigma_mps if imu_delta_v is not None else 0.0,
        )
        status = "ok" if int(iters) >= 0 else "native_failed"
        chunk_stats.append(
            {
                "chunk": chunk_idx,
                "start": start,
                "end": end,
                "n_epoch": seg_n,
                "n_state": seg_n_state,
                "iters": int(iters),
                "mse_pr": float(mse_pr),
                "status": status,
            }
        )
        if int(iters) < 0:
            return int(iters), float(mse_pr), chunk_stats
        fgo_state[start:end] = seg_state
        total_iters += int(iters)
        mse_values.append(float(mse_pr))
        print(
            f"  [chunk {chunk_idx}] epochs [{start}:{end}] iters={int(iters)} mse={float(mse_pr):.4g}",
            flush=True,
        )
    mean_mse = float(np.mean(mse_values)) if mse_values else float("nan")
    return total_iters, mean_mse, chunk_stats


def run_fgo_on_ppc_native(
    run_dir: Path,
    *,
    max_epochs: int = 300,
    start_epoch: int = 0,
    motion_sigma_m: float = 1.0,
    fgo_iters: int = 8,
    clock_drift_sigma_m: float = 1.0,
    doppler_mode: str = "off",
    export_csv: Path | None = None,
    systems: tuple[str, ...] = ("G",),
    doppler_gate_sigma: float = 0.0,
    doppler_huber_k: float = 0.0,
    imu_enabled: bool = False,
    imu_position_sigma_m: float = 0.0,
    imu_velocity_sigma_mps: float = 0.0,
    chunk_epochs: int = 0,
    elevation_mask_deg: float = 0.0,
    constellation_weighting: bool = False,
    constellation_sigma_scale: dict[str, float] | None = None,
) -> dict:
    """Run in-repo Ephemeris + PPCDatasetLoader FGO (no RTKLIB).

    ``systems``: constellation prefixes fed to ``PPCDatasetLoader`` (D2 audit;
    default ``("G",)`` reproduces the WP3a GPS-only backbone). Passing more
    than one system enables a per-constellation clock/ISB state
    (``n_clock = len(systems actually observed)``) mirroring the RTKLIB
    ``--multi-gnss`` path.
    ``doppler_gate_sigma``: robust per-epoch Doppler outlier gate (D1); ``0``
    disables it. ``doppler_huber_k``: native Huber-kernel threshold (m/s,
    Mahalanobis) passed straight to ``fgo_gnss_lm_vd``; ``0`` keeps pure L2.
    ``imu_enabled``: wire PPC ``imu.csv`` preintegrated deltas (D3) into the
    VD solver as ``imu_delta_p`` / ``imu_delta_v`` priors between epochs.
    ``chunk_epochs``: override the per-chunk epoch budget (0 = auto, i.e. the
    largest of 1000 epochs that fits ``_VD_MAX_STATE`` given ``n_clock``).
    The dense per-chunk LM solve cost grows roughly with the cube of
    ``n_state = chunk_epochs * state_stride``, so wide multi-clock states
    (D2) get dramatically cheaper with a smaller chunk (more, smaller solves)
    at the cost of slightly weaker inter-chunk coupling.
    ``elevation_mask_deg`` (WP3c work item 1): drop satellites below this
    elevation (computed from ``sat_ecef`` and a pass-1 per-epoch WLS
    position) before the pass-2 WLS/FGO solve; ``0`` disables (legacy
    behaviour, no mask).
    ``constellation_weighting`` / ``constellation_sigma_scale`` (WP3c work
    item 2b): when enabled, rescale pseudorange weights per-constellation
    (default ``DEFAULT_CONSTELLATION_SIGMA_SCALE``: BeiDou 1.5x sigma,
    GLONASS 2x sigma, GPS/Galileo/QZSS unchanged) before the pass-2
    WLS/FGO solve.
    """
    if doppler_mode not in {"off", "in-repo"}:
        raise ValueError(f"native path supports doppler off|in-repo, got {doppler_mode!r}")

    ref_p = run_dir / "reference.csv"
    if not ref_p.is_file():
        raise FileNotFoundError(f"Missing: {ref_p}")

    ref_tow, ref_ecef = _load_ppc_reference(ref_p)
    include_vel = doppler_mode == "in-repo"
    loader_kwargs: dict = {"include_sat_velocity": include_vel, "systems": systems}
    if max_epochs > 0:
        loader_kwargs["max_epochs"] = max_epochs
    if start_epoch > 0:
        loader_kwargs["start_epoch"] = start_epoch
    data = PPCDatasetLoader(run_dir).load_experiment_data(**loader_kwargs)

    n_epoch = int(data["n_epochs"])
    if n_epoch < 5:
        raise RuntimeError(f"Only {n_epoch} valid epochs")

    times = np.asarray(data["times"], dtype=np.float64)
    sat_counts = np.asarray(data["satellite_counts"], dtype=np.int32)
    max_sats = int(sat_counts.max())
    used_prns = data["used_prns"]

    sat_ecef = np.zeros((n_epoch, max_sats, 3), dtype=np.float64)
    pseudorange = np.zeros((n_epoch, max_sats), dtype=np.float64)
    weights = np.zeros((n_epoch, max_sats), dtype=np.float64)
    for t in range(n_epoch):
        ns = int(sat_counts[t])
        sat_ecef[t, :ns] = np.asarray(data["sat_ecef"][t], dtype=np.float64)
        pseudorange[t, :ns] = np.asarray(data["pseudoranges"][t], dtype=np.float64)
        weights[t, :ns] = np.asarray(data["weights"][t], dtype=np.float64)

    # D2: per-constellation clock/ISB. Map only the systems actually observed
    # to contiguous clock indices 0..n_clock-1 (keeps n_clock==1, sys_kind=None
    # for the default GPS-only path -- exact WP3a parity).
    constellations = tuple(sorted(data.get("constellations", ("G",))))
    if not constellations:
        constellations = ("G",)
    n_clock = len(constellations)
    sys_kind_arr: np.ndarray | None = None
    if n_clock > 1:
        sys_char_to_clock = {c: i for i, c in enumerate(constellations)}
        sys_kind_arr = np.zeros((n_epoch, max_sats), dtype=np.int32)
        for t in range(n_epoch):
            for i, sid in enumerate(used_prns[t]):
                sys_kind_arr[t, i] = sys_char_to_clock.get(sid[0] if sid else "G", 0)

    def _wls_per_epoch(w_arr: np.ndarray) -> np.ndarray:
        st_arr = np.zeros((n_epoch, 4), dtype=np.float64)
        for t in range(n_epoch):
            w = w_arr[t]
            idx = np.flatnonzero(w > 0)
            if idx.size < 4:
                continue
            st, _ = wls_position(
                sat_ecef[t, idx].reshape(-1), pseudorange[t, idx], w[idx], 25, 1e-9,
            )
            st_arr[t] = st
        return st_arr

    # Pass 1: unmasked/unweighted WLS, used only as the elevation observer
    # position for the elevation mask (WP3c work item 1) -- matches the
    # gsdc2023_bridge_config pattern of computing elevation from sat_ecef and
    # a WLS position rather than ground truth.
    wls_state = _wls_per_epoch(weights)

    elevation_mask_stats: dict[str, float | int] | None = None
    if elevation_mask_deg > 0.0:
        weights, elevation_mask_stats = _apply_elevation_mask(
            sat_ecef, weights, wls_state, elevation_mask_deg,
        )
    if constellation_weighting:
        weights = _apply_constellation_sigma_scaling(
            weights, used_prns, constellation_sigma_scale,
        )
    # Pass 2: re-solve WLS with the final (possibly masked/reweighted)
    # weights so the FGO seed and per-chunk kinematic state reflect them.
    if elevation_mask_deg > 0.0 or constellation_weighting:
        wls_state = _wls_per_epoch(weights)

    fgo_state = np.zeros((n_epoch, 7 + n_clock), dtype=np.float64)
    fgo_state[:, :3] = wls_state[:, :3]
    fgo_state[:, 6] = wls_state[:, 3]

    dt_arr = np.zeros(n_epoch, dtype=np.float64)
    fallback_dt = float(data.get("dt", 0.2))
    for t in range(n_epoch - 1):
        dt_arr[t] = float(times[t + 1] - times[t])
        if dt_arr[t] <= 0 or dt_arr[t] > 30:
            dt_arr[t] = fallback_dt

    sat_vel_arr: np.ndarray | None = None
    doppler_arr: np.ndarray | None = None
    doppler_w_arr: np.ndarray | None = None
    sat_clock_drift_arr: np.ndarray | None = None
    doppler_gate_stats: dict[str, float | int] | None = None
    if include_vel:
        sat_vel_arr = np.zeros((n_epoch, max_sats, 3), dtype=np.float64)
        doppler_arr = np.zeros((n_epoch, max_sats), dtype=np.float64)
        doppler_w_arr = np.zeros((n_epoch, max_sats), dtype=np.float64)
        sat_clock_drift_arr = np.zeros((n_epoch, max_sats), dtype=np.float64)
        wavelength_arr = _per_satellite_wavelength_m(used_prns, max_sats)
        for t in range(n_epoch):
            ns = int(sat_counts[t])
            sat_vel_arr[t, :ns] = np.asarray(data["sat_velocity"][t], dtype=np.float64)
            dop_hz = np.asarray(data["doppler_hz"][t], dtype=np.float64)
            doppler_arr[t, :ns] = _doppler_hz_to_range_rate(
                dop_hz, wavelength_m=wavelength_arr[t, :ns],
            )
            valid_dop = (
                np.isfinite(dop_hz) & (dop_hz != 0.0) & np.isfinite(wavelength_arr[t, :ns])
            )
            doppler_w_arr[t, :ns] = np.where(valid_dop, weights[t, :ns], 0.0)
            clk_drift = np.asarray(data["clock_drift"][t], dtype=np.float64)
            sat_clock_drift_arr[t, :ns] = clk_drift * C_LIGHT

        if doppler_gate_sigma > 0.0:
            doppler_w_arr, doppler_gate_stats = _gate_doppler_outliers_per_epoch(
                sat_ecef,
                sat_vel_arr,
                sat_clock_drift_arr,
                doppler_arr,
                doppler_w_arr,
                fgo_state,
                gate_sigma=doppler_gate_sigma,
            )

    imu_preint: IMUPreintegration | None = None
    if imu_enabled:
        imu_preint = load_ppc_imu_preintegration(run_dir, times, fgo_state[:, :3])

    # Wider multi-clock states (n_clock>1, D2) need a smaller per-chunk epoch
    # budget to stay under the native n_state cap; shrink the default chunk
    # size here rather than inside `_solve_fgo_vd_chunked` (which preserves
    # explicit-chunk_epochs callers' expectation of hitting the n_state_cap
    # guard, e.g. tests exercising oversized chunks directly).
    auto_chunk_epochs = min(_VD_MAX_EPOCHS_PER_CHUNK, _VD_MAX_STATE // _vd_state_stride(n_clock))
    eff_chunk_epochs = auto_chunk_epochs if chunk_epochs <= 0 else min(chunk_epochs, auto_chunk_epochs)

    iters, mse_pr, chunk_stats = _solve_fgo_vd_chunked(
        sat_ecef,
        pseudorange,
        weights,
        fgo_state,
        n_clock=n_clock,
        motion_sigma_m=motion_sigma_m,
        clock_drift_sigma_m=clock_drift_sigma_m,
        fgo_iters=fgo_iters,
        sat_vel=sat_vel_arr,
        doppler=doppler_arr,
        doppler_weights=doppler_w_arr,
        sat_clock_drift=sat_clock_drift_arr,
        dt=dt_arr,
        chunk_epochs=eff_chunk_epochs,
        sys_kind=sys_kind_arr,
        doppler_huber_k=doppler_huber_k,
        imu_preint=imu_preint,
        imu_position_sigma_m=imu_position_sigma_m,
        imu_velocity_sigma_mps=imu_velocity_sigma_mps,
    )

    err_wls_2d, err_fgo_2d = [], []
    err_wls_3d, err_fgo_3d = [], []
    for t in range(n_epoch):
        if np.linalg.norm(wls_state[t, :3]) < 1e3:
            continue
        tow = float(times[t])
        err_wls_2d.append(_nearest_ref_error_2d(tow, ref_tow, ref_ecef, wls_state[t]))
        err_fgo_2d.append(_nearest_ref_error_2d(tow, ref_tow, ref_ecef, fgo_state[t]))
        err_wls_3d.append(_nearest_ref_error_3d(tow, ref_tow, ref_ecef, wls_state[t]))
        err_fgo_3d.append(_nearest_ref_error_3d(tow, ref_tow, ref_ecef, fgo_state[t]))

    if export_csv is not None:
        export_trajectory_csv(export_csv, times, fgo_state[:, :3])

    rel_name = run_dir.name
    if len(run_dir.parents) >= 2:
        rel_name = f"{run_dir.parent.name}/{run_dir.name}"

    return {
        "run": rel_name,
        "n_epoch": n_epoch,
        "max_sats": max_sats,
        "median_sats": int(np.median(sat_counts)),
        "min_sats": int(sat_counts.min()),
        "n_clock": n_clock,
        "constellations": constellations,
        "fgo_iters": iters,
        "fgo_mse_pr": float(mse_pr),
        "rms_wls_2d": float(np.sqrt(np.mean(np.square(err_wls_2d)))),
        "rms_fgo_2d": float(np.sqrt(np.mean(np.square(err_fgo_2d)))),
        "rms_wls_3d": float(np.sqrt(np.mean(np.square(err_wls_3d)))),
        "rms_fgo_3d": float(np.sqrt(np.mean(np.square(err_fgo_3d)))),
        "p95_fgo_2d": float(np.percentile(err_fgo_2d, 95)),
        "export_spp": "(native in-repo)",
        "multi_gnss": n_clock > 1,
        "native": True,
        "doppler_mode": doppler_mode,
        "doppler_gate_sigma": doppler_gate_sigma,
        "doppler_gate_stats": doppler_gate_stats,
        "doppler_huber_k": doppler_huber_k,
        "imu_enabled": imu_enabled,
        "elevation_mask_deg": elevation_mask_deg,
        "elevation_mask_stats": elevation_mask_stats,
        "constellation_weighting": constellation_weighting,
        "export_csv": str(export_csv) if export_csv else "",
        "chunk_stats": chunk_stats,
        "times": times,
        "fgo_ecef": fgo_state[:, :3].copy(),
    }


def run_fgo_on_ppc(
    run_dir: Path,
    *,
    max_epochs: int = 300,
    el_mask_deg: float = 15.0,
    motion_sigma_m: float = 0.0,
    fgo_iters: int = 8,
    export_spp: Path | None = None,
    use_doppler: bool = False,
    doppler_mode: str = "off",
    multi_gnss: bool = False,
    use_vd: bool = False,
    clock_drift_sigma_m: float = 1.0,
) -> dict:
    """Run RTKLIB-aligned FGO on a PPC run and return results."""
    obs_p = run_dir / "rover.obs"
    nav_p = run_dir / "base.nav"
    ref_p = run_dir / "reference.csv"
    for need in (obs_p, nav_p, ref_p):
        if not need.is_file():
            raise FileNotFoundError(f"Missing: {need}")

    ref_tow, ref_ecef = _load_ppc_reference(ref_p)

    # Parse RINEX for epoch/satellite structure
    rinex = read_rinex_obs(obs_p)

    # Get RTKLIB measurements
    rtk_meas = None
    if export_spp is not None:
        rtk_meas = _run_rtklib_export(export_spp, obs_p, nav_p, el_mask_deg)

    # Determine which satellite system prefixes to accept
    _accept_prefixes = ("G",)
    if multi_gnss and rtk_meas is not None:
        _accept_prefixes = ("G", "E", "J")

    # Build epoch list from RINEX (also extract Doppler D1C if needed)
    epochs_data: list[tuple[float, list[str], np.ndarray, int]] = []
    doppler_data: list[dict[str, float]] = []  # per-epoch: sat_id -> D1C
    max_sats = 0
    for ep in rinex.epochs:
        pr_map: dict[str, float] = {}
        dop_map: dict[str, float] = {}
        for sat, obs in ep.observations.items():
            if not any(sat.startswith(p) for p in _accept_prefixes):
                continue
            if "C1C" in obs and obs["C1C"] and obs["C1C"] != 0.0:
                pr_map[sat] = obs["C1C"]
                d1c = obs.get("D1C", 0.0)
                if d1c and d1c != 0.0:
                    dop_map[sat] = float(d1c)
        if len(pr_map) < 4:
            continue
        tow = _datetime_to_gps_seconds_of_week(ep.time)
        wk = _datetime_to_gps_week(ep.time)
        sats = sorted(pr_map.keys())
        pr = np.array([pr_map[s] for s in sats], dtype=np.float64)
        epochs_data.append((tow, sats, pr, wk))
        doppler_data.append(dop_map)
        max_sats = max(max_sats, len(sats))
        if max_epochs > 0 and len(epochs_data) >= max_epochs:
            break

    n_epoch = len(epochs_data)
    if n_epoch < 5:
        raise RuntimeError(f"Only {n_epoch} valid epochs")

    # Build padded arrays
    sat_ecef = np.zeros((n_epoch, max_sats, 3), dtype=np.float64)
    pseudorange = np.zeros((n_epoch, max_sats), dtype=np.float64)
    weights = np.zeros((n_epoch, max_sats), dtype=np.float64)

    # Build sys_kind for multi-GNSS ISB
    has_multi = multi_gnss and rtk_meas is not None
    n_clock = 3 if has_multi else 1
    sys_kind_arr: np.ndarray | None = None
    if has_multi:
        sys_kind_arr = np.zeros((n_epoch, max_sats), dtype=np.int32)
        for t_sk, (_tow_sk, sats_sk, _pr_sk, _wk_sk) in enumerate(epochs_data):
            for si_sk, sid_sk in enumerate(sats_sk):
                prefix = sid_sk[0] if sid_sk else "G"
                sys_kind_arr[t_sk, si_sk] = SYS_ID_TO_KIND.get(prefix, 0)

    approx0 = rinex.header.approx_position.copy()

    for t, (tow, sats, pr_raw, wk) in enumerate(epochs_data):
        ns = len(sats)
        rx_est = approx0.astype(np.float64, copy=True)

        for _pass in range(2):
            pr_tmp = np.zeros(ns, dtype=np.float64)
            w_tmp = np.zeros(ns, dtype=np.float64)
            sat_buf = np.zeros((ns, 3), dtype=np.float64)

            for si, sid in enumerate(sats):
                if rtk_meas is not None:
                    row = rtk_meas.get((wk, round(float(tow), 4), sid))
                    if row is None:
                        continue
                    sat_buf[si, 0] = row["satx"]
                    sat_buf[si, 1] = row["saty"]
                    sat_buf[si, 2] = row["satz"]
                    pr_clean = (
                        row["prange_m"]
                        - row["iono_m"]
                        - row["trop_m"]
                        - row["sat_clk_m"]
                    )
                    pr_tmp[si] = pr_clean
                    if "el_rad" in row:
                        sin_el = max(math.sin(row["el_rad"]), 0.1)
                        w_tmp[si] = sin_el * sin_el
                    else:
                        w_tmp[si] = 0.5

            idx = np.flatnonzero(w_tmp > 0)
            if idx.size >= 4:
                st, _ = wls_position(
                    sat_buf[idx, :].reshape(-1), pr_tmp[idx], w_tmp[idx], 25, 1e-9,
                )
                rx_est = np.asarray(st[:3], dtype=np.float64).copy()

        sat_ecef[t, :ns] = sat_buf
        pseudorange[t, :ns] = pr_tmp
        weights[t, :ns] = w_tmp

    # WLS (single-clock for position seed; multi-clock handled by FGO)
    wls_state = np.zeros((n_epoch, 4), dtype=np.float64)
    for t2 in range(n_epoch):
        w = weights[t2]
        idx = np.flatnonzero(w > 0)
        if idx.size < 4:
            continue
        st, _ = wls_position(
            sat_ecef[t2, idx].reshape(-1), pseudorange[t2, idx], w[idx], 25, 1e-9,
        )
        wls_state[t2] = st

    # Compute Doppler-derived motion displacement from RTKLIB receiver velocity
    # RTKLIB pntpos estvel() uses Sagnac-corrected Doppler → accurate ECEF velocity
    motion_disp = None
    use_rtklib_doppler = use_doppler or doppler_mode == "rtklib"
    if use_rtklib_doppler and motion_sigma_m > 0 and rtk_meas is not None:
        motion_disp = np.zeros((n_epoch, 3), dtype=np.float64)
        for t_d in range(n_epoch - 1):
            tow_d = epochs_data[t_d][0]
            wk_d = epochs_data[t_d][3]
            sats_d = epochs_data[t_d][1]
            # Get receiver velocity from any satellite row at this epoch
            rx_vel = None
            for sid in sats_d:
                row = rtk_meas.get((wk_d, round(float(tow_d), 4), sid))
                if row is not None and "rx_vx" in row:
                    rx_vel = np.array([row["rx_vx"], row["rx_vy"], row["rx_vz"]])
                    break
            if rx_vel is None:
                continue
            dt_ep = epochs_data[t_d + 1][0] - epochs_data[t_d][0]
            if 0 < dt_ep < 10:
                motion_disp[t_d] = rx_vel * dt_ep

    # FGO — choose between standard and VD solver
    if use_vd:
        # --- Velocity-Doppler (VD) solver ---
        # State: [x,y,z, vx,vy,vz, c0,...,c_{K-1}, drift] -> 7 + n_clock columns
        fgo_state = np.zeros((n_epoch, 7 + n_clock), dtype=np.float64)
        fgo_state[:, :3] = wls_state[:, :3]  # position from WLS
        fgo_state[:, 6] = wls_state[:, 3]    # GPS clock bias

        # Store RTKLIB receiver velocity and initialize velocity state
        rx_vel_per_epoch = np.zeros((n_epoch, 3), dtype=np.float64)
        if rtk_meas is not None:
            for t_v in range(n_epoch):
                tow_v = epochs_data[t_v][0]
                wk_v = epochs_data[t_v][3]
                sats_v = epochs_data[t_v][1]
                for sid in sats_v:
                    row = rtk_meas.get((wk_v, round(float(tow_v), 4), sid))
                    if row is not None and "rx_vx" in row:
                        rv = np.array([row["rx_vx"], row["rx_vy"], row["rx_vz"]])
                        fgo_state[t_v, 3] = rv[0]
                        fgo_state[t_v, 4] = rv[1]
                        fgo_state[t_v, 5] = rv[2]
                        rx_vel_per_epoch[t_v] = rv
                        break

        # Build dt array (inter-epoch time differences)
        dt_arr = np.zeros(n_epoch, dtype=np.float64)
        for t_dt in range(n_epoch - 1):
            dt_arr[t_dt] = epochs_data[t_dt + 1][0] - epochs_data[t_dt][0]
            if dt_arr[t_dt] <= 0 or dt_arr[t_dt] > 30:
                dt_arr[t_dt] = 1.0  # fallback

        # Build satellite velocity and Doppler pseudorange-rate arrays
        sat_vel_arr = np.zeros((n_epoch, max_sats, 3), dtype=np.float64)
        doppler_arr = np.zeros((n_epoch, max_sats), dtype=np.float64)
        doppler_w_arr = np.zeros((n_epoch, max_sats), dtype=np.float64)

        if rtk_meas is not None:
            for t_d in range(n_epoch):
                tow_d = epochs_data[t_d][0]
                wk_d = epochs_data[t_d][3]
                sats_d = epochs_data[t_d][1]
                # Get receiver velocity for this epoch
                rx_vel = np.zeros(3)
                for sid in sats_d:
                    row = rtk_meas.get((wk_d, round(float(tow_d), 4), sid))
                    if row is not None and "rx_vx" in row:
                        rx_vel = np.array([row["rx_vx"], row["rx_vy"], row["rx_vz"]])
                        break

                for si, sid in enumerate(sats_d):
                    row = rtk_meas.get((wk_d, round(float(tow_d), 4), sid))
                    if row is None or "svx" not in row:
                        continue
                    sv = np.array([row["svx"], row["svy"], row["svz"]])
                    sat_vel_arr[t_d, si] = sv
                    # Compute pseudorange-rate (Doppler range rate):
                    # Kernel convention: e = (rx - sat) / r (sat-to-rx)
                    # pred = dot(e, sv - rv) + drift
                    # So doppler_obs should be dot((rx-sat)/r, sv-rv)
                    sat_pos = sat_ecef[t_d, si]
                    rx_pos = fgo_state[t_d, :3]
                    diff = rx_pos - sat_pos  # rx - sat (sat-to-rx direction)
                    rng = np.linalg.norm(diff)
                    if rng < 1e3:
                        continue
                    unit_vec = diff / rng
                    range_rate = np.dot(sv - rx_vel, unit_vec)
                    doppler_arr[t_d, si] = range_rate
                    # Weight same as pseudorange (sin^2 elevation)
                    doppler_w_arr[t_d, si] = weights[t_d, si]

        iters, mse_pr = fgo_gnss_lm_vd(
            sat_ecef, pseudorange, weights, fgo_state,
            sys_kind=sys_kind_arr,
            n_clock=n_clock,
            motion_sigma_m=motion_sigma_m,
            clock_drift_sigma_m=clock_drift_sigma_m,
            max_iter=fgo_iters, tol=1e-7,
            sat_vel=sat_vel_arr,
            doppler=doppler_arr,
            doppler_weights=doppler_w_arr,
            dt=dt_arr,
        )
    else:
        # --- Standard solver ---
        fgo_state = np.zeros((n_epoch, 3 + n_clock), dtype=np.float64)
        fgo_state[:, :4] = wls_state  # xyz + GPS clock from WLS
        iters, mse_pr = fgo_gnss_lm(
            sat_ecef, pseudorange, weights, fgo_state,
            sys_kind=sys_kind_arr,
            n_clock=n_clock,
            motion_sigma_m=motion_sigma_m, max_iter=fgo_iters, tol=1e-7,
            motion_displacement=motion_disp,
        )

    # Compute errors (skip epochs where WLS failed to converge)
    err_wls_2d, err_fgo_2d = [], []
    err_wls_3d, err_fgo_3d = [], []
    for t3 in range(n_epoch):
        if np.linalg.norm(wls_state[t3, :3]) < 1e3:
            continue  # WLS didn't converge
        tow = epochs_data[t3][0]
        err_wls_2d.append(_nearest_ref_error_2d(tow, ref_tow, ref_ecef, wls_state[t3]))
        err_fgo_2d.append(_nearest_ref_error_2d(tow, ref_tow, ref_ecef, fgo_state[t3]))
        err_wls_3d.append(_nearest_ref_error_3d(tow, ref_tow, ref_ecef, wls_state[t3]))
        err_fgo_3d.append(_nearest_ref_error_3d(tow, ref_tow, ref_ecef, fgo_state[t3]))

    rms_wls_2d = float(np.sqrt(np.mean(np.square(err_wls_2d))))
    rms_fgo_2d = float(np.sqrt(np.mean(np.square(err_fgo_2d))))
    rms_wls_3d = float(np.sqrt(np.mean(np.square(err_wls_3d))))
    rms_fgo_3d = float(np.sqrt(np.mean(np.square(err_fgo_3d))))
    p95_fgo_2d = float(np.percentile(err_fgo_2d, 95))

    return {
        "run": str(run_dir.relative_to(run_dir.parents[2])) if run_dir.parents[2].exists() else run_dir.name,
        "n_epoch": n_epoch,
        "max_sats": max_sats,
        "n_clock": n_clock,
        "fgo_iters": iters,
        "rms_wls_2d": rms_wls_2d,
        "rms_fgo_2d": rms_fgo_2d,
        "rms_wls_3d": rms_wls_3d,
        "rms_fgo_3d": rms_fgo_3d,
        "p95_fgo_2d": p95_fgo_2d,
        "export_spp": str(export_spp) if export_spp else "(off)",
        "multi_gnss": multi_gnss,
        "native": False,
        "doppler_mode": "rtklib" if use_rtklib_doppler else "off",
        "fgo_mse_pr": float(mse_pr),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ppc-root", type=Path, default=None, help="PPC-Dataset root dir")
    p.add_argument("--run", type=str, default="tokyo/run1", help="Run subdir (e.g. tokyo/run1)")
    p.add_argument("--all", action="store_true", help="Run all 6 PPC runs")
    p.add_argument("--max-epochs", type=int, default=300,
                   help="Cap usable epochs (0 = all)")
    p.add_argument("--start-epoch", type=int, default=0,
                   help="[--no-rtklib only] Skip this many usable epochs before starting "
                        "(for isolating a specific chunk, e.g. --start-epoch 5000 --max-epochs 1000)")
    p.add_argument("--elev", type=float, default=15.0)
    p.add_argument(
        "--motion-sigma-m",
        type=float,
        default=None,
        help="Motion factor sigma in metres (native backbone default: 1.0; RTKLIB default: 0.0)",
    )
    p.add_argument("--fgo-iters", type=int, default=8)
    p.add_argument("--no-rtklib", action="store_true", help="Use in-repo PPCDatasetLoader (no RTKLIB)")
    p.add_argument(
        "--doppler",
        choices=("off", "rtklib", "in-repo"),
        default="off",
        help="Doppler factor: off | rtklib (export_spp_meas) | in-repo (PPC loader)",
    )
    p.add_argument("--vd", action="store_true",
                   help="Use fgo_gnss_lm_vd solver (velocity-Doppler; required for --no-rtklib)")
    p.add_argument("--clock-drift-sigma-m", type=float, default=1.0,
                   help="Clock drift sigma for VD solver (default: 1.0)")
    p.add_argument("--multi-gnss", action="store_true",
                   help="Use GPS+Galileo+QZSS with ISB (n_clock=3) [RTKLIB path only]")
    p.add_argument(
        "--export-csv",
        type=Path,
        default=None,
        help="Write estimated trajectory CSV (tow, lat/lon/height, ECEF, fix)",
    )
    p.add_argument(
        "--systems",
        type=str,
        default="G",
        help="[--no-rtklib only] Constellation prefixes to load, e.g. 'G' or 'GRECJ' "
             "(D2 multi-GNSS audit; each letter maps to G=GPS,R=GLONASS,E=Galileo,"
             "C=BeiDou,J=QZSS). n_clock = number of systems actually observed.",
    )
    p.add_argument(
        "--doppler-gate-sigma",
        type=float,
        default=0.0,
        help="[--no-rtklib --doppler in-repo only] D1 robust per-epoch Doppler outlier "
             "gate in units of robust-sigma (median/MAD); 0 disables (default: 0).",
    )
    p.add_argument(
        "--doppler-huber-k",
        type=float,
        default=0.0,
        help="[--no-rtklib --doppler in-repo only] Native Huber kernel threshold (m/s, "
             "Mahalanobis) for the Doppler factor; 0 keeps pure L2 (default: 0).",
    )
    p.add_argument(
        "--imu",
        action="store_true",
        help="[--no-rtklib only] D3: wire PPC imu.csv preintegrated position/velocity "
             "deltas into the VD solver between GNSS epochs.",
    )
    p.add_argument("--imu-position-sigma-m", type=float, default=0.0,
                   help="IMU delta-position prior sigma in metres (0 disables the prior)")
    p.add_argument("--imu-velocity-sigma-mps", type=float, default=0.0,
                   help="IMU delta-velocity prior sigma in m/s (0 disables the prior)")
    p.add_argument(
        "--chunk-epochs",
        type=int,
        default=0,
        help="[--no-rtklib only] Override the per-chunk epoch budget (0 = auto). "
             "The dense per-chunk LM solve cost grows roughly as O(n_state^3); "
             "wide multi-clock states (--systems with >1 constellation, D2) get "
             "much cheaper with a smaller chunk (e.g. 250) at the cost of "
             "slightly weaker inter-chunk coupling.",
    )
    p.add_argument(
        "--elevation-mask-deg",
        type=float,
        default=0.0,
        help="[--no-rtklib only] WP3c work item 1: drop satellites below this "
             "elevation (computed from sat_ecef and a pass-1 per-epoch WLS "
             "position) before the pass-2 WLS/FGO solve; 0 disables (default).",
    )
    p.add_argument(
        "--constellation-weighting",
        action="store_true",
        help="[--no-rtklib only] WP3c work item 2b: rescale pseudorange weights "
             "per-constellation (default: BeiDou 1.5x sigma, GLONASS 2x sigma, "
             "GPS/Galileo/QZSS unchanged; see DEFAULT_CONSTELLATION_SIGMA_SCALE). "
             "Individual --pr-sigma-scale-* flags override the per-system default.",
    )
    for _sys_char, _sys_name in (("g", "GPS"), ("r", "GLONASS"), ("e", "Galileo"), ("c", "BeiDou"), ("j", "QZSS")):
        p.add_argument(
            f"--pr-sigma-scale-{_sys_char}",
            type=float,
            default=None,
            help=f"[--constellation-weighting only] {_sys_name} pseudorange sigma scale "
                 f"(1.0 = no change); overrides DEFAULT_CONSTELLATION_SIGMA_SCALE['{_sys_char.upper()}'].",
        )
    args = p.parse_args()

    ppc_root = _resolve_ppc_root(args.ppc_root)
    if not ppc_root.is_dir():
        print(f"PPC-Dataset not found: {ppc_root}")
        print("Download from: https://github.com/taroz/PPC-Dataset")
        sys.exit(1)

    export_spp = None if args.no_rtklib else _default_export_spp_meas()
    if args.doppler == "in-repo" and not args.no_rtklib:
        print("ERROR: --doppler in-repo requires --no-rtklib")
        sys.exit(1)
    if args.no_rtklib and not args.vd:
        print("ERROR: --no-rtklib native backbone requires --vd")
        sys.exit(1)
    if args.doppler == "rtklib" and export_spp is None:
        print("ERROR: --doppler rtklib requires RTKLIB export_spp_meas (omit --no-rtklib)")
        sys.exit(1)

    motion_sigma_m = args.motion_sigma_m
    if motion_sigma_m is None:
        motion_sigma_m = 1.0 if args.no_rtklib else 0.0

    runs = ALL_RUNS if args.all else [args.run]

    results = []
    for run_name in runs:
        run_dir = ppc_root / run_name
        if not run_dir.is_dir():
            print(f"  SKIP {run_name}: not found")
            continue
        try:
            if args.no_rtklib:
                r = run_fgo_on_ppc_native(
                    run_dir,
                    max_epochs=args.max_epochs,
                    start_epoch=args.start_epoch,
                    motion_sigma_m=motion_sigma_m,
                    fgo_iters=args.fgo_iters,
                    clock_drift_sigma_m=args.clock_drift_sigma_m,
                    doppler_mode=args.doppler,
                    export_csv=args.export_csv,
                    systems=tuple(args.systems),
                    doppler_gate_sigma=args.doppler_gate_sigma,
                    doppler_huber_k=args.doppler_huber_k,
                    imu_enabled=args.imu,
                    imu_position_sigma_m=args.imu_position_sigma_m,
                    imu_velocity_sigma_mps=args.imu_velocity_sigma_mps,
                    chunk_epochs=args.chunk_epochs,
                    elevation_mask_deg=args.elevation_mask_deg,
                    constellation_weighting=args.constellation_weighting,
                    constellation_sigma_scale={
                        sys_char.upper(): getattr(args, f"pr_sigma_scale_{sys_char}")
                        for sys_char in ("g", "r", "e", "c", "j")
                        if getattr(args, f"pr_sigma_scale_{sys_char}") is not None
                    }
                    or None,
                )
            else:
                r = run_fgo_on_ppc(
                    run_dir,
                    max_epochs=args.max_epochs,
                    el_mask_deg=args.elev,
                    motion_sigma_m=motion_sigma_m,
                    fgo_iters=args.fgo_iters,
                    export_spp=export_spp,
                    use_doppler=(args.doppler == "rtklib"),
                    doppler_mode=args.doppler,
                    multi_gnss=args.multi_gnss,
                    use_vd=args.vd,
                    clock_drift_sigma_m=args.clock_drift_sigma_m,
                )
            results.append(r)
            gnss_tag = f"clk={r['n_clock']}" if r.get("multi_gnss") else ""
            dop_tag = f"dop={r.get('doppler_mode', 'off')}"
            med_sats = r.get("median_sats")
            sats_tag = f"sats={r['max_sats']:2d}" if med_sats is None else f"sats={r['max_sats']:2d}(med={med_sats})"
            print(
                f"  {r['run']:20s}  epochs={r['n_epoch']:4d}  {sats_tag}  "
                f"WLS 2D={r['rms_wls_2d']:7.2f}m  FGO 2D={r['rms_fgo_2d']:7.2f}m  "
                f"P95={r['p95_fgo_2d']:7.2f}m  3D={r['rms_fgo_3d']:7.2f}m  "
                f"iters={r['fgo_iters']} mse={r.get('fgo_mse_pr', float('nan')):.4g}  "
                f"{gnss_tag} {dop_tag}"
            )
            gate_stats = r.get("doppler_gate_stats")
            if gate_stats:
                print(
                    f"    doppler gate: {gate_stats['n_obs_gated']}/{gate_stats['n_obs_total']} "
                    f"obs gated across {gate_stats['n_epochs_gated']} epochs",
                    flush=True,
                )
            elev_stats = r.get("elevation_mask_stats")
            if elev_stats:
                print(
                    f"    elevation mask ({elev_stats['min_elevation_deg']:.0f} deg): "
                    f"{elev_stats['n_obs_masked']}/{elev_stats['n_obs_total']} obs masked",
                    flush=True,
                )
            for chunk in r.get("chunk_stats", []):
                print(
                    f"    chunk {chunk['chunk']}: [{chunk['start']}:{chunk['end']}] "
                    f"n_state={chunk['n_state']} iters={chunk['iters']} "
                    f"mse={float(chunk['mse_pr']):.4g} status={chunk['status']}",
                    flush=True,
                )
        except Exception as e:
            print(f"  {run_name}: ERROR {e}")

    if len(results) > 1:
        print()
        print("Summary:")
        all_wls = [r["rms_wls_2d"] for r in results]
        all_fgo = [r["rms_fgo_2d"] for r in results]
        print(f"  Avg WLS 2D: {np.mean(all_wls):.2f}m  Avg FGO 2D: {np.mean(all_fgo):.2f}m")


if __name__ == "__main__":
    main()
