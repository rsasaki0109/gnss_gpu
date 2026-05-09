"""Time-differenced carrier phase (TDCP) velocity estimation.

Builds inter-epoch carrier-phase range change, removes satellite orbit and
clock motion using broadcast ephemeris velocities and drifts, then solves a
WLS for receiver displacement and epoch clock drift (in meters).

Assumes L1/E1-class carrier wavelength (1575.42 MHz) per measurement row.
Suitable for GPS / Galileo E1 / QZSS L1C; GLONASS FDMA needs per-channel
wavelengths (not handled here).
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

C_LIGHT = 299792458.0
L1_FREQ = 1575.42e6
L1_WAVELENGTH = C_LIGHT / L1_FREQ


def _meas_key(m: Any) -> tuple[int, int]:
    return (int(getattr(m, "system_id")), int(getattr(m, "prn")))


def _one_row_per_satellite(measurements: Sequence[Any]) -> dict[tuple[int, int], Any]:
    """gnssplusplus can emit multiple ``CorrectedMeasurement`` rows per satellite
    (e.g. L1 and L2). Pick one row per (system, prn) only when unambiguous:

    Prefer higher SNR. If several rows tie for best SNR (within ~0.05 dB-Hz),
    omit that satellite so TDCP does not pair different frequencies across epochs.
    """
    by_key: dict[tuple[int, int], list[Any]] = {}
    for m in measurements:
        k = _meas_key(m)
        by_key.setdefault(k, []).append(m)

    out: dict[tuple[int, int], Any] = {}
    eps = 0.05
    for k, lst in by_key.items():
        if len(lst) == 1:
            out[k] = lst[0]
            continue
        lst_sorted = sorted(lst, key=lambda m: float(getattr(m, "snr", 0.0)), reverse=True)
        best_snr = float(getattr(lst_sorted[0], "snr", 0.0))
        n_top = sum(
            1 for m in lst_sorted if abs(float(getattr(m, "snr", 0.0)) - best_snr) <= eps
        )
        if n_top != 1:
            continue
        out[k] = lst_sorted[0]
    return out


def _valid_carrier(phi: float) -> bool:
    return np.isfinite(phi) and abs(float(phi)) > 1e-12


def _solve_tdcp_wls(
    receiver_position: np.ndarray,
    prev_measurements: Sequence[Any],
    cur_measurements: Sequence[Any],
    dt: float,
    wavelength: float,
    min_sats: int,
    max_cycle_jump: float,
    max_postfit_rms_m: float,
    elevation_weight: bool = False,
    el_sin_floor: float = 0.1,
) -> tuple[np.ndarray, float] | None:
    """Internal: return (velocity [m/s], unweighted postfit RMS [m]) or None."""
    if dt <= 0 or prev_measurements is None or len(prev_measurements) == 0:
        return None

    rx = np.asarray(receiver_position, dtype=np.float64).ravel()[:3]

    prev_map = _one_row_per_satellite(prev_measurements)
    cur_map = _one_row_per_satellite(cur_measurements)
    common = prev_map.keys() & cur_map.keys()
    if len(common) < min_sats:
        return None

    h_rows: list[np.ndarray] = []
    y_vals: list[float] = []
    w_vals: list[float] = []

    for key in common:
        mp, mc = prev_map[key], cur_map[key]
        lp = float(getattr(mp, "carrier_phase"))
        lc = float(getattr(mc, "carrier_phase"))
        if not (_valid_carrier(lp) and _valid_carrier(lc)):
            continue
        if abs(lc - lp) > float(max_cycle_jump):
            continue

        sp = np.asarray(getattr(mp, "satellite_ecef"), dtype=np.float64).ravel()[:3]
        sc = np.asarray(getattr(mc, "satellite_ecef"), dtype=np.float64).ravel()[:3]
        vp = np.asarray(getattr(mp, "satellite_velocity"), dtype=np.float64).ravel()[:3]
        vc = np.asarray(getattr(mc, "satellite_velocity"), dtype=np.float64).ravel()[:3]

        sat_mid = 0.5 * (sp + sc)
        dx = sat_mid - rx
        rng = float(np.linalg.norm(dx))
        if rng < 1e3:
            continue
        los = dx / rng

        d_l_m = float(lc - lp) * float(wavelength)
        v_avg = 0.5 * (vp + vc)
        sat_range_change = float(np.dot(los, v_avg) * dt)
        drift_p = float(getattr(mp, "clock_drift"))
        drift_c = float(getattr(mc, "clock_drift"))
        drift_avg = 0.5 * (drift_p + drift_c)
        sat_clock_change = drift_avg * C_LIGHT * dt
        corrected = d_l_m - sat_range_change + sat_clock_change

        wp = float(getattr(mp, "weight", 1.0))
        wc = float(getattr(mc, "weight", 1.0))
        if wp <= 0 or wc <= 0 or not (np.isfinite(wp) and np.isfinite(wc)):
            w = 1.0
        else:
            w = float(np.sqrt(wp * wc))

        if elevation_weight:
            elp = float(getattr(mp, "elevation", float("nan")))
            elc = float(getattr(mc, "elevation", float("nan")))
            if np.isfinite(elp) and np.isfinite(elc):
                el = 0.5 * (elp + elc)
                s = max(float(np.sin(el)), float(el_sin_floor))
                w *= s * s

        h_rows.append(np.concatenate([los, np.ones(1)]))
        y_vals.append(corrected)
        w_vals.append(w)

    if len(h_rows) < min_sats:
        return None

    H = np.stack(h_rows, axis=0)
    y = np.asarray(y_vals, dtype=np.float64)
    sw = np.sqrt(np.asarray(w_vals, dtype=np.float64))
    Hw = H * sw[:, np.newaxis]
    yw = y * sw

    try:
        theta, _, rank, _ = np.linalg.lstsq(Hw, yw, rcond=None)
    except np.linalg.LinAlgError:
        return None

    if rank < 4:
        return None

    pred = H @ theta
    rms_res = float(np.sqrt(np.mean((pred - y) ** 2)))
    if not np.isfinite(rms_res) or rms_res > float(max_postfit_rms_m):
        return None

    delta_rx = theta[:3]
    if not np.all(np.isfinite(delta_rx)):
        return None
    vel = delta_rx / dt
    if np.linalg.norm(vel) > 50.0:
        return None
    return vel, rms_res


def estimate_velocity_from_tdcp(
    receiver_position: np.ndarray,
    prev_measurements: Sequence[Any],
    cur_measurements: Sequence[Any],
    dt: float,
    wavelength: float = L1_WAVELENGTH,
    min_sats: int = 4,
    max_cycle_jump: float = 20000.0,
    max_postfit_rms_m: float = 12.0,
    elevation_weight: bool = False,
    el_sin_floor: float = 0.1,
) -> np.ndarray | None:
    """Estimate ECEF velocity from TDCP using satellite motion / clock corrections.

    For each satellite tracked in both epochs (same ``system_id`` and ``prn``):

    - ``delta_L_m = (L_cur - L_prev) * wavelength``
    - ``sat_range_change = los · 0.5*(v_prev + v_cur) * dt``
    - ``sat_clock_change = 0.5*(drift_prev + drift_cur) * c * dt``
    - ``corrected = delta_L_m - sat_range_change + sat_clock_change``

    Solve ``los · delta_rx + delta_cb = corrected`` (meters) via WLS, then
    ``velocity = delta_rx / dt``.

    Parameters
    ----------
    receiver_position : (3,) or (4,)
        Approximate receiver ECEF [m] for line-of-sight (typically SPP).
    prev_measurements, cur_measurements
        Sequences of objects with attributes ``system_id``, ``prn``,
        ``satellite_ecef`` (3,), ``carrier_phase`` [cycles],
        ``satellite_velocity`` (3,) [m/s], ``clock_drift`` [s/s], and
        optionally ``weight`` (combined if present).
    dt : float
        Receiver time step [s]; must be positive.
    wavelength : float
        Carrier wavelength [m]. Default: GPS L1.
    min_sats : int
        Minimum number of satellites after filtering.
    max_cycle_jump : float
        Discard pairs whose raw carrier delta exceeds this many cycles (slip guard).
        Keep large enough for ~1 s epochs (satellite motion alone is often 10^2–10^3 cycles).
    max_postfit_rms_m : float
        Reject solution if unweighted RMS of |Hθ - y| exceeds this (meters).
        Guards against L1/L2 mixups and cycle-slip contamination.
    elevation_weight : bool
        If True and ``elevation`` is set on both measurements, multiply each row
        weight by ``max(sin(el_mean), el_sin_floor)**2`` (low-elevation TDCP rows
        downweighted; urban NLOS heuristic).
    el_sin_floor : float
        Minimum ``sin(elevation)`` when ``elevation_weight`` is active.

    Returns
    -------
    velocity : (3,) ndarray or None
        ECEF velocity [m/s], or None if underdetermined or unreasonable.
    """
    r = _solve_tdcp_wls(
        receiver_position,
        prev_measurements,
        cur_measurements,
        dt,
        wavelength,
        min_sats,
        max_cycle_jump,
        max_postfit_rms_m,
        elevation_weight,
        el_sin_floor,
    )
    return None if r is None else r[0]


def estimate_velocity_from_tdcp_with_metrics(
    receiver_position: np.ndarray,
    prev_measurements: Sequence[Any],
    cur_measurements: Sequence[Any],
    dt: float,
    wavelength: float = L1_WAVELENGTH,
    min_sats: int = 4,
    max_cycle_jump: float = 20000.0,
    max_postfit_rms_m: float = 12.0,
    elevation_weight: bool = False,
    el_sin_floor: float = 0.1,
) -> tuple[np.ndarray | None, float]:
    """Same as ``estimate_velocity_from_tdcp`` but also returns postfit RMS (meters).

    When TDCP is rejected, returns ``(None, nan)``.
    """
    r = _solve_tdcp_wls(
        receiver_position,
        prev_measurements,
        cur_measurements,
        dt,
        wavelength,
        min_sats,
        max_cycle_jump,
        max_postfit_rms_m,
        elevation_weight,
        el_sin_floor,
    )
    if r is None:
        return None, float("nan")
    return r[0], r[1]
