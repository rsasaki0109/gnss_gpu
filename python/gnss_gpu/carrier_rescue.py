"""Carrier-anchor and carrier fallback helpers for the PF smoother experiment."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from gnss_gpu import ParticleFilterDevice
from gnss_gpu.dd_quality import metric_sigma_scale
from gnss_gpu.pf_smoother_common import finite_float as _finite_float
from gnss_gpu.tdcp_velocity import C_LIGHT as TDCP_C_LIGHT

MUPF_L1_COMPAT_SYSTEM_IDS = frozenset({0, 2, 4})  # G/E/J share the L1/E1 frequency
MUPF_L1_WAVELENGTH_M = 299792458.0 / 1575.42e6

@dataclass
class CarrierBiasState:
    bias_cycles: float
    last_tow: float
    last_expected_cycles: float
    last_carrier_phase_cycles: float
    last_pseudorange_m: float
    last_receiver_state: np.ndarray
    last_sat_ecef: np.ndarray
    last_sat_velocity: np.ndarray
    last_clock_drift: float
    stable_epochs: int = 1


@dataclass
class CarrierAnchorAttempt:
    update: dict[str, object] | None = None
    stats: dict[str, float | int | None] | None = None
    rows_used: dict[tuple[int, int], dict[str, object]] = field(default_factory=dict)
    state: np.ndarray | None = None
    used: bool = False
    propagated_rows: int = 0


@dataclass
class CarrierFallbackAttempt:
    afv: dict[str, object] | None = None
    tracked_stats: dict[str, float | int | None] | None = None
    sigma_cycles: float | None = None
    sigma_scale: float = 1.0
    used: bool = False
    attempted_tracked: bool = False
    used_tracked: bool = False
    replaced_weak_dd: bool = False


def _collect_undiff_carrier_afv_inputs(
    measurements,
    sat_ecef: np.ndarray,
    pseudoranges: np.ndarray,
    *,
    snr_min: float,
    elev_min: float,
    spp_pos_check: np.ndarray,
    allowed_system_ids: set[int] | frozenset[int] | None = None,
):
    """Collect one carrier-phase row per satellite for undifferenced AFV."""

    by_key: dict[tuple[int, int], tuple[int, object]] = {}
    for idx, m in enumerate(measurements):
        key = (int(getattr(m, "system_id", 0)), int(getattr(m, "prn", 0)))
        prev = by_key.get(key)
        if prev is None or float(getattr(m, "snr", 0.0)) > float(getattr(prev[1], "snr", 0.0)):
            by_key[key] = (idx, m)

    cb_est_m = None
    if np.isfinite(spp_pos_check).all() and np.linalg.norm(spp_pos_check) > 1e6:
        cb_est_m = float(np.median(pseudoranges - np.linalg.norm(sat_ecef - spp_pos_check, axis=1)))

    cp_cycles = []
    cp_sat_ecef = []
    cp_weights = []
    for (system_id, _prn), (_idx, m) in sorted(by_key.items()):
        if allowed_system_ids is not None and int(system_id) not in allowed_system_ids:
            continue
        cp = float(getattr(m, "carrier_phase", 0.0))
        if cp == 0.0 or not np.isfinite(cp) or abs(cp) < 1e3:
            continue
        snr = float(getattr(m, "snr", 0.0))
        elev = float(getattr(m, "elevation", 0.0))
        if snr < snr_min and snr > 0.0:
            continue
        if 0.0 < elev < elev_min:
            continue
        sat_pos = np.asarray(m.satellite_ecef, dtype=np.float64).ravel()[:3]
        if cb_est_m is not None:
            rng = np.linalg.norm(sat_pos - spp_pos_check)
            res = abs(float(getattr(m, "corrected_pseudorange", 0.0)) - rng - cb_est_m)
            if res > 30.0:
                continue
        cp_cycles.append(cp)
        cp_sat_ecef.append(sat_pos)
        cp_weights.append(float(getattr(m, "weight", 1.0)))

    if not cp_cycles:
        return None
    return {
        "sat_ecef": np.asarray(cp_sat_ecef, dtype=np.float64),
        "carrier_phase_cycles": np.asarray(cp_cycles, dtype=np.float64),
        "weights": np.asarray(cp_weights, dtype=np.float64),
        "n_sat": int(len(cp_cycles)),
    }


def _tracked_carrier_row_support(
    tracker: dict[tuple[int, int], CarrierBiasState],
    key: tuple[int, int],
    row: dict[str, object],
    receiver_state: np.ndarray,
    tow: float,
    *,
    max_age_s: float,
    max_continuity_residual_m: float,
    min_stable_epochs: int,
):
    """Return tracked continuity metadata for a carrier row, or ``None`` if unsupported."""

    state = tracker.get(key)
    if state is None:
        return None
    age_s = float(tow) - float(state.last_tow)
    if age_s < 0.0 or age_s > float(max_age_s):
        return None
    if int(state.stable_epochs) < int(min_stable_epochs):
        return None
    sat_ecef_row = np.asarray(row["sat_ecef"], dtype=np.float64)
    if sat_ecef_row.shape != (3,) or not np.isfinite(sat_ecef_row).all():
        return None
    carrier_phase_cycles_row = float(row["carrier_phase_cycles"])
    wavelength_m = float(row["wavelength_m"])
    base_weight = float(row["weight"])
    if (
        not np.isfinite(carrier_phase_cycles_row)
        or not np.isfinite(wavelength_m)
        or wavelength_m <= 0.0
        or not np.isfinite(base_weight)
        or base_weight <= 0.0
    ):
        return None
    expected_cycles = (
        np.linalg.norm(sat_ecef_row - receiver_state[:3])
        + float(receiver_state[3])
    ) / wavelength_m
    if not np.isfinite(expected_cycles):
        return None
    tdcp_pseudorange_m = _carrier_tdcp_predicted_pseudorange_m(
        state,
        row,
        receiver_state,
        tow,
    )
    bias_based_pseudorange_m = wavelength_m * (
        carrier_phase_cycles_row - float(state.bias_cycles)
    )
    if not np.isfinite(bias_based_pseudorange_m):
        return None
    continuity_residual_m = (
        abs(tdcp_pseudorange_m - bias_based_pseudorange_m)
        if tdcp_pseudorange_m is not None
        else _carrier_bias_continuity_residual_m(
            state,
            carrier_phase_cycles_row,
            expected_cycles,
            wavelength_m,
        )
    )
    if not np.isfinite(continuity_residual_m):
        return None
    if continuity_residual_m > float(max_continuity_residual_m):
        return None
    stable_scale = min(float(state.stable_epochs) / max(float(min_stable_epochs), 1.0), 2.0)
    continuity_scale = 1.0 / (1.0 + (float(continuity_residual_m) / max(float(max_continuity_residual_m), 1e-6)) ** 2)
    weight_scale = stable_scale * continuity_scale
    if not np.isfinite(weight_scale) or weight_scale <= 0.0:
        return None
    return {
        "continuity_residual_m": float(continuity_residual_m),
        "stable_epochs": int(state.stable_epochs),
        "weight_scale": float(weight_scale),
    }


def _collect_tracked_undiff_carrier_afv_inputs(
    tracker: dict[tuple[int, int], CarrierBiasState],
    carrier_rows: dict[tuple[int, int], dict[str, object]],
    receiver_state: np.ndarray,
    tow: float,
    *,
    max_age_s: float,
    max_continuity_residual_m: float,
    min_stable_epochs: int,
    min_sats: int,
) -> tuple[dict[str, np.ndarray] | None, dict[str, float | int | None]]:
    """Collect undiff carrier AFV rows restricted to tracker-consistent satellites."""

    sat_ecef = []
    carrier_phase_cycles = []
    weights = []
    continuity_residuals_m = []
    stable_epochs_used = []

    for key, row in sorted(carrier_rows.items()):
        support = _tracked_carrier_row_support(
            tracker,
            key,
            row,
            receiver_state,
            tow,
            max_age_s=max_age_s,
            max_continuity_residual_m=max_continuity_residual_m,
            min_stable_epochs=min_stable_epochs,
        )
        if support is None:
            continue
        sat_ecef.append(np.asarray(row["sat_ecef"], dtype=np.float64))
        carrier_phase_cycles.append(float(row["carrier_phase_cycles"]))
        weights.append(float(row["weight"]) * float(support["weight_scale"]))
        continuity_residuals_m.append(float(support["continuity_residual_m"]))
        stable_epochs_used.append(int(support["stable_epochs"]))

    continuity_arr = np.asarray(continuity_residuals_m, dtype=np.float64)
    stable_arr = np.asarray(stable_epochs_used, dtype=np.float64)
    stats = {
        "n_sat": int(len(carrier_phase_cycles)),
        "continuity_median_m": float(np.median(continuity_arr)) if continuity_arr.size > 0 else None,
        "continuity_max_m": float(np.max(continuity_arr)) if continuity_arr.size > 0 else None,
        "stable_epochs_median": float(np.median(stable_arr)) if stable_arr.size > 0 else None,
    }
    if len(carrier_phase_cycles) < int(min_sats):
        return None, stats
    return {
        "sat_ecef": np.asarray(sat_ecef, dtype=np.float64),
        "carrier_phase_cycles": np.asarray(carrier_phase_cycles, dtype=np.float64),
        "weights": np.asarray(weights, dtype=np.float64),
        "n_sat": int(len(carrier_phase_cycles)),
    }, stats


def _collect_hybrid_tracked_undiff_carrier_afv_inputs(
    tracker: dict[tuple[int, int], CarrierBiasState],
    carrier_rows: dict[tuple[int, int], dict[str, object]],
    receiver_state: np.ndarray,
    tow: float,
    *,
    max_age_s: float,
    max_continuity_residual_m: float,
    min_stable_epochs: int,
    min_sats: int,
) -> tuple[dict[str, np.ndarray] | None, dict[str, float | int | None]]:
    """Collect same-band undiff AFV rows while upweighting tracker-consistent satellites."""

    sat_ecef = []
    carrier_phase_cycles = []
    weights = []
    continuity_residuals_m = []
    stable_epochs_used = []
    n_tracked_consistent_sat = 0

    for key, row in sorted(carrier_rows.items()):
        sat_ecef_row = np.asarray(row["sat_ecef"], dtype=np.float64)
        carrier_phase_cycles_row = float(row["carrier_phase_cycles"])
        base_weight = float(row["weight"])
        if (
            sat_ecef_row.shape != (3,)
            or not np.isfinite(sat_ecef_row).all()
            or not np.isfinite(carrier_phase_cycles_row)
            or not np.isfinite(base_weight)
            or base_weight <= 0.0
        ):
            continue
        weight_scale = 1.0
        support = _tracked_carrier_row_support(
            tracker,
            key,
            row,
            receiver_state,
            tow,
            max_age_s=max_age_s,
            max_continuity_residual_m=max_continuity_residual_m,
            min_stable_epochs=min_stable_epochs,
        )
        if support is not None:
            raw_weight_scale = float(support["weight_scale"])
            weight_scale = 1.0 + 0.25 * (raw_weight_scale - 1.0)
            weight_scale = float(np.clip(weight_scale, 0.75, 1.25))
            continuity_residuals_m.append(float(support["continuity_residual_m"]))
            stable_epochs_used.append(int(support["stable_epochs"]))
            n_tracked_consistent_sat += 1
        weight = base_weight * weight_scale
        if not np.isfinite(weight) or weight <= 0.0:
            continue
        sat_ecef.append(sat_ecef_row)
        carrier_phase_cycles.append(carrier_phase_cycles_row)
        weights.append(weight)

    continuity_arr = np.asarray(continuity_residuals_m, dtype=np.float64)
    stable_arr = np.asarray(stable_epochs_used, dtype=np.float64)
    stats = {
        "n_sat": int(len(carrier_phase_cycles)),
        "n_tracked_consistent_sat": int(n_tracked_consistent_sat),
        "continuity_median_m": float(np.median(continuity_arr)) if continuity_arr.size > 0 else None,
        "continuity_max_m": float(np.max(continuity_arr)) if continuity_arr.size > 0 else None,
        "stable_epochs_median": float(np.median(stable_arr)) if stable_arr.size > 0 else None,
    }
    if len(carrier_phase_cycles) < int(min_sats):
        return None, stats
    return {
        "sat_ecef": np.asarray(sat_ecef, dtype=np.float64),
        "carrier_phase_cycles": np.asarray(carrier_phase_cycles, dtype=np.float64),
        "weights": np.asarray(weights, dtype=np.float64),
        "n_sat": int(len(carrier_phase_cycles)),
    }, stats


def _select_same_band_carrier_rows(
    measurements,
    pseudoranges: np.ndarray,
    *,
    snr_min: float,
    elev_min: float,
    spp_pos_check: np.ndarray,
    wavelength_m: float,
    allowed_system_ids: set[int] | frozenset[int] | None = None,
) -> dict[tuple[int, int], dict[str, object]]:
    """Pick one compatible carrier row per satellite for carrier-bias reuse."""

    by_key: dict[tuple[int, int], object] = {}
    for m in measurements:
        key = (int(getattr(m, "system_id", 0)), int(getattr(m, "prn", 0)))
        prev = by_key.get(key)
        if prev is None or float(getattr(m, "snr", 0.0)) > float(getattr(prev, "snr", 0.0)):
            by_key[key] = m

    sat_ecef = np.asarray([m.satellite_ecef for m in measurements], dtype=np.float64)
    cb_est_m = None
    if np.isfinite(spp_pos_check).all() and np.linalg.norm(spp_pos_check) > 1e6:
        cb_est_m = float(np.median(pseudoranges - np.linalg.norm(sat_ecef - spp_pos_check, axis=1)))

    out: dict[tuple[int, int], dict[str, object]] = {}
    for key, m in sorted(by_key.items()):
        system_id = int(getattr(m, "system_id", 0))
        if allowed_system_ids is not None and system_id not in allowed_system_ids:
            continue
        cp = float(getattr(m, "carrier_phase", 0.0))
        if cp == 0.0 or not np.isfinite(cp) or abs(cp) < 1e3:
            continue
        snr = float(getattr(m, "snr", 0.0))
        elev = float(getattr(m, "elevation", 0.0))
        if snr < snr_min and snr > 0.0:
            continue
        if 0.0 < elev < elev_min:
            continue
        sat_pos = np.asarray(m.satellite_ecef, dtype=np.float64).ravel()[:3]
        if cb_est_m is not None:
            rng = np.linalg.norm(sat_pos - spp_pos_check)
            res = abs(float(getattr(m, "corrected_pseudorange", 0.0)) - rng - cb_est_m)
            if res > 30.0:
                continue
        out[key] = {
            "system_id": system_id,
            "prn": int(getattr(m, "prn", 0)),
            "sat_ecef": sat_pos,
            "sat_velocity": np.asarray(
                getattr(m, "satellite_velocity", np.zeros(3, dtype=np.float64)),
                dtype=np.float64,
            ).ravel()[:3],
            "clock_drift": float(getattr(m, "clock_drift", 0.0)),
            "carrier_phase_cycles": cp,
            "weight": float(getattr(m, "weight", 1.0)),
            "wavelength_m": float(wavelength_m),
            "measurement": m,
        }
    return out


def _carrier_bias_from_state(
    carrier_phase_cycles: float,
    sat_ecef: np.ndarray,
    receiver_state: np.ndarray,
    wavelength_m: float,
) -> tuple[float, float]:
    """Return (bias_cycles, expected_cycles) for a carrier row at a receiver state."""

    receiver_state = np.asarray(receiver_state, dtype=np.float64).ravel()
    receiver_pos = receiver_state[:3]
    clock_bias_m = float(receiver_state[3])
    expected_cycles = (np.linalg.norm(np.asarray(sat_ecef, dtype=np.float64) - receiver_pos) + clock_bias_m) / float(wavelength_m)
    bias_cycles = float(carrier_phase_cycles) - float(expected_cycles)
    return float(bias_cycles), float(expected_cycles)


def _predict_receiver_state_for_anchor(
    current_pf_state: np.ndarray,
    prev_pf_state: np.ndarray | None,
    velocity: np.ndarray | None,
    dt: float,
) -> np.ndarray:
    """Predict receiver state for carrier-anchor continuity on weak-DD epochs."""

    state = np.asarray(current_pf_state, dtype=np.float64).copy()
    if (
        prev_pf_state is None
        or velocity is None
        or dt <= 0.0
        or not np.all(np.isfinite(prev_pf_state))
        or not np.all(np.isfinite(velocity))
    ):
        return state
    state[:3] = np.asarray(prev_pf_state[:3], dtype=np.float64) + np.asarray(velocity, dtype=np.float64) * float(dt)
    state[3] = float(current_pf_state[3])
    return state


def _carrier_bias_continuity_residual_m(
    state: CarrierBiasState,
    carrier_phase_cycles: float,
    expected_cycles: float,
    wavelength_m: float,
) -> float:
    """Carrier continuity residual between consecutive epochs in meters."""

    delta_meas_cycles = float(carrier_phase_cycles) - float(state.last_carrier_phase_cycles)
    delta_expected_cycles = float(expected_cycles) - float(state.last_expected_cycles)
    return abs((delta_meas_cycles - delta_expected_cycles) * float(wavelength_m))


def _carrier_tdcp_predicted_pseudorange_m(
    state: CarrierBiasState,
    row: dict[str, object],
    receiver_state: np.ndarray,
    tow: float,
) -> float | None:
    """Predict current pseudorange-like value from receiver/satellite motion."""

    dt = float(tow) - float(state.last_tow)
    if dt <= 0.0 or not np.isfinite(dt):
        return None

    sat_prev = np.asarray(state.last_sat_ecef, dtype=np.float64).ravel()[:3]
    sat_cur = np.asarray(row["sat_ecef"], dtype=np.float64).ravel()[:3]
    sat_vel_prev = np.asarray(state.last_sat_velocity, dtype=np.float64).ravel()[:3]
    sat_vel_cur = np.asarray(row.get("sat_velocity", np.zeros(3)), dtype=np.float64).ravel()[:3]
    rx_prev = np.asarray(state.last_receiver_state, dtype=np.float64).ravel()[:3]
    rx_cur = np.asarray(receiver_state, dtype=np.float64).ravel()[:3]
    if not (
        np.all(np.isfinite(sat_prev))
        and np.all(np.isfinite(sat_cur))
        and np.all(np.isfinite(sat_vel_prev))
        and np.all(np.isfinite(sat_vel_cur))
        and np.all(np.isfinite(state.last_receiver_state))
        and np.all(np.isfinite(receiver_state))
        and np.isfinite(state.last_pseudorange_m)
    ):
        return None

    sat_mid = 0.5 * (sat_prev + sat_cur)
    rx_mid = 0.5 * (rx_prev + rx_cur)
    dx = sat_mid - rx_mid
    rng = float(np.linalg.norm(dx))
    if rng < 1e3:
        return None
    los = dx / rng

    delta_rx = rx_cur - rx_prev
    delta_cb = float(receiver_state[3]) - float(state.last_receiver_state[3])
    sat_range_change = float(np.dot(los, 0.5 * (sat_vel_prev + sat_vel_cur)) * dt)
    sat_clock_change = (
        0.5 * (float(state.last_clock_drift) + float(row.get("clock_drift", 0.0)))
        * TDCP_C_LIGHT
        * dt
    )
    return float(
        state.last_pseudorange_m
        + float(np.dot(los, delta_rx))
        + delta_cb
        + sat_range_change
        - sat_clock_change
    )


def _update_carrier_bias_tracker(
    tracker: dict[tuple[int, int], CarrierBiasState],
    carrier_rows: dict[tuple[int, int], dict[str, object]],
    receiver_state: np.ndarray,
    tow: float,
    *,
    blend_alpha: float,
    reanchor_jump_cycles: float,
    max_age_s: float,
    trusted: bool,
    max_continuity_residual_m: float | None = None,
) -> None:
    """Refresh per-satellite carrier bias state using a trusted epoch."""

    stale_keys = [
        key for key, state in tracker.items()
        if float(tow) - float(state.last_tow) > float(max_age_s)
    ]
    for key in stale_keys:
        tracker.pop(key, None)

    for key, row in carrier_rows.items():
        bias_cycles, expected_cycles = _carrier_bias_from_state(
            row["carrier_phase_cycles"],
            row["sat_ecef"],
            receiver_state,
            row["wavelength_m"],
        )
        prev = tracker.get(key)
        next_stable_epochs = 1
        if prev is None or not np.isfinite(prev.bias_cycles):
            next_bias = bias_cycles
        else:
            continuity_residual_m = _carrier_bias_continuity_residual_m(
                prev,
                row["carrier_phase_cycles"],
                expected_cycles,
                row["wavelength_m"],
            )
            continuity_bad = (
                max_continuity_residual_m is not None
                and continuity_residual_m > float(max_continuity_residual_m)
            )
            jump_bad = abs(bias_cycles - prev.bias_cycles) > float(reanchor_jump_cycles)
            if continuity_bad or jump_bad:
                if not trusted:
                    continue
                next_bias = bias_cycles
                next_stable_epochs = 1
            else:
                alpha = float(np.clip(blend_alpha, 0.0, 1.0))
                next_bias = (1.0 - alpha) * float(prev.bias_cycles) + alpha * float(bias_cycles)
                next_stable_epochs = int(prev.stable_epochs) + 1
        tracker[key] = CarrierBiasState(
            bias_cycles=float(next_bias),
            last_tow=float(tow),
            last_expected_cycles=float(expected_cycles),
            last_carrier_phase_cycles=float(row["carrier_phase_cycles"]),
            last_pseudorange_m=float(expected_cycles * float(row["wavelength_m"])),
            last_receiver_state=np.asarray(receiver_state, dtype=np.float64).copy(),
            last_sat_ecef=np.asarray(row["sat_ecef"], dtype=np.float64).copy(),
            last_sat_velocity=np.asarray(
                row.get("sat_velocity", np.zeros(3)), dtype=np.float64
            ).copy(),
            last_clock_drift=float(row.get("clock_drift", 0.0)),
            stable_epochs=int(next_stable_epochs),
        )


def _build_carrier_anchor_pseudorange_update(
    tracker: dict[tuple[int, int], CarrierBiasState],
    carrier_rows: dict[tuple[int, int], dict[str, object]],
    receiver_state: np.ndarray,
    tow: float,
    *,
    max_age_s: float,
    max_residual_m: float,
    max_continuity_residual_m: float,
    min_stable_epochs: int,
    min_sats: int,
) -> tuple[dict[str, np.ndarray] | None, dict[str, float | int | None], dict[tuple[int, int], dict[str, object]]]:
    """Create a pseudorange-like update from carrier rows and tracked biases."""

    sat_ecef = []
    pseudoranges = []
    weights = []
    abs_residual_m = []
    continuity_residuals_m = []
    max_age_seen_s = 0.0
    accepted_rows: dict[tuple[int, int], dict[str, object]] = {}

    for key, row in carrier_rows.items():
        state = tracker.get(key)
        if state is None:
            continue
        age_s = float(tow) - float(state.last_tow)
        if age_s < 0.0 or age_s > float(max_age_s):
            continue
        if int(state.stable_epochs) < int(min_stable_epochs):
            continue
        _bias_cycles_now, _expected_cycles = _carrier_bias_from_state(
            row["carrier_phase_cycles"],
            row["sat_ecef"],
            receiver_state,
            row["wavelength_m"],
        )
        tdcp_predicted_pseudorange_m = _carrier_tdcp_predicted_pseudorange_m(
            state,
            row,
            receiver_state,
            tow,
        )
        bias_based_pseudorange_m = float(row["wavelength_m"]) * (
            float(row["carrier_phase_cycles"]) - float(state.bias_cycles)
        )
        continuity_residual_m = (
            abs(tdcp_predicted_pseudorange_m - bias_based_pseudorange_m)
            if tdcp_predicted_pseudorange_m is not None
            else _carrier_bias_continuity_residual_m(
                state,
                row["carrier_phase_cycles"],
                _expected_cycles,
                row["wavelength_m"],
            )
        )
        if continuity_residual_m > float(max_continuity_residual_m):
            continue
        anchor_pseudorange_m = (
            0.5 * (tdcp_predicted_pseudorange_m + bias_based_pseudorange_m)
            if tdcp_predicted_pseudorange_m is not None
            else bias_based_pseudorange_m
        )
        anchor_expected_m = float(
            np.linalg.norm(np.asarray(row["sat_ecef"], dtype=np.float64) - receiver_state[:3])
            + float(receiver_state[3])
        )
        anchor_geom_residual_m = abs(anchor_pseudorange_m - anchor_expected_m)
        if anchor_geom_residual_m > float(max_residual_m):
            continue
        sat_ecef.append(np.asarray(row["sat_ecef"], dtype=np.float64))
        pseudoranges.append(float(anchor_pseudorange_m))
        stable_scale = min(float(state.stable_epochs) / max(float(min_stable_epochs), 1.0), 2.0)
        weights.append(float(row["weight"]) * stable_scale)
        abs_residual_m.append(float(anchor_geom_residual_m))
        continuity_residuals_m.append(float(continuity_residual_m))
        max_age_seen_s = max(max_age_seen_s, age_s)
        accepted_rows[key] = row

    if len(pseudoranges) < int(min_sats):
        return None, {
            "n_sat": int(len(pseudoranges)),
            "residual_median_m": None,
            "residual_max_m": None,
            "continuity_median_m": None,
            "continuity_max_m": None,
            "max_age_s": None if len(pseudoranges) == 0 else float(max_age_seen_s),
            "min_stable_epochs": None if len(pseudoranges) == 0 else int(min_stable_epochs),
        }, accepted_rows
    abs_residual_arr = np.asarray(abs_residual_m, dtype=np.float64)
    continuity_residual_arr = np.asarray(continuity_residuals_m, dtype=np.float64)
    return {
        "sat_ecef": np.asarray(sat_ecef, dtype=np.float64),
        "pseudoranges": np.asarray(pseudoranges, dtype=np.float64),
        "weights": np.asarray(weights, dtype=np.float64),
        "n_sat": int(len(pseudoranges)),
    }, {
        "n_sat": int(len(pseudoranges)),
        "residual_median_m": float(np.median(abs_residual_arr)),
        "residual_max_m": float(np.max(abs_residual_arr)),
        "continuity_median_m": float(np.median(continuity_residual_arr)),
        "continuity_max_m": float(np.max(continuity_residual_arr)),
        "max_age_s": float(max_age_seen_s),
        "min_stable_epochs": int(min_stable_epochs),
    }, accepted_rows


def _attempt_carrier_anchor_pseudorange_update(
    pf: ParticleFilterDevice,
    tracker: dict[tuple[int, int], CarrierBiasState],
    carrier_rows: dict[tuple[int, int], dict[str, object]],
    current_pf_state: np.ndarray,
    prev_pf_state: np.ndarray | None,
    velocity: np.ndarray | None,
    dt: float,
    tow: float,
    *,
    enabled: bool,
    dd_carrier_result: object | None,
    seed_dd_min_pairs: int,
    sigma_m: float,
    max_age_s: float,
    max_residual_m: float,
    max_continuity_residual_m: float,
    min_stable_epochs: int,
    min_sats: int,
) -> CarrierAnchorAttempt:
    """Try the carrier-anchor rescue path for weak DD carrier epochs."""

    attempt = CarrierAnchorAttempt()
    if not enabled or not carrier_rows:
        return attempt
    if dd_carrier_result is not None and int(getattr(dd_carrier_result, "n_dd", 0)) >= int(
        seed_dd_min_pairs
    ):
        return attempt

    attempt.state = _predict_receiver_state_for_anchor(
        current_pf_state,
        prev_pf_state,
        velocity,
        dt,
    )
    attempt.update, attempt.stats, attempt.rows_used = _build_carrier_anchor_pseudorange_update(
        tracker,
        carrier_rows,
        attempt.state,
        tow,
        max_age_s=max_age_s,
        max_residual_m=max_residual_m,
        max_continuity_residual_m=max_continuity_residual_m,
        min_stable_epochs=min_stable_epochs,
        min_sats=min_sats,
    )
    if attempt.update is None:
        return attempt

    pf.update(
        attempt.update["sat_ecef"],
        attempt.update["pseudoranges"],
        weights=attempt.update["weights"],
        sigma_pr=sigma_m,
    )
    attempt.used = True
    return attempt


def _prepare_dd_carrier_undiff_fallback(
    measurements,
    sat_ecef: np.ndarray,
    pseudoranges: np.ndarray,
    spp_pos_check: np.ndarray,
    tracker: dict[tuple[int, int], CarrierBiasState],
    carrier_rows: dict[tuple[int, int], dict[str, object]],
    carrier_state: np.ndarray | None,
    tow: float,
    *,
    enabled: bool,
    mupf_enabled: bool,
    dd_carrier_result: object | None,
    used_carrier_anchor: bool,
    snr_min: float,
    elev_min: float,
    fallback_sigma_cycles: float,
    fallback_min_sats: int,
    prefer_tracked: bool,
    tracked_min_stable_epochs: int,
    tracked_min_sats: int | None,
    tracked_continuity_good_m: float | None,
    tracked_continuity_bad_m: float | None,
    tracked_sigma_min_scale: float,
    tracked_sigma_max_scale: float,
    max_age_s: float,
    max_continuity_residual_m: float,
    allow_weak_dd: bool = False,
    weak_dd_max_pairs: int | None = None,
) -> CarrierFallbackAttempt:
    """Prepare the same-band undifferenced carrier fallback attempt."""

    attempt = CarrierFallbackAttempt()
    if not enabled or mupf_enabled or used_carrier_anchor:
        return attempt
    dd_pairs = int(getattr(dd_carrier_result, "n_dd", 0)) if dd_carrier_result is not None else 0
    dd_available = dd_pairs >= 3
    weak_dd = (
        allow_weak_dd
        and weak_dd_max_pairs is not None
        and dd_available
        and dd_pairs <= int(weak_dd_max_pairs)
    )
    if dd_available and not weak_dd:
        return attempt
    attempt.replaced_weak_dd = bool(weak_dd)

    tracked_assist_min_sats = int(1 if tracked_min_sats is None else tracked_min_sats)
    if prefer_tracked and carrier_rows and carrier_state is not None:
        attempt.attempted_tracked = True
        attempt.afv, attempt.tracked_stats = _collect_hybrid_tracked_undiff_carrier_afv_inputs(
            tracker,
            carrier_rows,
            carrier_state,
            tow,
            max_age_s=max_age_s,
            max_continuity_residual_m=max_continuity_residual_m,
            min_stable_epochs=tracked_min_stable_epochs,
            min_sats=fallback_min_sats,
        )
        tracked_consistent_n_sat = (
            int(attempt.tracked_stats.get("n_tracked_consistent_sat", 0))
            if attempt.tracked_stats is not None
            else 0
        )
        attempt.used_tracked = (
            attempt.afv is not None and tracked_consistent_n_sat >= tracked_assist_min_sats
        )

    if attempt.afv is None:
        attempt.afv = _collect_undiff_carrier_afv_inputs(
            measurements,
            sat_ecef,
            pseudoranges,
            snr_min=snr_min,
            elev_min=elev_min,
            spp_pos_check=spp_pos_check,
            allowed_system_ids=MUPF_L1_COMPAT_SYSTEM_IDS,
        )

    if attempt.afv is None or int(attempt.afv["n_sat"]) < int(fallback_min_sats):
        return attempt

    tracked_continuity_median_m = (
        _finite_float(attempt.tracked_stats.get("continuity_median_m"))
        if attempt.tracked_stats is not None
        else None
    )
    if (
        attempt.tracked_stats is not None
        and tracked_continuity_median_m is not None
        and tracked_continuity_good_m is not None
        and tracked_continuity_bad_m is not None
    ):
        attempt.sigma_scale = metric_sigma_scale(
            tracked_continuity_median_m,
            good_value=float(tracked_continuity_good_m),
            bad_value=float(tracked_continuity_bad_m),
            min_scale=float(tracked_sigma_min_scale),
            max_scale=float(tracked_sigma_max_scale),
        )

    attempt.sigma_cycles = float(fallback_sigma_cycles) * float(attempt.sigma_scale)
    return attempt


def _apply_dd_carrier_undiff_fallback(
    pf: ParticleFilterDevice,
    attempt: CarrierFallbackAttempt,
) -> CarrierFallbackAttempt:
    """Apply a prepared undifferenced carrier fallback attempt to the PF."""

    if attempt.afv is None or attempt.sigma_cycles is None:
        return attempt
    pf.update_carrier_afv(
        attempt.afv["sat_ecef"],
        attempt.afv["carrier_phase_cycles"],
        weights=attempt.afv["weights"],
        wavelength=MUPF_L1_WAVELENGTH_M,
        sigma_cycles=attempt.sigma_cycles,
    )
    attempt.used = True
    return attempt


def _attempt_dd_carrier_undiff_fallback(
    pf: ParticleFilterDevice,
    measurements,
    sat_ecef: np.ndarray,
    pseudoranges: np.ndarray,
    spp_pos_check: np.ndarray,
    tracker: dict[tuple[int, int], CarrierBiasState],
    carrier_rows: dict[tuple[int, int], dict[str, object]],
    carrier_state: np.ndarray | None,
    tow: float,
    *,
    enabled: bool,
    mupf_enabled: bool,
    dd_carrier_result: object | None,
    used_carrier_anchor: bool,
    snr_min: float,
    elev_min: float,
    fallback_sigma_cycles: float,
    fallback_min_sats: int,
    prefer_tracked: bool,
    tracked_min_stable_epochs: int,
    tracked_min_sats: int | None,
    tracked_continuity_good_m: float | None,
    tracked_continuity_bad_m: float | None,
    tracked_sigma_min_scale: float,
    tracked_sigma_max_scale: float,
    max_age_s: float,
    max_continuity_residual_m: float,
    allow_weak_dd: bool = False,
    weak_dd_max_pairs: int | None = None,
) -> CarrierFallbackAttempt:
    """Try the same-band undifferenced carrier fallback and apply it to the PF."""

    attempt = _prepare_dd_carrier_undiff_fallback(
        measurements,
        sat_ecef,
        pseudoranges,
        spp_pos_check,
        tracker,
        carrier_rows,
        carrier_state,
        tow,
        enabled=enabled,
        mupf_enabled=mupf_enabled,
        dd_carrier_result=dd_carrier_result,
        used_carrier_anchor=used_carrier_anchor,
        snr_min=snr_min,
        elev_min=elev_min,
        fallback_sigma_cycles=fallback_sigma_cycles,
        fallback_min_sats=fallback_min_sats,
        prefer_tracked=prefer_tracked,
        tracked_min_stable_epochs=tracked_min_stable_epochs,
        tracked_min_sats=tracked_min_sats,
        tracked_continuity_good_m=tracked_continuity_good_m,
        tracked_continuity_bad_m=tracked_continuity_bad_m,
        tracked_sigma_min_scale=tracked_sigma_min_scale,
        tracked_sigma_max_scale=tracked_sigma_max_scale,
        max_age_s=max_age_s,
        max_continuity_residual_m=max_continuity_residual_m,
        allow_weak_dd=allow_weak_dd,
        weak_dd_max_pairs=weak_dd_max_pairs,
    )
    return _apply_dd_carrier_undiff_fallback(pf, attempt)


def _should_replace_weak_dd_with_fallback(
    dd_carrier_result,
    dd_pseudorange_result,
    *,
    raw_afv_median_cycles: float | None,
    ess_ratio: float | None,
    weak_dd_max_pairs: int | None,
    weak_dd_max_ess_ratio: float | None,
    weak_dd_min_raw_afv_median_cycles: float | None,
    weak_dd_require_no_dd_pr: bool,
) -> bool:
    """Return True when a weak DD carrier epoch should try undiff fallback first."""

    if dd_carrier_result is None or int(getattr(dd_carrier_result, "n_dd", 0)) < 3:
        return False

    conds: list[bool] = []
    if weak_dd_max_pairs is not None:
        conds.append(int(getattr(dd_carrier_result, "n_dd", 0)) <= int(weak_dd_max_pairs))
    if weak_dd_max_ess_ratio is not None:
        conds.append(
            ess_ratio is not None
            and float(ess_ratio) <= float(weak_dd_max_ess_ratio)
        )
    if weak_dd_min_raw_afv_median_cycles is not None:
        conds.append(
            raw_afv_median_cycles is not None
            and float(raw_afv_median_cycles) >= float(weak_dd_min_raw_afv_median_cycles)
        )
    if weak_dd_require_no_dd_pr:
        conds.append(
            dd_pseudorange_result is None or int(getattr(dd_pseudorange_result, "n_dd", 0)) < 3
        )
    return bool(conds) and all(conds)


def _should_skip_low_support_dd_carrier(
    dd_carrier_result,
    dd_pseudorange_result,
    *,
    ess_ratio: float | None,
    spread_m: float | None,
    raw_afv_median_cycles: float | None,
    low_support_ess_ratio: float | None,
    low_support_max_pairs: int | None,
    low_support_max_spread_m: float | None,
    low_support_min_raw_afv_median_cycles: float | None,
    low_support_require_no_dd_pr: bool,
) -> bool:
    """Return True when DD carrier should be skipped in low-support epochs."""

    if dd_carrier_result is None or int(getattr(dd_carrier_result, "n_dd", 0)) < 3:
        return False

    conds: list[bool] = []
    if low_support_ess_ratio is not None:
        conds.append(
            ess_ratio is not None
            and float(ess_ratio) <= float(low_support_ess_ratio)
        )
    if low_support_max_pairs is not None:
        conds.append(int(getattr(dd_carrier_result, "n_dd", 0)) <= int(low_support_max_pairs))
    if low_support_max_spread_m is not None:
        conds.append(
            spread_m is not None
            and float(spread_m) <= float(low_support_max_spread_m)
        )
    if low_support_min_raw_afv_median_cycles is not None:
        conds.append(
            raw_afv_median_cycles is not None
            and float(raw_afv_median_cycles) >= float(low_support_min_raw_afv_median_cycles)
        )
    if low_support_require_no_dd_pr:
        conds.append(
            dd_pseudorange_result is None or int(getattr(dd_pseudorange_result, "n_dd", 0)) < 3
        )
    return bool(conds) and all(conds)


def _effective_dd_carrier_epoch_median_gate(
    dd_pseudorange_result,
    *,
    base_epoch_median_cycles: float | None,
    ess_ratio: float | None,
    spread_m: float | None,
    low_ess_epoch_median_cycles: float | None,
    low_ess_max_ratio: float | None,
    low_ess_max_spread_m: float | None,
    low_ess_require_no_dd_pr: bool,
) -> float | None:
    """Return the active DD-carrier epoch-median gate after contextual tightening."""

    limit = base_epoch_median_cycles
    if low_ess_epoch_median_cycles is None:
        return limit

    conds: list[bool] = []
    if low_ess_max_ratio is not None:
        conds.append(
            ess_ratio is not None
            and float(ess_ratio) <= float(low_ess_max_ratio)
        )
    if low_ess_max_spread_m is not None:
        conds.append(
            spread_m is not None
            and float(spread_m) <= float(low_ess_max_spread_m)
        )
    if low_ess_require_no_dd_pr:
        conds.append(
            dd_pseudorange_result is None or int(getattr(dd_pseudorange_result, "n_dd", 0)) < 3
        )
    if conds and all(conds):
        contextual_limit = float(low_ess_epoch_median_cycles)
        return contextual_limit if limit is None else min(float(limit), contextual_limit)
    return limit

def _propagate_carrier_bias_tracker_tdcp(
    tracker: dict[tuple[int, int], CarrierBiasState],
    carrier_rows: dict[tuple[int, int], dict[str, object]],
    receiver_state: np.ndarray,
    tow: float,
    *,
    blend_alpha: float,
    reanchor_jump_cycles: float,
    max_age_s: float,
    max_continuity_residual_m: float,
) -> int:
    """Advance per-satellite carrier state across weak-DD epochs using TDCP."""

    stale_keys = [
        key for key, state in tracker.items()
        if float(tow) - float(state.last_tow) > float(max_age_s)
    ]
    for key in stale_keys:
        tracker.pop(key, None)

    n_propagated = 0
    for key, row in carrier_rows.items():
        state = tracker.get(key)
        if state is None:
            continue
        tdcp_pseudorange_m = _carrier_tdcp_predicted_pseudorange_m(
            state,
            row,
            receiver_state,
            tow,
        )
        if tdcp_pseudorange_m is None or not np.isfinite(tdcp_pseudorange_m):
            continue
        bias_based_pseudorange_m = float(row["wavelength_m"]) * (
            float(row["carrier_phase_cycles"]) - float(state.bias_cycles)
        )
        continuity_residual_m = abs(tdcp_pseudorange_m - bias_based_pseudorange_m)
        if continuity_residual_m > float(max_continuity_residual_m):
            continue
        anchor_pseudorange_m = 0.5 * (
            float(tdcp_pseudorange_m) + float(bias_based_pseudorange_m)
        )
        propagated_bias = float(row["carrier_phase_cycles"]) - (
            float(anchor_pseudorange_m) / float(row["wavelength_m"])
        )
        if abs(propagated_bias - float(state.bias_cycles)) > float(reanchor_jump_cycles):
            continue
        alpha = float(np.clip(blend_alpha, 0.0, 1.0))
        next_bias = (1.0 - alpha) * float(state.bias_cycles) + alpha * float(propagated_bias)
        tracker[key] = CarrierBiasState(
            bias_cycles=float(next_bias),
            last_tow=float(tow),
            last_expected_cycles=float(anchor_pseudorange_m / float(row["wavelength_m"])),
            last_carrier_phase_cycles=float(row["carrier_phase_cycles"]),
            last_pseudorange_m=float(anchor_pseudorange_m),
            last_receiver_state=np.asarray(receiver_state, dtype=np.float64).copy(),
            last_sat_ecef=np.asarray(row["sat_ecef"], dtype=np.float64).copy(),
            last_sat_velocity=np.asarray(
                row.get("sat_velocity", np.zeros(3)), dtype=np.float64
            ).copy(),
            last_clock_drift=float(row.get("clock_drift", 0.0)),
            stable_epochs=int(state.stable_epochs) + 1,
        )
        n_propagated += 1
    return n_propagated
