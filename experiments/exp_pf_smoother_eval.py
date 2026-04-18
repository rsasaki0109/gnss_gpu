#!/usr/bin/env python3
"""Evaluate forward-backward particle smoothing on the gnssplusplus UrbanNav PF stack.

Pipeline per epoch (forward): predict (SPP or TDCP guide) → ``correct_clock_bias`` →
``update`` → optional ``position_update`` → ``store_epoch`` when smoothing is enabled.

After the forward pass, ``smooth()`` runs a backward PF and averages with forward
estimates. Metrics are computed on epochs aligned with UrbanNav ground-truth time tags
(same convention as ``exp_position_update_eval.py``).
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
for _p in (
    _PROJECT_ROOT / "python",
    _PROJECT_ROOT / "third_party" / "gnssplusplus" / "python",
    _PROJECT_ROOT / "third_party" / "gnssplusplus" / "build" / "python",
):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from evaluate import compute_metrics, ecef_errors_2d_3d, ecef_to_lla
from exp_urbannav_baseline import load_or_generate_data
from exp_urbannav_pf3d import PF_SIGMA_CB, PF_SIGMA_POS
from gnss_gpu import ParticleFilterDevice
from gnss_gpu.dd_quality import (
    combine_sigma_scales,
    dd_carrier_afv_cycles,
    dd_pseudorange_residuals_m,
    ess_gate_scale,
    gate_dd_carrier,
    gate_dd_pseudorange,
    metric_sigma_scale,
    pair_count_sigma_scale,
    spread_gate_scale,
)
from gnss_gpu.imu import ComplementaryHeadingFilter, load_imu_csv
from gnss_gpu.osm_constraint import OSMRoadNetwork
from gnss_gpu.tdcp_velocity import (
    C_LIGHT as TDCP_C_LIGHT,
    estimate_velocity_from_tdcp_with_metrics,
)

RESULTS_DIR = _SCRIPT_DIR / "results"
_MUPF_L1_COMPAT_SYSTEM_IDS = frozenset({0, 2, 4})  # G/E/J share the L1/E1 frequency
_MUPF_L1_WAVELENGTH_M = 299792458.0 / 1575.42e6
_CLI_PRESETS: dict[str, dict[str, object]] = {
    "odaiba_reference": {
        "description": "Smoother-first Odaiba reference: IMU stop-detect plus 0.18-cycle DD floor.",
        "argv": [
            "--runs", "Odaiba",
            "--n-particles", "100000",
            "--sigma-pos", "1.2",
            "--position-update-sigma", "1.9",
            "--predict-guide", "imu",
            "--imu-tight-coupling",
            "--imu-stop-sigma-pos", "0.1",
            "--residual-downweight",
            "--pr-accel-downweight",
            "--smoother",
            "--dd-pseudorange",
            "--dd-pseudorange-sigma", "0.5",
            "--dd-pseudorange-gate-adaptive-floor-m", "4.0",
            "--dd-pseudorange-gate-adaptive-mad-mult", "3.0",
            "--dd-pseudorange-gate-ess-min-scale", "0.9",
            "--dd-pseudorange-gate-ess-max-scale", "1.1",
            "--mupf-dd",
            "--mupf-dd-sigma-cycles", "0.20",
            "--mupf-dd-base-interp",
            "--mupf-dd-gate-adaptive-floor-cycles", "0.18",
            "--mupf-dd-gate-adaptive-mad-mult", "3.0",
            "--mupf-dd-gate-ess-min-scale", "0.9",
            "--mupf-dd-gate-ess-max-scale", "1.1",
            "--mupf-dd-gate-spread-min-scale", "0.88",
            "--mupf-dd-gate-spread-max-scale", "1.0",
            "--mupf-dd-gate-low-spread-m", "3.0",
            "--mupf-dd-gate-high-spread-m", "8.0",
            "--mupf-dd-skip-low-support-ess-ratio", "0.01",
            "--mupf-dd-skip-low-support-max-pairs", "4",
            "--mupf-dd-skip-low-support-min-raw-afv-median-cycles", "0.15",
            "--mupf-dd-fallback-undiff",
            "--mupf-dd-fallback-sigma-cycles", "0.10",
            "--mupf-dd-fallback-min-sats", "4",
            "--carrier-anchor",
            "--carrier-anchor-sigma-m", "0.25",
            "--carrier-anchor-max-residual-m", "0.80",
            "--carrier-anchor-max-continuity-residual-m", "0.50",
        ],
    },
    "odaiba_stop_detect": {
        "description": "Forward-stable Odaiba sibling: IMU stop-detect plus 0.25-cycle DD floor.",
        "argv": [
            "--runs", "Odaiba",
            "--n-particles", "100000",
            "--sigma-pos", "1.2",
            "--position-update-sigma", "1.9",
            "--predict-guide", "imu",
            "--imu-tight-coupling",
            "--imu-stop-sigma-pos", "0.1",
            "--residual-downweight",
            "--pr-accel-downweight",
            "--smoother",
            "--dd-pseudorange",
            "--dd-pseudorange-sigma", "0.5",
            "--dd-pseudorange-gate-adaptive-floor-m", "4.0",
            "--dd-pseudorange-gate-adaptive-mad-mult", "3.0",
            "--dd-pseudorange-gate-ess-min-scale", "0.9",
            "--dd-pseudorange-gate-ess-max-scale", "1.1",
            "--mupf-dd",
            "--mupf-dd-sigma-cycles", "0.20",
            "--mupf-dd-base-interp",
            "--mupf-dd-gate-adaptive-floor-cycles", "0.25",
            "--mupf-dd-gate-adaptive-mad-mult", "3.0",
            "--mupf-dd-gate-ess-min-scale", "0.9",
            "--mupf-dd-gate-ess-max-scale", "1.1",
            "--mupf-dd-gate-spread-min-scale", "0.88",
            "--mupf-dd-gate-spread-max-scale", "1.0",
            "--mupf-dd-gate-low-spread-m", "3.0",
            "--mupf-dd-gate-high-spread-m", "8.0",
            "--mupf-dd-skip-low-support-ess-ratio", "0.01",
            "--mupf-dd-skip-low-support-max-pairs", "4",
            "--mupf-dd-skip-low-support-min-raw-afv-median-cycles", "0.15",
            "--mupf-dd-fallback-undiff",
            "--mupf-dd-fallback-sigma-cycles", "0.10",
            "--mupf-dd-fallback-min-sats", "4",
            "--carrier-anchor",
            "--carrier-anchor-sigma-m", "0.25",
            "--carrier-anchor-max-residual-m", "0.80",
            "--carrier-anchor-max-continuity-residual-m", "0.50",
        ],
    },
    "odaiba_reference_guarded": {
        "description": "Odaiba reference + low-ESS smoother tail guard for weak smoothing tails.",
        "argv": [
            "--runs", "Odaiba",
            "--n-particles", "100000",
            "--sigma-pos", "1.2",
            "--position-update-sigma", "1.9",
            "--predict-guide", "imu",
            "--imu-tight-coupling",
            "--residual-downweight",
            "--pr-accel-downweight",
            "--smoother",
            "--dd-pseudorange",
            "--dd-pseudorange-sigma", "0.5",
            "--dd-pseudorange-gate-adaptive-floor-m", "4.0",
            "--dd-pseudorange-gate-adaptive-mad-mult", "3.0",
            "--dd-pseudorange-gate-ess-min-scale", "0.9",
            "--dd-pseudorange-gate-ess-max-scale", "1.1",
            "--mupf-dd",
            "--mupf-dd-sigma-cycles", "0.20",
            "--mupf-dd-base-interp",
            "--mupf-dd-gate-adaptive-floor-cycles", "0.18",
            "--mupf-dd-gate-adaptive-mad-mult", "3.0",
            "--mupf-dd-gate-ess-min-scale", "0.9",
            "--mupf-dd-gate-ess-max-scale", "1.1",
            "--mupf-dd-gate-spread-min-scale", "0.88",
            "--mupf-dd-gate-spread-max-scale", "1.0",
            "--mupf-dd-gate-low-spread-m", "3.0",
            "--mupf-dd-gate-high-spread-m", "8.0",
            "--mupf-dd-skip-low-support-ess-ratio", "0.01",
            "--mupf-dd-skip-low-support-max-pairs", "4",
            "--mupf-dd-skip-low-support-min-raw-afv-median-cycles", "0.15",
            "--mupf-dd-fallback-undiff",
            "--mupf-dd-fallback-sigma-cycles", "0.10",
            "--mupf-dd-fallback-min-sats", "4",
            "--carrier-anchor",
            "--carrier-anchor-sigma-m", "0.25",
            "--carrier-anchor-max-residual-m", "0.80",
            "--carrier-anchor-max-continuity-residual-m", "0.50",
            "--smoother-tail-guard-ess-max-ratio", "0.001",
            "--smoother-tail-guard-min-shift-m", "4.0",
        ],
    },
    "odaiba_best_accuracy": {
        "description": "Odaiba P50-minimizing config (200K + stop-sigma + anchor-sigma 0.15).",
        "argv": [
            "--runs", "Odaiba",
            "--n-particles", "200000",
            "--sigma-pos", "1.2",
            "--position-update-sigma", "1.9",
            "--predict-guide", "imu",
            "--imu-tight-coupling",
            "--imu-stop-sigma-pos", "0.1",
            "--residual-downweight",
            "--pr-accel-downweight",
            "--smoother",
            "--dd-pseudorange",
            "--dd-pseudorange-sigma", "0.5",
            "--dd-pseudorange-gate-adaptive-floor-m", "4.0",
            "--dd-pseudorange-gate-adaptive-mad-mult", "3.0",
            "--dd-pseudorange-gate-ess-min-scale", "0.9",
            "--dd-pseudorange-gate-ess-max-scale", "1.1",
            "--mupf-dd",
            "--mupf-dd-sigma-cycles", "0.20",
            "--mupf-dd-base-interp",
            "--mupf-dd-gate-adaptive-floor-cycles", "0.18",
            "--mupf-dd-gate-adaptive-mad-mult", "3.0",
            "--mupf-dd-gate-ess-min-scale", "0.9",
            "--mupf-dd-gate-ess-max-scale", "1.1",
            "--mupf-dd-gate-spread-min-scale", "0.88",
            "--mupf-dd-gate-spread-max-scale", "1.0",
            "--mupf-dd-gate-low-spread-m", "3.0",
            "--mupf-dd-gate-high-spread-m", "8.0",
            "--mupf-dd-skip-low-support-ess-ratio", "0.01",
            "--mupf-dd-skip-low-support-max-pairs", "4",
            "--mupf-dd-skip-low-support-min-raw-afv-median-cycles", "0.15",
            "--mupf-dd-fallback-undiff",
            "--mupf-dd-fallback-sigma-cycles", "0.10",
            "--mupf-dd-fallback-min-sats", "4",
            "--carrier-anchor",
            "--carrier-anchor-sigma-m", "0.15",
            "--carrier-anchor-max-residual-m", "0.80",
            "--carrier-anchor-max-continuity-residual-m", "0.50",
            "--smoother-tail-guard-ess-max-ratio", "0.001",
            "--smoother-tail-guard-min-shift-m", "4.0",
        ],
    },
}


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


def _finite_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


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
            allowed_system_ids=_MUPF_L1_COMPAT_SYSTEM_IDS,
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
        wavelength=_MUPF_L1_WAVELENGTH_M,
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


def _format_diag_value(value: object, fmt: str) -> str:
    out = _finite_float(value)
    if out is None:
        return "n/a"
    return format(out, fmt)


def _format_diag_metric(value: object, fmt: str, unit: str) -> str:
    out = _finite_float(value)
    if out is None:
        return "n/a"
    return f"{format(out, fmt)}{unit}"


def _diagnostics_output_path(base_path: Path, run_name: str, label: str, multiple_outputs: bool) -> Path:
    if not multiple_outputs:
        return base_path
    safe_run = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in run_name)
    safe_label = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in label)
    suffix = base_path.suffix or ".csv"
    return base_path.with_name(f"{base_path.stem}_{safe_run}_{safe_label}{suffix}")


def _write_epoch_diagnostics(rows: list[dict[str, object]], output_path: Path) -> None:
    if not rows:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _print_top_epoch_diagnostics(rows: list[dict[str, object]], top_k: int) -> None:
    if top_k <= 0 or not rows:
        return
    sort_key = "smoothed_error_2d" if any(_finite_float(r.get("smoothed_error_2d")) is not None for r in rows) else "forward_error_2d"
    ranked = sorted(
        rows,
        key=lambda row: _finite_float(row.get(sort_key)) if _finite_float(row.get(sort_key)) is not None else -1.0,
        reverse=True,
    )
    n_show = min(int(top_k), len(ranked))
    print(f"  [epoch_diag] worst {n_show} epochs by {sort_key}:")
    for row in ranked[:n_show]:
        print(
            "    "
            f"tow={_format_diag_value(row.get('tow'), '.1f')} "
            f"{sort_key}={_format_diag_metric(row.get(sort_key), '.2f', 'm')} "
            f"fwd={_format_diag_metric(row.get('forward_error_2d'), '.2f', 'm')} "
            f"smth={_format_diag_metric(row.get('smoothed_error_2d'), '.2f', 'm')} "
            f"shift={_format_diag_metric(row.get('smoothed_shift_3d_m'), '.2f', 'm')} "
            f"ess={_format_diag_value(row.get('gate_ess_ratio'), '.3f')} "
            f"spread={_format_diag_metric(row.get('gate_spread_m'), '.2f', 'm')} "
            f"dd_pr={int(row.get('dd_pr_kept_pairs') or 0)}/{int(row.get('dd_pr_input_pairs') or 0)} "
            f"dd_pr_med={_format_diag_metric(row.get('dd_pr_raw_abs_res_median_m'), '.2f', 'm')} "
            f"dd_cp={int(row.get('dd_cp_kept_pairs') or 0)}/{int(row.get('dd_cp_input_pairs') or 0)} "
            f"dd_cp_med={_format_diag_metric(row.get('dd_cp_raw_abs_afv_median_cycles'), '.3f', 'cy')}"
        )


def _apply_smoother_tail_guard(
    smoothed_aligned: np.ndarray,
    forward_aligned: np.ndarray,
    epoch_diagnostics: list[dict[str, object]] | None,
    *,
    ess_max_ratio: float | None = None,
    dd_carrier_max_pairs: int | None = None,
    dd_pseudorange_max_pairs: int | None = None,
    min_shift_m: float | None = None,
) -> tuple[np.ndarray, int]:
    if epoch_diagnostics is None:
        return smoothed_aligned, 0

    guarded = np.asarray(smoothed_aligned, dtype=np.float64).copy()
    applied = 0
    shift_m = np.linalg.norm(guarded - np.asarray(forward_aligned, dtype=np.float64), axis=1)
    guard_enabled = not (
        ess_max_ratio is None
        and dd_carrier_max_pairs is None
        and dd_pseudorange_max_pairs is None
        and min_shift_m is None
    )
    for i, row in enumerate(epoch_diagnostics):
        row["smoothed_shift_3d_m"] = float(shift_m[i])
        row["tail_guard_applied"] = False
        if not guard_enabled:
            continue
        conds: list[bool] = []
        if ess_max_ratio is not None:
            conds.append(_finite_float(row.get("gate_ess_ratio")) is not None and float(row["gate_ess_ratio"]) <= float(ess_max_ratio))
        if dd_carrier_max_pairs is not None:
            dd_cp_kept = row.get("dd_cp_kept_pairs")
            conds.append(dd_cp_kept is not None and int(dd_cp_kept) <= int(dd_carrier_max_pairs))
        if dd_pseudorange_max_pairs is not None:
            dd_pr_kept = row.get("dd_pr_kept_pairs")
            conds.append(dd_pr_kept is not None and int(dd_pr_kept) <= int(dd_pseudorange_max_pairs))
        if min_shift_m is not None:
            conds.append(float(shift_m[i]) >= float(min_shift_m))
        use_forward = bool(conds) and all(conds)
        row["tail_guard_applied"] = use_forward
        if use_forward:
            guarded[i] = forward_aligned[i]
            applied += 1
    return guarded, applied


def _find_base_obs_path(run_dir: Path) -> Path | None:
    for name in ("base_trimble.obs", "base.obs"):
        p = run_dir / name
        if p.exists():
            return p
    return None


def load_pf_smoother_dataset(run_dir: Path, rover_source: str = "trimble") -> dict[str, object]:
    """Load RINEX / UrbanNav ground-truth once for repeated PF runs (sweeps).

    Returns a dict with keys: ``epochs``, ``spp_lookup``, ``gt``, ``our_times``,
    ``first_pos``, ``init_cb``.  If ``imu.csv`` exists in *run_dir*, also
    includes ``imu_data`` (raw dict from :func:`load_imu_csv`).
    """
    from libgnsspp import preprocess_spp_file, solve_spp_file

    obs_path = str(run_dir / f"rover_{rover_source}.obs")
    nav_path = str(run_dir / "base.nav")

    epochs = preprocess_spp_file(obs_path, nav_path)
    sol = solve_spp_file(obs_path, nav_path)
    spp_records = [r for r in sol.records() if r.is_valid()]
    spp_lookup = {round(r.time.tow, 1): np.array(r.position_ecef_m) for r in spp_records}

    data = load_or_generate_data(run_dir, systems=("G", "E", "J"), urban_rover=rover_source)
    gt = data["ground_truth"]
    our_times = data["times"]

    first_pos = np.array(spp_records[0].position_ecef_m[:3], dtype=np.float64)
    init_meas = None
    for sol_epoch, measurements in epochs:
        if sol_epoch.is_valid() and len(measurements) >= 4:
            init_meas = measurements
            first_pos = np.array(sol_epoch.position_ecef_m[:3], dtype=np.float64)
            break
    if init_meas is None:
        raise RuntimeError(f"No valid epoch for init in {run_dir}")

    init_cb = float(
        np.median(
            [
                m.corrected_pseudorange
                - np.linalg.norm(np.asarray(m.satellite_ecef, dtype=np.float64) - first_pos)
                for m in init_meas
            ]
        )
    )
    result = {
        "epochs": epochs,
        "spp_lookup": spp_lookup,
        "gt": gt,
        "our_times": our_times,
        "first_pos": first_pos,
        "init_cb": init_cb,
    }

    # Load IMU data if available
    imu_path = run_dir / "imu.csv"
    if imu_path.exists():
        result["imu_data"] = load_imu_csv(imu_path)
        print(f"  [IMU] loaded {len(result['imu_data']['tow'])} samples from {imu_path}")
    else:
        result["imu_data"] = None

    return result


def _spp_heading_from_velocity(spp_vel_ecef: np.ndarray, lat: float, lon: float) -> float | None:
    """Compute heading (radians from north, clockwise) from SPP velocity in ECEF."""
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    sin_lon = math.sin(lon)
    cos_lon = math.cos(lon)
    # ECEF to ENU rotation
    ve = -sin_lon * spp_vel_ecef[0] + cos_lon * spp_vel_ecef[1]
    vn = (-sin_lat * cos_lon * spp_vel_ecef[0]
          - sin_lat * sin_lon * spp_vel_ecef[1]
          + cos_lat * spp_vel_ecef[2])
    speed = math.sqrt(ve ** 2 + vn ** 2)
    if speed < 0.5:  # too slow to get reliable heading
        return None
    return math.atan2(ve, vn)  # heading from north, clockwise


def run_pf_with_optional_smoother(
    run_dir: Path,
    run_name: str,
    *,
    n_particles: int,
    sigma_pos: float,
    sigma_pr: float,
    position_update_sigma: float | None,
    predict_guide: str,
    use_smoother: bool,
    rover_source: str = "trimble",
    max_epochs: int = 0,
    skip_valid_epochs: int = 0,
    sigma_pos_tdcp: float | None = None,
    sigma_pos_tdcp_tight: float | None = None,
    tdcp_tight_rms_max_m: float = 1.0e9,
    dataset: dict[str, object] | None = None,
    resampling: str = "megopolis",
    tdcp_elevation_weight: bool = False,
    tdcp_el_sin_floor: float = 0.1,
    tdcp_rms_threshold: float = 3.0,
    residual_downweight: bool = False,
    residual_threshold: float = 15.0,
    pr_accel_downweight: bool = False,
    pr_accel_threshold: float = 5.0,
    use_gmm: bool = False,
    gmm_w_los: float = 0.7,
    gmm_mu_nlos: float = 15.0,
    gmm_sigma_nlos: float = 30.0,
    doppler_position_update: bool = False,
    doppler_pu_sigma: float = 5.0,
    imu_tight_coupling: bool = False,
    imu_stop_sigma_pos: float | None = None,
    tdcp_position_update: bool = False,
    tdcp_pu_sigma: float = 0.5,
    tdcp_pu_rms_max: float = 3.0,
    mupf: bool = False,
    mupf_sigma_cycles: float = 0.05,
    mupf_snr_min: float = 25.0,
    mupf_elev_min: float = 0.15,
    dd_pseudorange: bool = False,
    dd_pseudorange_sigma: float = 0.75,
    dd_pseudorange_base_interp: bool = False,
    dd_pseudorange_gate_residual_m: float | None = None,
    dd_pseudorange_gate_adaptive_floor_m: float | None = None,
    dd_pseudorange_gate_adaptive_mad_mult: float | None = None,
    dd_pseudorange_gate_epoch_median_m: float | None = None,
    dd_pseudorange_gate_ess_min_scale: float = 1.0,
    dd_pseudorange_gate_ess_max_scale: float = 1.0,
    dd_pseudorange_gate_spread_min_scale: float = 1.0,
    dd_pseudorange_gate_spread_max_scale: float = 1.0,
    dd_pseudorange_gate_low_spread_m: float = 1.5,
    dd_pseudorange_gate_high_spread_m: float = 8.0,
    mupf_dd: bool = False,
    mupf_dd_sigma_cycles: float = 0.05,
    mupf_dd_base_interp: bool = False,
    mupf_dd_gate_afv_cycles: float | None = None,
    mupf_dd_gate_adaptive_floor_cycles: float | None = None,
    mupf_dd_gate_adaptive_mad_mult: float | None = None,
    mupf_dd_gate_epoch_median_cycles: float | None = None,
    mupf_dd_gate_low_ess_epoch_median_cycles: float | None = None,
    mupf_dd_gate_low_ess_max_ratio: float | None = None,
    mupf_dd_gate_low_ess_max_spread_m: float | None = None,
    mupf_dd_gate_low_ess_require_no_dd_pr: bool = False,
    mupf_dd_gate_ess_min_scale: float = 1.0,
    mupf_dd_gate_ess_max_scale: float = 1.0,
    mupf_dd_gate_spread_min_scale: float = 1.0,
    mupf_dd_gate_spread_max_scale: float = 1.0,
    mupf_dd_gate_low_spread_m: float = 1.5,
    mupf_dd_gate_high_spread_m: float = 8.0,
    mupf_dd_sigma_support_low_pairs: int | None = None,
    mupf_dd_sigma_support_high_pairs: int | None = None,
    mupf_dd_sigma_support_max_scale: float = 1.0,
    mupf_dd_sigma_afv_good_cycles: float | None = None,
    mupf_dd_sigma_afv_bad_cycles: float | None = None,
    mupf_dd_sigma_afv_max_scale: float = 1.0,
    mupf_dd_sigma_ess_low_ratio: float | None = None,
    mupf_dd_sigma_ess_high_ratio: float | None = None,
    mupf_dd_sigma_ess_max_scale: float = 1.0,
    mupf_dd_sigma_max_scale: float | None = None,
    carrier_anchor: bool = False,
    carrier_anchor_sigma_m: float = 0.25,
    carrier_anchor_min_sats: int = 4,
    carrier_anchor_max_age_s: float = 3.0,
    carrier_anchor_max_residual_m: float = 0.75,
    carrier_anchor_max_continuity_residual_m: float = 0.50,
    carrier_anchor_min_stable_epochs: int = 1,
    carrier_anchor_blend_alpha: float = 0.5,
    carrier_anchor_reanchor_jump_cycles: float = 4.0,
    carrier_anchor_seed_dd_min_pairs: int = 3,
    mupf_dd_fallback_undiff: bool = False,
    mupf_dd_fallback_sigma_cycles: float = 0.10,
    mupf_dd_fallback_min_sats: int = 4,
    mupf_dd_fallback_prefer_tracked: bool = False,
    mupf_dd_fallback_tracked_min_stable_epochs: int = 1,
    mupf_dd_fallback_tracked_min_sats: int | None = None,
    mupf_dd_fallback_tracked_continuity_good_m: float | None = None,
    mupf_dd_fallback_tracked_continuity_bad_m: float | None = None,
    mupf_dd_fallback_tracked_sigma_min_scale: float = 1.0,
    mupf_dd_fallback_tracked_sigma_max_scale: float = 1.0,
    mupf_dd_fallback_weak_dd_max_pairs: int | None = None,
    mupf_dd_fallback_weak_dd_max_ess_ratio: float | None = None,
    mupf_dd_fallback_weak_dd_min_raw_afv_median_cycles: float | None = None,
    mupf_dd_fallback_weak_dd_require_no_dd_pr: bool = False,
    mupf_dd_skip_low_support_ess_ratio: float | None = None,
    mupf_dd_skip_low_support_max_pairs: int | None = None,
    mupf_dd_skip_low_support_max_spread_m: float | None = None,
    mupf_dd_skip_low_support_min_raw_afv_median_cycles: float | None = None,
    mupf_dd_skip_low_support_require_no_dd_pr: bool = False,
    collect_epoch_diagnostics: bool = False,
    smoother_tail_guard_ess_max_ratio: float | None = None,
    smoother_tail_guard_dd_carrier_max_pairs: int | None = None,
    smoother_tail_guard_dd_pseudorange_max_pairs: int | None = None,
    smoother_tail_guard_min_shift_m: float | None = None,
    osm_map_constraint: bool = False,
    osm_road_file: Path | None = None,
    osm_road_sigma_m: float = 2.0,
    osm_road_huber_k: float = 2.0,
    osm_road_search_radius_m: float = 80.0,
    osm_road_max_segments: int = 96,
) -> dict[str, object]:
    if dd_pseudorange and use_gmm:
        raise ValueError("dd_pseudorange cannot be combined with --gmm")
    if osm_map_constraint and osm_road_sigma_m < 1.0:
        raise ValueError("osm_road_sigma_m must be >= 1.0m")
    if osm_map_constraint and osm_road_huber_k <= 0.0:
        raise ValueError("osm_road_huber_k must be positive")
    if osm_map_constraint and osm_road_search_radius_m <= 0.0:
        raise ValueError("osm_road_search_radius_m must be positive")
    if osm_map_constraint and osm_road_max_segments <= 0:
        raise ValueError("osm_road_max_segments must be positive")

    if dataset is None:
        ds = load_pf_smoother_dataset(run_dir, rover_source)
    else:
        ds = dataset

    epochs = ds["epochs"]
    spp_lookup = ds["spp_lookup"]
    gt = ds["gt"]
    our_times = ds["our_times"]
    first_pos = np.asarray(ds["first_pos"], dtype=np.float64)
    init_cb = float(ds["init_cb"])

    osm_network: OSMRoadNetwork | None = None
    if osm_map_constraint:
        if osm_road_file is None:
            raise ValueError("osm_map_constraint requires osm_road_file")
        osm_path = Path(osm_road_file)
        if not osm_path.exists():
            raise FileNotFoundError(f"OSM road file not found: {osm_path}")
        lat0, lon0, _ = ecef_to_lla(float(first_pos[0]), float(first_pos[1]), float(first_pos[2]))
        osm_network = OSMRoadNetwork.from_geojson(
            osm_path,
            origin_lat_deg=math.degrees(lat0),
            origin_lon_deg=math.degrees(lon0),
        )
        print(
            f"  [OSM] loaded {osm_network.segments.n_segments} road segments "
            f"from {osm_path}"
        )

    # --- IMU setup (for predict_guide in {"imu", "imu_spp_blend"}) ---
    imu_filter: ComplementaryHeadingFilter | None = None
    n_imu_used = 0
    n_imu_fallback = 0
    n_imu_stop_detected = 0
    if predict_guide in ("imu", "imu_spp_blend"):
        imu_data = ds.get("imu_data")
        if imu_data is None:
            imu_path = run_dir / "imu.csv"
            if imu_path.exists():
                imu_data = load_imu_csv(imu_path)
            else:
                raise RuntimeError(
                    f"predict_guide={predict_guide} requires IMU data but "
                    f"imu.csv not found in {run_dir}"
                )
        imu_filter = ComplementaryHeadingFilter(imu_data, alpha=0.05)
        # Initialize heading from first two SPP positions
        tow_keys = sorted(spp_lookup.keys())
        if len(tow_keys) >= 2:
            p0 = spp_lookup[tow_keys[0]][:3]
            p1 = spp_lookup[tow_keys[1]][:3]
            lat0, lon0, _ = ecef_to_lla(float(p0[0]), float(p0[1]), float(p0[2]))
            dt_init = tow_keys[1] - tow_keys[0]
            if dt_init > 0:
                spp_vel_init = (p1 - p0) / dt_init
                h = _spp_heading_from_velocity(spp_vel_init, lat0, lon0)
                if h is not None:
                    imu_filter.heading = h

    base_obs_path = _find_base_obs_path(run_dir) if (dd_pseudorange or mupf_dd) else None

    # --- DD pseudorange setup ---
    dd_pr_computer = None
    n_dd_pr_used = 0
    n_dd_pr_skip = 0
    n_dd_pr_gate_pairs_rejected = 0
    n_dd_pr_gate_epoch_skip = 0
    if dd_pseudorange:
        from gnss_gpu.dd_pseudorange import DDPseudorangeComputer

        if base_obs_path is None:
            raise RuntimeError(
                f"dd_pseudorange requires base station RINEX (expected base_trimble.obs or base.obs in {run_dir})"
            )
        dd_pr_computer = DDPseudorangeComputer(
            base_obs_path,
            rover_obs_path=run_dir / f"rover_{rover_source}.obs",
            interpolate_base_epochs=dd_pseudorange_base_interp,
        )
        print(f"  [DD-PR] base_pos = {dd_pr_computer.base_position}")

    # --- DD carrier phase setup ---
    dd_computer = None
    n_dd_used = 0
    n_dd_skip = 0
    n_dd_skip_support_guard = 0
    n_dd_sigma_relaxed = 0
    dd_sigma_scale_sum = 0.0
    n_carrier_anchor_used = 0
    n_carrier_anchor_propagated = 0
    n_dd_fallback_undiff_used = 0
    n_dd_fallback_tracked_attempted = 0
    n_dd_fallback_tracked_used = 0
    n_dd_fallback_weak_dd_replaced = 0
    n_dd_gate_pairs_rejected = 0
    n_dd_gate_epoch_skip = 0
    carrier_bias_tracker: dict[tuple[int, int], CarrierBiasState] = {}
    if mupf_dd:
        from gnss_gpu.dd_carrier import DDCarrierComputer
        if base_obs_path is not None:
            dd_computer = DDCarrierComputer(
                base_obs_path,
                rover_obs_path=run_dir / f"rover_{rover_source}.obs",
                interpolate_base_epochs=mupf_dd_base_interp,
            )
            print(f"  [DD] base_pos = {dd_computer.base_position}")
        else:
            print(f"  [DD] WARNING: no base station RINEX found in {run_dir}, DD disabled")

    pf = ParticleFilterDevice(
        n_particles=n_particles,
        sigma_pos=sigma_pos,
        sigma_cb=PF_SIGMA_CB,
        sigma_pr=sigma_pr,
        resampling=resampling,
        seed=42,
    )
    pf.initialize(first_pos, clock_bias=init_cb, spread_pos=10.0, spread_cb=100.0)

    if use_smoother:
        pf.enable_smoothing()

    forward_aligned: list[np.ndarray] = []
    all_gt: list[np.ndarray] = []
    aligned_indices: list[int] = []
    aligned_epoch_diagnostics: list[dict[str, object]] = []
    n_stored = 0
    n_tdcp_used = 0
    n_tdcp_fallback = 0
    n_imu_tight_used = 0
    n_imu_tight_skip = 0
    n_osm_road_used = 0
    n_osm_road_segments_sum = 0
    # PR acceleration weighting: need per-satellite PR history
    pr_history: dict[int, list[float]] = {}  # prn -> [pr(t-2), pr(t-1)]

    prev_tow = None
    prev_measurements: list | None = None
    prev_estimate: np.ndarray | None = None
    prev_pf_estimate: np.ndarray | None = None
    prev_pf_state: np.ndarray | None = None
    t0 = time.perf_counter()
    epochs_done = 0

    for sol_epoch, measurements in epochs:
        if not sol_epoch.is_valid() or len(measurements) < 4:
            continue
        if max_epochs and epochs_done >= skip_valid_epochs + max_epochs:
            break

        tow = sol_epoch.time.tow
        tow_key = round(tow, 1)
        dt = tow - prev_tow if prev_tow else 0.1

        velocity = None
        imu_velocity = None
        used_tdcp = False
        tdcp_rms = float("nan")
        used_imu = False
        imu_stop_detected = False
        used_imu_tight_epoch = False
        dd_pr_gate_stats = None
        dd_gate_stats = None
        dd_pr_gate_scale = None
        dd_cp_gate_scale = None
        dd_pr_input_pairs = 0
        dd_cp_input_pairs = 0
        dd_pr_raw_abs_res_median_m = None
        dd_pr_raw_abs_res_max_m = None
        dd_cp_raw_abs_afv_median_cycles = None
        dd_cp_raw_abs_afv_max_cycles = None
        dd_cp_sigma_support_scale = 1.0
        dd_cp_sigma_afv_scale = 1.0
        dd_cp_sigma_ess_scale = 1.0
        dd_cp_sigma_scale = 1.0
        dd_cp_sigma_cycles = None
        dd_cp_support_skip = False
        osm_road_update = None
        carrier_anchor_rows = {}
        anchor_attempt = CarrierAnchorAttempt()
        fallback_attempt = CarrierFallbackAttempt()

        if prev_tow is not None and dt > 0:
            # --- IMU-based velocity ---
            if predict_guide in ("imu", "imu_spp_blend") and imu_filter is not None:
                # Get current PF estimate for ECEF -> geodetic
                cur_est = np.asarray(pf.estimate()[:3], dtype=np.float64)
                lat_r, lon_r, _ = ecef_to_lla(
                    float(cur_est[0]), float(cur_est[1]), float(cur_est[2])
                )

                # Correct heading drift with SPP velocity if available
                pk = round(prev_tow, 1)
                spp_fd_vel_ecef: np.ndarray | None = None
                if tow_key in spp_lookup and pk in spp_lookup:
                    ddx = spp_lookup[tow_key][:3] - spp_lookup[pk][:3]
                    if np.all(np.isfinite(ddx)):
                        spp_fd_vel_ecef = ddx / dt
                        spp_heading = _spp_heading_from_velocity(
                            spp_fd_vel_ecef, lat_r, lon_r
                        )
                        if spp_heading is not None:
                            imu_filter.correct_heading_spp(spp_heading)

                # Get IMU velocity in ENU
                vel_enu = imu_filter.get_velocity_enu(prev_tow, tow)
                speed_enu = float(np.linalg.norm(vel_enu[:2]))

                if speed_enu > 0.01:
                    # Convert ENU velocity to ECEF
                    imu_vel_ecef = ComplementaryHeadingFilter.velocity_enu_to_ecef(
                        vel_enu, lat_r, lon_r
                    )

                    if predict_guide == "imu":
                        velocity = imu_vel_ecef
                        imu_velocity = imu_vel_ecef
                        used_imu = True
                        n_imu_used += 1
                    elif predict_guide == "imu_spp_blend":
                        # Blend IMU + SPP velocity (average)
                        if spp_fd_vel_ecef is not None:
                            spp_speed = float(np.linalg.norm(spp_fd_vel_ecef))
                            if spp_speed < 50:
                                velocity = 0.5 * imu_vel_ecef + 0.5 * spp_fd_vel_ecef
                            else:
                                # SPP velocity unreasonable, use IMU only
                                velocity = imu_vel_ecef
                        else:
                            # No SPP velocity, use IMU only
                            velocity = imu_vel_ecef
                        imu_velocity = velocity
                        used_imu = True
                        n_imu_used += 1
                else:
                    # IMU reports near-zero speed → stop detection
                    velocity = np.zeros(3)
                    imu_velocity = np.zeros(3)
                    used_imu = True
                    n_imu_used += 1
                    n_imu_stop_detected += 1
                    imu_stop_detected = True

            # --- TDCP-based velocity ---
            if predict_guide in ("tdcp", "tdcp_adaptive") and prev_measurements is not None:
                spp_pos_pre = np.array(sol_epoch.position_ecef_m[:3], dtype=np.float64)
                pk = round(prev_tow, 1)
                spp_fd_vel: np.ndarray | None = None
                if tow_key in spp_lookup and pk in spp_lookup:
                    ddx = spp_lookup[tow_key][:3] - spp_lookup[pk][:3]
                    if np.all(np.isfinite(ddx)):
                        spp_fd_vel = ddx / dt
                if np.all(np.isfinite(spp_pos_pre)):
                    tv, t_rms = estimate_velocity_from_tdcp_with_metrics(
                        spp_pos_pre,
                        prev_measurements,
                        measurements,
                        dt=dt,
                        elevation_weight=tdcp_elevation_weight,
                        el_sin_floor=tdcp_el_sin_floor,
                    )
                    if tv is not None and spp_fd_vel is not None:
                        if float(np.linalg.norm(tv - spp_fd_vel)) > 6.0:
                            tv = None
                    if tv is not None:
                        if predict_guide == "tdcp_adaptive" and t_rms >= tdcp_rms_threshold:
                            # Adaptive mode: postfit RMS too large, fall back
                            n_tdcp_fallback += 1
                        else:
                            velocity = tv
                            used_tdcp = True
                            tdcp_rms = float(t_rms)
                            n_tdcp_used += 1
                elif predict_guide == "tdcp_adaptive":
                    n_tdcp_fallback += 1

            # --- SPP finite-difference fallback ---
            if velocity is None and tow_key in spp_lookup:
                pk = round(prev_tow, 1)
                if pk in spp_lookup:
                    vel = (spp_lookup[tow_key][:3] - spp_lookup[pk][:3]) / dt
                    if np.all(np.isfinite(vel)) and np.linalg.norm(vel) < 50:
                        velocity = vel

        sig_predict = float(sigma_pos)
        if imu_stop_detected and imu_stop_sigma_pos is not None:
            sig_predict = float(imu_stop_sigma_pos)
        elif used_tdcp and sigma_pos_tdcp is not None:
            sig_predict = float(sigma_pos_tdcp)
        if (
            used_tdcp
            and sigma_pos_tdcp_tight is not None
            and np.isfinite(tdcp_rms)
            and tdcp_rms < float(tdcp_tight_rms_max_m)
        ):
            sig_predict = float(sigma_pos_tdcp_tight)

        pf.predict(velocity=velocity, dt=dt, sigma_pos=sig_predict)

        sat_ecef = np.array([m.satellite_ecef for m in measurements])
        pr = np.array([m.corrected_pseudorange for m in measurements])
        w = np.array([m.weight for m in measurements])
        if carrier_anchor:
            carrier_anchor_rows = _select_same_band_carrier_rows(
                measurements,
                pr,
                snr_min=mupf_snr_min,
                elev_min=mupf_elev_min,
                spp_pos_check=np.array(sol_epoch.position_ecef_m[:3], dtype=np.float64),
                wavelength_m=_MUPF_L1_WAVELENGTH_M,
                allowed_system_ids=_MUPF_L1_COMPAT_SYSTEM_IDS,
            )

        # --- Residual-based adaptive downweighting ---
        if residual_downweight:
            spp_pos3 = np.array(sol_epoch.position_ecef_m[:3], dtype=np.float64)
            if np.isfinite(spp_pos3).all() and np.linalg.norm(spp_pos3) > 1e6:
                # Estimate clock bias as median(PR - range)
                ranges = np.linalg.norm(sat_ecef - spp_pos3, axis=1)
                cb_est = float(np.median(pr - ranges))
                for i_m in range(len(measurements)):
                    residual = abs(pr[i_m] - ranges[i_m] - cb_est)
                    w[i_m] *= 1.0 / (1.0 + (residual / residual_threshold) ** 2)

        # --- Pseudorange acceleration downweighting ---
        if pr_accel_downweight:
            for i_m in range(len(measurements)):
                prn = int(getattr(measurements[i_m], "prn", 0))
                cur_pr = float(pr[i_m])
                hist = pr_history.get(prn, [])
                if len(hist) >= 2:
                    accel = abs(cur_pr - 2.0 * hist[-1] + hist[-2])
                    w[i_m] *= 1.0 / (1.0 + (accel / pr_accel_threshold) ** 2)
                # Update history (keep last 2)
                hist.append(cur_pr)
                if len(hist) > 2:
                    hist.pop(0)
                pr_history[prn] = hist

        dd_pr_result = None
        dd_carrier_result = None
        gate_pf_est = None
        gate_ess_ratio = None
        gate_spread_m = None
        need_gate_est = dd_pseudorange or mupf_dd
        need_gate_ess = (
            (dd_pseudorange_gate_ess_min_scale != 1.0 or dd_pseudorange_gate_ess_max_scale != 1.0)
            or (mupf_dd_gate_ess_min_scale != 1.0 or mupf_dd_gate_ess_max_scale != 1.0)
            or (mupf_dd_sigma_ess_low_ratio is not None and mupf_dd_sigma_ess_high_ratio is not None)
            or (mupf_dd_skip_low_support_ess_ratio is not None)
            or (mupf_dd_gate_low_ess_max_ratio is not None)
        )
        need_gate_spread = (
            (dd_pseudorange_gate_spread_min_scale != 1.0 or dd_pseudorange_gate_spread_max_scale != 1.0)
            or (mupf_dd_gate_spread_min_scale != 1.0 or mupf_dd_gate_spread_max_scale != 1.0)
            or (mupf_dd_skip_low_support_max_spread_m is not None)
            or (mupf_dd_gate_low_ess_max_spread_m is not None)
        )
        if need_gate_est:
            gate_pf_est = np.asarray(pf.estimate()[:3], dtype=np.float64)
        if need_gate_ess:
            gate_ess_ratio = pf.get_ess() / float(pf.n_particles)
        if need_gate_spread and gate_pf_est is not None:
            gate_spread_m = pf.get_position_spread(center=gate_pf_est)
        if dd_pseudorange and dd_pr_computer is not None:
            pf_est = gate_pf_est
            dd_pr_gate_scale = 1.0
            if gate_ess_ratio is not None:
                dd_pr_gate_scale *= ess_gate_scale(
                    gate_ess_ratio,
                    min_scale=dd_pseudorange_gate_ess_min_scale,
                    max_scale=dd_pseudorange_gate_ess_max_scale,
                )
            if gate_spread_m is not None:
                dd_pr_gate_scale *= spread_gate_scale(
                    gate_spread_m,
                    low_spread_m=dd_pseudorange_gate_low_spread_m,
                    high_spread_m=dd_pseudorange_gate_high_spread_m,
                    min_scale=dd_pseudorange_gate_spread_min_scale,
                    max_scale=dd_pseudorange_gate_spread_max_scale,
                )
            dd_pr_result = dd_pr_computer.compute_dd(
                tow,
                measurements,
                pf_est,
                rover_weights=w,
            )
            if dd_pr_result is not None:
                dd_pr_input_pairs = int(dd_pr_result.n_dd)
                if collect_epoch_diagnostics and dd_pr_result.n_dd > 0:
                    dd_pr_abs_res = np.abs(dd_pseudorange_residuals_m(dd_pr_result, pf_est))
                    dd_pr_raw_abs_res_median_m = float(np.median(dd_pr_abs_res))
                    dd_pr_raw_abs_res_max_m = float(np.max(dd_pr_abs_res))
            if dd_pr_result is not None and dd_pr_result.n_dd >= 3:
                dd_pr_result, dd_pr_gate_stats = gate_dd_pseudorange(
                    dd_pr_result,
                    pf_est,
                    pair_residual_max_m=dd_pseudorange_gate_residual_m,
                    adaptive_pair_floor_m=dd_pseudorange_gate_adaptive_floor_m,
                    adaptive_pair_mad_mult=dd_pseudorange_gate_adaptive_mad_mult,
                    epoch_median_residual_max_m=dd_pseudorange_gate_epoch_median_m,
                    threshold_scale=dd_pr_gate_scale,
                    min_pairs=3,
                )
                n_dd_pr_gate_pairs_rejected += dd_pr_gate_stats.n_pair_rejected
                if dd_pr_gate_stats.rejected_by_epoch:
                    n_dd_pr_gate_epoch_skip += 1

        if dd_pr_result is not None and dd_pr_result.n_dd >= 3:
            pf.update_dd_pseudorange(dd_pr_result, sigma_pr=dd_pseudorange_sigma)
            n_dd_pr_used += 1
        else:
            if dd_pseudorange:
                n_dd_pr_skip += 1
            pf.correct_clock_bias(sat_ecef, pr)
            if use_gmm:
                pf.update_gmm(sat_ecef, pr, weights=w,
                              w_los=gmm_w_los, mu_nlos=gmm_mu_nlos, sigma_nlos=gmm_sigma_nlos)
            else:
                pf.update(sat_ecef, pr, weights=w)

        # --- MUPF: carrier phase AFV update (after pseudorange) ---
        if mupf:
            # Collect carrier phase from gnssplusplus measurements
            # Filter: only use high-quality satellites (C/N0 + elevation)
            cp_cycles = []
            cp_sat_ecef = []
            cp_weights = []
            for m in measurements:
                cp = float(getattr(m, "carrier_phase", 0.0))
                if cp == 0.0 or not np.isfinite(cp) or abs(cp) < 1e3:
                    continue
                snr = float(getattr(m, "snr", 0.0))
                elev = float(getattr(m, "elevation", 0.0))
                # Skip low SNR (likely NLOS/multipath)
                if snr < mupf_snr_min and snr > 0:
                    continue
                # Skip low elevation (likely NLOS)
                if 0 < elev < mupf_elev_min:
                    continue
                # Also check pseudorange residual — large residual = likely NLOS
                spp_pos_check = np.array(sol_epoch.position_ecef_m[:3], dtype=np.float64)
                sat_pos = np.array(m.satellite_ecef, dtype=np.float64)
                if np.isfinite(spp_pos_check).all() and np.linalg.norm(spp_pos_check) > 1e6:
                    rng = np.linalg.norm(sat_pos - spp_pos_check)
                    cb_est_m = float(np.median(pr - np.linalg.norm(sat_ecef - spp_pos_check, axis=1)))
                    res = abs(m.corrected_pseudorange - rng - cb_est_m)
                    if res > 30.0:  # large residual = NLOS
                        continue
                cp_cycles.append(cp)
                cp_sat_ecef.append(m.satellite_ecef)
                cp_weights.append(m.weight)
            if len(cp_cycles) >= 4:
                cp_sat = np.array(cp_sat_ecef, dtype=np.float64)
                cp_arr = np.array(cp_cycles, dtype=np.float64)
                cp_w = np.array(cp_weights, dtype=np.float64)
                # Multi-step carrier phase AFV: progressively tighten sigma
                # Step 1: very loose (sigma=2.0 cycles ≈ 38cm) → coarse narrowing
                # Step 2: medium (sigma=0.5 cycles ≈ 10cm) → medium narrowing
                # Step 3: tight (sigma=target) → final precision
                mupf_sigmas = [2.0, 0.5, mupf_sigma_cycles]
                for sig in mupf_sigmas:
                    pf.resample_if_needed()
                    pf.update_carrier_afv(cp_sat, cp_arr, weights=cp_w,
                                          sigma_cycles=sig)

        # --- MUPF-DD: Double-Differenced carrier phase AFV update ---
        if mupf_dd and dd_computer is not None:
            pf_est = gate_pf_est
            dd_cp_gate_scale = 1.0
            if gate_ess_ratio is not None:
                dd_cp_gate_scale *= ess_gate_scale(
                    gate_ess_ratio,
                    min_scale=mupf_dd_gate_ess_min_scale,
                    max_scale=mupf_dd_gate_ess_max_scale,
                )
            if gate_spread_m is not None:
                dd_cp_gate_scale *= spread_gate_scale(
                    gate_spread_m,
                    low_spread_m=mupf_dd_gate_low_spread_m,
                    high_spread_m=mupf_dd_gate_high_spread_m,
                    min_scale=mupf_dd_gate_spread_min_scale,
                    max_scale=mupf_dd_gate_spread_max_scale,
                )
            dd_result = dd_computer.compute_dd(tow, measurements, pf_est)
            if dd_result is not None:
                dd_cp_input_pairs = int(dd_result.n_dd)
                if (
                    (
                        collect_epoch_diagnostics
                        or mupf_dd_skip_low_support_min_raw_afv_median_cycles is not None
                        or (
                            mupf_dd_sigma_afv_good_cycles is not None
                            and mupf_dd_sigma_afv_bad_cycles is not None
                        )
                    )
                    and dd_result.n_dd > 0
                ):
                    dd_cp_abs_afv = np.abs(dd_carrier_afv_cycles(dd_result, pf_est))
                    dd_cp_raw_abs_afv_median_cycles = float(np.median(dd_cp_abs_afv))
                    dd_cp_raw_abs_afv_max_cycles = float(np.max(dd_cp_abs_afv))
            if dd_result is not None and dd_result.n_dd >= 3:
                dd_cp_epoch_median_cycles = _effective_dd_carrier_epoch_median_gate(
                    dd_pr_result,
                    base_epoch_median_cycles=mupf_dd_gate_epoch_median_cycles,
                    ess_ratio=gate_ess_ratio,
                    spread_m=gate_spread_m,
                    low_ess_epoch_median_cycles=mupf_dd_gate_low_ess_epoch_median_cycles,
                    low_ess_max_ratio=mupf_dd_gate_low_ess_max_ratio,
                    low_ess_max_spread_m=mupf_dd_gate_low_ess_max_spread_m,
                    low_ess_require_no_dd_pr=mupf_dd_gate_low_ess_require_no_dd_pr,
                )
                dd_result, dd_gate_stats = gate_dd_carrier(
                    dd_result,
                    pf_est,
                    pair_afv_max_cycles=mupf_dd_gate_afv_cycles,
                    adaptive_pair_floor_cycles=mupf_dd_gate_adaptive_floor_cycles,
                    adaptive_pair_mad_mult=mupf_dd_gate_adaptive_mad_mult,
                    epoch_median_afv_max_cycles=dd_cp_epoch_median_cycles,
                    threshold_scale=dd_cp_gate_scale,
                    min_pairs=3,
                )
                n_dd_gate_pairs_rejected += dd_gate_stats.n_pair_rejected
                if dd_gate_stats.rejected_by_epoch:
                    n_dd_gate_epoch_skip += 1
            if dd_result is not None and dd_result.n_dd >= 3:
                if _should_skip_low_support_dd_carrier(
                    dd_result,
                    dd_pr_result,
                    ess_ratio=gate_ess_ratio,
                    spread_m=gate_spread_m,
                    raw_afv_median_cycles=dd_cp_raw_abs_afv_median_cycles,
                    low_support_ess_ratio=mupf_dd_skip_low_support_ess_ratio,
                    low_support_max_pairs=mupf_dd_skip_low_support_max_pairs,
                    low_support_max_spread_m=mupf_dd_skip_low_support_max_spread_m,
                    low_support_min_raw_afv_median_cycles=mupf_dd_skip_low_support_min_raw_afv_median_cycles,
                    low_support_require_no_dd_pr=mupf_dd_skip_low_support_require_no_dd_pr,
                ):
                    dd_result = None
                    dd_cp_support_skip = True
                    n_dd_skip_support_guard += 1
            if (
                dd_result is not None
                and dd_result.n_dd >= 3
                and _should_replace_weak_dd_with_fallback(
                    dd_result,
                    dd_pr_result,
                    raw_afv_median_cycles=dd_cp_raw_abs_afv_median_cycles,
                    ess_ratio=gate_ess_ratio,
                    weak_dd_max_pairs=mupf_dd_fallback_weak_dd_max_pairs,
                    weak_dd_max_ess_ratio=mupf_dd_fallback_weak_dd_max_ess_ratio,
                    weak_dd_min_raw_afv_median_cycles=mupf_dd_fallback_weak_dd_min_raw_afv_median_cycles,
                    weak_dd_require_no_dd_pr=mupf_dd_fallback_weak_dd_require_no_dd_pr,
                )
            ):
                replacement_attempt = _prepare_dd_carrier_undiff_fallback(
                    measurements,
                    sat_ecef,
                    pr,
                    np.array(sol_epoch.position_ecef_m[:3], dtype=np.float64),
                    carrier_bias_tracker,
                    carrier_anchor_rows,
                    np.asarray(pf.estimate(), dtype=np.float64),
                    tow,
                    enabled=mupf_dd_fallback_undiff,
                    mupf_enabled=mupf,
                    dd_carrier_result=dd_result,
                    used_carrier_anchor=False,
                    snr_min=mupf_snr_min,
                    elev_min=mupf_elev_min,
                    fallback_sigma_cycles=mupf_dd_fallback_sigma_cycles,
                    fallback_min_sats=mupf_dd_fallback_min_sats,
                    prefer_tracked=mupf_dd_fallback_prefer_tracked,
                    tracked_min_stable_epochs=mupf_dd_fallback_tracked_min_stable_epochs,
                    tracked_min_sats=mupf_dd_fallback_tracked_min_sats,
                    tracked_continuity_good_m=mupf_dd_fallback_tracked_continuity_good_m,
                    tracked_continuity_bad_m=mupf_dd_fallback_tracked_continuity_bad_m,
                    tracked_sigma_min_scale=mupf_dd_fallback_tracked_sigma_min_scale,
                    tracked_sigma_max_scale=mupf_dd_fallback_tracked_sigma_max_scale,
                    max_age_s=carrier_anchor_max_age_s,
                    max_continuity_residual_m=carrier_anchor_max_continuity_residual_m,
                    allow_weak_dd=True,
                    weak_dd_max_pairs=int(getattr(dd_result, "n_dd", 0)),
                )
                if replacement_attempt.afv is not None and replacement_attempt.sigma_cycles is not None:
                    fallback_attempt = _apply_dd_carrier_undiff_fallback(pf, replacement_attempt)
                    dd_result = None
            if dd_result is not None and dd_result.n_dd >= 3:
                if (
                    mupf_dd_sigma_support_low_pairs is not None
                    and mupf_dd_sigma_support_high_pairs is not None
                    and mupf_dd_sigma_support_max_scale > 1.0
                ):
                    dd_cp_sigma_support_scale = pair_count_sigma_scale(
                        int(dd_result.n_dd),
                        low_pairs=int(mupf_dd_sigma_support_low_pairs),
                        high_pairs=int(mupf_dd_sigma_support_high_pairs),
                        max_scale=mupf_dd_sigma_support_max_scale,
                    )
                if (
                    mupf_dd_sigma_afv_good_cycles is not None
                    and mupf_dd_sigma_afv_bad_cycles is not None
                    and mupf_dd_sigma_afv_max_scale > 1.0
                    and dd_cp_raw_abs_afv_median_cycles is not None
                ):
                    dd_cp_sigma_afv_scale = metric_sigma_scale(
                        float(dd_cp_raw_abs_afv_median_cycles),
                        good_value=float(mupf_dd_sigma_afv_good_cycles),
                        bad_value=float(mupf_dd_sigma_afv_bad_cycles),
                        max_scale=mupf_dd_sigma_afv_max_scale,
                    )
                if (
                    mupf_dd_sigma_ess_low_ratio is not None
                    and mupf_dd_sigma_ess_high_ratio is not None
                    and mupf_dd_sigma_ess_max_scale > 1.0
                    and gate_ess_ratio is not None
                ):
                    dd_cp_sigma_ess_scale = ess_gate_scale(
                        float(gate_ess_ratio),
                        low_ratio=float(mupf_dd_sigma_ess_low_ratio),
                        high_ratio=float(mupf_dd_sigma_ess_high_ratio),
                        min_scale=float(mupf_dd_sigma_ess_max_scale),
                        max_scale=1.0,
                    )
                dd_cp_sigma_scale = combine_sigma_scales(
                    dd_cp_sigma_support_scale,
                    dd_cp_sigma_afv_scale,
                    dd_cp_sigma_ess_scale,
                    max_scale=mupf_dd_sigma_max_scale,
                )
                dd_cp_sigma_cycles = float(mupf_dd_sigma_cycles) * float(dd_cp_sigma_scale)
                # Resample to concentrate particles before DD-AFV
                pf.resample_if_needed()
                pf.update_dd_carrier_afv(dd_result, sigma_cycles=dd_cp_sigma_cycles)
                dd_carrier_result = dd_result
                n_dd_used += 1
                if dd_cp_sigma_scale > 1.0 + 1e-9:
                    n_dd_sigma_relaxed += 1
                    dd_sigma_scale_sum += float(dd_cp_sigma_scale)
            else:
                n_dd_skip += 1
            if not fallback_attempt.used:
                anchor_attempt = _attempt_carrier_anchor_pseudorange_update(
                    pf,
                    carrier_bias_tracker,
                    carrier_anchor_rows,
                    np.asarray(pf.estimate(), dtype=np.float64),
                    prev_pf_state,
                    velocity,
                    dt,
                    tow,
                    enabled=carrier_anchor,
                    dd_carrier_result=dd_carrier_result,
                    seed_dd_min_pairs=carrier_anchor_seed_dd_min_pairs,
                    sigma_m=carrier_anchor_sigma_m,
                    max_age_s=carrier_anchor_max_age_s,
                    max_residual_m=carrier_anchor_max_residual_m,
                    max_continuity_residual_m=carrier_anchor_max_continuity_residual_m,
                    min_stable_epochs=carrier_anchor_min_stable_epochs,
                    min_sats=carrier_anchor_min_sats,
                )
                if anchor_attempt.used:
                    n_carrier_anchor_used += 1

                fallback_attempt = _attempt_dd_carrier_undiff_fallback(
                    pf,
                    measurements,
                    sat_ecef,
                    pr,
                    np.array(sol_epoch.position_ecef_m[:3], dtype=np.float64),
                    carrier_bias_tracker,
                    carrier_anchor_rows,
                    anchor_attempt.state,
                    tow,
                    enabled=mupf_dd_fallback_undiff,
                    mupf_enabled=mupf,
                    dd_carrier_result=dd_carrier_result,
                    used_carrier_anchor=anchor_attempt.used,
                    snr_min=mupf_snr_min,
                    elev_min=mupf_elev_min,
                    fallback_sigma_cycles=mupf_dd_fallback_sigma_cycles,
                    fallback_min_sats=mupf_dd_fallback_min_sats,
                    prefer_tracked=mupf_dd_fallback_prefer_tracked,
                    tracked_min_stable_epochs=mupf_dd_fallback_tracked_min_stable_epochs,
                    tracked_min_sats=mupf_dd_fallback_tracked_min_sats,
                    tracked_continuity_good_m=mupf_dd_fallback_tracked_continuity_good_m,
                    tracked_continuity_bad_m=mupf_dd_fallback_tracked_continuity_bad_m,
                    tracked_sigma_min_scale=mupf_dd_fallback_tracked_sigma_min_scale,
                    tracked_sigma_max_scale=mupf_dd_fallback_tracked_sigma_max_scale,
                    max_age_s=carrier_anchor_max_age_s,
                    max_continuity_residual_m=carrier_anchor_max_continuity_residual_m,
                )
            elif carrier_anchor and carrier_anchor_rows:
                anchor_attempt.state = np.asarray(pf.estimate(), dtype=np.float64)

            if fallback_attempt.attempted_tracked:
                n_dd_fallback_tracked_attempted += 1
            if fallback_attempt.used:
                n_dd_fallback_undiff_used += 1
            if fallback_attempt.used_tracked:
                n_dd_fallback_tracked_used += 1
            if fallback_attempt.used and fallback_attempt.replaced_weak_dd:
                n_dd_fallback_weak_dd_replaced += 1

        spp_pos = np.array(sol_epoch.position_ecef_m[:3], dtype=np.float64)
        if position_update_sigma is not None:
            if np.isfinite(spp_pos).all() and np.linalg.norm(spp_pos) > 1e6:
                pf.position_update(spp_pos, sigma_pos=position_update_sigma)

        # --- IMU tight-coupling dead-reckoning position update ---
        if (
            imu_tight_coupling
            and prev_pf_estimate is not None
            and imu_velocity is not None
            and np.isfinite(imu_velocity).all()
            and dt > 0
        ):
            imu_predicted_pos = prev_pf_estimate + imu_velocity * dt
            if len(measurements) > 0:
                ranges = np.linalg.norm(sat_ecef - spp_pos, axis=1)
                valid_mask = np.isfinite(ranges) & np.isfinite(pr)
                if np.any(valid_mask):
                    cb_est = float(np.median((pr - ranges)[valid_mask]))
                    residuals = np.abs((pr - ranges - cb_est)[valid_mask])
                    spp_residual_rms = float(np.sqrt(np.mean(residuals**2)))
                else:
                    spp_residual_rms = float("inf")
            else:
                spp_residual_rms = float("inf")

            n_sats = len(measurements)
            if n_sats < 6 or spp_residual_rms > 20.0:
                imu_pu_sigma = 3.0
            elif n_sats < 8 or spp_residual_rms > 10.0:
                imu_pu_sigma = 8.0
            else:
                imu_pu_sigma = 30.0

            if np.all(np.isfinite(imu_predicted_pos)):
                pf.position_update(imu_predicted_pos, sigma_pos=imu_pu_sigma)
                n_imu_tight_used += 1
                used_imu_tight_epoch = True
            else:
                n_imu_tight_skip += 1
        elif imu_tight_coupling:
            n_imu_tight_skip += 1

        # Doppler-predicted position update: propagate previous estimate by velocity
        if doppler_position_update and velocity is not None and prev_estimate is not None and dt > 0:
            doppler_predicted_pos = prev_estimate + velocity * dt
            pf.position_update(doppler_predicted_pos, sigma_pos=doppler_pu_sigma)

        # TDCP displacement position update: cm-level carrier phase constraint
        # This is the key to matching FGO performance without FGO.
        # TDCP velocity * dt = cm-level displacement from carrier phase.
        # Apply as tight position_update when TDCP RMS is good.
        if tdcp_position_update and used_tdcp and prev_estimate is not None and dt > 0:
            if np.isfinite(tdcp_rms) and tdcp_rms < tdcp_pu_rms_max:
                tdcp_displacement = velocity * dt  # velocity here is TDCP velocity
                tdcp_predicted_pos = prev_estimate + tdcp_displacement
                if np.all(np.isfinite(tdcp_predicted_pos)):
                    pf.position_update(tdcp_predicted_pos, sigma_pos=tdcp_pu_sigma)

        if osm_network is not None:
            osm_center = np.asarray(pf.estimate()[:3], dtype=np.float64)
            road_segments = osm_network.candidate_kernel_array(
                osm_center,
                radius_m=osm_road_search_radius_m,
                max_segments=osm_road_max_segments,
            )
            if road_segments.size > 0:
                osm_road_update = {
                    "road_segments_enu": road_segments,
                    "origin_ecef": osm_network.origin_ecef_m,
                    "east_basis": osm_network.east_basis,
                    "north_basis": osm_network.north_basis,
                    "sigma_road_m": float(osm_road_sigma_m),
                    "huber_k": float(osm_road_huber_k),
                }
                pf.osm_road_update(
                    road_segments,
                    osm_network.origin_ecef_m,
                    osm_network.east_basis,
                    osm_network.north_basis,
                    sigma_road_m=osm_road_sigma_m,
                    huber_k=osm_road_huber_k,
                )
                n_osm_road_used += 1
                n_osm_road_segments_sum += int(road_segments.shape[0])

        if use_smoother:
            spp_ref = (
                spp_pos
                if position_update_sigma is not None
                and np.isfinite(spp_pos).all()
                and np.linalg.norm(spp_pos) > 1e6
                else None
            )
            pf.store_epoch(
                sat_ecef,
                pr,
                w,
                velocity,
                dt,
                spp_ref=spp_ref,
                dd_pseudorange=dd_pr_result,
                dd_pseudorange_sigma=(
                    dd_pseudorange_sigma if dd_pr_result is not None and dd_pr_result.n_dd >= 3 else None
                ),
                dd_carrier=dd_carrier_result,
                dd_carrier_sigma=(
                    dd_cp_sigma_cycles
                    if dd_carrier_result is not None and dd_carrier_result.n_dd >= 3
                    else None
                ),
                carrier_anchor_pseudorange=anchor_attempt.update,
                carrier_anchor_sigma=(
                    carrier_anchor_sigma_m if anchor_attempt.update is not None else None
                ),
                carrier_afv=fallback_attempt.afv,
                carrier_afv_sigma=fallback_attempt.sigma_cycles,
                carrier_afv_wavelength=(
                    _MUPF_L1_WAVELENGTH_M if fallback_attempt.afv is not None else None
                ),
                osm_road=osm_road_update,
            )
            n_stored += 1

        pf_estimate_now = np.asarray(pf.estimate()[:3], dtype=np.float64).copy()
        if (
            carrier_anchor
            and carrier_anchor_rows
            and dd_carrier_result is not None
            and dd_carrier_result.n_dd >= int(carrier_anchor_seed_dd_min_pairs)
        ):
            _update_carrier_bias_tracker(
                carrier_bias_tracker,
                carrier_anchor_rows,
                np.asarray(pf.estimate(), dtype=np.float64),
                tow,
                blend_alpha=carrier_anchor_blend_alpha,
                reanchor_jump_cycles=carrier_anchor_reanchor_jump_cycles,
                max_age_s=carrier_anchor_max_age_s,
                trusted=True,
                max_continuity_residual_m=carrier_anchor_max_continuity_residual_m,
            )
        elif carrier_anchor and anchor_attempt.used and anchor_attempt.rows_used:
            _update_carrier_bias_tracker(
                carrier_bias_tracker,
                anchor_attempt.rows_used,
                np.asarray(pf.estimate(), dtype=np.float64),
                tow,
                blend_alpha=carrier_anchor_blend_alpha,
                reanchor_jump_cycles=carrier_anchor_reanchor_jump_cycles,
                max_age_s=carrier_anchor_max_age_s,
                trusted=False,
                max_continuity_residual_m=carrier_anchor_max_continuity_residual_m,
            )
        elif (
            carrier_anchor
            and carrier_anchor_rows
            and anchor_attempt.state is not None
            and not anchor_attempt.used
        ):
            anchor_attempt.propagated_rows = _propagate_carrier_bias_tracker_tdcp(
                carrier_bias_tracker,
                carrier_anchor_rows,
                anchor_attempt.state,
                tow,
                blend_alpha=carrier_anchor_blend_alpha,
                reanchor_jump_cycles=carrier_anchor_reanchor_jump_cycles,
                max_age_s=carrier_anchor_max_age_s,
                max_continuity_residual_m=carrier_anchor_max_continuity_residual_m,
            )
            n_carrier_anchor_propagated += int(anchor_attempt.propagated_rows)

        if epochs_done >= skip_valid_epochs:
            gt_idx = np.argmin(np.abs(our_times - tow))
            if abs(our_times[gt_idx] - tow) < 0.05:
                gt_now = np.asarray(gt[gt_idx], dtype=np.float64).copy()
                forward_aligned.append(pf_estimate_now.copy())
                all_gt.append(gt_now)
                if collect_epoch_diagnostics:
                    aligned_epoch_diagnostics.append({
                        "run": run_name,
                        "tow": float(tow),
                        "aligned_epoch_index": int(len(forward_aligned) - 1),
                        "store_epoch_index": int(n_stored - 1) if use_smoother else None,
                        "gt_index": int(gt_idx),
                        "n_measurements": int(len(measurements)),
                        "used_imu": bool(used_imu),
                        "used_tdcp": bool(used_tdcp),
                        "used_imu_tight": bool(used_imu_tight_epoch),
                        "used_dd_pseudorange": bool(dd_pr_result is not None and dd_pr_result.n_dd >= 3),
                        "used_dd_carrier": bool(dd_carrier_result is not None and dd_carrier_result.n_dd >= 3),
                        "gate_ess_ratio": _finite_float(gate_ess_ratio),
                        "gate_spread_m": _finite_float(gate_spread_m),
                        "dd_pr_input_pairs": int(dd_pr_input_pairs),
                        "dd_pr_kept_pairs": int(dd_pr_gate_stats.n_kept_pairs) if dd_pr_gate_stats is not None else int(dd_pr_input_pairs),
                        "dd_pr_pair_rejected": int(dd_pr_gate_stats.n_pair_rejected) if dd_pr_gate_stats is not None else 0,
                        "dd_pr_epoch_rejected": bool(dd_pr_gate_stats.rejected_by_epoch) if dd_pr_gate_stats is not None else False,
                        "dd_pr_gate_scale": _finite_float(dd_pr_gate_scale),
                        "dd_pr_gate_pair_threshold_m": _finite_float(dd_pr_gate_stats.pair_threshold) if dd_pr_gate_stats is not None else None,
                        "dd_pr_raw_abs_res_median_m": _finite_float(dd_pr_raw_abs_res_median_m),
                        "dd_pr_raw_abs_res_max_m": _finite_float(dd_pr_raw_abs_res_max_m),
                        "dd_pr_kept_abs_res_median_m": _finite_float(dd_pr_gate_stats.metric_median) if dd_pr_gate_stats is not None else None,
                        "dd_pr_kept_abs_res_max_m": _finite_float(dd_pr_gate_stats.metric_max) if dd_pr_gate_stats is not None else None,
                        "dd_cp_input_pairs": int(dd_cp_input_pairs),
                        "dd_cp_kept_pairs": int(dd_gate_stats.n_kept_pairs) if dd_gate_stats is not None else int(dd_cp_input_pairs),
                        "dd_cp_pair_rejected": int(dd_gate_stats.n_pair_rejected) if dd_gate_stats is not None else 0,
                        "dd_cp_epoch_rejected": bool(dd_gate_stats.rejected_by_epoch) if dd_gate_stats is not None else False,
                        "dd_cp_gate_scale": _finite_float(dd_cp_gate_scale),
                        "dd_cp_gate_pair_threshold_cycles": _finite_float(dd_gate_stats.pair_threshold) if dd_gate_stats is not None else None,
                        "dd_cp_raw_abs_afv_median_cycles": _finite_float(dd_cp_raw_abs_afv_median_cycles),
                        "dd_cp_raw_abs_afv_max_cycles": _finite_float(dd_cp_raw_abs_afv_max_cycles),
                        "dd_cp_kept_abs_afv_median_cycles": _finite_float(dd_gate_stats.metric_median) if dd_gate_stats is not None else None,
                        "dd_cp_kept_abs_afv_max_cycles": _finite_float(dd_gate_stats.metric_max) if dd_gate_stats is not None else None,
                        "dd_cp_sigma_support_scale": _finite_float(dd_cp_sigma_support_scale),
                        "dd_cp_sigma_afv_scale": _finite_float(dd_cp_sigma_afv_scale),
                        "dd_cp_sigma_ess_scale": _finite_float(dd_cp_sigma_ess_scale),
                        "dd_cp_sigma_scale": _finite_float(dd_cp_sigma_scale),
                        "dd_cp_sigma_cycles": _finite_float(dd_cp_sigma_cycles),
                        "dd_cp_support_skip": bool(dd_cp_support_skip),
                        "carrier_anchor_propagated_rows": (
                            int(anchor_attempt.propagated_rows)
                        ),
                        "carrier_anchor_n_sat": (
                            int(anchor_attempt.stats["n_sat"]) if anchor_attempt.stats is not None else 0
                        ),
                        "carrier_anchor_sigma_m": (
                            _finite_float(carrier_anchor_sigma_m) if anchor_attempt.update is not None else None
                        ),
                        "carrier_anchor_residual_median_m": (
                            _finite_float(anchor_attempt.stats["residual_median_m"]) if anchor_attempt.stats is not None else None
                        ),
                        "carrier_anchor_residual_max_m": (
                            _finite_float(anchor_attempt.stats["residual_max_m"]) if anchor_attempt.stats is not None else None
                        ),
                        "carrier_anchor_continuity_median_m": (
                            _finite_float(anchor_attempt.stats["continuity_median_m"]) if anchor_attempt.stats is not None else None
                        ),
                        "carrier_anchor_continuity_max_m": (
                            _finite_float(anchor_attempt.stats["continuity_max_m"]) if anchor_attempt.stats is not None else None
                        ),
                        "carrier_anchor_max_age_s": (
                            _finite_float(anchor_attempt.stats["max_age_s"]) if anchor_attempt.stats is not None else None
                        ),
                        "used_carrier_anchor": bool(anchor_attempt.used),
                        "used_dd_carrier_fallback": bool(fallback_attempt.used),
                        "used_dd_carrier_fallback_weak_dd": bool(fallback_attempt.used and fallback_attempt.replaced_weak_dd),
                        "attempted_dd_carrier_fallback_tracked": bool(fallback_attempt.attempted_tracked),
                        "used_dd_carrier_fallback_tracked": bool(fallback_attempt.used_tracked),
                        "dd_carrier_fallback_n_sat": (
                            int(fallback_attempt.afv["n_sat"]) if fallback_attempt.afv is not None else 0
                        ),
                        "dd_carrier_fallback_tracked_candidate_n_sat": (
                            int(fallback_attempt.tracked_stats.get("n_tracked_consistent_sat", fallback_attempt.tracked_stats.get("n_sat", 0)))
                            if fallback_attempt.tracked_stats is not None
                            else 0
                        ),
                        "dd_carrier_fallback_tracked_continuity_median_m": (
                            _finite_float(fallback_attempt.tracked_stats["continuity_median_m"])
                            if fallback_attempt.tracked_stats is not None
                            else None
                        ),
                        "dd_carrier_fallback_tracked_continuity_max_m": (
                            _finite_float(fallback_attempt.tracked_stats["continuity_max_m"])
                            if fallback_attempt.tracked_stats is not None
                            else None
                        ),
                        "dd_carrier_fallback_tracked_stable_epochs_median": (
                            _finite_float(fallback_attempt.tracked_stats["stable_epochs_median"])
                            if fallback_attempt.tracked_stats is not None
                            else None
                        ),
                        "dd_carrier_fallback_sigma_scale": _finite_float(fallback_attempt.sigma_scale),
                        "dd_carrier_fallback_sigma_cycles": _finite_float(fallback_attempt.sigma_cycles),
                        "forward_error_2d": None,
                        "forward_error_3d": None,
                        "smoothed_error_2d": None,
                        "smoothed_error_3d": None,
                        "smoothed_shift_3d_m": None,
                        "smoothing_improvement_2d": None,
                        "tail_guard_applied": False,
                    })
                if use_smoother:
                    aligned_indices.append(n_stored - 1)

        prev_tow = tow
        prev_measurements = list(measurements)
        prev_estimate = pf_estimate_now.copy()
        prev_pf_estimate = pf_estimate_now.copy()
        prev_pf_state = np.asarray(pf.estimate(), dtype=np.float64).copy()
        epochs_done += 1

    elapsed_ms = (time.perf_counter() - t0) * 1000

    if predict_guide == "tdcp_adaptive":
        total_tdcp = n_tdcp_used + n_tdcp_fallback
        print(
            f"  [tdcp_adaptive] TDCP used {n_tdcp_used}/{total_tdcp} epochs, "
            f"fallback {n_tdcp_fallback}/{total_tdcp} (rms_threshold={tdcp_rms_threshold:.1f}m)"
        )

    if predict_guide in ("imu", "imu_spp_blend"):
        total_imu = n_imu_used + n_imu_fallback
        print(
            f"  [{predict_guide}] IMU used {n_imu_used}/{total_imu} epochs, "
            f"fallback {n_imu_fallback}/{total_imu}"
        )
        if n_imu_stop_detected > 0:
            stop_sigma_str = f"{imu_stop_sigma_pos}" if imu_stop_sigma_pos is not None else "default"
            print(
                f"  [imu_stop_detect] stop epochs={n_imu_stop_detected}, "
                f"sigma_pos={stop_sigma_str}"
            )

    if imu_tight_coupling:
        total_tight = n_imu_tight_used + n_imu_tight_skip
        print(
            f"  [imu_tight] IMU position_update used {n_imu_tight_used}/{total_tight} epochs, "
            f"skip {n_imu_tight_skip}/{total_tight}"
        )

    if osm_network is not None:
        mean_segments = (
            n_osm_road_segments_sum / float(n_osm_road_used)
            if n_osm_road_used > 0
            else 0.0
        )
        print(
            f"  [osm_road] used {n_osm_road_used} epochs, "
            f"mean_segments={mean_segments:.1f}, "
            f"sigma={osm_road_sigma_m:.2f}m huber_k={osm_road_huber_k:.2f}"
        )

    if mupf_dd:
        total_dd = n_dd_used + n_dd_skip
        print(
            f"  [mupf_dd] DD-AFV used {n_dd_used}/{total_dd} epochs, "
            f"skip {n_dd_skip}/{total_dd}"
        )
        if (
            mupf_dd_gate_afv_cycles is not None
            or mupf_dd_gate_adaptive_floor_cycles is not None
            or mupf_dd_gate_adaptive_mad_mult is not None
            or mupf_dd_gate_epoch_median_cycles is not None
        ):
            print(
                f"  [mupf_dd_gate] pair_rejected={n_dd_gate_pairs_rejected} "
                f"epoch_skip={n_dd_gate_epoch_skip}"
            )
        if n_dd_skip_support_guard > 0:
            print(f"  [mupf_dd_support_skip] epochs={n_dd_skip_support_guard}")
        if n_dd_sigma_relaxed > 0:
            mean_sigma_scale = dd_sigma_scale_sum / float(n_dd_sigma_relaxed)
            print(
                f"  [mupf_dd_sigma_relax] epochs={n_dd_sigma_relaxed} "
                f"mean_scale={mean_sigma_scale:.3f}"
            )
        if n_carrier_anchor_used > 0:
            print(f"  [carrier_anchor] epochs={n_carrier_anchor_used}")
        if n_carrier_anchor_propagated > 0:
            print(f"  [carrier_anchor_tdcp] propagated_rows={n_carrier_anchor_propagated}")
        if n_dd_fallback_undiff_used > 0:
            print(f"  [mupf_dd_fallback_undiff] epochs={n_dd_fallback_undiff_used}")
        if n_dd_fallback_tracked_attempted > 0:
            print(f"  [mupf_dd_fallback_tracked_attempt] epochs={n_dd_fallback_tracked_attempted}")
        if n_dd_fallback_tracked_used > 0:
            print(f"  [mupf_dd_fallback_tracked] epochs={n_dd_fallback_tracked_used}")
        if n_dd_fallback_weak_dd_replaced > 0:
            print(f"  [mupf_dd_fallback_weak_dd] epochs={n_dd_fallback_weak_dd_replaced}")

    if dd_pseudorange:
        total_dd_pr = n_dd_pr_used + n_dd_pr_skip
        print(
            f"  [dd_pseudorange] used {n_dd_pr_used}/{total_dd_pr} epochs, "
            f"skip {n_dd_pr_skip}/{total_dd_pr}"
        )
        if (
            dd_pseudorange_gate_residual_m is not None
            or dd_pseudorange_gate_adaptive_floor_m is not None
            or dd_pseudorange_gate_adaptive_mad_mult is not None
            or dd_pseudorange_gate_epoch_median_m is not None
        ):
            print(
                f"  [dd_pseudorange_gate] pair_rejected={n_dd_pr_gate_pairs_rejected} "
                f"epoch_skip={n_dd_pr_gate_epoch_skip}"
            )

    forward_pos_full = np.array(forward_aligned, dtype=np.float64)
    gt_arr = np.array(all_gt, dtype=np.float64)

    result: dict[str, object] = {
        "run": run_name,
        "n_particles": n_particles,
        "predict_guide": predict_guide,
        "position_update_sigma": position_update_sigma,
        "use_smoother": use_smoother,
        "skip_valid_epochs": skip_valid_epochs,
        "sigma_pos_tdcp": sigma_pos_tdcp,
        "sigma_pos_tdcp_tight": sigma_pos_tdcp_tight,
        "tdcp_tight_rms_max_m": tdcp_tight_rms_max_m,
        "tdcp_elevation_weight": tdcp_elevation_weight,
        "tdcp_el_sin_floor": tdcp_el_sin_floor,
        "tdcp_rms_threshold": tdcp_rms_threshold,
        "doppler_position_update": doppler_position_update,
        "doppler_pu_sigma": doppler_pu_sigma,
        "dd_pseudorange": dd_pseudorange,
        "dd_pseudorange_sigma": dd_pseudorange_sigma,
        "dd_pseudorange_base_interp": dd_pseudorange_base_interp,
        "dd_pseudorange_gate_residual_m": dd_pseudorange_gate_residual_m,
        "dd_pseudorange_gate_adaptive_floor_m": dd_pseudorange_gate_adaptive_floor_m,
        "dd_pseudorange_gate_adaptive_mad_mult": dd_pseudorange_gate_adaptive_mad_mult,
        "dd_pseudorange_gate_epoch_median_m": dd_pseudorange_gate_epoch_median_m,
        "dd_pseudorange_gate_ess_min_scale": dd_pseudorange_gate_ess_min_scale,
        "dd_pseudorange_gate_ess_max_scale": dd_pseudorange_gate_ess_max_scale,
        "dd_pseudorange_gate_spread_min_scale": dd_pseudorange_gate_spread_min_scale,
        "dd_pseudorange_gate_spread_max_scale": dd_pseudorange_gate_spread_max_scale,
        "dd_pseudorange_gate_low_spread_m": dd_pseudorange_gate_low_spread_m,
        "dd_pseudorange_gate_high_spread_m": dd_pseudorange_gate_high_spread_m,
        "mupf_dd_base_interp": mupf_dd_base_interp,
        "mupf_dd_gate_afv_cycles": mupf_dd_gate_afv_cycles,
        "mupf_dd_gate_adaptive_floor_cycles": mupf_dd_gate_adaptive_floor_cycles,
        "mupf_dd_gate_adaptive_mad_mult": mupf_dd_gate_adaptive_mad_mult,
        "mupf_dd_gate_epoch_median_cycles": mupf_dd_gate_epoch_median_cycles,
        "mupf_dd_gate_low_ess_epoch_median_cycles": mupf_dd_gate_low_ess_epoch_median_cycles,
        "mupf_dd_gate_low_ess_max_ratio": mupf_dd_gate_low_ess_max_ratio,
        "mupf_dd_gate_low_ess_max_spread_m": mupf_dd_gate_low_ess_max_spread_m,
        "mupf_dd_gate_low_ess_require_no_dd_pr": mupf_dd_gate_low_ess_require_no_dd_pr,
        "mupf_dd_gate_ess_min_scale": mupf_dd_gate_ess_min_scale,
        "mupf_dd_gate_ess_max_scale": mupf_dd_gate_ess_max_scale,
        "mupf_dd_gate_spread_min_scale": mupf_dd_gate_spread_min_scale,
        "mupf_dd_gate_spread_max_scale": mupf_dd_gate_spread_max_scale,
        "mupf_dd_gate_low_spread_m": mupf_dd_gate_low_spread_m,
        "mupf_dd_gate_high_spread_m": mupf_dd_gate_high_spread_m,
        "mupf_dd_sigma_support_low_pairs": mupf_dd_sigma_support_low_pairs,
        "mupf_dd_sigma_support_high_pairs": mupf_dd_sigma_support_high_pairs,
        "mupf_dd_sigma_support_max_scale": mupf_dd_sigma_support_max_scale,
        "mupf_dd_sigma_afv_good_cycles": mupf_dd_sigma_afv_good_cycles,
        "mupf_dd_sigma_afv_bad_cycles": mupf_dd_sigma_afv_bad_cycles,
        "mupf_dd_sigma_afv_max_scale": mupf_dd_sigma_afv_max_scale,
        "mupf_dd_sigma_ess_low_ratio": mupf_dd_sigma_ess_low_ratio,
        "mupf_dd_sigma_ess_high_ratio": mupf_dd_sigma_ess_high_ratio,
        "mupf_dd_sigma_ess_max_scale": mupf_dd_sigma_ess_max_scale,
        "mupf_dd_sigma_max_scale": mupf_dd_sigma_max_scale,
        "carrier_anchor": carrier_anchor,
        "carrier_anchor_sigma_m": carrier_anchor_sigma_m,
        "carrier_anchor_min_sats": carrier_anchor_min_sats,
        "carrier_anchor_max_age_s": carrier_anchor_max_age_s,
        "carrier_anchor_max_residual_m": carrier_anchor_max_residual_m,
        "carrier_anchor_max_continuity_residual_m": carrier_anchor_max_continuity_residual_m,
        "carrier_anchor_min_stable_epochs": carrier_anchor_min_stable_epochs,
        "carrier_anchor_blend_alpha": carrier_anchor_blend_alpha,
        "carrier_anchor_reanchor_jump_cycles": carrier_anchor_reanchor_jump_cycles,
        "carrier_anchor_seed_dd_min_pairs": carrier_anchor_seed_dd_min_pairs,
        "mupf_dd_fallback_undiff": mupf_dd_fallback_undiff,
        "mupf_dd_fallback_sigma_cycles": mupf_dd_fallback_sigma_cycles,
        "mupf_dd_fallback_min_sats": mupf_dd_fallback_min_sats,
        "mupf_dd_fallback_prefer_tracked": mupf_dd_fallback_prefer_tracked,
        "mupf_dd_fallback_tracked_min_stable_epochs": mupf_dd_fallback_tracked_min_stable_epochs,
        "mupf_dd_fallback_tracked_min_sats": mupf_dd_fallback_tracked_min_sats,
        "mupf_dd_fallback_tracked_continuity_good_m": mupf_dd_fallback_tracked_continuity_good_m,
        "mupf_dd_fallback_tracked_continuity_bad_m": mupf_dd_fallback_tracked_continuity_bad_m,
        "mupf_dd_fallback_tracked_sigma_min_scale": mupf_dd_fallback_tracked_sigma_min_scale,
        "mupf_dd_fallback_tracked_sigma_max_scale": mupf_dd_fallback_tracked_sigma_max_scale,
        "mupf_dd_fallback_weak_dd_max_pairs": mupf_dd_fallback_weak_dd_max_pairs,
        "mupf_dd_fallback_weak_dd_max_ess_ratio": mupf_dd_fallback_weak_dd_max_ess_ratio,
        "mupf_dd_fallback_weak_dd_min_raw_afv_median_cycles": mupf_dd_fallback_weak_dd_min_raw_afv_median_cycles,
        "mupf_dd_fallback_weak_dd_require_no_dd_pr": mupf_dd_fallback_weak_dd_require_no_dd_pr,
        "mupf_dd_skip_low_support_ess_ratio": mupf_dd_skip_low_support_ess_ratio,
        "mupf_dd_skip_low_support_max_pairs": mupf_dd_skip_low_support_max_pairs,
        "mupf_dd_skip_low_support_max_spread_m": mupf_dd_skip_low_support_max_spread_m,
        "mupf_dd_skip_low_support_min_raw_afv_median_cycles": mupf_dd_skip_low_support_min_raw_afv_median_cycles,
        "mupf_dd_skip_low_support_require_no_dd_pr": mupf_dd_skip_low_support_require_no_dd_pr,
        "n_dd_pr_used": n_dd_pr_used,
        "n_dd_pr_skip": n_dd_pr_skip,
        "n_dd_pr_gate_pairs_rejected": n_dd_pr_gate_pairs_rejected,
        "n_dd_pr_gate_epoch_skip": n_dd_pr_gate_epoch_skip,
        "n_dd_used": n_dd_used,
        "n_dd_skip": n_dd_skip,
        "n_dd_gate_pairs_rejected": n_dd_gate_pairs_rejected,
        "n_dd_gate_epoch_skip": n_dd_gate_epoch_skip,
        "n_dd_skip_support_guard": n_dd_skip_support_guard,
        "n_dd_sigma_relaxed": n_dd_sigma_relaxed,
        "mean_dd_sigma_scale": (
            dd_sigma_scale_sum / float(n_dd_sigma_relaxed) if n_dd_sigma_relaxed > 0 else None
        ),
        "n_carrier_anchor_used": n_carrier_anchor_used,
        "n_carrier_anchor_propagated": n_carrier_anchor_propagated,
        "n_dd_fallback_undiff_used": n_dd_fallback_undiff_used,
        "n_dd_fallback_tracked_attempted": n_dd_fallback_tracked_attempted,
        "n_dd_fallback_tracked_used": n_dd_fallback_tracked_used,
        "n_dd_fallback_weak_dd_replaced": n_dd_fallback_weak_dd_replaced,
        "imu_tight_coupling": imu_tight_coupling,
        "n_imu_tight_used": n_imu_tight_used,
        "n_imu_tight_skip": n_imu_tight_skip,
        "n_tdcp_used": n_tdcp_used,
        "n_tdcp_fallback": n_tdcp_fallback,
        "n_imu_used": n_imu_used,
        "n_imu_fallback": n_imu_fallback,
        "n_imu_stop_detected": n_imu_stop_detected,
        "osm_map_constraint": osm_map_constraint,
        "osm_road_file": str(osm_road_file) if osm_road_file is not None else None,
        "osm_road_sigma_m": osm_road_sigma_m,
        "osm_road_huber_k": osm_road_huber_k,
        "osm_road_search_radius_m": osm_road_search_radius_m,
        "osm_road_max_segments": osm_road_max_segments,
        "n_osm_road_used": n_osm_road_used,
        "mean_osm_road_segments": (
            n_osm_road_segments_sum / float(n_osm_road_used)
            if n_osm_road_used > 0
            else None
        ),
        "elapsed_ms": elapsed_ms,
        "forward_metrics": None,
        "smoothed_metrics": None,
        "epoch_diagnostics": None,
        "n_tail_guard_applied": 0,
    }

    if len(forward_pos_full) == 0:
        return result

    result["forward_metrics"] = compute_metrics(forward_pos_full, gt_arr[: len(forward_pos_full)])
    if collect_epoch_diagnostics and aligned_epoch_diagnostics:
        forward_errors_2d, forward_errors_3d = ecef_errors_2d_3d(
            forward_pos_full,
            gt_arr[: len(forward_pos_full)],
        )
        for row, err2d, err3d in zip(aligned_epoch_diagnostics, forward_errors_2d, forward_errors_3d):
            row["forward_error_2d"] = float(err2d)
            row["forward_error_3d"] = float(err3d)
        result["epoch_diagnostics"] = aligned_epoch_diagnostics

    if use_smoother and n_stored > 0:
        smoothed_full, _forward_stored = pf.smooth(
            position_update_sigma=position_update_sigma,
        )
        if aligned_indices:
            idx = np.asarray(aligned_indices, dtype=np.int64)
            smoothed_aligned_arr = smoothed_full[idx]
            smoothed_aligned_arr, n_tail_guard_applied = _apply_smoother_tail_guard(
                smoothed_aligned_arr,
                forward_pos_full[: len(smoothed_aligned_arr)],
                aligned_epoch_diagnostics if collect_epoch_diagnostics else None,
                ess_max_ratio=smoother_tail_guard_ess_max_ratio,
                dd_carrier_max_pairs=smoother_tail_guard_dd_carrier_max_pairs,
                dd_pseudorange_max_pairs=smoother_tail_guard_dd_pseudorange_max_pairs,
                min_shift_m=smoother_tail_guard_min_shift_m,
            )
            result["n_tail_guard_applied"] = int(n_tail_guard_applied)
            result["smoothed_metrics"] = compute_metrics(
                smoothed_aligned_arr, gt_arr[: len(smoothed_aligned_arr)]
            )
            if collect_epoch_diagnostics and aligned_epoch_diagnostics:
                smoothed_errors_2d, smoothed_errors_3d = ecef_errors_2d_3d(
                    smoothed_aligned_arr,
                    gt_arr[: len(smoothed_aligned_arr)],
                )
                for row, err2d, err3d in zip(aligned_epoch_diagnostics, smoothed_errors_2d, smoothed_errors_3d):
                    row["smoothed_error_2d"] = float(err2d)
                    row["smoothed_error_3d"] = float(err3d)
                    if row["forward_error_2d"] is not None:
                        row["smoothing_improvement_2d"] = float(row["forward_error_2d"] - err2d)

    return result


def _expand_cli_preset_argv(argv: list[str]) -> list[str]:
    """Inline preset argv fragments so later user flags keep normal precedence."""

    expanded: list[str] = []
    i = 0
    while i < len(argv):
        token = argv[i]
        if token == "--preset":
            if i + 1 >= len(argv):
                raise ValueError("--preset requires a preset name")
            preset_name = argv[i + 1]
            preset = _CLI_PRESETS.get(preset_name)
            if preset is None:
                known = ", ".join(sorted(_CLI_PRESETS))
                raise ValueError(f"unknown preset '{preset_name}' (known: {known})")
            expanded.extend(preset["argv"])
            i += 2
            continue
        if token.startswith("--preset="):
            preset_name = token.split("=", 1)[1]
            preset = _CLI_PRESETS.get(preset_name)
            if preset is None:
                known = ", ".join(sorted(_CLI_PRESETS))
                raise ValueError(f"unknown preset '{preset_name}' (known: {known})")
            expanded.extend(preset["argv"])
            i += 1
            continue
        if token == "--list-presets":
            i += 1
            continue
        expanded.append(token)
        i += 1
    return expanded


def _print_cli_presets() -> None:
    print("Available presets:")
    for name in sorted(_CLI_PRESETS):
        print(f"  {name}: {_CLI_PRESETS[name]['description']}")


def _namespace_requests_epoch_diagnostics(args: argparse.Namespace) -> bool:
    return (
        args.epoch_diagnostics_out is not None
        or args.epoch_diagnostics_top_k > 0
        or args.smoother_tail_guard_ess_max_ratio is not None
        or args.smoother_tail_guard_dd_carrier_max_pairs is not None
        or args.smoother_tail_guard_dd_pseudorange_max_pairs is not None
        or args.smoother_tail_guard_min_shift_m is not None
    )


def _namespace_to_run_kwargs(
    args: argparse.Namespace,
    *,
    position_update_sigma: float | None,
    use_smoother: bool,
) -> dict[str, object]:
    return {
        "n_particles": args.n_particles,
        "sigma_pos": args.sigma_pos,
        "sigma_pr": args.sigma_pr,
        "position_update_sigma": position_update_sigma,
        "predict_guide": args.predict_guide,
        "use_smoother": use_smoother,
        "rover_source": args.urban_rover,
        "max_epochs": args.max_epochs,
        "skip_valid_epochs": args.skip_valid_epochs,
        "sigma_pos_tdcp": args.sigma_pos_tdcp,
        "sigma_pos_tdcp_tight": args.sigma_pos_tdcp_tight,
        "tdcp_tight_rms_max_m": args.tdcp_tight_rms_max,
        "tdcp_elevation_weight": args.tdcp_elevation_weight,
        "tdcp_el_sin_floor": args.tdcp_el_sin_floor,
        "tdcp_rms_threshold": args.tdcp_rms_threshold,
        "residual_downweight": args.residual_downweight,
        "residual_threshold": args.residual_threshold,
        "pr_accel_downweight": args.pr_accel_downweight,
        "pr_accel_threshold": args.pr_accel_threshold,
        "use_gmm": args.gmm,
        "gmm_w_los": args.gmm_w_los,
        "gmm_mu_nlos": args.gmm_mu_nlos,
        "gmm_sigma_nlos": args.gmm_sigma_nlos,
        "doppler_position_update": args.doppler_position_update,
        "doppler_pu_sigma": args.doppler_pu_sigma,
        "imu_tight_coupling": args.imu_tight_coupling,
        "imu_stop_sigma_pos": args.imu_stop_sigma_pos,
        "tdcp_position_update": args.tdcp_position_update,
        "tdcp_pu_sigma": args.tdcp_pu_sigma,
        "tdcp_pu_rms_max": args.tdcp_pu_rms_max,
        "mupf": args.mupf,
        "mupf_sigma_cycles": args.mupf_sigma_cycles,
        "mupf_snr_min": args.mupf_snr_min,
        "mupf_elev_min": args.mupf_elev_min,
        "dd_pseudorange": args.dd_pseudorange,
        "dd_pseudorange_sigma": args.dd_pseudorange_sigma,
        "dd_pseudorange_base_interp": args.dd_pseudorange_base_interp,
        "dd_pseudorange_gate_residual_m": args.dd_pseudorange_gate_residual_m,
        "dd_pseudorange_gate_adaptive_floor_m": args.dd_pseudorange_gate_adaptive_floor_m,
        "dd_pseudorange_gate_adaptive_mad_mult": args.dd_pseudorange_gate_adaptive_mad_mult,
        "dd_pseudorange_gate_epoch_median_m": args.dd_pseudorange_gate_epoch_median_m,
        "dd_pseudorange_gate_ess_min_scale": args.dd_pseudorange_gate_ess_min_scale,
        "dd_pseudorange_gate_ess_max_scale": args.dd_pseudorange_gate_ess_max_scale,
        "dd_pseudorange_gate_spread_min_scale": args.dd_pseudorange_gate_spread_min_scale,
        "dd_pseudorange_gate_spread_max_scale": args.dd_pseudorange_gate_spread_max_scale,
        "dd_pseudorange_gate_low_spread_m": args.dd_pseudorange_gate_low_spread_m,
        "dd_pseudorange_gate_high_spread_m": args.dd_pseudorange_gate_high_spread_m,
        "mupf_dd": args.mupf_dd,
        "mupf_dd_sigma_cycles": args.mupf_dd_sigma_cycles,
        "mupf_dd_base_interp": args.mupf_dd_base_interp,
        "mupf_dd_gate_afv_cycles": args.mupf_dd_gate_afv_cycles,
        "mupf_dd_gate_adaptive_floor_cycles": args.mupf_dd_gate_adaptive_floor_cycles,
        "mupf_dd_gate_adaptive_mad_mult": args.mupf_dd_gate_adaptive_mad_mult,
        "mupf_dd_gate_epoch_median_cycles": args.mupf_dd_gate_epoch_median_cycles,
        "mupf_dd_gate_low_ess_epoch_median_cycles": args.mupf_dd_gate_low_ess_epoch_median_cycles,
        "mupf_dd_gate_low_ess_max_ratio": args.mupf_dd_gate_low_ess_max_ratio,
        "mupf_dd_gate_low_ess_max_spread_m": args.mupf_dd_gate_low_ess_max_spread_m,
        "mupf_dd_gate_low_ess_require_no_dd_pr": args.mupf_dd_gate_low_ess_require_no_dd_pr,
        "mupf_dd_gate_ess_min_scale": args.mupf_dd_gate_ess_min_scale,
        "mupf_dd_gate_ess_max_scale": args.mupf_dd_gate_ess_max_scale,
        "mupf_dd_gate_spread_min_scale": args.mupf_dd_gate_spread_min_scale,
        "mupf_dd_gate_spread_max_scale": args.mupf_dd_gate_spread_max_scale,
        "mupf_dd_gate_low_spread_m": args.mupf_dd_gate_low_spread_m,
        "mupf_dd_gate_high_spread_m": args.mupf_dd_gate_high_spread_m,
        "mupf_dd_sigma_support_low_pairs": args.mupf_dd_sigma_support_low_pairs,
        "mupf_dd_sigma_support_high_pairs": args.mupf_dd_sigma_support_high_pairs,
        "mupf_dd_sigma_support_max_scale": args.mupf_dd_sigma_support_max_scale,
        "mupf_dd_sigma_afv_good_cycles": args.mupf_dd_sigma_afv_good_cycles,
        "mupf_dd_sigma_afv_bad_cycles": args.mupf_dd_sigma_afv_bad_cycles,
        "mupf_dd_sigma_afv_max_scale": args.mupf_dd_sigma_afv_max_scale,
        "mupf_dd_sigma_ess_low_ratio": args.mupf_dd_sigma_ess_low_ratio,
        "mupf_dd_sigma_ess_high_ratio": args.mupf_dd_sigma_ess_high_ratio,
        "mupf_dd_sigma_ess_max_scale": args.mupf_dd_sigma_ess_max_scale,
        "mupf_dd_sigma_max_scale": args.mupf_dd_sigma_max_scale,
        "carrier_anchor": args.carrier_anchor,
        "carrier_anchor_sigma_m": args.carrier_anchor_sigma_m,
        "carrier_anchor_min_sats": args.carrier_anchor_min_sats,
        "carrier_anchor_max_age_s": args.carrier_anchor_max_age_s,
        "carrier_anchor_max_residual_m": args.carrier_anchor_max_residual_m,
        "carrier_anchor_max_continuity_residual_m": args.carrier_anchor_max_continuity_residual_m,
        "carrier_anchor_min_stable_epochs": args.carrier_anchor_min_stable_epochs,
        "carrier_anchor_blend_alpha": args.carrier_anchor_blend_alpha,
        "carrier_anchor_reanchor_jump_cycles": args.carrier_anchor_reanchor_jump_cycles,
        "carrier_anchor_seed_dd_min_pairs": args.carrier_anchor_seed_dd_min_pairs,
        "mupf_dd_fallback_undiff": args.mupf_dd_fallback_undiff,
        "mupf_dd_fallback_sigma_cycles": args.mupf_dd_fallback_sigma_cycles,
        "mupf_dd_fallback_min_sats": args.mupf_dd_fallback_min_sats,
        "mupf_dd_fallback_prefer_tracked": args.mupf_dd_fallback_prefer_tracked,
        "mupf_dd_fallback_tracked_min_stable_epochs": args.mupf_dd_fallback_tracked_min_stable_epochs,
        "mupf_dd_fallback_tracked_min_sats": args.mupf_dd_fallback_tracked_min_sats,
        "mupf_dd_fallback_tracked_continuity_good_m": args.mupf_dd_fallback_tracked_continuity_good_m,
        "mupf_dd_fallback_tracked_continuity_bad_m": args.mupf_dd_fallback_tracked_continuity_bad_m,
        "mupf_dd_fallback_tracked_sigma_min_scale": args.mupf_dd_fallback_tracked_sigma_min_scale,
        "mupf_dd_fallback_tracked_sigma_max_scale": args.mupf_dd_fallback_tracked_sigma_max_scale,
        "mupf_dd_fallback_weak_dd_max_pairs": args.mupf_dd_fallback_weak_dd_max_pairs,
        "mupf_dd_fallback_weak_dd_max_ess_ratio": args.mupf_dd_fallback_weak_dd_max_ess_ratio,
        "mupf_dd_fallback_weak_dd_min_raw_afv_median_cycles": args.mupf_dd_fallback_weak_dd_min_raw_afv_median_cycles,
        "mupf_dd_fallback_weak_dd_require_no_dd_pr": args.mupf_dd_fallback_weak_dd_require_no_dd_pr,
        "mupf_dd_skip_low_support_ess_ratio": args.mupf_dd_skip_low_support_ess_ratio,
        "mupf_dd_skip_low_support_max_pairs": args.mupf_dd_skip_low_support_max_pairs,
        "mupf_dd_skip_low_support_max_spread_m": args.mupf_dd_skip_low_support_max_spread_m,
        "mupf_dd_skip_low_support_min_raw_afv_median_cycles": args.mupf_dd_skip_low_support_min_raw_afv_median_cycles,
        "mupf_dd_skip_low_support_require_no_dd_pr": args.mupf_dd_skip_low_support_require_no_dd_pr,
        "collect_epoch_diagnostics": _namespace_requests_epoch_diagnostics(args),
        "smoother_tail_guard_ess_max_ratio": args.smoother_tail_guard_ess_max_ratio,
        "smoother_tail_guard_dd_carrier_max_pairs": args.smoother_tail_guard_dd_carrier_max_pairs,
        "smoother_tail_guard_dd_pseudorange_max_pairs": args.smoother_tail_guard_dd_pseudorange_max_pairs,
        "smoother_tail_guard_min_shift_m": args.smoother_tail_guard_min_shift_m,
        "osm_map_constraint": args.osm_map_constraint,
        "osm_road_file": args.osm_road_file,
        "osm_road_sigma_m": args.osm_road_sigma_m,
        "osm_road_huber_k": args.osm_road_huber_k,
        "osm_road_search_radius_m": args.osm_road_search_radius_m,
        "osm_road_max_segments": args.osm_road_max_segments,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PF + optional forward-backward smoother (gnssplusplus stack)")
    parser.add_argument(
        "--preset",
        action="append",
        default=[],
        help="Apply a named CLI preset before parsing; later flags override preset values",
    )
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="List available CLI presets and exit",
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--runs", type=str, default="Odaiba")
    parser.add_argument("--n-particles", type=int, default=100_000)
    parser.add_argument("--sigma-pos", type=float, default=PF_SIGMA_POS)
    parser.add_argument("--sigma-pr", type=float, default=3.0)
    parser.add_argument(
        "--position-update-sigma",
        type=float,
        default=3.0,
        help="SPP soft constraint (m); use negative to disable",
    )
    parser.add_argument(
        "--predict-guide",
        choices=("spp", "tdcp", "tdcp_adaptive", "imu", "imu_spp_blend"),
        default="spp",
    )
    parser.add_argument("--smoother", action="store_true", help="Enable forward-backward smooth")
    parser.add_argument("--compare-both", action="store_true", help="Run with and without smoother")
    parser.add_argument("--max-epochs", type=int, default=0, help="Limit valid epochs (0 = no limit)")
    parser.add_argument(
        "--skip-valid-epochs",
        type=int,
        default=0,
        help="Process (burn-in) this many valid epochs before recording metrics; "
        "total processed = skip + max-epochs when max-epochs > 0",
    )
    parser.add_argument("--urban-rover", type=str, default="trimble")
    parser.add_argument(
        "--sigma-pos-tdcp",
        type=float,
        default=None,
        help="When TDCP velocity is accepted, use this predict sigma_pos (m); "
        "omit to use --sigma-pos for all epochs",
    )
    parser.add_argument(
        "--sigma-pos-tdcp-tight",
        type=float,
        default=None,
        help="If set and TDCP postfit RMS < --tdcp-tight-rms-max, use this sigma_pos",
    )
    parser.add_argument(
        "--tdcp-tight-rms-max",
        type=float,
        default=1.0e9,
        help="postfit RMS threshold (m) for --sigma-pos-tdcp-tight (default: disabled)",
    )
    parser.add_argument(
        "--tdcp-rms-threshold",
        type=float,
        default=3.0,
        help="Postfit RMS threshold (m) for tdcp_adaptive mode; "
        "epochs with RMS >= threshold fall back to Doppler/random-walk predict",
    )
    parser.add_argument(
        "--tdcp-elevation-weight",
        action="store_true",
        help="WLS row weight sin(el)^2 when measurements expose elevation (TDCP guide only)",
    )
    parser.add_argument(
        "--tdcp-el-sin-floor",
        type=float,
        default=0.1,
        help="Floor on sin(elevation) when --tdcp-elevation-weight is set",
    )
    parser.add_argument(
        "--residual-downweight",
        action="store_true",
        help="Downweight satellites with large SPP residuals (Cauchy-like)",
    )
    parser.add_argument(
        "--residual-threshold",
        type=float,
        default=15.0,
        help="Residual threshold (m) for Cauchy downweighting",
    )
    parser.add_argument(
        "--pr-accel-downweight",
        action="store_true",
        help="Downweight satellites with large pseudorange acceleration (multipath indicator)",
    )
    parser.add_argument(
        "--pr-accel-threshold",
        type=float,
        default=5.0,
        help="PR acceleration threshold (m) for Cauchy downweighting",
    )
    parser.add_argument("--gmm", action="store_true", help="Use GMM likelihood (LOS+NLOS mixture)")
    parser.add_argument("--gmm-w-los", type=float, default=0.7, help="GMM LOS weight")
    parser.add_argument("--gmm-mu-nlos", type=float, default=15.0, help="GMM NLOS mean bias (m)")
    parser.add_argument("--gmm-sigma-nlos", type=float, default=30.0, help="GMM NLOS sigma (m)")
    parser.add_argument(
        "--doppler-position-update",
        action="store_true",
        help="Apply a second position_update using Doppler-predicted position (prev_estimate + velocity*dt)",
    )
    parser.add_argument(
        "--doppler-pu-sigma",
        type=float,
        default=5.0,
        help="Sigma (m) for Doppler-predicted position_update constraint",
    )
    parser.add_argument(
        "--imu-tight-coupling",
        action="store_true",
        help="Apply IMU dead-reckoning position_update after SPP in each epoch",
    )
    parser.add_argument(
        "--imu-stop-sigma-pos",
        type=float,
        default=None,
        help="When IMU detects stop (speed<0.01 m/s), use this predict sigma_pos (m); "
        "omit to use --sigma-pos for stop epochs",
    )
    parser.add_argument(
        "--tdcp-position-update",
        action="store_true",
        help="Apply TDCP displacement as tight position_update (carrier-phase constraint)",
    )
    parser.add_argument("--tdcp-pu-sigma", type=float, default=0.5,
                        help="Sigma for TDCP displacement position_update (m)")
    parser.add_argument("--tdcp-pu-rms-max", type=float, default=3.0,
                        help="Max TDCP postfit RMS to apply displacement PU (m)")
    parser.add_argument("--mupf", action="store_true",
                        help="Multiple Update PF: carrier phase AFV update after pseudorange")
    parser.add_argument("--mupf-sigma-cycles", type=float, default=0.05,
                        help="Carrier phase AFV sigma in cycles (default 0.05 ≈ 1cm)")
    parser.add_argument("--mupf-snr-min", type=float, default=25.0,
                        help="Min C/N0 (dB-Hz) for carrier phase in MUPF")
    parser.add_argument("--mupf-elev-min", type=float, default=0.15,
                        help="Min elevation (rad) for carrier phase in MUPF (~8.6 deg)")
    parser.add_argument("--dd-pseudorange", action="store_true",
                        help="Use DD pseudorange as the primary weight update (requires base station RINEX)")
    parser.add_argument("--dd-pseudorange-sigma", type=float, default=0.75,
                        help="DD pseudorange sigma in meters (default 0.75)")
    parser.add_argument("--dd-pseudorange-base-interp", action="store_true",
                        help="Linearly interpolate 1 Hz base pseudorange to rover epoch before DD formation")
    parser.add_argument("--dd-pseudorange-gate-residual-m", type=float, default=None,
                        help="Drop DD pseudorange pairs whose abs residual exceeds this threshold (m)")
    parser.add_argument("--dd-pseudorange-gate-adaptive-floor-m", type=float, default=None,
                        help="Adaptive DD pseudorange pair gate floor in meters")
    parser.add_argument("--dd-pseudorange-gate-adaptive-mad-mult", type=float, default=None,
                        help="Adaptive DD pseudorange pair gate uses median + k*MAD with this k")
    parser.add_argument("--dd-pseudorange-gate-epoch-median-m", type=float, default=None,
                        help="Skip a DD pseudorange epoch when kept-pair median abs residual exceeds this threshold (m)")
    parser.add_argument("--dd-pseudorange-gate-ess-min-scale", type=float, default=1.0,
                        help="ESS-linked lower multiplier for DD pseudorange gate thresholds")
    parser.add_argument("--dd-pseudorange-gate-ess-max-scale", type=float, default=1.0,
                        help="ESS-linked upper multiplier for DD pseudorange gate thresholds")
    parser.add_argument("--dd-pseudorange-gate-spread-min-scale", type=float, default=1.0,
                        help="Spread-linked lower multiplier for DD pseudorange gate thresholds")
    parser.add_argument("--dd-pseudorange-gate-spread-max-scale", type=float, default=1.0,
                        help="Spread-linked upper multiplier for DD pseudorange gate thresholds")
    parser.add_argument("--dd-pseudorange-gate-low-spread-m", type=float, default=1.5,
                        help="Particle spread below this is treated as tightly converged for DD pseudorange gating")
    parser.add_argument("--dd-pseudorange-gate-high-spread-m", type=float, default=8.0,
                        help="Particle spread above this is treated as diffuse for DD pseudorange gating")
    parser.add_argument("--mupf-dd", action="store_true",
                        help="Use Double-Differenced carrier phase AFV (requires base station RINEX)")
    parser.add_argument("--mupf-dd-sigma-cycles", type=float, default=0.05,
                        help="DD carrier phase AFV sigma in cycles (default 0.05)")
    parser.add_argument("--mupf-dd-base-interp", action="store_true",
                        help="Linearly interpolate 1 Hz base carrier phase to rover epoch before DD formation")
    parser.add_argument("--mupf-dd-gate-afv-cycles", type=float, default=None,
                        help="Drop DD carrier pairs whose abs AFV exceeds this threshold (cycles)")
    parser.add_argument("--mupf-dd-gate-adaptive-floor-cycles", type=float, default=None,
                        help="Adaptive DD carrier pair gate floor in cycles")
    parser.add_argument("--mupf-dd-gate-adaptive-mad-mult", type=float, default=None,
                        help="Adaptive DD carrier pair gate uses median + k*MAD with this k")
    parser.add_argument("--mupf-dd-gate-epoch-median-cycles", type=float, default=None,
                        help="Skip a DD carrier epoch when kept-pair median abs AFV exceeds this threshold (cycles)")
    parser.add_argument("--mupf-dd-gate-low-ess-epoch-median-cycles", type=float, default=None,
                        help="Contextual DD carrier epoch-median AFV limit (cycles) used under low-ESS conditions")
    parser.add_argument("--mupf-dd-gate-low-ess-max-ratio", type=float, default=None,
                        help="Enable contextual DD carrier epoch-median gate when ESS ratio is at or below this threshold")
    parser.add_argument("--mupf-dd-gate-low-ess-max-spread-m", type=float, default=None,
                        help="Require PF spread to stay at or below this threshold when applying the contextual DD carrier epoch-median gate")
    parser.add_argument("--mupf-dd-gate-low-ess-require-no-dd-pr", action="store_true",
                        help="Require DD pseudorange to be absent before applying the contextual DD carrier epoch-median gate")
    parser.add_argument("--mupf-dd-gate-ess-min-scale", type=float, default=1.0,
                        help="ESS-linked lower multiplier for DD carrier gate thresholds")
    parser.add_argument("--mupf-dd-gate-ess-max-scale", type=float, default=1.0,
                        help="ESS-linked upper multiplier for DD carrier gate thresholds")
    parser.add_argument("--mupf-dd-gate-spread-min-scale", type=float, default=1.0,
                        help="Spread-linked lower multiplier for DD carrier gate thresholds")
    parser.add_argument("--mupf-dd-gate-spread-max-scale", type=float, default=1.0,
                        help="Spread-linked upper multiplier for DD carrier gate thresholds")
    parser.add_argument("--mupf-dd-gate-low-spread-m", type=float, default=1.5,
                        help="Particle spread below this is treated as tightly converged for DD carrier gating")
    parser.add_argument("--mupf-dd-gate-high-spread-m", type=float, default=8.0,
                        help="Particle spread above this is treated as diffuse for DD carrier gating")
    parser.add_argument("--mupf-dd-sigma-support-low-pairs", type=int, default=None,
                        help="Relax DD carrier sigma when kept DD pair count is at or below this threshold")
    parser.add_argument("--mupf-dd-sigma-support-high-pairs", type=int, default=None,
                        help="Return DD carrier sigma to baseline when kept DD pair count reaches this threshold")
    parser.add_argument("--mupf-dd-sigma-support-max-scale", type=float, default=1.0,
                        help="Maximum DD carrier sigma multiplier from sparse-support scaling")
    parser.add_argument("--mupf-dd-sigma-afv-good-cycles", type=float, default=None,
                        help="DD carrier raw abs AFV median at or below this keeps sigma at baseline")
    parser.add_argument("--mupf-dd-sigma-afv-bad-cycles", type=float, default=None,
                        help="DD carrier raw abs AFV median at or above this relaxes sigma toward max scale")
    parser.add_argument("--mupf-dd-sigma-afv-max-scale", type=float, default=1.0,
                        help="Maximum DD carrier sigma multiplier from raw AFV scaling")
    parser.add_argument("--mupf-dd-sigma-ess-low-ratio", type=float, default=None,
                        help="ESS ratio at or below this relaxes DD carrier sigma toward the ESS max scale")
    parser.add_argument("--mupf-dd-sigma-ess-high-ratio", type=float, default=None,
                        help="ESS ratio at or above this keeps DD carrier sigma at baseline")
    parser.add_argument("--mupf-dd-sigma-ess-max-scale", type=float, default=1.0,
                        help="Maximum DD carrier sigma multiplier from ESS-linked scaling")
    parser.add_argument("--mupf-dd-sigma-max-scale", type=float, default=None,
                        help="Optional clip on the combined DD carrier sigma multiplier")
    parser.add_argument("--carrier-anchor", action="store_true",
                        help="Seed per-satellite carrier biases from good DD epochs and reuse them as pseudorange-like updates when DD is weak")
    parser.add_argument("--carrier-anchor-sigma-m", type=float, default=0.25,
                        help="Sigma in meters for carrier-bias-conditioned anchored updates")
    parser.add_argument("--carrier-anchor-min-sats", type=int, default=4,
                        help="Minimum anchored carrier satellites required to apply the carrier-anchor update")
    parser.add_argument("--carrier-anchor-max-age-s", type=float, default=3.0,
                        help="Maximum age in seconds for a stored carrier bias before it is ignored")
    parser.add_argument("--carrier-anchor-max-residual-m", type=float, default=0.75,
                        help="Maximum anchored carrier residual in meters before a satellite is rejected")
    parser.add_argument("--carrier-anchor-max-continuity-residual-m", type=float, default=0.50,
                        help="Maximum inter-epoch carrier continuity residual in meters before a satellite is treated as slipped")
    parser.add_argument("--carrier-anchor-min-stable-epochs", type=int, default=1,
                        help="Minimum stable epochs required before a stored carrier bias can be reused")
    parser.add_argument("--carrier-anchor-blend-alpha", type=float, default=0.5,
                        help="EMA blending factor when refreshing a stored carrier bias on a trusted DD epoch")
    parser.add_argument("--carrier-anchor-reanchor-jump-cycles", type=float, default=4.0,
                        help="If refreshed carrier bias jumps by more than this many cycles, replace it instead of blending")
    parser.add_argument("--carrier-anchor-seed-dd-min-pairs", type=int, default=3,
                        help="Minimum kept DD carrier pairs required to trust an epoch for carrier-anchor seeding")
    parser.add_argument("--mupf-dd-fallback-undiff", action="store_true",
                        help="When DD carrier is unavailable, replay a same-band undifferenced carrier AFV update")
    parser.add_argument("--mupf-dd-fallback-sigma-cycles", type=float, default=0.10,
                        help="Sigma for undifferenced carrier AFV fallback used when DD carrier is unavailable")
    parser.add_argument("--mupf-dd-fallback-min-sats", type=int, default=4,
                        help="Minimum same-band carrier satellites required for undifferenced AFV fallback")
    parser.add_argument("--mupf-dd-fallback-prefer-tracked", action="store_true",
                        help="When carrier-anchor tracker is available, prefer tracker-consistent satellites for undifferenced AFV fallback")
    parser.add_argument("--mupf-dd-fallback-tracked-min-stable-epochs", type=int, default=1,
                        help="Minimum tracker stable epochs required before a satellite is eligible for tracked undiff AFV fallback")
    parser.add_argument("--mupf-dd-fallback-tracked-min-sats", type=int, default=None,
                        help="Minimum tracker-consistent satellites required before treating hybrid undiff fallback as tracked-assisted")
    parser.add_argument("--mupf-dd-fallback-tracked-continuity-good-m", type=float, default=None,
                        help="Tracked fallback continuity median at or below this tightens fallback sigma toward the tracked min scale")
    parser.add_argument("--mupf-dd-fallback-tracked-continuity-bad-m", type=float, default=None,
                        help="Tracked fallback continuity median at or above this relaxes fallback sigma toward the tracked max scale")
    parser.add_argument("--mupf-dd-fallback-tracked-sigma-min-scale", type=float, default=1.0,
                        help="Minimum multiplier for tracked undiff fallback sigma when continuity is very good")
    parser.add_argument("--mupf-dd-fallback-tracked-sigma-max-scale", type=float, default=1.0,
                        help="Maximum multiplier for tracked undiff fallback sigma when continuity degrades")
    parser.add_argument("--mupf-dd-fallback-weak-dd-max-pairs", type=int, default=None,
                        help="If set, try undiff carrier fallback before DD carrier update when kept DD carrier pairs are at or below this threshold")
    parser.add_argument("--mupf-dd-fallback-weak-dd-max-ess-ratio", type=float, default=None,
                        help="If set, weak-DD fallback replacement also requires PF ESS ratio to be at or below this threshold")
    parser.add_argument("--mupf-dd-fallback-weak-dd-min-raw-afv-median-cycles", type=float, default=None,
                        help="When weak-DD fallback replacement is enabled, require raw abs AFV median to be at or above this threshold")
    parser.add_argument("--mupf-dd-fallback-weak-dd-require-no-dd-pr", action="store_true",
                        help="Require DD pseudorange to be absent before replacing a weak DD carrier update with undiff fallback")
    parser.add_argument("--mupf-dd-skip-low-support-ess-ratio", type=float, default=None,
                        help="Skip DD carrier update when ESS ratio is at or below this threshold")
    parser.add_argument("--mupf-dd-skip-low-support-max-pairs", type=int, default=None,
                        help="Skip DD carrier update when kept DD carrier pairs are at or below this threshold")
    parser.add_argument("--mupf-dd-skip-low-support-max-spread-m", type=float, default=None,
                        help="Skip DD carrier update only when PF spread is at or below this threshold")
    parser.add_argument("--mupf-dd-skip-low-support-min-raw-afv-median-cycles", type=float, default=None,
                        help="Skip DD carrier update when raw abs AFV median is at or above this threshold")
    parser.add_argument("--mupf-dd-skip-low-support-require-no-dd-pr", action="store_true",
                        help="Require DD pseudorange to be absent when applying DD carrier low-support skip")
    parser.add_argument(
        "--epoch-diagnostics-out",
        type=Path,
        default=None,
        help="Optional CSV path for per-epoch diagnostics (tail analysis)",
    )
    parser.add_argument(
        "--epoch-diagnostics-top-k",
        type=int,
        default=0,
        help="Print the worst K aligned epochs by 2D error using per-epoch diagnostics",
    )
    parser.add_argument(
        "--smoother-tail-guard-ess-max-ratio",
        type=float,
        default=None,
        help="If set, fallback smoothed epochs to forward when ESS ratio is at or below this threshold",
    )
    parser.add_argument(
        "--smoother-tail-guard-dd-carrier-max-pairs",
        type=int,
        default=None,
        help="If set, require DD carrier kept-pair count to be at or below this threshold for smoother tail guard",
    )
    parser.add_argument(
        "--smoother-tail-guard-dd-pseudorange-max-pairs",
        type=int,
        default=None,
        help="If set, require DD pseudorange kept-pair count to be at or below this threshold for smoother tail guard",
    )
    parser.add_argument(
        "--smoother-tail-guard-min-shift-m",
        type=float,
        default=None,
        help="If set, require smoothed-vs-forward 3D shift to be at or above this threshold for smoother tail guard",
    )
    parser.add_argument(
        "--osm-map-constraint",
        action="store_true",
        help="Apply a soft OSM road-centerline Huber constraint to particles",
    )
    parser.add_argument(
        "--osm-road-file",
        type=Path,
        default=None,
        help="GeoJSON file containing OSM road centerlines",
    )
    parser.add_argument(
        "--osm-road-sigma-m",
        type=float,
        default=2.0,
        help="Road-centerline constraint sigma in meters (must be >= 1.0)",
    )
    parser.add_argument(
        "--osm-road-huber-k",
        type=float,
        default=2.0,
        help="Huber k for the road-centerline constraint",
    )
    parser.add_argument(
        "--osm-road-search-radius-m",
        type=float,
        default=80.0,
        help="Candidate road segment search radius around the current PF estimate",
    )
    parser.add_argument(
        "--osm-road-max-segments",
        type=int,
        default=96,
        help="Maximum nearby OSM road segments sent to the GPU per epoch",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if "--list-presets" in raw_argv:
        _print_cli_presets()
        return 0
    parser = build_arg_parser()
    args = parser.parse_args(_expand_cli_preset_argv(raw_argv))

    pos_sigma = args.position_update_sigma
    if pos_sigma < 0:
        pos_sigma = None

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    runs = [r.strip() for r in args.runs.split(",") if r.strip()]
    rows: list[dict[str, object]] = []

    for run_name in runs:
        run_dir = args.data_root / run_name
        print(f"\n{'='*60}\n  {run_name}\n{'='*60}")
        variants: list[tuple[str, bool]] = []
        if args.compare_both:
            variants.append(("forward_only", False))
            variants.append(("with_smoother", True))
        else:
            variants.append(("forward_only" if not args.smoother else "with_smoother", args.smoother))

        for label, use_sm in variants:
            print(
                f"  [{label}] guide={args.predict_guide} PU={pos_sigma} "
                f"sp_tdcp={args.sigma_pos_tdcp} smooth={use_sm}...",
                end=" ",
                flush=True,
            )
            out = run_pf_with_optional_smoother(
                run_dir,
                run_name,
                **_namespace_to_run_kwargs(
                    args,
                    position_update_sigma=pos_sigma,
                    use_smoother=use_sm,
                ),
            )
            fm = out["forward_metrics"]
            sm = out["smoothed_metrics"]
            ep_n = int(fm["n_epochs"]) if fm else 0
            ms_ep = float(out["elapsed_ms"]) / ep_n if ep_n else 0.0
            if fm:
                print(
                    f"FWD P50={fm['p50']:.2f}m RMS={fm['rms_2d']:.2f}m "
                    f"({ep_n} ep, {ms_ep:.2f}ms/ep)"
                )
            if sm:
                print(
                    f"       SMTH P50={sm['p50']:.2f}m RMS={sm['rms_2d']:.2f}m"
                )
                if int(out.get("n_tail_guard_applied", 0)) > 0:
                    print(f"       tail guard applied: {int(out['n_tail_guard_applied'])} epochs")
            epoch_diagnostics = out.get("epoch_diagnostics")
            if epoch_diagnostics:
                if args.epoch_diagnostics_top_k > 0:
                    _print_top_epoch_diagnostics(epoch_diagnostics, args.epoch_diagnostics_top_k)
                if args.epoch_diagnostics_out is not None:
                    diag_path = _diagnostics_output_path(
                        args.epoch_diagnostics_out,
                        run_name=run_name,
                        label=label,
                        multiple_outputs=(len(runs) * len(variants) > 1),
                    )
                    _write_epoch_diagnostics(epoch_diagnostics, diag_path)
                    print(f"       epoch diagnostics: {diag_path}")
            rows.append({
                "run": run_name,
                "variant": label,
                "predict_guide": args.predict_guide,
                "sigma_pos": args.sigma_pos,
                "sigma_pos_tdcp": args.sigma_pos_tdcp,
                "sigma_pos_tdcp_tight": args.sigma_pos_tdcp_tight,
                "tdcp_tight_rms_max_m": args.tdcp_tight_rms_max,
                "skip_valid_epochs": args.skip_valid_epochs,
                "tdcp_elevation_weight": args.tdcp_elevation_weight,
                "tdcp_el_sin_floor": args.tdcp_el_sin_floor,
                "tdcp_rms_threshold": args.tdcp_rms_threshold,
                "residual_downweight": args.residual_downweight,
                "residual_threshold": args.residual_threshold,
                "pr_accel_downweight": args.pr_accel_downweight,
                "pr_accel_threshold": args.pr_accel_threshold,
                "position_update_sigma": pos_sigma if pos_sigma is not None else "off",
                "doppler_position_update": args.doppler_position_update,
                "doppler_pu_sigma": args.doppler_pu_sigma,
                "imu_tight_coupling": args.imu_tight_coupling,
                "imu_stop_sigma_pos": args.imu_stop_sigma_pos,
                "dd_pseudorange": args.dd_pseudorange,
                "dd_pseudorange_sigma": args.dd_pseudorange_sigma,
                "dd_pseudorange_base_interp": args.dd_pseudorange_base_interp,
                "dd_pseudorange_gate_residual_m": args.dd_pseudorange_gate_residual_m,
                "dd_pseudorange_gate_adaptive_floor_m": args.dd_pseudorange_gate_adaptive_floor_m,
                "dd_pseudorange_gate_adaptive_mad_mult": args.dd_pseudorange_gate_adaptive_mad_mult,
                "dd_pseudorange_gate_epoch_median_m": args.dd_pseudorange_gate_epoch_median_m,
                "dd_pseudorange_gate_ess_min_scale": args.dd_pseudorange_gate_ess_min_scale,
                "dd_pseudorange_gate_ess_max_scale": args.dd_pseudorange_gate_ess_max_scale,
                "dd_pseudorange_gate_spread_min_scale": args.dd_pseudorange_gate_spread_min_scale,
                "dd_pseudorange_gate_spread_max_scale": args.dd_pseudorange_gate_spread_max_scale,
                "dd_pseudorange_gate_low_spread_m": args.dd_pseudorange_gate_low_spread_m,
                "dd_pseudorange_gate_high_spread_m": args.dd_pseudorange_gate_high_spread_m,
                "mupf": args.mupf,
                "mupf_dd": args.mupf_dd,
                "mupf_dd_base_interp": args.mupf_dd_base_interp,
                "mupf_dd_gate_afv_cycles": args.mupf_dd_gate_afv_cycles,
                "mupf_dd_gate_adaptive_floor_cycles": args.mupf_dd_gate_adaptive_floor_cycles,
                "mupf_dd_gate_adaptive_mad_mult": args.mupf_dd_gate_adaptive_mad_mult,
                "mupf_dd_gate_epoch_median_cycles": args.mupf_dd_gate_epoch_median_cycles,
                "mupf_dd_gate_low_ess_epoch_median_cycles": args.mupf_dd_gate_low_ess_epoch_median_cycles,
                "mupf_dd_gate_low_ess_max_ratio": args.mupf_dd_gate_low_ess_max_ratio,
                "mupf_dd_gate_low_ess_max_spread_m": args.mupf_dd_gate_low_ess_max_spread_m,
                "mupf_dd_gate_low_ess_require_no_dd_pr": args.mupf_dd_gate_low_ess_require_no_dd_pr,
                "mupf_dd_gate_ess_min_scale": args.mupf_dd_gate_ess_min_scale,
                "mupf_dd_gate_ess_max_scale": args.mupf_dd_gate_ess_max_scale,
                "mupf_dd_gate_spread_min_scale": args.mupf_dd_gate_spread_min_scale,
                "mupf_dd_gate_spread_max_scale": args.mupf_dd_gate_spread_max_scale,
                "mupf_dd_gate_low_spread_m": args.mupf_dd_gate_low_spread_m,
                "mupf_dd_gate_high_spread_m": args.mupf_dd_gate_high_spread_m,
                "mupf_dd_sigma_support_low_pairs": args.mupf_dd_sigma_support_low_pairs,
                "mupf_dd_sigma_support_high_pairs": args.mupf_dd_sigma_support_high_pairs,
                "mupf_dd_sigma_support_max_scale": args.mupf_dd_sigma_support_max_scale,
                "mupf_dd_sigma_afv_good_cycles": args.mupf_dd_sigma_afv_good_cycles,
                "mupf_dd_sigma_afv_bad_cycles": args.mupf_dd_sigma_afv_bad_cycles,
                "mupf_dd_sigma_afv_max_scale": args.mupf_dd_sigma_afv_max_scale,
                "mupf_dd_sigma_ess_low_ratio": args.mupf_dd_sigma_ess_low_ratio,
                "mupf_dd_sigma_ess_high_ratio": args.mupf_dd_sigma_ess_high_ratio,
                "mupf_dd_sigma_ess_max_scale": args.mupf_dd_sigma_ess_max_scale,
                "mupf_dd_sigma_max_scale": args.mupf_dd_sigma_max_scale,
                "carrier_anchor": args.carrier_anchor,
                "carrier_anchor_sigma_m": args.carrier_anchor_sigma_m,
                "carrier_anchor_min_sats": args.carrier_anchor_min_sats,
                "carrier_anchor_max_age_s": args.carrier_anchor_max_age_s,
                "carrier_anchor_max_residual_m": args.carrier_anchor_max_residual_m,
                "carrier_anchor_max_continuity_residual_m": args.carrier_anchor_max_continuity_residual_m,
                "carrier_anchor_min_stable_epochs": args.carrier_anchor_min_stable_epochs,
                "carrier_anchor_blend_alpha": args.carrier_anchor_blend_alpha,
                "carrier_anchor_reanchor_jump_cycles": args.carrier_anchor_reanchor_jump_cycles,
                "carrier_anchor_seed_dd_min_pairs": args.carrier_anchor_seed_dd_min_pairs,
                "mupf_dd_fallback_undiff": args.mupf_dd_fallback_undiff,
                "mupf_dd_fallback_sigma_cycles": args.mupf_dd_fallback_sigma_cycles,
                "mupf_dd_fallback_min_sats": args.mupf_dd_fallback_min_sats,
                "mupf_dd_fallback_prefer_tracked": args.mupf_dd_fallback_prefer_tracked,
                "mupf_dd_fallback_tracked_min_stable_epochs": args.mupf_dd_fallback_tracked_min_stable_epochs,
                "mupf_dd_fallback_tracked_min_sats": args.mupf_dd_fallback_tracked_min_sats,
                "mupf_dd_fallback_weak_dd_max_pairs": args.mupf_dd_fallback_weak_dd_max_pairs,
                "mupf_dd_fallback_weak_dd_max_ess_ratio": args.mupf_dd_fallback_weak_dd_max_ess_ratio,
                "mupf_dd_fallback_weak_dd_min_raw_afv_median_cycles": args.mupf_dd_fallback_weak_dd_min_raw_afv_median_cycles,
                "mupf_dd_fallback_weak_dd_require_no_dd_pr": args.mupf_dd_fallback_weak_dd_require_no_dd_pr,
                "mupf_dd_skip_low_support_ess_ratio": args.mupf_dd_skip_low_support_ess_ratio,
                "mupf_dd_skip_low_support_max_pairs": args.mupf_dd_skip_low_support_max_pairs,
                "mupf_dd_skip_low_support_max_spread_m": args.mupf_dd_skip_low_support_max_spread_m,
                "mupf_dd_skip_low_support_min_raw_afv_median_cycles": args.mupf_dd_skip_low_support_min_raw_afv_median_cycles,
                "mupf_dd_skip_low_support_require_no_dd_pr": args.mupf_dd_skip_low_support_require_no_dd_pr,
                "n_dd_skip_support_guard": int(out.get("n_dd_skip_support_guard", 0)),
                "n_dd_sigma_relaxed": int(out.get("n_dd_sigma_relaxed", 0)),
                "mean_dd_sigma_scale": out.get("mean_dd_sigma_scale"),
                "n_carrier_anchor_used": int(out.get("n_carrier_anchor_used", 0)),
                "n_carrier_anchor_propagated": int(out.get("n_carrier_anchor_propagated", 0)),
                "n_dd_fallback_undiff_used": int(out.get("n_dd_fallback_undiff_used", 0)),
                "n_dd_fallback_tracked_attempted": int(out.get("n_dd_fallback_tracked_attempted", 0)),
                "n_dd_fallback_tracked_used": int(out.get("n_dd_fallback_tracked_used", 0)),
                "n_dd_fallback_weak_dd_replaced": int(out.get("n_dd_fallback_weak_dd_replaced", 0)),
                "smoother_tail_guard_ess_max_ratio": args.smoother_tail_guard_ess_max_ratio,
                "smoother_tail_guard_dd_carrier_max_pairs": args.smoother_tail_guard_dd_carrier_max_pairs,
                "smoother_tail_guard_dd_pseudorange_max_pairs": args.smoother_tail_guard_dd_pseudorange_max_pairs,
                "smoother_tail_guard_min_shift_m": args.smoother_tail_guard_min_shift_m,
                "n_tail_guard_applied": int(out.get("n_tail_guard_applied", 0)),
                "osm_map_constraint": args.osm_map_constraint,
                "osm_road_file": str(args.osm_road_file) if args.osm_road_file is not None else None,
                "osm_road_sigma_m": args.osm_road_sigma_m,
                "osm_road_huber_k": args.osm_road_huber_k,
                "osm_road_search_radius_m": args.osm_road_search_radius_m,
                "osm_road_max_segments": args.osm_road_max_segments,
                "n_osm_road_used": int(out.get("n_osm_road_used", 0)),
                "mean_osm_road_segments": out.get("mean_osm_road_segments"),
                "smoother": use_sm,
                "n_particles": args.n_particles,
                "forward_p50": fm["p50"] if fm else None,
                "forward_p95": fm["p95"] if fm else None,
                "forward_rms_2d": fm["rms_2d"] if fm else None,
                "smoothed_p50": sm["p50"] if sm else None,
                "smoothed_p95": sm["p95"] if sm else None,
                "smoothed_rms_2d": sm["rms_2d"] if sm else None,
                "n_epochs": ep_n,
                "ms_per_epoch": ms_ep,
            })

    out_csv = RESULTS_DIR / "pf_smoother_eval.csv"
    if rows:
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nSaved: {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
