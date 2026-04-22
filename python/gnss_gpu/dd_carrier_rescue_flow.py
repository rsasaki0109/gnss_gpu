"""DD carrier rescue flow orchestration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gnss_gpu.carrier_rescue import (
    CarrierAnchorAttempt,
    CarrierBiasState,
    CarrierFallbackAttempt,
    _apply_dd_carrier_undiff_fallback,
    _attempt_carrier_anchor_pseudorange_update,
    _attempt_dd_carrier_undiff_fallback,
    _prepare_dd_carrier_undiff_fallback,
)
from gnss_gpu.pf_smoother_config import CarrierRescueConfig, MupfConfig


@dataclass(frozen=True)
class WeakDDCarrierFallbackDecision:
    fallback_attempt: CarrierFallbackAttempt
    dd_carrier_result: object | None


@dataclass(frozen=True)
class PostDDCarrierRescueDecision:
    anchor_attempt: CarrierAnchorAttempt
    fallback_attempt: CarrierFallbackAttempt


def apply_weak_dd_carrier_fallback_replacement(
    pf,
    measurements,
    sat_ecef: np.ndarray,
    pseudoranges: np.ndarray,
    spp_position_ecef: np.ndarray,
    tracker: dict[tuple[int, int], CarrierBiasState],
    carrier_rows: dict[tuple[int, int], dict[str, object]],
    carrier_state: np.ndarray,
    tow: float,
    dd_carrier_result,
    mupf: MupfConfig,
    config: CarrierRescueConfig,
) -> WeakDDCarrierFallbackDecision:
    attempt = _prepare_dd_carrier_undiff_fallback(
        measurements,
        sat_ecef,
        pseudoranges,
        np.asarray(spp_position_ecef, dtype=np.float64),
        tracker,
        carrier_rows,
        np.asarray(carrier_state, dtype=np.float64),
        tow,
        enabled=config.fallback_undiff,
        mupf_enabled=mupf.enabled,
        dd_carrier_result=dd_carrier_result,
        used_carrier_anchor=False,
        snr_min=mupf.snr_min,
        elev_min=mupf.elev_min,
        fallback_sigma_cycles=config.fallback_sigma_cycles,
        fallback_min_sats=config.fallback_min_sats,
        prefer_tracked=config.fallback_prefer_tracked,
        tracked_min_stable_epochs=config.fallback_tracked_min_stable_epochs,
        tracked_min_sats=config.fallback_tracked_min_sats,
        tracked_continuity_good_m=config.fallback_tracked_continuity_good_m,
        tracked_continuity_bad_m=config.fallback_tracked_continuity_bad_m,
        tracked_sigma_min_scale=config.fallback_tracked_sigma_min_scale,
        tracked_sigma_max_scale=config.fallback_tracked_sigma_max_scale,
        max_age_s=config.anchor_max_age_s,
        max_continuity_residual_m=config.anchor_max_continuity_residual_m,
        allow_weak_dd=True,
        weak_dd_max_pairs=int(getattr(dd_carrier_result, "n_dd", 0)),
    )
    if attempt.afv is not None and attempt.sigma_cycles is not None:
        attempt = _apply_dd_carrier_undiff_fallback(pf, attempt)
        return WeakDDCarrierFallbackDecision(
            fallback_attempt=attempt,
            dd_carrier_result=None,
        )
    return WeakDDCarrierFallbackDecision(
        fallback_attempt=attempt,
        dd_carrier_result=dd_carrier_result,
    )


def apply_post_dd_carrier_rescue(
    pf,
    measurements,
    sat_ecef: np.ndarray,
    pseudoranges: np.ndarray,
    spp_position_ecef: np.ndarray,
    tracker: dict[tuple[int, int], CarrierBiasState],
    carrier_rows: dict[tuple[int, int], dict[str, object]],
    current_pf_state: np.ndarray,
    prev_pf_state: np.ndarray | None,
    velocity: np.ndarray | None,
    dt: float,
    tow: float,
    dd_carrier_result,
    mupf: MupfConfig,
    config: CarrierRescueConfig,
    *,
    fallback_attempt: CarrierFallbackAttempt | None = None,
) -> PostDDCarrierRescueDecision:
    anchor_attempt = CarrierAnchorAttempt()
    fallback = fallback_attempt if fallback_attempt is not None else CarrierFallbackAttempt()

    current_pf_state = np.asarray(current_pf_state, dtype=np.float64)
    if not fallback.used:
        anchor_attempt = _attempt_carrier_anchor_pseudorange_update(
            pf,
            tracker,
            carrier_rows,
            current_pf_state,
            prev_pf_state,
            velocity,
            dt,
            tow,
            enabled=config.anchor_enabled,
            dd_carrier_result=dd_carrier_result,
            seed_dd_min_pairs=config.anchor_seed_dd_min_pairs,
            sigma_m=config.anchor_sigma_m,
            max_age_s=config.anchor_max_age_s,
            max_residual_m=config.anchor_max_residual_m,
            max_continuity_residual_m=config.anchor_max_continuity_residual_m,
            min_stable_epochs=config.anchor_min_stable_epochs,
            min_sats=config.anchor_min_sats,
        )

        fallback = _attempt_dd_carrier_undiff_fallback(
            pf,
            measurements,
            sat_ecef,
            pseudoranges,
            np.asarray(spp_position_ecef, dtype=np.float64),
            tracker,
            carrier_rows,
            anchor_attempt.state,
            tow,
            enabled=config.fallback_undiff,
            mupf_enabled=mupf.enabled,
            dd_carrier_result=dd_carrier_result,
            used_carrier_anchor=anchor_attempt.used,
            snr_min=mupf.snr_min,
            elev_min=mupf.elev_min,
            fallback_sigma_cycles=config.fallback_sigma_cycles,
            fallback_min_sats=config.fallback_min_sats,
            prefer_tracked=config.fallback_prefer_tracked,
            tracked_min_stable_epochs=config.fallback_tracked_min_stable_epochs,
            tracked_min_sats=config.fallback_tracked_min_sats,
            tracked_continuity_good_m=config.fallback_tracked_continuity_good_m,
            tracked_continuity_bad_m=config.fallback_tracked_continuity_bad_m,
            tracked_sigma_min_scale=config.fallback_tracked_sigma_min_scale,
            tracked_sigma_max_scale=config.fallback_tracked_sigma_max_scale,
            max_age_s=config.anchor_max_age_s,
            max_continuity_residual_m=config.anchor_max_continuity_residual_m,
        )
    elif config.anchor_enabled and carrier_rows:
        anchor_attempt.state = current_pf_state

    return PostDDCarrierRescueDecision(
        anchor_attempt=anchor_attempt,
        fallback_attempt=fallback,
    )
