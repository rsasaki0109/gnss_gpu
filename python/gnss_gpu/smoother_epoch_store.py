"""Build stored-epoch metadata for PF smoother replay."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from gnss_gpu.local_fgo_bridge import (
    _copy_dd_carrier_epoch,
    _copy_dd_pseudorange_epoch,
    _make_undiff_pr_epoch,
)
from gnss_gpu.pf_smoother_runtime import ForwardRunBuffers


@dataclass(frozen=True)
class SmootherEpochStoreInputs:
    spp_ref: np.ndarray | None
    dd_pseudorange: Any | None
    dd_pseudorange_sigma: float | None
    dd_pseudorange_source: str | None
    dd_carrier: Any | None
    dd_carrier_sigma: float | None
    carrier_anchor_pseudorange: Any | None
    carrier_anchor_sigma: float | None
    carrier_afv: Any | None
    carrier_afv_sigma: float | None
    carrier_afv_wavelength: float | None
    doppler_update: Any | None
    doppler_sigma_mps: float | None
    doppler_velocity_update_gain: float | None
    doppler_max_velocity_update_mps: float | None

    def as_store_kwargs(self) -> dict[str, Any]:
        return {
            "spp_ref": self.spp_ref,
            "dd_pseudorange": self.dd_pseudorange,
            "dd_pseudorange_sigma": self.dd_pseudorange_sigma,
            "dd_pseudorange_source": self.dd_pseudorange_source,
            "dd_carrier": self.dd_carrier,
            "dd_carrier_sigma": self.dd_carrier_sigma,
            "carrier_anchor_pseudorange": self.carrier_anchor_pseudorange,
            "carrier_anchor_sigma": self.carrier_anchor_sigma,
            "carrier_afv": self.carrier_afv,
            "carrier_afv_sigma": self.carrier_afv_sigma,
            "carrier_afv_wavelength": self.carrier_afv_wavelength,
            "doppler_update": self.doppler_update,
            "doppler_sigma_mps": self.doppler_sigma_mps,
            "doppler_velocity_update_gain": self.doppler_velocity_update_gain,
            "doppler_max_velocity_update_mps": self.doppler_max_velocity_update_mps,
        }


def build_smoother_epoch_store_inputs(
    *,
    spp_pos: np.ndarray,
    position_update_sigma: float | None,
    dd_pseudorange_result: Any | None,
    dd_pseudorange_sigma: float,
    used_widelane_epoch: bool,
    dd_carrier_result: Any | None,
    dd_carrier_sigma_cycles: float | None,
    anchor_attempt: Any,
    carrier_anchor_sigma_m: float,
    fallback_attempt: Any,
    carrier_afv_wavelength_m: float,
    doppler_update: Any | None,
    doppler_sigma_mps: float | None,
    doppler_velocity_update_gain: float,
    doppler_max_velocity_update_mps: float,
    min_pairs: int = 3,
) -> SmootherEpochStoreInputs:
    valid_dd_pr = _has_min_dd_pairs(dd_pseudorange_result, min_pairs)
    valid_dd_carrier = _has_min_dd_pairs(dd_carrier_result, min_pairs)
    anchor_update = getattr(anchor_attempt, "update", None)
    carrier_afv = getattr(fallback_attempt, "afv", None)

    return SmootherEpochStoreInputs(
        spp_ref=_spp_reference(spp_pos, position_update_sigma),
        dd_pseudorange=dd_pseudorange_result,
        dd_pseudorange_sigma=float(dd_pseudorange_sigma) if valid_dd_pr else None,
        dd_pseudorange_source=_dd_pseudorange_source(
            valid_dd_pr,
            used_widelane_epoch=used_widelane_epoch,
        ),
        dd_carrier=dd_carrier_result,
        dd_carrier_sigma=(
            float(dd_carrier_sigma_cycles)
            if valid_dd_carrier and dd_carrier_sigma_cycles is not None
            else None
        ),
        carrier_anchor_pseudorange=anchor_update,
        carrier_anchor_sigma=float(carrier_anchor_sigma_m) if anchor_update is not None else None,
        carrier_afv=carrier_afv,
        carrier_afv_sigma=getattr(fallback_attempt, "sigma_cycles", None),
        carrier_afv_wavelength=(
            float(carrier_afv_wavelength_m) if carrier_afv is not None else None
        ),
        doppler_update=doppler_update,
        doppler_sigma_mps=float(doppler_sigma_mps) if doppler_update is not None else None,
        doppler_velocity_update_gain=(
            float(doppler_velocity_update_gain) if doppler_update is not None else None
        ),
        doppler_max_velocity_update_mps=(
            float(doppler_max_velocity_update_mps)
            if doppler_update is not None
            else None
        ),
    )


def append_smoother_epoch_store(
    pf: Any,
    buffers: ForwardRunBuffers,
    *,
    sat_ecef: np.ndarray,
    pseudoranges: np.ndarray,
    weights: np.ndarray,
    spp_pos: np.ndarray,
    epoch_state: Any,
    dt: float,
    position_update_sigma: float | None,
    carrier_anchor_sigma_m: float,
    carrier_afv_wavelength_m: float,
    doppler_velocity_update_gain: float,
    doppler_max_velocity_update_mps: float,
    need_tdcp_motion: bool,
) -> SmootherEpochStoreInputs:
    store_inputs = build_smoother_epoch_store_inputs(
        spp_pos=spp_pos,
        position_update_sigma=position_update_sigma,
        dd_pseudorange_result=epoch_state.dd_pr_result,
        dd_pseudorange_sigma=epoch_state.dd_pr_sigma_epoch,
        used_widelane_epoch=epoch_state.used_widelane_epoch,
        dd_carrier_result=epoch_state.dd_carrier_result,
        dd_carrier_sigma_cycles=epoch_state.dd_cp_sigma_cycles,
        anchor_attempt=epoch_state.anchor_attempt,
        carrier_anchor_sigma_m=carrier_anchor_sigma_m,
        fallback_attempt=epoch_state.fallback_attempt,
        carrier_afv_wavelength_m=carrier_afv_wavelength_m,
        doppler_update=epoch_state.doppler_update_epoch,
        doppler_sigma_mps=epoch_state.doppler_sigma_epoch,
        doppler_velocity_update_gain=doppler_velocity_update_gain,
        doppler_max_velocity_update_mps=doppler_max_velocity_update_mps,
    )
    pf.store_epoch(
        sat_ecef,
        pseudoranges,
        weights,
        epoch_state.velocity,
        dt,
        **store_inputs.as_store_kwargs(),
    )
    buffers.append_smoother_motion(
        epoch_state.velocity,
        epoch_state.fgo_tdcp_motion_velocity,
        dt=dt,
        need_tdcp_motion=need_tdcp_motion,
    )
    fgo_receiver_pos = np.asarray(pf.estimate()[:3], dtype=np.float64)
    buffers.append_smoother_observations(
        _copy_dd_carrier_epoch(epoch_state.dd_carrier_result),
        _copy_dd_pseudorange_epoch(epoch_state.dd_pr_result),
        _make_undiff_pr_epoch(sat_ecef, pseudoranges, weights, fgo_receiver_pos),
    )
    return store_inputs


def _has_min_dd_pairs(result: Any | None, min_pairs: int) -> bool:
    return result is not None and int(getattr(result, "n_dd", 0)) >= int(min_pairs)


def _spp_reference(
    spp_pos: np.ndarray,
    position_update_sigma: float | None,
) -> np.ndarray | None:
    pos = np.asarray(spp_pos, dtype=np.float64)
    if (
        position_update_sigma is not None
        and np.isfinite(pos).all()
        and np.linalg.norm(pos) > 1e6
    ):
        return pos
    return None


def _dd_pseudorange_source(
    valid_dd_pr: bool,
    *,
    used_widelane_epoch: bool,
) -> str | None:
    if not valid_dd_pr:
        return None
    return "widelane" if used_widelane_epoch else "dd_pseudorange"
