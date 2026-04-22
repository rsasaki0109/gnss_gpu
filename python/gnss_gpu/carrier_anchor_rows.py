"""Carrier-anchor row selection helper."""

from __future__ import annotations

import numpy as np

from gnss_gpu.carrier_rescue import (
    MUPF_L1_COMPAT_SYSTEM_IDS,
    MUPF_L1_WAVELENGTH_M,
    _select_same_band_carrier_rows,
)
from gnss_gpu.pf_smoother_config import CarrierRescueConfig, MupfConfig


def select_carrier_anchor_rows(
    measurements,
    pseudoranges: np.ndarray,
    spp_position_ecef: np.ndarray,
    mupf: MupfConfig,
    config: CarrierRescueConfig,
) -> dict[tuple[int, int], dict[str, object]]:
    if not config.anchor_enabled:
        return {}
    return _select_same_band_carrier_rows(
        measurements,
        pseudoranges,
        snr_min=mupf.snr_min,
        elev_min=mupf.elev_min,
        spp_pos_check=np.asarray(spp_position_ecef, dtype=np.float64),
        wavelength_m=MUPF_L1_WAVELENGTH_M,
        allowed_system_ids=MUPF_L1_COMPAT_SYSTEM_IDS,
    )
