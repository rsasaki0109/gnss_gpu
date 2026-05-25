"""taroz/gsdc2023 ``obserrmodel.m`` SNR-based observation weighting.

Ports the ``"sn"`` (SNR) branch of ``obserrmodel.m`` and the
``sysfreq2sigtype.m`` lookup table to Python.  The single new mode
``"taroz_sn"`` for ``build_observation_matrix`` reproduces taroz's
σ_P / σ_D / σ_L computation:

::

    sn_base[i] = 10 ** (-(S[i] - prctile(S, 85, "all")) / sn_den)
    sigfactor[i] = sigtype_factor[sysfreq2sigtype(sys, freq)]
    σ_P[i] = P_sn_ratio * sn_base[i] * sigfactor[i]
    σ_D[i] = D_sn_ratio * sn_base[i]                 # NB: no sigfactor
    σ_L[i] = L_sn_ratio * sn_base[i] * sigfactor[i]

QZSS is folded into the GPS clock family (matching
``MATLAB_SIGNAL_CLOCK_KIND_L1``/``_L5`` in ``gsdc2023_signal_model``),
since taroz's ``sysfreq2sigtype.m`` enumerates only GPS/GLO/GAL/BDS and
QZSS would otherwise be tagged ``Other`` (sigfactor NaN, dropping the
observation entirely).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


SN_PERCENTILE_DEFAULT = 85.0
SN_DENOMINATOR_DEFAULT = 20.0
P_SN_RATIO_DEFAULT = 1.0
D_SN_RATIO_DEFAULT = 1.0 / 12.0
L_SN_RATIO_DEFAULT = 1.0 / 400.0


SIGTYPE_GPS_L1 = 0
SIGTYPE_GLO_G1 = 1
SIGTYPE_GAL_E1 = 2
SIGTYPE_BDS_B1 = 3
SIGTYPE_GPS_L5 = 4
SIGTYPE_GAL_E5 = 5
SIGTYPE_BDS_B2A = 6
SIGTYPE_OTHER = 7


SIGTYPE_FACTOR: tuple[float, ...] = (
    0.8,
    1.5,
    0.8,
    0.8,
    0.5,
    0.5,
    0.5,
    float("nan"),
)


CONSTELLATION_GPS = 1
CONSTELLATION_GLONASS = 3
CONSTELLATION_QZSS = 4
CONSTELLATION_BEIDOU = 5
CONSTELLATION_GALILEO = 6


def constellation_signal_to_sigtype(constellation_type: int, is_l5: bool) -> int:
    """Map Android constellation type + L1/L5 flag to taroz sigtype index.

    QZSS is folded into the matching GPS sigtype to avoid sigfactor=NaN.
    """
    if is_l5:
        if constellation_type == CONSTELLATION_GPS or constellation_type == CONSTELLATION_QZSS:
            return SIGTYPE_GPS_L5
        if constellation_type == CONSTELLATION_GALILEO:
            return SIGTYPE_GAL_E5
        if constellation_type == CONSTELLATION_BEIDOU:
            return SIGTYPE_BDS_B2A
        return SIGTYPE_OTHER
    if constellation_type == CONSTELLATION_GPS or constellation_type == CONSTELLATION_QZSS:
        return SIGTYPE_GPS_L1
    if constellation_type == CONSTELLATION_GLONASS:
        return SIGTYPE_GLO_G1
    if constellation_type == CONSTELLATION_GALILEO:
        return SIGTYPE_GAL_E1
    if constellation_type == CONSTELLATION_BEIDOU:
        return SIGTYPE_BDS_B1
    return SIGTYPE_OTHER


def sigtype_factor(sigtype: int) -> float:
    if sigtype < 0 or sigtype >= len(SIGTYPE_FACTOR):
        return float("nan")
    return SIGTYPE_FACTOR[sigtype]


def compute_sn_base(cn0_dbhz: float, p_percentile_dbhz: float, sn_den: float = SN_DENOMINATOR_DEFAULT) -> float:
    """Reproduce ``10^(-(S - prctile(S, ptile, "all")) / sn_den)``."""
    if not (np.isfinite(cn0_dbhz) and np.isfinite(p_percentile_dbhz) and sn_den > 0):
        return float("nan")
    return float(10.0 ** (-(float(cn0_dbhz) - float(p_percentile_dbhz)) / float(sn_den)))


def compute_pr_sigma_sn(
    cn0_dbhz: float,
    p_percentile_dbhz: float,
    sigtype: int,
    *,
    sn_den: float = SN_DENOMINATOR_DEFAULT,
    p_sn_ratio: float = P_SN_RATIO_DEFAULT,
) -> float:
    base = compute_sn_base(cn0_dbhz, p_percentile_dbhz, sn_den)
    factor = sigtype_factor(sigtype)
    if not np.isfinite(base) or not np.isfinite(factor):
        return float("nan")
    return float(p_sn_ratio * base * factor)


def compute_doppler_sigma_sn(
    cn0_dbhz: float,
    p_percentile_dbhz: float,
    *,
    sn_den: float = SN_DENOMINATOR_DEFAULT,
    d_sn_ratio: float = D_SN_RATIO_DEFAULT,
) -> float:
    base = compute_sn_base(cn0_dbhz, p_percentile_dbhz, sn_den)
    if not np.isfinite(base):
        return float("nan")
    return float(d_sn_ratio * base)


def compute_carrier_sigma_sn(
    cn0_dbhz: float,
    p_percentile_dbhz: float,
    sigtype: int,
    *,
    sn_den: float = SN_DENOMINATOR_DEFAULT,
    l_sn_ratio: float = L_SN_RATIO_DEFAULT,
) -> float:
    base = compute_sn_base(cn0_dbhz, p_percentile_dbhz, sn_den)
    factor = sigtype_factor(sigtype)
    if not np.isfinite(base) or not np.isfinite(factor):
        return float("nan")
    return float(l_sn_ratio * base * factor)


@dataclass(frozen=True)
class TripCN0Percentiles:
    """Per-frequency 85th-percentile SNR thresholds for a trip.

    ``np.nan`` for a frequency means no observations were collected for that
    frequency: any computation that falls into that bucket should fall back
    to the legacy weighting model.
    """

    l1: float
    l5: float

    def for_frequency(self, is_l5: bool) -> float:
        return float(self.l5 if is_l5 else self.l1)


def compute_trip_cn0_percentiles(
    cn0_dbhz_by_l5: tuple[np.ndarray, np.ndarray],
    *,
    percentile: float = SN_PERCENTILE_DEFAULT,
) -> TripCN0Percentiles:
    """Compute per-frequency 85th percentile across all collected SNR samples.

    The two arrays should already be filtered to L1 and L5 observations
    respectively; non-finite samples are removed before the percentile.
    """
    l1, l5 = cn0_dbhz_by_l5
    l1_arr = np.asarray(l1, dtype=np.float64).ravel()
    l5_arr = np.asarray(l5, dtype=np.float64).ravel()
    l1_arr = l1_arr[np.isfinite(l1_arr)]
    l5_arr = l5_arr[np.isfinite(l5_arr)]
    p_l1 = float(np.percentile(l1_arr, percentile)) if l1_arr.size else float("nan")
    p_l5 = float(np.percentile(l5_arr, percentile)) if l5_arr.size else float("nan")
    return TripCN0Percentiles(l1=p_l1, l5=p_l5)


def sigma_to_weight(sigma: float, *, eps: float = 1e-12) -> float:
    """Convert σ to 1/σ² weight, returning 0 on non-finite / non-positive σ."""
    if not np.isfinite(sigma) or sigma <= eps:
        return 0.0
    return float(1.0 / (sigma * sigma))


__all__ = [
    "CONSTELLATION_BEIDOU",
    "CONSTELLATION_GALILEO",
    "CONSTELLATION_GLONASS",
    "CONSTELLATION_GPS",
    "CONSTELLATION_QZSS",
    "D_SN_RATIO_DEFAULT",
    "L_SN_RATIO_DEFAULT",
    "P_SN_RATIO_DEFAULT",
    "SIGTYPE_BDS_B1",
    "SIGTYPE_BDS_B2A",
    "SIGTYPE_FACTOR",
    "SIGTYPE_GAL_E1",
    "SIGTYPE_GAL_E5",
    "SIGTYPE_GLO_G1",
    "SIGTYPE_GPS_L1",
    "SIGTYPE_GPS_L5",
    "SIGTYPE_OTHER",
    "SN_DENOMINATOR_DEFAULT",
    "SN_PERCENTILE_DEFAULT",
    "TripCN0Percentiles",
    "compute_carrier_sigma_sn",
    "compute_doppler_sigma_sn",
    "compute_pr_sigma_sn",
    "compute_sn_base",
    "compute_trip_cn0_percentiles",
    "constellation_signal_to_sigtype",
    "sigma_to_weight",
    "sigtype_factor",
]
