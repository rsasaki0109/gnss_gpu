"""Per-constellation broadcast time-scale alignment helpers.

Broadcast navigation records express their reference epochs (``toe`` /
``toc_seconds``) in the constellation's native time scale: GPST for GPS /
QZSS, GST (offset-free vs GPST) for Galileo, BDT for BeiDou, and UTC(SU)
for GLONASS.  This module centralises the small set of constants and pure
functions needed to fold a GPST receiver second-of-week into the broadcast
record's time scale, so each call-site can simply do::

    rx_sow = broadcast_rx_seconds_of_week(nav, recv_gpst_sow)
    dt = rx_sow - nav.toe

without re-deriving the offset for every record family.  Reference:
RTKLIB ``gpst2bdt`` / ``gpst2utc`` and the BeiDou ICD § 1.6.
"""

from __future__ import annotations

import math


GPS_WEEK_SEC = 604800.0
"""Seconds per GPS week (kept here so this module is import-safe on its own)."""

GPST_MINUS_BDT_S = 14.0
"""GPST minus BDT at BDT epoch alignment — BeiDou ICD §1.6 (RTKLIB constant)."""

GPST_MINUS_UTC_S = 18.0
"""GPST minus UTC offset used by GLONASS broadcast epochs (current as of 2017
leap-second insertion).  Updated manually when a new leap second is announced.
"""


def unwrap_week_seconds(value: float) -> float:
    """Fold a scalar second-of-week into ``[0, GPS_WEEK_SEC)``.

    Used to keep cross-week offset arithmetic numerically stable — for example
    when subtracting a BDT offset from a GPST sow near the week boundary.
    """

    folded = float(value)
    if not math.isfinite(folded):
        return folded
    while folded < 0.0:
        folded += GPS_WEEK_SEC
    while folded >= GPS_WEEK_SEC:
        folded -= GPS_WEEK_SEC
    return folded


def broadcast_rx_seconds_of_week(system: str, recv_gpst_sow: float) -> float:
    """GPST receiver sow folded into the broadcast record's time scale.

    ``system`` is the RINEX single-letter constellation identifier
    (``G``/``E``/``J``/``C``/``R``/``S``).  Only ``C`` and ``R`` apply
    non-zero offsets; the others fall through unchanged.
    """

    recv = float(recv_gpst_sow)
    if system == "C":
        return unwrap_week_seconds(recv - GPST_MINUS_BDT_S)
    if system == "R":
        return unwrap_week_seconds(recv - GPST_MINUS_UTC_S)
    return recv


__all__ = [
    "GPS_WEEK_SEC",
    "GPST_MINUS_BDT_S",
    "GPST_MINUS_UTC_S",
    "broadcast_rx_seconds_of_week",
    "unwrap_week_seconds",
]
