"""Signal, constellation, and clock-group helpers for GSDC2023 raw parity.

This module is deliberately free of trip I/O and solver state.  It owns the
small mapping rules that are shared by raw-bridge construction and parity
comparison tools.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from gnss_gpu.multi_gnss import (
    SYSTEM_BEIDOU,
    SYSTEM_GALILEO,
    SYSTEM_GPS,
    SYSTEM_QZSS,
)


DEFAULT_MULTI_GNSS_SIGNAL_TYPES = {
    1: "GPS_L1_CA",
    6: "GAL_E1_C_P",
    4: "QZS_L1_CA",
}
DUAL_FREQUENCY_SIGNAL_TYPES = {
    1: ("GPS_L1_CA", "GPS_L5_Q"),
    6: ("GAL_E1_C_P", "GAL_E5A_Q"),
    4: ("QZS_L1_CA", "QZS_L5_Q"),
}
CONSTELLATION_TO_SYS_KIND = {1: 0, 6: 1, 4: 2}
SYS_KIND_TO_MULTI_SYSTEM = {0: SYSTEM_GPS, 1: SYSTEM_GALILEO, 2: SYSTEM_QZSS, 3: SYSTEM_BEIDOU}
MATLAB_SIGNAL_CLOCK_DIM = 7
MATLAB_SIGNAL_CLOCK_KIND_L1 = {
    1: 0,  # GPS L1
    3: 1,  # GLONASS G1
    6: 2,  # Galileo E1
    5: 3,  # BeiDou B1
    4: 0,  # QZSS L1 shares the GPS clock family in the raw bridge.
}
MATLAB_SIGNAL_CLOCK_KIND_L5 = {
    1: 4,  # GPS L5
    6: 5,  # Galileo E5
    5: 6,  # BeiDou B2a
    4: 4,  # QZSS L5 shares the GPS L5 clock family in the raw bridge.
}
MATLAB_SIGNAL_KIND_TO_MULTI_SYSTEM = {
    0: SYSTEM_GPS,
    1: SYSTEM_GPS,
    2: SYSTEM_GALILEO,
    3: SYSTEM_BEIDOU,
    4: SYSTEM_GPS,
    5: SYSTEM_GALILEO,
    6: SYSTEM_BEIDOU,
}


def signal_types_for_constellation(
    constellation_type: int,
    signal_type: str,
    *,
    dual_frequency: bool,
) -> tuple[str, ...]:
    if dual_frequency:
        return DUAL_FREQUENCY_SIGNAL_TYPES.get(int(constellation_type), (signal_type,))
    return (signal_type,)


def multi_gnss_mask(df: pd.DataFrame, *, dual_frequency: bool = False) -> np.ndarray:
    signal = df["SignalType"].fillna("")
    mask = np.zeros(len(df), dtype=bool)
    if dual_frequency:
        for constellation_type, signal_types in DUAL_FREQUENCY_SIGNAL_TYPES.items():
            for signal_type in signal_types:
                mask |= (df["ConstellationType"] == constellation_type) & (signal == signal_type)
    else:
        for constellation_type, signal_type in DEFAULT_MULTI_GNSS_SIGNAL_TYPES.items():
            mask |= (df["ConstellationType"] == constellation_type) & (signal == signal_type)
    return mask


def signal_sort_rank(signal_type: str) -> int:
    sig = str(signal_type).upper()
    if "L5" in sig or "E5" in sig or "B2" in sig:
        return 1
    return 0


def is_l5_signal(signal_type: str) -> bool:
    return signal_sort_rank(signal_type) == 1


def slot_frequency_thresholds(
    slot_keys: list[tuple[int, int, str]] | tuple[tuple[int, int, str], ...],
    threshold: float,
    *,
    default_l1_threshold: float,
    default_l5_threshold: float,
) -> float | np.ndarray:
    threshold = float(threshold)
    if threshold <= 0.0 or abs(threshold - float(default_l1_threshold)) > 1e-9:
        return threshold
    values = np.full(len(slot_keys), threshold, dtype=np.float64)
    for idx, key in enumerate(slot_keys):
        if len(key) >= 3 and is_l5_signal(str(key[2])):
            values[idx] = float(default_l5_threshold)
    return values


def slot_sort_key(key: tuple[int, int] | tuple[int, int, str]) -> tuple[int, int, int, str]:
    constellation_type, svid = int(key[0]), int(key[1])
    signal_type = str(key[2]) if len(key) >= 3 else ""
    sort_order = {1: 0, 6: 1, 4: 2}
    return sort_order.get(int(constellation_type), 99), int(svid), signal_sort_rank(signal_type), signal_type


def clock_kind_for_observation(
    constellation_type: int,
    signal_type: str,
    *,
    dual_frequency: bool,
    multi_gnss: bool,
) -> int:
    if not dual_frequency:
        return CONSTELLATION_TO_SYS_KIND.get(int(constellation_type), 0)
    if multi_gnss:
        if signal_sort_rank(signal_type) > 0:
            return MATLAB_SIGNAL_CLOCK_KIND_L5.get(int(constellation_type), 0)
        return MATLAB_SIGNAL_CLOCK_KIND_L1.get(int(constellation_type), 0)
    if signal_sort_rank(signal_type) > 0:
        return MATLAB_SIGNAL_CLOCK_KIND_L5.get(int(constellation_type), 1)
    return 0


def multi_system_for_clock_kind(clock_kind: int, n_clock: int) -> int:
    if n_clock >= MATLAB_SIGNAL_CLOCK_DIM:
        return MATLAB_SIGNAL_KIND_TO_MULTI_SYSTEM.get(int(clock_kind), SYSTEM_BEIDOU)
    return SYS_KIND_TO_MULTI_SYSTEM.get(int(clock_kind), SYSTEM_BEIDOU)


def slot_frequency_label(signal_type: str) -> str:
    sig = str(signal_type).upper()
    return "L5" if "L5" in sig or "E5" in sig or "B2" in sig else "L1"


def factor_frequency_label(signal_type: str) -> str | None:
    sig = str(signal_type).upper()
    if "L5" in sig or "E5" in sig or "B2" in sig:
        return "L5"
    if "L1" in sig or "E1" in sig or "B1" in sig:
        return "L1"
    return None


def constellation_to_matlab_sys(constellation_type: int) -> int:
    if int(constellation_type) == 6:
        return 8
    return int(constellation_type)


def slot_pseudorange_common_bias_group_keys(
    slot_keys: list[tuple[int, int, str]] | tuple[tuple[int, int, str], ...],
) -> list[tuple[int, str]]:
    return [
        (constellation_to_matlab_sys(int(constellation_type)), slot_frequency_label(str(signal_type)))
        for constellation_type, _svid, signal_type in slot_keys
    ]


def slot_pseudorange_common_bias_groups(
    slot_keys: list[tuple[int, int, str]] | tuple[tuple[int, int, str], ...],
) -> np.ndarray:
    group_ids: list[int] = []
    group_index: dict[tuple[int, str], int] = {}
    for key in slot_pseudorange_common_bias_group_keys(slot_keys):
        group_id = group_index.get(key)
        if group_id is None:
            group_id = len(group_index)
            group_index[key] = group_id
        group_ids.append(group_id)
    return np.asarray(group_ids, dtype=np.int32)


def remap_pseudorange_isb_by_group(
    source_slot_keys: list[tuple[int, int, str]] | tuple[tuple[int, int, str], ...],
    source_isb_by_group: dict[int, float],
    target_slot_keys: list[tuple[int, int, str]] | tuple[tuple[int, int, str], ...],
) -> dict[int, float]:
    source_groups = slot_pseudorange_common_bias_groups(source_slot_keys)
    source_group_keys = slot_pseudorange_common_bias_group_keys(source_slot_keys)
    isb_by_key: dict[tuple[int, str], float] = {}
    for group_id, group_key in zip(source_groups, source_group_keys):
        value = source_isb_by_group.get(int(group_id))
        if value is not None and np.isfinite(value):
            isb_by_key.setdefault(group_key, float(value))

    target_groups = slot_pseudorange_common_bias_groups(target_slot_keys)
    target_group_keys = slot_pseudorange_common_bias_group_keys(target_slot_keys)
    remapped: dict[int, float] = {}
    for group_id, group_key in zip(target_groups, target_group_keys):
        value = isb_by_key.get(group_key)
        if value is not None:
            remapped.setdefault(int(group_id), float(value))
    return remapped
