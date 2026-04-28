"""GNSS logger bridge helpers for GSDC2023 raw parity.

This module owns the boundary between parsed ``gnss_log.txt`` observations and
the raw-bridge matrices.  It intentionally does not import
``gsdc2023_raw_bridge``; satellite-clock adjustment is injected by the caller.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.gsdc2023_signal_model import slot_frequency_label


GNSS_LOG_SYNTHETIC_PRODUCT_COLUMNS = (
    "SvPositionXEcefMeters",
    "SvPositionYEcefMeters",
    "SvPositionZEcefMeters",
    "SvVelocityXEcefMetersPerSecond",
    "SvVelocityYEcefMetersPerSecond",
    "SvVelocityZEcefMetersPerSecond",
    "SvClockBiasMeters",
    "SvClockDriftMetersPerSecond",
    "IonosphericDelayMeters",
    "TroposphericDelayMeters",
    "SvElevationDegrees",
    "SvAzimuthDegrees",
    "IsrbMeters",
)


SatClockAdjustmentFn = Callable[[int, int, str, dict[int, float]], float]


@dataclass(frozen=True)
class GnssLogPseudorangeProducts:
    pseudorange: np.ndarray
    weights: np.ndarray
    observable_pseudorange: np.ndarray


def gnss_log_signal_type(freq: str) -> str | None:
    freq_l = str(freq).upper()
    if freq_l == "L1":
        return "GPS_L1_CA"
    if freq_l == "L5":
        return "GPS_L5_Q"
    return None


def interpolated_raw_values(
    raw_frame: pd.DataFrame,
    *,
    time_key: int,
    svid: int,
    signal_type: str,
    columns: tuple[str, ...],
) -> dict[str, float]:
    subset = raw_frame[
        (pd.to_numeric(raw_frame["ConstellationType"], errors="coerce").fillna(0).astype(np.int64) == 1)
        & (pd.to_numeric(raw_frame["Svid"], errors="coerce").fillna(0).astype(np.int64) == int(svid))
        & (raw_frame["SignalType"].astype(str) == str(signal_type))
    ].copy()
    if subset.empty:
        return {}
    subset = subset.sort_values("utcTimeMillis")
    times = pd.to_numeric(subset["utcTimeMillis"], errors="coerce").to_numpy(dtype=np.float64)
    finite_time = np.isfinite(times)
    if not finite_time.any():
        return {}
    values: dict[str, float] = {}
    for column in columns:
        if column not in subset.columns:
            continue
        series = pd.to_numeric(subset[column], errors="coerce").to_numpy(dtype=np.float64)
        valid = finite_time & np.isfinite(series)
        if not valid.any():
            continue
        if np.count_nonzero(valid) == 1:
            values[column] = float(series[valid][0])
        else:
            valid_times = times[valid]
            valid_series = series[valid]
            target_time = float(time_key)
            if valid_times.min() <= target_time <= valid_times.max() and valid_times.size >= 3:
                nearest = np.argsort(np.abs(valid_times - target_time))[: min(8, valid_times.size)]
                x = (valid_times[nearest] - target_time) * 1.0e-3
                y = valid_series[nearest]
                degree = min(2, x.size - 1)
                coeff = np.polyfit(x, y, degree)
                values[column] = float(np.polyval(coeff, 0.0))
            else:
                values[column] = float(np.interp(target_time, valid_times, valid_series))
    return values


def _load_masked_gps_observations(
    log_path: Path,
    *,
    phone_name: str,
    require_p_ok: bool,
) -> pd.DataFrame:
    try:
        from experiments.gsdc2023_gnss_log_reader import (  # pylint: disable=import-outside-toplevel
            apply_gnss_log_signal_masks,
            load_gnss_log_observations,
        )
    except ImportError:
        return pd.DataFrame()

    observations = load_gnss_log_observations(log_path)
    if observations.empty:
        return pd.DataFrame()
    observations = observations[(observations["matlab_sys"] == 1) & observations["freq"].isin(("L1", "L5"))].copy()
    if observations.empty:
        return pd.DataFrame()
    observations = apply_gnss_log_signal_masks(observations, phone=phone_name)
    if require_p_ok:
        observations = observations[observations["P_ok"]].copy()
    return observations


def append_gnss_log_only_gps_rows(
    df: pd.DataFrame,
    raw_frame: pd.DataFrame,
    epoch_meta: pd.DataFrame,
    trip_dir: Path,
    *,
    phone_name: str,
    dual_frequency: bool,
) -> pd.DataFrame:
    log_path = Path(trip_dir) / "supplemental" / "gnss_log.txt"
    if not dual_frequency or not log_path.is_file():
        return df

    observations = _load_masked_gps_observations(log_path, phone_name=phone_name, require_p_ok=False)
    if observations.empty:
        return df

    existing: set[tuple[int, int, str]] = set()
    if not df.empty:
        for row in df.itertuples(index=False):
            freq = slot_frequency_label(str(getattr(row, "SignalType")))
            existing.add((int(getattr(row, "utcTimeMillis")), int(getattr(row, "Svid")), freq))

    epoch_lookup: dict[int, object] = {
        int(row.utcTimeMillis): row
        for row in epoch_meta.itertuples(index=False)
        if pd.notna(row.utcTimeMillis)
    }
    rows: list[dict[str, object]] = []
    for obs_row in observations.itertuples(index=False):
        time_key = int(getattr(obs_row, "utcTimeMillis"))
        svid = int(getattr(obs_row, "Svid"))
        freq = str(getattr(obs_row, "freq"))
        if (time_key, svid, freq) in existing:
            continue
        signal_type = gnss_log_signal_type(freq)
        if signal_type is None:
            continue
        product_values = interpolated_raw_values(
            raw_frame,
            time_key=time_key,
            svid=svid,
            signal_type=signal_type,
            columns=GNSS_LOG_SYNTHETIC_PRODUCT_COLUMNS,
        )
        required_products = (
            "SvPositionXEcefMeters",
            "SvPositionYEcefMeters",
            "SvPositionZEcefMeters",
            "SvClockBiasMeters",
            "IonosphericDelayMeters",
            "TroposphericDelayMeters",
            "SvElevationDegrees",
        )
        if any(column not in product_values or not np.isfinite(product_values[column]) for column in required_products):
            continue

        new_row = {column: np.nan for column in df.columns}
        epoch_row = epoch_lookup.get(time_key)
        if epoch_row is not None:
            for column in df.columns:
                if hasattr(epoch_row, column):
                    new_row[column] = getattr(epoch_row, column)
        new_row.update(product_values)
        new_row.update(
            {
                "MessageType": "Raw",
                "utcTimeMillis": time_key,
                "Svid": svid,
                "ConstellationType": 1,
                "SignalType": signal_type,
                "RawPseudorangeMeters": float(getattr(obs_row, "PseudorangeMeters")),
                "Cn0DbHz": float(getattr(obs_row, "Cn0DbHz")),
                "State": int(getattr(obs_row, "State")),
                "MultipathIndicator": float(getattr(obs_row, "MultipathIndicator")),
                "PseudorangeRateMetersPerSecond": float(getattr(obs_row, "PseudorangeRateMetersPerSecond")),
                "PseudorangeRateUncertaintyMetersPerSecond": float(
                    getattr(obs_row, "PseudorangeRateUncertaintyMetersPerSecond"),
                ),
                "AccumulatedDeltaRangeState": int(getattr(obs_row, "AccumulatedDeltaRangeState")),
                "AccumulatedDeltaRangeMeters": float(getattr(obs_row, "AccumulatedDeltaRangeMeters")),
                "AccumulatedDeltaRangeUncertaintyMeters": float(
                    getattr(obs_row, "AccumulatedDeltaRangeUncertaintyMeters"),
                ),
                "CarrierFrequencyHz": float(getattr(obs_row, "freq_hz")),
                "ArrivalTimeNanosSinceGpsEpoch": float(getattr(obs_row, "tow_rx_s")) * 1.0e9,
                "ReceivedSvTimeNanos": float(getattr(obs_row, "ReceivedSvTimeNanos")),
                "ReceivedSvTimeUncertaintyNanos": float(getattr(obs_row, "ReceivedSvTimeUncertaintyNanos")),
                "TimeOffsetNanos": float(getattr(obs_row, "TimeOffsetNanos")),
            },
        )
        rows.append(new_row)
        existing.add((time_key, svid, freq))

    if not rows:
        return df
    rows_frame = pd.DataFrame(rows, columns=df.columns)
    if df.empty:
        return rows_frame
    return pd.concat([df, rows_frame], ignore_index=True)


@lru_cache(maxsize=64)
def gnss_log_matlab_epoch_times_ms_cached(log_path_str: str) -> tuple[int, ...]:
    log_path = Path(log_path_str)
    if not log_path.is_file():
        return ()
    try:
        from experiments.gsdc2023_gnss_log_reader import (  # pylint: disable=import-outside-toplevel
            load_gnss_log_observations,
        )
    except ImportError:
        return ()

    observations = load_gnss_log_observations(log_path)
    if observations.empty or "utcTimeMillis" not in observations.columns:
        return ()
    times = pd.to_numeric(observations["utcTimeMillis"], errors="coerce").to_numpy(dtype=np.float64)
    times = times[np.isfinite(times)]
    if times.size == 0:
        return ()
    return tuple(int(round(float(t))) for t in np.unique(times))


def gnss_log_matlab_epoch_times_ms(trip_dir: Path) -> tuple[int, ...]:
    return gnss_log_matlab_epoch_times_ms_cached(str(Path(trip_dir) / "supplemental" / "gnss_log.txt"))


def gnss_log_corrected_pseudorange_products(
    trip_dir: Path,
    raw_frame: pd.DataFrame,
    times_ms: np.ndarray,
    slot_keys: tuple[tuple[int, int, str], ...],
    gps_tgd_m_by_svid: dict[int, float],
    *,
    phone_name: str,
    rtklib_tropo_m: np.ndarray | None = None,
    sat_clock_bias_m: np.ndarray | None = None,
    sat_clock_adjustment_m: SatClockAdjustmentFn | None = None,
) -> GnssLogPseudorangeProducts | None:
    if not slot_keys:
        return None
    if not any(int(key[0]) == 1 for key in slot_keys):
        return None
    log_path = Path(trip_dir) / "supplemental" / "gnss_log.txt"
    if not log_path.is_file():
        return None

    observations = _load_masked_gps_observations(log_path, phone_name=phone_name, require_p_ok=True)
    if observations.empty:
        return None

    slot_index = {key: idx for idx, key in enumerate(slot_keys)}
    time_index = {int(round(float(time_ms))): idx for idx, time_ms in enumerate(times_ms)}
    raw_lookup: dict[tuple[int, int, str], tuple[int, object]] = {}
    for row in raw_frame.itertuples(index=False):
        if int(getattr(row, "ConstellationType")) != 1:
            continue
        time_key = int(getattr(row, "utcTimeMillis"))
        if time_key not in time_index:
            continue
        signal_type = str(getattr(row, "SignalType"))
        key = (1, int(getattr(row, "Svid")), signal_type)
        slot_idx = slot_index.get(key)
        if slot_idx is None:
            continue
        freq = slot_frequency_label(signal_type)
        raw_lookup[(time_key, int(getattr(row, "Svid")), freq)] = (slot_idx, row)

    if not raw_lookup:
        return None

    shape = (len(times_ms), len(slot_keys))
    pseudorange = np.zeros(shape, dtype=np.float64)
    observable_pseudorange = np.zeros(shape, dtype=np.float64)
    weights = np.zeros(shape, dtype=np.float64)
    for row in observations.itertuples(index=False):
        time_key = int(getattr(row, "utcTimeMillis"))
        epoch_idx = time_index.get(time_key)
        raw_item = raw_lookup.get((time_key, int(getattr(row, "Svid")), str(getattr(row, "freq"))))
        if epoch_idx is None or raw_item is None:
            continue
        slot_idx, raw_row = raw_item
        signal_type = str(getattr(raw_row, "SignalType"))
        sat_clock_bias = np.nan
        if sat_clock_bias_m is not None and sat_clock_bias_m.shape == weights.shape:
            sat_clock_bias = float(sat_clock_bias_m[epoch_idx, slot_idx])
        if not np.isfinite(sat_clock_bias):
            adjustment_m = (
                sat_clock_adjustment_m(1, int(getattr(raw_row, "Svid")), signal_type, gps_tgd_m_by_svid)
                if sat_clock_adjustment_m is not None
                else 0.0
            )
            sat_clock_bias = float(getattr(raw_row, "SvClockBiasMeters")) + float(adjustment_m)
        observable_pr = float(getattr(row, "PseudorangeMeters"))
        corrected_pr = observable_pr + sat_clock_bias - float(getattr(raw_row, "IonosphericDelayMeters"))
        if rtklib_tropo_m is not None and rtklib_tropo_m.shape == weights.shape:
            tropo_m = float(rtklib_tropo_m[epoch_idx, slot_idx])
            if not np.isfinite(tropo_m):
                tropo_m = float(getattr(raw_row, "TroposphericDelayMeters"))
        else:
            tropo_m = float(getattr(raw_row, "TroposphericDelayMeters"))
        corrected_pr -= tropo_m
        if np.isfinite(corrected_pr):
            pseudorange[epoch_idx, slot_idx] = corrected_pr
            observable_pseudorange[epoch_idx, slot_idx] = observable_pr
            weights[epoch_idx, slot_idx] = 1.0

    if np.count_nonzero(weights) == 0:
        return None
    return GnssLogPseudorangeProducts(
        pseudorange=pseudorange,
        weights=weights,
        observable_pseudorange=observable_pseudorange,
    )
