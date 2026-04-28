"""Lightweight GSDC2023 ``gnss_log.txt`` observation reader.

The goal of this module is not to replace MatRTKLIB.  It mirrors the early
``gnsslog2obs.m`` / ``exobs.m`` steps that are needed for MATLAB parity audits:
parse ``Raw`` rows, apply the same initial row filters, derive L1/L5 observation
keys, and count finite P/D/L availability.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


C_LIGHT = 299_792_458.0
GPS_WEEK_SECONDS = 604_800.0
GPS_WEEK_NANOS = 604_800_000_000_000
DAY_SECONDS = 86_400.0

FREQ1_HZ = 1_575_420_000.0
FREQ5_HZ = 1_176_450_000.0
FREQ1_CMP_HZ = 1_561_098_000.0
FREQ1_GLO_HZ = 1_602_000_000.0
DFRQ1_GLO_HZ = 562_500.0

ANDROID_STATE_CODE_LOCK = (1 << 0) | (1 << 10)
ANDROID_STATE_TOW_OK = (1 << 3) | (1 << 14)
ANDROID_STATE_TOD_OK = (1 << 7) | (1 << 15)
ANDROID_ADR_STATE_VALID = 1 << 0
ANDROID_ADR_STATE_SLIP = (1 << 1) | (1 << 2)

SYS_GPS = 1
SYS_GLO = 2
SYS_QZS = 4
SYS_GAL = 8
SYS_CMP = 32

RAW_COLUMNS = [
    "utcTimeMillis",
    "TimeNanos",
    "LeapSecond",
    "TimeUncertaintyNanos",
    "FullBiasNanos",
    "BiasNanos",
    "BiasUncertaintyNanos",
    "DriftNanosPerSecond",
    "DriftUncertaintyNanosPerSecond",
    "HardwareClockDiscontinuityCount",
    "Svid",
    "TimeOffsetNanos",
    "State",
    "ReceivedSvTimeNanos",
    "ReceivedSvTimeUncertaintyNanos",
    "Cn0DbHz",
    "PseudorangeRateMetersPerSecond",
    "PseudorangeRateUncertaintyMetersPerSecond",
    "AccumulatedDeltaRangeState",
    "AccumulatedDeltaRangeMeters",
    "AccumulatedDeltaRangeUncertaintyMeters",
    "CarrierFrequencyHz",
    "CarrierCycles",
    "CarrierPhase",
    "CarrierPhaseUncertainty",
    "MultipathIndicator",
    "SnrInDb",
    "ConstellationType",
    "AgcDb",
    "BasebandCn0DbHz",
    "FullInterSignalBiasNanos",
    "FullInterSignalBiasUncertaintyNanos",
    "SatelliteInterSignalBiasNanos",
    "SatelliteInterSignalBiasUncertaintyNanos",
    "CodeType",
    "ChipsetElapsedRealtimeNanos",
]

_OBS_FIELDS = ("P", "D", "L")


def _read_raw_rows(log_path: Path) -> pd.DataFrame:
    rows: list[list[str]] = []
    with log_path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        for row in csv.reader(handle):
            if not row or row[0] != "Raw":
                continue
            values = row[1:]
            if len(values) < len(RAW_COLUMNS):
                values = values + [""] * (len(RAW_COLUMNS) - len(values))
            rows.append(values[: len(RAW_COLUMNS)])

    frame = pd.DataFrame(rows, columns=RAW_COLUMNS)
    for column in RAW_COLUMNS:
        if column == "CodeType":
            continue
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _apply_repeated_observation_workaround(frame: pd.DataFrame, log_path: Path) -> pd.DataFrame:
    path_text = str(log_path).lower()
    if "sm-a205u" not in path_text and "sm-a600t" not in path_text:
        return frame

    keep = np.zeros(len(frame), dtype=bool)
    starts = np.r_[0, np.flatnonzero(frame["utcTimeMillis"].to_numpy()[1:] != frame["utcTimeMillis"].to_numpy()[:-1]) + 1]
    stops = np.r_[starts[1:], len(frame)]
    for start, stop in zip(starts, stops):
        half = int((stop - start) / 2)
        keep[start : start + half] = True
    return frame.loc[keep].copy()


def _constellation_to_sys(constellation: pd.Series) -> pd.Series:
    mapping = {
        1: SYS_GPS,
        3: SYS_GLO,
        4: SYS_QZS,
        5: SYS_CMP,
        6: SYS_GAL,
    }
    return constellation.map(mapping).astype("Int64")


def _estimate_frequency(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    carrier = frame["CarrierFrequencyHz"].to_numpy(dtype=np.float64)
    const = frame["ConstellationType"].fillna(0).astype(np.int64).to_numpy()
    carrier_valid = np.isfinite(carrier)
    candidates = np.full((carrier.size, 3), np.nan, dtype=np.float64)
    candidates[:, 0] = FREQ1_HZ
    candidates[:, 1] = FREQ5_HZ
    candidates[:, 2] = FREQ1_CMP_HZ

    glo = (const == 3) & carrier_valid
    if np.any(glo):
        glo_channels = FREQ1_GLO_HZ + DFRQ1_GLO_HZ * np.arange(-7, 7, dtype=np.float64)
        nearest_glo = glo_channels[np.argmin(np.abs(carrier[glo, None] - glo_channels[None, :]), axis=1)]
        candidates[glo, 0] = nearest_glo
        candidates[glo, 1:] = np.nan

    delta = np.abs(carrier[:, None] - candidates)
    valid_delta = np.isfinite(delta).any(axis=1)
    nearest = np.full(carrier.size, np.nan, dtype=np.float64)
    freq_type = np.full(carrier.size, "", dtype=object)
    if np.any(valid_delta):
        nearest_idx = np.nanargmin(delta[valid_delta], axis=1)
        nearest[valid_delta] = candidates[np.flatnonzero(valid_delta), nearest_idx]
        freq_type[valid_delta] = np.where(np.isclose(nearest[valid_delta], FREQ5_HZ), "L5", "L1")
    return nearest, freq_type


def _unwrap(values: np.ndarray, period: float) -> np.ndarray:
    out = np.asarray(values, dtype=np.float64).copy()
    if out.size <= 1:
        return out
    delta = np.diff(out)
    jumps = period * np.cumsum(-np.round(delta / period))
    out[1:] = out[1:] + jumps
    return out


def _glonass_to_gps_seconds(glot: np.ndarray, gps_ref: np.ndarray) -> np.ndarray:
    day_of_week = np.floor(gps_ref / DAY_SECONDS)
    gpst = glot + day_of_week * DAY_SECONDS - 3.0 * 3600.0 + 18.0
    day_offset = day_of_week - np.floor(gpst / DAY_SECONDS)
    return gpst + day_offset * DAY_SECONDS


def _add_pseudorange_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        frame["PseudorangeMeters"] = np.array([], dtype=np.float64)
        return frame

    frame = frame.sort_values(["utcTimeMillis", "ConstellationType", "Svid", "CarrierFrequencyHz"]).copy()
    epoch_meta = frame.drop_duplicates("utcTimeMillis", keep="first").sort_values("utcTimeMillis")
    time_nanos = epoch_meta["TimeNanos"].to_numpy(dtype=np.int64)
    full_bias_nanos = epoch_meta["FullBiasNanos"].to_numpy(dtype=np.int64)
    bias_nanos = epoch_meta["BiasNanos"].to_numpy(dtype=np.float64)

    jump_idx = np.flatnonzero(np.abs(np.diff(time_nanos.astype(np.float64))) > 1.0e9) + 1
    base_idx = np.zeros(time_nanos.size, dtype=np.int64)
    for idx in jump_idx:
        base_idx[idx:] = idx
    base_bias_nanos = full_bias_nanos[base_idx]

    elapsed_nanos = time_nanos - base_bias_nanos
    week = np.floor(elapsed_nanos.astype(np.float64) / GPS_WEEK_NANOS).astype(np.int64)
    tow_rx_nanos = elapsed_nanos - week * GPS_WEEK_NANOS
    tow_rx = (
        tow_rx_nanos.astype(np.float64) / 1.0e9
        - bias_nanos / 1.0e9
    )
    epoch_tow = pd.Series(tow_rx, index=epoch_meta["utcTimeMillis"].astype(np.int64)).to_dict()

    frame["tow_rx_s"] = frame["utcTimeMillis"].astype(np.int64).map(epoch_tow).astype(np.float64)
    frame["tow_rx_s"] = frame["tow_rx_s"] - frame["TimeOffsetNanos"].fillna(0.0).to_numpy(dtype=np.float64) / 1.0e9
    frame["tow_tx_s"] = frame["ReceivedSvTimeNanos"].to_numpy(dtype=np.float64) / 1.0e9
    cmp_mask = frame["matlab_sys"].to_numpy(dtype=np.int64) == SYS_CMP
    frame.loc[cmp_mask, "tow_tx_s"] = frame.loc[cmp_mask, "tow_tx_s"] + 14.0
    glo_mask = frame["matlab_sys"].to_numpy(dtype=np.int64) == SYS_GLO
    if np.any(glo_mask):
        frame.loc[glo_mask, "tow_tx_s"] = _glonass_to_gps_seconds(
            frame.loc[glo_mask, "tow_tx_s"].to_numpy(dtype=np.float64),
            frame.loc[glo_mask, "tow_rx_s"].to_numpy(dtype=np.float64),
        )

    frame["PseudorangeMeters"] = np.nan
    for _, idx in frame.groupby(["matlab_sys", "Svid", "freq"], sort=False).groups.items():
        idx_list = list(idx)
        rx = _unwrap(frame.loc[idx_list, "tow_rx_s"].to_numpy(dtype=np.float64), GPS_WEEK_SECONDS)
        tx = _unwrap(frame.loc[idx_list, "tow_tx_s"].to_numpy(dtype=np.float64), GPS_WEEK_SECONDS)
        frame.loc[idx_list, "PseudorangeMeters"] = (rx - tx) * C_LIGHT
    return frame


def load_gnss_log_observations(log_path: Path) -> pd.DataFrame:
    """Load MATLAB-style long-form observations from ``gnss_log.txt`` Raw rows."""

    log_path = Path(log_path)
    frame = _read_raw_rows(log_path)
    if frame.empty:
        return pd.DataFrame()

    frame = frame[frame["TimeNanos"] != 0].copy()
    frame = _apply_repeated_observation_workaround(frame, log_path)
    frame.loc[frame["AccumulatedDeltaRangeMeters"] == 0.0, "AccumulatedDeltaRangeMeters"] = np.nan

    # Match gnsslog2obs.m initial constellation/data validity filters.
    frame = frame[~((frame["Svid"] > 24) & (frame["ConstellationType"] == 3))]
    frame = frame[~frame["ConstellationType"].isin([2, 4, 7])]
    frame = frame[~(frame["BiasUncertaintyNanos"] > 1.0e4)]
    frame = frame[~(frame["ReceivedSvTimeNanos"] < 1.0e10)].copy()
    if frame.empty:
        return frame

    frame["matlab_sys"] = _constellation_to_sys(frame["ConstellationType"])
    freq_hz, freq = _estimate_frequency(frame)
    frame["freq_hz"] = freq_hz
    frame["freq"] = freq
    frame["lambda_m"] = C_LIGHT / frame["freq_hz"]
    frame["epoch_index"] = pd.factorize(frame["utcTimeMillis"], sort=True)[0] + 1
    frame["D_cycles"] = -frame["PseudorangeRateMetersPerSecond"] / frame["lambda_m"]
    frame["L_cycles"] = frame["AccumulatedDeltaRangeMeters"] / frame["lambda_m"]
    frame = _add_pseudorange_columns(frame)
    return frame


def apply_gnss_log_signal_masks(
    observations: pd.DataFrame,
    *,
    phone: str = "",
    frequencies: Iterable[str] = ("L1", "L5"),
    min_cn0_dbhz: float = 20.0,
) -> pd.DataFrame:
    """Return observations with MATLAB ``exobs.m`` P/D/L validity columns."""

    frame = observations.copy()
    if frame.empty:
        for field in _OBS_FIELDS:
            frame[f"{field}_ok"] = np.array([], dtype=bool)
        return frame

    frame = frame[frame["freq"].isin(tuple(frequencies))].copy()
    cn0_ok = frame["Cn0DbHz"].to_numpy(dtype=np.float64) >= float(min_cn0_dbhz)
    multipath_ok = frame["MultipathIndicator"].fillna(0).to_numpy(dtype=np.float64) != 1.0
    state = frame["State"].fillna(0).astype(np.int64).to_numpy()
    adr_state = frame["AccumulatedDeltaRangeState"].fillna(0).astype(np.int64).to_numpy()
    is_glo = frame["matlab_sys"].fillna(0).astype(np.int64).to_numpy() == SYS_GLO

    p_state_ok = (state & ANDROID_STATE_CODE_LOCK) != 0
    p_state_ok &= np.where(is_glo, (state & ANDROID_STATE_TOD_OK) != 0, (state & ANDROID_STATE_TOW_OK) != 0)
    p = frame["PseudorangeMeters"].to_numpy(dtype=np.float64)
    frame["P_ok"] = cn0_ok & multipath_ok & p_state_ok & np.isfinite(p) & (p >= 1.0e7) & (p <= 4.0e7)
    frame["D_ok"] = (
        cn0_ok
        & multipath_ok
        & np.isfinite(frame["PseudorangeRateMetersPerSecond"].to_numpy(dtype=np.float64))
    )
    frame["L_ok"] = (
        cn0_ok
        & multipath_ok
        & np.isfinite(frame["AccumulatedDeltaRangeMeters"].to_numpy(dtype=np.float64))
        & ((adr_state & ANDROID_ADR_STATE_SLIP) == 0)
        & ((adr_state & ANDROID_ADR_STATE_VALID) != 0)
    )

    if phone.lower() in {"sm-a205u", "sm-a217m", "samsungs22ultra", "sm-s908b", "sm-a505g", "sm-a600t", "sm-a505u"}:
        frame.loc[frame["matlab_sys"] == SYS_GLO, "L_ok"] = False

    for _, idx in frame.sort_values("epoch_index").groupby(["matlab_sys", "Svid", "freq"], sort=False).groups.items():
        idx_list = list(idx)
        l_values = frame.loc[idx_list, "L_cycles"].to_numpy(dtype=np.float64)
        jump = np.r_[False, np.abs(np.diff(l_values)) > 2.0e4]
        if np.any(jump):
            frame.loc[np.asarray(idx_list, dtype=object)[jump].tolist(), "L_ok"] = False
    return frame


def gnss_log_observation_counts(
    log_path: Path,
    *,
    gps_only: bool = False,
    apply_signal_mask: bool = False,
    phone: str = "",
) -> dict[str, dict[str, int]]:
    """Count finite L1/L5 P/D/L availability from ``gnss_log.txt``."""

    obs = load_gnss_log_observations(log_path)
    if obs.empty:
        return {}
    obs = obs[obs["freq"].isin(("L1", "L5"))].copy()
    if gps_only:
        obs = obs[obs["matlab_sys"] == SYS_GPS].copy()
    if apply_signal_mask:
        obs = apply_gnss_log_signal_masks(obs, phone=phone)
        masks = {
            "P": obs["P_ok"],
            "D": obs["D_ok"],
            "L": obs["L_ok"],
        }
    else:
        masks = {
            "P": np.isfinite(obs["PseudorangeMeters"].to_numpy(dtype=np.float64)),
            "D": np.isfinite(obs["D_cycles"].to_numpy(dtype=np.float64)),
            "L": np.isfinite(obs["L_cycles"].to_numpy(dtype=np.float64)),
        }

    counts: dict[str, dict[str, int]] = {freq: {field: 0 for field in _OBS_FIELDS} for freq in ("L1", "L5")}
    for freq in ("L1", "L5"):
        freq_mask = obs["freq"].to_numpy(dtype=object) == freq
        for field in _OBS_FIELDS:
            counts[freq][field] = int(np.count_nonzero(np.asarray(masks[field]) & freq_mask))
    return counts


def gnss_log_signal_mask_frame(
    log_path: Path,
    *,
    phone: str = "",
    gps_only: bool = False,
) -> pd.DataFrame:
    """Return long-form P/D/L keys after MATLAB ``exobs.m`` signal masks."""

    obs = load_gnss_log_observations(log_path)
    if obs.empty:
        return pd.DataFrame(columns=["field", "freq", "epoch_index", "utcTimeMillis", "sys", "svid"])
    obs = obs[obs["freq"].isin(("L1", "L5"))].copy()
    if gps_only:
        obs = obs[obs["matlab_sys"] == SYS_GPS].copy()
    obs = apply_gnss_log_signal_masks(obs, phone=phone)

    rows: list[pd.DataFrame] = []
    base_cols = ["freq", "epoch_index", "utcTimeMillis", "matlab_sys", "Svid"]
    for field in _OBS_FIELDS:
        mask_col = f"{field}_ok"
        if mask_col not in obs.columns:
            continue
        sub = obs.loc[obs[mask_col], base_cols].copy()
        if sub.empty:
            continue
        sub.insert(0, "field", field)
        sub = sub.rename(columns={"matlab_sys": "sys", "Svid": "svid"})
        rows.append(sub)
    if not rows:
        return pd.DataFrame(columns=["field", "freq", "epoch_index", "utcTimeMillis", "sys", "svid"])
    out = pd.concat(rows, ignore_index=True)
    for col in ("epoch_index", "utcTimeMillis", "sys", "svid"):
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(np.int64)
    return out.drop_duplicates(["field", "freq", "epoch_index", "utcTimeMillis", "sys", "svid"])
