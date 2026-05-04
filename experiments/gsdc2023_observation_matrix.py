"""Raw observation loading and matrix helper primitives for GSDC2023."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.gsdc2023_imu import ecef_to_enu_relative as _ecef_to_enu_relative
from experiments.gsdc2023_tdcp import ADR_STATE_CYCLE_SLIP, ADR_STATE_RESET, ADR_STATE_VALID


RAW_GNSS_REQUIRED_COLUMNS = [
    "utcTimeMillis",
    "Svid",
    "ConstellationType",
    "SignalType",
    "RawPseudorangeMeters",
    "IonosphericDelayMeters",
    "TroposphericDelayMeters",
    "SvClockBiasMeters",
    "SvPositionXEcefMeters",
    "SvPositionYEcefMeters",
    "SvPositionZEcefMeters",
    "SvElevationDegrees",
    "Cn0DbHz",
    "WlsPositionXEcefMeters",
    "WlsPositionYEcefMeters",
    "WlsPositionZEcefMeters",
]
RAW_GNSS_OPTIONAL_COLUMNS = [
    "State",
    "MultipathIndicator",
    "SnrInDb",
    "RawPseudorangeUncertaintyMeters",
    "BiasUncertaintyNanos",
    "HardwareClockDiscontinuityCount",
    "ChipsetElapsedRealtimeNanos",
    "FullBiasNanos",
    "BiasNanos",
    "DriftNanosPerSecond",
    "ArrivalTimeNanosSinceGpsEpoch",
    "ReceivedSvTimeNanos",
    "PseudorangeRateMetersPerSecond",
    "PseudorangeRateUncertaintyMetersPerSecond",
    "AccumulatedDeltaRangeState",
    "AccumulatedDeltaRangeMeters",
    "AccumulatedDeltaRangeUncertaintyMeters",
    "SvVelocityXEcefMetersPerSecond",
    "SvVelocityYEcefMetersPerSecond",
    "SvVelocityZEcefMetersPerSecond",
    "SvClockDriftMetersPerSecond",
]
RAW_GNSS_COLUMNS = RAW_GNSS_REQUIRED_COLUMNS + RAW_GNSS_OPTIONAL_COLUMNS

BASELINE_BIAS_UNCERTAINTY_NANOS_MAX = 1.0e4
BASELINE_OUTLIER_WINDOW = 10
BASELINE_OUTLIER_THRESHOLD_FACTOR = 20.0
BASELINE_OUTLIER_FLOOR_M = 30.0
LIGHT_SPEED_MPS = 299792458.0
GPS_WEEK_SECONDS = 604800.0
GPS_LEAP_SECONDS = 18.0
EARTH_ROTATION_RATE_RAD_S = 7.2921151467e-5
GPS_EPOCH_UNIX_SECONDS = datetime(1980, 1, 6, tzinfo=timezone.utc).timestamp()
GPS_L1_FREQUENCY_HZ = 1575.42e6
GPS_L5_FREQUENCY_HZ = 1176.45e6
GPS_L5_SAT_PRODUCT_CLOCK_MATCH_THRESHOLD_M = 0.005
RTKLIB_SAT_VELOCITY_FORWARD_DIFF_HALF_STEP_S = 5.0e-4
OBS_MASK_MIN_CN0_DBHZ = 20.0
OBS_MASK_MIN_ELEVATION_DEG = 10.0
OBS_MASK_PSEUDORANGE_MIN_M = 1.0e7
OBS_MASK_PSEUDORANGE_MAX_M = 4.0e7
OBS_MASK_RESIDUAL_THRESHOLD_M = 20.0
OBS_MASK_RESIDUAL_THRESHOLD_L5_M = 15.0
OBS_MASK_DOPPLER_RESIDUAL_THRESHOLD_MPS = 3.0
OBS_MASK_PSEUDORANGE_DOPPLER_THRESHOLD_M = 40.0
OBS_MASK_PSEUDORANGE_DOPPLER_THRESHOLD_L5_M = 20.0
ANDROID_STATE_CODE_LOCK = (1 << 0) | (1 << 10)
ANDROID_STATE_TOW_OK = (1 << 3) | (1 << 14)
ANDROID_STATE_TOD_OK = (1 << 7) | (1 << 15)
CONSTELLATION_GLONASS = 3


@dataclass(frozen=True)
class TripArrays:
    times_ms: np.ndarray
    sat_ecef: np.ndarray
    pseudorange: np.ndarray
    weights: np.ndarray
    kaggle_wls: np.ndarray
    truth: np.ndarray
    max_sats: int
    has_truth: bool
    sys_kind: np.ndarray | None = None
    n_clock: int = 1
    sat_vel: np.ndarray | None = None
    sat_clock_bias_matrix: np.ndarray | None = None
    sat_clock_drift_mps: np.ndarray | None = None
    rtklib_iono_m: np.ndarray | None = None
    rtklib_tropo_m: np.ndarray | None = None
    doppler: np.ndarray | None = None
    doppler_weights: np.ndarray | None = None
    pseudorange_bias_weights: np.ndarray | None = None
    pseudorange_isb_by_group: dict[int, float] | None = None
    dt: np.ndarray | None = None
    tdcp_meas: np.ndarray | None = None
    tdcp_weights: np.ndarray | None = None
    n_sat_slots: int = 0
    slot_keys: tuple[tuple[int, int, str], ...] = ()
    elapsed_ns: np.ndarray | None = None
    clock_jump: np.ndarray | None = None
    clock_bias_m: np.ndarray | None = None
    clock_drift_mps: np.ndarray | None = None
    factor_dt_gap_count: int = 0
    stop_epochs: np.ndarray | None = None
    imu_preintegration: "IMUPreintegration | None" = None
    absolute_height_ref_ecef: np.ndarray | None = None
    absolute_height_ref_count: int = 0
    base_correction_count: int = 0
    observation_mask_count: int = 0
    residual_mask_count: int = 0
    doppler_residual_mask_count: int = 0
    pseudorange_doppler_mask_count: int = 0
    tdcp_consistency_mask_count: int = 0
    tdcp_geometry_correction_count: int = 0
    dual_frequency: bool = False


@dataclass(frozen=True)
class RawEpochObservation:
    time_ms: float
    group: pd.DataFrame
    baseline_xyz: np.ndarray
    truth_xyz: np.ndarray


@dataclass(frozen=True)
class ObservationMatrixProducts:
    times_ms: np.ndarray
    sat_ecef: np.ndarray
    pseudorange: np.ndarray
    pseudorange_observable: np.ndarray
    weights: np.ndarray
    pseudorange_bias_weights: np.ndarray
    sat_clock_bias_matrix: np.ndarray
    rtklib_iono_m: np.ndarray
    rtklib_tropo_m: np.ndarray
    kaggle_wls: np.ndarray
    truth: np.ndarray
    visible_max: int
    slot_keys: tuple[tuple[int, int, str], ...]
    n_clock: int
    elapsed_ns: np.ndarray | None = None
    sys_kind: np.ndarray | None = None
    clock_counts: np.ndarray | None = None
    clock_bias_m: np.ndarray | None = None
    clock_drift_mps: np.ndarray | None = None
    sat_vel: np.ndarray | None = None
    sat_clock_drift_mps: np.ndarray | None = None
    doppler: np.ndarray | None = None
    doppler_weights: np.ndarray | None = None
    adr: np.ndarray | None = None
    adr_state: np.ndarray | None = None
    adr_uncertainty: np.ndarray | None = None


def read_raw_gnss_csv(raw_path: Path, **kwargs) -> pd.DataFrame:
    kwargs.setdefault("low_memory", False)
    last_error: Exception | None = None
    for compression in ("zip", None):
        try:
            return pd.read_csv(raw_path, compression=compression, **kwargs)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"failed to read raw GNSS file: {raw_path}")


def _raw_gnss_usecols(raw_path: Path) -> list[str]:
    header = read_raw_gnss_csv(raw_path, nrows=0)
    available = set(header.columns)
    missing = [col for col in RAW_GNSS_REQUIRED_COLUMNS if col not in available]
    if missing:
        raise RuntimeError(f"device_gnss.csv missing columns: {missing}")
    return [col for col in RAW_GNSS_COLUMNS if col in available]


def load_raw_gnss_frame(raw_path: Path) -> pd.DataFrame:
    return read_raw_gnss_csv(raw_path, usecols=_raw_gnss_usecols(raw_path))


def _read_raw_gnss_csv_chunks(raw_path: Path, *, usecols: Sequence[str], chunksize: int) -> Iterable[pd.DataFrame]:
    last_error: Exception | None = None
    for compression in ("zip", None):
        try:
            reader = pd.read_csv(
                raw_path,
                compression=compression,
                usecols=list(usecols),
                chunksize=chunksize,
                low_memory=False,
            )
            for chunk in reader:
                yield chunk
            return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"failed to read raw GNSS file: {raw_path}")


def _filter_raw_gnss_frame_epoch_window(
    frame: pd.DataFrame,
    *,
    start_epoch: int,
    max_epochs: int,
    extra_epochs: int = 0,
) -> pd.DataFrame:
    if max_epochs <= 0:
        return frame
    stop_epoch = start_epoch + max_epochs + max(0, extra_epochs)
    if stop_epoch <= start_epoch:
        return frame.iloc[0:0].copy()
    epoch_times = pd.to_numeric(frame["utcTimeMillis"], errors="coerce").dropna().unique()
    selected_times = np.sort(epoch_times.astype(np.float64))[start_epoch:stop_epoch]
    if selected_times.size == 0:
        return frame.iloc[0:0].copy()
    return frame.loc[pd.to_numeric(frame["utcTimeMillis"], errors="coerce").isin(selected_times)].copy()


def _load_raw_gnss_frame_epoch_window_from_chunks(
    raw_path: Path,
    *,
    start_epoch: int,
    max_epochs: int,
    extra_epochs: int,
    chunksize: int,
    usecols: Sequence[str],
) -> pd.DataFrame:
    stop_epoch = start_epoch + max_epochs + max(0, extra_epochs)
    if stop_epoch <= start_epoch:
        return pd.DataFrame(columns=list(usecols))

    chunks: list[pd.DataFrame] = []
    time_rank: dict[float, int] = {}
    last_time: float | None = None
    for chunk in _read_raw_gnss_csv_chunks(raw_path, usecols=usecols, chunksize=chunksize):
        times = pd.to_numeric(chunk["utcTimeMillis"], errors="coerce")
        finite_times = times.dropna().to_numpy(dtype=np.float64)
        if finite_times.size:
            if np.any(finite_times[1:] < finite_times[:-1]) or (
                last_time is not None and finite_times[0] < last_time
            ):
                raise ValueError("device_gnss.csv utcTimeMillis is not sorted")
            last_time = float(finite_times[-1])
        for value in pd.unique(times.dropna()):
            key = float(value)
            if key not in time_rank:
                time_rank[key] = len(time_rank)

        ranks = times.astype(np.float64).map(time_rank)
        selected = ranks.ge(start_epoch) & ranks.lt(stop_epoch)
        if selected.any():
            chunks.append(chunk.loc[selected].copy())
        if len(time_rank) > stop_epoch:
            break

    if not chunks:
        return pd.DataFrame(columns=list(usecols))
    return pd.concat(chunks, ignore_index=True)


def load_raw_gnss_frame_epoch_window(
    raw_path: Path,
    *,
    start_epoch: int,
    max_epochs: int,
    extra_epochs: int = 0,
    chunksize: int = 5_000,
) -> pd.DataFrame:
    if start_epoch < 0:
        raise ValueError("start_epoch must be non-negative")
    if max_epochs <= 0:
        return load_raw_gnss_frame(raw_path)

    usecols = _raw_gnss_usecols(raw_path)
    try:
        return _load_raw_gnss_frame_epoch_window_from_chunks(
            raw_path,
            start_epoch=start_epoch,
            max_epochs=max_epochs,
            extra_epochs=extra_epochs,
            chunksize=chunksize,
            usecols=usecols,
        )
    except Exception:  # noqa: BLE001
        return _filter_raw_gnss_frame_epoch_window(
            load_raw_gnss_frame(raw_path),
            start_epoch=start_epoch,
            max_epochs=max_epochs,
            extra_epochs=extra_epochs,
        )


def android_state_tracking_ok(df: pd.DataFrame) -> np.ndarray:
    if "State" not in df.columns:
        return np.ones(len(df), dtype=bool)
    state = pd.to_numeric(df["State"], errors="coerce").fillna(0).astype(np.int64).to_numpy()
    code_lock = (state & ANDROID_STATE_CODE_LOCK) != 0
    if "ConstellationType" in df.columns:
        constellation = pd.to_numeric(df["ConstellationType"], errors="coerce").fillna(0).astype(np.int64).to_numpy()
        time_ok = np.where(
            constellation == CONSTELLATION_GLONASS,
            (state & ANDROID_STATE_TOD_OK) != 0,
            (state & ANDROID_STATE_TOW_OK) != 0,
        )
    else:
        time_ok = (state & ANDROID_STATE_TOW_OK) != 0
    return code_lock & time_ok


def apply_matlab_signal_observation_mask(
    df: pd.DataFrame,
    *,
    min_cn0_dbhz: float,
    min_elevation_deg: float,
) -> tuple[pd.DataFrame, int]:
    p_ok, _d_ok, _l_ok = matlab_signal_observation_masks(
        df,
        min_cn0_dbhz=min_cn0_dbhz,
        min_elevation_deg=min_elevation_deg,
    )
    return df.loc[p_ok].copy(), int(np.count_nonzero(~p_ok))


def matlab_signal_observation_masks(
    df: pd.DataFrame,
    *,
    min_cn0_dbhz: float,
    min_elevation_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if df.empty:
        empty = np.zeros(0, dtype=bool)
        return empty, empty, empty
    common = np.ones(len(df), dtype=bool)
    cn0 = pd.to_numeric(df["Cn0DbHz"], errors="coerce").to_numpy(dtype=np.float64)
    if min_cn0_dbhz > 0.0:
        common &= np.isfinite(cn0) & (cn0 >= float(min_cn0_dbhz))
    elevation = pd.to_numeric(df["SvElevationDegrees"], errors="coerce").to_numpy(dtype=np.float64)
    if min_elevation_deg > 0.0:
        common &= np.isfinite(elevation) & (elevation >= float(min_elevation_deg))
    if "MultipathIndicator" in df.columns:
        multipath = pd.to_numeric(df["MultipathIndicator"], errors="coerce").fillna(0).to_numpy(dtype=np.float64)
        common &= multipath != 1.0

    pr = pd.to_numeric(df["RawPseudorangeMeters"], errors="coerce").to_numpy(dtype=np.float64)
    p_ok = (
        common
        & android_state_tracking_ok(df)
        & np.isfinite(pr)
        & (pr >= OBS_MASK_PSEUDORANGE_MIN_M)
        & (pr <= OBS_MASK_PSEUDORANGE_MAX_M)
    )
    d_ok = common.copy()
    if "PseudorangeRateMetersPerSecond" in df.columns:
        doppler = pd.to_numeric(df["PseudorangeRateMetersPerSecond"], errors="coerce").to_numpy(dtype=np.float64)
        d_ok &= np.isfinite(doppler)
    else:
        d_ok[:] = False

    if "AccumulatedDeltaRangeState" in df.columns and "AccumulatedDeltaRangeMeters" in df.columns:
        adr_state = (
            pd.to_numeric(df["AccumulatedDeltaRangeState"], errors="coerce")
            .fillna(0)
            .astype(np.int64)
            .to_numpy()
        )
        adr = pd.to_numeric(df["AccumulatedDeltaRangeMeters"], errors="coerce").to_numpy(dtype=np.float64)
        l_ok = (
            common
            & np.isfinite(adr)
            & (adr != 0.0)
            & ((adr_state & ADR_STATE_VALID) != 0)
            & ((adr_state & (ADR_STATE_RESET | ADR_STATE_CYCLE_SLIP)) == 0)
        )
    else:
        l_ok = np.zeros(len(df), dtype=bool)
    return p_ok, d_ok, l_ok


def legacy_matlab_signal_observation_mask(
    df: pd.DataFrame,
    *,
    min_cn0_dbhz: float,
    min_elevation_deg: float,
) -> tuple[pd.DataFrame, int]:
    if df.empty:
        return df, 0
    mask = np.ones(len(df), dtype=bool)
    cn0 = pd.to_numeric(df["Cn0DbHz"], errors="coerce").to_numpy(dtype=np.float64)
    if min_cn0_dbhz > 0.0:
        mask &= np.isfinite(cn0) & (cn0 >= float(min_cn0_dbhz))
    elevation = pd.to_numeric(df["SvElevationDegrees"], errors="coerce").to_numpy(dtype=np.float64)
    if min_elevation_deg > 0.0:
        mask &= np.isfinite(elevation) & (elevation >= float(min_elevation_deg))
    pr = pd.to_numeric(df["RawPseudorangeMeters"], errors="coerce").to_numpy(dtype=np.float64)
    mask &= (
        np.isfinite(pr)
        & (pr >= OBS_MASK_PSEUDORANGE_MIN_M)
        & (pr <= OBS_MASK_PSEUDORANGE_MAX_M)
    )
    if "MultipathIndicator" in df.columns:
        multipath = pd.to_numeric(df["MultipathIndicator"], errors="coerce").fillna(0).to_numpy(dtype=np.float64)
        mask &= multipath != 1.0
    mask &= android_state_tracking_ok(df)
    return df.loc[mask].copy(), int(np.count_nonzero(~mask))


def build_epoch_metadata_frame(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "utcTimeMillis",
        "WlsPositionXEcefMeters",
        "WlsPositionYEcefMeters",
        "WlsPositionZEcefMeters",
    ]
    if "BiasUncertaintyNanos" in df.columns:
        cols.append("BiasUncertaintyNanos")
    if "HardwareClockDiscontinuityCount" in df.columns:
        cols.append("HardwareClockDiscontinuityCount")
    if "ChipsetElapsedRealtimeNanos" in df.columns:
        cols.append("ChipsetElapsedRealtimeNanos")
    if "FullBiasNanos" in df.columns:
        cols.append("FullBiasNanos")
    if "BiasNanos" in df.columns:
        cols.append("BiasNanos")
    if "DriftNanosPerSecond" in df.columns:
        cols.append("DriftNanosPerSecond")

    base_rows = df[cols].copy()
    fallback = base_rows.sort_values("utcTimeMillis").groupby("utcTimeMillis", as_index=False).first()
    if "BiasUncertaintyNanos" not in base_rows.columns:
        return fallback

    bias_ok = (~np.isfinite(base_rows["BiasUncertaintyNanos"])) | (
        base_rows["BiasUncertaintyNanos"] <= BASELINE_BIAS_UNCERTAINTY_NANOS_MAX
    )
    preferred_rows = base_rows[bias_ok]
    if preferred_rows.empty:
        return fallback

    preferred = preferred_rows.sort_values("utcTimeMillis").groupby("utcTimeMillis", as_index=False).first()
    merged = fallback.set_index("utcTimeMillis")
    preferred = preferred.set_index("utcTimeMillis")
    for col in preferred.columns:
        merged.loc[preferred.index, col] = preferred[col]
    return merged.reset_index()


def interpolate_series(times_ms: np.ndarray, values: np.ndarray) -> np.ndarray:
    out = np.asarray(values, dtype=np.float64).copy()
    if out.size == 0:
        return out
    finite = np.isfinite(out)
    if finite.all() or not finite.any():
        return out
    x = np.asarray(times_ms, dtype=np.float64)
    out[~finite] = np.interp(x[~finite], x[finite], out[finite])
    return out


def repair_baseline_wls(times_ms: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    repaired = np.asarray(xyz, dtype=np.float64).reshape(-1, 3).copy()
    if repaired.size == 0:
        return repaired

    finite_rows = np.isfinite(repaired).all(axis=1)
    if not finite_rows.any():
        return repaired

    for axis in range(3):
        repaired[:, axis] = interpolate_series(times_ms, repaired[:, axis])

    finite_rows = np.isfinite(repaired).all(axis=1)
    if not finite_rows.any():
        return repaired

    origin_xyz = repaired[np.flatnonzero(finite_rows)[0]]
    enu = _ecef_to_enu_relative(repaired, origin_xyz)
    outlier_mask = np.zeros(repaired.shape[0], dtype=bool)
    for axis in range(3):
        coord = enu[:, axis]
        median = (
            pd.Series(coord)
            .rolling(BASELINE_OUTLIER_WINDOW, center=True, min_periods=1)
            .median()
            .to_numpy(dtype=np.float64)
        )
        residual = np.abs(coord - median)
        mad = (
            pd.Series(residual)
            .rolling(BASELINE_OUTLIER_WINDOW, center=True, min_periods=1)
            .median()
            .to_numpy(dtype=np.float64)
        )
        threshold = np.maximum(
            BASELINE_OUTLIER_THRESHOLD_FACTOR * 1.4826 * mad,
            BASELINE_OUTLIER_FLOOR_M,
        )
        outlier_mask |= residual > threshold

    if repaired.shape[0] >= 5:
        for row_idx in range(3, repaired.shape[0] - 1):
            if not np.isfinite(repaired[row_idx - 3 : row_idx + 2]).all():
                continue
            stale_previous = (
                np.linalg.norm(repaired[row_idx - 1] - repaired[row_idx - 2]) < 0.05
                and np.linalg.norm(repaired[row_idx - 2] - repaired[row_idx - 3]) < 0.05
            )
            if not stale_previous:
                continue
            second_difference = np.linalg.norm(repaired[row_idx] - 0.5 * (repaired[row_idx - 1] + repaired[row_idx + 1]))
            jump_from_stale = np.linalg.norm(repaired[row_idx] - repaired[row_idx - 1])
            if second_difference > BASELINE_OUTLIER_FLOOR_M and jump_from_stale > 2.0 * BASELINE_OUTLIER_FLOOR_M:
                outlier_mask[row_idx] = True

    if not outlier_mask.any():
        return repaired

    repaired[outlier_mask] = np.nan
    for axis in range(3):
        repaired[:, axis] = interpolate_series(times_ms, repaired[:, axis])
    return repaired


def select_epoch_observations(
    epoch_time_keys: Iterable[float],
    grouped_observations: Mapping[float, pd.DataFrame],
    baseline_lookup: Mapping[int, np.ndarray],
    gt_times: np.ndarray | None,
    gt_ecef: np.ndarray | None,
    *,
    start_epoch: int,
    max_epochs: int,
    nearest_index_fn: Callable[[np.ndarray, float], int],
) -> list[RawEpochObservation]:
    empty_group = (
        next(iter(grouped_observations.values())).iloc[0:0].copy()
        if grouped_observations
        else pd.DataFrame()
    )
    epochs: list[RawEpochObservation] = []
    usable_epoch_index = 0
    for tow_ms in sorted(epoch_time_keys):
        if usable_epoch_index < start_epoch:
            usable_epoch_index += 1
            continue

        group = grouped_observations.get(float(tow_ms), empty_group)
        if group.empty:
            group = group.copy()
        else:
            group = group.sort_values(["ConstellationType", "Svid"]).copy()
        epoch_key = int(round(float(tow_ms)))
        if group.empty:
            baseline_xyz = baseline_lookup.get(epoch_key, np.full(3, np.nan, dtype=np.float64))
        else:
            baseline_xyz = group[
                ["WlsPositionXEcefMeters", "WlsPositionYEcefMeters", "WlsPositionZEcefMeters"]
            ].iloc[0].to_numpy(dtype=np.float64)
        if gt_times is not None and gt_ecef is not None:
            gt_idx = nearest_index_fn(gt_times, float(tow_ms))
            truth_xyz = gt_ecef[gt_idx]
        else:
            truth_xyz = np.full(3, np.nan, dtype=np.float64)
        epochs.append(
            RawEpochObservation(
                time_ms=float(tow_ms),
                group=group,
                baseline_xyz=baseline_xyz,
                truth_xyz=truth_xyz,
            ),
        )
        usable_epoch_index += 1
        if len(epochs) >= max_epochs:
            break
    return epochs


def _gps_received_sv_tow_s_from_row(row: Any) -> float:
    received_ns = float(getattr(row, "ReceivedSvTimeNanos", np.nan))
    if not np.isfinite(received_ns) or received_ns < 1.0e10:
        return float("nan")
    return float((received_ns * 1.0e-9) % GPS_WEEK_SECONDS)


def _row_has_usable_bridge_observation(row: Any) -> bool:
    flags = [
        getattr(row, "bridge_p_ok", None),
        getattr(row, "bridge_d_ok", None),
        getattr(row, "bridge_l_ok", None),
    ]
    known_flags = [flag for flag in flags if flag is not None and not pd.isna(flag)]
    if not known_flags:
        return True
    return any(bool(flag) for flag in known_flags)


def _row_is_gnss_log_only(row: Any) -> bool:
    flag = getattr(row, "bridge_gnss_log_only", False)
    if flag is None or pd.isna(flag):
        return False
    return bool(flag)


def _row_can_seed_gps_l1_product(row: Any) -> bool:
    if not _row_is_gnss_log_only(row):
        return True
    return _row_has_usable_bridge_observation(row)


def build_gps_l1_sat_time_lookup(
    epochs: Sequence[RawEpochObservation],
    *,
    gps_arrival_tow_s_from_row_fn: Callable[[Any], float],
    gps_sat_clock_bias_adjustment_m_fn: Callable[[int, int, str, Mapping[int, float]], float],
    gps_tgd_m_by_svid: Mapping[int, float],
    is_l5_signal_fn: Callable[[str], bool],
    include_masked_gnss_log_only: bool = False,
) -> dict[tuple[int, int], tuple[float, float, float, float]]:
    out: dict[tuple[int, int], tuple[float, float, float, float]] = {}
    for epoch_idx, epoch in enumerate(epochs):
        for row in epoch.group.itertuples(index=False):
            if int(row.ConstellationType) != 1 or is_l5_signal_fn(str(row.SignalType)):
                continue
            if not include_masked_gnss_log_only and not _row_can_seed_gps_l1_product(row):
                continue
            arrival_tow_s = gps_arrival_tow_s_from_row_fn(row)
            received_sv_tow_s = _gps_received_sv_tow_s_from_row(row)
            raw_pr = float(row.RawPseudorangeMeters)
            common_clock_m = float(row.SvClockBiasMeters) + gps_sat_clock_bias_adjustment_m_fn(
                int(row.ConstellationType),
                int(row.Svid),
                str(row.SignalType),
                gps_tgd_m_by_svid,
            )
            if np.isfinite(arrival_tow_s) and np.isfinite(raw_pr) and np.isfinite(common_clock_m):
                out[(epoch_idx, int(row.Svid))] = (arrival_tow_s, raw_pr, common_clock_m, received_sv_tow_s)
    return out


def build_gps_l1_sat_product_lookup(
    epochs: Sequence[RawEpochObservation],
    *,
    is_l5_signal_fn: Callable[[str], bool],
) -> dict[tuple[int, int], Any]:
    out: dict[tuple[int, int], Any] = {}
    for epoch_idx, epoch in enumerate(epochs):
        for row in epoch.group.itertuples(index=False):
            if int(row.ConstellationType) != 1 or is_l5_signal_fn(str(row.SignalType)):
                continue
            if not _row_can_seed_gps_l1_product(row):
                continue
            out[(epoch_idx, int(row.Svid))] = row
    return out


def build_sat_velocity_forward_difference_lookup(
    epochs: Sequence[RawEpochObservation],
    *,
    half_step_s: float = RTKLIB_SAT_VELOCITY_FORWARD_DIFF_HALF_STEP_S,
) -> dict[tuple[int, int, int, str], np.ndarray]:
    samples: dict[tuple[int, int, str], list[tuple[int, float, np.ndarray]]] = {}
    for epoch_idx, epoch in enumerate(epochs):
        for row in epoch.group.itertuples(index=False):
            try:
                velocity = np.array(
                    [
                        float(row.SvVelocityXEcefMetersPerSecond),
                        float(row.SvVelocityYEcefMetersPerSecond),
                        float(row.SvVelocityZEcefMetersPerSecond),
                    ],
                    dtype=np.float64,
                )
            except AttributeError:
                continue
            if not np.isfinite(velocity).all():
                continue
            key = (int(row.ConstellationType), int(row.Svid), str(row.SignalType))
            samples.setdefault(key, []).append((epoch_idx, float(epoch.time_ms), velocity))

    out: dict[tuple[int, int, int, str], np.ndarray] = {}
    for key, group_samples in samples.items():
        ordered = sorted(group_samples, key=lambda item: (item[1], item[0]))
        epoch_indices = [item[0] for item in ordered]
        times_s = np.array([item[1] for item in ordered], dtype=np.float64) * 1.0e-3
        velocities = np.vstack([item[2] for item in ordered])
        corrected = velocities.copy()
        if velocities.shape[0] >= 2 and np.all(np.diff(times_s) > 0.0):
            # RTKLIB ephpos reports velocity as a 1 ms forward difference,
            # which is centered half a millisecond after the raw velocity epoch.
            acceleration = np.zeros_like(velocities)
            for axis in range(3):
                acceleration[:, axis] = np.gradient(velocities[:, axis], times_s, edge_order=1)
            corrected = velocities + float(half_step_s) * acceleration
        constellation_type, svid, signal_type = key
        for epoch_idx, velocity in zip(epoch_indices, corrected):
            out[(int(epoch_idx), int(constellation_type), int(svid), str(signal_type))] = velocity
    return out


def build_slot_keys(
    epochs: Sequence[RawEpochObservation],
    *,
    slot_sort_key_fn: Callable[[tuple[int, int, str]], Any],
) -> tuple[tuple[int, int, str], ...]:
    return tuple(
        sorted(
            {
                (int(row.ConstellationType), int(row.Svid), str(row.SignalType))
                for epoch in epochs
                for row in epoch.group.itertuples(index=False)
            },
            key=slot_sort_key_fn,
        ),
    )


def _rtklib_tropo_for_satellite(
    rx_xyz: np.ndarray,
    sat_xyz: np.ndarray,
    *,
    ecef_to_lla_fn: Callable[[float, float, float], tuple[float, float, float]],
    elevation_azimuth_fn: Callable[[np.ndarray, np.ndarray], tuple[float, float]],
    rtklib_tropo_fn: Callable[[float, float, float], float],
) -> float:
    if not np.isfinite(rx_xyz).all() or not np.isfinite(sat_xyz).all():
        return float("nan")
    lat_rad, _lon_rad, alt_m = ecef_to_lla_fn(float(rx_xyz[0]), float(rx_xyz[1]), float(rx_xyz[2]))
    el_rad, _az_rad = elevation_azimuth_fn(rx_xyz, sat_xyz)
    tropo_m = rtklib_tropo_fn(float(lat_rad), float(alt_m), float(el_rad))
    if np.isfinite(tropo_m) and tropo_m > 0.0:
        return float(tropo_m)
    return float("nan")


def rtklib_klobuchar_iono_delay_m(
    alpha: Sequence[float],
    beta: Sequence[float],
    lat_rad: float,
    lon_rad: float,
    az_rad: float,
    el_rad: float,
    gps_tow_s: float,
) -> float:
    if (
        not np.isfinite(lat_rad)
        or not np.isfinite(lon_rad)
        or not np.isfinite(az_rad)
        or not np.isfinite(el_rad)
        or not np.isfinite(gps_tow_s)
        or el_rad <= 0.0
    ):
        return 0.0
    alpha_arr = np.asarray(alpha, dtype=np.float64).reshape(-1)
    beta_arr = np.asarray(beta, dtype=np.float64).reshape(-1)
    if alpha_arr.size < 4 or beta_arr.size < 4 or not np.isfinite(alpha_arr[:4]).all() or not np.isfinite(beta_arr[:4]).all():
        alpha_arr = np.array([0.1118e-07, -0.7451e-08, -0.5961e-07, 0.1192e-06], dtype=np.float64)
        beta_arr = np.array([0.1167e06, -0.2294e06, -0.1311e06, 0.1049e07], dtype=np.float64)
    psi = 0.0137 / (float(el_rad) / np.pi + 0.11) - 0.022
    phi = float(lat_rad) / np.pi + psi * np.cos(float(az_rad))
    phi = min(max(phi, -0.416), 0.416)
    lam = float(lon_rad) / np.pi + psi * np.sin(float(az_rad)) / np.cos(phi * np.pi)
    phi += 0.064 * np.cos((lam - 1.617) * np.pi)
    tt = 43200.0 * lam + float(gps_tow_s)
    tt -= np.floor(tt / 86400.0) * 86400.0
    slant = 1.0 + 16.0 * (0.53 - float(el_rad) / np.pi) ** 3
    amp = alpha_arr[0] + phi * (alpha_arr[1] + phi * (alpha_arr[2] + phi * alpha_arr[3]))
    per = beta_arr[0] + phi * (beta_arr[1] + phi * (beta_arr[2] + phi * beta_arr[3]))
    amp = max(float(amp), 0.0)
    per = max(float(per), 72000.0)
    x = 2.0 * np.pi * (tt - 50400.0) / per
    if abs(x) < 1.57:
        delay_s = 5.0e-9 + amp * (1.0 + x * x * (-0.5 + x * x / 24.0))
    else:
        delay_s = 5.0e-9
    return float(LIGHT_SPEED_MPS * slant * delay_s)


def signal_type_iono_scale(signal_type: str) -> float:
    freq_hz = GPS_L5_FREQUENCY_HZ if "L5" in str(signal_type).upper() else GPS_L1_FREQUENCY_HZ
    return float((GPS_L1_FREQUENCY_HZ / freq_hz) ** 2)


def _rtklib_iono_for_satellite(
    time_ms: float,
    rx_xyz: np.ndarray,
    sat_xyz: np.ndarray,
    signal_type: str,
    gps_iono_alpha_beta: tuple[Sequence[float], Sequence[float]] | None,
    *,
    ecef_to_lla_fn: Callable[[float, float, float], tuple[float, float, float]],
    elevation_azimuth_fn: Callable[[np.ndarray, np.ndarray], tuple[float, float]],
) -> float:
    if gps_iono_alpha_beta is None:
        return float("nan")
    if not np.isfinite(rx_xyz).all() or not np.isfinite(sat_xyz).all():
        return float("nan")
    alpha, beta = gps_iono_alpha_beta
    lat_rad, lon_rad, _alt_m = ecef_to_lla_fn(float(rx_xyz[0]), float(rx_xyz[1]), float(rx_xyz[2]))
    el_rad, az_rad = elevation_azimuth_fn(rx_xyz, sat_xyz)
    gps_abs_s = float(time_ms) * 1.0e-3 + GPS_LEAP_SECONDS - GPS_EPOCH_UNIX_SECONDS
    gps_tow_s = gps_abs_s % GPS_WEEK_SECONDS
    iono_l1 = rtklib_klobuchar_iono_delay_m(alpha, beta, lat_rad, lon_rad, az_rad, el_rad, gps_tow_s)
    iono_m = signal_type_iono_scale(signal_type) * iono_l1
    return float(iono_m) if np.isfinite(iono_m) and iono_m >= 0.0 else float("nan")


def recompute_rtklib_iono_matrix(
    times_ms: np.ndarray,
    receiver_xyz: np.ndarray,
    sat_ecef: np.ndarray,
    slot_keys: Sequence[tuple[int, int, str]],
    gps_iono_alpha_beta: tuple[Sequence[float], Sequence[float]] | None,
    *,
    ecef_to_lla_fn: Callable[[float, float, float], tuple[float, float, float]],
    elevation_azimuth_fn: Callable[[np.ndarray, np.ndarray], tuple[float, float]],
    initial_iono_m: np.ndarray | None = None,
) -> np.ndarray:
    shape = sat_ecef.shape[:2]
    out = (
        np.asarray(initial_iono_m, dtype=np.float64).copy()
        if initial_iono_m is not None and np.asarray(initial_iono_m).shape == shape
        else np.full(shape, np.nan, dtype=np.float64)
    )
    if gps_iono_alpha_beta is None:
        return out
    times = np.asarray(times_ms, dtype=np.float64).reshape(-1)
    rx = np.asarray(receiver_xyz, dtype=np.float64).reshape(-1, 3)
    for epoch_idx in range(min(shape[0], times.size, rx.shape[0])):
        for slot_idx, key in enumerate(slot_keys):
            constellation, _svid, signal_type = key
            if int(constellation) != 1:
                continue
            iono_m = _rtklib_iono_for_satellite(
                float(times[epoch_idx]),
                rx[epoch_idx],
                sat_ecef[epoch_idx, slot_idx],
                str(signal_type),
                gps_iono_alpha_beta,
                ecef_to_lla_fn=ecef_to_lla_fn,
                elevation_azimuth_fn=elevation_azimuth_fn,
            )
            if np.isfinite(iono_m):
                out[epoch_idx, slot_idx] = iono_m
    return out


def recompute_rtklib_tropo_matrix(
    kaggle_wls: np.ndarray,
    sat_ecef: np.ndarray,
    *,
    ecef_to_lla_fn: Callable[[float, float, float], tuple[float, float, float]],
    elevation_azimuth_fn: Callable[[np.ndarray, np.ndarray], tuple[float, float]],
    rtklib_tropo_fn: Callable[[float, float, float], float],
    initial_tropo_m: np.ndarray | None = None,
) -> np.ndarray:
    n_epoch, n_sat_slots = sat_ecef.shape[:2]
    out = (
        np.asarray(initial_tropo_m, dtype=np.float64).copy()
        if initial_tropo_m is not None
        else np.full((n_epoch, n_sat_slots), np.nan, dtype=np.float64)
    )
    for epoch_idx in range(n_epoch):
        rx_xyz = kaggle_wls[epoch_idx]
        if not np.isfinite(rx_xyz).all():
            continue
        for sat_idx in range(n_sat_slots):
            tropo_m = _rtklib_tropo_for_satellite(
                rx_xyz,
                sat_ecef[epoch_idx, sat_idx],
                ecef_to_lla_fn=ecef_to_lla_fn,
                elevation_azimuth_fn=elevation_azimuth_fn,
                rtklib_tropo_fn=rtklib_tropo_fn,
            )
            if np.isfinite(tropo_m):
                out[epoch_idx, sat_idx] = tropo_m
    return out


def fill_observation_matrices(
    epochs: Sequence[RawEpochObservation],
    *,
    source_columns: Iterable[str],
    baseline_lookup: Mapping[int, np.ndarray],
    weight_mode: str,
    multi_gnss: bool,
    dual_frequency: bool,
    tdcp_enabled: bool,
    adr_sign: float,
    elapsed_ns_lookup: Mapping[int, float] | None,
    hcdc_lookup: Mapping[int, float] | None,
    clock_bias_lookup: Mapping[int, float] | None,
    clock_drift_lookup: Mapping[int, float] | None,
    gps_tgd_m_by_svid: Mapping[int, float],
    gps_matrtklib_nav_messages: Mapping[int, Sequence[Any]] | None,
    gps_iono_alpha_beta: tuple[Sequence[float], Sequence[float]] | None = None,
    gps_arrival_tow_s_from_row_fn: Callable[[Any], float],
    gps_sat_clock_bias_adjustment_m_fn: Callable[[int, int, str, Mapping[int, float]], float],
    gps_matrtklib_sat_product_adjustment_fn: Callable[..., tuple[np.ndarray, np.ndarray | None, float, float | None] | None],
    clock_kind_for_observation_fn: Callable[..., int],
    is_l5_signal_fn: Callable[[str], bool],
    slot_sort_key_fn: Callable[[tuple[int, int, str]], Any],
    ecef_to_lla_fn: Callable[[float, float, float], tuple[float, float, float]],
    elevation_azimuth_fn: Callable[[np.ndarray, np.ndarray], tuple[float, float]],
    rtklib_tropo_fn: Callable[[float, float, float], float],
    matlab_signal_clock_dim: int,
) -> ObservationMatrixProducts:
    if not epochs:
        raise ValueError("at least one epoch is required")

    slot_keys = build_slot_keys(epochs, slot_sort_key_fn=slot_sort_key_fn)
    slot_index = {key: idx for idx, key in enumerate(slot_keys)}
    n_epoch = len(epochs)
    n_sat_slots = len(slot_keys)
    visible_max = max(len(epoch.group) for epoch in epochs)
    n_clock = matlab_signal_clock_dim if dual_frequency else (3 if multi_gnss else 1)

    sat_ecef = np.zeros((n_epoch, n_sat_slots, 3), dtype=np.float64)
    pseudorange = np.zeros((n_epoch, n_sat_slots), dtype=np.float64)
    pseudorange_observable = np.zeros((n_epoch, n_sat_slots), dtype=np.float64)
    weights = np.zeros((n_epoch, n_sat_slots), dtype=np.float64)
    pseudorange_bias_weights = np.zeros((n_epoch, n_sat_slots), dtype=np.float64)
    sat_clock_bias_matrix = np.full((n_epoch, n_sat_slots), np.nan, dtype=np.float64)
    rtklib_iono_m = np.full((n_epoch, n_sat_slots), np.nan, dtype=np.float64)
    rtklib_tropo_m = np.full((n_epoch, n_sat_slots), np.nan, dtype=np.float64)
    kaggle_wls = np.zeros((n_epoch, 3), dtype=np.float64)
    truth = np.zeros((n_epoch, 3), dtype=np.float64)
    times_ms = np.zeros(n_epoch, dtype=np.float64)
    elapsed_ns = np.full(n_epoch, np.nan, dtype=np.float64) if elapsed_ns_lookup is not None else None
    sys_kind = np.zeros((n_epoch, n_sat_slots), dtype=np.int32) if (multi_gnss or dual_frequency) else None
    clock_counts = np.full(n_epoch, np.nan, dtype=np.float64) if hcdc_lookup is not None else None
    clock_bias_m = np.full(n_epoch, np.nan, dtype=np.float64) if clock_bias_lookup is not None else None
    clock_drift_mps = np.full(n_epoch, np.nan, dtype=np.float64) if clock_drift_lookup is not None else None

    column_set = set(source_columns)
    has_sat_vel = {
        "SvVelocityXEcefMetersPerSecond",
        "SvVelocityYEcefMetersPerSecond",
        "SvVelocityZEcefMetersPerSecond",
    }.issubset(column_set)
    has_doppler = "PseudorangeRateMetersPerSecond" in column_set
    has_doppler_unc = "PseudorangeRateUncertaintyMetersPerSecond" in column_set
    has_sat_clock_drift = "SvClockDriftMetersPerSecond" in column_set
    sat_vel = np.zeros((n_epoch, n_sat_slots, 3), dtype=np.float64) if has_sat_vel else None
    sat_clock_drift_mps = np.zeros((n_epoch, n_sat_slots), dtype=np.float64) if has_sat_clock_drift else None
    doppler = np.zeros((n_epoch, n_sat_slots), dtype=np.float64) if has_doppler else None
    doppler_weights = np.zeros((n_epoch, n_sat_slots), dtype=np.float64) if has_doppler else None

    has_adr = tdcp_enabled and {"AccumulatedDeltaRangeMeters", "AccumulatedDeltaRangeState"}.issubset(column_set)
    adr = np.full((n_epoch, n_sat_slots), np.nan, dtype=np.float64) if has_adr else None
    adr_state = np.zeros((n_epoch, n_sat_slots), dtype=np.int32) if has_adr else None
    adr_uncertainty = (
        np.full((n_epoch, n_sat_slots), np.nan, dtype=np.float64)
        if has_adr and "AccumulatedDeltaRangeUncertaintyMeters" in column_set
        else None
    )

    gps_l1_sat_time_lookup = (
        build_gps_l1_sat_time_lookup(
            epochs,
            gps_arrival_tow_s_from_row_fn=gps_arrival_tow_s_from_row_fn,
            gps_sat_clock_bias_adjustment_m_fn=gps_sat_clock_bias_adjustment_m_fn,
            gps_tgd_m_by_svid=gps_tgd_m_by_svid,
            is_l5_signal_fn=is_l5_signal_fn,
        )
        if gps_matrtklib_nav_messages
        else {}
    )
    gps_l1_sat_time_gnss_log_lookup = (
        build_gps_l1_sat_time_lookup(
            epochs,
            gps_arrival_tow_s_from_row_fn=gps_arrival_tow_s_from_row_fn,
            gps_sat_clock_bias_adjustment_m_fn=gps_sat_clock_bias_adjustment_m_fn,
            gps_tgd_m_by_svid=gps_tgd_m_by_svid,
            is_l5_signal_fn=is_l5_signal_fn,
            include_masked_gnss_log_only=True,
        )
        if gps_matrtklib_nav_messages
        else {}
    )
    gps_l1_sat_product_lookup = build_gps_l1_sat_product_lookup(epochs, is_l5_signal_fn=is_l5_signal_fn)
    sat_velocity_forward_lookup = build_sat_velocity_forward_difference_lookup(epochs) if has_sat_vel else {}
    gps_sat_product_adjustment_cache: dict[
        tuple[int, int, str], tuple[np.ndarray, np.ndarray | None, float, float | None] | None
    ] = {}

    for epoch_idx, epoch in enumerate(epochs):
        tow_ms = epoch.time_ms
        epoch_key = int(tow_ms)
        kaggle_wls[epoch_idx] = baseline_lookup.get(epoch_key, epoch.baseline_xyz)
        truth[epoch_idx] = epoch.truth_xyz
        times_ms[epoch_idx] = tow_ms
        if elapsed_ns is not None and elapsed_ns_lookup is not None:
            elapsed_ns[epoch_idx] = elapsed_ns_lookup.get(epoch_key, np.nan)
        if clock_counts is not None and hcdc_lookup is not None:
            clock_counts[epoch_idx] = hcdc_lookup.get(epoch_key, np.nan)
        if clock_bias_m is not None and clock_bias_lookup is not None:
            clock_bias_m[epoch_idx] = clock_bias_lookup.get(epoch_key, np.nan)
        if clock_drift_mps is not None and clock_drift_lookup is not None:
            clock_drift_mps[epoch_idx] = clock_drift_lookup.get(epoch_key, np.nan)

        for row in epoch.group.itertuples(index=False):
            key = (int(row.ConstellationType), int(row.Svid), str(row.SignalType))
            sat_idx = slot_index[key]
            row_is_gps = int(row.ConstellationType) == 1
            row_is_gps_l5 = row_is_gps and is_l5_signal_fn(str(row.SignalType))
            p_ok = bool(getattr(row, "bridge_p_ok", True))
            d_ok = bool(getattr(row, "bridge_d_ok", True))
            l_ok = bool(getattr(row, "bridge_l_ok", True))
            p_bias_ok = bool(getattr(row, "bridge_p_bias_ok", p_ok))
            sat_product_adjustment = None
            raw_product_row = (
                gps_l1_sat_product_lookup.get((epoch_idx, int(row.Svid)), row)
                if row_is_gps_l5
                else row
            )
            raw_product_velocity = sat_velocity_forward_lookup.get(
                (
                    epoch_idx,
                    int(raw_product_row.ConstellationType),
                    int(raw_product_row.Svid),
                    str(raw_product_row.SignalType),
                ),
            )
            derived_common_clock_m = float(row.SvClockBiasMeters) + gps_sat_clock_bias_adjustment_m_fn(
                int(row.ConstellationType),
                int(row.Svid),
                str(row.SignalType),
                gps_tgd_m_by_svid,
            )
            if row_is_gps and gps_matrtklib_nav_messages:
                adjustment_key = (epoch_idx, int(row.Svid), str(row.SignalType))
                if adjustment_key in gps_sat_product_adjustment_cache:
                    sat_product_adjustment = gps_sat_product_adjustment_cache[adjustment_key]
                else:
                    if row_is_gps_l5:
                        adjustment_timing = gps_l1_sat_time_lookup.get((epoch_idx, int(row.Svid)))
                        if adjustment_timing is None and _row_is_gnss_log_only(row):
                            # MATLAB can use same-epoch GNSS-log L1 timing to
                            # generate GNSS-log-only L5 satellite products even
                            # when the L1 observation is inactive as a factor.
                            adjustment_timing = gps_l1_sat_time_gnss_log_lookup.get((epoch_idx, int(row.Svid)))
                        if adjustment_timing is None:
                            adjustment_timing = (
                                gps_arrival_tow_s_from_row_fn(row),
                                float(row.RawPseudorangeMeters),
                                derived_common_clock_m,
                                _gps_received_sv_tow_s_from_row(row),
                            )
                    else:
                        l1_timing = gps_l1_sat_time_lookup.get((epoch_idx, int(row.Svid)))
                        if l1_timing is None:
                            adjustment_timing = (
                                gps_arrival_tow_s_from_row_fn(row),
                                float(row.RawPseudorangeMeters),
                                derived_common_clock_m,
                                _gps_received_sv_tow_s_from_row(row),
                            )
                        else:
                            adjustment_timing = l1_timing
                    receiver_clock_bias_m = 0.0
                    if clock_bias_m is not None and np.isfinite(clock_bias_m[epoch_idx]):
                        receiver_clock_bias_m = float(clock_bias_m[epoch_idx])
                    sat_product_adjustment = gps_matrtklib_sat_product_adjustment_fn(
                        svid=int(row.Svid),
                        arrival_tow_s=float(adjustment_timing[0]),
                        l1_raw_pseudorange_m=float(adjustment_timing[1]),
                        derived_common_clock_m=float(adjustment_timing[2]),
                        nav_messages_by_svid=gps_matrtklib_nav_messages,
                        receiver_clock_bias_m=receiver_clock_bias_m,
                        received_sv_tow_s=float(adjustment_timing[3]),
                    )
                    gps_sat_product_adjustment_cache[adjustment_key] = sat_product_adjustment
            if sat_product_adjustment is not None:
                sat_xyz_adjusted, sat_vel_adjusted, sat_clock_bias_m, sat_clock_drift_adjusted = sat_product_adjustment
                l5_clock_already_matches = row_is_gps_l5 and (
                    abs(derived_common_clock_m - float(sat_clock_bias_m)) <= GPS_L5_SAT_PRODUCT_CLOCK_MATCH_THRESHOLD_M
                )
                if l5_clock_already_matches:
                    sat_clock_bias_m = derived_common_clock_m
                    if _row_is_gnss_log_only(raw_product_row):
                        sat_xyz = sat_xyz_adjusted
                    else:
                        sat_xyz = np.array(
                            [
                                float(raw_product_row.SvPositionXEcefMeters),
                                float(raw_product_row.SvPositionYEcefMeters),
                                float(raw_product_row.SvPositionZEcefMeters),
                            ],
                            dtype=np.float64,
                        )
                else:
                    sat_xyz = sat_xyz_adjusted
            else:
                sat_xyz = np.array(
                    [
                        float(raw_product_row.SvPositionXEcefMeters),
                        float(raw_product_row.SvPositionYEcefMeters),
                        float(raw_product_row.SvPositionZEcefMeters),
                    ],
                    dtype=np.float64,
                )
                sat_clock_bias_m = derived_common_clock_m
                sat_vel_adjusted = None
                sat_clock_drift_adjusted = None

            sat_ecef[epoch_idx, sat_idx] = sat_xyz
            sat_clock_bias_matrix[epoch_idx, sat_idx] = sat_clock_bias_m
            iono_m = float(row.IonosphericDelayMeters)
            recomputed_iono_m = _rtklib_iono_for_satellite(
                tow_ms,
                kaggle_wls[epoch_idx],
                sat_ecef[epoch_idx, sat_idx],
                str(row.SignalType),
                gps_iono_alpha_beta if row_is_gps else None,
                ecef_to_lla_fn=ecef_to_lla_fn,
                elevation_azimuth_fn=elevation_azimuth_fn,
            )
            if np.isfinite(recomputed_iono_m):
                iono_m = recomputed_iono_m
            rtklib_iono_m[epoch_idx, sat_idx] = iono_m
            pseudorange_observable[epoch_idx, sat_idx] = float(row.RawPseudorangeMeters)
            pseudorange[epoch_idx, sat_idx] = (
                float(row.RawPseudorangeMeters)
                + sat_clock_bias_m
                - iono_m
                - float(row.TroposphericDelayMeters)
            )

            tropo_m = _rtklib_tropo_for_satellite(
                kaggle_wls[epoch_idx],
                sat_ecef[epoch_idx, sat_idx],
                ecef_to_lla_fn=ecef_to_lla_fn,
                elevation_azimuth_fn=elevation_azimuth_fn,
                rtklib_tropo_fn=rtklib_tropo_fn,
            )
            if np.isfinite(tropo_m):
                rtklib_tropo_m[epoch_idx, sat_idx] = tropo_m

            if p_ok:
                if weight_mode == "sin2el":
                    sin_el = max(np.sin(np.deg2rad(float(row.SvElevationDegrees))), 0.1)
                    weights[epoch_idx, sat_idx] = sin_el * sin_el
                else:
                    weights[epoch_idx, sat_idx] = max(float(row.Cn0DbHz), 1.0)
            if p_bias_ok:
                pseudorange_bias_weights[epoch_idx, sat_idx] = 1.0

            if sys_kind is not None:
                sys_kind[epoch_idx, sat_idx] = clock_kind_for_observation_fn(
                    int(row.ConstellationType),
                    str(row.SignalType),
                    dual_frequency=dual_frequency,
                    multi_gnss=multi_gnss,
                )

            if sat_vel is not None:
                if sat_vel_adjusted is not None:
                    sat_vel[epoch_idx, sat_idx] = sat_vel_adjusted
                elif raw_product_velocity is not None:
                    sat_vel[epoch_idx, sat_idx] = raw_product_velocity
                else:
                    sat_vel[epoch_idx, sat_idx] = [
                        float(raw_product_row.SvVelocityXEcefMetersPerSecond),
                        float(raw_product_row.SvVelocityYEcefMetersPerSecond),
                        float(raw_product_row.SvVelocityZEcefMetersPerSecond),
                    ]

            if sat_clock_drift_mps is not None:
                sat_clock_drift = (
                    float(sat_clock_drift_adjusted)
                    if sat_clock_drift_adjusted is not None
                    else float(raw_product_row.SvClockDriftMetersPerSecond)
                )
                if np.isfinite(sat_clock_drift):
                    sat_clock_drift_mps[epoch_idx, sat_idx] = sat_clock_drift

            if doppler is not None and doppler_weights is not None:
                doppler_value = -float(row.PseudorangeRateMetersPerSecond)
                if np.isfinite(doppler_value):
                    doppler[epoch_idx, sat_idx] = doppler_value
                    sigma = 1.0
                    if has_doppler_unc:
                        sigma_unc = float(row.PseudorangeRateUncertaintyMetersPerSecond)
                        if np.isfinite(sigma_unc) and sigma_unc > 0.0:
                            sigma = sigma_unc
                    if d_ok:
                        doppler_weights[epoch_idx, sat_idx] = 1.0 / (max(sigma, 0.05) ** 2)

            if adr is not None and adr_state is not None:
                adr_value = adr_sign * float(row.AccumulatedDeltaRangeMeters)
                if l_ok and np.isfinite(adr_value) and adr_value != 0.0:
                    adr[epoch_idx, sat_idx] = adr_value
                    adr_state[epoch_idx, sat_idx] = int(row.AccumulatedDeltaRangeState)
                if l_ok and adr_uncertainty is not None:
                    adr_unc = float(row.AccumulatedDeltaRangeUncertaintyMeters)
                    if np.isfinite(adr_unc) and adr_unc > 0.0:
                        adr_uncertainty[epoch_idx, sat_idx] = adr_unc

    return ObservationMatrixProducts(
        times_ms=times_ms,
        sat_ecef=sat_ecef,
        pseudorange=pseudorange,
        pseudorange_observable=pseudorange_observable,
        weights=weights,
        pseudorange_bias_weights=pseudorange_bias_weights,
        sat_clock_bias_matrix=sat_clock_bias_matrix,
        rtklib_iono_m=rtklib_iono_m,
        rtklib_tropo_m=rtklib_tropo_m,
        kaggle_wls=kaggle_wls,
        truth=truth,
        visible_max=visible_max,
        slot_keys=slot_keys,
        n_clock=n_clock,
        elapsed_ns=elapsed_ns,
        sys_kind=sys_kind,
        clock_counts=clock_counts,
        clock_bias_m=clock_bias_m,
        clock_drift_mps=clock_drift_mps,
        sat_vel=sat_vel,
        sat_clock_drift_mps=sat_clock_drift_mps,
        doppler=doppler,
        doppler_weights=doppler_weights,
        adr=adr,
        adr_state=adr_state,
        adr_uncertainty=adr_uncertainty,
    )


def clock_jump_from_epoch_counts(epoch_counts: np.ndarray) -> np.ndarray:
    counts = np.asarray(epoch_counts, dtype=np.float64).copy()
    jumps = np.zeros(counts.size, dtype=bool)
    if counts.size <= 1:
        return jumps
    if np.isfinite(counts).any():
        counts = pd.Series(counts).ffill().bfill().to_numpy(dtype=np.float64)
    jumps[1:] = np.isfinite(counts[1:]) & np.isfinite(counts[:-1]) & (counts[1:] != counts[:-1])
    return jumps


def receiver_clock_bias_from_nanos(full_bias_nanos: float, bias_nanos: float) -> float:
    return -((float(full_bias_nanos) + float(bias_nanos)) * 1e-9 * LIGHT_SPEED_MPS)


def receiver_clock_bias_lookup_from_epoch_meta(epoch_meta: pd.DataFrame) -> dict[int, float]:
    if "FullBiasNanos" not in epoch_meta.columns:
        return {}
    out: dict[int, float] = {}
    base_full_bias: int | None = None
    base_hcdc = None
    prev_time_nanos: float | None = None
    prev_utc_ms: float | None = None
    meta = epoch_meta.sort_values("utcTimeMillis")
    for row in meta.itertuples(index=False):
        full_bias_value = getattr(row, "FullBiasNanos", np.nan)
        if pd.isna(full_bias_value):
            continue
        full_bias = int(full_bias_value)
        hcdc = getattr(row, "HardwareClockDiscontinuityCount", None)
        hcdc_key = None if pd.isna(hcdc) else int(hcdc)
        time_nanos = getattr(row, "TimeNanos", np.nan)
        time_jump = (
            prev_time_nanos is not None
            and pd.notna(time_nanos)
            and abs(float(time_nanos) - float(prev_time_nanos)) > 1.0e9
        )
        utc_ms = getattr(row, "utcTimeMillis", np.nan)
        utc_jump = (
            prev_time_nanos is None
            and prev_utc_ms is not None
            and pd.notna(utc_ms)
            and abs(float(utc_ms) - float(prev_utc_ms)) > 1000.0
        )
        if base_full_bias is None or (hcdc_key is not None and hcdc_key != base_hcdc) or time_jump or utc_jump:
            base_full_bias = full_bias
            base_hcdc = hcdc_key
        out[int(row.utcTimeMillis)] = float(full_bias - base_full_bias) * 1e-9 * LIGHT_SPEED_MPS
        if pd.notna(time_nanos):
            prev_time_nanos = float(time_nanos)
        if pd.notna(utc_ms):
            prev_utc_ms = float(utc_ms)
    return out


__all__ = [
    "ANDROID_STATE_CODE_LOCK",
    "ANDROID_STATE_TOD_OK",
    "ANDROID_STATE_TOW_OK",
    "BASELINE_BIAS_UNCERTAINTY_NANOS_MAX",
    "BASELINE_OUTLIER_FLOOR_M",
    "BASELINE_OUTLIER_THRESHOLD_FACTOR",
    "BASELINE_OUTLIER_WINDOW",
    "CONSTELLATION_GLONASS",
    "EARTH_ROTATION_RATE_RAD_S",
    "GPS_EPOCH_UNIX_SECONDS",
    "GPS_WEEK_SECONDS",
    "LIGHT_SPEED_MPS",
    "OBS_MASK_DOPPLER_RESIDUAL_THRESHOLD_MPS",
    "OBS_MASK_MIN_CN0_DBHZ",
    "OBS_MASK_MIN_ELEVATION_DEG",
    "OBS_MASK_PSEUDORANGE_DOPPLER_THRESHOLD_L5_M",
    "OBS_MASK_PSEUDORANGE_DOPPLER_THRESHOLD_M",
    "OBS_MASK_PSEUDORANGE_MAX_M",
    "OBS_MASK_PSEUDORANGE_MIN_M",
    "OBS_MASK_RESIDUAL_THRESHOLD_L5_M",
    "OBS_MASK_RESIDUAL_THRESHOLD_M",
    "RAW_GNSS_COLUMNS",
    "RAW_GNSS_OPTIONAL_COLUMNS",
    "RAW_GNSS_REQUIRED_COLUMNS",
    "RTKLIB_SAT_VELOCITY_FORWARD_DIFF_HALF_STEP_S",
    "GPS_LEAP_SECONDS",
    "GPS_L1_FREQUENCY_HZ",
    "GPS_L5_FREQUENCY_HZ",
    "ObservationMatrixProducts",
    "RawEpochObservation",
    "TripArrays",
    "android_state_tracking_ok",
    "apply_matlab_signal_observation_mask",
    "build_gps_l1_sat_product_lookup",
    "build_gps_l1_sat_time_lookup",
    "build_sat_velocity_forward_difference_lookup",
    "build_epoch_metadata_frame",
    "build_slot_keys",
    "clock_jump_from_epoch_counts",
    "fill_observation_matrices",
    "interpolate_series",
    "legacy_matlab_signal_observation_mask",
    "load_raw_gnss_frame",
    "load_raw_gnss_frame_epoch_window",
    "matlab_signal_observation_masks",
    "read_raw_gnss_csv",
    "receiver_clock_bias_from_nanos",
    "receiver_clock_bias_lookup_from_epoch_meta",
    "recompute_rtklib_iono_matrix",
    "recompute_rtklib_tropo_matrix",
    "rtklib_klobuchar_iono_delay_m",
    "signal_type_iono_scale",
    "repair_baseline_wls",
    "select_epoch_observations",
]
