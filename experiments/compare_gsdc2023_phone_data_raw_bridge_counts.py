#!/usr/bin/env python3
"""Compare MATLAB phone-data factor counts against raw-bridge counts.

This utility is intentionally narrow: it scans local trips, extracts the finite
observation counts stored in ``phone_data.mat`` for ``P/D/L/resPc/resD/resL``
on ``L1`` and ``L5``, then compares them against counts produced by the raw
bridge path that mirrors ``build_trip_arrays(... apply_observation_mask=True,
dual_frequency=True, use_tdcp=True)``.

The MATLAB ``phone_data_factor_counts.csv`` exports currently used by this
audit are GPS L1/L5 counts.  The default comparison scope is therefore GPS-only;
use ``--multi-gnss`` to include Galileo/QZSS bridge observations for a separate
coverage audit.

Trips without ``phone_data.mat`` are still included. Their phone-side counts are
left empty and the raw-bridge counts are still reported when ``device_gnss.csv``
is available.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from experiments.audit_gsdc2023_preprocessing_gap import TripSpec, discover_trip_specs
from experiments.gsdc2023_audit_output import (
    print_summary_and_output_dir as _print_summary_and_output_dir,
    timestamped_output_dir as _timestamped_output_dir,
    write_summary_json as _write_summary_json,
)
from experiments.gsdc2023_audit_cli import (
    add_data_root_arg as _add_data_root_arg,
    add_max_epochs_arg as _add_max_epochs_arg,
    add_multi_gnss_arg as _add_multi_gnss_arg,
    add_output_dir_arg as _add_output_dir_arg,
    nonnegative_max_epochs as _nonnegative_max_epochs,
    resolved_output_root as _resolved_output_root,
)
from experiments.gsdc2023_observation_matrix import (
    OBS_MASK_MIN_CN0_DBHZ,
    OBS_MASK_MIN_ELEVATION_DEG,
    OBS_MASK_RESIDUAL_THRESHOLD_L5_M,
    OBS_MASK_RESIDUAL_THRESHOLD_M,
    apply_matlab_signal_observation_mask as _apply_matlab_signal_observation_mask,
    load_raw_gnss_frame as _load_raw_gnss_frame,
)
from experiments.gsdc2023_raw_bridge import (
    DEFAULT_ROOT,
    DEFAULT_TDCP_CONSISTENCY_THRESHOLD_M,
    _build_trip_arrays,
)
from experiments.gsdc2023_signal_model import (
    factor_frequency_label as _freq_label,
    multi_gnss_mask as _multi_gnss_mask,
)
from experiments.validate_gsdc2023_phone_data import _load_mat, _observation_counts


_FIELDS = ("P", "D", "L", "resPc", "resD", "resL")
_FREQS = ("L1", "L5")
_FIELD_TO_BRIDGE_SOURCE = {
    "P": "pseudorange",
    "resPc": "pseudorange",
    "D": "doppler",
    "resD": "doppler",
    "L": "tdcp",
    "resL": "tdcp",
}


@dataclass(frozen=True)
class TripComparisonSummary:
    trip: str
    dataset_split: str
    course: str
    phone: str
    phone_data_present: bool
    raw_device_gnss_present: bool
    bridge_available: bool
    phone_error: str | None
    bridge_error: str | None
    bridge_total_p_count: int | None
    bridge_total_d_count: int | None
    bridge_total_l_count: int | None
    phone_total_p_count: int | None
    phone_total_d_count: int | None
    phone_total_l_count: int | None


def _phone_counts(trip_dir: Path) -> dict[str, dict[str, int]] | None:
    phone_data_path = trip_dir / "phone_data.mat"
    if not phone_data_path.is_file():
        return None
    exported_factor_counts_path = trip_dir / "phone_data_factor_counts.csv"
    if exported_factor_counts_path.is_file():
        return _phone_counts_from_csv(exported_factor_counts_path)
    exported_counts_path = trip_dir / "phone_data_observation_counts.csv"
    if exported_counts_path.is_file():
        return _phone_counts_from_csv(exported_counts_path)
    phone_data = _load_mat(phone_data_path)
    obs = phone_data.get("obs")
    if obs is None:
        return {}
    return _observation_counts(obs)


def _phone_counts_from_csv(path: Path) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {freq: {} for freq in _FREQS}
    frame = pd.read_csv(path)
    required = {"freq", "field", "count"}
    if not required.issubset(frame.columns):
        raise ValueError(f"phone count CSV must contain columns {sorted(required)}: {path}")
    for row in frame.itertuples(index=False):
        freq = str(getattr(row, "freq", "")).strip()
        field = str(getattr(row, "field", "")).strip()
        if freq not in _FREQS or field not in _FIELDS:
            continue
        count = getattr(row, "count")
        if pd.isna(count):
            continue
        counts[freq][field] = int(count)
    return {freq: freq_counts for freq, freq_counts in counts.items() if freq_counts}


def _total_phone_count(phone_counts: dict[str, dict[str, int]] | None, field: str) -> int | None:
    if phone_counts is None:
        return None
    values = [phone_counts.get(freq, {}).get(field) for freq in _FREQS]
    values = [int(value) for value in values if value is not None]
    if not values:
        return None
    return int(sum(values))


def _selected_bridge_frame(trip_dir: Path, batch: Any, *, multi_gnss: bool) -> pd.DataFrame:
    raw_path = trip_dir / "device_gnss.csv"
    raw_df = _load_raw_gnss_frame(raw_path)
    selected_times = {int(round(float(value))) for value in np.asarray(batch.times_ms, dtype=np.float64)}
    if selected_times:
        raw_df = raw_df[raw_df["utcTimeMillis"].astype(np.int64).isin(selected_times)]

    if multi_gnss:
        raw_df = raw_df[_multi_gnss_mask(raw_df, dual_frequency=True)]
    else:
        raw_df = raw_df[
            (raw_df["ConstellationType"] == 1)
            & raw_df["SignalType"].isin(["GPS_L1_CA", "GPS_L5_Q"])
        ]
    raw_df = raw_df[
        np.isfinite(raw_df["RawPseudorangeMeters"])
        & np.isfinite(raw_df["SvPositionXEcefMeters"])
        & np.isfinite(raw_df["SvPositionYEcefMeters"])
        & np.isfinite(raw_df["SvPositionZEcefMeters"])
        & np.isfinite(raw_df["SvElevationDegrees"])
        & np.isfinite(raw_df["SvClockBiasMeters"])
        & np.isfinite(raw_df["IonosphericDelayMeters"])
        & np.isfinite(raw_df["TroposphericDelayMeters"])
    ]
    raw_df, _ = _apply_matlab_signal_observation_mask(
        raw_df,
        min_cn0_dbhz=OBS_MASK_MIN_CN0_DBHZ,
        min_elevation_deg=OBS_MASK_MIN_ELEVATION_DEG,
    )
    if raw_df.empty:
        return raw_df
    return raw_df.sort_values(["utcTimeMillis", "ConstellationType", "Svid", "Cn0DbHz"]).groupby(
        ["utcTimeMillis", "ConstellationType", "Svid", "SignalType"],
        as_index=False,
    ).tail(1)


def _clock_jump_from_batch(batch: Any) -> np.ndarray | None:
    clock_jump = getattr(batch, "clock_jump", None)
    if clock_jump is None:
        return None
    arr = np.asarray(clock_jump, dtype=bool).reshape(-1)
    if arr.size == 0:
        return None
    return arr


def _bridge_counts_by_freq(batch: Any, selected_rows: pd.DataFrame) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {freq: {field: 0 for field in _FIELDS} for freq in _FREQS}
    slot_keys = getattr(batch, "slot_keys", None)
    if slot_keys:
        slot_freq = np.array([_freq_label(str(key[2])) for key in slot_keys], dtype=object)
        for freq in _FREQS:
            idx = np.flatnonzero(slot_freq == freq)
            if idx.size == 0:
                continue
            counts[freq]["P"] = int(np.count_nonzero(batch.weights[:, idx] > 0.0))
            counts[freq]["resPc"] = counts[freq]["P"]
            if batch.doppler_weights is not None:
                counts[freq]["D"] = int(np.count_nonzero(batch.doppler_weights[:, idx] > 0.0))
                counts[freq]["resD"] = counts[freq]["D"]
            if batch.tdcp_weights is not None:
                counts[freq]["L"] = int(np.count_nonzero(batch.tdcp_weights[:, idx] > 0.0))
                counts[freq]["resL"] = counts[freq]["L"]
        return counts

    if selected_rows.empty:
        return counts

    selected_rows = selected_rows.copy()
    selected_rows["freq"] = selected_rows["SignalType"].map(_freq_label)
    selected_rows = selected_rows[selected_rows["freq"].isin(_FREQS)]

    for freq, group in selected_rows.groupby("freq"):
        freq_counts = counts[freq]
        freq_counts["P"] = int(len(group))
        freq_counts["resPc"] = int(len(group))
        if "PseudorangeRateMetersPerSecond" in group.columns:
            doppler_valid = np.isfinite(group["PseudorangeRateMetersPerSecond"].to_numpy(dtype=np.float64))
            freq_counts["D"] = int(np.count_nonzero(doppler_valid))
            freq_counts["resD"] = int(np.count_nonzero(doppler_valid))

    if batch.tdcp_weights is None or len(batch.times_ms) <= 1:
        return counts

    key_cols = ["ConstellationType", "Svid", "SignalType"]
    epoch_groups = {
        int(round(float(epoch_time))): epoch_group.copy()
        for epoch_time, epoch_group in selected_rows.groupby("utcTimeMillis", sort=True)
    }
    times_ms = np.asarray(batch.times_ms, dtype=np.float64)
    clock_jump = _clock_jump_from_batch(batch)

    for epoch_idx in range(len(times_ms) - 1):
        dt_s = float(times_ms[epoch_idx + 1] - times_ms[epoch_idx]) * 1e-3
        if not np.isfinite(dt_s) or dt_s <= 0.0 or dt_s > 30.0:
            continue
        if clock_jump is not None and epoch_idx + 1 < clock_jump.size and bool(clock_jump[epoch_idx + 1]):
            continue
        left = epoch_groups.get(int(round(float(times_ms[epoch_idx]))))
        right = epoch_groups.get(int(round(float(times_ms[epoch_idx + 1]))))
        if left is None or right is None:
            continue
        left = left.set_index(key_cols)
        right = right.set_index(key_cols)
        common = left.index.intersection(right.index)
        if common.empty:
            continue
        for key in common:
            row0 = left.loc[key]
            row1 = right.loc[key]
            signal_type = str(key[2])
            freq = _freq_label(signal_type)
            if freq not in _FREQS:
                continue
            adr0 = float(row0.get("AccumulatedDeltaRangeMeters", np.nan))
            adr1 = float(row1.get("AccumulatedDeltaRangeMeters", np.nan))
            adr_state0 = int(row0.get("AccumulatedDeltaRangeState", 0))
            adr_state1 = int(row1.get("AccumulatedDeltaRangeState", 0))
            dop0 = float(row0.get("PseudorangeRateMetersPerSecond", np.nan))
            dop1 = float(row1.get("PseudorangeRateMetersPerSecond", np.nan))
            if not (
                np.isfinite(adr0)
                and np.isfinite(adr1)
                and np.isfinite(dop0)
                and np.isfinite(dop1)
                and bool(adr_state0 & 1)
                and bool(adr_state1 & 1)
                and not bool(adr_state0 & (2 | 4))
                and not bool(adr_state1 & (2 | 4))
            ):
                continue
            meas = float(adr1 - adr0)
            doppler_tdcp = 0.5 * (-dop0 - dop1) * dt_s
            if abs(meas - doppler_tdcp) > 1.5:
                continue
            counts[freq]["L"] += 1
            counts[freq]["resL"] += 1

    return counts


def _comparison_rows(
    trip: TripSpec,
    phone_counts: dict[str, dict[str, int]] | None,
    bridge_counts: dict[str, dict[str, int]] | None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for freq in _FREQS:
        for field in _FIELDS:
            phone_count = None if phone_counts is None else phone_counts.get(freq, {}).get(field)
            bridge_count = None if bridge_counts is None else bridge_counts.get(freq, {}).get(field)
            rows.append(
                {
                    "trip": trip.trip,
                    "dataset_split": trip.split,
                    "course": trip.course,
                    "phone": trip.phone,
                    "freq": freq,
                    "field": field,
                    "bridge_source": _FIELD_TO_BRIDGE_SOURCE[field],
                    "phone_data_present": phone_counts is not None,
                    "phone_count": None if phone_count is None else int(phone_count),
                    "bridge_count": None if bridge_count is None else int(bridge_count),
                    "count_delta": (
                        None
                        if phone_count is None or bridge_count is None
                        else int(bridge_count) - int(phone_count)
                    ),
                },
            )
    return rows


def _trip_summary(
    trip: TripSpec,
    phone_counts: dict[str, dict[str, int]] | None,
    bridge_counts: dict[str, dict[str, int]] | None,
    *,
    phone_data_present: bool,
    raw_device_gnss_present: bool,
    bridge_available: bool,
    phone_error: str | None,
    bridge_error: str | None,
) -> TripComparisonSummary:
    bridge_total_p = None if bridge_counts is None else int(sum(bridge_counts[freq]["P"] for freq in _FREQS))
    bridge_total_d = None if bridge_counts is None else int(sum(bridge_counts[freq]["D"] for freq in _FREQS))
    bridge_total_l = None if bridge_counts is None else int(sum(bridge_counts[freq]["L"] for freq in _FREQS))
    return TripComparisonSummary(
        trip=trip.trip,
        dataset_split=trip.split,
        course=trip.course,
        phone=trip.phone,
        phone_data_present=phone_data_present,
        raw_device_gnss_present=raw_device_gnss_present,
        bridge_available=bridge_available,
        phone_error=phone_error,
        bridge_error=bridge_error,
        bridge_total_p_count=bridge_total_p,
        bridge_total_d_count=bridge_total_d,
        bridge_total_l_count=bridge_total_l,
        phone_total_p_count=_total_phone_count(phone_counts, "P"),
        phone_total_d_count=_total_phone_count(phone_counts, "D"),
        phone_total_l_count=_total_phone_count(phone_counts, "L"),
    )


def _filter_trip_specs(specs: list[TripSpec], trips: Iterable[str] | None) -> list[TripSpec]:
    selectors = {str(trip).strip().strip("/") for trip in trips or []}
    selectors = {selector for selector in selectors if selector}
    if not selectors:
        return specs

    filtered: list[TripSpec] = []
    for spec in specs:
        candidates = {
            spec.trip,
            f"{spec.course}/{spec.phone}",
        }
        if candidates & selectors:
            filtered.append(spec)
    return filtered


def _matlab_epoch_window(spec: TripSpec, requested_max_epochs: int) -> tuple[int, int]:
    start_epoch = 0
    setting_max_epochs: int | None = None
    if spec.idx_start is not None and int(spec.idx_start) > 0:
        start_epoch = max(int(spec.idx_start) - 1, 0)
    if spec.idx_start is not None and spec.idx_end is not None and int(spec.idx_end) >= int(spec.idx_start):
        setting_max_epochs = int(spec.idx_end) - int(spec.idx_start) + 1

    if requested_max_epochs > 0 and setting_max_epochs is not None:
        max_epochs = min(int(requested_max_epochs), setting_max_epochs)
    elif requested_max_epochs > 0:
        max_epochs = int(requested_max_epochs)
    elif setting_max_epochs is not None:
        max_epochs = setting_max_epochs
    else:
        max_epochs = 1_000_000_000
    return start_epoch, max_epochs


def build_comparison_frames(
    data_root: Path,
    datasets: Iterable[str],
    *,
    trips: Iterable[str] | None = None,
    offset: int = 0,
    limit: int = 0,
    max_epochs: int = 0,
    multi_gnss: bool = False,
    pseudorange_residual_mask_m: float = OBS_MASK_RESIDUAL_THRESHOLD_M,
    pseudorange_residual_mask_l5_m: float = OBS_MASK_RESIDUAL_THRESHOLD_L5_M,
    doppler_residual_mask_mps: float = 3.0,
    tdcp_consistency_threshold_m: float = DEFAULT_TDCP_CONSISTENCY_THRESHOLD_M,
    pseudorange_doppler_mask_m: float = 40.0,
    matlab_residual_diagnostics_mask_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    datasets = list(datasets)
    specs = _filter_trip_specs(discover_trip_specs(data_root, datasets), trips)
    if offset > 0:
        specs = specs[int(offset):]
    if limit > 0:
        specs = specs[:limit]

    comparison_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    phone_data_trips = 0
    raw_device_gnss_trips = 0
    bridge_errors = 0
    phone_errors = 0
    matched_rows = 0
    total_rows = 0
    matched_phone_count_total = 0
    matched_bridge_count_total = 0
    matched_abs_delta_total = 0
    matched_signed_delta_total = 0
    delta_sums: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for spec in specs:
        trip_dir = spec.trip_dir
        phone_error = None
        phone_counts = None
        phone_data_present = (trip_dir / "phone_data.mat").is_file()
        raw_device_gnss_present = (trip_dir / "device_gnss.csv").is_file()

        if phone_data_present:
            try:
                phone_counts = _phone_counts(trip_dir)
                phone_data_trips += 1
            except Exception as exc:  # noqa: BLE001
                phone_error = str(exc)
                phone_counts = None
                phone_data_trips += 1
                phone_errors += 1

        bridge_counts = None
        bridge_error = None
        bridge_available = False
        if raw_device_gnss_present:
            try:
                start_epoch, bridge_max_epochs = _matlab_epoch_window(spec, max_epochs)
                batch = _build_trip_arrays(
                    trip_dir,
                    max_epochs=bridge_max_epochs,
                    start_epoch=start_epoch,
                    constellation_type=1,
                    signal_type="GPS_L1_CA",
                    weight_mode="sin2el",
                    multi_gnss=multi_gnss,
                    use_tdcp=True,
                    apply_observation_mask=True,
                    pseudorange_residual_mask_m=pseudorange_residual_mask_m,
                    pseudorange_residual_mask_l5_m=pseudorange_residual_mask_l5_m,
                    doppler_residual_mask_mps=doppler_residual_mask_mps,
                    tdcp_consistency_threshold_m=tdcp_consistency_threshold_m,
                    pseudorange_doppler_mask_m=pseudorange_doppler_mask_m,
                    matlab_residual_diagnostics_mask_path=matlab_residual_diagnostics_mask_path,
                    dual_frequency=True,
                )
                raw_device_gnss_trips += 1
                bridge_available = True
                selected_rows = _selected_bridge_frame(trip_dir, batch, multi_gnss=multi_gnss)
                bridge_counts = _bridge_counts_by_freq(batch, selected_rows)
                total_bridge_p = int(np.count_nonzero(batch.weights > 0.0))
                total_bridge_d = int(np.count_nonzero(batch.doppler_weights > 0.0)) if batch.doppler_weights is not None else 0
                total_bridge_l = int(np.count_nonzero(batch.tdcp_weights > 0.0)) if batch.tdcp_weights is not None else 0
                if bridge_counts is not None:
                    bridge_sum_p = sum(bridge_counts[freq]["P"] for freq in _FREQS)
                    bridge_sum_d = sum(bridge_counts[freq]["D"] for freq in _FREQS)
                    bridge_sum_l = sum(bridge_counts[freq]["L"] for freq in _FREQS)
                    if bridge_sum_p != total_bridge_p:
                        bridge_error = f"bridge P total mismatch: batch={total_bridge_p} freq_sum={bridge_sum_p}"
                    elif bridge_sum_d != total_bridge_d:
                        bridge_error = f"bridge D total mismatch: batch={total_bridge_d} freq_sum={bridge_sum_d}"
                    elif bridge_sum_l != total_bridge_l:
                        bridge_error = f"bridge L total mismatch: batch={total_bridge_l} freq_sum={bridge_sum_l}"
                    if bridge_error is not None:
                        bridge_errors += 1
            except Exception as exc:  # noqa: BLE001
                bridge_error = str(exc)
                bridge_counts = None
                bridge_errors += 1

        comparison_rows.extend(_comparison_rows(spec, phone_counts, bridge_counts))
        summary = _trip_summary(
            spec,
            phone_counts,
            bridge_counts,
            phone_data_present=phone_data_present,
            raw_device_gnss_present=raw_device_gnss_present,
            bridge_available=bridge_available,
            phone_error=phone_error,
            bridge_error=bridge_error,
        )
        summary_rows.append(summary.__dict__)

        for row in comparison_rows[-len(_FIELDS) * len(_FREQS):]:
            total_rows += 1
            if row["phone_count"] is not None and row["bridge_count"] is not None:
                matched_rows += 1
                phone_count = int(row["phone_count"])
                bridge_count = int(row["bridge_count"])
                count_delta = int(row["count_delta"])
                matched_phone_count_total += phone_count
                matched_bridge_count_total += bridge_count
                matched_abs_delta_total += abs(count_delta)
                matched_signed_delta_total += count_delta
                delta_sums[str(row["field"])][str(row["freq"])] += count_delta

    summary_payload: dict[str, object] = {
        "data_root": str(data_root),
        "datasets": list(datasets),
        "trip_filters": sorted(str(trip).strip().strip("/") for trip in trips or [] if str(trip).strip()),
        "offset": int(max(offset, 0)),
        "multi_gnss": bool(multi_gnss),
        "matlab_count_scope": "gps_l1_l5",
        "bridge_count_scope": "multi_gnss_l1_l5" if multi_gnss else "gps_l1_l5",
        "pseudorange_residual_mask_m": float(pseudorange_residual_mask_m),
        "pseudorange_residual_mask_l5_m": float(pseudorange_residual_mask_l5_m),
        "doppler_residual_mask_mps": float(doppler_residual_mask_mps),
        "tdcp_consistency_threshold_m": float(tdcp_consistency_threshold_m),
        "pseudorange_doppler_mask_m": float(pseudorange_doppler_mask_m),
        "matlab_residual_diagnostics_mask_path": (
            str(matlab_residual_diagnostics_mask_path)
            if matlab_residual_diagnostics_mask_path is not None
            else None
        ),
        "trip_count": int(len(specs)),
        "trips_with_phone_data": int(phone_data_trips),
        "trips_with_device_gnss": int(raw_device_gnss_trips),
        "bridge_errors": int(bridge_errors),
        "phone_errors": int(phone_errors),
        "comparison_rows": int(total_rows),
        "matched_rows": int(matched_rows),
        "matched_ratio": float(matched_rows / total_rows) if total_rows else None,
        "matched_phone_count_total": int(matched_phone_count_total),
        "matched_bridge_count_total": int(matched_bridge_count_total),
        "matched_abs_delta_total": int(matched_abs_delta_total),
        "matched_signed_delta_total": int(matched_signed_delta_total),
        "count_parity_ratio": (
            float(max(0.0, 1.0 - (matched_abs_delta_total / matched_phone_count_total)))
            if matched_phone_count_total > 0
            else None
        ),
        "delta_sums": {
            field: {freq: int(delta_sums[field].get(freq, 0)) for freq in _FREQS}
            for field in _FIELDS
        },
    }
    return pd.DataFrame(comparison_rows), pd.DataFrame(summary_rows), summary_payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    _add_data_root_arg(parser, default_root=DEFAULT_ROOT)
    parser.add_argument("--datasets", nargs="*", default=["train", "test"])
    parser.add_argument(
        "--trip",
        action="append",
        default=[],
        help="only compare this trip; accepts split/course/phone or course/phone and can be repeated",
    )
    _add_multi_gnss_arg(
        parser,
        default=False,
        help_text="include Galileo/QZSS bridge observations; default GPS-only scope matches MATLAB factor-count exports",
    )
    parser.add_argument("--limit", type=int, default=0, help="limit the number of discovered trips; 0 means no limit")
    parser.add_argument("--offset", type=int, default=0, help="skip this many discovered trips before applying --limit")
    _add_max_epochs_arg(parser, help_text="epoch limit passed to the raw bridge; 0 means all")
    parser.add_argument("--pseudorange-residual-mask-m", type=float, default=OBS_MASK_RESIDUAL_THRESHOLD_M)
    parser.add_argument("--pseudorange-residual-mask-l5-m", type=float, default=OBS_MASK_RESIDUAL_THRESHOLD_L5_M)
    parser.add_argument("--doppler-residual-mask-mps", type=float, default=3.0)
    parser.add_argument("--tdcp-consistency-threshold-m", type=float, default=DEFAULT_TDCP_CONSISTENCY_THRESHOLD_M)
    parser.add_argument("--pseudorange-doppler-mask-m", type=float, default=40.0)
    parser.add_argument(
        "--matlab-residual-diagnostics-mask",
        type=Path,
        default=None,
        help="optional phone_data_residual_diagnostics.csv used to force bridge P/D/L factor availability",
    )
    _add_output_dir_arg(parser)
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    out_dir = _timestamped_output_dir(
        _resolved_output_root(args),
        "gsdc2023_phone_data_raw_bridge_count_parity",
    )

    comparison_df, summary_df, summary = build_comparison_frames(
        data_root,
        args.datasets,
        trips=args.trip,
        offset=max(args.offset, 0),
        limit=max(args.limit, 0),
        max_epochs=_nonnegative_max_epochs(args),
        multi_gnss=args.multi_gnss,
        pseudorange_residual_mask_m=args.pseudorange_residual_mask_m,
        pseudorange_residual_mask_l5_m=args.pseudorange_residual_mask_l5_m,
        doppler_residual_mask_mps=args.doppler_residual_mask_mps,
        tdcp_consistency_threshold_m=args.tdcp_consistency_threshold_m,
        pseudorange_doppler_mask_m=args.pseudorange_doppler_mask_m,
        matlab_residual_diagnostics_mask_path=args.matlab_residual_diagnostics_mask,
    )
    comparison_df.to_csv(out_dir / "count_comparison.csv", index=False)
    summary_df.to_csv(out_dir / "trip_summary.csv", index=False)
    _write_summary_json(out_dir, summary)
    _print_summary_and_output_dir(summary, out_dir)


if __name__ == "__main__":
    main()
