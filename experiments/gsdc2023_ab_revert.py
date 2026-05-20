"""Row-level revert simulation for the GSDC2023 Kaggle bridge A/B.

Given the per-row haversine delta between a base submission (``bridge``) and a
candidate submission (``taroz``) plus the per-trip ``Disposition`` produced by
:mod:`gsdc2023_ab_gates`, this module computes the row-level delta that would
remain if every ``Disposition.REVERTED`` trip were rolled back to the base
submission's coordinates.

The simulation is purely tabular - it never re-reads the original submission
CSVs.  All computation lives in pure functions over a pandas ``DataFrame`` so
it can be tested with hand-built fixtures.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from experiments.gsdc2023_ab_gates import Disposition, TripDisposition


REQUIRED_ROW_DELTA_COLUMNS: tuple[str, ...] = ("tripId", "delta_m")


def _validate_row_delta_columns(row_delta: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_ROW_DELTA_COLUMNS if col not in row_delta.columns]
    if missing:
        raise ValueError(f"row_delta is missing columns: {missing}")


def dispositions_to_keep_map(
    dispositions: Iterable[TripDisposition],
) -> dict[str, bool]:
    """Map ``trip_id`` -> ``True`` when row deltas should be kept (not reverted).

    A trip is kept whenever its disposition is *not* ``REVERTED``: ``KEPT``
    trips retain their candidate-vs-base delta, and ``PASSTHROUGH`` trips do
    too because the gate has no opinion about them.
    """

    return {d.trip_id: d.disposition is not Disposition.REVERTED for d in dispositions}


def simulate_revert(
    row_delta: pd.DataFrame,
    dispositions: Iterable[TripDisposition],
    *,
    sim_column: str = "sim_delta_m",
) -> pd.DataFrame:
    """Return ``row_delta`` with an extra column zeroed for reverted trips.

    Rows whose ``tripId`` is missing from the dispositions are *kept* (no
    revert), matching the gate-side ``PASSTHROUGH`` semantics.  The returned
    DataFrame is a copy and never mutates the input.
    """

    _validate_row_delta_columns(row_delta)
    keep_map = dispositions_to_keep_map(dispositions)
    out = row_delta.copy()
    # ``map`` returns ``NaN`` for trips absent from ``keep_map``; treat those as
    # passthrough (keep).  Convert through ``True``/``False`` explicitly so we
    # do not rely on pandas' deprecated NA-to-bool downcast behavior.
    keep_series = out["tripId"].map(keep_map)
    keep = keep_series.where(keep_series.notna(), True).astype(bool).to_numpy()
    out[sim_column] = np.where(keep, out["delta_m"].to_numpy(), 0.0)
    return out


@dataclass(frozen=True)
class DeltaStats:
    """Row-aggregate stats for a single delta column."""

    changed_rows: int
    rows_gt_1m: int
    rows_gt_5m: int
    sum_m: float
    mean_m: float
    p95_m: float
    p99_m: float
    max_m: float


def compute_delta_stats(series: pd.Series, *, eps: float = 1e-9) -> DeltaStats:
    arr = np.asarray(series, dtype=float)
    if arr.size == 0:
        return DeltaStats(0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0)
    return DeltaStats(
        changed_rows=int((arr > eps).sum()),
        rows_gt_1m=int((arr > 1.0).sum()),
        rows_gt_5m=int((arr > 5.0).sum()),
        sum_m=float(arr.sum()),
        mean_m=float(arr.mean()),
        p95_m=float(np.percentile(arr, 95)),
        p99_m=float(np.percentile(arr, 99)),
        max_m=float(arr.max()),
    )


def per_trip_shift_proxy(row_delta: pd.DataFrame, *, column: str) -> pd.Series:
    """``(p50 + p95) / 2`` of the absolute delta within each trip.

    This is a deterministic proxy for the per-trip Kaggle score swing; it is
    used to summarise how much the candidate run drifts from the base per
    trip.  Returns a Series indexed by ``tripId``.
    """

    if column not in row_delta.columns:
        raise ValueError(f"{column!r} not in row_delta columns")

    def _proxy(values: pd.Series) -> float:
        arr = np.asarray(values, dtype=float)
        if arr.size == 0:
            return 0.0
        return 0.5 * (float(np.percentile(arr, 50)) + float(np.percentile(arr, 95)))

    return row_delta.groupby("tripId")[column].apply(_proxy)


__all__ = [
    "DeltaStats",
    "REQUIRED_ROW_DELTA_COLUMNS",
    "compute_delta_stats",
    "dispositions_to_keep_map",
    "per_trip_shift_proxy",
    "simulate_revert",
]
