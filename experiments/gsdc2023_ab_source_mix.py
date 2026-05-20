"""Per-trip selected_source_counts for the GSDC2023 Kaggle bridge A/B.

This module loads and diffs the per-trip selected_source_counts used by
``gsdc2023_raw_bridge`` between two submission runs (bridge vs. TaroZ).  All
functions are pure: loaders take only a JSON path or directory and return
plain dataclasses; diffs operate on those dataclasses.  Downstream signal
and gate modules consume the same dataclasses without re-reading files.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Iterable


_SOURCE_FIELDS: tuple[str, ...] = (
    "baseline",
    "fgo",
    "fgo_dd_carrier",
    "fgo_no_tdcp",
    "raw_wls",
    "interpolated",
    "interpolated_missing",
)


@dataclass(frozen=True)
class SourceCounts:
    """Per-trip ``selected_source_counts`` snapshot for one submission run."""

    trip_id: str
    n_epochs: int
    baseline: int = 0
    fgo: int = 0
    fgo_dd_carrier: int = 0
    fgo_no_tdcp: int = 0
    raw_wls: int = 0
    interpolated: int = 0

    @property
    def total_nonbaseline(self) -> int:
        return self.fgo + self.fgo_dd_carrier + self.fgo_no_tdcp + self.raw_wls + self.interpolated


@dataclass(frozen=True)
class SourceCountDiff:
    """``taroz - bridge`` per-trip source-count delta."""

    trip_id: str
    n_epochs_bridge: int
    n_epochs_taroz: int
    bridge: SourceCounts
    taroz: SourceCounts
    d_baseline: int
    d_fgo: int
    d_fgo_dd_carrier: int
    d_fgo_no_tdcp: int
    d_raw_wls: int

    @property
    def gained_no_tdcp(self) -> bool:
        return self.d_fgo_no_tdcp > 0

    @property
    def gained_dd_carrier(self) -> bool:
        return self.d_fgo_dd_carrier > 0


def _normalize_trip(trip: str) -> str:
    """Drop the ``train/`` / ``test/`` prefix so bridge and taroz keys align."""

    for prefix in ("train/", "test/"):
        if trip.startswith(prefix):
            return trip[len(prefix) :]
    return trip


def _counts_from_dict(
    trip_id: str,
    n_epochs: int,
    ssc: dict[str, Any] | None,
) -> SourceCounts:
    ssc = ssc or {}
    # ``interpolated_missing`` is the older name used by the bridge for missing
    # epochs filled by interpolation; treat it as ``interpolated``.
    interpolated = int(ssc.get("interpolated", 0) or 0) + int(ssc.get("interpolated_missing", 0) or 0)
    return SourceCounts(
        trip_id=trip_id,
        n_epochs=int(n_epochs or 0),
        baseline=int(ssc.get("baseline", 0) or 0),
        fgo=int(ssc.get("fgo", 0) or 0),
        fgo_dd_carrier=int(ssc.get("fgo_dd_carrier", 0) or 0),
        fgo_no_tdcp=int(ssc.get("fgo_no_tdcp", 0) or 0),
        raw_wls=int(ssc.get("raw_wls", 0) or 0),
        interpolated=interpolated,
    )


def load_taroz_source_counts(summary_path: Path) -> list[SourceCounts]:
    """Read the TaroZ submission ``summary.json`` and return per-trip counts."""

    with Path(summary_path).open() as fh:
        summary = json.load(fh)
    trip_metrics = summary.get("trip_metrics") or []
    rows = [
        _counts_from_dict(
            trip_id=_normalize_trip(tm["trip"]),
            n_epochs=tm.get("n_epochs", 0),
            ssc=tm.get("selected_source_counts"),
        )
        for tm in trip_metrics
    ]
    return rows


def load_bridge_source_counts(bridge_root: Path) -> list[SourceCounts]:
    """Walk the per-trip ``bridge_metrics.json`` tree of a bridge run."""

    root = Path(bridge_root)
    rows: list[SourceCounts] = []
    for path in sorted(root.rglob("bridge_metrics.json")):
        rel = path.relative_to(root)
        if len(rel.parts) < 3:
            # Expect ``<trip>/<phone>/bridge_metrics.json``
            continue
        trip = "/".join(rel.parts[:2])
        with path.open() as fh:
            metrics = json.load(fh)
        rows.append(
            _counts_from_dict(
                trip_id=trip,
                n_epochs=metrics.get("n_epochs", 0),
                ssc=metrics.get("selected_source_counts"),
            )
        )
    return rows


def diff_source_counts(
    bridge: Iterable[SourceCounts],
    taroz: Iterable[SourceCounts],
) -> list[SourceCountDiff]:
    """Join bridge and taroz by ``trip_id`` and report per-trip deltas.

    Missing trips on either side are treated as zero counts, so the union of
    trip ids is preserved.  The returned list is sorted by ``trip_id`` for
    deterministic output.
    """

    bridge_map = {sc.trip_id: sc for sc in bridge}
    taroz_map = {sc.trip_id: sc for sc in taroz}
    trip_ids = sorted(set(bridge_map) | set(taroz_map))
    diffs: list[SourceCountDiff] = []
    for trip_id in trip_ids:
        b = bridge_map.get(trip_id, SourceCounts(trip_id=trip_id, n_epochs=0))
        t = taroz_map.get(trip_id, SourceCounts(trip_id=trip_id, n_epochs=0))
        diffs.append(
            SourceCountDiff(
                trip_id=trip_id,
                n_epochs_bridge=b.n_epochs,
                n_epochs_taroz=t.n_epochs,
                bridge=b,
                taroz=t,
                d_baseline=t.baseline - b.baseline,
                d_fgo=t.fgo - b.fgo,
                d_fgo_dd_carrier=t.fgo_dd_carrier - b.fgo_dd_carrier,
                d_fgo_no_tdcp=t.fgo_no_tdcp - b.fgo_no_tdcp,
                d_raw_wls=t.raw_wls - b.raw_wls,
            )
        )
    return diffs


def aggregate_source_totals(counts: Iterable[SourceCounts]) -> dict[str, int]:
    """Sum each source column across all trips; useful for sanity prints."""

    totals = {field: 0 for field in _SOURCE_FIELDS}
    for sc in counts:
        totals["baseline"] += sc.baseline
        totals["fgo"] += sc.fgo
        totals["fgo_dd_carrier"] += sc.fgo_dd_carrier
        totals["fgo_no_tdcp"] += sc.fgo_no_tdcp
        totals["raw_wls"] += sc.raw_wls
        totals["interpolated"] += sc.interpolated
    return totals


__all__ = [
    "SourceCounts",
    "SourceCountDiff",
    "aggregate_source_totals",
    "diff_source_counts",
    "load_bridge_source_counts",
    "load_taroz_source_counts",
]
