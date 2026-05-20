"""Per-chunk ``fgo_no_tdcp`` promotion records for the Kaggle A/B audit.

The bridge writes a ``chunk_selection_records`` list per trip.  When the gate
chooses ``fgo_no_tdcp`` for a chunk, the corresponding record's
``gated_source`` field reflects the promotion.  This module collects those
promoted records (plus the no_tdcp candidate quality metrics) so the gate
module can inspect chunk-level signals separately from per-trip ones.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class PromotedNtdcChunk:
    """One chunk where ``gated_source == 'fgo_no_tdcp'``."""

    trip_id: str
    start_epoch: int
    end_epoch: int
    ntdc_quality_score: float
    ntdc_mse_pr: float
    ntdc_baseline_gap_max_m: float
    ntdc_step_p95_m: float
    ntdc_accel_p95_m: float

    @property
    def n_rows(self) -> int:
        return max(0, self.end_epoch - self.start_epoch)


def _normalize_trip(trip: str) -> str:
    for prefix in ("train/", "test/"):
        if trip.startswith(prefix):
            return trip[len(prefix) :]
    return trip


def _safe_float(value: Any) -> float:
    if value is None:
        return 0.0
    return float(value)


def _promoted_record(
    trip_id: str,
    record: dict[str, Any],
) -> PromotedNtdcChunk | None:
    if record.get("gated_source") != "fgo_no_tdcp":
        return None
    candidates = record.get("candidates") or {}
    ntdc = candidates.get("fgo_no_tdcp") or {}
    return PromotedNtdcChunk(
        trip_id=trip_id,
        start_epoch=int(record.get("start_epoch", 0) or 0),
        end_epoch=int(record.get("end_epoch", 0) or 0),
        ntdc_quality_score=_safe_float(ntdc.get("quality_score")),
        ntdc_mse_pr=_safe_float(ntdc.get("mse_pr")),
        ntdc_baseline_gap_max_m=_safe_float(ntdc.get("baseline_gap_max_m")),
        ntdc_step_p95_m=_safe_float(ntdc.get("step_p95_m")),
        ntdc_accel_p95_m=_safe_float(ntdc.get("accel_p95_m")),
    )


def extract_promoted_ntdc_chunks(
    trip_metrics: Iterable[dict[str, Any]],
) -> list[PromotedNtdcChunk]:
    """Pull every chunk whose final source is ``fgo_no_tdcp``.

    The records that did not promote to ``fgo_no_tdcp`` are filtered out so
    downstream gates only see chunks that actually contributed rows.
    """

    promoted: list[PromotedNtdcChunk] = []
    for tm in trip_metrics:
        trip_id = _normalize_trip(tm["trip"])
        for record in tm.get("chunk_selection_records") or []:
            entry = _promoted_record(trip_id, record)
            if entry is not None:
                promoted.append(entry)
    return promoted


def load_promoted_ntdc_chunks(summary_path: Path) -> list[PromotedNtdcChunk]:
    """Convenience wrapper to read promoted chunks straight from the JSON."""

    with Path(summary_path).open() as fh:
        summary = json.load(fh)
    return extract_promoted_ntdc_chunks(summary.get("trip_metrics") or [])


__all__ = [
    "PromotedNtdcChunk",
    "extract_promoted_ntdc_chunks",
    "load_promoted_ntdc_chunks",
]
