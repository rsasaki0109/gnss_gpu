"""Approximate sub-chunk merge for the GSDC2023 Kaggle A/B audit.

When the candidate run uses ``chunk_epochs=100`` but the base run used
``chunk_epochs=200``, the chunk-level gate sees different aggregate statistics
and can flip the ``gated_source`` decision (see Phase 76-77 audit).  This
module reconstructs ``ChunkCandidateQuality`` values for the *merged* (e.g.
adjacent pairwise) chunks from the candidate run's per-chunk records, then
re-evaluates the production gate (``select_gated_chunk_source``) on the
merged chunks.

Approximations and their consequences are explicit:

- ``mse_pr``, ``step_mean_m``, ``accel_mean_m``, ``baseline_gap_mean_m``: epoch
  count weighted average across the merged sub-chunks.  Exact when the input
  was itself an epoch-count weighted average, which is the case for the
  bridge's per-chunk MSE.
- ``step_p95_m``, ``accel_p95_m``, ``baseline_gap_p95_m``: upper-bounded by
  ``max`` across sub-chunks.  This *overestimates* the true p95 of the merged
  window, so a merged candidate that still passes the gate under this upper
  bound would pass the exact p95 too.  Conversely, a merged candidate that
  fails under this approximation may still pass with the true p95.
- ``baseline_gap_max_m``: exact via ``max``.
- ``bridge_jump_m``: kept from the first sub-chunk because the merged window's
  entry is the first sub-chunk's entry.
- ``quality_score``: recomputed from the above using the production formula
  in :func:`gsdc2023_chunk_selection.chunk_candidate_quality`.

The module intentionally does not duplicate the gate logic - it imports
``select_gated_chunk_source`` and ``ChunkSelectionRecord`` directly, so any
future change in the gate flows through automatically.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable

from experiments.gsdc2023_chunk_selection import (
    ChunkCandidateQuality,
    ChunkSelectionRecord,
    select_auto_chunk_source,
    select_gated_chunk_source,
)


_QS_WEIGHTS = (0.40, 0.10, 0.10, 0.10, 0.10, 0.20)
_QS_FLOORS = (10.0, 0.25, 0.5, 0.5, 0.5)
_GAP_P95_DIVISOR_FLOOR = 5.0


@dataclass(frozen=True)
class ChunkComparison:
    """Per-chunk comparison of bridge actual vs candidate-merged decision."""

    trip_id: str
    start_epoch: int
    end_epoch: int
    bridge_gated_source: str
    candidate_merged_gated_source: str
    matches: bool

    @property
    def n_rows(self) -> int:
        return max(0, self.end_epoch - self.start_epoch)


def _weighted_mean(values: list[float], weights: list[int]) -> float:
    total = float(sum(weights))
    if total <= 0:
        return 0.0
    return float(sum(v * w for v, w in zip(values, weights))) / total


def _quality_ratio(value: float, reference: float, floor: float) -> float:
    return float(value) / max(float(reference), float(floor))


def _recompute_quality_score(
    quality: ChunkCandidateQuality,
    baseline: ChunkCandidateQuality | None,
) -> float:
    """Mirror ``chunk_candidate_quality``'s quality_score formula."""

    if baseline is None:
        return 1.0
    w_mse, w_step_mean, w_step_p95, w_accel_p95, w_bridge, w_gap = _QS_WEIGHTS
    f_mse, f_step_mean, f_step_p95, f_accel_p95, f_bridge = _QS_FLOORS
    return (
        w_mse * _quality_ratio(quality.mse_pr, baseline.mse_pr, f_mse)
        + w_step_mean * _quality_ratio(quality.step_mean_m, baseline.step_mean_m, f_step_mean)
        + w_step_p95 * _quality_ratio(quality.step_p95_m, baseline.step_p95_m, f_step_p95)
        + w_accel_p95 * _quality_ratio(quality.accel_p95_m, baseline.accel_p95_m, f_accel_p95)
        + w_bridge * _quality_ratio(quality.bridge_jump_m, baseline.bridge_jump_m, f_bridge)
        + w_gap
        * (quality.baseline_gap_p95_m / max(baseline.step_p95_m, _GAP_P95_DIVISOR_FLOOR))
    )


def merge_candidate_quality(
    sub_qualities: list[ChunkCandidateQuality],
    sub_sizes: list[int],
    *,
    merged_baseline: ChunkCandidateQuality | None,
) -> ChunkCandidateQuality:
    """Approximate the merged-chunk ``ChunkCandidateQuality``."""

    if len(sub_qualities) != len(sub_sizes):
        raise ValueError("sub_qualities and sub_sizes must have equal length")
    if not sub_qualities:
        raise ValueError("at least one sub-chunk is required")

    def wmean(getter):
        return _weighted_mean([getter(q) for q in sub_qualities], sub_sizes)

    def upper(getter):
        return max(getter(q) for q in sub_qualities)

    merged = ChunkCandidateQuality(
        mse_pr=wmean(lambda q: q.mse_pr),
        step_mean_m=wmean(lambda q: q.step_mean_m),
        step_p95_m=upper(lambda q: q.step_p95_m),
        accel_mean_m=wmean(lambda q: q.accel_mean_m),
        accel_p95_m=upper(lambda q: q.accel_p95_m),
        bridge_jump_m=float(sub_qualities[0].bridge_jump_m),
        baseline_gap_mean_m=wmean(lambda q: q.baseline_gap_mean_m),
        baseline_gap_p95_m=upper(lambda q: q.baseline_gap_p95_m),
        baseline_gap_max_m=upper(lambda q: q.baseline_gap_max_m),
        quality_score=0.0,  # placeholder, recomputed below
    )
    quality_score = _recompute_quality_score(merged, merged_baseline)
    return ChunkCandidateQuality(
        mse_pr=merged.mse_pr,
        step_mean_m=merged.step_mean_m,
        step_p95_m=merged.step_p95_m,
        accel_mean_m=merged.accel_mean_m,
        accel_p95_m=merged.accel_p95_m,
        bridge_jump_m=merged.bridge_jump_m,
        baseline_gap_mean_m=merged.baseline_gap_mean_m,
        baseline_gap_p95_m=merged.baseline_gap_p95_m,
        baseline_gap_max_m=merged.baseline_gap_max_m,
        quality_score=quality_score,
    )


def _candidates_from_dict(raw: dict[str, Any]) -> dict[str, ChunkCandidateQuality]:
    out: dict[str, ChunkCandidateQuality] = {}
    for name, payload in (raw or {}).items():
        out[name] = ChunkCandidateQuality(
            mse_pr=float(payload.get("mse_pr", 0.0) or 0.0),
            step_mean_m=float(payload.get("step_mean_m", 0.0) or 0.0),
            step_p95_m=float(payload.get("step_p95_m", 0.0) or 0.0),
            accel_mean_m=float(payload.get("accel_mean_m", 0.0) or 0.0),
            accel_p95_m=float(payload.get("accel_p95_m", 0.0) or 0.0),
            bridge_jump_m=float(payload.get("bridge_jump_m", 0.0) or 0.0),
            baseline_gap_mean_m=float(payload.get("baseline_gap_mean_m", 0.0) or 0.0),
            baseline_gap_p95_m=float(payload.get("baseline_gap_p95_m", 0.0) or 0.0),
            baseline_gap_max_m=float(payload.get("baseline_gap_max_m", 0.0) or 0.0),
            quality_score=float(payload.get("quality_score", 1.0) or 1.0),
        )
    return out


def merge_adjacent_chunks_pairwise(
    chunks: list[dict[str, Any]],
) -> list[ChunkSelectionRecord]:
    """Merge adjacent sub-chunks pairwise into ``ChunkSelectionRecord`` instances.

    Each input is the raw dict shape stored by the bridge in
    ``trip_metrics[*].chunk_selection_records`` (i.e. ``{start_epoch,
    end_epoch, auto_source, gated_source, candidates}``).  When ``chunks`` has
    an odd number of entries, the final entry is emitted as a single-chunk
    record (no merge partner).
    """

    merged: list[ChunkSelectionRecord] = []
    i = 0
    while i < len(chunks):
        a = chunks[i]
        b = chunks[i + 1] if i + 1 < len(chunks) else None
        if b is None:
            sub = [_candidates_from_dict(a.get("candidates", {}))]
            sizes = [int(a["end_epoch"]) - int(a["start_epoch"])]
            start = int(a["start_epoch"])
            end = int(a["end_epoch"])
        else:
            sub = [
                _candidates_from_dict(a.get("candidates", {})),
                _candidates_from_dict(b.get("candidates", {})),
            ]
            sizes = [
                int(a["end_epoch"]) - int(a["start_epoch"]),
                int(b["end_epoch"]) - int(b["start_epoch"]),
            ]
            start = int(a["start_epoch"])
            end = int(b["end_epoch"])

        names = set().union(*[set(c) for c in sub])
        merged_candidates: dict[str, ChunkCandidateQuality] = {}
        merged_baseline = None
        if "baseline" in names and all("baseline" in c for c in sub):
            merged_baseline = merge_candidate_quality(
                [c["baseline"] for c in sub], sizes, merged_baseline=None
            )
            merged_candidates["baseline"] = merged_baseline
        for name in sorted(names - {"baseline"}):
            if not all(name in c for c in sub):
                # Skip candidates that aren't present in every sub-chunk; we
                # cannot honestly merge a quality from missing data.
                continue
            merged_candidates[name] = merge_candidate_quality(
                [c[name] for c in sub], sizes, merged_baseline=merged_baseline
            )
        # ``select_auto_chunk_source`` requires a ``baseline`` candidate.  If
        # the input records lack one in any sub-chunk (the merge step above
        # dropped it because it must be present in *every* sub-chunk), fall
        # back to ``baseline`` as the auto source.  This keeps the simulation
        # honest about what the production gate can decide.
        auto = (
            select_auto_chunk_source(merged_candidates)
            if "baseline" in merged_candidates
            else "baseline"
        )
        merged.append(
            ChunkSelectionRecord(
                start_epoch=start,
                end_epoch=end,
                auto_source=auto,
                candidates=merged_candidates,
            )
        )
        i += 2 if b is not None else 1
    return merged


def predict_merged_gated_sources(
    candidate_chunks: list[dict[str, Any]],
    *,
    gated_threshold: float = 500.0,
    allow_raw_wls_on_mi8_baseline_jump: bool = False,
    raw_wls_max_gap_m: float | None = None,
    allow_fgo_raw_wls_proxy_rescue: bool = False,
) -> list[tuple[int, int, str]]:
    """Predict ``gated_source`` per merged chunk using the production gate.

    Returns ``[(start_epoch, end_epoch, gated_source), ...]``.  The gate
    parameters mirror ``select_gated_chunk_source``'s signature so the caller
    can match the candidate run's config.
    """

    merged = merge_adjacent_chunks_pairwise(candidate_chunks)
    out: list[tuple[int, int, str]] = []
    for r in merged:
        if "baseline" not in r.candidates:
            # Production gate requires a baseline candidate; default to
            # ``baseline`` when the merge step could not reconstruct one.
            out.append((int(r.start_epoch), int(r.end_epoch), "baseline"))
            continue
        out.append(
            (
                int(r.start_epoch),
                int(r.end_epoch),
                select_gated_chunk_source(
                    r,
                    gated_threshold,
                    allow_raw_wls_on_mi8_baseline_jump=allow_raw_wls_on_mi8_baseline_jump,
                    raw_wls_max_gap_m=raw_wls_max_gap_m,
                    allow_fgo_raw_wls_proxy_rescue=allow_fgo_raw_wls_proxy_rescue,
                ),
            )
        )
    return out


def _epoch_to_source(
    chunks: Iterable[dict[str, Any]],
    n_epochs: int,
    key: str = "gated_source",
) -> list[str | None]:
    out: list[str | None] = [None] * n_epochs
    for r in chunks:
        s = int(r["start_epoch"])
        e = int(r["end_epoch"])
        src = r.get(key, "baseline")
        for i in range(s, min(e, n_epochs)):
            out[i] = src
    return out


def compare_bridge_to_merged_candidate(
    *,
    trip_id: str,
    n_epochs: int,
    bridge_chunks: list[dict[str, Any]],
    merged_predictions: list[tuple[int, int, str]],
) -> list[ChunkComparison]:
    """Align bridge actual chunks against merged-candidate predictions by epoch.

    Returns a per-merged-chunk record indicating whether the predicted
    ``gated_source`` matches the bridge's actual ``gated_source`` for the same
    epoch span.  Mismatches are the chunks where chunk_epochs alone does not
    explain the regression.
    """

    bridge_source = _epoch_to_source(bridge_chunks, n_epochs, key="gated_source")
    out: list[ChunkComparison] = []
    for start, end, pred in merged_predictions:
        start = max(0, min(start, n_epochs))
        end = max(0, min(end, n_epochs))
        if start >= end:
            continue
        sources = [s for s in bridge_source[start:end] if s is not None]
        if not sources:
            continue
        # Bridge chunk for this region; use majority vote in case of
        # boundary mismatch.
        from collections import Counter

        bridge_src = Counter(sources).most_common(1)[0][0]
        out.append(
            ChunkComparison(
                trip_id=trip_id,
                start_epoch=start,
                end_epoch=end,
                bridge_gated_source=bridge_src,
                candidate_merged_gated_source=pred,
                matches=(bridge_src == pred),
            )
        )
    return out


def load_chunk_records_from_summary(
    summary_path: Path,
) -> dict[str, dict[str, Any]]:
    """Return ``{trip_id: {"n_epochs": int, "chunks": [raw dicts]}}``."""

    with Path(summary_path).open() as fh:
        summary = json.load(fh)
    out: dict[str, dict[str, Any]] = {}
    for tm in summary.get("trip_metrics") or []:
        trip = tm["trip"]
        for prefix in ("train/", "test/"):
            if trip.startswith(prefix):
                trip = trip[len(prefix) :]
                break
        out[trip] = {
            "n_epochs": int(tm.get("n_epochs", 0) or 0),
            "chunks": list(tm.get("chunk_selection_records") or []),
        }
    return out


def load_bridge_chunk_records(
    bridge_root: Path,
) -> dict[str, dict[str, Any]]:
    """Return ``{trip_id: {"n_epochs": int, "chunks": [...]}}`` from bridge dir."""

    root = Path(bridge_root)
    out: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("bridge_metrics.json")):
        rel = path.relative_to(root)
        if len(rel.parts) < 3:
            continue
        trip = "/".join(rel.parts[:2])
        with path.open() as fh:
            metrics = json.load(fh)
        out[trip] = {
            "n_epochs": int(metrics.get("n_epochs", 0) or 0),
            "chunks": list(metrics.get("chunk_selection_records") or []),
        }
    return out


__all__ = [
    "ChunkComparison",
    "compare_bridge_to_merged_candidate",
    "load_bridge_chunk_records",
    "load_chunk_records_from_summary",
    "merge_adjacent_chunks_pairwise",
    "merge_candidate_quality",
    "predict_merged_gated_sources",
]
