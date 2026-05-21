"""Per-trip whitelist gates for the GSDC2023 Kaggle bridge A/B.

Two gates are derived directly from the Phase 73-75 audit:

- ``DDAnchorGate``: keep ``fgo_dd_carrier`` only when DD anchor coverage is high
  and no ``fgo_no_tdcp`` is mixed into the same trip.
- ``NoTdcpCoexistGate``: keep ``fgo_no_tdcp`` only when ``fgo_dd_carrier`` is
  absent and DD anchor coverage is high.

The combined gate is purely a logical OR over the two: a trip is retained iff
at least one of its active non-baseline sources passes its own gate.  Trips
that have *no* DD/no_tdcp activity in the candidate submission are
``passthrough``: gates never block them, but they also do not constitute a
reason to keep the submission's row delta.

Inputs are the dataclasses from :mod:`gsdc2023_ab_source_mix` and
:mod:`gsdc2023_ab_dd_signals`; gates do not touch JSON.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from experiments.gsdc2023_ab_dd_signals import DDSignals
from experiments.gsdc2023_ab_source_mix import SourceCounts


class Disposition(str, Enum):
    """Per-trip outcome under a gate."""

    PASSTHROUGH = "passthrough"  # no DD / no_tdcp activity; gate is silent
    KEPT = "kept"  # at least one active source passes its gate
    REVERTED = "reverted"  # active source(s) all blocked; revert to bridge


@dataclass(frozen=True)
class DDAnchorGate:
    """Keep ``fgo_dd_carrier`` iff anchor coverage >= floor AND no_tdcp absent."""

    min_anchor_coverage: float = 0.6
    require_no_tdcp_absent: bool = True

    def decide(self, signals: DDSignals, counts: SourceCounts) -> bool:
        if counts.fgo_dd_carrier <= 0:
            return False
        if signals.anchor_coverage < self.min_anchor_coverage:
            return False
        if self.require_no_tdcp_absent and counts.fgo_no_tdcp > 0:
            return False
        return True


@dataclass(frozen=True)
class NoTdcpCoexistGate:
    """Keep ``fgo_no_tdcp`` iff no DD-carrier and anchor coverage >= floor."""

    min_anchor_coverage: float = 0.6
    require_no_dd_carrier: bool = True

    def decide(self, signals: DDSignals, counts: SourceCounts) -> bool:
        if counts.fgo_no_tdcp <= 0:
            return False
        if signals.anchor_coverage < self.min_anchor_coverage:
            return False
        if self.require_no_dd_carrier and counts.fgo_dd_carrier > 0:
            return False
        return True


@dataclass(frozen=True)
class CombinedGate:
    """Logical OR of the DD and no_tdcp gates with a passthrough sentinel."""

    dd: DDAnchorGate = DDAnchorGate()
    ntdc: NoTdcpCoexistGate = NoTdcpCoexistGate()

    def has_activity(self, counts: SourceCounts) -> bool:
        return counts.fgo_dd_carrier > 0 or counts.fgo_no_tdcp > 0

    def decide(self, signals: DDSignals, counts: SourceCounts) -> Disposition:
        if not self.has_activity(counts):
            return Disposition.PASSTHROUGH
        if self.dd.decide(signals, counts) or self.ntdc.decide(signals, counts):
            return Disposition.KEPT
        return Disposition.REVERTED


@dataclass(frozen=True)
class TripDisposition:
    """Result of applying ``CombinedGate`` to a single trip."""

    trip_id: str
    disposition: Disposition
    dd_pass: bool
    ntdc_pass: bool


def apply_gate(
    gate: CombinedGate,
    signals: Iterable[DDSignals],
    counts: Iterable[SourceCounts],
) -> list[TripDisposition]:
    """Join ``DDSignals`` and ``SourceCounts`` by ``trip_id`` and apply gate.

    Missing entries on either side fall back to ``passthrough`` so the result
    always covers the union of trip ids.  Trip order follows the union sort.
    """

    signal_map = {s.trip_id: s for s in signals}
    count_map = {c.trip_id: c for c in counts}
    trip_ids = sorted(set(signal_map) | set(count_map))
    out: list[TripDisposition] = []
    for trip_id in trip_ids:
        sig = signal_map.get(trip_id)
        cnt = count_map.get(trip_id)
        if sig is None or cnt is None:
            out.append(TripDisposition(trip_id, Disposition.PASSTHROUGH, False, False))
            continue
        dd_pass = gate.dd.decide(sig, cnt)
        ntdc_pass = gate.ntdc.decide(sig, cnt)
        out.append(
            TripDisposition(
                trip_id=trip_id,
                disposition=gate.decide(sig, cnt),
                dd_pass=dd_pass,
                ntdc_pass=ntdc_pass,
            )
        )
    return out


def disposition_counts(dispositions: Iterable[TripDisposition]) -> dict[str, int]:
    counts: dict[str, int] = {d.value: 0 for d in Disposition}
    for entry in dispositions:
        counts[entry.disposition.value] += 1
    return counts


__all__ = [
    "CombinedGate",
    "DDAnchorGate",
    "Disposition",
    "NoTdcpCoexistGate",
    "TripDisposition",
    "apply_gate",
    "disposition_counts",
]
