"""Per-trip DD-carrier signal extraction for the GSDC2023 Kaggle A/B audit.

``DDSignals`` collects the four ``dd_carrier_*`` fields the bridge writes per
trip plus a derived ``anchor_coverage`` (``dd_anchor_epochs / n_epochs``) used
by the whitelist gates.  Extraction is a pure transform over the raw
``trip_metrics`` dicts returned by the TaroZ submission summary so the gate
module never has to touch JSON.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class DDSignals:
    """Per-trip DD-carrier support summary."""

    trip_id: str
    n_epochs: int
    dd_anchor_epochs: int
    dd_dd_epochs: int
    dd_base_snapped_epochs: int
    dd_pairs_mean: float

    @property
    def anchor_coverage(self) -> float:
        """Fraction of trip epochs with an accepted DD anchor."""

        if self.n_epochs <= 0:
            return 0.0
        return self.dd_anchor_epochs / float(self.n_epochs)


def _normalize_trip(trip: str) -> str:
    for prefix in ("train/", "test/"):
        if trip.startswith(prefix):
            return trip[len(prefix) :]
    return trip


def _signals_from_trip_metrics(tm: dict[str, Any]) -> DDSignals:
    return DDSignals(
        trip_id=_normalize_trip(tm["trip"]),
        n_epochs=int(tm.get("n_epochs", 0) or 0),
        dd_anchor_epochs=int(tm.get("dd_carrier_accepted_anchor_epochs", 0) or 0),
        dd_dd_epochs=int(tm.get("dd_carrier_dd_epochs", 0) or 0),
        dd_base_snapped_epochs=int(tm.get("dd_carrier_base_snapped_epochs", 0) or 0),
        dd_pairs_mean=float(tm.get("dd_carrier_dd_pairs_mean", 0.0) or 0.0),
    )


def extract_dd_signals(trip_metrics: Iterable[dict[str, Any]]) -> list[DDSignals]:
    """Convert the raw ``trip_metrics`` list into ``DDSignals`` dataclasses."""

    return [_signals_from_trip_metrics(tm) for tm in trip_metrics]


def load_dd_signals_from_summary(summary_path: Path) -> list[DDSignals]:
    """Convenience wrapper that reads the TaroZ summary JSON from disk."""

    with Path(summary_path).open() as fh:
        summary = json.load(fh)
    return extract_dd_signals(summary.get("trip_metrics") or [])


__all__ = [
    "DDSignals",
    "extract_dd_signals",
    "load_dd_signals_from_summary",
]
