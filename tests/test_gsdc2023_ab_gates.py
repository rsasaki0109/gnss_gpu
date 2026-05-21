from __future__ import annotations

from experiments.gsdc2023_ab_dd_signals import DDSignals
from experiments.gsdc2023_ab_gates import (
    CombinedGate,
    DDAnchorGate,
    Disposition,
    NoTdcpCoexistGate,
    apply_gate,
    disposition_counts,
)
from experiments.gsdc2023_ab_source_mix import SourceCounts


def _signals(trip: str, *, anchor_cov: float, n_epochs: int = 1000) -> DDSignals:
    anchor = int(round(anchor_cov * n_epochs))
    return DDSignals(
        trip_id=trip,
        n_epochs=n_epochs,
        dd_anchor_epochs=anchor,
        dd_dd_epochs=anchor,
        dd_base_snapped_epochs=anchor,
        dd_pairs_mean=5.0,
    )


def _counts(trip: str, *, baseline: int = 800, dd: int = 0, no_tdcp: int = 0) -> SourceCounts:
    return SourceCounts(
        trip_id=trip,
        n_epochs=1000,
        baseline=baseline,
        fgo_dd_carrier=dd,
        fgo_no_tdcp=no_tdcp,
    )


# --- DDAnchorGate ----------------------------------------------------------


def test_dd_gate_requires_dd_rows_present():
    gate = DDAnchorGate()
    assert gate.decide(_signals("a", anchor_cov=0.9), _counts("a", dd=0)) is False


def test_dd_gate_passes_when_anchor_above_floor_and_no_tdcp_absent():
    gate = DDAnchorGate(min_anchor_coverage=0.6)
    assert gate.decide(_signals("a", anchor_cov=0.65), _counts("a", dd=200)) is True


def test_dd_gate_blocks_when_anchor_below_floor():
    gate = DDAnchorGate(min_anchor_coverage=0.6)
    assert gate.decide(_signals("a", anchor_cov=0.45), _counts("a", dd=200)) is False


def test_dd_gate_blocks_when_no_tdcp_coexists_by_default():
    gate = DDAnchorGate()
    assert gate.decide(_signals("a", anchor_cov=0.9), _counts("a", dd=200, no_tdcp=100)) is False


def test_dd_gate_can_disable_no_tdcp_coexistence_check():
    gate = DDAnchorGate(require_no_tdcp_absent=False)
    assert gate.decide(_signals("a", anchor_cov=0.9), _counts("a", dd=200, no_tdcp=100)) is True


# --- NoTdcpCoexistGate -----------------------------------------------------


def test_ntdc_gate_requires_no_tdcp_rows_present():
    gate = NoTdcpCoexistGate()
    assert gate.decide(_signals("a", anchor_cov=0.9), _counts("a", no_tdcp=0)) is False


def test_ntdc_gate_passes_when_dd_absent_and_anchor_above_floor():
    gate = NoTdcpCoexistGate(min_anchor_coverage=0.6)
    assert gate.decide(_signals("a", anchor_cov=0.7), _counts("a", no_tdcp=100)) is True


def test_ntdc_gate_blocks_when_dd_coexists():
    gate = NoTdcpCoexistGate()
    assert gate.decide(_signals("a", anchor_cov=0.9), _counts("a", dd=100, no_tdcp=100)) is False


def test_ntdc_gate_blocks_when_anchor_below_floor():
    gate = NoTdcpCoexistGate(min_anchor_coverage=0.6)
    assert gate.decide(_signals("a", anchor_cov=0.5), _counts("a", no_tdcp=100)) is False


# --- CombinedGate ----------------------------------------------------------


def test_combined_returns_passthrough_when_no_activity():
    gate = CombinedGate()
    assert gate.decide(_signals("a", anchor_cov=0.9), _counts("a")) is Disposition.PASSTHROUGH


def test_combined_returns_kept_when_dd_passes():
    gate = CombinedGate()
    assert gate.decide(_signals("a", anchor_cov=0.75), _counts("a", dd=200)) is Disposition.KEPT


def test_combined_returns_kept_when_ntdc_passes_alone():
    gate = CombinedGate()
    assert gate.decide(_signals("a", anchor_cov=0.75), _counts("a", no_tdcp=100)) is Disposition.KEPT


def test_combined_returns_reverted_when_both_fail():
    gate = CombinedGate()
    assert (
        gate.decide(_signals("a", anchor_cov=0.45), _counts("a", dd=200, no_tdcp=100))
        is Disposition.REVERTED
    )


def test_apply_gate_aligns_signals_and_counts_by_trip_id():
    gate = CombinedGate()
    signals = [
        _signals("a", anchor_cov=0.75),
        _signals("b", anchor_cov=0.5),
    ]
    counts = [
        _counts("a", dd=200),
        _counts("b", dd=200),
        _counts("c"),  # passthrough
    ]
    dispositions = apply_gate(gate, signals, counts)
    by_trip = {d.trip_id: d for d in dispositions}
    assert by_trip["a"].disposition is Disposition.KEPT
    assert by_trip["b"].disposition is Disposition.REVERTED
    assert by_trip["c"].disposition is Disposition.PASSTHROUGH


def test_apply_gate_emits_passthrough_when_one_side_missing():
    gate = CombinedGate()
    dispositions = apply_gate(gate, [_signals("a", anchor_cov=0.9)], [])
    assert [d.trip_id for d in dispositions] == ["a"]
    assert dispositions[0].disposition is Disposition.PASSTHROUGH


def test_disposition_counts_returns_value_keys():
    dispositions = [
        # (trip, dd_pass, ntdc_pass) only for completeness; the disposition is what matters
    ]
    from experiments.gsdc2023_ab_gates import TripDisposition

    dispositions = [
        TripDisposition("a", Disposition.KEPT, True, False),
        TripDisposition("b", Disposition.KEPT, False, True),
        TripDisposition("c", Disposition.REVERTED, False, False),
        TripDisposition("d", Disposition.PASSTHROUGH, False, False),
        TripDisposition("e", Disposition.PASSTHROUGH, False, False),
    ]
    counts = disposition_counts(dispositions)
    assert counts == {"passthrough": 2, "kept": 2, "reverted": 1}
