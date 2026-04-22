from dataclasses import dataclass

import numpy as np

import gnss_gpu.carrier_bias_tracker_update as tracker_update
from gnss_gpu.carrier_rescue import CarrierAnchorAttempt
from gnss_gpu.pf_smoother_config import CarrierRescueConfig


@dataclass
class _DummyDDResult:
    n_dd: int


def test_update_carrier_bias_tracker_after_epoch_skips_when_disabled(monkeypatch):
    calls = []
    monkeypatch.setattr(
        tracker_update,
        "_update_carrier_bias_tracker",
        lambda *args, **kwargs: calls.append(kwargs),
    )

    propagated = tracker_update.update_carrier_bias_tracker_after_epoch(
        {},
        {(0, 1): {}},
        CarrierAnchorAttempt(),
        np.zeros(4, dtype=np.float64),
        10.0,
        _DummyDDResult(4),
        CarrierRescueConfig(anchor_enabled=False),
    )

    assert propagated == 0
    assert calls == []


def test_update_carrier_bias_tracker_after_epoch_trusts_strong_dd_epoch(monkeypatch):
    calls = []

    def fake_update(*args, **kwargs):
        calls.append({"rows": args[1], "state": args[2], **kwargs})

    monkeypatch.setattr(tracker_update, "_update_carrier_bias_tracker", fake_update)

    rows = {(0, 1): {"row": True}}
    propagated = tracker_update.update_carrier_bias_tracker_after_epoch(
        {},
        rows,
        CarrierAnchorAttempt(),
        np.ones(4, dtype=np.float64),
        10.0,
        _DummyDDResult(4),
        CarrierRescueConfig(anchor_enabled=True, anchor_seed_dd_min_pairs=3),
    )

    assert propagated == 0
    assert calls[0]["rows"] is rows
    assert calls[0]["trusted"] is True
    np.testing.assert_allclose(calls[0]["state"], np.ones(4))


def test_update_carrier_bias_tracker_after_epoch_refreshes_anchor_rows_untrusted(monkeypatch):
    calls = []
    monkeypatch.setattr(
        tracker_update,
        "_update_carrier_bias_tracker",
        lambda *args, **kwargs: calls.append({"rows": args[1], **kwargs}),
    )

    anchor_attempt = CarrierAnchorAttempt(
        used=True,
        rows_used={(0, 2): {"row": True}},
    )

    propagated = tracker_update.update_carrier_bias_tracker_after_epoch(
        {},
        {},
        anchor_attempt,
        np.zeros(4, dtype=np.float64),
        10.0,
        None,
        CarrierRescueConfig(anchor_enabled=True),
    )

    assert propagated == 0
    assert calls[0]["rows"] is anchor_attempt.rows_used
    assert calls[0]["trusted"] is False


def test_update_carrier_bias_tracker_after_epoch_propagates_when_anchor_state_available(monkeypatch):
    calls = []

    def fake_propagate(*args, **kwargs):
        calls.append({"rows": args[1], "state": args[2], **kwargs})
        return 2

    monkeypatch.setattr(
        tracker_update,
        "_propagate_carrier_bias_tracker_tdcp",
        fake_propagate,
    )

    rows = {(0, 1): {}, (0, 2): {}}
    anchor_attempt = CarrierAnchorAttempt(
        state=np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
    )

    propagated = tracker_update.update_carrier_bias_tracker_after_epoch(
        {},
        rows,
        anchor_attempt,
        np.zeros(4, dtype=np.float64),
        10.0,
        None,
        CarrierRescueConfig(anchor_enabled=True, anchor_blend_alpha=0.6),
    )

    assert propagated == 2
    assert anchor_attempt.propagated_rows == 2
    assert calls[0]["rows"] is rows
    np.testing.assert_allclose(calls[0]["state"], [1.0, 2.0, 3.0, 4.0])
    assert calls[0]["blend_alpha"] == 0.6
