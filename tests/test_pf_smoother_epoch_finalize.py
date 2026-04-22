import numpy as np

import gnss_gpu.pf_smoother_epoch_finalize as epoch_finalize
from gnss_gpu.pf_smoother_config import CarrierRescueConfig
from gnss_gpu.pf_smoother_epoch_history import ForwardEpochHistory
from gnss_gpu.pf_smoother_epoch_state import create_epoch_forward_state
from gnss_gpu.pf_smoother_forward_stats import ForwardRunStats
from gnss_gpu.pf_smoother_runtime import ForwardRunBuffers


class _FakeParticleFilter:
    def __init__(self, state):
        self.state = np.asarray(state, dtype=np.float64)
        self.estimate_calls = 0

    def estimate(self):
        self.estimate_calls += 1
        return self.state


def test_finalize_forward_epoch_updates_tracker_alignment_and_history(monkeypatch):
    captured = {}

    def fake_update_tracker(
        tracker,
        carrier_rows,
        anchor_attempt,
        receiver_state,
        tow,
        dd_carrier_result,
        config,
    ):
        captured["tracker"] = tracker
        captured["carrier_rows"] = carrier_rows
        captured["receiver_state"] = receiver_state.copy()
        captured["tow"] = tow
        captured["dd_carrier_result"] = dd_carrier_result
        captured["config"] = config
        return 2

    monkeypatch.setattr(
        epoch_finalize,
        "update_carrier_bias_tracker_after_epoch",
        fake_update_tracker,
    )

    pf = _FakeParticleFilter([1.0, 2.0, 3.0, 4.0])
    buffers = ForwardRunBuffers()
    history = ForwardEpochHistory()
    stats = ForwardRunStats()
    state = create_epoch_forward_state(0.75)
    state.carrier_anchor_rows = {(0, 1): {"row": 1}}
    state.dd_carrier_result = object()
    tracker = {}
    config = CarrierRescueConfig(anchor_enabled=True)
    measurements = [{"sat": 1}]

    result = epoch_finalize.finalize_forward_epoch(
        pf,
        buffers,
        history,
        stats,
        carrier_bias_tracker=tracker,
        tow=10.0,
        measurements=measurements,
        run_name="Odaiba",
        gt=np.asarray([[5.0, 6.0, 7.0]], dtype=np.float64),
        our_times=np.asarray([10.0], dtype=np.float64),
        skip_valid_epochs=0,
        use_smoother=False,
        collect_epoch_diagnostics=False,
        epoch_state=state,
        rbpf_velocity_kf=False,
        gate_ess_ratio=0.5,
        gate_spread_m=2.0,
        carrier_anchor_sigma_m=0.25,
        carrier_rescue_config=config,
    )
    measurements.append({"sat": 2})

    assert pf.estimate_calls == 1
    assert result.carrier_anchor_propagated_rows == 2
    np.testing.assert_allclose(result.pf_estimate_now, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(result.pf_state_now, [1.0, 2.0, 3.0, 4.0])
    assert result.alignment.aligned is True
    assert stats.n_carrier_anchor_propagated == 2
    np.testing.assert_allclose(buffers.forward_aligned[0], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(buffers.gt_aligned[0], [5.0, 6.0, 7.0])
    assert history.prev_tow == 10.0
    assert history.prev_measurements == [{"sat": 1}]
    np.testing.assert_allclose(history.prev_estimate, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(history.prev_pf_state, [1.0, 2.0, 3.0, 4.0])
    assert history.epochs_done == 1
    assert captured["tracker"] is tracker
    assert captured["carrier_rows"] is state.carrier_anchor_rows
    assert captured["dd_carrier_result"] is state.dd_carrier_result
    assert captured["config"] is config
    np.testing.assert_allclose(captured["receiver_state"], [1.0, 2.0, 3.0, 4.0])


def test_finalize_forward_epoch_advances_history_when_alignment_skips(monkeypatch):
    monkeypatch.setattr(
        epoch_finalize,
        "update_carrier_bias_tracker_after_epoch",
        lambda *args, **kwargs: 0,
    )

    pf = _FakeParticleFilter([9.0, 8.0, 7.0, 6.0])
    buffers = ForwardRunBuffers()
    history = ForwardEpochHistory()
    stats = ForwardRunStats()

    result = epoch_finalize.finalize_forward_epoch(
        pf,
        buffers,
        history,
        stats,
        carrier_bias_tracker={},
        tow=10.0,
        measurements=[],
        run_name="Odaiba",
        gt=np.asarray([[5.0, 6.0, 7.0]], dtype=np.float64),
        our_times=np.asarray([10.0], dtype=np.float64),
        skip_valid_epochs=1,
        use_smoother=False,
        collect_epoch_diagnostics=False,
        epoch_state=create_epoch_forward_state(0.75),
        rbpf_velocity_kf=False,
        gate_ess_ratio=0.5,
        gate_spread_m=2.0,
        carrier_anchor_sigma_m=0.25,
        carrier_rescue_config=CarrierRescueConfig(),
    )

    assert result.alignment.aligned is False
    assert buffers.forward_aligned == []
    assert history.epochs_done == 1
    np.testing.assert_allclose(history.prev_estimate, [9.0, 8.0, 7.0])
    assert stats.n_carrier_anchor_propagated == 0
