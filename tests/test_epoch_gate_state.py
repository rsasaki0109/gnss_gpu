import numpy as np

from gnss_gpu.epoch_gate_state import compute_epoch_gate_state
from gnss_gpu.pf_smoother_config import PfSmootherConfig


class FakeParticleFilter:
    n_particles = 100

    def __init__(self):
        self.estimate_calls = 0
        self.ess_calls = 0
        self.spread_calls = 0

    def estimate(self):
        self.estimate_calls += 1
        return np.array([1.0, 2.0, 3.0, 4.0])

    def get_ess(self):
        self.ess_calls += 1
        return 20.0

    def get_position_spread(self, *, center):
        self.spread_calls += 1
        np.testing.assert_allclose(center, [1.0, 2.0, 3.0])
        return 4.0


def _config(**overrides):
    values = {
        "n_particles": 100,
        "sigma_pos": 1.2,
        "sigma_pr": 3.0,
        "position_update_sigma": 1.9,
        "predict_guide": "imu",
        "use_smoother": True,
    }
    values.update(overrides)
    return PfSmootherConfig(**values)


def test_epoch_gate_state_skips_pf_queries_when_all_gates_disabled():
    pf = FakeParticleFilter()

    state = compute_epoch_gate_state(pf, _config().parts())

    assert state.pf_estimate is None
    assert state.ess_ratio is None
    assert state.spread_m is None
    assert state.dd_pr_gate_scale == 1.0
    assert state.dd_cp_gate_scale == 1.0
    assert pf.estimate_calls == 0
    assert pf.ess_calls == 0
    assert pf.spread_calls == 0


def test_epoch_gate_state_computes_pr_scale_from_enabled_dd_pr_gates():
    pf = FakeParticleFilter()
    config = _config(
        dd_pseudorange=True,
        dd_pseudorange_gate_ess_min_scale=0.5,
        dd_pseudorange_gate_ess_max_scale=1.0,
        dd_pseudorange_gate_spread_min_scale=0.5,
        dd_pseudorange_gate_spread_max_scale=1.0,
        dd_pseudorange_gate_low_spread_m=2.0,
        dd_pseudorange_gate_high_spread_m=6.0,
    )

    state = compute_epoch_gate_state(pf, config.parts())

    np.testing.assert_allclose(state.pf_estimate, [1.0, 2.0, 3.0])
    assert state.ess_ratio == 0.2
    assert state.spread_m == 4.0
    assert state.dd_pr_gate_scale < 1.0
    assert state.dd_cp_gate_scale == 1.0
    assert pf.estimate_calls == 1
    assert pf.ess_calls == 1
    assert pf.spread_calls == 1


def test_epoch_gate_state_computes_cp_scale_from_enabled_dd_carrier_gates():
    pf = FakeParticleFilter()
    config = _config(
        mupf_dd=True,
        mupf_dd_gate_ess_min_scale=0.5,
        mupf_dd_gate_ess_max_scale=1.0,
        mupf_dd_gate_spread_min_scale=0.5,
        mupf_dd_gate_spread_max_scale=1.0,
        mupf_dd_gate_low_spread_m=2.0,
        mupf_dd_gate_high_spread_m=6.0,
    )

    state = compute_epoch_gate_state(pf, config.parts())

    assert state.dd_pr_gate_scale == 1.0
    assert state.dd_cp_gate_scale < 1.0
    assert state.ess_ratio == 0.2
    assert state.spread_m == 4.0


def test_epoch_gate_state_fetches_spread_for_widelane_min_spread_gate():
    pf = FakeParticleFilter()
    config = _config(
        widelane=True,
        widelane_gate_min_spread_m=3.0,
    )

    state = compute_epoch_gate_state(pf, config.parts())

    np.testing.assert_allclose(state.pf_estimate, [1.0, 2.0, 3.0])
    assert state.ess_ratio is None
    assert state.spread_m == 4.0
