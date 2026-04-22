from types import SimpleNamespace

import numpy as np

import gnss_gpu.particle_filter_device as pfd


class _FakeBackwardPF:
    instances = []

    def __init__(self, **_kwargs):
        self.calls = []
        self._estimate = np.array([10.0, 20.0, 30.0, 0.0], dtype=np.float64)
        self.__class__.instances.append(self)

    def initialize(self, init_pos, **_kwargs):
        self.calls.append(("initialize", np.asarray(init_pos, dtype=np.float64).copy()))

    def predict(self, **_kwargs):
        self.calls.append(("predict", None))

    def update_dd_pseudorange(self, dd_result, sigma_pr=0.75, resample=True):
        self.calls.append(("dd_pr", int(dd_result.n_dd), float(sigma_pr), bool(resample)))

    def correct_clock_bias(self, sat_ecef, pseudoranges):
        self.calls.append(("clock", len(pseudoranges)))

    def update(self, sat_ecef, pseudoranges, weights=None, sigma_pr=None):
        self.calls.append(("pr", len(pseudoranges), sigma_pr))

    def estimate(self):
        return self._estimate


def _smooth_shell(source: str | None):
    sat_ecef = np.array(
        [
            [20_000_000.0, 0.0, 0.0],
            [0.0, 20_000_000.0, 0.0],
            [0.0, 0.0, 20_000_000.0],
            [20_000_000.0, 20_000_000.0, 0.0],
        ],
        dtype=np.float64,
    )
    epoch = {
        "estimate": np.array([1.0, 2.0, 3.0], dtype=np.float64),
        "sat_ecef": sat_ecef,
        "pseudoranges": np.full(4, 20_000_000.0, dtype=np.float64),
        "weights": np.ones(4, dtype=np.float64),
        "velocity": np.zeros(3, dtype=np.float64),
        "dt": 0.1,
        "spp_ref": None,
        "dd_pseudorange": {"n_dd": 3},
        "dd_pseudorange_sigma": 0.1,
        "dd_pseudorange_source": source,
        "dd_carrier": None,
        "dd_carrier_sigma": None,
        "carrier_anchor_pseudorange": None,
        "carrier_anchor_sigma": None,
        "carrier_afv": None,
        "carrier_afv_sigma": None,
        "carrier_afv_wavelength": None,
        "doppler_update": None,
        "doppler_sigma_mps": None,
        "doppler_velocity_update_gain": None,
        "doppler_max_velocity_update_mps": None,
    }
    return SimpleNamespace(
        _smooth_enabled=True,
        _smooth_epochs=[epoch],
        n_particles=16,
        sigma_pos=1.0,
        sigma_cb=300.0,
        sigma_pr=3.0,
        nu=0.0,
        resampling="systematic",
        ess_threshold=0.5,
        seed=7,
        per_particle_nlos_gate=False,
        per_particle_nlos_dd_pr_threshold_m=10.0,
        per_particle_nlos_dd_carrier_threshold_cycles=0.5,
        per_particle_nlos_undiff_pr_threshold_m=30.0,
        per_particle_huber=False,
        per_particle_huber_dd_pr_k=1.5,
        per_particle_huber_dd_carrier_k=1.5,
        per_particle_huber_undiff_pr_k=1.5,
        sigma_vel=0.0,
        velocity_guide_alpha=1.0,
        rbpf_velocity_kf=False,
        velocity_process_noise=0.0,
        _velocity_init_sigma=0.0,
    )


def test_smooth_can_skip_widelane_dd_pseudorange_replay(monkeypatch):
    original_pf = pfd.ParticleFilterDevice
    monkeypatch.setattr(pfd, "ParticleFilterDevice", _FakeBackwardPF)

    _FakeBackwardPF.instances = []
    original_pf.smooth(
        _smooth_shell("widelane"),
        position_update_sigma=None,
        skip_widelane_dd_pseudorange=False,
    )
    assert any(call[0] == "dd_pr" for call in _FakeBackwardPF.instances[-1].calls)

    _FakeBackwardPF.instances = []
    original_pf.smooth(
        _smooth_shell("widelane"),
        position_update_sigma=None,
        skip_widelane_dd_pseudorange=True,
    )
    calls = [call[0] for call in _FakeBackwardPF.instances[-1].calls]
    assert "dd_pr" not in calls
    assert "clock" in calls
    assert "pr" in calls


def test_smooth_skip_widelane_keeps_regular_dd_pseudorange_replay(monkeypatch):
    original_pf = pfd.ParticleFilterDevice
    monkeypatch.setattr(pfd, "ParticleFilterDevice", _FakeBackwardPF)

    _FakeBackwardPF.instances = []
    original_pf.smooth(
        _smooth_shell("dd_pseudorange"),
        position_update_sigma=None,
        skip_widelane_dd_pseudorange=True,
    )
    assert any(call[0] == "dd_pr" for call in _FakeBackwardPF.instances[-1].calls)
