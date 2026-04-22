from types import SimpleNamespace

import numpy as np

from gnss_gpu.pf_smoother_epoch_state import create_epoch_forward_state
from gnss_gpu.pf_smoother_runtime import ForwardRunBuffers
from gnss_gpu.smoother_epoch_store import (
    append_smoother_epoch_store,
    build_smoother_epoch_store_inputs,
)


def test_build_smoother_epoch_store_inputs_keeps_valid_epoch_metadata():
    spp_pos = np.array([1.0e6, 2.0e6, 3.0e6])
    dd_pr = SimpleNamespace(n_dd=3)
    dd_carrier = SimpleNamespace(n_dd=4)
    anchor_update = {"sat_ecef": np.zeros((4, 3))}
    afv = {"sat_ecef": np.zeros((4, 3))}
    doppler = {"sat_ecef": np.zeros((4, 3))}

    store_inputs = build_smoother_epoch_store_inputs(
        spp_pos=spp_pos,
        position_update_sigma=2.0,
        dd_pseudorange_result=dd_pr,
        dd_pseudorange_sigma=0.5,
        used_widelane_epoch=True,
        dd_carrier_result=dd_carrier,
        dd_carrier_sigma_cycles=0.12,
        anchor_attempt=SimpleNamespace(update=anchor_update),
        carrier_anchor_sigma_m=0.25,
        fallback_attempt=SimpleNamespace(afv=afv, sigma_cycles=0.08),
        carrier_afv_wavelength_m=0.19,
        doppler_update=doppler,
        doppler_sigma_mps=0.4,
        doppler_velocity_update_gain=0.3,
        doppler_max_velocity_update_mps=8.0,
    )

    kwargs = store_inputs.as_store_kwargs()
    assert kwargs["spp_ref"] is not None
    np.testing.assert_allclose(kwargs["spp_ref"], spp_pos)
    assert kwargs["dd_pseudorange"] is dd_pr
    assert kwargs["dd_pseudorange_sigma"] == 0.5
    assert kwargs["dd_pseudorange_source"] == "widelane"
    assert kwargs["dd_carrier"] is dd_carrier
    assert kwargs["dd_carrier_sigma"] == 0.12
    assert kwargs["carrier_anchor_pseudorange"] is anchor_update
    assert kwargs["carrier_anchor_sigma"] == 0.25
    assert kwargs["carrier_afv"] is afv
    assert kwargs["carrier_afv_sigma"] == 0.08
    assert kwargs["carrier_afv_wavelength"] == 0.19
    assert kwargs["doppler_update"] is doppler
    assert kwargs["doppler_sigma_mps"] == 0.4
    assert kwargs["doppler_velocity_update_gain"] == 0.3
    assert kwargs["doppler_max_velocity_update_mps"] == 8.0


def test_build_smoother_epoch_store_inputs_drops_ineligible_metadata():
    store_inputs = build_smoother_epoch_store_inputs(
        spp_pos=np.array([1.0, 2.0, 3.0]),
        position_update_sigma=2.0,
        dd_pseudorange_result=SimpleNamespace(n_dd=2),
        dd_pseudorange_sigma=0.5,
        used_widelane_epoch=False,
        dd_carrier_result=SimpleNamespace(n_dd=2),
        dd_carrier_sigma_cycles=0.12,
        anchor_attempt=SimpleNamespace(update=None),
        carrier_anchor_sigma_m=0.25,
        fallback_attempt=SimpleNamespace(afv=None, sigma_cycles=0.08),
        carrier_afv_wavelength_m=0.19,
        doppler_update=None,
        doppler_sigma_mps=None,
        doppler_velocity_update_gain=0.3,
        doppler_max_velocity_update_mps=8.0,
    )

    kwargs = store_inputs.as_store_kwargs()
    assert kwargs["spp_ref"] is None
    assert kwargs["dd_pseudorange_sigma"] is None
    assert kwargs["dd_pseudorange_source"] is None
    assert kwargs["dd_carrier_sigma"] is None
    assert kwargs["carrier_anchor_sigma"] is None
    assert kwargs["carrier_afv_wavelength"] is None
    assert kwargs["doppler_sigma_mps"] is None
    assert kwargs["doppler_velocity_update_gain"] is None
    assert kwargs["doppler_max_velocity_update_mps"] is None


def test_append_smoother_epoch_store_updates_pf_and_fgo_buffers():
    class FakeParticleFilter:
        def __init__(self):
            self.calls = []

        def store_epoch(self, *args, **kwargs):
            self.calls.append((args, kwargs))

        def estimate(self):
            return np.array([0.0, 0.0, 0.0, 1.0])

    pf = FakeParticleFilter()
    buffers = ForwardRunBuffers()
    state = create_epoch_forward_state(0.5)
    state.velocity = np.array([1.0, 2.0, 3.0])
    sat_ecef = np.array(
        [
            [2000.0, 0.0, 0.0],
            [0.0, 2100.0, 0.0],
            [0.0, 0.0, 2200.0],
            [2300.0, 2300.0, 0.0],
        ],
        dtype=np.float64,
    )
    pseudoranges = np.array([2005.0, 2105.0, 2205.0, 3258.0], dtype=np.float64)
    weights = np.ones(4, dtype=np.float64)

    store_inputs = append_smoother_epoch_store(
        pf,
        buffers,
        sat_ecef=sat_ecef,
        pseudoranges=pseudoranges,
        weights=weights,
        spp_pos=np.array([1.0, 2.0, 3.0]),
        epoch_state=state,
        dt=0.5,
        position_update_sigma=None,
        carrier_anchor_sigma_m=0.25,
        carrier_afv_wavelength_m=0.19,
        doppler_velocity_update_gain=0.3,
        doppler_max_velocity_update_mps=8.0,
        need_tdcp_motion=True,
    )

    assert store_inputs.dd_pseudorange_sigma is None
    assert len(pf.calls) == 1
    args, kwargs = pf.calls[0]
    assert args[0] is sat_ecef
    assert args[1] is pseudoranges
    assert args[2] is weights
    np.testing.assert_allclose(args[3], [1.0, 2.0, 3.0])
    assert args[4] == 0.5
    assert kwargs["spp_ref"] is None
    assert buffers.n_stored == 1
    assert buffers.stored_dd_carrier == [None]
    assert buffers.stored_dd_pseudorange == [None]
    assert buffers.stored_undiff_pr[0] is not None
