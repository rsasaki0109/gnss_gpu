"""Exact atmospheric batching, original scalar callbacks and complete IQ."""

import copy
import math
import struct
import types

import numpy as np
import pytest

from gnss_gpu import atmosphere
from gnss_gpu.urban_signal_sim import UrbanSignalSimulator


class _ScalarAtmosphere(atmosphere.AtmosphereCorrection):
    """A subclass retains the original per-satellite callback route."""


class _CaptureIQ:
    sampling_freq = 2.6e6

    def generate_epoch(self, channels, n_samples=None):
        return np.zeros(2*int(n_samples), dtype=np.float32)


def _assert_exact(a, b):
    assert type(a) is type(b)
    if isinstance(a, np.ndarray):
        assert a.dtype == b.dtype and a.shape == b.shape
        assert a.tobytes() == b.tobytes()
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            _assert_exact(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b, strict=True):
            _assert_exact(x, y)
    elif isinstance(a, (float, np.floating)):
        assert struct.pack('d', a) == struct.pack('d', b)
    else:
        assert a == b


def _rays(count, mode='mixed', rx=None):
    rx = np.array([6378137., 0., 0.]) if rx is None else np.asarray(rx, dtype=np.float64)
    up = rx/np.linalg.norm(rx)
    east = np.cross([0., 0., 1.], up)
    if np.linalg.norm(east) < 1e-10:
        east = np.array([0., 1., 0.])
    east /= np.linalg.norm(east)
    limits = dict(all=(10., 80.), mixed=(-30., 80.), none=(-80., -5.))[mode]
    el = np.radians(np.linspace(*limits, count))
    return rx, rx+20200e3*(np.sin(el)[:, None]*up+np.cos(el)[:, None]*east)


def _metadata(atmo, rx, sat, **kwargs):
    sim = UrbanSignalSimulator(elevation_mask_deg=0.)
    sim.sim = _CaptureIQ()
    return sim.compute_epoch(rx, sat, atmo_correction=atmo, n_samples=32, **kwargs)


@pytest.mark.parametrize('count', [1, 12, 32])
@pytest.mark.parametrize('visibility', ['all', 'mixed', 'none'])
def test_builtin_batches_only_visible_satellites(monkeypatch, count, visibility):
    calls = []

    def correction(lla, el):
        calls.append((np.array(lla, copy=True), np.array(el, copy=True)))
        return 3.+np.asarray(el)*.5

    monkeypatch.setattr(atmosphere, '_HAS_GPU', True)
    monkeypatch.setattr(atmosphere, 'tropo_correction_batch', correction, raising=False)
    rx, sat = _rays(count, visibility)
    actual = _metadata(atmosphere.AtmosphereCorrection(), rx, sat)
    batch_calls = calls.copy()
    calls.clear()
    expected = _metadata(_ScalarAtmosphere(), rx, sat)
    _assert_exact(actual, expected)
    visible = actual['visible']
    assert len(batch_calls) == int(visible.any())
    assert len(calls) == np.count_nonzero(visible)
    if visible.any():
        np.testing.assert_array_equal(batch_calls[0][1], actual['elevations'][visible])
        assert all(row[1].shape == (1,) for row in calls)


@pytest.mark.parametrize('mode', ['cpu', 'subclass', 'instance', 'class'])
@pytest.mark.parametrize('visibility', ['mixed', 'none'])
def test_overrides_and_cpu_keep_scalar_calls(monkeypatch, mode, visibility):
    calls = []

    def correction(*args):
        calls.append(np.array(args[-1], copy=True))
        return 3.+np.asarray(args[-1])*.5

    monkeypatch.setattr(atmosphere, '_HAS_GPU', mode != 'cpu')
    monkeypatch.setattr(atmosphere, 'tropo_correction_batch', correction, raising=False)
    monkeypatch.setattr(atmosphere, '_tropo_saastamoinen_cpu', correction)
    original = atmosphere.AtmosphereCorrection.tropo

    def wrapped(self, *args):
        return original(self, *args)

    if mode == 'class':
        monkeypatch.setattr(atmosphere.AtmosphereCorrection, 'tropo', wrapped)
    atmo = _ScalarAtmosphere() if mode == 'subclass' else atmosphere.AtmosphereCorrection()
    if mode == 'instance':
        atmo.tropo = types.MethodType(wrapped, atmo)
    rx, sat = _rays(12, visibility)
    result = _metadata(atmo, rx, sat)
    assert len(calls) == np.count_nonzero(result['visible'])
    assert all(el.shape == (1,) for el in calls)


def test_custom_ionosphere_callback_order_and_parameters():
    calls = []

    class Custom:
        def tropo(self, lla, el):
            calls.append(('tropo', float(el), lla.copy()))
            return 2.

        def iono(self, lla, az, el, gps_time, **params):
            calls.append(('iono', float(el), gps_time, params, lla.copy()))
            return 1.

    rx, sat = _rays(12)
    params = dict(alpha=[1.]*4, beta=[2.]*4)
    result = _metadata(Custom(), rx, sat, gps_time=123., iono_params=params)
    expected_el = result['elevations'][result['visible']]
    assert [c[0] for c in calls] == ['tropo', 'iono']*len(expected_el)
    for i, el in enumerate(expected_el):
        assert calls[2*i][1] == calls[2*i+1][1] == el
        assert calls[2*i+1][2:4] == (123., params)


@pytest.mark.cuda
@pytest.mark.gpu
@pytest.mark.parametrize('gpu_atmosphere', [False, True])
@pytest.mark.parametrize('count', [1, 12, 32])
@pytest.mark.parametrize('rx', [[6378137., 0., 0.], [6381137., 0., 0.], [-3954922., 3354936., 3699777.], [0., 0., 6356752.314245]])
def test_real_native_iq_and_all_metadata_match_scalar(monkeypatch, gpu_atmosphere, count, rx):
    if not atmosphere._HAS_GPU:
        pytest.skip('Native atmosphere build required for this parity check')
    pytest.importorskip('gnss_gpu._gnss_gpu_signal_sim')
    monkeypatch.setattr(atmosphere, '_HAS_GPU', gpu_atmosphere)
    rx, sat = _rays(count, 'all', rx)
    results = []
    sims = []
    try:
        for atmo in (atmosphere.AtmosphereCorrection(), _ScalarAtmosphere()):
            sim = UrbanSignalSimulator(elevation_mask_deg=-90.)
            sim.sim.noise_seed = 37
            sims.append(sim)
            results.append(sim.compute_epoch(rx, sat, sat_clk=np.linspace(-1e-7, 1e-7, count),
                sat_vel=np.ones((count, 3))*1234., rx_vel=[3., 2., 1.], n_samples=2600, atmo_correction=atmo))
        _assert_exact(*results)
    finally:
        for sim in sims:
            sim.sim.release_gpu()


@pytest.mark.cuda
@pytest.mark.gpu
@pytest.mark.parametrize('batch_size', [1, 4])
@pytest.mark.parametrize('count', [1, 12])
def test_trajectory_iq_seed_and_epoch_boundaries_match_scalar(batch_size, count):
    if not atmosphere._HAS_GPU:
        pytest.skip('Native atmosphere build required')
    pytest.importorskip('gnss_gpu._gnss_gpu_signal_sim')
    results, sims = [], []
    try:
        rx, sat = _rays(count, 'all')
        positions = np.repeat(rx[None], 9, axis=0)
        positions[:, 0] += np.arange(9)
        satellites = np.repeat(sat[None], 9, axis=0)
        for atmo in (atmosphere.AtmosphereCorrection(), _ScalarAtmosphere()):
            sim = UrbanSignalSimulator(elevation_mask_deg=0.)
            sim.sim.noise_seed = 37
            sims.append(sim)
            results.append(list(sim.simulate_trajectory(positions, satellites, n_samples=31,
                atmo_correction=atmo, batch_size=batch_size)))
        _assert_exact(*results)
    finally:
        for sim in sims:
            sim.sim.release_gpu()


@pytest.mark.cuda
@pytest.mark.gpu
def test_elevation_clamp_and_returned_storage_match_original_scalar():
    if not atmosphere._HAS_GPU:
        pytest.skip('Native atmosphere build required')
    rx = np.array([6378137., 0., 0.])
    el = np.radians([-30., 0., math.nextafter(2., 0.), 2., math.nextafter(2., math.inf), 10., 90.])
    sat = rx+20200e3*np.stack((np.sin(el), np.cos(el), np.zeros(len(el))), axis=1)
    sim = UrbanSignalSimulator(elevation_mask_deg=-90.)
    sim.sim = _CaptureIQ()
    first = sim.compute_epoch(rx, sat, n_samples=32, atmo_correction=atmosphere.AtmosphereCorrection())
    snapshot = copy.deepcopy(first)
    # Match the same below-horizon mask in the original scalar route.
    reference = UrbanSignalSimulator(elevation_mask_deg=-90.)
    reference.sim = _CaptureIQ()
    expected = reference.compute_epoch(rx, sat, n_samples=32, atmo_correction=_ScalarAtmosphere())
    _assert_exact(first, expected)
    sim.compute_epoch(rx+1., sat, n_samples=32, atmo_correction=atmosphere.AtmosphereCorrection())
    _assert_exact(first, snapshot)
