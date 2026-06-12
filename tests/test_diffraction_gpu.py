"""Parity tests for the CUDA knife-edge diffraction kernel vs the CPU model."""

from types import SimpleNamespace

import numpy as np
import pytest

from gnss_gpu.diffraction import compute_diffraction_paths

try:
    from gnss_gpu.diffraction import compute_diffraction_paths_gpu
    from gnss_gpu._gnss_gpu_diffraction import diffraction_candidates  # noqa: F401

    _HAS_GPU = True
except Exception:
    _HAS_GPU = False

pytestmark = pytest.mark.skipif(
    not _HAS_GPU, reason="GPU diffraction extension not available")

_KW = dict(max_paths=4, max_ray_edge_distance_m=40.0,
           max_excess_path_m=150.0, max_edge_range_m=400.0)


def _make_edges(n, seed):
    rng = np.random.default_rng(seed)
    s = rng.uniform(-150.0, 150.0, (n, 3))
    e = s + rng.uniform(-30.0, 30.0, (n, 3))
    return SimpleNamespace(
        start=s, end=e, midpoint=0.5 * (s + e),
        length_m=np.linalg.norm(e - s, axis=1),
        dihedral_deg=np.full(n, 90.0), is_boundary=np.zeros(n, bool), size=n)


def _make_sats(rx, n, seed):
    rng = np.random.default_rng(seed)
    d = rng.uniform(-1.0, 1.0, (n, 3))
    return rx + d / np.linalg.norm(d, axis=1, keepdims=True) * 2.2e7


def _assert_parity(cpu, gpu):
    assert len(cpu) == len(gpu)
    for c, g in zip(cpu, gpu):
        assert len(c) == len(g)
        for pc, pg in zip(c, g):
            assert pc.edge_id == pg.edge_id
            assert pc.excess_delay == pytest.approx(pg.excess_delay, abs=1e-6)
            assert pc.amplitude == pytest.approx(pg.amplitude, abs=1e-9)
            assert pc.fresnel_v == pytest.approx(pg.fresnel_v, abs=1e-9)
            np.testing.assert_allclose(
                pc.diffraction_point, pg.diffraction_point, atol=1e-6)


def test_gpu_matches_cpu_small_and_medium():
    rx = np.array([0.0, 0.0, 0.0])
    sats = _make_sats(rx, 8, seed=1)
    for n_edge in (50, 500, 2000):
        edges = _make_edges(n_edge, seed=n_edge)
        cpu = compute_diffraction_paths(rx, sats, edges, **_KW)
        gpu = compute_diffraction_paths_gpu(rx, sats, edges, **_KW)
        _assert_parity(cpu, gpu)


def test_gpu_empty_inputs():
    rx = np.array([0.0, 0.0, 0.0])
    sats = _make_sats(rx, 3, seed=2)
    edges = _make_edges(10, seed=2)
    empty = SimpleNamespace(
        start=np.empty((0, 3)), end=np.empty((0, 3)), midpoint=np.empty((0, 3)),
        length_m=np.empty(0), dihedral_deg=np.empty(0),
        is_boundary=np.empty(0, bool), size=0)

    assert compute_diffraction_paths_gpu(rx, np.empty((0, 3)), edges) == []
    assert compute_diffraction_paths_gpu(rx, sats, empty) == [[], [], []]
    assert compute_diffraction_paths_gpu(rx, sats, edges, max_paths=0) == [[], [], []]


def test_gpu_respects_max_paths():
    rx = np.array([0.0, 0.0, 0.0])
    sats = _make_sats(rx, 4, seed=5)
    edges = _make_edges(3000, seed=9)
    gpu = compute_diffraction_paths_gpu(rx, sats, edges, **{**_KW, "max_paths": 2})
    assert all(len(p) <= 2 for p in gpu)
    # Each satellite's paths are sorted strongest-first.
    for paths in gpu:
        amps = [p.amplitude for p in paths]
        assert amps == sorted(amps, reverse=True)
