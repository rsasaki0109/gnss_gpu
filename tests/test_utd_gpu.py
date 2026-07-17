"""Parity tests for the CUDA UTD (Kouyoumjian-Pathak) diffraction kernel vs the
CPU model in gnss_gpu.utd_diffraction.

Numerical parity with compute_utd_diffraction_paths is the acceptance bar:
same selected edges, excess_delay to 1e-9 m (relative), amplitude and
attenuation_db to 1e-6 (absolute). The tolerances are looser than the
knife-edge GPU parity tests (tests/test_diffraction_gpu.py) because the CPU
Fresnel integral (gnss_gpu.diffraction.fresnel_integral) is itself only a
finite-step trapezoidal approximation; see src/diffraction/utd.cu for the
device-side approximation notes.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from gnss_gpu.utd_diffraction import compute_utd_diffraction_paths

try:
    from gnss_gpu.utd_diffraction import compute_utd_diffraction_paths_gpu
    from gnss_gpu._gnss_gpu_diffraction import utd_diffraction_candidates  # noqa: F401

    _HAS_GPU = True
except Exception:
    _HAS_GPU = False

pytestmark = pytest.mark.skipif(
    not _HAS_GPU, reason="GPU UTD diffraction extension not available")

_KW = dict(max_paths=4, max_ray_edge_distance_m=40.0,
           max_excess_path_m=150.0, max_edge_range_m=400.0)


def _make_boxes(n_boxes, seed):
    """A handful of axis-aligned box "buildings" as wedge edges with two
    face directions each (the vertical edges of the box), similar in spirit
    to the wedge fixtures in tests/test_utd_diffraction.py and
    tests/test_urban_utd_diffraction.py.
    """
    rng = np.random.default_rng(seed)
    starts = []
    ends = []
    face_a = []
    face_b = []
    wedge_n = []

    for _ in range(n_boxes):
        cx, cy = rng.uniform(-150.0, 150.0, 2)
        hw = rng.uniform(5.0, 20.0)
        hd = rng.uniform(5.0, 20.0)
        h = rng.uniform(10.0, 60.0)

        corners = [
            (cx - hw, cy - hd), (cx + hw, cy - hd),
            (cx + hw, cy + hd), (cx - hw, cy + hd),
        ]
        normals = [
            (0.0, -1.0), (1.0, 0.0), (0.0, 1.0), (-1.0, 0.0),
        ]
        for i in range(4):
            x, y = corners[i]
            starts.append([x, y, -5.0])
            ends.append([x, y, -5.0 + h])
            fa = normals[i]
            fb = normals[(i - 1) % 4]
            face_a.append([fa[0], fa[1], 0.0])
            face_b.append([fb[0], fb[1], 0.0])
            wedge_n.append(1.5)  # convex 90-degree corner

    n = len(starts)
    start = np.asarray(starts, dtype=float)
    end = np.asarray(ends, dtype=float)
    return SimpleNamespace(
        start=start, end=end, midpoint=0.5 * (start + end), size=n,
        face_dir_a=np.asarray(face_a, dtype=float),
        face_dir_b=np.asarray(face_b, dtype=float),
        wedge_n=np.asarray(wedge_n, dtype=float),
    )


def _make_sats(rx, n, seed):
    rng = np.random.default_rng(seed)
    d = rng.uniform(-1.0, 1.0, (n, 3))
    d[:, 2] = np.abs(d[:, 2]) + 0.2  # keep satellites above the horizon
    return rx + d / np.linalg.norm(d, axis=1, keepdims=True) * 2.2e7


def _assert_parity(cpu, gpu, excess_abs=1e-6, amp_abs=1e-6):
    # excess_delay = d1 + d2 - direct_dist subtracts three ECEF-scale
    # distances (~1e7 m for GPS orbital ranges) that nearly cancel, so the
    # float64 noise floor for this quantity is ~2^-52 * 1e7 ~= 2e-9 m even
    # between two independently-correct implementations that don't share
    # the exact same operation order (e.g. numpy's dnrm2-backed
    # np.linalg.norm vs a direct sqrt(dot(v, v)) on the device). 1e-6 m
    # matches the tolerance already established for the same quantity in
    # tests/test_diffraction_gpu.py's knife-edge GPU parity tests.
    assert len(cpu) == len(gpu)
    for c, g in zip(cpu, gpu):
        assert len(c) == len(g)
        for pc, pg in zip(c, g):
            assert pc.edge_id == pg.edge_id
            assert pc.excess_delay == pytest.approx(pg.excess_delay, abs=excess_abs)
            assert pc.amplitude == pytest.approx(pg.amplitude, abs=amp_abs)
            assert pc.attenuation_db == pytest.approx(
                pg.attenuation_db, abs=amp_abs * 50.0)
            assert pc.beta0 == pytest.approx(pg.beta0, abs=1e-6)
            assert pc.phi == pytest.approx(pg.phi, abs=1e-6)
            assert pc.phi_p == pytest.approx(pg.phi_p, abs=1e-6)
            assert pc.wedge_n == pytest.approx(pg.wedge_n, abs=1e-9)
            np.testing.assert_allclose(
                pc.diffraction_point, pg.diffraction_point, atol=1e-6)


@pytest.mark.parametrize("mode", ["absorbing", "soft", "hard"])
def test_gpu_matches_cpu_small_scene(mode):
    rx = np.array([0.0, 0.0, 1.5])
    sats = _make_sats(rx, 12, seed=1)
    edges = _make_boxes(4, seed=1)
    cpu = compute_utd_diffraction_paths(rx, sats, edges, mode=mode, **_KW)
    gpu = compute_utd_diffraction_paths_gpu(rx, sats, edges, mode=mode, **_KW)
    _assert_parity(cpu, gpu)


def test_gpu_matches_cpu_medium_scene():
    rx = np.array([10.0, -5.0, 1.5])
    sats = _make_sats(rx, 20, seed=7)
    edges = _make_boxes(25, seed=7)
    cpu = compute_utd_diffraction_paths(rx, sats, edges, mode="absorbing", **_KW)
    gpu = compute_utd_diffraction_paths_gpu(rx, sats, edges, mode="absorbing", **_KW)
    _assert_parity(cpu, gpu)


def test_gpu_matches_cpu_half_plane_shadow_sweep():
    # Reuse the canonical half-plane fixture from test_utd_diffraction.py
    # across the full lit -> deep-shadow sweep.
    start = np.array([[0.0, 0.0, -100.0]])
    end = np.array([[0.0, 0.0, 100.0]])
    edges = SimpleNamespace(
        start=start, end=end, midpoint=0.5 * (start + end), size=1,
        face_dir_a=np.array([[0.0, -1.0, 0.0]]),
        face_dir_b=np.array([[0.0, -1.0, 0.0]]),
        wedge_n=np.array([2.0]))
    sat = np.array([-1.0e7, 0.0, 0.0])
    kw = dict(max_ray_edge_distance_m=30.0, max_excess_path_m=200.0, max_paths=1)
    for yr in (-12.0, -6.0, -3.0, -1.0, -0.1, 0.05, 1.0, 3.0, 6.0, 12.0):
        rx = np.array([50.0, yr, 0.0])
        cpu = compute_utd_diffraction_paths(rx, sat, edges, mode="absorbing", **kw)
        gpu = compute_utd_diffraction_paths_gpu(rx, sat, edges, mode="absorbing", **kw)
        _assert_parity(cpu, gpu)


def test_gpu_matches_cpu_wedge_variants():
    start = np.array([[0.0, 0.0, -100.0]])
    end = np.array([[0.0, 0.0, 100.0]])
    sat = np.array([-1.0e7, 0.0, 0.0])
    rx = np.array([50.0, 6.0, 0.0])
    kw = dict(max_ray_edge_distance_m=30.0, max_excess_path_m=200.0, max_paths=1)
    for n_val in (1.2, 1.5, 1.8, 2.0, 2.5, 3.0):
        edges = SimpleNamespace(
            start=start, end=end, midpoint=0.5 * (start + end), size=1,
            face_dir_a=np.array([[0.0, -1.0, 0.0]]),
            face_dir_b=np.array([[-1.0, 0.0, 0.0]]),
            wedge_n=np.array([n_val]))
        cpu = compute_utd_diffraction_paths(rx, sat, edges, mode="hard", **kw)
        gpu = compute_utd_diffraction_paths_gpu(rx, sat, edges, mode="hard", **kw)
        _assert_parity(cpu, gpu)


def test_gpu_empty_inputs():
    rx = np.array([0.0, 0.0, 0.0])
    sats = _make_sats(rx, 3, seed=2)
    edges = _make_boxes(2, seed=2)
    empty = SimpleNamespace(
        start=np.empty((0, 3)), end=np.empty((0, 3)), midpoint=np.empty((0, 3)),
        face_dir_a=np.empty((0, 3)), face_dir_b=np.empty((0, 3)),
        wedge_n=np.empty(0), size=0)

    assert compute_utd_diffraction_paths_gpu(rx, np.empty((0, 3)), edges) == []
    assert compute_utd_diffraction_paths_gpu(rx, sats, empty) == [[], [], []]
    assert compute_utd_diffraction_paths_gpu(rx, sats, edges, max_paths=0) == [
        [] for _ in range(3)]


def test_gpu_respects_max_paths_and_sorted():
    rx = np.array([0.0, 0.0, 1.5])
    sats = _make_sats(rx, 6, seed=5)
    edges = _make_boxes(30, seed=9)
    gpu = compute_utd_diffraction_paths_gpu(
        rx, sats, edges, **{**_KW, "max_paths": 2})
    assert all(len(p) <= 2 for p in gpu)
    for paths in gpu:
        amps = [p.amplitude for p in paths]
        assert amps == sorted(amps, reverse=True)


def test_gpu_default_wedge_n_matches_cpu():
    # edges.wedge_n omitted entirely from the array (size 0) -> default 2.0
    # for every edge on both CPU and GPU (utd_diffraction._wedge_n_at).
    rx = np.array([0.0, 0.0, 1.5])
    sats = _make_sats(rx, 5, seed=11)
    edges = _make_boxes(6, seed=11)
    edges.wedge_n = np.empty(0)
    cpu = compute_utd_diffraction_paths(rx, sats, edges, mode="absorbing", **_KW)
    gpu = compute_utd_diffraction_paths_gpu(rx, sats, edges, mode="absorbing", **_KW)
    _assert_parity(cpu, gpu)


def test_gpu_invalid_mode_raises():
    edges = _make_boxes(1, seed=3)
    rx = np.array([0.0, 0.0, 1.5])
    sats = _make_sats(rx, 2, seed=3)
    with pytest.raises(ValueError):
        compute_utd_diffraction_paths_gpu(rx, sats, edges, mode="bogus")
