"""Tests for reflection+diffraction composite propagation paths.

Physics is validated through geometric identities that must hold for any valid
composite path, regardless of the specific scene:

  * total 3-segment path length == d1 + d2 (image-method identity), so
    excess_delay == (|rx-P1|+|P1-P2|+|P2-sat|) - |sat-rx|
  * the reflection point lies exactly on the reflecting triangle plane
  * Snell's law of reflection: angle of incidence == angle of reflection
  * the diffraction point lies exactly on the edge segment
  * amplitude == knife_edge_amplitude(fresnel_v)
"""

from types import SimpleNamespace

import numpy as np
import pytest

from gnss_gpu.reflection_diffraction import (
    ReflectionDiffractionPath,
    compute_reflection_diffraction_paths,
    knife_edge_amplitude,
)

# Big vertical wall in the plane x = 5 (outward +x normal), spanning y, z.
_WALL = np.array(
    [[5.0, -50.0, -50.0], [5.0, 50.0, -50.0], [5.0, 0.0, 50.0]], dtype=float)
_TRIS = _WALL[None, :, :]
_RX = np.array([0.0, 0.0, 0.0])
_SAT = np.array([-20.0, 60.0, 0.0])


def _edges(start, end):
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    return SimpleNamespace(
        start=start, end=end, midpoint=0.5 * (start + end), size=start.shape[0])


def _wall_normal():
    n = np.cross(_WALL[1] - _WALL[0], _WALL[2] - _WALL[0])
    return n / np.linalg.norm(n)


def _assert_path_identities(path, rx, sat, edges):
    n = _wall_normal()
    P1, P2 = path.reflection_point, path.diffraction_point
    seg = [rx, P1, P2, sat] if path.order == "RD" else [rx, P2, P1, sat]
    length = sum(np.linalg.norm(seg[i + 1] - seg[i]) for i in range(3))
    excess_true = length - np.linalg.norm(sat - rx)
    assert path.excess_delay == pytest.approx(excess_true, abs=1e-6)

    # Reflection point on the wall plane.
    assert float(np.dot(P1 - _WALL[0], n)) == pytest.approx(0.0, abs=1e-9)

    # Snell reflection at P1.
    if path.order == "RD":
        inc, out = P1 - rx, P2 - P1
    else:
        inc, out = P1 - P2, sat - P1
    inc = inc / np.linalg.norm(inc)
    out = out / np.linalg.norm(out)
    ang_in = np.arccos(abs(np.dot(inc, n)))
    ang_out = np.arccos(abs(np.dot(out, n)))
    assert ang_in == pytest.approx(ang_out, abs=1e-9)
    assert path.incidence_angle == pytest.approx(ang_in, abs=1e-9)

    # Diffraction point on the edge segment.
    e0 = edges.start[path.edge_id]
    e1 = edges.end[path.edge_id]
    t = np.dot(P2 - e0, e1 - e0) / np.dot(e1 - e0, e1 - e0)
    assert np.linalg.norm((e0 + t * (e1 - e0)) - P2) == pytest.approx(0.0, abs=1e-9)

    # Amplitude consistency.
    assert path.amplitude == pytest.approx(
        knife_edge_amplitude(path.fresnel_v), abs=1e-12)
    assert 0.0 <= path.amplitude <= 1.0
    assert path.attenuation_db >= 0.0


def test_rd_path_identities():
    # Vertical edge straddling the rx-image ray; yields a reflect-then-diffract path.
    edges = _edges([[0.0, 20.0, -3.0]], [[0.0, 20.0, 3.0]])
    res = compute_reflection_diffraction_paths(_TRIS, edges, _RX, _SAT, max_paths=8)
    assert len(res) == 1
    rd = [p for p in res[0] if p.order == "RD"]
    assert len(rd) == 1
    p = rd[0]
    # Hand-derived geometry: reflection at [5,10,0], diffraction at [0,20,0].
    np.testing.assert_allclose(p.reflection_point, [5.0, 10.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(p.diffraction_point, [0.0, 20.0, 0.0], atol=1e-9)
    _assert_path_identities(p, _RX, _SAT, edges)


def test_dr_path_identities():
    # Edge near the rx -> sat-image ray on the near side of the wall.
    edges = _edges([[3.0, 6.0, -3.0]], [[3.0, 6.0, 3.0]])
    res = compute_reflection_diffraction_paths(
        _TRIS, edges, _RX, _SAT, max_paths=8, orders=("DR",))
    dr = [p for p in res[0] if p.order == "DR"]
    assert len(dr) == 1
    p = dr[0]
    np.testing.assert_allclose(p.reflection_point, [5.0, 10.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(p.diffraction_point, [3.0, 6.0, 0.0], atol=1e-9)
    _assert_path_identities(p, _RX, _SAT, edges)


def test_orders_filter():
    edges = _edges(
        [[0.0, 20.0, -3.0], [3.0, 6.0, -3.0]],
        [[0.0, 20.0, 3.0], [3.0, 6.0, 3.0]])
    rd_only = compute_reflection_diffraction_paths(
        _TRIS, edges, _RX, _SAT, max_paths=8, orders=("RD",))
    dr_only = compute_reflection_diffraction_paths(
        _TRIS, edges, _RX, _SAT, max_paths=8, orders=("DR",))
    assert all(p.order == "RD" for p in rd_only[0])
    assert all(p.order == "DR" for p in dr_only[0])
    assert rd_only[0] and dr_only[0]


def test_max_paths_and_sorting():
    edges = _edges(
        [[0.0, 20.0, -3.0], [3.0, 6.0, -3.0]],
        [[0.0, 20.0, 3.0], [3.0, 6.0, 3.0]])
    res = compute_reflection_diffraction_paths(
        _TRIS, edges, _RX, _SAT, max_paths=1)
    assert len(res[0]) == 1
    full = compute_reflection_diffraction_paths(
        _TRIS, edges, _RX, _SAT, max_paths=8)
    amps = [p.amplitude for p in full[0]]
    assert amps == sorted(amps, reverse=True)


def test_multi_satellite_shape():
    edges = _edges([[0.0, 20.0, -3.0]], [[0.0, 20.0, 3.0]])
    sats = np.vstack([_SAT, [1.0e7, 2.0e7, 3.0e7]])
    res = compute_reflection_diffraction_paths(_TRIS, edges, _RX, sats, max_paths=4)
    assert len(res) == 2
    assert all(isinstance(p, ReflectionDiffractionPath) for p in res[0])


def test_empty_and_degenerate_inputs():
    edges = _edges([[0.0, 20.0, -3.0]], [[0.0, 20.0, 3.0]])
    # No satellites.
    assert compute_reflection_diffraction_paths(
        _TRIS, edges, _RX, np.empty((0, 3))) == []
    # max_paths <= 0.
    assert compute_reflection_diffraction_paths(
        _TRIS, edges, _RX, _SAT, max_paths=0) == [[]]
    # No triangles / no edges.
    assert compute_reflection_diffraction_paths(None, edges, _RX, _SAT) == [[]]
    assert compute_reflection_diffraction_paths(_TRIS, None, _RX, _SAT) == [[]]
    empty_edges = _edges(np.empty((0, 3)), np.empty((0, 3)))
    assert compute_reflection_diffraction_paths(
        _TRIS, empty_edges, _RX, _SAT) == [[]]
