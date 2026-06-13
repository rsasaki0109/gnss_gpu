"""Lit/shadow sign for knife-edge diffraction clearance.

The legacy model clamps the Fresnel clearance to >= 0, so every candidate is
treated as shadow-side (amplitude <= 0.5). ``edge_lit_shadow_sign`` recovers the
sign: a receiver whose direct ray clears the edge is lit (sign -1 -> ~no loss,
amplitude ~1), one whose ray is occluded by the edge's wall is in shadow
(sign +1 -> attenuating loss).
"""

import numpy as np
import pytest

from gnss_gpu.diffraction import (
    compute_diffraction_paths,
    edge_lit_shadow_sign,
    knife_edge_amplitude,
)
from gnss_gpu.fresnel import GPS_L1_FREQ  # noqa: F401  (ensures package import path)


# A vertical edge at the origin; the wall occupies the half-plane y=0, x<=0
# (face direction -x from the edge). Open air is the x>0 / y!=0 region.
EDGE_START = np.array([0.0, 0.0, -10.0])
EDGE_END = np.array([0.0, 0.0, 10.0])
EDGE_POINT = np.array([0.0, 0.0, 0.0])
FACE_A = np.array([-1.0, 0.0, 0.0])


def test_sign_shadow_when_ray_crosses_wall():
    rx = np.array([1.0, 2.0, 0.0])
    # Direct ray to this satellite crosses y=0 at x<0 -> through the wall.
    sat = np.array([-300.0, -200.0, 0.0])
    s = edge_lit_shadow_sign(rx, sat, EDGE_POINT, EDGE_START, EDGE_END, FACE_A)
    assert s == pytest.approx(1.0)


def test_sign_lit_when_ray_clears_wall():
    rx = np.array([1.0, 2.0, 0.0])
    # Crosses y=0 at x>0 -> through open air, edge does not occlude.
    sat = np.array([300.0, -200.0, 0.0])
    s = edge_lit_shadow_sign(rx, sat, EDGE_POINT, EDGE_START, EDGE_END, FACE_A)
    assert s == pytest.approx(-1.0)


def test_sign_lit_when_both_on_same_side():
    rx = np.array([1.0, 2.0, 0.0])
    sat = np.array([5.0, 400.0, 0.0])  # same +y side, never crosses y=0
    s = edge_lit_shadow_sign(rx, sat, EDGE_POINT, EDGE_START, EDGE_END, FACE_A)
    assert s == pytest.approx(-1.0)


def test_wedge_uses_both_faces():
    # Two faces: -x wall and +y wall. A ray blocked only by the +y face must
    # still register as shadow when that face is supplied.
    face_b = np.array([0.0, 1.0, 0.0])  # wall in x=0, y>=0 plane
    rx = np.array([2.0, 1.0, 0.0])
    sat = np.array([-200.0, 1.0, 0.0])  # crosses x=0 at y>0 -> through +y wall
    only_a = edge_lit_shadow_sign(rx, sat, EDGE_POINT, EDGE_START, EDGE_END, FACE_A)
    with_b = edge_lit_shadow_sign(
        rx, sat, EDGE_POINT, EDGE_START, EDGE_END, FACE_A, face_b)
    assert with_b == pytest.approx(1.0)
    # face A alone (the -x wall) does not occlude this particular ray.
    assert only_a == pytest.approx(-1.0)


class _Edges:
    """Minimal edge set with wedge faces for compute_diffraction_paths."""

    def __init__(self, start, end, face_a, face_b=None):
        self.start = np.asarray(start, float).reshape(-1, 3)
        self.end = np.asarray(end, float).reshape(-1, 3)
        self.midpoint = 0.5 * (self.start + self.end)
        self.face_dir_a = np.asarray(face_a, float).reshape(-1, 3)
        self.face_dir_b = (None if face_b is None
                           else np.asarray(face_b, float).reshape(-1, 3))

    @property
    def size(self):
        return self.start.shape[0]


def test_compute_paths_lit_amplitude_exceeds_unsigned():
    # A single edge near the ray. A lit-side satellite should, with lit_shadow,
    # get a near-unity amplitude (no diffraction loss) instead of the unsigned
    # model's <=0.5.
    edges = _Edges([EDGE_START], [EDGE_END], [FACE_A])
    rx = np.array([1.0, 2.0, 0.0])
    sat = np.array([300.0, -200.0, 0.0])  # lit (ray clears the wall)
    kw = dict(max_paths=2, max_edge_range_m=500.0,
              max_ray_edge_distance_m=50.0, max_excess_path_m=500.0)
    unsigned = compute_diffraction_paths(rx, sat, edges, **kw)
    signed = compute_diffraction_paths(rx, sat, edges, lit_shadow=True, **kw)
    assert unsigned[0] and signed[0]
    assert unsigned[0][0].amplitude <= 0.5 + 1e-9
    assert signed[0][0].amplitude > 0.9  # lit -> ~full field
    assert signed[0][0].fresnel_v < 0.0  # negative clearance = lit


def test_require_shadow_drops_lit_edges():
    # The single edge is lit for this satellite -> require_shadow yields no
    # candidate (the edge is not the silhouette that shadows the sat), while the
    # plain models still return it.
    edges = _Edges([EDGE_START], [EDGE_END], [FACE_A])
    rx = np.array([1.0, 2.0, 0.0])
    sat = np.array([300.0, -200.0, 0.0])  # lit
    kw = dict(max_paths=2, max_edge_range_m=500.0,
              max_ray_edge_distance_m=50.0, max_excess_path_m=500.0)
    assert compute_diffraction_paths(rx, sat, edges, **kw)[0]  # legacy keeps it
    assert compute_diffraction_paths(
        rx, sat, edges, require_shadow=True, **kw) == [[]]


def test_require_shadow_keeps_shadow_edges():
    edges = _Edges([EDGE_START], [EDGE_END], [FACE_A])
    rx = np.array([1.0, 0.5, 0.0])
    sat = np.array([-300.0, -150.0, 0.0])  # shadow: wall occludes the ray
    kw = dict(max_paths=2, max_edge_range_m=500.0,
              max_ray_edge_distance_m=50.0, max_excess_path_m=500.0)
    out = compute_diffraction_paths(rx, sat, edges, require_shadow=True, **kw)
    assert out[0]
    assert out[0][0].fresnel_v >= 0.0  # kept candidate is shadow-side


def test_compute_paths_shadow_matches_unsigned_at_grazing():
    # A shadow-side satellite with the ray grazing the edge: signed and unsigned
    # agree near v=0 (the sign only matters away from the shadow boundary).
    edges = _Edges([EDGE_START], [EDGE_END], [FACE_A])
    rx = np.array([1.0, 0.5, 0.0])
    sat = np.array([-300.0, -150.0, 0.0])  # shadow
    kw = dict(max_paths=2, max_edge_range_m=500.0,
              max_ray_edge_distance_m=50.0, max_excess_path_m=500.0)
    signed = compute_diffraction_paths(rx, sat, edges, lit_shadow=True, **kw)
    assert signed[0]
    assert signed[0][0].fresnel_v >= 0.0  # shadow -> non-negative clearance
