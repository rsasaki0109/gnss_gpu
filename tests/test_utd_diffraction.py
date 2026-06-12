"""Tests for the Kouyoumjian-Pathak UTD wedge diffraction model.

Validated against known physical properties of UTD:
  * the Fresnel transition function F(x) -> 0 at x=0 and -> 1 for large x
  * reciprocity of the diffraction coefficient: D(phi, phi') == D(phi', phi)
  * for a half-plane (n=2, absorbing edge) UTD reproduces the knife-edge model
  * the diffracted amplitude is ~0.5 at the shadow boundary and decays into shadow
"""

import math
from types import SimpleNamespace

import numpy as np
import pytest

from gnss_gpu.utd_diffraction import (
    UTDDiffractionPath,
    compute_utd_diffraction_paths,
    fresnel_transition,
    utd_coefficient,
    GPS_L1_WAVELENGTH,
)
from gnss_gpu.diffraction import compute_diffraction_paths

_K = 2.0 * math.pi / GPS_L1_WAVELENGTH


def _half_plane_edge():
    start = np.array([[0.0, 0.0, -100.0]])
    end = np.array([[0.0, 0.0, 100.0]])
    return SimpleNamespace(
        start=start, end=end, midpoint=0.5 * (start + end), size=1,
        face_dir_a=np.array([[0.0, -1.0, 0.0]]),
        face_dir_b=np.array([[0.0, -1.0, 0.0]]),
        wedge_n=np.array([2.0]))


def test_fresnel_transition_limits():
    assert fresnel_transition(0.0) == 0.0 + 0.0j
    assert abs(fresnel_transition(50.0)) == pytest.approx(1.0, abs=2e-3)
    assert abs(fresnel_transition(100.0)) == pytest.approx(1.0, abs=1e-3)
    # Small argument: magnitude well below 1.
    assert abs(fresnel_transition(0.01)) < 0.3


def test_coefficient_reciprocity():
    for mode in ("absorbing", "soft", "hard"):
        for n in (2.0, 1.5):
            a = utd_coefficient(2.3, 0.7, math.pi / 2, n, _K, 5.0, mode)
            b = utd_coefficient(0.7, 2.3, math.pi / 2, n, _K, 5.0, mode)
            assert abs(a - b) < 1e-12


def test_soft_hard_average_is_absorbing():
    args = (2.1, 0.9, math.pi / 2, 1.5, _K, 4.0)
    soft = utd_coefficient(*args, "soft")
    hard = utd_coefficient(*args, "hard")
    absorbing = utd_coefficient(*args, "absorbing")
    assert abs(0.5 * (soft + hard) - absorbing) < 1e-12


def test_n2_absorbing_matches_knife_edge():
    edges = _half_plane_edge()
    sat = np.array([-1.0e7, 0.0, 0.0])
    kw = dict(max_ray_edge_distance_m=30.0, max_excess_path_m=200.0, max_paths=1)
    for yr in (-12.0, -6.0, -3.0, -1.0, 1.0, 3.0, 6.0, 12.0):
        rx = np.array([50.0, yr, 0.0])
        ke = compute_diffraction_paths(rx, sat, edges, **kw)
        ut = compute_utd_diffraction_paths(rx, sat, edges, mode="absorbing", **kw)
        assert ke[0] and ut[0]
        # Agreement within ~0.5 dB across the shadow region.
        ratio_db = 20.0 * math.log10(ut[0][0].amplitude / ke[0][0].amplitude)
        assert abs(ratio_db) < 0.5


def test_amplitude_half_at_shadow_boundary_and_decays():
    edges = _half_plane_edge()
    sat = np.array([-1.0e7, 0.0, 0.0])
    kw = dict(max_ray_edge_distance_m=30.0, max_excess_path_m=200.0, max_paths=1)
    grazing = compute_utd_diffraction_paths(
        np.array([50.0, 0.05, 0.0]), sat, edges, mode="absorbing", **kw)
    deep = compute_utd_diffraction_paths(
        np.array([50.0, 20.0, 0.0]), sat, edges, mode="absorbing", **kw)
    assert grazing[0] and deep[0]
    # Near the shadow boundary the diffracted field is about half the incident.
    assert grazing[0][0].amplitude == pytest.approx(0.5, abs=0.06)
    # Deep in the shadow the amplitude is much smaller, attenuation larger.
    assert deep[0][0].amplitude < 0.1
    assert deep[0][0].attenuation_db > grazing[0][0].attenuation_db


def test_path_fields_and_excess_delay():
    edges = _half_plane_edge()
    sat = np.array([-1.0e7, 0.0, 0.0])
    rx = np.array([50.0, 5.0, 0.0])
    res = compute_utd_diffraction_paths(
        rx, sat, edges, mode="absorbing",
        max_ray_edge_distance_m=30.0, max_excess_path_m=200.0, max_paths=1)
    assert len(res) == 1 and res[0]
    p = res[0][0]
    assert isinstance(p, UTDDiffractionPath)
    q = p.diffraction_point
    d1 = np.linalg.norm(q - rx)
    d2 = np.linalg.norm(sat - q)
    direct = np.linalg.norm(sat - rx)
    assert p.excess_delay == pytest.approx(d1 + d2 - direct, abs=1e-6)
    assert p.wedge_n == 2.0
    assert 0.0 < p.beta0 <= math.pi
    assert p.attenuation_db == pytest.approx(
        -20.0 * math.log10(p.amplitude), abs=1e-9)


def test_wedge_vs_halfplane_differ():
    # A 90-degree convex wedge (n=1.5) diffracts differently from a half-plane.
    sat = np.array([-1.0e7, 0.0, 0.0])
    rx = np.array([50.0, 6.0, 0.0])
    kw = dict(max_ray_edge_distance_m=30.0, max_excess_path_m=200.0, max_paths=1)
    half = _half_plane_edge()
    wedge = _half_plane_edge()
    wedge.face_dir_b = np.array([[-1.0, 0.0, 0.0]])  # second face -> 90deg corner
    wedge.wedge_n = np.array([1.5])
    a_half = compute_utd_diffraction_paths(rx, sat, half, mode="absorbing", **kw)
    a_wedge = compute_utd_diffraction_paths(rx, sat, wedge, mode="absorbing", **kw)
    assert a_half[0] and a_wedge[0]
    assert a_half[0][0].amplitude != pytest.approx(a_wedge[0][0].amplitude, abs=1e-6)


def test_empty_and_disabled_inputs():
    edges = _half_plane_edge()
    sat = np.array([-1.0e7, 0.0, 0.0])
    assert compute_utd_diffraction_paths(
        np.array([50.0, 5.0, 0.0]), np.empty((0, 3)), edges) == []
    assert compute_utd_diffraction_paths(
        np.array([50.0, 5.0, 0.0]), sat, edges, max_paths=0) == [[]]
