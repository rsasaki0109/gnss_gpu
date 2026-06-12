from types import SimpleNamespace

import numpy as np

from gnss_gpu.diffraction import (
    compute_diffraction_paths,
    fresnel_integral,
    fresnel_v,
    knife_edge_amplitude,
    knife_edge_loss_db,
)


def make_edges():
    start = np.array([[100.0, 0.0, 0.0]])
    end = np.array([[100.0, 0.0, 50.0]])
    return SimpleNamespace(
        start=start,
        end=end,
        midpoint=0.5 * (start + end),
        length_m=np.array([50.0]),
        dihedral_deg=np.array([90.0]),
        is_boundary=np.array([False]),
        size=1,
    )


def make_empty_edges():
    empty = np.empty((0, 3), dtype=float)
    return SimpleNamespace(
        start=empty,
        end=empty,
        midpoint=empty,
        length_m=np.empty(0),
        dihedral_deg=np.empty(0),
        is_boundary=np.empty(0, dtype=bool),
        size=0,
    )


def make_many_edges():
    x = np.array([80.0, 100.0, 120.0])
    start = np.column_stack([x, np.zeros_like(x), np.zeros_like(x)])
    end = np.column_stack([x, np.zeros_like(x), np.full_like(x, 50.0)])
    return SimpleNamespace(
        start=start,
        end=end,
        midpoint=0.5 * (start + end),
        length_m=np.full(3, 50.0),
        dihedral_deg=np.full(3, 90.0),
        is_boundary=np.zeros(3, dtype=bool),
        size=3,
    )


def test_knife_edge_loss_db_itu_shape_and_monotonicity():
    values = np.array([-2.0, -0.78, 0.0, 1.0, 3.0])
    loss = knife_edge_loss_db(values)

    assert loss[0] == 0.0
    assert loss[1] == 0.0
    assert abs(float(knife_edge_loss_db(0.0)) - 6.0) < 0.3
    assert float(knife_edge_loss_db(1.0)) > float(knife_edge_loss_db(0.0))
    assert np.all(np.diff(loss) >= -1.0e-12)


def test_knife_edge_amplitude_range_and_ordering():
    values = np.array([-2.0, 0.0, 1.0, 3.0])
    amplitude = knife_edge_amplitude(values)

    assert np.all(amplitude >= 0.0)
    assert np.all(amplitude <= 1.0)
    assert abs(amplitude[0] - 1.0) < 1.0e-12
    assert amplitude[3] < amplitude[2] < amplitude[1] < amplitude[0]


def test_fresnel_integral_basic_properties():
    c0, s0 = fresnel_integral(0.0)
    assert abs(c0) < 1.0e-12
    assert abs(s0) < 1.0e-12

    c, s = fresnel_integral(np.array([-1.0, 1.0]))
    assert abs(c[0] + c[1]) < 1.0e-4
    assert abs(s[0] + s[1]) < 1.0e-4

    c5, s5 = fresnel_integral(5.0)
    assert abs(c5 - 0.5) < 0.1
    assert abs(s5 - 0.5) < 0.1


def test_fresnel_v_matches_closed_form():
    expected = np.sqrt((2.0 / 0.19) * (200.0 / 10000.0))
    actual = fresnel_v(1.0, 100.0, 100.0, wavelength_m=0.19)

    assert abs(actual - expected) < 1.0e-12
    assert fresnel_v(0.0, 100.0, 100.0, wavelength_m=0.19) == 0.0


def test_compute_diffraction_paths_generates_path_for_ray_edge():
    rx = np.array([0.0, 0.0, 0.0])
    sat = np.array([[200.0, 0.0, 20.0]])

    paths_by_sat = compute_diffraction_paths(rx, sat, make_edges())

    assert len(paths_by_sat) == 1
    assert len(paths_by_sat[0]) == 1

    path = paths_by_sat[0][0]
    assert 0.0 < path.amplitude <= 1.0
    assert path.excess_delay >= 0.0
    assert path.edge_id == 0
    assert path.diffraction_point.shape == (3,)


def test_compute_diffraction_paths_empty_inputs():
    rx = np.array([0.0, 0.0, 0.0])
    sat = np.array([[200.0, 0.0, 20.0]])

    assert compute_diffraction_paths(rx, np.empty((0, 3)), make_edges()) == []
    assert compute_diffraction_paths(rx, sat, make_empty_edges()) == [[]]


def test_compute_diffraction_paths_respects_max_paths():
    rx = np.array([0.0, 0.0, 0.0])
    sat = np.array([[200.0, 0.0, 20.0]])

    paths_by_sat = compute_diffraction_paths(rx, sat, make_many_edges(), max_paths=2)

    assert len(paths_by_sat) == 1
    assert len(paths_by_sat[0]) == 2
