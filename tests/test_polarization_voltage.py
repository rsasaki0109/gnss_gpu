"""Published MPSim field composition checked via an independent basis change."""

import numpy as np
import pytest

from gnss_gpu.fresnel import (
    circular_reflection_voltage, complex_permittivity, reflection_coefficient_complex,
)


def test_general_circular_fields_match_linear_basis_boundary_conditions():
    theta, er, el = .7, .8+.2j, .1-.2j
    hr, hl = .5+.3j, .1j
    eps = complex_permittivity("concrete")
    root = np.sqrt(eps-np.sin(theta)**2)
    rp = (eps*np.cos(theta)-root)/(eps*np.cos(theta)+root)
    rs = (np.cos(theta)-root)/(np.cos(theta)+root)
    # Unitary circular/linear basis transformation, evaluated independently.
    transform = np.array([[1., -1j], [1., 1j]])/np.sqrt(2)
    incident_linear = transform.conj().T@np.array([er, el])
    reflected_linear = np.diag([rp, rs])@incident_linear
    reflected_circular = transform@reflected_linear
    expected = np.dot([hr, hl], reflected_circular)
    actual = circular_reflection_voltage(theta, incident_rhcp=er, incident_lhcp=el,
                                         receiver_rhcp=hr, receiver_lhcp=hl)
    assert actual == pytest.approx(expected, abs=1e-15)


def test_ideal_rhcp_limit_matches_existing_copol_coefficient():
    angles = np.linspace(0., np.pi/2, 31)
    np.testing.assert_array_equal(circular_reflection_voltage(angles),
                                  reflection_coefficient_complex(angles, polarization="rhcp"))


def test_normal_incidence_cross_response_uses_voltage_not_power_gain():
    # eps=4: R_parallel=+1/3, R_perp=-1/3; reflection is pure opposite hand.
    ideal = circular_reflection_voltage(0., 4+0j)
    assert abs(ideal) < 1e-15
    opposite = circular_reflection_voltage(0., 4+0j, receiver_rhcp=0., receiver_lhcp=1.)
    leakage = circular_reflection_voltage(0., 4+0j, receiver_lhcp=.1j)
    assert leakage == pytest.approx(1j/30)
    assert abs(leakage/opposite)**2 == pytest.approx(.01)  # -20 dB power response


def test_receive_response_phase_changes_coherent_power():
    a = circular_reflection_voltage(.8, receiver_lhcp=.1)
    b = circular_reflection_voltage(.8, receiver_lhcp=-.1)
    assert abs(a)**2 != pytest.approx(abs(b)**2, abs=1e-5)


@pytest.mark.parametrize("kwargs", [dict(receiver_lhcp=np.nan), dict(incident_rhcp=np.inf),
                                       dict(receiver_rhcp="bad"), dict(freq_hz=0),
                                       dict(material=complex(np.inf)), dict(incidence_angle_rad=-.1),
                                       dict(incidence_angle_rad=np.nan)])
def test_invalid_polarization_inputs_are_rejected(kwargs):
    values = dict(incidence_angle_rad=.5)
    values.update(kwargs)
    with pytest.raises(ValueError):
        circular_reflection_voltage(**values)


def test_wall_leakage_changes_reflected_iq_and_preserves_direct_wave():
    pytest.importorskip("numba")
    from gnss_gpu import CpuSignalSimulator
    from gnss_gpu.wall_signal_sim import VerticalWall, simulate_wall_signal

    sim = CpuSignalSimulator(noise_floor_db=-10000)
    ideal_wall = VerticalWall(0., -100., 100., 0., 100.)
    real_wall = VerticalWall(0., -100., 100., 0., 100., antenna_lhcp_response=.1j)
    args = dict(rx_position_m=[-10., 0., 2.], rx_velocity_mps=[0., 0., 0.],
                transmitter_m=[-100., 0., 2.], n_samples=4096,
                integrate_samples=True, reference_range_m=90.)
    ideal = simulate_wall_signal(sim, ideal_wall, **args)
    ideal_direct = simulate_wall_signal(sim, ideal_wall, include_reflection=False, **args)
    full = simulate_wall_signal(sim, real_wall, **args)
    direct = simulate_wall_signal(sim, real_wall, include_reflection=False, **args)
    reflected = simulate_wall_signal(sim, real_wall, include_direct=False, **args)
    np.testing.assert_array_equal(ideal.direct_range_m, full.direct_range_m)
    np.testing.assert_array_equal(ideal_direct.iq, direct.iq)
    np.testing.assert_allclose(ideal.iq, ideal_direct.iq, atol=2e-17)
    assert np.max(np.abs(reflected.iq)) > .01
    np.testing.assert_allclose(full.iq, direct.iq+reflected.iq, atol=1.2e-7)
    assert not np.array_equal(ideal.iq, full.iq)


def test_wall_rejects_leakage_outside_the_defined_circular_basis():
    from gnss_gpu.wall_signal_sim import VerticalWall
    with pytest.raises(ValueError, match="RHCP"):
        VerticalWall(0., -10., 10., 0., 10., polarization="parallel", antenna_lhcp_response=.1)
    with pytest.raises(ValueError, match="finite"):
        VerticalWall(0., -10., 10., 0., 10., antenna_lhcp_response=np.nan)
