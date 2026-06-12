import math

import numpy as np
import pytest

from gnss_gpu.fresnel import (
    complex_permittivity,
    fresnel_coefficients,
    reflection_coefficient,
)


def test_normal_incidence_sign_convention_and_rhcp_copol_zero():
    eps = complex_permittivity("concrete")
    r_par, r_perp = fresnel_coefficients(0.0, eps)

    sqrt_eps = np.sqrt(eps)
    expected_mag = abs((sqrt_eps - 1.0) / (sqrt_eps + 1.0))

    assert abs(abs(r_par) - abs(r_perp)) < 1e-12
    assert abs(r_par + r_perp) < 1e-12
    assert abs(abs(r_par) - expected_mag) < 1e-12
    assert abs(abs(r_perp) - expected_mag) < 1e-12

    rhcp = reflection_coefficient(0.0, "concrete", polarization="rhcp")
    assert rhcp == pytest.approx(0.0, abs=1e-12)


def test_grazing_incidence_linear_and_rhcp_copol_approach_one():
    theta = math.radians(89.9)

    for polarization in ("parallel", "perpendicular", "rhcp", "average"):
        coeff = reflection_coefficient(theta, "concrete", polarization=polarization)
        assert coeff > 0.98
        assert coeff <= 1.0

    # With the specified circular decomposition, R_par and R_perp share the
    # same -1 grazing limit, so RHCP cross-pol tends to zero for dielectrics.
    cross = reflection_coefficient(theta, "concrete", polarization="rhcp_cross")
    assert cross < 0.02


def test_concrete_rhcp_increases_with_incidence_angle():
    c0 = reflection_coefficient(math.radians(0.0), "concrete", polarization="rhcp")
    c45 = reflection_coefficient(math.radians(45.0), "concrete", polarization="rhcp")
    c89 = reflection_coefficient(math.radians(89.0), "concrete", polarization="rhcp")

    assert c0 == pytest.approx(0.0, abs=1e-12)
    assert c0 < c45 < c89


@pytest.mark.parametrize("angle", [0.0, math.radians(37.0), math.radians(89.0)])
def test_metal_reflects_linear_pols_fully_but_rejects_rhcp_copol(angle):
    # A perfect conductor gives |R_par| = |R_perp| = 1, so an incident RHCP wave
    # is flipped entirely to LHCP: linear / cross-pol magnitudes -> 1 while the
    # RHCP co-pol coefficient (R_par + R_perp)/2 -> 0.
    for polarization in ("parallel", "perpendicular", "rhcp_cross", "average"):
        coeff = reflection_coefficient(angle, "metal", polarization=polarization)
        assert coeff == pytest.approx(1.0, abs=1e-3)

    copol = reflection_coefficient(angle, "metal", polarization="rhcp")
    assert copol == pytest.approx(0.0, abs=1e-3)


def test_complex_permittivity_concrete_loss_sign():
    eps = complex_permittivity("concrete")

    assert eps.real == pytest.approx(5.31, abs=1e-12)
    assert eps.imag < 0.0


def test_tuple_and_complex_materials_work():
    theta = math.radians(45.0)

    coeff_tuple = reflection_coefficient(
        theta,
        (5.31, 0.0326),
        polarization="rhcp",
    )

    eps = complex_permittivity((5.31, 0.0326))
    coeff_complex = reflection_coefficient(
        theta,
        eps,
        polarization="rhcp",
    )

    assert coeff_tuple == pytest.approx(coeff_complex, rel=1e-12, abs=1e-12)

    r_par, r_perp = fresnel_coefficients(theta, eps)
    assert isinstance(r_par, complex)
    assert isinstance(r_perp, complex)


def test_array_angle_input_supported():
    angles = np.radians(np.array([0.0, 45.0, 89.0]))
    coeffs = reflection_coefficient(angles, "concrete", polarization="rhcp")

    assert isinstance(coeffs, np.ndarray)
    assert coeffs.shape == angles.shape
    assert coeffs[0] < coeffs[1] < coeffs[2]


def test_urban_reflector_material_makes_amplitude_angle_dependent():
    """reflector_material drives a physical (angle-dependent) replica amplitude."""
    from gnss_gpu.raytrace import BuildingModel
    from gnss_gpu.urban_signal_sim import UrbanSignalSimulator

    class _CaptureSignalGenerator:
        def __init__(self, sampling_freq=2.6e6):
            self.sampling_freq = sampling_freq
            self.channels = None

        def generate_epoch(self, channels, n_samples=None):
            self.channels = [dict(ch) for ch in channels]
            count = int(n_samples) if n_samples is not None else int(self.sampling_freq * 1e-3)
            return np.zeros(2 * count, dtype=np.float32)

    class _StubBuildingModel:
        def __init__(self, mesh):
            self._mesh = mesh

        def check_los(self, rx, sats):
            sats = np.asarray(sats).reshape(-1, 3)
            return np.ones(sats.shape[0], dtype=bool)

        def compute_reflection_paths(self, rx, sats, max_paths=4):
            return self._mesh.compute_reflection_paths(rx, sats, max_paths=max_paths)

    mesh = BuildingModel.create_box(np.array([0.05, 0.0, 0.0]), 0.1, 20.0, 20.0)
    rx = np.array([-10.0, -3.0, 0.0])
    sats = np.array([[-20.0, 9.0, 0.0]])

    usim = UrbanSignalSimulator(
        building_model=_StubBuildingModel(mesh),
        elevation_mask_deg=-90.0,
        max_reflection_paths=1,
        reflector_material="concrete",
        reflection_polarization="rhcp",
    )
    usim.sim = _CaptureSignalGenerator()

    result = usim.compute_epoch(rx, sats, prn_list=[3], n_samples=16)

    path = result["reflection_paths"][0][0]
    expected = reflection_coefficient(path.incidence_angle, "concrete", polarization="rhcp")
    replica_amp = result["channels"][1]["amplitude"]

    # Direct (LOS) amplitude is 1.0, so replica amplitude == coeff, and it must
    # differ from the legacy fixed 0.5.
    assert replica_amp == pytest.approx(float(expected))
    assert abs(replica_amp - 0.5) > 1e-6
