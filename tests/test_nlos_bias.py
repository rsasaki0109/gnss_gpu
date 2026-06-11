"""Tests for the true-NLOS pseudorange bias model."""

import math

import pytest

from gnss_gpu.validation.nlos_bias import (
    ReflectionReplica,
    nlos_bias_m,
    path_cn0_dbhz,
    pooled_nlos_bias_m,
    predict_nlos_bias_samples_m,
    reflection_replicas,
    select_tracked_path,
)


class _Path:
    def __init__(self, amplitude, excess_delay):
        self.amplitude = amplitude
        self.excess_delay = excess_delay


class _ReflPath:
    """Mimics raytrace.ReflectionPath (no amplitude; carries incidence/triangle)."""

    def __init__(self, excess_delay, incidence_angle, triangle_id=0):
        self.excess_delay = excess_delay
        self.incidence_angle = incidence_angle
        self.triangle_id = triangle_id


def test_path_cn0():
    assert path_cn0_dbhz(1.0, 45.0) == pytest.approx(45.0)
    assert path_cn0_dbhz(0.5, 45.0) == pytest.approx(45.0 - 6.0206, abs=1e-3)
    assert path_cn0_dbhz(0.1, 45.0) == pytest.approx(25.0, abs=1e-6)


def test_select_strongest_above_threshold():
    paths = [_Path(0.5, 40.0), _Path(0.3, 25.0), _Path(0.7, 60.0)]
    t = select_tracked_path(paths, cn0_los_dbhz=45.0, cn0_threshold_dbhz=28.0)
    assert t.amplitude_ratio == 0.7
    assert t.excess_delay_m == 60.0


def test_threshold_gates_weak_paths():
    # amplitude 0.05 -> 45 - 26 = 19 dBHz, below a 28 dBHz threshold.
    paths = [_Path(0.05, 30.0)]
    assert select_tracked_path(paths, cn0_threshold_dbhz=28.0) is None
    assert nlos_bias_m(paths, cn0_threshold_dbhz=28.0) is None
    # Lowering the threshold makes it trackable.
    assert nlos_bias_m(paths, cn0_threshold_dbhz=15.0) == 30.0


def test_nlos_bias_is_tracked_excess_delay():
    paths = [_Path(0.4, 35.0), _Path(0.6, 80.0)]
    assert nlos_bias_m(paths) == 80.0  # strongest replica's excess delay


def test_predict_samples_drops_untrackable():
    per_sat = [
        [_Path(0.5, 40.0)],         # trackable -> 40
        [_Path(0.02, 25.0)],        # too weak -> dropped
        [_Path(0.3, 70.0), _Path(0.1, 20.0)],  # strongest trackable -> 70
        [],                         # no paths -> dropped
    ]
    samples = predict_nlos_bias_samples_m(per_sat, cn0_threshold_dbhz=28.0)
    assert samples == [40.0, 70.0]


def test_amplitude_model_changes_trackability():
    # Same geometry (excess delay), different amplitude models. The weaker
    # (knife-edge-like) model drops the path; the stronger (UTD-like) keeps it.
    excess = 55.0
    weak = [[_Path(0.04, excess)]]   # 45 - 28 = 17 dBHz < 28 -> dropped
    strong = [[_Path(0.25, excess)]]  # 45 - 12 = 33 dBHz > 28 -> tracked
    assert predict_nlos_bias_samples_m(weak, cn0_threshold_dbhz=28.0) == []
    assert predict_nlos_bias_samples_m(strong, cn0_threshold_dbhz=28.0) == [55.0]


def test_reflection_replicas_amplitude_matches_fresnel():
    from gnss_gpu.fresnel import reflection_coefficient

    angle = math.radians(50.0)
    [rep] = reflection_replicas([_ReflPath(42.0, angle, triangle_id=3)],
                                material="concrete", polarization="rhcp")
    assert rep.excess_delay == 42.0
    assert rep.triangle_id == 3
    expected = reflection_coefficient(angle, material="concrete",
                                      polarization="rhcp")
    assert rep.amplitude == pytest.approx(expected)
    assert 0.0 <= rep.amplitude <= 1.0


def test_reflection_replicas_ground_material_override():
    angle = math.radians(70.0)
    paths = [_ReflPath(20.0, angle, triangle_id=-1)]  # ground plane
    from gnss_gpu.fresnel import reflection_coefficient

    [rep] = reflection_replicas(paths, material="concrete",
                                ground_material="wet_ground", polarization="rhcp")
    assert rep.amplitude == pytest.approx(
        reflection_coefficient(angle, material="wet_ground", polarization="rhcp"))


def test_pooled_bias_prefers_strong_reflection_over_weak_diffraction():
    # A grazing diffraction replica (weak, small delay) and a wall reflection
    # (strong, large delay). The receiver tracks the strongest -> reflection,
    # giving a tens-of-metres bias the diffraction-only model cannot reach.
    diff = [_Path(0.10, 8.0)]                # 45 - 20 = 25 dBHz
    refl = reflection_replicas([_ReflPath(34.0, math.radians(55.0))],
                               material="concrete")
    assert refl[0].amplitude > 0.10          # reflection is the stronger replica
    bias = pooled_nlos_bias_m(diff, refl, cn0_threshold_dbhz=20.0)
    assert bias == 34.0


def test_pooled_bias_falls_back_to_diffraction_when_no_reflection():
    diff = [_Path(0.12, 9.0)]
    assert pooled_nlos_bias_m(diff, [], cn0_threshold_dbhz=20.0) == 9.0
    assert pooled_nlos_bias_m([], [], cn0_threshold_dbhz=20.0) is None
