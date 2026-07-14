from types import SimpleNamespace

import numpy as np

from experiments.exp_ppc_ctrbpf_fgo import (
    _apply_fixed_lag_output,
    _apply_particle_mode_emission,
    _attach_particle_mode_diagnostics,
    _emission_prediction,
)
from gnss_gpu.particle_fixed_lag import FixedLagSmootherOutput
from gnss_gpu.particle_modes import extract_particle_modes, select_particle_mode


def _accepted_selection():
    rng = np.random.default_rng(3)
    particles = rng.normal([5.0, 1.0, -2.0], 0.1, size=(500, 3))
    result = extract_particle_modes(
        particles,
        np.zeros(500),
        voxel_size_m=0.5,
        min_core_cell_mass=0.002,
    )
    return result, select_particle_mode(result)


def test_diagnostic_policy_never_changes_output():
    _, selection = _accepted_selection()
    original = np.array([100.0, 200.0, 300.0])

    output, source = _apply_particle_mode_emission(
        original, "pf_hybrid_emit", "diagnostic", selection
    )

    np.testing.assert_array_equal(output, original)
    assert source == "pf_hybrid_emit"


def test_emit_changes_only_pf_sourced_output():
    _, selection = _accepted_selection()
    original = np.array([100.0, 200.0, 300.0])

    pf_output, pf_source = _apply_particle_mode_emission(
        original, "pf_hybrid_emit", "emit", selection
    )
    hybrid_output, hybrid_source = _apply_particle_mode_emission(
        original, "hybrid", "emit", selection
    )

    assert np.linalg.norm(pf_output - selection.position) == 0.0
    assert pf_source == "pf_hybrid_emit_mode"
    np.testing.assert_array_equal(hybrid_output, original)
    assert hybrid_source == "hybrid"


def test_rejected_selection_does_not_change_pf_output():
    selection = SimpleNamespace(accepted=False, position=np.array([1.0, 2.0, 3.0]))
    original = np.array([100.0, 200.0, 300.0])
    output, source = _apply_particle_mode_emission(original, "pf", "emit", selection)
    np.testing.assert_array_equal(output, original)
    assert source == "pf"


def test_constant_velocity_emission_prediction():
    positions = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [np.nan] * 3])
    times = np.array([0.0, 1.0, 1.5])
    np.testing.assert_allclose(_emission_prediction(positions, times, 2), [3.0, 0.0, 0.0])
    assert _emission_prediction(positions, times, 0) is None


def test_mode_diagnostics_include_selection_and_separation():
    rng = np.random.default_rng(8)
    particles = np.vstack(
        (
            rng.normal([0.0, 0.0, 0.0], 0.1, size=(700, 3)),
            rng.normal([6.0, 0.0, 0.0], 0.1, size=(300, 3)),
        )
    )
    result = extract_particle_modes(
        particles,
        np.zeros(1000),
        voxel_size_m=0.5,
        min_core_cell_mass=0.003,
    )
    selection = select_particle_mode(result)
    row = {}
    _attach_particle_mode_diagnostics(row, result, selection)

    assert row["pf_mode_count"] == 2
    assert row["pf_mode_selection_accepted"] is True
    assert row["pf_mode_selected_mass"] > 0.65
    assert row["pf_mode_top2_separation_m"] > 5.5
    assert row["pf_mode_analyzed_particles"] == 1000


def _smoother_output(position, covariance=None, rewrite_allowed=True):
    return FixedLagSmootherOutput(
        epoch_index=0,
        position=np.asarray(position, dtype=np.float64),
        covariance=(
            np.eye(3) * 0.04
            if covariance is None
            else np.asarray(covariance, dtype=np.float64)
        ),
        path_count=8,
        unique_oldest_particles=4,
        terminal_particle_count=20,
        terminal_conditioned=True,
        rewrite_allowed=rewrite_allowed,
    )


def test_fixed_lag_output_rewrites_only_pf_source_with_safety_gates():
    positions = np.array([[0.0, 0.0, 0.0]])
    sources = ["pf"]
    applied, reason, correction, max_std = _apply_fixed_lag_output(
        positions,
        sources,
        _smoother_output([1.0, 0.0, 0.0]),
        max_std_m=1.0,
        max_correction_m=2.0,
        min_unique_particles=2,
    )
    assert applied and reason == "accepted"
    assert correction == 1.0
    assert max_std == 0.2
    np.testing.assert_allclose(positions[0], [1.0, 0.0, 0.0])
    assert sources == ["pf_ffbsi"]

    positions[:] = 0.0
    sources[:] = ["hybrid"]
    applied, reason, *_ = _apply_fixed_lag_output(
        positions,
        sources,
        _smoother_output([1.0, 0.0, 0.0]),
        max_std_m=1.0,
        max_correction_m=2.0,
        min_unique_particles=2,
    )
    assert not applied and reason == "non_pf_source"
    np.testing.assert_array_equal(positions[0], [0.0, 0.0, 0.0])


def test_fixed_lag_output_abstains_on_path_spread_and_terminal_gate():
    positions = np.array([[0.0, 0.0, 0.0]])
    sources = ["pf"]
    applied, reason, *_ = _apply_fixed_lag_output(
        positions,
        sources,
        _smoother_output([1.0, 0.0, 0.0], covariance=np.eye(3) * 9.0),
        max_std_m=1.0,
        max_correction_m=2.0,
        min_unique_particles=2,
    )
    assert not applied and reason == "path_spread"

    applied, reason, *_ = _apply_fixed_lag_output(
        positions,
        sources,
        _smoother_output([1.0, 0.0, 0.0], rewrite_allowed=False),
        max_std_m=1.0,
        max_correction_m=2.0,
        min_unique_particles=2,
    )
    assert not applied and reason == "terminal_abstain"
