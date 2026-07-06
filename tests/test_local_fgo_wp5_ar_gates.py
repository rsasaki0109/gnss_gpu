"""Unit tests for WP5 work item 3: local_fgo.py AR validation gates.

Two new gates, both backward-compatible (disabled by default / no-op when
inputs are absent), added on top of ``solve_local_fgo_with_lambda``:

1. Segment-length gate (``LambdaFixConfig.min_epochs``, already existed but
   was silently applied with no reject counter) -- now tracked explicitly
   via ``_estimate_lambda_fixes``'s ``n_segments_rejected_short`` /
   ``min_segment_epochs`` and rolled up into the top-level summary's
   ``n_segments_rejected_short``.
2. DD-pseudorange cross-check (``LambdaFixConfig.ddpr_reject_threshold``,
   new) -- ``_ddpr_cross_check`` compares the DD-pseudorange residual RMS
   before/after tentatively applying a batch of new LAMBDA fixes and vetoes
   the whole batch if the code-based (carrier-independent) residual gets
   worse, mirroring inuex35's DDPR cross-validation / post-AR cost gate.

This repo's venv has no GTSAM bindings (confirmed in WP4/WP5), so
``solve_local_fgo``/``solve_local_fgo_with_lambda`` transparently exercise
the NumPy/SciPy sparse-LM fallback here -- i.e. these tests cover the exact
backend that WP5's full sweep actually runs.
"""

from __future__ import annotations

import numpy as np

from gnss_gpu.local_fgo import (
    DDCarrierEpoch,
    DDPseudorangeEpoch,
    LambdaFixConfig,
    LocalFgoConfig,
    LocalFgoProblem,
    LocalFgoWindow,
    _ddpr_cross_check,
    _estimate_lambda_fixes,
    solve_local_fgo,
    solve_local_fgo_with_lambda,
)


WAVELENGTH_M = 299792458.0 / 1575.42e6


def _dd_expected_m(x, sat_k, sat_ref, base_k, base_ref) -> float:
    return float(
        np.linalg.norm(sat_k - x) - np.linalg.norm(sat_ref - x) - base_k + base_ref
    )


def _make_dd_pr_epoch(sat_k, sat_ref, base_k, base_ref, dd_value_m) -> DDPseudorangeEpoch:
    return DDPseudorangeEpoch(
        dd_pseudorange_m=np.array([dd_value_m]),
        sat_ecef_k=sat_k.reshape(1, 3),
        sat_ecef_ref=sat_ref.reshape(1, 3),
        base_range_k=np.array([base_k]),
        base_range_ref=np.array([base_ref]),
        weights=np.array([1.0]),
    )


# ---------------------------------------------------------------------------
# Gate 2: DD-pseudorange cross-check (`_ddpr_cross_check`), pure unit tests.
# ---------------------------------------------------------------------------


def _geometry():
    sat_k = np.array([0.0, 20_000_000.0, 0.0])
    sat_ref = np.array([20_000_000.0, 0.0, 0.0])
    base_pos = np.zeros(3)
    base_k = float(np.linalg.norm(sat_k - base_pos))
    base_ref = float(np.linalg.norm(sat_ref - base_pos))
    return sat_k, sat_ref, base_k, base_ref


def test_ddpr_cross_check_disabled_by_zero_threshold_always_accepts():
    sat_k, sat_ref, base_k, base_ref = _geometry()
    window = LocalFgoWindow(0, 1)
    x_before = np.zeros((2, 3))
    x_after = np.array([[1000.0, 0.0, 0.0], [1000.0, 0.0, 0.0]])  # huge, "obviously worse" move
    dd_pr = [
        _make_dd_pr_epoch(sat_k, sat_ref, base_k, base_ref, _dd_expected_m(np.zeros(3), sat_k, sat_ref, base_k, base_ref))
        for _ in range(2)
    ]
    accepted, rms_before, rms_after = _ddpr_cross_check(
        dd_pr, window, [0, 1], x_before, x_after, threshold=0.0
    )
    assert accepted is True
    assert rms_before is None and rms_after is None


def test_ddpr_cross_check_no_data_always_accepts():
    window = LocalFgoWindow(0, 1)
    x_before = np.zeros((2, 3))
    x_after = np.array([[1000.0, 0.0, 0.0], [1000.0, 0.0, 0.0]])
    accepted, rms_before, rms_after = _ddpr_cross_check(
        None, window, [0, 1], x_before, x_after, threshold=0.1
    )
    assert accepted is True
    assert rms_before is None and rms_after is None


def test_ddpr_cross_check_rejects_when_residual_worsens():
    sat_k, sat_ref, base_k, base_ref = _geometry()
    window = LocalFgoWindow(0, 1)
    x_before = np.zeros((2, 3))
    x_after = np.array([[5.0, 0.0, 0.0], [5.0, 0.0, 0.0]])  # moved away, code residual must grow
    # DD-PR observation exactly matches x_before -> residual_before == 0.
    dd_value = _dd_expected_m(np.zeros(3), sat_k, sat_ref, base_k, base_ref)
    dd_pr = [_make_dd_pr_epoch(sat_k, sat_ref, base_k, base_ref, dd_value) for _ in range(2)]
    accepted, rms_before, rms_after = _ddpr_cross_check(
        dd_pr, window, [0, 1], x_before, x_after, threshold=0.05
    )
    assert accepted is False
    assert rms_before == 0.0
    assert rms_after is not None and rms_after > 0.0


def test_ddpr_cross_check_accepts_when_residual_improves():
    sat_k, sat_ref, base_k, base_ref = _geometry()
    window = LocalFgoWindow(0, 1)
    # True position is at (5, 0, 0); "before" is biased at the origin.
    true_pos = np.array([5.0, 0.0, 0.0])
    x_before = np.zeros((2, 3))
    x_after = np.tile(true_pos, (2, 1))
    dd_value = _dd_expected_m(true_pos, sat_k, sat_ref, base_k, base_ref)
    dd_pr = [_make_dd_pr_epoch(sat_k, sat_ref, base_k, base_ref, dd_value) for _ in range(2)]
    accepted, rms_before, rms_after = _ddpr_cross_check(
        dd_pr, window, [0, 1], x_before, x_after, threshold=0.05
    )
    assert accepted is True
    assert rms_before is not None and rms_after is not None
    assert rms_after < rms_before


def test_ddpr_cross_check_ignores_epochs_outside_touched_set():
    sat_k, sat_ref, base_k, base_ref = _geometry()
    window = LocalFgoWindow(0, 2)
    x_before = np.zeros((3, 3))
    # Only epoch 2 (untouched) moves catastrophically; touched set is {0, 1}.
    x_after = x_before.copy()
    x_after[2] = [9999.0, 0.0, 0.0]
    dd_value = _dd_expected_m(np.zeros(3), sat_k, sat_ref, base_k, base_ref)
    dd_pr = [_make_dd_pr_epoch(sat_k, sat_ref, base_k, base_ref, dd_value) for _ in range(3)]
    accepted, rms_before, rms_after = _ddpr_cross_check(
        dd_pr, window, [0, 1], x_before, x_after, threshold=0.01
    )
    assert accepted is True
    assert rms_before == 0.0 and rms_after == 0.0


# ---------------------------------------------------------------------------
# Gate 1: segment-length gate accept/reject counters (`_estimate_lambda_fixes`).
# ---------------------------------------------------------------------------


def _dd_carrier_epoch(sat_k, sat_ref, base_k, base_ref, cycles_value) -> DDCarrierEpoch:
    return DDCarrierEpoch(
        dd_carrier_cycles=np.array([cycles_value]),
        sat_ecef_k=sat_k.reshape(1, 3),
        sat_ecef_ref=sat_ref.reshape(1, 3),
        base_range_k=np.array([base_k]),
        base_range_ref=np.array([base_ref]),
        wavelengths_m=np.array([WAVELENGTH_M]),
        weights=np.array([1.0]),
        sat_ids=("G01",),
        ref_sat_ids=("G02",),
    )


def test_segment_length_gate_rejects_short_track_and_counts_it():
    sat_k, sat_ref, base_k, base_ref = _geometry()
    n_epochs = 3  # shorter than min_epochs=5
    positions = np.zeros((n_epochs, 3))
    dd_value = _dd_expected_m(np.zeros(3), sat_k, sat_ref, base_k, base_ref) / WAVELENGTH_M
    dd_epochs = [_dd_carrier_epoch(sat_k, sat_ref, base_k, base_ref, dd_value) for _ in range(n_epochs)]
    config = LambdaFixConfig(min_epochs=5, ratio_threshold=3.0)
    fixes, info = _estimate_lambda_fixes(dd_epochs, positions, LocalFgoWindow(0, n_epochs - 1), config)
    assert fixes == {}
    assert info["n_segments"] == 0
    assert info["n_segments_rejected_short"] == 1
    assert info["min_segment_epochs"] == 5


def test_segment_length_gate_accepts_long_enough_track():
    sat_k, sat_ref, base_k, base_ref = _geometry()
    n_epochs = 6  # >= min_epochs=5
    positions = np.zeros((n_epochs, 3))
    dd_value = _dd_expected_m(np.zeros(3), sat_k, sat_ref, base_k, base_ref) / WAVELENGTH_M
    dd_epochs = [_dd_carrier_epoch(sat_k, sat_ref, base_k, base_ref, dd_value) for _ in range(n_epochs)]
    config = LambdaFixConfig(min_epochs=5, ratio_threshold=3.0)
    fixes, info = _estimate_lambda_fixes(dd_epochs, positions, LocalFgoWindow(0, n_epochs - 1), config)
    assert len(fixes) > 0
    assert info["n_segments"] == 1
    assert info["n_segments_rejected_short"] == 0


# ---------------------------------------------------------------------------
# End-to-end: `solve_local_fgo_with_lambda` (NumPy fallback backend).
# ---------------------------------------------------------------------------


def _build_biased_seed_problem(bias_m: float, n_epochs: int = 6):
    sat_ref = np.array([20_000_000.0, 0.0, 0.0])
    sat_k = np.array([0.0, 20_000_000.0, 0.0])
    base_pos = np.zeros(3)
    base_k = float(np.linalg.norm(sat_k - base_pos))
    base_ref = float(np.linalg.norm(sat_ref - base_pos))
    x_true = np.zeros((n_epochs, 3))
    x_seed = np.tile([bias_m, 0.0, 0.0], (n_epochs, 1))
    dd_value = _dd_expected_m(x_true[0], sat_k, sat_ref, base_k, base_ref) / WAVELENGTH_M
    dd_carrier = [_dd_carrier_epoch(sat_k, sat_ref, base_k, base_ref, dd_value) for _ in range(n_epochs)]
    problem = LocalFgoProblem(
        initial_positions_ecef=x_seed,
        prior_positions_ecef=x_seed,
        window=LocalFgoWindow(0, n_epochs - 1),
        motion_deltas_ecef=np.zeros((n_epochs - 1, 3)),
        dd_carrier=dd_carrier,
    )
    config = LocalFgoConfig(
        prior_sigma_m=1.0,
        motion_sigma_m=0.5,
        dd_sigma_cycles=0.2,
        dd_fixed_sigma_cycles=0.05,
        max_iterations=30,
    )
    return problem, config, sat_k, sat_ref, base_k, base_ref


def test_lambda_ddpr_gate_disabled_by_default_matches_pre_wp5_behaviour():
    """Backward-compat: default LambdaFixConfig (ddpr_reject_threshold=0) fixes as before."""
    problem, config, *_ = _build_biased_seed_problem(bias_m=45.09)
    lam_cfg = LambdaFixConfig(min_epochs=3, max_iterations=1, ratio_threshold=3.0)
    assert lam_cfg.ddpr_reject_threshold == 0.0
    result, info = solve_local_fgo_with_lambda(problem, config, lam_cfg)
    assert info["n_fixed"] > 0
    assert info["n_ddpr_rejected_iterations"] == 0
    assert info["n_ddpr_rejected_observations"] == 0
    assert result.factor_counts["dd_carrier_fixed"] > 0


def test_lambda_ddpr_gate_rejects_a_self_consistent_but_worse_fix():
    """A LAMBDA fix that is internally self-consistent (huge ratio) but makes the
    independent DD-pseudorange residual worse must be vetoed by the gate, leaving
    the pre-iteration (float) result unchanged."""
    problem, config, sat_k, sat_ref, base_k, base_ref = _build_biased_seed_problem(bias_m=45.09)

    x_before = solve_local_fgo(problem, config).positions_ecef.copy()

    # Sanity check: without the gate, this scenario *does* produce a fix
    # (reproduces the WP4 self-consistency bug we are now guarding against).
    lam_disabled = LambdaFixConfig(min_epochs=3, max_iterations=1, ratio_threshold=3.0)
    _result_disabled, info_disabled = solve_local_fgo_with_lambda(problem, config, lam_disabled)
    assert info_disabled["n_fixed"] > 0

    # DD-pseudorange observations are constructed to exactly match x_before
    # (residual_before == 0 identically) so that any post-fix displacement
    # at all is a relative degradation, deterministically triggering the gate.
    dd_pr = [
        _make_dd_pr_epoch(sat_k, sat_ref, base_k, base_ref, _dd_expected_m(x, sat_k, sat_ref, base_k, base_ref))
        for x in x_before
    ]
    gated_problem = LocalFgoProblem(
        initial_positions_ecef=problem.initial_positions_ecef,
        prior_positions_ecef=problem.prior_positions_ecef,
        window=problem.window,
        motion_deltas_ecef=problem.motion_deltas_ecef,
        dd_carrier=problem.dd_carrier,
        dd_pseudorange=dd_pr,
    )
    lam_gated = LambdaFixConfig(min_epochs=3, max_iterations=1, ratio_threshold=3.0, ddpr_reject_threshold=0.01)
    result_gated, info_gated = solve_local_fgo_with_lambda(gated_problem, config, lam_gated)

    assert info_gated["n_fixed"] == 0
    assert info_gated["n_ddpr_rejected_iterations"] == 1
    assert info_gated["n_ddpr_rejected_observations"] > 0
    np.testing.assert_allclose(result_gated.positions_ecef, x_before)

    iteration = info_gated["iterations"][0]
    assert iteration["ddpr_gate_accepted"] is False
    assert iteration["ddpr_rms_before"] == 0.0
    assert iteration["ddpr_rms_after"] > 0.0
