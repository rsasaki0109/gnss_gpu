"""Unit tests for the FGO 2-stage residual re-solve (VD path) and its guard.

These exercise the pure residual / mask / guard-cost helpers and the
``two_stage_residual_resolve_vd`` orchestration with a mock solver, so no native
CUDA solve is needed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
for p in (REPO, REPO / "python"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from experiments.gsdc2023_raw_bridge import (  # noqa: E402
    _pr_huber_guard_cost,
    _pr_linearized_residual,
    _pr_resolve_thresholds,
    _pr_two_stage_mask,
    two_stage_residual_resolve_vd,
)


def _state(pos, c0, n_extra_clock=0):
    """Build a single-epoch VD state row: [x,y,z, vx,vy,vz, c0, (extra clocks)]."""
    row = [*pos, 0.0, 0.0, 0.0, c0, *([0.0] * n_extra_clock)]
    return np.asarray([row], dtype=np.float64)


def test_linearized_residual_matches_hand_computation():
    # los . (x - ref) + clk, residual = z - that.
    los = np.asarray([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], dtype=np.float64)
    ref = np.asarray([[0.0, 0.0, 0.0]], dtype=np.float64)
    state = _state([2.0, 5.0, 0.0], c0=1.0)
    z = np.asarray([[10.0, 10.0]], dtype=np.float64)
    resid = _pr_linearized_residual(z, los, ref, state, sys_kind=None, n_clock=1)
    # pred = [2,5] + clk(1) = [3, 6]; resid = [10-3, 10-6] = [7, 4]
    np.testing.assert_allclose(resid, [[7.0, 4.0]])


def test_linearized_residual_uses_per_system_isb():
    los = np.zeros((1, 2, 3), dtype=np.float64)  # geometry contributes nothing
    ref = np.zeros((1, 3), dtype=np.float64)
    # n_clock=7 layout: clocks at state cols 6..12
    row = [0, 0, 0, 0, 0, 0, 3.0, 0, 0, 0, 0, 9.0, 0]  # c0=3, c5=9
    state = np.asarray([row], dtype=np.float64)
    z = np.asarray([[3.0, 12.0]], dtype=np.float64)
    sk = np.asarray([[0, 5]], dtype=np.int64)  # sat0 -> c0 only; sat1 -> c0 + c5
    resid = _pr_linearized_residual(z, los, ref, state, sys_kind=sk, n_clock=7)
    # sat0 clk=3 -> resid 0; sat1 clk=3+9=12 -> resid 0
    np.testing.assert_allclose(resid, [[0.0, 0.0]], atol=1e-9)


def test_resolve_thresholds_l1_l5_split():
    sk = np.asarray([[0, 4, 5, 6, 2]], dtype=np.int64)
    thr = _pr_resolve_thresholds(sk, sk.shape, threshold_l1_m=20.0, threshold_l5_m=15.0)
    np.testing.assert_allclose(thr, [[20.0, 15.0, 15.0, 15.0, 20.0]])
    # no sys_kind -> all L1
    thr2 = _pr_resolve_thresholds(None, (1, 3), 20.0, 15.0)
    np.testing.assert_allclose(thr2, [[20.0, 20.0, 20.0]])


def test_two_stage_mask_zeroes_outliers():
    resid = np.asarray([[1.0, 2.0, 50.0, -3.0, 0.5, 100.0]], dtype=np.float64)
    w = np.ones((1, 6), dtype=np.float64)
    thr = np.full((1, 6), 20.0)
    new_w, n_masked = _pr_two_stage_mask(resid, w, thr, min_keep=4)
    assert n_masked == 2
    assert new_w[0, 2] == 0.0 and new_w[0, 5] == 0.0
    assert list(new_w[0, [0, 1, 3, 4]]) == [1.0, 1.0, 1.0, 1.0]


def test_two_stage_mask_respects_min_keep_worst_first():
    # 6 rows all over threshold, min_keep=5 -> only the single worst dropped
    resid = np.asarray([[30.0, 40.0, 25.0, 90.0, 35.0, 50.0]], dtype=np.float64)
    w = np.ones((1, 6), dtype=np.float64)
    thr = np.full((1, 6), 20.0)
    new_w, n_masked = _pr_two_stage_mask(resid, w, thr, min_keep=5)
    assert n_masked == 1
    assert new_w[0, 3] == 0.0  # the 90 m row
    assert int((new_w[0] > 0).sum()) == 5


def test_two_stage_mask_skips_thin_epochs():
    resid = np.asarray([[50.0, 60.0, 70.0]], dtype=np.float64)  # only 3 rows
    w = np.ones((1, 3), dtype=np.float64)
    thr = np.full((1, 3), 20.0)
    new_w, n_masked = _pr_two_stage_mask(resid, w, thr, min_keep=5)
    assert n_masked == 0
    np.testing.assert_allclose(new_w, w)


def test_huber_guard_cost_caps_outliers():
    # one inlier (3 m) quadratic, one outlier (100 m) capped linear at c=20
    resid = np.asarray([[3.0, 100.0]], dtype=np.float64)
    w = np.ones((1, 2), dtype=np.float64)
    thr = np.full((1, 2), 20.0)
    cost = _pr_huber_guard_cost(resid, w, thr)
    expected = 0.5 * 9.0 + 20.0 * (100.0 - 10.0)  # 4.5 + 1800
    assert cost == pytest.approx(expected)
    # zero-weight rows ignored
    w2 = np.asarray([[0.0, 1.0]], dtype=np.float64)
    cost2 = _pr_huber_guard_cost(resid, w2, thr)
    assert cost2 == pytest.approx(20.0 * 90.0)


class _MockSolver:
    """Mutates state in place per call; returns (iters, mse). Call 1 = pass-1,
    call 2 = pass-2 (sets the clock so the guard sees the prescribed fit)."""

    def __init__(self, pass1_c0, pass2_c0, iters1=5, iters2=7):
        self.calls = 0
        self.pass1_c0 = pass1_c0
        self.pass2_c0 = pass2_c0
        self.iters1 = iters1
        self.iters2 = iters2
        self.weights_seen = []

    def __call__(self, sat_ecef, pseudorange, weights, state, **kwargs):
        self.calls += 1
        self.weights_seen.append(np.asarray(weights).copy())
        c0 = self.pass1_c0 if self.calls == 1 else self.pass2_c0
        state[:, 6] = c0
        return (self.iters1 if self.calls == 1 else self.iters2), 0.0


def _vd_kwargs():
    los = np.zeros((1, 6, 3), dtype=np.float64)  # geometry contributes nothing
    ref = np.zeros((1, 3), dtype=np.float64)
    return dict(pr_linearization_ref_ecef=ref, pr_linearization_los_ecef=los,
                sys_kind=None, n_clock=1)


def test_resolve_accepts_when_guard_cost_improves():
    # pass-1 c0=0 -> resid = z; one big outlier. pass-2 c0=1 tightens inliers.
    z = np.asarray([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 100.0]], dtype=np.float64)
    w = np.ones((1, 7), dtype=np.float64)
    state = _state([0.0, 0.0, 0.0], c0=0.0)
    los = np.zeros((1, 7, 3)); ref = np.zeros((1, 3))
    kw = dict(pr_linearization_ref_ecef=ref, pr_linearization_los_ecef=los,
              sys_kind=None, n_clock=1)
    solver = _MockSolver(pass1_c0=0.0, pass2_c0=1.0)
    iters, _mse = two_stage_residual_resolve_vd(
        solver, None, z, w, state,
        threshold_l1_m=20.0, threshold_l5_m=15.0, min_keep=5, vd_kwargs=kw)
    assert solver.calls == 2
    assert iters == 12  # iters1 + iters2
    assert state[0, 6] == 1.0  # pass-2 state kept
    # pass-2 ran on masked weights (outlier dropped)
    assert solver.weights_seen[1][0, 6] == 0.0


def test_resolve_rejects_when_guard_cost_worsens():
    z = np.asarray([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 100.0]], dtype=np.float64)
    w = np.ones((1, 7), dtype=np.float64)
    state = _state([0.0, 0.0, 0.0], c0=0.0)
    los = np.zeros((1, 7, 3)); ref = np.zeros((1, 3))
    kw = dict(pr_linearization_ref_ecef=ref, pr_linearization_los_ecef=los,
              sys_kind=None, n_clock=1)
    solver = _MockSolver(pass1_c0=0.0, pass2_c0=-50.0)  # pass-2 makes everything worse
    iters, _mse = two_stage_residual_resolve_vd(
        solver, None, z, w, state,
        threshold_l1_m=20.0, threshold_l5_m=15.0, min_keep=5, vd_kwargs=kw)
    assert solver.calls == 2
    assert iters == 5  # only pass-1 iters
    assert state[0, 6] == 0.0  # restored to pass-1 state


def test_resolve_keeps_pass2_when_guard_disabled():
    # Same setup as the reject test, but guard=False -> pass-2 is always kept,
    # even though its full-set Huber cost is worse. This is the production default:
    # the dense-urban win raises the robust cost yet lowers position error.
    z = np.asarray([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 100.0]], dtype=np.float64)
    w = np.ones((1, 7), dtype=np.float64)
    state = _state([0.0, 0.0, 0.0], c0=0.0)
    los = np.zeros((1, 7, 3)); ref = np.zeros((1, 3))
    kw = dict(pr_linearization_ref_ecef=ref, pr_linearization_los_ecef=los,
              sys_kind=None, n_clock=1)
    solver = _MockSolver(pass1_c0=0.0, pass2_c0=-50.0)  # would be rejected by the guard
    iters, _mse = two_stage_residual_resolve_vd(
        solver, None, z, w, state,
        threshold_l1_m=20.0, threshold_l5_m=15.0, min_keep=5, guard=False, vd_kwargs=kw)
    assert solver.calls == 2
    assert iters == 12  # iters1 + iters2 (pass-2 kept)
    assert state[0, 6] == -50.0  # pass-2 state kept despite worse Huber cost


def test_resolve_noop_without_fixed_linearization():
    z = np.asarray([[100.0, 100.0, 100.0, 100.0, 100.0, 100.0]], dtype=np.float64)
    w = np.ones((1, 6), dtype=np.float64)
    state = _state([0.0, 0.0, 0.0], c0=0.0)
    solver = _MockSolver(pass1_c0=0.0, pass2_c0=1.0)
    kw = dict(sys_kind=None, n_clock=1)  # no ref/los
    iters, _mse = two_stage_residual_resolve_vd(
        solver, None, z, w, state,
        threshold_l1_m=20.0, threshold_l5_m=15.0, min_keep=5, vd_kwargs=kw)
    assert solver.calls == 1  # never re-solved
    assert iters == 5


def test_resolve_noop_when_nothing_masked():
    z = np.asarray([[1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 4.0]], dtype=np.float64)  # all < 20
    w = np.ones((1, 7), dtype=np.float64)
    state = _state([0.0, 0.0, 0.0], c0=0.0)
    los = np.zeros((1, 7, 3)); ref = np.zeros((1, 3))
    kw = dict(pr_linearization_ref_ecef=ref, pr_linearization_los_ecef=los,
              sys_kind=None, n_clock=1)
    solver = _MockSolver(pass1_c0=0.0, pass2_c0=1.0)
    iters, _mse = two_stage_residual_resolve_vd(
        solver, None, z, w, state,
        threshold_l1_m=20.0, threshold_l5_m=15.0, min_keep=5, vd_kwargs=kw)
    assert solver.calls == 1  # nothing masked -> no re-solve
    assert iters == 5


def test_resolve_restores_state_on_pass2_failure():
    z = np.asarray([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 100.0]], dtype=np.float64)
    w = np.ones((1, 7), dtype=np.float64)
    state = _state([0.0, 0.0, 0.0], c0=0.0)
    los = np.zeros((1, 7, 3)); ref = np.zeros((1, 3))
    kw = dict(pr_linearization_ref_ecef=ref, pr_linearization_los_ecef=los,
              sys_kind=None, n_clock=1)

    class _FailingPass2(_MockSolver):
        def __call__(self, sat_ecef, pseudorange, weights, state, **kwargs):
            self.calls += 1
            if self.calls == 1:
                state[:, 6] = 0.0
                return 5, 0.0
            state[:, 6] = 999.0  # corrupt, then signal failure
            return -1, 0.0

    solver = _FailingPass2(0.0, 0.0)
    iters, _mse = two_stage_residual_resolve_vd(
        solver, None, z, w, state,
        threshold_l1_m=20.0, threshold_l5_m=15.0, min_keep=5, vd_kwargs=kw)
    assert solver.calls == 2
    assert iters == 5
    assert state[0, 6] == 0.0  # restored despite pass-2 corrupting then failing
