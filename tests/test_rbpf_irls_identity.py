"""Bit-exact CPU/GPU IRLS identity shortcuts and complete D236 state parity."""

from copy import deepcopy
import os
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
from bench_rbpf_feedback_mass import _bind_fixture_callbacks  # noqa: E402
from rbpf_feedback_cases import make_feedback_case  # noqa: E402
from rbpf_feedback_mass import NativeFeedbackMassStep  # noqa: E402
from rbpf_irls_identity import IdentityBatchScorer, IRLSIdentityStep  # noqa: E402
from rbpf_likelihood_batch import BatchScorer  # noqa: E402
from rbpf_likelihood_cases import make_case  # noqa: E402
from rbpf_move_cases import comparison, state  # noqa: E402


@pytest.fixture(scope="module")
def extensions():
    names = dict(
        extension="GNSS_GPU_RBPF_CANONICAL_EXT",
        parent_extension="GNSS_GPU_RBPF_PARENT_EXT",
        mass_extension="GNSS_GPU_RBPF_MASS_EXT",
    )
    if any(not os.environ.get(n) for n in names.values()):
        pytest.skip("Explicit unchanged canonical/parent/mass extensions required")
    return {k: os.environ[v] for k, v in names.items()}


@pytest.fixture(scope="module")
def scorers():
    old, new = (
        os.environ.get("GNSS_GPU_RBPF_PROBE_DLL"),
        os.environ.get("GNSS_GPU_RBPF_IDENTITY_DLL"),
    )
    if not old or not new:
        pytest.skip("Both explicit standalone DLLs required")
    with BatchScorer(old) as reference, IdentityBatchScorer(new) as candidate:
        yield reference, candidate


def exact(a, b):
    r = comparison(a, b)
    assert (
        r["discrete_equal"] and r["numeric_bits_equal"] and r["max_abs_error"] == 0.0
    ), r


def check_batch(scorers, hs, problem, oracle, mode):
    a, b = scorers
    before = deepcopy((problem, hs, oracle.cluster_states))
    va, ma = a.score(hs, problem, oracle, mode, diagnostics=True)
    vb, mb = b.score(hs, problem, oracle, mode, diagnostics=True)
    exact(va, vb)
    exact(ma, mb)
    exact(before, (problem, hs, oracle.cluster_states))
    if "statuses" in ma:
        assert b.last_skipped.shape == ma["statuses"].shape
        assert np.all((b.last_skipped >= 0) & (b.last_skipped <= 2))
        if oracle.cfg.huber_c <= 0:
            assert not np.any(b.last_skipped)
    else:
        assert b.last_skipped.size == 0
    return ma


@pytest.mark.parametrize("seed", [20260710, 20260711, 20260712])
@pytest.mark.parametrize("rows,groups", [(16, 1), (64, 4), (128, 8)])
@pytest.mark.parametrize(
    "uniform,huber,cluster",
    [
        (True, True, False),
        (False, True, False),
        (True, False, False),
        (False, False, False),
        (True, True, True),
        (False, True, True),
    ],
)
def test_all_scores_intermediates_status_and_input_bits(
    scorers, seed, rows, groups, uniform, huber, cluster
):
    oracle, p, hs = make_case(
        8,
        rows=rows,
        groups=groups,
        seed=seed,
        marg_uniform=uniform,
        huber=huber,
        cluster=cluster,
    )
    for mode in ("native_cpu", "gpu"):
        details = check_batch(scorers, hs, p, oracle, mode)
        assert details["scalar_fallbacks"] == 0


def zero_residual_case(rows=64, groups=4):
    oracle, p, hs = make_case(2, rows=rows, groups=groups, duplicates=False)
    h = hs[1]
    dd = oracle._pair_dd(h, p)
    assert np.all(np.isfinite(dd))
    p.rows_v0[:] = 0.0
    p.rows_bmeas[p.rows_iscp] = dd[p.rows_pair[p.rows_iscp]]
    # Include signed-zero matrix entries in the identity proof domain.
    p.rows_J[0, 0] = -0.0
    p.rows_J[0, 1] = 0.0
    return oracle, p, [h, deepcopy(h)]


@pytest.mark.parametrize("mode", ["native_cpu", "gpu"])
@pytest.mark.parametrize("rows,groups", [(16, 1), (64, 4), (128, 8)])
def test_exact_fixed_point_skips_two_duplicate_normal_solves(
    scorers, mode, rows, groups
):
    oracle, p, hs = zero_residual_case(rows, groups)
    details = check_batch(scorers, hs, p, oracle, mode)
    assert details["unique"] == 1 and details["scalar_fallbacks"] == 0
    assert scorers[1].last_skipped.tolist() == [2]


@pytest.mark.parametrize("mode", ["native_cpu", "gpu"])
def test_changed_weights_retain_original_work_and_no_irls_control(scorers, mode):
    oracle, p, hs = make_case(8, marg_uniform=False, duplicates=False)
    p.rows_bmeas[p.rows_iscp] += np.linspace(-20.0, 20.0, len(p.pairs))
    check_batch(scorers, hs, p, oracle, mode)
    assert np.any(scorers[1].last_skipped == 0)
    oracle.cfg.huber_c = 0.0
    check_batch(scorers, hs, p, oracle, mode)
    assert not np.any(scorers[1].last_skipped)


@pytest.mark.parametrize("rows,count", [(0, 1), (1, 3), (63, 7), (127, 21)])
def test_empty_and_odd_row_shapes_and_partial_blocks(scorers, rows, count):
    oracle, p, hs = make_case(count, rows=128, groups=8)
    for name, value in vars(p).items():
        if name.startswith("rows_"):
            setattr(p, name, value[:rows].copy())
    for mode in ("native_cpu", "gpu"):
        check_batch(scorers, hs, p, oracle, mode)
        if rows == 0:
            assert np.all(scorers[1].last_skipped == 2)


@pytest.mark.parametrize("kind", ["asymmetric", "singular", "tiny", "huge"])
def test_prior_scales_asymmetry_and_original_failure_bits(scorers, kind):
    oracle, p, hs = make_case(8, duplicates=False)
    if kind == "asymmetric":
        p.Ppos[0, 1] += 0.05
    elif kind == "singular":
        p.Ppos[:] = 0.0
    elif kind == "tiny":
        p.Ppos[:] = np.eye(3) * 1e-300
    else:
        p.Ppos[:] = np.eye(3) * 1e300
    with np.errstate(all="ignore"):
        for mode in ("native_cpu", "gpu"):
            details = check_batch(scorers, hs, p, oracle, mode)
            if kind == "singular":
                assert details["scalar_fallbacks"] == len(hs)


def test_no_stale_counters_after_empty_unsupported_or_capacity_fallback(scorers):
    old, new = scorers
    oracle, p, hs = zero_residual_case()
    check_batch(scorers, hs, p, oracle, "gpu")
    assert new.last_skipped.tolist() == [2]
    before = new.counts()
    check_batch(scorers, [], p, oracle, "gpu")
    assert new.counts() == before and new.last_skipped.size == 0
    p.Ppos = p.Ppos.astype(np.float32)
    check_batch(scorers, hs, p, oracle, "gpu")
    assert new.counts() == before and new.last_skipped.size == 0
    oracle, p, hs = make_case(8, duplicates=False)
    with (
        BatchScorer(os.environ["GNSS_GPU_RBPF_PROBE_DLL"], capacity=1) as old_limited,
        IdentityBatchScorer(
            os.environ["GNSS_GPU_RBPF_IDENTITY_DLL"], capacity=1
        ) as limited,
    ):
        va, ma = old_limited.score(hs, p, oracle, "gpu")
        vb, mb = limited.score(hs, p, oracle, "gpu")
        exact(va, vb)
        exact(ma, mb)
        assert mb["unsupported"] and mb["scalar_fallbacks"] == 8
        assert limited.last_skipped.size == 0 and not any(limited.counts().values())


@pytest.mark.parametrize("mode", ["scalar", "native_cpu", "gpu"])
@pytest.mark.parametrize("seed", [20260710, 20260711, 20260712])
@pytest.mark.parametrize(
    "options",
    [
        {},
        dict(cluster=True),
        dict(shadow=True, spawn=True),
        dict(feedback=False, preload=False, resample=False),
    ],
)
def test_four_epoch_all_state_metadata_rng_and_graph_events(
    extensions, scorers, mode, seed, options
):
    a, ra, ca = make_feedback_case(seed=seed, **options)
    b, rb, cb = make_feedback_case(seed=seed, **options)
    old = NativeFeedbackMassStep(a, scorers[0], mode, **extensions)
    call = IRLSIdentityStep(b, scorers[1], mode, **extensions)
    a.step, b.step = old, call
    for ep in range(4):
        ra.epoch2 = rb.epoch2 = ep
        if options.get("feedback", True):
            _bind_fixture_callbacks(ra)
            _bind_fixture_callbacks(rb)
        oa, ob = old(ra, deepcopy(ca)), call(rb, deepcopy(cb))
        exact(state(a, ra, oa), state(b, rb, ob))
        exact(old.last, {k: v for k, v in call.last.items() if k != "irls_identity"})
        s = call.last["irls_identity"]
        if mode == "scalar":
            assert not any(s.values())
        else:
            assert (
                s["native_batches"] == 1
                and s["native_hypotheses"] == old.last["batch"]["unique"]
            )
            assert 0 <= s["identical_hypotheses"] <= s["native_hypotheses"]
            assert (
                s["identical_hypotheses"]
                <= s["skipped_normal_solves"]
                <= 2 * s["identical_hypotheses"]
            )


def test_strict_policy_preserves_original_partial_state(extensions, scorers):
    a, ra, ca = make_feedback_case()
    b, rb, cb = make_feedback_case()
    call = IRLSIdentityStep(b, scorers[1], mode="gpu", **extensions)
    with np.errstate(under="raise"):
        for fn, rtk, ctx in ((a.step, ra, ca), (call, rb, cb)):
            with pytest.raises(FloatingPointError):
                fn(rtk, ctx)
    exact(state(a, ra, None), state(b, rb, None))
    assert call.last == dict(original_step_fallback="numpy_error_policy")
