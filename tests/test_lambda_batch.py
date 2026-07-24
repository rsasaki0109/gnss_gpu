"""Tests for the batched LAMBDA/MLAMBDA CUDA module (requires CUDA GPU).

The kernel is a faithful batched port of cssrlib's ``mlambda.py``; the
authoritative parity evidence is the WP15 harness replaying 34,569 real
pipeline calls (see gnss_gpu/results/wp15/WP15_REPORT.md). The tests
here are self-contained: brute-force ILS cross-checks on small random
problems, PAR-path semantics, batch/single consistency, and (when
cssrlib and/or the captured-input npz are available) direct parity.
"""

from pathlib import Path

import numpy as np
import pytest

try:
    from gnss_gpu.lambda_batch import (
        HAS_LAMBDA_BATCH,
        MlambdaResult,
        lambda_batch_max_n,
        mlambda_batch,
    )
except ImportError:
    HAS_LAMBDA_BATCH = False

pytestmark = pytest.mark.skipif(
    not HAS_LAMBDA_BATCH, reason="lambda_batch CUDA module not available")

_CAPTURE_NPZ = Path(
    r"C:\Users\rsasa\Workspace\old\repro_tc_fgo\results\wp15"
    r"\mlambda_capture_r2_3000.npz")

try:
    from cssrlib.mlambda import mlambda as _cssrlib_mlambda
    HAS_CSSRLIB = True
except ImportError:
    HAS_CSSRLIB = False


def _random_problem(n, seed, scale=0.05):
    """A well-conditioned float-ambiguity problem."""
    rng = np.random.default_rng(seed)
    ahat = rng.normal(0.0, 5.0, n)
    A = rng.normal(0.0, 1.0, (n, n))
    Q = scale * (A @ A.T) + np.eye(n) * scale
    return ahat, Q


def _brute_force_ils(ahat, Q, radius=3):
    """Exhaustive ILS over an integer box around round(ahat)."""
    from itertools import product

    n = ahat.size
    Qinv = np.linalg.inv(Q)
    center = np.rint(ahat).astype(int)
    best = []
    for offs in product(range(-radius, radius + 1), repeat=n):
        z = center + np.array(offs)
        r = z - ahat
        best.append((float(r @ Qinv @ r), tuple(z)))
    best.sort()
    return best


def test_max_n_reported():
    assert lambda_batch_max_n() >= 36  # covers every observed pipeline n


def test_top24_candidate_count_for_wp29():
    ahat, Q = _random_problem(8, 2029)
    r = mlambda_batch([ahat], [Q], ncands=24, parmode=1)[0]
    assert r.status == 0
    assert r.afix.shape == (8, 24)
    assert r.s.shape == (24,)
    assert np.all(np.diff(r.s) >= 0.0)


def test_identity_covariance_rounds_to_nearest():
    ahat = np.array([1.2, -3.4, 0.6, 7.9])
    Q = np.eye(4) * 0.01
    r = mlambda_batch([ahat], [Q], ncands=2, parmode=1)[0]
    assert r.status == 0
    assert r.nfix == 4
    np.testing.assert_array_equal(r.afix[:, 0], np.rint(ahat))
    assert r.s[0] <= r.s[1]


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_matches_brute_force_ils(seed):
    n = 4
    ahat, Q = _random_problem(n, seed)
    r = mlambda_batch([ahat], [Q], ncands=2, parmode=1)[0]
    assert r.status == 0
    ref = _brute_force_ils(ahat, Q)
    np.testing.assert_array_equal(r.afix[:, 0], np.array(ref[0][1], float))
    np.testing.assert_array_equal(r.afix[:, 1], np.array(ref[1][1], float))
    np.testing.assert_allclose(r.s, [ref[0][0], ref[1][0]], rtol=1e-9)


def test_non_positive_definite_reports_status():
    n = 5
    ahat = np.zeros(n)
    Q = np.eye(n)
    Q[2, 2] = -1.0
    r = mlambda_batch([ahat], [Q], ncands=2, parmode=1)[0]
    assert r.status == 1  # cssrlib raises LambdaError here


def test_parmode2_accepts_confident_problem():
    ahat = np.array([1.1, -2.05, 3.02, 0.98])
    Q = np.eye(4) * 1e-4  # bootstrapped success rate ~1
    r = mlambda_batch([ahat], [Q], ncands=2, parmode=2, P0=0.995)[0]
    assert r.status == 0
    assert r.nfix == 4
    assert r.Ps > 0.995
    np.testing.assert_array_equal(r.afix[:, 0], np.rint(ahat))


def test_parmode2_rejects_hopeless_problem():
    rng = np.random.default_rng(7)
    n = 6
    ahat = rng.normal(0, 5, n)
    Q = np.eye(n) * 100.0  # bootstrapped success rate ~0
    r = mlambda_batch([ahat], [Q], ncands=2, parmode=2, P0=0.995)[0]
    assert r.status == 0
    assert r.nfix == 0
    assert r.s.size == 0        # cssrlib returns an empty list
    assert np.isnan(r.Ps)
    assert r.afix.ndim == 1     # cssrlib returns the 1-D float vector


def test_batch_equals_single_calls():
    problems = [_random_problem(n, seed)
                for seed, n in enumerate([3, 5, 8, 12, 20, 7, 4])]
    batch = mlambda_batch([a for a, _ in problems],
                          [q for _, q in problems], ncands=2, parmode=1)
    for (a, q), rb in zip(problems, batch):
        rs = mlambda_batch([a], [q], ncands=2, parmode=1)[0]
        assert rs.status == rb.status
        assert rs.nfix == rb.nfix
        np.testing.assert_array_equal(rs.afix, rb.afix)
        np.testing.assert_array_equal(rs.s, rb.s)


def test_rejects_non_finite_input():
    ahat = np.array([1.0, np.nan])
    Q = np.eye(2)
    with pytest.raises(ValueError):
        mlambda_batch([ahat], [Q])


def test_result_type():
    ahat, Q = _random_problem(4, 42)
    r = mlambda_batch([ahat], [Q])[0]
    assert isinstance(r, MlambdaResult)


@pytest.mark.skipif(not HAS_CSSRLIB, reason="cssrlib not installed")
@pytest.mark.parametrize("parmode", [1, 2])
def test_parity_vs_cssrlib_synthetic(parmode):
    """Parity with the CPU reference on random problems.

    The decorrelation search itself is bit-identical (--fmad=false +
    exact operation-order transcription); the small matrix products
    around it (zhat = Z.T @ ahat etc.) use naive accumulation vs BLAS
    on the CPU, so pure-float outputs may differ by ~1 ulp on random
    synthetic inputs. Integer outputs must match EXACTLY; floats to
    1e-9 relative. (On the 34,569 real pipeline calls captured for
    WP15 every output was bit-identical -- see WP15_REPORT.md.)
    """
    for seed in range(30):
        n = 3 + (seed % 14)
        ahat, Q = _random_problem(n, 1000 + seed,
                                  scale=0.02 + 0.01 * (seed % 5))
        r = mlambda_batch([ahat], [Q], ncands=2, parmode=parmode)[0]
        afix_c, s_c, nfix_c, ps_c = _cssrlib_mlambda(
            ahat, Q, ncands=2, parmode=parmode)
        assert r.status == 0
        assert r.nfix == nfix_c
        np.testing.assert_allclose(np.asarray(r.s),
                                   np.asarray(s_c, float), rtol=1e-9)
        afix_c = np.asarray(afix_c, float)
        afix_g = np.asarray(r.afix)
        assert afix_g.shape == afix_c.shape
        np.testing.assert_allclose(afix_g, afix_c, rtol=0, atol=1e-6)
        int_mask = afix_c == np.rint(afix_c)
        np.testing.assert_array_equal(afix_g[int_mask], afix_c[int_mask])
        if not (np.isnan(r.Ps) and np.isnan(ps_c)):
            assert r.Ps == pytest.approx(ps_c, rel=1e-9)


@pytest.mark.slow
@pytest.mark.skipif(not _CAPTURE_NPZ.exists(),
                    reason="WP15 captured-input npz not present")
def test_parity_vs_captured_pipeline_outputs():
    """Replay real pipeline (ahat, Qahat) inputs captured from a run2
    replay and require the CPU-recorded outputs to match exactly."""
    d = np.load(_CAPTURE_NPZ)
    n_arr = d["n"]
    parmode = d["parmode"]
    err = d["err"]
    nfix_cpu = d["nfix"]
    ahat_flat = d["ahat_flat"]
    q_flat = d["Q_flat"]
    afix_flat = d["afix_flat"]
    s_flat = d["s_flat"]

    ahat_l, q_l, afix_l, s_l = [], [], [], []
    ao = qo = fo = so = 0
    for i, n in enumerate(n_arr):
        n = int(n)
        ahat_l.append(ahat_flat[ao:ao + n])
        q_l.append(q_flat[qo:qo + n * n].reshape(n, n))
        if err[i]:
            afix_l.append(None)
            s_l.append(None)
        else:
            sz = n * 2 if (parmode[i] == 1 or nfix_cpu[i] > 0) else n
            a = afix_flat[fo:fo + sz]
            afix_l.append(a.reshape(n, 2) if sz == n * 2 else a)
            fo += sz
            s_n = 2 if (parmode[i] == 1 or nfix_cpu[i] > 0) else 0
            s_l.append(s_flat[so:so + s_n])
            so += s_n
        ao += n
        qo += n * n

    rng = np.random.default_rng(0)
    valid = [i for i in range(len(n_arr)) if not err[i]]
    sample = rng.choice(valid, size=min(2000, len(valid)), replace=False)
    for pm in (1, 2):
        ids = [i for i in sample if parmode[i] == pm]
        if not ids:
            continue
        res = mlambda_batch([ahat_l[i] for i in ids],
                            [q_l[i] for i in ids],
                            ncands=2, parmode=pm,
                            P0=float(d["P0"][ids[0]]))
        for i, r in zip(ids, res):
            assert r.status == 0
            assert r.nfix == nfix_cpu[i]
            if s_l[i] is not None and len(s_l[i]):
                np.testing.assert_array_equal(np.asarray(r.s), s_l[i])
            np.testing.assert_array_equal(np.asarray(r.afix),
                                          np.asarray(afix_l[i]))
