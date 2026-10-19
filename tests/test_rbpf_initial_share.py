"""Initial-system sharing preserves all score/diagnostic/PF/RNG bits."""

from contextlib import ExitStack
from copy import deepcopy
import ctypes
import hashlib
import os
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
from rbpf_initial_share import InitialShareScorer  # noqa: E402
from rbpf_register_lu import RegisterLUScorer  # noqa: E402
from rbpf_likelihood_cases import make_case  # noqa: E402
from rbpf_irls_replay import load_corpus, oracle_from_snapshot  # noqa: E402
from bench_rbpf_irls_replay import prepare, digest  # noqa: E402
from test_rbpf_register_lu import (  # noqa: E402,F401
    exact,
    test_all_numeric_bits_same_backend,
    test_alignment_empty_rows_and_completed_host_readback,
    test_original_fallbacks_errors_warnings,
    test_explicit_selector_rejects_invalid_modes_without_affecting_score,
    test_complete_pf_four_epochs_all_saved_state_rng_and_graph_events,
)


@pytest.fixture(scope="module")
def scorers():
    old, new = (
        os.environ.get(k)
        for k in ("GNSS_GPU_RBPF_PROBE_DLL", "GNSS_GPU_RBPF_REGISTER_DLL")
    )
    if not old or not new:
        pytest.skip("Explicit old/candidate standalone DLLs required")
    with ExitStack() as stack:
        yield (
            stack.enter_context(RegisterLUScorer(old)),
            stack.enter_context(InitialShareScorer(new, enabled=False)),
            stack.enter_context(InitialShareScorer(new, enabled=True)),
        )


def full_case(count=16, *, huber=True):
    oracle, problem, hs = make_case(
        2, rows=128, groups=8, duplicates=False, huber=huber
    )
    h = hs[1]
    g = next(iter(h))
    sat = list(h[g])[1]
    result = []
    for i in range(count):
        candidate = deepcopy(h)
        candidate[g][sat] += i
        result.append(candidate)
    return oracle, problem, result


@pytest.mark.parametrize("huber", [False, True])
@pytest.mark.parametrize(
    "kind", ["same", "centres", "prior_ulp", "prior_signed_zero", "support"]
)
def test_actual_shared_path_and_bitwise_key_separation(scorers, huber, kind):
    oracle, problem, hs = full_case(huber=huber)
    if kind == "support":
        items = [(g, sat) for g, z in hs[0].items() for sat in list(z)[1:]]
        for i, h in enumerate(hs):
            g, sat = items[i]
            del h[g][sat]
    elif kind in ("centres", "prior_ulp", "prior_signed_zero"):
        oracle.cfg.cluster_relin = 1
        base = np.diag([2.0, 3.0, 4.0]) if kind == "prior_signed_zero" else problem.Ppos
        for i, h in enumerate(hs):
            prior = base.copy()
            if kind == "prior_ulp":
                for _ in range(i):
                    prior[0, 0] = np.nextafter(prior[0, 0], np.inf)
            if kind == "prior_signed_zero":
                prior[0, 1] = -0.0 if i % 2 else 0.0
            oracle.cluster_states[oracle.basin_id(h)] = dict(
                H=deepcopy(h),
                P=prior,
                pos=problem.pos + np.array([i * 0.01, -0.02, 0.03]),
            )
    before = digest((hs, problem, oracle.cluster_states))
    expected = scorers[0].score(hs, problem, oracle, "gpu", diagnostics=True)
    for selected in scorers[1:]:
        exact(expected, selected.score(hs, problem, oracle, "gpu", diagnostics=True))
        usage = selected.usage()
        assert usage["count"] == len(hs)
        if selected.enabled and kind not in ("prior_ulp", "support"):
            assert usage["duplicates"] > 0 and usage["members"] == len(hs)
            assert usage["launches"] == 2 and np.all(usage["used"] == 1)
            assert np.all(
                (usage["owners"] >= 0) & (usage["owners"] <= np.arange(len(hs)))
            )
            if kind == "prior_signed_zero":
                assert usage["owners"][0] != usage["owners"][1]
        else:
            assert not np.any(usage["used"])
            assert usage["duplicates"] == 0 and usage["launches"] == 1
    assert before == digest((hs, problem, oracle.cluster_states))


def raw_inputs():
    oracle, p, hs = full_case(8)
    rows = np.ascontiguousarray(
        np.column_stack((p.rows_J, p.rows_v0, p.rows_sig, p.rows_lam, p.rows_bmeas))
    )
    groups = {}
    meta = np.empty((len(rows), 3), np.int32)
    for r, cp in enumerate(p.rows_iscp):
        pi = int(p.rows_pair[r])
        g = p.pairs[pi][0] if cp else None
        if cp and g not in groups:
            groups[g] = len(groups)
        meta[r] = int(cp), pi, groups[g] if cp else -1
    pairs = np.ascontiguousarray(np.column_stack((p.y, np.diag(p.Qb))))
    dd = np.ascontiguousarray([oracle._pair_dd(h, p) for h in hs])
    parameters = np.tile(np.r_[np.zeros(3), p.Ppos.ravel()], (len(hs), 1))
    cfg = np.array(
        [
            oracle.cfg.beta,
            oracle.cfg.marg_uniform,
            oracle.cfg.huber_c,
            oracle.cfg.sig_group,
            oracle.cfg.mix_eps,
            oracle.cfg.mix_sigma_cp,
            oracle.cfg.mix_sigma_pr,
        ]
    )
    return rows, meta, pairs, dd, parameters, cfg


@pytest.mark.parametrize(
    "kind",
    [
        "ordinary",
        "rhs_nan",
        "rhs_inf",
        "rhs_huge",
        "one_rhs_nan",
        "one_rhs_huge",
        "matrix_overflow",
        "prior_singular",
        "prior_nan",
        "no_irls",
        "zero_rhs",
        "nb65",
        "nb128",
        "signed_zero",
        "tiny_prior",
        "empty_rows",
    ],
)
def test_raw_partial_failure_guard_alignment_and_rhs_bits(scorers, kind):
    rows, meta, pairs, dd, parameters, cfg = raw_inputs()
    if kind in ("rhs_nan", "rhs_inf", "rhs_huge"):
        rows[0, 3] = dict(rhs_nan=np.nan, rhs_inf=np.inf, rhs_huge=1e300)[kind]
    elif kind in ("one_rhs_nan", "one_rhs_huge"):
        parameters[-1, 0] = np.nan if kind == "one_rhs_nan" else 1e300
    elif kind == "matrix_overflow":
        rows[:, :3] *= 2.0**600
    elif kind in ("prior_singular", "prior_nan", "tiny_prior"):
        prior = (
            np.zeros((3, 3))
            if kind == "prior_singular"
            else np.eye(3) * (np.nan if kind == "prior_nan" else 5e-324)
        )
        parameters[:, 3:] = prior.ravel()
    elif kind == "no_irls":
        cfg[2] = 0.0
    elif kind == "zero_rhs":
        rows[:, 3] = rows[:, 6] = 0.0
        dd[:] = 0.0
    elif kind in ("nb65", "nb128"):
        extra = (65 if kind == "nb65" else 128) - dd.shape[1]
        pairs = np.ascontiguousarray(np.vstack((pairs, np.zeros((extra, 2)))))
        dd = np.ascontiguousarray(
            np.column_stack((dd, np.full((len(dd), extra), np.nan)))
        )
    elif kind == "signed_zero":
        parameters[::2, :3] = -0.0
    elif kind == "empty_rows":
        rows, meta = rows[:0].copy(), meta[:0].copy()
    saved = [x.copy() for x in (rows, meta, pairs, dd, parameters, cfg)]
    outcomes = []
    for selected in scorers:
        output, statuses, diagnostics = (
            np.empty(len(dd)),
            np.empty(len(dd), np.int32),
            np.empty((len(dd), 135)),
        )
        assert (
            selected.dll.rbpf_score(
                selected.handle,
                rows,
                meta,
                pairs,
                dd,
                parameters,
                cfg,
                len(dd),
                len(rows),
                dd.shape[1],
                1,
                output,
                statuses,
                ctypes.c_void_p(diagnostics.ctypes.data),
            )
            == 0
        )
        outcomes.append(
            (
                output,
                dict(
                    unique=len(dd),
                    scalar_fallbacks=0,
                    statuses=statuses,
                    diagnostic=diagnostics,
                ),
            )
        )
    for result in outcomes[1:]:
        exact(outcomes[0], result)
    for x, y in zip((rows, meta, pairs, dd, parameters, cfg), saved, strict=True):
        assert x.tobytes() == y.tobytes()
    usage = scorers[-1].usage()
    if kind in ("one_rhs_nan", "one_rhs_huge"):
        assert usage["used"][-1] == 0 and np.all(usage["used"][:-1] == 1)
    if kind in (
        "rhs_nan",
        "rhs_inf",
        "rhs_huge",
        "matrix_overflow",
        "prior_singular",
        "prior_nan",
        "tiny_prior",
        "empty_rows",
    ):
        assert not np.any(usage["used"])


def test_no_cross_call_cache_owned_results_and_invalid_api(scorers):
    _, off, on = scorers
    assert on.dll.rbpf_initial_share_enable(None, 0) == -1
    assert on.dll.rbpf_initial_share_enable(on.handle, -1) == -1
    assert on.dll.rbpf_initial_share_enable(on.handle, 2) == -1
    layout = on.memory_layout()
    assert layout == dict(template_bytes=1992, key_bytes=96, owner_bytes=4)
    for rows in (128, 64, 128, 16, 128):
        oracle, p, hs = (
            full_case()
            if rows == 128
            else make_case(8, rows=rows, groups=4 if rows == 64 else 1)
        )
        expected = off.score(hs, p, oracle, "gpu", diagnostics=True)
        result = on.score(hs, p, oracle, "gpu", diagnostics=True)
        exact(expected, result)
        previous = deepcopy(result)
        usage = on.usage()
        assert bool(np.any(usage["used"])) == (rows == 128)
        on.score(hs[:1], p, oracle, "native_cpu")
        assert on.usage()["launches"] == 0 and not np.any(on.usage()["used"])
        exact(previous, result)


def test_enabled_workspace_lifetime_and_returned_array_ownership():
    oracle, p, hs = full_case(8)
    for capacity in (8, 32, 512):
        for _ in range(8):
            owner = InitialShareScorer(
                os.environ["GNSS_GPU_RBPF_REGISTER_DLL"], capacity, enabled=True
            )
            result = owner.score(hs, p, oracle, "gpu", diagnostics=True)
            assert owner.usage()["duplicates"] > 0
            saved = deepcopy(result)
            owner.close()
            owner.close()
            exact(saved, result)


@pytest.mark.parametrize("cluster", [0, 1])
def test_all_197_actual_batches_diagnostics_inputs_and_gpu_use(scorers, cluster):
    path = (
        ROOT
        / f"dist/rbpf_irls_replay_capture_20261016/r3_seed0_capture_cpu_c{cluster}.corpus.jsonl.gz"
    )
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "7172b4db3f50435f24c19c92056d22baf436442cf3315a02ee18a46271128e66"
        if cluster == 0
        else "75d747605bde5bc389d14ce07ed34e3163d5c0f2b0d42f2348e54c7e2183ba18"
    )
    corpus = load_corpus(path)
    assert len(corpus) == 197
    oracles = [
        oracle_from_snapshot(
            corpus[0]["inputs"],
            os.environ["GNSS_GPU_RBPF_CANONICAL_EXT"],
            os.environ["GNSS_GPU_RBPF_PARENT_EXT"],
        )
        for _ in scorers
    ]
    used = 0
    for entry in corpus:
        outcomes = []
        for selected, oracle in zip(scorers, oracles, strict=True):
            hs, p = prepare(oracle, entry["inputs"])
            before = digest(
                (hs, p, oracle.cfg, oracle._cluster_state.__self__.cluster_states)
            )
            outcomes.append(selected.score(hs, p, oracle, "gpu", diagnostics=True))
            assert before == digest(
                (hs, p, oracle.cfg, oracle._cluster_state.__self__.cluster_states)
            )
        for result in outcomes[1:]:
            exact(outcomes[0], result)
        used += int(scorers[-1].usage()["used"].sum())
    assert used > 0
