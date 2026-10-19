"""Pinned setup/lifetime guards and exact original completed score behavior."""

from contextlib import ExitStack
from copy import deepcopy
import ctypes
import os
from pathlib import Path
import sys
import warnings

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
from rbpf_likelihood_cases import make_case  # noqa: E402
from rbpf_pinned_transport import PinnedTransportScorer, TracedPinnedTransportScorer  # noqa: E402
from rbpf_register_lu import RegisterLUScorer  # noqa: E402


@pytest.fixture(scope="module")
def paths():
    old, new = os.environ.get("GNSS_RBPF_BASELINE_DLL"), os.environ.get("GNSS_RBPF_PINNED_DLL")
    if not old or not new:
        pytest.skip("Explicit corrected original/pinned-candidate DLLs required")
    return old, new


@pytest.fixture(scope="module")
def scorers(paths):
    with ExitStack() as stack:
        yield [stack.enter_context(RegisterLUScorer(paths[0])),
            *[stack.enter_context(klass(paths[1], transport=transport))
                for klass in (PinnedTransportScorer, TracedPinnedTransportScorer)
                for transport in ("pageable", "pinned")]]


def exact(left, right):
    a, ad = left
    b, bd = right
    assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
    assert ad["unique"] == bd["unique"] and ad["scalar_fallbacks"] == bd["scalar_fallbacks"]
    assert ad.keys() == bd.keys()
    for key in ad.keys() - {"statuses", "diagnostic"}:
        assert ad[key] == bd[key]
    for key in ("statuses", "diagnostic"):
        assert (key in ad) == (key in bd)
        if key in ad:
            assert ad[key].dtype == bd[key].dtype and ad[key].shape == bd[key].shape and ad[key].tobytes() == bd[key].tobytes()


@pytest.mark.parametrize("value", [None, True, 1, "async", ""])
def test_invalid_selection_rejected_before_allocation(value):
    with pytest.raises(ValueError):
        PinnedTransportScorer("nonexistent.dll", transport=value)


@pytest.mark.parametrize("transport", ["pageable", "pinned"])
def test_selection_metadata_and_immutable_setup(paths, transport):
    with PinnedTransportScorer(paths[1], transport=transport) as scorer:
        assert scorer.transport == transport and sum(scorer.staging_bytes.values()) == 1_143_352
        select = scorer.dll.rbpf_host_staging_select
        assert select(scorer.handle, int(transport == "pinned")) == -2
        assert select(scorer.handle, 1-int(transport == "pinned")) == -2
        assert select(scorer.handle, 7) == -1 and select(None, 0) == -1
        validate = scorer.dll.rbpf_host_staging_validate
        validate.argtypes, validate.restype = [ctypes.c_void_p], ctypes.c_int
        assert validate(scorer.handle) == int(transport == "pinned")
        data = np.full(5, 123, dtype=np.uint64)
        info = scorer.dll.rbpf_host_staging_info
        assert info(scorer.handle, data, 1) == -1 and np.all(data == 123)
        assert info(None, data, 5) == -1 and np.all(data == 123)


@pytest.mark.parametrize("seed,rows,groups,cluster", [(20260710,16,1,False), (20260711,64,4,False),
    (20260712,128,8,False), (20260710,16,1,True), (20260711,64,4,True), (20260712,128,8,True)])
def test_all_numeric_bits_CPU_GPU(scorers, seed, rows, groups, cluster):
    oracle, problem, hypotheses = make_case(8, rows=rows, groups=groups, seed=seed, cluster=cluster)
    for mode in ("native_cpu", "gpu"):
        expected = scorers[0].score(hypotheses, problem, oracle, mode, diagnostics=True)
        for scorer in scorers[1:]:
            exact(expected, scorer.score(hypotheses, problem, oracle, mode, diagnostics=True))
        assert all(s.last_native_trace["full_GPU_path"] is (mode == "gpu") for s in scorers[3:])


@pytest.mark.parametrize("cluster", [0, 1])
def test_every_saved_actual_input_exact(scorers, cluster):
    from bench_rbpf_irls_replay import digest, prepare
    from rbpf_irls_replay import load_corpus, oracle_from_snapshot
    corpus = load_corpus(ROOT / f"dist/rbpf_irls_replay_capture_20261016/r3_seed0_capture_cpu_c{cluster}.corpus.jsonl.gz")
    assert len(corpus) == 197
    oracle = oracle_from_snapshot(corpus[0]["inputs"], ROOT / os.environ["GNSS_RBPF_CANONICAL_EXT"], ROOT / os.environ["GNSS_RBPF_PARENT_EXT"])
    for entry in corpus:
        hypotheses, problem = prepare(oracle, entry["inputs"])
        data = (hypotheses, problem, oracle.cfg, oracle._cluster_state.__self__.cluster_states)
        before = digest(data)
        expected = scorers[0].score(hypotheses, problem, oracle, "gpu", diagnostics=True)
        for scorer in scorers[1:]:
            exact(expected, scorer.score(hypotheses, problem, oracle, "gpu", diagnostics=True))
        assert before == digest(data)


@pytest.mark.parametrize("rows,count", [(0,1), (1,3), (63,7), (127,21)])
def test_alignment_changing_shapes_and_return_ownership(scorers, rows, count):
    oracle, problem, hypotheses = make_case(count, rows=128, groups=8)
    for name, value in vars(problem).items():
        if name.startswith("rows_"):
            setattr(problem, name, value[:rows].copy())
    expected = scorers[0].score(hypotheses, problem, oracle, "gpu", diagnostics=True)
    for scorer in scorers[1:]:
        result = scorer.score(hypotheses, problem, oracle, "gpu", diagnostics=True)
        exact(expected, result)
        saved = result[0].tobytes()
        scorer.score(hypotheses[:1], problem, oracle, "gpu")
        assert result[0].tobytes() == saved


@pytest.mark.parametrize("kind", ["dtype", "capacity", "rows", "groups", "empty", "singular", "strict", "warning", "invalid"])
def test_original_fallbacks_preserved(scorers, kind):
    oracle, problem, hypotheses = make_case(8, rows=130 if kind == "rows" else 64,
        groups=9 if kind == "groups" else 4)
    if kind == "dtype":
        problem.rows_J = problem.rows_J.astype(np.float32)
    elif kind == "capacity":
        for scorer in scorers:
            scorer.capacity = 1
    elif kind == "empty":
        hypotheses = []
    elif kind == "singular":
        problem.Ppos[:] = 0
    elif kind == "invalid":
        oracle.cfg.mix_eps = 0
    elif kind == "strict":
        hypotheses = hypotheses[:2]
    try:
        outcomes = []
        for scorer in scorers:
            with np.errstate(all="raise" if kind == "strict" else "warn", under="warn" if kind == "warning" else "ignore"):
                with warnings.catch_warnings(record=True) as seen:
                    warnings.simplefilter("always")
                    try:
                        result = scorer.score(hypotheses, problem, oracle, "gpu")
                    except (ValueError, FloatingPointError) as error:
                        result, failure = None, (type(error), str(error))
                    else:
                        failure = None
                    outcomes.append((result, failure, [(v.category, str(v.message)) for v in seen]))
        for result, failure, seen in outcomes[1:]:
            assert (failure, seen) == outcomes[0][1:]
            if result is not None:
                exact(outcomes[0][0], result)
    finally:
        for scorer in scorers:
            scorer.capacity = 512


@pytest.mark.parametrize("transport", ["pageable", "pinned"])
def test_allocation_lifecycle_and_double_close(paths, transport):
    oracle, problem, hypotheses = make_case(4, rows=16, groups=1)
    for _ in range(12):
        scorer = PinnedTransportScorer(paths[1], capacity=8, transport=transport)
        result = scorer.score(hypotheses, problem, oracle, "gpu")
        assert len(result[0]) == len(hypotheses)
        scorer.close()
        scorer.close()
        assert scorer.handle is None


@pytest.mark.parametrize("candidate,AA,expected", [
    ([80]*5, [100]*5, True), ([95]*5, [100]*5, False),
    ([80]*5, [120]*5, False), ([80,110,80,80,80], [100]*5, True),
    ([80,80,80,110,110], [100]*5, False), ([80]*5, [100,100,100,110,110], False)])
def test_gate_keeps_losing_rounds_and_rejects_inconsistent_controls(candidate, AA, expected):
    from bench_rbpf_pinned_transport import contrasts
    totals = {(rd,0,mode):value for rd in range(5)
        for mode,value in (("old",100),("pageable",100),("pinned",candidate[rd]),("old_AA",AA[rd]))}
    result = contrasts(totals,1)
    assert result["component_screen_passed"] is expected
    assert result["contrasts"]["pinned_vs_old"]["round_reduction_percent"] == pytest.approx([100-v for v in candidate])
    assert result["contrasts"]["old_AA"]["round_reduction_percent"] == pytest.approx([100-v for v in AA])


def test_unmarked_candidate_reuses_original_score_code():
    assert PinnedTransportScorer.score.__code__ is RegisterLUScorer.score.__code__
