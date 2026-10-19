"""Same-backend numeric/state bits,complete batches,ownership and draining."""

from copy import deepcopy
import gc
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
from bench_rbpf_feedback_mass import _bind_fixture_callbacks  # noqa: E402
from rbpf_async_score import AsyncScorer, AST_PROOF as SCORE_PROOF  # noqa: E402
from rbpf_discovery_pipeline import DiscoveryPipelineStep, AST_PROOF  # noqa: E402
from rbpf_feedback_cases import make_feedback_case  # noqa: E402
from rbpf_feedback_mass import NativeFeedbackMassStep  # noqa: E402
from rbpf_likelihood_cases import make_case  # noqa: E402
from rbpf_move_cases import comparison, state  # noqa: E402
from rbpf_move_plan_checked import MovePlanStep  # noqa: E402
from rbpf_move_plan_validation import check_metadata  # noqa: E402
from rbpf_register_lu import RegisterLUScorer  # noqa: E402


@pytest.fixture(scope="module")
def deps():
    old = json.loads((ROOT / "dist/rbpf_move_plan_checked_tests_20261018.plan.json").read_text(encoding="utf8"))["extensions"]
    build = json.loads((ROOT / "dist/rbpf_async_score_build_20261018.json").read_text(encoding="utf8"))
    return dict(old, ASYNC_DLL=build["dll_path"])


@pytest.fixture(scope="module")
def scorers(deps):
    with RegisterLUScorer(ROOT / deps["DLL"]) as old, AsyncScorer(ROOT / deps["ASYNC_DLL"]) as active, AsyncScorer(ROOT / deps["ASYNC_DLL"], blocked=True) as blocked:
        yield old, active, blocked


def exact(a, b):
    value = comparison(a, b)
    assert value["discrete_equal"] and value["numeric_bits_equal"] and value["max_abs_error"] == 0, value


def controller(inputs, deps, scorer, backend="gpu", kind="pipeline"):
    runtime, rtk, _ = inputs
    _bind_fixture_callbacks(rtk)
    cls = {"original": NativeFeedbackMassStep, "plan": MovePlanStep, "pipeline": DiscoveryPipelineStep}[kind]
    call = cls(runtime, scorer, backend, extension=ROOT / deps["CANONICAL_EXT"],
        parent_extension=ROOT / deps["PARENT_EXT"], mass_extension=ROOT / deps["MASS_EXT"])
    runtime.step = call
    return call


def test_AST_recovery_and_original_numeric_prefix(deps):
    assert SCORE_PROOF["original_score_AST_restored_exact"] and AST_PROOF["original_moves_AST_restored_exact"]
    old = (ROOT / "benchmarks/probes/rbpf_register_lu_synced/probe.cu").read_bytes()
    new = (ROOT / "benchmarks/probes/rbpf_async_score/probe.cu").read_bytes()
    assert new.startswith(old)


@pytest.mark.parametrize("shape", ((16, 2, False), (64, 4, False), (128, 8, True)))
@pytest.mark.parametrize("policy", ((True, True), (False, True), (True, False)))
def test_score_bits_and_full_diagnostics(scorers, shape, policy):
    old, active, blocked = scorers
    rows, groups, cluster = shape
    oracle, problem, hypotheses = make_case(8, rows=rows, groups=groups, cluster=cluster,
        marg_uniform=policy[0], huber=policy[1])
    expected = old.score(hypotheses, problem, oracle, "gpu", diagnostics=True)
    for scorer in (active, blocked):
        exact(expected, scorer.score(hypotheses, problem, oracle, "gpu", diagnostics=True))
        job = scorer.begin(hypotheses, problem, oracle, diagnostics=True)
        assert scorer.active is job
        actual = scorer.finish(job)
        exact(expected, actual)
        assert scorer.active is None


@pytest.mark.parametrize("backend", ("scalar", "native_cpu", "gpu"))
@pytest.mark.parametrize("cluster", (False, True))
@pytest.mark.parametrize("seed", (20260710, 20260713))
def test_four_full_epochs_exact(deps, scorers, backend, cluster, seed):
    old, active, blocked = scorers
    initial = make_feedback_case(64, 6, cluster=cluster, seed=seed, shadow=True, spawn=True)
    inputs = [deepcopy(initial) for _ in range(4)]
    calls = [controller(x, deps, s, backend, kind) for x, s, kind in zip(inputs,
        (old, old, blocked, active), ("original", "plan", "pipeline", "pipeline"), strict=True)]
    policy = gc.isenabled(), gc.get_threshold()
    for epoch in range(4):
        states, metadata = [], []
        for x, call in zip(inputs, calls, strict=True):
            x[1].epoch2 = epoch
            out = call(x[1], x[2])
            states.append(state(*x[:2], out))
            metadata.append(deepcopy(call.last))
        for value in states[1:]:
            exact(states[0], value)
        check_metadata(metadata[0], metadata[1])
        for value in metadata[2:]:
            pipeline = value.pop("discovery_pipeline")
            check_metadata(metadata[0], value)
            assert pipeline["discoveries"] == 1 and pipeline["drains"] == 0
            assert pipeline["complete_launches"] == pipeline["finishes"] == int(backend == "gpu")
            assert pipeline["records_after_begin"] == (value["requested_moves"] if backend == "gpu" else 0)
        assert all(c._view is c._oracle is None for c in calls)
    assert policy == (gc.isenabled(), gc.get_threshold())


@pytest.mark.parametrize("variant", ("ordered", "huge", "aliases", "empty_groups"))
def test_complete_discovery_order_and_particle_record_ownership(deps, scorers, variant):
    old, active, _ = scorers
    initial = make_feedback_case(16, 4)
    if variant == "ordered":
        for p in initial[0].particles[::2]:
            p.H = dict(reversed([(g, dict(reversed(list(z.items())))) for g, z in p.H.items()]))
    elif variant == "huge":
        for p in initial[0].particles:
            p.H = {g: {s: z + (1 << 200) for s, z in zs.items()} for g, zs in p.H.items()}
    elif variant == "aliases":
        initial[0].particles[1].H = initial[0].particles[0].H
    else:
        for p in initial[0].particles[::2]:
            p.H[(99, 0)] = {}
    inputs = [deepcopy(initial), deepcopy(initial)]
    calls = [controller(inputs[0], deps, old, kind="plan"), controller(inputs[1], deps, active)]
    snapshots = []
    for x, call in zip(inputs, calls, strict=True):
        numeric = call.staging_engine.stage
        def capture(*args, numeric=numeric, call=call):
            held = numeric(*args)
            keys, hypotheses, lookup, pending, staged = args[-5:]
            for p, moves in zip(call.runtime.particles, staged, strict=True):
                assert all(h is p.H for _, _, h, tag in moves if tag in ("survive", "stay"))
            releases = [h for moves in staged for _, _, h, tag in moves if tag == "release"]
            assert len({id(h) for h in releases}) == len(releases)
            snapshots.append(deepcopy((keys, hypotheses, lookup, pending, staged, held)))
            if type(call) is DiscoveryPipelineStep:
                assert call.staging_engine.job is active.active and active.active is not None
                assert len(call.staging_engine.expected) == len(pending)
            return held
        call.staging_engine.stage = capture
        call(x[1], x[2])
    exact(snapshots[0], snapshots[1])


def test_inputs_snapshot_and_output_lifetime(deps):
    oracle, problem, hypotheses = make_case(6)
    with RegisterLUScorer(ROOT / deps["DLL"]) as old, AsyncScorer(ROOT / deps["ASYNC_DLL"]) as scorer:
        expected = old.score(hypotheses, problem, oracle, "gpu", diagnostics=True)
        job = scorer.begin(hypotheses, problem, oracle, diagnostics=True)
        # Native begin snapshots packed inputs before it returns. Targets are
        # written only at finish;all argument references are retained by job.
        for index in (1, 2, 3, 4, 5, 6):
            job.arguments[index].fill(0)
        actual = scorer.finish(job)
        exact(expected, actual)
        saved = deepcopy(actual)
        scorer.finish(scorer.begin(hypotheses, problem, oracle))
        exact(saved, actual)


def test_outstanding_reuse_cancel_and_close(deps):
    oracle, problem, hypotheses = make_case(6)
    scorer = AsyncScorer(ROOT / deps["ASYNC_DLL"])
    try:
        job = scorer.begin(hypotheses, problem, oracle)
        with pytest.raises(RuntimeError):
            scorer.begin(hypotheses, problem, oracle)
        with pytest.raises(RuntimeError):
            scorer.score(hypotheses, problem, oracle)
        assert scorer.dll.rbpf_async_begin(scorer.async_handle, *job.arguments[1:]) == -2
        assert scorer.cancel(job) == scorer.cancel(job) == 0
        with pytest.raises(ValueError):
            scorer.finish(job)
        assert scorer.dll.rbpf_async_finish(scorer.async_handle) == -1
        job = scorer.begin(hypotheses, problem, oracle)
        scorer.close()
        assert job.closed and scorer.active is None and not scorer.async_handle and not scorer.handle
        scorer.close()
    finally:
        scorer.close()


def test_stage_exception_drains_before_scope_and_workspace_reuse(deps, scorers):
    _, active, _ = scorers
    inputs = make_feedback_case(16, 4)
    call = controller(inputs, deps, active)
    original = call.staging_engine.stage
    error = ValueError("materialization boundary")
    def fail(*args):
        original(*args)
        assert active.active is not None
        raise error
    call.staging_engine.stage = fail
    before = active.counts["drains"]
    with pytest.raises(ValueError) as caught:
        call(inputs[1], inputs[2])
    assert caught.value is error and active.counts["drains"] == before + 1
    assert active.active is None and call.staging_engine.job is None
    assert call._view is call._oracle is None
    assert "_pipeline_score" not in call._staging_clones[0].__globals__


def test_empty_cpu_and_unsupported_capacity_keep_original_fallback(deps):
    oracle, problem, hypotheses = make_case(6)
    with AsyncScorer(ROOT / deps["ASYNC_DLL"], capacity=1) as scorer:
        for requested, mode in (([], "gpu"), (hypotheses, "gpu"), (hypotheses, "native_cpu")):
            expected = scorer.score(requested, problem, oracle, mode)
            job = scorer.begin(requested, problem, oracle, mode)
            exact(expected, scorer.finish(job))
        assert scorer.counts["begins"] == 0


def test_native_submission_error_runs_original_scalar_fallback_once(deps, monkeypatch):
    oracle, problem, hypotheses = make_case(6)
    with AsyncScorer(ROOT / deps["ASYNC_DLL"]) as scorer:
        monkeypatch.setattr(scorer.dll, "rbpf_async_begin", lambda *args: -99)
        value, detail = scorer.finish(scorer.begin(hypotheses, problem, oracle))
        exact(value, scorer.scalar(hypotheses, problem, oracle))
        assert detail["native_error"] == -99 and scorer.active is None


def test_strict_error_and_skip_preserve_original_partial_state(deps, scorers):
    old, active, _ = scorers
    for strict in (False, True):
        initial = make_feedback_case(8, 4)
        inputs = [deepcopy(initial), deepcopy(initial)]
        calls = [controller(inputs[0], deps, old, kind="original"), controller(inputs[1], deps, active)]
        if not strict:
            for x in inputs:
                x[2]["problem"].ok = False
                x[2]["problem"].reason = "skip"
        results, errors = [], []
        before = active.counts["begins"]
        with np.errstate(all="raise") if strict else np.errstate():
            for x, call in zip(inputs, calls, strict=True):
                out = None
                try:
                    out = call(x[1], x[2])
                    errors.append(None)
                except FloatingPointError as error:
                    errors.append((type(error).__name__, str(error)))
                results.append((state(*x[:2], out), call.last))
        exact(results[0], results[1])
        assert errors[0] == errors[1] and bool(errors[0]) == strict
        assert active.counts["begins"] == before
