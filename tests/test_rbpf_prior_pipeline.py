"""Earlier prior work must retain every operation and each RNG draw."""

from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
import test_rbpf_discovery_pipeline as checked  # noqa: E402
from rbpf_prior_pipeline import PriorPipelineStep, AST_PROOF, _PREPARE  # noqa: E402
from rbpf_feedback_cases import make_feedback_case  # noqa: E402
from rbpf_move_cases import state  # noqa: E402
from rbpf_move_plan_validation import check_metadata  # noqa: E402

deps, scorers = checked.deps, checked.scorers
exact, controller = checked.exact, checked.controller


def prior_call(inputs, deps, scorer, backend="gpu"):
    runtime, rtk, ctx = inputs
    from bench_rbpf_feedback_mass import _bind_fixture_callbacks
    _bind_fixture_callbacks(rtk)
    call = PriorPipelineStep(runtime, scorer, backend, extension=ROOT / deps["CANONICAL_EXT"],
        parent_extension=ROOT / deps["PARENT_EXT"], mass_extension=ROOT / deps["MASS_EXT"])
    runtime.step = call
    return call


def test_original_prior_operations_and_rounding_bits():
    assert AST_PROOF["original_moves_AST_restored_exact"]
    assert not AST_PROOF["probability_or_normalizer_cache"]
    for priors in ([.25, .1, .03, .002], [1e-100, 1e100, .17], [.1] * 4096):
        records = [(p, i, {}, "stay") for i, p in enumerate(priors)]
        pri = np.array(priors, dtype=float)
        pri /= pri.sum()
        exact(np.log(pri), _PREPARE(records))


@pytest.mark.parametrize("backend", ("scalar", "native_cpu", "gpu"))
@pytest.mark.parametrize("cluster", (False, True))
@pytest.mark.parametrize("seed", (20260710, 20260713))
def test_four_full_epochs_prior_work_exact(deps, scorers, backend, cluster, seed):
    old, active, blocked = scorers
    initial = make_feedback_case(64, 6, cluster=cluster, seed=seed, shadow=True, spawn=True)
    inputs = [deepcopy(initial) for _ in range(3)]
    calls = [controller(inputs[0], deps, old, backend, "original"),
        prior_call(inputs[1], deps, blocked, backend), prior_call(inputs[2], deps, active, backend)]
    for epoch in range(4):
        snapshots, metadata = [], []
        for x, call in zip(inputs, calls, strict=True):
            x[1].epoch2 = epoch
            out = call(x[1], x[2])
            snapshots.append(state(*x[:2], out))
            metadata.append(deepcopy(call.last))
        for value in snapshots[1:]:
            exact(snapshots[0], value)
        for value in metadata[1:]:
            prior = value.pop("prior_pipeline")
            pipeline = value.pop("discovery_pipeline")
            check_metadata(metadata[0], value)
            assert prior["prepared_particles"] == 64 and prior["literal_weight_calls"] == 0
            assert prior["prepared_records"] == value["requested_moves"]
            assert pipeline["complete_launches"] == int(backend == "gpu")
        assert all("_pipeline_priors" not in c._staging_clones[0].__globals__ for c in calls[1:])


@pytest.mark.parametrize("kind", ("custom_rng", "unsafe_prior", "numpy_hook", "profile"))
def test_unsupported_prior_schedule_uses_literal_weights(deps, scorers, monkeypatch, kind):
    old, active, _ = scorers
    initial = make_feedback_case(16, 4, resample=False)
    inputs = [deepcopy(initial), deepcopy(initial)]
    calls = [controller(inputs[0], deps, old, kind="plan"), prior_call(inputs[1], deps, active)]
    if kind == "custom_rng":
        class Rng:
            def choice(self, n, p):
                p[:] = 0.0
                p[0] = 1.0
                return 0
            def random(self):
                return .25
        for x in inputs:
            x[0].rng = Rng()
    elif kind == "unsafe_prior":
        for x in inputs:
            x[0].cfg.p_adopt = 1e110
    elif kind == "numpy_hook":
        original = np.log
        monkeypatch.setattr(np, "log", lambda *a, **kw: original(*a, **kw))
    def profile(frame, event, arg):
        pass
    previous = sys.getprofile()
    if kind == "profile":
        sys.setprofile(profile)
    try:
        outputs = []
        for x, call in zip(inputs, calls, strict=True):
            out = call(x[1], x[2])
            outputs.append(([(p.H, p.logw, p.last_move) for p in x[0].particles], out))
        exact(outputs[0], outputs[1])
        note = calls[1].last["prior_pipeline"]
        assert note["prepared_particles"] == 0 and note["literal_weight_calls"] == 1
    finally:
        if kind == "profile":
            assert sys.getprofile() is profile
            sys.setprofile(previous)
