"""Continuous RTK bindings retain the tested estimator and owned GPU jobs."""

from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
import test_rbpf_discovery_pipeline as checked  # noqa: E402
from bench_rbpf_feedback_mass import _bind_fixture_callbacks  # noqa: E402
from rbpf_async_score import AsyncScorer  # noqa: E402
from rbpf_discovery_pipeline_route import make_scorer, make_step  # noqa: E402
from rbpf_discovery_pipeline_validation import check_pipeline_metadata  # noqa: E402
from rbpf_feedback_cases import make_feedback_case  # noqa: E402
from rbpf_move_cases import state  # noqa: E402
from rbpf_move_plan_route import VARIANTS  # noqa: E402
from rbpf_register_lu import RegisterLUScorer  # noqa: E402

deps = checked.deps


def test_continuous_rtk_arm_bindings_preserve_full_pf_state(deps):
    initial = make_feedback_case(64, 6, cluster=False)
    inputs = {v: deepcopy(initial) for v in VARIANTS}
    with ExitStack() as stack:
        calls, scorers = {}, {}
        for variant, x in inputs.items():
            asynchronous = variant in ("off", "candidate")
            scorer = stack.enter_context(make_scorer(ROOT / deps["ASYNC_DLL" if asynchronous else "DLL"], variant))
            assert type(scorer) is (AsyncScorer if asynchronous else RegisterLUScorer)
            if asynchronous:
                assert scorer.safe() and scorer.blocked is (variant == "off")
            call = make_step(x[0], scorer, "gpu", variant, extension=ROOT / deps["CANONICAL_EXT"],
                parent_extension=ROOT / deps["PARENT_EXT"], mass_extension=ROOT / deps["MASS_EXT"])
            x[0].step = call
            _bind_fixture_callbacks(x[1])
            calls[variant], scorers[variant] = call, scorer
        for epoch in range(2):
            states = {}
            for variant, x in inputs.items():
                x[1].epoch2 = epoch
                call = calls[variant]
                output = call(x[1], x[2])
                assert call.route_calls == epoch + 1 and call.route_last == (x[1], output)
                states[variant] = state(*x[:2], output)
            for variant in VARIANTS:
                checked.exact(states["reference"], states[variant])
            for variant in ("off", "candidate"):
                check_pipeline_metadata(calls["reference"].last, calls[variant].last)
                assert scorers[variant].counts == dict(begins=epoch + 1, completions=epoch + 1, drains=0, immediate=0, errors=0)


def test_invalid_route_arm_or_backend_is_rejected():
    runtime = SimpleNamespace(cfg=SimpleNamespace(cluster_relin=0))
    with pytest.raises(ValueError, match="Unknown"):
        make_scorer(None, "missing")
    for mode, variant in (("gpu", "missing"), ("scalar", "reference"), ("native_cpu", "candidate")):
        with pytest.raises(ValueError, match="declared"):
            make_step(runtime, None, mode, variant)
