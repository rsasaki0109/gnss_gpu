"""Coarse-only spans preserve all original PF state and caller metadata."""

from copy import deepcopy
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
from bench_rbpf_feedback_mass import _bind_fixture_callbacks  # noqa: E402
from bench_rbpf_process_profile_field import ProcessProfileStep  # noqa: E402
from rbpf_feedback_cases import make_feedback_case  # noqa: E402
from rbpf_feedback_mass import NativeFeedbackMassStep  # noqa: E402
from rbpf_move_cases import state  # noqa: E402
import test_rbpf_call_profile as support  # noqa: E402
from test_rbpf_native_staging import exact  # noqa: E402

extensions, scorer = support.extensions, support.scorer


@pytest.mark.parametrize("mode", ["scalar", "native_cpu", "gpu"])
@pytest.mark.parametrize("seed", [20260710, 20260711])
@pytest.mark.parametrize(
    "options",
    [
        {},
        dict(cluster=True),
        dict(shadow=True, spawn=True),
        dict(feedback=False, preload=False, resample=False),
    ],
)
def test_four_epochs_without_function_tracing(extensions, scorer, mode, seed, options):
    a, ra, ca = make_feedback_case(seed=seed, **options)
    b, rb, cb = make_feedback_case(seed=seed, **options)
    old = NativeFeedbackMassStep(a, scorer, mode, **extensions)
    new = ProcessProfileStep(b, scorer, mode, **extensions)
    a.step, b.step = old, new
    for epoch in range(4):
        ra.epoch2 = rb.epoch2 = epoch
        if options.get("feedback", True):
            _bind_fixture_callbacks(ra)
            _bind_fixture_callbacks(rb)
        oa, ob = old(ra, deepcopy(ca)), new(rb, deepcopy(cb))
        exact(state(a, ra, oa), state(b, rb, ob))
        exact(old.last, new.last)
        assert sys.getprofile() is None and new.monitor.current is None
    assert len(new.monitor.records) == 4
    assert all(set(r["spans"]) == {"pf"} for r in new.monitor.records)
