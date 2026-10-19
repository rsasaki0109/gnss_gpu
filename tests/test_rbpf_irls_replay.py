"""On/off controls share one binary; both preserve original batch results."""

from copy import deepcopy
import os
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
from rbpf_irls_replay import ControlledIdentityScorer  # noqa: E402
from rbpf_likelihood_batch import BatchScorer  # noqa: E402
from rbpf_likelihood_cases import make_case  # noqa: E402
from test_rbpf_irls_identity import check_batch, exact, zero_residual_case  # noqa: E402


@pytest.fixture(scope="module", params=[False, True])
def scorers(request):
    old, new = (
        os.environ.get("GNSS_GPU_RBPF_PROBE_DLL"),
        os.environ.get("GNSS_GPU_RBPF_REPLAY_DLL"),
    )
    if not old or not new:
        pytest.skip("Explicit original and controlled standalone DLLs required")
    with (
        BatchScorer(old) as reference,
        ControlledIdentityScorer(new, enabled=request.param) as candidate,
    ):
        yield reference, candidate


@pytest.mark.parametrize("mode", ["native_cpu", "gpu"])
@pytest.mark.parametrize("rows,groups", [(16, 1), (64, 4), (128, 8)])
@pytest.mark.parametrize(
    "uniform,huber,cluster", [(1, 1, 0), (0, 1, 0), (1, 0, 0), (0, 0, 1), (1, 1, 1)]
)
def test_original_score_normal_solution_logdet_status_and_input_bits(
    scorers, mode, rows, groups, uniform, huber, cluster
):
    oracle, p, hs = make_case(
        8, rows=rows, groups=groups, marg_uniform=uniform, huber=huber, cluster=cluster
    )
    check_batch(scorers, hs, p, oracle, mode)
    if not scorers[1].enabled:
        assert not np.any(scorers[1].last_skipped)


@pytest.mark.parametrize("mode", ["native_cpu", "gpu"])
@pytest.mark.parametrize("rows,count", [(0, 1), (1, 3), (63, 7), (127, 21)])
def test_partial_blocks_and_empty_shapes(scorers, mode, rows, count):
    oracle, p, hs = make_case(count, rows=128, groups=8)
    for name, value in vars(p).items():
        if name.startswith("rows_"):
            setattr(p, name, value[:rows].copy())
    check_batch(scorers, hs, p, oracle, mode)


@pytest.mark.parametrize("mode", ["native_cpu", "gpu"])
def test_fixed_point_control_and_stale_fallback(scorers, mode):
    oracle, p, hs = zero_residual_case()
    check_batch(scorers, hs, p, oracle, mode)
    assert scorers[1].last_skipped.tolist() == [2 if scorers[1].enabled else 0]
    before = scorers[1].counts()
    check_batch(scorers, [], p, oracle, mode)
    assert scorers[1].counts() == before and not scorers[1].last_skipped.size
    p.Ppos = p.Ppos.astype(np.float32)
    check_batch(scorers, hs, p, oracle, mode)
    assert scorers[1].counts() == before and not scorers[1].last_skipped.size


@pytest.mark.parametrize("kind", ["singular", "asymmetric", "tiny", "huge"])
def test_prior_and_status_fallback_bits(scorers, kind):
    oracle, p, hs = make_case(8, duplicates=False)
    if kind == "singular":
        p.Ppos[:] = 0.0
    elif kind == "asymmetric":
        p.Ppos[0, 1] += 0.05
    else:
        p.Ppos[:] = np.eye(3) * (1e-300 if kind == "tiny" else 1e300)
    with np.errstate(all="ignore"):
        for mode in ("native_cpu", "gpu"):
            check_batch(scorers, hs, p, oracle, mode)


def test_capacity_same_scalar_path(scorers):
    oracle, p, hs = make_case(8, duplicates=False)
    with BatchScorer(os.environ["GNSS_GPU_RBPF_PROBE_DLL"], capacity=1) as old:
        for enabled in (False, True):
            with ControlledIdentityScorer(
                os.environ["GNSS_GPU_RBPF_REPLAY_DLL"], capacity=1, enabled=enabled
            ) as new:
                before = deepcopy((hs, p, oracle.cluster_states))
                exact(old.score(hs, p, oracle), new.score(hs, p, oracle))
                exact(before, (hs, p, oracle.cluster_states))
                assert not any(new.counts().values()) and not new.last_skipped.size


@pytest.mark.parametrize("mode", ["native_cpu", "gpu"])
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
def test_complete_filter_state_rng_graph_and_existing_metadata(
    scorers, mode, seed, options
):
    from bench_rbpf_feedback_mass import _bind_fixture_callbacks
    from rbpf_feedback_cases import make_feedback_case
    from rbpf_feedback_mass import NativeFeedbackMassStep
    from rbpf_move_cases import state

    extensions = {
        key: os.environ[env]
        for key, env in (
            ("extension", "GNSS_GPU_RBPF_CANONICAL_EXT"),
            ("parent_extension", "GNSS_GPU_RBPF_PARENT_EXT"),
            ("mass_extension", "GNSS_GPU_RBPF_MASS_EXT"),
        )
    }
    a, ra, ca = make_feedback_case(seed=seed, **options)
    b, rb, cb = make_feedback_case(seed=seed, **options)
    old = NativeFeedbackMassStep(a, scorers[0], mode, **extensions)
    new = NativeFeedbackMassStep(b, scorers[1], mode, **extensions)
    a.step, b.step = old, new
    for epoch in range(4):
        ra.epoch2 = rb.epoch2 = epoch
        if options.get("feedback", True):
            _bind_fixture_callbacks(ra)
            _bind_fixture_callbacks(rb)
        oa, ob = old(ra, deepcopy(ca)), new(rb, deepcopy(cb))
        exact(state(a, ra, oa), state(b, rb, ob))
        exact(old.last, new.last)
