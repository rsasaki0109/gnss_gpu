"""Exact diagnostics plus explicit boundary-perturbation controls."""
from contextlib import ExitStack
import ctypes
import os
from pathlib import Path
import sys
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
from rbpf_lu_phases import LUPhaseScorer  # noqa: E402
from rbpf_likelihood_batch import BatchScorer  # noqa: E402
from rbpf_likelihood_cases import make_case  # noqa: E402
from bench_rbpf_lu_phases import stage_cycles  # noqa: E402
from test_rbpf_register_lu import (  # noqa: E402
    exact, test_all_numeric_bits_same_backend,
    test_alignment_empty_rows_and_completed_host_readback,
    test_original_fallbacks_errors_warnings,
    test_complete_pf_four_epochs_all_saved_state_rng_and_graph_events,
)


class ProfiledView(LUPhaseScorer):
    def __init__(self, dll, capacity=512, control=None):
        super().__init__(dll, capacity)
        self.control = control

    def score(self, hypotheses, problem, oracle, mode="gpu", diagnostics=False):
        return self.profiled_score(hypotheses, problem, oracle, mode, diagnostics, self.control)[:2]


@pytest.fixture(scope="module")
def scorers():
    old = os.environ.get("GNSS_GPU_RBPF_PROBE_DLL")
    new = os.environ.get("GNSS_GPU_RBPF_PHASES_DLL")
    if not old or not new:
        pytest.skip("Explicit standalone control/phase DLLs required")
    with ExitStack() as stack:
        yield (stack.enter_context(BatchScorer(old)), stack.enter_context(LUPhaseScorer(new)),
            stack.enter_context(ProfiledView(new, control="copy")),
            stack.enter_context(ProfiledView(new, control="barriers")),
            stack.enter_context(ProfiledView(new)))


@pytest.mark.parametrize("huber", [False, True])
@pytest.mark.parametrize("groups", [1, 8])
def test_phase_topology_ownership_and_reset(scorers, huber, groups):
    oracle, problem, hs = make_case(21, rows=128, groups=groups, huber=huber, cluster=True)
    scorer = scorers[1]
    expected = scorers[0].score(hs, problem, oracle, "gpu", diagnostics=True)
    values, details, trace = scorer.profiled_score(hs, problem, oracle, diagnostics=True)
    exact(expected, (values, details))
    blocks = stage_cycles(trace, huber)
    assert len(blocks) == details["unique"]
    assert trace.shape == (details["unique"], 48)
    saved = trace.copy()
    for control in ("copy", "barriers"):
        assert scorer.profiled_score(hs[:1], problem, oracle, control=control)[2] is None
        assert scorer.dll.rbpf_profile_read(scorer.handle, None, 0) == 0
    assert trace.tobytes() == saved.tobytes()
    assert scorer.profiled_score(hs, problem, oracle, mode="native_cpu")[2] is None
    for invalid in (-1, 4):
        assert scorer.dll.rbpf_profile_select(scorer.handle, invalid) == -1
    assert scorer.dll.rbpf_profile_select(None, 0) == -1
    assert scorer.dll.rbpf_profile_read(scorer.handle, None, -1) == -1
    with LUPhaseScorer(os.environ["GNSS_GPU_RBPF_PHASES_DLL"]) as temporary:
        temporary.profiled_score(hs[:1], problem, oracle)
    temporary.close()
    with pytest.raises(RuntimeError):
        temporary.profiled_score(hs, problem, oracle)


KINDS = ("spd", "asymmetric", "rowpivot", "ties", "singular", "duplicate", "negative", "nan", "inf", "overflow", "subnormal", "signedzero", "mixed_scale", "ill_conditioned", "rhs_nan", "rhs_inf", "rhs_huge", "pivot_ties_later", "near_singular", "rhs_signedzero")


@pytest.mark.parametrize("n", range(3, 12))
@pytest.mark.parametrize("kind", KINDS)
def test_raw_phase_LU_pivots_solution_logdet_status_exact(scorers, n, kind):
    rng = np.random.default_rng(20261017 + n * 100 + KINDS.index(kind))
    data = np.zeros((32, 132))
    for case in data:
        a = rng.normal(size=(n, n))
        matrix = a @ a.T + np.eye(n) * 0.01
        rhs = rng.normal(size=n)
        if kind == "asymmetric":
            matrix += rng.normal(size=(n, n)) * 0.02
        elif kind == "rowpivot":
            matrix = matrix[np.roll(np.arange(n), 2)]
        elif kind == "ties":
            matrix[:2, 0] = [2., -2.]
            matrix[2:, 0] = 0.
        elif kind == "singular":
            matrix[:] = 0.
        elif kind == "duplicate":
            matrix[-1] = matrix[-2]
        elif kind == "negative":
            matrix[0] *= -1.
        elif kind in ("nan", "inf"):
            matrix[int(rng.integers(n)), int(rng.integers(n))] = np.nan if kind == "nan" else np.inf
        elif kind == "overflow":
            matrix = rng.choice([-1., 1.], size=(n, n)) * (np.finfo(float).max / 2)
        elif kind == "subnormal":
            matrix = np.eye(n)
            matrix[0, 0] = np.nextafter(0., 1.)
            rhs[0] = np.nextafter(0., 1.)
        elif kind == "signedzero":
            matrix = np.eye(n)
            zero = rng.choice([0., -0.], size=(n, n))
            matrix[np.eye(n) == 0] = zero[np.eye(n) == 0]
            rhs = rng.choice([0., -0.], size=n)
        elif kind == "mixed_scale":
            scale = np.exp2(rng.integers(-60, 61, size=n).astype(float))
            matrix *= scale[:, None] * scale[None, :]
        elif kind == "ill_conditioned":
            index = np.arange(n, dtype=float)
            matrix = 1. / (index[:, None] + index[None, :] + 1.)
            matrix += np.eye(n) * (2. ** -50)
        elif kind == "rhs_nan":
            rhs[int(rng.integers(n))] = np.nan
        elif kind == "rhs_inf":
            rhs[int(rng.integers(n))] = np.inf
        elif kind == "rhs_huge":
            rhs *= np.finfo(float).max / 16
        elif kind == "pivot_ties_later":
            matrix = np.eye(n)
            matrix[1, 1] = 2.
            matrix[2, 1] = -2.
            matrix[1, 2] = 1.
        elif kind == "near_singular":
            matrix[-1] = matrix[-2] + rng.normal(size=n) * (2. ** -50)
        elif kind == "rhs_signedzero":
            rhs = rng.choice([0., -0.], size=n)
        case[:121].reshape(11, 11)[:n, :n] = matrix
        case[121:121+n] = rhs
    dims = np.full(len(data), n, dtype=np.int32)
    output = np.empty((len(data), 3, 144))
    status = np.empty((len(data), 3), dtype=np.int32)
    f = scorers[-1].dll.rbpf_phase_solve_probe
    doubles = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    integers = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
    f.argtypes = [doubles, integers, ctypes.c_int, doubles, integers, np.ctypeslib.ndpointer(dtype=np.uint64, flags="C_CONTIGUOUS")]
    f.restype = ctypes.c_int
    trace = np.empty((len(data), 3, 10), dtype=np.uint64)
    before = data.tobytes(), dims.tobytes()
    assert f(data, dims, len(data), output, status, trace) == 0
    assert before == (data.tobytes(), dims.tobytes())
    assert output[:, 0].tobytes() == output[:, 1].tobytes()
    assert status[:, 0].tobytes() == status[:, 1].tobytes()
    assert output[:, 0].tobytes() == output[:, 2].tobytes()
    assert status[:, 0].tobytes() == status[:, 2].tobytes()
    sentinel = np.iinfo(np.uint64).max
    assert np.all(trace[:, :2] == sentinel)
    if n != 11:
        assert np.all(trace[:, 2] == sentinel)
    else:
        for t, code in zip(trace[:, 2], status[:, 2], strict=True):
            visited = [i for i, v in enumerate(t[:8]) if v != sentinel]
            assert visited in ([0, 1], list(range(4)), list(range(8)))
            assert all(int(t[b]) >= int(t[a]) for a, b in zip(visited, visited[1:]))
            assert int(t[8]) in (1, 2)
            assert 0 <= int(t[9]) <= 11
            assert (int(t[8]) == 2) == (visited == [0, 1])
            if int(t[8]) == 1:
                assert (code == 1) == (visited == list(range(4)))
