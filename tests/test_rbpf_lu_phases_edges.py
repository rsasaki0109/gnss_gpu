"""Block-wide abort,partial factorization and mixed independent block batches."""

import ctypes
import os
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
from rbpf_lu_phases import LUPhaseScorer  # noqa: E402


def compare_raw(data, dims):
    dll = os.environ.get("GNSS_GPU_RBPF_PHASES_DLL")
    if not dll:
        pytest.skip("Explicit standalone phase DLL required")
    with LUPhaseScorer(dll) as scorer:
        f = scorer.dll.rbpf_phase_solve_probe
        doubles = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
        integers = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
        ticks = np.ctypeslib.ndpointer(dtype=np.uint64, flags="C_CONTIGUOUS")
        f.argtypes = [doubles, integers, ctypes.c_int, doubles, integers, ticks]
        f.restype = ctypes.c_int
        output = np.empty((len(data), 3, 144))
        status = np.empty((len(data), 3), dtype=np.int32)
        trace = np.empty((len(data), 3, 10), dtype=np.uint64)
        before = data.tobytes(), dims.tobytes()
        assert f(data, dims, len(data), output, status, trace) == 0
        assert before == (data.tobytes(), dims.tobytes())
        for variant in (1, 2):
            assert output[:, 0].tobytes() == output[:, variant].tobytes()
            assert status[:, 0].tobytes() == status[:, variant].tobytes()
        sentinel = np.iinfo(np.uint64).max
        assert np.all(trace[:, :2] == sentinel)
        for n, code, t in zip(dims, status[:, 2], trace[:, 2], strict=True):
            if n != 11:
                assert np.all(t == sentinel)
                continue
            visited = [i for i, v in enumerate(t[:8]) if v != sentinel]
            assert visited in ([0, 1], list(range(4)), list(range(8)))
            assert all(int(t[b]) >= int(t[a]) for a, b in zip(visited, visited[1:]))
            assert int(t[8]) in (1, 2) and 0 <= int(t[9]) <= 11
            assert (int(t[8]) == 2) == (visited == [0, 1])
            if int(t[8]) == 1:
                assert (code == 1) == (visited == list(range(4)))
        return output, status


@pytest.mark.parametrize("failed_column", range(11))
@pytest.mark.parametrize("pivoted", [False, True], ids=["plain", "rowpivot"])
def test_partial_factorization_failure_and_shared_block_handoffs(failed_column, pivoted):
    rng = np.random.default_rng(20261017 + failed_column * 2 + int(pivoted))
    data = np.zeros((32, 132))
    for item in data:
        matrix = item[:121].reshape(11, 11)
        matrix[:] = rng.choice([0., -0.], size=(11, 11))
        for row in range(failed_column):
            matrix[row, row:] = rng.uniform(-0.2, 0.2, size=11 - row)
            matrix[row, row] = 2. ** (row + 1)
        if pivoted:
            matrix[:] = matrix[rng.permutation(11)]
        item[121:] = rng.normal(size=11)
    output, status = compare_raw(data, np.full(32, 11, dtype=np.int32))
    assert np.all(status == 1)
    assert np.all(np.isfinite(output[:, :, 0]))
    if failed_column == 0:
        assert np.all(output[:, :, 0] == 0.)
    else:
        assert np.all(output[:, :, 0] > 0.)
    assert np.all(output[:, :, 133 + failed_column:] == 0.)


@pytest.mark.parametrize("count", [1, 63, 128, 256])
def test_mixed_dimensions_status_and_maximum_raw_batch_are_independent(count):
    rng = np.random.default_rng(20261017 + count)
    data = np.zeros((count, 132))
    dims = np.empty(count, dtype=np.int32)
    for i, item in enumerate(data):
        n = 11 if i == 0 else 3 + i % 9
        dims[i] = n
        a = rng.normal(size=(n, n))
        matrix = a @ a.T + np.eye(n)
        rhs = rng.normal(size=n)
        if i and i % 7 == 0:
            matrix[:] = 0.
        elif i and i % 11 == 0:
            rhs[0] = np.inf
        elif i and i % 13 == 0:
            matrix[0] *= -1.
        item[:121].reshape(11, 11)[:n, :n] = matrix
        item[121:121 + n] = rhs
    output, status = compare_raw(data, dims)
    assert np.all(status[0] == 0) and np.all(np.isfinite(output[0]))


@pytest.mark.parametrize("row", [0, 3, 6])
@pytest.mark.parametrize("exponent", [54, 100, 499])
@pytest.mark.parametrize("sign", [1., -1.])
def test_backward_subtract_order_cancellation_bits(row, exponent, sign):
    rng = np.random.default_rng(20261017)
    data = np.zeros((32, 132))
    for item in data:
        a = item[:121].reshape(11, 11)
        a[:] = rng.choice([0., -0.], size=(11, 11))
        np.fill_diagonal(a, 1.)
        a[row, row + 1:row + 4] = [sign * 2. ** exponent, -sign * 2. ** exponent, sign]
        item[121:] = 1.
        item[121 + row] = sign
    dims = np.full(32, 11, dtype=np.int32)
    output, status = compare_raw(data, dims)
    assert np.all(status == 0)
    assert np.all(output[:, :, 1 + row] == -sign)


@pytest.mark.parametrize("tiny,rhs", [(np.nextafter(0., 1.), 1.),
    (np.nextafter(0., 1.), 2. ** 499), (np.finfo(float).tiny, 1.),
    (np.finfo(float).tiny, 2. ** 499)])
def test_phase_finite_guarded_backsolve_overflow_and_signed_zero(tiny, rhs):
    rng = np.random.default_rng(20261017)
    for position in (0, 5, 10):
        data = np.zeros((32, 132))
        for item in data:
            matrix = item[:121].reshape(11, 11)
            matrix[:] = rng.choice([0., -0.], size=(11, 11))
            np.fill_diagonal(matrix, 1.)
            matrix[position, position] = tiny
            item[121:] = rng.choice([-1., 1.], size=11) * rhs
        output, status = compare_raw(data, np.full(32, 11, dtype=np.int32))
        expected = 0 if tiny == np.finfo(float).tiny and rhs == 1. else 3
        assert np.all(status == expected)
