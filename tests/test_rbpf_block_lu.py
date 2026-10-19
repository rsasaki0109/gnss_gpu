"""Block-wide abort,partial factorization and mixed independent block batches."""

import ctypes
import os
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
from rbpf_register_lu import RegisterLUScorer  # noqa: E402


def compare_raw(data, dims):
    new = os.environ.get("GNSS_GPU_RBPF_REGISTER_DLL")
    safe = os.environ.get("GNSS_GPU_RBPF_REGISTER_SYNCED_DLL")
    if not new or not safe:
        pytest.skip("Explicit block LU and synchronized reference DLLs required")
    results = []
    for dll in (safe, new):
        with RegisterLUScorer(dll) as scorer:
            f = scorer.dll.rbpf_register_solve_probe
            doubles = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
            integers = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
            f.argtypes = [doubles, integers, ctypes.c_int, doubles, integers]
            f.restype = ctypes.c_int
            output = np.empty((len(data), 2, 144))
            status = np.empty((len(data), 2), dtype=np.int32)
            before = data.tobytes(), dims.tobytes()
            assert f(data, dims, len(data), output, status) == 0
            assert before == (data.tobytes(), dims.tobytes())
            assert output[:, 0].tobytes() == output[:, 1].tobytes()
            assert status[:, 0].tobytes() == status[:, 1].tobytes()
            results.append((output, status))
    assert results[0][0].tobytes() == results[1][0].tobytes()
    assert results[0][1].tobytes() == results[1][1].tobytes()
    return results[1]


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
