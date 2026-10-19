"""Cancellation distinguishes ascending-k backward subtraction from reordered sums."""

import ctypes
import os
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
from rbpf_register_lu import RegisterLUScorer  # noqa: E402


@pytest.mark.parametrize("row", [0, 3, 6])
@pytest.mark.parametrize("exponent", [54, 100, 499])
@pytest.mark.parametrize("sign", [1., -1.])
def test_backward_subtract_order_cancellation_bits(row, exponent, sign):
    new = os.environ.get("GNSS_GPU_RBPF_REGISTER_DLL")
    safe = os.environ.get("GNSS_GPU_RBPF_REGISTER_SYNCED_DLL")
    if not new or not safe:
        pytest.skip("Explicit candidate and synchronized register DLLs required")
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
    results = []
    for dll in (safe, new):
        with RegisterLUScorer(dll) as scorer:
            f = scorer.dll.rbpf_register_solve_probe
            doubles = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
            integers = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
            f.argtypes = [doubles, integers, ctypes.c_int, doubles, integers]
            f.restype = ctypes.c_int
            output = np.empty((32, 2, 144))
            status = np.empty((32, 2), dtype=np.int32)
            assert f(data, dims, 32, output, status) == 0
            assert output[:, 0].tobytes() == output[:, 1].tobytes()
            assert np.all(status == 0)
            # The first subtract loses the unit;the next cancels;the last remains.
            # Reversing the three terms would yield zero instead.
            assert np.all(output[:, :, 1 + row] == -sign)
            results.append((output, status))
    assert results[0][0].tobytes() == results[1][0].tobytes()
    assert results[0][1].tobytes() == results[1][1].tobytes()
