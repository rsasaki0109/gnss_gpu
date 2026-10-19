"""Finite guarded inputs whose back substitution can overflow or propagate NaN."""

import ctypes
import os
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
from rbpf_register_lu import RegisterLUScorer  # noqa: E402


@pytest.mark.parametrize("tiny,rhs", [(np.nextafter(0., 1.), 1.),
    (np.nextafter(0., 1.), 2. ** 499), (np.finfo(float).tiny, 1.),
    (np.finfo(float).tiny, 2. ** 499)])
def test_finite_guarded_backsolve_overflow_nan_and_signed_zero_bits(tiny, rhs):
    dll = os.environ.get("GNSS_GPU_RBPF_REGISTER_DLL")
    if not dll:
        pytest.skip("Explicit standalone register LU DLL required")
    rng = np.random.default_rng(20261017)
    with RegisterLUScorer(dll) as scorer:
        f = scorer.dll.rbpf_register_solve_probe
        doubles = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
        integers = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
        f.argtypes = [doubles, integers, ctypes.c_int, doubles, integers]
        f.restype = ctypes.c_int
        for position in (0, 5, 10):
            data = np.zeros((32, 132))
            for item in data:
                matrix = item[:121].reshape(11, 11)
                matrix[:] = rng.choice([0., -0.], size=(11, 11))
                np.fill_diagonal(matrix, 1.)
                matrix[position, position] = tiny
                item[121:] = rng.choice([-1., 1.], size=11) * rhs
            assert np.all(np.isfinite(data)) and np.max(np.abs(data)) <= 2. ** 500
            dims = np.full(32, 11, dtype=np.int32)
            output = np.empty((32, 2, 144))
            status = np.empty((32, 2), dtype=np.int32)
            assert f(data, dims, 32, output, status) == 0
            assert output[:, 0].tobytes() == output[:, 1].tobytes()
            assert status[:, 0].tobytes() == status[:, 1].tobytes()
            expected = 0 if tiny == np.finfo(float).tiny and rhs == 1. else 3
            assert np.all(status == expected)
