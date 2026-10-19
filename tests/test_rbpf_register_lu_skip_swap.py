"""Mixed pivot permutations compare generic, corrected register and skip exchange."""

import ctypes
import os
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
from rbpf_register_lu import RegisterLUScorer  # noqa: E402


@pytest.mark.parametrize("pattern", ["identity", "pairs", "reverse", "cycle", "alternating", "signed_zero"])
def test_mixed_pivot_permutations_match_corrected_register_and_original(pattern):
    new = os.environ.get("GNSS_GPU_RBPF_REGISTER_DLL")
    safe = os.environ.get("GNSS_GPU_RBPF_REGISTER_SYNCED_DLL")
    if not new or not safe:
        pytest.skip("Explicit candidate and corrected register DLLs required")
    rng = np.random.default_rng(20261017)
    data = np.zeros((32, 132))
    for item in data:
        a = np.triu(rng.uniform(-0.1, 0.1, size=(11, 11)))
        np.fill_diagonal(a, rng.uniform(2., 4., size=11))
        perm = np.arange(11)
        if pattern == "pairs":
            perm[:10] = perm[:10].reshape(5, 2)[:, ::-1].ravel()
        elif pattern == "reverse":
            perm = perm[::-1]
        elif pattern == "cycle":
            perm = np.roll(perm, 3)
        elif pattern == "alternating":
            perm = perm[[0, 2, 1, 3, 6, 5, 4, 7, 10, 9, 8]]
        elif pattern == "signed_zero":
            a[np.tril_indices(11, -1)] = rng.choice([0., -0.], size=55)
        a = a[perm].copy()
        inversions = sum(int(perm[i] > perm[j]) for i in range(11) for j in range(i + 1, 11))
        if inversions % 2:
            a[0] *= -1.
        item[:121] = a.ravel()
        item[121:] = rng.normal(size=11)
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
            results.append((output, status))
    assert results[0][0].tobytes() == results[1][0].tobytes()
    assert results[0][1].tobytes() == results[1][1].tobytes()
    pivots = results[1][0][:, 1, 133:144]
    moving = pivots != np.arange(11)
    if pattern in ("identity", "signed_zero"):
        assert not moving.any()
    else:
        assert moving.any() and (~moving).any()
