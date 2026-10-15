"""Isolated replay contracts do not require the GTSAM experiment environment."""

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1] / "experiments/rbpf_fgo"
sys.path.insert(0, str(HERE))
from benchmark_likelihood_replay import cpu_reference, reference_call  # noqa: E402
from likelihood_cuda import CudaLikelihood, prepare  # noqa: E402
from likelihood_cuda_packed import PackedCudaLikelihood, prepare as prepare_packed  # noqa: E402


def problem(group=False, batch=2):
    n = 4 if group else 3
    basis = np.zeros((6, n))
    basis[:, :3] = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    if group:
        basis[3:, 3] = 1
    cp = np.array([False, False, False, True, True, True])
    arrays = dict(A0=basis, rows_J=basis[:, :3], rows_v0=np.array([1., 0., -2., 0., 0., 0.]),
                  rows_sig=np.array([1., 1., 1., .008, .008, .008]), rows_lam=np.array([0., 0., 0., .19, .24, .19]),
                  rows_bmeas=np.array([0., 0., 0., 2.1, 3., 1.]), rows_pair=np.array([-1, -1, -1, 0, 1, 2]),
                  rows_iscp=cp, dds=np.full((batch, 3), np.nan), centres=np.zeros((batch, 3)),
                  priors=np.repeat(np.eye(3)[None], batch, axis=0))
    if batch > 1:
        arrays["dds"][1] = [2, np.nan, 1]
        arrays["centres"][1] = [.1, -.05, .2]
    config = dict(beta=.5, huber_c=3., sig_group=.02, mix_eps=.1, mix_sigma_cp=.05, mix_sigma_pr=10.)
    return arrays, config


def test_prepare_owns_contiguous_inputs():
    arrays, config = problem(True)
    packed, _ = prepare(arrays, config)
    for name, value in packed.items():
        assert value.flags.c_contiguous
        assert not np.shares_memory(value, arrays[name])
    arrays["dds"][:] = 100
    assert np.isnan(packed["dds"][0]).all()


@pytest.mark.parametrize("name,value", [
    ("A0", np.zeros((6, 17))), ("rows_J", np.zeros((5, 3))), ("centres", np.zeros((2, 4))),
    ("priors", np.zeros((2, 2, 2))), ("rows_v0", np.ones(5)),
    ("rows_sig", np.zeros(6)), ("rows_lam", np.zeros(6)),
    ("rows_pair", np.full(6, 99)), ("rows_J", np.full((6, 3), np.nan)),
])
def test_invalid_arrays_rejected(name, value):
    arrays, config = problem()
    arrays[name] = value
    with pytest.raises(ValueError):
        prepare(arrays, config)


@pytest.mark.parametrize("name,value", [("beta", 0), ("sig_group", -1), ("mix_eps", 0),
                                        ("mix_eps", 1), ("mix_sigma_cp", 0), ("huber_c", float("nan"))])
def test_invalid_configuration_rejected(name, value):
    arrays, config = problem()
    config[name] = value
    with pytest.raises(ValueError):
        prepare(arrays, config)


def test_cpu_reference_imports_without_gtsam():
    arrays, config = problem(True)
    packed, config = prepare(arrays, config)
    reference = cpu_reference(HERE / "rbpf_fgo.py")
    output = reference_call(reference, packed, config)()
    assert [value.shape for value in output] == [(2,), (2, 3), (2, 3, 3)]
    assert all(np.isfinite(value).all() for value in output)


@pytest.fixture(params=["separate", "packed"])
def gpu(request):
    if importlib.util.find_spec("numba") is None:
        pytest.skip("Numba is optional")
    from numba import cuda
    if not cuda.is_available():
        pytest.skip("CUDA unavailable")
    evaluator = CudaLikelihood() if request.param == "separate" else PackedCudaLikelihood()
    yield evaluator
    evaluator.close()


def gpu_inputs(gpu, arrays, config):
    return prepare_packed(arrays, config) if isinstance(gpu, PackedCudaLikelihood) else prepare(arrays, config)


@pytest.mark.parametrize("group,huber", [(False, 0), (False, 3), (True, 0), (True, 3)])
def test_cuda_numerics_and_workspace_reuse(gpu, group, huber):
    reference = cpu_reference(HERE / "rbpf_fgo.py")
    for size in [2, 9, 1]:
        arrays, config = problem(group, size)
        config["huber_c"] = huber
        packed, config = gpu_inputs(gpu, arrays, config)
        expected = reference_call(reference, packed, config)()
        observed = gpu.evaluate(packed, config)
        for a, b in zip(observed, expected):
            np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-9)


def test_cuda_singular_failure_and_recovery(gpu):
    arrays, config = problem(True)
    arrays["priors"][0] = 0
    packed, config = gpu_inputs(gpu, arrays, config)
    with pytest.raises(FloatingPointError):
        gpu.evaluate(packed, config)
    arrays["priors"][0] = np.eye(3)
    packed, config = gpu_inputs(gpu, arrays, config)
    assert np.isfinite(gpu.evaluate(packed, config)[0]).all()


def test_closed_evaluator_rejects_calls(gpu):
    arrays, config = problem()
    packed, config = gpu_inputs(gpu, arrays, config)
    gpu.close()
    gpu.close()
    with pytest.raises(RuntimeError, match="closed"):
        gpu.evaluate(packed, config)
