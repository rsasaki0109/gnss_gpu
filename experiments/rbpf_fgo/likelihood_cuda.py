"""Experimental fused FP64 likelihood replay; the live filter is unchanged.

Consumes captured uniform-lattice likelihood inputs, including real scalar
weight-cache misses. CUDA/Numba is imported only when constructing the evaluator.
Singular/unsupported inputs raise rather than silently substituting a result.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

MAX_STATE = 16
FLOAT_INPUTS = ("A0", "rows_J", "rows_v0", "rows_sig", "rows_lam", "rows_bmeas",
                "dds", "centres", "priors")


def prepare(arrays, config):
    """Validate and own packed inputs; perform outside repeated replay clocks."""
    values = {name: np.array(arrays[name], dtype=np.float64, order="C", copy=True) for name in FLOAT_INPUTS}
    values["rows_pair"] = np.array(arrays["rows_pair"], dtype=np.int32, order="C", copy=True)
    values["rows_iscp"] = np.array(arrays["rows_iscp"], dtype=np.int32, order="C", copy=True)
    if values["A0"].ndim != 2 or values["dds"].ndim != 2:
        raise ValueError("A0 and dds must be matrices")
    rows, state = values["A0"].shape
    batch, pairs = values["dds"].shape
    if rows < 1 or batch < 1 or pairs < 1 or not 3 <= state <= MAX_STATE:
        raise ValueError("nonempty batch/rows/pairs and state size 3..16 required")
    expected = dict(rows_J=(rows, 3), centres=(batch, 3), priors=(batch, 3, 3))
    expected.update({name: (rows,) for name in ["rows_v0", "rows_sig", "rows_lam", "rows_bmeas", "rows_pair", "rows_iscp"]})
    for name, shape in expected.items():
        if values[name].shape != shape:
            raise ValueError(f"invalid {name} shape")
    for name in FLOAT_INPUTS:
        if name != "dds" and not np.all(np.isfinite(values[name])):
            raise ValueError(f"nonfinite {name}")
    cp = values["rows_iscp"] != 0
    if np.any(values["rows_sig"] <= 0) or np.any(values["rows_lam"][cp] <= 0):
        raise ValueError("positive noise scales/carrier wavelengths required")
    if np.any(values["rows_pair"][cp] < 0) or np.any(values["rows_pair"][cp] >= pairs):
        raise ValueError("carrier pair index out of range")
    parameters = {name: float(config[name]) for name in
                  ["beta", "huber_c", "sig_group", "mix_eps", "mix_sigma_cp", "mix_sigma_pr"]}
    if not all(math.isfinite(x) for x in parameters.values()):
        raise ValueError("finite configuration required")
    if parameters["beta"] <= 0 or parameters["sig_group"] <= 0 or not 0 < parameters["mix_eps"] < 1:
        raise ValueError("invalid beta/group prior/mixture probability")
    if min(parameters["mix_sigma_cp"], parameters["mix_sigma_pr"]) <= 0:
        raise ValueError("positive mixture scales required")
    return values, parameters


@lru_cache(maxsize=1)
def _compile():
    from numba import cuda, float64, int32

    @cuda.jit(device=True)
    def factor(a, piv, n):
        sign, logdet = 1.0, 0.0
        for k in range(n):
            p, largest = k, abs(a[k, k])
            for i in range(k + 1, n):
                if abs(a[i, k]) > largest:
                    p, largest = i, abs(a[i, k])
            if largest == 0 or not math.isfinite(largest):
                return 0.0, 0.0
            piv[k] = p
            if p != k:
                for j in range(n):
                    a[k, j], a[p, j] = a[p, j], a[k, j]
                sign = -sign
            diagonal = a[k, k]
            if diagonal < 0:
                sign = -sign
            logdet += math.log(abs(diagonal))
            for i in range(k + 1, n):
                a[i, k] /= diagonal
                for j in range(k + 1, n):
                    a[i, j] -= a[i, k] * a[k, j]
        return sign, logdet

    @cuda.jit(device=True)
    def solve(a, piv, rhs, n):
        for k in range(n):
            p = piv[k]
            rhs[k], rhs[p] = rhs[p], rhs[k]
        for i in range(n):
            for j in range(i):
                rhs[i] -= a[i, j] * rhs[j]
        for i in range(n - 1, -1, -1):
            for j in range(i + 1, n):
                rhs[i] -= a[i, j] * rhs[j]
            rhs[i] /= a[i, i]

    @cuda.jit(fastmath=False)
    def kernel(A0, J, v0, sig, wavelength, measured, dds, centres, priors, pair, cp,
               B, m, n, nb, beta, huber, group_sigma, eps, sigma_cp, sigma_pr,
               output_ll, output_sol, output_cov, status):
        bidx = cuda.grid(1)
        if bidx >= B:
            return
        output_ll[bidx] = -1e12
        status[bidx] = 1
        for i in range(3):
            output_sol[bidx * 3 + i] = 0.0
            for j in range(3):
                output_cov[bidx * 9 + i * 3 + j] = 1.0 if i == j else 0.0
        matrix = cuda.local.array((MAX_STATE, MAX_STATE), float64)
        piv = cuda.local.array(MAX_STATE, int32)
        rhs = cuda.local.array(MAX_STATE, float64)
        solution = cuda.local.array(MAX_STATE, float64)
        prior_inv = cuda.local.array((3, 3), float64)
        for i in range(3):
            for j in range(3):
                matrix[i, j] = priors[bidx * 9 + i * 3 + j]
        sign_prior, logdet_prior = factor(matrix, piv, 3)
        if sign_prior <= 0:
            return
        for col in range(3):
            for i in range(3):
                rhs[i] = 1.0 if i == col else 0.0
            solve(matrix, piv, rhs, 3)
            for i in range(3):
                prior_inv[i, col] = rhs[i]
        inverse_group = 1.0 / (group_sigma * group_sigma)
        logdet_prior += 2.0 * (n - 3) * math.log(group_sigma)
        iterations = 3 if huber > 0 else 1
        logdet_lam, sign_lam = 0.0, 0.0
        for iteration in range(iterations):
            for i in range(n):
                rhs[i] = 0.0
                for j in range(n):
                    matrix[i, j] = prior_inv[i, j] if i < 3 and j < 3 else (inverse_group if i == j else 0.0)
            for r in range(m):
                z, held = 0.0, False
                if cp[r]:
                    z = dds[bidx * nb + pair[r]]
                    held = math.isfinite(z)
                    if not held:
                        continue
                noise = sig[r] / math.sqrt(beta) if held else sig[r]
                value = wavelength[r] * (measured[r] - z) if held else v0[r]
                for j in range(3):
                    value -= centres[bidx * 3 + j] * J[r * 3 + j]
                weight = 1.0
                if iteration > 0:
                    residual = value
                    for j in range(n):
                        residual -= A0[r * n + j] * solution[j]
                    absolute = abs(residual / noise)
                    denominator = absolute if absolute > 1e-12 else 1e-12
                    ratio = huber / denominator
                    weight = ratio if ratio < 1.0 else 1.0
                scale = math.sqrt(weight) / noise
                for i in range(n):
                    ai = A0[r * n + i] * scale
                    rhs[i] += ai * (value * scale)
                    for j in range(n):
                        matrix[i, j] += ai * (A0[r * n + j] * scale)
            sign_lam, logdet_lam = factor(matrix, piv, n)
            if sign_lam <= 0:
                return
            solve(matrix, piv, rhs, n)
            for i in range(n):
                solution[i] = rhs[i]
        ll = 0.0
        for r in range(m):
            z, held = 0.0, False
            if cp[r]:
                z = dds[bidx * nb + pair[r]]
                held = math.isfinite(z)
                if not held:
                    lam = wavelength[r] if wavelength[r] > 1e-3 else 1e-3
                    ll -= math.log(lam)
                    continue
            noise = sig[r] / math.sqrt(beta) if held else sig[r]
            value = wavelength[r] * (measured[r] - z) if held else v0[r]
            for j in range(3):
                value -= centres[bidx * 3 + j] * J[r * 3 + j]
            for j in range(n):
                value -= A0[r * n + j] * solution[j]
            residual = value / noise
            outside_scale = sigma_cp if cp[r] else sigma_pr
            outside = outside_scale if outside_scale > noise else noise
            a = math.log(1.0 - eps) - 0.5 * residual * residual - math.log(noise)
            out = residual * noise / outside
            c = math.log(eps) - 0.5 * out * out - math.log(outside)
            largest = a if a > c else c
            ll += largest + math.log1p(math.exp(-abs(a - c))) - 0.5 * math.log(2.0 * math.pi)
        prior_cost = 0.0
        for i in range(n):
            for j in range(n):
                v = prior_inv[i, j] if i < 3 and j < 3 else (inverse_group if i == j else 0.0)
                prior_cost += solution[i] * v * solution[j]
        ll -= 0.5 * (prior_cost + logdet_lam + logdet_prior)
        if not math.isfinite(ll):
            return
        for i in range(3):
            output_sol[bidx * 3 + i] = solution[i] + centres[bidx * 3 + i]
        for col in range(3):
            for i in range(n):
                rhs[i] = 1.0 if i == col else 0.0
            solve(matrix, piv, rhs, n)
            for i in range(3):
                output_cov[bidx * 9 + i * 3 + col] = rhs[i]
        for i in range(3):
            for j in range(i):
                v = 0.5 * (output_cov[bidx * 9 + i * 3 + j] + output_cov[bidx * 9 + j * 3 + i])
                output_cov[bidx * 9 + i * 3 + j] = v
                output_cov[bidx * 9 + j * 3 + i] = v
        output_ll[bidx] = ll
        status[bidx] = 0

    return cuda, kernel


class CudaLikelihood:
    """One CUDA stream and growing buffers; instances are not thread-safe."""

    def __init__(self):
        self.cuda, self.kernel = _compile()
        if not self.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        self.stream = self.cuda.stream()
        self.buffers = {}
        self.closed = False

    def close(self):
        if not self.closed:
            self.stream.synchronize()
            self.buffers.clear()
            self.closed = True

    def _buffer(self, name, size, dtype):
        array = self.buffers.get(name)
        if array is None or array.size < size:
            array = self.buffers[name] = self.cuda.device_array(size, dtype=dtype, stream=self.stream)
        return array[:size]

    def evaluate(self, packed, config):
        """Upload packed inputs, run FP64 evaluation, read all results and synchronize."""
        if self.closed:
            raise RuntimeError("likelihood evaluator is closed")
        B, nb = packed["dds"].shape
        m, n = packed["A0"].shape
        inputs = {}
        for name, host in packed.items():
            device = self._buffer(name, host.size, host.dtype)
            device.copy_to_device(host.reshape(-1), stream=self.stream)
            inputs[name] = device
        ll = self._buffer("ll", B, np.float64)
        sol = self._buffer("sol", B * 3, np.float64)
        cov = self._buffer("cov", B * 9, np.float64)
        status = self._buffer("status", B, np.int32)
        self.kernel[(B + 31) // 32, 32, self.stream](
            *[inputs[name] for name in ["A0", "rows_J", "rows_v0", "rows_sig", "rows_lam", "rows_bmeas",
                                      "dds", "centres", "priors", "rows_pair", "rows_iscp"]],
            B, m, n, nb, config["beta"], config["huber_c"], config["sig_group"],
            config["mix_eps"], config["mix_sigma_cp"], config["mix_sigma_pr"], ll, sol, cov, status)
        outputs = [array.copy_to_host(stream=self.stream) for array in [ll, sol, cov, status]]
        self.stream.synchronize()
        if np.any(outputs[3]):
            raise FloatingPointError("unsupported singular/nonfinite likelihood system; use the CPU reference")
        return outputs[0], outputs[1].reshape(B, 3), outputs[2].reshape(B, 3, 3)
