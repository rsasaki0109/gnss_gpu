"""Replay variant sharing the FP64 kernel, with two uploads and one readback."""

import numpy as np

from likelihood_cuda import CudaLikelihood, FLOAT_INPUTS, prepare as prepare_arrays


def prepare(arrays, config):
    packed, config = prepare_arrays(arrays, config)
    packed["_float_payload"] = np.concatenate([packed[name].reshape(-1) for name in FLOAT_INPUTS])
    packed["_int_payload"] = np.concatenate([packed[name].reshape(-1) for name in ["rows_pair", "rows_iscp"]])
    return packed, config


class PackedCudaLikelihood(CudaLikelihood):
    def evaluate(self, packed, config):
        if self.closed:
            raise RuntimeError("likelihood evaluator is closed")
        B, nb = packed["dds"].shape
        m, n = packed["A0"].shape
        inputs = {}
        for payload, names in [("_float_payload", FLOAT_INPUTS), ("_int_payload", ["rows_pair", "rows_iscp"])]:
            host = packed[payload]
            device = self._buffer(payload, host.size, host.dtype)
            device.copy_to_device(host, stream=self.stream)
            offset = 0
            for name in names:
                size = packed[name].size
                inputs[name] = device[offset:offset+size]
                offset += size
        output = self._buffer("packed_output", B * 14, np.float64)
        # Status uses exact 0/1 values in the same f64 readback buffer.
        ll, sol = output[:B], output[B:4*B]
        cov, status = output[4*B:13*B], output[13*B:14*B]
        self.kernel[(B + 31) // 32, 32, self.stream](
            *[inputs[name] for name in ["A0", "rows_J", "rows_v0", "rows_sig", "rows_lam", "rows_bmeas",
                                      "dds", "centres", "priors", "rows_pair", "rows_iscp"]],
            B, m, n, nb, config["beta"], config["huber_c"], config["sig_group"],
            config["mix_eps"], config["mix_sigma_cp"], config["mix_sigma_pr"], ll, sol, cov, status)
        result = output.copy_to_host(stream=self.stream)
        self.stream.synchronize()
        if np.any(result[13*B:]):
            raise FloatingPointError("unsupported singular/nonfinite likelihood system; use the CPU reference")
        return result[:B], result[B:4*B].reshape(B, 3), result[4*B:13*B].reshape(B, 3, 3)
