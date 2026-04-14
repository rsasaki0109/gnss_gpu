#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

python3 - <<'PY'
from gnss_gpu.signal_sim import SignalSimulator
from gnss_gpu.acquisition import Acquisition
from gnss_gpu._gnss_gpu_signal_sim import generate_signal
from gnss_gpu._gnss_gpu_acq import acquire_parallel

sim = SignalSimulator(noise_seed=1)
acq = Acquisition(sampling_freq=sim.sampling_freq, intermediate_freq=sim.intermediate_freq)

assert callable(generate_signal)
assert callable(acquire_parallel)
assert sim.sampling_freq > 0
assert acq.doppler_step > 0

print("CUDA import smoke passed")
PY
