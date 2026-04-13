#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

python3 - <<'PY'
from gnss_gpu.signal_sim import SignalSimulator
from gnss_gpu.acquisition import Acquisition

sim = SignalSimulator(noise_seed=1)
n_samples = int(sim.sampling_freq * 1e-3)
channels = [{
    "prn": 1,
    "code_phase": 0.0,
    "carrier_phase": 0.0,
    "doppler_hz": 750.0,
    "amplitude": 1.0,
    "nav_bit": 1,
}]

iq = sim.generate_epoch(channels, n_samples=n_samples)
signal = iq[0::2].copy()

acq = Acquisition(
    sampling_freq=sim.sampling_freq,
    intermediate_freq=sim.intermediate_freq,
)
results = acq.acquire(signal, prn_list=[1])

assert len(results) == 1
assert results[0]["acquired"], results

print("CUDA runtime roundtrip passed")
PY

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests/test_signal_sim.py
