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

# Suites that need the compiled extensions. They skip on the CPU-only
# runners (full-suite.yml), so this workflow is where they actually run.
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q \
  tests/test_signal_sim.py \
  tests/test_raytrace.py \
  tests/test_bvh.py \
  tests/test_raytrace_bvh_wrapper.py \
  tests/test_pf_wrapper.py \
  tests/test_pf_device_wrapper.py \
  tests/test_cuda_streams.py \
  tests/test_pf_device_resampling.py \
  tests/test_svgd_wrapper.py \
  tests/test_pf3d.py \
  tests/test_pf3d_wrapper.py \
  tests/test_pf3d_bvh.py \
  tests/test_pf3d_bvh_short_segment.py \
  tests/test_multipath.py \
  tests/test_city_model_validator.py

# Exercise the PPC CT-RBPF segment runner end to end on synthetic data.
PYTHONPATH=python python3 experiments/fingerprint_ctrbpf_segment.py --methods pf,pf+pu,rbpf+pu,pf+hybrid --epochs 20 > /dev/null
