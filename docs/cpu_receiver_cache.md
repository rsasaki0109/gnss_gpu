# CPU receiver reference-template reuse

A subsequent [polynomial-preprocessing experiment](cpu_receiver_polynomial.md)
measures an additional improvement relative to this cached implementation.
The timing results below remain the original cache experiment.

The delay fitter now reuses immutable C/A reference data across acquisition
and frequency-refinement calls. The receiver's search, fitting equations,
root solver, gate, confirmation rules and state updates remain the same.

## Why this helps

A 32-block receiver profile made 796 calls to `estimate_ca_delay` and 5,572
calls to `_averaged_template`. Rebuilding aperture templates took about 0.257 s
of the profiled 0.729 s. Frequency refinement repeatedly evaluates the same
PRN, sampling geometry and nearby delay intervals at different carriers.

The fitter now retains:

* Sample indices and the conjugate FFT of the zero-delay reference, keyed by
  PRN and integer samples per chip (16 LRU entries).
* The two affine aperture basis arrays and their three Gram coefficients,
  keyed by PRN, samples per chip and interval center (64 LRU entries).

Current IQ is still normalized and carrier-wiped for every evaluation. Its FFT,
projections, polynomial roots, fitted amplitude, delay and detection score are
computed from that observation. Cached arrays have NumPy's writeable flag
disabled and are used as read-only references; no IQ or fitted decision is
retained by these caches. Concurrent frequency searches can share them.

Periods above **8,192 samples** bypass both retained caches. At the largest
supported cached period (8,184 samples), maximum retained NumPy array payload
is approximately **11 MiB**, excluding active temporaries and Python bookkeeping.
This bounds retained reference data, not total process memory. Cold misses and
working sets exceeding capacity may see less benefit. Release cached data with:

```python
from gnss_gpu.code_delay_cpu import clear_ca_delay_cache
clear_ca_delay_cache()
```

Ongoing fits retain valid references to arrays already in use; clearing changes
future reuse, not an observation's meaning.

## Frozen end-to-end comparison

The benchmark reuses the previous
[continuous batch/receiver recording](cpu_batch_receiver_validation.md):
128 contiguous 1 ms blocks, 12 composite PRNs, a 16 ms signal outage and one
PRN-1 receiver with fixed grid −500/0/+500 Hz. Both versions use the same batch
IQ generator and calibrated gate. Only the delay fitter is switched, in a
benchmark-local context that restores the original functions afterward.

The saved recording was created with the pre-change fitter. Its source hash
matches the saved baseline module. The regenerated IQ SHA-256 must match the
recording before timing. All **199 frozen noise-calibration scores match
exactly**, and every field of all **128 receiver results** matches the baseline
and saved trace, including complex candidate amplitudes and delay/frequency
states. The same comparisons hold on every timed trial.

Seven repetitions alternate old/new order after warmup, with tests completed
before timing. The local environment has four CPU cores; generation uses four
Numba threads and receiver search uses one worker. Independent stage medians:

| Stage | Before | After | Before / after |
| --- | ---: | ---: | ---: |
| IQ generation | 50.47 ms | 50.81 ms | approximately unchanged |
| Receiver construction and updates | 2.206 s | 1.038 s | 2.12× |
| Generation + receiver | 2.257 s | 1.089 s | 2.07× |

Stage medians need not sum to the total median. Calibration, JIT, comparisons,
scoring and serialization are excluded. A separate receiver pass with these
caches cleared took 1.293 s; that is one cache-empty pass, not a cold-start
distribution or a timing bound. Final cache counters were 25,471 reference
hits / 1 miss and 75,288 interval hits / 1,128 misses, with capacities respected.

The earlier comparison changed only generation and did not demonstrate a
repeatable whole-pipeline gain. This comparison isolates a receiver-side
change and shows a larger improvement on the frozen workload. It still does
not establish universal speedup or real-time operation: 128 ms of input took
about 1.09 s to generate and process on the measured path. Timing repetitions
reuse one synthetic recording, not seven independent GNSS trials.

The existing physical limitations remain: rectangular point-sampled signals
are followed by an aperture-based fitter, only one PRN is followed in the
composite, and there is no PLL/FLL/DLL, nav decoding or PVT validation. Exact
computational equivalence is not proof of real RF accuracy, rain validity or
an integrity bound.

## Validation and reproduction

**179 tests pass** across waveform generation, batching, fractional-delay
physical checks, detection, acquisition, tracking, wall propagation and
two-path fitting. Cache-specific checks cover cold/warm/uncached equality at
2/3/4/8/20 samples per chip, read-only arrays, bounded capacity, eviction,
explicit clearing and oversized-period bypass. Existing parallel acquisition
checks also preserve serial results and caller IQ.

The saved baseline is
`output/cpu-receiver-speed/baseline_code_delay_cpu.py`, SHA-256
`b41d0961b767f932a3a4e696eb802e0ce06141bdb514d7a37f17a9494018603a`.
Retain the pre-change `output/cpu-batch-receiver/` recording for the frozen
comparison; regenerating it is a different provenance record.

```bash
PYTHONPATH=python:. python benchmarks/bench_cpu_receiver_cache.py \
  --baseline-source output/cpu-receiver-speed/baseline_code_delay_cpu.py \
  --frozen-dir output/cpu-batch-receiver --repeats 7
```

`output/cpu-receiver-speed/comparison.json` contains source/input hashes, raw
timings for all 14 runs, exact-result checks and cache statistics. Calibration
scores and the complete receiver trace remain in the frozen recording.
