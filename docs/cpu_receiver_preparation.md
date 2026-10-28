# Reusing observation preparation within a carrier search

A subsequent [54-scenario stress matrix](cpu_receiver_stress.md) checks exact
receiver equivalence across strengths, offsets and interruptions, and exposes
the existing out-of-grid carrier failure mode.

Acquisition and golden-section carrier refinement previously converted IQ to
float64, checked every sample, built a complex signal and normalized its scale
at every tested frequency. They now prepare one immutable observation per
search or refinement call. Each frequency trial creates its own carrier-wiped
array; energy, FFT, projections, polynomial roots and estimates are still
computed separately in the original arithmetic order.

The prepared observation owns its signal data, is local to the call and is not
stored in global caches. Concurrent grid workers only read it. Caller snapshot
behavior remains in acquisition and refinement. Acquisition and refinement
prepare separate observations; the tracker does not retain IQ between blocks.
The public `estimate_ca_delay` interface still validates and prepares each call.
All-zero observations still validate the supplied carrier before returning None.

## Fixed-input comparison

This experiment compares the complete receiver before and after this change.
The baseline already includes template caching and the earlier removal of
[duplicate polynomial preprocessing](cpu_receiver_polynomial.md). The frozen
128-block synthetic recording, whole-grid gate and 199 noise-calibration
blocks are the same as in the earlier experiments.

Seven sequential repetitions alternate old/new order after both paths have
been warmed. Generation uses four Numba threads and receiver search uses one
worker in the four-core environment. No tests run during the timing trials.

| Stage, median | Before | After | Speed ratio |
| --- | ---: | ---: | ---: |
| Receiver construction and updates | 0.963 s | 0.813 s | 1.185× |
| IQ generation + receiver | 1.014 s | 0.867 s | 1.170× |

Unchanged generation measured 50.31/52.16 ms. Stage medians need not sum to
the total median. Calibration, warmup, comparisons and serialization are
outside the timers. Every field of all 128 receiver results matched the frozen
trace exactly on every timed trial, and both receiver versions reproduced all
199 frozen calibration scores exactly. IQ buffers matched bit for bit.

These repetitions reuse one synthetic recording; they are not independent
GNSS trials. The result does not establish general speedup or RF accuracy.
128 ms of input still requires about 0.87 s including generation here, so this
path is not real time. Ratios from separate experiments should not be multiplied
into a claimed measured cumulative speedup.

## Verification and reproduction

187 relevant tests pass. New checks compare shared preparation against separate
public fitter calls for every carrier bin, including strided float32/float64
input and float64 scales of 1e-200/1e200. They also verify parallel equality,
ownership after caller mutation, read-only prepared samples, unchanged samples
after concurrent fits, and invalid-carrier rejection on zero IQ.

Saved pre-change modules are in `output/cpu-receiver-preparation-speed/`:
`code_delay_cpu.py`, `acquisition_cpu.py`, and `tracking_cpu.py`. The benchmark
loads these together with their original dependency bindings, restores the
import registry and selects the old/current tracker only within the experiment.
All 14 raw timing rows and baseline/candidate/benchmark hashes are recorded in
`comparison.json`. The original frozen recording remains untouched.

```bash
PYTHONPATH=python:. python benchmarks/bench_cpu_receiver_preparation.py \
  --baseline-dir output/cpu-receiver-preparation-speed --repeats 7
```

The earlier `bench_cpu_receiver_cache.py` remains a fitter-only comparison. Its
adapter deliberately performs preparation on every call for both versions,
so it does not measure this search-level reuse. The adapter's result equality
and restoration after normal exit and exceptions were checked separately.
