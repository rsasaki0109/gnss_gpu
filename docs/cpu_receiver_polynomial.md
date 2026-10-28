# Removing duplicate polynomial preprocessing

The next [observation-preparation experiment](cpu_receiver_preparation.md)
reuses IQ conversion and normalization across carrier trials. This page retains
the polynomial experiment's original measurements.

After reference-template caching, a warmed 32-block profile still spent about
0.057 s of 0.571 s in `np.trim_zeros` and its helpers, across 2,388 calls.
Each call processed just three stationary-polynomial coefficients. The next
operation, `np.roots`, already removes leading and trailing zero coefficients.

The delay fitter now passes the normalized polynomial directly to `np.roots`.
Normalization, the root solver, root ordering, candidate filtering and score
selection remain unchanged. This also handles a quadratic degenerating into
a linear or constant polynomial. No closed-form replacement or approximation
was introduced.

## Frozen comparison

The baseline is the previous, template-cached fitter, rather than the original
uncached implementation. The benchmark reuses the same frozen 128-block IQ,
199 calibration scores, gate and complete receiver trace described in
[the earlier cache experiment](cpu_receiver_cache.md).

Seven repetitions alternate baseline/candidate order, with one receiver worker
and four generation threads on the same four-core environment. Stage medians:

| Stage | Previous cached fitter | Duplicate preprocessing removed | Speed ratio |
| --- | ---: | ---: | ---: |
| Receiver | 1.100 s | 0.969 s | 1.135× |
| IQ generation + receiver | 1.151 s | 1.018 s | 1.131× |

IQ generation medians were 50.08/55.72 ms; that code did not change. Stage
medians need not sum to the total median. Calibration, warmup, result comparisons
and serialization are excluded from timings. Every timed IQ buffer and every
field of all 128 receiver results matched exactly. All 199 frozen noise scores
also matched exactly. The earlier experiment's output was preserved.

The 179 relevant tests passed. A separate check compared direct and explicitly
trimmed roots for 1,962 finite coefficient triples, including leading/trailing
zeros, a constant, repeated roots and complex roots, with exact equality on
NumPy 2.3.4. This sampled check is not exhaustive floating-point verification.

These are additional gains relative to the cached fitter in this experiment;
ratios from separate timing sessions should not be multiplied into a measured
cumulative claim. The same synthetic-input and physical-model limitations apply.
Processing 128 ms of input still takes about one second here.

## Reproduction

The saved pre-change module is
`output/cpu-receiver-roots-speed/baseline_code_delay_cpu.py`. Run:

```bash
PYTHONPATH=python:. python benchmarks/bench_cpu_receiver_cache.py \
  --baseline-source output/cpu-receiver-roots-speed/baseline_code_delay_cpu.py \
  --frozen-dir output/cpu-batch-receiver --repeats 7 \
  --output output/cpu-receiver-roots-speed/comparison.json
```

The report records baseline/candidate/benchmark hashes, all 14 timing rows and
the frozen-input identity. `root_equivalence.json` records the separate roots
check, its random seed, NumPy version and explicit special cases. Its random
corpus used `default_rng(20261024)`: 2,000 normally distributed triples, each
coefficient independently replaced with zero with probability 0.3, excluding
all-zero triples, plus the eight explicit cases in the report; each triple was
normalized by its maximum absolute coefficient before both root calls.
