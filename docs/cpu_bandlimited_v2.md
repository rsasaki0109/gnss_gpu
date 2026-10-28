# Numerical tie handling in the bandlimited two-path fit

The [independent replication](cpu_bandlimited_replication.md) exposed six lost
range estimates at 55 dB-Hz, wide bandwidth, 0.5-chip gap and pi/2 phase. The
single-path estimates all succeeded, but the selected estimator rejected six
two-path fits for optimizer nonconvergence.

## Cause and bounded correction

Instrumenting all twelve optimization starts showed that every one of these
six records also had a converged candidate at essentially the same solution.
The failed winner's objective was only 7.1e-15 to 2.8e-14 lower; the delay
parameters differed by at most 2.82e-9 chip. Selecting the smallest floating-point
objective unconditionally allowed a line-search `ABNORMAL` result to beat an
equivalent converged result.

`bandlimited_multipath_v2_cpu.py` preserves the frozen v1 implementation and changes
only multistart winner selection. When the minimum-objective candidate failed,
a successful alternative may replace it only if:

- objective values agree within `64 * float64_eps * max(1, abs(best_objective))`;
- both first-delay and excess-delay parameters agree within 1e-7 chip (about
  0.03 mm);
- the alternative's objective and parameters are finite.

The best such converged candidate is used. A different solution, a materially
worse objective, or absence of a converged equivalent preserves the failure.
The original BIC, boundary and local-information checks still run afterward.
This does not declare a failed optimizer successful or bypass ambiguity checks.
It also does not prove that any optimizer found a global minimum.

The six known development failures now select the two-path solution and produce
errors below 10 m. Those cases motivated the change and are regression tests,
not held-out evidence. Twelve tests pass, covering those six records, equivalent
ties, rejection of worse/different/nonfinite alternatives, and preservation of
existing successes/all-failed candidates.

## Fresh validation design

Before evaluation, a new protocol freezes the code and all 20 previous signal
conditions with 64 fresh replicas per cell (IDs 20000–20063). These are disjoint
from development IDs 0–7 and the preceding replication IDs 10000–10127.
Each of 1,280 observations is fitted by both v1 and v2, using four CPU processes.
Synthesis is repeated for v2 and its observation hash must match v1 exactly.

Success remains an available absolute direct-range error below 10 m; all attempts
stay in the denominator. Every gain, loss, nonconvergence and unavailable result
is reported. Statistical intervals/tests are exploratory and pointwise; the
study is a descriptive regression screen, not a population noninferiority test.
Replicas are paired across all cells: the pooled counts are not 1,280 independent
trials. Carrier/nav and coarse-delay assistance remain ideal/known. No acquisition,
tracking, PVT, real RF or forest/rain claim follows from this test.

## Results

Across all 1,280 paired observations, v2 recovers **30 available estimates**:
unavailable counts fall from 37 to 7. Of those recovered estimates, 27 have error
below 10 m and three do not. No previously available selected output changes,
including its numerical error. There are 27 success gains and zero success losses
in this finite grid. This does not prove zero population regression risk.

Selected-method success counts, v1 -> v2, all out of 64 attempts:

| C/N0 | One-sided bandwidth | Direct only | 0.1 chip, phase 0 | 0.1 chip, pi/2 | 0.5 chip, phase 0 | 0.5 chip, pi/2 |
| --- | --- | --- | --- | --- | --- | --- |
| 37 | 0.921 MHz | 46 -> 46 | 15 -> 15 | 23 -> 23 | 0 -> 0 | 32 -> 32 |
| 37 | 3.683 MHz | 60 -> 60 | 15 -> 15 | 27 -> 27 | 57 -> 59 | 56 -> 61 |
| 55 | 0.921 MHz | 64 -> 64 | 0 -> 0 | 43 -> 44 | 60 -> 62 | 62 -> 64 |
| 55 | 3.683 MHz | 64 -> 64 | 59 -> 64 | 61 -> 64 | 59 -> 64 | 62 -> 64 |

For the condition that exposed the original regression (55 dB-Hz, wide bandwidth,
0.5 chip, pi/2), both fresh failures recover, changing 62/64 to 64/64. Close echoes
at 37 dB-Hz remain difficult: this numerical correction does not resolve their
physical/model-selection limitations. All four direct-only controls are unchanged.

The completed audit verifies source hashes, preservation of previous experiment
artifacts, all record identities, recomputed cell reports, preservation of formerly
available estimates, and exact replay of two serialized records. The original
runner's last audit assertion initially compared JSON lists with in-memory tuples;
the computations and saved results were complete. The executed source is archived
as `executed_validation_source.py`, with its original protocol hash. The runner
now normalizes both comparison operands; the separate audit verifies that this
single comparison correction is its only post-run source change. No estimator,
signal generation, task selection or analysis setting changed after evaluation.

## Reproduction

```bash
source /workspace/.gnss-env/activate.sh
pytest -q tests/test_bandlimited_multipath_v2.py
python experiments/validate_cpu_bandlimited_v2.py
python experiments/audit_cpu_bandlimited_v2.py
```

The output directory `output/cpu-bandlimited-v2` must be empty. The run saves
protocol/source hashes before evaluation, all paired records, cell summaries and
an audit. It verifies all original source and artifact hashes, record identities,
and exact preservation of every previously available v1 selected output. Two
records are replayed in full using the same implementation. Original experiments
remain loadable and reproducible; v2 has not replaced v1 in their entry points.
