# Integration time versus close-reflection separation

This experiment tests whether more coherent integration helps the 0.1-chip
reflection cases that remain difficult after the [v2 numerical fix](cpu_bandlimited_v2.md).
The v2 estimator and all selection/assistance settings are frozen. Only the
amount of coherently averaged data changes: 20, 80 or 200 ms.

## Ideal coherence and matched noise

Each replica generates ten static 20 ms observations of the same isolated PRN 1
passband spectrum. Carrier and navigation polarity are **perfectly removed**.
The first 1/4/10 blocks are averaged for 20/80/200 ms, respectively. Per-block
complex spectral noise is independent with variance `1/(C/N0*0.020)`, so the
average has variance `1/(C/N0*T)` without increasing signal amplitude or changing
its post-filter power. Shared spectral harmonics and random initial delay/phase
are paired across bandwidths and echo conditions.

This is the preceding ideal periodic spectral Gaussian model, not raw RF
simulation over 200 ms. There is no motion, changing reflection phase/amplitude,
residual carrier, clock drift, data-bit uncertainty or finite-filter transient.
GPS L1 C/A navigation bits last 20 ms; extending integration beyond that requires
correct bit wipeoff. The experiment assumes it and does not implement or verify
a receiver capable of providing it. Longer integration also increases latency.

## Frozen comparison

There are 64 fresh replicas per condition, using a new seed domain `0x434F4845`.
The grid has direct C/N0 37 dB-Hz, one-sided bandwidths 0.921/3.683 MHz,
direct-only controls, and voltage-ratio-0.8 echoes at gaps 0.1/0.5 chip and phases
0/pi/2. Across three durations this produces 1,920 estimates in 30 cells. The
64 independent replica IDs are shared across cells; the 1,920 outputs are not
independent trials. Signal truth is used only for synthesis and error scoring.

Success is an available absolute direct-range error <10 m, with all 64 attempts
in the denominator. Unavailable selected estimates count as failures. At each
duration, the single-path baseline and selected-method results are both saved.
Two primary paired comparisons were fixed before evaluation: selected-method
success at 200 versus 20 ms, wide bandwidth and a 0.1-chip echo, separately for
phases zero and pi/2. Exact two-sided McNemar tests receive a Bonferroni correction
for the two-comparison family at 0.05. Other comparisons, including 80 ms, are
exploratory. Binomial 95% intervals and paired bootstrap intervals are pointwise,
not simultaneous. A finite run with no failures does not establish zero risk.

## Results

The two primary comparisons do not support a uniform close-path improvement:

| Phase, 0.1-chip gap, wide bandwidth | 20 ms successes | 200 ms successes | Paired gains / losses | Adjusted p |
| --- | ---: | ---: | ---: | ---: |
| 0 | 12/64 | 12/64 | 7 / 7 | 1.0 |
| pi/2 | 27/64 | 64/64 | 37 / 0 | 2.91e-11 |

For pi/2 the observed increase is 57.8 percentage points, with a descriptive
pointwise paired bootstrap interval of +45.3 to +70.3 points. The 200 ms success
proportion has a pointwise exact 95% interval of 94.4–100%, not a guarantee of
perfect performance. For phase zero the paired difference is zero, with a
pointwise bootstrap interval of -10.9 to +10.9 points. No improvement is supported.

All selected-method success counts follow, each out of 64 attempts:

| One-sided B | Duration | Direct only | 0.1 chip, phase 0 | 0.1 chip, pi/2 | 0.5 chip, phase 0 | 0.5 chip, pi/2 |
| --- | --- | --- | --- | --- | --- | --- |
| 0.921 MHz | 20 ms | 41 | 18 | 29 | 5 | 29 |
| 0.921 MHz | 80 ms | 62 | 8 | 19 | 20 | 48 |
| 0.921 MHz | 200 ms | 64 | 4 | 22 | 36 | 60 |
| 3.683 MHz | 20 ms | 58 | 12 | 27 | 58 | 58 |
| 3.683 MHz | 80 ms | 64 | 5 | 53 | 63 | 64 |
| 3.683 MHz | 200 ms | 64 | 12 | 64 | 64 | 64 |

The narrow-band 0.1-chip, same-phase condition actually loses threshold successes
with longer averaging, from 18 to 4. All those fits retain the single-path model;
reduced random noise does not remove its multipath bias and can reduce the chance
of a noise realization placing an estimate inside the 10 m success window.
In the narrow-band pi/2 close-path cell, 200 ms has 18 unavailable estimates,
all counted as failures. There are no unavailable estimates in the wide-band grid.

An exploratory diagnostic helps identify the next development target: for the
wide-band, 200 ms, 0.1-chip, same-phase condition, raw two-path estimates have
RMSE 7.68 m across all 64 attempts, but only 15/64 records select that model.
The final selected method succeeds in 12/64. This suggests the fixed BIC selection
rule deserves investigation; it does **not** justify always returning two paths,
which can fit a fictitious echo in direct-only signals. A replacement rule needs
independent calibration on single-path controls and untouched signal evaluation,
including failures. This diagnostic was examined after the primary comparisons
and is not a predeclared hypothesis test.

![Integration duration and range successes](../output/cpu-coherent-multipath/comparison.png)

Longer ideal integration helps some close echoes, but the effect depends on
relative phase and model selection. It is not a blanket remedy, and these results
do not establish that a real receiver can maintain 200 ms coherence.

## Reproduction and checks

```bash
source /workspace/.gnss-env/activate.sh
pytest -q tests/test_coherent_multipath.py
python experiments/run_cpu_coherent_multipath.py
python experiments/audit_cpu_coherent_multipath.py
```

`output/cpu-coherent-multipath` must be empty for the experiment. Protocol and
hashes are written before evaluation; all candidate errors, failures, observation
hashes and selection diagnostics are saved in JSONL. The script produces cell
and paired-comparison summaries and PNG/PDF figures. Three tests check the
integration noise variance and preserved signal, exact shared-harmonic pairing,
and independent/reproducible replica generation. The audit checks all identities,
denominators, paired contingency tables, source and prior-artifact hashes, and
exactly replays two tasks (six estimator outputs) with the same implementation.
No earlier source or experimental output is overwritten.
