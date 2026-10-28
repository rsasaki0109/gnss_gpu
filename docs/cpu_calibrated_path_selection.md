# Conditional calibration of one-path versus two-path selection

The [integration-duration study](cpu_coherent_multipath.md) found usable raw
two-path fits that the fixed BIC rule often rejected for close, same-phase echoes.
This study calibrates the selection score on independent **single-path signals**
before evaluating any new echo data. The underlying v2 delay optimizer and its
convergence, boundary and local-uncertainty guards remain unchanged.

## Scope and statistical meaning

The context is fixed: isolated PRN 1, one-sided cutoff 3.683 MHz, direct C/N0
37 dB-Hz, 200 ms ideal coherent integration and the default coarse-delay search.
The null is a real direct signal with **no echo**, not noise-only input. Direct
delay is uniform [-0.3,0.3] chip, common phase is uniform [-pi,pi], and noise is
the preceding periodic spectral Gaussian model. Carrier/nav wipeoff is perfect.

The statistic is the frozen v2 BIC improvement, calculated from the full
multistart two-path and single-path fits. The original rule accepts a preferred
two-path model when the statistic exceeds zero. The new rule uses the maximum
of 299 independently generated single-path calibration scores, with **strict
exceedance**. The gate is written and hashed before held-out fitting starts.

For iid calibration records, the maximum-order-statistic construction gives
`P(null exceedance probability > 0.01) <= 0.99**299 < 0.05`. Thus the conditional
null exceedance probability is at most 1% with at least 95% calibration confidence.
This is a marginal statement over the specified nuisance-parameter distribution,
not uniform control for every delay/phase. Final two-path selection also requires
the unchanged local guards, so its null rate cannot exceed the raw gate-exceedance
rate. Neither is a receiver acquisition false-alarm or positioning-integrity rate.

`fit_calibrated_paths` uses the fixed bandwidth and default v2 fit. It cannot
verify the input's actual C/N0, coherence or channel distribution. Its gate must
not be transferred to different signal powers, integration durations, bandwidths,
assistance or interference without recalibration. The simulation is not evidence
that a physical receiver meets these assumptions.

## Independent evaluation

Calibration IDs 40000–40298, held-out single-path IDs 50000–50298 and echo IDs
60000–60127 are disjoint under the existing coherent-signal seed domain. Echo
replicas are paired across four conditions: voltage ratio 0.8, delays 0.1/0.5
chip and phases 0/pi/2. There are 299 calibration fits, 299 held-out single-path
fits and 512 held-out echo fits, with 128 independent replicas per echo cell.

One primary comparison was fixed before calibration: selected range success
under the calibrated gate versus the original BIC gate, for 0.1-chip same-phase
echoes. An exact two-sided McNemar test uses alpha 0.05; other comparisons are
exploratory. Success requires an available absolute direct-range error <10 m.
All attempts, including ambiguous estimates, remain in the denominator. Score
exceedance, actual two-path selection and range success are separate outcomes.
Reported exact binomial/paired bootstrap intervals are pointwise, not simultaneous.

## Results and tradeoff

The frozen calibrated gate is **-12.057079** on the BIC-improvement scale,
compared with zero for the original selector. This is an empirically calibrated
score gate; a negative gate is not a claim that BIC itself prefers two paths.

| Held-out condition | Original range successes | Calibrated range successes | Calibrated two-path selections |
| --- | ---: | ---: | ---: |
| Direct only | 299/299 | 298/299 | 2/299 |
| 0.1-chip echo, phase 0 | 37/128 | 86/128 | 103/128 |
| 0.1-chip echo, pi/2 | 127/128 | 127/128 | 128/128 |
| 0.5-chip echo, phase 0 | 128/128 | 128/128 | 128/128 |
| 0.5-chip echo, pi/2 | 128/128 | 128/128 | 128/128 |

The primary close, same-phase comparison has 49 paired gains and zero losses:
success increases from 28.9% to 67.2%, a difference of 38.3 percentage points.
The exact paired p-value is 3.55e-15. The calibrated success proportion has a
pointwise exact 95% interval of 58.3–75.2%; the descriptive paired bootstrap
interval for the difference is +29.7 to +46.9 points. No estimates are unavailable
in any held-out cell. The method still fails the 10 m criterion in 42/128 of the
primary echo records; selecting two paths does not guarantee correct ranging.

The held-out direct-only group has **2/299 false two-path selections (0.67%)**.
One of these produces range error >=10 m, causing one loss against the original
selector. The false-selection count equals the raw score-exceedance count here.
Its two-sided exact 95% interval is **0.081–2.395%**: this finite validation sample
does not independently establish that the probability is below 1%. The 1%/95%
statement above comes from the order-statistic calibration construction and its
distributional assumptions, not from that empirical interval. It allows rare
errors and is not an integrity guarantee.

This provides a controlled improvement for the predeclared close-reflection
condition, with an observed direct-only regression that must remain visible.
The calibration cannot be applied directly to forests, rain, moving channels,
unknown carrier/navigation data, or other C/N0 values. The next validation target
is robustness of the gate to realistic departures from this fixed single-path
calibration distribution, before integration into acquisition or positioning.

## Reproduction and audit

```bash
source /workspace/.gnss-env/activate.sh
pytest -q tests/test_calibrated_path_selection.py
python experiments/run_cpu_calibrated_path_selection.py
python experiments/audit_cpu_calibrated_path_selection.py
```

The output directory `output/cpu-calibrated-path-selection` must be empty. The
run saves its protocol before calibration, all calibration scores/fits, the frozen
gate before evaluation, all held-out fits and failures, summaries and completion
hashes. Ten tests check sample-size requirements, strict threshold ties,
promotion below the original BIC threshold, retention of all local guards and
invalid calibration records. The audit recomputes the maximum threshold, checks
disjoint identities and all selected outcomes/denominators, verifies source and
prior-artifact hashes, and replays one calibration record and one echo record,
including the public inference entry point.
