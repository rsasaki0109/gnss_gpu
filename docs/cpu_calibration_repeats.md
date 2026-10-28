# Sensitivity to independent calibration data

The [2x2 ablation](cpu_noise_ablation.md) placed a small observed white-noise
sensitivity loss on the broader-calibration step, but did not establish that
loss statistically. This experiment repeats calibration three times with
independent new data while holding the weighted estimator and evaluation
observations fixed.

## Design and scope

Each independent calibration draw contains 477 single-path observations per
profile (white, center, edge), matching the preceding ablation's sample size
and maximum-score rule. The weighted fit is unchanged. Each draw produces a
white-only gate and a multi-profile gate equal to the largest of the three
profile maxima. The 4,293 calibration observations use disjoint replica-ID
ranges beginning at 200000/210000/220000, with separate 1000-ID offsets per
profile; none appeared in earlier runs. All six gates are frozen before
evaluation and none is selected or adjusted using echo results.

For each calibration draw, `3*0.99**477` bounds the probability that any of its
three stratum exceedance rates exceeds 1%. This is a per-draw conditional
calibration statement. There is no 95% simultaneous claim over all nine strata
across the three draws. White-only gates only cover the white stratum; broad
gates cover the three specified synthetic profiles, not arbitrary channels.

Fresh evaluation has 128 single-path observations per profile and 128 white-noise
same-phase echo observations (voltage ratio 0.8, gap 0.1 chip), for 512 observations.
The same observations are reused across all six gates to isolate variation due
to calibration. This yields 3,072 selection outcomes, **not 3,072 independent
observations**. Null/echo evaluation IDs 240000–240127 and 250000–250127 are
disjoint from calibration and previous experiments.

The three predeclared primary paired comparisons are multi-profile versus
white-only success on the common echo set, one comparison per calibration draw.
Exact two-sided McNemar tests use a three-test Bonferroni correction. All successes
require available absolute direct-range error <10 m, with unavailable estimates
counted as failures. Descriptive binomial/bootstrap intervals are pointwise.
Pooling all draws' success counts would conceal the shared evaluation data and
is not used for significance testing.

Only three calibration realizations are sampled. Their threshold spread is
descriptive, not a precise population distribution or confidence interval for
future calibration performance. Signal conditions remain ideal: static aligned
PRN 1, reference C/N0 37 dB-Hz, 200 ms with perfect carrier/nav wipeoff, one-sided
cutoff 3.683 MHz and fixed assistance. No acquisition, tracking or PVT claim is
made. Spectral noise is synthetic rather than measured forest/rain interference.

## Results

The three independent calibrations produce substantially different frozen
thresholds despite identical algorithms and sample sizes:

| Calibration draw | White-only threshold | Multi-profile threshold | Profile setting multi threshold | White-only successes | Multi-profile successes |
| --- | ---: | ---: | --- | ---: | ---: |
| 1 | -5.886326 | -5.886326 | white | 57/128 | 57/128 |
| 2 | -10.385978 | -5.362683 | center | 73/128 | 54/128 |
| 3 | -10.554947 | -10.554947 | white | 74/128 | 74/128 |

All success counts use the **same** fresh white-noise, 0.1-chip, same-phase echo
observations. No model inputs, fitted candidates or true signal parameters change
between the six selections. No echo estimates are unavailable.

Draw 2 loses 19 successes and gains none when calibration coverage expands:
the success difference is -14.8 percentage points. Its exact two-sided paired
p-value is 3.81e-6, or **1.14e-5 after three-test Bonferroni correction**. The
descriptive pointwise paired-bootstrap interval is -21.1 to -8.59 points.
Draws 1 and 3 have identical white/multi gates and identical outcomes (adjusted
p = 1). Their equality is caused by the white stratum already supplying the
largest calibration score, not evidence that broader calibration is universally
free of a sensitivity cost.

Even white-only calibration varies from 57 to 74 successful ranges on the same
128 records. Multi-profile calibration varies from 54 to 74 (42.2–57.8%). These
are observed ranges over three calibration draws, not confidence intervals for
all future calibrations. This identifies calibration-set variation as a material
source of the previously observed sensitivity changes. It does not establish
that broad calibration always degrades performance or that weighting alone
caused the earlier regression.

Every direct-only profile has 128/128 range successes and zero false two-path
selections under all six gates. Each gate/profile cell still has a pointwise
exact 95% false-selection upper bound of 2.84%; sharing the same null observations
across gates does not multiply the independent validation sample size. No new
claim of an empirically verified sub-1% rate follows from these zero counts.

![Threshold and held-out success variation](../output/cpu-calibration-repeats/comparison.png)

The maximum-order-statistic calibration remains valid under its assumptions,
but its conservative threshold can be sensitive to rare calibration records.
The next candidate is an order-statistic rule using more than the single maximum,
with a larger calibration sample and an exact binomial tolerance bound chosen
before looking at evaluation outcomes. Discarding the maximum or selecting the
best-performing gate from these three draws would invalidate the intended
calibration procedure; neither action was taken here.

## Reproduction and audit

```bash
source /workspace/.gnss-env/activate.sh
pytest -q tests/test_calibration_repeats.py
python experiments/run_cpu_calibration_repeats.py
python experiments/audit_cpu_calibration_repeats.py
python experiments/plot_cpu_calibration_repeats.py
```

The destination `output/cpu-calibration-repeats` must be empty. The protocol is
saved before calibration; every fit and block hash, all frozen gates and every
selection outcome are preserved. Two tests check that calibration groups are
not pooled, the correct per-group gates are produced, and scoring truth cannot
change selection. The audit checks source/prior hashes, group independence,
identities, all gates, choices and denominators, then replays one calibration
record from each draw and two held-out records with the same implementation.
