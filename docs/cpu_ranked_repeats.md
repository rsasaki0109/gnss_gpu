# Repeated independent calibration of the ranked gate

The [first ranked-gate experiment](cpu_ranked_calibration.md) improved sensitivity
but did not test calibration-to-calibration variability. This experiment preserves
that initial calibration exactly and adds two independent new calibration draws.
All three draws are evaluated on one new shared validation set.

## What changes and what stays fixed

The estimator, 101-bin noise smoothing, assistance, signal distributions and
fourth-largest gate rule are frozen. Each draw has 1,000 single-path records per
white/center/edge profile. Both methods combine their three stratum thresholds
by taking the maximum: one uses the largest score per stratum; the other uses
the fourth-largest. No gate is selected based on echo results.

The initial 3,000 records and gates are reused by reference, with file hashes
and exact gate equality checked. The two new draws use disjoint IDs beginning
400000/410000/420000 and 430000/440000/450000, respectively: 6,000 new fits total.
The initial draw is a previously inspected development reference, not newly
prospective evidence. The two new draws independently check it; a range computed
over all three is descriptive rather than a population variance estimate.

Each draw retains the same conditional calibration failure bound
`3*BinomialCDF(3;1000,.01) = 0.03021796`. This is a per-draw joint statement over
its three specified noise strata, not a 95% simultaneous guarantee over all
three calibration draws. The maximum gate is stricter on the same data and is
covered whenever that draw's ranked-gate bound holds.

New evaluation uses 128 single-path records per profile (IDs 470000–470127)
and 128 white-noise, same-phase 0.1-chip echoes with voltage ratio 0.8
(IDs 480000–480127). All six method/draw combinations use these same 512
observations. There are 3,072 selection outcomes, not 3,072 independent samples.
All six gates are frozen before the new evaluation begins. Success means an
available absolute direct-range error <10 m; unavailable estimates remain failures.

Three primary paired comparisons test ranked versus maximum success on the
common echo set, one per calibration draw. Exact two-sided McNemar tests use
a three-test Bonferroni correction. Shared observations do not invalidate that
correction, but cannot be pooled as independent evidence. Pointwise intervals
are descriptive. Only three draws are available, so lower observed threshold
spread cannot establish a general reduction in calibration variance.

All signals remain static isolated PRN 1 at 37 reference dB-Hz, with ideal
carrier/nav wipeoff for 200 ms, a 3.683 MHz one-sided cutoff and fixed coarse
assistance. This does not validate RF dynamics, forest/rain physics, acquisition,
tracking or positioning performance.

## Results

All three gates are evaluated on the same new white-noise close-echo observations:

| Calibration draw | Maximum threshold | Ranked threshold | Maximum successes | Ranked successes | Adjusted paired p |
| --- | ---: | ---: | ---: | ---: | ---: |
| Preserved initial draw | 1.224553 | -11.266411 | 35/128 | 89/128 | 3.33e-16 |
| New draw 1 | -3.470958 | -10.820802 | 51/128 | 89/128 | 2.18e-11 |
| New draw 2 | 0.795095 | -10.597230 | 37/128 | 89/128 | 1.33e-15 |

Each predeclared paired comparison supports improvement after the three-test
Bonferroni correction: gains are 54, 38 and 52, with zero success losses in each
comparison. The ranked success proportion is 69.5%, with pointwise exact 95%
interval 60.8–77.4%. The identical 89/128 counts are repeated selections of the
same observations, not three independent estimates to combine into a narrower
confidence interval. Thirty-nine of 128 echoes still miss the 10 m criterion.
All outputs are available; this is not a complete separation solution.

Observed calibration variability over these three draws:

| Metric | Maximum gate | Ranked gate |
| --- | ---: | ---: |
| Threshold range | 4.6955 | 0.6692 |
| Sample threshold standard deviation | 2.5959 | 0.3407 |
| Echo success-count range on shared data | 16 | 0 |

These statistics describe a smaller spread for the ranked gate in this experiment.
They do not establish its population variance or prove that every future
calibration is stable. The preserved first draw was previously inspected and
included a large maximum; the two new draws provide additional prospective
evidence, but the total number of calibration realizations remains small.

For each of the three noise profiles and all six gates, held-out direct-only
controls have 128/128 range successes, zero false two-path selections and zero
unavailable estimates. Each 0/128 false-selection count has a pointwise exact
95% upper bound of 2.84%. Shared data cannot be pooled to claim a lower bound.
The earlier experiment's 1–3 false selections per 299 records remain valid
observations and are not superseded by zero counts on this smaller new sample.

![Repeated-calibration comparison](../output/cpu-ranked-repeats/comparison.png)

The new data support retaining the ranked rule as the experimental static-model
candidate: it shows higher close-echo sensitivity and less observed threshold
spread here. Further calibration draws would be needed for a precise stability
claim. A separate important limitation is the 200 ms static-signal assumption:
time-varying reflections can contaminate the block-difference noise estimate,
so dynamic-channel validation is needed before claiming realistic receiver gains.
No acquisition, tracking or PVT entry point has been changed.

## Reproduction and audit

```bash
source /workspace/.gnss-env/activate.sh
pytest -q tests/test_ranked_repeats.py
python experiments/run_cpu_ranked_repeats.py
python experiments/audit_cpu_ranked_repeats.py
python experiments/plot_cpu_ranked_repeats.py
```

The destination `output/cpu-ranked-repeats` must be empty. Protocol, all new
calibration records, hashes of the preserved original records, all frozen gates,
new evaluation records, outcome summaries and descriptive variability statistics
are saved. Two tests verify per-draw paired denominators and across-draw rather
than across-observation variability calculations. The audit checks all gates,
independent calibration groups, shared evaluation identities and denominators,
source/prior hashes, then replays a calibration record from each new draw and
two evaluation records with the same implementation.

The reused single-run summary contains an unadjusted
`primary_improvement_supported` flag. For this repeated study, conclusions use
`adjusted_primary_p`; the audit writes explicit corrected decisions to
`family_decisions.json` to distinguish the two scopes.
