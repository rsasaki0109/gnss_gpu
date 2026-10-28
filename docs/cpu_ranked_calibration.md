# Fourth-largest-score calibration with an exact tolerance bound

The [calibration-repeat study](cpu_calibration_repeats.md) showed that a maximum
score gate could move substantially between independent calibration sets. This
change uses a rank fixed by the sample size and required probability budget,
not by observed signal performance. The noise-aware estimator remains frozen.

## Rank selection and guarantee

For n iid single-path null scores, a threshold equal to the kth-largest score,
with strict exceedance, satisfies

`P(null tail probability > alpha) <= BinomialCDF(k-1; n, alpha)`.

This follows because a threshold below the true (1-alpha) quantile requires
fewer than k calibration samples above that quantile. Strict ties are
conservative. The procedure chooses the largest permissible k based only on
n, alpha and delta, before looking at any scores or echo outcomes.

With n=1,000 per stratum, alpha=0.01 and delta=0.05/3, **k=4** is allowed:
`BinomialCDF(3;1000,.01)=0.01007265`. Rank 5 would have failure bound 0.0286864,
which exceeds the per-stratum budget. The combined gate is the maximum of the
three fourth-largest scores from white/center/edge noise. A union bound gives
joint calibration failure probability <=0.030218, below 0.05. Thus the same
conditional 1% false-score-exceedance budget holds in each of these three
specified distributions with at least 95% calibration confidence.

The maximum comparator uses **the same 1,000 records per stratum** and is always
at least as strict as the ranked gate, so the ranked gate's guarantee event also
covers it. This comparison changes only the gate rule, not sample size, fitting
or calibration coverage. It does not discard calibration observations or choose
a favorable gate based on held-out results. Final two-path selection still
requires the existing convergence, boundary and uncertainty checks.

The fourth-largest score is unaffected if just the already-largest score becomes
larger. That is a structural robustness property, not proof of lower statistical
variance across calibration draws. Its threshold can still vary, and this study
uses only one fresh calibration realization. Larger n supplies the probability
budget needed for using a rank below the maximum; simply removing a maximum from
the old small sample would not preserve its original calibration guarantee.

## Frozen evaluation

The context remains isolated PRN 1, 37 reference dB-Hz, 200 ms ideal static
carrier/nav coherence, one-sided cutoff 3.683 MHz and fixed delay assistance.
All 3,000 calibration records are fresh (profile-specific ID ranges starting
300000/310000/320000). The rank and full task list are saved before calibration.
Both combined gates are saved before any held-out fitting begins.

Evaluation includes 299 fresh single-path records per noise profile and 128
same-phase, 0.1-chip, voltage-ratio-0.8 echo records per profile: 1,281 observations.
Calibration, null and echo groups use disjoint IDs. Evaluation replicas are
paired across profiles and methods, not independent across all records. One
primary comparison is fixed: ranked versus maximum range success for white-noise
same-phase echoes, using an exact two-sided McNemar test at 0.05. Other comparisons
are exploratory. All attempts remain in denominators; unavailable estimates
fail the available absolute error <10 m success criterion. Binomial/bootstrap
intervals are pointwise, not simultaneous.

The guarantee is conditional on the specified iid null distributions, estimated
noise-weighting pipeline and static signal assumptions. It does not transfer to
arbitrary noise spectra, motion, unknown carrier/nav data, acquisition or PVT.

## Results and cost of reduced conservatism

The frozen per-profile scores and combined gates are:

| Profile | Maximum | Fourth largest |
| --- | ---: | ---: |
| white | -10.688040 | -13.015057 |
| center | 1.224553 | -12.393318 |
| edge | -9.379708 | -11.266411 |
| **Combined maximum across profiles** | **1.224553** | **-11.266411** |

The large center-profile maximum makes the comparator particularly conservative
in this calibration draw. It remains in the saved calibration data and was not
removed as an outlier. The fourth-largest rule was fixed before these values
were generated.

Held-out same-phase close-echo success counts, all out of 128:

| Profile | Maximum gate | Ranked gate |
| --- | ---: | ---: |
| white (primary) | 29 | 75 |
| center | 74 | 111 |
| edge | 20 | 62 |

The primary white-noise comparison has 46 paired gains and no losses, increasing
success from 22.7% to 58.6% (+35.9 percentage points). Its exact paired p-value
is 2.84e-14; the descriptive pointwise paired-bootstrap interval is +28.1 to
+44.5 points. The ranked success proportion has pointwise exact 95% interval
49.6–67.2%. Center/edge results are exploratory. All estimates are available in
these cells, but the ranked method still fails the 10 m criterion on 53/128 of
the primary records. Model selection is not a guarantee of accurate ranging.

Direct-only validation reveals the cost:

| Profile | False paths: maximum -> ranked | Range successes: maximum -> ranked | Ranked false-path exact 95% interval |
| --- | ---: | ---: | --- |
| white | 0 -> 3 / 299 | 299 -> 297 / 299 | 0.207–2.904% |
| center | 0 -> 2 / 299 | 299 -> 297 / 299 | 0.081–2.395% |
| edge | 0 -> 1 / 299 | 299 -> 298 / 299 | 0.008–1.849% |

There are no unavailable estimates in any held-out cell. False-path counts
equal raw gate-exceedance counts in this run. In white noise, 3/299 is about
1.003%; a finite observed fraction slightly over 1% neither demonstrates a
population-budget violation nor establishes compliance. All three exact
intervals extend above 1%. The conditional tolerance guarantee is a statement
about the calibration procedure under its iid/model assumptions, not a claim
that this evaluation independently proves the realized gate's rate is below 1%.

![Maximum versus ranked calibration](../output/cpu-ranked-calibration/comparison.png)

This single fresh comparison supports the ranked rule as a less conservative
alternative with improved close-echo sensitivity while retaining the specified
theoretical calibration bound. It also produces observed direct-only regressions
that must not be hidden. Empirical threshold stability across repeated independent
calibrations remains untested for the new rule, and real dynamics/model mismatch
remain outside the guarantee. No production acquisition/PVT path was changed.

## Reproduction and checks

```bash
source /workspace/.gnss-env/activate.sh
pytest -q tests/test_ranked_path_gate_cpu.py
python experiments/run_cpu_ranked_calibration.py
python experiments/audit_cpu_ranked_calibration.py
python experiments/plot_cpu_ranked_calibration.py
```

The destination `output/cpu-ranked-calibration` must be empty. Protocol, all
calibration and evaluation fits/hashes, per-stratum rank metadata, frozen gates
and all outcomes are saved. Ten tests check the exact binomial boundary, maximum
special case, order statistic, input preservation/permutation, invariance to
increasing the largest score, ties and invalid inputs. The audit independently
sorts the calibration scores, recomputes the joint bound, checks all gates,
identities, disjoint groups, selections/denominators and source/prior hashes,
then replays three calibration and two evaluation records with the same model.
