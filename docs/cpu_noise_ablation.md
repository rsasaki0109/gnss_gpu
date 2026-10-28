# Separating noise weighting from calibration coverage

The [noise-aware experiment](cpu_noise_aware_multipath.md) improved center-noise
behavior but reduced white-noise close-echo sensitivity. This fresh paired 2x2
experiment separates the two changes while keeping both delay estimators fixed.

| Arm | Delay fitting | Calibration coverage |
| --- | --- | --- |
| unweighted_white | Frozen unweighted v2 | White noise only |
| unweighted_multi | Frozen unweighted v2 | White, center and edge noise |
| weighted_white | Estimated spectral noise weighting | White noise only |
| weighted_multi | Estimated spectral noise weighting | White, center and edge noise |

The weighted fit estimates noise from ten aligned blocks, as before. No true
noise profile or true range is supplied to either estimator. Calibration labels
identify which synthetic nuisance distribution a null record belongs to; they
are not used to select an arm or gate during inference. Both raw fits are computed
on the same observation, and all four selection outcomes are retained.

## Frozen design

Signal conditions stay fixed at PRN 1, reference C/N0 37 dB-Hz, 200 ms ideal
carrier/nav coherence, one-sided cutoff 3.683 MHz and default delay assistance.
All data are newly generated with replica IDs disjoint from earlier experiments:
calibration 100000–100476/101000–101476/102000–102476; null evaluation 110000–110127;
echo evaluation 120000–120063. The existing seed domain is unchanged, but these
IDs have not been used before. Evaluation IDs are shared across profiles/phases,
so cell outcomes are paired rather than independent across the whole grid.

Each of three noise profiles has 477 calibration observations fitted by both
estimators: 1,431 observations, six score strata. For each estimator/profile pair,
the gate is the maximum score. White-only arms use the respective white maximum;
multi-profile arms use the maximum of that estimator's three maxima. Strict
exceedance and `6 * 0.99**477 <= 0.05` give joint calibration confidence of at least
95% that each specified stratum's exceedance rate at its own gate is <=1%.
White-only arms do not inherit control in the colored-noise strata. Multi-profile
arms cover the three specified strata, not arbitrary spectra or dynamics.

All four gates are frozen before any evaluation. Evaluation uses 128 null
observations per profile and 64 echo observations per profile/phase (0 and pi/2),
with echo voltage ratio 0.8 and excess delay 0.1 chip: 768 observations and 3,072
selection outcomes. Success requires an available absolute direct-range error
<10 m; all attempts remain in the denominator.

The two primary paired contrasts are fixed on **white-noise, same-phase echoes**:

- `weighted_white - unweighted_white`: change fitting while holding calibration
  coverage and calibration observations fixed;
- `weighted_multi - weighted_white`: change calibration coverage while holding
  the weighted fit fixed.

Exact two-sided McNemar tests use a two-test Bonferroni correction. Remaining
contrasts/cells are exploratory. Pointwise binomial/bootstrap intervals are not
simultaneous. The weighting contrast includes the estimator-specific white
recalibration; using an identical numeric gate across different score models
would not be a matched false-selection comparison. This study samples one fresh
calibration realization, not the distribution of outcomes over many calibrations.

## Results

Fresh frozen thresholds on each estimator's BIC-improvement score:

| Arm | Threshold |
| --- | ---: |
| unweighted_white | -11.447675 |
| unweighted_multi | 27.836653 |
| weighted_white | -11.420132 |
| weighted_multi | -9.709779 |

At the same fit, expanding calibration coverage only raises the threshold here.
Across different fits, equal numeric thresholds would not imply equal null
behavior. The especially large unweighted multi-profile threshold is driven by
its colored-noise score distribution and suppresses close-echo selection.

Primary contrasts on white-noise, same-phase 0.1-chip echoes:

| Contrast | Successes before -> after | Paired gains / losses | Bonferroni-adjusted p |
| --- | ---: | ---: | ---: |
| Weighting, white-only calibration | 41/64 -> 41/64 | 0 / 0 | 1.0 |
| Coverage expansion, weighted fitting | 41/64 -> 37/64 | 0 / 4 | 0.25 |

The observed four lost successes occur on the calibration-coverage step, not
the weighting step, in this dataset. **The four-record difference is not
statistically established** at the predeclared family level. Equal success
outcomes in the first contrast also do not prove equivalent population behavior.
The paired percentile bootstrap degenerates for all-equal differences and can
exclude zero for a few one-directional losses even when the discrete exact test
does not reject; it is descriptive and does not override the primary exact test.
This result suggests an explanation of the preceding regression but does not
prove that the same attribution holds across calibration draws.

All same-phase 0.1-chip echo range successes, out of 64 attempts:

| Noise profile | Unweighted / white | Unweighted / multi | Weighted / white | Weighted / multi |
| --- | ---: | ---: | ---: | ---: |
| white | 41 | 1 | 41 | 37 |
| center | 36 | 1 | 56 | 56 |
| edge | 35 | 1 | 30 | 25 |

For pi/2 echoes, all four arms have 64/64 successes in white and center noise,
and 63/64 in edge noise. Every echo estimate is available in this run. The
weighted multi-profile arm therefore retains much more same-phase sensitivity
than unweighted multi-profile calibration, but is not uniformly better than
the less broadly calibrated arms. Edge-profile same-phase losses remain visible.
These additional contrasts are exploratory.

False two-path selections in direct-only controls, out of 128 attempts:

| Noise profile | Unweighted / white | Unweighted / multi | Weighted / white | Weighted / multi |
| --- | ---: | ---: | ---: | ---: |
| white | 0 | 0 | 0 | 0 |
| center | 17 | 0 | 1 | 0 |
| edge | 1 | 0 | 0 | 0 |

The unweighted white-only arm has 114/128 center-noise range successes and eight
unavailable estimates; its edge-noise success count is 127/128. Weighted
white-only calibration has one unavailable estimate in each white/edge null
cell (127/128 successes), and 128/128 center successes despite one false path
selection. Both multi-profile arms have 128/128 range successes and no unavailable
null estimates in every profile. A 0/128 observed false-selection count still
has a pointwise exact 95% upper bound of 2.84%, not a verified zero or sub-1% rate.

![Four-arm ablation](../output/cpu-noise-ablation/comparison.png)

The practical tradeoff is clearer: applying broad null calibration without
noise weighting removes most of the desired same-phase echo sensitivity, while
estimated weighting makes broad calibration considerably less costly. The
white-noise regression seen previously is not enough to conclude that weighting
itself is harmful. Replication over independent calibration sets is needed to
measure how much the extreme-order-statistic threshold varies and whether the
small coverage-related loss persists. No arm is selected for production/PVT
deployment from this finite idealized experiment.

## Reproduction and audit

```bash
source /workspace/.gnss-env/activate.sh
pytest -q tests/test_noise_ablation.py
python experiments/run_cpu_noise_ablation.py
python experiments/audit_cpu_noise_ablation.py
python experiments/plot_cpu_noise_ablation.py
```

The output directory `output/cpu-noise-ablation` must be empty. The run saves its
protocol before calibration, every raw fit and block hash, per-stratum and arm
gates before evaluation, all outcomes and contrasts, and completion hashes.
Three tests check the correct 2x2 gate mapping, unchanged paired inputs and the
minimum calibration size for the six-stratum confidence budget. The audit checks
all thresholds, identities, disjoint groups, arm mappings, reports/denominators
and source/prior hashes, then exactly replays three calibration and two evaluation
records using the same implementation. Previous research artifacts are preserved.
