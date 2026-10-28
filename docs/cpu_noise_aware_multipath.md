# Estimated noise weighting and full-pipeline recalibration

The [frozen-gate stress study](cpu_path_selection_stress.md) exposed frequent
false echoes when noise power concentrated near baseband center. This change
estimates spectral noise variance from observations and weights both the data
and code templates before fitting. It then calibrates a new gate for the entire
noise-estimation/fitting procedure. Previous estimators and gates are preserved.

## Receiver input and estimator

The receiver gets ten aligned 20 ms spectral blocks, with known carrier and
navigation polarity removed. It receives **no true noise profile, C/N0, delay
or reflection parameters**. At each harmonic, it subtracts the across-block
mean and estimates sample variance from the residuals. It smooths this variance
over 101 adjacent 1 kHz harmonics, correcting for the number of bins near passband
edges, and floors at 1e-6 of the smoothed mean. Relative weights have mean one.
Both the averaged observation and every fitted signal template are divided by
the square root of these weights. The existing numerical-tie, boundary and
local-information guards remain in force.

The constant signal cancels from block deviations. Under the stationary proper
Gaussian model, the block mean is independent of the sample deviations; the
same-record variance estimate therefore does not use true range or exploit a
fitted signal residual to decide the noise shape. Finite variance estimation and
smoothing still affect score distributions, so the old gate is not reused.
The smoothing bandwidth and floor were fixed before calibration/evaluation.

This estimator relies on a static, aligned signal over 200 ms. Motion, changing
echo amplitude/phase, residual Doppler or unremoved navigation bits can appear
in the deviations and be mistaken for noise. It is not yet a practical receiver
front end for arbitrary recorded RF. The white/center/edge profiles are synthetic
smooth spectra, not measured rain or forest noise.

## Calibration and validation protocol

The signal context stays fixed at PRN 1, 37 reference dB-Hz, 3.683 MHz one-sided
cutoff, 200 ms ideal coherence and the default coarse-delay assistance. Each of
the three single-path noise strata gets 408 fresh calibration records. Its gate
is the maximum raw fit score. The combined threshold is the largest of all three
stratum maxima. Strict exceedance and `3 * 0.99**408 <= 0.05` establish a marginal
1% exceedance budget in each specified stratum jointly with at least 95%
calibration confidence, assuming iid records within each stratum. The guarantee
does not extend to arbitrary spectra, powers, dynamics or assistance errors.

The 1,224-record calibration completes and its gate is saved before held-out
evaluation. Evaluation has 128 new single-path records per noise profile and
64 new records per close-echo condition: voltage ratio 0.8, excess delay 0.1 chip,
phases zero and pi/2, across three profiles (768 held-out observations total).
Calibration, null-evaluation and echo seed groups are disjoint. Evaluation IDs
are paired across profiles/echo phases, not independent across all 768 records.

The baseline is the old unweighted estimator with its original conditional gate
on the exact same averaged blocks. The comparison changes **weighting and
calibration together**; it does not isolate whitening from recalibration alone.
Two primary paired success comparisons are fixed: same-phase echoes in white
and center-profile noise, using exact two-sided McNemar tests with a two-test
Bonferroni correction. Other comparisons are exploratory. A success is an
available absolute range error <10 m; unavailable estimates remain failures.
Pointwise exact/paired-bootstrap intervals are not simultaneous.

## Results: improved colored-noise behavior, reduced white-noise sensitivity

The combined gate is **-7.358247**, stricter than the old -12.057079 gate on
its own unweighted score. Scores come from different weighted likelihoods;
the numeric thresholds alone cannot isolate the source of a performance change.

Held-out direct-only controls, all out of 128 attempts:

| Profile | False two-path selections, old -> new | Range successes, old -> new | Unavailable, old -> new |
| --- | ---: | ---: | ---: |
| white | 1 -> 0 | 128 -> 128 | 0 -> 0 |
| center | 15 -> 0 | 116 -> 128 | 7 -> 0 |
| edge | 0 -> 0 | 128 -> 128 | 0 -> 0 |

Every new direct-only cell has zero observed false two-path selections, but its
pointwise exact 95% upper bound remains **2.84%**. These samples do not independently
establish a rate below 1%; the separate conditional 1%/95% statement comes from
the stratified calibration construction and its assumptions.

Close-echo range successes (available error <10 m), all out of 64 attempts:

| Noise profile | Phase 0, old -> new | Phase pi/2, old -> new |
| --- | ---: | ---: |
| white | 45 -> 37 | 64 -> 64 |
| center | 39 -> 49 | 64 -> 64 |
| edge | 32 -> 27 | 61 -> 63 |

The two predeclared primary comparisons show **opposite directions**:

- Center-profile, same-phase echoes improve by 15.6 percentage points
  (11 paired gains, one loss), adjusted p = 0.01270. The descriptive pointwise
  paired bootstrap interval is +6.25 to +25.0 points.
- White-profile, same-phase echoes regress by 12.5 percentage points
  (one gain, nine losses), adjusted p = 0.04297. The descriptive pointwise
  paired bootstrap interval is -21.9 to -3.13 points.

These are paired comparisons on the same fresh blocks. The white-profile result
must not be hidden by pooling it with the colored-noise improvement. The new
method is not uniformly better. The exploratory edge same-phase cell also loses
five successes. One new edge/pi/2 estimate is unavailable; it remains a failure
in the denominator. All other new echo cells have 64 available estimates.

![Noise-aware held-out results](../output/cpu-noise-aware-multipath/comparison.png)

Estimated noise weighting plus recalibration resolves the observed center-noise
false-echo problem on this finite grid, while costing sensitivity in other
conditions. The next useful experiment is a frozen ablation separating estimated
weighting from the stricter multi-profile calibration, using fresh calibration
and evaluation splits. A profile-adaptive gate cannot simply be chosen after
examining signal success: the profile decision and resulting complete receiver
would require calibration together. No production/acquisition/PVT entry point
has been switched to this experimental estimator.

## Reproduction and checks

```bash
source /workspace/.gnss-env/activate.sh
pytest -q tests/test_noise_aware_multipath_cpu.py
python experiments/run_cpu_noise_aware_multipath.py
python experiments/audit_cpu_noise_aware_multipath.py
python experiments/plot_cpu_noise_aware_multipath.py
```

The destination `output/cpu-noise-aware-multipath` must be empty. All calibration
and evaluation records, block hashes, raw/selected fits and failures are saved,
alongside the pre-run protocol, frozen per-stratum/combined gates and summary.
Eight tests check estimation of all three spectra, invariance to an unknown
constant signal/complex scaling, exact equivalence to v2 under unit weights and
invalid inputs. The audit recomputes all gate maxima, verifies the joint bound,
all identities, disjoint groups, choices/denominators and source/prior hashes,
then exactly replays three calibration and two evaluation records. This is
limited replay with the same implementation, not independent RF validation.
