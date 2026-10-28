# Range scatter and weighted position uncertainty

The [median-reference recovery experiment](position_recovery_simulation.md)
did not improve overall publication availability. This step therefore tests
measurement uncertainty before changing another position-publication rule.
It adds an opt-in weighted least-squares estimator and a causal range-scatter
model. Existing position solvers and publication defaults remain available.
Neither estimator in this experiment applies a position-publication gate.

## Receiver-only range-scatter model

`range_uncertainty_cpu.CausalRangeUncertainty` tracks fresh measured ranges
separately for each PRN. It uses only PRN, measured pseudorange and timestamp;
it does not convert acquisition scores into C/N0 or ranging confidence.
The same prior-only sigma is returned regardless of the current observation's
value, and the current difference enters history only for the next update.
Thus a current outlier cannot excuse itself by inflating its own uncertainty.

The [fixed policy](position_uncertainty_policy.json) uses the previous 32 range
differences and requires eight differences before leaving a 10 m cold prior.
For differences d, the Gaussian-consistent robust scale is
`s_delta = 1.4826022185 * median(abs(d - median(d)))`.
The output sigma is `s_delta / sqrt(2 * gain)`, clipped to [2,100] m, with
configured receiver smoothing gain 0.25. These limits and windows are
engineering settings, not a learned fit or a calibrated confidence bound.
Gaps above 5 ms restart the prior; histories can be explicitly reset.

The smoothing correction follows directly from the stationary scalar model
`x_t = (1-a)*x_(t-1) + a*z_t`, with independent white input noise z:

```text
Cov(x_t, x_(t-1)) = (1-a)*Var(x_t)
Var(x_t - x_(t-1)) = 2*a*Var(x_t)
sigma_x = sigma_difference / sqrt(2*a)
```

This relation assumes equally spaced successful updates with stationary
linear smoothing and white input errors. Carrier-aided prediction, bounded
updates, short gaps, acquisition/reset, time-varying signal power and nonlinear
code fitting violate parts of that model. With 32 correlated differences, a
MAD estimate is itself noisy. Its median subtraction also removes a constant
range slope. A persistent bias can be arbitrarily large with small estimated
scatter: an explicit test adds 200 m without changing any sigma.

## Weighted fit and local covariance

`position_uncertainty_cpu.solve_weighted_pseudorange_position` minimizes

```text
sum_i ((rho_i - norm(sat_i - position) - clock_m) / sigma_i)^2
```

It solves position and receiver clock jointly using weighted linearized least
squares. With final design H and diagonal R = diag(sigma_i^2), the reported
four-state covariance is `(H.T @ inv(R) @ H)^-1`, evaluated with an SVD.
The covariance is not rescaled by postfit residuals. Its upper-left 3×3 block
is the marginal position covariance, including uncertainty coupled to clock;
inverting only the position design would incorrectly treat clock as known.
Four nondegenerate observations can yield a covariance but have zero residual
degrees of freedom. Rank failure/nonconvergence returns no solution.

```python
from gnss_gpu.position_uncertainty_cpu import solve_weighted_pseudorange_position

result = solve_weighted_pseudorange_position(satellites_m, ranges_m, sigma_m)
if result is not None:
    print(result.solution.position_m, result.solution.clock_bias_m)
    print(result.covariance_m2[:3, :3])
```

Independent heterogeneous Gaussian-noise tests check that weighting reduces
position error and that empirical normalized position-error energy is near its
three-dimensional expectation. Further tests cover equal-weight equivalence,
sigma/covariance scaling, smoothing correction, reset/gaps, invalid input and
truth/visibility/weather/acquisition-score independence. These controlled
tests do not establish validity for real multipath or rainy-weather signals.

## Paired signal-to-position experiment

All previously inspected seeds **24683–24688** are development data: 60 episodes
and 30,000 input epochs. Policy bytes and source hashes are saved before any
replay or fresh generation. New seeds **24689/24690** generate 16 episodes and
8,000 epochs: open/wall geometry crossed with dry, 6 dB loss, 30 dB loss stress
and combined antenna stress. The same channel receiver, geometry, acquisition
thresholds and prescribed wetting/drying schedule apply. No setting is changed
after viewing these fresh results.

Both estimators receive identical fresh pseudoranges. The equal-weight
comparator uses the RMS of current per-range sigmas as its common sigma:
uniform scaling preserves the original raw point estimate while providing a
covariance in the same declared noise units. Weighted estimation instead uses
each range's own sigma. Equal-weight availability and positions are checked
against the saved raw receiver within one micrometre. This comparison does
not reuse the previous publication gate or its six-satellite minimum.

After receiver-only estimation, synthetic truth scores errors and range-sigma
coverage. The true clock is zero in this generator. Position-error energy is
`e.T @ inv(C_position) @ e`; a nominal 95% Gaussian ellipsoid contains truth
when this is <=7.8147279033 (chi-square with three degrees of freedom).
The plotted coverage is the observed fraction among available estimates,
not an independent-trial confidence interval. Missing epochs remain missing;
all-input unavailable counts and >100 m errors accompany conditional RMSE/P95.
Cold priors are included and readiness is retained per range and position.

```bash
PYTHONPATH=python:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python experiments/run_cpu_position_uncertainty.py --workers 4
PYTHONPATH=python:. python -m pytest tests/test_position_uncertainty_cpu.py tests/test_range_uncertainty_cpu.py tests/test_cpu_position_uncertainty.py -q
```

Prerequisites are the wet-antenna, position-quality and reference-recovery
outputs described in the linked guides. `output/cpu-position-uncertainty/`
saves both estimators at every epoch, range sigmas/errors/readiness, episode
and stage summaries, fresh channel observations/raw positions, frozen policy,
source/input hashes and `position_uncertainty.png` (RMSE and ellipsoid coverage).

## Frozen-model results

The completed replay retains 76,000 two-estimator position records, 288,252
fresh range-sigma/error records, 152 episode summaries and 760 stage summaries.
It also saves 96,000 new channel records and 8,000 new raw receiver positions.
The maximum equal-weight replay difference is **3.73e-9 m**, below the 1e-6 m
check. All saved source/input/policy hashes match; no model setting changed
after inspecting the fresh results.

| Fresh-noise dataset | Estimator | Input epochs | Available | 3D RMSE [m] | 3D P95 [m] | Errors >100 m | Nominal 95% position coverage |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| All cases | Equal | 8,000 | 6,120 | 13.73 | 24.63 | 12 | 93.22% |
| All cases | Weighted | 8,000 | 6,120 | 12.63 | 21.61 | 12 | 93.22% |
| Open | Equal | 4,000 | 3,058 | 7.73 | 9.98 | 4 | 95.81% |
| Open | Weighted | 4,000 | 3,058 | 7.62 | 9.93 | 4 | 94.15% |
| Wall | Equal | 4,000 | 3,062 | 17.80 | 30.97 | 8 | 90.63% |
| Wall | Weighted | 4,000 | 3,062 | 16.15 | 27.30 | 8 | 92.29% |

The 3D RMSE decreases about **8.0%** overall and **9.2%** near the wall on
this fixed test set, with no change in availability (1,880 missing epochs).
These are conditional-error metrics; no missing fix is treated as zero error.
This does not reproduce the previous gate's >100 m rejection: **all 12 large
errors remain**, and the largest error remains 173.45 m. Weighting is not an
outlier rejection policy or a substitute for one.

| Fresh case, both new seeds | Equal RMSE [m] | Weighted RMSE [m] | Weighted nominal 95% coverage |
| --- | ---: | ---: | ---: |
| Open, dry | 3.54 | 3.54 | 96.19% |
| Open, 6 dB loss | 5.41 | 5.48 | 91.48% |
| Open, 30 dB loss stress | 11.52 | 11.27 | 94.73% |
| Open, combined stress | 11.54 | 11.28 | 94.73% |
| Wall, dry | 12.73 | 10.77 | 92.59% |
| Wall, 6 dB loss | 16.80 | 13.84 | 90.88% |
| Wall, 30 dB loss stress | 22.25 | 21.70 | 93.43% |
| Wall, combined stress | 22.22 | 21.26 | 93.25% |

The improvement is not universal: open/6 dB loss has slightly larger RMSE and
only 91.48% coverage with weighting. The pooled equal comparator assumes a
common range variance, whereas the weighted covariance assumes heterogeneous
independent variances; their coverage difference is not an isolated test of
matrix weighting with otherwise identical covariance assumptions.

Even when every used range has enough prior differences, weighted fresh
coverage is 93.16% overall (94.28% open, 92.04% wall). Cold priors therefore
do not explain the remaining undercoverage. A mean normalized position-error
energy of 3.29, near its ideal expectation of three, does not establish a
correct 95% tail. There are no independent-epoch confidence claims here.

On the larger, previously inspected development set, RMSE is 13.34 to 12.59 m,
>100 m errors are 36 to 32, and weighted nominal coverage is 92.26% versus
93.23% for equal weights. Development gains in large-error count do not recur
in the fresh holdout; both outcomes are retained.

This provides an opt-in weighted-estimation baseline, **not a calibrated
uncertainty-based publication rule**. Future work should address persistent
bias, transient/reacquisition error and uncertainty in short histories before
using covariance to relax continuity checks. Seeds 24689/24690 have now been
inspected and cannot validate a subsequent revised model as unseen data.

## Interpretation boundary

Linearized covariance assumes the supplied variances are appropriate and
range errors are locally unbiased and independent across satellites. It omits
shared errors, persistent multipath/antenna bias, wrong code-period peaks and
uncertainty in estimated sigma itself. Positive covariance and low postfit
residual do not establish an accurate fix. These ellipsoids are model diagnostics,
not protection levels or a safe publication rule. The two fresh noise seeds,
paired scenarios, serial tracking states, one static geometry per environment
and uncalibrated stress coefficients do not validate real rain or dynamic motion.
