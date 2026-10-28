# Frozen path-selection gate under power and noise-shape changes

This study stress-tests the [conditionally calibrated selector](cpu_calibrated_path_selection.md)
without recalibrating its threshold or changing its estimator. All observations
contain exactly one direct path. Any published two-path solution is therefore
a false echo selection; this is distinct from score exceedance, unavailable
range and range error.

## Controlled distribution changes

Twelve cells combine reference C/N0 values 31/37/43/55 dB-Hz with three spectral
noise profiles. The one-sided cutoff remains 3.683 MHz, integration remains
200 ms with ideal carrier/navigation wipeoff, and the default coarse-delay
assistance remains fixed. PRN 1 direct delay and common phase have the same
distributions as the original calibration. The fixed gate is -12.057079.

The expected per-harmonic noise variance is
`w(f) / (10**(reference_C/N0/10) * 0.200)`. With `x = f/max(abs(f))`, the raw
weights are:

- white: `1`;
- center: `1 + 9*exp(-0.5*(12*x)**2)`;
- edge: `1 + 9*x**4`.

Every profile is normalized to arithmetic mean one. Thus expected total
passband noise energy is equal across profiles at fixed reference C/N0.
Colored profiles have frequency-dependent PSD: the reference C/N0 labels use
the mean passband PSD and must not be read as a frequency-independent C/N0.
Equal total noise does not imply equal delay information or score distributions.
Independent spectral coefficients with nonuniform variance yield colored noise
in time. The receiver intentionally continues to assume equal spectral variance,
so this is a model-mismatch experiment, not a colored-noise-aware receiver.
These smooth profiles are synthetic stress cases, not measured forest, rain,
antenna or RF interference spectra.

There are 128 fresh replica IDs per cell and 1,536 fitted records. A new seed
domain `0x53545253` separates them from calibration and previous evaluations.
The same underlying Gaussian draws, direct delay and common phase are paired
across cells. There are 128 independent replicas per cell, not 1,536 independent
replicas. The protocol and full task list were frozen before evaluation.

## Statistical interpretation

The original 1%/95% calibration guarantee applies only to its single-path
37 dB-Hz white-noise distribution. It is not assumed to transfer here.
For each of the 12 cells, a predeclared one-sided exact binomial test asks whether
the false two-path selection probability exceeds 1%. Bonferroni correction
controls the 12-test family at 0.05 even though cells share replicas.
Failure to reject does not establish a rate <=1%. Exact binomial 95% intervals
are pointwise rather than simultaneous. All attempts, including unavailable
estimates, stay in the denominator; raw score-exceedance counts are also saved.

## Results

The frozen white-noise gate **does not retain its conditional false-selection
control for the center-heavy noise profile**. All four center-profile cells
reject a false-selection rate <=1% after the predeclared 12-test Bonferroni
correction. Adjusted p-values are 2.58e-13 at 31/43/55 dB-Hz and 3.89e-12 at
37 dB-Hz. These cells share random replicas, so this is not four independent
replications; the family correction remains valid under that dependence.

| Reference C/N0 | Profile | Gate exceedances | False two-path selections | Unavailable | Range successes: original BIC -> calibrated |
| --- | --- | ---: | ---: | ---: | ---: |
| 31 | white | 0/128 | 0/128 | 0/128 | 128 -> 128 |
| 31 | center | 27/128 | 17/128 | 10/128 | 125 -> 114 |
| 31 | edge | 1/128 | 0/128 | 1/128 | 127 -> 126 |
| 37 | white | 0/128 | 0/128 | 0/128 | 128 -> 128 |
| 37 | center | 26/128 | 16/128 | 10/128 | 126 -> 115 |
| 37 | edge | 1/128 | 1/128 | 0/128 | 128 -> 127 |
| 43 | white | 0/128 | 0/128 | 0/128 | 128 -> 128 |
| 43 | center | 27/128 | 17/128 | 10/128 | 126 -> 115 |
| 43 | edge | 0/128 | 0/128 | 0/128 | 128 -> 128 |
| 55 | white | 0/128 | 0/128 | 0/128 | 128 -> 128 |
| 55 | center | 27/128 | 17/128 | 10/128 | 126 -> 115 |
| 55 | edge | 0/128 | 0/128 | 0/128 | 128 -> 128 |

Range success means available absolute error <10 m, out of all 128 attempts.
At 37 dB-Hz, the center-profile false-selection rate is 12.5%, with pointwise
exact 95% interval 7.3–19.5%. The 17/128 cells give 13.3%, interval 7.9–20.4%.
The original BIC rule selects false echoes in 3/128 center-profile records at
each strength; it is less affected than the calibrated gate, but is not proven
to control the rate at 1% either. The calibrated selector also withholds ten
additional ranges per center-profile cell when its score passes but local
guards fail. Neither those failures nor the resulting range-success losses
are omitted from the table.

White-profile cells produce no false two-path selections in these samples.
Each 0/128 cell still has a pointwise 95% upper bound of 2.84%; non-rejection
cannot establish that the 1% budget holds at other signal strengths. Edge-heavy
noise produces one false selection at 37 dB-Hz and a withheld range at 31 dB-Hz.

![Frozen-gate false echo selections](../output/cpu-path-selection-stress/comparison.png)

Equal total noise power was insufficient to preserve the selection statistic:
the noise spectrum matters. This is a distributional limitation of the gate and
the equal-variance likelihood assumption, not a contradiction of the original
white-noise calibration statement. The next implementation target is estimation
of the noise spectrum and covariance-aware fitting/whitening, followed by fresh
full-pipeline single-path calibration. Using the simulator's true noise profile
as receiver input would be an oracle benchmark, not a validated practical fix.

## Reproduction and checks

```bash
source /workspace/.gnss-env/activate.sh
pytest -q tests/test_spectral_noise_cpu.py
python experiments/stress_cpu_path_selection.py
python experiments/audit_cpu_path_selection_stress.py
```

The destination `output/cpu-path-selection-stress` must be empty. Protocol,
all fitted records and observation hashes, summaries and PNG/PDF plots are
saved. Eight tests verify positive normalized noise profiles, symmetry to
floating-point precision, simulated Gaussian energy, profile direction and
invalid inputs. The audit checks the frozen gate and prior-artifact/source
hashes, all identities and denominators, all corrected tests and exact replay
of four records spanning all three profiles. No earlier gate or output changes.
