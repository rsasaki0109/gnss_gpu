# Wet-troposphere correction residuals: IQ to height and clock

`wet_delay_cpu.WetDelayResidualLayer` independently injects a controlled
**actual-minus-applied zenith wet-delay residual**. It extends the
[rain/water-film experiment](rain_simulation.md), while retaining separate
controls for rain attenuation, wet-surface reflection and neutral atmospheric
delay. Rain rate is not used to infer this residual.

## Model and receiver conventions

For a flat slab of height H and signed full-column zenith residual dZWD:

```text
residual refractivity = dZWD/H
excess range [m]      = (dZWD/H) * length_inside_slab [m]
excess range rate     = d(excess range)/dt
```

A complete straight-ray crossing maps to `dZWD/sin(elevation)`. A receiver
inside the slab traverses only the remaining portion; the experiment's
receiver is at z=1.5 m and the slab covers z=0..3000 m. Reflected paths sum the
two finite legs, including the low-altitude receiver-to-wall portion. Positive
residual means under-correction, negative means over-correction; negative
residual is not negative physical water vapour. The nominal atmosphere is
treated as already removed, so the IQ injection represents its remaining
correction error. The selected +/-0.1 and +/-0.3 m levels are illustrative
sensitivity controls, not observed rain-linked distributions.

Both code and carrier use `geometric_range + wet_excess_range` with the same
positive path-delay convention. Analytic derivatives of slab clipping and the
moving specular point contribute to code rate and Doppler. This avoids adding
a delay to code alone or omitting its time derivative from carrier frequency.
Inverse-range amplitude and path geometry remain geometric. This is a
nondispersive weak-refraction approximation with straight rays, not bending or
a refracted-ray solution. Sharp slab boundaries have derivative cusps; the
centered-aperture integration remains locally affine at each sample center.

```python
from gnss_gpu.wet_delay_cpu import WetDelayResidualLayer

wet = WetDelayResidualLayer(zenith_residual_m=.3, bottom_m=0., top_m=3000.)
# Existing simulator call, with an independent optional parameter:
block = simulate_wall_signal(..., wet_delay_layer=wet)
```

`WallSignalBlock.direct_range_m`/`reflected_range_m` and range rates remain
**geometric**. New arrays expose `direct_wet_delay_m`, `reflected_wet_delay_m`,
`direct_wet_delay_rate_mps`, and `reflected_wet_delay_rate_mps`. Propagation-range
properties add the excess to the corresponding geometric range. Doppler
properties include the wet-delay derivative; they still exclude intermediate
frequency. Invalid reflected paths have zero wet-delay diagnostics and do not
contribute IQ. None/zero residual preserves the default waveform exactly.

This is a derived flat-layer sensitivity model. The previously reviewed
Sanwale et al. [preprint](https://arxiv.org/abs/2605.01553v1) Eq. (7) supplies
the conceptual separation of geometric and atmospheric delay, but the slab,
clipping derivatives and experimental residual levels are our implementation.
No Niell/GMF or Saastamoinen mapping coefficients are claimed here. Humidity,
temperature, pressure, curved-Earth ray paths, horizontal gradients and time
evolution are not modeled. [Provenance and assumptions](rain_sources.json)
record the distinction from the separately checked P.838 attenuation model.

## Controlled comparison

```bash
PYTHONPATH=python:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python experiments/run_cpu_wet_delay_positioning.py --workers 4
PYTHONPATH=python:. python -m pytest tests/test_wet_delay_cpu.py tests/test_cpu_wet_delay_positioning.py -q
```

The experiment retains the rain study's 12 static synthetic source directions,
16 receiver poses, eight noise realizations per pose, known PRN/zero carrier,
55 dB-Hz nominal direct C/N0 and 2.046 MHz centered-aperture IQ. Each snapshot
requires two confirmed 1 ms observations. Per-PRN noise gates are frozen before
evaluation and use the same 199-block calibration procedure. A fresh noise seed
24682 is used; threshold/confirmation settings are not retuned. Position and
clock are solved with the existing equal-weight nonlinear least-squares model
and the same coarse initial position, without truth-based rejection or weights.
The known 22 Mm reference still selects the integer code period.

Open-space scenarios include zero residual, +/-0.1 and +/-0.3 m, rain
100 mm/h alone, and rain plus +0.3 m residual. Wall scenarios additionally
compare rain with a 1 mm water film, with and without +0.3 m residual. Film
thickness, rain rate and delay residual are independent controls. All
unavailable solutions, channel failures, large errors and paired counts are
retained; conditional quantiles do not replace availability.

Common random numbers are paired **in the direct carrier frame**: across
scenarios, a unit-modulus rotation of generated circular Gaussian noise
matches the modeled direct atmospheric phase change. Its marginal white-noise
distribution and power are unchanged. This is an artificial Monte Carlo
variance-reduction coupling, not an assertion that physical noise follows the
atmosphere. It avoids phase-driven noise resampling obscuring a small code-delay
perturbation. Known simulated delay is used only in noise generation, never in
the receiver. Thus paired-shift distributions depend on this chosen coupling;
they are not real-weather uncertainty intervals. Plotted P05–P95 bands are
distribution quantiles, **not confidence intervals**.

An explicitly separate **noiseless ALL-LOS range-domain control** forms
geometric range plus the prescribed wet excess and solves position/clock. It
shows geometry coupling without IQ/noise/multipath. It is a diagnostic and must
not be reported as measured receiver performance.

## Initial results

For pose zero in the noiseless geometric control:

| Zenith residual [m] | Height shift [m] | Receiver clock bias, distance [m] |
| ---: | ---: | ---: |
| -0.30 | -0.7751 | -1.0221 |
| -0.10 | -0.2584 | -0.3407 |
| 0 | approximately 0 | approximately 0 |
| +0.10 | +0.2584 | +0.3407 |
| +0.30 | +0.7751 | +1.0221 |

The residual is elevation dependent, so clock estimation cannot absorb it as
a single common range offset. The resulting height amplification depends on
this constellation; it is not a universal factor for wet delay.

In noisy IQ comparisons, every scenario has 128/128 solved positions. For
wet residual alone, median paired height shifts are:

| Zenith residual [m] | Open-space height shift P50 [m] | Wall height shift P50 [m] |
| ---: | ---: | ---: |
| -0.30 | -0.7810 | -0.7876 |
| -0.10 | -0.2603 | -0.2625 |
| +0.10 | +0.2603 | +0.2625 |
| +0.30 | +0.7809 | +0.7875 |

These paired state changes are effect sizes, not increases in absolute error.
Absolute height-error P95 is 16.93 m (open reference) versus 17.37 m (+0.3 m
residual), and 56.31 m (wall reference) versus 56.97 m (+0.3 m residual).
With rain plus film at the wall, P95 is 67.25 m without the additional residual
and 67.99 m with it. Horizontal errors can move either way under a signed
residual. Wet reflection changes and atmospheric delay are not assumed to add
linearly in the nonlinear receiver/multipath solution.

Successful solving and confirmation are not integrity guarantees. This fresh
seed has one >100 m 3D error out of 128 in each wall condition without film and
two in film conditions; none are dropped. No channel with neither a geometric
direct nor reflected path is used in this particular confirmed run. Large
remaining errors can therefore arise with available paths and poor geometry/
noise/multipath, not only blocked-source false capture. One seed and sixteen
prescribed poses do not establish a false-fix or rainy-weather error bound.

`output/cpu-wet-delay-positioning/` contains 2,048 position records, 24,576 final
channel decisions, 80 separate geometric controls, 2,388 calibration scores,
16 summaries, complete seeds/model settings/source hashes and
`wet_delay_positioning.png`. Clock biases and paired clock shifts are retained
in metres. All conditional metrics report their counts.

The remaining weather work includes measured/estimated water-vapour profiles
and correction error statistics, realistic mapping/ray bending, evolving
wet-delay gradients, wet antenna/radome effects and validation on real rainy
days with independent position truth. This stage quantifies a controlled wet-
delay residual mechanism; it does not infer that residual from rainfall rate.
