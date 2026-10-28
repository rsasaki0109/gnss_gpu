# Rain attenuation, wet-wall sensitivity and CPU positioning

The CPU simulator can now compare rain-path attenuation and an independent
wet-wall film model, then propagate acquired code delays into a 3D position
and receiver-clock solution. These are controlled synthetic sensitivity tests,
not a calibrated prediction of real rainy-weather GNSS performance.

## Two separate physical controls

**Rain in the propagation path.** `rain_cpu.circular_rain_coefficients` uses the
ITU-R P.838-3 frequency fits with circular polarization. The coefficients were
checked against [ITU-Rpy 0.4.0](https://github.com/inigodelportillo/ITU-Rpy/tree/094abb618c2a7a8cc145e93940f4571650bc2cf5),
commit `094abb618c2a7a8cc145e93940f4571650bc2cf5`. The primary ITU PDF has not
been downloaded or claimed reviewed. [Source provenance](rain_sources.json)
records reviewed scope, URLs and hashes; the [MIT notice](itu_rpy_notice.txt)
preserves attribution for the numerical coefficient tables.

```text
gamma [dB/km] = k(f) * R[mm/h]^alpha(f)
A [dB]       = gamma * length_inside_rain_layer[km]
voltage gain = 10^(-A/20)
```

For GPS L1, k=0.000056009072377142174 and alpha=0.9567177715408957. At
100 mm/h, gamma is about **0.004589 dB/km**. A vertical 3 km rain column gives
0.01377 dB; a shallow slant path gives more loss. `RainLayer` clips each finite
segment to a prescribed horizontal slab, rather than applying rain attenuation
to the entire satellite-to-receiver distance. The direct segment and both
reflected legs are integrated separately. The circular-polarization scalar loss
is also an approximation for reflected, potentially elliptical fields: rain
depolarization is not modeled. Noise is added **after** signal attenuation.
No random phase jitter or metre-scale delay is invented from rainfall rate.

**Water on the wall.** `VerticalWall.water_film_thickness_m` adds an ideal smooth
air/water/substrate layer. Separate s/p Fresnel coefficients are composed as

```text
r_front = (r01 + r12*exp(-2j*k0*q_water*d))
          / (1 + r01*r12*exp(-2j*k0*q_water*d))
q_water = sqrt(eps_water - sin(theta)^2)
```

The phasor convention is exp(+j*omega*t) with passive eps.imag<=0. In IQ,
geometry still references the original substrate plane. A factor
exp(+2j*k0*cos(theta)*d) accounts for the air segment replaced by film, before
the existing circular receive-basis combination. This makes a film of air
replacing air leave the waveform unchanged. Reflected-point geometry is held
fixed in the thin-film approximation; boundary displacement/diffraction is not
resolved. Film reflection is evaluated at the carrier frequency and held
constant over the signal band; dielectric film group delay is not modeled.

The default water relative permittivity **80-8j is an explicit illustrative
assumption**, not measured wet concrete, a temperature-dependent water model or
a fitted field experiment. Film thickness 0.1/1 mm is an independent control.
There is **no rainfall-rate-to-film-depth law**: wet-after-rain and rain-with-dry-
surface are intentionally separate ablations. Reflection magnitude and phase
both change, so wetting need not increase every individual ranging error.
There is no infiltration, partial wetness, droplets, roughness or wet radome.

Both controls default to no change: `rain_layer=None` and zero film depth keep
the existing propagation path. Per-sample `direct_rain_loss_db` and
`reflected_rain_loss_db` expose diagnostic geometric-path losses. Loss on a
disabled/occluded path does not imply that it contributes received power.

```python
from gnss_gpu import CpuSignalSimulator
from gnss_gpu.rain_cpu import RainLayer
from gnss_gpu.wall_signal_sim import VerticalWall, simulate_wall_signal

sim = CpuSignalSimulator(2.046e6, noise_floor_db=-10000)
wall = VerticalWall(0., -100., 100., 0., 100.,
                    water_film_thickness_m=.001, water_permittivity=80-8j)
block = simulate_wall_signal(
    sim, wall, [-20., 0., 1.5], [0., 0., 0.], [-1e7, 1e7, 1e7], 2046,
    reference_range_m=22e6, integrate_samples=True,
    rain_layer=RainLayer(rate_mm_h=100., bottom_m=0., top_m=3000.))
print(block.direct_rain_loss_db)
```

## IQ to position experiment

```bash
PYTHONPATH=python:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python experiments/run_cpu_rain_positioning.py --workers 4
PYTHONPATH=python:. python -m pytest tests/test_rain_cpu.py tests/test_positioning_cpu.py tests/test_cpu_rain_positioning.py -q
```

The experiment uses 12 fixed synthetic local-ENU source directions at ranges
near 22 Mm. Sixteen prescribed receiver poses near x=-20 m each get eight
independent noise realizations, for 128 trials per scenario. Every channel has
nominal pre-aperture direct C/N0 55 dB-Hz, 2.046 MHz sampling and known zero
carrier. PRNs are known, and IQ is synthesized **separately per channel**;
multiaccess interference is not part of this experiment. Geometry and all
measurement-generation parameters are saved in metadata.

Open-space controls remove all wall intersections and reflections. Wall cases
use concrete at x=0, y=[-100,100], z=[0,100] m, with an illustrative receive
LHCP voltage response 0.1j. Some source directions are blocked. Scenarios are:

- dry baseline; 25 or 100 mm/h with a dry surface;
- a 0.1 or 1 mm film with no atmospheric rain;
- 100 mm/h combined with a 1 mm film.

Each PRN's single-carrier delay search is calibrated with 199 independent
noise-only blocks, target marginal false-alarm probability 1%, before any
weather evaluation. The same frozen gates apply to all environments/weather.
The existing `CpuCaTracker` requires two consecutive frequency/code-consistent
detections per snapshot. These use two independent noise blocks, so the minimum
observation is 2 ms. In this static, zero-carrier scene the noiseless waveform
repeats exactly each millisecond; replaying it with fresh noise is equivalent
to synthesizing the next identical period. This is per-snapshot confirmation,
not a continuous moving receiver or a navigation-bit-decoding experiment.

Code delays become pseudoranges using a known common 22 Mm reference that
selects their integer code period. This experiment does not solve code-period
ambiguity. `positioning_cpu.solve_pseudorange_position` uses NumPy nonlinear
least squares for receiver x/y/z plus clock bias in metres. It requires four
accepted channels and full rank, starts at the same coarse position (-25,5,0),
and uses equal weights. True position, path visibility and wetness are **never**
used for receiver rejection or weighting. Nonconvergence/insufficient channels
remain unavailable observations, not zero-error solutions. A solved position
is not by itself an integrity-validated fix.

Standardized noise is identical across weather and environment comparisons for
each pose/trial/PRN/confirmation block. Scenarios are paired, not independent
extra realizations. There are 16 synthetic geometries, not 128 independent city
routes. Per-PRN gate calibration does not imply a sequence/PVT false-alarm bound.
Spawned CPU workers parallelize poses and limit BLAS/Numba threads per worker.

## Initial results and interpretation

With confirmed-mode noise seed 24681, all 128/128 trials in each scenario
produce positions, with no >100 m position errors in this run. Selected results:

| Environment / control | Horizontal P50 [m] | Horizontal P95 [m] | 3D P95 [m] | Median paired 3D position shift [m] |
| --- | ---: | ---: | ---: | ---: |
| Open, dry | 4.231 | 8.487 | 16.410 | 0 |
| Open, 25 mm/h | 4.234 | 8.494 | 16.423 | 0.0066 |
| Open, 100 mm/h | 4.242 | 8.511 | 16.461 | 0.0249 |
| Wall, dry | 15.186 | 37.279 | 57.958 | 0 |
| Wall, 100 mm/h, dry surface | 15.214 | 37.311 | 58.072 | 0.0483 |
| Wall, 0.1 mm film, no atmospheric rain | 15.287 | 36.943 | 57.988 | 0.473 |
| Wall, 1 mm film, no atmospheric rain | 15.601 | 39.274 | 62.579 | 5.862 |
| Wall, 100 mm/h plus 1 mm film | 15.614 | 39.357 | 62.637 | 5.899 |

Paired position shift is `norm(position_case-position_dry)` for identical
inputs/noise. It is an **effect size, not the increase in positioning error**.
For example, 5.862 m of paired shift does not mean the 3D error grew by 5.862 m.
Every condition can move individual errors in either direction through coherent
multipath. The 100 mm/h direct-path losses in this constellation are only
0.01397–0.04023 dB. The small atmospheric-only change and larger water-film
sensitivity are consistent with the chosen mechanisms; they are not a claim
that all real rainy-weather GNSS degradation is small or caused by wet walls.

The one-block diagnostic is deliberately retained:

```bash
PYTHONPATH=python:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python experiments/run_cpu_rain_positioning.py --workers 4 --confirmation-blocks 1 --signal-seed 24680 --output-dir output/cpu-rain-positioning-raw
```

That seed informed the decision to connect the already implemented two-block
confirmation. In the raw wall dry baseline, 9/128 solved positions exceed
100 m and horizontal P95 is about 11.9 km because blocked-source noise peaks
are accepted. This is a receiver false-capture failure, not rain causing
kilometre errors. The confirmed run uses a different noise seed 24681; the two
receiver modes are not presented as a paired causal performance comparison.
Neither result is filtered using geometric truth, nor are thresholds retuned
on evaluation. The new seed's zero observed false captures is not a guaranteed
false-fix probability. Thresholds and confirmation parameters remain fixed.

`output/cpu-rain-positioning-confirmed/` contains 1,152 position records,
13,824 channel decisions, 2,388 calibration scores, all nine summaries,
source/settings/seed metadata and `rain_positioning.png`. The raw directory
retains the corresponding failed decisions and positions. Availability and
>100 m counts accompany conditional error quantiles. JSON provenance is in
[rain_sources.json](rain_sources.json); the full coefficient-source notice is
in [itu_rpy_notice.txt](itu_rpy_notice.txt).

## Scope still missing for real weather studies

The [wet antenna/radome sensitivity model](wet_antenna_simulation.md) now adds
independent time/direction-dependent signal loss, code bias and carrier phase,
with continuous code-following and position outage/recovery measurements.
Its coefficients and fast replay timing are prescribed, not empirically fitted.

The separate [wet-delay residual experiment](wet_delay_simulation.md) now adds
a signed neutral delay to code and carrier, with geometry-dependent mapping,
analytic delay rate and position/clock sensitivity. The baseline rain/water-film
experiment above continues to hold that residual at zero.

Rain rate is not a replacement for water-vapour pressure or residual zenith wet
delay. This experiment holds atmospheric group delay fixed and does not model
wet tropospheric residuals, rain phase/group delay, antenna/radome wetting,
water-driven antenna phase-centre changes or receiver noise-temperature changes.
It also omits orbit/transmit-time/Sagnac/ionosphere effects. These controls should
be added separately with measurements or defensible parameter priors before
predicting real rainy-day position errors. Soil/puddle reflections, temporal
drying/water films and spatial rain cells are further extensions. The current
result is a reproducible controlled experiment in attenuation and wet-surface
reflection, including their receiver/geometry dependencies.
