# Wet antenna/radome sensitivity and drying replay

`wet_antenna_cpu.WetAntennaResponse` adds independently prescribed signal loss,
code-epoch bias and carrier phase intercept, with a continuous wetting/drying
cycle and optional direction dependence. This is a **phenomenological sensitivity
model awaiting empirical calibration**, not a rainfall law or an electromagnetic
solution for a particular antenna/radome.

## Response contract

For each arriving path, wetness w(t) multiplies each supplied coefficient:

```text
q(ray)       = 1 - clip(ray_up / |ray|, 0, 1)
loss_dB(t)   = w(t) * (loss_db + elevation_loss_db*q)
code_bias(t) = w(t) * (code_bias_m + elevation_code_bias_m*q)
phase(t)     = w(t) * (phase_bias_rad + elevation_phase_bias_rad*q)
```

The simple q basis is a low-elevation sensitivity shape, with no azimuth
dependence and a flat lower-hemisphere response. It is not a measured gain or
phase-centre pattern. A direct and reflected ray use their **own arrival
directions**, so phase/loss changes can also change coherent multipath.
Existing circular polarization response remains a separate setting.

Signal voltage is multiplied by `10^(-loss_dB/20)` **before** receiver noise.
The code envelope receives the extra delay `code_bias_m/c`. Carrier phase
receives the independent `phase_bias_rad` intercept, not an automatic
`-2*pi*f_L1*code_bias_m/c` shift. These are separate local group-delay slope and
phase-intercept controls, analogous to a first-order narrowband response
`H(nu,t) = A(t)*exp(j*phase(t))*exp(-j*2*pi*nu*group_delay(t))` about the modeled
received carrier. This is not a full frequency-dependent causal RF filter or
a general time-varying convolution. Signed biases are relative to a dry
reference; absolute hardware delay is not inferred.

Analytic derivatives of w and ray direction modify the **code rate** through
the code-bias rate and **carrier Doppler** through `phase_rate/(2*pi)`.
Amplitude is held locally constant within each sample aperture, as with the
existing geometric path gain. Thus a code-only bias does not spuriously change
RF phase, and a carrier-phase-only control does not pretend to be a code delay.

Wetness follows cubic smoothstep: 0→1 during wetting, one during the plateau,
then 1→0 during drying, with zero endpoint derivatives. Default schedule:

- wetting begins at 50 ms and finishes at 100 ms;
- full wetness lasts until 200 ms;
- drying finishes at 400 ms; the replay ends at 500 ms.

These are deliberately **compressed diagnostic times**, not observations of
real wetting or drying. The model API exposes all four timing parameters for
later calibration. The supplied experiment fixes the matching stage/recovery
windows to this schedule. Rainfall rate, temperature and film thickness do not
implicitly select any response coefficient or drying time.

```python
from gnss_gpu.wet_antenna_cpu import WetAntennaResponse

antenna = WetAntennaResponse(
    wet_start_s=.05, wetting_s=.05, drying_start_s=.2, drying_s=.2,
    loss_db=6., code_bias_m=.3, elevation_code_bias_m=.3,
    phase_bias_rad=1., elevation_phase_bias_rad=.5)
# Existing CPU simulator call:
block = simulate_wall_signal(..., antenna_response=antenna)
```

The optional response composes with the existing `rain_layer`, wall film and
`wet_delay_layer` parameters, which retain distinct meanings. This experiment
holds those additional weather controls off to isolate antenna response.
None/zero response leaves IQ unchanged. Outside the profile, the response is
exactly dry; receiver state can retain a temporary history of the preceding
wet period.

`WallSignalBlock` exposes direct/reflected antenna loss, code bias, carrier
phase, code-bias rate and phase-rate arrays, with names such as
`direct_antenna_code_bias_m` and `direct_antenna_phase_rate_rad_s`.
Propagation-range properties remain geometric plus atmospheric excess and do
not silently include the receiver's separate code bias. Doppler properties
include the antenna phase derivative. Disabled reflected paths do not acquire
an observable antenna response.

## Continuous receiver and position experiment

```bash
PYTHONPATH=python:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python experiments/run_cpu_wet_antenna.py --workers 4
PYTHONPATH=python:. python experiments/plot_cpu_wet_antenna_errors.py
PYTHONPATH=python:. python -m pytest tests/test_wet_antenna_cpu.py tests/test_cpu_wet_antenna.py -q
```

The seven fixed response cases are dry, 6 dB loss, **30 dB loss stress case**,
0.3 m common code bias, 0.3 m low-elevation code-bias coefficient, a carrier-phase
pattern (1 rad common plus 0.5 rad low-elevation coefficient), and combined
30 dB/code/phase stress. These amplitudes are independently prescribed inputs,
not measured rainy-day antenna errors. At peak wetness, 30 dB loss takes the
nominal direct C/N0 from 55 to 25 dB-Hz with fixed receiver N0.

Each case uses 500 **contiguous** 1 ms blocks, 12 fixed synthetic sources and
the same static receiver (-20,-1,1.5) m. Open space and a concrete wall are
separate geometries; wall bounds are x=0, y=[-100,100], z=[0,100] m, with the
existing illustrative LHCP receive response 0.1j. Two fresh noise seeds,
24683/24684, are paired across scenarios. No multiaccess interference or
orbital motion is introduced; PRN channels remain isolated. Block-aligned
alternating 50 bit/s nav data have unknown polarity to the receiver.

The original per-PRN 199-block single-carrier noise calibration is frozen
before the replay. Each channel uses default two-block confirmation,
exponential following and three-block coast/loss policy throughout the entire
episode; state is not reset at wetting or drying boundaries. The receiver
receives no wetness, bias, path visibility or stage labels.

The nominal carrier is known zero in this static experiment. A fixed zero-
carrier search is used; a few-Hz antenna-induced phase transient is not
separately searched here. The receiver fits unknown complex amplitude per
block and follows code delay. **It is not a PLL/carrier-phase navigation
receiver**, and the loss/recovery measurements concern code-following and
fresh-channel position availability, not carrier-cycle slips.

Position/clock least squares uses only channels with a **fresh measurement**,
not predicted coasting states, and requires at least four. No truth-based
rejection or weights are used. The known 22 Mm common range reference still
selects the code period. All unavailable positions are retained, together with
conditional error quantiles and large-error counts. A numerical position is
not an integrity-validated fix.

Common random numbers are paired in the **direct antenna carrier frame**:
generated circular Gaussian noise is rotated by the modeled direct phase at
each sample. This preserves marginal white-noise power after attenuation and
reduces variance from arbitrary phase/noise coupling between counterfactual
scenarios. It is not physical noise tracking or receiver truth aiding. Noise
power/temperature is otherwise held fixed. Serial blocks and paired cases
must not be counted as independent weather observations or used to claim
confidence intervals for real antennas.

## Metrics and controls

Initial position latency is measured at the end of the first available block.
Loss/reacquisition events and all six receiver state labels are retained per
channel. Channel-use counts have **all 12 input channels per block** as their
denominator, including blocked wall directions; wall/open rates therefore need
their respective dry controls for interpretation. Each record includes
geometric visibility for evaluation, never receiver selection.

Recovery is reported only when the fully wet 100–200 ms plateau contained a
position outage. It means five consecutive available positions after drying
begins; the recovery decision time is the **end of the fifth block**. No-loss
and not-recovered cases are distinct and use missing latency, not zero.
This is a short availability criterion, not proof that all later fixes are
correct or uninterrupted. Later outages remain in the records.

Paired height and clock shifts compare identical time/seed/geometry blocks
with their dry control **only when both positions exist**. Missing wet fixes
remain missing, with explicit counts. A separate noiseless IQ test verifies
that a common 0.3 m code bias enters the clock, while the chosen
`0.3*(1-sin(elevation))` pattern gives approximately +0.3 m height and +0.3 m
clock. This particular direction basis is strongly coupled to vertical
position/clock geometry; it does not establish a universal wet-antenna error.

The output directory `output/cpu-wet-antenna/` retains 14,000 position blocks,
168,000 channel records, 28 episode summaries, 140 stage summaries, 2,388
calibration scores, profile/receiver/noise settings and source hashes. The
plot shows the first seed's open-space loss/recovery and separate bias/phase
controls; CSV records include both seeds and wall cases. Results are sensitivity
tests for prescribed coefficients, not a fitted antenna/radome prediction.

## Initial results

Dry and 6 dB-loss cases produce positions in 499/500 blocks for both seeds and
both geometries; the initial block is withheld for confirmation. Initial
position latency is 2 ms. During the fully wet 100-block plateau, all mild-loss
and bias/phase cases retain 100/100 positions, while every 30 dB/combined stress
case has **0/100 positions**. Fully wet horizontal error P95 illustrates the
precision cost even when availability is unchanged:

| Environment | Seed | Dry horizontal P95 [m] | 6 dB loss horizontal P95 [m] |
| --- | ---: | ---: | ---: |
| Open | 24683 | 3.27 | 6.63 |
| Open | 24684 | 2.77 | 5.95 |
| Wall | 24683 | 14.93 | 25.26 |
| Wall | 24684 | 11.96 | 18.73 |

The 30 dB and combined stress cases recover the five-position availability
criterion **114–115 ms after drying begins**, i.e. at 314–315 ms in this
compressed replay, before the prescribed response is fully dry at 400 ms.
Per-episode positions are available in 264–265/500 blocks, not a complete track.
This recovery time is a consequence of the fixed profile, C/N0 and receiver
settings; it is not an observed drying time of an actual antenna.

In the first seed's open-space wet plateau, the common 0.3 m code-only case
has median paired clock shift 0.3003 m and approximately zero height shift.
The 0.3 m elevation-pattern case instead has median height shift 0.2999 m and
clock shift 0.3000 m. These agree with the separately tested noiseless IQ
controls. Small wall deviations remain because of the finite-sampling,
multipath and receiver model. The phase-pattern case has approximately zero
steady code-position shift here: constant per-channel phase is fitted as an
unknown complex amplitude, and a vertical-wall far-field reflection changes
azimuth much more than elevation in this elevation-only basis. This does
**not** imply that wet-antenna phase is harmless in carrier-phase positioning
or with other measured antenna patterns.

Reappearance of a numerical position is not recovery of accuracy. Across both
seeds, the open 30 dB case has 529/1,000 available blocks, 471 unavailable blocks
and **five available positions with >100 m 3D error**, with maximum about
250.6 m. The wall case has 528/1,000 available, 472 unavailable and **four >100 m
positions**, maximum about 182.4 m. Combined stress has similar counts. These
errors, especially during weak-signal transitions, are retained; they are not
excluded as outliers or reclassified as missing. Counts refer to serial blocks,
not independent failure trials.

`wet_antenna.png` shows channel availability and the separate height/clock
effects. `wet_antenna_errors.png`, produced from the saved CSV by the second
command above, shows **all** available 3D errors for the four loss/stress cases,
with unavailable positions left as gaps. Its provenance file hashes the input
CSV, metadata and renderer. The recorded generation-source snapshots are also
archived under `generation_sources/`; metadata notes a later documentation-only
wall-module edit whose numeric AST was verified unchanged.

Future calibration needs dry/wet C/N0 changes, code and carrier residuals by
satellite direction, measured wetting/drying time series and independent position
truth. The [position-publication study](position_quality_simulation.md) now
checks redundant observations, DOP, residuals and temporal consistency on raw
recovery candidates, reporting both withheld fixes and fresh-noise holdout errors.
It does not turn the original replay into an integrity-certified receiver.

Further RF work needs receiver noise-temperature changes, a full antenna Jones
pattern, material dispersion, real radome geometry and carrier-phase tracking.
