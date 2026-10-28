# CPU GPS signal simulation

For the paper-linked propagation model, verified sources and polarization
extension, see [GNSS signal simulation literature](gnss_signal_simulation_literature.md).

`CpuSignalSimulator` produces interleaved float32 I/Q without CUDA. Install
`numpy` and `numba` (the existing `realtime` extra declares Numba); in the cloud
development environment, run `source /workspace/.gnss-env/activate.sh`.
The first generation compiles the Numba kernel. Later calls reuse it.

```python
from gnss_gpu import CpuSignalSimulator

sim = CpuSignalSimulator(sampling_freq=2.6e6, noise_seed=42)
iq = sim.generate_test_signal(prn=1, doppler=1200, cn0_dbhz=45, duration_s=0.02)
sim.write_bin(iq, "output/gps-l1.float32", fmt="float32")
```

The model implements GPS L1 C/A PRNs 1–32 with rectangular BPSK chips,
complex carrier phase, constant Doppler and code Doppler
`1.023e6 * (1 + doppler_hz / 1575.42e6)`. It accepts multiple channels,
including coherent delayed replicas for multipath. Other constellations raise
an error. It does not substitute synthetic codes for unsupported systems.

`generate_test_signal` calibrates complex white noise to C/N0 in dB-Hz:
each I/Q component has variance `sampling_freq / (2 * 10**(cn0_dbhz/10))`
for a unit-amplitude satellite. In contrast, the constructor's `noise_floor_db`
is the per-component noise standard deviation in dB relative to unit amplitude,
matching the existing CUDA wrapper's convention. CUDA random samples and the
CPU generator's samples are not expected to match.

For navigation-bit transitions and continuous noiseless blocks:

```python
sim = CpuSignalSimulator(noise_floor_db=-10000)  # noise underflows to zero
channels = [dict(prn=1, code_phase=0., carrier_phase=0., doppler_hz=1200.,
                 amplitude=1., nav_bit=1, nav_bits=[1, -1, 1], nav_bit_rate=50.)]
a = sim.generate_epoch(channels, n_samples=52000)
b = sim.generate_epoch(channels, n_samples=52000, start_sample=52000)
```

Phases and navigation bits are referenced to sample zero. Positive code phase
advances the spreading code. A propagation delay `tau` therefore adds
`-1.023e6*tau` chips and `-2*pi*1575.42e6*tau` carrier radians to a replica.
Navigation bit sequences must cover the requested interval; missing bits are
rejected instead of repeating the final bit. Without `nav_bits`, `nav_bit`
stays constant. `generate_epochs` creates independent epochs at sample zero by
default, as the CUDA API does. For continuous blocks use `start_sample` on
individual calls or explicit `start_samples` on a CPU batch.

Nonzero fixed noise seeds repeat noise per call and are independent of thread
count. Noise is not continuous across independently generated streaming blocks.
The kernel parallelizes over output samples, accumulating channels in fixed
order; changing thread count preserves the generated float32 samples exactly.
Set `NUMBA_NUM_THREADS=4` before starting Python, or call
`numba.set_num_threads(4)` within its configured limit.

Urban simulation can explicitly select the new generator:

```python
from gnss_gpu import UrbanSignalSimulator
urban = UrbanSignalSimulator(signal_backend="cpu")
# urban.compute_epoch(rx_ecef, sat_ecef, prn_list=[...]) returns actual IQ.
```

This selection changes IQ generation. Geometry/atmosphere backends retain their
own requirements: a CUDA-backed building model still needs CUDA. Open-sky urban
IQ generation works on CPU, and CPU-compatible scene adapters can supply LOS
and propagation paths. Orbit evolution, receiver dynamics, navigation-message
encoding, front-end band limiting and oscillator/clock impairments are not
implemented by the waveform generator. Supply phases/path parameters from the
appropriate models and avoid treating constant-parameter blocks as full dynamic
RF simulation. The default urban backend remains CUDA.

Validation and a repeatable benchmark:

```bash
PYTHONPATH=python python -m pytest tests/test_signal_sim_cpu.py -q
PYTHONPATH=python NUMBA_NUM_THREADS=4 python benchmarks/bench_signal_sim_cpu.py
```

Tests cover PRN 1's specified initial chips, code repetition, independent FFT
acquisition of a known delay, coherent multipath summation, a NumPy waveform
reference with code Doppler and bit transitions, block continuity, C/N0 noise
statistics, and exact serial/parallel equality. Benchmark medians exclude first
compilation, include channel packing/output checks, and report the measured
thread count and signal duration. They describe waveform synthesis, not urban
geometry throughput or end-to-end real-time performance.

## Short-block CPU validation optimization

On 2026-10-24, profiling 12-channel, 1 ms streaming calls found substantial
Python-side work in `np.isin` validation of navigation bits, including empty
sequences. The generator now skips membership work for empty sequences and
uses direct comparisons for the two allowed values in nonempty sequences.
Dimension, value and interval-coverage checks remain in place. That validation
change left the Numba kernel, summation order, navigation timing and noise
generation unchanged.

The [comparison benchmark](../benchmarks/bench_cpu_signal_validation.py) runs a
saved pre-change module and the updated implementation in the same process,
alternating their order over nine repetitions after JIT warmup. Each trial
generates 260,000 complex samples (0.1 s at 2.6 MHz), including channel packing,
output checks and output concatenation. This machine has a four-core CPU quota.
Results below use four Numba threads and disabled noise:

| Workload | Before, median | After, median | Before / after |
| --- | ---: | ---: | ---: |
| 1 channel, 100 × 1 ms blocks | 13.43 ms | 9.27 ms | 1.45× |
| 12 channels, 100 × 1 ms blocks, constant nav bit | 81.83 ms | 46.25 ms | 1.77× |
| 12 channels, 100 × 1 ms blocks, navigation transitions | 76.37 ms | 54.54 ms | 1.40× |
| 12 channels, one 100 ms block | 23.75 ms | 24.60 ms | 0.97× |

The 12-channel constant-bit short-block case also improves 1.42× with one
thread (97.58 → 68.70 ms). Every generated I/Q sample in the benchmark matches
the saved implementation bit for bit, and one/four-thread outputs also match.
Long-block timings show no improvement; do not extrapolate the short-block
gain to other workloads or claim a universal speedup. Medians describe this
machine and these trials, not a statistical performance bound. Geometry,
orbit calculation, tracking and position solving are outside this benchmark.

The saved baseline source is
`output/cpu-signal-speed/baseline_signal_sim_cpu.py`, SHA-256
`c47f322e7075a86217a7aeefda778e068b3ea292ec343f08681e3ad5c8b1068c`.
Full timings, versions, thread counts, waveform checks and source hashes are
saved in `output/cpu-signal-speed/comparison.json`. To rerun against that
saved pre-change source:

```bash
PYTHONPATH=python:. python benchmarks/bench_cpu_signal_validation.py \
  --baseline-source output/cpu-signal-speed/baseline_signal_sim_cpu.py \
  --repeats 9
```

Signal synthesis, finite-wall and moving-wall regression checks pass: 45 tests.
They cover independent waveform calculations, code/carrier/nav continuity,
noise statistics, thread equality, propagation dynamics and rejected invalid
inputs. Additional navigation regressions cover NaN/fractional bits, empty
non-vector arrays, and explicit empty sequences holding the constant bit.

## Per-channel constants in the waveform kernel

The next optimization computes each channel's constant code step and carrier
angular rate once before the parallel sample loop. Previously, both were
recomputed inside the channel loop for every output sample. The arithmetic
order for each constant, sample timing, channel summation and final float32
conversion are retained; no phase recurrence, fast-math or approximation is
introduced. Validation and noise generation retain the prior behavior.

The comparison baseline **already includes** the short-block validation change
above, so these measurements show its additional benefit, not a repeat of the
earlier baseline comparison. The same benchmark alternates execution order
over **11 repetitions**, with first-call JIT outside the timed intervals:

| Workload | Threads | Before, median | After, median | Before / after |
| --- | ---: | ---: | ---: | ---: |
| 12 channels, one 100 ms block | 1 | 55.81 ms | 48.30 ms | 1.16× |
| 12 channels, one 100 ms block | 4 | 23.32 ms | 20.64 ms | 1.13× |
| 12 channels, 100 × 1 ms blocks | 4 | 41.45 ms | 38.94 ms | 1.06× |
| 12 channels, 100 × 1 ms blocks with nav transitions | 4 | 53.35 ms | 50.24 ms | 1.06× |

This is measured CPU waveform-generation performance on this environment's
four-core quota, not a full receiver/geometry real-time claim or a general
speed guarantee. Each trial generates 260,000 complex samples without noise.
All eight workload/thread combinations in the full report match the saved
baseline bit for bit, including one/four-thread equivalence.

An additional 48 waveform comparisons cover 2.046/2.6 MHz sampling,
0/350 kHz intermediate frequency, mixed positive/negative/zero Doppler,
different PRNs and amplitudes, constant and changing nav bits, nonzero block
starts, seeded noise on/off and one/four threads. Every compared I/Q sample is
bitwise equal. Independent waveform, coherent propagation, acquisition and
tracking regression suites pass: **83 tests**.

The saved baseline is `output/cpu-kernel-speed/baseline_signal_sim_cpu.py`,
SHA-256 `e35960d95619cdcbcb9c1b2168eeeb7eecb1b5481f0583bb24e6fd7ea17d585d`.
Timing samples and source/version hashes are in
`output/cpu-kernel-speed/constant-hoist-comparison.json`; the additional
equivalence cases are in `output/cpu-kernel-speed/waveform_equivalence.json`.
Reproduce the timing comparison with:

```bash
PYTHONPATH=python:. python benchmarks/bench_cpu_signal_validation.py \
  --baseline-source output/cpu-kernel-speed/baseline_signal_sim_cpu.py \
  --repeats 11 --output output/cpu-kernel-speed/constant-hoist-comparison.json
```

## Independent-epoch CPU batches

`generate_epochs` now has a shared batch path for multiple short epochs with
more than one active Numba thread. It packs each epoch's channels once, shares
one 32-PRN code bank, and dispatches 256-sample tiles across epochs in one JIT
call. The kernel writes directly into the final batch array, avoiding a second
full-IQ allocation/copy for `np.stack` on this path. Channel parameters and
navigation arrays still scale with the total number of input channels. The
code bank adds a fixed approximately 256 KiB allocation; this is not a measured
whole-process peak-memory claim.

The existing API accepts different channel counts, empty epochs and repeated
PRNs. Every epoch retains input channel order and its own noise seed:

```python
sim = CpuSignalSimulator(noise_floor_db=-20)
iq = sim.generate_epochs([channels_a, [], channels_b], n_samples=2600,
                         noise_seeds=[31, 42, 53])
# iq.shape == (3, 5200); empty channels still receive configured receiver noise
```

Without `start_samples`, these are **independent epochs starting at sample
zero**, as before. They are not an implicitly continuous stream. For continuous
code/carrier/nav timing, use `generate_epoch(..., start_sample=...)` or provide
one explicit absolute offset per batch row, with unchanged sample-zero channel
definitions:

```python
blocks = sim.generate_epochs([channels]*3, n_samples=2600,
                             start_samples=[0, 2600, 5200],
                             noise_seeds=[31, 32, 33])
```

Navigation sequences must cover the last absolute sample, not just one block.
Offsets may be in arbitrary order for independent simulations; a contiguous
receiver requires them in consecutive order. Offsets alone do not integrate
phase for arbitrarily changing Doppler/trajectory parameters: the caller must
provide consistent sample-zero phase references. Noiseless contiguous batches
match one long generation exactly; seeded noisy blocks match individual calls
at the same offsets/seeds, not one long RNG draw. Fixed nonzero per-epoch seeds reproduce individual calls exactly;
`None` and zero retain the existing nondeterministic seed convention.

Measurements of the initial always-batched version showed regressions for one
thread and long epochs. The final dispatch therefore retains the prior
per-epoch path for one active thread or more than **8,192 samples per epoch**.
A single epoch also uses the single-epoch kernel directly. This cutoff is a
conservative engineering choice, not a universal optimum for every CPU.

The batch baseline already contains both preceding CPU optimizations.
Final measurements alternate baseline/candidate order over **17 repetitions**
in one process, exclude JIT warmup, and include validation, packing, synthesis,
noise and output assembly. Tests completed before timing. This environment has
a four-core CPU quota; the following use four Numba threads:

| Workload | Before, median | After, median | Before / after |
| --- | ---: | ---: | ---: |
| 128 × 2,600 samples, 12 channels, no noise | 43.08 ms | 40.27 ms | 1.07× |
| Same with seeded receiver noise | 58.18 ms | 50.28 ms | 1.16× |
| 128 × 2,600 samples, varying 0/1/4/12/32 channels | 36.05 ms | 34.64 ms | 1.04× |
| 8 × 26,000 samples, 12 channels (fallback) | 18.01 ms | 17.97 ms | 1.00× |

All ten workload/thread combinations match the saved baseline bit for bit,
and one/four-thread results match. One-thread cases use the fallback and show
no material speed improvement. Results vary between runs; these are local
medians, not general performance bounds or end-to-end receiver timings.

Signal, wall propagation, acquisition and tracking regressions pass: **98 tests**.
New tests cover ragged/empty epochs, repeated PRNs, navigation transitions,
per-epoch uint64 seeds, output row independence, invalid later epochs, and both
sides of the batch-dispatch cutoff.

The baseline is `output/cpu-batch-speed/baseline_signal_sim_cpu.py`, SHA-256
`210e4ff947a101d97af252f48cca929b2554f2080994a8e4d2779e413ee2864f`.
Full final timing samples, waveform checks and source/version hashes are in
`output/cpu-batch-speed/comparison.json`. Reproduce with:

```bash
PYTHONPATH=python:. python benchmarks/bench_cpu_signal_batch.py \
  --baseline-source output/cpu-batch-speed/baseline_signal_sim_cpu.py \
  --repeats 17
```

The subsequent [generation-to-receiver validation](cpu_batch_receiver_validation.md)
checks contiguous batch IQ, calibrated acquisition, code/carrier-state following,
outage and reacquisition, including separate generation/receiver timing.

## Moving receiver beside a finite wall

`gnss_gpu.wall_signal_sim` connects sample-wise path geometry directly to IQ.
`VerticalWall` describes a finite opaque rectangle at local `x=x_m`, with y/z
bounds, material and polarization. Coordinates are in a shared Cartesian frame
in metres; they need not be ECEF. Receiver and transmitter positions may follow
prescribed constant-acceleration trajectories. The default transmitter is
stationary and receiver acceleration is zero. Both direct and image-source specular path lengths are
recomputed at every receive sample. The complex Fresnel coefficient determines
reflection amplitude **and phase**, and fields scale inversely with path length.
The spreading code and RF carrier use the same propagation delay, so changing
range generates carrier and code Doppler without restarting phases per block.

```python
import numpy as np
from gnss_gpu import CpuSignalSimulator
from gnss_gpu.wall_signal_sim import VerticalWall, simulate_wall_signal

sim = CpuSignalSimulator(noise_floor_db=-10000)
wall = VerticalWall(0., -30., 5., 0., 100., material="concrete")
receiver = np.array([-20., -20., 1.5])
satellite = np.array([-1e7, 1e7, 1e7])
reference = float(np.linalg.norm(satellite-receiver))
block = simulate_wall_signal(
    sim, wall, receiver, [1., 5., 0.], satellite, n_samples=2600,
    reference_range_m=reference,
)
# Next contiguous block: keep the sample-zero position and reference unchanged.
next_block = simulate_wall_signal(
    sim, wall, receiver, [1., 5., 0.], satellite, n_samples=2600,
    start_sample=2600, reference_range_m=reference,
)
```

`block.iq` is interleaved float32 I/Q. The remaining fields expose per-sample
geometric ranges, visibility flags, reflection points and complex coefficients.
Inactive reflections have NaN points and zero coefficients; the image range is
still reported but contributes no signal. `reference_range_m` removes one fixed
common delay from code and carrier (receiver clock/time alignment). It does not
change geometric ranges. Do not reset that reference or the initial receiver
position between contiguous blocks. Noiseless blocks match concatenated
generation exactly; seeded noise retains the per-call behavior described above.

Run the reproducible moving-wall example:

```bash
PYTHONPATH=python python examples/demo_cpu_wall_signal.py
PYTHONPATH=python python -m pytest tests/test_wall_signal_sim.py tests/test_code_delay_cpu.py -q
```

The example writes `output/cpu-wall-signal/route.csv` and `route.png`. It samples
101 receiver poses over two seconds, synthesizing a 1 ms IQ interval at each.
These are separated observation windows; the simulator can also generate
contiguous blocks. The route crosses a finite reflection footprint and compares
direct-only, reflection-only and composite code-delay acquisition using the
same PRN and reference. No true path delay is supplied to either acquisition
estimator. Only local C/A code and sampling frequency are used. The initial
common range is removed explicitly as the experiment's clock alignment.

The diagnostic has 20 samples/chip at 20.46 MHz, or 14.65 m per coarse delay bin.
It compares the earlier three-bin parabolic peak interpolation with a receiver
aperture template fit on **the same integrated IQ**. The four panels show the
route, isolated reflection excess, direct-only estimator error against geometry
at the observation midpoint, and composite-minus-direct multipath bias for each
estimator. The CSV includes geometric delays, both estimates, baseline error and
normalized waveform-fit residual. Those residuals describe model mismatch; they
are not a confidence interval or signal-detection statistic.

### Sampling aperture and fractional delay

Instantaneous rectangular-chip samples cannot distinguish some sub-sample
delays once unknown carrier phase is absorbed in the amplitude. Correlation
interpolation alone cannot recover information absent from those samples.
`simulate_wall_signal(..., integrate_samples=True)` instead averages the complex
RF field over a centered aperture of duration `1/sampling_freq`. The code-edge
and carrier integrals are analytical for affine delay during each aperture.
Range rate is computed from geometry; amplitude, Fresnel coefficient and
visibility are held at the aperture center. The integrated RF carrier includes
the resulting per-path Doppler. This mode requires at least two samples/chip.
Default `integrate_samples=False` retains instantaneous sampling.

The aperture is a rectangular integrate-and-dump observation, with a sinc
frequency response, not a brick-wall anti-alias filter. Paths switching at a
wall edge are still switched at the aperture center. Receiver-motion curvature
within a nanosecond-scale aperture is neglected. Noise is added after the
observation using the existing per-I/Q convention.

```python
from gnss_gpu.code_delay_cpu import estimate_ca_delay

# At an integer number of samples/chip, use a 1 ms integrated IQ block.
estimate = estimate_ca_delay(integrated_iq, sampling_freq=20.46e6, prn=1)
if estimate is not None:
    print(estimate.delay_m, estimate.fit_residual_fraction)
```

The estimator starts with circular FFT acquisition, then fits fractional delay
and an unknown complex amplitude near its strongest peak. For integer samples
per chip, aperture-averaged C/A templates are affine between half-integer sample
delays. The fit maximizes normalized coherent power by checking interval ends
and analytical stationary points, rather than fitting a parabola to three
correlation bins. No building geometry, true delay or carrier phase is supplied.
Returned delay is modulo one code period, wrapped into +/-0.5 ms. All-zero input
returns `None`; nonzero noise can still produce a peak, so this is not a detector.

The fit models a single nearly constant path and assumes negligible RF carrier
rotation within one sampling aperture. `carrier_hz` can wipe a supplied residual
carrier before fitting, but there is no carrier search or code-Doppler
correction. High intermediate frequencies require a more complete matched
front-end template. The wall demo has a near-zero baseband carrier and short
observation windows. Multipath, fading, model errors and noise can bias the fit;
the composite-minus-direct estimate is still estimator-specific.

Tests recover known fractional delays with unknown phase/amplitude, verify RF
aperture integration against dense independent quadrature, cover periodic delay
ambiguity and seeded noise, and preserve exact moving-block equality. The
noiseless single-path calibration is an ideal synthetic check, not a prediction
of real receiver accuracy. The previous raw-sample and geometry checks remain.

This model holds one navigation bit constant, uses one prescribed transmitter
and one ideal two-sided opaque wall, and has no diffraction, transmission,
roughness, antenna pattern or a full RF front-end filter. Specular paths switch abruptly at the
rectangle boundary. A blocked path is removed rather than attenuated into a
fabricated direct signal. The model is a controlled test scene, not a full city
or dynamic navigation-message simulator.

For a moving satellite and accelerating receiver, supply initial velocities
and accelerations relative to sample zero:

```python
block = simulate_wall_signal(
    sim, wall, receiver, [1., 5., 0.], satellite, n_samples=2600,
    transmitter_velocity_mps=[1200., 2500., -500.],
    rx_acceleration_mps2=[0., 20., 0.],
    transmitter_acceleration_mps2=[-.1, .2, -.3],
    reference_range_m=reference,
)
```

Ranges, range rates and Doppler are derived from the same geometry. See the
[full-text model mapping](gnss_signal_simulation_literature.md#full-text-driven-extension-moving-transmitter-and-accelerating-receiver)
and `examples/demo_cpu_signal_dynamics.py`. This does not add satellite orbit,
transmit-time, Earth-rotation, clock or atmospheric modeling. Doppler can now
be much larger than the near-zero carrier assumed by the simple code-delay
estimator; use an appropriate carrier wipe/search instead of applying its
zero-carrier configuration blindly. Previously saved white-noise detector
thresholds do not cover a new multi-frequency search.

## Noise and C/N0 Monte Carlo

Run the paired noise experiment from the repository root after activating the
CPU environment (or installing `scripts/ci/requirements-full-suite.txt`, which
provides the scientific dependencies, Matplotlib and threadpoolctl):

```bash
PYTHONPATH=python:. python experiments/run_cpu_wall_noise.py --workers 4
PYTHONPATH=python:. python -m pytest tests/test_cpu_wall_noise.py -q
```

The default experiment uses C/N0 values 25, 35, 45 and 55 dB-Hz, three nearby
static receiver poses and 32 random trials per pose: 96 trials per condition
and C/N0, or 1,152 rows across direct-only, direct-plus-reflection and noise-only
conditions. Each observation is 1 ms at 20.46 MHz with integrate-and-dump
sampling. Receiver x offsets of 0, 0.05 and 0.10 m change the reflection's
relative carrier phase. The finite concrete wall, transmitter, poses, Fresnel
coefficients, relative carrier phases and reference-clock alignments are saved.
The 4.5 m residual direct delay is non-integer in samples and known only to the
generator and scorer; acquisition receives no truth or geometry.

Independent white Gaussian I/Q components have variance
`fs/(2*10**(cn0_dbhz/10))`. C/N0 is referenced to unit direct carrier power before
sample-aperture chip-edge attenuation. Receiver noise stays unchanged when a
reflection is added; composite signal power is not renormalized. A reflection
can improve or degrade acquisition depending on its phase and added signal
power, so a pooled comparison must not be read as a universal multipath penalty.
Use the per-pose summary to inspect these cases.

Trials share the exact same noise array across the three conditions. Standard
normal draws are also reused with different scaling across C/N0 levels. Draws
are independent across pose/trial keys derived from the explicit root seed.
Worker scheduling does not choose seeds. Spawn processes avoid forking the
Numba/OpenMP pool used for scene synthesis; each acquisition task uses one BLAS
thread. Serial execution temporarily limits and then restores BLAS threads.
No GPU is required, and serial/spawn trial outputs are tested for exact equality.

Outputs under `output/cpu-wall-noise/` are:

- `trials.csv`: every estimate, expected delay, wrapped error, fit residual,
  condition, seed/trial key, noise scale, status and bad/missing flag.
- `summary.csv`: pooled signed bias, RMSE, median/P95/maximum absolute error,
  returned/missing counts and bad-or-missing fraction.
- `per_pose_summary.csv`: the same statistics separated by receiver pose.
- `metadata.json`: parameters, pairing rules, runtime versions and source-file
  SHA-256 hashes for the code used in the experiment.
- `noise_comparison.png`: median/P95 errors, bad/missing fractions and a
  noise-only negative control.

Wrong code peaks remain in all error statistics; no post-hoc inlier selection
is performed. Errors are wrapped modulo the 1 ms C/A period into +/-0.5 ms,
so this experiment does not resolve integer code-period ambiguity. P95 uses
the higher order statistic from all available errors. Missing estimates are
counted separately and included in the bad-or-missing fraction, whose denominator
is every trial. The default bad-error threshold is 30 m (`--threshold-m`), an
evaluation threshold with truth, not a signal-detection threshold.

The noise-only condition deliberately has no true delay, ranging error or
bad/missing classification. Nonzero noise still returns a strongest correlation
peak: peak return is **not** detection. These controls are labeled by the nominal
C/N0 noise scaling; a noise-only signal has no physical C/N0. Fit residuals also
do not establish detection or calibrated uncertainty. No measurement gate has
been added to the acquisition API by this experiment.

This is a finite Monte Carlo sample of three static poses with one ideal wall,
constant navigation polarity and a single 1 ms observation. The tails depend
on the trial count and seed; use `--trials` and `--root-seed` to extend or repeat
the experiment. It does not validate live tracking, longer coherent integration,
receiver RF filtering or performance on real GNSS recordings.

## Noise-calibrated signal detection

`estimate_ca_delay` still returns its strongest candidate without deciding
whether a signal exists. `gnss_gpu.code_detection_cpu` adds an explicit gate;
rejected candidates remain available for diagnostics. Callers must check
`accepted` before treating a candidate as a detected signal.

```python
from gnss_gpu.code_delay_cpu import estimate_ca_delay
from gnss_gpu.code_detection_cpu import (
    ca_detection_statistic, calibrate_ca_threshold, detect_ca_signal,
)

# Supply independent, representative noise-only 1 ms blocks for this search.
scores = [ca_detection_statistic(estimate_ca_delay(iq, 20.46e6))
          for iq in calibration_noise_blocks]
threshold = calibrate_ca_threshold(scores, 20.46e6, false_alarm_probability=.01)
result = detect_ca_signal(observation_iq, threshold)
if result.accepted:
    print(result.candidate.delay_m)
```

The statistic is `N * (1 - fit_residual_fraction)`, normalized fitted coherent
energy from the complete strongest-peak search and fractional fit. It is
invariant to overall IQ amplitude. Calibration must use the same PRN, sample
rate, carrier setting, observation length and estimator implementation as
evaluation. Threshold objects retain the search settings. This is one PRN and
one carrier setting, not a joint false-alarm limit for multiple PRNs or Doppler
bins. Colored noise, interference and RF recordings need representative new
calibration and validation.

For `n` independent calibration scores and target `alpha`, the threshold is
the `k = ceil((n+1)*(1-alpha))`th ascending score. A candidate must have a score
**strictly greater** than this value; ties are rejected. Under exchangeable
calibration and future noise scores, the marginal rank false-alarm bound is
`(n+1-k)/(n+1)`. This averages over calibration sets; it does not guarantee the
operating false-alarm probability of one particular saved threshold. Too few
calibration blocks for the requested tail probability raise an error. The code
does not fit a theoretical distribution to correlated code-phase bins.

Run independent calibration and evaluation:

```bash
PYTHONPATH=python:. python experiments/run_cpu_wall_detection.py --workers 4
PYTHONPATH=python:. python -m pytest tests/test_code_detection_cpu.py tests/test_cpu_wall_detection.py -q
```

Defaults use 1,000 noise-only calibration blocks, target false-alarm probability
1%, 1,000 held-out noise blocks and the existing three-pose/four-C/N0 signal
experiment with 32 trials per pose. SeedSequence domain keys for calibration,
held-out noise and signal poses cannot overlap. Calibration is completed and
the threshold frozen before evaluation. Changing evaluation count does not
retune the threshold. All null observations are independent of signal trials;
the same standardized signal noise reused across C/N0 is not counted as
additional independent held-out evidence.

Files in `output/cpu-wall-detection/` retain calibration statistics, every
held-out decision, every signal candidate including rejections, summary CSV,
parameters/source hashes, and `detection_comparison.png`. The held-out false-
alarm count is reported with a 95% Wilson binomial interval. Signal summaries
show acceptance, rejection and acceptance with <=30 m or >30 m truth error, all
using **every signal trial** as denominator. The 30 m evaluation threshold is
not used for detection. Accepted-only median error is explicitly labeled
conditional and includes its sample count; it must not replace the ungated
error statistics or hide rejected observations.

Acceptance is evidence against the calibrated noise model, not proof that the
range is correct. Multipath and low-SNR observations can pass the gate with a
bad delay. The experiment keeps these cases visible and does not add a tracking
or ranging-integrity guarantee. The existing noise-only experiment remains an
ungated reference.

## Unknown-carrier acquisition on CPU

`gnss_gpu.acquisition_cpu` extends the one-period aperture delay estimator to a
fixed grid of received carrier frequencies (IF plus Doppler). It evaluates the
complete delay fit at every frequency and selects the largest normalized fitted
coherent-energy score. The output frequency is a grid value; there is no
frequency interpolation or tracking loop. Unknown complex amplitude and phase
are fitted, and the receiver does not receive true geometry, delay or frequency.

```python
from gnss_gpu.acquisition_cpu import (
    CaSearch, acquire_ca_signal, calibrate_ca_search, detect_ca_search,
)

search = CaSearch(2.046e6, tuple(range(-6000, 6001, 500)), prn=1)
scores = [acquire_ca_signal(iq, search).statistic for iq in calibration_noise_blocks]
threshold = calibrate_ca_search(scores, search, false_alarm_probability=.01)
result = detect_ca_search(observation_iq, threshold, workers=4)
if result.accepted:
    print(result.candidate.carrier_hz, result.candidate.delay.delay_m)
```

Calibrate using **whole-grid maximum scores**, independently of evaluation.
Single-frequency thresholds do not control the larger search. The immutable
threshold stores the grid, PRN and sampling rate and always reuses them for
evaluation. Calibration provenance remains the caller's responsibility: a list
of scores alone cannot prove how they were obtained. The existing marginal
rank bound and exchangeability assumptions apply. More PRNs, changed grids,
colored noise or changed estimators require matching calibration.

`workers=4` in the API threads independent carrier bins with deterministic
selection order; it does not promise a speedup for small FFTs. The experiment
below instead uses spawned CPU processes across independent noise blocks.
Keep BLAS/OMP threads at one to avoid oversubscription. The code requires 1 ms
centered integrate-and-dump IQ at >=2 integer samples/chip, constant nav-bit
polarity, negligible code drift and acceleration-induced carrier drift within
that block. Carrier rotation within each aperture is neglected in the fitting
template, requiring carrier frequency small relative to sample rate. The
generated aperture IQ includes that rotation, so model mismatch remains.

```bash
PYTHONPATH=python:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python experiments/run_cpu_carrier_search.py --workers 4
PYTHONPATH=python:. python -m pytest tests/test_acquisition_cpu.py tests/test_cpu_carrier_search.py -q
```

The experiment uses the moving-transmitter/accelerating-receiver geometry
linked to Sanwale et al. Eq. (8), 25 frequencies from -6 to +6 kHz in 500 Hz
steps, 199 calibration nulls, 400 independently seeded held-out nulls, and
264 noisy isolated-path blocks. Four noise realizations at 11 times for each
path and nominal C/N0 35/45/55 dB-Hz give 44 observations per group. C/N0 refers
to the unit direct path before aperture: reflected power is reduced by Fresnel
voltage and spreading, so its actual C/N0 is lower. Noise is paired across C/N0;
these are not independent repetitions for aggregate confidence intervals.

For seed 20261024 the frozen threshold is 15.667522. Held-out false alarms are
**9/400 = 2.25%**, with 95% Wilson interval **1.19–4.22%**. This exceeds the
nominal 1% target for this particular calibration; the threshold is not retuned
after evaluation. The marginal rank bound averages over calibration sets and
does not guarantee each frozen threshold's operating false-alarm rate. This
small-calibration experiment is not evidence that the 1% operating target was
met. Direct-path acceptance is 1/44, 42/44, 44/44 at the three C/N0 levels;
accepted delays exceeding 30 m are respectively 1, 7, 0. Reflected-only blocks
are all rejected under this nominal-direct-power noise setting. Rejected
candidates are retained and must not be interpreted as measured Doppler.

`output/cpu-carrier-search/` contains all calibration scores, null decisions,
signal candidates including rejections and wrong peaks, group counts, a plot,
the frozen threshold, domain seeds, trajectories, assumptions and source hashes.
Truth is used only for evaluation. Counts of rejected and accepted-but-wrong
observations remain separate; acceptance is not a range-integrity guarantee.

## Carrier refinement and contiguous-block following

`gnss_gpu.tracking_cpu.refine_ca_carrier` maximizes the existing fractional-delay
profile score continuously between the coarse candidate's neighboring frequency
bins using bounded golden-section search (default 0.1 Hz bracket tolerance,
32 iterations). It refits code delay at each frequency and retains the coarse
candidate if refinement finds no improvement. This is local optimization,
assuming a locally unimodal peak, with no guarantee against multipath or false
peaks. A single-bin grid cannot be refined, and the bracket never extends beyond
the original grid. Bracket tolerance is an optimizer setting, not frequency
accuracy or uncertainty.

`CpuCaTracker` follows frequency and wrapped code delay across contiguous 1 ms
blocks. The receiver gets only IQ, sample timestamps, the calibrated search
configuration and optional known IF. It performs the **original full coarse
search and gate on every block**, then refines only gate-accepted candidates.
The gate never uses the larger refined score, so refinement does not silently
change the detection statistic or require reusing an invalid coarse threshold.
The search is still evaluated in full; this implementation does not promise
that tracking reduces computation relative to repeated acquisition.

```python
from gnss_gpu.tracking_cpu import CpuCaTracker

# threshold is the whole-grid noise-calibrated CaSearchThreshold from above.
tracker = CpuCaTracker(threshold, intermediate_freq=0., gain=.25, coast_blocks=3)
for start_sample, iq in contiguous_blocks:
    state = tracker.update(iq, start_sample=start_sample)
    if state.measurement_used:
        print(state.status, state.carrier_hz, state.delay_samples)
```

State updates exponentially smooth accepted measurements. Code delay prediction
uses `delta_delay_samples = -N * (carrier_hz - IF_hz) / f_L1`, where `N` is one
block's sample count. Wrapped delay and frequency innovations must fit the
configured limits (defaults 1 sample and 500 Hz). Carrier prediction is constant;
acceleration therefore causes smoothing lag. These gates do not prove that the
correct path is followed. `detection_accepted` means that the coarse score passed
its noise gate; `measurement_used` additionally requires the innovation gates.
Returned coarse/refined candidates retain diagnostics even when an innovation
is rejected.

States are `searching`, `confirming`, `acquired`, `tracking`, `coasting`, and `lost`. Coasting
outputs are predictions with `measurement_used=False`. After three consecutive
unused measurements the state is discarded and carrier/delay outputs are
`None`. Acquisition and reacquisition require two consecutive consistent
accepted observations by default, as described below. `confirmation_blocks=1`
retains the historical single-block behavior. `missed_blocks` counts consecutive
unused observations. Timestamp gaps, overlap and malformed IQ raise an error
without advancing the state; call `reset()` before processing a gap.

The procedure estimates each block's unknown complex amplitude independently.
It supports nav polarity changes **between** aligned blocks, while assuming
constant polarity within each block. It does not unwrap carrier phase or
implement a PLL/FLL/DLL, bit synchronization, ephemeris processing or PVT.
The per-block rank false-alarm bound does not imply a false-track bound over a
sequence. Repeated observations create repeated opportunities for false capture.

```bash
PYTHONPATH=python:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python experiments/run_cpu_tracking.py --workers 4
PYTHONPATH=python:. python -m pytest tests/test_tracking_cpu.py -q
```

This historical baseline demo explicitly uses `confirmation_blocks=1` (also its
CLI default), while `CpuCaTracker` now defaults to two-block confirmation. Use
`--confirmation-blocks 2` to change the demo. The experiment calibrates 199 independent noise blocks before evaluating a
200 ms direct-only trajectory at nominal pre-aperture C/N0 55 dB-Hz. It uses
the moving transmitter (1,000 m/s, acceleration 2 m/s²) and receiver (5 m/s,
acceleration 200 m/s²) from the dynamics example, alternating block-aligned
50 bit/s nav bits, and a 20 ms signal outage from 80 to 100 ms with noise still
present. Neither outage flags nor geometric truth are supplied to the receiver.
CPU processes parallelize independent calibration observations; stateful
tracking remains sequential in time. Seed/domain keys separate calibration
from evaluation. These are illustrative settings, not reproduced paper figures.

With seed 20261025, all 180 signal-present blocks are used. On those same blocks,
carrier RMSE is **139.40 Hz coarse, 25.26 Hz refined, 10.97 Hz smoothed**. The
smoothed delay RMSE is **2.32 m**. These are conditional synthetic results for
one trajectory and noise realization, not real receiver accuracy. During the
20 signal-absent blocks, **one noise-only block is falsely acquired** (block 85),
with about 100 km wrapped-delay error, then discarded after three unused
observations. There are three acquisition events, 178 tracking blocks, four
coasting blocks, two loss events and 13 searching blocks. This failure remains
in the complete CSV and plot; the implementation is not claimed to solve
false reacquisition. No parameters or threshold are retuned after evaluation.

`output/cpu-tracking/` contains `rows.csv`, calibration scores, `summary.json`,
`metadata.json` with the frozen threshold, seeds, parameters and source hashes,
and `tracking.png`. The plot includes outage false capture and coasting errors
on symmetric logarithmic axes, rather than clipping them to the normal error
range. Error summaries explicitly select used signal-present blocks and report
their count alongside all signal-present/absent counts.

## Multi-block acquisition confirmation

`CpuCaTracker(..., confirmation_blocks=2)` now requires two consecutive coarse-
gate-accepted, refined candidates consistent in frequency and wrapped code delay
before publishing a state. This is the API default. Frequency and code-delay
consistency use the existing 500 Hz and 1 sample innovation bounds, including
the same carrier/IF-aided delay prediction as normal tracking. Nav polarity may
change between blocks. There is no geometry/truth input to confirmation.

The first accepted candidate has status `confirming`, `confirmation_count=1`,
`measurement_used=False`, and `carrier_hz=delay_samples=None`. Its coarse and
refined candidates remain available for diagnostics. A rejected block clears
the pending chain; an inconsistent accepted candidate replaces it and restarts
the count at one. On the second consistent accepted block, status becomes
`acquired`, the current measurement initializes the state, and the count is two.
Normal tracking reports count zero. `missed_blocks` includes confirmation
waiting because those observations have not yet updated a published state.
`reset()` and loss discard pending evidence; reacquisition requires a new chain.
`confirmation_blocks=1` provides the original behavior, and larger positive
integer counts are supported. Timestamp or IQ validation errors leave pending
evidence unchanged so the same timestamp can be retried.

This adds at least one block of acquisition latency. Low-C/N0 missed detections
or inconsistent estimates can make the wait longer, or prevent acquisition.
Confirmation does not change the coarse detection statistic or retune its
threshold. It is an engineering consistency check, **not** an `alpha²`
false-track bound: search peaks, repeated observations and state decisions
require sequence-level validation. A consistent false path can still acquire.

```bash
PYTHONPATH=python:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python experiments/run_cpu_tracking_confirmation.py --workers 4
PYTHONPATH=python:. python -m pytest tests/test_tracking_confirmation_cpu.py tests/test_cpu_tracking_confirmation.py -q
```

The paired comparison keeps the 199-block calibration, complete coarse grid,
gain, innovation bounds and trajectory fixed. One- and two-block modes receive
identical IQ in each case. Both initial acquisition and return at 100 ms include
the decision block in their latency: an immediate one-block acquisition takes
1 ms; two consecutive blocks take 2 ms. Missing acquisition is `None`, not zero.
All confirming, rejected, coasting and wrong estimates remain in the CSV and
all-block denominators.

Seed 20261025 reproduces the failure that informed this change and is labeled
**regression**. Seeds 20261026/20261027 are new evaluation realizations; the
two-block setting is fixed before evaluating them. The 45/55 dB-Hz conditions
reuse standardized noise within a seed, including the outage, so their null
outcomes must not be counted as independent evidence. Results are:

| Seed | C/N0 [dB-Hz] | Outage false acquisitions, 1→2 blocks | Used / 180 signal-present blocks, 1→2 | Initial decision [ms], 1→2 | Return decision [ms], 1→2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 20261025 (regression) | 45 | 1→0 | 167→165 | 1→2 | 1→2 |
| 20261025 (regression) | 55 | 1→0 | 180→178 | 1→2 | 1→2 |
| 20261026 (evaluation) | 45 | 0→0 | 157→154 | 2→3 | 1→2 |
| 20261026 (evaluation) | 55 | 0→0 | 180→178 | 1→2 | 1→2 |
| 20261027 (evaluation) | 45 | 1→0 | 170→169 | 1→2 | 2→2 |
| 20261027 (evaluation) | 55 | 1→0 | 179→178 | 1→2 | 2→2 |

The observed false capture is avoided in the regression and fresh-seed cases,
with fewer used signal blocks and delayed acquisition. At 45 dB-Hz, evaluation
seed 20261027 still has **three used signal-present blocks with >30 m delay
error in both modes**. Confirmation therefore does not provide ranging
integrity. Two new seeds and 40 unique fresh-seed outage noise blocks are a
small synthetic check, not a demonstrated operational false-track probability.
Parameters are not selected or retuned using these evaluation outcomes.

`output/cpu-tracking-confirmation/` retains 2,400 per-mode block records,
12 case summaries, the frozen calibration scores, trajectory/settings/source
hashes, regression/evaluation labels and `confirmation.png`. Spawned CPU workers
parallelize independent cases; each track's time evolution remains sequential.

## Continuous tracking with specular multipath

`experiments/run_cpu_multipath_tracking.py` extends the earlier isolated-path
and independent-pose experiments to contiguous 1 ms tracking observations.
Three independent receiver states process the same route with (1) direct-only
IQ, (2) direct plus ideal-RHCP reflected IQ, and (3) direct plus reflected IQ
with a receive LHCP voltage response `0.1j`. That response is an illustrative
-20 dB power leakage at +90 degrees, not a measured antenna pattern. Direct
amplitude and input noise are identical across these diagnostic propagation
variants; the reflected field is added coherently before noise.

The receiver starts at `(-20,-20,1.5)` m, moves at `(1,20,0)` m/s, and receives
from a stationary point transmitter at `(-1e7,1e7,1e7)` m in the common Cartesian
frame. A finite concrete wall lies at x=0, y in [-30,5], z in [0,100] m. During
the 400 ms trajectory the specular point exits its edge: block 263 contains the
last visible reflected samples, and block 264 is fully outside. Both direct
and composite receivers continue tracking across that transition. This is a
prescribed synthetic geometry, not satellite orbit propagation or a measured
urban route. The hard visibility mask has no diffraction/edge-field model.

```bash
PYTHONPATH=python:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python experiments/run_cpu_multipath_tracking.py --workers 4
PYTHONPATH=python:. python -m pytest tests/test_cpu_multipath_tracking.py -q
```

The five datasets are one deterministic noiseless control and two fresh seeds
(20261028/20261029) at each of nominal direct-path C/N0 55/65 dB-Hz. They use
2.046 MHz centered-aperture IQ and alternating block-aligned 50 bit/s nav data.
All 15 tracks retain the default two-block confirmation and the original gain
and innovation limits. A fixed near-zero-IF carrier prior of -1 to +1 kHz in
500 Hz steps is calibrated on 199 independent noise blocks with seed 20261030
before evaluating tracks. This narrower prior is specific to the slow receiver
and stationary transmitter experiment, not general satellite acquisition.
The exact path Doppler, delay and reflection visibility are never passed to
the receiver, and the threshold/settings are not retuned on evaluation results.

The same additive noise is used across all three propagation/antenna variants
within a dataset. Standardized noise is also reused across C/N0 for each seed;
these comparisons are paired, not independent additional null trials. The
noiseless control is deterministic and is not counted as a random realization.
CPU workers parallelize independent datasets and limit Numba and BLAS to one
thread per worker. Each track remains sequential in time.

Every track uses measurements in 399/400 blocks, with the initial confirmation
block retained as unused. There are no coasting, loss or >30 m used-delay-error
blocks in this short experiment. The single-path receiver nonetheless follows
a biased state in multipath:

| Dataset | Direct delay RMSE [m] | Ideal-RHCP composite delay RMSE [m] | Ideal-RHCP paired peak bias [m] | LHCP-leakage composite delay RMSE [m] | LHCP-leakage paired peak bias [m] |
| --- | ---: | ---: | ---: | ---: | ---: |
| Noiseless | <0.000001 | 2.72 | 5.72 | 2.77 | 5.79 |
| 20261028, 55 dB-Hz | 2.01 | 3.01 | 6.12 | 3.08 | 6.57 |
| 20261028, 65 dB-Hz | 0.63 | 2.65 | 5.84 | 2.71 | 5.97 |
| 20261029, 55 dB-Hz | 2.12 | 3.44 | 6.70 | 3.48 | 6.61 |
| 20261029, 65 dB-Hz | 0.67 | 2.78 | 5.98 | 2.83 | 6.02 |

RMSE is relative to geometric direct delay at block midpoint, conditional on
the stated 399 used measurements. The tiny noiseless direct-only residual is
an internal consistency result for the matched synthetic model, not a physical
accuracy claim. Paired bias is the wrapped composite-minus-direct tracked code
delay, evaluated only when **both** receivers use a measurement. Missing pairs
are retained as missing values with explicit pair counts. The measured peak
depends on route, polarization, receiver smoothing and the sampling model; it
is not a universal multipath bound. Antenna leakage does not monotonically
increase or reduce error, because its complex phase changes field interference.

After reflection disappears the instantaneous direct/composite waveforms are
identical, while smoothed states can still differ due to history. For the
noiseless ideal-RHCP case, the first fully reflection-free block retains about
-0.052 m of paired state bias before it decays. Two-block confirmation verifies
temporal consistency and does not remove these multipath errors. There is no
multipath mitigation or ranging-integrity guarantee here.

`output/cpu-multipath-tracking/` stores 6,000 per-mode block records, 15 summaries,
calibration scores, the frozen threshold, seeds, complete settings/source
hashes, and `multipath_tracking.png`. Records include rejected/pending states,
individual delay errors, paired biases, fitted residual fractions and per-path
geometric Doppler/visibility. The returned reflection visibility describes
geometry even in the direct-only diagnostic; its reflected waveform is disabled
by the mode switch. A composite waveform has no unique geometric carrier, so
the two path Dopplers are recorded separately. Fitted residual is waveform
mismatch, not a calibrated multipath alarm or ranging uncertainty. These two
seeds and one idealized wall do not establish real RF or PVT performance.

## Experimental two-path waveform fitting

`gnss_gpu.multipath_fit_cpu.fit_ca_two_paths` adds a local two-path parameter
estimator after acquisition. It fits
`z[n] = a1*C(n-tau1) + a2*C(n-tau2) + noise` after wiping the acquired common
carrier. `C` is the same centered-aperture C/A template as the single-path
estimator; complex amplitudes include unknown carrier phase and nav polarity.
The receiver receives no wall geometry, true delays, path separation, path count
or visibility. This is an experimental receiver implementation, not a claimed
reproduction of a cited paper's mitigation algorithm.

```python
from gnss_gpu.multipath_fit_cpu import fit_ca_two_paths

# carrier_hz comes from the existing coarse acquisition + refinement.
result = fit_ca_two_paths(iq, sampling_freq, prn=1, carrier_hz=carrier_hz)
if result is not None:
    print(result.selected_model, result.reason, result.delay_samples)
```

An integer-sample coarse pair search initializes up to eight projected
Gauss-Newton fits, each with at most 20 iterations. Amplitudes are solved by
linear least squares at each delay pair. The first delay is searched within
half a chip of the strongest single-path peak, and excess delay is bounded
between 0.05 sample and one chip. The earliest component must be stronger in
voltage magnitude than the later one. These are generic **LOS-dominant priors**,
not geometry-derived truth. They can be wrong in NLOS or when the strongest
peak is an echo. Delays are ordered in a local unwrapped neighborhood and
returned modulo one code period; a wrap can make the first returned value
numerically greater than the second.

For `N` complex samples, BIC compares 3 versus 6 real parameters using `2*N`
real white-noise observations. A shared numerical residual floor prevents
perfect-fit roundoff from becoming evidence for an extra path. A preferred
two-path fit also needs a full-rank projected delay Jacobian, acceptable local
delay standard deviation, and no solution on a template knot or search bound.
The API standard-deviation limit defaults to 0.25 sample; the experiment below
uses a fixed 5 m limit converted to samples at each rate. The approximate
standard deviation comes from the fitted residual variance and local curvature.
It excludes model error, alternative minima and uncertainty in the carrier.

`selected_model` is `single_path`, `two_path` or `ambiguous`. `delay_samples`
returns the single-model estimate, the estimated earliest component, or `None`,
respectively. All candidate delays, amplitudes, residuals, BIC difference, local
condition number and standard deviation remain available. Undefined local
conditioning is represented as `None`. A selected single model does **not**
establish that multipath is absent. These checks are diagnostics, not calibrated
signal detection, global separation or an integrity guarantee. The original
coarse gate is applied before fitting and never tested against the larger
two-path fitted score.

Resolution matters: two aperture templates with delays in the same affine
sample-delay interval can span the same two-dimensional waveform space for
many different delay pairs. Both amplitudes and delays are then not uniquely
recoverable, even with a nearly zero residual. A static noiseless 2.046 MHz
control explicitly tests this rank deficiency; a 20.46 MHz control separates
the same physical paths. Sampling-rate changes here also change the idealized
front-end bandwidth/noise model, so higher rate is not free resolution in a
real bandwidth-limited RF receiver.

```bash
PYTHONPATH=python:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python experiments/run_cpu_two_path_fit.py --workers 4
PYTHONPATH=python:. python -m pytest tests/test_multipath_fit_cpu.py tests/test_cpu_two_path_fit.py -q
```

The experiment samples 20 one-millisecond blocks at 20 ms intervals along the
previous moving-wall route. It compares both estimators on identical direct-
only and ideal-RHCP multipath IQ at 2.046/20.46 MHz. Each rate has a deterministic
noiseless control and two new seeds (3101/3102) at nominal direct C/N0
55/65/75 dB-Hz. Independent 199-block noise calibration for each rate freezes
the same -1 to +1 kHz carrier search before evaluation. Same-rate modes and
C/N0 settings share standardized noise, while noise differs between rates.
Each snapshot estimates its carrier from IQ. No true carrier, delay, excess
path length or visibility is supplied to the estimator. All fit parameters
are fixed before evaluating these seeds and are not retuned on their outcomes.

These are **per-block waveform estimates**, and the corrected delays have not
been inserted into the continuous tracker. They must not be compared directly
against the previous smoothed tracking RMSE. Representative multipath results
below compare the single and selected estimators on exactly the same available
blocks; missing results remain in counts:

| Rate | Dataset | Two / single / ambiguous out of 20 | Single RMSE on selected blocks [m] | Selected RMSE [m] |
| --- | --- | ---: | ---: | ---: |
| 2.046 MHz | Noiseless | 0 / 0 / 20 | — | — |
| 2.046 MHz | Seed 3101, 75 dB-Hz | 2 / 13 / 5 | 3.07 | **19.17** |
| 2.046 MHz | Seed 3102, 75 dB-Hz | 1 / 11 / 8 | 3.20 | **3.21** |
| 20.46 MHz | Noiseless | 14 / 6 / 0 | 1.06 | 0.00022 |
| 20.46 MHz | Seed 3101, 55 dB-Hz | 0 / 20 / 0 | 1.91 | 1.91 |
| 20.46 MHz | Seed 3102, 55 dB-Hz | 0 / 20 / 0 | 1.94 | 1.94 |
| 20.46 MHz | Seed 3101, 65 dB-Hz | 6 / 13 / 1 | 1.15 | **1.84** |
| 20.46 MHz | Seed 3102, 65 dB-Hz | 8 / 12 / 0 | 1.12 | **1.35** |
| 20.46 MHz | Seed 3101, 75 dB-Hz | 11 / 6 / 3 | 0.82 | 0.19 |
| 20.46 MHz | Seed 3102, 75 dB-Hz | 14 / 6 / 0 | 1.05 | 0.60 |

The high-rate, strong-signal cases improve, but this is not a general mitigation
success. At 65 dB-Hz both new high-rate seeds get worse. At 2.046 MHz/75 dB-Hz,
seed 3101 has a selected error exceeding 30 m; on its two corrected blocks the
RMSE is 51.91 m despite passing the local checks. This exposes the limits of
local variance/rank and the model-selection rule. No failed blocks are dropped
to claim an overall improvement, and no tuning is performed to remove them.

The constant-delay/common-carrier model also neglects code drift and
differential path Doppler during each moving block. Even direct-only noiseless
moving IQ at the low rate can prefer an ambiguous two-path approximation of
that small mismatch; all 20 such direct-only blocks are retained as ambiguous.
The tiny noiseless high-rate residuals are synthetic consistency results, not
physical ranging accuracy. The fitter remains a diagnostic prototype pending
better model mismatch handling and independent sequence-level validation.

`output/cpu-two-path-fit/` retains 560 snapshot/mode records, 28 summaries,
398 calibration scores, all missing/ambiguous decisions, acquired carriers,
candidate delays, residuals/BIC/local uncertainty, fixed settings and source
hashes, plus `two_path_fit.png`. Summaries include all detected blocks, selected
counts, and matched-subset baseline errors. There is no new RF, tracking-loop,
PVT or ranging-integrity claim.

## Rain and wet-surface positioning sensitivity

The [rain simulation guide](rain_simulation.md) adds P.838-3 rain-path
attenuation, an independently controlled water-film Fresnel model and a
NumPy-only position/receiver-clock solver. Paired open-space/wall experiments
compare dry surfaces, rain, water films and their combination. The guide records
source provenance, 2 ms acquisition confirmation, raw false-capture failures,
position error/availability and the missing wet-atmosphere/antenna mechanisms.

The [wet-delay residual extension](wet_delay_simulation.md) independently adds
signed zenith correction errors through a flat refractivity slab to both code
and carrier, including their derivatives. Its paired IQ-to-position experiment
records height/clock coupling, large remaining errors and a separate noiseless
geometric control; rainfall rate does not determine the prescribed residual.

The [wet antenna/radome study](wet_antenna_simulation.md) adds independent
time-varying loss, code bias and carrier-phase response, including direction
dependence and analytic rates. Continuous wetting/drying replays retain code-
following loss/reacquisition, unavailable positions and height/clock effects.
All coefficients remain uncalibrated sensitivity/stress controls.

The [position-publication experiment](position_quality_simulation.md) replays
the same receiver measurements through frozen redundancy, DOP, residual and
temporal checks, then evaluates fresh-noise IQ. It reports large-error reduction
together with withheld positions and recovery delay; the new gate remains an
opt-in experimental policy rather than a certified integrity bound.

The [reference-recovery extension](position_recovery_simulation.md) retains
those ordinary checks and adds median continuity references plus stronger-quality
recovery clusters. It compares all three publication policies on fresh noise,
including output gaps and coherent-wrong-cluster limitations.

The [uncertainty validation](position_uncertainty_simulation.md) adds causal
per-range scatter estimates, a weighted position/clock solver and local
covariance. Paired fresh-noise replays test both position error and nominal
ellipsoid coverage before introducing another publication policy.
