# Sampling, bandwidth and two-path information

The preceding [weak multipath experiment](cpu_weak_multipath.md) found many
ambiguous fits at two samples/chip. This study adds an **ideal bandlimited signal
model and local Cramér–Rao bound (CRLB)** to distinguish sampling density from
physical bandwidth. It does not run or improve the acquisition/positioning
receiver. Existing frozen receiver thresholds and results are unchanged.

## Model and interpretation

PRN 1 is a periodic rectangular C/A waveform. Its exact Fourier-series
coefficient at harmonic k is the 1023-chip DFT coefficient divided by 1023,
multiplied by `sinc(k/1023) exp(-i*pi*k/1023)`. A unity-gain ideal filter retains
harmonics with `|f| <= B`; **B is the one-sided baseband cutoff**, so the full
two-sided bandwidth is 2B. Code power is not normalized after filtering.
Direct-path C/N0 is defined before the filter.

The observation is an ideal periodic spectral Gaussian model: each retained
complex Fourier coefficient has independent proper complex noise of variance
`N0/T`, with T = 20 ms. This assumes coherent repetitions with known carrier and
navigation polarity. The corresponding filtered time samples have correlated
noise. It is not a transient simulation of a finite physical filter; finite
record boundary effects, oscillator error, quantization and colored interference
are excluded. Sampling must satisfy `fs > 2B`. Increasing fs at fixed B preserves
the same information in this model, rather than adding independent measurements.

Both delays and both complex path amplitudes are unknown (six real parameters).
For template matrix A and amplitude-weighted delay derivative D, amplitude
nuisance parameters are eliminated by orthogonal projection:

`J_delay = 2 * (C/N0) * T * Re[D^H (I - P_A) D]`.

The diagonal of its inverse gives the local delay variance lower bound, converted
from chips to metres. Known simulation parameters locate the point where this
theoretical information is evaluated; they are not inputs to a receiver estimator.
Coincident paths produce a singular amplitude matrix and **no finite bound**.
Near-singular delay information likewise returns unavailable instead of using a
pseudoinverse. The relative eigenvalue cutoff of 1e-12 is numerical, not a
physical resolution threshold.

CRLB concerns locally unbiased estimation under this exact model. It does not
predict global path selection, detection, availability, bias or achieved RMSE.
Small bounds do not guarantee successful separation; large bounds do not forbid
a biased or prior-constrained estimator from having lower error at a given point.
The bandwidth model differs from the earlier centered-aperture model, so their
numbers are not a paired receiver improvement comparison.

## Results

The frozen grid contains 1,872 evaluations: 13 valid sampling/bandwidth pairs,
two C/N0 levels (37/55 dB-Hz), three voltage ratios (0.3/0.8/1.5), three relative
phases (0/pi/2/pi), and eight excess delays (0/0.01/0.02/0.05/0.1/0.2/0.5/1 chip).
Sampling rates are 2/4/8/16 samples/chip; cutoff B is 0.5/0.9/1.8/3.6 times the
chip rate. Three aliasing combinations are explicitly skipped. The 234 coincident
path evaluations remain in the output as singular. These are deterministic
parameter-grid evaluations, not independent Monte Carlo trials.

At fixed bandwidth and path conditions, every valid sampling rate gives the
same bound. An independent time-domain six-parameter Fisher calculation also
matches the spectral result at all four sampling rates.

For 37 dB-Hz and echo/direct voltage 0.8, the direct-range standard-deviation
lower bounds are:

| One-sided cutoff B | 0.1-chip gap, phase 0 | 0.1-chip gap, phase pi/2 | 0.5-chip gap, phase 0 |
| --- | ---: | ---: | ---: |
| 0.512 MHz | 3345.97 m | 178.23 m | 136.20 m |
| 0.921 MHz | 880.32 m | 80.84 m | 38.01 m |
| 1.841 MHz | 109.75 m | 23.81 m | 8.05 m |
| 3.683 MHz | 21.25 m | 9.24 m | 5.67 m |

Higher bandwidth adds information while power/noise conventions stay fixed.
Close, same-phase paths remain difficult. At 55 dB-Hz these local bounds scale
down by `10**(18/20)`; this scaling is an ideal Fisher result, not evidence that
the existing receiver achieves it. Phase pi gives the same marginal bounds as
phase zero here, although actual estimator behavior can differ.

![Local lower bounds versus excess delay](../output/cpu-multipath-information/comparison.png)

The next receiver experiment should use bandlimited synthesis **and matching
templates/noise treatment**, then recalibrate the complete acquisition pipeline
before a paired signal evaluation. Feeding these signals to the old aperture
fitter would confound bandwidth with model mismatch. No research novelty or
real forest/rain performance is established by this theoretical study alone.

## Reproduction and validation

```bash
source /workspace/.gnss-env/activate.sh
pytest -q tests/test_multipath_information_cpu.py
python experiments/run_cpu_multipath_information.py --output output/cpu-multipath-information-new
```

The destination must be empty. The experiment saves its protocol before the
grid evaluation, source hashes, all finite and singular records, a summary, and
PNG/PDF plots. Fifteen tests pass: analytic Fourier coefficients versus independent
rectangular-wave quadrature, finite-difference delay derivatives, four sampled
time-domain full Fisher inversions, C/N0/duration scaling, bandwidth/sampling
controls, singular paths and invalid inputs. The saved source hashes are verified
at the end of the run. No old experimental outputs are overwritten.
