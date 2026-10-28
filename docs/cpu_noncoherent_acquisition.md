# Noncoherent C/A acquisition and a matched-horizon experiment

A follow-up [larger independent null validation](cpu_noncoherent_calibration_validation.md)
uses 1,999 calibration and 1,999 evaluation records and compares old/new
thresholds on the same fresh noise stream. The original measurements below
are preserved.

A new CPU detector accumulates correlation power over multiple 1 ms periods
at a common carrier and integer circular code-delay bin. This is a conventional
noncoherent acquisition method, not a new research algorithm. It supplements
the existing receiver; it has not been integrated into its tracking states.

## Detector and API

For every frequency and delay, correlate each period against the centered
aperture C/A reference, square its complex magnitude, and add those powers
within a fixed window. Only then choose the strongest bin. Adding independent
per-period maxima would combine potentially different signals or noise peaks,
and is deliberately not used.

For K periods of N complex samples, the window statistic is:

```
N*K * sum_j |<carrier_wiped_period_j, shifted_reference>|^2
      / (reference_energy * sum_j period_energy_j)
```

The decision score is the maximum over all configured windows, frequencies
and integer delays. A single overall input scaling is removed before complex
construction to keep finite very large/small IQ well conditioned. Every window
is normalized by its own energy. All-zero IQ returns no candidate; zero-energy
windows cannot win against positive scores.

```python
from gnss_gpu.acquisition_cpu import CaSearch
from gnss_gpu.noncoherent_acquisition_cpu import (
    CaNoncoherentSearch, acquire_ca_noncoherent,
    calibrate_ca_noncoherent, detect_ca_noncoherent,
)

config = CaNoncoherentSearch(
    CaSearch(2_046_000., (-1000., -500., 0., 500., 1000.)),
    periods_per_window=4, windows_per_decision=5,
)
# Each noise_record has shape (20, 4092), interleaved real I/Q per 1 ms row.
noise_scores = [acquire_ca_noncoherent(record, config).statistic
                for record in independent_calibration_noise]
threshold = calibrate_ca_noncoherent(noise_scores, config)
detection = detect_ca_noncoherent(observation, threshold)
```

The threshold binds the period count, window count and search grid. Its reused
rank metadata's sampling frequency describes a 1 ms row, not the decision
duration. Changing any search/integration configuration requires fresh scores
and calibration. Rejected candidates remain available for audit.

Whole-period phase or navigation-polarity flips do not cancel the accumulated
powers. A transition **inside** a 1 ms period can still destroy its coherent
projection, as a counterexample test demonstrates. The caller must provide
chronological contiguous periods. There is no bit synchronization, fractional
delay fitting, carrier refinement, code-drift compensation or tracking loop.

## Matched 20 ms comparison

The experiment compares 20 separate 1 ms windows against five 4 ms accumulation
windows on the same 20 ms observation record. Both use this new integer-delay
detector, so the only integration change is K and the associated window count.
Neither is the earlier fractional-delay tracker; its 21 ms acquisition rates
must not be substituted as this experiment's baseline.

Each method is calibrated using the maximum score of its **entire 20 ms
record**, including all windows/carriers/delays. The same 199 independent
calibration records feed both methods, with separate scores and thresholds.
The nominal rank target is 1% per complete decision, rather than per window.
Held-out evaluation uses another 199 records with a disjoint seed domain.

All 144 signal scenarios regenerate the previous sensitivity inputs and must
match their full-IQ hashes before the first 20 ms are selected. This horizon
precedes the interruption. Neither truth nor bit signs are supplied to detection.
The matched detectors are evaluated without any tracking confirmation rule.

Offline, a declared coarse criterion requires detection, carrier error at most
250 Hz (half the grid spacing), and wrapped code lag within two samples of the
scene's initial phase lag. This is a coarse peak/plausibility check, not a
ranging, navigation or integrity bound. Over this horizon the maximum tested
code drift is about 0.018 sample; the test does not estimate an absolute delay
error or resolve the point-sampling/aperture-model mismatch.

Counts passing detection **and** this criterion out of 12 paired replicas:

| Nominal C/N0 | 170 Hz: 1 ms / 4 ms | 700 Hz: 1 ms / 4 ms |
| --- | --- | --- |
| 35 dB-Hz | 0 / 1 | 0 / 1 |
| 37 dB-Hz | 1 / 9 | 1 / 8 |
| 39 dB-Hz | 4 / 11 | 1 / 11 |
| 41 dB-Hz | 11 / 12 | 7 / 11 |
| 43 dB-Hz | 12 / 12 | 9 / 11 |
| 45 dB-Hz | 12 / 12 | 10 / 11 |

**The empirical false-alarm rates are not matched.** Held-out noise passed
1/199 decisions for 1 ms and 6/199 for 4 ms. The thresholds were 16.894306 and
23.139309. All seven noise passes were discordant between methods. These small
calibration/evaluation samples do not establish a gain at equal false-alarm
probability. The rank guarantee is marginal over random calibration sets; it
is not a guarantee for these particular frozen thresholds. The original result
is retained, rather than tuning a threshold to remove the observed noise passes.

![Exploratory noncoherent results and noise-pass caveat](../output/cpu-noncoherent-sensitivity/noncoherent_sensitivity.png)

Pointwise Wilson intervals use independent replica events within each condition.
The conditions share noise/phase draws within replicas and are not independent
comparisons. Twelve replicas do not establish a sensitivity specification.
More calibration records and a fresh, larger held-out stream are needed before
claiming performance at matched false-alarm rates.

## Checks and reproduction

```bash
PYTHONPATH=python:. python experiments/run_cpu_noncoherent_sensitivity.py
PYTHONPATH=python:. python experiments/plot_cpu_noncoherent_sensitivity.py
```

`output/cpu-noncoherent-sensitivity/` contains every signal and held-out-noise
decision, all calibration scores, complete search/threshold metadata, input and
source hashes, aggregated counts, and PNG/PDF figures. Existing recordings and
the original receiver were preserved.

22 new detector/experiment tests and the existing 224 relevant tests pass.
Checks include an independent direct circular-inner-product reference, rejection
of summing different period peaks, phase/polarity invariance, within-period bit
failure, unknown carrier/delay recovery, extreme scales, input preservation,
zero windows, invalid shapes/configurations, strict gate ties, metadata checks,
3/8/20 samples per chip including cache bypass, wrapped phase criteria and
failed-replica denominators.

The signal experiment uses navigation edges aligned to period boundaries and
does not validate arbitrary bit timing, dynamics or multipath. No real-time
speed, sustained tracking improvement, RF accuracy, rain effect, PVT result or
research novelty is claimed.
