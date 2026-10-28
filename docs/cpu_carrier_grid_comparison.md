# Recalibrated wider carrier search

The next [weak-signal sensitivity experiment](cpu_receiver_sensitivity.md)
keeps the wide threshold frozen and evaluates 144 new scenarios with
replica-level acquisition rates and pointwise uncertainty intervals.

The earlier [stress matrix](cpu_receiver_stress.md) exposed a true 700 Hz
carrier estimated at the +500 Hz endpoint. This experiment compares the fixed
three-bin grid (−500/0/+500 Hz) with a fixed five-bin grid
(−1000/−500/0/+500/+1000 Hz). The wide grid applies to every scenario; truth
and boundary diagnostics do not select a search at runtime.

Each complete grid receives its own rank threshold from 199 noise-only blocks.
The two calibrations share the same noise draws for a paired comparison, but
use their own whole-grid maximum scores. The target is 1% per-block marginal
false alarm under the calibration assumptions, not track-level integrity or
a guaranteed error rate for one saved threshold. Wider search is never tested
against the original three-bin threshold.

Both grids are tested on the same additional 199 independent noise-only blocks.
Signal testing regenerates all 54 original scenarios and requires their IQ
hashes to match the saved recording. It retains weak-signal failures, short
coasting, lost/reacquired states, and all per-scenario denominators.

Errors are reported both conditionally on each receiver's fresh measurements
and on the exact intersection of signal-present blocks used by both methods.
The latter comparison prevents a lower conditional error caused solely by
changing which observations are available. It is still conditional on selection
and does not describe errors on missed observations. Missing errors remain null.

## Observed results

All 54 regenerated IQ arrays matched the original hashes. Narrow-grid
calibration scores, threshold and held-out null counts were reproduced exactly.
The narrow threshold was 15.429738 and the separately calibrated wide threshold
was 15.681290, using the same 199 calibration noise draws.

Pooling paired replicas and both outage durations, with each common-used block
weighted equally:

| Nominal C/N0 | True Doppler | Common fresh blocks | Narrow carrier MAE | Wide carrier MAE |
| --- | ---: | ---: | ---: | ---: |
| 45 dB-Hz | 0 Hz | 345 | 23.79 Hz | 23.79 Hz |
| 45 dB-Hz | 170 Hz | 343 | 26.83 Hz | 26.83 Hz |
| 45 dB-Hz | 700 Hz | 335 | 200.00 Hz | 22.82 Hz |
| 55 dB-Hz | 0 Hz | 345 | 7.38 Hz | 7.38 Hz |
| 55 dB-Hz | 170 Hz | 345 | 8.01 Hz | 8.01 Hz |
| 55 dB-Hz | 700 Hz | 345 | 200.00 Hz | 7.03 Hz |

All 35 dB-Hz scenarios still had no fresh measurements: no error estimate is
available there. At 45 dB-Hz and 700 Hz, total fresh signal-present measurements
increased from 337 to 343 out of 354 blocks, but only 335 were common to both
methods. At 55 dB-Hz and 700 Hz, both used the same 345/354 blocks. None of the
wide-grid fresh estimates was flagged at its ±1000 Hz boundary in this matrix.
This does not establish correctness for arbitrary frequencies or waveforms.

![Carrier errors on common fresh blocks](../output/cpu-carrier-grid-comparison/paired_carrier_error.png)

On the 199 held-out pure-noise blocks, narrow detection passed 0 and wide
detection passed 1; neither used any fresh tracking measurement. This is a
small observed count, not evidence for a generally lower or higher false-alarm
probability. No signal-absent outage block was detected or used in either
matrix. The pure-noise stream is separate from the paired outage draws.

Median receiver-only time on the fixed 64 ms timing scenario was 0.516 s for
the narrow grid and 0.603 s for the wide grid: approximately **16.7% more time**.
These are seven warm repetitions of one scenario, not a general overhead bound.
The wider fixed grid resolves the demonstrated boundary limitation at additional
search cost; it does not improve the weak-signal failures or establish real-time
processing.

## Reproduction

```bash
PYTHONPATH=python:. python experiments/compare_cpu_carrier_grids.py
PYTHONPATH=python:. python experiments/plot_cpu_carrier_grids.py
```

Inputs under `output/cpu-receiver-stress/` remain untouched. Outputs under
`output/cpu-carrier-grid-comparison/` contain the wide receiver's complete
`wide_traces.jsonl`, per-scenario `summary.json` / `summary.csv`, and `report.json`
with both thresholds, all calibration scores, held-out null outcomes, hashes
and raw timings. Common-used observation indices are retained in JSON.
The plot is exported as PNG and PDF.

Receiver-only timing uses one predeclared 64 ms scenario: 55 dB-Hz, 700 Hz,
8 ms outage, replica 0. Both paths are warmed, then seven repetitions alternate
grid order. Result equality to the recorded narrow trace or initial wide trace
is checked after each timer ends. No signal generation, calibration, comparison
or serialization is inside these timers. Tests finish before timing begins.

The 216 relevant tests pass. New checks verify separate complete-grid rank
thresholds, that additional bins cannot lower a null maximum, exclusion of
outage/unavailable blocks from paired errors, and null errors for an empty
intersection of used observations.

This remains an exploratory three-replica synthetic comparison. Conditions
share noise within replicas, the point-sampling/aperture-model mismatch remains,
and there is no navigation, PVT, rain, RF or integrity validation. The wide grid
still has finite boundaries; it is not a universal carrier solution.
