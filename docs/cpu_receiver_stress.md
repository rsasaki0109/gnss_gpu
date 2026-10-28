# Exploratory receiver stress matrix

The subsequent [search-boundary diagnostic](cpu_carrier_support.md) identifies
all 682 used 700 Hz measurements as boundary-limited in this saved matrix,
without changing the receiver's decisions or threshold.

The optimized receiver is evaluated across conditions, rather than only on the
single frozen performance recording. The matrix varies nominal primary C/N0
(35, 45, 55 dB-Hz), primary Doppler (0, 170, 700 Hz) and all-signal outage
duration (2, 8 ms), with three independent noise/phase replicas: **54 scenarios**,
each 64 contiguous 1 ms blocks. This small matrix is exploratory.

The same three-bin carrier grid (−500/0/+500 Hz), two-block confirmation and
three-unused-block loss rule apply throughout. A complete-grid threshold is
calibrated once from 199 separate white-noise blocks at a nominal 1% per-block
false-alarm target. Another 199 disjoint noise-only blocks are held out.
Neither labels nor truth enter the receiver.

Noise power is fixed. Primary amplitude is changed to obtain the nominal C/N0;
the other 11 PRNs retain fixed amplitudes outside the outage. C/N0 here follows
the generator's per-I/Q white-noise convention and is not a receiver estimate.
The outage mutes all 12 PRNs but leaves noise running. It models an idealized
signal interruption, not rainfall or a physically modeled obstruction.

Each replica has independent noise and initial carrier phases. Conditions
within a replica share these draws for paired comparisons; the 54 conditions
are not 54 independent trials. Absolute sample starts, code Doppler and global
navigation-bit timing continue across blocks.

## Interpreting the measurements

The completed matrix produced these exploratory observations:

| Condition | Observed behavior across the three replicas |
| --- | --- |
| 35 dB-Hz, all tested offsets/outages | No fresh measurement or pre-outage acquisition |
| 45/55 dB-Hz, 2 ms outage | All acquired before interruption; no loss event during interruption |
| 45/55 dB-Hz, 8 ms outage | All lost state during interruption and reacquired at the second post-return block end |
| 700 Hz, 45/55 dB-Hz | Fresh-measurement fraction 0.940–0.984, but conditional carrier MAE approximately 200 Hz |

The 700 Hz result exposes a failure mode: the constrained estimate reaches the
+500 Hz grid endpoint while the tracker continues using its measurements.
Internal tracking state and high update fraction do not establish carrier
correctness. A next improvement should expose search-boundary ambiguity or
evaluate a broader search, with calibration repeated for that complete search.
Simply widening a grid while retaining the old threshold would invalidate the
whole-search calibration.

Across all scenarios, no absent-signal block was accepted or used. The separate
held-out pure-noise stream also had 0/199 accepted blocks and no fresh updates.
Zero observed events in this small sample does not establish zero false-alarm
probability or validate a rare-event bound for the selected threshold.

All 54 batch/individual IQ comparisons were exact. With the archived baseline
enabled, every field of all **3,456 receiver block results** matched the previous
receiver implementation exactly. These comparisons test the optimization across
conditions; they do not make either receiver physically correct.

![Fresh-measurement fraction across the exploratory matrix](../output/cpu-receiver-stress/availability.png)

Fresh-measurement fraction counts `measurement_used` over every signal-present
block, including startup. It measures the receiver's internal acceptance and
update behavior, not correct ranging or navigation availability. Carrier mean
absolute error is reported only for fresh measurements and is therefore
conditioned on selection. A high fresh-measurement fraction can coexist with
a wrong carrier.

Initial acquisition is counted only before the outage. First fresh measurement
after return is eligible only if the receiver had used a measurement before
the outage. Reacquisition is separately eligible only if a loss event occurred
during the outage. Ineligible or unobserved events remain null, rather than
being replaced with zero or dropped to create a favorable mean. All latencies
are observation-block end times in milliseconds, not wall-clock processing
latencies. The supplied summaries retain per-replica denominators and outcomes.

The 700 Hz condition lies outside the fixed search grid. Refinement is bounded
by that grid and cannot estimate the true frequency. This deliberately tests
the distinction between internal tracking state and frequency correctness.

## Reproduction and audit

```bash
PYTHONPATH=python:. python experiments/run_cpu_receiver_stress.py \
  --baseline-dir output/cpu-receiver-preparation-speed
```

The optional baseline directory contains the three complete receiver modules
saved before observation-preparation reuse. With this flag, every scenario
compares every receiver result field exactly against that baseline. Batch and
individual waveform outputs are also compared bit for bit for every scenario.
Without the flag, baseline equality is reported as null, not asserted.

Outputs under `output/cpu-receiver-stress/`:

* `metadata.json`: fixed threshold, all calibration scores, disjoint seed
  domains, source and per-scenario IQ hashes, held-out null counts and caveats.
* `summary.json` / `summary.csv`: all 54 per-replica outcomes, retaining failed
  acquisitions and missing/ineligible recovery times.
* `traces.jsonl`: every receiver result field for every signal scenario.
* `availability.png`: mean fresh-measurement fraction for each condition,
  averaged over the three paired replicas.

192 relevant tests pass, including new checks that short coasting is not
reacquisition, post-outage first acquisition is not recovery, loss/recovery
timing uses decision-block ends, waveform parameters remain continuous, and
calibration/null/signal seed domains are disjoint.

Synthesis still uses point-sampled rectangular C/A while the delay fitter
assumes a centered sampling aperture. There is no absolute delay-accuracy
claim, PLL/FLL/DLL, navigation decoding, PVT or RF validation. Per-block noise
calibration is not a track-level false-alarm or integrity guarantee. Three
replicas cannot establish population performance or rare-event probabilities.
