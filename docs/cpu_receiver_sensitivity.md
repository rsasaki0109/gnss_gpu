# Exploratory 1 ms receiver sensitivity

The next [noncoherent detector experiment](cpu_noncoherent_acquisition.md)
implements power accumulation and compares 1/4 ms windows over equal 20 ms
records. Its integer-delay detectors and decision-level gates are separate
from the fractional-delay tracker evaluated here.

The wider search corrected the earlier 700 Hz boundary failure, but 35 dB-Hz
signals remained unavailable. This experiment maps the transition over nominal
35/37/39/41/43/45 dB-Hz, at 170 and 700 Hz, with **12 replicas per condition**:
144 scenarios and 9,216 one-millisecond receiver blocks.

The five-bin ±1000 Hz grid and threshold are loaded from the previous
[recalibrated grid comparison](cpu_carrier_grid_comparison.md). They are frozen
before the new observations are evaluated. No threshold is tuned to the results.
Replicas 100–111 use signal/noise/phase draws distinct from the earlier replicas
0–2. Conditions within one replica share draws for paired comparisons. Each
scenario starts with a fresh tracker; state never carries between conditions.

Every scenario is 64 ms, with an 8 ms all-signal interruption starting at 21 ms.
The acquisition event is therefore defined as an internal acquired state reached
by the end of the first 21 signal-present observation blocks. Failure to acquire
by this deadline is retained as failure for this event, even if a later block
is acquired. It does not mean acquisition would fail for an unlimited duration.

## Outcomes and uncertainty

The completed matrix produced these counts out of 12 replicas per condition:

| Nominal C/N0 | 170 Hz: acquired / acquired with carrier criterion | 700 Hz: acquired / acquired with carrier criterion | Mean fresh-update fraction, 170 / 700 Hz |
| --- | --- | --- | --- |
| 35 dB-Hz | 0 / 0 | 0 / 0 | 0.000 / 0.000 |
| 37 dB-Hz | 0 / 0 | 0 / 0 | 0.000 / 0.000 |
| 39 dB-Hz | 1 / 0 | 0 / 0 | 0.004 / 0.003 |
| 41 dB-Hz | 9 / 5 | 8 / 5 | 0.112 / 0.089 |
| 43 dB-Hz | 12 / 7 | 12 / 8 | 0.719 / 0.677 |
| 45 dB-Hz | 12 / 9 | 12 / 10 | 0.951 / 0.954 |

For example, `9 / 5` means 9/12 reached internal acquisition by 21 ms, but only
5/12 both acquired and satisfied the first-acquisition carrier criterion.
Fresh-update fractions include all 56 signal-present blocks per scenario,
including startup and post-outage reacquisition. They do not measure navigation
availability or prove that the updates are correct.

At 41 dB-Hz, an initial acquisition event was often followed by few usable
updates: only 9–11% of signal-present blocks on average. Thus an acquisition
event alone overstates sustained tracking performance. At 43/45 dB-Hz all
replicas acquired by the deadline, but not all initial carrier estimates met
the declared 100 Hz criterion. The criterion does not turn those misses into
proven false locks; it only records this carrier-error outcome.

The new held-out noise stream had 1/199 detection passages and 0 fresh tracking
updates. Zero noise updates does not establish a track-level false-alarm bound.
All 144 signal traces (9,216 blocks) and the full noise trace are saved.

![Exploratory event rates and pointwise intervals](../output/cpu-receiver-sensitivity/sensitivity.png)

Two replica-level events are reported separately:

* Internal acquisition before interruption, with no accuracy criterion.
* That same acquisition, with absolute carrier error at the **first** acquisition
  at most 100 Hz. True carrier is used only for offline scoring.

The 100 Hz criterion is declared for this experiment. It is a carrier-only
plausibility check, not a code-delay, navigation or integrity validation. A later
improvement cannot rescue an initially incorrect acquisition for this event.

Pointwise 95% Wilson intervals use 12 independent replica outcomes for each
condition. They never treat correlated tracking blocks as independent trials.
Conditions share draws within replicas, so the intervals are not independent
or simultaneous across all conditions. Even 0/12 successes has an upper interval
near 0.2425, and 12/12 has a lower interval near 0.7575. This small experiment
cannot establish a precise acquisition threshold or rare-event probability.

The per-scenario table also retains fresh update counts, status counts, failed
acquisitions, missing recovery times and conditional carrier errors. Median
initial-acquisition latency uses only successful pre-outage acquisitions and is
explicitly labeled conditional; failures remain in the event denominators.

A separate 199-block noise-only stream uses a new seed domain. Its complete
trace is saved. The frozen detection gate applies without truth or noise labels.

## Reproduction and checks

```bash
PYTHONPATH=python:. python experiments/run_cpu_receiver_sensitivity.py
PYTHONPATH=python:. python experiments/plot_cpu_receiver_sensitivity.py
```

Outputs under `output/cpu-receiver-sensitivity/` include all receiver results in
`traces.jsonl`, `null_trace.json`, per-replica `summary.json` / `summary.csv`,
condition-level `aggregate.json`, and PNG/PDF figures. `metadata.json` records
the frozen threshold, threshold-report hash, source/IQ hashes, configuration,
replica offset, null seed domain and observed null counts. Earlier recordings
are not overwritten. The existing scene seed function and source are recorded
for deterministic regeneration.

224 relevant tests pass. New checks cover Wilson endpoints and invalid counts,
incorrect or post-deadline acquisitions, failed-replica denominators, conditional
latencies, preservation of frozen threshold metadata and rejection of a changed
search grid.

Nominal C/N0 follows the simulator's white-noise convention. Other PRNs retain
fixed power outside the all-signal outage. Point-sampled synthesis versus the
aperture delay fitter remains a shared limitation. This is a baseline for a
future longer-integration comparison, not a demonstrated integration gain,
RF sensitivity specification, rain result or positioning validation.
