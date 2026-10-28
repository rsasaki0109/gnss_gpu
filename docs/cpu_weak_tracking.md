# Weak-signal following over continuous 240 ms records

The new 37 dB-Hz continuous-signal experiment follows all 32 replicas at each
carrier and reacquires all 32 outage replicas after return. This extends the
earlier isolated acquisition result to coarse state following over 240 ms;
it does not establish physical receiver lock or precise ranging.

The next [assisted range and position experiment](cpu_weak_positioning.md)
connects fresh states to explicit code-period resolution and a position/clock
solution, and quantifies the current integer-range precision limit.

`CpuWeakCaTracker` connects the calibrated multiple-seed weak acquisition API
to a state follower operating on contiguous, nonoverlapping 20 ms windows.
Every window runs the same complete nuisance cancellation and coherent search;
there is no state-dependent search narrowing or threshold adjustment. Two
consecutive consistent candidates establish a state. Carrier and wrapped code
delay then receive exponential updates. Code prediction uses the full elapsed
window and estimated carrier minus known IF as L1 Doppler.

An unused measurement produces a predicted `coasting` state. Two consecutive
unused windows discard that state as `lost`. Fresh acquisition then requires
two consistent candidates again. Pending candidates do not publish a state;
predictions never count as fresh measurements. Timestamp gaps require `reset()`.
Invalid IQ or timestamps leave pending/state history unchanged.

This is coarse blockwise state following, without PLL/FLL/DLL, bit decoding,
carrier-phase continuity, precise ranging or positioning. Window estimates refer
to the window's first row but become available only after its last row. With two
20 ms confirmations, the earliest initial acquisition is available after 40 ms.
The report includes these decision latencies rather than backdating estimates.

## Frozen protocol

`experiments/run_cpu_weak_tracking.py` renders each full 240 ms record with
absolute sample indices, continuous code/carrier evolution and independent
per-row white noise. The target navigation polarity alternates every 20 ms;
the random initial sample offset makes its reversal occur inside a row.
Nuisance PRNs 2–12 keep constant polarity and the earlier fixed Doppler model.
No independent 20 ms recordings are stitched together.

There are 32 replicas for each 37 dB-Hz target carrier (170/700 Hz) and condition
(continuous or target-only outage from 80 to 120 ms). Other satellites have
20 dB more power than nominal, as in the prior near-far model. Those four cases
use the same noise and initial parameters within a replica for paired controls.
Three additional 32-sequence target-absent controls contain white noise,
nominal other PRNs or stronger other PRNs. New signal/null seed domains are
separate from the earlier calibration and evaluation domains.

The existing gate in `output/cpu-multiseed-acquisition` remains frozen. Source
and calibration artifact hashes are checked at the start/end of the experiment.
This reuse preserves the complete per-window algorithm; it supplies **no
sequence-level false-alarm guarantee**, and no new sequence gate is fitted.
The continuous null sequences empirically test state publication and expose
temporal dependence. In particular, a previous 1% marginal null budget cannot
be reported as a 1% false-track bound, nor can independent-window probabilities
be multiplied without evidence.

All replicas remain in the totals, including failed initial acquisition.
Reacquisition requires acquisition before 80 ms and an actual loss during the
outage. A late first acquisition is not counted as recovery. Accuracy summaries
refer only to states updated by fresh measurements; coasting predictions are
reported separately. The report contains maximum carrier and circular code
errors so a state-publication count is not mistaken for accuracy.

## Results

The completed run contains **224 sequences and 2,688 window outcomes**.
Each row below has 32 independently seeded replicas. Fresh updates include
the acquisition decision but exclude pending confirmation and coasting.

| Carrier / target case | Initial acquisition | Fresh / signal-present windows | Outage loss | Reacquisition |
| --- | ---: | ---: | ---: | ---: |
| 170 Hz, continuous | 32/32 | 352/384 | — | — |
| 700 Hz, continuous | 32/32 | 352/384 | — | — |
| 170 Hz, 40 ms outage | 32/32 | 256/320 | 32/32 | 32/32 |
| 700 Hz, 40 ms outage | 32/32 | 256/320 | 32/32 | 32/32 |

Every initial acquisition is available at 40 ms. Every outage reacquisition
is available 40 ms after the signal returns, at relative time 160 ms. There
are no accepted or used measurements in the 64 absent windows per outage
case. Confirmation accounts for the unused signal-present windows: one initial
confirmation window, plus one after return in outage cases. No signal-present
window is silently removed from the denominator.

Across fresh states, the maximum carrier errors are 5 Hz at 170 Hz and 0 Hz at
700 Hz, reflecting the 25 Hz carrier grid. Maximum circular code errors against
the nominal initial phase evolved by code Doppler are 0.996 and 0.979 samples.
One sample at 2.046 MHz corresponds to about 146.5 m of code-range; these are
coarse states, not precise pseudorange measurements.

| Entirely target-absent stratum | Windows passing acquisition gate | Sequences publishing any state |
| --- | ---: | ---: |
| White noise | 1/384 | 0/32 |
| Nominal other PRNs | 1/384 | 0/32 |
| Other PRNs +20 dB | 1/384 | 0/32 |

Each observed gate passage is insufficient to confirm a state. Zero observed
false published sequences does not establish zero risk: the pointwise 95%
Wilson interval for each 0/32 sequence count is approximately 0–10.72%. Each
32/32 signal count has interval 89.28–100%. The sequence data are descriptive,
not a calibrated track-level false-alarm bound or a superiority comparison.

![Fresh updates and sequence-level null outcomes](../output/cpu-weak-tracking/weak_tracking.png)

## Use and reproduce

```python
from gnss_gpu.weak_acquisition_cpu import load_weak_ca_experiment
from gnss_gpu.weak_tracking_cpu import CpuWeakCaTracker

gate = load_weak_ca_experiment("output/cpu-multiseed-acquisition")
tracker = CpuWeakCaTracker(gate)
result = tracker.update(iq_20ms, start_sample=absolute_start_sample)
if result.measurement_used:
    print(result.carrier_hz, result.delay_samples)
```

`iq_20ms` contains 20 contiguous rows, each holding interleaved real I/Q for
one millisecond (4,092 real values at 2.046 MHz). Submit each next window with
an increment of 40,920 complex samples. The returned delay wraps modulo one
2,046-sample C/A period. Coasting states are predictions, available with
`measurement_used=False`.

```bash
source /workspace/.gnss-env/activate.sh
python experiments/run_cpu_weak_tracking.py
python experiments/audit_cpu_weak_tracking.py
python experiments/plot_cpu_weak_tracking.py
```

The run refuses a nonempty output directory. Use `--output` for a fresh run;
`--replicas` controls replication count. `output/cpu-weak-tracking` holds the
protocol, input IQ hashes, all window/state outcomes, per-sequence summaries,
aggregate report and PNG/PDF figure. An earlier partial development run is
retained in `output/cpu-weak-tracking-draft`; it was stopped to correct the
reacquisition denominator rule and supplies no completed performance result.

`audit.json` records verified identities for all 224 sequences, all 2,688 window
timestamps/outcomes, recomputed summaries, source/calibration hashes and exact
regenerated IQ/receiver traces for replica zero of all seven cases. This replay
is a regression check, not additional independent validation data.

43 relevant tests pass, including actual IQ integration, contiguous timestamps,
transactional failed updates, confirmation, wrapped prediction with known IF,
coasting/loss/reacquisition and keeping late initial acquisitions out of recovery
counts. The earlier acquisition/tracker implementations and frozen experiments
are preserved.

## Limits

The continuous scenes still assume a constant target carrier, a single reversal
per decision and the earlier finite nuisance model. Integer aperture delays,
point-sampled synthesis and heuristic consistency bounds remain approximations.
Neither a fresh update nor the status `tracking` establishes physical lock.
Only 32 independent sequences per null stratum are tested; the 12 decisions
inside each sequence are dependent and must not be treated as 384 independent
track trials. Pointwise sequence intervals are not simultaneous. There is no
claim of RF performance, real-time throughput, dynamic tracking, rain effects,
positioning accuracy or research novelty.
