# Position publication checks during weak-signal recovery

`position_quality_cpu.PositionPublicationGate` is a receiver-observable filter
on candidate position/clock solutions. It addresses large errors seen during
the [wet-antenna stress replay](wet_antenna_simulation.md), while explicitly
measuring the additional missing positions and publication delay.
This is an engineering quality policy, **not** certified integrity monitoring,
a protection level or a bound on false-fix probability.

## Inputs and fixed policy

The gate receives only source coordinates, receiver pseudoranges and timestamp.
It solves the same equal-weight position/clock problem as the raw baseline.
Weather flags, stage labels, geometric visibility, true position and simulated
biases are not inputs. A test changes truth/visibility/stage fields in saved
receiver records and verifies that publication decisions do not change.

The concrete [default policy](position_quality_policy.json) is:

| Check | Setting |
| --- | ---: |
| Fresh pseudoranges | at least 6 |
| PDOP | at most 6 |
| Postfit residual RMS | at most 15 m |
| Position jump allowance | 20 m + 15 m/s × elapsed time |
| Published-anchor maximum age | 2 s |
| Confirmation after initialization or any rejection | 5 good consecutive candidates |

The defaults were fixed before generating new evaluation seeds. They are
engineering thresholds, not a statistical fit or a searched optimum.
Receiver-only development dry geometry has PDOP approximately 1.64 in open
space and 4.41 near the wall; inspected postfit RMS values were approximately
1.7 and 2.7 m. No true-position error was used inside the gate, and no policy
retuning is performed after holdout evaluation.

Four pseudoranges solve the four unknowns x/y/z/clock without redundant
residuals. A wrong four-satellite solution can therefore have nearly zero
postfit residual. Six observations supply two residual degrees of freedom.
That helps but does not establish integrity against coherent or multiple
faults. DOP is computed from the equal-weight design matrix at the candidate
using its singular values. PDOP is dimensionless and rotation invariant;
HDOP/VDOP diagnostics assume x/y horizontal and z up. Postfit sigma is missing
when residual degrees of freedom are zero, not reported as perfect precision.

Every candidate first passes satellite-count, rank/solution, PDOP and residual
checks, then position continuity. Only the last **published** position is the
anchor. A rejected candidate cannot move it. The allowed jump is a noise margin
plus a configured speed envelope over elapsed time; it is not a calibrated
motion likelihood or a literal instantaneous speed test. It can withhold
legitimate motion outside the assumed envelope, especially after gaps.

Any rejected/missing epoch breaks confirmation. Five consecutive valid,
mutually consistent candidates are needed to initialize or resume publication.
After that, good epochs publish immediately until another rejection.
The gate returns no position/clock for rejected or pending epochs: it never
relabels a previous fix or predicted position as a new measurement. Candidate
solutions and diagnostics remain available separately for audit.

Consecutive means consecutive supplied epochs. This replay supplies every
millisecond, including empty observations. The API requires strictly increasing
time but does not impose a fixed cadence; applications that skip missing epochs
must report an empty update or reset confirmation rather than silently treating
separated samples as an uninterrupted observation sequence.

The anchor expires after two seconds without publication, requiring a new
bootstrap. This avoids permanent denial after a large real displacement, but
also removes the old absolute reference. A persistent wrong but coherent
range field, slow drift, or an initially wrong consistent solution can still
pass. A test explicitly demonstrates such a 200 m wrong bootstrap. Small
direction-dependent antenna biases can also pass; this gate does not correct
their physical cause or promise absolute accuracy.

```python
from gnss_gpu.position_quality_cpu import PositionPublicationGate

gate = PositionPublicationGate()
decision = gate.update(satellite_positions_m, measured_pseudoranges_m, time_s=t)
if decision.accepted:
    print(decision.position_m, decision.clock_bias_m)
else:
    print(decision.reason)  # position_m and clock_bias_m are None
```

The same coarse initial position (-25,5,0) is used for every raw candidate and
the gated candidate. Gate history does not alter the channel receiver or its
pseudoranges, so this compares publication decisions rather than a different
measurement estimator. Invalid input/time is rejected before state advances.

## Development replay and fresh-noise holdout

First generate the previous wet-antenna replay if its outputs are absent:

```bash
PYTHONPATH=python:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python experiments/run_cpu_wet_antenna.py --workers 4
PYTHONPATH=python:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python experiments/run_cpu_position_quality.py --workers 4
PYTHONPATH=python:. python experiments/plot_cpu_position_quality_availability.py
PYTHONPATH=python:. python -m pytest tests/test_position_quality_cpu.py tests/test_cpu_position_quality.py -q
```

The new script writes the exact policy bytes and SHA-256 to
`output/cpu-position-quality/frozen_policy.json` **before** development replay
or held-out generation. It replays all prior receiver-channel observations from
seeds 24683/24684 as development data. The gate sees only channels marked as
fresh measurements, their PRNs and code delays. It reconstructs raw candidates
and checks that availability and positions agree with the original receiver
output within 1 micrometre before scoring quality decisions.

Held-out IQ uses two new, disjoint seeds **24685/24686**, open/wall geometries
and four predetermined cases: dry, 6 dB loss, 30 dB loss stress and combined
stress. The same source geometry, antenna profile, C/N0, channel receiver and
frozen per-PRN acquisition thresholds apply. There is no threshold scan or
retuning on holdout. New channels and their raw positions are saved so the
publication policy can be audited separately from IQ generation.

The gate is evaluated on exactly the same raw candidate stream as its baseline.
Only after decisions are made is known synthetic position used to score errors.
Every raw candidate, rejection/pending reason, PDOP, residual, confirmation
count and anchor jump/age remains in the output. An unavailable or withheld
position remains missing; its error is not replaced by zero. All-input counts
of unavailable positions and >100 m published errors accompany conditional
P95/max errors. The 100 m evaluation threshold is not a policy parameter.

Initial publication latency is measured at the end of the first available
block. For both modes, the same retrospective five-consecutive-position metric
from the original replay reports recovery after drying begins. This is
**additional to** the gate's own five-candidate publication confirmation; the
two latencies must not be confused. No wet-plateau outage and no recovery
remain distinct reasons with missing latency rather than zero.

This is one static geometry per environment with two new noise seeds, not
independent city drives. Counterfactual scenarios share standardized noise and
serial receiver states are correlated. No binomial/independent-frame confidence
claim or real rainy-weather integrity conclusion follows from these counts.
The original compressed 0.5 s wetting/drying schedule and uncalibrated antenna
coefficients remain sensitivity/stress inputs.

`output/cpu-position-quality/` stores development and heldout raw/gated
position records, episode/stage summaries, all rejection reasons, fresh heldout
channels/raw positions, frozen policy, input/source hashes and a figure showing
both errors and withheld-position gaps. Missing output is a deliberate cost
of the publication policy, not evidence of zero error.

## Frozen-policy results, including failure to resume

Raw replay agrees exactly with the saved receiver: maximum candidate-position
difference is **0.0 m**. No policy setting changed after inspecting holdout.
The counts below include every input block; paired scenarios and serial states
are not independent failure trials:

| Dataset | Mode | Input blocks | Published | Withheld/unavailable | Published errors >100 m |
| --- | --- | ---: | ---: | ---: | ---: |
| Development | Raw | 14,000 | 12,094 | 1,906 | 18 |
| Development | Quality gate | 14,000 | 11,439 | 2,561 | 0 |
| Fresh-noise holdout | Raw | 8,000 | 6,102 | 1,898 | 12 |
| Fresh-noise holdout | Quality gate | 8,000 | 5,103 | 2,897 | 0 |

The observed >100 m outputs disappear, but holdout publication availability
falls from **76.3% to 63.8%**. This is not a free improvement, nor proof of
zero false fixes. For the 30 dB stress case, the common retrospective recovery
metric shows the delay cost:

| Seed / geometry | Raw published / 500 | Gated / 500 | Raw recovery after drying starts [ms] | Gated recovery [ms] |
| --- | ---: | ---: | ---: | ---: |
| 24685 / open | 265 | 250 | 113 | 123 |
| 24686 / open | 264 | 243 | 113 | 129 |
| 24685 / wall | 264 | 167 | 113 | 135 |
| 24686 / wall | 262 | 198 | 117 | 160 |

Combined stress has the same publication counts and recovery timing in these
seeds. Initial raw publication is at 2 ms; gated publication is at 6–7 ms.
Holdout gated rejection/pending counts are 1,944 for too few satellites,
655 for anchor jump, 292 confirming, and six for residual RMS. No PDOP rejection
occurs in this particular holdout: redundancy and continuity dominate here.
Of the 999 extra withheld raw candidates, 655 are rejected for anchor jump,
292 are pending confirmation, 46 lack the new six-satellite minimum and six
fail the residual check.

**A serious availability failure remains.** For seed 24686 near the wall with
only 6 dB loss, raw publishes 499/500 positions but the gate publishes only
116/500. In the final fully dry 100 ms, raw publishes 100/100 and the gate
publishes **0/100**. A previously published noisy anchor can disagree with
later otherwise admissible candidates; its two-second expiry exceeds this
0.5-second replay, so the continuity policy can prevent resumption. The early
five-position recovery metric does not mean later continuity is restored.
This is retained rather than hidden by a threshold change or a new definition
of recovery.

Consequently this remains an **opt-in comparison prototype**, not a replacement
for the existing position output or a production-ready recovery policy. Anchor
uncertainty/expiry and independent reinitialization require further work. The
present holdout has now been inspected; changing those policies requires new
held-out realizations rather than reusing these seeds as unseen validation.

`position_quality.png` compares raw/gated error timelines with gaps for
withheld fixes. `position_quality_availability.png` shows all held-out episode
counts and the explicitly post-evaluation dry-tail diagnostic. The latter is
rendered from saved data, has separate input/renderer hashes, and does not
recompute or tune the gate. Files retain 44,000 raw/gated epoch records,
88 episode summaries, 440 stage summaries, all rejection reasons, 96,000 new
held-out channel observations, 8,000 held-out raw positions and the frozen
policy/source/input metadata. Published large errors and missing outputs are
both first-class results.

The [reference-recovery follow-up](position_recovery_simulation.md) treats all
these inspected seeds as development data, preserves this baseline and tests
median references plus sustained candidate clusters on new noise seeds.
