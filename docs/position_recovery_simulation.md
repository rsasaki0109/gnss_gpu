# Position reference recovery after weak-signal outages

This opt-in extension addresses the noisy-reference failure in the
[previous publication policy](position_quality_simulation.md). It changes
publication decisions on the same raw position candidates; the RF simulator,
channel measurements and equal-weight position/clock solver are unchanged.
It is an engineering stability policy, not an integrity guarantee.

## Fixed reference and recovery rules

The [frozen settings](position_recovery_policy.json) retain every ordinary
check: six observations, PDOP <=6, residual RMS <=15 m, jump allowance
20 m + 15 m/s × reference age, two-second reference expiry and five-candidate
confirmation after any rejection. Calling `PositionPublicationGate()` without
a recovery configuration retains the previous policy.

Two optional changes are evaluated together:

1. Use the coordinatewise median of the latest 15 published positions as the
   continuity reference, reducing sensitivity to one noisy accepted solution.
2. When the ordinary jump check blocks a candidate, collect a separate
   stronger-quality recovery window. Each candidate needs at least seven
   observations, PDOP <=5 and residual RMS <=5 m. The 25-candidate window must
   span at least 24 ms, with no inter-candidate gap exceeding 5 ms. Adjacent
   candidates must also satisfy the ordinary jump/speed allowance.

The cluster center is its coordinatewise median. Its 90th-percentile distance
and current candidate distance must both be <=10 m + 15 m/s × window span.
A poor-quality epoch, excessive gap or interrupted jump rejection clears the
window. A qualifying cluster replaces the reference and seeds its 15-position
history from the most recent cluster candidates. Publication then still needs
the ordinary five-candidate confirmation, counting the qualifying epoch first.
Replacing a reference alone does not produce a position.

Published output is always the **current raw candidate**, including its clock
estimate. The median is a continuity reference, not a smoothed reported fix.
The input remains only satellite coordinates, measured pseudoranges and time;
true position, weather stage and visibility are used only for scoring afterward.
Tests verify that changing those extra receiver-record fields cannot change
publication decisions. Missing/rejected fixes retain missing errors.

```python
from gnss_gpu.position_quality_cpu import PositionPublicationGate, PositionRecoveryConfig

gate = PositionPublicationGate(recovery_config=PositionRecoveryConfig())
decision = gate.update(satellite_positions_m, measured_pseudoranges_m, time_s=t)
if decision.accepted:
    print(decision.position_m, decision.clock_bias_m)
else:
    print(decision.reason, decision.recovery_count, decision.reanchored)
```

At 1 ms cadence the recovery reference can be replaced on the 25th qualifying
candidate, with publication no earlier than the 29th. This is serial stability,
not 25 independent pieces of statistical evidence. Applications must supply
missing epochs or reset as described in the previous guide. The published
median history counts samples rather than elapsed time and can retain older
positions across gaps until reference expiry/recovery; moving receivers and
irregular cadence need separate validation.

## Paired replay and unseen noise

Previously inspected seeds 24683–24686 are all development data here: 44
episodes and 22,000 input epochs. Their original saved files are retained.
The new script saves exact policy bytes before any replay or generation, then
generates fresh IQ for seeds **24687/24688**. These 16 episodes cover open/wall
geometry and dry, 6 dB loss, 30 dB loss stress and combined stress. All three
policies receive the same candidate stream: raw, previous gate, recovery gate.
No policy changes are made after looking at the fresh results.

```bash
PYTHONPATH=python:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python experiments/run_cpu_position_recovery.py --workers 4
PYTHONPATH=python:. python -m pytest tests/test_position_recovery_cpu.py tests/test_cpu_position_recovery.py tests/test_position_quality_cpu.py tests/test_cpu_position_quality.py -q
```

Generate the prerequisites with the wet-antenna and position-quality scripts
in the [previous guide](position_quality_simulation.md) if needed. The output
directory `output/cpu-position-recovery/` contains all candidate/publication
epochs for three modes, per-episode and stage summaries, rejection reasons,
recovery-window counts/radius, reference source and replacement events. It
also saves fresh channel observations/raw positions, frozen policy bytes,
source/input SHA-256 hashes and an availability comparison figure. Raw replay
is checked against the saved receiver positions within one micrometre.

The same initial-publication and retrospective five-position recovery metrics
are used for every mode. An early recovery does not establish sustained
availability: dry-tail counts and all later gaps remain visible.

## Results: local improvement, overall availability regression

The policy stayed unchanged after fresh IQ generation. All 90,000 three-mode
epoch records, 180 episode summaries and 900 stage summaries are saved.
The 96,000 fresh channel observations and 8,000 raw positions allow independent
replay; maximum raw-position replay difference is **0.0 m**. Saved source,
input and policy hashes match the files used in this run.

| Dataset | Mode | Input epochs | Published | Unavailable | Published errors >100 m |
| --- | --- | ---: | ---: | ---: | ---: |
| Development | Raw | 22,000 | 18,196 | 3,804 | 30 |
| Development | Previous gate | 22,000 | 16,542 | 5,458 | 0 |
| Development | Recovery gate | 22,000 | 16,135 | 5,865 | 0 |
| Fresh holdout | Raw | 8,000 | 6,104 | 1,896 | 6 |
| Fresh holdout | Previous gate | 8,000 | 5,360 | 2,640 | 0 |
| Fresh holdout | Recovery gate | 8,000 | 5,262 | 2,738 | 0 |

Fresh publication availability falls from **67.0% to 65.8%** relative to the
previous gate. Both gates withhold the six >100 m raw outputs; the extension
does not demonstrate additional large-error rejection in these seeds. Open
geometry gains 78 published epochs (2,872 to 2,950), but wall geometry loses
176 (2,488 to 2,312). Jump rejections decrease from 371 to 308 while pending
confirmation increases from 329 to 490, accounting for the net 98 lost outputs.

The motivating development case (24686, wall, 6 dB loss) improves from 116 to
304 outputs out of 500, and its final dry 100 epochs improve from 0 to 90.
This local gain does **not** generalize to the aggregate. In fresh seed 24688,
all four wall scenarios publish 83/100 dry-tail positions with the extension,
versus 100/100 with the previous gate. In fresh seed 24687, wall/dry publication
falls from 495 to 438 out of 500 despite the absence of imposed wet loss.

**No recovery-cluster reference replacement occurs in either dataset.** The
longest qualifying window is ten candidates in development and nine in fresh
holdout, below the frozen requirement of 25. The observed changes therefore
come from the median reference, not demonstrated successful cluster recovery.
Controlled tests do exercise cluster replacement, new confirmation, rejection
on gaps/spread/insufficient observations and a coherent wrong-cluster failure.
The stronger rule is not loosened after seeing these results.

Some recovery timings improve: open/24688/30 dB stress goes from 200 to 127 ms
after drying starts. Others worsen: wall/24687/30 dB stress goes from 144 to
158 ms. These are the same retrospective five-position metric, not proof of
sustained recovery. `position_recovery.png` shows every fresh episode's output
count rather than selecting only favorable cases.

The extension remains opt-in and **is not adopted as the default**. Subsequent
work should model pseudorange/reference uncertainty and confirmation behavior,
then use further unseen realizations. Seeds 24687/24688 are now inspected data
and must not be presented as fresh validation of a revised policy.

The [uncertainty follow-up](position_uncertainty_simulation.md) adds a causal
range-scatter model and weighted position/covariance diagnostics, using further
new noise seeds before proposing changes to the publication gate.

## Limits

A coherent wrong cluster can satisfy these checks and replace the reference;
a test explicitly demonstrates publication of a 200 m wrong solution. Small
biases and gradual drift can also pass. Neither zero observed >100 m errors nor
an improved output fraction establishes a protection level or a false-fix bound.
The two fresh seeds share one static geometry per environment, paired noise
between scenarios and correlated tracking states. Antenna loss/bias inputs and
compressed wetting/drying times remain uncalibrated sensitivity/stress controls.
This does not validate real rainy-weather receiver behavior or dynamic motion.
