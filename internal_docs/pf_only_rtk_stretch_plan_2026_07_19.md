# PF-only RTK full-run stretch plan (2026-07-19)

## Objective

Raise `<50 cm` full-run coverage to at least 81% on Tokyo run1 and 86% on
Nagoya run1 with one truth-free configuration, no runtime FGO, and declared
FIX false rate at or below 1%. The WP30 M4 1200-window report remains a frozen
baseline and is not overwritten.

## Denominator correction

The production PPC loader was replayed without `max_epochs`. After satellite,
ephemeris and reference-time gates, the actual 5 Hz denominators are:

| run | usable epochs | TOW range | required correct epochs |
|---|---:|---:|---:|
| Tokyo run1 | 11,924 | 187470.0--189860.0 | >=9,659 (81%) |
| Nagoya run1 | 7,583 | 550380.0--551910.0 | >=6,522 (86%) |

The earlier 1,200-epoch artifacts cover only the first 240 seconds. They are
valid window benchmarks, but must not be described as dataset-full results.

## Work packages

### S1 -- denominator and baseline lock

- hash input RINEX/reference files and record loader epoch counts;
- replay the common GPU candidate generator and auto selector on every epoch;
- report candidate oracle recall separately from selected output coverage.

Gate: ordered trajectory row count equals the usable denominator; no truth
fields enter candidate generation or selection; false FIX <=1%.

### S2 -- gap decomposition

Partition every error epoch into candidate-supply miss, selector miss,
transition/outage loss, and absolute-anchor miss. Report contiguous failure
blocks, acquisition delay, candidate survival and recovery latency.

Gate: every lost target epoch has exactly one primary failure class and an
evidence artifact; no threshold changes are made from truth labels.

Status 2026-07-19: in progress. Nagoya run1 candidate union is only
1,802/7,583 (23.76%), with a longest supply-miss block of 3,511 epochs. The
motion audit also found one 13.8 s recorded-time gap and multiple TDCP outage
runs. These are now separately telemetered rather than treated as ordinary
0.2 s intervals.

### S3 -- common recovery policy

Develop run-independent gates from TDCP/Doppler/IMU motion consistency,
multi-frequency/widelane integer consistency, carrier absolute evidence,
satellite integrity and static-stop anchors. Candidate proposal and output
selection remain separate; recovery suppresses FIX until consistency returns.

Gate: each promoted block has causal support, passes held-out evidence checks,
and does not increase false FIX above 1% on either full run.

Status 2026-07-19: active shadow development. Truth-free coarse-to-fine static
search plus temporal/wide-lane/road/height evidence produced four accepted
Nagoya anchors. A gyro-shaped TDCP gap fill and anchor endpoint closure raised
Nagoya full `<50 cm` coverage from 598/7,583 (7.89%) to 2,265/7,583 (29.87%)
with zero declared FIX and zero false FIX. Short-evidence and weak-separation
anchors are fail-closed; several audit failures are retained as negative
evidence rather than promoted.

### S4 -- freeze and locked benchmark

Freeze one configuration before the final two-run replay. Hash configuration,
inputs, intermediate evidence and final trajectories. Include CPU/GPU parity,
runtime, p99 latency and bounded-memory audit.

Gate: Tokyo run1 >=9,659/11,924 and Nagoya run1 >=6,522/7,583, both with false
FIX <=1%, full denominator, truth-free selection and no runtime FGO. Otherwise
publish an honest negative with the measured limiting failure class.
