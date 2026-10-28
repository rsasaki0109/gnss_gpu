# Fresh bit-timing validation under a common false-alarm budget

A subsequent [other-satellite null experiment](cpu_other_prn_nulls.md) tests the
scope limitation of the white-noise budget. Strong target-absent satellite
signals produce near-far false detections; a fixed stratified gate suppresses
them at a measured weak-signal sensitivity cost.

The previous comparison used a marginal rank calibration and previously observed
signal scores. This experiment introduces an upper-tail tolerance calibration
and new signal/noise records. Its scope is a common **upper false-alarm budget**,
not equality of the methods' unknown true false-alarm probabilities.

## Tolerance calibration

For n IID representative null decision scores, use the kth ascending score and
accept only strictly larger evaluation scores. Choose the smallest k satisfying:

```
P[Binomial(n, alpha) <= n-k] <= delta
```

Then the probability over calibration samples that the frozen threshold's true
conditional null passage probability exceeds alpha is at most delta. This is
an order-statistic tolerance result; continuous scores give the exact tail,
and ties with strict acceptance make it conservative. It requires IID records
and the same null distribution during calibration and operation.

The experiment sets alpha=0.01 and delta=0.025 for each method. By the union
bound, both frozen thresholds meet the upper budget with calibration probability
at least 0.95, without requiring independence between methods. Calibration uses
4,999 new shared null records; the chosen rank is 4,963, with binomial failure
probability approximately 0.02331 per method. The combined bound is below 0.05.
This probability concerns calibration draws, not a per-record confidence score.

The NumPy/standard-library implementation uses log-domain binomial summation
and a binary search for the rank. It introduces no mandatory SciPy dependency.
`calibrate_ca_noncoherent_tolerance` and `detect_ca_noncoherent_tolerance` live
in `gnss_gpu.noncoherent_tolerance_cpu`; existing marginal calibration remains
available and the original detector's numerical calculations are unchanged.

Both methods search the same ±1000 Hz five-bin grid and the same 20 ms horizon.
The 1 ms method searches 20 windows; the 4 ms method searches five windows.
Calibration includes each complete decision's maximum over all windows,
frequencies and integer delays. Thresholds are frozen before 4,999 further
independent null records or any new signal outcomes are evaluated.

**The null is all signals absent with simulated Gaussian white noise.** The
budget does not cover a target-absent receiver with other satellites, colored
noise, RF interference or a different distribution.

## Fresh signal and bit controls

Each 35/37/39/41/43/45 dB-Hz and 170/700 Hz condition has 32 fresh replicas,
starting at replica 1000 with new noise seed domains. Each replica is evaluated
twice with identical underlying noise, phases, geometry and sample starts:

* `constant`: PRN 1 has constant navigation polarity.
* `flip`: PRN 1 has one forced sign reversal inside a 1 ms period.

Sample-zero offsets are drawn uniformly over a 20 ms navigation bit, excluding
offsets aligned to 1 ms boundaries. Absolute code and carrier time continue
through the record. The simulator's existing navigation-bit indexing generates
the reversal; there is no oracle bit wipe or new simulator timing shortcut.
Other 11 PRNs have constant polarity to isolate the primary's bit transition.
This is a forced-reversal stress test, not random complete navigation messages.

There are 768 fresh 20 ms signal records. Both detectors are evaluated on every
record. Offline criteria remain carrier error <=250 Hz and circular phase lag
within two samples of the nominal starting phase, adjusted for the randomized
absolute sample offset. Small code drift within a record is not fitted. This
coarse criterion is not ranging, tracking or positioning accuracy.

The primary endpoints are declared in the saved protocol: 37 dB-Hz with forced
reversal at 170 and 700 Hz. A two-sided exact paired sign test compares discordant
criterion outcomes, with a Bonferroni factor of two for this family. These
endpoints were motivated by previous exploration but fixed before these fresh
signals were evaluated. Other conditions are exploratory. Wilson intervals
describe pointwise replica event rates, not simultaneous coverage.

## Observed results

The frozen tolerance thresholds were 17.572386 (1 ms) and 23.429654 (4 ms).
All calibration, evaluation and signal records completed without retuning.

| Method | Fresh white-noise passes | Observed rate | Pointwise 95% Wilson interval |
| --- | ---: | ---: | ---: |
| 1 ms windows | 18/4999 | 0.36% | 0.23–0.57% |
| 4 ms windows | 41/4999 | 0.82% | 0.61–1.11% |

Both passed 6 records, only 1 ms passed 12, only 4 ms passed 35, and neither
passed 4,946. The observed rates differ. In particular, the held-out interval
for 4 ms still extends above 1%; held-out data alone do not establish that its
true rate is below the budget. The separate tolerance-calibration statement
depends on the IID model and probability over calibration draws.

At 37 dB-Hz, counts meeting detection and the coarse carrier/phase criterion
were as follows, out of 32 fresh replicas per cell:

| Bit control | True carrier | 1 ms | 4 ms |
| --- | ---: | ---: | ---: |
| Constant polarity | 170 Hz | 0 | 22 |
| Forced within-period reversal | 170 Hz | 0 | 18 |
| Constant polarity | 700 Hz | 0 | 14 |
| Forced within-period reversal | 700 Hz | 1 | 14 |

Raw detection passages in the primary reversal conditions were 0/32 versus
25/32 at 170 Hz, and 1/32 versus 23/32 at 700 Hz. The criterion therefore matters;
passing detection alone does not establish the right coarse carrier/code peak.

For the two declared reversal endpoints, discordant criterion outcomes favored
4 ms by 18 versus 0 and 13 versus 0, respectively. Bonferroni-adjusted exact
two-sided paired p-values were approximately 0.0000153 and 0.000488. These
provide model-specific evidence of different criterion-event probabilities on
fresh paired signals. They do not establish gain at exactly equal true
false-alarm rates, physical receiver accuracy, or general GNSS performance.

The forced reversal reduces some 4 ms successes relative to constant polarity:
noncoherent accumulation does not eliminate within-period cancellation. Its
weak-signal benefit remains visible in these primary stress conditions despite
that limitation. Results for all strengths, including less favorable cells,
remain in the full output and figure.

![Fresh paired signal results with pointwise uncertainty](../output/cpu-noncoherent-bit-validation/bit_validation.png)

## Reproduction and checks

```bash
PYTHONPATH=python:. python experiments/run_cpu_noncoherent_bit_validation.py
PYTHONPATH=python:. python experiments/plot_cpu_noncoherent_bit_validation.py
```

Outputs under `output/cpu-noncoherent-bit-validation/` retain the protocol,
source hashes, frozen threshold metadata, every calibration/evaluation score
and IQ hash, every fresh signal outcome, aggregate event rates and paired
statistics. The protocol is written before calibration and thresholds before
evaluation. No outcome-dependent threshold tuning is performed. This is local
experiment bookkeeping, not an external registration.

265 relevant tests pass. New checks include direct binomial probability sums,
known large-sample ranks and their minimality, insufficient sample counts,
strict gate ties, exact within-period flip timing in isolated noiseless IQ,
batch/individual signal equality, paired-noise preservation, unchanged other
satellite polarities, exact paired probabilities and primary-family adjustment.

Point-sampled generation versus aperture templates, integer delays, the small
number of signal replicas and the simulated null all limit interpretation.
There is no fractional carrier refinement, sustained tracking, PVT, rain, RF
validation or algorithmic novelty claim. Meeting a common upper budget does
not demonstrate performance at exactly equal false-alarm rates.
