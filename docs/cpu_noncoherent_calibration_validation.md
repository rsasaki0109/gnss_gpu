# Larger independent null validation

The next [fresh bit-timing validation](cpu_noncoherent_bit_validation.md) uses
upper-tail tolerance calibration, 4,999 calibration/evaluation records each,
and 768 fresh paired signals. It tests a common upper false-alarm budget rather
than claiming equality of the methods' true false-alarm probabilities.

The first noncoherent experiment observed 1/199 noise passages for 1 ms and
6/199 for 4 ms. This follow-up uses **1,999 calibration records and 1,999 new
evaluation records**, each 20 ms, with disjoint seed domains from the earlier
experiment and from each other. The detectors, grids and decision horizon are
unchanged. Both methods share each noise record for paired comparison.

The run writes `protocol.json` before calibration and `frozen_thresholds.json`
before evaluation. Configuration, input/source hashes and the 1% marginal
decision-level calibration target are fixed before the new outcomes are seen.
Evaluation results are not used to adjust either threshold. This is local
experiment bookkeeping, not an externally registered protocol.

## New frozen-threshold results

| Detector | Old threshold | New threshold | New noise passages | Observed rate | Pointwise 95% Wilson interval |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 ms windows | 16.894306 | 16.806368 | 19/1999 | 0.95% | 0.61–1.48% |
| 4 ms windows | 23.139309 | 22.807714 | 28/1999 | 1.40% | 0.97–2.02% |

Of the fresh records, both methods passed 2, only 1 ms passed 17, only 4 ms
passed 26, and neither passed 1,954. The observed difference is about 0.45
percentage point, retaining the pairing of outcomes.

These intervals estimate each newly frozen threshold's conditional white-noise
passage probability. They are pointwise, not simultaneous. Overlapping intervals
do not establish equivalence, and the experiment does not prove either rate is
at most 1%. The rank calibration guarantee remains marginal over calibration
sets; only one new calibration set was evaluated here.

## Separate threshold effects from evaluation-sample effects

The original 199-record evaluation and this 1,999-record evaluation use different
noise. Their observed rates cannot be compared as if only thresholds changed.
Applying both old and new thresholds to the **same new candidate scores** gives:

| Detector | Original data, old threshold | Fresh data, old threshold | Same fresh data, new threshold |
| --- | ---: | ---: | ---: |
| 1 ms | 1/199 | 16/1999 | 19/1999 |
| 4 ms | 6/199 | 18/1999 | 28/1999 |

Both new thresholds are lower, so they add noise passages on the same stream:
3 for 1 ms and 10 for 4 ms, with none removed. Therefore the lower observed
4 ms rate relative to the original 6/199 is **not a demonstrated calibration
improvement**. A larger calibration sample reduces sampling uncertainty in the
estimated tail; it does not guarantee a higher threshold or lower passage rate.

![Original and fresh noise data with paired threshold controls](../output/cpu-noncoherent-calibration-validation/null_validation.png)

## Signal rethresholding

All 144 previously observed signal candidate scores are re-evaluated against
the new thresholds. Candidate frequencies, integer delays, winning windows,
score definitions and offline criteria are preserved; detection and combined
criterion flags are recomputed. Source/configuration hashes are checked before
cached scores can be reused.

At 37 dB-Hz, the coarse carrier/phase criterion remains 1/12 for 1 ms versus
9/12 at 170 Hz and 8/12 at 700 Hz for 4 ms. This is **rethresholding of existing
signal data**, not an independent signal validation. With unequal observed
noise passage rates, it is still not a gain demonstrated at equal false-alarm
probability. The earlier results and their raw data remain intact.

## Reproduction and verification

```bash
PYTHONPATH=python:. python experiments/validate_cpu_noncoherent_calibration.py
PYTHONPATH=python:. python experiments/audit_cpu_noncoherent_thresholds.py
```

Outputs under `output/cpu-noncoherent-calibration-validation/` include:

* `protocol.json` and `frozen_thresholds.json`, with configuration/source/input
  hashes and the settings frozen before evaluation.
* `calibration.csv`: both scores and an IQ hash for every calibration record.
* `evaluation.jsonl`: both candidates, acceptance flags and IQ hashes for all
  fresh records; no RF IQ storage is required for deterministic regeneration.
* `report.json`: observed null counts, pointwise Wilson intervals, paired
  outcomes and threshold comparisons.
* `signal_decisions.jsonl` / `aggregate.json`: all rethresholded signal outcomes.
* `same_stream_threshold_comparison.json` and PNG/PDF plots: the paired
  old/new-threshold control on the fresh evaluation stream.

The audit verifies the validation protocol and prior report hashes, checks the
frozen threshold file, and recomputes new acceptance before comparing thresholds.
Seven calibration/audit tests and the previous 246 relevant tests pass. Checks
cover paired counts, uncertainty after zero observed events, strict threshold
ties, candidate preservation, inconsistent cached scores, disjoint seed domains,
same-stream monotonicity and saved-acceptance consistency.

This follow-up adds independent **noise** evidence and better uncertainty
reporting. It does not establish general sensitivity, sustained tracking gain,
rain effects, PVT, RF accuracy or integrity. Fresh signal replicas and repeated
calibration sets remain useful next validation steps.
