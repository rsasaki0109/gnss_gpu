# Weak 37 dB-Hz capture: implemented fix and fresh validation

The declared simulation capture criterion is met: with other satellites boosted
by 20 dB, at least 24/32 fresh replicas passed detection and the existing coarse
carrier/phase check at **both** 170 and 700 Hz. The observation horizon remains
20 ms. This resolves the specified experimental capture failure; it is not a
guarantee of a 75% population detection probability, RF sensitivity or tracking.

## What changed

Two issues needed attention:

* Summing 1 ms correlation powers discards phase information needed at 37 dB-Hz.
  The new detector coherently combines period correlations, searching residual
  carrier in 25 Hz steps and an unknown single navigation-bit reversal.
* A nuisance code delay can cross an integer sample during the record. Subtracting
  a single fixed-delay waveform then leaves structured other-satellite energy.
  The refined projection uses adjacent delay columns, with complex coefficients
  fitted separately per period, without a true code-clock or delay input.

For each coarse carrier (250 Hz spacing), the detector first shortlists 64
integer delay bins by noncoherent power. It then searches coherent sums with
constant polarity or a single reversal inside any 1 ms row. The possible
transition row is omitted and opposite signs are applied on its two sides.
The true edge, carrier, code lag and sign are never supplied to inference.
`shortlist_size=None` enables exhaustive delay search for reference checks.
Pruning can miss a coherent peak; the shortlist selection is included in calibration.

Carrier correction between rows leaves a small within-row rotation loss.
Integer aperture templates, a constant target carrier/delay and at most one
bit reversal are model assumptions. There is no bit decoding or tracking loop.

## Preserved first attempt and diagnosed residual

The first coherent attempt is retained under `output/cpu-weak-acquisition-solution/`.
It obtained 23/32 and 24/32 on the primary conditions and therefore **failed**
the predeclared criterion. The criterion was not lowered.

In a development calibration record, PRN 2's sampled code lag changed from
−143 to −144 during the record. The fixed dictionary could not represent both.
Adding neighboring delay columns reduced that record's target-absent coherent
score from 62.89 to 13.85. This is a paired development diagnostic, not a held-out
performance claim. It motivated the local-delay model before fresh validation.

The revised run uses different calibration, evaluation and signal seed domains.
Its aggregate counts must not be compared to the first attempt as if only the
algorithm changed; the held-out draws also changed. A matched baseline remains
inside the revised run: original cancellation + 4 ms noncoherent acquisition,
compared with local-delay cancellation + coherent bit-edge acquisition on the
same IQ. Both preprocessing and acquisition change in that pipeline comparison.

## Fresh results

Each of the three finite null strata has 499 calibration and 499 evaluation
records. The common 1% conditional false-alarm upper budget is tolerance-calibrated
with a 5% joint calibration-failure allocation over two methods and three strata.
All window, carrier, delay, shortlist, nuisance selection and bit-edge trials
belong to the complete decision being calibrated. Gates are frozen before
evaluation; they are not tuned to make empirical false-alarm counts match.

The revised target thresholds are 30.8133 for the noncoherent baseline and
59.8474 for the coherent pipeline. Their score definitions differ, so threshold
magnitudes alone are not a sensitivity comparison.

| Strong-other-satellite condition | Matched 4 ms baseline | Revised coherent pipeline |
| --- | ---: | ---: |
| 37 dB-Hz, 170 Hz | 2/32 | **27/32** |
| 37 dB-Hz, 700 Hz | 1/32 | **24/32** |
| 41 dB-Hz, 170 Hz | 30/32 | 32/32 |
| 41 dB-Hz, 700 Hz | 28/32 | 32/32 |

These counts require detection **and** the unchanged coarse criterion: carrier
error <=250 Hz and circular code lag within two samples of the nominal initial
phase. True parameters are used only for offline scoring. For the new primary
37 dB-Hz counts, pointwise 95% Wilson intervals are approximately 68.2–93.1%
and 57.9–86.7%. Thus the observed 75% criterion is met, but the lower confidence
limits are not at least 75%; no such population guarantee is claimed.

Target-absent passages on the new 499-record streams:

| Null stratum | Matched baseline | Revised pipeline |
| --- | ---: | ---: |
| White noise only | 0/499 | 0/499 |
| Nominal other PRNs | 0/499 | 0/499 |
| Other PRNs +20 dB | 1/499 | 1/499 |

Observed passage counts match, but this does not prove equality of the true
false-alarm probabilities. The 1/499 pointwise interval is about 0.035–1.13%,
and the upper endpoint for 0/499 is about 0.764%. The tolerance guarantee relies
on representative IID null records and covers only the declared strata, not
arbitrary interference, other Dopplers, RF hardware or dependent observations.

![Primary capture and independent null outcomes](../output/cpu-weak-acquisition-refined/weak_acquisition.png)

## Use the calibrated complete pipeline

```python
from gnss_gpu.weak_acquisition_cpu import load_weak_ca_experiment, detect_weak_ca

gate = load_weak_ca_experiment("output/cpu-weak-acquisition-refined")
result = detect_weak_ca(iq, gate)  # shape (20, 4092), interleaved real I/Q
if result.detection.accepted:
    print(result.detection.candidate)
```

The loaded calibration binds nuisance cancellation, local-delay radius and
bit-edge search. It verifies the protocol/frozen-gate hashes and three principal
processor source hashes before loading. These are local artifact identity checks,
not authentication of an untrusted report or a claim of exhaustive dependency
pinning. The existing one-PRN tracker is not silently switched to this detector.

`calibrate_weak_ca` accepts streamed null records per stratum and executes the
entire preprocessing/search path internally. It returns a bound threshold used
by `detect_weak_ca`; a separate runtime preprocessing override is not accepted.
Changed settings or signal/null models require fresh calibration.

The operational API reproduced all **64 primary held-out detections and residual
IQ hashes exactly** against the experiment. `api_parity.json` records this check.

## Reproduction and remaining scope

```bash
PYTHONPATH=python:. python experiments/solve_cpu_weak_acquisition.py       # preserved failed first model
PYTHONPATH=python:. python experiments/solve_cpu_weak_acquisition_v2.py    # revised fresh validation
PYTHONPATH=python:. python experiments/plot_cpu_weak_acquisition.py
```

The successful output directory is `output/cpu-weak-acquisition-refined/`.
It retains the protocol, frozen gates, source hashes, all 1,497 calibration and
1,497 evaluation records, all 384 fresh signal outcomes, aggregate counts,
runtime versions, API parity and PNG/PDF figures. Earlier results are preserved.

294 relevant tests pass. Checks include direct time-domain coherent-score
agreement, unknown within-period edges, constant polarity, strict gate ties,
shortlists, a nuisance epoch slip, target retention in a controlled mixture,
radius-zero equivalence, streamed full-pipeline calibration, configuration
mismatches, calibration loading/hash checks and unchanged failed-replica denominators.

Some 37 dB-Hz captures still fail. Coarse nuisance discovery can miss components;
one revised calibration record missed PRN 4, keeping the final gate conservative.
The subsequent [multiple-seed discovery study](cpu_multiseed_acquisition.md)
investigates this failure with a fresh paired calibration and evaluation.
Unknown within-period nuisance bit changes, multiple target reversals, carrier
dynamics, code drift beyond the local span and point/aperture mismatch remain
unvalidated. No sustained tracking, precise ranging, real-time throughput,
positioning, rain, real RF or research novelty claim follows from this result.
