# Blind other-satellite cancellation prototype

The [weak-capture fix](cpu_weak_acquisition_solution.md) adds adjacent-delay
nuisance projection and coherent bit-edge search. Fresh 37 dB-Hz near-far
captures meet its declared simulation criterion; both the failed first attempt
and successful revised validation are preserved.

The previous fixed maximum gate rejected near-far cross-correlation detections
but also suppressed weak targets. This change adds one-pass estimation and
joint nuisance projection before target acquisition. True satellite carrier,
code delay, amplitude and presence labels are not supplied to the estimator.

## Estimation and subtraction

The fixed nuisance catalog is PRNs 2–12; target PRN 1 is explicitly excluded.
For each nuisance PRN:

1. Search integer circular delays and −2000…+2000 Hz in 250 Hz steps, using
   noncoherent point-template power on the first four 1 ms rows.
2. Retain peaks above the predeclared heuristic statistic of 35.
3. Project all 20 rows onto the coarse template. Remove the coarse carrier's
   phase across rows, then estimate residual carrier from squared successive
   complex projections. Squaring tolerates whole-period bit-sign reversals.
4. Require phase coherence >=0.5, a frequency correction <=125 Hz and a refined
   carrier inside the search interval.

These are heuristic nuisance-selection rules, not calibrated presence gates.
Squared phases have a ±250 Hz alias range, and incorrect peaks/aliases can
still pass. All selection steps are included in the final detector calibration.

Selected point-code/carrier templates form one joint dictionary. Per 1 ms row,
a complex least-squares coefficient is fitted for every selected PRN; the
joint projection is subtracted. This avoids sequential subtraction order but
is not iterative interference cancellation. A Gram-condition limit rejects
ill-conditioned dictionaries. Input IQ is copied and never mutated.

The model uses rectangular point-sampled C/A, integer delay and a fixed delay
over the record. It does not fit code Doppler, within-period nuisance bit edges,
fractional apertures or multipath. A component's complex amplitude can vary
between rows. Selected frequencies, delays, phase coherence and fitted RMS
amplitudes are retained for audit. Cached point references are immutable and
bounded; oversized periods bypass caching.

**Projection removes noise and can remove target energy too.** A smaller
residual is not proof of more accurate acquisition or target preservation.
Excluding PRN 1 from the catalog prevents directly fitting its code, but does
not eliminate cross-correlation leakage into nuisance estimates.

## API

```python
from gnss_gpu.ca_cancellation_cpu import CaCancellationConfig, cancel_other_ca

config = CaCancellationConfig(2_046_000.)
result = cancel_other_ca(iq, config)  # 20 rows of interleaved 1 ms IQ
residual_iq = result.iq
```

The residual is float64 real interleaved IQ. No-component and zero cases retain
the input values in a private copy. The result also records fitted components,
removed-energy fraction, Gram condition and status. It is a preprocessing result,
not a calibrated detection or tracking state.

## Complete-pipeline comparison

Raw and cancelled variants of both 1/4 ms target detectors are calibrated
separately. Both see the same 20 ms input records; cancellation gains no extra
observation duration. Its carrier refinement uses the whole record, so this is
an offline decision, not a causal low-latency tracker.

Each of the three earlier null strata (white-only, nominal other PRNs, and
other PRNs +20 dB) has 599 calibration and 499 fresh evaluation records. The
joint 5% calibration-failure budget is divided over four policies and three
strata. Each policy receives a fixed maximum of its stratum tolerance thresholds.
At n=599 the selected per-stratum rank is the maximum score. The calibration
therefore covers nuisance discovery, frequency fitting, projection, residual
normalization and the complete target search, not just a detector on untouched
white noise. Old raw thresholds are not reused as calibrated residual gates.

Target-present controls use 37/41/45 dB-Hz, 170/700 Hz and 24 fresh paired
replicas per profile/condition: 432 records. They retain a forced within-period
primary bit reversal; other satellites' bits stay constant. True parameters
are used only for synthesis and offline coarse carrier/phase scoring.

All protocols, selection rules and thresholds are fixed before evaluation.
No runtime profile label or truth selects a threshold. The guarantee applies
only to the declared IID null models; unknown interference, nuisance bit edges,
code drift or a different receiver/front end can invalidate it.

## Observed results

The maximum calibrated target thresholds were:

| Detector | Raw pipeline | Cancellation pipeline |
| --- | ---: | ---: |
| 1 ms | 28.913 | 20.197 |
| 4 ms | 46.342 | 34.362 |

Near-far calibration records set these maxima. Both pipelines use the same
calibration counts, confidence allocation and fresh paired records; the residual
gate is newly calibrated, rather than borrowing the lower historical white gate.

All white-only and nominal-other-satellite target-absent evaluation passages were
0/499 for every policy. In the +20 dB other-satellite stratum:

| Detector | Raw passages | After cancellation |
| --- | ---: | ---: |
| 1 ms | 0/499 | 0/499 |
| 4 ms | 5/499 | 1/499 |

The observed 4 ms rates are about 1.00% and 0.20%, with pointwise Wilson intervals
approximately 0.43–2.32% and 0.035–1.13%. These small counts do not prove equality
of false-alarm rates or an upper bound below 1% from evaluation data alone.
Zero observed passages also does not mean zero probability.

For the 4 ms method with strong other satellites, counts passing detection
**and** the coarse carrier/phase criterion out of 24 fresh paired replicas were:

| Nominal target C/N0 | Carrier | Raw pipeline | Cancellation pipeline |
| --- | ---: | ---: | ---: |
| 37 dB-Hz | 170 Hz | 0 | 1 |
| 37 dB-Hz | 700 Hz | 0 | 0 |
| 41 dB-Hz | 170 Hz | 1 | 24 |
| 41 dB-Hz | 700 Hz | 1 | 14 |
| 45 dB-Hz | 170 Hz | 21 | 24 |
| 45 dB-Hz | 700 Hz | 18 | 21 |

The 41 dB-Hz result demonstrates an exploratory improvement in this finite
synthetic near-far family, while 37 dB-Hz remains largely unavailable. This is
not a matched-true-false-alarm superiority claim or sustained tracking result.

On target-absent evaluation records, median selected component counts were
0 (white), 1 (nominal others) and 10 (strong others). Median removed-energy
fractions were 0%, 0.65% and 57.5%. The numbers describe the fitted projection;
they do not estimate actual interference power or prove that target energy
was untouched. Improvements on white-only controls can arise from the lower
recalibrated maximum gate even when no subtraction occurs there.

![Calibrated raw and cancellation pipeline outcomes](../output/cpu-ca-cancellation/cancellation_results.png)

## Reproduction and checks

```bash
PYTHONPATH=python:. python experiments/run_cpu_ca_cancellation.py
PYTHONPATH=python:. python experiments/plot_cpu_ca_cancellation.py
```

Outputs under `output/cpu-ca-cancellation/` include the protocol and source hashes,
frozen per-pipeline thresholds, every calibration/evaluation candidate and
acceptance, nuisance-fit audit data, raw/residual IQ hashes, every signal
outcome, aggregate tables and PNG/PDF figures. Existing experiments are retained.
The cancellation configuration must remain fixed with its calibration; changing
the catalog, discovery settings or model needs recalibration of the full pipeline.

`source_snapshot/` retains every hash-verified source used by the recorded run.
Afterward, the cancellation result's optional condition annotation was changed
to `typing.Optional` for the project's Python >=3.9 declaration. The numerical
function bytecode stayed identical; nine extra input cases matched archived and
current IQ and fit diagnostics exactly. `source_compatibility.json` records both
hashes and this check. Recorded source hashes refer to the snapshot; no numerical
results were overwritten by the annotation-only compatibility fix. Python 3.9
itself was not run in this environment.

283 relevant tests pass. New cases verify blind noiseless carrier/delay recovery,
unknown amplitudes, whole-period polarity reversals, joint multi-PRN fitting,
one controlled weak target's retention, input preservation, extreme scales,
explicit target exclusion, empty catalog and zero behavior, invalid inputs,
strict detection ties and audit serialization without IQ array duplication.

This is a conventional projection-based prototype, not a novel algorithm or
a full GNSS receiver. There is no sustained tracking, fractional ranging, PVT,
rain or real RF validation, and no universal target-preservation guarantee.
