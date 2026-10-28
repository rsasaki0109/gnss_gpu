# Target-absent other-satellite nulls

The next [blind cancellation experiment](cpu_ca_cancellation.md) estimates
other-satellite waveforms from IQ and calibrates the complete residual detector.
It improves the 41 dB-Hz near-far outcomes while retaining explicit null controls.

The earlier tolerance guarantee covered white noise with all signals absent.
This experiment removes only PRN 1 and leaves other satellites present, then
measures both false passages and the sensitivity cost of guarding these nulls.

## Fixed null family and calibration

Three declared strata are evaluated:

| Stratum | PRN 1 | PRNs 2–12 |
| --- | --- | --- |
| `white` | Absent | Absent |
| `other_prns` | Absent | Amplitude 0.4/PRN, constant bit polarity |
| `near_far_20db` | Absent | Ten times the nominal amplitude, hence +20 dB power |

All use the same white-noise power as earlier experiments. Other satellites'
code phases and carrier phases are independently randomized per record;
Dopplers stay at the earlier fixed PRN-dependent values. Absolute sample starts
are randomized within 20 ms. Profile pairs share underlying random draws, with
separate domains for calibration, evaluation and target-present signals.

The +20 dB label is relative to **other satellites' nominal powers**, not to
the target. For example PRN 2 has amplitude 0.2 nominally and 2.0 in the strong
stratum. This is a synthetic near-far stress control, not a measured RF scene.

Each method/stratum gets 999 calibration records and a 1% upper-tail tolerance
threshold. The 5% joint calibration failure budget is divided over two methods
and three strata: delta=0.05/6 each. At n=999 the selected rank is 997; its
binomial failure bound is approximately 0.002702 per calibration.

For each detector, one **maximum of the three thresholds** is frozen and used
on every record. No runtime profile or true satellite-presence label selects
the gate. By the union bound, the declared per-stratum budget holds jointly
with at least the requested calibration confidence under the IID model. Raising
a threshold to the maximum cannot increase a stratum's false-alarm rate.

This covers the three specific distributions and independent mixtures of them,
not every interference level between nominal and +20 dB, unseen Dopplers,
time-correlated RF scenes or all possible satellites. A mixture-average
calibration alone would not provide this per-stratum protection.

## Paired comparisons

The historical white-noise tolerance thresholds are preserved as the baseline.
All target-absent candidates are scored once and passed through both historical
and new fixed gates on the exact same IQ. Each stratum has 999 fresh evaluation
records. Calibration and evaluation data never overlap; thresholds are frozen
before evaluation outcomes are seen.

Target-present controls add PRN 1 at 37/41/45 dB-Hz and 170/700 Hz, with 32
fresh paired replicas per cell and profile: 576 records. Each has one forced
primary bit reversal inside a 1 ms period. Other satellites' parameters and
noise draws are retained when the target is inserted. The same coarse
carrier/nominal-phase criterion as earlier experiments separates a raw detection
from a candidate near the intended peak.

The white baseline and new gates have different calibration counts and family
confidence allocations. This compares two complete policies; it does not
isolate the effect of adding strata while all other calibration details stay
fixed. Sensitivity outcomes are exploratory and retain failures.

## Observed results and sensitivity cost

On the same 999 fresh target-absent records per profile, the number of detection
passages was:

| Null stratum | 1 ms white gate | 1 ms stratified gate | 4 ms white gate | 4 ms stratified gate |
| --- | ---: | ---: | ---: | ---: |
| White noise | 7 | 0 | 12 | 0 |
| Nominal other PRNs | 8 | 0 | 15 | 0 |
| Other PRNs +20 dB | 612 | 0 | 999 | 1 |

The strong-other-satellite condition exposes a failure outside the historical
white-noise guarantee: the target is absent, but the fixed PRN-1 detector passes
every 4 ms decision. This is a detection failure in the declared simulation,
not a demonstrated false positioning fix or a real RF measurement. Ordinary
other-satellite results have much smaller changes and their small-sample rates
alone do not establish a violation of the historical white-noise model.

The new scalar thresholds are 27.171221 for 1 ms and 49.460494 for 4 ms, compared
with historical 17.572386 and 23.429654. The near-far stratum sets both maxima.
Its evaluation passage counts fall to 0/999 and 1/999. Zero observed passages
does not prove zero probability: the pointwise Wilson upper endpoint for 0/999
is about 0.383%. For 1/999 the interval is approximately 0.018–0.565%.

This protection costs sensitivity on the same target-present IQ. At 170 Hz,
counts meeting detection **and** the coarse carrier/phase criterion out of 32
fresh replicas for the 4 ms method are:

| Other-satellite condition | Target C/N0 | White gate | Stratified gate |
| --- | ---: | ---: | ---: |
| Nominal | 37 dB-Hz | 11 | 0 |
| Nominal | 41 dB-Hz | 29 | 9 |
| Nominal | 45 dB-Hz | 32 | 32 |
| +20 dB power | 37 dB-Hz | 1 | 0 |
| +20 dB power | 41 dB-Hz | 7 | 1 |
| +20 dB power | 45 dB-Hz | 30 | 20 |

All tested 37 dB-Hz criterion successes disappear under the stratified maximum
gate, including white-only target controls. This is the cost of one scalar gate
covering a strong-nuisance family, not evidence that the weak target became
physically absent. Raising a threshold alone cannot preserve the earlier
weak-signal gain while also rejecting these near-far peaks in this experiment.
Better signal discrimination or estimated other-satellite cancellation is a
natural next step; no cancellation has been implemented here.

![Target-absent passages and descriptive target-present sensitivity](../output/cpu-other-prn-nulls/other_prn_tradeoff.png)

## API and reproduction

```python
from gnss_gpu.stratified_tolerance_cpu import (
    calibrate_ca_stratified_tolerance, detect_ca_stratified_tolerance,
)

threshold = calibrate_ca_stratified_tolerance(
    scores_by_stratum, config,
    max_false_alarm_probability=.01,
    joint_failure_budget=.05, methods_in_family=2,
)
result = detect_ca_stratified_tolerance(iq, threshold)
```

The configuration and immutable per-stratum calibrations are retained with the
maximum threshold. Construction checks that it dominates every stratum, uses
matching search configurations and stays inside the allocated family budget.

```bash
PYTHONPATH=python:. python experiments/run_cpu_other_prn_nulls.py
PYTHONPATH=python:. python experiments/plot_cpu_other_prn_nulls.py
```

`output/cpu-other-prn-nulls/` contains the protocol and source/baseline hashes,
all calibration/evaluation candidates and IQ hashes, frozen threshold details,
every target-present result, aggregated outcomes, and PNG/PDF figures. Original
experiments are preserved. The plot shows all null strata and descriptive
170 Hz signal curves; the complete 700 Hz outcomes remain in the raw tables.

273 relevant tests pass. Added checks verify maximum-gate dominance, confidence
allocation, configuration mismatch rejection, invalid families, absence of
PRN 1, exact +20 dB amplitude scaling, matched nuisance parameters when inserting
the target, independent domains, strict gate ties and batch/individual equality.

There is no interference cancellation or multi-satellite joint acquisition in
this change. A conservative fixed gate may suppress weak target signals along
with cross-correlation detections. The declared null family, constant nuisance
bit signs, point-sampling/aperture mismatch and coarse delay bins limit physical
interpretation. No sustained tracking, PVT, rain, real RF or novelty claim is made.
