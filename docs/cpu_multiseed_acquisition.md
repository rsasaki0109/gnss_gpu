# Multiple coarse seeds for nuisance discovery

The predeclared empirical improvement criterion is met on fresh simulation
records: the 37 dB-Hz near-far coarse capture counts are **28/32 -> 32/32** at
170 Hz and **32/32 -> 32/32** at 700 Hz. These are matched comparisons on the
new stream, not comparisons with the different replicas in the earlier run.

The experimental nuisance canceller refines the three highest-scoring coarse
carrier candidates per PRN, rather than discarding a satellite when the single
highest-scoring candidate fails its frequency-cell check. Each carrier retains
its own best delay. The same discovery-statistic, phase-coherence and frequency
correction limits apply to every seed. Among valid seeds, the largest full-record
projection energy chooses one carrier/delay per PRN. Ties retain discovery-score
order, then ascending coarse carrier. There are no true-parameter inputs.

The selected satellites enter the same joint per-period fit, adjacent-delay
refinement and coherent bit-edge detector as the previous pipeline. More seeds
can select spurious components and remove target/noise energy. Reduced residual
energy alone is not evidence of improved detection. The complete expanded
search therefore receives a new calibration; the old gate is never reused.

The historical processor and experiment artifacts are preserved. The alternative
lives in `python/gnss_gpu/ca_multiseed_cpu.py`; `coarse_candidates=1` calls the
historical implementation exactly. This keeps the previous calibrated model's
source identity intact.

## Development diagnostic

The previous revised run's calibration record `near_far_20db/281` missed PRN 4.
The multiple-seed search recovers it near 598 Hz from the 500 Hz coarse cell.
The target-absent coherent score changes from 59.8474 to 17.8893. This record was
used to develop the change and is **not** held-out evidence. The reproducible
audit checks the raw IQ hash and original candidate against the preserved run.

## Fresh paired validation

`experiments/validate_cpu_multiseed_acquisition.py` fixes new, separate seed
domains for calibration, evaluation and signals. Both pipelines see the same IQ,
20 ms horizon and coherent search; only nuisance discovery differs. The three
null strata remain white noise, nominal PRNs 2–12 and those PRNs with 20 dB more
power than nominal. They have randomized code/carrier phases, fixed Dopplers,
constant nuisance polarity and white noise. They are not arbitrary RF models.

Each method uses 499 calibration records per stratum. A stratum tolerance gate
with a 1% conditional false-alarm upper budget is fitted for each method; the
single operational gate is the maximum across strata. The 5% joint calibration
failure budget is split over two methods and three strata. The gate includes
all nuisance candidates, projections, delay shortlists and coherent hypotheses.
499 independent evaluation records per stratum follow, without retuning.

Signals use 32 new replicas per profile, C/N0 (37/41 dB-Hz) and carrier
(170/700 Hz). The primary empirical criterion, written before evaluation,
requires at least 24/32 coarse captures at both 37 dB-Hz near-far conditions,
no decrease in either primary count and at least two additional primary captures
in total versus the matched single-seed pipeline. A coarse capture requires
detection, carrier error <=250 Hz and circular lag error <=2 samples relative
to nominal initial phase. True values enter only synthesis and offline scoring.

The complete run contains 1,497 calibration records, 1,497 independent null
evaluation records and 384 signal records. Both methods use freshly calibrated
gates, frozen before evaluation. Their maximum thresholds are 55.0403 for the
single seed and 20.5167 for three seeds. The near-far calibration-stratum gate
falls from 55.0403 to 20.3411; the nominal-other-PRN stratum now sets the maximum.

| Other satellites +20 dB | Single seed | Three seeds |
| --- | ---: | ---: |
| 37 dB-Hz, 170 Hz | 28/32 | **32/32** |
| 37 dB-Hz, 700 Hz | 32/32 | **32/32** |
| 41 dB-Hz, 170 Hz | 32/32 | 32/32 |
| 41 dB-Hz, 700 Hz | 32/32 | 32/32 |

At 170 Hz, four paired primary replicas change from failure to success and none
change in the opposite direction. At 700 Hz, all primary replicas pass both
methods. Each 32/32 pointwise 95% Wilson interval is approximately 89.3–100%;
this does not establish universal success. No superiority test was predeclared.

| Independent target-absent stratum | Single seed | Three seeds |
| --- | ---: | ---: |
| White noise | 0/499 | 0/499 |
| Nominal other PRNs | 0/499 | 0/499 |
| Other PRNs +20 dB | 3/499 | 1/499 |

The near-far pointwise 95% Wilson intervals are approximately 0.205–1.752% and
0.035–1.126%. The upper endpoint for 0/499 is 0.764%. These finite observations
do not establish lower or equal true false-alarm probabilities; both gates use
the same calibrated upper budget. The results are preserved without retuning.

![Matched capture counts and independent null outcomes](../output/cpu-multiseed-acquisition/weak_acquisition.png)

## Use and reproduce

```python
from gnss_gpu.weak_acquisition_cpu import load_weak_ca_experiment, detect_weak_ca

gate = load_weak_ca_experiment("output/cpu-multiseed-acquisition")
result = detect_weak_ca(iq, gate)  # 20 contiguous 1 ms rows, real interleaved IQ
```

The loader binds `coarse_candidates=3`, nuisance settings, local-delay radius
and the target detector to the saved calibration. It checks artifact hashes
and the four principal processor source hashes. These are local identity checks,
not authentication or exhaustive dependency pinning. The previous v2 calibration
continues to load with one candidate. For another model, use `calibrate_weak_ca`
with representative null record streams and `coarse_candidates=3` to obtain a
fresh bound gate. There is no runtime preprocessing override in `detect_weak_ca`.

```bash
source /workspace/.gnss-env/activate.sh
python experiments/audit_cpu_multiseed_discovery.py
python experiments/validate_cpu_multiseed_acquisition.py
python experiments/audit_cpu_multiseed_validation.py
python experiments/plot_cpu_weak_acquisition.py --directory output/cpu-multiseed-acquisition
```

The validation refuses a nonempty output directory; use `--output` for a fresh
replication. The output retains protocols, frozen thresholds, source hashes,
per-record outcomes, nuisance components, residual IQ hashes and aggregate data.

The artifact audit checks all record identities, disjoint IQ hashes across the
three splits, every saved acceptance flag, regenerated frozen gates and signal
aggregates. The operational API reproduces all 64 primary held-out detections
and residual IQ hashes exactly. `audit.json` retains this check and runtime
versions. The 39 relevant tests pass, including recovery from an adjacent coarse
cell, exact historical one-seed equivalence, controlled target retention,
extreme-scale inputs, streamed calibration, artifact loading and both API modes.

## Scope

The subsequent [continuous weak-signal following experiment](cpu_weak_tracking.md)
connects this frozen gate to disjoint-window state following, including target
outage, reacquisition and sequence-level null controls.

An empirical capture-count improvement does not establish a population success
rate, statistically significant superiority or equal true false-alarm rates.
Pointwise Wilson intervals are not simultaneous. Calibration depends on IID,
representative draws from the three declared null models. Multiple initial
candidates do not resolve squared-phase aliases, wrong-delay peaks or arbitrary
near-far interference. Within-row nuisance bit changes, carrier dynamics,
multiple target bit transitions, code drift outside the local span and the
point/aperture model mismatch remain unvalidated. This is coarse simulation
capture, without precise ranging, sustained tracking, positioning, rain or RF
validation; it establishes no research novelty or real-time throughput claim.
