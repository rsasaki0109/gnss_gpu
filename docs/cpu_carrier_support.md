# Carrier search-boundary diagnostics

A subsequent [recalibrated wider-grid comparison](cpu_carrier_grid_comparison.md)
reduces the demonstrated 700 Hz carrier error and measures the additional
processing cost and held-out noise detections.

The stress matrix exposed a receiver that continues using a +500 Hz estimate
for a true +700 Hz carrier. A new diagnostic describes an estimate's support
within the configured search interval, separately from detection and tracking.
It does not change the grid, threshold, estimates or state updates.

```python
from gnss_gpu.carrier_support_cpu import assess_ca_carrier_support

result = tracker.update(iq, start_sample=start)
diagnostic = assess_ca_carrier_support(
    result.refined_candidate, tracker.threshold.search,
    boundary_margin_hz=0.1,
)
print(result.status, diagnostic.status, diagnostic.boundary_side)
```

Use the current refined candidate. A smoothed carrier or a coasting prediction
can hide a measurement hitting the search endpoint. The diagnostic's statuses:

| Status | Meaning |
| --- | --- |
| `not_estimated` | No current refined candidate; predictions are not assessed |
| `fixed_frequency` | Only one grid bin; frequency was fixed, with no resolving interval |
| `boundary` | Estimate is within the inclusive margin of an endpoint |
| `interior` | Estimate is farther than the margin from both endpoints |

The result also contains the estimated carrier, distance to the nearest
endpoint, chosen margin and boundary side (`lower`, `upper`, `both`, or None).
The default 0.1 Hz margin matches the default refinement tolerance. If a caller
changes refinement tolerance, it should choose its diagnostic margin explicitly.
This margin is a numerical policy, not an uncertainty or confidence interval.

## Audit of the saved stress matrix

The diagnostic was applied to the existing 54-scenario, 3,456-block
[stress recording](cpu_receiver_stress.md). No receiver or waveform rerun was
needed. The audit annotates the stored refined candidates and preserves the
original input files. Among fresh measurements:

| True carrier | Fresh measurements | Boundary flagged | Interior |
| --- | ---: | ---: | ---: |
| 0 Hz | 690 | 0 | 690 |
| 170 Hz | 688 | 0 | 688 |
| 700 Hz | 682 | 682 | 0 |

The diagnostic uses only the estimated carrier and search definition; true
carrier labels are used afterward for this evaluation. In this small matrix,
every used 700 Hz measurement was flagged and its conditional carrier error
remained approximately 200 Hz. The diagnostic exposes the limitation; it does
not correct the frequency or automatically reject a measurement.

A true carrier exactly at either grid endpoint also triggers. Physical-waveform
tests cover true ±500 Hz edges, outside-grid ±700 Hz carriers and interior
±170 Hz carriers. Therefore `boundary` does **not** prove that the true carrier
is outside the search, and `interior` does **not** establish accuracy. The three
paired replicas do not establish general sensitivity, specificity or integrity.

## Reproduction and checks

```bash
PYTHONPATH=python:. python experiments/audit_cpu_carrier_support.py \
  --recording-dir output/cpu-receiver-stress \
  --output output/cpu-carrier-support --boundary-margin-hz 0.1
```

`diagnostics.jsonl` stores every per-block diagnostic alongside the original
status and measurement-used flag. `summary.json` / `summary.csv` retain each
scenario's counts and conditional errors. `report.json` records hashes of the
input recording, diagnostic and audit source. The audit rejects duplicate,
missing and incomplete scenarios; it does not require access to the original
IQ buffers. The stored hashes identify inputs, rather than authenticating them.

21 diagnostic/audit tests pass, in addition to the 192 existing relevant tests.
Checks include both endpoints, inclusive and zero margins, overlapping margins,
single-bin and missing estimates, invalid input, physical waveform cases, input
preservation and duplicate-record rejection. Receiver dataclasses and numerical
decisions are unchanged, so the earlier frozen comparisons remain usable.

A wider search could address the 700 Hz failure, but requires its own complete
grid calibration. This diagnostic neither widens the search nor treats the
existing threshold as valid for a different search.
