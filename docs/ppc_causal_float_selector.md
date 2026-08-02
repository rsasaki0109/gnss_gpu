# PPC causal FLOAT trajectory selector

## Scope and decision contract

This milestone improves the complete PPC trajectory while preserving the
existing IMU PF/FGO tracker as the only FIX authority. It uses only rover/base
GNSS, broadcast navigation data, and IMU. LiDAR, camera, map data, and reference
positions are not estimator inputs.

The output decision order is fixed:

1. Preserve every safe tracker FIX position and Status=4 exactly.
2. At other epochs, use the `gnss_fuse` candidate when its current native
   status is fixed, but emit that position as Status=3.
3. Also use candidate FLOAT positions when at least 90% of the previous 500
   available candidate epochs had native fixed status.
4. Otherwise retain the existing safe primary FLOAT position. Clear candidate
   health across gaps longer than 1 second.

Candidate Status=4 is therefore only a causal health observation. It never
becomes final FIX authority. The composer verifies the safe-output summary and
candidate manifest hashes before reading either trajectory.

## Frozen candidate generator

`experiments/run_ppc_float_candidates.py` invokes one configuration on every
route:

```text
--preset low-cost
--library-fix-integrity-gate
--integrity-disjoint-ensemble
--integrity-satellite-par-consensus-promotion
--integrity-satellite-par-surplus-validation
--integrity-satellite-par-surplus-min-fixed-pairs 8
--integrity-satellite-par-surplus-aperture-lt1 0.1
--integrity-satellite-par-surplus-aperture-1to2 0.1
--integrity-satellite-par-surplus-aperture-gt2 0.1
--integrity-satellite-par-acquisition-streak 1
```

Tokyo uses lever arm `0.31,0,0.55`; Nagoya uses
`0.593,0.670,1.216`. The promoted executable SHA-256 is
`b1d7c0172401063a6b6e0a3b40bb782485771beb3e9715aca1cce1decce5796e`.
The runner writes a pre-run manifest containing the full command, executable
hash, and rover/base/navigation/IMU hashes before starting the estimator.
Reference paths cannot be passed to this process.

The historical WP176 run1 trajectory was useful as a diagnostic candidate,
but its command was not recorded and the reconstructed first-300-epoch run did
not reproduce it. It is not used in the promoted result.

## Selection protocol

- Development: Tokyo/Nagoya run1 only. A coarse grid over 25, 50, 100, 250,
  500, and 1000 epoch windows and 0.3 through 0.9 fixed fractions selected the
  500/0.9 policy. It improved all ten contiguous run1 audit blocks.
- Validation: the frozen policy was run once on both run2 routes. Both route
  scores improved and safe FIX positions remained identical. Two time blocks
  degraded, including -4.10 points in one Tokyo block, so block stability is a
  recorded limitation rather than a claimed success.
- Sealed evaluation: the same policy was run on run3 without tuning after
  validation. Estimator and selected-output hashes were fixed before the
  post-estimator audit opened a reference trajectory. Both sealed routes
  improved.

## Measured development and validation results

The metric is the official forward-only traveled-distance fraction with 3-D
error at most 0.5 m. Deltas are percentage points.

| Route | Previous safe | Selected | Delta | Final FIX | False FIX | >1 m false FIX |
|---|---:|---:|---:|---:|---:|---:|
| Tokyo run1 | 65.248163% | 67.448316% | +2.200153 | 1,257 | 0 | 0 |
| Nagoya run1 | 51.453362% | 52.639691% | +1.186329 | 1,160 | 0 | 0 |
| Tokyo run2 | 81.169169% | 81.183315% | +0.014147 | 1,907 | 0 | 0 |
| Nagoya run2 | 28.946183% | 28.964625% | +0.018442 | 1,344 | 0 | 0 |
| Tokyo run3 | 78.282034% | 78.911778% | +0.629744 | 5,154 | 0 | 0 |
| Nagoya run3 | 46.032560% | 46.184515% | +0.151955 | 209 | 0 | 0 |

The run1 mean rises from 58.350763% to 60.044003%. The run2 mean rises from
55.057676% to 55.073970%. This is a trajectory improvement, not an increase in
safe FIX availability.

Across all six routes, the official arithmetic-mean score rises from
58.521912% to **59.222040%** (+0.700128 points), adding 389.68 m of passing
distance. The final stream retains all 11,031 safe FIX epochs with zero false
FIX and zero false FIX above 1 m. It remains 10.777960 points below 70%,
19.477960 points below 78.7%, and 20.777960 points below 80%.

The Release replay produced position files with the same SHA-256 as the
development/validation/sealed score inputs on every route. Candidate runtime
p95 was 28.37--50.31 ms across routes. All route p95 values passed 100 ms;
99.825% or more epochs passed 100 ms on every route. The two isolated maxima
above 100 ms were 130.58 ms on Tokyo run2 and 118.66 ms on Tokyo run3.

## Reproduction

Generate truth-free candidates (routes may be restricted during development):

```powershell
python experiments/run_ppc_float_candidates.py `
  --binary third_party/gnssplusplus/build-win/apps/Release/gnss_fuse.exe `
  --expected-binary-sha256 b1d7c0172401063a6b6e0a3b40bb782485771beb3e9715aca1cce1decce5796e `
  --dataset-root E:/datasets/PPC-Dataset-data `
  --output-root Testing/ppc_float_candidates --jobs 2
```

Compose one route only after its safe summary and candidate manifest exist:

```powershell
python experiments/compose_ppc_causal_float_selector.py `
  --route <tokyo_run1> `
  --safe-output <safe_output.csv> --safe-summary <safe_output.json> `
  --float-candidate-pos <float_candidate.pos> `
  --candidate-manifest <run_manifest.json> `
  --output <selected.csv> --summary <selected.json>
```

Only the separate audit command receives reference data:

```powershell
python experiments/audit_ppc_causal_float_selector.py `
  --safe-output <safe_output.csv> --safe-summary <safe_output.json> `
  --selected-output <selected.csv> --selector-summary <selected.json> `
  --reference <reference.csv> --output <audit.json>
```

## Safety and limitations

The audit fails if the selected FIX set or any selected FIX position differs
from the safe input. Tests also cover malicious candidate FIX positions,
non-finite values, health reset after an outage, duplicate safe epochs, and
manifest tampering. Candidate generation remains CPU-bound; two routes are run
in parallel for throughput. The already validated CUDA PF hypothesis reduction
and cuSOLVER FGO paths remain unchanged, and no GPU speedup is claimed for this
linear-time composer.

The public PPC routes are not blind private data. Even if the final public
score improves, this evidence cannot establish world SOTA. The 70%, 78.7%, and
80% targets are reported strictly from the final six-route official audit.
Machine-readable route scores, output hashes, latency, and target gaps are
committed in `docs/ppc_causal_float_selector_evidence.json`.
