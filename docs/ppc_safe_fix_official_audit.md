# PPC GNSS+IMU official-score and safe-FIX audit

This document freezes the evidence contract used for the PPC GNSS+IMU work.
Only rover/base GNSS, navigation data, and IMU are estimator inputs. LiDAR,
camera, extra routes, and reference truth are not estimator inputs.

## Scoring contract

`python/gnss_gpu/ppc_score.py` implements the driven-distance ratio whose 3D
ECEF error is at most 0.5 m. The reference trajectory always defines the full
distance denominator: an omitted, duplicate, or non-finite estimate cannot
increase the score. `experiments/evaluate_ppc_official_score.py` aligns by GPS
TOW and reports missing epochs and wrong FIX counts. The six-route suite is the
arithmetic mean of the six route distance scores; the pooled distance ratio is
reported separately as a diagnostic.

Reference data is opened only by post-estimator scoring. The native
`gnss_fgo_parity` harness also receives `--ref` because its CSV writer requires
it, but source order is `run(problem_imu, ...)` first and
`dumpEpochCsv(result, ref_rows, ...)` afterward. The reference is not passed to
the FGO solve.

## Full six-route replay

The native GTSAM fixed-lag replay used the documented shipping preset: 5 s
forward-only lag, tactical IMU preintegration, multi-frequency DD
pseudorange/carrier, TDCP, partial LAMBDA, fix-and-hold, CMC screening,
CP-hold, exception recovery, DDPR anchors, FDE, elevation-dependent variance,
FIX demotion, and surplus-satellite validation.

| Route | Existing safe K8 IMU | Native FGO + safe FIX | Public diagnostic |
|---|---:|---:|---:|
| Tokyo run1 | 65.25% | 54.47% | 65.46% |
| Tokyo run2 | 81.15% | 79.83% | 81.25% |
| Tokyo run3 | 78.21% | 67.24% | 78.38% |
| Nagoya run1 | 51.47% | 35.19% | 62.03% |
| Nagoya run2 | 28.95% | 30.68% | 30.68% |
| Nagoya run3 | 46.03% | 21.89% | 46.45% |
| Mean of six | **58.509%** | **48.214%** | **60.707%** |

The safe-FIX composer never inherits the primary trajectory's FIX label. Only
the independently gated IMU PF/FGO tracker can emit status 4. Across the safe
six-route evaluations it emitted 7,760 FIX epochs, all 7,760 were within
0.5 m, and no FIX exceeded 1 m error.

The 60.707% column is an offline public-data diagnostic, not a promotion
result. It combines known route candidates (WP176 on run1, the native result
on Nagoya run2, and the existing primary elsewhere), so the route choice has
seen the scored public data. It demonstrates remaining candidate headroom but
must not be represented as a deployable selector or sealed score. The honest
production-safe result remains 58.509% until one truth-free policy passes
route-blocked validation.

The later run1 safe-union diagnostic raises Tokyo FIX availability to
6,488/11,928 (54.393%) and Nagoya to 5,277/7,602 (69.416%), with zero false
FIX in both cities. Its official trajectory scores are 65.477% and 62.029%,
respectively. This is not yet a six-route promotion: matching active/monitor
artifacts have not been reproduced for run2/run3 under one recorded command.
The implementation and next ablations are documented in
`docs/ppc_pf_fgo_research_plan.md`.

The later frozen candidate-supply policy ranks partial-ambiguity candidates by
measurement quality before native LAMBDA enumeration. In a full-length,
truth-free six-route replay with the same top-K 8, two-epoch streak, one-epoch
gap tolerance, and native fixed-lag IMU FGO, correct safe FIX increased from
7,475/48,778 (15.325%) to 8,626/48,778 (17.684%). False FIX and false FIX above
1 m both remained zero. All six route audits and the aggregate integrity gate
passed. This is a 1,151-epoch (+15.4% relative) safe-FIX availability gain; it
does not by itself replace the 58.509% official trajectory-score result above.
The detailed ablation, per-route counts, CUDA timing, and artifact hash are in
`docs/ppc_candidate_supply_ablation.md`.

The native shipping result by itself scored 46.331% and emitted 4,629 wrong
FIX epochs, including 1,608 above 1 m. It is therefore rejected as FIX
authority. Separating its position trajectory from FIX authority removes all
wrong FIXes but does not recover the score regression on five routes.

## PF/FGO and GPU gates

The tested PF/FGO path covers multiple ambiguity basins, conditioned FGO
hypotheses, top-K native ambiguity ingestion, FFBSi history, promotion gates,
DD pseudorange/carrier factors, Doppler/TDCP, cycle-slip fail-closed behavior,
robust pseudorange loss, persistent ambiguities across DD reference changes,
CMC/FDE, and elevation-adaptive variance.

CUDA MultiSD exercised the forced GPU solver successfully. CPU/GPU acceptance
and integer decisions were identical, the maximum ECEF delta in the checked
parity artifact was 3.4e-9 m, and two repeated GPU smoke runs produced exactly
the same summary. Performance promotion is not met: on this GTX 1660 Ti the
measured GPU p95 was 34.16 ms versus 28.65 ms CPU for the small batch (1.19x
slower), and a 300-epoch audit measured 61.67 ms versus 12.82 ms (4.81x
slower). CUDA remains an opt-in parity-checked backend, not a claimed speedup.

## Promotion rules

A future candidate may replace the production-safe baseline only when all of
the following are true:

1. One frozen, truth-free runtime policy is used for every route.
2. Route-blocked development/validation improves the mean-of-six score and no
   held-out route regresses beyond its declared budget.
3. Wrong FIX above 1 m remains zero on every held-out route and injected
   outage, NLOS, cycle-slip, and satellite-loss tests fail closed.
4. CPU/GPU decisions are identical within the numerical tolerance and repeated
   runs are deterministic.
5. GPU speed is measured end-to-end and is faster than CPU for the promoted
   workload; kernel-only timing is insufficient.
6. The 70%, 78.7%, and 80% score gates are reported as failed until actual
   evidence crosses them.

There is no unused/sealed route in the current workspace. Consequently an
80% sealed claim cannot be produced honestly from these files; it requires an
externally held PPC-format route or a challenge-side evaluation unavailable to
the estimator and tuning loop.
