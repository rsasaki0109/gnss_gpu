# PPC PF/FGO performance plan

This plan is restricted to rover/base GNSS, broadcast navigation data, and IMU.
It does not introduce LiDAR, camera, map, reference trajectory, or another
route as an estimator input.

## Evidence-driven priorities

1. **Reproducibility before another sweep.** Every six-route run writes its
   executable hash, complete truth-free command, and input hashes before the
   solver starts. The historical WP176 run1 command was not preserved and a
   reconstruction produced 263 rather than the recorded 284 FIX epochs over
   the first 300 Tokyo epochs. Those reconstructed results are rejected.
2. **Increase correct ambiguity-basin supply.** Keep the existing RBPF top-K
   integer hypotheses and conditional FGO, but measure oracle availability at
   each gate: native MultiSD generation, DD holdout, temporal confirmation,
   IMU consistency, and final promotion. Work on the first gate that loses a
   correct candidate; do not relax the final integrity gate to manufacture FIX.
3. **Persistent ambiguity propagation.** Preserve ambiguity nodes across the
   fixed-lag boundary and DD reference changes, with cycle-slip generation IDs.
   This follows the 2025 raw-observation RTK/INS FGO result that reports an
   11.2% 3-D improvement from ambiguity propagation and recommends a 4 s
   window. The current implementation already has persistent ambiguity keys;
   the next experiment is a frozen 4 s versus 5 s route-blocked ablation.
4. **Two-stage, causal code screening.** Stage 1 compares code innovation with
   Doppler/TDCP motion; stage 2 compares DD code with the IMU-preintegrated
   prediction. Carrier arcs remain in the graph under robust loss unless a
   cycle slip is declared. This adapts the 2025 two-stage method without its
   odometer input.
5. **Adaptive robust loss as a shadow ablation.** Add Barron's general loss for
   DD pseudorange only, choose its shape from past residuals, and keep Huber
   carrier/TDCP factors. Promote it only if city- and time-blocked validation
   improves score with zero wrong FIX. The reported adaptive Barron FGO gain is
   useful motivation, not evidence on PPC.
6. **GPU only at useful batch sizes.** Retain exact CPU/GPU integer-decision
   parity and determinism gates. Batch hypotheses/routes to amortize transfers;
   do not claim acceleration until end-to-end p95 is lower than CPU.

## Promotion matrix

One frozen policy must run on all six routes. Report mean official PPC score,
each route score, correct/false/>1 m false FIX counts, candidate attrition,
runtime p50/p95/max, hashes, and fault injection. Public run1/run2/run3 results
are development/validation evidence only because no unused sealed route exists.
The 70%, 78.7%, and 80% gates remain failed until directly measured.

## Recorded full-six GNSS+IMU evidence (2026-08-02)

The frozen truth-free policy (`top_k=8`, FIX streak 2, validation-gap
tolerance 1, native fixed-lag IMU FGO, CPU mode) completed all six PPC routes.
The executable hash was
`98b5252fe5a81659249f7344b1aae83d6564758d1799b8e6b93b23dbec5c9e57`.
Reference trajectories were opened only by the post-estimator audit processes.

| Route | Official denominator | Correct FIX | FIX rate | False FIX | >1 m false FIX | Correct-candidate oracle |
|---|---:|---:|---:|---:|---:|---:|
| Nagoya run1 | 6,748 | 910 | 13.49% | 0 | 0 | 44.36% |
| Nagoya run2 | 6,763 | 926 | 13.69% | 0 | 0 | 53.83% |
| Nagoya run3 | 3,275 | 193 | 5.89% | 0 | 0 | 37.52% |
| Tokyo run1 | 10,070 | 743 | 7.38% | 0 | 0 | 39.48% |
| Tokyo run2 | 8,383 | 1,337 | 15.95% | 0 | 0 | 59.70% |
| Tokyo run3 | 13,539 | 3,366 | 24.86% | 0 | 0 | 73.39% |
| **Total** | **48,778** | **7,475** | **15.32%** | **0** | **0** | **56.69%** |

The aggregate is recorded at
`Testing/basin_fgo_full_recorded_v2_aggregate.json`; its six input summaries
must have identical estimator hashes and policy fields. This 15.32% figure is
the conservative PF/FGO candidate path alone, not the previously reported
run1-only safe union with the library FIX path. It therefore must not be
presented as the overall library FIX rate.

The full-route result changes the next priority: candidate supply is the first
limiting gate. A correct top-8 candidate existed in only 14,815 of 26,134
evaluated epochs (56.69%). Validation retained a correct candidate in 12,753
epochs, while unique validation retained 9,109. The temporal PF promoted 7,475
and rejected the two unique-pass wrong candidates, preserving zero false FIX.
Increasing top-K through batched GPU hypothesis evaluation and improving
ambiguity proposals should be tested before relaxing validation. A truth-free
common-mode satellite leave-one-out shadow check added only two unique correct
epochs on full Nagoya run3, so it is not promoted.

The 70%, 78.7%, and 80% goals remain unproven on a six-route safe union. The
dataset has no unused sealed route, so these public-route results establish
neither a blind private score nor world SOTA.

## Promoted candidate supply plus IMU continuity (2026-08-02)

Quality-ranked PAR raised the uniform full-six result to 8,626/48,778. The
previously validated native IMU aperture was then composed with that stream.
A two-epoch validation-gap tolerance was selected using run1/run2 only and
confirmed on the untouched run3 blocks before the full replay. The frozen
policy is top-K 8 quality-ranked PAR, native IMU aperture 0.30 m with a 0.05 m
winner margin, FIX/IMU streak 2, gap tolerance 2, posterior gamma 0.99, and CPU
solver mode.

It produces 9,964/48,778 correct FIX (20.427%), zero false FIX, and zero false
FIX above 1 m. All six routes improve over the original 7,475 result, and eight
deterministic GNSS/IMU fault audits pass. This closes the current composition
milestone but not the stretch goal.

The first of those milestones is now implemented: deterministic disjoint
carrier partitions with a 0.02 dual-winner margin and a 75% detailed carrier
pass-consistency guard raise safe-FIX availability to 10,794/48,778 (22.129%)
with both false counters zero. All six routes improve, and eight GNSS/IMU fault
audits remain fail-closed. The 25% milestone (at least 12,195 correct epochs)
therefore remains 1,401 epochs away. The quality-ranked candidate oracle
contains 15,069 correct-candidate epochs, leaving 4,275 epochs of theoretical
headroom above current authority.

The next work should target causal IMU motion-consensus likelihood and
persistent ambiguity evidence across DD-reference changes, selected on run1/2
and evaluated once on run3 before another full replay. Candidate count or FIX
gates must not be relaxed without route-blocked and fault-injection evidence.

The causal IMU motion-consensus milestone is now complete. A two-partition
median offset predictor, using only prior safe FIX anchors, raises authority to
11,031/48,778 (22.615%) with zero false FIX. Development run1/run2 gained 197
epochs; the frozen run3 evaluation gained 40. Eight fault cases remained
fail-closed, and a repeated full replay was decision-identical. The 25% target
is now 1,164 epochs away, while the 15,069-epoch candidate oracle leaves 4,038
epochs of theoretical headroom.

The official trajectory score remains 58.521912% before and after the motion
consensus because all 237 newly authoritative positions replace primary FLOAT
epochs already inside 0.5 m. The next milestone must therefore prioritize the
FLOAT trajectory rather than only integer labels: a truth-free causal selector
between the existing primary, native FGO, and PF conditional positions, using
innovation/health features frozen on run1 and evaluated on run2/run3. Persistent
ambiguity evidence across DD-reference changes remains the next FIX-side track.

That FLOAT milestone is now complete. A 500-epoch causal health window uses the
`gnss_fuse` candidate only when its native fixed fraction is at least 90%, or
when the current candidate itself is fixed; candidate status is always emitted
as FLOAT and the safe IMU PF/FGO tracker remains the sole FIX authority. The
policy was selected on run1, validated without retuning on run2, and evaluated
once on run3. All six route scores improved, including both sealed routes. The
official mean rises to 59.222040% (+0.700128 points, +389.68 m passing
distance), with the same 11,031 safe FIX epochs and zero false FIX. Release
replay was position-hash identical on every route and had 28.37--50.31 ms p95
candidate runtime. The 70% target remains 10.777960 points away; details and
the reproducible truth boundary are in `docs/ppc_causal_float_selector.md`.

## Primary references

- T. Suzuki, *Open-Source Factor Graph Optimization Package for GNSS: Examples
  and Applications* (2025), https://arxiv.org/abs/2502.08158 and
  https://github.com/taroz/gtsam_gnss
- *Factor Graph Optimization-Based RTK/INS Integration With Raw Observations
  for Robust Positioning in Urban Canyons* (2025),
  https://ieeexplore.ieee.org/document/11029173
- B. Song et al., *Two stage GNSS outlier detection for factor graph
  optimization based GNSS-RTK/INS/odometer fusion* (2025),
  https://arxiv.org/abs/2510.00524
- E. Ahmadi et al., *Adaptive Factor Graph-Based Tightly Coupled GNSS/IMU
  Fusion for Robust Positioning* (2025), https://arxiv.org/abs/2511.23017
- W. Wen and L.-T. Hsu, GraphGNSSLib,
  https://github.com/weisongwen/GraphGNSSLib
- GTSAM fixed-lag smoothing and IMU preintegration,
  https://github.com/borglab/gtsam
