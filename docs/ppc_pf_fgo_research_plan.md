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

The next pre-registered milestone is 25% safe-FIX availability (at least
12,195 correct epochs) with both false counters still zero. The quality-ranked
candidate oracle contains 15,069 correct-candidate epochs, leaving theoretical
headroom, but only 9,964 are safely promoted. Work should therefore target
truth-free disambiguation when multiple candidates pass: two disjoint satellite
holdouts, IMU motion-consensus likelihood, and persistent ambiguity evidence
across DD-reference changes. Candidate count or FIX gates must not be relaxed
without route-blocked and fault-injection evidence.

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
