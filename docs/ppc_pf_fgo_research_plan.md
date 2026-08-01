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
