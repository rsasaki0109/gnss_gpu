# WP26 — independent relative-motion evidence

Date: 2026-07-18. Parent: `pf_only_rtk_scaleup_plan_2026_07_18.md`.

## Objective

Add causal, truth-free epoch-to-epoch motion evidence that can distinguish live
ambiguity basins without reusing their cumulative DDPR/DDCP posterior. Start
with slip-aware rover TDCP, then cross-check and fuse Doppler and IMU
preintegration.

## First increment

1. Repair the drifted PPC TDCP runner/solver API while preserving existing
   solver defaults.
2. Return displacement, post-fit residual, used/rejected satellites, and a 3D
   covariance from a robust TDCP WLS with receiver-clock nuisance state.
3. Iteratively reject satellite-specific phase discontinuities using post-fit
   residuals; do not interpret raw carrier delta as receiver motion.
4. Add external displacement/covariance support to the WP25 transition filter.
5. Record the TDCP motion stream independently, join it to the truth-free basin
   replay, and compare candidate-motion / no-motion / TDCP-motion arms.

## Gates

- Synthetic TDCP tests recover displacement/covariance and reject a slipped
  satellite while retaining a solvable clean subset.
- PPC clean-interval velocity error, NIS, use rate, and post-fit RMS are
  reported without using truth for acceptance.
- Correct-live-basin selection improves over WP25's best 10/165 result and
  high-gamma wrong selections decrease.
- Leave-one-source-out results separate TDCP, Doppler, and IMU contributions.
- No motion source is connected to production FIX until held-out false-FIX is
  `<=1%`; default trusted output remains identical.
