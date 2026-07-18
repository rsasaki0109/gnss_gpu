# WP27 — satellite integrity and absolute basin evidence

Date: 2026-07-18. Parent: `pf_only_rtk_scaleup_plan_2026_07_18.md`.

## Objective

Break the coherent-translation ambiguity that relative motion cannot observe.
Use truth-free satellite-domain absolute evidence, beginning with pivot-invariant
robust DD pseudorange consensus over the live ambiguity basins.

## First increment

1. Reconstruct constellation-local single-difference innovation coordinates
   from each conventional DD result.
2. Score all alternative satellite pivots with a robust pairwise Cauchy cost,
   trimming a bounded number of contaminated pairs.
3. Make the score invariant to the originally selected DD reference satellite.
4. Replay the score over WP25's 20,565 live basin states and measure correct
   candidate selection, oracle availability, support, and confidence.
5. Keep the score diagnostic-only until cross-run calibration passes.

## Gates

- Synthetic tests cover pivot invariance, one biased satellite, multiple
  constellations, insufficient support, and finite normalized scores.
- Tokyo run3/200 correct-live-basin selection exceeds WP25's best 10/165.
- Confident wrong selection decreases and no production FIX is introduced.
- If DDPR consensus remains ambiguous, measure whether per-satellite residual
  histories expose persistent biased/blocked modes before adding a latent
  integrity state.
