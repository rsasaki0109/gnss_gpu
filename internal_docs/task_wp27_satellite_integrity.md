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

## Online diagnostic increment

1. Add an opt-in causal arm to `exp_wp23b_basin_ar.py`.
2. Feed only multi-pivot DDPR scores into the observation term; never reuse the
   basin PF's cumulative/current DDPR/DDCP likelihood.
3. Feed robust TDCP displacement and covariance into the transition term.
4. Record both sources under the separate `integrity_lineage` evidence target.
5. Keep the selected candidate, gamma, and error diagnostic-only. The existing
   output position, MAP posterior, and trusted commit policy must be bit-identical
   to a control run.

Online gate: Tokyo run3/200 must reproduce the offline 91/200 diagnostic result,
accept 199 TDCP intervals and 40 DDPR anchors, pass the evidence audit, and have
zero operational-field mismatch plus identical trajectory SHA-256.

## Frozen three-run calibration audit

Apply the run3-selected parameters unchanged to Tokyo run1/run2/run3 first 200
epochs. Record live-basin oracle availability, non-chaining 0.5 m position-ball
mass, Brier score, 10-bin ECE, existing float/DDPR guard distances, and Wilson
95% false-rate upper bounds. The diagnostic grid is fixed to gamma
`0/.05/.1/.15/.2/.25/.3`, dwell `1/3/5/10/20`, float guard `.25/.5/1 m`, and
DDPR guard `.5/1/1.75 m`. This grid may reject promotion; it must not select a
run-local production policy.

Promotion gate: one common configuration must retain nonzero correct coverage
on every run, observed false rate `<=1%`, and a Wilson 95% upper bound `<=1%`.
Otherwise keep the arm diagnostic and proceed to satellite-specific failure
attribution.

## Satellite attribution and latent-cost arm

At each 1 Hz anchor, compute pivot-invariant incident pair cost per satellite at
the causal DDPR guard position. Measure every leave-one-satellite-out score only
for truth-joined attribution. The deployable diagnostic rule may exclude the
single maximum-cost satellite; it may not use the oracle exclusion identity.

Also evaluate a causal EMA of satellite cost as the minimal persistent
`clean/biased` state. EMA memory is development-only until frozen transfer to
Nagoya. Both arms remain disconnected from output/FIX and must preserve the
existing evidence and replay audits.
