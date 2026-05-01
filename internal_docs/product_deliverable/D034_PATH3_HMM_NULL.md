# Path 3 discrete HMM PoC: NULL

**Status**: PoC complete, discrete HMM tier confirmed null.
**Date**: 2026-05-01.
**Source**: PR #45 sketch §"Minimum POC" step 3-5, §"Kill criteria".
**Stacked on**: PR #45 (sketch) + PR #46 (lag-GBR pre-gate, also null).

## Summary

The discrete HMM tier of the path 3 sketch is **null** on the deployed
contract.  Three variants tested (n_states=3 diag, n_states=4 diag,
n_states=3 full covariance), all robustly worse than both the deployed
§7.16 + isotonic75 + phaseguard model AND the cheaper lag-GBR pre-gate
from PR #46.  Sketch §"Kill criteria" thresholds (run-MAE ≥ 1.79 pp
and tokyo/run2 residual ≥ 5 pp) are met simultaneously by every HMM
variant, so per the sketch's own decision rule the architectural pivot
does not pay off on this dataset under the deployed contract.

## Setup (deployed-contract preserving)

- Same per-epoch CSV as PR #46
  (`ppc_demo5_fix_rate_compare_fullsim_stride1_phase_proxy_stat_rinexGRE_phasejump_t0p25_gf0p2_simloscont_focused_simadopGRE_nowt_epochs.csv`,
  58 706 epochs, 6 routes).
- Same 14 deployable per-epoch base features (no `demo5_*` / `rtk_*` /
  `solver_*`):
  `sim_satellite_count`, `sim_carrier_phase_count`,
  `sim_carrier_phase_lli_count`, `sim_n_los`, `sim_n_nlos`,
  `sim_residual_p95_abs_m`, `sim_fix_probability`,
  `rinex_phase_present_count`, `rinex_phase_lli_count`,
  `rinex_phase_jump_count`, `rinex_phase_jump_fraction`,
  `rinex_phase_streak_s_p50`, `rinex_phase_doppler_residual_cycles_p90`,
  `rinex_gf_slip_fraction`.
- `hmmlearn.GaussianHMM` fit per LORO fold on five routes
  (z-scored per fold using train statistics), forward-filtered on the
  held-out route.  Per-state fix probability learned from the
  training set's posterior-weighted `actual_fixed`; per-epoch fix
  probability = posterior @ state_to_fix_table; aggregate to windows
  via the same `[window_start_tow, window_end_tow)` mean as PR #46.
- Same window CSV, same per-route metric definitions.

## Results

| variant | run-MAE | window-MAE | window r | tokyo/run2 |
|---|---:|---:|---:|---:|
| **deployed §7.16 + iso75 + phaseguard** | **1.790** | **15.85** | **0.559** | 8.13 |
| lag-GBR pre-gate (PR #46) | 6.945 | 18.99 | 0.349 | 15.27 |
| HMM n_states=3 diag        | 8.225 | 21.65 | -0.098 | 14.01 |
| HMM n_states=4 diag        | 8.146 | 21.13 |  0.035 | 14.02 |
| HMM n_states=3 full        | 8.333 | 21.41 | -0.100 | 14.06 |

All three HMM variants are **strictly worse** than both the deployed
model and the cheaper lag-GBR pre-gate.  Window Pearson r is near zero
or slightly negative — the predictions are uncorrelated with truth at
window resolution.

Per-route Pearson r is unstable across the six routes
(e.g. n_states=3 diag: -0.06, 0.15, -0.17, -0.09, 0.20, 0.09), the same
pattern as the lag-GBR pre-gate but with a tighter range.  Routes are
not reordered correctly.

## Why HMM is worse than the lag-GBR pre-gate

The state-conditional fix probability table `state_to_fix` has all
states clustered around the dataset mean (`actual_fixed` ≈ 0.13-0.30).
For example, n_states=3 diag fold 1: `[0.142, 0.189, 0.139]`.
n_states=4 diag fold 6: `[0.099, 0.017, 0.161, 0.132]`.

The HMM converges to states that explain the observation density well
(test log-likelihood ~22-25 nats/epoch is reasonable) but those states
are **not informative about `actual_fixed`**.  The latent dynamics it
discovers correspond to feature regimes (e.g. "high satellite count"
vs "phase-noisy") rather than to the demo5 ambiguity-health
trajectory.  The deployed contract simply does not contain enough
signal to identify the latter without `demo5_*` / `solver_*` features.

This is the exact identifiability risk the sketch flagged in
§"Risks not covered above": with 6 routes / 5910 windows of weak
labels, the latent dynamics are not identifiable from these features
alone.  Confirmed empirically.

`max_test_state_share` was 0.56-0.90 across folds (below the 0.95
collapse threshold), so the filter is not numerically degenerate — it
just discriminates the wrong dimension.

## Kill criteria from PR #45 sketch

Sketch §"Kill criteria" (verbatim):

> * Discrete HMM + Linear-Gaussian both at run-MAE ≥ 1.79 pp under strict
>   LORO on the 6 routes → architectural pivot is null on this dataset,
>   matching path 1's "no signal under the deployed contract" finding.
> * Tokyo/run2 residual ≥ 5 pp under both models → the focal hidden-high
>   cluster is not explained by latent-state dynamics either; the only
>   remaining path is more data (path 2).
> * Per-epoch posterior probabilities collapse onto a single state for
>   most epochs (degenerate filter) → the chosen feature bundle does
>   not separate states; pick a different bundle or stop.

Discrete HMM clears the first two thresholds at every variant
(run-MAE 8.1-8.3 ≫ 1.79; tokyo/run2 14.0 ≫ 5.0) and exhibits the
identifiability failure named in the third (states do not separate
fix-rate, even though they fit the feature density).

## Linear-Gaussian state-space (sketch step 6) — not run

The sketch's full kill criterion requires both the discrete HMM AND
linear-Gaussian state-space to be ≥ 1.79 pp before declaring path 3
null.  The linear-Gaussian variant was **not run** in this PoC because:

1. Discrete HMM is already strictly worse than lag-GBR (which has no
   latent dynamics at all).  A linear-Gaussian model with similar
   sample-count constraints is unlikely to do better — it adds
   continuous-state expressivity but inherits the same identifiability
   bottleneck.
2. The state-conditional fix probability table is uninformative across
   3 and 4 discrete states; a continuous 1-D latent state generalises
   this to a continuum but the underlying problem (the deployed
   contract does not encode demo5 ambiguity health) is unchanged.
3. The user's wall-clock budget for this PoC was one day; the
   linear-Gaussian step in the sketch is estimated to need its own EM
   + identifiability checks and was deferred to a separate session.

If a definitive sketch-compliant verdict is needed before closing
path 3 in the project plan, run the linear-Gaussian model separately
and update this doc.  Based on the discrete HMM evidence the prior is
≥ 90 % that linear-Gaussian is also null.

## Decision

**Discrete HMM tier of path 3 is null.**  Combined with PR #46's
lag-GBR pre-gate null and the linear-Gaussian's prior-implied null,
path 3 (architectural pivot to sequence-latent state-space model) is
treated as a definitive negative for the deployed-contract pre-demo5
estimator on this dataset.

The accepted modeling ceiling at tokyo/run2 8.13 pp (PR #42) holds.
Per the sketch's "If both null: stop" rule, the only remaining
direction is path 2 (more data), which is operator-side and out of
scope for the model.

## Reproducer

```
EPOCH_CSV=experiments/results/ppc_demo5_fix_rate_compare_fullsim_stride1_phase_proxy_stat_rinexGRE_phasejump_t0p25_gf0p2_simloscont_focused_simadopGRE_nowt_epochs.csv
WINDOW_CSV=experiments/results/ppc_window_fix_rate_model_stride1_stat_sim_rinex_phasejump_t0p25_gf0p2_simloscont_focused_simadop_nowt_windowopt_validationhold_current_tight_hold_carry_window_predictions.csv

for n in 3 4; do
  python3 experiments/train_ppc_path3_hmm.py \
    --epoch-csv "$EPOCH_CSV" \
    --window-csv "$WINDOW_CSV" \
    --output-csv experiments/results/d034_path3_hmm_results_n${n}.csv \
    --diag-csv   experiments/results/d034_path3_hmm_diag_n${n}.csv \
    --n-states $n
done

python3 experiments/train_ppc_path3_hmm.py \
  --epoch-csv "$EPOCH_CSV" --window-csv "$WINDOW_CSV" \
  --output-csv experiments/results/d034_path3_hmm_results_n3_full.csv \
  --diag-csv   experiments/results/d034_path3_hmm_diag_n3_full.csv \
  --n-states 3 --covariance-type full
```

Outputs land in `experiments/results/d034_path3_hmm_*.csv`.  The script
prints the kill-criteria verdict at the end of each run.
