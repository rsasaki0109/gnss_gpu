# D-034 path 3 pre-gate: null result on lag-feature GBR

**Status**: null — temporal context with simple lag features does not lift
the deployed run-MAE.  Path 3 (HMM / state-space) prior is weakened but
not conclusively ruled out.
**Date**: 2026-05-01.
**Source**: `D034_PATH3_SKETCH.md` POC step "If discrete HMM null: try
Linear-Gaussian state-space.  If both null: stop."  This evaluation runs
a *cheaper* gate before either full HMM or hand-rolled state-space, to
test whether per-epoch temporal context contains usable signal at all
under the deployed contract.

## Why a pre-gate

PR #45's sketch hypothesised that window aggregation discards the
temporal trajectory needed to predict hidden-high cases like
tokyo/run2 windows 23-27 (carry-over success).  The natural full
test is an HMM on per-epoch features.  ``hmmlearn`` is not currently
installed in this environment and an unscoped install is not
authorised; before paying the cost of a hand-rolled HMM
implementation, we run the simplest possible "does temporal context
help?" test that uses only ``sklearn`` (already installed).

Lag-feature GBR is strictly weaker than an HMM on the same features
(no explicit latent state, no transition dynamics), so it is a
necessary condition: if lag features alone do not lift run-MAE, an
HMM is unlikely to either, and the path-3 kill criterion ("discrete
+ linear-Gaussian both ≥ 1.79 pp") applies *a fortiori*.  If lag
features show a signal, the case for installing ``hmmlearn`` and
running a real state-space POC strengthens.

## Setup

* Per-epoch input CSV
  (`ppc_demo5_fix_rate_compare_fullsim_stride1_phase_proxy_stat_rinexGRE_phasejump_t0p25_gf0p2_simloscont_focused_simadopGRE_nowt_epochs.csv`):
  58 706 epochs across 6 routes, 339 columns.
* 14 deployable per-epoch base features: simulator outputs
  (`sim_satellite_count`, `sim_n_los`, `sim_residual_p95_abs_m`,
  `sim_fix_probability`, etc.) and RINEX phase/cycle-slip quality
  (`rinex_phase_jump_*`, `rinex_phase_streak_s_p50`,
  `rinex_phase_doppler_residual_cycles_p90`, `rinex_gf_slip_fraction`).
  Strictly excludes ``demo5_*`` / ``rtk_*`` / ``solver_*`` to preserve
  the deployed contract.
* Rolling windows: 30 s and 120 s; statistics: mean, std, max; all
  shifted by one epoch (past-only).
* Total feature dimension: 14 base + 14 × 2 × 3 lag = **98 features**.
* Target: per-epoch ``actual_fixed`` ∈ {0, 1}.
* Model: ``sklearn.GradientBoostingRegressor`` with library defaults
  (no hyperparameter tuning).
* CV: strict leave-one-route-out across ``(city, run)``.
* Aggregation: per-epoch fix probability → per-window predicted FIX
  rate (mean of probabilities across epochs in the window's
  ``[start_tow, end_tow)``) → per-route weighted by
  ``sim_matched_epochs``.

## Results

### Aggregate

| metric | pre-gate (lag GBR) | deployed §7.16+iso+phaseguard | delta |
| --- | ---: | ---: | ---: |
| run-MAE | **6.945 pp** | 1.790 pp | **+5.16** |
| window-MAE | 18.99 pp | 15.85 pp | +3.14 |
| window Pearson r | 0.349 | 0.559 | -0.21 |

### Per route

| route | actual | pred | err | window MAE | route r |
| --- | ---: | ---: | ---: | ---: | ---: |
| nagoya/run1 | 11.5 | 20.6 |  9.13 | 18.33 |  0.47 |
| nagoya/run2 | 16.2 |  9.0 |  7.20 | 17.71 |  0.05 |
| nagoya/run3 |  7.9 |  9.3 |  1.39 | 10.61 | -0.27 |
| tokyo/run1  | 10.9 | 19.5 |  8.60 | 18.11 |  0.08 |
| tokyo/run2  | 29.0 | 13.8 | **15.27** | 24.43 |  0.52 |
| tokyo/run3  | 24.0 | 23.9 |  0.10 | 20.41 |  0.59 |

### Diagnosis

* The model **under-predicts tokyo/run2 by 15.27 pp** — *worse* than
  the deployed model's 8.13 pp under-prediction (the focal hidden-high
  case PR #42 declared an accepted ceiling).  Adding temporal context
  in this specific architecture made tokyo/run2 worse, not better.
* Per-route Pearson r is unstable (-0.27 on nagoya/run3, +0.59 on
  tokyo/run3): the model is not learning a route-transferable signal.
* nagoya/run1 and tokyo/run1 are over-predicted (~9 pp); nagoya/run2
  and tokyo/run2 are under-predicted (~7 and ~15 pp).  This is the
  pattern of a model that has fit the dataset mean and is unable to
  separate good-conditions-good-fix from bad-conditions-good-fix.

## Implication for path 3

The pre-gate is null: lag-feature GBR at run-MAE 6.95 pp is well
above the path-3 sketch's kill criterion of run-MAE ≥ 1.79 pp.

This shifts the prior toward path 3 being null but does *not*
conclusively rule it out.  Reasons an HMM could still help where lag
features did not:

1. **Explicit latent dynamics.**  An HMM models transitions between
   discrete regimes and integrates the entire past sequence into a
   posterior state estimate, where lag features just summarise raw
   observations within fixed windows.
2. **Sample efficiency.**  An HMM with 3-4 states has O(K²) transition
   parameters plus per-state emissions, whereas the GBR fits ~98
   features × ensemble depth on 50 K samples — different overfitting
   profile.
3. **Interpretability check.**  An HMM's posterior state assignments
   can be inspected; if they map to physical regimes (open sky /
   canyon / viaduct), the model class is at least consistent with
   the data even if metrics don't beat the deployed reference.

Reasons the gate result is genuinely informative:

1. **Same feature signal.**  Whatever signal an HMM would extract from
   lag-shifted ``sim_*``/``rinex_*`` features is present in the GBR's
   input.  If the GBR cannot find a transferable signal, the HMM has
   to rely entirely on its inductive bias, which may not survive
   contact with the small sample (5 910 epochs / route, 6 routes).
2. **Focal failure unchanged.**  Tokyo/run2 went from 8.13 pp under
   deployed to 15.27 pp under the gate.  A successful path 3 model
   would need to fix tokyo/run2 specifically, and lag features failed
   in the *opposite direction*.

## Recommendation

Three forward options, in increasing cost and decreasing prior of
success:

1. **Stop here.**  Accept path 3 null on this dataset, treat the
   deployed §7.16 stack at 1.79 pp as the practical ceiling under
   the deployed contract, and re-prioritise toward path 2 (more PPC
   data) when operator-side capacity allows.  Lowest cost,
   acknowledges the cumulative null-result evidence (PR #44 closing
   path 1, this PR weakening path 3).
2. **Hand-rolled 3-state HMM in numpy.**  ~1 day work, no install
   needed, gives a real test of the latent-state inductive bias.
   Honest expectation: also null, but the result would be definitive
   for the discrete-HMM family.
3. **Authorise ``hmmlearn`` + ``pykalman`` install** and run the full
   path-3 POC as sketched.  ~2-3 day work.  Recommended only if the
   user explicitly prefers a strong negative result over a weakened
   prior.

The author's vote is **option 1**, conditional on the user's appetite
for confirmation effort.  Option 2 is the right second-best if the
weakened-but-not-conclusive prior bothers anyone.

## Reproduce

```bash
EPOCH_CSV=experiments/results/ppc_demo5_fix_rate_compare_fullsim_stride1_phase_proxy_stat_rinexGRE_phasejump_t0p25_gf0p2_simloscont_focused_simadopGRE_nowt_epochs.csv
WINDOW_CSV=experiments/results/ppc_window_fix_rate_model_stride1_stat_sim_rinex_phasejump_t0p25_gf0p2_simloscont_focused_simadop_nowt_windowopt_validationhold_current_tight_hold_carry_window_predictions.csv

python3 experiments/train_ppc_path3_pregate_lag_features.py \
    --epoch-csv "$EPOCH_CSV" \
    --window-csv "$WINDOW_CSV" \
    --output-csv experiments/results/d034_path3_pregate_results.csv
```

Runtime ≈ 15.5 min on the development host (single CPU; six folds, 50
K-sample GBR each).
