# D-034 path 1 LORO evaluation: solver-state lightweight wrapper

**Status**: positive result — path 1 lifts the 8.13 pp tokyo/run2 ceiling.
**Caveat**: contract change required (predictor becomes a *post-demo5* QA
tool rather than a *pre-demo5* estimator).
**Date**: 2026-05-01.

## Background

PR #42 (`docs: tokyo/run2 8.13pp residual is an accepted modeling ceiling`)
flagged three out-of-scope paths to lift the ceiling.  PR #43 introduced
the opt-in interface for path 1 (`experiments/solver_state_wrapper.py`,
six-column allowlist).  This document records the LORO experiment that
followed, executed via
`experiments/train_ppc_solver_state_wrapper_loro.py`.

## Setup

* Input: the same 197-window CSV consumed by the deployed §7.16 stack
  (`ppc_window_fix_rate_model_..._validationhold_current_tight_hold_carry_window_predictions.csv`).
* Model: a single sklearn `GradientBoostingRegressor` with library
  defaults (no hyperparameter tuning), trained on
  `actual_fix_rate_pct` directly.
* CV: strict leave-one-route-out across `(city, run)`, six folds.
* Three feature variants:

| variant | features | description |
| --- | --- | --- |
| `baseline_no_solver_state` | 5873 | sim + RINEX + meta features after `_is_metadata_or_label`; matches the deployed contract. |
| `treatment_with_curated_six` | 5879 | baseline plus the six curated columns surfaced via `SolverStateWrapper`. |
| `curated_six_only` | 6 | the curated columns alone, no other features. |

The architecture is intentionally simple so that the only thing varying
between baseline and treatment is the curated-six feature set.  No
isotonic calibration, no phase-delta guard, no transition-surrogate
classifier stack.  Hyperparameter tuning was deliberately omitted to
avoid the post-hoc threshold-guard / leak failure mode flagged in past
PPC experiments.

## Results

### Aggregate

| variant | run-MAE (pp) | window-MAE (pp) | window Pearson r |
| --- | ---: | ---: | ---: |
| `baseline_no_solver_state` | 12.389 | 25.068 | 0.007 |
| `treatment_with_curated_six` | **1.431** | **5.052** | **0.925** |
| `curated_six_only` | 1.658 | 4.764 | 0.927 |
| deployed §7.16 + isotonic75 + phaseguard | 1.790 | 15.847 | 0.559 |

The deployed reference is *not* an apples-to-apples comparison: it uses
a transition-surrogate classifier stack with isotonic calibration and a
RINEX phase-delta guard, all of which are absent from the variants
above.  It is included only as a contemporaneous benchmark.

### Per route

| route | actual | baseline pred | baseline err | treatment pred | treatment err | curated-six pred | curated-six err | deployed err (PR #42) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| nagoya/run1 | 11.5 | 44.9 | 33.4 | 11.1 | 0.4 | 10.0 | 1.5 | 0.23 |
| nagoya/run2 | 16.2 | 19.5 | 3.3 | 15.1 | 1.0 | 14.7 | 1.5 | 1.44 |
| nagoya/run3 |  7.9 | 15.9 | 8.1 |  9.3 | 1.4 | 10.4 | 2.5 | 0.17 |
| tokyo/run1 | 10.9 | 20.6 | 9.7 | 10.2 | 0.8 |  9.7 | 1.2 | 0.50 |
| tokyo/run2 | 29.0 | 17.5 |11.5 | 25.6 | 3.4 | 27.5 | **1.50** | **8.13** |
| tokyo/run3 | 24.0 | 15.6 | 8.4 | 25.6 | 1.6 | 25.8 | 1.8 | 0.28 |

Tokyo/run2 — the previously-accepted-ceiling case — drops from 8.13 pp
under the deployed model to 3.39 pp under `treatment_with_curated_six`
and **1.50 pp under `curated_six_only`**.  This is the ceiling lift PR
#42 path 1 promised.

## Honest caveats

### The signal is near-tautological

The curated six are demo5's own ambiguity-fix-state outputs:
`solver_demo5_ratio_*` are the per-window aggregate ambiguity ratio,
`rtk_lock_p90_p50*` measure the persistence of carrier-phase locks.
demo5's actual FIX rate is, almost by construction, highly correlated
with these.  The curated_six_only Pearson r of 0.93 (window-level)
reflects this.

In other words, predicting "did demo5 succeed at window N" using
"demo5's confidence at window N" is barely a prediction problem; it
is closer to a regression of self-reported confidence onto
self-reported success.  The 1.50 pp tokyo/run2 lift is real, but it
is the lift of a tool that runs *after* demo5, not before.

### Contract change implications

The deployed contract (README §1) reads:

> "The model does not use any demo5 solver internals (`rtk_*`,
> `solver_demo5_*`) as runtime features; those columns appear only as
> classification targets during training."

Adopting path 1 reverses this for the curated six.  The deployment
implication is one of:

1. **Run demo5 first, then run the predictor as a QA tool**.  This
   makes the predictor a *post-demo5 quality validator* — useful for
   e.g. "out of these 6 routes I just processed, which look like they
   have hidden-high or false-high windows worth a second look".  Both
   the offline-QA and route-ranking use cases (README §2) collapse
   into this scenario, since both already assume demo5 has been run.
2. **Estimate the curated six from RINEX/sim alone** (a real
   "lightweight wrapper" rather than direct exposure).  This is a
   different, much harder research path; it would not benefit from
   the 1.43 pp metric reported here.
3. **Drop the predictor entirely for the post-demo5 case**, since
   demo5 already reports per-window FIX directly.  The predictor's
   value is the smoothing / aggregation behaviour and the route-level
   summary, not novel information.

Whichever route is taken, the 1.79 pp pre-demo5 estimator is *not*
made obsolete by this result; it remains the only model that produces
a route-level FIX rate from deployable features alone.

### `curated_six_only` ≈ `treatment_with_curated_six`

The two treatment variants are within 0.23 pp run-MAE of each other.
The 5873 deployable features add essentially nothing over the curated
six.  This is consistent with the curated six dominating the regression
trees in the GBR — once the model can read demo5's confidence
directly, the rest of the feature space is noise relative to it.

A corollary: the elaborate transition-surrogate-stack architecture
(§7.16) is unnecessary if the curated six are exposed as runtime
features.  A six-feature GBR matches its run-MAE (1.66 vs 1.79 pp).
The deployed stack's value is its window-level smoothing
(`window_pearson_r = 0.559` and `window_mae = 15.85` pp); this
coarseness is the price paid for not having the curated six as
features.

### Possible leakage we did not check

* The curated six are derived per-window inside each route.
  Cross-route leakage is impossible (LORO splits at the route
  boundary).  Within a held-out route, a predictor that uses the same
  route's solver state will of course do well at predicting that
  route's outcomes — that's the tautology above, not a CV bug.
* Hyperparameters: GBR defaults only.  No tuning, so no test-fold
  selection bias.

## Recommendation

Adopt path 1 as a *parallel* model class — a "post-demo5 QA" predictor
that runs after demo5 and surfaces per-window confidence aggregates.
Do not retire the deployed §7.16 stack; the two address different use
cases.  Document the contract trade-off explicitly in
`product_deliverable/README.md`.

This positive result resolves D-034's open item ("Solver-state wrapper
を opt-in する research training script の設計") for the experimental
phase.  Whether to *deploy* the post-demo5 model and whether/how to
expose its outputs in the deliverable bundle (`route_level_fix_rate_prediction.csv`,
`window_level_details.csv`) are product questions left to the
maintainer.

## Reproduce

```bash
INPUT=experiments/results/ppc_window_fix_rate_model_stride1_stat_sim_rinex_phasejump_t0p25_gf0p2_simloscont_focused_simadop_nowt_windowopt_validationhold_current_tight_hold_carry_window_predictions.csv

python3 experiments/train_ppc_solver_state_wrapper_loro.py \
    --input-csv "$INPUT" \
    --output-csv experiments/results/d034_loro_results.csv
```

Runtime ~3 minutes 15 seconds on the development host (single core,
no GPU).
