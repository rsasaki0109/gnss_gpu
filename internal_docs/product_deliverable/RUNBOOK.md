# PPC FIX-Rate Predictor — Operational Runbook

**Last updated**: 2026-04-24
**Target audience**: operator running the predictor on new PPC/taroz data
**Prerequisites**: familiar with the `gnss_gpu` repository layout and Python 3.12 environment

---

## 1. When to run

Run the predictor when any of these happen:

- A new PPC/taroz run has been processed through the epoch CSV pipeline.
- You want to re-score the existing 6 runs after any upstream pipeline
  change (validationhold surrogate, refinedgrid base model).
- The last deliverable CSV is older than one month and you want to
  refresh for a review.

Do NOT run just because time has passed.  The model is frozen per
D-030; re-running without new data gives identical output.

## 2. One-shot reproduction (bundled 6-run test set)

```bash
cd /media/sasaki/aiueo/ai_coding_ws/gnss_cuda_sim_ws/gnss_gpu
python3 experiments/predict.py
python3 experiments/build_product_dashboard.py  # refresh dashboard.html
```

Takes 2-4 minutes.  Outputs land in:

- `experiments/results/ppc_window_fix_rate_model_..._alpha75_meta_run45_window_predictions.csv`
- `internal_docs/product_deliverable/route_level_fix_rate_prediction.csv`
- `internal_docs/product_deliverable/window_level_details.csv`
- `internal_docs/product_deliverable/dashboard.html` — open in a browser

## 3. Running on new data

### 3.1 Prerequisites

Before calling `predict.py`, the new run's data must already be
preprocessed through the upstream pipeline into:

- an epoch CSV with `gnss_gpu` simulator features, RINEX aggregates,
  demo5 solver state (`rtk_*`, `solver_demo5_*`) and demo5 actual FIX
  labels
- a window predictions CSV with `base_pred_fix_rate_pct`,
  `corrected_pred_fix_rate_pct`, and the full windowopt feature set
- a refinedgrid base prediction CSV (same city/run keys)

This preprocessing is out of scope for this runbook.  See
`internal_docs/plan.md` §4.1 for the upstream pipeline.

### 3.2 Command

```bash
python3 experiments/predict.py \
  --epochs-csv path/to/new_epochs.csv \
  --window-csv path/to/new_window_predictions.csv \
  --base-prefix <prefix_of_refinedgrid_predictions> \
  --results-prefix <custom_prefix_for_this_run>
```

All four paths are required when running on new data.

### 3.3 Caveat about LORO retraining

`predict.py` retrains the nested stack from scratch each call.  When
new runs are included, they participate as additional LORO outer folds.
This means "prediction for the new run" is produced while the new run
is held out and the rest of the dataset trains the stack.

There is no saved single-model artefact.  If runtime becomes a concern,
refactor `train_ppc_solver_transition_surrogate_nested_stack.py` to
support a `--no-loro --save-model` mode (future work; see D-033).

## 4. Interpreting the output

### 4.1 route_level_fix_rate_prediction.csv

One row per `(city, run)` with columns:

- `actual_fix_rate_pct` — demo5 observed FIX rate
- `baseline_pred_fix_rate_pct` — §2.2 conservative deployable baseline
- `adopted_pred_fix_rate_pct` — §7.16 adopted model prediction
- `adopted_abs_error_pp` — absolute error of the adopted prediction
- `confidence_tier` — `high` / `medium` / `low`
- `confidence_note` — reason for the tier

### 4.2 Confidence tiers

- `high`: route contains no focus-case windows; expected error <= 3 pp.
  Trustable as-is.
- `medium`: route contains hidden-high or mild false-lift windows.
  Expected error 3-8 pp.  Use with caution on boundary decisions.
- `low`: route contains false-high or strong false-lift windows.
  The adopted model partially suppresses these but residual error
  can exceed 8 pp.  Review `window_level_details.csv` for the specific
  focus cases.

### 4.3 window_level_details.csv

One row per window.  Use for diagnostics when a `low` tier run needs
drilling down.  The `focus_case_tag` column marks known failure
archetypes (`false_high`, `hidden_high`, `false_lift`,
`false_lift_mild`, `false_lift_resolved`).

## 5. What to report when predictions look off

If the adopted model's prediction on a new run deviates more than 5 pp
from actual, investigate in this order:

1. Check `window_level_details.csv` for unexpected `focus_case_tag` hits
   — are the failures in known archetypes or new?
2. Check the validationhold window summary
   (`experiments/results/ppc_validationhold_window_summary.csv`)
   for the run: are `clean_streak_s_at_start`, `hold_ready_frac`,
   `validation_block_score_p90` within the expected ranges for similar
   routes?
3. Check the baseline prediction (`baseline_pred_fix_rate_pct` column):
   is the baseline itself off, or is the adopted correction off?
4. Check the refinedgrid base CSV the adopted model built on; if that
   is off, re-run the upstream refinedgrid fit.

If the issue is a NEW failure archetype (actual and predicted disagree
for reasons not in the documented focus cases), that is a data-side
discovery worth investigating before retraining.

## 6. When to retrain / re-scan thresholds

Per D-030 / D-033:

- ≤ 1-2 new runs: do nothing.  Expect prediction quality to match the
  reported metrics.
- 3-9 new runs: re-run `predict.py`, check whether aggregate metrics
  stay within the documented ranges (run MAE ~3.2 pp, corr ~0.55).
- ≥ 10 new runs: re-scan the `hold_ready_thr` dimension
  (§7.13 / §7.14) and the residual alpha (§7.16).  The current
  thresholds were informed by Tokyo run2 behaviour and may drift.
- ≥ 10 new runs AND ≥ 1 new false-high example: revisit D-031;
  evaluate whether the high-prediction reject rule can be learned as
  an LORO model rather than stay a domain prior.
- ≥ 50 new runs: revisit D-033; consider structural paradigm changes
  (per-satellite state machine, deep learning) that were deferred.

## 7. Files that matter

| Path | Purpose |
| --- | --- |
| `experiments/predict.py` | pipeline entrypoint |
| `experiments/build_product_deliverable.py` | deliverable CSV builder |
| `experiments/build_product_dashboard.py` | HTML dashboard renderer |
| `internal_docs/product_deliverable/dashboard.html` | generated dashboard (open in browser) |
| `experiments/augment_ppc_epochs_with_validation_hold_surrogate.py` | epoch-level surrogate |
| `experiments/analyze_ppc_validation_hold_surrogate_windows.py` | window aggregation |
| `experiments/rebuild_validationhold_flag_thresholds.py` | threshold preset switcher |
| `experiments/augment_ppc_windows_with_validationhold_features.py` | feature augmenter |
| `experiments/train_ppc_solver_transition_surrogate_nested_stack.py` | strict nested stack trainer |
| `internal_docs/plan.md` | research history and rationale |
| `internal_docs/decisions.md` | D-030 through D-033 for this model |
| `internal_docs/product_deliverable/README.md` | user-facing scope & metrics |
| `internal_docs/product_deliverable/RUNBOOK.md` | this file |

## 8. Who to ask

- Model / algorithm questions: see `internal_docs/plan.md` sections 7.7
  through 7.20 first.  Adoption rationale is in `decisions.md` D-030
  through D-033.
- Upstream pipeline questions (epoch CSV / refinedgrid base): out of
  scope for this runbook.

## 9. Known operational gotchas

- The nested stack is randomness-sensitive via `--random-state`
  (default 2034).  If you need deterministic reruns on the same data,
  keep the default and do not pass `--random-state` on the command
  line.
- The `rebuild_validationhold_flag_thresholds.py` script overwrites
  its output CSV silently.  If you experiment with alternate presets,
  name the output files explicitly and do not point them at the
  canonical `ppc_validationhold_window_summary_current_tight_hold.csv`
  path unless you intend to replace it.
- Aggregate error on the dataset is +0.26 pp (not 0).  This is the
  expected calibration of the adopted model; do not try to re-bias
  it away with a post-hoc offset without understanding the per-route
  distribution first.
