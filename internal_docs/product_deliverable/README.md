# PPC demo5 FIX-Rate Predictor — Product Deliverable

**Status**: internal research prototype, route-level deliverable
**Last updated**: 2026-04-24
**Adopted model**: §7.16 `transition_surrogate_nested_et80_validationhold_current_tight_hold_carry_alpha75_meta_run45`
**Source plan**: `internal_docs/plan.md` sections 7.7 through 7.16

---

## 1. What this predicts

Per-route predicted RTKLIB demo5 FIX rate on PPC/taroz GNSS data, given
only deployable simulator (`gnss_gpu`) features and RINEX observations.

The model does **not** use any demo5 solver internals (`rtk_*`,
`solver_demo5_*`) as runtime features; those columns appear only as
classification targets during training.

## 2. Scope and confidence

### In-scope use cases (confidence `high` or `medium`)

- Offline QA: after a taroz/PPC run, predict the FIX rate the data would
  produce under RTKLIB demo5.
- Route ranking: compare two routes and predict which would give a
  higher demo5 FIX rate.
- Dataset-level aggregate prediction: across a set of similar runs,
  predict the overall FIX rate within a few percentage points.

### Out-of-scope use cases

- Individual 30 s window FIX-rate predictions.  Typical window weighted
  MAE is 17 pp and tail errors reach 60+ pp on known failure cases.
- Real-time / online inference.  The current pipeline is an offline
  batch chain; see section 6 below.
- Data sources different from taroz/PPC Tokyo/Nagoya.  The model was
  tuned on 6 runs across 2 cities; generalisation to other
  cities/environments is untested.
- Safety-critical decisions.  This is a statistical estimate with
  documented failure modes in section 5.

## 3. Metrics (strict nested LORO on 6 runs / 197 windows)

Under product-relevant metrics:

| metric | adopted §7.16 | baseline §2.2 |
| --- | --- | --- |
| run MAE | **3.202 pp** | 4.436 pp |
| window correlation | **0.551** | 0.401 |
| overall aggregate error | **+0.26 pp** on 17.90 % dataset FIX rate | — |
| window weighted MAE | 17.087 pp | 18.046 pp |

Per-route error on the test set (see
`route_level_fix_rate_prediction.csv`):

| city / run | actual | predicted | \|err\| | confidence |
| --- | --- | --- | --- | --- |
| nagoya / run1 | 11.5 % | 14.5 % | 3.03 pp | medium |
| nagoya / run2 | 16.2 % | 21.0 % | 4.80 pp | low |
| nagoya / run3 |  7.9 % |  6.8 % | 1.04 pp | high |
| tokyo  / run1 | 10.9 % | 12.1 % | 1.15 pp | low |
| tokyo  / run2 | 29.0 % | 20.8 % | 8.25 pp | low  |
| tokyo  / run3 | 24.0 % | 25.0 % | 0.94 pp | low  |

Confidence tiers are auto-detected per route from the presence of
focus-case windows (see `build_product_deliverable.py::_classify_window`
thresholds).  A `high` tier means no material failure-mode window is
present; `medium` means the route contains hidden-high windows
(actual high, corrected under-predicts by 40+ pp); `low` means the
route contains at least one false-high or false-lift window.

Three of six routes are within 3 pp of actual (nagoya/run3, tokyo/run1,
tokyo/run3); a fourth (nagoya/run1 at 3.03 pp) is borderline.  Five of
six are within 5 pp.  The one exception is tokyo/run2 at 8.25 pp, which
contains the Tokyo run2 w7-w9 false-high cluster documented in section
5.

## 4. Inputs, outputs, and files

### Inputs required at training time

- Processed PPC epoch CSV with `gnss_gpu` simulator features, RINEX
  aggregates, demo5 solver state (for targets), and demo5 actual FIX
  labels.
- Refined-grid base prediction CSV
  (`..._windowopt_baseerror15_refinedgrid_window_predictions.csv`).

### Pipeline outputs at inference time

- `route_level_fix_rate_prediction.csv` — one row per `(city, run)`:
  actual vs predicted FIX rate, aggregate error, qualitative
  confidence tier.
- `window_level_details.csv` — per-window predictions with focus-case
  annotations, intended for diagnostics.

### Files shipped with this deliverable

- `internal_docs/product_deliverable/README.md` — this file.
- `internal_docs/product_deliverable/RUNBOOK.md` — operational runbook.
- `internal_docs/product_deliverable/route_level_fix_rate_prediction.csv`
- `internal_docs/product_deliverable/window_level_details.csv`
- `internal_docs/product_deliverable/dashboard.html` — self-contained
  HTML report (open in a browser) with metrics summary, per-run bar
  chart, actual-vs-predicted scatter, and focus-case detail table.
- `internal_docs/product_deliverable/plots/` — static PNG figures per
  run that overlay the predicted FIX rate onto the actual demo5
  per-epoch FIX/NO-FIX trajectory:
  - `{city}_{run}_timeseries.png`: predicted vs actual FIX rate over
    elapsed seconds, with the rolling per-epoch FIX fraction and a
    quality-flag (Q) strip.
  - `{city}_{run}_trajectory.png`: lat/lon scatter coloured by demo5 Q
    (FIX = green, FLOAT = orange, DGPS = purple), with per-window
    predicted FIX % annotated at each window start.
  - `summary_grid.png`: 2x3 panel overview of all 6 runs.
- `experiments/build_product_deliverable.py` — script that produces the
  two CSVs above from the adopted model's `window_predictions.csv`.
  Focus-case windows are detected by threshold-based classification
  (`_classify_window`), so new runs with similar failure archetypes
  are tagged automatically without updating a hardcoded list.
- `experiments/build_simulation_vs_actual_plots.py` — renders the PNG
  figures under `plots/` from the adopted predictions and the demo5
  .pos files (`experiments/results/demo5_pos/<city>_<run>/rtklib.pos`).
  When the epoch classifier predictions CSV is present it also
  overlays a blue line showing the per-epoch P(FIX) (15 s smoothed).
- `experiments/train_ppc_epoch_fix_classifier.py` — experimental
  per-epoch classifier (null result, see
  `product_deliverable/EPOCH_CLASSIFIER.md`); NOT part of the adopted
  pipeline.  Kept to produce the blue diagnostic line on the
  time-series plots.
- `internal_docs/product_deliverable/EPOCH_CLASSIFIER.md` — null-result
  write-up for the epoch classifier and future-work notes.
- `experiments/_common.py` — shared helpers used across the experiment
  scripts (e.g. `_is_metadata_or_label`).
- `experiments/build_product_dashboard.py` — script that produces the
  HTML dashboard from those CSVs.
- `experiments/predict.py` — pipeline runner for fresh input data; see
  section 6.

## 5. Known failure modes

All focus-case windows are documented in `window_level_details.csv`
under `focus_case_tag`.

- **Tokyo run2 w7 / w9** (`false_high`): base prediction is 63-74 %
  because simulator continuity looks clean, but demo5 refuses to FIX
  due to RINEX phase jumps.  Adopted model reduces to 39.5 / 71.3 %;
  still inflated.  A diagnostic reject rule
  (`rinex_phase_jump_ge0p5cy_count_max >= 1`) catches both cleanly but
  cannot be validated as a learned LORO model (only these two
  positives exist in the dataset) and is therefore declared a domain
  prior in §5.3.
- **Tokyo run2 w23 - w27** (`hidden_high`): actual FIX is 75-100 % but
  deployable features under-predict.  Adopted model lifts to 32-50 %,
  a significant gain over baseline 16 %, but still far from actual.
- **Tokyo run3 w17** (`false_lift`): deployable readiness peaks
  (`hold_carry_score_mean = 11.68`) but demo5 refuses.  Adopted model
  lifts to 19 % against 0 % actual.  A stricter hold_ready threshold
  (0.55) catches this case at the cost of the w25-w27 lifts; see
  §7.14.
- **Nagoya run2 false-lift cluster**: partially absorbed by the
  classifier; predictions land at 9-21 % rather than the direct lift
  rule's 95 %.

## 6. How to reproduce

### From scratch (full retrain + deliverable build)

```bash
# 1. Epoch-level validationhold surrogate
python3 experiments/augment_ppc_epochs_with_validation_hold_surrogate.py

# 2. Window-level aggregation
python3 experiments/analyze_ppc_validation_hold_surrogate_windows.py

# 3. Build window CSV with current_tight_hold preset
python3 experiments/rebuild_validationhold_flag_thresholds.py \
  --preset current_tight_hold \
  --output-csv experiments/results/ppc_validationhold_window_summary_current_tight_hold.csv
python3 experiments/augment_ppc_windows_with_validationhold_features.py \
  --validationhold-csv experiments/results/ppc_validationhold_window_summary_current_tight_hold.csv \
  --output-csv experiments/results/ppc_window_fix_rate_model_stride1_stat_sim_rinex_phasejump_t0p25_gf0p2_simloscont_focused_simadop_nowt_windowopt_validationhold_current_tight_hold_carry_window_predictions.csv

# 4. Train adopted strict-best model with residual alpha=0.75
python3 experiments/train_ppc_solver_transition_surrogate_nested_stack.py \
  --window-csv experiments/results/ppc_window_fix_rate_model_stride1_stat_sim_rinex_phasejump_t0p25_gf0p2_simloscont_focused_simadop_nowt_windowopt_validationhold_current_tight_hold_carry_window_predictions.csv \
  --base-prefix ppc_window_fix_rate_model_stride1_stat_sim_rinex_phasejump_t0p25_gf0p2_simloscont_focused_simadop_nowt_windowopt_baseerror15_refinedgrid \
  --classifier-include-run-position \
  --alphas 0.75 \
  --residual-clip-pp 50 \
  --max-run-mae-pp 4.5 \
  --max-abs-aggregate-error-pp 2.0 \
  --results-prefix ppc_window_fix_rate_model_stride1_stat_sim_rinex_phasejump_t0p25_gf0p2_simloscont_focused_simadop_nowt_solver_transition_surrogate_nested_et80_validationhold_current_tight_hold_carry_alpha75_meta_run45

# 5. Build deliverable CSVs
python3 experiments/build_product_deliverable.py
```

### Inference on a new epoch CSV

See `experiments/predict.py --help`.  Caveat: the pipeline retrains the
nested stack from scratch with the new data included as an additional
LORO outer fold.  True single-model inference requires saving a
full-data-fit stack, which is not currently implemented.

## 7. Maintainer notes

- The adopted threshold set (0.60 hold_ready, 0.60 pass, 0.45
  hold_strict, 1.25 block_p90) is a local optimum on this dataset
  (§7.14 / §7.15).  Re-scan if the training data grows by more than
  a couple of runs.
- The `alpha=0.75` residual setting is Pareto-optimal on run MAE +
  correlation but 0.09 pp worse on wmae than `alpha=0.5`.  Switch via
  the `--alphas` argument if the consumer metric changes.
- The `clean_streak_s` carry counter (§7.11) is the single most
  impactful deployable feature added since the baseline.  Any further
  epoch-level surrogate redesign should preserve a pure cumulative-
  counter state variable.
- `validationhold_high_pred_reject_signal` is declared a domain prior
  (§5.3).  Do not treat as a learned gate until more false-high
  windows exist in the dataset.
