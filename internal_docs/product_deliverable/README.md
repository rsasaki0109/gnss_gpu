# PPC demo5 FIX-Rate Predictor — Product Deliverable

**Status**: internal research prototype, route-level deliverable with saved one-shot fresh-data batch inference, bootstrap raw-source preparation, source-manifest validation, and actionability annotations
**Last updated**: 2026-04-30
**Adopted model**: §7.16 transition stack + 0.75-blended isotonic calibration + phase-delta guard `transition_surrogate_nested_et80_validationhold_current_tight_hold_carry_alpha75_isotonic75_phaseguard_meta_run45`
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
- Fresh-data offline inference: prepare and score new PPC/taroz
  preprocessed epoch/window/base CSVs with the committed full-data-fit
  model artifact, without retraining the nested LORO stack.
- Online-compatible scoring: score prepared windows in route order using
  only current/past model probability state, when the planned route
  window count is known.
- Source bundle contract validation: bind raw PPC source run
  directories to derived epoch/window/base CSVs, check that their
  route/window keys match, and run product batch inference from the
  manifest.
- Bootstrap raw-source preparation: parse PPC RINEX/reference run
  directories into label-free epoch/window/base CSVs and a derived
  manifest that the product inference path can consume.  Simulator and
  refinedgrid-only features are neutral-filled, so this is a degraded
  bridge for product wiring, not the calibrated upstream feature
  pipeline.
- Split inference input preparation: build the pre-augmented window CSV
  separately when debugging intermediate validationhold features.
- Window-level operational screening: use `window_action == "use"` for
  normal diagnostics, `review` for hidden-high windows, and `abstain`
  to remove known false-high/false-lift windows from automated
  window-level actions.

### Out-of-scope use cases

- Unscreened individual 30 s window FIX-rate predictions.  Typical
  window weighted MAE is about 16 pp and tail errors reach 60+ pp on known
  failure cases; use `window_action` before acting on a window.
- Full calibrated GNSS/RINEX/simulator feature extraction from source
  files.  The product path includes a bootstrap raw-source bridge, but
  the research simulator/refinedgrid extraction remains upstream.
- Full real-time feature extraction from raw RINEX/simulator source
  files.  The product model can score prepared windows online, but the
  upstream feature extraction is still batch/offline.
- Data sources different from taroz/PPC Tokyo/Nagoya.  The model was
  tuned on 6 runs across 2 cities; generalisation to other
  cities/environments is untested.
- Safety-critical decisions.  This is a statistical estimate with
  documented failure modes in section 5.

## 3. Metrics (strict nested LORO on 6 runs / 197 windows)

Under product-relevant metrics:

| metric | adopted phaseguard | baseline §2.2 |
| --- | --- | --- |
| run MAE | **1.790 pp** | 4.436 pp |
| window correlation | **0.559** | 0.401 |
| overall aggregate error | **-1.16 pp** on 17.90 % dataset FIX rate | — |
| window weighted MAE | **15.847 pp** | 18.046 pp |

The committed single-model artifact is a full-data fit for deployment,
not an independent validation fold.  The final isotonic calibration is
selected on strict leave-one-run-out predictions from the previous
§7.16 model, then blended 0.75 with the original alpha75 prediction to
keep correlation and tail error closer to the pre-calibrated model.
A deployable phase-delta guard then caps predictions at 20 % when
`rinex_phase_raw_delta_cycles_p50_p75 >= 426.419`, selected on strict
LORO as a RINEX phase-instability prior.
Use the strict nested LORO table above as the generalisation estimate.

Per-route error on the test set (see
`route_level_fix_rate_prediction.csv`):

| city / run | actual | predicted | \|err\| | confidence | action |
| --- | --- | --- | --- | --- | --- |
| nagoya / run1 | 11.5 % | 11.7 % | 0.23 pp | medium | review |
| nagoya / run2 | 16.2 % | 17.6 % | 1.44 pp | medium | review |
| nagoya / run3 |  7.9 % |  8.0 % | 0.17 pp | high | ok |
| tokyo  / run1 | 10.9 % | 10.4 % | 0.50 pp | medium | review |
| tokyo  / run2 | 29.0 % | 20.9 % | 8.13 pp | low | review_required |
| tokyo  / run3 | 24.0 % | 23.7 % | 0.28 pp | low | review_required |

Confidence tiers are auto-detected per route from the presence of
focus-case windows (see `build_product_deliverable.py::_classify_window`
thresholds).  A `high` tier means no material failure-mode window is
present; `medium` means the route contains hidden-high windows
(actual high, corrected under-predicts by 40+ pp); `low` means the
route contains at least one false-high or false-lift window.
`route_action` is separate from the numeric prediction: `ok` means all
windows are usable, `review` means hidden-high windows are present, and
`review_required` means at least one window is abstained from automated
window-level action.

Five of six routes are within 1.5 pp of actual.  The one exception is
tokyo/run2 at 8.13 pp, which contains the Tokyo run2 w7/w9 false-high
cluster and the w23-w27 hidden-high cluster documented in section 5.
Compared with the previous isotonic75 artifact, the phase-delta guard
reduces focus false-high windows from 9 to 3, abstain windows from 9 to
3, and low-confidence routes from 5 to 2.

## 4. Inputs, outputs, and files

### Inputs required by the product command

- Committed adopted window prediction CSV:
  `experiments/results/ppc_window_fix_rate_model_..._alpha75_isotonic75_phaseguard_meta_run45_window_predictions.csv`.
  `python3 experiments/predict.py` uses this frozen artifact by default
  to refresh `route_level_fix_rate_prediction.csv`,
  `window_level_details.csv`, and `dashboard.html`.
- Committed full-data-fit product model artifact:
  `experiments/results/ppc_window_fix_rate_model_..._alpha75_isotonic75_phaseguard_meta_run45_product_model.pkl.gz`.
  `python3 experiments/predict.py --batch-inference` loads this artifact
  after preparing fresh inputs; split `--inference` can also score an
  already prepared window CSV without training.

### Inputs required for fresh-data preparation / inference

- Epoch feature CSV with deployable simulator/RINEX columns and
  `(city, run, gps_tow)`.
- Window feature CSV with matching `(city, run, window_index)`,
  `window_start_tow`, `window_end_tow`, and `sim_matched_epochs`.
- Refined-grid base prediction CSV with matching `(city, run,
  window_index)` keys.

These inputs do not need `actual_fix_rate_pct`, `actual_fixed`,
`rtk_*`, or `solver_demo5_*` columns at inference time.

### Optional source manifest for fresh-data inference

`--source-bundle-prepare` accepts a raw JSON manifest with PPC run
directories, parses their RINEX/reference inputs, and writes bootstrap
epoch/window/base CSVs plus a derived source manifest.  Unsupported
simulator/refinedgrid features are neutral-filled so the saved product
model can run.  This bootstrap path validates `base.nav` presence as
part of the PPC source bundle, but it does not run broadcast-ephemeris
satellite propagation, the calibrated simulator, or refinedgrid base
prediction.

`--source-bundle-check` and `--source-bundle-inference` accept the
derived JSON manifest that declares raw PPC source run directories and
the derived product input CSVs.  The manifest validator checks that each
raw run directory has the PPC source files (`rover.obs`, `base.obs`,
`base.nav`, `reference.csv`), that the derived epoch/window CSVs contain
those `(city, run)` keys, that every product window has a matching base
prediction row, and that prepared/final output paths do not collide.
Relative paths in manifests resolve from the manifest file's directory.

### Inputs required for full retraining

- Processed PPC epoch CSV with `gnss_gpu` simulator features, RINEX
  aggregates, demo5 solver state (for targets), and demo5 actual FIX
  labels.
- Refined-grid base prediction CSV
  (`..._windowopt_baseerror15_refinedgrid_window_predictions.csv`).

### Pipeline outputs at inference time

- `route_level_fix_rate_prediction.csv` — one row per `(city, run)`:
  actual vs predicted FIX rate, aggregate error, qualitative
  confidence tier, actionability counts, and route action.
- `window_level_details.csv` — per-window predictions with focus-case
  annotations plus `window_action` (`use`, `review`, or `abstain`),
  intended for diagnostics and operational screening.
- In `--batch-inference`, `--inference`, or `--online-inference` mode,
  `<prefix>_route_predictions.csv` and
  `<prefix>_window_predictions.csv` are written.  When labels are not
  present, these files contain predictions only and omit actual/error
  columns.
- In `--prepare-inference` mode, a prepared window CSV is written.  It
  contains the deployable window features, validationhold aggregates,
  and `base_pred_fix_rate_pct` for direct use with
  `--inference --use-window-base-prediction`.

### Files shipped with this deliverable

- `internal_docs/product_deliverable/README.md` — this file.
- `internal_docs/product_deliverable/RUNBOOK.md` — operational runbook.
- `internal_docs/product_deliverable/route_level_fix_rate_prediction.csv`
- `internal_docs/product_deliverable/window_level_details.csv`
- `internal_docs/product_deliverable/dashboard.html` — self-contained
  HTML report (open in a browser) with metrics summary, per-run bar
  chart, actual-vs-predicted scatter, and focus-case detail table.
- `experiments/results/ppc_window_fix_rate_model_stride1_stat_sim_rinex_phasejump_t0p25_gf0p2_simloscont_focused_simadop_nowt_solver_transition_surrogate_nested_et80_validationhold_current_tight_hold_carry_alpha75_isotonic75_phaseguard_meta_run45_window_predictions.csv`
  — committed frozen calibrated window prediction artifact used by
  `experiments/predict.py` in default product mode.
- `experiments/results/ppc_window_fix_rate_model_stride1_stat_sim_rinex_phasejump_t0p25_gf0p2_simloscont_focused_simadop_nowt_solver_transition_surrogate_nested_et80_validationhold_current_tight_hold_carry_alpha75_isotonic75_phaseguard_meta_run45_product_model.pkl.gz`
  — committed saved full-data-fit model artifact with 0.75-blended
  final isotonic calibration and phase-delta prediction guard, used by
  `experiments/predict.py --inference`.
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
- `experiments/product_inference_model.py` — fits and runs the saved
  single-model product inference artifact.
- `experiments/product_source_bundle.py` — validates source manifests
  that bind raw PPC run directories to derived epoch/window/base product
  input CSVs.
- `experiments/product_raw_source_prepare.py` — bootstrap raw PPC
  source preparation into model-schema-compatible epoch/window/base CSVs
  and a derived source manifest.  Neutral-fills unsupported
  simulator/refinedgrid-only features.
- `experiments/analyze_ppc_validation_hold_surrogate_windows.py` —
  aggregates validationhold epoch state to windows.  It supports
  label-free inference inputs; strict metrics are written only when
  actual labels are present.
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
- `internal_docs/product_deliverable/PAPER_STYLE_EVAL.md` — evaluation
  lined up with Furukawa & Kubo (2019) "Prediction of Fixing of
  RTK-GNSS Positioning in Multipath Environment Using Radiowave
  Propagation Simulation" (IPNTJ Vol.10 No.2).  Includes the
  threshold-swept matching-rate table, Fig.9/10-style figures, and a
  methodology comparison.
- `internal_docs/product_deliverable/REFLECTION_POC_NULL.md` —
  null-result write-up for the canyon proxy, BVH multipath, and
  simplified antenna gain + NLoS attenuation probes.  These are
  research diagnostics only; none replace the adopted §7.16 model.
- `experiments/build_paper_style_eval.py` — computes the pooled
  matching-rate sweep and renders the side-by-side Fig.10-style
  RTK-FIXED maps under `plots/{city}_{run}_fix_comparison_map.png`.
- `experiments/exp_ppc_reflection_poc.py`,
  `experiments/aggregate_reflection_features.py`, and
  `experiments/augment_window_csv_with_reflection.py` — reflection
  feature extraction, pooling, and §7.16 merge scripts for the null
  BVH multipath probe.
- `experiments/exp_ppc_antenna_attenuation_features.py` and
  `experiments/augment_window_csv_with_antenna.py` — simplified
  antenna gain + -25 dB NLoS attenuation extraction and §7.16 merge
  scripts for the §7.25 null probe.
- `experiments/_common.py` — shared helpers used across the experiment
  scripts (e.g. `_is_metadata_or_label`).
- `experiments/build_product_dashboard.py` — script that produces the
  HTML dashboard from those CSVs.
- `experiments/predict.py` — product entrypoint.  Default mode refreshes
  deliverable outputs from the frozen calibrated predictions;
  `--batch-inference` prepares and scores fresh inputs in one command;
  `--source-bundle-prepare` derives bootstrap CSVs and a manifest from
  raw PPC source runs;
  `--source-bundle-inference` validates a source manifest before
  running the same batch path;
  `--online-inference` scores prepared windows in causal mode with a
  planned route length; split `--prepare-inference` / `--inference` are
  available for debugging; `--retrain` remains the research LORO
  pipeline.

## 5. Known failure modes

All focus-case windows are documented in `window_level_details.csv`
under `focus_case_tag`.  The adjacent `window_action` column marks
`false_high` / `false_lift` as `abstain`, `hidden_high` as `review`,
and normal or resolved windows as `use`.

- **Tokyo run2 w7 / w9** (`false_high`): base prediction is high
  because simulator continuity looks clean, but demo5 refuses to FIX
  due to RINEX phase jumps.  The adopted phase-delta guard does not
  catch these two residual false-high windows because they do not match
  the selected phase-delta cap condition.
- **Tokyo run2 w23 - w27** (`hidden_high`): actual FIX is 75-100 % but
  deployable features under-predict.  The adopted model lifts part of
  this segment, but still under-predicts several high-FIX windows.
- **Tokyo run3 w28** (`false_high`): residual inflated prediction on a
  low-actual window after isotonic calibration and the phase-delta
  guard.
- **Nagoya run2 false-lift cluster**: partially absorbed by the
  classifier and now mostly tagged `false_lift_resolved`; phase-delta
  active windows are capped at 20 %.

## 6. How to reproduce

### Product refresh from the frozen adopted predictions

```bash
python3 experiments/predict.py
```

Use `--check-only` for a no-write preflight.  This is the recommended
operator path from a clean checkout.

### From scratch (full retrain + deliverable build)

The full retrain path requires the large preprocessed epoch/window CSVs
and refined-grid base prediction CSV, which are not part of the compact
product deliverable.

```bash
python3 experiments/predict.py \
  --retrain \
  --epochs-csv path/to/epochs.csv \
  --window-csv path/to/window_predictions.csv \
  --base-prefix path/to/refinedgrid_prefix_or_predictions.csv \
  --results-prefix ppc_..._my_run
```

### One-shot inference from new preprocessed CSVs

```bash
python3 experiments/predict.py \
  --batch-inference \
  --epochs-csv path/to/preprocessed_epochs.csv \
  --window-csv path/to/window_features.csv \
  --base-prefix path/to/refinedgrid_prefix_or_predictions.csv \
  --prepare-prefix experiments/results/my_run_prepare \
  --prepared-window-csv experiments/results/my_run_prepared_window_predictions.csv \
  --inference-output-prefix experiments/results/my_run_product
```

This produces a label-free prepared window CSV containing
`base_pred_fix_rate_pct` and validationhold features, then scores it
with the committed `*_product_model.pkl.gz` artifact.  It does not run
demo5 and does not require actual labels.  The output files are
`experiments/results/my_run_product_route_predictions.csv` and
`experiments/results/my_run_product_window_predictions.csv`.
The prepared CSV path must be distinct from the final
`*_window_predictions.csv` output path.

### Source-manifest inference from a PPC source bundle

Create an example derived manifest:

```bash
python3 experiments/product_source_bundle.py template \
  --output source_manifest.example.json
```

To start from raw PPC run directories, prepare bootstrap product CSVs
first:

```bash
python3 experiments/predict.py \
  --source-bundle-prepare \
  --source-manifest raw_source_manifest.json \
  --source-output-prefix experiments/results/my_raw_source
```

The command writes
`experiments/results/my_raw_source_source_manifest.json`, which can be
used by the validation and inference commands below.  This raw-source
bridge is label-free and product-schema-compatible, but it neutral-fills
features that require the full simulator/refinedgrid research pipeline.
The derived manifest also records `raw_source_prepare` metadata,
including total and per-run epoch/window/base row counts.  The
source-bundle validator checks those counts when the metadata is
present, so stale or partially replaced derived CSVs fail before
scoring.

Validate the raw source runs and derived product CSV contract:

```bash
python3 experiments/predict.py \
  --source-bundle-check \
  --source-manifest path/to/source_manifest.json
```

Validate and then run the same one-shot batch inference path:

```bash
python3 experiments/predict.py \
  --source-bundle-inference \
  --source-manifest path/to/source_manifest.json
```

The derived manifest fixes the product input bundle by proving the raw
source directories and derived epoch/window/base CSVs refer to the same
PPC runs/windows.

### Split preparation for debugging

```bash
python3 experiments/predict.py \
  --prepare-inference \
  --epochs-csv path/to/preprocessed_epochs.csv \
  --window-csv path/to/window_features.csv \
  --base-prefix path/to/refinedgrid_prefix_or_predictions.csv \
  --prepare-prefix experiments/results/my_run_prepare \
  --prepared-window-csv experiments/results/my_run_prepared_window_predictions.csv
```

This is the same preparation stage used by `--batch-inference`.

### Inference on a prepared window CSV

```bash
python3 experiments/predict.py \
  --inference \
  --window-csv experiments/results/my_run_prepared_window_predictions.csv \
  --use-window-base-prediction \
  --inference-output-prefix experiments/results/my_run_product
```

This path loads the committed `*_product_model.pkl.gz` artifact and does
not retrain.  The output files are
`experiments/results/my_run_product_route_predictions.csv` and
`experiments/results/my_run_product_window_predictions.csv`.

If the prepared window CSV does not contain `base_pred_fix_rate_pct`,
use `--base-prefix path/to/refinedgrid_prefix_or_predictions.csv`
instead of `--use-window-base-prediction`.

### Online-compatible inference on prepared windows

```bash
python3 experiments/predict.py \
  --online-inference \
  --window-csv experiments/results/my_run_prepared_window_predictions.csv \
  --use-window-base-prediction \
  --planned-window-count 32 \
  --inference-output-prefix experiments/results/my_run_online_product
```

This scores the same saved product model with causal current/past
probability state.  The adopted artifact uses route-position meta
features, so online mode needs a planned route length.  For multi-route
inputs, add a `planned_window_count` column to the prepared window CSV
instead of passing a single `--planned-window-count`.

Raw source-file feature extraction is still upstream of this command.

## 7. Maintainer notes

- The adopted threshold set (0.60 hold_ready, 0.60 pass, 0.45
  hold_strict, 1.25 block_p90) is a local optimum on this dataset
  (§7.14 / §7.15).  Re-scan if the training data grows by more than
  a couple of runs.
- The 0.75-blended final isotonic calibrator plus phase-delta guard
  improves run MAE and weighted MAE over raw §7.16 alpha=0.75
  predictions.  Re-evaluate this post-calibrator/guard pair when the
  training set grows.
- The `clean_streak_s` carry counter (§7.11) is the single most
  impactful deployable feature added since the baseline.  Any further
  epoch-level surrogate redesign should preserve a pure cumulative-
  counter state variable.
- `validationhold_high_pred_reject_signal` is declared a domain prior
  (§5.3).  Do not treat as a learned gate until more false-high
  windows exist in the dataset.
