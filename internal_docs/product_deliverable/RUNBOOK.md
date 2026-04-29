# PPC FIX-Rate Predictor — Operational Runbook

**Last updated**: 2026-04-29
**Target audience**: operator running the predictor on new PPC/taroz data
**Prerequisites**: familiar with the `gnss_gpu` repository layout and Python 3.12 environment

---

## 1. When to run

Run the predictor when any of these happen:

- A new PPC/taroz run has been processed through the epoch CSV pipeline,
  or you need the bootstrap raw-source bridge to derive product CSVs
  from PPC RINEX/reference run directories.
- You want to re-score the existing 6 runs after any upstream pipeline
  change (validationhold surrogate, refinedgrid base model).
- The last deliverable CSV is older than one month and you want to
  refresh for a review.

Do NOT run just because time has passed.  The model is frozen per
D-030; re-running without new data gives identical output.

## 2. One-shot product refresh (bundled 6-run test set)

```bash
cd /media/sasaki/aiueo/ai_coding_ws/gnss_cuda_sim_ws/gnss_gpu
python3 experiments/predict.py
```

This is the frozen product path.  It reads the committed adopted §7.16
window predictions and refreshes the operator-facing outputs.  It
should finish in a few seconds from a clean checkout.

Outputs land in:

- `experiments/results/ppc_window_fix_rate_model_..._alpha75_meta_run45_window_predictions.csv` — frozen input artifact
- `experiments/results/ppc_window_fix_rate_model_..._alpha75_meta_run45_product_model.pkl.gz` — saved inference model artifact
- `internal_docs/product_deliverable/route_level_fix_rate_prediction.csv`
- `internal_docs/product_deliverable/window_level_details.csv`
- `internal_docs/product_deliverable/dashboard.html` — open in a browser

For a preflight-only check:

```bash
python3 experiments/predict.py --check-only
```

## 3. Fresh-data batch inference without retraining

### 3.1 Prerequisites

Before calling `predict.py --batch-inference`, the new run's data must already be
preprocessed through the upstream pipeline into:

- an epoch feature CSV with deployable simulator/RINEX columns and
  `(city, run, gps_tow)`
- a window feature CSV with `(city, run, window_index)`,
  `window_start_tow`, `window_end_tow`, and `sim_matched_epochs`
- a refinedgrid base prediction CSV (same city/run keys)

These files do not need `actual_fix_rate_pct`, `actual_fixed`, `rtk_*`,
or `solver_demo5_*` columns.  Those are training/evaluation labels, not
runtime inputs.

### 3.2 Bootstrap raw-source preparation

Use `--source-bundle-prepare` when the operator receives raw PPC
RINEX/reference run directories but not derived product CSVs yet.  The
command emits label-free epoch/window/base CSVs and a derived source
manifest for the normal source-bundle inference path.

Raw manifest shape:

```json
{
  "runs": [
    {
      "city": "nagoya",
      "run": "run1",
      "run_dir": "/path/to/PPC-Dataset/nagoya/run1"
    }
  ],
  "outputs": {
    "prepare_prefix": "experiments/results/my_run_prepare",
    "prepared_window_csv": "experiments/results/my_run_prepared_window_predictions.csv",
    "inference_output_prefix": "experiments/results/my_run_product"
  }
}
```

Prepare bootstrap CSVs and the derived manifest:

```bash
python3 experiments/predict.py \
  --source-bundle-prepare \
  --source-manifest raw_source_manifest.json \
  --source-output-prefix experiments/results/my_raw_source
```

Outputs:

- `experiments/results/my_raw_source_epochs.csv`
- `experiments/results/my_raw_source_window_features.csv`
- `experiments/results/my_raw_source_base_window_predictions.csv`
- `experiments/results/my_raw_source_source_manifest.json`

This path parses raw PPC RINEX/reference inputs, but it is intentionally
degraded: features that require the calibrated simulator/refinedgrid
research pipeline are neutral-filled so the saved product model can run.
Use it to fix the product inference wiring from raw source bundles, not
as a replacement for the upstream calibrated feature pipeline.

### 3.3 Source manifest check

Use a source manifest when the operator receives a PPC source bundle and
derived product CSVs together.  The manifest binds raw PPC run
directories to the epoch/window/base CSVs so the product command can
reject mismatched bundles before scoring.

Create a template:

```bash
python3 experiments/product_source_bundle.py template \
  --output source_manifest.example.json
```

Minimal manifest shape:

```json
{
  "runs": [
    {
      "city": "nagoya",
      "run": "run1",
      "run_dir": "/path/to/PPC-Dataset/nagoya/run1"
    }
  ],
  "derived_inputs": {
    "epochs_csv": "/path/to/preprocessed_epochs.csv",
    "window_csv": "/path/to/window_features.csv",
    "base_prediction_csv": "/path/to/refinedgrid_window_predictions.csv"
  },
  "outputs": {
    "prepare_prefix": "experiments/results/my_run_prepare",
    "prepared_window_csv": "experiments/results/my_run_prepared_window_predictions.csv",
    "inference_output_prefix": "experiments/results/my_run_product"
  }
}
```

Validate only:

```bash
python3 experiments/predict.py \
  --source-bundle-check \
  --source-manifest path/to/source_manifest.json
```

Validate and run one-shot batch inference:

```bash
python3 experiments/predict.py \
  --source-bundle-inference \
  --source-manifest path/to/source_manifest.json
```

The derived source manifest confirms that the raw PPC source directories
and derived epoch/window/base CSVs describe the same product input
bundle.  Relative paths in the manifest resolve from the manifest file's
directory.

### 3.4 One-shot batch command

```bash
python3 experiments/predict.py \
  --batch-inference \
  --epochs-csv path/to/preprocessed_epochs.csv \
  --window-csv path/to/window_features.csv \
  --base-prefix path/to/refinedgrid_prefix_or_predictions.csv \
  --prepare-prefix experiments/results/<custom_prefix_for_this_run>_prepare \
  --prepared-window-csv experiments/results/<custom_prefix_for_this_run>_prepared_window_predictions.csv \
  --inference-output-prefix experiments/results/<custom_prefix_for_this_run>
```

`--base-prefix`
accepts either a bare prefix under `experiments/results`, an explicit
prefix path, or the full `*_window_predictions.csv` path.

Outputs:

- `<custom_prefix_for_this_run>_route_predictions.csv`
- `<custom_prefix_for_this_run>_window_predictions.csv`
- `<custom_prefix_for_this_run>_prepared_window_predictions.csv`

The prepared CSV contains the original window features, validationhold
aggregates/flags, and `base_pred_fix_rate_pct`.  The route/window
prediction files are the operator-facing fresh-data result.
Keep `--prepared-window-csv` distinct from the final
`<custom_prefix_for_this_run>_window_predictions.csv`; the command
errors rather than overwrite the prepared feature file.

For a preflight-only check:

```bash
python3 experiments/predict.py \
  --batch-inference \
  --epochs-csv path/to/preprocessed_epochs.csv \
  --window-csv path/to/window_features.csv \
  --base-prefix path/to/refinedgrid_prefix_or_predictions.csv \
  --check-only
```

### 3.4 Split prepare command

Use this when inspecting intermediate validationhold features before
scoring:

```bash
python3 experiments/predict.py \
  --prepare-inference \
  --epochs-csv path/to/preprocessed_epochs.csv \
  --window-csv path/to/window_features.csv \
  --base-prefix path/to/refinedgrid_prefix_or_predictions.csv \
  --prepare-prefix experiments/results/<custom_prefix_for_this_run>_prepare \
  --prepared-window-csv experiments/results/<custom_prefix_for_this_run>_prepared_window_predictions.csv
```

### 3.5 Split inference command

Use this when the prepared CSV already exists:

```bash
python3 experiments/predict.py \
  --inference \
  --window-csv experiments/results/<custom_prefix_for_this_run>_prepared_window_predictions.csv \
  --use-window-base-prediction \
  --inference-output-prefix experiments/results/<custom_prefix_for_this_run>
```

Outputs:

- `<custom_prefix_for_this_run>_route_predictions.csv`
- `<custom_prefix_for_this_run>_window_predictions.csv`

When the inference input has no actual labels, the output omits
actual/error columns and contains predictions only.

### 3.6 Online-compatible inference command

Use this when prepared windows arrive in route order and the planned
route length is known:

```bash
python3 experiments/predict.py \
  --online-inference \
  --window-csv experiments/results/<custom_prefix_for_this_run>_prepared_window_predictions.csv \
  --use-window-base-prediction \
  --planned-window-count 32 \
  --inference-output-prefix experiments/results/<custom_prefix_for_this_run>_online
```

The online scorer uses only the current window and previous windows for
the product model's temporal probability features.  The adopted model
also uses route-position meta features, so it needs either
`--planned-window-count` for a single route or a `planned_window_count`
column for multi-route inputs.  Use `--online-use-input-run-length`
only for offline parity/smoke checks where the complete input file is
already available.

### 3.7 Full retrain / research scoring

Use `predict.py --retrain` only when deliberately rebuilding the
research LORO pipeline:

```bash
python3 experiments/predict.py \
  --retrain \
  --epochs-csv path/to/epochs.csv \
  --window-csv path/to/window_predictions.csv \
  --base-prefix path/to/refinedgrid_prefix_or_predictions.csv \
  --results-prefix <custom_prefix_for_this_run>
```

`--retrain` trains from scratch and is not required for normal
fresh-data scoring.

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

### 4.4 Inference route/window prediction files

`predict.py --inference` writes route and window prediction files under
the requested `--inference-output-prefix`.

Route file columns:

- `baseline_pred_fix_rate_pct` — refined-grid base prediction
- `adopted_pred_fix_rate_pct` — saved product model prediction
- `actual_fix_rate_pct`, `adopted_abs_error_pp`,
  `adopted_signed_error_pp` — present only when labels were provided

Window file columns:

- `base_pred_fix_rate_pct`, `residual_pred_pp`,
  `corrected_pred_fix_rate_pct`
- transition-surrogate probability columns ending in `_prob`
- actual/error columns only when labels were provided

### 4.5 Prepared inference window CSV

`predict.py --prepare-inference` writes a window CSV suitable for
`--inference --use-window-base-prediction`.

Important columns:

- `base_pred_fix_rate_pct` — refined-grid base prediction merged by
  `(city, run, window_index)`
- `validationhold_high_pred_reject_flag`,
  `validationhold_low_pred_lift_flag` — deployable flags after the
  selected preset
- validationhold aggregate columns such as `hold_ready_frac`,
  `hold_strict_ready_frac`, `clean_streak_s_at_start`, and
  `validation_block_score_p90`

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
- 3-9 new runs: re-run `predict.py --retrain`, check whether aggregate metrics
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
| `experiments/predict.py` | product entrypoint; default is frozen deliverable refresh, `--batch-inference` prepares and scores fresh data in one command, `--source-bundle-prepare` derives bootstrap CSVs and a manifest from raw PPC source runs, `--source-bundle-inference` validates a derived source manifest before running the same batch path, `--online-inference` scores prepared windows causally with planned route length, split `--prepare-inference` / `--inference` are available for debugging, `--retrain` runs the full LORO pipeline |
| `experiments/product_inference_model.py` | fit/run saved single-model product inference artifact, including online-compatible scoring |
| `experiments/product_source_bundle.py` | validate source manifests that bind raw PPC run directories to derived epoch/window/base product input CSVs |
| `experiments/product_raw_source_prepare.py` | bootstrap raw PPC source preparation into model-schema-compatible epoch/window/base CSVs and a derived source manifest; neutral-fills unsupported simulator/refinedgrid-only features |
| `experiments/results/ppc_window_fix_rate_model_..._alpha75_meta_run45_window_predictions.csv` | committed adopted §7.16 window predictions used by default product mode |
| `experiments/results/ppc_window_fix_rate_model_..._alpha75_meta_run45_product_model.pkl.gz` | committed full-data-fit inference artifact used by `predict.py --inference` |
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
- Calibrated upstream pipeline questions (simulator extraction /
  refinedgrid base): out of scope for this runbook.  The bootstrap
  raw-source bridge exists only to derive product-compatible CSVs with
  neutral-filled unsupported features.

## 9. Known operational gotchas

- The nested stack is randomness-sensitive via `--random-state`
  (default 2034).  If you need deterministic reruns on the same data,
  keep the default and do not pass `--random-state` on the command line.
  This only applies to `predict.py --retrain`; the default frozen
  product flow is deterministic and does not train.
- The saved `*_product_model.pkl.gz` artifact is a full-data fit for
  deployment.  Use the strict nested LORO metrics in `README.md` as the
  generalisation estimate; do not treat full-data smoke metrics as an
  independent validation result.
- `--source-bundle-prepare` is label-free and parses raw PPC
  RINEX/reference inputs, but it does not run the calibrated simulator
  or refinedgrid base pipeline.  Unsupported features are neutral-filled.
- `--batch-inference`, `--source-bundle-inference`, and
  `--prepare-inference` are label-free, but they assume the
  epoch/window/base product CSVs already exist.  Use
  `--source-bundle-prepare` first when starting from raw source
  directories only.
- `--source-bundle-check` validates source-file presence and
  route/window key consistency.  It is a bundle-contract check, not a
  raw-feature extraction pipeline.
- In `--batch-inference`, do not point `--prepared-window-csv` at the
  final `<prefix>_window_predictions.csv` output.  Use a name such as
  `*_prepared_window_predictions.csv` so the prepared feature file and
  final prediction file remain separate.
- `--online-inference` is online for the saved product scoring stage,
  not for raw feature extraction.  The window row must already contain
  deployable simulator/RINEX features and `base_pred_fix_rate_pct` (or
  be paired with `--base-prefix`).
- The adopted product artifact uses planned route position.  If the
  route length is not known at runtime, do not use online mode as a
  strict live predictor; first provide a planned window count from the
  route plan.
- The `rebuild_validationhold_flag_thresholds.py` script overwrites
  its output CSV silently.  If you experiment with alternate presets,
  name the output files explicitly and do not point them at the
  canonical `ppc_validationhold_window_summary_current_tight_hold.csv`
  path unless you intend to replace it.
- Aggregate error on the dataset is +0.26 pp (not 0).  This is the
  expected calibration of the adopted model; do not try to re-bias
  it away with a post-hoc offset without understanding the per-route
  distribution first.
