# Epoch-level FIX classifier — experimental result

**Status**: experimental, not adopted.  The window-level §7.16 predictor
remains the adopted strict-best model.

## Motivation

The adopted §7.16 model predicts FIX rate per 30-s window.  An
alternative paradigm is to predict `P(actual_fixed = 1)` at each 0.2 s
epoch directly, then aggregate to window-level if needed.  Epoch-level
prediction has ~150x more training samples (58 706 epochs vs 197
windows) and offers a finer temporal resolution for product display.

This section records the null result from a first attempt.

## Script

`experiments/train_ppc_epoch_fix_classifier.py`

- Model: `HistGradientBoostingClassifier`
  (max_iter=300, learning_rate=0.05, max_depth=6, min_samples_leaf=50,
  l2=0.5)
- Features: 305 deployable per-epoch columns (excludes `rtk_*`,
  `solver_*`, `demo5_*`, `actual_fixed`, metadata; includes all
  validationhold surrogate state variables).
- Cross-validation: strict LORO by run (6 folds).

## Results

Per-run AUC varies dramatically:

| run | epochs | AUC | log-loss | Brier | demo5 FIX fraction |
| --- | --- | --- | --- | --- | --- |
| nagoya/run1 |  7651 | 0.776 | 0.467 | 0.143 | 11.1 % |
| nagoya/run2 |  9451 | 0.815 | 0.339 | 0.099 | 15.8 % |
| nagoya/run3 |  5201 | 0.525 | 0.370 | 0.101 |  8.0 % |
| tokyo/run1  | 11951 | 0.546 | 0.457 | 0.123 | 10.1 % |
| tokyo/run2  |  9151 | 0.571 | 1.029 | 0.251 | 28.1 % |
| tokyo/run3  | 15301 | 0.455 | 0.991 | 0.223 | 23.8 % |
| **OVERALL** | 58706 | **0.543** | 0.660 | 0.166 | 17.3 % |

Nagoya runs 1 and 2 train well (AUC 0.78-0.82), but held-out Tokyo
runs and nagoya/run3 fall to near-random.  Tokyo/run3 AUC 0.455 is
slightly *below* random — the model learns patterns that do not
transfer across the city/run boundary.

Window-level aggregation (mean of per-epoch predictions within each
30-s window) gives:

| metric | epoch classifier | adopted phaseguard |
| --- | --- | --- |
| weighted window MAE | 20.30 pp | **15.85 pp** |
| run MAE | 8.90 pp | **1.79 pp** |
| window correlation | 0.162 | **0.559** |

The epoch classifier is strictly worse on every aggregate metric than
the adopted window-level calibrated model.

## Why the epoch classifier is worse

1. **Per-epoch labels are noisy from the model's point of view.**
   demo5 has lock-carry inertia: once it enters FIX, it can hold
   through several epochs of poor geometry.  That is an internal
   solver state, not a property of the epoch's deployable features.
2. **The LORO boundary is still per-run.**  Epoch temporal correlation
   within a run means the effective independent-sample count is far
   smaller than 58 706.  LORO across only 6 runs still faces the same
   generalisation barrier as the window-level model.
3. **Non-stationarity across runs.**  Tokyo run3 in particular has a
   pattern of FIX behaviour that is not predicted by the 5 training
   runs, matching the window-level observation that Tokyo run2 / run3
   are the dominant failure cases.

## Value as a diagnostic

Despite the poor aggregate metrics, the per-epoch P(FIX) line is
informative as a visualisation overlay:

- It shows where the classifier believes conditions are favourable
  within a run, at a finer time resolution than the 30-s window
  predictions.
- In high-AUC runs (nagoya/run1, run2) the epoch line tracks the
  actual FIX fraction reasonably well.
- In low-AUC runs (tokyo/run3) the epoch line's inflation over the
  actual FIX fraction makes the model's over-confidence visible to
  the operator.

## Adoption

Not adopted as the reported model.  Predictions and metrics are
retained under
`experiments/results/ppc_epoch_fix_classifier_{predictions,metrics,window_aggregated}.csv`
and overlaid as a blue line on
`internal_docs/product_deliverable/plots/*_timeseries.png` for visual
diagnostic use only.

## Future work to improve the epoch classifier

- Add a per-run calibration step so AUC / log-loss per run are
  comparable (current model has very different scales across runs).
- Add a sequential / state-space model (e.g. an HMM over FIX / FLOAT
  transitions) to capture demo5 lock-carry inertia explicitly.
- Add per-epoch temporal features (e.g. FIX-state auto-regressive
  proxies) that are deployable and capture the carry state.
- Collect more runs so the LORO held-out distribution does not
  bottleneck the model's transfer.
