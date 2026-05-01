# Path 3 sketch: sequence-latent demo5 success model

**Status**: research design draft, not yet adopted.
**Date**: 2026-05-01.
**Source**: PR #42 path 3 ("Architectural pivot: model demo5 success directly,
e.g. as a sequence-level latent variable, rather than predicting the FIX
rate from per-window stats. Multi-week research.")

## Why path 3 (and why not path 1 or path 2)

Three observations from PR #42 + PR #44 pin down where the deployed
1.79 pp ceiling comes from:

1. **Window aggregates lose temporal information.**  demo5's success at
   window N depends on its trajectory through windows 0..N-1 (lock
   build-up, ambiguity convergence, carry-over from a clean stretch).
   A window's deployable features describe the window in isolation;
   the §7.16 stack glues windows back together with hand-crafted
   `hold_carry`/`clean_streak` counters, but those are coarse.
2. **Hidden-high cases are carry-over success.**  Tokyo/run2 windows
   23-27 sit in feature-bad conditions where demo5 nonetheless FIXes
   at 75-100 %.  The most parsimonious explanation is that demo5
   entered window 23 in a high-confidence ambiguity state and rode
   through.  No window-level feature can show this without
   reconstructing solver state.
3. **Path 1's "lift" is solver state read-back, not prediction.**  PR
   #44 confirmed that surfacing demo5's own ambiguity-fix-state
   indicators as runtime features lifts the ceiling, but only because
   it tells the model what demo5 already concluded.  The original
   pre-demo5 problem is unchanged.

Together these say: the original problem requires a model that
**reconstructs the latent ambiguity-health trajectory from deployable
features**, not one that aggregates per-window summaries.

## Model class

Discrete HMM as a starting point.  Per-epoch hidden state ∈ {0, 1, 2, 3}
representing roughly {no-lock, marginal, locked, locked-converged}.
Per-epoch observations are a small bundle (5-10 features) of
deployable signals: simulator LoS density, sim DOP, RINEX phase
quality summary, antenna effective dB, and the existing `hold_carry`
counter.

Train per-route (or with light pooling) with Baum-Welch.  At
inference, run forward filter over the test route, integrate posterior
state probabilities into per-window FIX rate predictions.  Aggregate
to route-level for the deployed metric.

If the discrete HMM is too coarse (likely), step up to a
linear-Gaussian state-space (Kalman-style) with a continuous
ambiguity-health scalar.  An RNN/Transformer is the last resort —
sample count (5910 epochs / 6 routes) is too small to learn deep
temporal dynamics without strong inductive bias.

## Data requirements

* Per-epoch CSVs (not window-aggregated).  These already exist
  upstream of the window-aggregation step
  (`augment_ppc_epochs_with_validation_hold_surrogate.py` outputs).
* Per-epoch demo5 success label.  Available from RTKLIB demo5 output
  used to compute the window-level FIX rates.
* Same 6 routes; ~5910 epochs total (197 windows × 30 epochs nominal).
* Two-handful of deployable per-epoch features (NOT the 5873-feature
  window vector — overfitting risk for a small-sample state-space
  model).

No new data collection.  Path 3 is reanalysis of existing PPC runs.

## Minimum POC

1. Extract per-epoch CSV from existing augmentation outputs (likely
   already on disk, may need a small repackager).
2. Pick 5-10 features by ablation against the deployed §7.16 stack's
   feature importances.
3. Fit a 3-4 state HMM with `hmmlearn` (CPU, seconds per fit).
4. Strict LORO across the 6 routes.
5. Aggregate per-epoch posterior FIX probability to per-window and
   per-route, compare run-MAE / window-MAE to deployed 1.79 pp / 15.85
   pp.
6. If discrete HMM null: try Linear-Gaussian state-space with
   `pykalman` or hand-rolled.
7. If both null: stop.  The architectural pivot does not pay off on
   this dataset, only path 2 (more data) remains.

Estimated wall-clock: 2-3 days for an honest POC, including the data
extraction.

## Kill criteria (what makes us stop)

* Discrete HMM + Linear-Gaussian both at run-MAE ≥ 1.79 pp under strict
  LORO on the 6 routes → architectural pivot is null on this dataset,
  matching path 1's "no signal under the deployed contract" finding.
* Tokyo/run2 residual ≥ 5 pp under both models → the focal hidden-high
  cluster is not explained by latent-state dynamics either; the only
  remaining path is more data (path 2).
* Per-epoch posterior probabilities collapse onto a single state for
  most epochs (degenerate filter) → the chosen feature bundle does
  not separate states; pick a different bundle or stop.

## What success looks like (so we don't overclaim)

* run-MAE ≤ 1.50 pp under strict LORO **on the deployed contract**
  (no rtk_*/solver_* features) → first genuine improvement to the
  pre-demo5 estimator since PR #42 closed paths 1-3 of the
  hidden-high attacks.
* Tokyo/run2 residual ≤ 5 pp **without any post-hoc threshold guard
  selected on aggregate metrics** → the hidden-high cluster is at
  least partially explained by latent-state dynamics.
* Per-state interpretability check: cluster the routes' epoch-level
  posterior state assignments and verify they correspond to plausible
  physical regimes (e.g. open sky, urban canyon, viaduct) rather than
  random partitions.

## Risks not covered above

* **State-space identifiability**: with 6 routes / 5910 epochs and
  weak labels, the latent dynamics may not be identifiable.  The
  POC fit must be checked for collapsed states and unstable
  transition matrices before any LORO claim.
* **Per-epoch label noise**: demo5's per-epoch FIX flag is noisier
  than the window-aggregate.  A windowed posterior may be needed for
  a stable target; if so, we lose some of the temporal-resolution
  advantage.
* **Per-epoch feature alignment**: the existing pipeline aggregates
  to windows.  Re-pulling per-epoch features for the exact same 197
  windows requires care to avoid time-stamp drift.

## Relation to D-035

Path 3 attacks the **pre-demo5 estimator** problem (still stuck at
1.79 pp).  D-035's path 1 model is for the **post-demo5 QA tool**
problem (1.50 pp on tokyo/run2).  Both can land in the deliverable as
parallel model classes; they answer different questions.
