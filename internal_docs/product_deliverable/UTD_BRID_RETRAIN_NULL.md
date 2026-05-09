# UTD edge candidates + bldg+brid: NULL

**Status**: bldg+brid UTD edge candidate retrain confirmed null,
strictly worse than the bldg-only PoC.
**Date**: 2026-05-01.
**Source**: `UTD_DIFFRACTION_POC.md` (bldg-only null)
+ `PLATEAU_BRIDGE_INTEGRATION.md` (bridge loader)
+ `PLATEAU_BRIDGE_RETRAIN_NULL.md` (§7.25b antenna+brid null).

## Summary

Re-running the UTD edge candidate feature pipeline on the
EGM96-corrected bldg+brid mesh and retraining the §7.16 nested stack
**fails the `--max-run-mae-pp 4.5` selection guardrail** and is
strictly worse than both the bldg-only UTD PoC and the deployed
adopted model.

| variant | wmae pp | run MAE pp | corr |
|---|---:|---:|---:|
| §7.16 + iso75 + phaseguard (adopted) | **15.85** | **1.79** | **0.559** |
| UTD edge bldg-only ridge α=0.75 (existing PoC) | 18.264 | 5.168 | 0.471 |
| **UTD edge bldg+brid ridge α=0.75** | **18.539** | **5.863** | **0.468** |
| UTD edge bldg+brid extra_trees α=0.75 | 18.866 | 5.709 | 0.365 |
| conservative baseline (no model) | 18.046 | 4.436 | 0.401 |

Both bldg+brid variants exceed the 4.5 pp guardrail (5.709 / 5.863).
Adding bridges to the bldg-only PoC **regresses** run MAE by
**+0.695 pp** and weighted MAE by **+0.275 pp**, with essentially
unchanged window correlation.

## Why bridges hurt the UTD edge feature

The bridge mesh adds many edge candidates in regions densely covered
by elevated viaducts (Tokyo Monorail, Yurikamome, Shuto Expressway 11
along the 浜松町〜芝浦 trajectory). Tokyo run2 focus-window comparison
of `utd_score_nlos_sum_mean`:

| window | actual | §7.16 corrected | bldg-only score | bldg+brid score | Δ |
|---|---:|---:|---:|---:|---:|
| w7 (false-high)   |   0.0% | 39.5% | 0.418 |  **115.355** | +114.937 |
| w9 (false-high)   |   0.0% | 56.7% | 2.276 |  **101.631** |  +99.355 |
| w23 (hidden-high) | 100.0% | 32.0% | 7.592 |   39.198 |  +31.606 |
| w24 (hidden-high) | 100.0% | 32.0% | 0.000 |   35.833 |  +35.833 |
| w25 (hidden-high) | 100.0% | 48.1% | 4.470 |   70.171 |  +65.701 |
| w26 (hidden-high) |  96.7% | 50.3% | 8.765 |   71.779 |  +63.014 |
| w27 (hidden-high) |  75.3% | 39.2% | 0.773 |   12.589 |  +11.817 |

Under bldg-only the hidden-high windows w23/w26 had the highest score
(7-9), false-high w7/w9 the lowest (0.4-2.3). With bridges this
**rank inverts**: false-high windows now lead with 102-115 while
hidden-high w23-w27 sit at 13-72. The features are activating MORE on
the false-high windows than on the hidden-high ones because the
elevated viaduct geometry is densest where the false-high windows
sit, not where demo5 actually fails to FIX. The classifier sees a
stronger occluder signal that is **anti-correlated** with the failure
it needs to predict.

This is the same pattern as the §7.25b bldg+brid antenna retrain
(`PLATEAU_BRIDGE_RETRAIN_NULL.md`): bridges shift values significantly
but in directions that describe the visibility environment, not demo5
ambiguity health. Adding more visibility-equivalent signal makes the
classifier *more* confidently wrong on the very windows where the
8.13 pp ceiling lives.

## Pooled univariate is marginal but real

Pooled correlations on 197 LORO-evaluation windows did improve over
bldg-only:

| feature | r vs actual (bldg-only) | r vs actual (bldg+brid) | Δ |
|---|---:|---:|---:|
| utd_min_excess_path_m_mean        | +0.147 | **+0.212** | +0.065 |
| utd_candidate_nlos_sat_count_mean | -0.184 | **-0.201** | +0.017 |
| utd_min_fresnel_v_mean            | (n/a)  | +0.186 | -- |
| utd_min_excess_path_m_min         | (n/a)  | +0.172 | -- |
| utd_score_nlos_sum_max            | -0.171 | -0.103 | -0.068 |
| utd_candidate_count_total_max     | (n/a)  | -0.029 | -- |

Two features now exceed |r| ≥ 0.20, one of them passing the
Bonferroni-adjusted alpha=0.05 bar (≈ 0.21). Correlations vs §7.16
error also strengthen (nlos_sat_count_max: +0.044 → +0.170;
min_excess_path_m_min: -0.051 → -0.176).

But the retrain selection test is decisive: even with the marginal
univariate improvement, the residual head built on the larger feature
set generalises *worse* under LORO. The +0.695 pp regression confirms
that the feature shift is reorienting the classifier toward
visibility-dense regions that are not the actual demo5 failure
windows.

## Decision

**bldg+brid UTD edge candidates is null and strictly worse than
bldg-only UTD edge candidates.** Combined with:

- §7.25 bldg-only antenna null (4.795 pp);
- §7.25b bldg+brid antenna null (5.023 pp, +0.228 over bldg-only);
- UTD edge bldg-only null (5.168 pp);
- **UTD edge bldg+brid null (5.863 pp, +0.695 over bldg-only)**;
- path 1 solver-state lightweight wrapper null (PR #43-44);
- path 3 architectural pivot null (HMM PoC 739a8fa).

Two clear patterns now hold across all simulator-side feature
attacks:

1. The 14 antenna features and the 20 UTD edge features both live in
   the same equivalence class as §7.16's existing `sim_los_*` and
   validationhold visibility families. Adding them produces nested
   stack regressions because the residual head sees correlated
   restatements of an already-available signal.
2. Adding bridges to either family makes the regression **worse**,
   not better, because bridges densify the visibility signal in
   false-high regions (along elevated viaducts) without resolving the
   demo5 ambiguity-health information that hidden-high windows
   actually depend on.

The 8.13 pp tokyo/run2 ceiling (PR #42 accepted) holds. The remaining
plausible directions, in order:

1. **Full UTD diffraction coefficient** (Keller wedge + received-power
   model, 7-14 day): per the bldg-only PoC's own kill criterion ("If
   this is null, full UTD is less likely to help"), the prior on full
   UTD breaking the ceiling on the deployed contract is now well below
   10 %. The marginal +0.065 univariate r improvement from bridges
   does not change this; the structural problem is still that
   visibility-class features cannot identify demo5 ambiguity health.
2. **PLATEAU LoD3 + manual elevated OBJ** (3-5 day): same equivalence
   class as bridges; expected to produce the same pattern (regress on
   false-high regions with dense added geometry).
3. **More routes / more data** (path 2, operator-side): out of scope.
4. **LASSO / sparse selection over the existing 5867 features**:
   could in principle find a sparse classifier that suppresses the
   visibility-confounded subspace. Untried as of 2026-05-01; priors
   weak (existing classifier already does feature subsampling).
5. **Multi-frequency (L1/L2/L5) features**: PPC dataset is L1-only,
   operator-side.

ML / feature-engineering side of the simulator search space is
treated as **fully exhausted** for the deployed pre-demo5 contract.

## Reproducer

```bash
# 1. Per-run UTD edge extraction with --include-bridges (6 runs)
for run in nagoya/run1 nagoya/run2 nagoya/run3 tokyo/run1 tokyo/run2 tokyo/run3; do
  city=${run%/*}; rname=${run##*/}
  preset=$([ "$city" = tokyo ] && echo tokyo23 || echo nagoya)
  zone=$([ "$city" = tokyo ] && echo 9 || echo 7)
  python3 experiments/exp_ppc_utd_edge_diffraction_features.py \
    --run-dir "$PPC_DATASET_ROOT/$run" \
    --preset "$preset" --plateau-zone "$zone" \
    --epoch-stride 60 --edge-voxel-size-m 5 --max-candidate-edges 50000 \
    --include-bridges \
    --results-prefix "ppc_utd_edges_BRID_egm96_s60_${city}_${rname}"
done

# 2. Aggregate to pooled per_window CSV
python3 experiments/aggregate_utd_features.py \
  --prefix ppc_utd_edges_BRID_egm96_s60 \
  --output experiments/results/ppc_utd_edges_BRID_egm96_pooled_per_window.csv

# 3. Augment §7.16 base window CSV with utd_edge_* features
python3 experiments/augment_window_csv_with_utd.py \
  --base-csv  experiments/results/ppc_window_..._validationhold_current_tight_hold_carry_window_predictions.csv \
  --utd-csv   experiments/results/ppc_utd_edges_BRID_egm96_pooled_per_window.csv \
  --output-csv experiments/results/ppc_window_..._with_utd_BRID_egm96_window_predictions.csv

# 4. Retrain nested stack
python3 experiments/train_ppc_solver_transition_surrogate_nested_stack.py \
  --window-csv experiments/results/ppc_window_..._with_utd_BRID_egm96_window_predictions.csv \
  --base-prefix ppc_window_fix_rate_model_..._windowopt_baseerror15_refinedgrid \
  --classifier-include-run-position \
  --alphas 0.75 --residual-clip-pp 50 \
  --max-run-mae-pp 4.5 --max-abs-aggregate-error-pp 2.0 \
  --results-prefix ppc_window_..._alpha75_utd_BRID_egm96_meta_run45
```

Outputs in `experiments/results/...alpha75_utd_BRID_egm96_meta_run45_*.csv`.
