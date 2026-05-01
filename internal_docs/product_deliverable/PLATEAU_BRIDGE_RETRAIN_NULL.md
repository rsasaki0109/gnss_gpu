# PLATEAU bldg+brid antenna retrain: NULL

**Status**: bldg+brid retrain confirmed null on the deployed-contract
nested stack.
**Date**: 2026-05-01.
**Source**: PR #38 (bridge loader) + `PLATEAU_BRIDGE_INTEGRATION.md`
follow-up.
**Stacked on**: §7.25 bldg-only antenna null (`REFLECTION_POC_NULL.md`,
run MAE 4.795 pp).

## Summary

Re-running the §7.25 antenna-augmented nested stack with the
EGM96-corrected bldg+brid mesh (instead of bldg-only) **does not pass**
the `--max-run-mae-pp 4.5` selection guardrail and is slightly **worse**
than the bldg-only antenna run on every metric except window
correlation. The 8.13 pp tokyo/run2 ceiling reported in
`README.md` §3 holds.

| variant | wmae pp | run MAE pp | corr |
|---|---:|---:|---:|
| §7.16 + iso75 + phaseguard (adopted) | **15.85** | **1.79** | **0.559** |
| bldg-only antenna alpha=0.75 (§7.25) | 17.886 | 4.795 | 0.459 |
| **bldg+brid antenna ridge alpha=0.75** | **17.182** | **5.023** | **0.508** |
| bldg+brid antenna extra_trees alpha=0.75 | 18.336 | 5.080 | 0.383 |
| conservative baseline (no model) | 18.046 | 4.436 | 0.401 |

Both bldg+brid variants exceed the 4.5 pp guardrail (5.023 / 5.080 vs
4.5), so neither is adoptable. Window correlation **improves** on the
bldg-only run (+0.049) and weighted MAE improves (-0.704), but route MAE
**regresses** by +0.228 pp.

## Setup (preserves `with_antenna` schema)

- Base CSV: `..._validationhold_current_tight_hold_carry_with_antenna_window_predictions.csv`
  (197 evaluation windows, 14 `ant_*` columns from the bldg-only
  antenna run).
- BRID per-window CSVs: 6 files
  `ppc_antenna_features_BRID_egm96_<city>_<run>_per_window.csv`
  (395 rows total; 197 match on `(city, run, window_index)`).
- Merge script: `experiments/merge_brid_into_window_csv.py` replaces
  the 14 `ant_*` values with the BRID equivalents on each matched row;
  unmatched rows would keep bldg-only values, but all 197 windows
  matched.
- Train script: `experiments/train_ppc_solver_transition_surrogate_nested_stack.py`
  with `--alphas 0.75 --residual-clip-pp 50 --max-run-mae-pp 4.5
  --max-abs-aggregate-error-pp 2.0 --classifier-include-run-position`,
  same `--base-prefix` as the adopted run.

## Why bridges did not help

The bldg+brid mesh shifts the 14 `ant_*` feature **values** as predicted
in `PLATEAU_BRIDGE_INTEGRATION.md` (Tokyo run2 mean
`ant_eff_db_p50_mean` -4.013 → -8.303, mean `ant_nlos_at_high_elev_count_mean`
+0.450 → +2.348). It does **not** add a new feature class.

The §7.25 diagnosis from `REFLECTION_POC_NULL.md` applies unchanged:
"the new features mostly restate elevation and LoS visibility, already
represented by the §7.16 `sim_los_*` and validationhold feature
families." Adding bridges to the antenna pipeline produces a richer
visibility signal but does not break out of the equivalence class with
the existing visibility features.

The +0.049 window-correlation lift over bldg-only confirms bridges are
real occluders that the input layer can see, but the gain at the
window level does not propagate to the route aggregate that the
selection guardrail measures.

## Decision

**bldg+brid antenna retrain is null.** Combined with:
- §7.25 bldg-only antenna null (4.795 pp);
- path 1 solver-state lightweight wrapper null (PR #43-44);
- path 3 architectural pivot null (PR #45 sketch + PR #46 lag-GBR
  + commit 739a8fa HMM PoC);
- §7.17 / §7.18 / §7.19 / §7.20 / §7.24.x null entries.

The adopted §7.16 + iso75 + phaseguard model (`run45`,
run MAE 1.79 pp / window MAE 15.85 / corr 0.559) and the tokyo/run2
**8.13 pp ceiling** are confirmed against every ML and physics-proxy
attack tested to date.

Remaining plausible directions, in order:

1. **UTD diffraction** (7-14 day): missing physics from Furukawa 2019;
   requires edge detection on the PLATEAU+brid mesh and a UTD GPU
   implementation. This is the only physics-side path with prior > 10 %
   of moving the ceiling.
2. **PLATEAU LoD3 + manual high-elevation OBJ** (3-5 day):
   Tokyo Monorail / Daiba viaducts that the LoD2 brid mesh does not
   cover at full geometric fidelity.
3. **More routes / more data** (path 2, operator-side): out of scope
   for the model.
4. **LASSO / sparse selection over the 5867 §7.16 features**: untried
   as of 2026-05-01, but priors are weak — the existing classifier
   already does its own feature subsampling.
5. **Multi-frequency (L1/L2/L5) features**: PPC dataset is L1-only,
   so this is also operator-side.

ML / feature-engineering side of the search space is treated as
exhausted.

## Reproducer

```bash
# 1. Merge BRID per_window CSVs into the augmented window CSV
python3 experiments/merge_brid_into_window_csv.py \
  --window-csv experiments/results/ppc_window_fix_rate_model_stride1_stat_sim_rinex_phasejump_t0p25_gf0p2_simloscont_focused_simadop_nowt_windowopt_validationhold_current_tight_hold_carry_with_antenna_window_predictions.csv \
  --brid-dir experiments/results \
  --output-csv experiments/results/ppc_window_fix_rate_model_stride1_stat_sim_rinex_phasejump_t0p25_gf0p2_simloscont_focused_simadop_nowt_windowopt_validationhold_current_tight_hold_carry_with_antenna_BRID_egm96_window_predictions.csv

# 2. Retrain nested stack on the BRID-merged CSV
python3 experiments/train_ppc_solver_transition_surrogate_nested_stack.py \
  --window-csv experiments/results/...with_antenna_BRID_egm96_window_predictions.csv \
  --base-prefix ppc_window_fix_rate_model_stride1_stat_sim_rinex_phasejump_t0p25_gf0p2_simloscont_focused_simadop_nowt_windowopt_baseerror15_refinedgrid \
  --classifier-include-run-position \
  --alphas 0.75 --residual-clip-pp 50 \
  --max-run-mae-pp 4.5 --max-abs-aggregate-error-pp 2.0 \
  --results-prefix ppc_window_fix_rate_model_..._alpha75_antenna_BRID_egm96_meta_run45
```

Outputs in `experiments/results/...alpha75_antenna_BRID_egm96_meta_run45_*.csv`.
