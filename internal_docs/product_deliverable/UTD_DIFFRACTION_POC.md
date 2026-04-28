# UTD edge-diffraction candidate features

This is the first implementation step for the UTD follow-up identified
after PR #12.  It is not yet a full Uniform Theory of Diffraction
solver.  The goal is to create a reproducible, deployable feature
family that tests whether PLATEAU building-edge geometry carries signal
that the existing LoS, BVH reflection, and simplified antenna probes do
not.

## What is implemented

`experiments/utd_edge_features.py` provides pure geometry helpers:

- extract candidate diffraction edges from the PLATEAU triangle mesh;
- keep non-coplanar shared edges, optionally keeping boundary edges
  because PLATEAU polygons are often not topologically welded;
- remove coplanar fan-triangulation diagonals;
- thin edges by midpoint voxel for tractable PPC sweeps;
- score receiver-satellite rays by nearest edge distance, excess path,
  Fresnel-like distance, and an exponential candidate score.

`experiments/exp_ppc_utd_edge_diffraction_features.py` runs those
helpers for one PPC run and writes:

- `experiments/results/ppc_utd_edges_s<stride>_<city>_<run>_per_epoch.csv`
- `experiments/results/ppc_utd_edges_s<stride>_<city>_<run>_per_window.csv`

The default stride is 60 epochs and the default aggregation window is
60 seconds, matching the §7.16 product-deliverable window IDs.  This is
intended as the first full six-run sweep before spending engineering
time on GPU UTD coefficients.

## Physical interpretation

For each satellite ray, the script looks for candidate building edges
near the receiver-to-satellite line.  A candidate contributes when:

- its closest point lies in front of the receiver;
- it is within `--max-edge-range-m` of the receiver;
- it is within `--max-ray-edge-distance-m` of the ray;
- the two-segment path excess
  `|rx-edge| + |sat-edge| - |rx-sat|` is below
  `--max-excess-path-m`.

The exported feature family includes candidate satellite counts,
NLoS-only candidate counts, minimum excess path, minimum edge distance,
minimum Fresnel-like parameter, and aggregate scores.

This is intentionally closer to a knife-edge / wedge-candidate detector
than to a final received-power model.  If this is null, full UTD is less
likely to help the current 197-window LORO stack.  If it is non-null,
the next step is a real UTD coefficient and attenuation model.

## Runbook

One-run smoke, Tokyo run2:

```bash
python3 experiments/exp_ppc_utd_edge_diffraction_features.py \
  --run-dir /media/sasaki/aiueo/ai_coding_ws/datasets/PPC-Dataset-data/tokyo/run2 \
  --preset tokyo23 \
  --plateau-zone 9 \
  --epoch-stride 60 \
  --results-prefix ppc_utd_edges_s60_tokyo_run2
```

Nagoya uses the Nagoya PLATEAU preset:

```bash
python3 experiments/exp_ppc_utd_edge_diffraction_features.py \
  --run-dir /media/sasaki/aiueo/ai_coding_ws/datasets/PPC-Dataset-data/nagoya/run1 \
  --preset nagoya \
  --plateau-zone 7 \
  --epoch-stride 60 \
  --results-prefix ppc_utd_edges_s60_nagoya_run1
```

After all six per-run CSVs exist:

```bash
python3 experiments/aggregate_utd_features.py --prefix ppc_utd_edges_s60

python3 experiments/augment_window_csv_with_utd.py
```

Then retrain the §7.16 nested stack on the UTD-augmented CSV:

```bash
python3 experiments/train_ppc_solver_transition_surrogate_nested_stack.py \
  --window-csv experiments/results/ppc_window_fix_rate_model_stride1_stat_sim_rinex_phasejump_t0p25_gf0p2_simloscont_focused_simadop_nowt_windowopt_validationhold_current_tight_hold_carry_with_utd_window_predictions.csv \
  --base-prefix ppc_window_fix_rate_model_stride1_stat_sim_rinex_phasejump_t0p25_gf0p2_simloscont_focused_simadop_nowt_windowopt_baseerror15_refinedgrid \
  --classifier-include-run-position \
  --alphas 0.75 \
  --residual-clip-pp 50 \
  --max-run-mae-pp 4.5 \
  --max-abs-aggregate-error-pp 2.0 \
  --results-prefix ppc_window_fix_rate_model_..._utd_alpha75_meta_run45
```

Compare against the adopted §7.16 row:

| variant | wmae pp | run MAE pp | corr |
| --- | --- | --- | --- |
| §7.16 adopted | **17.087** | **3.202** | **0.551** |
| §7.16 + UTD edge candidates, ridge alpha=0.75 | 18.264 | 5.168 | 0.471 |
| §7.16 + UTD edge candidates, ExtraTrees alpha=0.75 | 18.275 | 5.168 | 0.404 |
| selected under guardrails | 18.046 | 4.436 | 0.401 |

The guardrail-selected row falls back to the conservative base model
because both UTD-augmented residual variants exceed
`--max-run-mae-pp 4.5`.  Relative to §7.16, the UTD edge features are a
clear null / regression: weighted MAE worsens by at least +1.18 pp, run
MAE worsens by +1.97 pp, and correlation drops by at least -0.080.

## Full six-run result

All six runs were extracted at epoch stride 60 with
`--edge-voxel-size-m 5` and `--max-candidate-edges 50000`.

| run | sampled epochs | windows | retained edges | notes |
| --- | ---: | ---: | ---: | --- |
| nagoya/run1 | 126 | 26 | 50,000 | 4,494 boundary |
| nagoya/run2 | 158 | 32 | 50,000 | 5,980 boundary |
| nagoya/run3 | 87 | 18 | 50,000 | 8,250 boundary |
| tokyo/run1 | 198 | 40 | 50,000 | one trailing window is outside the §7.16 product CSV |
| tokyo/run2 | 152 | 31 | 50,000 | focus run |
| tokyo/run3 | 255 | 51 | 50,000 | 12 boundary |

The pooled UTD CSV has 198 windows; 197 join with the adopted §7.16
product-deliverable window set.

Univariate pooled correlations against demo5 actual FIX rate and §7.16
prediction error are weak:

| feature | r vs actual FIX | r vs §7.16 error |
| --- | ---: | ---: |
| `utd_candidate_nlos_sat_count_max` | -0.195 | +0.044 |
| `utd_candidate_nlos_sat_count_mean` | -0.184 | +0.037 |
| `utd_candidate_sat_count_mean` | -0.176 | -0.023 |
| `utd_score_nlos_sum_max` | -0.171 | +0.018 |
| `utd_score_sum_max` | -0.156 | -0.028 |
| `utd_min_excess_path_m_mean` | +0.147 | -0.051 |

No UTD edge feature reaches |r| = 0.20 on the pooled 197-window set.
That is already below the useful-feature threshold before considering
the 20 parallel feature correlations.  The retrain result above is the
decisive selection test: the features do not improve the nested stack.

Artifacts:

- `experiments/results/ppc_utd_edges_s60_<city>_<run>_per_epoch.csv`
- `experiments/results/ppc_utd_edges_s60_<city>_<run>_per_window.csv`
- `experiments/results/ppc_utd_edges_pooled_per_window.csv`
- `experiments/results/ppc_window_fix_rate_model_..._utd_alpha75_meta_run45_*.csv`

## Initial Tokyo run2 smoke result

The implementation was smoke-tested on the hardest focus run:

```bash
python3 experiments/exp_ppc_utd_edge_diffraction_features.py \
  --run-dir /media/sasaki/aiueo/ai_coding_ws/datasets/PPC-Dataset-data/tokyo/run2 \
  --preset tokyo23 \
  --plateau-zone 9 \
  --epoch-stride 60 \
  --edge-voxel-size-m 5 \
  --max-candidate-edges 50000 \
  --results-prefix ppc_utd_edges_s60_tokyo_run2
```

Runtime on Tokyo run2:

| stage | result |
| --- | --- |
| RINEX / nav load | 45.9 s, 9106 epochs |
| PLATEAU + BVH build | 105.7 s, 2,200,133 triangles |
| edge extraction | 121.4 s, 50,000 retained edges |
| epoch scoring | 18.1 s, 152 sampled epochs |
| output windows | 31 windows |

The initial per-window smoke outputs are committed as:

- `experiments/results/ppc_utd_edges_s60_tokyo_run2_per_epoch.csv`
- `experiments/results/ppc_utd_edges_s60_tokyo_run2_per_window.csv`

Tokyo run2-only correlations are not a model-selection result, but they
are a useful early diagnostic:

| feature | r vs actual FIX | r vs §7.16 error |
| --- | --- | --- |
| `utd_candidate_nlos_sat_count_max` | -0.330 | +0.268 |
| `utd_candidate_count_nlos_max` | -0.329 | +0.222 |
| `utd_score_nlos_sum_max` | -0.299 | +0.200 |
| `utd_candidate_sat_count_max` | +0.245 | -0.231 |

Focus windows:

| window | actual | §7.16 corrected | nlos sat mean | nlos candidate mean | nlos score mean |
| --- | --- | --- | --- | --- | --- |
| w7 | 0.0 % | 39.5 % | 0.6 | 6.0 | 0.418 |
| w9 | 0.0 % | 56.7 % | 0.4 | 3.2 | 2.276 |
| w23 | 100.0 % | 32.0 % | 0.8 | 28.8 | 7.592 |
| w24 | 100.0 % | 32.0 % | 0.0 | 0.0 | 0.000 |
| w25 | 100.0 % | 48.1 % | 0.4 | 13.8 | 4.470 |
| w26 | 96.7 % | 50.3 % | 0.8 | 17.0 | 8.765 |
| w27 | 75.3 % | 39.2 % | 0.2 | 1.6 | 0.773 |

Interpretation: the features are active and directionally related to
some Tokyo run2 residuals, but they do not cleanly separate false-high
w7/w9 from hidden-high w23-w27.  The all-six-run pool and §7.16 retrain
above supersede this smoke diagnostic and make the UTD edge-candidate
family a null result for the current product model.

## Implementation limits

- This does not compute Keller / UTD wedge coefficients.
- Edge visibility is approximated by geometric proximity to the
  receiver-satellite ray; it does not yet ray-trace receiver-to-edge and
  edge-to-satellite legs separately.
- Boundary edges are included by default because PLATEAU LOD2 topology
  is not guaranteed to be welded.  Use `--exclude-boundary-edges` for a
  stricter welded-edge-only ablation.
- Runtime is controlled by `--edge-voxel-size-m`,
  `--max-candidate-edges`, `--max-edge-range-m`, and
  `--epoch-stride`.

## Validation

Pure geometry unit tests:

```bash
pytest -q tests/test_utd_edge_features.py
```

The tests verify that coplanar fan diagonals are removed, welded box
edges survive, and an edge exactly on a receiver-satellite ray produces
a positive candidate score.
