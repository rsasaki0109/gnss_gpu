# Reflection / canyon proxy PoC (null result)

Tested two routes for adding multipath / diffraction information to the
gnss_gpu PPC simulator after the §7.20 plateau and the §6 paper-style
spatial-disagreement diagnostic.  Both ended as null.

## Test 1: BVH first-order multipath via `compute_multipath`

Script: `experiments/exp_ppc_reflection_poc.py`

Result: **infrastructure blocker.**

- `gnss_gpu/bvh.py::BVHAccelerator.compute_multipath` exists but
  imports `gnss_gpu._bvh.raytrace_multipath_bvh`, which is **not
  present in the compiled `_bvh.cpython-*.so`**.  ImportError.
- Falling back to `gnss_gpu.raytrace.BuildingModel.compute_multipath`
  (no BVH, brute-force ray-vs-all-triangles in CUDA) hits **CUDA OOM
  at the first epoch** with 2.2M triangles × ~25 satellites.  Even on
  a 16 GB device the kernel allocates more than the available free
  memory.
- Two valid recovery paths exist but were declined for this PoC:
  - Compile the BVH multipath C++ kernel (build-system change, 30-60
    min if `nvcc` and `pybind11` are wired up).
  - Per-epoch triangle filtering (keep only triangles within ~500 m
    of the receiver before passing to brute-force multipath).

## Test 2: Heuristic canyon / edge proxy (BVH check_los only)

Script: `experiments/exp_ppc_canyon_proxy_poc.py`

This avoids the multipath kernel by using only the BVH `check_los`
primitive that *is* compiled.  Two deployable proxies are computed:

- **`canyon_blocked_count`**: cast 16 low-elevation rays in a circle
  from the receiver, count how many are blocked by buildings within
  150 m.  0 = open sky, 16 = surrounded.
- **`edge_near_sat_count`**: for each tracked satellite, also test
  `check_los` at +/- 3° azimuth perturbations.  A satellite whose
  three samples disagree is "near a building edge" → diffraction or
  multipath risk.

Tokyo run2, stride 5, 1822 epochs, 21 min wall (BVH build 3 min +
loop 21 min for ~165k LoS calls; Python per-call overhead dominates).

### Numbers

Per-window pooled correlations against demo5 actual FIX rate:

| feature | corr vs actual_fix_rate_pct | corr vs §7.16 error (corrected − actual) |
| --- | --- | --- |
| canyon_blocked_count_mean | -0.01 | +0.07 |
| canyon_blocked_count_max | -0.08 | +0.16 |
| canyon_blocked_fraction_mean | -0.01 | +0.07 |
| edge_near_sat_count_mean | **-0.18** | **+0.24** |
| edge_near_sat_fraction_mean | -0.12 | +0.17 |
| edge_near_sat_count_max | -0.16 | +0.22 |

Focus-window inspection:

| window | actual | §7.16 corrected | canyon mean / max | edge mean |
| --- | --- | --- | --- | --- |
| w7 (false-high) | 0.0 % | 39.5 % | 7.47 / 8 | 2.70 |
| w9 (false-high) | 0.0 % | 56.7 % | 3.67 / 8 | 0.57 |
| w23 (hidden-high) | 100.0 % | 32.0 % | 2.03 / 4 | 1.07 |
| w24 (hidden-high) | 100.0 % | 32.0 % | 4.00 / 4 | 1.90 |
| w25 (hidden-high) | 100.0 % | 48.1 % | 5.13 / 7 | 1.47 |
| w26 (hidden-high) | 96.7 % | 50.3 % | 4.83 / 7 | 0.30 |
| w27 (hidden-high) | 75.3 % | 39.2 % | 4.00 / 4 | 0.00 |

w7 alone has the predicted "deep canyon → multipath → demo5 fails"
signature: 7.47 of 16 azimuth bins blocked plus 2.7 sats near building
edges, while actual FIX is 0 %.  But w9 (also a false-high window in
the same area) is moderate, and the hidden-high cluster w23-w27 is
inconsistent — w24 has high canyon score and high edge density yet
demo5 successfully fixes.

The aggregate correlations of -0.01 to -0.18 against actual FIX, and
+0.07 to +0.24 against §7.16 prediction error, are far below the
useful-feature threshold for a 30-window LORO regression at this
sample size.

### Interpretation

- The §7.16 failures on Tokyo run2 are **not dominated by simple
  geometric multipath risk** that a heuristic LoS-only proxy can
  capture.
- The remaining gap to demo5's behaviour likely involves **validation
  / hold internal state** (already partially captured by the
  validationhold surrogate adopted in §7) rather than residual
  multipath geometry.
- Closing the rest of the gap empirically would require either:
  - real first-order multipath modelling at scale (compile the BVH
    multipath kernel),
  - UTD-style diffraction modelling (Furukawa 2019's approach;
    significant engineering), or
  - 3D model patches for elevated-road / monorail structures
    omitted from PLATEAU LOD2.

## Conclusion

No additional gain is available from "free" open-data simulator
upgrades inside the current build / dataset.  §7.16 remains the
reported best.  The PoC scripts are committed under `experiments/`
so this null result is reproducible and the work does not need to be
repeated for the next dataset increment.

## Addendum: BVH multipath actually tested after rebuild

The BVH multipath binding was subsequently rebuilt from source
(`cmake --build build --target _bvh`) because the kernel and pybind
registration both existed but the compiled `.so` was stale.  With
the rebuilt binding, `BVHAccelerator.compute_multipath` runs
successfully at ~6 s/epoch on the 2.2M-triangle Tokyo run2 PLATEAU
set.

Stride 200 (46 epochs) over Tokyo run2 gives per-window reflection
features.  Correlations against demo5 actual FIX (n=24 windows after
merge with §7.16 predictions):

| feature | corr vs actual FIX | corr vs §7.16 error |
| --- | --- | --- |
| reflection_count_mean | **-0.08** | +0.04 |
| excess_delay_m_max_mean | -0.19 | +0.12 |
| nlos_count_mean | +0.04 | +0.16 |

Focus-window decisive examples:

- Tokyo run2 w9 (actual 0 %): reflection count = 0, excess delay = 0.
  The simulator sees no multipath yet demo5 fails.
- Tokyo run2 w24 / w25 (actual 100 %): reflection count = 16-18,
  excess delay ~18 m.  Simulator sees heavy multipath yet demo5
  fixes cleanly.
- Tokyo run2 w26 (actual 97 %): reflection = 0, excess = 0.
  Simulator agrees with demo5.

The reflection count is nearly uncorrelated (and slightly
anti-correlated on the focus windows) with demo5 FIX success.  Raw
geometric reflections against the LOD2 building set do not
distinguish where demo5 fails.

Hypothesis for why: BVH multipath flags reflections against every
triangle (including ground-plane facets and small features) without
attenuation / antenna-pattern weighting.  demo5's ambiguity
validation appears robust to most of the reflections that this
model reports.  Furukawa 2019's 88 % result likely depends on the
UTD diffraction + antenna gain + signal-attenuation modelling that
the gnss_gpu BVH kernel does not implement, rather than on
first-order geometric reflections alone.

Consequence: even with BVH multipath successfully available, it does
not provide a useful feature for the §7.16 nested stack on this
dataset.  The combined conclusion across all simulator-side
PoCs — brute-force OOM, canyon heuristic, BVH multipath — is that
§7.16 has extracted the available simulation signal for this
feature pipeline.

Artifacts for reproducibility:
- `experiments/results/ppc_reflection_bvh_s200_tokyo_run2_per_epoch.csv`
- `experiments/results/ppc_reflection_bvh_s200_tokyo_run2_per_sat.csv`
- `experiments/results/ppc_reflection_bvh_s200_tokyo_run2_per_window.csv`

## Final attempt: integrate BVH multipath features into the §7.16 stack

After confirming the reflection features are weakly anti-correlated
with §7.16 prediction error, the experiment was extended to all 6
runs at stride 60 (~75 min total compute) and the resulting per-window
reflection features were merged into the §7.16-augmented window CSV
and the nested stack was re-trained.

Pooled correlations across 197 LORO-merged windows:

| feature | corr vs actual FIX | corr vs §7.16 error |
| --- | --- | --- |
| reflection_count_mean | +0.014 | +0.024 |
| reflection_count_max | +0.003 | +0.042 |
| excess_delay_m_max_mean | +0.179 | -0.193 |
| excess_delay_m_max_max | +0.186 | -0.201 |
| excess_delay_m_p90_mean | +0.179 | -0.193 |
| nlos_count_mean | -0.142 | +0.110 |
| sat_count_mean | +0.333 | -0.161 |

`sat_count_mean` is the strongest single feature at +0.333 — but it
is a basic visibility count that the §7.16 pipeline already has via
many `sim_*` aggregates.

Retrained §7.16 nested stack with the 7 reflection features added as
`refl_*` columns:

| variant | wmae pp | run MAE pp | corr |
| --- | --- | --- | --- |
| §7.16 adopted | **17.087** | **3.202** | **0.551** |
| §7.16 + reflection features | 18.358 | 4.510 | 0.428 |

Adding the reflection features **strictly worsens every metric**
by 1.2-1.3 pp on wmae / run MAE.  This matches the §7.17 hold_age
null where introducing additional correlated features dilutes the
ridge feature sampling.

Combined conclusion across every simulator-side approach in the
session — brute-force OOM, canyon heuristic, BVH multipath
correlation, BVH multipath retrain — is that §7.16 has fully
extracted the deployable simulation signal in the current dataset
and pipeline.  Further gain requires either:

- 7+ days of UTD diffraction + antenna-pattern + signal-attenuation
  modelling (Furukawa 2019's full machinery),
- 3-5 days of LOD3 PLATEAU + manual elevated-road / monorail
  geometry patching for each focus failure window, or
- Additional run-level data growth (rejected by the user).

§7.16 (`current_tight_hold + carry + alpha=0.75`,
17.087 / **3.202** / **0.551**) remains the reported best strict
research direction.  No further empirical improvements are
achievable with the resources available in-session.

Artifacts for the all-runs reflection experiment:
- `experiments/exp_ppc_reflection_poc.py` (BVH multipath PoC)
- `experiments/aggregate_reflection_features.py` (pooled correlation)
- `experiments/augment_window_csv_with_reflection.py` (merge into §7.16 CSV)
- `experiments/results/ppc_reflection_bvh_s60_<city>_<run>_per_*.csv` (× 6 runs × 3 levels)
- `experiments/results/ppc_reflection_bvh_pooled_per_window.csv`
- `experiments/results/ppc_window_fix_rate_model_..._with_reflection_alpha75_meta_run45_*.csv`
