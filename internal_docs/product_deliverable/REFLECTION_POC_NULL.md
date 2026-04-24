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
