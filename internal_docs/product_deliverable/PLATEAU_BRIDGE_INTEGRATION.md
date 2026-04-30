# PLATEAU bridge geometry integration (Phase 1)

This is the first staged step toward the "PLATEAU LOD3 + 高架手動 OBJ"
follow-up identified in the PR-#12-era handoff and reaffirmed after the
§7.25 antenna null and the UTD edge-candidate null
(`UTD_DIFFRACTION_POC.md`).

The product-level motivation is the residual Tokyo run2 8.13 pp route
MAE that survives the §7.16 + isotonic75 + phaseguard adopted model
(`README.md` §3): five of six routes are within 1.5 pp of actual, and
the lone exception is the Tokyo run2 hidden-high cluster w23-w27 in the
浜松町〜芝浦 area (lat 35.6532-35.6568, lon 139.7925-139.7943, ellipsoidal
height ~40-43 m).  Geographic inspection places those windows directly
along the Tokyo Monorail elevated track and the 首都高 11 号線 viaducts.

## Why bridges, not LOD3

`fetch_plateau_subset.py` only extracts `udx/bldg/` GML files, and
`load_plateau` only parses `bldg:Building` features.  PLATEAU Tokyo23
also ships:

- 850 `udx/brid/` GML files at LOD2 (full 3D bridge geometry);
- 669 `udx/tran/` GML files at LOD1 (planar road centerlines, no 3D
  superstructure);
- smaller `frn`, `fld`, `dem`, `luse`, `urf`, `lsld`, `htd` collections.

For the missing-occluder problem in Tokyo run2, the LOD2 bridge files
are the only ones that contribute new ray-blocking surfaces.  Tran is
LOD1 only, and roof-detail LOD3 buildings are unrelated to the elevated
viaduct gap.

Inspection of `53393683_brid_6697_op.gml` (the third-level mesh that
contains all five hidden-high windows) shows 20 `brid:Bridge` features
with 7,376 triangles, an XY bbox that fully covers the trajectory, and
ellipsoidal heights in [1.6 m, 28.4 m] (median 9.1 m, p90 18.6 m).
Those numbers match elevated monorail tracks (~10-15 m above ground)
and expressway viaducts (~15-25 m).  By contrast the bldg-only mesh for
the same tile has 125,245 triangles up to 186 m, so the bridge mesh is
about 6 % of the building volume but located precisely where the
trajectory is.

## What this step delivers

This commit is the loader / fetch tooling only.  It does not yet
re-run the simulator pipeline or retrain §7.16, because those need a
fresh editable install of `gnss_gpu` rebuilt against this branch (the
sister tree `/media/sasaki/aiueo/ai_coding_ws/gnss_gpu` is currently
pinned to a different feature branch via
`/home/sasaki/.local/lib/python3.12/site-packages/_gnss_gpu_editable.pth`,
and that path's compiled bindings shadow our worktree at import time).

Concretely:

- `experiments/fetch_plateau_subset.py` gets a new `--include-bridges`
  flag.  When supplied, after the existing `udx/bldg/` extraction it
  also matches `udx/brid/` GML files for the same trajectory mesh codes
  and writes them into the same output directory.  The manifest gains
  `include_bridges`, `n_bridge_entries`, and `bridge_entries` fields.
- `python/gnss_gpu/io/citygml.py::parse_citygml` accepts a new
  `kind="bldg"` / `kind="brid"` argument; the brid kind reuses the
  existing `_extract_polygons` and `_determine_lod` helpers and
  returns the bridges as `Building` dataclasses (the geometry shape is
  identical, only the root element name differs).
- `python/gnss_gpu/io/plateau.py::PlateauLoader.load_directory` and
  `load_plateau` get a new `include_bridges=False` flag.  When True, the
  directory loader picks up `_brid_*.gml` files in addition to
  `_bldg_*.gml` and merges their triangles into the same
  `BuildingModel`.  Single-file `load_plateau` infers the kind from the
  filename (`_brid_` → bridge, anything else → building), so existing
  callers are unaffected.
- `tests/test_plateau.py` gets a minimal bridge fixture and two new
  tests (`test_parse_bridge_kind`, `test_directory_loader_skips_bridges_by_default`)
  verifying that the default kind="bldg" path ignores bridges, that
  kind="brid" extracts them as Building dataclasses with the right LoD,
  and that the directory loader honours `include_bridges`.  All 18
  existing plateau tests still pass.

## How to fetch bridges for a run

```bash
python3 experiments/fetch_plateau_subset.py \
  --run-dir /media/sasaki/aiueo/ai_coding_ws/datasets/PPC-Dataset-data/tokyo/run2 \
  --preset tokyo23 \
  --output-dir /tmp/plateau_segment_cache_tokyo_run2_bldg_brid \
  --mesh-radius 1 \
  --include-bridges
```

This writes both `_bldg_*.gml` and `_brid_*.gml` files, plus a manifest
with both file lists.  The same directory is then suitable for
`load_plateau(dir, include_bridges=True)`.

## Phase 2 — what is needed to actually move route MAE

To convert this Phase 1 loader work into a measurable route-MAE delta,
the full PPC simulator pipeline has to be re-run on top of the bridge
mesh and the §7.16 + isotonic75 + phaseguard nested stack has to be
retrained.  In rough order:

1. **Editable install rebuild**.  The sister tree at
   `/media/sasaki/aiueo/ai_coding_ws/gnss_gpu` is currently the pip
   editable install for `gnss_gpu` and is on
   `feature/ppc-realtime-turing-target`.  Rebuild against this branch
   (or use a separate venv for the PPC LoD2-bridge experiment) so that
   `from gnss_gpu.io.plateau import load_plateau` resolves to the
   `include_bridges`-aware version.
2. **Trajectory-aligned bridge fetch for all six runs**.  Use the new
   `--include-bridges` flag on `fetch_plateau_subset.py` for the three
   Tokyo runs.  Nagoya runs do not have meaningful elevated structures
   on the trajectory, but extracting them for completeness costs
   nothing (bridges in Nagoya23 will be ports, expressways, etc.).
3. **Per-run feature extraction with bldg+brid BVH**.  The same
   `exp_ppc_*_features.py` family (antenna / reflection / utd) can be
   re-run with `include_bridges=True` passed through to
   `load_plateau`.  Expect BVH-build time to grow ~6 % per run.
4. **Compare LoS counts at Tokyo run2 w23-w27 epochs first**.  Even
   before retraining, compare bldg-only vs bldg+brid `check_los`
   results at five representative epochs.  If the bridge mesh blocks
   no satellites that the building mesh did not already block, retrain
   is unlikely to help — abort early.
5. **Retrain §7.16 + isotonic + phaseguard**.  Use the same nested
   stack pipeline as the antenna and UTD probes, with
   `--max-run-mae-pp 4.5` as the selection guardrail.  Compare against
   the adopted model (run MAE 1.790 pp, weighted MAE 15.847 pp,
   correlation 0.559).
6. **If the retrain improves Tokyo run2 specifically**, the new
   adopted model would be `..._bldgbrid_alpha75_isotonic75_phaseguard_meta_run45`.
   Targeted success criterion: Tokyo run2 route error from 8.13 pp
   to <= 5 pp without regression on the other five runs.

## Phase 2 outcome — dense-sampled check (2026-04-30, v3)

> **Update (2026-04-30, after dense re-sampling):** The v2 numbers
> (1 flip in 9 hand-picked epochs) were unrepresentative.  Re-running
> at every-second sampling across the full hidden-high / false-high
> windows (script: `experiments/check_bldg_vs_brid_los_v3.py`,
> 214 epochs total) shows a much stronger and **window-bimodal**
> bridge contribution:
>
> | window | epochs | bldg-only NLoS / sat-slots | bldg+brid NLoS / sat-slots | flips | flip/epoch |
> | --- | ---: | ---: | ---: | ---: | ---: |
> | w7 (false-high)     | 41 | 564/1319 (43%) | 610/1319 (46%) | **46** | **1.122** |
> | w9 (false-high)     | 41 | 425/1239 (34%) | 425/1239 (34%) |   0 |   0.000 |
> | w23 (hidden-high)   | 81 | 1213/2475 (49%) | 1267/2475 (51%) | **54** | **0.667** |
> | w26-w27 (hidden)    | 51 |  159/1889 (8%) |  160/1889 (8%) |   1 |   0.020 |
> | **total**           | 214 | 2361/6922 (34%) | 2462/6922 (36%) | **101** | **0.472** |
>
> Verdict (v3): **w7 and w23 are clearly bridge-affected** (>0.65
> flips/epoch).  **w9 and w26-w27 are not** (<0.05 flips/epoch).
> The aggregate 0.472 flips/epoch sits at the upper edge of the
> "WEAK / plausibly worth retrain" band defined in the script's
> decision rule.  Phase 2 retrain on bldg+brid is now justifiable
> for the hidden-high w23 segment in particular, where 4.3 % of
> all elev>5° sat-slots flip from LoS to NLoS due to bridges.
> The earlier "abort" recommendation (v1, geoid-bug) and
> "inconclusive" recommendation (v2, 9-epoch) are both retracted.

## Phase 2 outcome — sampled check, 1 flip in 9 epochs (v2, superseded)

> **Important caveat (added 2026-04-30, after first Codex review):** The
> first run of this check (`check_bldg_vs_brid_los.py`) used PLATEAU
> heights as if they were ellipsoidal.  PLATEAU CityGML EPSG:6697 in
> fact carries **orthometric heights above Tokyo Bay mean sea level
> (TP)**, while `reference.csv` carries **ellipsoidal heights**.  At
> Tokyo Hamamatsucho the geoid undulation N ≈ +36.7 m, so the loaded
> mesh sat ~37 m below the rover and produced a spurious 0-flip
> "abort" signal.  All numbers below come from the corrected v2 run
> (`check_bldg_vs_brid_los_v2.py`), which monkey-patches
> `PlateauLoader._lla_to_ecef` to add a constant +36.7 m before
> ECEF conversion.  A proper fix should bake a geoid lookup into the
> loader itself; see "Follow-ups" below.

The Phase 2 step 4 early-abort check was run end-to-end against the
actual Tokyo run2 dataset.  After the geoid correction, it returns a
**weak / inconclusive** signal: bridges contribute 1 extra LoS→NLoS
flip out of 87 total bldg+brid blocks across 9 sampled epochs.

Scripts:
- `experiments/check_bldg_vs_brid_los.py` (uncorrected — kept for
  reproducibility; do **not** read its 0-flip result as evidence).
- `experiments/check_bldg_vs_brid_los_v2.py` (geoid-corrected; this
  is the version whose numbers are reported below).

Both are standalone (only depend on `gnss_gpu.io.rinex`,
`gnss_gpu.io.nav_rinex`, `gnss_gpu.ephemeris`, `gnss_gpu.bvh`, and
the new `gnss_gpu.io.plateau` bridge support).

Inputs (v2):
- 9 representative epochs covering both clusters: w7 (TOW 177210,
  177230), w9 (177270, 177290), w23 (177700, 177750), w26-w27
  (177800, 177810, 177830).
- bldg mesh: 2,200,133 triangles from 18 PLATEAU `_bldg_` GML files
  (mesh-radius 1 around the trajectory, fetched via
  `fetch_plateau_subset.py --include-bridges`).
- bldg+brid mesh: 2,308,241 triangles (+108,108 brid tris from 13
  `_brid_` GMLs; bridges are ~5 % of building volume in this area).
- Geoid correction: constant N = +36.7 m applied to PLATEAU alt
  before ECEF conversion (GSI Geoid 2011 nominal value for Tokyo
  Minato-ku; varies < 0.1 m/km across the trajectory).

Result table (per-epoch, geoid-corrected):

| TOW | n_sat (elev > 5°) | bldg-only NLoS | bldg+brid NLoS | extra flips |
| ---: | ---: | ---: | ---: | ---: |
| 177210 (w7)  | 32 | 13 | 14 | **1** (G03, elev 23.1°) |
| 177230 (w7)  | 34 | 18 | 18 | 0 |
| 177270 (w9)  | 28 |  8 |  8 | 0 |
| 177290 (w9)  | 35 |  8 |  8 | 0 |
| 177700 (w23) | 29 | 11 | 11 | 0 |
| 177750 (w23) | 31 | 19 | 19 | 0 |
| 177800 (w26) | 34 |  2 |  2 | 0 |
| 177810 (w27) | 38 |  4 |  4 | 0 |
| 177830 (w27) | 38 |  3 |  3 | 0 |
| **total** | 299 | **86** | **87** | **1** |

Sanity check: after geoid correction, the loaded mesh ECEF z range
spans the rover height (rover z = 3,697,019 m, mesh tri z range =
[3,695,900, 3,700,023]); the bldg-only check now produces 86 NLoS
results across 9 epochs, confirming the geometry is engaged
(versus 0 in the buggy v1 run).

Interpretation (weak signal, sampled check):

- The bridge mesh produces **1** extra NLoS in 9 epochs — only at
  TOW 177210 (w7), where the Tokyo Monorail / Yurikamome viaduct
  blocks GPS PRN 03 at 23° elevation.  None of the w23-w27 hidden-
  high cluster epochs flipped, despite that being the original
  motivation.
- This is *not enough samples* to claim either way.  9 epochs out of
  ~9,000 in the run is a 0.1 % sample, and the 8.13 pp run-MAE
  residual could plausibly come from a denser bridge interaction at
  other epochs not sampled here.
- It is also not enough to motivate the full 6-run feature
  extraction + §7.16 retrain (~1-2 h compute) by itself: 1 flip in
  87 blocks is a ~1 % geometric correction, well below the noise
  floor of the existing isotonic75 calibration.

Conclusion for Phase 2 (revised, weaker than original):

- **Sampled check is inconclusive.**  Earlier conclusion that
  "occluder geometry is not the cause" was over-stated for 9
  epochs and was further invalidated by the geoid bug.
- Recommended next step before any retrain: dense sampling
  (every 10th epoch across all hidden-high windows) on the
  geoid-corrected loader to get a real per-second flip rate.
- Phase 1 loader and `--include-bridges` plumbing remain useful
  infrastructure regardless; they cost nothing when not enabled.

Follow-ups (open after v3 dense-sampling):

1. ~~Add a real geoid model to `PlateauLoader._lla_to_ecef`.~~ Done.
   `PlateauLoader` and `load_plateau` accept a `geoid_correction`
   kwarg with **default `"egm96"`** (Codex review round 2 P1 #1: the
   silent no-correction path is what produced the original Phase 2
   abort bug, so the default was hardened from `None` to `"egm96"`).
   Other accepted values: a constant float (m), a callable
   `N(lat_deg, lon_deg)`, or `None` (explicit opt-out, fires a
   one-time `UserWarning`).  Missing pyproj → `ImportError` from the
   default path -- intentional "fail loud".  v2/v3 scripts use
   `geoid_correction="egm96"`; results match the previous `+36.7 m`
   monkey-patch within ~0.5 m (EGM96 vs GSI Geoid 2011 in Tokyo).
   A Tokyo regression test asserts that an EGM96-corrected mesh
   ground tri is within 5 m of the rover's ellipsoidal ground.
2. **Phase 2 retrain is now justified.**  See "Antenna feature
   pipeline propagation" below for the antenna-feature evidence
   that bridges produce a measurable signal at the input layer
   of the §7.16 nested stack.  Once the per-window features for
   all six runs are merged with the existing window CSV, run
   `predict.py --retrain` with `--max-run-mae-pp 4.5` as the
   selection guardrail.

## Antenna feature pipeline propagation (2026-04-30)

`exp_ppc_antenna_attenuation_features.py` grew `--include-bridges`
and `--geoid-correction` flags that thread directly into the new
`PlateauLoader(kinds=..., geoid_correction=...)` API.  Running it on
Tokyo run2 with `--include-bridges --geoid-correction egm96` and
comparing against the same script with `--geoid-correction egm96`
alone (so the only delta is the bridge mesh) gives:

| window | epochs | NLoS Δ | high-elev-NLoS Δ | eff_p50 Δ |
| --- | ---: | ---: | ---: | ---: |
| w7  (false-high)   |   41 | **+19 (+5.3%)**  | **+16 (+9.0%)**  | **−1.28 dB** |
| w9  (false-high)   |   41 | 0                | 0                |  0           |
| w23 (hidden-high)  |   81 | **+21 (+3.5%)**  | **+16 (+5.9%)**  | **−0.59 dB** |
| w26-27 (hidden)    |   51 | +1               | 0                |  0           |
| **whole run**      | 1592 | **+3227 (+62%)** | **+2353 (+110%)** | **−3.84 dB** |

(Stride 5; reference rover at the truth state per epoch; elev>0°
all-sat counts, no per-PRN filter.)

Critical observation: the bridge effect is **not localised to the
hidden-high windows**.  Across the whole run the bldg+brid mesh
adds +3,227 NLoS results (+62 % over the bldg-only baseline) and
shifts the median effective gain by **−3.84 dB**.  The earlier
v3 dense LoS check (1.122 / 0.667 flips/epoch in w7 / w23) was
underestimating the antenna pipeline's response because it used
elev>5° and counted only sats actually observed by the rover in
each epoch; the antenna feature extractor counts all geometric
sats and includes low-elevation ones, where bridges are most
effective occluders.  This is the input-layer signal that will
feed the §7.16 retrain.

### 6-run sweep (all PPC runs, 2026-04-30)

After the Tokyo run2 spot check, the same bldg vs bldg+brid antenna
extraction was run on all six PPC runs (stride 5, EGM96 correction
in both arms).  Result by run:

| run | epochs | ΔNLoS | Δhigh-elev-NLoS | Δeff_p50 |
| --- | ---: | ---: | ---: | ---: |
| tokyo/run1   | 2369 | +224 (+2.8 %)  | +181 (+3.5 %)    | −0.21 dB |
| tokyo/run2   | 1822 | **+3 227 (+57 %)**  | **+2 353 (+105 %)** | **−3.36 dB** |
| tokyo/run3   | 3059 | **+6 129 (+88 %)**  | **+4 713 (+165 %)** | **−3.61 dB** |
| nagoya/run1  | 1508 | 0              | 0                | 0        |
| nagoya/run2  | 1886 | 0              | 0                | 0        |
| nagoya/run3  | 1041 | 0              | 0                | 0        |
| **all 6**    | **11 685** | **+9 580 (+25 %)** | **+7 247 (+35 %)** | **−1.51 dB** |

Two clean signals:
- **Tokyo run2/run3 are heavily bridge-affected** (Tokyo Monorail,
  Yurikamome, Shuto Expressway viaducts on the trajectory).
- **Nagoya runs have zero bridge contribution** (no elevated
  structures crossed by the trajectory, as predicted by the
  Phase 1 README note).
- Tokyo run1 sits in between (+2.8 %).

This is the cleanest "do bridges matter at all in this dataset"
result we can produce without committing to a full retrain.

### Predictive power of bridge-aware features

The 14 antenna `_per_window` features were correlated against
`actual_fix_rate_pct` on the 197 evaluation windows in the §7.16
augmented window CSV.  Comparing bldg-only vs bldg+brid (both
EGM96), seven of fourteen features become measurably better
predictors when the bridge mesh is included:

| feature | r(bldg) | r(bldg+brid) | Δr |
| --- | ---: | ---: | ---: |
| eff_db_p10_mean              | +0.207 | +0.212 | +0.005 |
| eff_db_p50_mean              | +0.028 | +0.071 | **+0.043** |
| eff_db_p90_mean              | −0.013 | +0.044 | +0.031 |
| eff_db_mean_mean             | +0.132 | +0.158 | +0.026 |
| usable_count_mean            | +0.253 | +0.277 | +0.024 |
| marginal_count_mean          | +0.290 | +0.324 | +0.034 |
| nlos_at_high_elev_count_mean | −0.188 | −0.209 | +0.022 |
| **usable_count_min**         | +0.219 | +0.106 | **−0.113** |
| **eff_db_max_max**           | +0.144 | +0.081 | −0.063 |
| nlos_at_high_elev_count_max  | −0.139 | −0.070 | −0.069 |

`_mean` aggregations consistently improve (Δr ≈ +0.02..+0.04 each).
`_min` / `_max` aggregations degrade, because bridge effects spike
the per-window distribution and break aggregations the model had
learned over a smoother bldg-only signal.  Net: **weak positive**.

This is consistent with the LoS-check result: bridges are real
occluders that the input layer can see, but the absolute
correlation with FIX rate is modest (|r| < 0.33 in either case),
so the route-MAE delta from a full §7.16 retrain on bldg+brid
features is bounded above by a few hundredths of a percentage
point per run, not a percentage point.

Open follow-up (operator-driven): merge the 6 `*_BRID_egm96_*_per_window.csv`
files into the master window CSV (preserving the existing
`ant_*` schema) and run `predict.py --retrain --window-csv <new>
--epochs-csv <preprocessed phasejump epochs.csv>
--base-prefix <refinedgrid prefix>` with `--max-run-mae-pp 4.5`
as the guardrail.  Compare against the adopted model's run MAE
(1.79 pp) and route MAE.  The preprocessed epochs.csv is in
`/tmp/ppc_plots_worktree/experiments/results/` (177 MB), so this
is a single operator-side command, not a from-scratch rebuild.

To re-run the dense check:

```bash
python3 experiments/check_bldg_vs_brid_los_v3.py    # ~1 min, 214 epochs
```

Or the older 9-epoch v2 (geoid-corrected, faster):

```bash
python3 experiments/check_bldg_vs_brid_los_v2.py
```

## Why the residual was not closed (2026-04-30, post-merge follow-up)

After PR #38 was merged we ran three additional attacks on the
tokyo/run2 8.13 pp residual.  All three returned negative results
in the tested feature pool and model classes at this sample size;
together they point at a likely ceiling under the current
architecture.

### Attack 1: linear correction over the input pool

A Lasso (LassoCV) was fit to the 197-window residual
(`actual - corrected_pred`) using a focused 96-feature subset
(`sim_adop_*_min`, `rinex_phase_streak_*_min/p25`, the
`validationhold_high_pred_reject_flag`).  In-sample R² on residual
was 0.436 with 31 non-zero coefficients.  Applied as an additive
correction:

`route Δ` is the signed change in `|actual - prediction|`: positive
means the adjusted prediction is closer to actual, negative means
it moved further away.

| route       | actual | corrected | adjusted | route Δ |
| ----------- | ---:   | ---:      | ---:     | ---:    |
| nagoya/run1 | 11.51  | 11.74     | 13.02    | -1.28   |
| nagoya/run2 | 16.17  | 17.61     | 20.67    | -3.06   |
| nagoya/run3 |  7.87  |  8.03     |  8.31    | -0.28   |
| tokyo/run1  | 10.92  | 10.42     |  9.67    | -0.75   |
| tokyo/run2  | 29.02  | 20.89     | 21.34    | **+0.45** |
| tokyo/run3  | 24.01  | 23.73     | 26.04    | -1.75   |

Tokyo run2 improves by 0.45 pp; the other five routes get worse by
0.28..3.06 pp.  The "lift" direction the in-sample fit found is
not transferable across routes -- and even within sample it only
shaves 5 % of the tokyo/run2 residual.

### Attack 2: hidden-high binary classifier (LORO)

Defined hidden-high as `actual ≥ 75 % AND corrected_pred ≤ 50 %`,
giving 15 positives across the 6 route folds (one fold has zero
positives), with per-fold counts 3 / 1 / 0 / 2 / 5 / 4 out of 197
windows total.  Trained a `GradientBoostingClassifier(n_estimators=200,
max_depth=3, learning_rate=0.05)` on the same 5,884-feature pool
with leave-one-route-out:

| held-out route | AUC   | n_pos / n_total |
| ---            | ---:  | ---:            |
| nagoya/run1    | 0.819 | 3 / 26          |
| nagoya/run2    | **0.258** | 1 / 32      |
| nagoya/run3    | --    | 0 / 18          |
| tokyo/run1     | 0.662 | 2 / 39          |
| tokyo/run2     | 0.615 | 5 / 31          |
| tokyo/run3     | **0.330** | 4 / 51      |
| **pooled OOF** | **0.595** | 15 / 197    |

(`pooled OOF` AUC is computed on the concatenated out-of-fold
predictions across all 6 folds, not as a macro/weighted mean of the
per-fold AUCs.)

Two of the held-out folds (nagoya/run2 with `n_pos=1`, tokyo/run3
with `n_pos=4`) score below chance (AUC < 0.5), and top-15 flagged
precision is 13 % (vs 7.6 % positive-rate floor).  At this sample
size the existing feature pool does not yield a transferable
hidden-high classifier across routes -- whether this reflects an
intrinsic ceiling or just sparse-positive variance is not separable
from one LORO sweep.

### Attack 3: per-window uncertainty abstention

Trained a `GradientBoostingRegressor` to predict `|residual|` with
the same LORO setup, then abstained on the top-X % most uncertain
windows.  The aggregate route MAE went **up** at every abstention
rate from 5 % to 30 %:

| drop top | windows kept | mean abs route err |
| ---:     | ---:         | ---:               |
| 0 %      | 197          | **1.79**           |
| 5 %      | 188          | 2.86               |
| 10 %     | 178          | 3.52               |
| 15 %     | 168          | 3.40               |
| 20 %     | 158          | 2.90               |
| 25 %     | 148          | 2.48               |
| 30 %     | 138          | 2.98               |

Inspecting the tokyo/run2 windows ranked by predicted
uncertainty: the model puts the false-high windows (w7, w9 --
residuals -48, -77) at the top, but does not consistently rank the
hidden-high cluster (w23, w25, w26) above several lower-residual
windows -- so dropping the most-uncertain windows removes some
hidden-high cases together with low-residual ones, and the route
MAE rises rather than falls.  The input layer carries phase-jump
warnings that flag false-high cases, but the tested feature pool
does not separate hidden-high cases from genuinely bad observation
conditions.

### Joint conclusion

All three attacks fail the same way: hidden-high windows are
those where demo5 reaches/maintains FIX while the runtime feature
layer (simulator outputs + RINEX summary stats + antenna eff_db,
with or without bridge-aware NLoS) describes the conditions as
bad.  None of the tested attacks found a transferable signal at
the input layer that separates these cases from the many windows
that are also "feature-bad" and where demo5 does not FIX.  Under
the current architecture (no `rtk_*` / `solver_demo5_*` columns as
runtime features) and at this sample size, we treat the 8.13 pp
tokyo/run2 residual as an **accepted ceiling**, conditional on the
out-of-scope paths below.

Three out-of-scope paths to lift this ceiling:

1. **Solver-state lightweight wrapper**: expose a curated subset of
   demo5 ambiguity-fix-state indicators as runtime features.
   Medium-size PR, but
   it changes the model's input contract.
2. **Additional PPC data collection**: more runs in viaduct-heavy
   trajectories so the hidden-high cluster stops being 5/197 and
   becomes statistically learnable.  Operator-side.
3. **Architectural pivot**: model demo5 success directly (e.g. as
   a sequence-level latent variable) rather than predicting the
   FIX rate from per-window stats.  Multi-week research.

The product deliverable at run-MAE 1.79 pp (with tokyo/run2
flagged `route_action="review_required"`) is the best result we
have obtained on this dataset under the tested feature pool and
model classes.

## Implementation notes

- The bridge fixture in `tests/test_plateau.py` deliberately uses the
  PLATEAU `<filename>_brid_<crs>_op.gml` naming convention so the
  directory-loader filename heuristic exercises the same code path as
  real PLATEAU data.
- `parse_citygml(kind="brid")` returns `Building` dataclasses rather
  than introducing a new `Bridge` type.  Bridges and buildings have
  identical CityGML 2.0 geometry serialisation (LoD-tagged solid /
  multi-surface with `gml:posList` rings); a new dataclass would have
  the same fields and only complicate the downstream triangulation
  code.
- `udx/tran` is intentionally not parsed.  Tran is LOD1 (planar
  centerlines) and contributes no occluder volume above the road
  surface.  If future PLATEAU releases ship LOD2 tran with viaduct
  superstructure as separate from `brid`, the same `_FEATURE_KINDS`
  table can grow a `("tran", "Road")` entry trivially.
- This loader change is a strict superset of the previous behaviour:
  every existing call site continues to load buildings only, so the
  adopted model artifacts and CSVs do not move until the deliberate
  retrain in step 5 above.
