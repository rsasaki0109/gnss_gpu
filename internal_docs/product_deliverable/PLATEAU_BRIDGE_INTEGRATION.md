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
   `PlateauLoader` and `load_plateau` now accept a `geoid_correction`
   kwarg accepting `None` / `"egm96"` (pyproj-driven) / a constant
   float / a callable.  Without it, a one-time UserWarning fires.
   The v2/v3 scripts use `geoid_correction="egm96"`; results match
   the previous `+36.7 m` monkey-patch within ~0.5 m (EGM96 vs
   GSI Geoid 2011 difference in Tokyo).  A Tokyo regression test
   asserts that an EGM96-corrected mesh ground tri is within 5 m
   of the rover's ellipsoidal ground.
2. **Phase 2 retrain is now justified for w7 and w23.**  Run the
   full feature extraction with `kinds=("bldg", "brid")` and
   `geoid_correction="egm96"`, then re-fit §7.16 + isotonic75 +
   phaseguard with the `--max-run-mae-pp 4.5` selection guardrail.
   The expected gain is bounded above by the 4.3 % sat-slot flip
   rate in w23 plus 3.5 % in w7, applied through the existing
   FIX classifier.

To re-run the dense check:

```bash
python3 experiments/check_bldg_vs_brid_los_v3.py    # ~1 min, 214 epochs
```

Or the older 9-epoch v2 (geoid-corrected, faster):

```bash
python3 experiments/check_bldg_vs_brid_los_v2.py
```

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
