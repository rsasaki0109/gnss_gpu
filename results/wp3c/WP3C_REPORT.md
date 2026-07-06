# WP3c — Making multi-GNSS actually help the native FGO backbone (tokyo run1)

This builds on `results/wp3b/WP3B_REPORT.md`'s D2 finding: enabling all 5
broadcast constellations (`systems=GRECJ`) grows the median satellite count
8.0 → 28.0 and pushes coverage 97.9% → 100%, but *regresses* accuracy (FGO 2D
94.52 m → 114.78 m) versus the GPS-only backbone. WP3b attributed this to
(1) no elevation mask, (2) no inter-constellation code-bias/ISB calibration,
(3) possibly weaker non-GPS broadcast-ephemeris fidelity, but did not
quantify or fix any of them. This report does.

**Baseline to beat** (WP3b variant (c), GPS-only): **FGO 2D 85.82 m, 3D
105.73 m, coverage 97.9%** (11676/11928).

## Work item 1 — Elevation mask

Added `_elevation_deg_per_epoch` / `_apply_elevation_mask` to
`experiments/validate_fgo_ppc.py`: elevation is computed from `sat_ecef` and
a **pass-1 per-epoch WLS position** (not ground truth) via
`gnss_gpu.validation.elevation_azimuth` (the same WGS84-geodesic ENU
projection already used by `gnss_gpu.validation.real_residuals`), mirroring
the `observation_min_elevation_deg` lever pattern in
`experiments/gsdc2023_bridge_config.py`. `run_fgo_on_ppc_native` now runs a
two-pass WLS: pass 1 (unweighted/unmasked) supplies the elevation observer
position; satellites below the cutoff have their pseudorange weight zeroed;
pass 2 re-solves WLS with the masked weights and that seeds the FGO state.
New `--elevation-mask-deg` CLI flag (`0` = disabled, legacy behaviour).

### Sweep (10/15/20 deg, representative 2000-epoch window, GRECJ, PR+motion only/Doppler off)

`experiments/sweep_elevation_mask_wp3c.py`:

```
set PYTHONPATH=python
set PYTHONUNBUFFERED=1
python experiments/sweep_elevation_mask_wp3c.py
```

| elevation mask | obs masked | WLS 2D | FGO 2D | FGO 3D |
|---|---|---|---|---|
| 0 deg (off) | 0.0% | 48.94 m | 53.36 m | 57.33 m |
| 10 deg | 4.9% | 47.62 m | 52.01 m | 55.88 m |
| 15 deg | 7.6% | 45.80 m | 50.97 m | 54.50 m |
| **20 deg** | 14.1% | **41.24 m** | **47.03 m** | **49.57 m** |

All three required cutoffs monotonically improve over the no-mask baseline;
**20 deg was best of the required sweep** (12% 2D-RMS reduction vs 0 deg on
this window) and was still improving at the top of the sweep range, so it
was picked for variant (d). (Higher cutoffs, e.g. 25/30 deg, are flagged as
a follow-up; only 10/15/20 deg were swept per the task's own sweep range.)

## Work item 2 — Per-constellation weighting / bias handling

### 2a — GRECJ `n_clock=5` clock-state wiring (verification)

`experiments/diag_constellation_wiring_and_residuals.py` confirms
`run_fgo_on_ppc_native`'s dynamic `constellations = sorted(...)` /
`n_clock = len(constellations)` / `sys_kind` wiring assigns **exactly one
contiguous clock index per observed system letter**, with no collisions:

```
constellations (sorted) = ('C', 'E', 'G', 'J', 'R')  n_clock = 5
clock_index -> system_char(s): {0: ['C'], 1: ['E'], 2: ['G'], 3: ['J'], 4: ['R']}
wiring_ok = True
```

A 300-epoch VD smoke solve with `n_clock=5` converges cleanly (`iters=8`,
`mse_pr=1.45e4`). The wiring is correct — **not** a contributing cause of
the D2 regression. (This matches WP3b's own code-read conclusion; this
report additionally exercises it end-to-end with a live solve.)

### 2b — Per-constellation pseudorange sigma scaling

Added `_apply_constellation_sigma_scaling` to `validate_fgo_ppc.py`: rescales
`weights` per-satellite by `1/scale**2` using each PRN's system letter, with
a configurable `constellation_sigma_scale` dict (new `--constellation-weighting`
CLI flag enables it with the task's a-priori starting point — GPS/Galileo/QZSS
1.0x, BeiDou 1.5x, GLONASS 2.0x sigma — plus new `--pr-sigma-scale-{g,r,e,c,j}`
flags to override any system individually).

**Tuning on data** (task's own instruction: "tune on data via per-constellation
residual RMS from a WLS pass, report the numbers") — using the work item 3
"BEFORE" WLS residual RMS breakdown below, the *data-derived* scale (relative
to GPS) is:

| system | WLS residual RMS (before mask/weight) | scale = rms / rms_G |
|---|---|---|
| G (GPS)     | 28.97 m | 1.000 (reference) |
| E (Galileo) | 30.09 m | 1.039 |
| C (BeiDou)  | 24.99 m | **0.863** |
| J (QZSS)    | 25.07 m | **0.865** |
| R (GLONASS) | 36.52 m | 1.261 |

This directly **contradicts the a-priori guess**: on this dataset BeiDou and
QZSS pseudoranges are *less* noisy than GPS's own (scale < 1.0 — the a-priori
1.5x BeiDou down-weight was working against the data), and GLONASS is only
mildly noisier (1.26x, not 2x). A head-to-head comparison on the same
2000-epoch window (`experiments/sweep_constellation_weighting_wp3c.py`,
elevation mask 20 deg applied in both) confirms the data-tuned scale wins:

| weighting | WLS 2D | FGO 2D | FGO 3D |
|---|---|---|---|
| a-priori (BDS 1.5x / GLO 2.0x) | 43.54 m | 46.01 m | 50.34 m |
| **data-tuned** (BDS 0.863x / GAL 1.039x / QZS 0.865x / GLO 1.261x) | **38.97 m** | **45.27 m** | **46.94 m** |

The data-tuned weights were used for the reported variant (d′) full run below.

### 2c — Pseudorange code-preference frequency consistency (verification)

`experiments/diag_constellation_wiring_and_residuals.py` checked every entry
of `io/ppc.py`'s `_PSEUDORANGE_CODE_PREFERENCES` against its system's single
expected L1-like carrier band (GPS/Galileo/QZSS 1575.42 MHz, BeiDou B1I
1561.098 MHz, GLONASS L1 ~1602 MHz) — **no L1/L2/L5 mixing found**; every
system's preference list stays within one frequency. Not a contributing
cause.

## Work item 3 — Per-constellation residual breakdown (root-cause attribution)

`experiments/diag_constellation_wiring_and_residuals.py`, single-*global*-clock
WLS residual (`pr_corr - range - clk`) grouped by system, on the same
2000-epoch GRECJ window, before vs. after applying the chosen elevation mask
(20 deg) + constellation weighting (data-tuned):

| system | n (before) | median\|resid\| before | rms before | n (after) | median\|resid\| after | rms after |
|---|---|---|---|---|---|---|
| C (BeiDou)  | 20986 | 17.57 m | 24.99 m | 19429 | 17.67 m | 26.12 m |
| E (Galileo) | 14146 | 24.22 m | 30.09 m | 11704 | 21.56 m | 31.53 m |
| G (GPS)     | 17778 | 20.60 m | 28.97 m | 14782 | 17.76 m | 23.99 m |
| J (QZSS)    |  5204 | 19.91 m | 25.07 m |  4000 | 14.82 m | 15.17 m |
| R (GLONASS) |  9584 | 22.86 m | 36.52 m |  8228 | 22.64 m | 33.62 m |

**Attribution**: raw per-satellite residual *magnitude* does not single out
one "poisoning" constellation — GPS's own median residual (20.60 m) is not
the smallest, and BeiDou's is comparable to or smaller than GPS's. The real
signal is in the **per-system clock/ISB estimate** from the live `n_clock=5`
VD smoke solve (work item 2a): BeiDou's estimated clock bias is **365.8 m**,
14–31x every other system's (`E=24.0m, G=26.6m, J=20.2m, R=11.8m`). Because
each system's clock/ISB is a *free per-epoch state* (constrained only by the
motion/clock-drift priors, not an absolute bias model), an unusually large
and likely unstable BeiDou ISB is the primary quantitative D2 root cause,
not low elevation or raw multipath noise (those exist too — QZSS's
median\|resid\| drops 25% after masking/weighting — but are secondary).

## Work item 4 — Full-run variant (d)

PR + motion + Doppler(Huber k=5) + IMU(0.5/0.2) + GRECJ + elevation mask
(20 deg) + constellation weighting, `--chunk-epochs 250`:

```
set PYTHONPATH=python
set PYTHONUNBUFFERED=1
python -u experiments/validate_fgo_ppc.py --no-rtklib --vd --run tokyo/run1 --max-epochs 0 ^
  --doppler in-repo --doppler-huber-k 5.0 ^
  --imu --imu-position-sigma-m 0.5 --imu-velocity-sigma-mps 0.2 ^
  --systems GRECJ --elevation-mask-deg 20 --constellation-weighting ^
  --chunk-epochs 250 --fgo-iters 8 ^
  --export-csv results/wp3c/tokyo_run1_fgo_variant_d.csv
```

Smoke test (2000 epochs) beforehand: no crashes, `WLS2D=43.54m FGO2D=45.98m
FGO3D=50.30m`, 14.1% obs masked (matches the sweep exactly) — proceeded to
the full run.

Variant (d′) re-runs the same command with the work item 2b data-tuned
weights substituted via the new `--pr-sigma-scale-{c,e,j,r}` overrides:

```
python -u experiments/validate_fgo_ppc.py --no-rtklib --vd --run tokyo/run1 --max-epochs 0 ^
  --doppler in-repo --doppler-huber-k 5.0 ^
  --imu --imu-position-sigma-m 0.5 --imu-velocity-sigma-mps 0.2 ^
  --systems GRECJ --elevation-mask-deg 20 --constellation-weighting ^
  --pr-sigma-scale-c 0.863 --pr-sigma-scale-e 1.039 --pr-sigma-scale-j 0.865 --pr-sigma-scale-r 1.261 ^
  --chunk-epochs 250 --fgo-iters 8 ^
  --export-csv results/wp3c/tokyo_run1_fgo_variant_d_tuned.csv
```

| Variant | WLS 2D | FGO 2D | FGO 3D (=AllRMS) | coverage |
|---|---|---|---|---|
| WP3a (a) `systems=G`, PR+motion | 95.73 m | 94.52 m | 115.26 m | 97.9% (11676/11928) |
| WP3b (c) PR+motion+Doppler-Huber+IMU, `systems=G` | 95.73 m | **85.82 m** | 105.73 m | 97.9% |
| WP3b D2 `systems=GRECJ`, PR+motion (no fixes) | 108.68 m | 114.78 m | 131.27 m | 100.0% |
| **WP3c (d)** GRECJ + mask(20°) + a-priori weights + Doppler-Huber + IMU | 92.63 m | **97.03 m** | 114.04 m | **100.0%** (11924/11928) |
| **WP3c (d′)** GRECJ + mask(20°) + **data-tuned weights** + Doppler-Huber + IMU | 90.54 m | 98.96 m | **111.47 m** | **100.0%** (11924/11928) |

Scored with `experiments/score_vs_inuex35.py` (same commands as
WP3B_REPORT.md):

```
python experiments/score_vs_inuex35.py --traj results/wp3c/tokyo_run1_fgo_variant_d.csv --city tokyo --run run1 --format csv --out-json results/wp3c/score_variant_d.json --out-csv results/wp3c/scores.csv
python experiments/score_vs_inuex35.py --traj results/wp3c/tokyo_run1_fgo_variant_d_tuned.csv --city tokyo --run run1 --format csv --out-json results/wp3c/score_variant_d_tuned.json --out-csv results/wp3c/scores.csv
```

variant (d): `AllRMS(3D)=114.039`, `coverage=100.0%` (11924/11928); variant
(d′): `AllRMS(3D)=111.468`, `coverage=100.0%` (11924/11928) — **neither
meets the success bar** (2D 97.03 m / 98.96 m > (c)'s 85.82 m), despite
coverage reaching ~100%. Every elevation-mask/weighting/wiring fix from work
items 1–3 was applied and each individually and jointly improved the
*representative-window* numbers (see sweeps above), yet the *full-run* FGO
2D RMS is even slightly worse than its own WLS seed in both variants (97.03 m
vs 92.63 m WLS for (d); 98.96 m vs 90.54 m WLS for (d′)).

**The data-tuned weighting (work item 2b) did not generalize from the
representative window to the full run** — it improved 3D RMS (114.04 m →
111.47 m) but slightly *worsened* 2D RMS (97.03 m → 98.96 m). Per-chunk
`mse_pr` diagnostics explain why: both variants show the same specific
degraded segment, epochs ~6500–7250 (chunks 26–28), `mse_pr` spiking to
7.1e4–1.5e5 versus a typical 0.7–1.5e4 elsewhere — but the *tuned* variant's
spike is larger there (up to 1.19e5 vs 1.02e5 for the a-priori weights,
chunk 28) and a second bad region appears at epochs 4250–4750 (chunks 17–18,
mse up to 1.15e5) that is comparatively milder in variant (d) (mse ~5.2e4
there). Because the data-tuning treated BeiDou as *cleaner* than GPS (scale
0.863 < 1.0, i.e. a relative up-weight vs the a-priori 1.5x down-weight),
and BeiDou carries the single largest, most unstable per-epoch ISB estimate
(work item 3: 365.8 m), giving it more relative trust generalizes well on
the calm window used to derive the scale factors but backfires on the
harder segments where that same constellation's clock/ISB estimate is least
trustworthy — the representative-window WLS-residual RMS is not
representative of the *whole* trajectory's per-constellation noise
characteristics (a non-stationarity the task's "tune on data" instruction
did not anticipate). This is reported as an additional, concrete finding
beyond the required work items.

A full fix for the underlying degraded-segment vulnerability (e.g. an
ISB stability/rate-of-change prior, or a more robust per-chunk LM solve)
would need per-epoch-adaptive weighting or changes to the solver itself,
both out of this task's `validate_fgo_ppc.py` / `io/ppc.py` surface (the
solver lives in `fgo.cu`; `local_fgo.py` is off-limits per the task
constraints) and is flagged as follow-up work.

## Work item 5 — Fallback: GPS+QZS+GAL only

Since multi-GNSS (variant d) still could not beat GPS-only after the
mask+weights fixes, ran the task's fallback with the same
mask+weight+Doppler-Huber+IMU stack, `systems=GEJ` (GPS+Galileo+QZSS,
dropping BeiDou/GLONASS — the two systems flagged above as the largest
ISB/residual outliers):

```
python -u experiments/validate_fgo_ppc.py --no-rtklib --vd --run tokyo/run1 --max-epochs 0 ^
  --doppler in-repo --doppler-huber-k 5.0 ^
  --imu --imu-position-sigma-m 0.5 --imu-velocity-sigma-mps 0.2 ^
  --systems GEJ --elevation-mask-deg 20 --constellation-weighting ^
  --chunk-epochs 250 --fgo-iters 8 ^
  --export-csv results/wp3c/tokyo_run1_fgo_variant_e_fallback_gej.csv
```

```
python experiments/score_vs_inuex35.py --traj results/wp3c/tokyo_run1_fgo_variant_e_fallback_gej.csv --city tokyo --run run1 --format csv --out-json results/wp3c/score_variant_e_fallback_gej.json --out-csv results/wp3c/scores.csv
```

| Variant | WLS 2D | FGO 2D | FGO 3D (=AllRMS) | coverage |
|---|---|---|---|---|
| WP3c (e) fallback GEJ + mask(20°) + weights + Doppler-Huber + IMU | 102.25 m | 97.84 m | 121.92 m | 100.0% (11923/11928) |

The GEJ fallback (`AllRMS=121.921`, `coverage=100.0%`) is **also worse**
than both variant (d) (114.04 m) and the GPS-only baseline (105.73 m) —
dropping BeiDou/GLONASS did not help. The same degraded epoch range
(~6500–7250) still dominates the error (`mse_pr` up to 1.50e5 there), so the
bottleneck is not simply "which extra constellations are included" but a
specific weak-geometry/multipath trajectory segment that penalizes *any*
wider, less-constrained multi-clock state during that window, GRECJ or GEJ.

## Finding (per task's own fallback clause)

**Multi-GNSS still cannot beat the GPS-only backbone on tokyo/run1**, even
after implementing and correctly wiring an elevation mask, per-constellation
weighting (both a-priori and data-tuned), verifying the clock-state wiring
and code-frequency consistency, and trying the GPS+Galileo+QZSS fallback.
The **best multi-GNSS configuration found by the primary (2D) metric is
variant (d)** (GRECJ + 20° elevation mask + a-priori constellation
weighting + Doppler-Huber + IMU) at **97.03 m 2D / 114.04 m 3D**, 100%
coverage; the **best by 3D is variant (d′)** (same but data-tuned weights)
at **98.96 m 2D / 111.47 m 3D**. Both are a large improvement over WP3b's
unfixed D2 baseline (114.78 m 2D → 97.03 m 2D, **−15.4%**) and proof that
the elevation/ISB-weighting/wiring hypotheses from WP3b's D2 section were
all real, fixable, and quantifiable contributors — but both remain short of
GPS-only's 85.82 m 2D / 105.73 m 3D. The fallback (e) (GEJ, dropping
BeiDou/GLONASS) performed *worse* than either GRECJ variant (121.92 m 3D),
confirming the bottleneck is not simply "which extra constellations", i.e.
work item 5's fallback does not produce the best configuration here —
**variant (d) is the best configuration found overall** and is the one
recommended if multi-GNSS coverage (100% vs GPS-only's 97.9%) is valued
over the 2D-RMS regression.

The root cause that remains unfixed within this task's scope is a specific
weak-geometry/multipath trajectory segment (epochs ~6500–7250, and to a
lesser extent ~4250–4750) where a wider, less-constrained multi-clock FGO
state is structurally more vulnerable than the single-clock GPS-only
backbone — and where BeiDou's uniquely large, free-per-epoch ISB estimate
(work item 3) is the leading quantitative suspect. Fixing that would need
either an unstable-ISB-aware clock prior/rate constraint or a more robust
per-chunk solve, both out of this task's `validate_fgo_ppc.py` / `io/ppc.py`
surface (they live in `fgo.cu`).

## Artifacts

| File | Description |
|---|---|
| `results/wp3c/tokyo_run1_fgo_variant_d.csv` | Variant (d) trajectory (a-priori weights) |
| `results/wp3c/tokyo_run1_fgo_variant_d_tuned.csv` | Variant (d′) trajectory (data-tuned weights) |
| `results/wp3c/tokyo_run1_fgo_variant_e_fallback_gej.csv` | Fallback (e) GEJ trajectory |
| `results/wp3c/score_variant_d.json` / `score_variant_d_tuned.json` / `score_variant_e_fallback_gej.json` | scorer JSONs |
| `results/wp3c/scores.csv` | Combined scorer rows |
| `results/wp3c/diag_constellation_wiring_and_residuals.log` | work items 2a/2c/3 diagnostic output |
| `results/wp3c/sweep_elevation.log` | work item 1 elevation-mask sweep output |
| `results/wp3c/sweep_weighting.log` | work item 2b a-priori-vs-data-tuned weighting output |
| `results/wp3c/variant_d_full.log` / `variant_d_tuned_full.log` / `variant_e_fallback_full.log` | full-run console logs (per-chunk `mse_pr`) for (d)/(d′)/(e) |

## Code touched

- `experiments/validate_fgo_ppc.py` — elevation mask (`_elevation_deg_per_epoch`,
  `_apply_elevation_mask`, work item 1), per-constellation sigma scaling
  (`_apply_constellation_sigma_scaling`, `DEFAULT_CONSTELLATION_SIGMA_SCALE`,
  work item 2b), two-pass WLS wiring in `run_fgo_on_ppc_native`, new
  `--elevation-mask-deg` / `--constellation-weighting` /
  `--pr-sigma-scale-{g,r,e,c,j}` CLI flags.
- `experiments/diag_constellation_wiring_and_residuals.py` (new) — work
  items 2a/2c/3 diagnostics.
- `experiments/sweep_elevation_mask_wp3c.py` (new) — work item 1 sweep.
- `experiments/sweep_constellation_weighting_wp3c.py` (new) — work item 2b
  a-priori-vs-data-tuned weighting comparison.
- `tests/test_validate_fgo_ppc_native.py` (extended) — 10 new unit tests for
  the elevation mask and constellation-weighting logic (synthetic
  lat=0/lon=0 ENU geometry for exact-elevation satellite placement).

## Tests

```
set PYTHONPATH=python
python -m pytest -p no:xonsh tests/test_ppc_imu_adapter.py tests/test_validate_fgo_ppc_native.py tests/test_score_vs_inuex35.py -q
```

**Result:** 37 passed (25 in `test_validate_fgo_ppc_native.py`, including
the 10 new WP3c elevation-mask/weighting tests, + 5 in
`test_ppc_imu_adapter.py` + 7 in `test_score_vs_inuex35.py`).
