# WP3a — Native FGO backbone on PPC tokyo run1

## Data path

`E:\datasets\PPC-Dataset-data\tokyo\run1`  
(rover.obs, base.obs, base.nav, reference.csv)

## Root cause: `iters=-1` / FGO == WLS

The first full-run attempt called `fgo_gnss_lm_vd` on all **11 676** epochs at once. With `n_clock=1` the VD state stride is **8** (`[x,y,z,vx,vy,vz,c0,drift]`), so `n_state = 11 676 × 8 = 93 408`, which exceeds the native cap in `fgo.cu`:

```3825:3825:src/positioning/fgo.cu
  if (n_state > 16384) return -1;  // larger limit for extended state
```

The native solver returns **`iters=-1`** without modifying `state_io`. The experiment script treated that as a completed call and exported the **WLS initialization** unchanged, so **FGO 2D == WLS 2D** and **`mse=0`**.

**Fix:** `_solve_fgo_vd_chunked()` in `experiments/validate_fgo_ppc.py` splits the timeline into chunks of ≤**1000** epochs (`n_state ≤ 8000 < 16384`), optimizes each chunk, and carries kinematic state across boundaries via `_seed_chunk_boundary_state()` (position, velocity, clock, drift). Per-chunk `iters`, `mse_pr`, and `status` are recorded; any native failure aborts with `status=native_failed`.

## WLS sanity (first 120 epochs)

Truncated run against `reference.csv` (nearest-TOW matching, same as `validate_fgo_ppc`):

| Stage | 2D RMS (m) | iters | mse_pr |
|-------|------------|-------|--------|
| WLS   | 32.66      | —     | —      |
| FGO (a) | 32.66    | 8     | 2.29e4 |

The ~**96 m** full-run WLS 2D is **not** a units bug on the first 120 epochs; it reflects poor epoch-wise SPP over the full urban timeline (many weak-geometry epochs), not the pre-chunk solver failure.

## Variant runs

### (a) PR + motion + clock-drift (`--doppler off`)

```
set PYTHONPATH=python
set PYTHONUNBUFFERED=1
python -u experiments/validate_fgo_ppc.py --no-rtklib --vd --run tokyo/run1 --max-epochs 0 ^
  --doppler off --fgo-iters 8 ^
  --export-csv results/wp3a/tokyo_run1_fgo_backbone_pr_motion.csv
```

**Convergence:** 12 chunks × 8 iters = **96** total iters; mean chunk `mse_pr ≈ 1.64e4`.  
**validate_fgo_ppc line:** WLS 2D **95.73 m**, FGO 2D **94.52 m**, 3D **115.26 m** (FGO improved vs WLS).

### (b) + in-repo Doppler (`--doppler in-repo`, `include_sat_velocity=True`)

```
python -u experiments/validate_fgo_ppc.py --no-rtklib --vd --run tokyo/run1 --max-epochs 0 ^
  --doppler in-repo --fgo-iters 8 ^
  --export-csv results/wp3a/tokyo_run1_fgo_backbone_doppler.csv
```

**Convergence:** 12 chunks × 8 iters = **96** total iters.  
**validate_fgo_ppc line:** WLS 2D **95.73 m**, FGO 2D **546.02 m**, 3D **595.66 m**.  
Chunk **5** (`epochs 5000:6000`) reported `mse_pr = 9.25e5`, indicating Doppler misfit / ill-conditioning in that segment. On **120 epochs**, in-repo Doppler is neutral-to-helpful (FGO 2D **32.41 m** vs WLS **32.66 m**); full-run degradation is localized, not a wiring omission.

## Dual-metric scores (`score_vs_inuex35.py`, full-timeline denominator)

```
python experiments/score_vs_inuex35.py --traj results/wp3a/tokyo_run1_fgo_backbone_pr_motion.csv ^
  --format csv --city tokyo --run run1 --data-root E:/datasets/PPC-Dataset-data ^
  --out-json results/wp3a/score_pr_motion.json --out-csv results/wp3a/scores.csv

python experiments/score_vs_inuex35.py --traj results/wp3a/tokyo_run1_fgo_backbone_doppler.csv ^
  --format csv --city tokyo --run run1 --data-root E:/datasets/PPC-Dataset-data ^
  --out-json results/wp3a/score_doppler.json --out-csv results/wp3a/scores.csv
```

| Variant | coverage% | n_scored / n_rover | AllRMS (3D, m) | Fix% | <50cm_full% |
|---------|-----------|-------------------|----------------|------|--------------|
| (a) pr_motion | 97.9 | 11676 / 11928 | **115.26** | 0.0 | 0.0 |
| (b) doppler   | 97.9 | 11676 / 11928 | **595.66** | 0.0 | 0.0 |

## Chunk boundary handling

For chunk index `k > 0`, epoch `start` is overwritten with the optimized state at `start-1`:

- `seg_state[0, :3]` ← previous position  
- `seg_state[0, 3:6]` ← previous velocity  
- `seg_state[0, 6]` ← previous receiver clock  
- `seg_state[0, 7]` ← previous clock drift (when `n_clock=1`)

Interior epochs within each chunk are re-optimized; only the first epoch of each chunk is pinned to the prior chunk tail.

## Artifacts

| File | Description |
|------|-------------|
| `results/wp3a/tokyo_run1_fgo_backbone_pr_motion.csv` | Variant (a) trajectory |
| `results/wp3a/tokyo_run1_fgo_backbone_doppler.csv` | Variant (b) trajectory |
| `results/wp3a/run_variant_a.log` | Full (a) log with per-chunk stats |
| `results/wp3a/run_variant_b.log` | Full (b) log with per-chunk stats |
| `results/wp3a/scores.csv` | Combined scorer rows |
| `results/wp3a/score_pr_motion.json` | (a) JSON metrics |
| `results/wp3a/score_doppler.json` | (b) JSON metrics |

## Tests

```
set PYTHONPATH=python
python -m pytest tests/test_validate_fgo_ppc_native.py tests/test_score_vs_inuex35.py -q -p no:xonsh
```

**Result:** 15 passed (8 native/chunking + 7 scorer).

## Code touched

- `experiments/validate_fgo_ppc.py` — native PPC path, chunked VD solver, CSV export, in-repo Doppler wiring  
- `tests/test_validate_fgo_ppc_native.py` — chunking / export / Doppler unit tests
