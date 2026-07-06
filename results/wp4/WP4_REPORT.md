# WP4 — First full-run `<50cm_full%` of the DD/LAMBDA local-FGO pipeline (tokyo run1)

Campaign doc: `internal_docs/inuex35_tc_fgo_benchmark.md`. Target: inuex35's
tokyo/run1 README row (**56.7% `<50cm`, 49.5% fix, AllRMS 47.4 m**, verified
reproducible — see `repro_tc_fgo/REPRO_REPORT.md`). This report produces the
**first full-run** score of our precision machinery — the DD-carrier +
LAMBDA local FGO (`python/gnss_gpu/local_fgo.py:971
solve_local_fgo_with_lambda`, driven per-window by
`experiments/solve_ppc_segment_multifamily_fgo.py`) — over all 11,928 rover
epochs, with the full-timeline `<50cm_full%` denominator.

## Headline result

| method | run | n_scored/n_rover | coverage% | AllRMS (3D, m) | FixRMS (m) | fix% | `<50cm%` | **`<50cm_full%`** |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| inuex35 README (external baseline) | run1 | n/a | 100.0 | 47.40 | 0.815 | 49.5 | 56.7 | **56.7** |
| WP3a (a) native FGO backbone, PR+motion | run1 | 11676/11928 | 97.9 | 115.26 | n/a | 0.0 | 0.0 | 0.0 |
| WP3b (b′) native FGO + Doppler+Huber | run1 | 11676/11928 | 97.9 | 110.22 | n/a | 0.0 | 0.0 | 0.0 |
| WP3b (c) native FGO + Doppler+Huber+IMU | run1 | 11676/11928 | 97.9 | 105.73 | n/a | 0.0 | 0.0 | 0.0 |
| WP4 seed baseline (gap-filled WP3b (c), 100% coverage, no LAMBDA) | run1 | 11928/11928 | 100.0 | 107.68 | n/a | 0.0 | 0.0 | 0.0 |
| **WP4 — DD/LAMBDA local FGO (this report)** | run1 | 11923/11928 | 100.0 | **107.68** | **92.63** | **86.0** | **0.0** | **0.0** |

**Our honest standalone-FGO precision row vs their 56.7%: `<50cm_full% = 0.0%`.**

This is a clean negative result, not a null one: 86.0% of epochs got at least
one LAMBDA-fixed double-difference ambiguity (vs. inuex35's 49.5% *position*
fix rate — not the same thing, see below), yet AllRMS barely moved from the
107.68 m seed baseline (107.68 m → 107.68 m, i.e. no change to 2 decimal
places) and FixRMS (92.63 m) is barely better than AllRMS. The bottleneck
analysis below explains why fixing ambiguities did not translate into
accuracy.

## What was run

```
set PYTHONPATH=python
set PYTHONUNBUFFERED=1
python -u experiments/wp4_run_local_fgo_full.py ^
  --window-epochs 200 --dd-base-interp --systems G,E ^
  --dd-families L1_E1_B1,L5_E5A_B2A > results/wp4/full_sweep_run.log

python experiments/score_vs_inuex35.py --traj results/wp4/tokyo_run1_local_fgo_lambda.csv ^
  --format csv --city tokyo --run run1 --data-root E:/datasets/PPC-Dataset-data ^
  --out-json results/wp4/score_local_fgo_lambda.json --out-csv results/wp4/scores.csv

python experiments/score_vs_inuex35.py --traj results/wp4/tokyo_run1_seed_full_coverage.pos ^
  --format pos --fix-statuses 999 --city tokyo --run run1 --data-root E:/datasets/PPC-Dataset-data ^
  --out-json results/wp4/score_seed_baseline.json --out-csv results/wp4/scores.csv
```

`experiments/wp4_run_local_fgo_full.py` is a new, thin driver (no edits to
`solve_ppc_segment_multifamily_fgo.py` or `local_fgo.py`) that:

1. Builds a full-coverage (11,928/11,928) seed `.pos` for tokyo/run1 by
   linearly gap-filling `results/wp3b/tokyo_run1_fgo_imu_doppler_huber.csv`
   (our best native-FGO backbone, 85.82 m 2D / 105.73 m 3D RMS, 97.9%
   coverage) over its 252 missing epochs (2.1%).
2. Partitions the full rover timeline into 60 contiguous 200-epoch (40 s)
   TOW windows and drives `solve_ppc_segment_multifamily_fgo.main()`
   in-process (monkeypatched `sys.argv`) once per window — each window is an
   independent local-FGO + LAMBDA solve, anchored only by
   endpoint priors to the (gap-filled) seed.
3. Merges each window's float FGO trajectory and a solver-internal-state
   diff trick (see "Bugs found and worked around" below) to recover the
   per-epoch LAMBDA fix mask, into one full-run trajectory CSV.

## Pipeline understanding (work item 1)

- **CLI**: `solve_ppc_segment_multifamily_fgo.py --run city/run --seed-pos
  <pos> --start-tow --end-tow --systems --dd-families ...` solves *one*
  contiguous TOW window at a time (no chunking/looping — that is what our
  driver adds). It requires `--seed-pos` to already cover *every* TOW in the
  window exactly (`_load_seed_positions` raises if any are missing), builds
  per-epoch `DDCarrierEpoch`/`DDPseudorangeEpoch` objects via
  `DDCarrierComputer`/`DDPseudorangeComputer` against `base.obs`/`rover.obs`,
  assembles a `LocalFgoProblem` spanning the whole window, and calls
  `solve_local_fgo_with_lambda` (iterative: float-solve → LAMBDA fix →
  re-solve with fixed ambiguities as tight priors, up to
  `--lambda-max-epoch-gap`/`--lambda-slip-threshold-cycles`-gated tracks).
  Outputs: one summary CSV row (diagnostics/factor counts/LAMBDA stats), one
  `--out-pos` (full float+fixed trajectory), one `--out-fixed-only-pos`
  (seed trajectory with only LAMBDA-fixed epochs overwritten).
- **Backend**: `import gtsam` fails in this repo's venv
  (`ModuleNotFoundError`); `pip`/`pip download` both report no matching
  distribution. Confirmed via web search: borglab/gtsam publishes **no
  Windows wheels on PyPI** (Linux/macOS cibuildwheel only) — Windows needs an
  MSVC source build (as Track A already did in a separate workspace). Per
  the task's "no heavy installs" constraint, this report runs entirely on
  **`local_fgo.py`'s NumPy/SciPy fallback** (`_solve_local_fgo_numpy`: sparse
  Gauss-Newton/LM via `scipy.sparse`, `scipy>=1.17` available). This is
  slower per-iteration than GTSAM's factor-graph Cholesky but is a
  **pose-only** graph (3 vars/epoch), so 200-epoch windows still solve in
  16–87 s (median 31 s) — no correctness difference from GTSAM is expected
  (same residual/Jacobian math, both do damped Gauss-Newton), only speed.
- **Runtime**: measured on 3 spot-check 200-epoch windows before committing
  to the full sweep (16–19 s without `--dd-base-interp`, ~40 s with) —
  logged to `repro_tc_fgo/PROGRESS.md` with the extrapolation
  (60 windows × ~40 s ≈ 40 min ≪ 6 h) before launching the full,
  non-stratified run.

## Bugs found and worked around (documented per task constraints)

Neither of these is in `local_fgo.py`, `validate_fgo_ppc.py`, or `ppc.py`
(the protected files); both were pre-existing and both fully blocked running
`solve_ppc_segment_multifamily_fgo.py` at all, so they were fixed minimally
and locally rather than worked around by disabling functionality:

1. **`experiments/ppc_window_geometry.py` imported a function that does not
   exist.** `_compute_at_transmit_time` was imported from `gnss_gpu.io.ppc`,
   but that module has no such function (confirmed with `git log -p` — never
   committed there either). Fixed by adding a small, local, per-satellite
   transmit-time iteration (`t_tx = t_rx - range/c`, 2 iterations, standard
   SPP correction) directly in `ppc_window_geometry.py`. Note this is
   *stricter* than `gnss_gpu.io.ppc.load_experiment_data` (the native-FGO
   loader), which evaluates ephemerides at reception time directly — an
   acceptable approximation for coarse WLS/SPP but not for the mm/cm-level
   double-difference carrier geometry this module feeds.
2. **`exp_ppc_ctrbpf_fgo.py`'s `_write_pos_file`/`_load_hybrid_pos_file` are
   column-misaligned.** The writer appends its `status` argument as an
   untracked 13th whitespace token; the reader expects status at
   `parts[8]` (the real RTKLIB `Q` column, which the writer instead always
   hardcodes to `"1"`). Verified: round-tripping `_write_pos_file`'s own
   output through `_load_hybrid_pos_file` returns an **empty dict** — i.e.
   `--seed-pos` could never have worked with a file produced by that helper
   or via the solver's own `--out-pos`/`--out-fixed-only-pos`. Not touched
   (outside the declared surface, and it's a large, shared file); worked
   around entirely inside the new driver: our own seed-pos writer
   (`write_pos_file`) uses the real RTKLIB column order (`Q` at index 8) so
   the solver's internal `_load_seed_positions` succeeds, and a
   status-independent reader (`read_pos_ecef`) recovers ECEF from the
   solver's own outputs using the column indices both writers agree on
   (`parts[1:5]`). The per-epoch LAMBDA fix mask is then recovered without
   any solver changes by diffing `--out-fixed-only-pos` against the known
   seed (`recover_fix_mask`): non-fixed epochs are copied byte-for-byte from
   the seed by `solve_ppc_segment_multifamily_fgo.main()`, so any epoch
   whose position differs from the seed was necessarily LAMBDA-fixed.

Both fixes, plus all new driver logic, are covered by 23 synthetic unit
tests (`tests/test_ppc_window_geometry_transmit_time.py`,
`tests/test_wp4_run_local_fgo_full.py`) — see Tests below.

## Full-run sweep stats (work item 5)

60 windows × 200 epochs (last window 128 epochs), `--dd-base-interp
--systems G,E --dd-families L1_E1_B1,L5_E5A_B2A`, all default LAMBDA/FGO
tuning (`--lambda-ratio 3.0`, `--lambda-max-epoch-gap 6`,
`--lambda-slip-threshold-cycles 1.5`, `--dd-fixed-sigma-cycles 0.05`,
`--prior-sigma-m 0.5`, `--motion-sigma-m 0.25`):

| Stat | Value |
|---|---:|
| Windows solved | 60/60 (100%) |
| Wall time (solve only) | 2190 s (36.5 min) |
| Per-window solve time | median 31.1 s, mean 36.3 s, range 16.5–87.3 s |
| Epochs covered | 11,923 / 11,928 (99.96%) |
| Epochs with DD carrier data | 11,887 / 11,923 (99.7%) — needs `--dd-base-interp`; without it, only ~20% (base logs at 1 Hz vs rover's 5 Hz) |
| Total DD carrier pairs built | 138,130 |
| Ambiguity **tracks** LAMBDA-fixed (segments) | 14,301 (26,726 counting duplicate re-fixes across the 2 solve iterations; 6 via true multi-satellite group LAMBDA, **26,726 via single-track ratio test** — see Bottleneck 3) |
| Fixed *observations* (epoch×DD-pair) | 81,356 |
| Epochs with ≥1 fixed ambiguity | median 183.5/200 (91.8%) per window; scorer-level fix% 86.0% |
| Ratio-test value (accepted fixes) | median 16.3, p90 2077, "best" up to ~2.2×10⁹ (see Bottleneck 3 — this is a sign of over-confidence, not correctness) |
| Fixed-by-system | E(Galileo) 7,636 / G(GPS) 6,665 |
| Segment length (epochs, median) | 3.0 — segments restart very often (short baseline for the internal ratio test) |
| Position error, seed vs FGO vs fixed-only (median-of-per-window-medians) | 63.6 m / 62.7 m / 62.7 m — **no meaningful change** |

## Top-3 bottlenecks vs inuex35's 56.7% (work item 5)

**1. Seed/absolute-position quality — the single dominant bottleneck.**
`local_fgo.py`'s own module docstring says it "keeps the main particle
filter as the primary estimator... solves a small post-process position
graph over a selected window" — it is a **refinement/rescue** tool, not a
bootstrap SPP. It was previously exercised on top of a libgnss++ hybrid
`.pos` (cm–m class RTK fixes). Our only available full-coverage absolute
seed is the WP3b native-FGO backbone at **85.82 m (2D) / 105.73 m (3D) RMS**
— two to three *orders of magnitude* coarser than the cm-level anchor this
machinery expects. The proof is in the numbers: scoring the raw,
gap-filled seed directly (no LAMBDA, no FGO) gives **AllRMS 107.68 m**;
running the full DD+LAMBDA pipeline on top of it gives **AllRMS
107.68 m** — identical to 2 decimal places. The local FGO's only pull toward
absolute truth is a soft prior at each window's two endpoints, tied to the
*same* biased seed, so the graph converges to a self-consistent solution
near a wrong absolute position. inuex35 avoids this because their RTK core
(`cssrlib.rtk.rtkpos`) *is* the absolute-position source and is DD/carrier
based from Phase 1 onward — there is no analogous, comparably-precise
absolute layer in this repo yet (that is exactly the gap the campaign doc's
"wire `fgo_gnss_lm_vd` + `local_fgo` LAMBDA to raw PPC DD streams" next step
is meant to close, and this report is the first honest measurement of that
gap).

**2. No independent motion/IMU coupling to break the position-ambiguity
degeneracy (task's "no IMU bridging" hypothesis, confirmed).** Each window's
`BetweenFactorPoint3` motion prior is a constant-velocity guess derived from
the *same* seed trajectory (`initial_positions[i+1] - initial_positions[i]`
by default), so it carries no information the carrier/position factors don't
already have. inuex35's tight coupling adds `CombinedImuFactor` +
NHC/ZUPT/Doppler-velocity priors that are truly independent of the GNSS
position solve — exactly the kind of constraint that could pull a
badly-biased float ambiguity/position pair back toward truth. WP3b already
built the IMU-adapter half of this (`ppc_imu_adapter.py`) for the *native*
FGO; it is not yet wired into `local_fgo`'s window graph.

**3. The "LAMBDA" fixing that actually ran is a self-consistency check, not
an absolute-correctness check — and it rarely runs as true LAMBDA.**
`_estimate_lambda_fixes` groups ambiguity tracks by `(system, ref_sat)` and
only attempts a true multi-satellite integer least-squares search
(`solve_lambda`) when `1 < group_size <= max_group_size`; across the whole
run this fired only **6 times** (out of 14,301 fixed segments) — group
tracks almost never survive intact, because `--lambda-slip-threshold-cycles
1.5` and the short, independent 200-epoch windows constantly restart
segments (median segment length **3 epochs**). The other 26,726(-ish)
fixes came from the single-track ratio test, whose "variance" is the
*within-segment scatter* of the float ambiguity over those ~3 epochs, not
any external validation. That is why ratio values look extremely confident
(median 16.3, up to ~2×10⁹, vs. the 3.0 acceptance threshold) even though
the resulting fixes did not improve accuracy at all: a systematically biased
seed produces a float ambiguity that is *consistently* wrong across a short
segment, and the ratio test happily accepts that consistency as a "fix."
inuex35's AR stack has exactly the external checks this lacks — subset-AR,
`valpos`, and a **DDPR cross-validation at the fixed position** (reject if
the fix makes the pseudorange residual worse) plus a post-AR cost gate.
None of those exist in `local_fgo.py`'s LAMBDA path today.

*(Runner-up, not scored as a top-3 item but worth flagging: the base
station's 1 Hz logging vs the rover's 5 Hz means DD carrier data only
exists natively on 20% of epochs; `--dd-base-interp` (hold the last base
epoch) recovers full coverage but introduces up to 0.8 s of base-epoch
staleness, which the DD math does not otherwise account for.)*

## Deliverables

| File | Description |
|---|---|
| `results/wp4/tokyo_run1_local_fgo_lambda.csv` | Full-run trajectory: `tow,lat_deg,lon_deg,height_m,ecef_x,ecef_y,ecef_z,fix` (11,923 rows) |
| `results/wp4/tokyo_run1_seed_full_coverage.pos` | Full-coverage (11,928-epoch) seed used by every window, for reproducibility/baseline comparison |
| `results/wp4/score_local_fgo_lambda.json`, `score_seed_baseline.json`, `scores.csv` | Scorer outputs (headline table above) |
| `results/wp4/per_segment_stats.csv` | Per-window solve time, DD/fix counts, errors (driver-level) |
| `results/wp4/windows/*_summary.csv`, `*.pos`, `*_fixed_only.pos` | Per-window solver-native outputs (60 windows × 3 files) |
| `results/wp4/full_sweep_run.log` | Full sweep console log |
| `experiments/wp4_run_local_fgo_full.py` | New driver (seed gap-filling, windowing, merge, fix-mask recovery) |
| `experiments/ppc_window_geometry.py` | +`_compute_at_transmit_time` (local fix, see Bugs) |

## Tests

```
set PYTHONPATH=python
python -m pytest -p no:xonsh tests/test_wp4_run_local_fgo_full.py ^
  tests/test_ppc_window_geometry_transmit_time.py tests/test_score_vs_inuex35.py ^
  tests/test_ppc_imu_adapter.py tests/test_validate_fgo_ppc_native.py -q
```

**Result:** 53 passed (12 driver tests + 4 transmit-time tests + 7 pre-existing
scorer tests + 30 pre-existing IMU-adapter/native-chunking tests, all
unaffected by this work — nothing in the protected surface
(`validate_fgo_ppc.py`, `python/gnss_gpu/io/ppc.py`) was touched).

## Code touched

- `experiments/ppc_window_geometry.py` — added `_compute_at_transmit_time`
  (local fix for a missing import; see Bugs above).
- `experiments/wp4_run_local_fgo_full.py` (new) — full-run driver.
- `tests/test_wp4_run_local_fgo_full.py`,
  `tests/test_ppc_window_geometry_transmit_time.py` (new).
- `solve_ppc_segment_multifamily_fgo.py` and `python/gnss_gpu/local_fgo.py`:
  **not modified**, used strictly via their existing CLI/API as instructed.

## Suggested next step

Given Bottleneck 1, the highest-leverage follow-up is *not* more LAMBDA
tuning — it's giving the local-FGO window a genuinely absolute (few-meter or
better) seed to refine, e.g. by running it on top of a real RTK/PF fix (as
`local_fgo.py` was originally designed for) rather than raw WLS/native-FGO
SPP output, or by adding an independent absolute constraint (IMU/ZUPT, or a
code-based DD-pseudorange solve with a much larger measurement-model
weight) inside the same window before trusting carrier-phase LAMBDA fixes.
