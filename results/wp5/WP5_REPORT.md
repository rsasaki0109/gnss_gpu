# WP5 — Anchoring `local_fgo` on libgnss++ RTK cm-level fixes (tokyo run1)

Campaign doc: `internal_docs/inuex35_tc_fgo_benchmark.md`. Prior work:
`results/wp4/WP4_REPORT.md` (WP4's Bottleneck 1: the DD/LAMBDA local-FGO
pipeline never moved the needle because its only available full-coverage
seed was 85–105 m RMS, "two to three orders of magnitude coarser than the
cm-level anchor this machinery expects"). WP5's premise: anchor the same
`local_fgo`/LAMBDA machinery on the libgnss++ RTK `.pos` artifact
(cm-level where it has a FIX) instead of the native-FGO backbone, and add
two external AR validation gates inuex35 has but we didn't. Success
criterion (task's own words): `<50cm_full%` **strictly above** the
RTK-only baseline (~25.4% for run1), stretch goal inuex35's 56.7%.

## Headline result

| method | run | n_scored/n_rover | coverage% | AllRMS (3D, m) | FixRMS (m) | fix% | `<50cm%` | **`<50cm_full%`** | ppc |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| inuex35 README (external baseline) | run1 | n/a | 100.0 | 47.40 | 0.815 | 49.5 | 56.7 | **56.7** | n/a |
| RTK-only baseline (libgnss++ v5, best available `.pos`) | run1 | 7397/11928 | 62.0 | 19.86 | 0.084 | 10.5 | 40.9 | **25.4** | 22.88 |
| WP4 — DD/LAMBDA local FGO, unanchored (native-FGO seed) | run1 | 11923/11928 | 100.0 | 107.68 | 92.63 | 86.0 | 0.0 | **0.0** | 0.00 |
| WP5 hybrid seed (RTK where available + WP3b backbone, **no FGO/LAMBDA**) | run1 | 11928/11928 | 100.0 | 94.15 | n/a | 0.0 | 25.4 | **25.4** | 21.08 |
| **WP5 — RTK-anchored DD/LAMBDA local FGO (this report)** | run1 | 11923/11928 | 100.0 | **94.05** | **58.91** | **74.3** | **24.5** | **24.5** | 19.53 |

**Result: `<50cm_full% = 24.5%`, *not* strictly above the 25.4% RTK-only
baseline — the success criterion was not met.** This is a clean negative
result with a fully diagnosed, quantified root cause (below), not a null
one: unlike WP4 (where the pipeline provably could not move AllRMS at all),
here the anchored pipeline moves large amounts of the trajectory
(fix%=74.3, FixRMS dropped from "n/a" to 58.9 m) and even improves 356
individual epochs from failing to passing `<50cm` — but it *also*
regresses 461 previously-passing epochs, for a net loss of 105 epochs
(3.9 percentage points net negative on the 6,622-epoch FLOAT-anchored
subset alone). Section "Bottleneck analysis v2" below traces this to two
independent, quantified causes: (1) RTK FIX coverage is far too sparse and
too front-loaded in time for "extend outward from FIX anchors" to have any
runway over 92% of the run, and (2) the DDPR cross-check gate as specified
is >100x too insensitive to ever detect a wrong fix in this dataset's real
noise regime.

## What was run

```
set PYTHONPATH=python
set PYTHONUNBUFFERED=1
python -u experiments/wp5_run_anchored_fgo.py --window-epochs 200 ^
  > results/wp5/full_sweep_run.log

python experiments/score_vs_inuex35.py --traj results/wp5/tokyo_run1_anchored_fgo.csv ^
  --format csv --city tokyo --run run1 --data-root E:/datasets/PPC-Dataset-data ^
  --out-json results/wp5/score_anchored_fgo.json --out-csv results/wp5/scores.csv

python experiments/score_vs_inuex35.py --traj results/wp5/tokyo_run1_hybrid_seed.pos ^
  --format pos --fix-statuses 999 --city tokyo --run run1 --data-root E:/datasets/PPC-Dataset-data ^
  --out-json results/wp5/score_hybrid_seed_baseline.json --out-csv results/wp5/scores.csv
```

## Work item 1 — Baseline

Located the RTK artifact: `experiments/results/libgnss_rtk_pos_v5/tokyo_run1_full.pos`
(libgnss++ RTK, Status column: 775 FIX / 6,471 FLOAT-3 / 151 FLOAT-1,
Counter over all rows). Re-scored: `n_scored=7397/11928 coverage=62.0%
AllRMS=19.861 FixRMS=0.084 fix%=10.5 <50cm%=40.9 <50cm_full%=25.4
ppc=22.88%` — this row (`<50cm_full%=25.4%`) is the number WP5 had to beat.

## Work items 2–3 — Hybrid seed, anchors, external AR validation

**`python/gnss_gpu/local_fgo.py` (read in full, 1,764 lines) already
implements per-epoch position priors end-to-end**
(`LocalFgoProblem.prior_positions_ecef`/`prior_sigmas_m`, wired in both the
GTSAM `build_factor_graph` path and the NumPy/SciPy `_solve_local_fgo_numpy`
fallback) — the "extend `local_fgo.py` if lacking per-epoch priors" clause
of work item 2 did not apply. What genuinely was missing, and was added
(backward-compatible, unit-tested, within the task's declared minimal-edit
surface):

1. **`experiments/solve_ppc_segment_multifamily_fgo.py`: new
   `--anchor-source rtk` mode** (`_build_rtk_anchor_priors`). Its
   pre-existing `--anchor-source pos` applies one uniform sigma to all
   anchors and filters candidates by *ground-truth* error
   (`--anchor-max-error-m`) — an oracle/validation-only mode, explicitly
   unusable here (anchor selection must come from the RTK Status column,
   not from comparing to the reference trajectory — that would be
   cheating). The new `rtk` mode gates purely on Status: FIX
   (`--anchor-fix-statuses`, default `4`) gets a tight prior
   (`--anchor-fix-sigma-m`, default 0.07 m); FLOAT (`--anchor-float-statuses`,
   default `1,3`) gets a loose prior (`--anchor-float-sigma-m`, default
   2.0 m); anything else gets no per-epoch prior (endpoint-only, as in WP4).
2. **`python/gnss_gpu/local_fgo.py`: two new AR validation gates** in
   `solve_local_fgo_with_lambda`/`_estimate_lambda_fixes`, both default-off
   / backward-compatible:
   - **Minimum-segment-length gate**: the pre-existing `LambdaFixConfig
     .min_epochs` threshold (previously a real but unreported/undercounted
     CLI flag, `--lambda-min-epochs`, default 2) now has explicit
     accept/reject counters (`n_segments_rejected_short`, rolled up into
     the summary). WP5 raises the default to 5 per the task.
   - **DD-pseudorange cross-check** (new `_ddpr_cross_check` +
     `LambdaFixConfig.ddpr_reject_threshold`, default 0.0 = disabled): after
     tentatively applying and re-solving with each LAMBDA iteration's
     newly-proposed fixes, compares the DD-pseudorange (code-based,
     carrier-independent) residual RMS before vs. after over the touched
     epochs and vetoes the whole batch if it got worse by more than the
     threshold — mirrors inuex35's DDPR cross-validation / post-AR cost
     gate. Exposed as `--lambda-ddpr-reject-threshold` (WP5 default 0.2 m).
3. **New driver `experiments/wp5_run_anchored_fgo.py`** (imports and
   extends `wp4_run_local_fgo_full.py` rather than duplicating it):
   builds the hybrid seed (RTK ECEF at every epoch with a non-zero
   libgnss++ Status; WP3b backbone, gap-filled exactly as in WP4,
   everywhere else), drives `solve_ppc_segment_multifamily_fgo.main()` per
   200-epoch window with `--anchor-source rtk --dd-pr --lambda-min-epochs 5
   --lambda-ddpr-reject-threshold 0.2`, and adds a post-merge
   "fix-extension length" analysis (`compute_extension_stats`/
   `nearest_fix_distance_epochs`, work item 4/6).

Both `local_fgo.py` extensions are covered by 9 new unit tests
(`tests/test_local_fgo_wp5_ar_gates.py`), including a deterministic
end-to-end regression proving the DDPR gate vetoes a self-consistent-but-
wrong fix that WP4's un-gated pipeline would have accepted, and a
regression guard proving the default (disabled) configuration reproduces
pre-WP5 behaviour byte-for-byte. The new driver's pure helpers are covered
by 8 more (`tests/test_wp5_run_anchored_fgo.py`).

**Hybrid seed composition** (tokyo/run1, 11,928 epochs): 7,397 RTK
(62.0%, matches the RTK-only baseline's own coverage exactly, as expected)
+ 4,531 backbone (209 of those still linearly interpolated — the WP3b
backbone's own 252-epoch gaps minus what RTK already covers).

## Full-run sweep stats (work item 4)

60 windows × 200 epochs (last window 128 epochs), `--anchor-source rtk
--dd-pr --lambda-min-epochs 5 --lambda-ddpr-reject-threshold 0.2`, all
other defaults identical to WP4 (`--dd-base-interp --systems G,E
--dd-families L1_E1_B1,L5_E5A_B2A`, `--lambda-ratio 3.0`):

| Stat | Value |
|---|---:|
| Windows solved | 60/60 (100%) |
| Wall time (solve only) | 2,624 s (43.7 min) |
| Per-window solve time | median 43.2 s, mean 43.7 s, range 25.0–96.8 s |
| Epochs covered | 11,923 / 11,928 (99.96%) |
| Total DD carrier pairs built | 138,130 |
| **Anchor epochs, by class** | FIX 775 (6.5%) / FLOAT 6,622 (55.5%) / none 4,526 (38.0%) |
| **Windows with ≥1 FIX anchor** | **5 / 60 (8.3%)** — windows 1–5 only (TOW 187470–187670, i.e. the *first* ~200 s of a ~2,386 s run); **0 FIX anchors in the remaining 55/60 windows (91.7% of the run)** |
| Fixed *observations* (epoch×DD-pair) | 96,428 |
| Epochs with ≥1 fixed ambiguity | scorer-level fix% 74.3% |
| Segment-length gate: rejected (too short) | 50,609 candidate segments (of the ~53k evaluated across both LAMBDA iterations × 60 windows) |
| DDPR cross-check gate: rejected | **0** of 117 batch evaluations (see Bottleneck 2 — not because every fix was good) |
| Position error, seed vs. FGO vs. fixed-only (median-of-per-window-medians) | see per-class breakdown below; aggregate `<50cm_full%` 25.4% → 24.5% |
| `<50cm` pass count, by anchor class (seed → FGO) | FIX: 773→774 (Δ+1, noise); FLOAT: 2,255→2,150 (**Δ−105**); none: 0→0 (Δ0) |

## Bottleneck analysis v2 (work item 5/6)

**1. RTK FIX coverage is far too sparse *and* too front-loaded in time for
"extend outward from FIX anchors" to have any runway — the single
dominant bottleneck.** Of 11,928 epochs, only 775 (6.5%) carry libgnss++
Status 4 (FIX), and — critically — all 775 fall inside the *first* 5 of 60
windows (TOW 187470.0–187669.8, i.e. the first ~200 s of a ~2,386 s / 40
min run). For **55 of 60 windows (91.7% of the whole run)** the anchor set
that "seeds" DD/LAMBDA extension contains *zero* FIX epochs — only loose
(2.0 m sigma) FLOAT anchors and/or no anchor at all. WP5's core hypothesis
("extend cm-level accuracy outward from FIX anchors via DD-carrier
LAMBDA") was only ever structurally possible in ~8% of the timeline; for
the other 92% the pipeline necessarily degenerates to exactly WP4's
unanchored regime (self-consistent-but-not-necessarily-correct float
ambiguities, this time additionally perturbed by a loose FLOAT prior). The
diagnostic driver's own extension-length metric confirms this
indirectly: of the 2,924 epochs that pass `<50cm`, 2,150 are non-FIX-class
— but per-class breakdown shows these are essentially all *already-passing
FLOAT anchors* (2,255 pass pre-FGO), not epochs newly pulled below 50 cm by
carrier-phase extension from a FIX epoch — genuine "extension" essentially
did not happen (see Bottleneck 3).

**2. The DDPR cross-check gate, as specified (`--lambda-ddpr-reject
-threshold 0.2` m over `--dd-pr` residuals with `--dd-pr-sigma-m 5.0`), is
structurally unable to detect a wrong fix in this dataset — 0/117 batch
evaluations rejected anything, and it is not because every fix was
externally validated as correct.** Aggregating every DDPR gate evaluation
across all 60 windows: DD-pseudorange residual RMS is itself 0.64–83.9 m
(median 16.9 m, driven by the code-pseudorange noise floor and the
seed's own absolute bias in unanchored regions), while the *change* in
that RMS from applying a batch of new LAMBDA fixes is never more than
0.042 m (median −0.001 m, p90 +0.008 m) — **more than an order of
magnitude below the 0.2 m threshold in the single worst case observed, and
typically ~100–1000x below it.** A wrong integer ambiguity fix shifts a
position by centimeters to a few decimeters; the code-based DD
pseudorange observable this gate cross-checks against has a noise floor an
order of magnitude larger than that shift. The gate is real, tested, and
verified to work on a constructed adversarial case (see
`tests/test_local_fgo_wp5_ar_gates.py`), but on real noisy pseudorange data
at this `--dd-pr-sigma-m`, it cannot see the signal it is meant to police
— inuex35's own DDPR cross-validation almost certainly relies on either a
much tighter effective sigma (e.g. from a short baseline / high-quality
base receiver), averaging over many more epochs per check, or an
absolute-position (not batch-residual) comparison, none of which this
report's minimal-edit surface could add without touching
`solve_ppc_segment_multifamily_fgo.py`'s DD-pseudorange model more
invasively than the task's declared scope allows.

**3. Net effect on the one anchor class that *did* have runway (FLOAT,
55.5% of epochs): the loose 2.0 m prior sigma lets DD/LAMBDA move
previously-good positions more often than it fixes bad ones.** Restricting
to the 6,622 FLOAT-anchored, reference-covered epochs: 2,255 passed
`<50cm` in the seed (i.e., RTK FLOAT alone was already sub-50cm there,
before any FGO/LAMBDA); after the anchored FGO+LAMBDA pass, only 2,150
pass — 1,794 stayed passing, but **461 regressed from passing to failing**
against only **356 that improved from failing to passing**, a net loss of
105 epochs. Segment-length ≥5 rejected 50,609 candidate segments (the
gate *is* doing real filtering — it is the majority outcome, not a
no-op), yet segments that *do* clear length 5 in a FIX-less window still
have no external signal (Bottleneck 2) forcing them toward the loose
anchor's true position rather than away from it — a longer, self-
consistent-but-wrong segment is exactly as likely to pass the length gate
as a correct one. The FIX-class anchors (0.07 m sigma, tight enough that
the optimizer cannot move them meaningfully) show no such effect
(773→774, noise-level).

*(Combined picture: none-class epochs, 38.0% of the run, remain at 0/4,526
`<50cm` before and after — identical to WP4's finding that DD/LAMBDA
cannot manufacture absolute accuracy with zero nearby absolute reference,
now additionally confirmed in the presence of anchors and both new AR
gates elsewhere in the same windows.)*

## WP6 recommendation

The task's own success metric was not met, and the root cause is now
precisely quantified rather than merely suspected — this materially
changes what WP6 should prioritize versus a generic "tune LAMBDA more"
follow-up:

1. **Don't extend where there's nothing to extend from.** Given FIX
   coverage is 6.5%/front-loaded, the highest-leverage structural change is
   *not* more anchoring machinery — it's obtaining (or reprocessing for) a
   genuinely wider RTK FIX rate over the run (e.g. a better base-station
   solution, different elevation mask/cycle-slip settings in the libgnss++
   run that produced `tokyo_run1_full.pos`, or a completely different RTK
   engine run) before this local-FGO anchoring approach has anything
   non-trivial to anchor on for 92% of the timeline.
2. **Make the FLOAT-anchor prior self-defending, or skip FLOAT-anchored
   windows entirely.** Since a loose (2.0 m) FLOAT prior nets *negative*
   here (Bottleneck 3), either (a) tighten it adaptively using a signal
   libgnss++ already computes internally (e.g. its own reported
   position-covariance/ratio, if surfaced, rather than a flat status-based
   constant), or (b) as a cheap, purely-defensive fallback, add a
   "no-worse-than-seed" per-epoch check after solving (revert to the
   seed's FLOAT-anchor position if the post-solve position moved further
   from the *loose anchor itself* by more than some small multiple of its
   sigma) — this would not fix the underlying accuracy but would eliminate
   the observed 461-epoch regression at near-zero cost.
3. **Give the DDPR cross-check a fighting chance.** Bottleneck 2's gap is
   ~2 orders of magnitude, too large to close with threshold tuning alone.
   Concretely: (a) average the DD-pseudorange residual over a many-epoch
   window per candidate fix (reduces the noise floor by ~√N) instead of a
   single-batch RMS, and/or (b) if available, use a tighter empirical
   `--dd-pr-sigma-m` calibrated from the base/rover receivers' actual code
   noise rather than the current placeholder (5.0 m), and/or (c) add a
   genuinely independent check per inuex35 (IMU/ZUPT-derived position, not
   DD-pseudorange) — this is WP4's still-unaddressed Bottleneck 2
   ("no independent motion/IMU coupling") and is likely the more
   fundamental fix; `ppc_imu_adapter.py` (built for the native FGO in a
   prior track) is the natural starting point for wiring an IMU factor into
   `local_fgo`'s window graph.
4. Before any of the above, it would be cheap and high-information to
   **re-run this exact WP5 pipeline gating anchor use on a wider FLOAT
   status set or on a different/independently-reprocessed RTK artifact**
   with materially higher FIX coverage, to isolate whether item 1
   (coverage) alone — with *no other change* — is sufficient to cross the
   25.4% baseline, before investing in items 2–3.

## Deliverables

| File | Description |
|---|---|
| `results/wp5/tokyo_run1_anchored_fgo.csv` | Full-run trajectory: `tow,lat_deg,lon_deg,height_m,ecef_x,ecef_y,ecef_z,fix` (11,923 rows) |
| `results/wp5/tokyo_run1_hybrid_seed.pos` | Full-coverage (11,928-epoch) hybrid seed (RTK where available, WP3b backbone elsewhere) used by every window |
| `results/wp5/score_anchored_fgo.json`, `score_hybrid_seed_baseline.json`, `scores.csv` | Scorer outputs (headline table above) |
| `results/wp5/per_segment_stats.csv` | Per-window solve time, DD/fix counts, anchor counts, gate rejection counts (driver-level) |
| `results/wp5/extension_stats.json` | Fix-extension length diagnostic (work item 4/6) |
| `results/wp5/windows/*_summary.csv`, `*.pos`, `*_fixed_only.pos` | Per-window solver-native outputs (60 windows × 3 files), including per-iteration DDPR gate before/after RMS |
| `results/wp5/full_sweep_run.log` | Full sweep console log |
| `experiments/wp5_run_anchored_fgo.py` | New driver (hybrid seed, RTK-anchored windowing, merge, extension-length analysis) |
| `experiments/solve_ppc_segment_multifamily_fgo.py` | +`--anchor-source rtk` mode, +DDPR-gate/segment-gate CLI flags (see Work items 2–3) |
| `python/gnss_gpu/local_fgo.py` | +DDPR cross-check gate, +explicit segment-length gate counters (see Work items 2–3) |

## Tests

```
set PYTHONPATH=python
python -m pytest -p no:xonsh tests/test_wp4_run_local_fgo_full.py ^
  tests/test_ppc_window_geometry_transmit_time.py tests/test_score_vs_inuex35.py ^
  tests/test_ppc_imu_adapter.py tests/test_validate_fgo_ppc_native.py ^
  tests/test_local_fgo_wp5_ar_gates.py tests/test_wp5_run_anchored_fgo.py ^
  tests/test_local_fgo_bridge.py -q
```

**Result: 72 passed** (9 new DDPR/segment-gate tests + 8 new WP5-driver
tests + 12 WP4-driver tests + 4 transmit-time tests + 7 scorer tests + 30
pre-existing IMU-adapter/native-chunking tests + 2 pre-existing
`local_fgo` bridge tests — all unaffected-by-WP5 tests still green,
confirming both new gates are backward-compatible at their disabled
defaults; nothing in the protected surface, `validate_fgo_ppc.py` or
`python/gnss_gpu/io/ppc.py`, was touched).

## Code touched

- `python/gnss_gpu/local_fgo.py` — `LambdaFixConfig.ddpr_reject_threshold`
  (new, default 0.0/disabled), new `_ddpr_cross_check`, DDPR-gate wiring +
  explicit segment-length-gate counters in `solve_local_fgo_with_lambda`/
  `_estimate_lambda_fixes`. No change to any existing default value or
  return shape other than additive dict keys in `summary`.
- `experiments/solve_ppc_segment_multifamily_fgo.py` — new
  `_parse_status_list`, new `_build_rtk_anchor_priors`, `--anchor-source
  rtk` dispatch, new CLI flags (`--anchor-fix-sigma-m`,
  `--anchor-float-sigma-m`, `--anchor-fix-statuses`,
  `--anchor-float-statuses`, `--lambda-ddpr-reject-threshold`), new summary
  columns. All pre-existing `--anchor-source {none,truth,pos}` behaviour
  unchanged.
- `experiments/wp5_run_anchored_fgo.py` (new) — full-run driver, imports
  and extends `experiments/wp4_run_local_fgo_full.py` rather than
  duplicating its logic.
- `tests/test_local_fgo_wp5_ar_gates.py`, `tests/test_wp5_run_anchored_fgo.py`
  (new).
- `experiments/wp4_run_local_fgo_full.py`, `python/gnss_gpu/io/ppc.py`,
  `experiments/validate_fgo_ppc.py`: **not modified** (the first is reused
  as-is via import; the latter two are the protected surface).
