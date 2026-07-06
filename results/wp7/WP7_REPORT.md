# WP7 — NLOS-weighted RTK: wiring PLATEAU ray-traced LOS/NLOS into libgnss++ DD processing (tokyo run1-3)

Prior work: `results/wp6/WP6_REPORT.md` (esp. stages 6-8's root-cause chain:
the remaining catastrophic failure mode after AR-config tuning is
**FLOAT-filter divergence in urban-canyon segments (10-170 m, internally
indistinguishable from good fixes)** — its own "Next-bottleneck
recommendation" #1 names `tow≈188990-189070` on tokyo/run1 specifically).
Benchmark doc: `internal_docs/inuex35_tc_fgo_benchmark.md` (Track H row: our
unique asset vs inuex35 is the PLATEAU 3D-city-mesh ray-traced LOS/NLOS
classification, LOS median error 1.0 m / AUC 0.92).

**Objective:** get the existing per-satellite PLATEAU LOS/NLOS
classification into the libgnss++ RTK engine's own measurement weighting
(pseudorange/carrier sigma inflation, applied to both the float KF and the
DD formation used by AR), on the theory that this cuts float divergence at
its NLOS/multipath source, plus wire the two WP6-identified dead knobs
(`--arfilter`, `--hold-ratio-threshold`) properly.

## Headline result

**Negative/mixed result for NLOS soft-weighting**, isolated cleanly from a
**positive, unintentional side-effect from the dead-knob fix**:

| effect | run1 `<50cm_full%` | run2 `<50cm_full%` | run3 `<50cm_full%` |
|---|---:|---:|---:|
| WP6 winner, dead knobs still forced off (=WP6's own number) | 26.92 | 43.13 | 43.72 |
| + dead-knob wiring fix (same CLI, `--preset low-cost`'s own settings now active) | 26.64 (−0.28pp) | n/a (not separately isolated on run2/3) | n/a |
| + best NLOS mapping on top (continuous, floor=0.5) | 22.13 (**−4.51pp**) | 45.68 (**+2.51pp**) | 37.31 (**−6.40pp**) |

The single NLOS mapping selected on run1 (least-bad of 10 grid points
tried) **regresses run1 and run3 but improves run2** when applied verbatim
— it does not generalize as a net win. The canyon segment itself
(tow 188990–189070) shows **no meaningful reduction in float divergence**
under any tested NLOS config (mean position error stays ≈119 m, see
"Canyon segment deep-dive" below): sigma inflation is the wrong tool for
~100+ m biased pseudoranges. Both dead knobs are now correctly wired
(unit-tested) and are bit-identical to pre-fix behavior when their
underlying config values are unchanged; their default-preset-driven
activation is a real, small, separately-reported effect, not a WP7 bug.

## 1. Asset verification

Searched `python/gnss_gpu/{nlos_mask,bvh,raytrace}.py`,
`experiments/{build_per_epoch_nlos_csv,prepare_pf_nlos_production}.py`,
`internal_docs/nlos_pf_measurement_wiring.md`, and git log for PRs #117/#118
("NLOS Waves 1-4: PF/DD measurement-layer soft weights").

**Per-epoch, per-satellite LOS/NLOS classifications already exist for all
three tokyo runs**, generated in a prior phase (Phase 33) by
`experiments/build_per_epoch_nlos_csv.py` (PLATEAU 3D-city-mesh BVH
ray-trace, ground-truth rover positions from `reference.csv`, satellite
ECEF from broadcast ephemeris):

```
experiments/results/plateau_nlos_phase33/tokyo_run1_per_epoch_nlos.csv  (17.3 MB)
experiments/results/plateau_nlos_phase33/tokyo_run2_per_epoch_nlos.csv  (16.5 MB)
experiments/results/plateau_nlos_phase33/tokyo_run3_per_epoch_nlos.csv  (28.4 MB)
```

Header: `tow,epoch_idx,prn,is_los,system,svid,elevation_deg,receiver_source,receiver_time_delta_s`
(`is_los` ∈ {0,1}, `prn` already in the `G05`/`C11`/`R05`/`E04`/`J02` format
`libgnss::SatelliteId::toString()` emits — no reformatting needed). No mesh
regeneration was required; **no blocker to log**.

## 2. Interface + mapping design

New, self-contained module (`third_party/gnssplusplus/{include,src}/libgnss++/algorithms/nlos_weights.{hpp,cpp}`,
unit-tested in `tests/test_nlos_weights.cpp`, 14 tests):

- `loadNlosWeightsCsv(path)` — header-driven, case-insensitive CSV parser.
  Accepts **both** contracts: the module's native `tow,sat,los_prob`
  (`los_prob` ∈ [0,1]) and the existing `tow,epoch_idx,prn,is_los,...`
  contract emitted by `build_per_epoch_nlos_csv.py` (boolean `is_los`
  mapped to `los_prob` ∈ {0.0, 1.0}) — the phase-33 CSVs work as-is.
- `lookupLosProb(table, tow, sat_id, tow_tolerance_s)` — nearest-tow lookup
  (mirrors `gnss_gpu.nlos_mask.lookup_nlos_sets`'s tolerance semantics).
  Missing `(tow, sat)` pairs default to `los_prob = 1.0` (LOS, no
  inflation) — a partial mask only ever down-weights satellites it has
  evidence for, never silently excludes others.
- `nlosVarianceInflationFactor(los_prob, mode, ...)` — two mapping flags,
  both flags-first per the task's "start simple, make the mapping a flag":
  - `two-tier`: `los_prob < threshold` (default 0.5) → sigma
    ×`inflation` (default 3.0), else no-op.
  - `continuous`: `sigma² *= 1/max(los_prob, floor)` (task's own suggested
    mapping), `floor` (default 0.05) keeps `los_prob=0` finite.
  - `off` (default): always returns `1.0` regardless of inputs.

Wired into `RTKProcessor::buildMeasurementBlocks()` (`rtk.cpp`): both the
reference-satellite and each other satellite's phase/code variance are
multiplied by `nlos_variance_factor(sat)` before being placed into the DD
measurement rows used by **both** the float KF update and the AR (LAMBDA)
candidate set — exactly the "applied to BOTH" requirement, because DD row
construction is shared code for both consumers.

CLI (`gnss_solve`): `--nlos-weights <csv>`, `--nlos-weight-mode
{off,two-tier,continuous}` (default `off`), `--nlos-two-tier-threshold`,
`--nlos-two-tier-inflation`, `--nlos-continuous-floor`,
`--nlos-tow-tolerance`. `--nlos-weight-mode` without `--nlos-weights` is a
hard CLI error (fail fast, not silently inert).

### Bit-identical verification (absent-flag requirement)

Bisected the 3-hunk diff (epoch-time cache write; NLOS variance-factor
lambda + multiply; dead-knob wiring) and rebuilt+hashed each increment
individually on a 300-epoch tokyo/run1 slice:

| build | SHA-256 of output `.pos` |
|---|---|
| pre-WP7 code (clean rebuild) | `dd7da1…71630` |
| + epoch-time cache write only | `dd7da1…71630` (**identical**) |
| + NLOS variance-factor code (flags absent) | `dd7da1…71630` (**identical**) |
| + dead-knob wiring (flags at their pre-fix numeric defaults) | `cc1ff6…eddcc7` (differs — see §3) |

**The NLOS code path is bit-identical when its flags are absent**, verified
down to the SHA-256 level, not just matching solution counts. This also
independently reproduced the pre-existing `wp6_winner_jumprate_2.3.pos`
byte-for-byte (`sha256 a32cd878…70946647` both before and after the WP7
C++ changes, full run1), confirming both (a) the WSL build is itself
deterministic and (b) the codebase had zero incidental drift from the WP7
diff outside the intentionally-changed lines.

## 3. Dead-knob wiring (measured separately from NLOS, per the task's own instruction)

Replaced the hardcoded `2.0` at `rtk.cpp:2326`'s fix-and-hold ratio
relaxation with `rtk_config_.hold_ambiguity_ratio_threshold` (falls back to
`2.0` if the config value is non-finite or ≤ 0), and added
`rtk_ar_evaluation::passesArFilter(...)` as an extra AND-gate at both AR
acceptance call sites (full-set and subset-search). Both are no-ops at
their `RTKConfig` struct defaults (`enable_ar_filter=false`,
`hold_ambiguity_ratio_threshold=2.0`) — verified above.

**Surprise, root-caused during bit-identical verification**: `--preset
low-cost` (the WP6 winner's own base profile) has *always* set
`enable_ar_filter=true` / `ar_filter_margin=0.35` /
`hold_ratio_threshold=2.5` internally (`gnss_solve.cpp`'s
`applyRTKTuningPreset`) — this is exactly the WP6 "dead knob" finding:
these preset defaults, and WP6's own `--arfilter --arfilter-margin 0.35
--min-hold-count 8 --hold-ratio-threshold 2.6` "v5 baseline flags" probe,
were **silently discarded** by the old hardcoded-`2.0`/never-called-
`passesArFilter` code (`results/wp6/final/run1/baseline_log.txt` shows
`c0_bare_preset` and `c0_v5_baseline_flags` producing byte-identical
output). Now that both knobs are wired, re-running the *exact* WP6 winner
CLI (`--preset low-cost --max-pos-jump-rate 2.3`, no other flags) **no
longer reproduces the historical `.pos` byte-for-byte**, because the
preset's own long-dormant settings activate:

| run1, tokyo | fixed | AllRMS | FixRMS | fix% | `<50cm_full%` | ppc |
|---|---:|---:|---:|---:|---:|---:|
| `--no-arfilter --hold-ratio-threshold 2.0` (forces pre-fix values explicitly) | 1130 | 19.476 | 0.3114 | 15.10 | **26.92** | 24.19 |
| same CLI as WP6 winner, no override (dead knobs now active) | 1130 | 19.471 | 0.3128 | 15.10 | **26.64** | 23.65 |
| same knobs made explicit (`--arfilter --arfilter-margin 0.35 --min-hold-count 8 --hold-ratio-threshold 2.6`) | 1130 | 19.471 | 0.3128 | 15.10 | **26.64** | 23.65 |

The forced-off row is byte-identical (SHA-256-verified, §2) to
`results/wp6/final/run1/wp6_winner_jumprate_2.3.pos`. The dead-knob
activation itself is a **small, real regression on run1** in isolation
(−0.28pp `<50cm_full%`, −0.54pp ppc, same fixed count) — `--preset
low-cost`'s stricter subset-AR margin apparently trades a few marginal
fixes for none gained on this run. This is adopted as the WP7 baseline
(next section) because it is what `--preset low-cost` has *always claimed*
to do; masking it again would just re-introduce a second dead knob.

New tests: `tests/test_rtk_smoke.cpp` gained 4 cases
(`ArFilterDisabledByDefaultMatchesPreWiringBehavior`,
`ArFilterWithLargeMarginSuppressesFixesOnceEnabled`,
`HoldAmbiguityRatioThresholdIsHonoredNotHardcodedTwo`,
`HoldAmbiguityRatioThresholdDefaultMatchesLegacyHardcodedValue`) exercising
both wirings end-to-end against the bundled kinematic RTK fixture (in
addition to the existing pure-function tests in
`test_rtk_ar_evaluation.cpp`). These `GTEST_SKIP()` in this environment
because `third_party/gnssplusplus/data/{rover,base}_kinematic.obs` is not
present here — same pre-existing gap as 8 other `RTKSmokeTest`/
`RTKRealDataTest`/`SPPTest` cases (47 skips total, 0 failures, see §6); the
logic is validated on real PPC data instead (§4-§5, all `returncode: 0`).

## 4. Run1 evaluation: NLOS sigma-inflation sweep

Base config for every candidate below: `--preset low-cost
--max-pos-jump-rate 2.3` (WP6 winner, dead knobs now active — "WP7
baseline"), full run1 timeline (11928 rover epochs; WP6's own coarse-slice
lesson: only full-run numbers are trustworthy). Driver:
`experiments/sweep_libgnss_rtk_wp7.py` (extends
`sweep_libgnss_rtk_wp6.py`), sweep CSVs in `results/wp7/sweep/run1_stage{0,1,2}.csv`.

| candidate | fixed | coverage% | AllRMS | FixRMS | fix% | `<50cm_full%` | ppc |
|---|---:|---:|---:|---:|---:|---:|---:|
| **WP7 baseline (no NLOS)** | 1130 | 62.75 | 19.471 | 0.313 | 15.10 | **26.64** | 23.65 |
| two-tier, thr 0.5, ×2 | 1124 | 62.51 | 18.809 | 0.303 | 15.08 | 19.03 | 10.73 |
| two-tier, thr 0.5, ×3 | 1128 | 61.94 | 18.208 | 0.320 | 15.27 | 14.94 | 8.99 |
| two-tier, thr 0.5, ×5 | 1089 | 61.09 | 17.503 | 0.337 | 14.94 | 13.66 | 7.29 |
| two-tier, thr 0.5, ×10 | 1084 | 61.30 | 17.117 | 0.445 | 14.82 | 17.98 | 15.06 |
| two-tier, thr 0.5, ×20 | 1066 | 60.99 | 17.564 | 0.302 | 14.65 | 20.59 | 18.92 |
| continuous, floor 0.5 | 1128 | 61.83 | 19.076 | 0.311 | 15.29 | **22.13** | 13.15 |
| continuous, floor 0.2 | 1131 | 62.48 | 18.828 | 0.303 | 15.18 | 17.44 | 9.96 |
| continuous, floor 0.1 | 1126 | 61.99 | 18.099 | 0.319 | 15.23 | 14.70 | 8.76 |
| continuous, floor 0.05 | 1099 | 61.83 | 17.481 | 0.305 | 14.90 | 13.81 | 7.41 |
| continuous, floor 0.01 | 1084 | 61.30 | 17.117 | 0.445 | 14.82 | 17.98 | 15.06 |

**Every single one of the 10 tested points is worse than the no-NLOS
baseline on `<50cm_full%` and ppc** (and mostly on coverage too — fewer
epochs get *any* valid solution, not just fewer FIX epochs). `FixRMS`
stays under the 0.5 m budget throughout (worst case 0.445 m), so the
*regression is not disguised as a quality win* — it is a straightforward
loss on the task's primary metric. `continuous floor=0.5` (mildest
inflation tested, effective variance cap ×2) is the least-bad point and is
carried forward as "the best mapping" for §5's generalization step, per
the task's instruction to pick one — but it is **not a recommended
default**.

### Canyon segment deep-dive (tow 188990–189070)

Only 18-23 of the ~400 possible epochs in this 80 s window get *any* valid
solution under any config (severe coverage gap, matching WP6's own
characterization of this segment), and **none of them are FIX** in any
tested configuration (`canyon_fix_pct = 0.0` throughout) — this segment is
purely a FLOAT-quality question, exactly as WP6 framed it.

| config | epochs w/ any solution | canyon AllRMS (m) |
|---|---:|---:|
| WP7 baseline (no NLOS) | 18 | 125.61 |
| two-tier ×2 (mildest two-tier) | 23 (+5) | 122.69 (−2.9) |
| two-tier ×3/×5/×10/×20 | 23/19/19/19 | 122.69/125.22/125.22/125.22 |
| continuous floor=0.5 (selected "best") | 18 (+0) | 125.61 (+0.00) |
| continuous floor=0.2/0.1/0.05/0.01 | 23/23/19/19 | 122.69/122.69/125.22/125.22 |

**No config meaningfully reduces the float divergence in this segment.**
The mildest two-tier setting recovers 5 more epochs at a near-identical
error level; every other tested config leaves the ~119 m mean / 146 m max
error untouched. Mechanistically this makes sense: a pseudorange biased by
~100+ m of NLOS/multipath needs either exclusion or an enormous variance
inflation to be out-weighted in a weighted-least-squares/Kalman update —
the ×2 to ×20 (two-tier) / ×2 to ×100 (continuous) sigma multipliers tested
here are far too small next to a >100x-baseline-variance bias. **Sigma
inflation is the wrong tool for this specific failure mode**; see §7.

## 5. Generalization to run2/run3 (best mapping applied verbatim, no per-run tuning)

`continuous, floor=0.5` (the least-bad run1 candidate), same CLI on
run2/run3, no per-run tuning (`results/wp7/sweep/run{2,3}_stage3.csv`):

| run | config | fixed | coverage% | AllRMS | FixRMS | fix% | `<50cm_full%` | ppc | vs inuex35 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| run1 | WP7 baseline | 1130 | 62.75 | 19.471 | 0.313 | 15.10 | 26.64 | 23.65 | 56.7 |
| run1 | + NLOS | 1128 | 61.83 | 19.076 | 0.311 | 15.29 | **22.13 (−4.51pp)** | 13.15 | |
| run2 | WP7 baseline | 843 | 70.66 | 9.335 | 0.125 | 13.04 | 43.17 | 48.05 | 69.9 |
| run2 | + NLOS | 1362 | 70.59 | 9.106 | 0.241 | 21.08 | **45.68 (+2.51pp)** | 49.06 | |
| run3 | WP7 baseline | 666 | 84.15 | 5.637 | 0.081 | 5.17 | 43.71 | 40.67 | 67.9 |
| run3 | + NLOS | 669 | 84.02 | 5.655 | 0.106 | 5.20 | **37.31 (−6.40pp)** | 38.71 | |

**Does not generalize as a net win.** Run2 shows a genuinely large
improvement (+519 fixed solutions, +2.51pp `<50cm_full%`, FixRMS still well
under budget at 0.241 m) — but run1 and, especially, run3 regress (−4.51pp
and **−6.40pp** respectively). A single mapping picked on one run's coarse
sweep does not transfer; this is the same "coarse-slice doesn't generalize"
lesson WP6 already learned the hard way for AR-config knobs (stage 3/4
there), now recurring for the NLOS mapping. **Recommendation: do not ship
NLOS weighting as a default-on flag** in its current form — it is
correctly wired, fully optional/bit-identical when off, and available for
further research, but per-run sign flips rule it out as a blind, one-size
default.

## 6. Existing test suites

- **C++ (`third_party/gnssplusplus/build/tests/run_tests`, gtest — upgraded
  the environment's stale 1.8.0 apt package to 1.14.0 to get `GTEST_SKIP()`
  support, matching what the existing suite already assumed):
  `276 tests, 229 passed, 0 failed, 47 skipped`** (skips are all pre-existing
  missing-fixture-data gaps — `rover_kinematic.obs`/Odaiba/RTKLIB-reference
  files are not present in this checkout — not WP7 regressions; every
  `returncode: 0` run against real PPC data in §4/§5 is the actual
  validation for the new C++ code, per this repo's own established
  convention of "C++/config changes are validated by the score, not unit
  tests" for engine-level behavior, while the NLOS pure-function logic
  itself (14 cases in `test_nlos_weights.cpp`) and the dead-knob smoke
  tests do run/pass where fixtures exist).
- **Python** (`tests/test_sweep_libgnss_rtk_wp7.py`, new): **8 passed, 0
  failed** — covers the sweep driver's own argv-building (WP6-winner-base
  args ordering, `--nlos-weights` insertion/omission) and canyon-segment
  filtering logic, mirroring `test_sweep_libgnss_rtk_wp6.py`'s scope.

## 7. Next-bottleneck recommendation

1. **Sigma inflation cannot fix ~100+ m-biased NLOS pseudoranges** (§4's
   canyon deep-dive) — the mappings tested here (×2–×100 effective
   variance) are an order of magnitude too weak, and pushing them much
   higher starts costing coverage/AR pair-count instead (the two-tier ×5/×10
   sweep already shows this). The next experiment should try **hard
   exclusion** of `los_prob` below a strict threshold from the DD pair set
   entirely (not just downweighting) — losing a biased satellite's
   redundancy is likely cheaper than keeping it in with any finite weight.
2. **The float divergence itself may not be (only) NLOS/multipath** — since
   even aggressive NLOS downweighting leaves the canyon segment's ~119 m
   error untouched, WP6's own open question ("an undetected cycle slip the
   existing `gf_slip_count`/`doppler_slip_*` counters missed?") is still
   unanswered and should be investigated directly with
   `--debug-epoch-log` on this specific segment before spending more effort
   on NLOS-specific levers.
3. **The per-run sign flip (run2 win, run1/run3 loss) suggests satellite
   geometry/redundancy interacts with the NLOS mask non-uniformly** — a
   run-level covariate (e.g. mean visible-satellite count, canyon
   dwell-time fraction) might predict which runs benefit, which would be
   worth characterizing before trying a third mapping family.
4. Dead-knob wiring's own small regression (§3) suggests `--preset
   low-cost`'s `ar_filter_margin=0.35`/`hold_ratio_threshold=2.5` defaults
   were never actually tuned against real data (they were inert) — now
   that they work, they are a legitimate (if minor) re-tuning target,
   separate from NLOS.

## Deliverables

- `results/wp7/WP7_REPORT.md` (this file)
- `results/wp7/sweep/run1_stage{0,1,2}.csv`, `run{2,3}_stage3.csv` (18
  candidate rows total)
- `results/wp7/final/run{1,2,3}/*.pos` (all sweep candidates' trajectories)
- `results/wp7/final/run{1,2,3}/wp7_{baseline,best_nlos}_score.json`,
  `run1/wp7_deadknobs_off_score.json`
- `experiments/results/plateau_nlos_phase33/tokyo_run{1,2,3}_per_epoch_nlos.csv`
  (pre-existing NLOS masks used as-is, no regeneration needed)
- `experiments/sweep_libgnss_rtk_wp7.py` + `tests/test_sweep_libgnss_rtk_wp7.py` (8 tests, all passing)
- C++: `third_party/gnssplusplus/include/libgnss++/algorithms/nlos_weights.hpp`,
  `src/algorithms/nlos_weights.cpp`, `tests/test_nlos_weights.cpp` (new, 14
  tests); minimal flag-guarded diffs to `rtk.hpp`/`rtk.cpp`/`gnss_solve.cpp`/
  `tests/test_rtk_smoke.cpp` (+4 tests) / `CMakeLists.txt`×2
