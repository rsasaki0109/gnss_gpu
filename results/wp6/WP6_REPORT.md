# WP6 — Raising the libgnss++ RTK FIX rate on tokyo run1

Campaign doc: `internal_docs/inuex35_tc_fgo_benchmark.md` ("Campaign insight
(2026-07-06)"). Prior work: `results/wp5/WP5_REPORT.md` ("Bottleneck
analysis v2" / "WP6 recommendation" — RTK FIX coverage is far too sparse
(6.5%) and too front-loaded in time for WP5's anchoring to have any runway).

**Objective (task's own words): maximize `<50cm_full%` on tokyo run1
(baseline 25.4%), subject to FixRMS ≤ 0.5 m.**

## Headline result

| method | run | n_scored/n_rover | coverage% | AllRMS (m) | **FixRMS (m)** | fix% | `<50cm%` | **`<50cm_full%`** | ppc |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| inuex35 README (external) | run1 | n/a | 100.0 | 47.40 | 0.815 | 49.5 | 56.7 | **56.7** | n/a |
| **RTK-only baseline (v5, bare `--preset low-cost`)** | run1 | 7397/11928 | 62.0 | 19.86 | 0.084 | 10.5 | 40.9 | **25.4** | 22.88 |
| **WP6 winner (`--max-pos-jump-rate 2.3`)** | run1 | 7485/11928 | 62.8 | 19.48 | **0.311** | 15.1 | 42.9 | **26.92** | 24.19 |
| inuex35 README (external) | run2 | n/a | 100.0 | 32.08 | 0.277 | 60.8 | 69.9 | **69.9** | n/a |
| RTK-only baseline | run2 | 6466/9151 | 70.7 | 9.31 | 0.049 | 12.6 | 51.1 | **36.11** | 43.47 |
| WP6 winner (same config, no per-run tuning) | run2 | 6457/9151 | 70.6 | 9.34 | **0.254** | 13.1 | 61.1 | **43.13** | 48.01 |
| inuex35 README (external) | run3 | n/a | 100.0 | 34.52 | 0.211 | 59.4 | 67.9 | **67.9** | n/a |
| RTK-only baseline | run3 | 12833/15301 | 83.9 | 5.28 | 0.048 | 6.4 | 52.1 | **43.70** | 40.66 |
| WP6 winner (same config, no per-run tuning) | run3 | 12875/15301 | 84.1 | 5.64 | **0.081** | 5.2 | 52.0 | **43.72** | 40.69 |

**Result: run1 `<50cm_full%` 25.39% → 26.92% (+1.53 pp), run2 36.11% →
43.13% (+7.03 pp), run3 43.70% → 43.72% (flat, +0.02 pp) — all three stay
inside the FixRMS ≤ 0.5 m budget, and no run regresses.** The gain is real
but modest relative to inuex35's 56.7/69.9/67.9%; see "Next-bottleneck
recommendation" below for why a bigger lever than AR-config tuning is
needed to close that gap, and "Bottleneck analysis" for the specific,
diagnosed reason every *other* tested knob failed.

## Work item 1 — Provenance (found, reproduced)

`experiments/results/libgnss_rtk_pos_v5/tokyo_run1_full.pos` was produced by
two matching driver scripts:
`experiments/scripts/run_libgnss_rtk_tokyo_run1.sh` and
`experiments/run_libgnss_rtk_wsl.py`'s `TOKYO_PROFILE`, both invoking the
WSL-built `third_party/gnssplusplus/build/apps/gnss_solve` as:

```bash
gnss_solve --rover rover.obs --base base.obs --nav base.nav --skip-epochs 0 \
  --out tokyo_run1_full.pos --no-kml --preset low-cost \
  --arfilter --arfilter-margin 0.35 --min-hold-count 8 --hold-ratio-threshold 2.6
```

Reproduced verbatim (`results/wp6/baseline_repro/tokyo_run1_full.pos`, 206–283 s
wall): engine summary 775/7397 fixed (10.48%); re-scored with
`score_vs_inuex35.py` → `n_scored=7397/11928 coverage=62.0% AllRMS=19.861
FixRMS=0.084 fix%=10.5 <50cm%=40.9 <50cm_full%=25.4 ppc=22.88%` — an exact
match to WP5's/the campaign doc's baseline row. **Baseline locked in.**

Control experiment (`stage0`, `results/wp6/WP6_sweep_table_all.csv` rows
`c0_bare_preset` / `c0_v5_baseline_flags`): bare `--preset low-cost` and the
v5 baseline's exact extra flags produce a **byte-identical** engine summary
and score on the full run — see "3 of 4 v5 flags are dead knobs" below for
why.

## Work item 2 — Knob inventory (file:line citations)

Enumerated via `gnss_solve --help` (~100 AR/fix/guard flags) cross-referenced
against `apps/gnss_solve.cpp` (CLI parsing + app-level post-processing),
`src/algorithms/rtk.cpp`/`rtk_validation.cpp` (core RTK/AR logic), and
`include/libgnss++/algorithms/rtk.hpp` (`RTKConfig` struct).

| Knob | CLI parse | Config field | Actually used? | Where |
|---|---|---|---|---|
| `--ratio` (AR ratio threshold) | `gnss_solve.cpp:1205` | `RTKConfig::ratio_threshold` | ✅ real | LAMBDA acceptance test, `rtk.cpp` |
| `--min-hold-count` | `gnss_solve.cpp:1240` | `min_hold_count` | ✅ real | `canAttemptHoldFix`, `rtk.cpp:1863`-area |
| `--hold-ratio-threshold` | `gnss_solve.cpp:1243` | `hold_ambiguity_ratio_threshold` (`rtk.hpp:59`) | ❌ **dead** | assigned in `gnss_solve.cpp:1823` but never read anywhere in `src/`; the real fix-and-hold ratio is a **hardcoded literal `2.0`** at `rtk.cpp:2326`, independent of this config |
| `--arfilter` / `--no-arfilter` / `--arfilter-margin` | ~`gnss_solve.cpp:1210` | `enable_ar_filter`/`ar_filter_margin` (`rtk.hpp:54-55`) | ❌ **dead** | the gate they should drive, `rtk_ar_evaluation::passesArFilter` (`rtk_ar_evaluation.cpp:16-26`, fully implemented + unit-tested), is only ever called from its own test file — the RTK subset-AR path in `rtk.cpp` never invokes it |
| `--min-ar-sats` | `gnss_solve.cpp:1219` | `min_ar_sats` | ⚠️ **narrow** | only wired in the GLONASS-autocal branch (`rtk.cpp:1770-1772`); the general path's satellite-count gate is a **hardcoded `nb < 4`** DD-pair-count floor at `rtk.cpp:2242`/`2307`/`3492`, independent of this setting |
| `--elevation-mask-deg` | `gnss_solve.cpp:1246` | `elevation_mask` | ✅ real | satellite obs filter, `rtk.cpp:627` |
| `--glonass-ar {off,on,autocal}` | `gnss_solve.cpp:1293` | `glonass_ar_mode` | ✅ real | `usesGlonassAutocal` (`rtk.cpp:53`) gates `REAL_STATES` (extra inter-channel-bias filter states) vs `BASE_STATES` (`rtk.cpp:2231`,`3481`); `autocal` calibrates FDMA ICB online, `on` does not (empirically catastrophic — see stage1 below) |
| `--max-consec-float-reset` / `--max-consec-nonfix-reset` | `gnss_solve.cpp:1383/1385` | `max_consecutive_float_for_reset`/`max_consecutive_nonfix_for_reset` | ✅ real | ambiguity-state reset gate, `rtk.cpp:1268-1411` |
| `--max-pos-jump` | `gnss_solve.cpp:1310` | `max_position_jump_m` | ✅ real | rejects any new fix whose position differs from `last_fixed_position_` by more than the limit (`rtk_validation::exceedsAbsoluteJump`, `rtk.cpp:3171-3186`); default 5.0 m |
| `--max-pos-jump-min` / **`--max-pos-jump-rate`** | `gnss_solve.cpp:1312/1314` | `max_position_jump_min_m`/`max_position_jump_rate_mps` | ✅ real (**the WP6 lever**) | `adaptiveJumpLimit(dt, min, rate) = max(min, rate·dt_since_last_fix)` (`rtk_validation.cpp:38-40`); `position_jump_limit = max(max_pos_jump_m, adaptive_limit)` (`rtk.cpp:3172-3184`) — grows the allowed jump with elapsed staleness instead of a flat cap |
| (unnamed, no CLI flag) fixed-vs-float jump gate | — | — | ✅ real, **hardcoded 20.0 m** | `rtk.cpp:1816-1839`: rejects a new fix if it differs from the *same-epoch* FLOAT solution by >20 m (`reject_reason="fixed_float_jump"`) — a different, always-on check from `--max-pos-jump`'s cross-epoch comparison |
| (unnamed, no CLI flag) fix-history jump gate | — | — | ✅ real, **hardcoded 0.1/0.2 m** | `rtk_validation::exceedsFixHistoryJump` (`rtk_validation.cpp:61-74`): only engages once `consecutive_fix_count_ >= 3`, static/kinematic threshold not configurable |
| `--demote-fixed-status-{max-ratio,nis-per-obs,post-rms,gate-ratio}`, `--{min,max}-demote-fixed-status-baseline` | `gnss_solve.cpp:1352-1363` | app-level `SolveConfig` fields (not `RTKConfig`) | ✅ real, but **app-level relabel only** | `shouldDemoteFixedStatus` (`gnss_solve.cpp:239-290`): rewrites the *output* Status column from FIX→FLOAT post-hoc based on ratio/NIS/post-fit-residual/baseline-length quality gates — never touches the actual ECEF position |

**No missing AR capability was flagged as needing implementation** (task
item 2's "if a capability is missing, note it — do not implement C++
features" clause): fix-and-hold, partial/subset AR, per-constellation AR
gating (GLONASS autocal), elevation mask, and cycle-slip detection
(`gf_slip_count`/`doppler_slip_*`/`lli_slip_*` in the debug telemetry) all
exist and are wired; the gap is that **3 of the v5 baseline's 4
"tuned" flags are dead knobs** (`--arfilter`, `--arfilter-margin`,
`--hold-ratio-threshold`), and a **base interpolation** flag was not
investigated (out of scope once the real bottleneck was found — see below).

## Work item 3 — Staged sweep

### Stage 1–4: coarse slice, then full-run reality check (a documented negative result)

Per the task's "if a run takes >5 min, use a representative 1/3 slice
including mid-run urban canyon" instruction (full runs took 200–410 s here,
under the 5-min trigger but slice-sweeping was still faster for a wide
grid): ran a 15-candidate coarse grid (`--ratio`×{2.4,2.6,2.8,3.0},
`--min-hold-count`×{3,5,8}, `--max-consec-{float,nonfix}-reset`×{5,10},
`--glonass-ar`×{on,autocal}, `--elevation-mask-deg`×{10,20}) on a
representative slice (epochs 4000–8000, spanning two urban-canyon segments
flagged by prior work). Single standout: **`--glonass-ar autocal`** lifted
the slice from 1→51 fixed at FixRMS 0.478 m; plain `on` (no inter-channel
bias calibration) gave only 3 fixed at FixRMS 29.7 m (confirms FDMA ICB
*must* be autocal'd). Every other knob was a no-op on this slice — later
explained by the slice landing in a segment gated by the hardcoded `nb<4`
floor, independent of ratio/hold-count/min-ar-sats. Refined
`--glonass-ar autocal` combos (stage2); best was `--glonass-ar autocal
--ratio 2.8` (56 fixed, FixRMS 0.473 m — best on this slice).

**Stage 3 generalization check on a second, independent slice (epochs
8000–11928) — critical, and it failed:** the "winner" performed *worse*
than baseline there (`<50cm_full%` 1.65% vs baseline's 6.77%). **Stage 4
full-run confirmation: `--glonass-ar autocal` REGRESSES the full run** (278
fixed vs baseline's 775, `<50cm_full%` 14.6% vs 25.4%) — adding GLONASS to
the AR-eligible pool destabilizes far more of the easy front-loaded fixes
than it recovers in hard segments. **This is an explicit, honest negative
result and a methodology lesson: the coarse-slice stage actively misled
here** — every full-run decision from stage 5 onward was made only on full
run1 evidence.

### Stage 5: every wired knob, full run — all no-ops or net-negative

`--ratio 2.4`/`2.6` alone: 777/776 fixed vs baseline's 775 (noise;
contradicts the engine's own `docs/benchmarks.md` claim of 54.4% fix on
tokyo run1 from `--ratio 2.4` alone — does not reproduce on this
build/commit). `--max-consec-{float,nonfix}-reset 10`: identical 775 fixed
but ~1400 fewer scored epochs (drops `<50cm_full%` to ~13.4%).
`--elevation-mask-deg 10`: 775 fixed, 24.97% vs 25.39% (noise). **Every
wired knob tested at this stage was either a no-op or net-negative.**

### Root-cause diagnosis: `--debug-epoch-log`

`gnss_solve --debug-epoch-log` emits a ~55-column per-epoch AR/validation
telemetry CSV. On the baseline full run1: of 8247 AR-attempted epochs, 775
got fixed and 7475 did not — but **4437 of those not-fixed epochs already
had a LAMBDA-resolved candidate with `full_ratio ≥ 3.0`** (AR itself
succeeded) that was still not applied. **4409/4437 (99.4%) carry
`reject_reason == "max_position_jump"`**, and **3874/4437 (87%) of those
land after the first 300 s of the run.** Mechanism: once a fix streak
breaks, `last_fixed_position_` (the reference `--max-pos-jump` compares
against) goes stale — from wherever the vehicle was during its last fix,
potentially far away/long ago in the urban canyon — so nearly every later
well-resolved fix candidate gets vetoed for "jumping" away from that stale
reference, **permanently locking the pipeline out of re-fixing for the rest
of the run.** `--ratio` tuning (stage 5) could never have found this: AR
itself was already succeeding; the veto is strictly downstream.

### Stage 6–7: a flat disable overshoots, and post-hoc demotion cannot rescue it

`--max-pos-jump 0` (disabled): 775→4200 fixed, `<50cm_full%` 25.4%→31.7%,
but **FixRMS blows up to 14.38 m** (budget ≤0.5 m). `--max-pos-jump
{15,30}`: full no-ops (identical to baseline) — the stale-reference jumps
this run needs to clear are ≫30 m. Root cause of the blowup (new offline
merge of `--debug-epoch-log` telemetry against per-epoch truth error,
`results/wp6/sweep/merged_fixed_analysis.csv`): of 4200 fixed epochs, 724
are ≥0.5 m error (mean 15.1 m), and a small tail of ~20–30 catastrophic
epochs (up to 170 m, clustered in one ~80 s urban-canyon segment,
tow≈188990–189070) dominate the RMS. **These catastrophic epochs are
internally self-consistent**: `fixed_float_jump_m ≈ 0` (the fixed solution
agrees with the FLOAT filter — the FLOAT state itself was already badly
diverged, not just the AR step) with plausible-looking `selected_ratio`
(up to 30) and pair count (5–13). **No AR-quality metric distinguishes them
from good fixes after the fact.**

Confirmed empirically: swept `--demote-fixed-status-{post-rms,nis-per-obs,
max-ratio}` on top of `--max-pos-jump 0` (post-rms∈{0.02,0.03,0.05,0.5,
1.0,2.0}, nis-per-obs∈{1,2,5}, max-ratio=5, and a post-rms+nis combo) — best
FixRMS achieved was still **12.9 m** (nis-per-obs=1, keeping 2529/4200 as
FIX); tight post-rms thresholds (0.02/0.03) demoted nearly everything (kept
only 20/4200) yet those 20 survivors still averaged **FixRMS=125 m**. Post-
hoc demotion is the wrong tool for a float-filter-divergence bug (all 11
stage-7 rows are in `results/wp6/WP6_sweep_table_all.csv` for the record).

### Stage 8: the adaptive jump gate — the actual winner

Pivoted to `--max-pos-jump-rate <m/s>` (already wired, see knob table):
grows the allowed jump with elapsed staleness instead of a flat disable —
keeps the baseline's tight 5 m gate for fresh streaks, permits recovery
after a long gap only up to a physically-plausible displacement for the
given `dt`. Bisected on the full run:

| `--max-pos-jump-rate` | fixed | FixRMS (m) | `<50cm_full%` | verdict |
|---:|---:|---:|---:|---|
| 1.0 | 815 | 0.079 | 25.44 | safe, flat |
| 2.0 | 819 | 0.063 | 25.49 | safe, flat |
| **2.3** | **1130** | **0.311** | **26.92** | **safe, real gain — WINNER** |
| 2.35 | 1257 | 8.76 | 26.79 | budget-blown |
| 2.4 | 1391 | 9.12 | 26.90 | budget-blown |
| 2.45 | 1491 | 9.05 | 26.76 | budget-blown |
| 2.5 | 1407 | 9.56 | 26.91 | budget-blown |
| 2.7 | 1131 | 11.40 | 21.90 | budget-blown |
| 3.0 | 1868 | 7.52 | 28.24 | budget-blown |
| 4.0 | 2134 | 13.57 | 23.47 | budget-blown |
| 5.0 | 2566 | 12.45 | 26.62 | budget-blown |
| 10.0 | 4017 | 13.18 | 32.72 | budget-blown |
| 20.0 | 4498 | 8.76 | 34.02 | budget-blown |
| 30.0 | 4511 | 8.82 | 34.07 | budget-blown |

**The safe/unsafe boundary is razor-thin** (2.3 safe, 2.35 already
blown) — the same catastrophic float-divergence segment identified in
stage 6 gets admitted the moment the jump budget is generous enough to
reach it, and once admitted it dominates RMS regardless of how many other
(good) epochs also get admitted. Combining `--max-pos-jump-rate 5/10` with
`--demote-fixed-status-post-rms 1.0` did not help (FixRMS unchanged at
12.4/13.4 m) — confirms stage 7's finding that this specific failure mode
is undetectable by internal quality metrics, at any jump-rate setting.

**Winner: `--max-pos-jump-rate 2.3`** (bare `--preset low-cost` otherwise).
Full sweep table (78 candidate rows, every stage): `results/wp6/WP6_sweep_table_all.csv`.

## Work item 4 — Best config: full run1 + run2/run3 generalization

Saved to `results/wp6/final/run{1,2,3}/`:
`wp6_winner_jumprate_2.3.pos` + `wp6_winner_score.json` (winner),
`c0_bare_preset.pos` + `baseline_score.json` (baseline, for the same-run
comparison), `sweep_table.csv` (per-run engine+score summary).

| run | metric | baseline | winner | Δ |
|---|---|---:|---:|---:|
| run1 | `<50cm_full%` | 25.39 | 26.92 | **+1.53 pp** |
| run1 | FixRMS (m) | 0.084 | 0.311 | within budget |
| run2 | `<50cm_full%` | 36.11 | 43.13 | **+7.03 pp** |
| run2 | FixRMS (m) | 0.049 | 0.254 | within budget |
| run3 | `<50cm_full%` | 43.70 | 43.72 | +0.02 pp (flat) |
| run3 | FixRMS (m) | 0.048 | 0.081 | within budget |

**No per-run tuning** — the exact same `--max-pos-jump-rate 2.3` flag was
used on all three runs. Generalizes positively on run1/run2, is flat (not
negative) on run3. Run3's baseline already has far higher natural fix
coverage (43.7% vs run1's 25.4%) — the front-loaded/stale-reference problem
this lever fixes is inherently less severe there, so there is little
headroom left for this specific lever on that run; it did not regress it
either.

(Note: while assembling these results, a same-out-dir/same-filename race
was caught — two backgrounded runs of differing epoch counts finished out
of launch order and silently overwrote each other's `.pos` file. Re-ran
into per-run subdirectories and verified each file's TOW range matches its
run before scoring; flagging here since it is exactly the kind of quiet
correctness bug this task's own "score every candidate" discipline is
designed to catch.)

## Work item 5 — WP5-compounding check (stretch)

Fed `results/wp6/final/run1/wp6_winner_jumprate_2.3.pos` into
`experiments/wp5_run_anchored_fgo.py --rtk-pos <that file>` (unchanged
driver, per task constraint) on the full run1, 200-epoch windows:

```bash
set PYTHONPATH=python
set PYTHONUNBUFFERED=1
python -u experiments/wp5_run_anchored_fgo.py --window-epochs 200 \
  --rtk-pos results/wp6/final/run1/wp6_winner_jumprate_2.3.pos \
  --out-dir results/wp6/wp5_compounding \
  > results/wp6/wp5_compounding/full_sweep_run.log
```

| method | run | n_scored/n_rover | coverage% | AllRMS (m) | FixRMS (m) | fix% | `<50cm%` | `<50cm_full%` |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| WP5 original (anchored on v5 baseline `.pos`, from WP5_REPORT.md) | run1 | 11923/11928 | 100.0 | 94.05 | 58.91 | 74.3 | 24.5 | 24.5 |
| **WP5 driver anchored on WP6 winner `.pos` (this stretch check)** | run1 | 11923/11928 | 100.0 | 93.70 | 58.23 | 74.5 | 25.6 | **25.6** |
| *(for reference)* WP6 RTK-only winner, no FGO | run1 | 7485/11928 | 62.8 | 19.48 | 0.31 | 15.1 | 42.9 | **26.92** |

**Compounding gives a small further gain over WP5's original anchored run
(24.5%→25.6%, +1.1 pp — consistent with feeding it a modestly better/wider
FIX supply: 1130 vs 775 FIX epochs), but the anchored-FGO pipeline still
ends up *below* just using the WP6 winner's raw RTK `.pos` directly
(25.6% vs 26.92%).** `extension_stats.json` for this run: 1130 FIX-anchored
epochs, 6355 FLOAT-anchored, 4438 with no anchor at all (endpoint-only);
3050 epochs pass `<50cm` after FGO vs 1939 of those specifically *because*
of extension from an anchor (`n_extension_pass_lt_50cm`) — real signal, but
still not enough to beat the plain RTK output. This reinforces WP5's own
"Bottleneck analysis v2": the anchored-FGO machinery's ceiling is bounded
by FIX coverage/spread more than by anything in the FGO/LAMBDA code itself,
and WP6's fix-supply gain (though real) was too small to flip that
conclusion. Artifacts: `results/wp6/wp5_compounding/` (note: the driver's
`--out-csv`/`--stats-json`/`--per-segment-csv` default into `results/wp5/`
regardless of `--out-dir` — this run's outputs were moved out to
`results/wp6/wp5_compounding/*_wp6winner.*` immediately after completion,
and `results/wp5/`'s original WP5 artifacts were regenerated with the
driver's own defaults to undo the accidental overwrite; see PROGRESS.md).

## Next-bottleneck recommendation

Closing the remaining gap to inuex35 (56.7/69.9/67.9%) needs a bigger lever
than AR-config tuning:

1. **The float filter itself diverges in specific urban-canyon segments**
   (stage 6/7's root cause) — this is upstream of AR entirely (the FIX
   agreed with an already-wrong FLOAT). No `RTKConfig` knob touches the
   float KF's own robustness (outlier rejection, process noise, multipath
   weighting). This is the single highest-leverage next investigation:
   instrument *why* the float filter diverges at tow≈188990–189070
   specifically (satellite geometry change? multipath? an undetected cycle
   slip that the existing `gf_slip_count`/`doppler_slip_*` counters missed?).
2. **The reset/coverage tax is real and unaddressed**: every
   `--max-consec-*-reset` setting traded ~25–27% of coverage for a handful
   of clean fixes — a smarter reset (e.g., re-seed from the *current*
   FLOAT rather than discarding to raw code-only) could recover that
   coverage without the quality cost.
3. **`--min-ar-sats` is a dead end outside the GLONASS-autocal branch** — the
   hardcoded `nb<4` floor (`rtk.cpp:2242`) is the real general-path gate.
   If more AR opportunities in low-satellite-count canyon segments are
   wanted, this would need a genuine (out-of-scope-for-this-task) C++
   change, not a config knob.
4. Per WP5's own finding, **anchoring quality still depends on fix
   *spread*, not just count** — `frac_fix_after_warmup` for the winner is
   28.3% (vs baseline's 0%), a real improvement, but still low; the
   adaptive-jump-rate lever's razor-thin safe margin (2.3 vs 2.35) means
   there is little room to push this further without a genuine float-filter
   fix (point 1).

## Deliverables

- `results/wp6/WP6_REPORT.md` (this file)
- `results/wp6/WP6_sweep_table_all.csv` (78 candidate rows, all stages)
- `results/wp6/final/run{1,2,3}/wp6_winner_jumprate_2.3.pos` + `wp6_winner_score.json`
- `results/wp6/final/run{1,2,3}/c0_bare_preset.pos` + `baseline_score.json`
- `results/wp6/baseline_repro/tokyo_run1_full.pos` + `score_baseline.json` (v5 reproduction)
- `results/wp6/sweep/merged_fixed_analysis.csv` (per-epoch AR-telemetry-vs-truth-error merge used for the stage 6/7 root-cause diagnosis)
- `experiments/sweep_libgnss_rtk_wp6.py` + `tests/test_sweep_libgnss_rtk_wp6.py` (13 tests, all passing)
- `results/wp6/wp5_compounding/` (stretch goal artifacts)

## Constraints honored

No git commits. `python/gnss_gpu/local_fgo.py`, `experiments/validate_fgo_ppc.py`,
`python/gnss_gpu/io/ppc.py` not modified. No libgnss++ C++ rebuild was
needed — every candidate in this sweep is a config-only CLI variant of the
already-built `gnss_solve` binary.
