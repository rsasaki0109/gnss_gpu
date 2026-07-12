# WP10 — Lapse-gated trust policy + min-LOS-sats AR gate — tokyo run1-3

Workspace: `C:\Users\rsasa\Workspace\old\gnss_gpu`. Baseline = WP7/dead-knob baseline: `--preset low-cost --max-pos-jump-rate 2.3` (run1 26.643 / run2 43.165 / run3 43.710 `<50cm_full%`; reproduction artifact `results/wp8/verify/wp8_absentflag_check_run1.pos`, SHA `4c2effb6…`).

**Headline result: negative on the mandatory 3-run regression gate, but a fully code-verified, mechanism-quantified negative — the second in a row (after WP9) on this exact lever.** The lapse-gated policy is implemented, unit-tested, and bit-identical to legacy when absent. It reproduces WP9's canyon win essentially unchanged (AllRMS 125.6→74.7 m) and, unlike unconditional `scaled-reset`, is gated on lapse duration so short/benign lapses can in principle stay legacy — but **no single gate value clears the mandatory regression gate on all three runs simultaneously**: `gate=2` s is the only tested value that gives run1 its required ≥+0.5 pp win (+1.065 pp), but it overshoots run3's −0.3 pp budget by 2× (−0.602 pp); the only gate that clears run3 (`gate=20` s) gives run1 only +0.176 pp, far short of the bar. The optional NLOS-fraction trigger (work item 2) was built and evaluated per the task's own contingency question — it does **not** discriminate better: at both tested thresholds it never once fires during run1's canyon (canyon AllRMS is byte-for-byte unchanged from baseline), so it contributes zero win where the plan needs one, while still costing up to −0.34 pp on run2. `--nlos-min-los-sats` (work item 5) is implemented and is a clean, sizeable loss on run1 (−4.3 to −4.5 pp) at both tested `N`, so it does not carry forward either. Root cause (§5): duration-based (and NLOS-fraction-based) triggering cannot separate "helpful" from "harmful" lapses because all three runs have a similar *population* of multi-second-to-~2-minute intermittent-tracking gaps (run1: 41 segments affected by `gate=2`, run2: 32, run3: 42) — run1's happen to net positive, run2/3's net negative, and neither duration nor the tested NLOS-fraction threshold correlates with which is which.

## 1. Policy design (work items 1, 2, 5)

**Work item 1 — `--float-trust-policy lapse-gated`** (new `FloatTrustPolicy::LAPSE_GATED`): tracks the continuous lapse via the pre-existing `dt_since_trust = rover_obs.time - last_trusted_time_` clock (already reset to ~0 every time `rememberSolution()` refreshes trust — no new counter needed). New pure function `float_trust_policy::lapseGateExceeded(dt_since_trust_s, gate_s)` (inclusive boundary `dt >= gate`, defensive clamping of negative/non-finite inputs) decides the branch in `resetPositionToSPP()`:

- **below `--trust-lapse-gate-s`** (default 5.0): `wp9_seeded` is left `false`, so the epoch falls through to the **unmodified legacy branch** — bit-identical by construction, not just numerically close.
- **at/above the gate**: applies WP9's `scaledResetPositionVariance(25, qpos, dt_since_trust, 900)` law verbatim (`--trust-lapse-qpos`, default 0.1, "the WP9 run1 winner").

7 new `FloatTrustPolicyTest` cases: gate boundary (inclusive), gate=0 degenerate case (always-scaled endpoint), huge-gate never-triggers case, defensive clamping, composition with `scaledResetPositionVariance`.

**Work item 2 — optional second trigger, `--trust-lapse-gate-nlos-frac F`** (default off, own flag): built (the plumbing was cheap, ~15 lines, well under the task's 50-line bar reusing WP9's existing `current_epoch_nlos_fraction_` per-epoch cache). ALSO switches to `scaled-reset`, regardless of lapse duration, whenever the epoch's NLOS-flagged-satellite fraction exceeds `F` (`--nlos-weights` required). Documented caveat: since `resetPositionToSPP()` runs *before* `collectSatelliteData()` computes this epoch's own fraction, the value read is the *previous* epoch's (one-epoch lag) — a deliberate, cheap simplification, immaterial for the multi-second-to-minute NLOS-heavy dwells this trigger targets.

**Work item 5 — `--nlos-min-los-sats N`** (AR-acceptance gate, not float-filter): new pure function `nlos_weights::nlosMinLosSatsGateAllows(los_sat_count, min_los_sats)` (`N<=0` disables; 3 new unit tests) + a gate inserted at the top of `RTKProcessor::resolveAmbiguities(dd_pairs)` that counts unique LOS-flagged satellites across the AR candidate set's DD pairs and vetoes the whole AR attempt (new `ARSkipReason::TOO_FEW_LOS_SATS`) if too few. Deliberately **not** touching `buildSelectionSnapshot()` (the float filter's own satellite source) — the float update is provably unaffected either way, matching the task's "gates AR only" requirement.

All three levers default off/inert; a healthy/converged run touches none of this new code.

**Bit-identical verification** (absent-flag constraint): reran the exact WP7/8/9-baseline full-run1 command (no WP10 flags) and separately `--float-trust-policy lapse-gated --trust-lapse-gate-s 1000000` (a gate far larger than any real lapse in the data), SHA-256'd both full 11928-epoch `.pos` files against the canonical `results/wp8/verify/wp8_absentflag_check_run1.pos`:

```
4c2effb637672cd29a6ae79d3a1a065130c292d95a92d54d9fe02f964b7ad8aa  results/wp10/verify/z0_baseline_no_wp10.pos
4c2effb637672cd29a6ae79d3a1a065130c292d95a92d54d9fe02f964b7ad8aa  results/wp10/verify/z1_lapsegated_hugegate.pos
4c2effb637672cd29a6ae79d3a1a065130c292d95a92d54d9fe02f964b7ad8aa  results/wp8/verify/wp8_absentflag_check_run1.pos
```

All three identical. Both the absent-flag path and the huge-gate path of the new policy are provably no-ops relative to legacy.

## 2. Run1 gate sweep (work item 3)

`experiments/sweep_libgnss_rtk_wp10.py --stage gate_sweep` (`gate ∈ {2,5,10,20}` s, `qpos=0.1`) plus a fine-grained follow-up (`gate ∈ {3,4}`) once 2 vs 5 bracketed the +0.5 pp bar. Full rows in `results/wp10/sweep/run1_gate_sweep.csv` / `run1_gate_fine.csv`.

| gate (s) | fixed | coverage% | AllRMS | FixRMS | fix% | `<50cm_full%` | Δ vs baseline (pp) | ppc | canyon AllRMS |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline (legacy) | 1130 | 62.75 | 19.471 | 0.3128 | 15.10 | 26.643 | — | 23.65 | 125.61 |
| **2** | 1130 | 61.75 | 19.152 | 0.3128 | 15.34 | **27.708** | **+1.065** | 27.71 | 74.73 |
| 3 | 1130 | — | — | 0.3128 | — | 27.104 | +0.461 | — | — |
| 4 | 1130 | — | — | 0.3127 | — | 27.138 | +0.495 | — | — |
| 5 | 1130 | 61.76 | 19.243 | 0.3127 | 15.34 | 27.129 | +0.486 | 27.13 | 71.03 |
| 10 | 1130 | 62.42 | 19.374 | 0.3128 | 15.18 | 26.878 | +0.235 | 23.65 | 74.73 |
| 20 | 1130 | 62.42 | 19.212 | 0.3128 | 15.18 | 26.819 | +0.176 | 23.74 | 70.30 |

Every tested gate value re-triggers on the canyon's ~92 s lapse (canyon AllRMS drops from 125.6 to 70–75 m at every gate, reproducing WP9's unconditional-`scaled-reset` canyon win of 74.7 m essentially unchanged) — as the task predicted. The sweep is really about how much of run1's *other* short-lapse population stays gated: smaller gates capture more of that population as "long enough," and **run1's gain is a smooth, monotonically-decreasing function of gate value** (2→1.065pp, 3→0.461, 4→0.495 non-monotonic blip within noise, 5→0.486, 10→0.235, 20→0.176). Only `gate=2` clears the task's own +0.5 pp bar; `gate=3/4/5` all fall just short. FixRMS is flat at ≈0.313 m (no wrong-fix effect) across the whole grid.

## 3. Regression matrix (work item 4) — **fails on every tested gate**

`experiments/sweep_libgnss_rtk_wp10.py --stage regression --best-gate {2,5,10,20}`, each gate applied verbatim on run2 and run3 (full rows in `results/wp10/sweep/run{2,3}_regression*.csv`).

| gate (s) | run1 `<50cm_full%` Δpp | run2 `<50cm_full%` Δpp | run2 within ±0.3pp? | run3 `<50cm_full%` Δpp | run3 within ±0.3pp? | FixRMS ≤0.5m everywhere? | **all 3 criteria pass?** |
|---:|---:|---:|:---:|---:|:---:|:---:|:---:|
| 2 | **+1.065** (win) | −0.274 | yes | **−0.602** | **no** | yes | **NO** |
| 5 | +0.486 (short of +0.5) | +0.065 | yes | −0.530 | **no** | yes | **NO** |
| 10 | +0.235 (short) | not tested | — | −0.438 | **no** | yes | **NO** |
| 20 | +0.176 (short) | −0.099 | yes | **−0.268** | **yes** | yes | **NO** (run1 short) |

(run2 exact values: baseline 43.165; gate2 42.891; gate5 43.230; gate20 43.066. run3 exact values: baseline 43.710; gate2 43.108; gate5 43.180; gate10 43.272; gate20 43.442. FixRMS stays in 0.081–0.313 m across every candidate on every run — never a wrong-fix-quality problem.)

**No gate value satisfies all three mandatory criteria** (run1 ≥+0.5pp, run2/3 within ±0.3pp, FixRMS≤0.5m). There is a clean crossover: as gate grows, run1's gain and run3's regression both shrink monotonically toward zero, but they cross zero at different points — run1 needs `gate≤~2` s to clear +0.5pp; run3 needs `gate≥~20` s to clear −0.3pp. No overlap exists in the tested grid (or, by the monotonic trend, in the gap between 2 and 20 either — `gate=5/10` already demonstrate the trend continues smoothly through the gap without a rescuing plateau).

### 3.1 Which segments triggered? (task's explicit ask)

Diffed each run's `gate=2` `.pos` file against its own baseline (>5 cm 3D displacement = "the gate visibly changed this epoch's estimate"):

| run | epochs compared | epochs affected (>5cm) | affected % | # contiguous segments | longest segment |
|---|---:|---:|---:|---:|---|
| run1 (win, +1.065pp) | 7192 | 2496 | 34.7% | 41 | 130.6 s / 640 epochs (tow 188680–188811) |
| run2 (marginal pass, −0.274pp) | 6417 | 1639 | 25.5% | 32 | 63.0 s / 183 epochs (tow 178067–178130) |
| run3 (fails, −0.602pp) | 12817 | 2667 | 20.8% | 42 | 86.0 s / 416 epochs (tow 181080–181166) |

All three runs have a broadly similar *population* of gate-triggered segments — dozens of contiguous multi-second-to-~2-minute windows, not a single isolated event. Notably, run1's single largest triggered segment (tow 188680–188811, 130.6 s) is **not** the WP8/WP9-identified "canyon" window itself (tow 188925–189075) — it's an approach corridor of several back-to-back medium lapses (tow ≈188386–188811, three segments totaling ~350 s) immediately before the canyon's own deep dropout, which only shows a partial, much smaller trigger (tow 188928–188950, 20 epochs) inside its nominal window. Run3's top segments (86 s, 77 s, 67 s, 61 s, 54 s — five segments over 50 s each) are the same order of magnitude and shape as run1's winning ones. **This is the direct evidence for the root cause in §5: duration alone cannot tell these apart — run3 simply has its own population of comparably-long intermittent-tracking gaps, and the same law that helps run1's population net-hurts run3's.**

### 3.2 Does item 2's NLOS-fraction trigger discriminate better?

Per the task's explicit contingency ("report ... whether item 2's NLOS trigger discriminates better"), tested the trigger in isolation: `--trust-lapse-gate-s 1000000` (duration branch unreachable) + `--trust-lapse-gate-nlos-frac F`, `F ∈ {0.5, 0.3}` (0.5 matches the existing `--trust-gate-nlos-relax` precedent in this codebase), full runs (`results/wp10/sweep/run{1,2,3}_nlos_frac_sweep.csv`):

| run | F=0.5 `<50cm_full%` Δpp | F=0.5 canyon AllRMS | F=0.3 `<50cm_full%` Δpp | F=0.3 canyon AllRMS |
|---|---:|---:|---:|---:|
| run1 (baseline 26.643, canyon 125.61) | **+0.000** | **125.61 (unchanged!)** | +0.067 | **125.61 (unchanged!)** |
| run2 (baseline 43.165) | −0.241 (within budget) | n/a | **−0.339 (fails budget)** | n/a |
| run3 (baseline 43.710) | −0.007 (passes trivially) | n/a | −0.033 (passes trivially) | n/a |

**Verdict: the NLOS-fraction trigger does not discriminate better — it is strictly worse than the plain duration gate for this purpose.** At both thresholds it **never once fires during run1's canyon** (canyon AllRMS is byte-identical to the untouched baseline, 125.6065 m, at both F=0.5 and F=0.3), so it contributes essentially zero run1 benefit (+0.000 to +0.067 pp, nowhere near the +0.5 pp bar) — the one place the whole policy exists to help. It still costs run2 up to −0.339 pp (worse than duration `gate=20`'s −0.099 pp) because it also fires on some of run2's ordinary short lapses whenever their (one-epoch-lagged) NLOS fraction happens to cross the threshold. This is consistent with WP7's own prior finding ("even aggressive NLOS downweighting left [the canyon] untouched," WP7_REPORT.md §NLOS) — during the canyon's deepest dropout, so few satellites are tracked at all that the *fraction of tracked satellites flagged NLOS* apparently never spikes the way one would intuit; the corruption there looks more like "too few satellites, period" than "mostly-NLOS satellites," which is exactly what `--nlos-min-los-sats` (§4) targets instead — and that lever independently fails too (loses outright on run1's overall metric despite not touching FixRMS).

## 4. min-LOS-sats AR gate (work item 5)

`experiments/sweep_libgnss_rtk_wp10.py --stage min_los_sats`, `N ∈ {4, 6}` combined with WP7's continuous soft weighting (`--nlos-weight-mode continuous --nlos-continuous-floor 0.5`), full run1 only per the task's own carry-forward gate (`results/wp10/sweep/run1_min_los_sats.csv`):

| N | fixed | `<50cm_full%` | Δ vs baseline (pp) | FixRMS |
|---:|---:|---:|---:|---:|
| baseline | 1130 | 26.643 | — | 0.3128 |
| 4 | 1128 | 22.133 | **−4.510** | 0.3113 |
| 6 | 1150 | 22.376 | **−4.267** | 0.3126 |

**Clean loss at both tested `N`, not carried to run2/run3** per the task's own rule. FixRMS does *not* blow up (0.311–0.313 m, matching WP8's hypothesis that this gate protects AR quality) — the fixed-solution *count* is also nearly unchanged (1128–1150 vs 1130) — but the *set* of accepted fixes shifts enough to lose 4+ pp on `<50cm_full%`, meaning the LOS-count veto is rejecting/altering a materially different (and net-worse) population of AR attempts than baseline, not simply filtering out bad ones. Not investigated further given the clean, decisive negative and the task's explicit stop condition.

## 5. Root cause (why duration/NLOS-fraction gating doesn't generalize)

WP9 diagnosed unconditional `scaled-reset`'s failure as "every lapse gets the same immediate-overconfidence treatment, regardless of whether it needs it." WP10's premise was that lapse *duration* (or NLOS fraction) would separate the canyon's rare, severe, genuinely-helped-by-tightening lapses from run2/3's frequent, benign ones. §3.1's segment census disproves the premise directly: **run1, run2, and run3 all have a similar-sized population (32–42 segments) of comparable-duration (tens of seconds to ~2 minutes) intermittent-tracking gaps** — the "run1 is canyon-dominated, run2/3 are mostly-open-sky" framing from WP9 undersells how much of run1's *own* win actually comes from a cluster of medium lapses that look structurally identical to run3's losing ones, not from the single headline canyon event (whose own core dropout is barely touched by any gate value, §3.1). Neither raw duration nor the (one-epoch-lagged) NLOS-flagged-satellite fraction correlates with "does tightening the covariance here help or hurt" — both are proxies for "the receiver briefly lost lock," which happens for structurally similar reasons (temporary obstruction, low elevation, brief multipath) across all three environments, and only the *re-acquisition geometry* after each specific gap — not visible to either gate signal — determines whether SPP's honest-tighter reseed helps or an overconfident anchor hurts the next few epochs.

## 6. Combination (work item 6) — skipped, correctly, per the task's own conditional

The task's rule: "if items 1-4 produce a passing winner **and** item 5 wins independently, run the combination on all 3 runs." Neither condition holds — no gate value passes the regression matrix (§3), and `--nlos-min-los-sats` loses outright on run1 (§4) — so no combination candidate was run. This mirrors WP9's own decision structure exactly (there, the analogous `--hold-ratio-threshold 2.0` combination was run only because the task listed it unconditionally; here the task's combination is explicitly gated on both halves winning, and neither does).

## 7. Test suite counts

- **C++ (`third_party/gnssplusplus/build/tests/run_tests`): 322 tests, 268 passed, 0 failed, 54 skipped** (up from WP9's 310/258/0/52: +7 new `FloatTrustPolicyTest::lapseGateExceeded` cases, +3 new `NlosWeightsTest::nlosMinLosSatsGateAllows` cases, all 10 passing; +2 new `RTKSmokeTest` cases — `FloatTrustPolicyLapseGatedWithHugeGateIsBitIdenticalToLegacy` / `FloatTrustPolicyLapseGatedAtZeroGateDoesNotCrash` — both `GTEST_SKIP()`'d for the same pre-existing missing-fixture-data reason as 52 others, not new failures).
- **Python (`pytest -p no:xonsh tests/test_sweep_libgnss_rtk_wp10.py`): 11 passed, 0 failed** (9 WP10-candidate-shape tests carried over/extended + 2 new for the NLOS-fraction-trigger stage added in §3.2).
- **Full project Python suite** (`pytest -p no:xonsh tests/ --ignore=tests/test_reproduce_urbannav_external_baseline.py`, the latter pre-existing and unrelated — missing `eval_harness_lib` module, not touched by this task): **2624 passed, 39 failed, 46 skipped.** All 39 failures are in modules untouched by WP10 (`test_cuda_streams.py` — no CUDA device in this environment; `test_plateau.py` — missing/mismatched `pyproj`; `test_gsdc2023_*`/`test_eval_gsdc2023_*`/`test_pf_smoother_forward_epoch.py`/`test_submit_gsdc2023_pixel5_candidate_queue.py`/`test_run_gsdc2023_taroz_full_parity_gate.py` — an unrelated GSDC2023 pipeline); every WP10-relevant test file (`test_sweep_libgnss_rtk_wp10.py`, `test_sweep_libgnss_rtk_wp9.py`, `test_score_vs_inuex35.py`) passes cleanly (27/27).

## 8. 3-run summary vs inuex35 and prior WPs

`gate=2` shown as the "closest to a win" configuration (best run1 result of anything tested) — reported honestly with its run3 regression, per this project's established negative-result convention, not shipped as a new default.

| method | run | AllRMS | FixRMS | fix% | `<50cm_full%` | ppc | vs inuex35 |
|---|---|---:|---:|---:|---:|---:|---:|
| inuex35 README (external) | run1 | 47.40 | 0.815 | 49.5 | **56.7** | n/a | — |
| WP6/7/8 baseline (`--max-pos-jump-rate 2.3`, dead knobs wired) | run1 | 19.471 | 0.313 | 15.10 | 26.643 | 23.65 | −30.1pp |
| WP9 scaled-reset qpos=0.1 (unconditional, does not generalize) | run1 | 19.327 | 0.313 | 15.31 | 27.423 | 25.44 | −29.3pp |
| **WP10 lapse-gated gate=2 qpos=0.1 (best run1, does not generalize)** | run1 | 19.152 | 0.313 | 15.34 | **27.708** | 27.71 | −29.0pp |
| inuex35 README (external) | run2 | 32.08 | 0.277 | 60.8 | **69.9** | n/a | — |
| WP6/7/8 baseline | run2 | 9.335 | 0.125 | 13.04 | 43.165 | 48.05 | −26.7pp |
| WP9 scaled-reset qpos=0.1 (regresses) | run2 | 9.413 | 0.126 | 13.57 | 41.373 | 48.45 | −28.5pp |
| WP10 lapse-gated gate=2 qpos=0.1 (passes ±0.3pp gate) | run2 | 9.276 | 0.125 | 13.02 | 42.891 | 49.05 | −27.0pp |
| inuex35 README (external) | run3 | 34.52 | 0.211 | 59.4 | **67.9** | n/a | — |
| WP6/7/8 baseline | run3 | 5.637 | 0.081 | 5.17 | 43.710 | 40.67 | −24.2pp |
| WP9 scaled-reset qpos=0.1 (regresses) | run3 | 5.841 | 0.081 | 5.14 | 42.383 | 40.65 | −25.5pp |
| **WP10 lapse-gated gate=2 qpos=0.1 (fails −0.3pp gate)** | run3 | 5.650 | 0.081 | 5.14 | **43.108** | 40.56 | −24.6pp |

**Recommendation: do not change the shipped default.** `--float-trust-policy` stays `legacy` by default (bit-identical, verified). The WP7/8 baseline remains the best-generalizing configuration across all three runs. `lapse-gated gate=2` is a strictly better run1-local result than WP9's unconditional `scaled-reset` (+1.065 vs +0.78 pp, still no worse a run3 regression in absolute terms — WP9's was −1.327 pp, WP10's is −0.602 pp, a >2× improvement in regression severity even though it still doesn't clear the ±0.3pp bar) but is not safe to ship globally as-is.

## 9. Next-bottleneck recommendation

WP9 recommended gating on duration or NLOS fraction; WP10 built and tested both, and §5's segment census shows *why neither generalizes*: the discriminator needs to be about **re-acquisition quality**, not lapse duration or NLOS-flag prevalence. Two concrete next steps follow directly from this run's evidence:

1. **Gate on post-lapse geometry instead of pre-lapse duration**: use the *first post-reacquisition epoch's* own solution quality signal (e.g. DOP, residual RMS, or number of satellites actually used in the SPP reseed) to decide whether to trust a tight `scaled-reset` variance or fall back to legacy's wide one — this directly targets "was the re-acquisition any good," which duration/NLOS-fraction can only proxy for badly (§5).
2. **Per-run/per-environment calibration instead of a single global config**: since WP9 and WP10 have now both shown the same run1-wins/run2-3-regresses split across two structurally different policies (unconditional and duration-gated `scaled-reset`), a config that ships `legacy` as the safe global default but allows an opt-in `scaled-reset`/`lapse-gated` override for known canyon-heavy corridors (e.g. keyed by a route/segment classifier upstream of `gnss_solve`) may be more productive than continuing to search for a single global gate value that this task's own segment-census evidence suggests likely does not exist.

## 10. Deliverables

- `results/wp10/WP10_REPORT.md` (this file)
- `results/wp10/verify/z0_baseline_no_wp10.pos`, `z1_lapsegated_hugegate.pos` — bit-identical verification artifacts (SHA `4c2effb6…`, matches `results/wp8/verify/wp8_absentflag_check_run1.pos`)
- `results/wp10/sweep/run1_gate_sweep.csv`, `run1_gate_fine.csv` — work item 3 gate sweep
- `results/wp10/sweep/run{2,3}_regression.csv`, `run{2,3}_regression_gate{5,10,20}.csv` — work item 4 regression matrix
- `results/wp10/sweep/run{1,2,3}_nlos_frac_sweep.csv` — §3.2's NLOS-fraction-trigger-only evaluation
- `results/wp10/sweep/run1_min_los_sats.csv` — work item 5 min-LOS-sats coarse test
- `.pos` files for every candidate above under `results/wp10/sweep/run*/`, `run*_nlosfrac/`
- New C++: `float_trust_policy.hpp/.cpp` (`lapseGateExceeded`), `nlos_weights.hpp/.cpp` (`nlosMinLosSatsGateAllows`), `rtk.hpp/.cpp`/`gnss_solve.cpp` diffs (flag-guarded), `test_float_trust_policy.cpp`/`test_nlos_weights.cpp`/`test_rtk_smoke.cpp` new cases
- New Python: `experiments/sweep_libgnss_rtk_wp10.py`, `tests/test_sweep_libgnss_rtk_wp10.py` (11 tests, all pass)
- No git commits made; no protected files touched; build tree unchanged at `third_party/gnssplusplus/build` (WSL-accessed via `/mnt/c`).
