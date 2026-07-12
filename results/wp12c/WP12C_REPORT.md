# WP12c — Schur-complement marginalization + window-width sweep on TC-FGO float — tokyo run1 probes

Workspace: `C:\Users\rsasa\Workspace\old\gnss_gpu`. Builds on WP12b (`--recovery --anchor-fix --dynamic-dd-rebuild --dd-carrier --persistent-ambiguities`). WP12c implements proper Schur-complement marginalization (`--schur-marginal`, replacing the diagonal σ=0.2 m front-state prior) and sweeps window width {5, 10, 25} with and without it, all on the run1 4 000-epoch probe.

**Headline result: stage-1 gate FAIL, but with the campaign's most important mechanism finding so far.** The Schur marginal is now *mathematically verified correct* (closed-form acceptance tests, see §2) and delivers a **10× open-sky float improvement (0.44 → 0.04 m RMS)** — yet **every** memory mechanism (correct Schur, window 5→25, both combined) *loses or ties* on 4k-probe AllRMS, because the probe's error budget is dominated by degraded-GNSS segments where the best available *absolute* information (DD-code least squares) is only ~28 m. **Cross-epoch memory cannot manufacture absolute accuracy that no factor in the graph possesses.** This redirects the campaign (§6): stop chasing full-length AllRMS < 20 m — even inuex35's own AllRMS is 47.4 m — and go after `<50cm_full%` directly by enabling AR *only where the float is already sub-meter*, which the telemetry shows is exactly the open-sky regime we now dominate.

## 1. Gate verdicts

| gate | criterion | result | pass? |
|---|---|---|:---:|
| Stage 1 (float health) | run1 AllRMS < 20 m at ~100 % coverage | best 4k-probe config remains WP12b no-schur win5: **109.7 m**; all WP12c variants 125–156 m | **NO** |
| Stage 2 (AR quality) | `<50cm_full%` > 26.6, FixRMS ≤ 0.8 m | not attempted (stage 1 unmet; LAMBDA stays off per WP12b rule) | **N/A** |

No full-length runs were paid for: no probe beat the reference, per the pre-registered decision rule.

## 2. Schur implementation: bug postmortem, then verified correct

First iteration (predecessor v1) scored **1 365.9 m** (vs 109.7 reference) with recovery firing 951×: the marginal's prior *mean* was taken from stale window seeds instead of the converged LM values, so each new window was dragged toward the previous window's *pre-optimization* state. The mean fix brought it to 211.6 m; remaining issues (bias-block interaction 232.3 m, index mapping across the slide) were then closed against **closed-form acceptance tests** (`tests/test_tc_fgo.py`):

- `test_schur_complement_matches_analytic_gaussian_marginal` / `test_schur_chain_three_state_exact_conditioning` — exact conditioning on toy Gaussians;
- `test_sliding_schur_matches_joint_solve_constant_velocity` — two consecutive marginalized windows reproduce the joint batch solve (fixed-lag smoother equivalence);
- `test_schur_marginal_cleared_on_recovery_path_in_runner_logic` — marginal wiped on recovery/generation bump.

**pytest `tests/test_tc_fgo.py`: 24 passed** (18 carried + 6 new). Conditioning guards: eigenvalue floor `1e-6`, optional cap from the in-window Hessian (`schur_info_cap_ratio`), nav-only front-block option. The final implementation is *correct by construction test*; §3's negatives are therefore behavioral facts, not bugs.

## 3. Probe ablation (run1, first 4 000 rover epochs → 3 900 scored)

| config | AllRMS (m) | `<50cm%` | `<50cm_full%` | recovery | verdict |
|---|---:|---:|---:|---:|---|
| **no-schur win5 (WP12b reference)** | **109.7** | 15.6 | 5.1 | 259 | still the 4k-probe best |
| no-schur win25 (1k probe, WP12b) | (1.9 on 1k) | 66.6 | 5.6 | 7 | did **not** generalize, see next row |
| **no-schur win25, 4k (decisive test)** | **125.4** | 17.1 | 5.6 | 263 | window width alone does not survive the drift region |
| schur v1 (broken mean) | 1 365.9 | 1.7 | 0.6 | 951 | postmortem §2 |
| schur mean-fix (win5+bias) | 211.6 | 9.3 | 3.0 | 280 | partial fix |
| schur fixed, win5 | 156.4 | 20.8 | 6.8 | 257 | correct math, still loses |
| schur fixed, win10 | 149.9 | 20.8 | 6.8 | 257 | |
| schur fixed, win25 | 137.7 | 20.8 | 6.8 | 271 | best schur, still loses |

Wall cost: win25 ≈ 0.7 s/epoch (7× win5); schur overhead at win5 is small (~10 %).

## 4. Telemetry signatures (the mechanism, quantified)

Same segments/method as WP12b §2.1 (`results/wp12c/_telemetry_analysis.py`):

| config | ep 0–999 RMS | ep 500–799 (open sky) | ep 1000–1499 (drift onset) | recovery @fire → +10 ep |
|---|---:|---:|---:|---|
| WP12b no-schur win5 | 2.32 m | 0.44 m | 17.20 m | 27.8 → 25.7 m (stuck) |
| schur fixed win5 | **1.19 m** | **0.04 m** | 16.39 m | 28.5 → 26.0 m (stuck) |
| schur fixed win10 | 1.00 m | 0.04 m | 15.46 m | 28.7 → 25.5 m (stuck) |
| no-schur win25 | 1.88 m | — | 15.01 m | 28.1 → 23.4 m (stuck) |
| schur fixed win25 | 0.98 m | — | 13.83 m | 28.6 → 23.3 m (stuck) |

Verdicts on the two pre-registered signatures:

- **(a) post-recovery tightening: did NOT flip.** All configs re-seed at ~28 m and sit at ~23–26 m ten epochs later. The reseed lands at DD-code-LS accuracy *in that geometry*, and no graph memory can improve on it because no factor in the graph knows better.
- **(b) drift-onset flattening: marginal.** 17.2 → 13.8 m at best (schur+win25 combined), nowhere near the sub-meter continuation the gate needs.
- **Bonus (not pre-registered): open-sky float is now 0.04 m RMS** — the correct Schur marginal makes converged stretches essentially carrier-smooth. This is the asset §6 builds on.

## 5. Honest conclusion

The WP11→WP12c arc chased one hypothesis — "position cross-epoch memory is the bottleneck" — to its end. With memory now *provably correct* (fixed-lag-smoother-equivalent) and windows up to 5 s, the 4k probe still fails, because the drift region's problem is **absolute-information quality** (NLOS-corrupted code, ~28 m DD-LS), the same enemy WP7–WP10 fought inside the RTK engine. inuex35 does not solve it either: their run1 AllRMS is **47.4 m**. They win the campaign metric with **49.5 % FIX epochs at 0.815 m**, i.e. by fixing carrier ambiguities wherever the float is locally sane and letting the bad stretches be bad.

## 6. Recommended WP12d — AR where the float is sub-meter (the inuex35 shape, finally)

1. **Quality-gated LAMBDA**: enable `--lambda-ar` only on windows whose marginal position σ and post-fit DD-CP/PR residuals certify sub-meter float (thresholds from §4's open-sky telemetry). This dodges the WP4/WP12b self-consistency trap by construction — AR is simply *not offered* a biased float.
2. **Validation stack on accepted fixes** (inuex35 port, WP12b wiring exists): ratio 3.0 → subset-AR → DDPR cross-validation at the fixed position → fix-and-hold with post-AR cost gate; recovery/generation bump wipes holds.
3. Ship config: schur fixed win5 (open sky 0.04 m at ~0.15 s/ep) as the float; win25 not worth 7× cost for −19 m AllRMS.
4. Success = stage-2 gate directly: run1 `<50cm_full%` > 26.6 with FixRMS ≤ 0.8 m; fix% reported honestly. The probe's `<50cm%` is already 20.8 % with *zero* fixes — every correctly fixed open-sky epoch adds on top.

## 7. Deliverables

- `results/wp12c/WP12C_REPORT.md` (this file)
- `results/wp12c/probe_*.pos` + `*_telemetry.csv` + `*.score.json` (all §3 rows), `ablation_4000_summary.json`
- `results/wp12c/_telemetry_analysis.py` (§4 generator)
- Code: `python/gnss_gpu/tc_fgo.py` (`TcSchurMarginal`, `schur_complement_marginalize`, `schur_front_block_marginalize`, conditioning guards, runner flag `--schur-marginal` + `--window-epochs`)
- Tests: `tests/test_tc_fgo.py` — **24 passed** (6 new Schur acceptance tests)

*Process note: WP12c was executed across three agent sessions interrupted by infrastructure errors; probes and scores were preserved on disk and the final synthesis (this report) was assembled by the coordinating session. Full-length 3-run scoring was intentionally not performed (pre-registered rule: no full runs until a probe beats the 109.7 m reference).*
