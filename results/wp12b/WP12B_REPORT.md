# WP12b — DD carrier + persistent ambiguities on TC-FGO float — tokyo run1-3

Workspace: `C:\Users\rsasa\Workspace\old\gnss_gpu`. Builds on WP12a (`--recovery --anchor-fix`). WP12b ports DD carrier-phase factors, in-window N-continuity, LAMBDA/subset-AR validation (off for shipped float config), dynamic DD rebuild, and **cross-window ambiguity memory** (`TcAmbiguityBank` + `BetweenFactorDouble`-style priors).

**Headline result: both gates FAIL — carrier phase and persistent ambiguities are wired and tested, but the float remains ~110 m on the run1 4k probe and ~229 m full-length; LAMBDA on this float is the WP4 self-consistency trap (80 % “fix”, FixRMS 23.7 m).** Mechanism: **position drift from weak cross-epoch memory**, not AR tuning. Open-sky stretches already achieve sub-meter float (ep 500–800 RMS **0.44 m**); degradation is gradual then catastrophic after ~epoch 3500.

## 1. Gate verdicts

| gate | criterion | run1 result | pass? |
|---|---|---|:---:|
| **Stage 1 (float health)** | probe/full AllRMS **< 20 m** at ~100 % coverage | probe **109.7 m** (4k ep); full **228.6 m** (97.9 % cov) | **NO** |
| **Stage 2 (AR quality)** | run1 `<50cm_full%` **> 26.6** with FixRMS **≤ 0.8 m** | AR not enabled in shipped config; prior-agent +lambda-ar: fix **80.2 %**, FixRMS **23.7 m**, `<50cm_full%` **6.8 %** | **NO** |

**Verdict: WP12b gate not met.** Closest full-length accuracy is run3 (**27.3 m** AllRMS) — near WP12a run3 but still above the 20 m bar. Do **not** re-enable LAMBDA until probe float is meter-level with FixRMS meaning.

Best shipped float config: `--recovery --anchor-fix --dynamic-dd-rebuild --dd-carrier --persistent-ambiguities` (no `--lambda-ar`).

## 2. Probe ablation table (run1, first 4 000 rover epochs → 3 900 scored)

| config | AllRMS (m) | FixRMS (m) | fix % | `<50cm_full%` | notes |
|---|---:|---:|---:|---:|---|
| wp12a baseline (recovery+anchor) | 111.9 | n/a | 0 | 6.8 | prior agent |
| +dynamic-dd-rebuild | 111.9 | n/a | 0 | 6.8 | inert on float |
| +dd-carrier (float amb) | 110.4 | n/a | 0 | 6.8 | −1.5 m AllRMS |
| full (+lambda-ar) | 110.4 | 23.7 | 80.2 | 6.8 | wrong fixes (WP4) |
| best (window-state fix) | 110.2 | 23.3 | 80.1 | 6.8 | prior agent |
| strict AR (min_epochs=8) | 110.2 | n/a | 0 | 6.8 | kills all fixes |
| **+persistent-ambiguities** | **109.7** | n/a | 0 | **5.1** | cross_window_prior mean **19.7** |
| window_epochs=25 (1k ep only) | **1.9** | n/a | 0 | 5.6 | 728 s / 1k ep (7.3× cost) |

Artifact: `results/wp12b/ablation_4000_summary.json`, telemetry `probe_persist_4000_telemetry.csv`.

### 2.1 Telemetry mechanism (persist probe)

| segment | n ep | pos err RMS (m) | notes |
|---|---:|---:|---|
| ep 0–999 | 1 000 | 2.3 | RTK anchor region |
| ep 500–799 | 300 | **0.44** | open-sky — float **can** be sub-meter |
| ep 1000–1499 | 500 | 17.2 | drift onset |
| ep 3500–3899 | 400 | **337** | catastrophic tail |
| recovery @ fire | 259 | mean **27.8** | code-level reseed |
| +10 ep after recovery | — | mean **25.0** | no tightening to meter-level |

`<50cm_full%` frozen at **6.8 %** across all prior-agent configs = RTK anchor epochs (first ~200 s). Outside anchor: **3.9 %** of epochs `< 1 m`.

## 3. Hypothesis verdicts (investigation order)

| # | hypothesis | verdict | evidence |
|---|---|---|---|
| **1** | Per-window disposable ambiguities | **partially confirmed, insufficient** | `TcAmbiguityBank` + cross-window priors (mean 19.7 factors/ep) → probe AllRMS **109.7** vs **110.4** (−0.7 m); `<50cm_full%` regresses 6.8→5.1 % |
| **2** | `<50cm_full%` = anchor only; structural mid-run failure | **confirmed** | ep 500–800 RMS **0.44 m** with 38 DD factors; error–DD correlation 0.77 in wider band reflects drift into recovery, not absent factors |
| **3** | Recovery reseeds to ~meters, graph stays there | **confirmed** | recovery mean err **27.8 m**; +10 ep **25.0 m** — never re-enters sub-meter without anchor |
| **4** | 5-epoch window too short | **promising, costly** | window=25 on first 1k ep: AllRMS **1.9 m** vs **~2.3 m** (window=5); **728 s** vs **~140 s** projected — cubic LM cost |

**Primary mechanism:** weak **position** cross-epoch memory (naive σ=0.2 m marginal + 5-epoch window + IMU-dominated segments), identical root cause to WP11/WP12a. Carrier phase cannot pull a ~100 m-biased position; ambiguities absorb misclosure self-consistently. AR ratio test on that float is WP4’s self-consistency check, not independent fixing.

### 3.1 Code — persistent ambiguity bank (WP12b addition)

Cross-window memory and priors:

```641:720:python/gnss_gpu/tc_fgo.py
@dataclass
class TcAmbiguityBank:
    """Cross-window float ambiguity memory keyed by DD pair (inuex35 amb_gen pattern)."""
    generation: int = 0
    estimates: dict[tuple[str, str, str, str], AmbiguityEstimate] = field(default_factory=dict)
    ...
```

Prior factors in the LM stack:

```1174:1188:python/gnss_gpu/tc_fgo.py
    if layout is not None and x_amb is not None and layout.cross_window_priors:
        nav_base = _window_nav_dim(n, config)
        for amb_idx, (prior_val, prior_sigma) in layout.cross_window_priors.items():
            ...
            counts["n_cross_window_prior"] += 1
```

Recovery bumps generation (wipes bank), mirroring inuex35 `amb_gen` slip handling:

```352:358:experiments/wp12_run_tc_fgo.py
                if recovery_fired:
                    ...
                    if ambiguity_bank is not None:
                        ambiguity_bank.bump_generation()
```

### 3.2 Why LAMBDA is inert on accuracy

```1392:1405:python/gnss_gpu/tc_fgo.py
    fixes, ar_info = _estimate_lambda_fixes(dd_carrier_padded, positions_ecef, win, lam_cfg)
    if not fixes:
        ...
```

With float position ~100 m wrong, LAMBDA resolves integer ambiguities consistent with the wrong geometry → ratio passes, FixRMS **23.7 m**, `<50cm_full%` unchanged at **6.8 %**. Strict `lambda_min_epochs=8` rejects all fixes (fix **0 %**) without improving AllRMS.

## 4. Full 3-run table (best float config)

Scored with `experiments/score_vs_inuex35.py`. Comparison rows from campaign doc / WP12a.

| method | run | n_scored | coverage% | AllRMS | FixRMS | fix% | `<50cm_full%` | ppc_official% |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| inuex35 README | run1 | n/a | 100.0 | 47.40 | 0.815 | 49.5 | **56.7** | n/a |
| inuex35 README | run2 | n/a | 100.0 | 32.08 | 0.277 | 60.8 | **69.9** | n/a |
| inuex35 README | run3 | n/a | 100.0 | 34.52 | 0.211 | 59.4 | **67.9** | n/a |
| WP7/8 RTK baseline | run1 | 7 485 | 62.8 | 19.47 | 0.313 | 15.1 | **26.6** | 23.65 |
| WP7/8 RTK baseline | run2 | 6 466 | 70.7 | 9.33 | 0.125 | 13.0 | **43.2** | 48.05 |
| WP7/8 RTK baseline | run3 | 12 875 | 84.1 | 5.64 | 0.081 | 5.2 | **43.7** | 40.67 |
| WP12a best | run1 | 11 676 | 97.9 | 234.7 | n/a | 0 | **7.1** | 2.44 |
| WP12a best | run2 | 9 148 | 100.0 | 103.1 | n/a | 0 | **9.5** | 9.55 |
| WP12a best | run3 | 15 291 | 99.9 | 27.6 | n/a | 0 | **7.7** | 1.67 |
| **WP12b best** | run1 | 11 676 | **97.9** | **228.6** | n/a | 0 | **5.2** | 1.88 |
| **WP12b best** | run2 | 9 148 | **100.0** | **69.3** | n/a | 0 | **9.4** | 5.49 |
| **WP12b best** | run3 | 15 291 | **99.9** | **27.3** | n/a | 0 | **2.7** | 1.13 |

Artifacts: `results/wp12b/full_run{1,2,3}.pos`, `full_run{1,2,3}_score.json`, logs.

### 4.1 Run1 canyon (tow 188 990–189 070)

| segment | n_scored | AllRMS (m) | `<50cm_full%` |
|---|---:|---:|---:|
| WP12b best | 401 | **118.5** | 0.0 |
| WP12a best (ref) | 401 | 120.9 | 0.0 |

Carrier + persist does not fix canyon; recovery bounds blow-up vs WP11 but anchor remains WP10 RTK float.

## 5. Execution

```powershell
$env:PYTHONPATH="python"
# 4000-epoch probe (best float stack)
python experiments/wp12_run_tc_fgo.py --run tokyo/run1 --max-epochs 4000 `
  --export-pos results/wp12b/probe_persist_4000.pos `
  --recovery --anchor-fix --dynamic-dd-rebuild --dd-carrier --persistent-ambiguities `
  --telemetry-csv results/wp12b/probe_persist_4000_telemetry.csv

# Full 3-run (parallel)
python experiments/wp12_run_tc_fgo.py --run tokyo/run1 --export-pos results/wp12b/full_run1.pos `
  --recovery --anchor-fix --dynamic-dd-rebuild --dd-carrier --persistent-ambiguities
python experiments/wp12_run_tc_fgo.py --run tokyo/run2 --export-pos results/wp12b/full_run2.pos `
  --baseline-pos results/wp10/sweep/run2/b0_baseline_no_wp10.pos `
  --recovery --anchor-fix --dynamic-dd-rebuild --dd-carrier --persistent-ambiguities
python experiments/wp12_run_tc_fgo.py --run tokyo/run3 --export-pos results/wp12b/full_run3.pos `
  --baseline-pos results/wp10/sweep/run3/b0_baseline_no_wp10.pos `
  --recovery --anchor-fix --dynamic-dd-rebuild --dd-carrier --persistent-ambiguities
```

| run | epochs | recovery | wall (s) | s/epoch |
|---|---:|---:|---:|---:|
| probe persist 4k | 3 900 | 259 | 539 | 0.14 |
| probe win25 1k | 1 000 | 7 | 728 | 0.73 |
| run1 full | 11 676 | 1 098 | 1 387 | 0.12 |
| run2 full | 9 148 | 473 | 1 132 | 0.12 |
| run3 full | 15 291 | 555 | 1 738 | 0.11 |

## 6. Honest gaps / blockers

1. **Position memory, not ambiguity memory, is the bottleneck** — persistent ambiguities are necessary but not sufficient; marginal prior remains diagonal σ=0.2 m with no Schur complement.
2. **5-epoch window** cannot match inuex35 ISAM2 effective history; window=25 helps early segment (1.9 m) at 7× cost — needs sparse/incremental solver for full-run viability.
3. **IMU biases frozen outside LM** — attitude/bias errors compound through canyon; `--optimize-imu-biases` scaffolded but not in best config.
4. **LAMBDA blocked** until probe AllRMS < 20 m; current 80 % fix rate is actively misleading.
5. **run3 `<50cm_full%` regression** (7.7→2.7 %) from carrier factors in anchor-poor run — monitor per-run anchor supply before enabling dd-carrier globally.

## 7. Recommended WP12c

1. **Incremental / Schur marginalization** (or pilot ISAM2 binding) — give position real cross-epoch memory, not just ambiguities.
2. **Wider window (25–50 ep) with profiling** — confirm drift reduction on full 4k probe; block-sparse LM or chunking required.
3. **Promote IMU biases in-graph** + GNSS-quality IMU σ inflation (WP12a knobs, off in best config).
4. **Re-enable LAMBDA only after** probe AllRMS < 20 m **and** FixRMS ≤ 0.8 m on a held-out AR smoke — never optimize fix % alone.

## 8. Deliverables

- `results/wp12b/WP12B_REPORT.md` (this file)
- `results/wp12b/ablation_4000_summary.json`
- `results/wp12b/probe_*` + `full_run{1,2,3}.*`
- `results/wp12b/full_run1_canyon_score.json`
- Code: `python/gnss_gpu/tc_fgo.py`, `experiments/wp12_run_tc_fgo.py`
- Tests: `tests/test_tc_fgo.py` — **18 passed**
