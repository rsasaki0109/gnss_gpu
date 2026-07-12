# WP12e — Dense RTK anchoring + calibrated cert AR on TC-FGO Schur float — tokyo run1-3

Workspace: `C:\Users\rsasa\Workspace\old\gnss_gpu`. Builds on WP12d (feedback-fixed cert-tight AR; `<50cm_full%` stuck at **7.0 %** on run1). WP12e implements the WP12d §6 mandate: audit WP7 baseline anchor supply, densify with truth-honest FLOAT priors (`--anchor-float`, per-row σ from PDOP/nsat/ratio + `--anchor-sigma-scale`), and re-calibrate full-length cert (`--ar-cert-min-epochs-since-recovery 25`, `--ar-cert-max-epochs-since-anchor 50`) to stop the WP12d drift-tail FixRMS leak (60 m → **1.59 m** on run1).

**Headline result: stage-2 gate FAIL.** Dense anchors materially improve float geometry (probe AllRMS **156 → 67 m**) and full-run FixRMS purity (run1 **60 → 1.59 m**), but `<50cm_full%` moves only **7.0 → 7.2 %** on run1 — the campaign metric is still capped by *where* sub-meter accuracy exists, not anchor count. FLOAT anchors cover **96.6 %** of run1 within 50 epochs yet carry **21 m** truth RMS in degraded stretches; cert still fires almost exclusively in open-sky (marginal σ gate). Canyon (tow 188990–189070) unchanged: **0 FIX**, **115 m** RMS.

## 1. Gate verdicts

| gate | criterion | run1 result | pass? |
|---|---|---|:---:|
| **Stage 2 (AR quality)** | `<50cm_full%` **> 26.6** with FixRMS **≤ 0.8 m** | full run1 **7.2 %**, FixRMS **1.59 m** (probe cert+hold: FixRMS **0.40 m**, `<50cm_full%` **6.9 %**) | **NO** |

Shipped config: `--recovery --anchor-fix --anchor-float --dynamic-dd-rebuild --dd-carrier --persistent-ambiguities --schur-marginal --lambda-ar --ar-cert-max-pos-sigma 0.15 --ar-cert-max-dd-pr-rms 1.0 --ar-cert-min-epochs-since-recovery 25 --ar-cert-max-epochs-since-anchor 50` (full validation stack on).

## 2. Anchor supply audit (WP7 baseline `.pos`, truth vs `reference.csv`)

Baselines: `results/wp10/sweep/run{1,2,3}/a0|b0_baseline_no_wp10.pos`. FIX = Status==4; FLOAT = Status∈{1,3}.

| run | FIX anchors | FLOAT anchors | FIX after ep1000 | FIX truth RMS (m) | FLOAT truth RMS (m) | FIX-only within 50 ep | FIX+FLOAT within 50 ep |
|---|---:|---:|---:|---:|---:|---:|---:|
| run1 | 1 130 (9.5 %) | 6 355 (53.3 %) | 320 (28 % of FIX) | **0.31** (tail 0.58) | **21.1** (med 0.72) | **13.1 %** | **96.6 %** |
| run2 | 843 (9.2 %) | 5 623 (61.5 %) | 323 | **0.12** | **10.0** | 14.8 % | **99.5 %** |
| run3 | 666 (4.4 %) | 12 209 (79.8 %) | 2 | **0.08** | **5.8** | 6.0 % | **100 %** |

**Key findings:**

1. **FIX supply is sparse outside open-sky** — run1 has only **320** FIX epochs after ep1000; FIX-only anchor deserts cover **87 %** of the timeline at 50-epoch radius.
2. **FLOAT densification removes deserts** — FIX+FLOAT within 50 ep reaches **96–100 %**, but FLOAT truth RMS is **5–21 m** in drift (median sub-meter masks a heavy tail).
3. **`<50cm` RTK candidates far from FIX** — run1: **1 996 / 3 178** epochs with RTK truth `<50 cm` have no FIX within 50 ep; many coincide with FLOAT rows whose σ we honestly set to **2–5 m**, i.e. they cannot certify a sub-meter TC-FGO float for AR.
4. **Approach not doomed on coverage, doomed on quality** — anchor deserts do *not* coincide with all remaining `<50cm` candidates once FLOAT is included; the blocker is FLOAT *accuracy* in canyon/drift (WP8 reset loop), not anchor spacing.

Artifact: `anchor_audit.json`, generator `_anchor_audit.py`.

## 3. Probe ablation (run1, first 4 000 rover epochs → 3 900 scored)

| stage | AllRMS (m) | FixRMS (m) | fix % | `<50cm_full%` | n_fix |
|---|---:|---:|---:|---:|---:|
| WP12d cert-tight+hold (ref) | 156.4 | 0.71 | 14.7 | 6.8 | 572 |
| **float dense anchor** (no AR) | **66.7** | — | 0 | 6.8 | 0 |
| **cert dense + hold** | **66.7** | **0.40** | 14.9 | **6.9** | 580 |

Dense FLOAT+FIX anchors cut probe AllRMS **2.3×** vs WP12d but `<50cm_full%` is unchanged: improved geometry does not add epochs to the full-rover denominator. Cert telemetry (ep1000–3999 drift): **105** cert-pass / **10** fixed vs **704** total cert-pass — AR remains an open-sky phenomenon.

Artifact: `ablation_4000_summary.json`, `probe_*_4000_telemetry.csv`.

## 4. Full-length 3-run table

| method | run | cov % | AllRMS (m) | FixRMS (m) | fix % | `<50cm_full%` |
|---|---|---:|---:|---:|---:|---:|
| inuex35 README | run1 | 100 | 47.4 | 0.815 | 49.5 | **56.7** |
| inuex35 README | run2 | 100 | 32.1 | 0.277 | 60.8 | **69.9** |
| inuex35 README | run3 | 100 | 34.5 | 0.211 | 59.4 | **67.9** |
| WP7 RTK | run1 | 62.0 | 19.9 | 0.084 | 10.5 | 25.4 |
| WP7 RTK | run2 | 70.7 | 9.3 | 0.049 | 12.6 | 43.2 |
| WP7 RTK | run3 | 83.9 | 5.3 | 0.048 | 6.4 | 43.7 |
| WP12d cert-tight+hold | run1 | 97.9 | 235.2 | 60.1 | 5.3 | 7.0 |
| WP12d cert-tight+hold | run2 | 100.0 | 69.1 | 2.47 | 10.3 | 10.2 |
| WP12d cert-tight+hold | run3 | 99.9 | 26.9 | 9.90 | 27.0 | 4.6 |
| **WP12e dense anchor+cert** | **run1** | **97.9** | **206.3** | **1.59** | **5.2** | **7.2** |
| **WP12e dense anchor+cert** | **run2** | **100.0** | **56.6** | **2.30** | **13.5** | **11.7** |
| **WP12e dense anchor+cert** | **run3** | **99.9** | **23.7** | **3.46** | **23.5** | **5.9** |

Scores: `full_run{1,2,3}.score.json`, summary `full_3run_summary.json`.

### 4.1 Run1 canyon (tow 188990–189070)

| metric | WP12d | WP12e |
|---|---:|---:|
| n epochs | 401 | 401 |
| n FIX | 0 | **0** |
| RMS (m) | 118.4 | **115.0** |
| median (m) | 62.3 | 61.5 |

Artifact: `run1_canyon_188990_189070.json`.

## 5. Honest conclusion

WP12e proves the campaign's **FIX-supply hypothesis half-right**: spreading anchors (FIX+FLOAT) gives the TC-FGO graph absolute information almost everywhere and stops catastrophic wrong-AR blow-ups (FixRMS 60 → 1.6 m), but it **cannot raise `<50cm_full%`** because FLOAT rows in drift/canyon are honestly **multi-metre** anchors — inflating σ does not make the float sub-meter, and the cert correctly refuses AR there. inuex35's edge remains **49.5 % FIX at 0.815 m** from IMU-tight RTK+AR, not denser loose FLOAT priors on a Python TC-FGO port.

**Recommended next workstream (per campaign doc):** abandon further TC-FGO anchor tuning; pursue **PF/RBPF milestone 2** (multi-modal float through canyon memory loss) or **RTK-engine FIX-supply** (WP6 jump-gate axis exhausted at app level; needs structural trust/IMU coupling like inuex35, not more `.pos` rows).

## 6. Deliverables

- `results/wp12e/WP12E_REPORT.md` (this file)
- `anchor_audit.json`, `ablation_4000_summary.json`, `full_3run_summary.json`
- Probe/full `*.pos`, `*.score.json`, `*_telemetry.csv`
- Helpers: `_anchor_audit.py`, `_run_ablation.py`, `_run_full_parallel.py`, `_fix_truth_analysis.py`
- Code: `python/gnss_gpu/tc_fgo.py` (anchor-proximity cert), `experiments/wp12_run_tc_fgo.py` (dense anchors), `experiments/wp5_run_anchored_fgo.py` (extended `.pos` loader, σ helpers)
- Tests: `tests/test_tc_fgo.py` — **32 passed** (+1 anchor-distance cert test)
