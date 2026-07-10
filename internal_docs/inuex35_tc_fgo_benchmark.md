# inuex35 tightly-coupled-gnss-imu-fgo — benchmark target & comparison

Last updated: 2026-07-10 (WP14 — verified closure, see "Campaign closure (2026-07-10)").

This is the working document for the campaign to **beat
[inuex35/tightly-coupled-gnss-imu-fgo](https://github.com/inuex35/tightly-coupled-gnss-imu-fgo)
on the shared Tokyo PPC benchmark**. Sequencing decided 2026-07-04:

1. **Polish standalone FGO first** (GNSS+IMU tight coupling — `fgo_gnss_lm_vd`
   full stack: DD + IMU preintegration + LAMBDA AR) until it beats their numbers.
2. **Then PF+FGO hybrid** as the second milestone.

Win criterion: side-by-side on **both metric families** (their
AllRMS/FixRMS/fix%/<50cm *and* our `ppc_score` OFFICIAL%), same runs, same
epochs, coverage reported honestly. Headline metric: **<50cm%**.

## Target numbers (their README, Tokyo PPC, full length, defaults, NF=3)

| run | length | AllRMS | FixRMS | fix % | <50 cm |
|---|---|---:|---:|---:|---:|
| tokyo/run1 | 11,928 ep | 47.40 m | 0.815 m | 49.5 % | 56.7 % |
| tokyo/run2 |  9,151 ep | 32.08 m | 0.277 m | 60.8 % | 69.9 % |
| tokyo/run3 | 15,301 ep | 34.52 m | 0.211 m | 59.4 % | 67.9 % |

Dataset identity confirmed: their CLI (`run_imu_gnss_tc.py rover.obs base.obs
base.nav imu.csv reference.csv`) consumes exactly the layout of our checked-in
`datasets/PPC-Dataset-data/tokyo/run{1,2,3}/` (Septentrio mosaic-X5 5 Hz rover,
1 Hz base, 100 Hz MEMS IMU, 5 Hz ground truth). Same data, fair fight.

## Reproduction verdict (Track A, 2026-07-04)

Their README numbers are **real and reproducible**. Full report:
`C:\Users\rsasa\Workspace\old\repro_tc_fgo\REPRO_REPORT.md` (plain MSVC build of
`inuex35/gtsam@develop`, Boost/TBB/MKL all OFF, cssrlib fork @dd90eb0).

| run | metric | README | reproduced |
|---|---|---:|---:|
| run1 | <50 cm | 56.7 % | 52.4 % (−4.3 pp; FixRMS +19 % — LAMBDA AR platform variance, no seed) |
| run2 | <50 cm | 69.9 % | **69.9 % (exact, all 4 metrics)** |
| run3 | <50 cm | 67.9 % | **67.9 % (exact, all 4 metrics)** |

Re-scorable per-epoch trajectories: `repro_run{1,2,3}.csv` (LLH) and
`results\tc_run{1,2,3}.npz` (ECEF + smode + err3d) under the repro workspace.
Implication: target numbers stand as-is; run1's −4.3 pp shows AR outcomes have
platform-level variance of a few pp, so wins smaller than ~5 pp on a single run
should be treated as noise.

## How their numbers should be read (the win strategy)

FixRMS 0.2–0.8 m at fix 50–60 % but **AllRMS 32–47 m**: when they fix, they are
decimeter-accurate; when they don't (40–50 % of epochs), the solution blows up
by tens of meters. The attack surface is therefore:

- **(a) raise fix rate** beyond their ~50–61 % with equal or better fix purity, and/or
- **(b) crush the non-fix blow-ups** — their float/dead-reckoning epochs are the
  entire AllRMS story.

Our unique assets map directly onto (b) and partially (a):

| Our asset | Where it hits |
|---|---|
| PLATEAU 3D-mesh ray-traced NLOS priors (LOS median 1.0 m, AUC 0.92) | They have **no city-model prior** — only reactive residual-based `sat_badness`. Feeding NLOS masks into DD weights cleans the float solution *and* the AR candidate set → (a)+(b) |
| GPU wide-window batch FGO (`fgo_gnss_lm_vd`) | They smooth over a **1.0 s fixed lag only** (ISAM2 `IncrementalFixedLagSmoother`). A wide/batch window bridges their blow-up stretches → (b) |
| PF/RBPF + robust kernels at 100K–1M particles | Multi-modal float tracking through canyon stretches where their single-hypothesis FLS diverges → (b); milestone 2 |
| `local_fgo.py` LAMBDA + ratio test + per-window fix machinery | Already ours; needs wiring to raw PPC DD streams → (a) |

Honest difficulty note: their pipeline is heavily engineered (hundreds of
tuned knobs, multi-stage AR validation, recovery FSMs — see below). Beating
FixRMS purity head-on is a real fight; the structural edges are the NLOS prior
and window width, not tuning skirmishes.

## Architecture close-read (2026-07-04, source at their `main`)

Two-layer design: **cssrlib supplies the RTK core** (their `ImuGnssTc` extends
`cssrlib.rtk.rtkpos` — `resamb_lambda`, `zdres`/`sdres`, `valpos`, DD prep all
inherited), **GTSAM supplies the estimator**.

- **Two phases** (`runner.py`, `tightly_coupled.py`): Phase 1 GNSS-only Pose3
  RTK collects `n_collect=5` fixes while stationary; transition at speed >
  `vel_thresh=1.0 m/s` seeds heading from the fix-path, roll/pitch from gravity,
  biases from the stationary IMU window; Phase 2 is the moving TC pipeline.
- **Estimator**: `gtsam.IncrementalFixedLagSmoother`, **lag = 1.0 s** (~5 epochs
  at 5 Hz), relinearizeSkip=10, threshold=0.05. Sequential, not batch.
- **States** per epoch: `Pose3` (body, base-anchored local ENU via `ecef_T_nav`),
  `Vel`, `imuBias.ConstantBias`; ambiguities as scalar `Double` values keyed
  `n(gen*1e6 + sat*10 + freq)`. Lever arm handled inside the factors.
- **Factors** (`buildfactor/factors.py`, `imu_preintegration.py`):
  - `DoubleDifferencePseudorangeFactorArm` / `DoubleDifferenceCarrierPhaseFactorArm`
    (custom C++, see reproduction below), earth-rotation corrected;
    a pure-Python `CustomFactor` DDCP variant folds *held* integer N into a
    constant to drop factor arity (perf trick, `factors.py:132-215`).
  - `gtsam.CombinedImuFactor` + `BetweenFactorConstantBias` + per-epoch bias
    prior; IMU **integration covariance inflated by last DDPR residual²/dt**
    (`imu_preintegration.py:53-72`) — GNSS quality throttles IMU trust.
  - N-continuity: `BetweenFactorDouble` chain (σ=0.01 cyc fixed / 0.1 float).
  - NHC (lateral/vertical vel ≈ 0), ZUPT/ZARU, Doppler velocity prior,
    bootstrap DDPR pose priors for the first ~20 Phase-2 epochs.
- **AR** (`optimize/ar.py`): cssrlib `resamb_lambda` (RTKLIB mode), ratio 3.0,
  **fix-and-hold** (`ar_mode=3`) with a deep acceptance stack:
  eligibility (DD-fraction gate) → precheck skip → per-sat residual gate →
  LAMBDA → **subset-AR** (drop up to 2 ranked-bad sats, keep best ratio) →
  RTKLIB `valpos` → **DDPR cross-validation at the fixed position**
  (reject if residual worsens) → context reject (fragile nb≤6 fixes during
  cp-hold/ddpr-bad) → weak-fix gates → hold, then **post-AR cost gate**
  (un-hold if post-fit DDPR RMS degrades > threshold).
- **Robustness / recovery** (`preprocess/`, `validation/`): RTKLIB-demo5
  `varerr` el/SNR weighting; `sat_badness` σ-inflation from residual history;
  CP-vs-PR innovation gate; post-fit FDE (PR>4 m / CP>0.5 m vs median, rejected
  CP treated as slip via `amb_gen` bump); **DDPR-sanity FSM** — persistent
  post-fit RMS > 3 m (catastrophic 15 m fast path) triggers DDPR-only LS anchor,
  anchor-vs-IMU consistency check, ambiguity wipe + CP-hold + PIM break, and
  IMU-predicted pose fallback, all multi-gated (GDOP, persistence,
  multipath-dominance ratio).
- **Config surface**: ~200 knobs in `config.py` `TcConfig`, env-var driven,
  with a tuned `tokyo_mode2_satbad_cponly` preset. The README defaults are the
  reported configuration.

## Reproduction recipe (verified facts, 2026-07-04)

- The PyPI `gtsam` wheel **lacks** the DD factors (their `requirements.txt`
  says so explicitly). The factor sources live in the fork
  **`inuex35/gtsam`, branch `develop`**:
  `gtsam/navigation/PseudorangeFactor.h` (`DoubleDifferencePseudorangeFactorArm`
  ~L720), `CarrierPhaseFactor.h` (`DoubleDifferenceCarrierPhaseFactorArm`
  ~L534), `GPSFactor.h` (`GPSFactorArm` ~L125), and all are exposed to Python
  in `gtsam/navigation/navigation.i` (~L673/L839). Building that fork with
  `-DGTSAM_BUILD_PYTHON=ON` is the reproduction path.
- cssrlib must be the fork:
  `pip install -e "git+https://github.com/inuex35/cssrlib-numba.git@dd90eb0#egg=cssrlib"`.
- Run pattern: `LEVER_ARM=0.31,0,0.55 SAVE_NPZ=... python examples/run_imu_gnss_tc.py
  rover.obs base.obs base.nav imu.csv reference.csv` (env-var config).
- Their metric definitions (from `examples/run_imu_gnss_tc.py`): err3d =
  ‖sol−ref‖ at nearest-TOW reference row; FIX = `smode==4`; <50cm over all
  scored epochs; they process every rover epoch (IMU-only propagation when
  base/sats missing), so **coverage ≈ 100 % by construction** — our comparisons
  must state coverage explicitly.

## Workstreams

| WS | What | Where | Status (2026-07-04) |
|---|---|---|---|
| Track A | Reproduce their pipeline: build `inuex35/gtsam@develop` (plain MSVC, build tree on E:, **no vcpkg** — a vcpkg/MKL attempt filled the C: drive), run tokyo run1→3, `REPRO_REPORT.md` | `C:\Users\rsasa\Workspace\old\repro_tc_fgo\` (outside this repo), specs `TASK_A.md`–`TASK_A3.md` | **done** — see "Reproduction verdict" below |
| Track C | WP3a: native `fgo_gnss_lm_vd` backbone on raw tokyo run1 (PR+motion, then +Doppler), scored with dual-metric scorer | this repo `results/wp3a/`, spec `TASK_C.md` + `TASK_C2.md` | **done** — `WP3A_REPORT.md`; root-caused native cap `n_state>16384 → iters=-1` silent WLS passthrough; fixed with 1000-ep chunking |
| Track D | WP3b: Doppler robustness + multi-GNSS audit + IMU adapter | this repo `results/wp3b/`, spec `TASK_D.md` | **done** — `WP3B_REPORT.md`; backbone chain 94.52→90.04 (Huber Doppler) →85.82 m 2D (+loose IMU priors); GRECJ=28 sats median but *regresses* accuracy without elevation mask / inter-constellation bias calibration (coverage 100%); host LM solve is dense O(n_state³) → `--chunk-epochs 250` workaround, block-sparse solver = backlog |
| Track B | Dual-metric scorer `experiments/score_vs_inuex35.py` (+ 7 tests, passing) + shootout table | this repo, specs `TASK_B.md` + `TASK_B2.md` | **done** — corrected table below |
| Track E | WP3c: elevation mask + per-constellation weighting to make GRECJ beat GPS-only backbone | this repo `results/wp3c/`, spec `TASK_E.md` | **done** — `WP3C_REPORT.md`; documented negative: even with 20° mask + data-calibrated weights, multi-GNSS (97–111 m) cannot beat GPS-only (85.82 m). Root cause = time-varying BeiDou ISB in specific canyon segments (ep ~6500–7250), unfixable by constellation selection; ISB-stability prior = solver backlog. GPS-only variant (c) remains the best backbone; multi-GNSS buys 100 % coverage at ~25 m AllRMS cost |
| Track F | WP4: first full-run <50cm_full% of DD/LAMBDA `local_fgo` pipeline | this repo `results/wp4/`, spec `TASK_F.md` | **done** — `WP4_REPORT.md`; **<50cm_full% = 0.0%** (clean negative result, see below) |
| Track G | WP5: anchor `local_fgo` windows on libgnss++ RTK fixes + AR validation gates | this repo `results/wp5/`, spec `TASK_G.md` | **done** — `WP5_REPORT.md`; 24.5% vs 25.4% baseline (miss). Root cause quantified: RTK FIX = 775/11928 epochs (6.5%), ALL in the first 200 s → 55/60 windows have zero anchors; loose FLOAT priors net −105 epochs; DDPR gate blind (code noise 17 m vs cm-level fix shifts, 0/117 rejects) |
| Track H | WP6: raise the libgnss++ RTK FIX rate | `results/wp6/`, spec `TASK_H.md` | **done** — `WP6_REPORT.md`; winner `--max-pos-jump-rate 2.3`: run1 25.4→26.9 %, run2 36.1→**43.1 %** (+7 pp), run3 flat, all FixRMS ≤ 0.31 m, no regressions. Found 3 dead CLI knobs (arfilter/hold-ratio never read by rtk.cpp; v5's tuning flags were no-ops), and the front-load mechanism: stale `last_fixed_position_` + 5 m jump guard vetoed 99.4 % of 4437 ratio≥3.0-resolved epochs. Catastrophic wrong fixes = float-filter divergence in one canyon segment, internally indistinguishable (fixed≈float, good ratio) — only the adaptive jump gate discriminates. WP5-compounding recheck: anchored FGO (25.6 %) still below raw WP6 pos (26.9 %) |
| Track I | WP7: PLATEAU ray-traced NLOS weights wired into the RTK engine's DD weighting + properly wire the dead arfilter/hold-ratio knobs | `results/wp7/`, spec `TASK_I.md` | **done** — `WP7_REPORT.md`; **NLOS soft-weighting = clean negative**: all 10 sigma-inflation points regress run1 (26.64→13.7–22.1 %); best mapping (continuous floor 0.5) applied verbatim: run2 **+2.51 pp** (43.17→45.68 %, +519 fixes) but run1 −4.51 pp / run3 −6.40 pp → per-run sign flip, not shippable as default. Canyon (tow 188990–189070) untouched by any config (~119 m float error; only 18–23/~400 epochs get *any* solution, 0 FIX): σ-inflation ×2–×100 is an order of magnitude too weak vs 100 m-biased pseudoranges → **hard exclusion is the right tool**. Dead knobs now correctly wired + unit-tested (bit-identical off, SHA-256-bisected); surprise: `--preset low-cost` had always *claimed* arfilter/hold-ratio settings that the old code silently discarded — activating them = −0.28 pp on run1 (adopted as honest baseline). C++ suite 276/229 pass/0 fail, +14 nlos_weights tests, +4 smoke tests, Python +8 |
| Track J | WP8: NLOS hard exclusion + canyon forensics + live-knob retune | `results/wp8/`, spec `TASK_J.md` | **done** — `WP8_REPORT.md`; exclusion = clean negative (all candidates regress, FixRMS blown 4–8×; smaller sat set weakens AR's self-checking → wrong fixes; phase-33 mask is boolean-only so the threshold axis was inert); retune = near-miss (+0.277 pp with `--hold-ratio-threshold 2.0`, below the 0.3 pp bar; `--arfilter-margin` a complete no-op on this run); **centerpiece = canyon forensics, definitive code-cited root cause**: `resetPositionToSPP()` (rtk.cpp:1473) runs unconditionally every epoch and resets float position covariance to 900 m²/axis unless the previous epoch refreshed "trust" (`rememberSolution` rtk.cpp:3741 — FIXED, or FLOAT w/ ≥5 sats + small jump). Canyon: 73.5 % of epochs in the wide-reset regime (vs 0 % open-sky), NLOS residuals (median 20.5 m, max 641 m) corrupt SPP seed + DD update → trust never earned → self-reinforcing loop, **zero cross-epoch position memory for ~92 s**. Slips real but secondary; `adaptive_position_jump` rejects (63.6 %) a downstream symptom. This also explains why inuex35 wins: their IMU tight coupling supplies exactly the cross-epoch memory our filter discards each epoch. New float-covariance/NIS debug telemetry now in `--debug-epoch-log`. C++ 289/239/0 fail; Python +19 tests |
| Track K | WP9: fix the float-filter trust/reset policy (cv-predict / scaled-reset) | `results/wp9/`, spec `TASK_K.md` | **done** — `WP9_REPORT.md`; negative at the "single global config" level but **mechanism confirmed**: `scaled-reset qpos=0.1` fixes the canyon exactly as predicted (canyon AllRMS 125.6→74.7 m, run1 26.64→**27.42 %** +0.78 pp, FixRMS budgets kept) yet fails the 3-run gate (run2 −1.79 pp, run3 −1.33 pp) — root-caused: its dt=0 variance is a fixed 25 m² for *every* lapse, so run2/3's frequent short benign lapses suffer immediate overconfidence, and no qpos rescues both sides (qpos=100 ≈ legacy everywhere). `cv-predict` lost outright (finite-difference velocity collapses to ~0 between trust refreshes). `hold-ratio 2.0` alone = wash (WP8's +0.277 pp required margin 0.0/0.2 *together*, discrepancy documented). C++ 310/258/0 fail; clear WP10 shape: **gate scaled-reset on lapse duration / NLOS fraction instead of applying it unconditionally** |
| Track L | WP10: lapse-gated trust policy + `--nlos-min-los-sats` AR gate | `results/wp10/`, spec `TASK_L.md` | **done** — `WP10_REPORT.md`; negative on the 3-run gate (third in a row on this lever): `gate=2 s` gives run1 +1.065 pp but run3 −0.602 pp; NLOS-fraction trigger never fires in the canyon; `--nlos-min-los-sats` a clean run1 loss (−4 pp). Root cause: lapse duration/NLOS fraction don't correlate with help-vs-hurt — all 3 runs share a similar 32–42-segment lapse population; only post-gap re-acquisition geometry decides. **RTK-engine knob axis now exhausted → campaign pivoted to the TC-FGO port (Tracks M+)** |
| Track M | WP11: TC-FGO float skeleton (GNSS DDPR + IMU preintegration, 5-epoch numpy LM) | `results/wp11/`, `python/gnss_gpu/tc_fgo.py` | **done** — `WP11_REPORT.md`; **gate FAIL**: smoke 8.7 m @2k ep but full run1 AllRMS **12 148 m** (coverage 97.9 %); run2 1 273 m / run3 21 410 m; `<50cm_full%` 0.2 / 2.1 / 2.2 vs inuex35 56.7 / 69.9 / 67.9. Root cause = no AR, naive 0.2 m marginal prior, no recovery FSM; IMU propagation achieves coverage but not trustworthy cross-epoch memory. 6 Python tests |
| Track N | WP12a: stabilize float estimator (diagnostics-first, recovery FSM, anchors) | `results/wp12a/`, `experiments/wp12_run_tc_fgo.py` | **done** — `WP12A_REPORT.md`; **gate FAIL** but mechanism confirmed: recovery bug fixed (`max_shift_m=50` → 5000); post-fix probe **173 m** / full run1 **235 m**, `<50cm_full%` **7.1 / 9.5 / 7.7 %**. Huber+marginal σ=0.2 m overpower DD at km misclosure. 11 Python tests |
| Track O | WP12b: DD carrier + persistent ambiguities + LAMBDA validation on TC-FGO | `results/wp12b/`, `python/gnss_gpu/tc_fgo.py` | **done** — `WP12B_REPORT.md`; **both gates FAIL**: probe AllRMS **109.7 m** (persistent amb −0.7 m vs carrier-only), full run1 **228.6 m** / run2 69.3 m / run3 27.3 m; LAMBDA on ~110 m float = WP4 trap (80 % fix, FixRMS 23.7 m). Mechanism = **position cross-epoch memory** (naive marginal + 5-ep window), not AR; open-sky ep 500–800 RMS **0.44 m** proves float can work; window=25 → **1.9 m** AllRMS on 1k ep at 7× cost. 18 Python tests |
| Track P | WP12c: Schur-complement sliding-window marginalization (`--schur-marginal`) + window sweep + win25 decisive probe | `results/wp12c/`, `python/gnss_gpu/tc_fgo.py` | **done** — `WP12C_REPORT.md`; stage-1 gate FAIL but decisive mechanism finding: Schur now *provably correct* (closed-form + fixed-lag-equivalence tests, 24 Python tests) and makes open-sky float **0.04 m RMS** (10× better), yet every memory mechanism loses on the 4k probe (fixed Schur 137–156 m, win25-no-schur **125.4 m** vs 109.7 m reference — win25's 1k-probe 1.9 m does not survive the drift region). Recovery reseeds land at ~28 m (DD-code-LS quality there) and *stay*: **memory cannot manufacture absolute accuracy no factor possesses**. Note inuex35's own AllRMS is 47.4 m — they don't solve this either; they win via 49.5 % fixes at 0.815 m. Campaign reframed: stop chasing full-length AllRMS, target `<50cm_full%` directly |
| Track Q | WP12d: quality-gated LAMBDA AR on Schur float (cert → subset → DDPR → hold) | `results/wp12d/`, `python/gnss_gpu/tc_fgo.py` | **done** — `WP12D_REPORT.md`; stage-2 gate **FAIL** but feedback defect fixed (held-integer Jacobian sign/λ scaling); post-fix 1 351/3 900 probe epochs shift on re-solve, unit test cm convergence; `<50cm_full%` still **6.8 %** probe / **7.0 %** run1 (cert only passes where float already sub-meter); full run1 FixRMS **60 m** on drift wrong-fixes. **31 Python tests** |
| Track R | WP12e: dense RTK FIX+FLOAT anchoring + anchor-proximity cert calibration | `results/wp12e/`, `python/gnss_gpu/tc_fgo.py` | **done** — `WP12E_REPORT.md`; stage-2 gate **FAIL**; dense anchors cut probe AllRMS **156→67 m**, full run1 FixRMS **60→1.6 m**, but `<50cm_full%` **7.0→7.2 %** (FLOAT truth RMS **21 m** in drift); canyon 0 FIX / 115 m unchanged. **32 Python tests** |
| next | **PF/RBPF milestone 2** or **RTK-engine structural FIX/IMU coupling** (inuex35-shaped absolute core) — TC-FGO anchor axis exhausted | campaign doc §PF hybrid | recommended |

## Campaign closure (2026-07-10) — WP14 verified: the goal is met in libgnss++

**Verdict: inuex35 beaten, locally verified.** The gnssplusplus (libgnss++)
`FGOBackend::GTSAM` tightly-coupled GNSS/IMU backend (upstream develop@09fec9a,
submodule bumped in c9fbc75, our Phase18/WP9-10 carried as PR
[gnssplusplus-library#284](https://github.com/rsasaki0109/gnssplusplus-library/pull/284))
reproduces its README parity numbers **exactly (all 9 digits)** on a local
MSVC + GTSAM 4.3.0 build — full Tokyo runs, coverage 99.8–100 %:

| run | libgnss++ `<50cm` (2D) | fix % | FixRMS | inuex35 `<50cm` | inuex35 fix % |
|---|---:|---:|---:|---:|---:|
| run1 | **56.8** | **54.7** | 0.89 m | 56.7 | 49.5 |
| run2 | **80.5** | **78.0** | 0.63 m | 69.9 | 60.8 |
| run3 | **72.8** | **72.2** | 0.29 m | 67.9 | 59.4 |

Honest caveat (full detail in `results/wp14/WP14_REPORT.md`): the upstream
README's `<50cm` side-by-side compares its **2D-horizontal** metric against
inuex35's **3D** numbers. Under matched 3D definitions vs the same-machine
inuex35 repro: **fix-rate wins all 3 runs decisively** (identically defined:
54.7/78.0/72.2 vs 46.9/60.8/59.4); `<50cm` = run2 clear win (+7.4 pp),
run1/run3 −2.9/−1.9 pp (inside the known ±4.3 pp LAMBDA platform-variance
band = statistical ties). Bottom line: **fix-rate all-win + `<50cm`
1-win-2-tie under the strictest reading; full win as published.**

**The Python standalone campaign (WP13a–WP13s, `repro_tc_fgo/`)** rebuilt the
same machinery from scratch (DD-PR+DD-CP, IMU tight coupling, joint-marginal
AR, accept gate, slip resets, conditioned holds, recovery) and reached
best-per-run **36.3 / 51.1 / 63.3** `<50cm_full%` — **beating WP7 RTK
(25.4/43.2/43.7) on all three runs** — with cm-median fix purity and the
canyon fully cracked (747 fixes, ≤1 false). Its 19 work-package diagnostic
chain (reports under `repro_tc_fgo/results/wp13*/`) is what localized the
mechanisms that the C++ backend's urban stack embodies. NLOS priors were
**empirically falsified at all three injection layers** (WP7/WP8/WP13b) —
the city-model edge thesis is retired.

Next (2026-07-10): WP15 CUDA acceleration of Python hot paths (in flight);
optional Python parity levers V9/B14 (recovery-float) if the standalone is
pursued to full tc/ parity.

## Campaign closure (2026-07-07, superseded by the 2026-07-10 closure above)

**Verdict: inuex35 not beaten on `<50cm_full%`.** Best self-contained TC-FGO stack (WP12e) vs inuex35 README targets:

| run | inuex35 `<50cm_full%` | WP7 RTK | **WP12e TC-FGO** | gap (pp) |
|---|---:|---:|---:|---:|
| run1 | **56.7** | 25.4 | **7.2** | −49.5 |
| run2 | **69.9** | 43.2 | **11.7** | −58.2 |
| run3 | **67.9** | 43.7 | **5.9** | −62.0 |

**What we built (Tracks M–R):** end-to-end Python TC-FGO port — `python/gnss_gpu/tc_fgo.py` + `experiments/wp12_run_tc_fgo.py`, IMU preintegration, Schur marginalization, recovery FSM, persistent ambiguities, quality-gated LAMBDA (subset-AR / DDPR / hold), dense RTK anchoring. **32 unit tests passing.** Open-sky float **0.04 m RMS**; cert-tight AR FixRMS **0.40 m** on probe.

**What we learned (mechanism chain, all code-verified):**

1. **RTK-engine axis (WP6–WP10):** knob tuning cannot raise FIX supply; canyon trust-reset loop (`resetPositionToSPP`) needs IMU coupling, not more flags.
2. **TC-FGO float (WP11–WP12c):** coverage 100 % achievable; drift tail needs recovery + honest memory; Schur marginal is correct but cannot create absolute accuracy no factor possesses (~28 m DD-code floor in degraded geometry).
3. **AR (WP12b–WP12d):** LAMBDA on biased float = self-consistency trap (WP4 replay); quality gating + Jacobian fix makes AR honest but cert fires only where float is already sub-meter.
4. **Anchors (WP12e):** FIX+FLOAT densification improves AllRMS and FixRMS purity but `<50cm_full%` immobile — FLOAT truth RMS **21 m** in drift; cert correctly refuses AR in canyon (0 FIX, 115 m).

**inuex35's actual edge:** **49.5 % of epochs FIX at 0.815 m** from IMU-tight RTK+AR with cross-epoch memory — not FGO sophistication. Their AllRMS is 47.4 m; we were wrong to treat AllRMS < 20 m as the campaign gate.

**Recommended next campaign (outside this closure):**

- **Option A — PF/RBPF milestone 2** ([`internal_docs/inuex35_tc_fgo_benchmark.md`](internal_docs/inuex35_tc_fgo_benchmark.md) original milestone 2): multi-modal float through canyon memory loss; unique asset vs inuex35.
- **Option B — RTK structural:** IMU-tight coupling inside libgnss++ (replace per-epoch SPP reset with propagated state); re-use WP12e TC-FGO as scorer only.
- **Do not continue:** TC-FGO anchor/AR tuning, RTK knob sweeps, or naive LAMBDA on degraded float.

Reports: `results/wp{10,11,12a,12b,12c,12d,12e}/WP*_REPORT.md`.

**Campaign insight (2026-07-06)**: the decisive gap vs inuex35 is not FGO
sophistication — it is **RTK FIX supply** (theirs 49.5 % of epochs at 0.815 m
FixRMS; ours 6.5 % at 0.048 m, front-loaded in the open-sky start). Our fixes
are 17× purer; we can afford to trade purity for quantity. WP5's anchoring
machinery is built, tested, and waiting — it becomes useful exactly when WP6
delivers fixes spread across the timeline.

## WP4 negative result (2026-07-05) — why DD/LAMBDA over an SPP seed cannot work

Full-run local_fgo+LAMBDA over the WP3b backbone seed: AllRMS 107.68 m →
107.68 m (identical to the seed to 2 dp), <50cm_full% = 0.0 despite "fixing"
on 86 % of epochs. `results/wp4/WP4_REPORT.md` bottlenecks:
1. **Seed absolute quality dominates** — local_fgo is a refinement layer; its
   only absolute pull is endpoint priors tied to the same ~100 m-biased seed.
   inuex35's absolute core is DD/carrier RTK (`cssrlib rtkpos`) from phase 1.
2. **No independent motion/IMU constraint** in the window graph (motion prior
   is derived from the seed itself — zero new information).
3. **The LAMBDA path is a self-consistency check**: true multi-sat group
   search fired 6× in the whole run vs 26,726 single-track ratio-test accepts
   (median segment 3 epochs; ratio values up to 2×10⁹ on a biased seed). No
   subset-AR / valpos / DDPR cross-validation like inuex35 has.
Also found: `solve_ppc_segment_multifamily_fgo.py`'s import chain was broken
(never ran end-to-end before) and `exp_ppc_ctrbpf_fgo.py`'s pos writer/reader
are column-misaligned (seed round-trip returned empty dict).

Results tables from Track A/B get appended here as they land.

## Track B baseline shootout (corrected, 2026-07-04)

Scorer fixes: libgnss++ `.pos` FIX = Status==4 (default `--fix-statuses 4`; RTKLIB
Q files use `--fix-statuses 1`). Added `<50cm_full%` (full rover-epoch denominator;
missing epochs count as failures). AllRMS remains over scored epochs only; coverage
reported explicitly.

| method | run | n_scored | coverage% | AllRMS | FixRMS | fix% | <50cm% | <50cm_full% | ppc_official% |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| inuex35 README (external baseline) | run1 | n/a | 100.0 | 47.40 | 0.815 | 49.5 | 56.7 | 56.7 | n/a |
| inuex35 README (external baseline) | run2 | n/a | 100.0 | 32.08 | 0.277 | 60.8 | 69.9 | 69.9 | n/a |
| inuex35 README (external baseline) | run3 | n/a | 100.0 | 34.52 | 0.211 | 59.4 | 67.9 | 67.9 | n/a |
| libgnss_ctrbpf_pos/tokyo_run1_RBPF-velKF+DD+gate+hybrid.pos | run1 | 1200 | 10.1 | 12.04 | 12.042 | 100.0 | 29.5 | 3.0 | 36.73 |
| libgnss_ctrbpf_pos/tokyo_run1_RBPF-velKF+DD+gate.pos | run1 | 1200 | 10.1 | 53.98 | 53.979 | 100.0 | 0.0 | 0.0 | 0.00 |
| libgnss_ctrbpf_pos/tokyo_run1_RBPF-velKF+DD.pos | run1 | 120 | 1.0 | 44.45 | 44.448 | 100.0 | 0.0 | 0.0 | 0.00 |
| libgnss_ctrbpf_pos/tokyo_run2_RBPF-velKF+DD+gate+hybrid.pos | run2 | 1200 | 13.1 | 16.99 | 16.988 | 100.0 | 9.2 | 1.2 | 4.67 |
| libgnss_ctrbpf_pos/tokyo_run3_RBPF-velKF+DD+gate+hybrid.pos | run3 | 1200 | 7.8 | 24.09 | 24.090 | 100.0 | 41.4 | 3.2 | 33.63 |
| libgnss_rtk_pos_v5/tokyo_run1_full.pos | run1 | 7397 | 62.0 | 19.86 | 0.084 | 10.5 | 40.9 | 25.4 | 22.88 |
| libgnss_rtk_pos_v5/tokyo_run2_full.pos | run2 | 6466 | 70.7 | 9.31 | 0.049 | 12.6 | 51.1 | 36.1 | 43.47 |
| libgnss_rtk_pos_v5/tokyo_run3_full.pos | run3 | 12833 | 83.9 | 5.28 | 0.048 | 6.4 | 52.1 | 43.7 | 40.66 |
| libgnss_rtk_wave2/* (all configs; identical to v5) | run1 | 7397 | 62.0 | 19.86 | 0.084 | 10.5 | 40.9 | 25.4 | 22.88 |
| libgnss_rtk_wave2/* (all configs; identical to v5) | run2 | 6466 | 70.7 | 9.31 | 0.049 | 12.6 | 51.1 | 36.1 | 43.47 |
| libgnss_rtk_wave2/* (all configs; identical to v5) | run3 | 12833 | 83.9 | 5.28 | 0.048 | 6.4 | 52.1 | 43.7 | 40.66 |
| pf_nlos_oracle_hybrid/tokyo_run1_full.pos | run1 | 11951 | 100.2 | 0.00 | n/a | 100.0 | 100.0 | 100.2 | 100.00 |
| pf_nlos_smoke_pos/tokyo_run1_RBPF-velKF+DD+gate+hybrid+rtkdiag_pf.pos | run1 | 1200 | 10.1 | 104.92 | 104.919 | 100.0 | 0.0 | 0.0 | 0.00 |

**Key takeaway:** libgnss RTK run3 still beats inuex35 on AllRMS (5.28 m vs 34.52 m)
at 83.9% coverage, but `<50cm_full%` (43.7%) is well below inuex35's 67.9% once
missing epochs are counted as failures. FixRMS for libgnss RTK is now sensible
(0.05–0.08 m on fixed epochs) with status==4 fix% at 6–13%.

## Relation to existing PPC work

The PPC selector/ranker production line (Phase71, 86.21 % OFFICIAL — see
[`ppc_current_status.md`](ppc_current_status.md)) is a *different* game: it
ensembles many external candidate sources. This campaign is about a **single
self-contained estimator** beating a single self-contained estimator on the
same raw inputs. GICI-derived numbers (95–100 % <1 m) remain reference-only
(GPL-3.0 — see development policy in README): they must not seed or tune our
solver, but they do show the dataset's headroom.
