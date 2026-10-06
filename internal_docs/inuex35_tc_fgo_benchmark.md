# inuex35 tightly-coupled-gnss-imu-fgo — benchmark target & comparison

Last updated: 2026-07-11 (RB-FGO-PF milestone 2).

This is the working document for the campaign to **beat
[inuex35/tightly-coupled-gnss-imu-fgo](https://github.com/inuex35/tightly-coupled-gnss-imu-fgo)
on the shared Tokyo PPC benchmark**. Sequencing decided 2026-07-04:

1. **Polish standalone FGO first** (GNSS+IMU tight coupling — `fgo_gnss_lm_vd`
   full stack: DD + IMU preintegration + LAMBDA AR) until it beats their numbers.
2. **Then PF+FGO hybrid** as the second milestone.

Win criterion: side-by-side on **both metric families** (their
AllRMS/FixRMS/fix%/<50cm *and* our `ppc_score` OFFICIAL%), same runs, same
epochs, coverage reported honestly. Headline metric: **<50cm%**.

## RB-FGO-PF milestone 2 (2026-07-11) — inuex35 beaten

Our Rao-Blackwellized particle filter over integer-ambiguity basins beats the
inuex35 README result on all three Tokyo PPC runs using the matched-3D official
scorer:

| `<50cm_full%` (3D) | run1 | run2 | run3 |
|---|---:|---:|---:|
| inuex35 README | 56.7 | 69.9 | 67.9 |
| libgnss++ GTSAM (2D reference) | 56.8 | 80.5 | 72.8 |
| **RB-FGO-PF (ours, 3D)** | **59.6** | **78.7** | **78.1** |

The runtime is in `experiments/rbpf_fgo/`: a verbatim copy of `repro_tc_fgo`
at `fa57b7f` with pinned-dependency setup, the ship command and a copy check.

Shipped FixRMS is 0.062/0.042/0.059 m, median fixed error is about 3 cm,
fix rate is 37.8/71.1/69.4%, and PPC OFFICIAL is 57.8/80.0/82.2%. The canyon
probe produced 283 fixes and no false fix. The shipped result uses output
guards `RBPF_FIX_VOTE_DD=6` and `RBPF_FIX_VOTE_DPR=3.5` and, since
2026-10-06, the `nb >= 12` report floor (see below; it was `nb >= 9`).

Under the original `nb >= 9` floor, the run 3 false-fix rate was 3.09%, above
the approximately 2% target, due to 281 false fixes in a coherent 0.53 m shift
at tow `[179700,179900)`. Posterior confidence is approximately 0.9999, while
DDPR is affected by the roughly 1 m multipath-bias floor; DDPR hold-release
and challenger spawns proved unshippable. Per-cluster relinearization (WP19)
and the later WP20–37 campaign did not produce a shippable fix either; see
`repro_tc_fgo/results/wp19`–`wp37`.

### Report floor raised to `nb >= 12` (2026-10-06)

The floor relabels a fixed epoch as float when fewer than `nb` ambiguities are
resolved; positions do not move, so `<50cm_full%` and PPC OFFICIAL are
unchanged. WP18 declined `nb >= 11` because picking a floor per run after the
fact is cherry-picking. The floor was re-selected leave-one-run-out on the
saved WP18 full runs (`repro_tc_fgo/results/wp18/full_r{1,2,3}/run*.npz`):
for each held-out run, the floor in {9..14} that maximises
`kept good fixes − λ · kept false fixes` on the other two runs.

- Testing on run1 or run2 (run3 in training): `nb >= 12` for λ from 5 to 20;
  at λ = 50, `nb >= 12` (run1) and `nb >= 13` (run2).
- Testing on run3 (only run1/run2 in training, which hold 51 false fixes in
  total): `nb >= 9` for λ ≤ 10, `nb >= 10` at λ = 20, `nb >= 13` at λ = 50.
  The run3 failure is not visible from the other two runs unless a false fix
  is weighted as about 50 good ones.
- Simple support rules from the per-epoch LAMBDA (`ar_raw_nb`, `ar_ratio`)
  were in the search and never chosen over the plain floor.

Official scorer (`experiments/score_vs_inuex35.py`, 3D), `nb >= 9` → `nb >= 12`
(relabelled with `results/wp18/relabel_nb9_pos.py <pos> <npz> <out> 12`):

| run | fix % | FixRMS | false-fix % | `<50cm_full%` | PPC OFFICIAL |
|---|---:|---:|---:|---:|---:|
| run1 | 43.1 → 37.8 | 0.104 → 0.062 m | 0.59 → 0.29 | 59.6 | 57.80 |
| run2 | 75.1 → 71.1 | 0.121 → 0.042 m | 0.37 → 0.00 | 78.7 | 80.02 |
| run3 | 73.7 → 69.4 | 0.150 → 0.059 m | 3.09 → 0.05 | 78.1 | 82.19 |

This fixes the integrity of the reported fixes, not their number: the 0.53 m
block is still output as a 0.53 m position, now labelled float.

### Outputting the FLOAT position on disagreement: negative (WP38, 2026-10-06)

The float error in that block is lower (0.48 m vs 0.53 m), so WP38 tested
outputting the FLOAT position whenever it disagrees with the shipped FIX.
`repro_tc_fgo` now saves the float ECEF (`flt_xyz`) per epoch. The three WP18
runs were rerun; their shipped output was bit-identical. Rules of the form
"switch when the 3D / horizontal / vertical `|fix - float|` > tau (0.2–1.5 m)
and `nb <= nbmax`" were selected leave-one-run-out.

- Oracle ceiling (switch exactly the false fixes whose float is < 0.5 m):
  +0.50 / +0.14 / +2.08 pp `<50cm_full%`.
- Leave-one-run-out: −8 / 0 / +2 epochs (−0.07 / 0.00 / +0.01 pp). The best
  in-sample rule (horizontal `d > 1.5 m`) is net −6 epochs.
- Reason: in the block, FIX and FLOAT agree to 0.09 m median. The float
  carries the same bias and only sits just under the 0.5 m threshold.
  Correct fixes disagree with float by 0.1–0.3 m at p90 and about 1 m at p99.

The 0.53 m block is a bias common to the float and the fixed solution. Moving
it needs a different measurement or bias model, not a FIX/FLOAT selection rule
(`repro_tc_fgo/results/wp38/WP38_REPORT.md`).

### Per-satellite cause, NLOS-AR exclusion and seed spread (WP39, 2026-10-06)

**Cause.** DD code and carrier residuals per satellite at the reference
position (`repro_tc_fgo/wp39_block_sat_residuals.py`) trace the block to two
satellites going NLOS. They pass the existing gates (elevation 15°, CNR
25 dB-Hz, 4 m code FDE).

- E04 f1/f2: elevation 56° but SNR 27–29 dB-Hz, about 4.4 m of DD code
  error, and a carrier fraction shift of −0.4 to −0.5 cycles.
- G09 L1: normal code, but a +0.47-cycle carrier shift.

The PLATEAU ray-traced mask agrees: both satellites are LOS before the block
and NLOS in it. The bias starts when the DD satellite count drops to 7–10
(tow about 179735) and lasts until the near-total outage at about 179843.

**Oracle mask.** The existing PLATEAU masks
(`gnss_gpu/experiments/results/plateau_nlos_phase33`) were ray-traced at the
ground-truth position (`receiver_source=reference`), so results that use them
are not admissible. A non-oracle mask was ray-traced at the WP18 output
position (`build_per_epoch_nlos_csv.py --receiver-pos-file`). It agrees with
the oracle on 94.2 / 99.0 / 98.7% of satellite-epochs. The runtime reads it via
`NLOS_MASK_DIR`.

**NLOS layers.** First 2400 run3 epochs, oracle mask, inside the block:

- Excluding NLOS satellites from AR (`NLOS_AR_EXCLUDE=1`): false fixes
  261 → 1.
- Inflating their sigma 5x (`NLOS_MEAS_MODE=2`): 261 → 1.
- Dropping their DD factors (`NLOS_MEAS_MODE=1`): unchanged.

With the non-oracle mask over full runs, the sigma inflation loses 1.4 pp
OFFICIAL on run3 and is rejected.

**Seed spread.** The WP18 configuration and AR exclusion with the non-oracle
mask, 3 seeds each (20260710 shipped, 20260711, 20260712), official scorer,
`nb >= 12`. Values are mean ± sample std:

| run | metric | WP18 | + NLOS-AR exclusion | difference per seed |
|---|---|---:|---:|---|
| run1 | PPC OFFICIAL | 56.69 ± 1.22 | 59.64 ± 0.67 | +1.90 / +2.07 / +4.88 |
| run1 | `<50cm_full%` | 58.80 ± 1.39 | 60.75 ± 0.28 | +1.48 / +1.01 / +3.35 |
| run2 | PPC OFFICIAL | 81.22 ± 1.35 | 82.05 ± 1.24 | +1.03 / −1.01 / +2.48 |
| run2 | `<50cm_full%` | 78.36 ± 0.63 | 78.80 ± 0.54 | −0.13 / −0.32 / +1.79 |
| run3 | PPC OFFICIAL | 83.57 ± 1.33 | 83.87 ± 0.51 | +1.95 / −1.56 / +0.49 |
| run3 | `<50cm_full%` | 81.15 ± 2.69 | 82.39 ± 0.83 | +3.39 / −0.70 / +1.04 |

FixRMS means are 0.23 / 0.044 / 0.060 m for WP18 and 0.24 / 0.073 / 0.072 m
with exclusion. False-fix means are 0.59 / 0.01 / 0.12% and 0.77 / 0.13 / 0.20%.

- **The 0.53 m block is seed-specific.** WP18 false fixes inside it are
  261 / 0 / 0 across the three seeds. NLOS E04/G09 make a stochastic basin lock
  possible, and the shipped seed happens to hit it. The shipped single-seed
  numbers carry about ±1–3 pp of seed spread; for example, run3 `<50cm_full%`
  is 78.1 / 83.2 / 82.1 across seeds. Even the three-seed mean stays above
  inuex35 on every run.
- **NLOS-AR exclusion helps reliably only on run1** (+2.95 pp OFFICIAL, every
  seed positive). The run2/run3 gains (+0.83 / +0.30) are inside the spread.
  It costs 1–3 cm of FixRMS and +0.1–0.2 pp of false fixes, and it needs
  PLATEAU data plus a first pass. It is kept as an optional map-aided setting;
  the default is unchanged.

Comparisons of RB-FGO-PF variants need several seeds
(`repro_tc_fgo/results/wp39/WP39_REPORT.md`).

### Map-free NLOS flag from the C/N0 deficit: negative (WP40, 2026-10-06)

WP40 replaced the PLATEAU mask with a single-pass, map-free flag. A satellite
is flagged when the rover's C/N0 is more than 8 dB below the open-sky base
station's for the same satellite and epoch, averaged over the tracked
frequencies. The threshold is shared by all runs and matches the mask's NLOS
rate (24–30%). The flag recovers 69–80% of the mask's NLOS sat-epochs and
also flags E04/G09 in the run3 block. It feeds the same AR exclusion.

Over three seeds the mean PPC OFFICIAL is 56.21 / 81.88 / 83.89. That compares
with 56.69 / 81.22 / 83.57 for WP18 and 59.64 / 82.05 / 83.87 with the PLATEAU
mask. The run1 gain does not carry over, and run2/run3 stay inside the seed
spread, so the flag is not adopted
(`repro_tc_fgo/results/wp40/WP40_REPORT.md`).

### Real time, N = 128, and cluster shadows over three seeds (WP41–42, 2026-10-07)

- **Real time** (5 Hz data): with one process on an idle machine, N = 64
  runs at 13.4 synced epochs/s (2.7x real time) and N = 128 at 9.6 epochs/s
  (1.9x). These are averages; worst-case latency was not measured. Of an
  epoch, the PF takes 59% (likelihood 27%), and the ISAM2 update itself only
  2.6%.
- **Batched or GPU likelihood does not pay at N = 64.** Only about 21
  distinct hypotheses reach the likelihood per epoch. A numpy-batched version
  matched the scalar one to 4e-14 but was about 20% slower.
- **N = 128 on run3, 3 seeds:** it removes the shipped-seed lock (block false
  fixes 0 / 3 / 0), but the mean is unchanged (OFFICIAL 83.62 vs 83.57).
- **Cluster-conditioned shadows (WP19–22), 3 seeds:** the 3-run mean
  OFFICIAL drops from 73.83 to 70.69, and every seed is worse on runs 1 and 3.
  The WP22 rejection stands.

The open direction is a real factor graph per top basin (its own holds and
history), not a position memory; it is designed but not started. The handoff
note, together with seed-queue, scoring and aggregation tools, is in
`repro_tc_fgo/results/wp42/WP42_REPORT.md` and `repro_tc_fgo/tools/`.

Gamma is calibrated where coherent multipath shifts are absent (96.3–97.1%
full-scale accuracy for gamma >= 0.99), not universally perfect. Run 1 AllRMS
is 19.5 m because of its tunnel float tail; its fixed layer is unaffected. The
CUDA batch-LAMBDA path is 0.71x at batch size one because the architecture
normally submits one shared problem per epoch, so CPU remains the default.
FFBSi was descoped because basin-lineage smoothing requires new design work.

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
