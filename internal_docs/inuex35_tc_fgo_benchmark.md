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

### Tracked lineage beliefs and dead-reckoned gap fill (WP43, 2026-10-10)

WP43 is the new shipped configuration. It changes only what is written out:
the particle weights, holds and FIX decisions are identical to WP18. The
work ran in `experiments/rbpf_fgo/` (no longer a verbatim copy). The
platform was Linux, Python 3.12, 4 cores and 16 GB, using the PPC data from
the official OneDrive archive.

**Where WP18 loses epochs.** `error_budget.py` splits the full denominator.
Three-seed means, as % of rover epochs:

| run | FIX ok | FLOAT ok | false FIX | FLOAT, MAP basin ok | FLOAT lost | no output |
|---|---:|---:|---:|---:|---:|---:|
| run1 | 35.35 | 24.75 | 0.16 | 3.69 | 27.70 | 8.35 |
| run2 | 64.08 | 14.83 | 0.11 | 3.38 | 8.72 | 8.88 |
| run3 | 68.54 | 13.32 | 0.07 | 2.91 | 9.72 | 5.43 |

- **No output (5–9%).** These are rover epochs the graph abandons (fewer
  than 6 satellites, GDOP, solve failure). The scorer counts them as
  failures.
- **FLOAT lost.** These are mostly segments with 6–15 DD pairs where the
  shared float is 1–2 m off. In run1, about 15 pp is the tunnel, which has
  no carrier at all.
- **Diagnostic on run2's float-bad epochs.** `RBPF_DBG4` was extended with
  the candidate Hamming distance and the held basin. Over the first 6,200
  epochs, the truth basin is among the LAMBDA top-K in 54% of float-bad
  epochs. Its single-epoch likelihood beats the best held basin in only
  20%. The loss is the measurement model under multipath, not candidate
  supply.

**Mechanism.** Each particle lineage carries a Gaussian belief of the
antenna position (`RBPF_TRACK=2`). The belief is predicted between epochs
with the shared graph's IMU motion increment (an odometer of committed
epochs; a PIM break or a gap over 2 s resets it). It is updated with the
epoch's DD rows under the lineage's own integers. The model is the WP18
likelihood (Huber, mixture, group nuisance) with the belief as the prior
instead of the 1 m float prior. Empty lineages carry a code-only belief
(`RBPF_TRACK_EMPTY=1`).

The reported position (`RBPF_TRACK_OUT=2`) is the MAP basin's mass-weighted
belief whenever one exists, on FIX and FLOAT epochs alike. On rover epochs
the graph abandons, `--dr-fill 1` outputs the last reported position plus
the IMU-predicted antenna motion (smode 6, written as float).

This is not the WP19–22/WP42 cluster shadow. That fed a remembered position
into the weights and centred the next epoch's prior on the previous
absolute position, without the vehicle's motion. WP43 predicts with the IMU
and leaves the weights alone.

**Result.** Official matched-3D scorer, `nb >= 12` floor, seeds
20260710/11/12. WP18 comes from shadow runs whose FIX labels and `nb` were
checked identical to the WP43 runs, with the gap-fill rows removed and the
float positions kept. Values are mean ± sample std:

| run | metric | WP18 | WP43 | per seed |
|---|---|---:|---:|---|
| run1 | `<50cm_full%` | 60.10 ± 1.83 | **63.07 ± 1.37** | +3.94 / +2.34 / +2.64 |
| run1 | PPC OFFICIAL | 57.13 ± 2.46 | **63.68 ± 1.75** | +7.99 / +5.51 / +6.14 |
| run2 | `<50cm_full%` | 78.91 ± 1.12 | **82.65 ± 1.22** | +4.06 / +3.48 / +3.67 |
| run2 | PPC OFFICIAL | 82.22 ± 1.56 | **85.08 ± 1.28** | +2.75 / +2.60 / +3.22 |
| run3 | `<50cm_full%` | 81.86 ± 0.33 | **86.65 ± 0.24** | +4.81 / +4.86 / +4.69 |
| run3 | PPC OFFICIAL | 83.47 ± 0.66 | **86.94 ± 0.27** | +3.02 / +3.75 / +3.64 |

- **Every seed improves on every run.** WP43 is above libgnss++ GTSAM
  (56.8 / 80.5 / 72.8) and inuex35 (56.7 / 69.9 / 67.9) on all three runs.
- **FIX quality.** The FIX labels do not change. The tracked FIX positions
  are slightly better: FixRMS 0.178 / 0.073 / 0.055 m against
  0.187 / 0.075 / 0.060 m, and false fixes 0.33 / 0.16 / 0.10% against
  0.47 / 0.17 / 0.11%.
- **Fix rate over scored epochs.** It drops (35.5 / 64.2 / 68.6%) only
  because the gap-fill rows enlarge the denominator. The FIX count is
  unchanged.
- **The two parts add up.** Tracked output alone gives +1.42 / +2.10 / +2.57
  pp `<50cm_full%`; the gap fill alone gives +1.32 / +1.14 / +2.10 pp.
- **Knobs.** They were chosen on seed 20260710 with `RBPF_TRACK_VARIANTS`
  (several shadow trackers on one filter):
  - Random walk: 0.02 m per epoch beat 0.01, 0.05 and 0.1.
  - Posterior floor: 0.01 m beat 0.05.
  - Tracking the empty lineages beat not tracking them on all three runs.

  Seeds 11 and 12 were then run once with the chosen setting.
- **Offline alternative.** On top of the gap fill, interpolating the
  FIX–float offset over FLOAT epochs (`output_policies.py interp`, which
  uses future epochs) adds +1.3 pp. The causal tracked output adds
  +2.1–2.4 pp in the same comparison (without the empty lineages).

**Not adopted.**

- **`RBPF_TRACK=1`: tracked beliefs also inside the weights.** Every particle
  x move is solved in one batched kernel (`_loglik_batch`, which matches the
  scalar likelihood to 6e-10), about 400–500 solves per epoch instead of
  about 21. On run1 (seed 20260710) it raised the fix rate from 37.3% to
  43.3%. It also locked a 0.72 m wrong basin after the tunnel: false fixes
  2.39% against 0.56%, `<50cm_full%` 58.20 against 58.89. A lineage's tight
  belief gives an incumbent basin an Occam advantage of about 10 nats per
  epoch over a fresh adoption, so the first basin to enter wins.
- **N = 256 particles (seed 20260710).** With the WP43 output, run1
  scores 62.06% / 63.72 against 62.83% / 63.60 at N = 64, and run2 scores
  81.72% / 85.04 against 83.65% / 86.25. The PF time is 3.5x (210 against
  60 ms per step on run1; 311 against 89 ms on run2, both runs carrying six
  shadow variants).

**Real time.** On 3,000 run2 epochs in a single process: WP18 runs at 11.6
epochs/s and WP43 at 10.7 epochs/s, both above 2x real time at 5 Hz. The PF
step goes from 55.5 to 62.6 ms, with about 37 belief updates per epoch.

**GPU.** The WP43 gain does not need more compute. More particles do not
help, and the compute-heavy `RBPF_TRACK=1` hurts, so a GPU port is still not
justified by accuracy. The batched kernel is where one would start if a
future variant needs thousands of hypothesis solves per epoch.

Reproduction: `experiments/rbpf_fgo/run_ship.sh` (now WP43 by default;
`RBPF_SHIP=wp18` for the old output), `run_queue.sh`, `score_runs.py`,
`output_policies.py`, `error_budget.py`.

Gamma is calibrated where coherent multipath shifts are absent (96.3–97.1%
full-scale accuracy for gamma >= 0.99), not universally perfect. Run 1 AllRMS
is 19.5 m because of its tunnel float tail; its fixed layer is unaffected. The
CUDA batch-LAMBDA path is 0.71x at batch size one because the architecture
normally submits one shared problem per epoch, so CPU remains the default.
FFBSi was descoped because basin-lineage smoothing requires new design work.

### FIX rate at a wrong-FIX budget (WP44–WP45, 2026-10-11/12)

**Goal.** Raise the correct-FIX share of rover epochs while every run keeps
its wrong-FIX share (3D error ≥ 0.5 m) at or below 1% of its fixes.
`fix_integrity.py` scores every system the same way.

**Result (shipped WP45, one configuration for all runs, 3 seeds).**

| Correct FIX (wrong / FIX) | run1 | run2 | run3 |
|---|---:|---:|---:|
| inuex35 TC-FGO (local reproduction) | 46.7% (5.72%) | 59.7% (1.78%) | 58.2% (2.06%) |
| libgnss++ GTSAM FGO (`gnss_fgo_parity`) | 37.7% (29.9%) | 77.3% (1.57%) | 68.2% (1.56%) |
| libgnss++ KF-RTK, raw | 68.0% (8.76%) | 76.7% (1.80%) | 71.6% (3.01%) |
| libgnss++ KF-RTK, in-sample gate to ≤ 1% | 52.8% (0.97%) | 76.4% (0.65%) | 68.3% (0.96%) |
| libgnss++ KF-RTK, LORO gate (cap 0.7%) | 61.6% (2.66%) | 58.3% (0.13%) | 59.8% (0.17%) |
| RB-FGO-PF WP43 | 35.4% (0.33%) | 64.1% (0.16%) | 68.5% (0.10%) |
| **RB-FGO-PF WP45** | **66.6 ± 1.0% (0.56%)** | **83.9 ± 0.2% (0.09%)** | **84.3 ± 0.1% (0.04%)** |

Worst seed wrong/FIX: 0.63 / 0.17 / 0.06%. PPC OFFICIAL 80.8 / 92.7 /
90.4%, `<50cm_full%` 80.1 / 90.2 / 94.7%. The KF-RTK baseline is
`ppc-demo --solver rtk --preset low-cost --ratio 2.4
--max-subset-ar-drop-steps 18 --rtk-snr-weighting --no-arfilter`. Its gates
use only the `.pos` telemetry (ratio, NumSat, NIS, outliers).

**What changed.**

1. **Satellite set (the main gain).** The runner asked cssrlib for three
   bands per system. cssrlib's `qcedit` drops a satellite when any of the
   requested bands fails, and the DD factors use only two. So every
   satellite without L5, and every BeiDou satellite without B1C (BDS-2,
   most IGSO/GEO), was discarded. On the run1 epochs the PF missed while
   libgnss++ KF-RTK fixed them, the KF used a median 15 satellites and the
   graph 7–8. `--nf 2 --band-priority 'C:2,6;R:-'` (BeiDou B1I + B3I,
   GLONASS off) raises one sampled run1 epoch from 15 to 25 satellites.
   With the old label (`nb >= 12`) run1 goes from 35.4% to 58.7% correct
   FIX, and the wrong-FIX share falls to 0.00%. `--nf 2` with the default
   B1C + B1I gives 44.1%.
2. **Basin persistence (`RBPF_KEEP=1`).** Graph re-seeds no longer empty
   the basins; only slip / MW / CMC / FDE-on-carrier releases do.
3. **FIX label rule.** The label is
   `nb_at[0.95] >= 5 or (nb_at_ne[0.95] >= 12 and ne_mass >= 0.1) or
   (base LAMBDA fix with >= 5 ambiguities)`, behind the PF output guards.
   `nb_at[t]` counts the MAP basin's pairs whose agreement mass across all
   particles exceeds t. `nb_at_ne` is the same over the non-empty particles,
   renormalised, and `ne_mass` is their mass. `fix_rule_loro.py` selects
   the rule on two runs and applies it to the third; all three folds pick
   this rule. `--fix-rule` applies it in the runner and matches the offline
   labels row for row.

The run1-only recovery flags of WP43 are not used: with the sanity reset on
run1 the three seeds score 67.1% (0.54%), without it 66.6% (0.56%), and
the same flags on run2/run3 cost 0.1–2.3 pp.

**Not adopted.**

- **`RBPF_TRACK=1` with a probation window** (score on the loose float prior
  for 20 s after a belief reset): run1 41.0% (1.17%) against 42.2%
  (0.26%) under the same label grid. Without the window 41.1% (1.70%).
- **Ambiguity-function label (`RBPF_AFV=1`):** FIX when ≥ 95% of the
  epoch's DD carrier rows sit within 0.15 cycle of an integer at the
  reported position. With ≥ 12 rows it adds +0.9 / +1.6 / +0.8 pp before
  WP45 and +0.3–0.6 pp after it. With 8–10 rows run3 reaches 1.56% wrong:
  the held rows are integer at the position solved from them, so with few
  rows a self-consistent wrong integer set passes. Kept as a diagnostic.
- **Independent-satellite witness (Fredeluces et al. 2024):** carrier rows
  of satellites outside the MAP basin. Too few per epoch (0.4–1.7) to
  separate correct from wrong candidates. libgnss++'s own audits reject
  the same idea, for the same shared urban multipath basin.
- **Lower `nb` floors scaled by the number of pairs:** at most +2.5 pp on
  run1, with +60–140 wrong fixes per run.
- **External KF-RTK as a FIX witness:** labelling FIX where libgnss++
  KF-RTK fixes within 0.2 m of the PF position gave 51.7 / 74.8 / 74.5%
  before WP45. That is a two-engine system, not the PF. Used only to locate
  the run1 losses that led to item 1.
- **Elevation mask 10°** via `--env ELMIN_DEG=10`: bit-identical output, so
  the setting does not reach the filter. Not investigated further.

**Literature and OSS read for this round.** PPC2024 results (1st TURING:
public 78.7%, private 85.6%; 2nd TUMSAT; 3rd AIST). These scores are the
distance share within 0.5 m, FLOAT included, over Tokyo and Nagoya. The
rules required real-time processing (no future observations). The private
score covers a hidden half of the six runs. During the challenge, ground
truth existed only for two separate training runs. Also:
TURING's gici-open fork (LAMBDA + WL/NL, FIX accepted only if the
re-optimised range cost does not rise, ZUPT); Fredeluces, Ozeki, Kubo,
El-Mowafy, Sensors 24(9) 2712, 2024 (velocity-propagated float prior,
independent-satellite ambiguity check, constellation-subset retry); Niimi
et al. 2025 (arXiv 2506.03537, RBPF without integer resolution); and the
libgnss++ audits (`ppc_fix_integrity_audit.md`,
`fgo_disjoint_ar_shadow_plan.md`, `fgo_multiepoch_ar_fix_rate_audit.md`).

Reproduction: `bash run_ship.sh <name> <run> 0 --rb-seed <seed>` (WP45 is
the default), then `python fix_integrity.py --traj
x:<run>:results/<name>/run<run>.pos:pos`. The rule selection is
`python fix_rule_loro.py <prefixes>` on runs that carry the WP44
diagnostics.

### Nagoya validation of WP45 (WP46, 2026-10-12) — does not transfer

WP45 was selected on Tokyo only. Nagoya, same configuration with the
Nagoya lever arm (`--env LEVER_ARM=0.593,0.670,1.216`, FLU), 3 seeds:

| Correct FIX (wrong / FIX) | nagoya1 | nagoya2 | nagoya3 |
|---|---:|---:|---:|
| WP45 label rule (`3,5,3,12,0.1,5`) | 69.3% (0.70%) | 53.4% (**8.53%**) | 18.5% (**16.2%**) |
| strictest grid rule (`5,12,5,99,0.1,99`, ≈ `nb >= 12`) | 64.1% (0.00%) | 47.5% (0.77%) | 14.9% (**6.76%**) |
| libgnss++ KF-RTK, raw (same `ppc-demo` flags as Tokyo) | 64.9% (14.8%) | 48.6% (2.15%) | 29.8% (29.4%) |

- The WP45 term `nb_at[0.95] >= 5` is wrong on 9.7% (nagoya2) and
  11.8% (nagoya3) of the epochs it labels. A leave-one-run-out over all six
  runs (`fix_rule_loro.py --set city:run:prefix`) finds no grid rule that
  keeps every training run at ≤ 0.7% wrong: nagoya3 locks wrong basins
  with high agreement mass. Its misses have a median 12 satellites in the
  graph against 17 in libgnss++ KF-RTK, and the PF position is off by a
  median 5.3 m there.
- **Reference offset, not a model error.** At the Nagoya reference
  position only 32–44% of DD carrier rows sit within 0.1 cycle of an
  integer (Tokyo 73–78%). The L2/L1 residual ratio is 1.0, so this is not
  ionosphere. A Saastamoinen differential troposphere, a constant
  base-coordinate offset and the base-epoch age (1 Hz base) do not explain
  it either. A per-epoch 3D offset absorbs it (2.8 → 0.9 cm residual). The
  carrier-consistent antenna lies about 4 cm to the vehicle's left of the
  reference point and about 4 cm below it, consistently over runs. The
  reference point is therefore offset from the antenna phase centre by a
  few cm. That is far below the 0.5 m FIX threshold.
- **Elevation mask did not reach the filter.** cssrlib runs the base QC on
  `base_nav`, copied before the rover masks are set, so `ELMIN_DEG` below
  15° had no effect. `QC_BASE_MASK=1` applies it. Mask 10° (seed 20260710)
  gives nagoya1 67.3% (0.00%), nagoya2 47.0% (4.04%), nagoya3 18.2%
  (3.75%) and Tokyo run1 59.4% (0.00%) under the strict rule. The result is
  mixed, and Nagoya seeds spread by 7 pp, so it is not adopted on one seed.
- **`QC_ANY_BAND=1`** keeps satellites with one usable band. On its own it
  costs Tokyo run1 10 pp. Combined with mask 10° it gives nagoya3 26.8%
  (0.57%) on one seed. Not adopted.

**Shipped WP46 label (rule reselection).** Leave-one-run-out over Tokyo
1–3 and Nagoya 1–2 (Nagoya 3 excluded, no rule holds it), 3 seeds, grid
extended to F = 15, training cap 0.7%. Four of the five folds select
`nb_at[0.95] >= 15 or (nb_at_ne[0.95] >= 15 and ne_mass >= 0.1)` (no
shared-graph term). The fold that holds out Nagoya 2 has no hard run in its
training set and selects WP45's rule again (Nagoya 2 then 8.53% wrong).
`run_ship.sh` now ships the four-fold rule:

| WP46 label, 3 seeds | Tokyo 1 | Tokyo 2 | Tokyo 3 | Nagoya 1 | Nagoya 2 | Nagoya 3 |
|---|---:|---:|---:|---:|---:|---:|
| correct FIX | 56.1% | 76.4% | 75.8% | 61.0% | 46.5% | 12.3% |
| wrong / FIX (worst seed) | 0.02% (0.04) | 0.02% (0.06) | 0.00% (0.01) | 0.03% (0.06) | 0.21% (0.37) | 6.15% (8.12) |

With a 1.0% cap the folds pick `nb_at[0.98] >= 12 or (nb_at_ne[0.95] >= 15
and ne_mass >= 0.1)`. That gives 59.9 / 78.9 / 78.6 / 65.1 / 48.2%, but
Nagoya 2 reaches 1.45% on its worst seed, so it is not shipped. The
stricter label costs Tokyo 8–10 pp of correct FIX against WP45. Positions,
PPC OFFICIAL and `<50cm_full%` are unchanged.

Open: Nagoya 3 locks wrong basins with high agreement mass (6.2% wrong
even under the strictest label). That needs a filter-level fix: the
satellite set (mask 10°, one-band satellites) over several seeds, and the
metre-level float drift in narrow streets.

### Nagoya 3 (Sakae) investigation (WP47, 2026-10-12) — first pass

Seed 20260710 unless noted; WP46 label (`3,15,3,15,0.1,99`).

| configuration | Tokyo 1 | Tokyo 3 | Nagoya 3 |
|---|---:|---:|---:|
| WP46 (shipped) | 56.8% (0.01%) | 75.9% (0.01%) | 12.6% (8.12%) |
| mask 10° (`QC_BASE_MASK=1 ELMIN_DEG=10`) | 59.4%* | – | 14.6% (3.48%), 3 seeds |
| + one-band satellites (`QC_ANY_BAND=1`), 3 seeds | 45.9% (0.10%) | 68.3% (0.09%) | **22.7% (0.59%)** |
| + one-band, code only (`QC_ANY_BAND_CODE_ONLY=1`) | 44.8% (0.17%) | 67.6% (0.05%) | **25.1% (0.23%)**; s11 23.3% (0.00%) |
| + code only, PR sigma ×10 | 51.2% (0.03%) | 69.9% (0.05%) | 25.0% (0.08%) |
| + code only ×10, full-band reference | 50.6% (0.00%) | 73.3% (0.09%) | – |
| one-band only below 8 / 12 full-band satellites | 52.1 / 51.7% | 76.4 / 76.9% | 15.6 / 12.7% (7.5 / 5.2%) |
| one-band C/N0 ≥ 35 / 40 dB-Hz | 51.3 / 50.7% | – | 11.1 / 15.6% (7.5 / 5.7%) |

\* strictest label.

- **Mechanism on Nagoya 3.** Too few satellites carry both bands in Sakae, so
  the shared graph often becomes unsolvable. PF-active epochs are 79%
  (Tokyo 93–94%), and the dead-reckoned share is 2.4%. Keeping one-band
  satellites raises PF-active epochs to 94–95%. Their pseudorange alone is
  enough (code-only mode). Nagoya 3 then holds ≤ 0.8% wrong FIX on every
  seed tried, with correct FIX nearly doubled.
- **Why it is not shipped.** The same satellites cost Tokyo 1 6–12 pp and
  Tokyo 3 3–8 pp of correct FIX. The one-band population is the same in
  both cities: about 3.5 per epoch, mostly BDS B3I-only and GPS L1-only. So
  no system or C/N0 filter separates "helps" from "hurts". The harm is
  partly the DD reference choice: the highest satellite (one per system,
  shared by all bands) could be a one-band satellite, which drops that
  band's carrier pairs. The full-band reference fix recovers Tokyo 3 but not
  Tokyo 1. On Tokyo 1 the reference still flips between satellites whose L2
  flickers, and the PF pair count is about the same, so the remaining loss
  is probably graph ambiguity churn on reference changes.
- Gating the one-band satellites on the full-band count loses the Nagoya 3
  gain. The graph needs them continuously, not only in the sparse epochs.

(Superseded: the Tokyo loss was not reference churn. See the next section.)

### WP47 shipped (2026-10-13): one-band code-only satellites, residual attribution fix

**Root cause of the Tokyo harm.** The per-satellite DD-pseudorange residual
map (`_main_ddpr_per_sat`, `_main_ddpr_per_sat_pr`) charged each pair's
residual to both its reference and its target satellite. A one-band
code-only satellite is down-weighted, so the float does not follow its
multipath and its pair residual stays large. That residual was charged to
the full-band reference too. The per-satellite gate (3 m) and the
persist-bad quarantine (2 m, 4 epochs) then removed the reference, and with
it every carrier pair of its system. Pairs that involve a one-band
satellite are now charged to the one-band satellite alone and kept out of
the epoch RMS. A satellite missing a band at the base also counts as
one-band (it is never the reference). Neither change has any effect
without `QC_ANY_BAND=1`.

Seed 20260710, WP46 label:

| configuration | Tokyo 1 | Tokyo 3 | Nagoya 3 |
|---|---:|---:|---:|
| WP46 | 56.8% (0.01%) | 75.9% (0.01%) | 12.6% (8.12%) |
| one-band code-only ×10, before the fix | 52.2% (0.02%) | 73.3% (0.09%) | 21.7% (0.09%) |
| same, after the fix | 55.9% (0.04%) | 75.2% (0.01%) | 20.9% (0.00%) |
| mask 10° alone | 57.0% (0.00%) | – | 14.4% (2.09%) |

Without the 10° mask, Tokyo 2 loses the float for minutes in its hard
segment (178260–178420 s, float error up to 200 m) on two of three seeds
(wrong FIX 2.40 / 1.56 / 0.49%). With the mask, that does not happen.
With the mask (WP46 label, 3 seeds), Tokyo is unchanged
(56.4 / 76.8 / 75.7%), Nagoya 1 goes 61.0 → 66.2% and Nagoya 3 goes
12.3 → 26.2%. But Nagoya 2 and 3 keep deterministic wrong spans, at the
same times on every seed: Nagoya 2 at 556840–556862 s (a self-consistent
basin 0.95 m off, with the shared float itself about 1 m off; WP46 trips on
part of it too) and Nagoya 3 at 554402–554408 s (0.53 m).

**Label re-selection.** A grid search over the `fix_rule_loro.py` grid and
the veto grid on all six runs × three seeds, requiring every run and every
seed to stay at or below 1% wrong, picks `5,15,3,99,0.1,9`:
`nb_at[0.99] >= 15`, or the shared graph's own LAMBDA fix with at least 9
ambiguities. The non-empty-mass term and the vetoes are off.

| WP47 (3 seeds), correct FIX (wrong / FIX, per seed) | Tokyo 1 | Tokyo 2 | Tokyo 3 | Nagoya 1 | Nagoya 2 | Nagoya 3 |
|---|---:|---:|---:|---:|---:|---:|
| WP47 filter + WP47 label | 56.9% (0.01 / 0.01 / 0.01) | 78.0% (0.01 / 0.01 / 0.01) | 76.1% (0.02 / 0.05 / 0.04) | 65.4% (0.00 / 0.02 / 0.02) | 46.3% (0.99 / 0.93 / 0.20) | 26.3% (0.35 / 0.67 / 0.44) |
| WP46 filter + WP47 label | 57.0% | 77.5% | 76.6% | 60.8% | 46.4% (1.79 / 0.71 / 0.11) | 12.3% (5.94 / 4.56 / 5.42) |
| WP46 (shipped before) | 56.1% | 76.4% | 75.8% | 61.0% | 46.5% (0.37 / 0.13 / 0.14) | 12.3% (8.12 / 3.98 / 6.35) |
| PPC OFFICIAL, WP47 | 80.5% | 91.8% | 88.3% | 66.9% | 55.1% | 71.0% |
| PPC OFFICIAL, WP46 | 80.8% | 92.7% | 90.4% | 65.4% | 55.3% | 20.2% |

Caveats:

- The label is selected in-sample, on the same six runs. Held out
  leave-one-run-out, Nagoya 2 is the weak fold: trained on the other five
  runs, the selection picks a looser rule that labels 3.5–4.7% of Nagoya 2
  fixes wrong. Every other fold picks `nb_at[0.99] >= 15` with the DDPR
  veto and stays under 0.3%. The 0.99% worst seed on Nagoya 2 is at the
  cap.
- Tokyo 3 PPC OFFICIAL drops 2.1 pp (90.4 → 88.3%): the extra code-only
  satellites move some float positions.
- With the WP46 label on the WP47 filter, a DDPR-quality veto
  (`ddpr_qual > 1.0`) clears the Nagoya 3 spans (wrong 0.00 / 0.00 /
  0.29%) but not Nagoya 2.

### WP48 (2026-10-13): shared-graph measurement model

**Where the FIX rate is lost (WP47, 3 seeds).** On Tokyo, epochs within
5 cm of the reference are already almost all labelled FIX: Tokyo 1 has
56.9% correct FIX, 56.0% within 5 cm, 68.4% within 10 cm, 80.5% within
50 cm. Label changes alone gain little. An agreement-fraction term (`>= 90%`
of at least 8 pairs) adds 0.3–1.4 pp held out. A gradient-boosting
classifier on 60+ causal PF features, nested leave-one-run-out, labels
fewer fixes than the rule at the same wrong-FIX cap (Tokyo 1: 52.3 vs
56.9%). The rest is epochs the filter does not resolve: Tokyo 1 10.8% at
0.1–0.5 m and 18.7% at 0.5 m or more (tunnel), Nagoya 2 / 3 39% / 34% at
0.5 m or more, with the shared float more than 1 m off in about 30% of
epochs.

**Doppler.** The Doppler velocity prior (tc/'s LS, isotropic 0.5 m/s)
was worse than it looked. Against the reference velocity (first 3500
epochs), its horizontal error had p95 2.0 m/s and a vertical bias of
-0.40 m/s (Tokyo 1) / +0.36 m/s (Nagoya 2). Per-satellite residuals traced
it to the BeiDou GEO satellites (C01–C05, C59/C60), whose Doppler is
3–22 m/s off (their broadcast-orbit velocity). Every other system agrees
within a MAD of 0.03–0.10 m/s. The prior was also applied to the body
velocity without the lever-arm rotation term. Tightening the old prior's
sigma to 0.2 / 0.1 m/s cost up to 6 pp. `DOP_V2=1` drops GLONASS and
BeiDou GEO, weights rows by C/N0, rejects residuals above 0.5 m/s, uses
the solve's covariance, and removes the lever-arm term. The horizontal
p95 becomes 0.37 m/s (Tokyo 1) and 0.50 m/s (Nagoya 2).

Single-seed screening (Tokyo 1 / Nagoya 2 / Nagoya 3, WP47 label): Huber
on DD-PR and DD-CP +0.5 / +2.3 / +0.4 pp; the C/N0 pseudorange term cuts
the share of epochs with the float more than 1 m off (Nagoya 3:
25.7 → 20.4%) with FIX about unchanged; Doppler v2 +0.8 / -0.8 / -1.4 pp.
Each is within the Nagoya seed spread alone; together:

| 3 seeds, WP47 label | Tokyo 1 | Tokyo 2 | Tokyo 3 | Nagoya 1 | Nagoya 2 | Nagoya 3 |
|---|---:|---:|---:|---:|---:|---:|
| WP47 correct FIX | 56.9% | 78.0% | 76.1% | 65.4% | 46.3% | 26.3% |
| WP48 filter, correct FIX | 56.5% | 79.0% | 77.6% | 66.7% | 47.8% | 27.8% |
| WP47 within 50 cm (all epochs) | 80.5% | 90.3% | 92.4% | 80.8% | 59.9% | 65.2% |
| WP48 filter, within 50 cm | 84.9% | 92.0% | 95.0% | 85.1% | 60.2% | 73.4% |

**Label.** Re-selected over the six runs × three seeds with every seed at
or below 0.7% wrong: `5,15,5,15,0.1,5,1.0`, i.e. `nb_at[0.99] >= 15`, or
`nb_at_ne[0.99] >= 15` with non-empty mass 0.1+, or the shared graph's
LAMBDA fix with 5+, all vetoed while the DDPR quality exceeds 1.0 (new
optional 7th `--fix-rule` field). A 1.0% per-seed cap would allow
`3,15,3,15,0.1,5,1.0` (+1.1 pp mean, worst seed 0.99%); not taken.

| WP48 (3 seeds), correct FIX (wrong / FIX per seed) | Tokyo 1 | Tokyo 2 | Tokyo 3 | Nagoya 1 | Nagoya 2 | Nagoya 3 |
|---|---:|---:|---:|---:|---:|---:|
| correct FIX | 56.4% (0.01 / 0.01 / 0.01) | 78.7% (0.01 / 0.01 / 0.01) | 74.8% (0.03 / 0.03 / 0.03) | 66.5% (0.04 / 0.10 / 0.04) | 47.8% (0.62 / 0.55 / 0.46) | 27.4% (0.00 / 0.21 / 0.00) |
| PPC OFFICIAL | 84.3% | 93.8% | 91.8% | 74.5% | 56.1% | 78.8% |
| PPC OFFICIAL, WP47 | 80.5% | 91.8% | 88.3% | 66.9% | 55.1% | 71.0% |

Leave-one-run-out (with the veto grid), the Nagoya 2 fold still picks a
looser rule and labels 5.5% of its fixes wrong; every other fold stays at
or below 0.6% on the mean. Nagoya 2's self-consistent ~0.95 m basin
(556840–556862 s) remains the open problem.

### WP49 (2026-10-13): wide-lane fixes — negative result

`RBPF_WL=1` (diagnostic, default off) forms DD wide-lane integers
`N_f0 - N_f1` per pair from the epoch's float DD ambiguities, runs a
two-candidate LAMBDA on them, and solves a WL carrier position (epoch-local
LS with the DD-PR rows and the relaxed float prior). It runs on the WP48
configuration and leaves the PF unchanged. Seed 20260710:

| | Tokyo 1 | Tokyo 2 | Tokyo 3 | Nagoya 1 | Nagoya 2 | Nagoya 3 |
|---|---:|---:|---:|---:|---:|---:|
| WP48 correct FIX | 56.5% | 78.5% | 74.6% | 66.5% | 47.9% | 27.7% |
| WL position within 0.5 m (all epochs) | 70.3% | 85.4% | 86.2% | 78.6% | 55.9% | 57.9% |
| extra WL labels, ratio 5, rms 10 cm, n 6 | +8.2% (10.8% wrong) | +4.7% (3.4%) | +7.6% (4.1%) | +6.6% (1.0%) | +3.4% (44.8%) | +11.0% (3.7%) |
| extra WL labels, ratio 5, rms 5 cm, n 8 | +2.4% (2.1%) | +1.9% (0.6%) | +3.9% (4.9%) | +2.7% (0.5%) | +1.0% (11.9%) | +2.5% (0.8%) |
| same, held 5 epochs | +1.0% (0%) | +1.0% (0%) | +2.6% (5.1%) | +1.3% (0%) | +0.4% (0%) | +0.5% (0%) |

The headroom exists (the WL position is within 0.5 m on 7–30 pp more
epochs than the FIX label), but the per-epoch WL ratio test on the shared
graph's covariance does not separate right from wrong WL integer sets.
Wrong sets persist for tens of epochs where the float is biased by
multipath (Tokyo 3, and Nagoya 2 at 40%+ wrong). The WL position also does
not veto Nagoya 2's wrong basin: the wrong cm-level fixes there are a
median 0.23 m from the WL position, because the float is biased the same
way. Not shipped. A usable WL layer would need WL integer hypotheses held
over time (PF basins at the WL level, or Melbourne–Wübbena averaging), and
it would hit the same code-multipath bias.

**Wide-lane RB-PF (`RBPF_WLPF=1`).** A second PF instance runs the same
particle machinery on the wide-lane problem: WL floats `T y`, covariance
`T Q T'`, WL carrier rows next to the DD-PR rows. Its WL integer sets are
held over time, and a slip on either band drops the satellite. It feeds
nothing back, and the main PF's output is bit-identical with it on.
Labelling the epochs the cm-level rule leaves unfixed by the WL PF's own
agreement (seed 20260710):

| extra WL labels | Tokyo 1 | Tokyo 2 | Tokyo 3 | Nagoya 1 | Nagoya 2 | Nagoya 3 |
|---|---:|---:|---:|---:|---:|---:|
| `nb_at[0.99] >= 5` | +5.2% (20.3% wrong) | +3.8% (9.5%) | +6.6% (5.2%) | +4.8% (15.1%) | +3.1% (48.2%) | +7.2% (5.1%) |
| `nb_at[0.99] >= 8` | +0.9% (0.9%) | +0.8% (3.8%) | +2.8% (7.5%) | +0.8% (12.5%) | +0.4% (0%) | +1.1% (0%) |

The WL PF converges where the NL PF already does, and where the NL PF does
not, its WL basins are wrong as often. Holding WL hypotheses over time does
not remove the bias of the shared float. Not shipped. The 3D-map (PLATEAU)
NLOS route stays untested here: the building data hosts are not reachable
from this environment.

### WP50 (2026-10-13): hysteresis label and integer carry

**Why time.** A wrong basin is the true trajectory shifted by a fixed
offset δ. While the satellite geometry is static, carrier phase cannot
reveal δ, and in urban canyons the code is biased the same way for tens of
seconds. The libgnss++ WP176 route (a separate, heavily validated KF-RTK in
this repository) reaches the same wall: Tokyo run1 54.4% / Nagoya run1 69.4%
correct FIX at ~0% wrong (RB-FGO-PF WP48: 56.4% / 66.5%), and its report
concludes that a new independent discriminator is needed. Two properties of
time were unused:

- Wrong fixes are short flickers. Dropping FIX labels in fix-runs shorter
  than 2 s cuts Nagoya 2's wrong share from 0.54% to 0.04% at a cost of
  1.7 pp (3 seeds).
- A correct integer set stays valid while the carrier stays locked. Many
  unfixed epochs are followed, with no data gap, by a correct FIX within
  15 s (+11 to +28 pp upper bound, before slips).

**Hysteresis label.** Search over entry rule, continuation rule, entry
length and bridge (6 runs × 3 seeds, every seed at or below 0.7% wrong):
enter after the WP48 rule held 5 epochs, stay while `nb_at[0.90] >= 6` (or
the non-empty form, or the graph's LAMBDA fix with 5+), bridging 2 failed
epochs. Mean 58.6 → 60.9%, worst seed 0.62 → 0.21%. Leave-one-run-out, 5
of 6 folds pick this rule and stay at or below 0.01% wrong held out; the
Nagoya 2 fold picks an entry length of 1 (nothing else in training shows why
it is needed) and gets 2.6% wrong. The 1 s entry is therefore a design
choice, not a selected one.

**Integer carry** (`integer_carry.py`, parameters fixed a priori). The
single-difference integers of a FIX epoch (from its reported position and
the raw DD carrier of the row dump) are carried while a carrier-only LS
still fits them. A pair is dropped above 0.15 cycle (slip), a new satellite
joins only within 0.10 cycle, and the carry stops below 8 pairs, on a data
gap, on weak geometry, or when the DD pseudoranges disagree by more than
3 m (median). Seed 10 sensitivity: residual-RMS limits from 2 to 5 cm give
identical results; 6 pairs lets Tokyo 1 reach 0.34% wrong, 10 pairs loses
0.3–0.7 pp; backward windows beyond 15 s add at most 1.2 pp. Seeding the
carry from the plain WP48 rule propagates its wrong fixes (Nagoya 2: 146
wrong carried epochs). Seeding it from the hysteresis label, no carried
epoch came from a wrong source.

| 3 seeds, correct FIX (wrong / FIX, worst seed) | Tokyo 1 | Tokyo 2 | Tokyo 3 | Nagoya 1 | Nagoya 2 | Nagoya 3 |
|---|---:|---:|---:|---:|---:|---:|
| WP48 | 56.4% (0.01%) | 78.7% (0.01%) | 74.8% (0.03%) | 66.5% (0.10%) | 47.8% (0.62%) | 27.4% (0.21%) |
| WP50 real time (hysteresis + forward carry) | 61.8% (0.04%) | 82.5% (0.03%) | 77.9% (0.00%) | 70.4% (0.00%) | 51.1% (0.20%) | 29.7% (0.13%) |
| WP50 + backward carry, 15 s fixed lag | 65.3% (0.05%) | 85.0% (0.03%) | 81.1% (0.00%) | 74.6% (0.00%) | 54.2% (0.19%) | 38.4% (0.10%) |
| PPC OFFICIAL, real time | 84.3% | 93.8% | 91.8% | 74.5% | 56.1% | 78.8% |
| PPC OFFICIAL, 15 s fixed lag | 84.6% | 94.2% | 92.4% | 75.0% | 57.8% | 79.4% |

With the 15 s lag, the correct-FIX rate exceeds libgnss++ KF-RTK's raw FIX
rate on 5 of 6 runs (KF-RTK: 68.0 / 76.7 / 71.6 / 64.9 / 48.6 / 29.8%, at
1.8–29% wrong). The label layer does not feed the PF: `RBPF_SHIP=wp50` on
Tokyo 1 (seed 20260710) is bit-identical to `replay_carry.py` on the
row-dump run, and the 3-seed table is that replay on the row-dump runs of
seeds 20260710–12.

### WP51 (2026-10-13): backward PF pass for the post-processed output

**Scoring correction for WP50.** The WP50 table above was scored with
`fix_rule_loro.prepare`, which also drops FIX epochs that fail the PF's
output guard (reported position more than 5 m from the code-only DDPR
position) and dead-reckoned epochs. The live label applies the guard to
PF-labelled epochs only; carried epochs keep FIX in the `.pos` output.
Scored as written (`score_label.py`), WP50 gives real time 63.1 / 82.8 /
78.8 / 70.6 / 52.2 / 31.0% and 15 s lag 67.3 / 85.7 / 82.5 / 75.3 / 55.4 /
41.6%, with wrong-FIX rates within 0.01 pp of the table. WP48 and earlier
are unaffected: their label already includes the guard. From WP51 on, every
FIX number is the output as written.

**Backward pass.** `wp16_run_rbpf.py --reverse 1` runs the unchanged filter
on a time-mirrored copy of the run (`reverse_time.py`):
- τ = t_first + t_last − t; Doppler, satellite velocity and gyro negated;
- satellite states at the true transmission time (`cssrlib.rtk.satposs`
  patched);
- LLI values moved to each satellite's previous observation;
- IMU stamps mirrored (each 10 ms sample keeps its physical interval);
- bootstrap heading turned by π.

Two Phase-1 changes apply in reverse mode only. The bootstrap is 100 epochs
(20 s) instead of 200, since most runs end parked. `PHASE1_STUCK_N=5` hands
over to Phase 2 after 5 failed epochs once 20 have committed. The
mirrored Tokyo 2 and Nagoya 3 bootstraps wedged on the Phase-1 smoother's
`map::at` (a stale ambiguity key), a latent bug forward runs had not hit.
Forward runs are bit-identical (Tokyo 2, 600 epochs, every column).
Smoke test on the last 300 s of Tokyo 2: backward 98.1% correct FIX, 0
wrong; forward 99.0%.

Backward label alone (3 seeds, worst seed in parentheses): Tokyo 61.3 / 82.1
/ 67.8% (0.08 / 0.00 / 5.28%), Nagoya 64.5 / 53.7 / 42.9% (1.23 / 1.02 /
0.10%). The label rule was selected on forward data; on time-reversed data
it is noticeably less safe on Nagoya 1/2 and one Tokyo 3 seed. Nagoya 2's
wrong segment (556831–556839 s) is the same wrong basin the forward pass
holds (|F − B| = 1 cm).

**Combining.** On the forward run's epochs and rows (`combine_fb.py`):
1. backward integer carry from the forward hysteresis label, no lag limit;
2. a backward FIX segment (gaps up to 0.6 s) is added if the step-1 label is
   FIX within 0.3 m on at least 5 of its epochs and never further; on
   disagreement, the forward label wins;
3. backward and forward carry from every FIX epoch.

| 3 seeds, correct FIX (mean wrong / FIX; worst seed) | Tokyo 1 | Tokyo 2 | Tokyo 3 | Nagoya 1 | Nagoya 2 | Nagoya 3 |
|---|---:|---:|---:|---:|---:|---:|
| WP50 real time (as written) | 63.1% (0.05; 0.05) | 82.8% (0.02; 0.03) | 78.8% (0.00; 0.00) | 70.6% (0.00; 0.00) | 52.2% (0.13; 0.20) | 31.0% (0.04; 0.12) |
| WP50 15 s lag (as written) | 67.3% (0.05; 0.06) | 85.7% (0.02; 0.03) | 82.5% (0.00; 0.00) | 75.3% (0.00; 0.00) | 55.4% (0.13; 0.19) | 41.6% (0.03; 0.09) |
| backward carry, no lag limit | 67.3% | 85.7% | 82.5% | 77.5% | 55.5% | 45.2% |
| + carry re-seeded from carried epochs | 68.4% | 86.1% | 83.1% | 78.8% | 56.0% | 46.9% |
| **WP51** (+ anchored backward pass) | **69.3% (0.10; 0.15)** | **87.1% (0.02; 0.03)** | **83.6% (0.02; 0.02)** | **79.9% (0.00; 0.00)** | **56.8% (0.23; 0.35)** | **55.8% (0.16; 0.23)** |
| PPC OFFICIAL, WP50 15 s lag | 84.6% | 94.2% | 92.4% | 75.0% | 57.8% | 79.4% |
| PPC OFFICIAL, WP51 | 84.8% | 94.7% | 92.9% | 76.3% | 58.8% | 81.5% |

The anchor threshold is not critical (k = 1 / 5 / 10 agree within 0.1 pp).
Variants that were rejected (same 18 runs):
- dropping the forward fix on a conflict: Tokyo 3 loses 3 pp, because one
  backward seed is 5.3% wrong;
- adding every backward fix that the forward output does not contradict:
  1.7% wrong on Tokyo 3, 1.1% on Nagoya 2;
- FIX where both PFs report their own fix within 3–10 cm: about +2 pp on
  Tokyo 1, 4.5–4.8% wrong on Nagoya 2 (shared wrong basin);
- averaging the two float positions where neither is fixed: fewer epochs
  within 0.5 m on Tokyo 1, Tokyo 3 and Nagoya 1.

The backward pass helps most on Nagoya 3 (+8.9 pp over the carry steps);
alone it fixes 42.9% of epochs there, against 31.0% forward. Compute: the post-processed output runs the
filter twice (backward runs take as long as forward ones).

### WP52 (2026-10-13): carry feedback into the shared graph — negative result

`RBPF_CARRY_FB=1` holds the forward carry's integer set in the shared graph
(`RbpfFgo.carry_feedback`, the same `_committed` / `_inject_hold` path as
the PF's own feedback) on carried epochs, and keeps the PF's mass release
off those pairs while the carry lasts; `=2` also holds it on labelled FIX
epochs. The aim was to keep the float anchored where the PF's agreement
dips, so the real-time label recovers sooner. Default off.

| seed 20260710, correct FIX (wrong / FIX) | Tokyo 1 | Nagoya 2 | Nagoya 3 |
|---|---:|---:|---:|
| WP50 real time | 63.4% (0.05%) | 52.4% (0.12%) | 32.0% (0.00%) |
| `RBPF_CARRY_FB=1` | 63.6% (0.09%) | 52.6% (0.48%) | 29.0% (0.00%) |
| `RBPF_CARRY_FB=2` | – | – | 29.0% (0.00%) |

Forward carry episodes are rare and short: 2–3.5% of epochs, median 2
epochs (0.4 s), at most 53–149 epochs (Tokyo 1, Nagoya 2, Nagoya 3). The
feedback acted on 17–367 epochs, so the changes above are within the seed
spread, except that the wrong share rose on Nagoya 2. The hard stretches
end the carry within a second, which is exactly where anchoring would have
been needed. Not pursued further.

### WP53 (2026-10-13): no FIX entry while parked — negative result

Nagoya 2's wrong basin (556831–556862 s) falls in a 45 s stop (reference
speed 0 from 556807 to 556853 s). While parked, code multipath is a
constant bias and the geometry barely changes, so a basin about 1 m off is
indistinguishable. Test (label replay on the 18 WP48 row-dump runs): veto
the hysteresis entry while a causal IMU detector says the vehicle is
stationary (last 1 s: |acc| std < 0.05 m/s², median |gyro| < 0.01 rad/s).
Against the reference speed it is 97–99% precise; it finds 90–94% of the
parked epochs on Nagoya but only 13–32% on Tokyo, whose IMU vibrates more
when parked. Parked epochs are 33–55% of every run.

| 3 seeds, real time, correct FIX (worst wrong / FIX) | Tokyo 1 | Tokyo 2 | Tokyo 3 | Nagoya 1 | Nagoya 2 | Nagoya 3 |
|---|---:|---:|---:|---:|---:|---:|
| WP50 | 63.1% (0.05%) | 82.8% (0.03%) | 78.8% (0.00%) | 70.6% (0.00%) | 52.2% (0.20%) | 31.0% (0.12%) |
| no entry while parked | 62.7% (0.05%) | 81.5% (0.03%) | 78.5% (0.00%) | 55.4% (0.00%) | 40.7% (0.26%) | 26.0% (0.00%) |

Nagoya loses 5–15 pp, and Nagoya 2's wrong share does not fall: the wrong
basin is chosen during the stop but labelled only after the vehicle starts
moving. Not pursued.

### WP54 (2026-10-13): ambiguity-fit float selection (post-processed)

PPC OFFICIAL is distance-weighted and ignores the FIX label. Distance-weighted
breakdown of the WP51 output (3-seed mean, % of reference distance; strict
form, every reference epoch weighted by the distance since the previous one):

| | Tokyo 1 | Tokyo 2 | Tokyo 3 | Nagoya 1 | Nagoya 2 | Nagoya 3 |
|---|---:|---:|---:|---:|---:|---:|
| correct FIX | 69.6 | 86.8 | 80.7 | 58.9 | 43.0 | 44.0 |
| float within 0.5 m | 15.0 | 7.9 | 12.2 | 17.4 | 15.8 | 37.4 |
| float 0.5–2 m | 10.2 | 4.5 | 4.8 | 13.9 | 18.6 | 18.1 |
| float over 2 m | 4.8 | 0.8 | 2.4 | 7.3 | 22.2 | 0.3 |
| missing | 0.4 | 0.0 | 0.0 | 2.5 | 0.0 | 0.0 |

The strict and the `score_runs` (lenient) official scores differ by at most
0.3 pp. Oracles over candidate positions on unfixed epochs: better of
forward / backward 87.0% (6-run mean), plus their mean 87.9%, plus either
pass's graph float 90.1%, plus the code-only position 90.1%.

Selectors (no truth at run time; the classifiers are trained leave-one-run-
out, all three seeds of the held-out run excluded):

| PPC OFFICIAL | Tokyo 1 | Tokyo 2 | Tokyo 3 | Nagoya 1 | Nagoya 2 | Nagoya 3 | mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| WP51 | 84.5 | 94.7 | 92.9 | 76.3 | 58.8 | 81.5 | 81.5 |
| pass that fixed more recently | 90.3 | 95.5 | 88.9 | 73.6 | 61.4 | 84.1 | 82.3 |
| closer to the code-only position (1 m) | 86.4 | 95.6 | 91.6 | 77.2 | 60.2 | 84.4 | 82.6 |
| classifier, forward vs backward | 89.4 | 95.6 | 91.6 | 78.9 | 62.9 | 88.7 | 84.5 |
| classifier, five candidates | 89.9 | 95.6 | 91.9 | 81.7 | 61.9 | 89.9 | 85.2 |
| classifier, five candidates + AF features | 91.3 | 95.9 | 92.5 | 82.3 | 63.5 | 92.1 | 86.3 |
| AF max, single epoch | 91.1 | 95.6 | 91.9 | 79.1 | 63.7 | 89.5 | 85.2 |
| **AF, ±5 epochs, margin 0.1 (`combine_fb.py`)** | **92.0** | **95.6** | **93.0** | **79.5** | **64.2** | **90.9** | **85.9** |

(The last row is `score_runs` on the tool's output; the others are the
scratch evaluation on the reference grid, which agrees within 0.4 pp.) AF is
mean cos(2π frac(b0 − J·(x − pos0)/λ)) over the forward row dump. A shallow
decision tree (depth 2–3) reaches only 81.5–82.2%. Nine AF variants (window
0 / ±2 / ±5, margin 0 / 0.05 / 0.1) scored 85.1–85.8% on the scratch grid.
FIX labels and FIX positions are identical to WP51 on all 18 runs; per-seed
spread of the official score is ±0.1 (Tokyo 1) to ±3.2 pp (Nagoya 1).

### WP55 (2026-10-14): real-time ambiguity-fit position choice; Nagoya 2

**Real time.** Causal candidates: reported position, graph float, MAP and
mixture tracked beliefs (all in the live loop). Oracle over them: 79.8 →
84.0% (6-run mean, scratch weighting); with the code-only position 84.0%.
AF averaged over the last N epochs, margin m (scratch weighting): N = 1 /
3 / 6 / 11, m = 0 / 0.05 / 0.1 / 0.2 all give 81.7–82.1%. Shipped N = 6,
m = 0.1 (`--af-select 6`, `RBPF_SHIP=wp55`, now the default):

| PPC OFFICIAL (`score_runs`, 3 seeds) | Tokyo 1 | Tokyo 2 | Tokyo 3 | Nagoya 1 | Nagoya 2 | Nagoya 3 | mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| WP50 | 84.3 ± 0.9 | 93.8 ± 0.1 | 91.8 ± 0.8 | 74.5 ± 0.9 | 56.1 ± 1.0 | 78.8 ± 2.5 | 79.9 |
| WP55 | 85.5 ± 0.8 | 93.7 ± 0.1 | 92.3 ± 0.8 | 77.2 ± 1.5 | 57.5 ± 1.4 | 83.7 ± 2.7 | 81.6 |

Live check (Nagoya 3, seed 20260710, 1500 epochs): `pos_src` and positions
equal the offline causal replay on the run's own row dump (142 epochs
changed), and the graph float is identical to the WP48 row-dump run.

**Windowed AF offset search (negative).** Shift the last W epochs of the
forward trajectory by a common ENU offset (±3 m horizontal, ±1.5 m
vertical, 5 cm grid) and take the offset with the highest mean AF. Nagoya 2,
seed 20260710, every 50th unfixed epoch (79 epochs), W = 5: within 0.5 m on
18% (forward position 22%); restricting to AF peaks ≥ 0.6 / 0.7 / 0.8 gives
27 / 34 / 39% against 31 / 38 / 43% for the forward position.

**Nagoya 2's remaining loss.** Post-processed (WP54, seed 20260710), 17.1%
of the distance is more than 2 m off and 17.4% is 0.5–2 m off. The 2 m+ share is almost all
one stretch, 555966–556367 s (seed 20260710): 2–7 DD satellites, 0–8
carrier pairs, DDPR quality at its cap (30), position 3–20 m off on IMU dead
reckoning in both passes (two stops of 77 s and 66 s inside). Blending
forward and backward positions there by time since each pass's last FIX
(inverse (t + t0)^p, epochs with at most 2 / 4 / 6 pairs): Nagoya 2 +0.5 to
+1.1 pp, but Tokyo 3 −2.2 to −4.8 pp and Nagoya 1 up to −3.2 pp (6-run mean
84.6–86.1% against 85.8%). Not shipped.

### WP56 (2026-10-14): float-consistency check on the FIX label

**Correction (WP57, same day).** The Tokyo 3 backward numbers below (4.0–5.3%
wrong, float within 0.5 m on 52%) came from a backward-pass IMU bias bug, not
from the label. With the bias fixed, the WP50 label is wrong on 0.06% of
backward Tokyo 3 fixes and its float is within 0.5 m on 93.6%. On the fixed
backward runs the check is justified by Nagoya 1 (1.23 → 0.04%) and the
forward runs (0.20 → 0.08%). Its threshold sweep and leave-one-run-out are
redone in the WP57 section. The rest of this section is the original record.

**Why.** The WP50 label was chosen on the forward runs; the backward runs
(WP51, 18 runs) are data it was not tuned on. Replayed truth-free on them,
the WP50 label equals the live backward `rule_fix` on all 18 runs, and is
wrong on 4.8% of Tokyo 3 fixes (4.01 / 5.28 / 5.03% per seed), 1.15% of
Nagoya 1 and 0.90% of Nagoya 2 (3-seed means).

**Where the wrong fixes are.** All labelled FIX epochs, 18 runs per
direction, by distance from the fix to the graph float:

| fix − float | forward FIX | forward wrong | backward FIX | backward wrong |
|---|---:|---:|---:|---:|
| < 0.2 m | 114,381 | 6 (0.01%) | 90,470 | 151 (0.17%) |
| 0.2–0.5 m | 2,711 | 15 (0.55%) | 11,894 | 161 (1.35%) |
| 0.5–1 m | 243 | 14 (5.8%) | 5,412 | 289 (5.3%) |
| ≥ 1 m | 37 | 3 (8.1%) | 6,443 | 1,293 (20.1%) |

Other splits (backward): carried epochs 13.0% wrong (forward 0.34%);
epochs where the entry rule does not hold 10.7% (forward 0.22%); MAP pairs
above 0.90 under 6: 20.9%. The backward float is far worse on two runs:

| graph float within 0.5 m (median error) | Tokyo 1 | Tokyo 2 | Tokyo 3 | Nagoya 1 | Nagoya 2 | Nagoya 3 |
|---|---:|---:|---:|---:|---:|---:|
| forward | 81.0% (0.05 m) | 91.5% (0.03 m) | 94.5% (0.03 m) | 86.1% (0.14 m) | 57.6% (0.24 m) | 70.5% (0.18 m) |
| backward | 82.9% (0.08 m) | 90.8% (0.04 m) | 52.1% (0.46 m) | 62.5% (0.25 m) | 59.7% (0.20 m) | 78.2% (0.14 m) |

Backward Tokyo 3 has 60 s blocks with median float error 0.5–12.6 m
throughout; backward Nagoya 1 has 88 / 155 / 193 m blocks (seed 20260710).
Moving epochs show no timing offset (along-track error / speed: +0.002 s
backward, −0.0002 s forward on Tokyo 3). Cause not found. The position
carried from the hysteresis entry also separates them (backward: ≥ 0.5 m
from it, 560 of 1,713 epochs wrong; forward 6 of 227), but it is unavailable
on 9,060 backward FIX epochs.

**Rule.** `--fix-flt-max 0.5`: no label (entry or stay) while
‖reported position − graph float‖ > 0.5 m. The carry is not vetoed.

| Variant (correct FIX mean, worst-seed wrong) | forward 6-run mean | forward worst | backward 6-run mean | backward worst (Tokyo 3 / Nagoya 1 / Nagoya 2) |
|---|---:|---:|---:|---:|
| WP50 | 63.09% | 0.20% | 62.05% | 5.28 / 1.23 / 1.02% |
| label 0.5 m (**shipped**) | 62.84% | 0.08% | 59.61% | 0.41 / 0.04 / 1.02% |
| carry 0.5 m only | 63.07% | 0.20% | 61.38% | 3.11 / 1.15 / 1.02% |
| label and carry 0.5 m | 62.75% | 0.08% | 56.31% | 0.48 / 0.05 / 1.02% |
| label and carry 0.4 m | 62.51% | 0.04% | 55.14% | 0.42 / 0.05 / 1.02% |
| label and carry 0.6 m | 62.84% | 0.20% | 57.33% | 0.66 / 0.09 / 1.02% |
| label 0.5 m, carry 1.0 m | 62.80% | 0.08% | 57.77% | 0.47 / 0.04 / 1.02% |

Label-only threshold sweep (forward mean / forward worst / backward worst
without Nagoya 2): none 63.09 / 0.20 / 5.28%; 1.0 m 63.01 / 0.20 / 3.10%;
0.75 m 62.92 / 0.20 / 2.89%; 0.6 m 62.91 / 0.20 / 0.56%; 0.5 m 62.84 / 0.08
/ 0.41%; 0.4 m 62.76 / 0.06 / 0.38%; 0.3 m 62.60 / 0.06 / 0.16%.

Leave-one-run-out over the threshold (fold = both directions of one run;
every training pseudo-run at or below the cap, pooled over seeds). With a
0.7% cap no threshold is feasible once Nagoya 2's backward run (about 0.9%
pooled at every threshold) is in training. With a 1.0% cap:

| held out | threshold | forward | backward |
|---|---:|---:|---:|
| Tokyo 1 | 0.6 m | 63.0% (0.05%) | 60.9% (0.06%) |
| Tokyo 2 | 0.6 m | 82.3% (0.02%) | 81.7% (0.00%) |
| Tokyo 3 | 1.0 m | 78.7% (0.00%) | 63.9% (2.42%) |
| Nagoya 1 | 0.6 m | 70.5% (0.00%) | 61.3% (0.04%) |
| Nagoya 2 | 0.6 m | 52.1% (0.13%) | 53.2% (0.91%) |
| Nagoya 3 | 0.6 m | 31.0% (0.00%) | 42.4% (0.06%) |

Only Tokyo 3 has fixes 0.6–1.0 m off the float, so a data-chosen threshold
misses it when Tokyo 3 is held out. The shipped 0.5 m is the correct-FIX
tolerance and was fixed by design.

**Remaining backward wrong fixes.** Nagoya 2: the 0.95 m basin, entered at
556839.4 s going backward and held to 556831 s (40–47 epochs per seed); its
float is in the same basin (median fix − float 0.05 m). Tokyo 3: 6 / 23 / 35
epochs per seed, spread over 5–8 FIX segments with errors 0.5–1.2 m.

**All outputs (3 seeds; correct FIX, mean wrong, worst-seed wrong).**

| output | Tokyo 1 | Tokyo 2 | Tokyo 3 | Nagoya 1 | Nagoya 2 | Nagoya 3 | mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| real time WP50 | 63.1 (0.05 / 0.05) | 82.8 (0.02 / 0.03) | 78.8 (0.00 / 0.00) | 70.6 (0.00 / 0.00) | 52.2 (0.13 / 0.20) | 31.0 (0.04 / 0.12) | 63.09 |
| real time WP56 | 62.9 (0.04 / 0.04) | 82.2 (0.02 / 0.03) | 78.6 (0.00 / 0.00) | 70.5 (0.00 / 0.00) | 52.0 (0.07 / 0.08) | 30.8 (0.00 / 0.00) | 62.84 |
| 15 s lag WP50 | 67.3 (0.05 / 0.06) | 85.7 (0.02 / 0.03) | 82.5 (0.00 / 0.00) | 75.3 (0.00 / 0.00) | 55.4 (0.13 / 0.19) | 41.6 (0.03 / 0.09) | 67.97 |
| 15 s lag WP56 | 67.3 (0.04 / 0.04) | 85.7 (0.02 / 0.03) | 82.5 (0.00 / 0.00) | 75.3 (0.00 / 0.00) | 55.4 (0.07 / 0.08) | 40.9 (0.00 / 0.00) | 67.83 |
| post-processed WP51 | 69.3 (0.10 / 0.15) | 87.1 (0.02 / 0.03) | 83.6 (0.02 / 0.02) | 79.9 (0.00 / 0.00) | 56.8 (0.23 / 0.35) | 55.8 (0.16 / 0.23) | 72.11 |
| post-processed WP56 | 69.3 (0.10 / 0.18) | 87.1 (0.02 / 0.03) | 84.6 (0.05 / 0.12) | 79.9 (0.00 / 0.00) | 56.9 (0.18 / 0.24) | 54.9 (0.14 / 0.18) | 72.11 |
| backward WP50 | 61.3 (0.08 / 0.08) | 82.1 (0.00 / 0.00) | 67.8 (4.77 / 5.28) | 64.5 (1.15 / 1.23) | 53.7 (0.90 / 1.02) | 42.9 (0.06 / 0.10) | 62.05 |
| backward WP56 | 60.8 (0.05 / 0.08) | 81.4 (0.00 / 0.00) | 59.9 (0.24 / 0.41) | 61.0 (0.03 / 0.04) | 53.2 (0.91 / 1.02) | 41.3 (0.06 / 0.10) | 59.61 |

The post-processed output combines the WP56 forward and backward labels
(`combine_fb.py`, unchanged). Tokyo 3 gains 1.0 pp because backward
segments that used to clash with the forward label now pass the anchored
union.

| PPC OFFICIAL (`score_runs`, 3 seeds) | Tokyo 1 | Tokyo 2 | Tokyo 3 | Nagoya 1 | Nagoya 2 | Nagoya 3 | mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| real time (WP55 choice) | 85.5 ± 0.8 | 93.7 ± 0.1 | 92.3 ± 0.8 | 77.2 ± 1.5 | 57.5 ± 1.4 | 83.7 ± 2.8 | 81.6 |
| 15 s lag | 84.6 ± 0.8 | 94.2 ± 0.1 | 92.3 ± 0.7 | 75.0 ± 1.1 | 57.8 ± 1.1 | 79.3 ± 2.7 | 80.5 |
| post-processed (WP54 choice) | 92.0 ± 0.0 | 95.6 ± 0.3 | 92.9 ± 0.6 | 79.5 ± 3.2 | 64.3 ± 1.2 | 90.9 ± 1.3 | 85.9 |

All within 0.07 pp of WP55 / WP50 / WP54.

**How it was computed.**
- Forward: `replay_carry.py --flt-max 0.5` on the WP48 row-dump runs
  (`dmp_*`); with `--flt-max 0` this reproduces the WP50 replays exactly.
- Backward: the same replay on the WP50 backward runs. On their carried
  epochs (about 5%) the PF's own position was not stored, so the check
  there uses the carried position.
- Live check, `RBPF_SHIP=wp56`, seed 20260710, first 1,500 epochs of
  Tokyo 1, Tokyo 3 and Nagoya 3:
  - `hyst_fix`, `carry_fwd`, `rule_fix`, `pos_src` and positions equal the
    replay, including the 3 (Tokyo 1) and 4 (Tokyo 3) epochs the check
    removes there;
  - the graph float equals the WP48 dump runs.

### WP57 (2026-10-14): backward-pass IMU bias from a parked window

**Cause.** `_transition_to_phase2` takes the IMU bias (and the tilt) from
the first 500 IMU samples (5 s), assuming the vehicle is parked. A backward
pass reads the mirrored IMU, so that window is the last 5 s of the run.
Last 5 s per run (true time): Tokyo 3 mean yaw rate 3.99 deg/s, gyro std
2.2–2.7 deg/s, accel std 0.55–0.70 m/s² (turning); Tokyo 1 gyro std up to
1.13 deg/s with the mean unchanged; the four other runs parked (std ≤ 0.45
deg/s). The backward Tokyo 3 bias at the end of the run was [−0.19, −0.19,
−4.5] deg/s against the mirrored forward bias ≈ [−0.01, −0.24, +0.02]; the
other five backward runs mirror the forward bias.

**Fix.** `IMU_STATIC_SEARCH=1` (set by `--reverse`): the bias comes from the
first 500-sample window (250-sample steps) whose per-axis gyro std is under
0.0087 rad/s and accel std under 0.2 m/s² (`IMU_STATIC_GYRO_STD`,
`IMU_STATIC_ACC_STD`). The windows chosen are: backward Tokyo 3 at 60 s
(bias [−0.014, −0.245, 0.023] deg/s), backward Tokyo 1 at 2.5 s, and index 0
for every other backward run and every forward run. Forward runs keep the
option off. The four other backward runs and all forward runs are
unchanged, because the same first window is chosen there.

**Reruns** (`RBPF_SHIP=wp50 --reverse 1`, 3 seeds, `bw57TK_*`), label
replayed with the WP56 check:

| backward, 3 seeds | float within 0.5 m (median) | correct FIX, WP50 label (worst wrong) | correct FIX, WP56 label (worst wrong) |
|---|---:|---:|---:|
| Tokyo 1, before | 82.9% (0.08 m) | 61.3% (0.08%) | 60.8% (0.08%) |
| Tokyo 1, WP57 | 84.0% (0.08 m) | 62.3% (0.09%) | 61.9% (0.10%) |
| Tokyo 3, before | 52.1% (0.46 m) | 67.8% (5.28%) | 59.9% (0.41%) |
| Tokyo 3, WP57 | 93.6% (0.05 m) | 80.2% (0.06%) | 80.0% (0.07%) |

Short check (2,500 backward epochs of Tokyo 3, seed 20260710): float within
0.5 m 49.6 → 97.5%, correct FIX 52.7 → 83.9%.

**Post-processed** (WP56 forward label, `combine_fb.py` unchanged):

| 3 seeds | Tokyo 1 | Tokyo 2 | Tokyo 3 | Nagoya 1 | Nagoya 2 | Nagoya 3 | mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| correct FIX, WP56 | 69.3 (0.10 / 0.18) | 87.1 (0.02 / 0.03) | 84.6 (0.05 / 0.12) | 79.9 (0.00 / 0.00) | 56.9 (0.18 / 0.24) | 54.9 (0.14 / 0.18) | 72.11 |
| correct FIX, WP57 | 69.6 (0.10 / 0.18) | 87.1 (0.02 / 0.03) | 85.3 (0.02 / 0.05) | 79.9 (0.00 / 0.00) | 56.9 (0.18 / 0.24) | 54.9 (0.14 / 0.18) | 72.26 |
| PPC OFFICIAL, WP56 | 92.0 ± 0.0 | 95.6 ± 0.3 | 92.9 ± 0.6 | 79.5 ± 3.2 | 64.3 ± 1.2 | 90.9 ± 1.3 | 85.9 |
| PPC OFFICIAL, WP57 | 92.1 ± 0.1 | 95.6 ± 0.3 | 95.2 ± 0.7 | 79.5 ± 3.2 | 64.3 ± 1.2 | 90.9 ± 1.3 | 86.3 |

(correct FIX: mean wrong / worst-seed wrong in parentheses.)

**WP56 check re-evaluated on the fixed backward runs.**
- Backward wrong FIX: 342 of 118,710, of which 170 are 0.5 m or more from
  the float (forward: 17 of 38).
- Label-only threshold sweep, as forward mean / backward worst without
  Nagoya 2:
  - none: 63.09 / 1.23% (Nagoya 1)
  - 1.0 m: 63.01 / 0.12%
  - 0.75 m: 62.92 / 0.10%
  - 0.6 m: 62.91 / 0.10%
  - 0.5 m: 62.84 / 0.10%
  - 0.4 m: 62.76 / 0.23%
  - 0.3 m: 62.60 / 0.10%
- Leave-one-run-out with a 1.0% pooled cap:
  - TK1 / TK2 / TK3 / NG2 / NG3 folds pick 1.0 m and hold out at 0.05 /
    0.00 / 0.04 / 0.90 / 0.06% backward.
  - The NG1 fold picks no check and holds out at 1.16% backward. Only
    Nagoya 1 shows the failure.
- With a 0.7% cap no threshold is feasible once Nagoya 2 is in training.

**Nagoya 1 backward (not a bug).** Two observations rule out the obvious
causes:
- **Observation files.** No base or rover gap there (rover: one 10 s gap at
  550597 s).
- **Arc ends.** Forward row dumps, 6 runs × 7,238 arcs of 21+ epochs, |DD
  residual| at truth for the k-th epoch from an arc's start / end:
  - pseudorange p90: 2.73 / 2.05 m at k = 0, and 2.30 / 1.68 m at k = 7
    (mid-arc 1.43 m);
  - carrier p90: 0.065 / 0.091 m at k = 0, and 0.063 / 0.064 m at k = 7.

  Arc ends are not worse, so there is no case for trimming them.

What the epochs show instead:
- At 551349 s (backward order), satellites 41 (−20 m DD pseudorange
  residual on one band) and 24 (+9 to +16 m) enter as new arcs.
- The backward float goes 0.2 → 3.2 → 4.4 m with 12–14 satellites. The
  forward float holds 0.13–0.19 m with the same satellites, whose
  ambiguities were set earlier.
- 551334–551316 s has 0–2 satellites (an outage in both directions). The
  backward float enters it 9–20 m off and comes out 330 m off at 551314 s.
- It then holds no carrier ambiguity (`nsat_dd` 0, 1,697 "no valid DD"
  against 165 forward) until about 551140 s.

**Sanity reset on it (negative).** `--sanity-enable 1 --env SANITY_PR_ONLY=1`
plus the WP13r recovery flags were tried on the first 3,300 backward epochs
of Nagoya 1 (seed 20260710):
- the failed stretch recovers within 20 s (0.1 m by 551240 s);
- but the float within 0.5 m falls 66.2 → 53.7%, and correct FIX 72.8 →
  59.4% (wrong 0.00 → 0.05%).

Not shipped.

### WP58 (2026-10-14, negative): fewer-pair FIX entry and one-band carriers

**Where the real-time epochs go** (WP56 label, WP55 positions, 3 seeds, % of
rover epochs; A FIX correct, B FIX wrong, C unfixed with the reported
position within 0.5 m, D unfixed with the float within 0.5 m but the
reported position not, E unfixed with both off and an outage or < 4
carrier pairs, F unfixed with both off and 4+ pairs):

| run | A | B | C | D | E | F |
|---|---:|---:|---:|---:|---:|---:|
| Tokyo 1 | 62.9 | 0.0 | 22.6 | 0.9 | 4.0 | 9.6 |
| Tokyo 2 | 82.2 | 0.0 | 10.5 | 0.8 | 3.8 | 2.8 |
| Tokyo 3 | 78.6 | 0.0 | 17.5 | 0.6 | 2.0 | 1.3 |
| Nagoya 1 | 70.5 | 0.0 | 16.4 | 1.1 | 6.0 | 5.4 |
| Nagoya 2 | 52.0 | 0.0 | 8.5 | 0.4 | 9.6 | 29.4 |
| Nagoya 3 | 30.8 | 0.0 | 46.3 | 1.8 | 8.4 | 12.8 |

Post-processed (WP57), C is 21.8 / 8.3 / 12.5 / 8.5 / 6.3 / 34.6%. C epochs
are float quality, not lost fixes:
- **Carrier pairs:** median 9–13 (A epochs 24–40).
- **MAP pairs above 0.99:** median 0–6 (A epochs 19–37).
- **Reported error:** mostly 5–50 cm. Below 5 cm: 5.0 / 3.2 / 6.9 / 0.5 /
  0.1 / 2.3%.

**Fewer-pair entry (label only, replay of 36 runs).** A second entry branch
needs `nb_at[0.99] >= K` for L consecutive epochs, with the PF guards, the
DD-pseudorange quality veto and the WP56 float check. Results are forward
means (Nagoya 3) / Nagoya 2 worst-seed wrong:

| K / L | forward | backward |
|---|---|---|
| none | 62.84 (30.8) / 0.08% | 63.14 / 1.02% |
| 8 / 5 | 66.98 (41.6) / 13.47% | 66.93 / 10.83% |
| 10 / 5 | 66.10 (39.1) / 13.41% | 66.68 / 3.78% |
| 12 / 5 | 64.98 (36.0) / 4.61% | 65.70 / 3.57% |
| 8 / 25 | 62.96 (30.8) / 0.08% | 63.91 / 1.02% |
| 10–12 / 25–100 | 62.84–62.90 / 0.08% | 63.15–63.49 / 1.02% |

Per epoch, without hysteresis, unfixed epochs with ≥ 10 pairs above 0.99
would add 7.6 / 3.1 / 7.1 / 5.3 / 2.5 / 13.5 pp correct, but 2.57% of Nagoya
2's epochs wrong. The wrong basin agrees on few pairs as readily as the right
one.

**One-band satellites in the C epochs.** Seed 20260710, `qcedit`
instrumented with a wrapper, no code change:

| | Tokyo 1 (first 4,000 epochs) | Nagoya 3 |
|---|---|---|
| one-band satellites, median: C / A epochs | 5 / 2 | 2 / 2 |
| C epochs with ≥ 15 carrier pairs: now → with one pair per one-band satellite | 8.9 → 49.0% | 25.2 → 47.5% |

**Probe A: carrier on** (`--env QC_ANY_BAND_CODE_ONLY=0`). Seed 20260710,
full forward runs, live WP56 label, compared with the same seed's WP56:

| | correct FIX | wrong / FIX | float within 0.5 m | PPC OFFICIAL |
|---|---:|---:|---:|---:|
| Tokyo 1 | 63.2 → 57.2% | 0.04 → 0.04% | 80.7 → 72.5% | 86.1 → 77.4% |
| Tokyo 3 | 78.2 → 73.9% | 0.00 → 0.02% | 94.5 → 84.2% | 91.8 → 89.3% |
| Nagoya 2 | 52.4 → 51.4% | 0.08 → 2.10% | 57.8 → 58.4% | 58.4 → 57.6% |
| Nagoya 3 | 32.0 → 15.1% | 0.00 → 0.00% | 69.7 → 18.6% | 82.7 → 28.8% |

Carrier rows at truth in probe A, comparing one-band with two-band rows:

| | Tokyo 1 | Nagoya 3 |
|---|---|---|
| fraction more than 0.25 cycle off an integer | 31.9% / 6.6% | 42.3% / 24.3% |
| integer jumps per 1,000 pair-epochs | 96.6 / 8.5 | 97.2 / 19.4 |

The surviving band is mostly reflected or diffracted, and has no dual-band
slip check.

**Probe B: carrier only at 40 dB-Hz or more** (rover C/N0 on the band;
wrapper, no code change):

| | correct FIX | wrong / FIX | float within 0.5 m | PPC OFFICIAL |
|---|---:|---:|---:|---:|
| Tokyo 1 | 63.2 → 58.7% | 0.04 → 0.06% | 80.7 → 77.1% | 86.1 → 79.7% |
| Nagoya 2 | 52.4 → 51.0% | 0.08 → 2.05% | 57.8 → 60.5% | 58.4 → 59.7% |
| Nagoya 3 | 32.0 → 30.1% | 0.00 → 0.00% | 69.7 → 70.7% | 82.7 → 87.9% |

Not shipped: correct FIX falls everywhere, Nagoya 2 exceeds 1% wrong and
Tokyo 1 positions get worse.

**GLONASS** is not a quick source of pairs. The bases are Trimble (Tokyo
Alloy, Nagoya NetR9) and the rover a Septentrio mosaic-X5, so integer
GLONASS ambiguities would need inter-frequency bias handling.

### WP59 (2026-10-14): real-time ensemble of forward runs (map-free)

**Goal.** Beat the PPC2024 winner (TURING, 85.6% private) under the
challenge rules: real time, no future observations. PPC OFFICIAL as in
`rt_ensemble.py` (distance share within 0.5 m, missing epochs failing).

**Loss anatomy first** (3-seed ensemble, AF choice, 82.95%; pp of run
distance, mean over runs): no output 0.49, dead reckoning 2.12, fewer than
5 DD satellites 5.86, 5+ satellites 8.58. By error type: height only 3.67,
horizontal 0.5–3 m 9.66, over 3 m 3.22. Height-only losses are mostly high
(median +0.46 to +1.02 m). Nagoya 2 loses 21.2 pp of its distance in one
375 s canyon stretch (555996–556371 s, 2–7 satellites).

**Members** (all real time, same epochs): WP56 seeds 20260710–12 (`af56*`)
and the one-band carrier variant at 40 dB-Hz, seed 20260710 (`w58b_*`,
`QC_ANY_BAND_CODE_ONLY=0 QC_ANY_BAND_CP_CNR=40`).

| PPC OFFICIAL, real time | Tokyo 1 | Tokyo 2 | Tokyo 3 | Nagoya 1 | Nagoya 2 | Nagoya 3 | mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| WP56 one run (3-seed mean) | 85.2 | 93.7 | 92.3 | 77.2 | 57.5 | 83.7 | 81.61 |
| 3 seeds, AF choice | 85.7 | 93.8 | 92.2 | 78.2 | 59.7 | 88.2 | 82.95 |
| + one-band member | 85.9 | 93.9 | 93.0 | 78.6 | 62.1 | 89.1 | 83.76 |
| learned choice (LORO) | 86.3 | 94.9 | 93.3 | 79.3 | 61.1 | 90.2 | 84.19 |
| + height bias (LORO), rover-gap rows | 86.9 | 95.8 | 93.7 | 80.2 | 61.1 | 90.4 | **84.69** |

- **Learned choice.** HistGradientBoosting on 46 causal features per
  candidate (AF over 1/3/6/12/30 epochs and relative to the best / member 0,
  FIX flags, PF state, agreement counts within 0.1/0.3/0.6 m, distance to
  the median / member 0 / float / fixing members, step jump, time since the
  last FIX), trained on the other five runs (epochs where candidates
  disagree, distance-weighted). Oracle over the same 16 candidates: 87.8%.
- **Height bias.** Every LORO fold picks 0.10 or 0.15 m (grid 0–0.3 m).
- **Time halves** (each run split at its middle epoch, 84.69% row): 82.4 /
  88.1%. TURING 78.7 / 85.6 (public / private); all three top teams scored
  higher on private.
- **FIX label** (2+ map-free members fixed within 0.2 m of the output):
  62.7 / 82.1 / 77.7 / 70.0 / 52.0 / 31.8% correct, at most 0.04% wrong.
  Any single member's label: Nagoya 2 1.77% wrong.

**Rejected** (seed 20260710 unless stated):
- `P2_MIN_NS=4` (commit epochs with 4–5 common satellites): Nagoya 1 first
  1,400 epochs 59.1 → 60.4%, full run 76.5 → 75.5% (one 20 s block −1.0 pp,
  PF spread); Tokyo 1 85.7 → 85.4%. As a fifth member +0.1 / +0.4 pp.
- `P2_IMU_ONLY=1` (commit GNSS-less epochs from IMU, NHC, ZUPT): 58.3% on
  those 1,400 epochs.
- `DOP_LOWSAT_SIG=0.1` (Doppler from 2–4 satellites): 3 firings in 1,400
  epochs.
- `--gap-fill` alone: Tokyo 1 +0.3 pp, Nagoya 1 0 (anchor 4 s stale).
- FIX-offset carry: −0.2 to +0.07 pp; medoid / median choice: 81.7 /
  82.1% (AF 83.76%); AF + agreement bonus: 83.87%.

### WP60 (2026-10-14): PLATEAU NLOS traced from the predicted position (map-aided)

**Data.** PLATEAU CityGML (Tokyo 23 wards 2022, Nagoya 2022) building and
bridge tiles along each run, fetched by range requests
(`experiments/fetch_plateau_subset.py --mesh-radius 1 --include-bridges`):
Tokyo 0.75–1.2 GB of GML per run, Nagoya 0.6–0.7 GB (no bridge tiles). Heights to
the ellipsoid with EGM96 over each run (Tokyo 36.2–36.4 m, Nagoya
37.2–37.3 m). 2.2–3.6 M triangles per Tokyo run.

**Tracer.** `plateau_los.py`: 8 m grid, per-cell triangle lists and top
heights, 2D DDA, Moller-Trumbore, numba; surfaces within 3 m ignored (bridge
decks: Tokyo 2 zenith blocked on 1,354 of 9,151 epochs, 7 with buildings
only, 854 with the 3 m rule, at 7–13 m above the antenna). NLOS share of
satellite-epochs (masks traced from the first-pass track): 27–40% (Tokyo
2 and 3: 27%). Nagoya 2's canyon: 14–22 of 24–30 satellites.

**Filter.** `NLOS_ONLINE=1` (trace from the IMU-predicted antenna position
each epoch), `NLOS_MEAS_MODE=3` (exclude NLOS pairs while 6+ satellites stay
LOS, else 5x sigma), `NLOS_REF_LOS=1`. Mask from a first map-free pass's
positions, excluding (`nl1`) or inflating (`nl2`): Nagoya 2 59.1 / 59.2%
(single run 58.5%); excluding cost 2.4 pp of Tokyo 1's first 4,500 epochs.

| One run, 3-seed mean | Tokyo 1 | Tokyo 2 | Tokyo 3 | Nagoya 1 | Nagoya 2 | Nagoya 3 | mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| PPC OFFICIAL, WP56 | 85.2 | 93.7 | 92.3 | 77.2 | 57.5 | 83.7 | 81.61 |
| PPC OFFICIAL, WP60 | 89.7 | 94.4 | 92.8 | 75.4 | 58.5 | 83.2 | 82.33 |
| correct FIX, WP60 (`score_label.py`) | 61.3 | 80.2 | 80.3 | 71.6 | 45.2 | 25.5 | |
| wrong / FIX, WP60 (worst seed) | 0.07 (0.10) | 0.02 (0.03) | 0.01 (0.02) | 0.01 (0.04) | 5.84 (7.06) | 0.00 (0.00) | |

Tokyo 1's +3.8 pp (seed 20260710) sits in one stretch, 189530–189610 s:
WP56 1.3–6 m off, WP60 0.1–0.4 m. Nagoya 2's wrong fixes are one 48 s
stretch (556816–556864 s), the known 0.95 m basin, now with the float
equally off (0.93 m), so the WP56 float check does not fire.

| Ensemble, PPC OFFICIAL, real time | Tokyo 1 | Tokyo 2 | Tokyo 3 | Nagoya 1 | Nagoya 2 | Nagoya 3 | mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| WP59 (4 map-free runs), learned, height, gaps | 86.9 | 95.8 | 93.7 | 80.2 | 61.1 | 90.4 | 84.69 |
| + WP60 seed 20260710 | 91.7 | 96.5 | 94.4 | 81.7 | 62.1 | 90.3 | 86.13 |
| + WP60 3 seeds, AF choice | 90.9 | 94.8 | 93.5 | 80.6 | 62.1 | 90.5 | 85.41 |
| **+ WP60 3 seeds, learned choice** | 92.2 | 96.2 | 94.4 | 81.6 | 62.1 | 91.0 | **86.25** |
| WP60 3 seeds + one-band member only | 91.8 | 96.2 | 94.1 | 81.0 | 62.1 | 91.0 | 86.05 |

Time halves of the 86.25% row: 83.3 / 90.3%. FIX label (map-free members
only): 62.0 / 82.3 / 77.8 / 70.0 / 52.0 / 31.4%, at most 0.06% wrong.
Nagoya 2's canyon: no candidate of the 28 within 0.5 m on more than 0.2 of
its 21.2 pp (best candidate median 2.35 m).

**Operational note.** The cloud container was reclaimed while idle and
killed the running members; long runs need a live watcher.

### WP61 (2026-10-18): 3D-map candidate scoring (negative)

**Question.** Nagoya 2's 375 s canyon (555996–556371 s) has no candidate
within 0.5 m. Can PLATEAU-predicted LOS / NLOS states and reflection delays
score candidate positions well enough to find one? Everything below is
`wp61_3dma.py` on 1 Hz epochs (base epochs), first band, GLONASS excluded.

**At the ground truth** (`truth`, `summary`). Each satellite is traced for
LOS, and for NLOS the shortest valid single-bounce path on a PLATEAU wall
is found (`plateau_reflect.py`, walls within 60 m). DD code residuals to the
highest LOS satellite of each system:

| DD code residual at truth | Nagoya 2 canyon | the other five runs |
|---|---:|---:|
| NLOS share (map) | 63% | 20–34% |
| map-LOS satellites, median abs residual | 0.68 m | 0.31–0.60 m |
| map-NLOS with a reflection: median residual / median predicted extra path | 18.1 / 17.7 m | 3.8–15.9 / 15.2–21.9 m |
| correlation, residual vs predicted extra path | 0.42 | 0.18–0.56 |
| map-NLOS without a reflection: median residual | 13.7 m | 1.2–8.5 m |

The map's LOS states are reliable; the predicted delays are not. On the
other runs they run 1.4–5.6× the measured error (diffraction and paths the
LOD model lacks).

**Densities** (`calib`, Tokyo 1–3 at truth, applied to Nagoya). Residual
minus a per-epoch clock and per-system bias (stable: Galileo 2.7 m, BeiDou
−1.7 to −1.8 m against GPS, all three runs): LOS 10/50/90% −0.5 / 0.0 /
0.5 m; NLOS −1.0 / 2.0 / 28.0 m; NLOS minus the predicted extra path
−27.5 / −6.5 / 9.5 m. C/N0 medians LOS 34–47, NLOS 26–35 dB-Hz by
elevation band.

**Candidate scoring** (`grid`). For each epoch, a horizontal grid of
candidates ±12 m around the real-time output (the 86.25% ensemble) at the
output's height. Each candidate has its own LOS set; the receiver clock is
the best of a ±3 m search around the median LOS residual; the per-system
biases are estimated before the canyon at the output (causal). The score is
the sum of −log densities; the best candidate is reported.

| Nagoya 2 canyon, 376 epochs | horizontal error median | < 1 m | < 2 m | median E / N |
|---|---:|---:|---:|---:|
| real-time output | 1.85 m | 24.5% | 52.1% | −1.33 / 0.27 m |
| code residual by class, 0.5 m grid | 1.97 m | 17.0% | 51.3% | −0.01 / 1.34 m |
| + C/N0, 1 m grid | 1.83 m | 19.1% | 55.9% | 0.07 / 1.33 m |
| + C/N0, last 5 epochs summed | 1.98 m | 19.1% | 51.1% | −0.01 / 1.34 m |
| reflection delay for NLOS with a path | 1.91 m | 14.1% | 53.5% | −0.43 / 1.11 m |
| C/N0 + reflection, 5 epochs | 1.99 m | 15.4% | 50.8% | −0.08 / 1.33 m |
| code residual, grid centred on the truth | 1.80 m | 17.3% | 57.2% | 0.00 / 1.50 m |

No variant beats the output. Even centred on the truth the best candidate
is 1.8 m off. Every variant is about 1.3–1.5 m north of the truth with
east unbiased, whatever the street direction. That points to a
registration offset between PLATEAU and the reference frame, not tested
further.

**As a selector feature** (`selloss`). The ensemble's choice loses 3.3 pp
against an oracle choice among its 28 candidates (89.5%). Split by the
distance from the output to the nearest candidate within 0.5 m (pp of run
distance, mean of the six runs): < 0.5 m 1.43, 0.5–1 m 1.10, 1–1.5 m 0.30,
1.5–2 m 0.22, 2–3 m 0.14, > 3 m 0.09. A score that resolves 1.5–2 m can
reach at most the last three, 0.45 pp.

**Carrier at the truth** (`carrier`). A < 0.5 m position in the canyon needs
carrier phase. Map-LOS satellites: median 7 of 19 (after the canyon: 14 of
20). LOS DD carrier pairs: median 3 with one reference per system, 2 of
them continuous for 10 s (residual change < 3 cm between epochs); epochs
with 4 / 5 such pairs: 23% / 11%. One reference shared by GPS, Galileo and
QZSS (same L1 / E1 wavelength): 4 pairs, 41% / 16%. After the canyon: 9
pairs, 72% / 70%.

**Outcome.** Not adopted (D-060). `plateau_reflect.py` and `wp61_3dma.py`
stay as tools.


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
