# Changelog

## Unreleased

## v0.4.0

- Added the interactive PLATEAU/UrbanNav browser simulation with OpenStreetMap,
  default GPS/Galileo/BeiDou/QZSS/GLONASS, path inspection and signal history.
- Added route A/B evaluation, missing-aware summaries, CSV/JSON/GeoJSON exports
  and original-epoch, receiver-specific recorded signal-strength comparison.
- Consolidated exact-output native PF/ray/simulation/RB-FGO-PF optimizations and
  strict Float64 browser WebAssembly SIMD tracing with shared geometry.
- Updated package and release assets to 0.4.0, guarded release tags against
  package-version mismatches, and labelled archived v0.3 audit results historical.
- See [release notes](RELEASE_NOTES_v0.4.0.md) for scope and accuracy limitations.

The research entries below record work included since v0.3.0, including rejected
methods and experimental configurations; they are not all production defaults.

- **RB-FGO-PF WP61, negative** (`experiments/rbpf_fgo/wp61_3dma.py`,
  `plateau_reflect.py`): scoring candidate positions by their PLATEAU
  LOS / NLOS states, C/N0 and single-bounce wall-reflection delays does not
  beat the real-time output in Nagoya 2's canyon (best candidate 1.8–2.0 m
  median horizontal error vs 1.85 m; 1.8 m even centred on the truth).
  Map-LOS satellites are reliable (0.3–0.7 m median code error at the
  truth); the predicted reflection delays are not (correlation 0.18–0.56).
  Not adopted. D-060.
- **RB-FGO-PF WP60** (`experiments/rbpf_fgo/`, map-aided, off by default):
  NLOS satellites from the PLATEAU 3D city model, traced every epoch from the
  IMU-predicted antenna position (`NLOS_ONLINE=1`, `NLOS_TRI_NPZ`; CPU tracer
  `plateau_los.py`). NLOS pairs leave the double differences while 6+
  satellites stay in line of sight, else they are down-weighted
  (`NLOS_MEAS_MODE=3`); references are line-of-sight satellites
  (`NLOS_REF_LOS=1`); surfaces within 3 m of the antenna are ignored (bridge
  decks). Three such runs added to the WP59 ensemble: real-time PPC OFFICIAL
  92.2 / 96.2 / 94.4 / 81.6 / 62.1 / 91.0%, mean 86.3% (PPC2024 winner:
  85.6% private), FIX label from the map-free runs at most 0.06% wrong. Their
  own label is not used (Nagoya 2: 5.84% wrong). D-059.
- **RB-FGO-PF WP59** (`experiments/rbpf_fgo/rt_ensemble.py`, map-free):
  real-time ensemble of forward runs (three seeds and a one-band carrier
  variant, `QC_ANY_BAND_CP_CNR=40`); per epoch a classifier trained on the
  other five runs picks a member's candidate, unfixed outputs drop by a
  0.10–0.15 m height bias chosen the same way, and rover gaps get IMU
  dead-reckoned rows (`wp16_run_rbpf.py --gap-fill`). PPC OFFICIAL 84.7%
  (single run: 81.6%). Rejected: committing epochs with 4+ satellites
  (`P2_MIN_NS`), IMU-only epochs (`P2_IMU_ONLY`), Doppler from 2–4
  satellites (`DOP_LOWSAT_SIG`), medoid / median choice. D-058.
- **Docs: PPC2024 comparison corrected.** PPC2024 required real-time
  processing with no future observations, and ranked teams on a hidden half
  of the six runs (winner 85.6% private, 78.7% public). The post-processed
  output (86.3%) uses future data and had been compared with 85.6%; that
  comparison is withdrawn. Under the challenge rules the comparable number
  is the real-time output, 81.6% on all six runs. Report, README, plan,
  D-057.

- **RB-FGO-PF WP58** (negative result, record only): where the real-time
  epochs go, and two ways to fix more of them, both rejected.
  - **Where the epochs go.** 8.5–46.3% of epochs have a correct position
    but no FIX; they have 9–13 carrier pairs against 24–40 in FIX epochs.
  - **Fewer-pair FIX entry.** At 1 s it marks 4.6–13.5% of Nagoya 2's fixes
    wrong; at 5 s or longer it adds at most 0.1 pp.
  - **One-band satellites' carrier** (single seed). It lowers correct FIX
    on every run tried (Nagoya 3 32.0 → 15.1%) and marks 2.1% of Nagoya
    2's fixes wrong; at truth those carriers jump integer 5–11 times as
    often as two-band ones.
  - **The same, gated at 40 dB-Hz.** Nagoya 3 PPC OFFICIAL rises to 87.9%,
    but correct FIX still falls and Nagoya 2 reaches 2.05% wrong.
- **RB-FGO-PF WP57** (`experiments/rbpf_fgo/`): the backward pass takes the
  IMU bias from the first parked 5 s window (`IMU_STATIC_SEARCH=1`, set by
  `--reverse`) instead of the run's last 5 s. Tokyo 3 ends mid-turn, and the
  backward gyro bias had come out 4.5 deg/s off for the whole run. Backward
  Tokyo 3: graph float within 0.5 m 52.1 → 93.6%, correct FIX 59.9 → 80.0%.
  Post-processed PPC OFFICIAL 92.1 / 95.6 / 95.2 / 79.5 / 64.3 / 90.9%, mean
  86.3% (WP56: 85.9%). Forward runs unchanged. This also corrects WP56's
  evidence: its 5.3% backward Tokyo 3 wrong share came from this bug; the
  check is still justified by backward Nagoya 1 (1.23 → 0.04%).
- **RB-FGO-PF WP56** (`experiments/rbpf_fgo/`, new default
  `RBPF_SHIP=wp56`): `--fix-flt-max 0.5` withholds the FIX label while the
  reported position is more than 0.5 m from the shared graph's float (the
  correct-FIX tolerance, not tuned); the integer carry is unchanged. On
  the backward pass, which the label was not tuned on, the worst wrong
  share falls from 5.28% to 0.41% (Nagoya 2's wrong basin, 1.02%, aside).
  Real time 62.9 / 82.2 / 78.6 / 70.5 / 52.0 / 30.8% correct FIX (−0.25 pp
  mean), worst seed 0.08% wrong (WP50: 0.20%); post-processed mean
  unchanged at 72.1%, worst 0.24% (0.35%). PPC OFFICIAL unchanged.
  `replay_carry.py --flt-max` replays it.
- **RB-FGO-PF WP55** (`experiments/rbpf_fgo/`, new default
  `RBPF_SHIP=wp55`): the ambiguity-fit position choice now also runs in
  real time (`--af-select 6`, `af_select.py`), among the reported position,
  the graph float and the PF's tracked beliefs over the last 6 epochs.
  Real-time PPC OFFICIAL 85.5 / 93.7 / 92.3 / 77.2 / 57.5 / 83.7%, mean
  81.6% (WP50: 79.9%); FIX labels unchanged. Negative results recorded: a
  windowed carrier offset search, and forward/backward blending in
  low-satellite stretches. Nagoya 2's remaining loss is a 6.5-minute
  stretch with 2–7 satellites.
- **RB-FGO-PF WP54** (`experiments/rbpf_fgo/combine_fb.py`): on epochs the
  post-processed label leaves unfixed, report whichever of the forward,
  backward, mean and float positions best fits the raw DD carrier modulo
  whole cycles. PPC OFFICIAL (3 seeds) 92.0 / 95.6 / 93.0 / 79.5 / 64.2 /
  90.9%, mean 85.9% (WP51: 81.5%).
  No training data; FIX labels unchanged.
- **RB-FGO-PF WP53** (negative result, record only): vetoing FIX entry
  while an IMU detector says the vehicle is parked costs Nagoya 5–15 pp and
  does not remove Nagoya 2's wrong basin, which is chosen during a stop but
  labelled after the vehicle moves.
- **RB-FGO-PF WP52** (`experiments/rbpf_fgo/`, negative result): optional
  `RBPF_CARRY_FB=1|2` holds the integer carry's set in the shared graph.
  Carry episodes last a median of 0.4 s, so it does not raise the
  real-time FIX rate (seed 20260710: Tokyo 1 63.4 → 63.6%, Nagoya 3 32.0 →
  29.0%, Nagoya 2 wrong share 0.12 → 0.48%). Default off.
- **RB-FGO-PF WP51** (`experiments/rbpf_fgo/`): a post-processed output
  from a second, backward pass. `wp16_run_rbpf.py --reverse 1` runs the
  unchanged filter on a time-mirrored copy of the run (`reverse_time.py`),
  and `combine_fb.py` adds the backward fixes that the forward label
  confirms, then carries integers both ways. Correct FIX (3 seeds): 69.3 /
  87.1 / 83.6 / 79.9 / 56.8 / 55.8% on Tokyo 1–3 / Nagoya 1–3, every seed at
  or below 0.35% wrong (15 s lag: 67.3 / 85.7 / 82.5 / 75.3 / 55.4 /
  41.6%). Two independent passes still agree on Nagoya 2's wrong basin, so
  their agreement alone is never a FIX. Forward runs are unchanged.
- **Scoring correction (WP50).** The published WP50 FIX numbers also dropped
  carried epochs that failed the PF's output guard; the `.pos` output keeps
  them. As written, WP50 scores 63.1 / 82.8 / 78.8 / 70.6 / 52.2 / 31.0% in
  real time and 67.3 / 85.7 / 82.5 / 75.3 / 55.4 / 41.6% with the 15 s lag
  (wrong-FIX rates within 0.01 pp). The report and records now use these.
- **RB-FGO-PF technical report** (`docs/rbpf_fgo_report.md`): method,
  evaluation protocol, results WP43–WP50, negative results (learned labels,
  wide-lane, more particles), limits and next steps.
  `experiments/rbpf_fgo/score_label.py` scores the `rule_fix` label (the
  report's FIX columns).
- **RB-FGO-PF WP50** (`experiments/rbpf_fgo/`): the PF decides integer sets,
  and carrier continuity carries them. `run_ship.sh` now runs the WP48 filter
  with a hysteresis label (`--fix-hyst 5,2,2,6,2,6,0.1,5`: enter after the
  WP48 rule held 5 epochs, stay while 6+ pairs agree above 0.90) and a
  forward integer carry (`--carry 1`, `RBPF_CARRY=1`, `integer_carry.py`).
  The carry keeps the last FIX's single-difference integers while the raw DD
  carrier fits them (8+ pairs, 0.15 cycle), and reports those epochs FIX at
  the carried-integer position. `integer_smoother.py` runs the same carry
  backward for a 15 s fixed-lag output. Correct FIX (3 seeds), real time:
  Tokyo 61.8 / 82.5 / 77.9%, Nagoya 70.4 / 51.1 / 29.7%. With the 15 s lag:
  65.3 / 85.0 / 81.1%, 74.6 / 54.2 / 38.4%. Every seed stays at or below
  0.20% wrong (WP48: 56.4 / 78.7 / 74.8 / 66.5 / 47.8 / 27.4%, worst 0.62%).
  `replay_carry.py` replays the label layer on a row-dump run.
  `RBPF_SHIP=wp48` reproduces the previous release.
- **RB-FGO-PF WP49** (`experiments/rbpf_fgo/`, negative result): wide-lane
  diagnostics, default off. `RBPF_WL=1` adds a per-epoch wide-lane LAMBDA
  fix and WL carrier position. `RBPF_WLPF=1` runs a second PF on the
  wide-lane problem, with WL integer sets held over time. Neither labels
  enough extra correct fixes at or below 1% wrong (about +1 pp; up to 48%
  wrong where the shared float is biased), so the shipped WP48 configuration
  is unchanged (D-047).
- **RB-FGO-PF WP48** (`experiments/rbpf_fgo/`): measurement model of the
  shared float graph. BeiDou GEO Doppler rows were 3–22 m/s off the
  reference velocity and biased the Doppler velocity prior (about 0.4 m/s
  vertical). The prior was also applied to the body velocity without the
  lever-arm rotation term. `DOP_V2=1` solves the Doppler velocity from
  per-satellite rows without GLONASS and BeiDou GEO, C/N0-weighted, and uses
  the solve's own covariance (`DOPPLER_LEVER_COMP=1` adds the lever-arm
  term). Horizontal p95 velocity error goes from about 2.0 to 0.4–0.5 m/s.
  WP48 also turns on Huber losses (`HUBER_PR=1.345`, `HUBER_CP=2.0`) and a
  C/N0 pseudorange variance term (`PR_CNR_ERR=0.5`). The label is
  re-selected so that every seed stays at or below 0.7% wrong
  (`--fix-rule 5,15,5,15,0.1,5,1.0`; the new 7th field vetoes labels while
  the DDPR quality exceeds 1.0). Correct FIX (3 seeds): Tokyo
  56.4 / 78.7 / 74.8%, Nagoya 66.5 / 47.8 / 27.4% (worst seed 0.62% wrong).
  PPC OFFICIAL: Tokyo 84.3 / 93.8 / 91.8%, Nagoya 74.5 / 56.1 / 78.8%
  (WP47: 80.5 / 91.8 / 88.3%, 66.9 / 55.1 / 71.0%). `RBPF_SHIP=wp47`
  reproduces the previous release.
- **RB-FGO-PF WP47** (`experiments/rbpf_fgo/`): Nagoya 3. Too few
  satellites kept both bands in Sakae, so the shared graph often had too few
  carrier pairs. `run_ship.sh` now keeps satellites with one usable band as
  code-only rows (`QC_ANY_BAND`, `QC_ANY_BAND_CODE_ONLY`,
  `QC_ANY_BAND_PR_SCALE=10`), puts both receivers on a 10° mask
  (`QC_BASE_MASK`, `ELMIN_DEG=10`), and re-selects the label over all six
  runs (`--fix-rule 5,15,3,99,0.1,9`). One-band satellites are never the DD
  reference, and their pair residuals are charged to them alone. Charging
  them to the reference as well gated or quarantined the reference and cost
  Tokyo 1 up to 12 pp. Correct FIX (3 seeds): Tokyo 56.9 / 78.0 / 76.1%,
  Nagoya 65.4 / 46.3 / 26.3%. Wrong FIX stays at or below 1% of fixes on
  every run and seed (worst 0.99%, Nagoya 2). PPC OFFICIAL on Nagoya 3 goes
  from 20.2% to 71.0%, and on Tokyo 3 from 90.4% to 88.3%.
  `RBPF_SHIP=wp46` reproduces the previous release.
- **RB-FGO-PF WP46** (`experiments/rbpf_fgo/`): validated on Nagoya. WP45's
  Tokyo-selected FIX label marked 8.5% / 16.2% of Nagoya 2 / 3 fixes wrong.
  `run_ship.sh` now ships a label rule selected leave-one-run-out over Tokyo
  1–3 and Nagoya 1–2 (`--fix-rule 3,15,3,15,0.1,99`). Correct FIX (3 seeds):
  Tokyo 56.1 / 76.4 / 75.8%, Nagoya 61.0 / 46.5 / 12.3%. Wrong FIX stays at
  or below 0.37% of fixes on every seed except Nagoya 3 (6.2%). Positions
  are unchanged. Also added: `PPC_CITY=nagoya` (lever arm and scorers),
  `fix_rule_loro.py --set city:run:prefix`, and the default-off shared-graph
  QC options `QC_BASE_MASK` and `QC_ANY_BAND`. `RBPF_SHIP=wp45` reproduces
  the previous label.
- **RB-FGO-PF WP45** (`experiments/rbpf_fgo/`): `run_ship.sh` now runs one
  configuration on all runs. The runner requests two bands per satellite.
  With three, cssrlib dropped every satellite without L5, and BeiDou used
  BDS-3-only B1C; WP45 uses BeiDou B1I + B3I and drops GLONASS. Basins
  survive graph re-seeds, and the `.pos` FIX flag is a leave-one-run-out
  selected rule on the PF's agreement mass (`--fix-rule`). Tokyo PPC correct
  FIX (3 seeds) goes from 35.4 / 64.1 / 68.5% to 66.6 / 83.9 / 84.3%, with
  0.56 / 0.09 / 0.04% of fixes wrong. PPC OFFICIAL goes from 63.7 / 85.1 /
  86.9% to 80.8 / 92.7 / 90.4%. `RBPF_SHIP=wp43` reproduces the previous
  release.
- **RB-FGO-PF WP43** (`experiments/rbpf_fgo/`): the shipped run now reports
  each epoch's MAP basin tracked position belief. Every particle lineage
  keeps its own IMU-predicted Gaussian position belief, updated under its
  own integers. On rover epochs the graph abandons, the run writes an IMU
  dead-reckoned position. FIX decisions are unchanged. Tokyo PPC
  `<50cm_full%` (3 seeds) goes from 60.1 / 78.9 / 81.9 to 63.1 / 82.7 / 86.7,
  and PPC OFFICIAL from 57.1 / 82.2 / 83.5 to 63.7 / 85.1 / 86.9.
  `RBPF_SHIP=wp18` reproduces the previous output.
- **Behavior change**: `ParticleFilterDevice(resampling="megopolis")`, the
  default, now runs the corrected Megopolis (Chesser et al. 2021) with
  `megopolis_iterations=60`. The previous kernel did not converge to the weight
  distribution and remains available as `"megopolis_legacy"` for reproducing
  old results. On PPC (6 routes) PF P50 improves by 0.7 m; on UrbanNav
  Odaiba the forward RMS improves by 0.6 m and the smoothed output is unchanged
  within seed noise.
- `ParticleFilterDevice(resampling="megopolis_coalesced")`: opt-in Megopolis
  (Chesser et al. 2021) with ancestor-weight acceptance and block-shared,
  coalesced proposals. It converges to the weight distribution as iterations
  grow, unlike the default `"megopolis"`, and runs 3-13x faster (9 ms at
  B=15, 18 ms at B=240 for 1M particles). New `megopolis_iterations` argument
  (default 15). `ParticleFilterDevice` now rejects unknown `resampling` values.
- `ParticleFilterDevice` Megopolis resampling now mixes int32 ancestor indices
  and gathers the 16-double particle state once instead of copying it on
  every iteration. Bit-identical output; 1M-particle `predict → update` goes
  from 153 ms to 32 ms on a Turing GPU.
- Fixed a GPU memory leak in the host-buffered `ParticleFilter` Megopolis
  resampling: 4 of 9 device buffers were never freed (32 MB per call at 1M
  particles).

## 0.3.0 - 2026-07-29

- Added immutable evaluation contracts and mandatory negative holdouts.
- Added truth-free evidence detection, DDPR profiles, and weighted arc screens.
- Added IMU/map multi-hypothesis outage recovery.
- Added persistent CUDA batching, adaptive particle budgets, and runtime audits.
- Added cross-domain validation and fail-closed adaptation guidance.
- Added a ROS 2 lifecycle package with watchdog and deterministic replay.
- Added CUDA/ROS 2 containers, release automation, a public audit, and a
  deterministic reproducibility archive with benchmark, ablation, and failure
  gallery records.

See [the v0.3.0 release notes](RELEASE_NOTES_v0.3.0.md) for audited results and
known limits.
