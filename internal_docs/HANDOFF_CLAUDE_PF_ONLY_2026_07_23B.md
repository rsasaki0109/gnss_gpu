# HANDOFF → Claude: PF-only RTK stretch campaign — 2026-07-23 (session B)

Supersedes the immediate-action sections of
`internal_docs/HANDOFF_CLAUDE_PF_ONLY_2026_07_23.md`. All invariants from that
document remain in force unchanged (PF-only, no runtime FGO, truth-free
selection, full denominator, false FIX <= 1%, frozen gates, immutable M4).

## 1. Authoritative current production state

### Tokyo run1

- Production: **WP160** (2026-07-24; supersedes WP157)
- `<50cm_full`: **3744 / 11924 = 31.398859443139887%**
- FIX epochs: 0; false FIX epochs: 0; declared false FIX rate: 0%
- Production trajectory:
  `results/wp31/tokyo_run1_wp160_screened_stability_full_trajectory.csv`
- Trajectory SHA-256:
  `13A03483C56EC7E2D4472108DD630B70BBE69B77B3DC4E6132EEA7C45917C226`
- Benchmark:
  `results/wp31/tokyo_run1_wp160_screened_stability_full_benchmark.json`
- Benchmark SHA-256:
  `A9B154D4F100FC8D6ADF3925AF777C060684797CEB50D9684F6B2B27D05CEF33`
- Promotion lock:
  `internal_docs/wp160_tokyo_screened_stability_promotion_2026_07_24.json`
- Prior state (WP157, 3689/11924 = 30.94%) trajectory
  `results/wp31/tokyo_run1_wp157_stability_cppr_full_trajectory.csv`
  (`59E2D5B65555F0D9C82CFCF18C7328E06EB579216B055C4EA6F549BA62DE0FF4`)
  remains on disk as the WP160 application input.

### Nagoya run1 — unchanged

- Production: WP100, 5274/7583 = 69.55%. See the earlier handoff.

### M4 immutable hashes — unchanged and re-verified throughout this session

- `internal_docs/wp30_m4_production_config.json`
  `66A5FF3F1919C4B0F9ED95A5EFA38865B518C9E03E6FD2652B7A0456A1F89486`
- `internal_docs/wp30_m4_tokyo_evidence_ledger.json`
  `9D756F447304C30B73694225F1CEEA1A82DE864F8D968D449928662582DF098C`

README and `docs/assets/figures/pf_only_rtk_stretch.svg` currently report
Tokyo 31.40% / Nagoya 69.55%.

## 2. What this session did (WP140–WP157)

Promotions (all via the frozen chain: 17-sample GSI cache -> rank-0/1/2
refits -> [fine grid if initial pools are zero-gain] -> sanitize ->
cross-basis pool -> rank-0/2 cross refits -> WP53 consensus -> WP138
stability selector -> hash-recomputing promoter with WP129 + Nagoya holdouts
-> full-denominator application requiring gained>0, lost=0):

| WP | Segment | Gain | Note |
|---|---|---:|---|
| WP148 | 1100-1155 | +55 | WP108 redo; direct chain; 55/55 audit |
| WP150 | 11550-11605 | +55 | WP110 redo; direct chain; 55/55 audit |
| WP153 | 11605-11660 | +5 | WP116 redo; fine grid; selector chose a 5/55 candidate while 55/55 candidates existed (recorded, no substitution) |
| WP157 | 1155-1210 | +53 | WP127 "borderline structural" reversed by the 17-sample cache (floor 4.012 -> 3.84 m); 53/55 audit |

Negative locks written this session (all `internal_docs/wp1NN_tokyo_*_2026_07_23.json`):

- Structural DDPR/spread rejections: WP140 (11220-11275), WP142 (2530-2585),
  WP144 (4785-4840, 45 m floor), WP145 (11165-11220), WP146 (3025-3080,
  oracle 0/55), WP147 (7095-7150).
- Posterior-selection failures with demonstrated useful supply: WP141
  (5115-5170), WP143 (5060-5115), WP149 (990-1045), WP152 (935-990), WP154
  (10945-11000), plus WP151 (11440-11495, low 19-22/55 oracle ceiling).
- WP155 (1540-1595): permanent GSI 1 m laser coverage gap (all 17 samples
  resolve to the 10 m DEM); fail-closed, machinery-independent.
- WP156 (10890-10945): **first measured fine-grid-era false acceptance** of
  the WP138 selector — accepted candidate audits 0/55 at 5.75 m; application
  measured gained 0 and production was not advanced. Its selection artifacts
  are a new unsafe-acceptance holdout candidate alongside WP129
  (`internal_docs/wp156_tokyo_zero_gain_acceptance_rejection_2026_07_23.json`).

Analysis results locked in this session's conversation (no repo file):

- The quorum-2-of-3 pairwise-disagreement selector variant was fully
  simulated (parity-checked replica): holdouts stay safe but it accepts
  zero-gain winners on WP139/WP141 — **dead, do not revisit**.
- All 9 useful candidates in WP139/141/143 die at the single frozen gate
  `rank0_to_rank2_m <= 0.10 m` inside the WP131 fuse step; in WP139 the one
  surviving useful candidate loses to the winner on every stored metric.
- Full truth audit of production vs the WP106 scan (217 blocks):
  143 all-bad blocks (7865 epochs), 21 partial (538 epochs, good-epoch
  headroom mostly < 0.3 m — constant offsets risk losses), 53 all-good.

## 3. Current campaign position

Every all-bad block in the WP106 scan has now been assessed at least once,
and every pre-fine-grid supply-miss/posterior block has been re-assessed with
the current frozen machinery. The re-assessment queue is **empty**.

Remaining recovery inventory:

1. Six posterior-failure blocks with demonstrated 46-55/55 useful supply
   (WP139, 141, 143, 149, 152, 154 — about 300 epochs of realistic ceiling).
   Blocked solely by the frozen WP138 selector's ranking/margins. Any new
   selector must be separately named, keep every absolute gate, and now has
   THREE mandatory holdouts: WP129 (unsafe pool), Nagoya WP53
   (missing-evidence abstain), WP156 (zero-gain acceptance). The prior
   analysis shows the existing stored metrics cannot separate useful from
   zero-gain candidates — a genuinely new truth-free evidence family
   (e.g. temporal split-half consistency) is required, not a re-weighting.
2. Structural DDPR/spread blocks (~120 blocks / ~6.9k epochs incl. this
   session's six): unreachable under the constant-offset model; oracle
   ceilings as low as 0/55. Requires an offset-model extension (affine or
   per-epoch), which touches production-configuration invariants — a user
   decision, not a WP.
3. Partial blocks (538 epochs): tight good-epoch headroom; constant offsets
   are risky. Low priority.

Tokyo target 81% needs +5969 more epochs; category 1 alone cannot reach it.
The structural category (2) dominates the remaining gap. Raising the ceiling
therefore requires either the offset-model extension or a fundamentally
better carrier/DDPR consistency treatment; both are design decisions to
present to the user before starting.

## 3b. Exploratory measurements (2026-07-24, Tracks A/B) — all negative, do not repeat

Track A — affine ceiling probe (truth-derived model-free upper bounds,
scratchpad script `wp31_wpXA_affine_ceiling_probe.py`):

- 3025-3080: constant bound 13/55 vs affine bound 36/55 (11.7 m linear
  drift) — the ONLY block where an affine offset model materially raises the
  ceiling.
- 7095-7150: 29 -> 35; 4785-4840: 30 -> 37; 11220-11275: 55 -> 55 (already
  geometry-perfect). For DDPR-floor blocks the limit is evidence quality,
  not offset-model shape. An affine production extension is therefore LOW
  priority (~1 block of upside) and still needs user approval.

Track B — new-evidence-family tests on WP149 (990-1045, labels known):

- Stride-phase cross-consistency is UNMEASURABLE on PPC Tokyo: base.obs is
  1 Hz, rover 5 Hz, so each block has exactly one whole-second-aligned
  stride phase with any DD evidence (this is also why the WP106 scan
  "selects" a phase). Artifacts `tokyo_run1_wpXB_*` are the empty proofs.
- Split-half consistency (stride 10, phases 0 vs 5 — both whole-second,
  disjoint halves) FAILS in the informative direction: the useful 53/55
  candidate has the LARGEST split disagreement (0.0552 m) while zero-gain
  candidates sit at 0.012-0.052 m; every candidate passes all per-candidate
  gates in both halves. Artifacts `tokyo_run1_wpXB2_*`. Caveat discovered:
  re-seeding fitted offsets through the 0.05 m dedup radius can silently
  drop/replace candidates (WP149 seed 5 collided with seed 4).

Combined with the earlier proofs (metric re-weighting impossible;
quorum-2/3 accepts zero-gain), the consistent conclusion is that block-level
geometric self-consistency CANNOT identify the correct basin: zero-gain
basins look stable precisely because they agree with the biased DDPR
evidence. The open root-cause question is why DDPR carries 10-45 m floors
and biases on specific segments.

## 3c. DDPR floor anatomy (2026-07-24) — BREAKTHROUGH, next campaign axis

Per-satellite DDPR residual anatomy at truth positions, block 11220-11275
(floor 17.9 m, oracle 55/55) vs healthy control 1100-1155 (2.07 m):

- The floor is caused ENTIRELY by seven satellites: G06, G07, G13, G15
  (stable continuous NLOS biases, 55-64 m; G13 drifts smoothly 17->32 m) and
  C26, C39, C42 (intermittent/bimodal 30-70 m switching, near-truth in other
  epochs). Excluding exactly these seven: 17.918 -> 1.858 m, matching the
  healthy control. Galileo/QZSS are clean (0.47-0.48 m); the reference
  satellite is innocent (proven by triple-differences); block-wide base-data
  bias is ruled out. Constellation RMS: G 29.4 m, C 12.0 m, E 0.48 m,
  J 0.47 m.
- Implication: DDPR-floor structural blocks are likely recoverable by a
  TRUTH-FREE per-satellite robust screen (triple-difference mutual-consistency
  clustering cancels the reference and NLOS magnitudes 30-70 m dwarf the
  ~8 m production position error), NOT by an offset-model change. A follow-up
  measurement is testing whether simple robust rules recover the same
  culprit set from production positions only, with the healthy block as
  false-positive control. If it passes, the next design item is a separately
  named screen + frozen-chain WP with the three holdouts (WP129, Nagoya
  WP53, WP156). Scratchpad scripts: ddpr_residual_anatomy.py,
  ddpr_residual_gps_detail.py, ddpr_residual_bds_detail.py,
  ddpr_residual_named_drop.py.

## 3d. WP158-WP160 (2026-07-24): screen built, selector validated, first structural-block recovery

The 3c follow-up PASSED and was carried to production the same day:

- **WP158 screen** (`experiments/build_wp158_ddpr_satellite_screen.py`,
  schema `wp158_ddpr_satellite_screen_v1`): truth-free per-epoch/per-system
  triple-difference mutual-consistency clustering on production positions
  (edge 5.0 m, outlier fraction 0.2). Recovers the full 11220-11275 culprit
  set plus C24/C40 from production positions alone. The refit gained an
  opt-in `--exclude-ddpr-satellites` flag (default off, DDPR pairs only,
  carrier untouched, recorded in every output artifact as
  `ddpr_excluded_satellites`). Tests:
  `tests/test_wp158_ddpr_satellite_screen.py` (7),
  `tests/test_wp159_screened_stability_consensus.py` (6); frozen suite
  unchanged (11).
- **WP158 chain** on 11220-11275: screened refits re-enter the frozen gates
  (DDPR floor 16.6-17.9 -> 1.52-1.62 m, 55/55 supply in the rank-1 pool);
  the WP138 selector ranks the correct basin FIRST for the first time but
  rejects it solely on the cross-refit-disagreement family rank (10 > 4)
  and 13.3% runner margin. No substitution; locked in
  `internal_docs/wp158_tokyo_stability_cppr_posterior_rejection_2026_07_24.json`.
  Conclusion: basis-swap stability is anti-correlated with correct basins
  under the screen.
- **WP159 selector**
  (`experiments/select_wp159_screened_stability_consensus.py`, schema
  `wp159_screened_stability_consensus_v1`): WP138 clone with
  `cross_refit_disagreement_m` removed from RANKING only (its 0.10 m
  absolute eligibility gate and all other absolute gates unchanged), scoped
  to screened chains via mandatory non-empty `ddpr_excluded_satellites`
  (fail-closed reason `screen_evidence_required`). Validation: 10-chain
  simulation matrix (`results/wp31/*_wp159_sim_*.json`) — all unscreened
  chains fail closed, only screened WP158 accepts; plus a screened-regime
  zero-gain holdout: 10890-10945 (the WP156 block) re-run through the FULL
  screened chain (its own screen flags C24,C26,C42,E04,E05,G07,G15,G20,G30)
  is REJECTED by WP159 (family-rank gate) and by a WP138 comparison run
  (`tokyo_run1_wp159_stability_consensus_10890_10945.json`,
  `tokyo_run1_wp159_wp138comparison_10890_10945.json`).
- **WP160 promotion** (`experiments/promote_wp159_screened_stability_consensus.py`,
  holdouts WP129 + Nagoya WP53 both `screen_evidence_required`): candidate 1
  (ranks 3/1/1, runner margin 1.0) applied to WP157 — gained 55, lost 0,
  FIX/false-FIX 0, M4 intact. Tokyo 30.94% -> 31.40%. This is the FIRST
  recovery of a formerly "structural" DDPR-floor block; the structural
  category (§3 item 2) is therefore no longer categorically unreachable.

**Next inventory for the screen+WP159 chain** (in rough order of expected
yield): DDPR-floor blocks with recorded oracle ceilings — 11165-11220
(WP145, 31/55): **assessed as WP161 2026-07-24, honest rejection** (screen
restores 1.49-1.70 m floors and all per-rank gates pass, but WP53
cross-basis consensus fails on a disagreement/spread trade-off and WP159
fails closed with a single mode; locked in
`internal_docs/wp161_tokyo_screened_consensus_rejection_2026_07_24.json`);
2530-2585 (WP142, 43/55): **assessed as WP162 2026-07-24, honest
rejection** (all refit ranks pass at 1.43-1.53 m floors; exactly one
candidate passes every WP53 supply gate but a single qualifying candidate
cannot establish the 0.2 runner margin, so WP53 fails closed at margin 0.0
and WP159 follows with a single mode; locked in
`internal_docs/wp162_tokyo_screened_consensus_rejection_2026_07_24.json`);
7095-7150 (WP147, 29/55 constant-model bound): **assessed as WP163
2026-07-24, honest rejection at the REFIT stage** (screen drops the floor
to 1.27-1.38 m but every hypothesis at every rank fails the 0.5 m
block-spread gate at 1.07-7.91 m — intra-block drift, an offset-model
limitation matching the Track A affine probe; locked in
`internal_docs/wp163_tokyo_screened_refit_rejection_2026_07_24.json`);
4785-4840 (WP144, 30/55): **assessed as WP164 2026-07-24, honest
rejection at the REFIT stage** (screen flags 15 satellites incl. Galileo;
floor collapses ~45 -> 5.33-5.44 m but stays above the 4.0 m gate with
tight spreads <= 0.14 m — thin surviving evidence, the inverse of WP163's
spread failure; locked in
`internal_docs/wp164_tokyo_screened_refit_rejection_2026_07_24.json`).

WP121-126-era floors, re-prioritized by the diagnostic ceilings recorded
in their 2026-07-23 locks: 11275-11330 (WP123, ceiling 55, floor 19.6 m,
ADJACENT to the recovered WP160 block) and 1430-1485 (WP122, ceiling 55,
floor 4.79 m) are the two blocks matching the WP160 promotion pattern
(ceiling 55 + NLOS floor) — both dispatched as WP165/WP166 on
2026-07-24 evening. 11275-11330 (WP165): **honest rejection** — lowest
screened floors of the campaign (1.017 m, eight excluded satellites
largely shared with WP160's set) and all refits clean, but the rank-1
pool yields only two gate-passing hypotheses, WP53 has one
supply-passing candidate with no runner (margin 0.0, WP162's shape) and
WP159 fails closed on a single mode; locked in
`internal_docs/wp165_tokyo_screened_consensus_rejection_2026_07_24.json`.
1430-1485 (WP166): **honest posterior rejection**,
the deepest roll-out chain yet — light 4-satellite screen, every gate
passes, WP53 ACCEPTS (margin 32.5%), and WP159 dies solely on the
runner-margin check (16.7% vs 20%) with its top pick (candidate 10)
differing from the WP53 pick (candidate 1); locked in
`internal_docs/wp166_tokyo_screened_posterior_rejection_2026_07_24.json`.
Audit fields were NOT consulted; a margin change would be a new selector
requiring a new name and full holdout revalidation. Then 1375-1430 (WP167 2026-07-24: **honest rejection** — clean refits at
1.05-1.16 m floors, TWO supply-passing WP53 candidates but separated by
only 0.6% against the 20% margin, statistical twins; locked in
`internal_docs/wp167_tokyo_screened_consensus_rejection_2026_07_24.json`),
1320-1375 (WP168/WP171 2026-07-24: **FIRST WP159 zero-gain false
acceptance** — WP159 accepts candidate 2 at stability ranks 1/1/1 and
margin 1.33, the promoter passes, and the application gate catches it:
gained 0, lost 0, production NOT advanced; the WP168 selection triple is
a NEW MANDATORY unsafe-acceptance holdout for any WP159 successor;
locked in
`internal_docs/wp171_tokyo_screened_zero_gain_acceptance_rejection_2026_07_24.json`).
4895-4950 (WP169 2026-07-24: **honest rejection** — 13-satellite screen,
clean refits at 0.84-1.01 m, WP53 accepts with 58.7% margin but WP159
fails closed at mode_count 0; locked in
`internal_docs/wp169_tokyo_screened_consensus_rejection_2026_07_24.json`).
1265-1320 (WP170 2026-07-24: **honest rejection** — campaign's lowest
screened floor 0.871 m and first roll-out chain to reach the WP159
ranking with 3 modes, but WP53 fails on a 0.76% margin and WP159 fails
the family-rank check with margin exactly at 0.2; locked in
`internal_docs/wp170_tokyo_screened_posterior_rejection_2026_07_24.json`). Each needs: build screen -> screened rank-0/1/2 refits ->
screened cross-basis chain -> WP159 -> WP159 promoter. Blocks whose
ceiling is far below 55 will likely fail the gates or the selector
honestly; run them anyway to lock the outcome. 3025-3080 remains
affine-only (WP146, oracle 0/55 constant) and still requires a user
decision.

**FINAL roll-out conclusion — all ten screened blocks assessed
(2026-07-24 close)**: WP160 promoted (+55); WP161-167, WP169, WP170
honest rejections; WP168/WP171 a zero-gain FALSE ACCEPTANCE caught by
the application gate. Detailed map:

| Block | WP | Ceiling | Outcome / failing stage |
|---|---|---|---|
| 11220-11275 | WP160 | 55 | **PROMOTED +55** (Tokyo 30.94 -> 31.40%) |
| 11165-11220 | WP161 | 31 | WP53 disagreement/spread trade-off |
| 2530-2585 | WP162 | 43 | WP53 lone qualifying basin, margin 0.0 |
| 7095-7150 | WP163 | 29 | refit block-spread (intra-block drift) |
| 4785-4840 | WP164 | 30 | refit residual floor (thin evidence) |
| 11275-11330 | WP165 | 55 | WP53 lone qualifying basin, margin 0.0 |
| 1430-1485 | WP166 | 55 | WP159 runner margin 16.7% vs 20% |
| 1375-1430 | WP167 | 52 | WP53 statistical twins, margin 0.6% |
| 1320-1375 | WP168/171 | 46 | **WP159 FALSE ACCEPT, application gained 0** |
| 4895-4950 | WP169 | 39 | WP159 mode_count 0 (WP53 had accepted) |

Conclusions frozen into the campaign record: (1) the WP158 screen
reliably collapses NLOS DDPR floors (0.84-1.7 m from 4.8-45 m) on 8 of
10 blocks — evidence quality is SOLVED for most structural blocks;
(2) neither ceiling height nor screened floor predicts promotability —
the binding constraint is the frozen demand for opposed, margined,
multi-mode consensus; (3) WP168/171 proves a lone dominant-but-wrong
basin on a screened chain can be unopposed AND maximally stable
(stability ranks 1/1/1, margin 1.33), so relaxing the consensus demand
without new evidence families is UNSAFE — the WP168 triple joins WP129,
Nagoya WP53, and WP156 as a mandatory holdout; (4) only the
full-denominator application gate (gained>0, lost=0) separates WP160's
true basin from WP168's false one truth-freely at the final step, and
it held. Any further recovery on these nine blocks requires either a
genuinely new truth-free evidence family for basin identity (the same
open problem as the §3 category-1 posterior blocks) or an offset-model
extension (WP163-type drift blocks) — both are user decisions. The
screened roll-out queue is EMPTY.

## 4. Environment notes (unchanged)

CWD `C:\Users\rsasa\Workspace\old\gnss_gpu`, PowerShell, branch
`agent/wp23b-basin-ar`, dirty worktree is intentional, no `git reset --hard`,
no pushes without an explicit user request. PPC data at
`E:/datasets/PPC-Dataset-data/tokyo/run1`. Selector tests:
`python -m pytest tests/test_wp53_cross_basis_consensus.py tests/test_wp131_cross_basis_cppr_consensus.py tests/test_wp133_cppr_anchor_consensus.py tests/test_wp138_stability_cppr_consensus.py -q`
(no selector code was modified this session).
