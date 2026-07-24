# WP31 Tokyo run1 static/outage chain (2026-07-19)

## Locked constraints

- M4 locked artifacts remain immutable.
- Production selection is truth-free, PF-only, and has no runtime FGO.
- Ground truth is used only after selection for audit metrics.
- All rates use the full Tokyo run1 denominator of 11,924 epochs.
- No new position anchor declares FIX; declared/false FIX remain 0/0.

## Full replay and displacement baseline

- Full PF replay: 924/11,924 oracle basin candidates below 50 cm (7.75%), but
  only 197 selected positions below 50 cm (1.65%).
- Unanchored TDCP smoother: 565/11,924 (4.738%).
- Two-anchor gyro gap-fill: 773/11,924 (6.483%).

This separates candidate supply from posterior selection and shows that the
dominant Tokyo limitation is absolute reacquisition after outages.

## Truth-free static detector

`experiments/detect_wp31_static_stops.py` freezes these gates:

- at least 8 DD observations,
- post-fit RMS at most 0.05 m,
- TDCP speed below 0.05 m/s,
- bridge at most 15 epochs and 3 s,
- stop length at least 40 epochs,
- reliable raw fraction at least 0.65.

Tokyo run1 contains 14 detected stops. The PF basin supplies a sub-50 cm
candidate only at stops 1-61 and 604-755; all stops from 1867 onward require
new proposal/reacquisition logic.

## Accepted anchor chain

1. Stop 1-61: candidate 0, `clear_widelane`, audit 0.052 m.
2. Stop 604-755: candidates 5/3 compact parent marginal, 0.025 m spread,
   audit 0.050 m.
3. Stop 1867-2066: child 18 selected by cached GSI ground height:
   calibration antenna-height spread 0.0045 m, selected height residual
   0.0475 m, runner gap 0.170 m, audit 0.315 m.
4. Stop 2354-2464: child 41 is the unique intersection of GSI height residual
   at most 0.15 m and OSM road distance at most 1.0 m; audit 0.410 m.
5. Stop 5866-6021: candidate 47 is the mean-radius representative of a
   three-radius direction cluster selected by calibrated OSM road offset and
   carrier-temporal consensus; audit 0.202 m.
6. Stops 7562-7859 and 7918-7970: candidate 16 at each stop, selected jointly
   by GSI height, raw TDCP, and local road continuity; audits 0.126/0.335 m.
7. Stop 9218-9540: candidate 25, selected by calibrated road eligibility plus
   carrier-temporal cube consensus; audit 0.276 m.
8. Stop 9883-10248: candidate 68, the mean-radius representative of a
   three-radius direction cluster; audit 0.385 m.
9. Stop 11844-11924: candidate 7, selected by GSI height and OSM loop-revisit
   uniqueness; audit 0.195 m.

The GSI selector is network-free at production time. Raw DEM/geoid responses,
query URLs, acquisition times, input hashes, and the GSIGEO2011 model name are
stored in `results/wp31/tokyo_run1_gsi_height_cache*.json`.

## Full-denominator progress

| configuration | sub-50 cm epochs | full rate | FIX / false FIX |
|---|---:|---:|---:|
| two anchors | 773 | 6.483% | 0 / 0 |
| three anchors | 1,278 | 10.718% | 0 / 0 |
| four anchors | 1,324 | 11.104% | 0 / 0 |
| six anchors | 1,770 | 14.844% | 0 / 0 |
| seven anchors | 1,966 | 16.488% | 0 / 0 |
| eight anchors | 2,355 | 19.750% | 0 / 0 |
| nine anchors | 2,798 | 23.465% | 0 / 0 |
| ten anchors | 3,007 | 25.218% | 0 / 0 |

The fourth anchor adds only 46 epochs because the 2773-4081 and 3403-4081
weak-observation intervals destroy the forward motion prior before stop 4518.

## Fail-closed outage experiments

### Stop 4518-4612

- Unseeded PF/static recurring modes: best audit 32.18 m; no anchor supply.
- GSI height correction reduces the coarse error to about 11.4 m horizontal.
- Hierarchical 12 m -> 3 m -> 1 m shells eventually supply exactly one
  sub-50 cm child (child 49, audit 0.459 m).
- It is rejected: wide-lane ranks it poorly, OSM prefers another child, and
  carrier-temporal integrity ranks it 11th. Evidence is only 19 epochs.

### Stop 9218-9540

- Unseeded recurring modes share an approximately +18 m height error.
- GSI height correction plus 2 m and 1 m shells supplies child 25 at 0.276 m.
- The original wide-lane gate rejects it: residual 0.825 m (rank 9), above
  the 0.5 m absolute gate. A later predeclared carrier-temporal cube gate
  recovers it without using wide-lane as an absolute selector; see below.
- The long stop provides 64 evidence epochs, so failure is posterior
  ambiguity rather than observation count.

The rejected 4518 experiment remains proposal evidence only. The initial 9218
rejection is retained as an ablation; only the later consensus artifact is an
accepted position override.

## Next development unit: multi-stop joint selector

Single-stop evidence is insufficient after the Tokyo long outage. The next
implementation should jointly score static candidate sets at 4518, 5866,
7562, 9218, 9883, and 11844 with:

1. cached GSI ground-height calibration at each stop,
2. forward and reverse TDCP/gyro displacement residuals between stops,
3. OSM as a bounded auxiliary score rather than an absolute selector,
4. global parent comparison before any conditioned child can be accepted,
5. a unique-path margin and leave-one-stop-out stability gate,
6. fail-closed output unless every accepted node has a sub-0.5 m audit result
   after the truth-free path is frozen.

The first measurable gate was revised to any later globally selected safe
anchor pair. Stops 7562 and 7918 satisfy it. Stop 4518 remains rejected; 9218
is recovered only by the later calibrated carrier-temporal gate.

## Joint-selector prototype update

The compact PF cache
`results/wp31/tokyo_run1_static_pf_stop_candidates.json` freezes 14 stop
candidate sets (24 candidates each) against the full basin CSV SHA-256. It is
generated by a pandas chunk reader, avoiding repeated full Python-CSV scans.

`experiments/select_wp31_static_tdcp_joint_pair.py` implements a fail-closed
two-stop gate:

- the connecting edge must meet the configured raw-TDCP fraction (currently
  1.0),
- candidates must pass cached GSI height residual <= 0.15 m,
- best pair edge residual must be <= 0.5 m,
- either the TDCP runner gap must be >= 0.1 m, or a short edge (<= 50 m) must
  additionally pass nearest-road-distance continuity <= 0.5 m with a runner
  gap >= 0.1 m.

Stops 7562-7859 and 7918-7970 are connected by 60/60 raw TDCP intervals. The
coarse PF pair recovers the best actual parent modes, and hierarchical GSI
height correction plus 4 m and 2 m common-shift shells supplies a pair with
audit errors 0.126 m and 0.335 m after DD refinement. The truth-free joint
selector also ranks that pair first with edge residual 0.360 m, but the runner
is 0.399 m (gap 0.039 m), so TDCP alone remains correctly rejected.

Absolute OSM road distance selects the wrong common translation and is not an
absolute selector. On this 18.54 m, 60/60 raw-TDCP edge, however, the change in
nearest-road distance is an independent local continuity measurement. The
correct pair changes by 0.137 m; the runner changes by 0.441 m, giving a
0.305 m uniqueness margin. The frozen selector therefore accepts candidate 16
at both stops with reason `tdcp_gsi_road_continuity_unique`.

`experiments/build_wp31_tdcp_gyro_gap_fill.py` accepts that joint artifact only
when it is selected with the exact supported reason, then expands it into two
position spans. Full replay improves from 1,324 to 1,770 sub-50 cm epochs
(11.104% to 14.844%) over all 11,924 epochs. Declared FIX and false FIX remain
0/0. The remaining Tokyo gap to 81% is 7,889 sub-50 cm epochs.

## 2354-7562 motion-backbone audit

The accepted endpoints were also passed to the truth-free endpoint-closed
IMU/Doppler route builder. Heading correction strides 10, 15, and 20 pass the
predeclared heading-coherence (<15 deg) and speed-scale (0.8-1.2) gates. They
are rejected after audit: each route has a roughly 61-62 m median error over
5,303 epochs and only 158-164 sub-50 cm epochs. A single global heading/scale
closure is therefore not a usable production backbone for this outage.

An exploratory `all_filled` closure distributes endpoint residual over all
gyro-filled intervals while preserving every raw-TDCP row. It yields 1,882
sub-50 cm epochs (15.783%), but was evaluated after the four/six-anchor result
and is not promoted or locked. The `all_filled` variant remains excluded from
the accepted trajectory; the later seven-anchor gain below uses the unchanged
`longest` closure mode.

## Final-stop loop revisit and seven-anchor replay

The recurring PF parents at stop 11844-11924 contain no sub-50 cm candidate;
the best parent is 1.42 m away. A candidate-median GSI query, independent of
candidate ID and truth, gives DEM 1.5 m and geoid 36.4763 m. Parent 7 is the
unique closest-height parent with a 1.358 m correction-margin over the runner.
After height correction its center audits at 0.245 m; DD refinement moves only
0.065 m, has normalized RMS 0.260 over 865 observations, and audits at 0.195 m.

The accepted first anchor supplies an independent local-map fingerprint for
this physical revisit. Parent 7's nearest-road offset-vector change is 2.468 m
versus 7.053 m for the runner (4.585 m gap), its reference-position distance is
2.487 m, and the nearest-road feature moves only 0.304 m. The fail-closed
selector accepts only when the GSI and OSM winners agree and all DD, height,
distance, and runner gates pass. Its reason is
`gsi_height_osm_loop_revisit_unique`.

Adding this endpoint produces 1,966/11,924 sub-50 cm epochs (16.488%), with
declared/false FIX still 0/0. The remaining Tokyo gap to 81% is 7,693 epochs.

## Calibrated carrier-temporal recovery

Five already accepted Tokyo anchors freeze the production road-distance
calibration set at 0.362, 0.815, 1.180, 4.819, and 4.955 m. With a fixed 0.15 m
margin, only candidates in 0.212-5.105 m are carrier-ranked. The integrity
artifact evaluates the full 6 sigma by 4 minimum-arc-sample grid; no truth or
candidate audit enters selection.

At 9218-9540, candidate 25 wins 12/24 carrier-temporal grid cells, the runner
wins 8, and both the carrier Cauchy and temporal-arc metrics select candidate
25. The fixed cube gate requires at least 12 wins, a gap of at least 4, and
agreement of both carrier metrics. It accepts candidate 25 with reason
`gsi_osm_carrier_temporal_cube_consensus`; frozen audit is 0.276 m.

At 5866-6021, GSI height uniquely identifies parent 19. Its corrected center
is still 6.69 m away, but a 32-direction shell at 6.5/6.7/6.9 m supplies one
complete direction cluster. Direction 14 wins 18/24 cells versus 6 for the
runner, with 0.20 m radius-cluster spread. The accepted mean-radius
representative is candidate 47 at 0.202 m audit.

### Stop 9883 recovery

At 9883-10248, cached GSI height identifies parent 18 but requires a 42.61 m
vertical correction. An ENU-horizontal shell around that corrected parent is
progressively refined from 8 to 32 and finally 64 uniform directions. The
64-direction, 4.5/4.7/4.9 m shell supplies three sub-50 cm candidates (audit
0.385, 0.418, and 0.450 m), proving candidate supply.

Wide-lane alone still rejects the supplied cluster: its median-absolute ranks
are 64, 71, and 80 of 193. Under the independent calibrated-road eligibility
gate, however, direction 3 wins 16/24 carrier-temporal cells versus 5 for the
runner. All three radii are present and their spread is 0.20 m. The accepted
mean-radius representative is candidate 68, reason
`gsi_osm_carrier_temporal_direction_consensus`, audit 0.385 m.

Adding 9218 and 9883 raises the replay to 2,798/11,924 (23.465%). Adding 5866
produces the current ten-anchor result, 3,007/11,924 (25.218%), with declared
and false FIX unchanged at 0/0. The remaining Tokyo gap to 81% is 6,652
sub-50 cm epochs.

## Intermediate-stop follow-up (2026-07-22)

The same GSI-height, horizontal-shell, calibrated-road, and 24-cell
carrier-temporal procedure was applied without relaxed gates:

- 6865-6912 supplies a complete sub-50 cm direction-27 cluster (best audit
  0.129 m), but carrier direction 16 wins only 12/24 and the correct direction
  wins 5. The 12-vs-5 result misses both the 16-win and 8-win-gap gates, so the
  stop fails closed.
- 6945-7076 supplies a complete sub-50 cm direction-30 cluster (best audit
  0.142 m), but production evidence selects adjacent direction 29 at 18/24.
  Its frozen audit is 1.061 m. DD refinement improves only to 1.016 m, so the
  stop is rejected and is not passed to the trajectory builder.

These results expose a roughly one-direction carrier bias on the short 6945
stop. Any next selector must resolve that bias using an independent,
predeclared production measurement rather than audit-based direction tuning.

## Dominant long-gap route bridge (2026-07-22)

The ten-anchor `longest` closure forces the full 486.56 m residual between
2354 and 5866 into epochs 3403-4081. This produces errors up to 212.8 m inside
the gap. The deterministic `all_filled` ablation improves the full result from
3,007 to 3,138 epochs (26.317%), but remains exploratory because mode choice
was compared after audit.

`bridge_long_gyro_routes` now implements a stricter truth-free alternative.
For each accepted-anchor pair it selects only a duration-dominant gyro gap of
at least 30 s, derives its left boundary forward from the left anchor and its
right boundary backward from the right anchor, and attempts an endpoint-closed
gyro/Doppler curved route. Application requires longest/runner duration ratio
at least 2, Doppler-heading p95 at most 15 degrees, and speed scale 0.8-1.2.

The Tokyo 3403-4081 gap is duration-unique (136.4 s versus 52.2 s, ratio
2.613), but the route has heading p95 33.84 degrees and speed scale 1.400.
It therefore fails closed and leaves the trajectory unchanged. This proves
that the 2354-5866 outage cannot be recovered safely from gyro/Doppler alone;
the next route proposal must add independent road-network or carrier evidence.

## OSM-constrained particle route (2026-07-22)

`build_wp31_osm_particle_route_bridge.py` adds a deterministic, truth-free OSM
particle route with a cached 10,058-road-geometry network. Particle state is
heading, constant gyro bias, and Doppler speed scale. The likelihood uses only
the independently calibrated 0.212-5.105 m road-offset band; accepted anchors
provide the endpoints. Ground truth is loaded only after the route and all
production diagnostics have been frozen.

A single-gap 4,096-particle run over 3403-4081 is rejected. Its left boundary
has already accumulated about 81 m error before the gap; preclosure endpoint
error is 112.4 m, road-distance p95 is 60.1 m, and audit has zero sub-50 cm
epochs. This demonstrates that the two long gaps cannot be solved independently.

The multi-gap run covers all epochs 2464-5866. It fixes 1,801 reliable TDCP
steps as observed displacements and particle-propagates only the remaining
1,601 weak steps. With 1,024 particles, route scale is 1.0004 and preclosure
endpoint error falls to 17.24 m, but road-distance p95 is 14.92 m and posterior
runner separation is zero after resampling. Frozen audit is 66/3,403 sub-50 cm
epochs (1.94%). Restricting endpoint closure to uncertain steps raises supply
to 98/3,403 (2.88%) but road p95 remains 15.19 m and median error is 16.32 m.
Both variants fail closed.

OSM centerline distance improves endpoint consistency but cannot distinguish
parallel roads and intersections at the required lane-level accuracy. The next
posterior must therefore add carrier/DD evidence along the route; further road
sigma or particle-count tuning is not justified by these results.

## Route carrier/DD posterior audit (2026-07-22)

A full 11,924-epoch assignment-max-marginal lineage was regenerated from the
WP31 basin graph so route evidence no longer depends on the old 1,200-epoch
smoke artifact. The lineage itself declares 10 FIX epochs with zero false FIX.

At 5-epoch stride, the OSM multi-gap route beats the current route in a single
carrier block, 2475-2500. Requiring at least eight fixed-integer carrier rows
and simultaneous DDPR improvement applies only 26 epochs and leaves the full
result unchanged at 3,007/11,924. From 3380-3400, the OSM route has strongly
better carrier and DDPR ratios, but only three matching carrier rows per epoch;
it is therefore diagnostic evidence, not an accepted block.

A moving-epoch fixed-integer refinement was then run with three independent
seeds (current trajectory, OSM particle route, and selected basin). At epoch
3380 the single-epoch gates all pass (three carrier rows, 13 DDPR rows, 0.439 m
seed-solution spread, 0.471 cycle carrier RMS, and 3.403 m DDPR RMS), yet the
frozen audit error is 16.06 m. This is direct evidence that a single-epoch
three-row integer solution can be confidently wrong.

The production gate now additionally requires three consecutive metric passes
and both refined 5-epoch displacements to agree with TDCP within 0.5 m. Epoch
3380 is isolated; the following edge residuals are 3.20, 1.96, 4.64, and
2.58 m. The temporal gate rejects every candidate, preventing the false anchor.

## Moving block ambiguity resupply (2026-07-22)

`refine_wp31_moving_block_ambiguity.py` tests whether carrier integers shared
over a moving block can generate a new absolute anchor instead of reusing the
wrong per-epoch basin lineage. The solver estimates one integer per segmented
DD signal arc, refines a common ECEF route offset with carrier and DDPR, and
requires four time-block solutions to agree. Candidate selection never reads
truth; a truth-seeded branch is explicitly non-eligible and reports only the
measurement ceiling after production hypotheses are frozen.

The initial 3250-3405 constant-offset run fails closed. It has 31 evidence
epochs, 430 carrier rows, and 421 DDPR rows, but all 24 hypotheses have DDPR
RMS 20-39 m and block spread 12-72 m. Audit confirms zero sub-50 cm epochs.
The cause is localized: the route's truth-relative offset is stable to 1.28 m
over 3250-3300 and 0.36 m over 3350-3405, but changes rapidly in 3300-3350.

On the valid 3350-3405 short window, splitting phase arcs at observation gaps
or route-referenced jumps changes the joint float offset from the obviously
invalid `[-32.84, 121.02, 55.13] m` to `[11.54, 17.92, 20.68] m`. A road-height
prior further reduces the best of the top-32 integer candidate audits to
6.71 m, but supplies no sub-50 cm candidate; the LAMBDA ratio is only 1.13.
These candidates remain rejected.

The non-eligible truth-seeded ceiling is positive and sharply separates model
viability from candidate supply. It converges from `[15.56, 18.68, 0.49] m`
to `[15.56, 18.74, 0.46] m`, has carrier RMS 0.238 cycles, DDPR RMS 9.124 m,
four-block spread 0.080 m, median audit 0.215 m, and 55/55 sub-50 cm epochs.
The absolute 4 m DDPR gate therefore rejects even the correct solution in this
urban block; it must not be relaxed from audit. The next independent proposal
must inject causal anchor-to-road heading/cross-track evidence before integer
search, then use a predeclared relative DDPR improvement and uniqueness gate.
Increasing the raw LAMBDA candidate count or tuning height sigma further is
not justified.

Primary artifacts are
`tokyo_run1_moving_block_ambiguity_3250_3405.json`,
`tokyo_run1_moving_block_segmented_lambda_3350_3405.json`,
`tokyo_run1_moving_block_heightprior05_lambda_3350_3405.json`, and
`tokyo_run1_moving_block_oracle_ceiling_3350_3405.json` under `results/wp31`.

### OSM translation resupply follow-up

An OSM translation proposal now performs a coarse/fine common-XY road-shape
search, then a carrier-coherence search within the best road valleys. On the
3350-3405 development block, the full 4,056-point local pool contains a 55/55
sub-50 cm grid point (median audit 0.208 m). Even after retaining only four
truth-free proposal-score winners per road mode, seed 52 remains: refinement
has carrier RMS 0.192 cycles, DDPR RMS 9.119 m, road p95 0.784 m, four-block
spread 0.068 m, median audit 0.439 m, and 39/55 sub-50 cm epochs. This is the
first moving-block candidate supplied below 50 cm without basin lineage.

After observing that development result, a relative gate was frozen at
carrier RMS <=0.20 cycles, DDPR RMS ratio to the unchanged route <=0.65, road
p95 <=1.0 m, and four-block spread <=0.10 m. The reproducible gate auditor
selects only seed 52 on the development block. It selects nothing on the
3250-3300 holdout; that window's complete local pool has no sub-50 cm point.
It also selects nothing on 2520-2575, where OSM's top road valleys omit the
nearby true road mode, and on 3680-3735, where carrier/DDPR evidence is absent.

The three abstentions show no false acceptance, but they do not provide an
independent positive validation. Seed 52 is therefore not promoted into the
accepted ten-anchor trajectory. The next proposal must always include the
identity/near-current road mode alongside globally best OSM valleys, then
freeze a new gate before another positive holdout. Relevant artifacts are
`tokyo_run1_moving_block_osmlocal_supplyaudit_3350_3405.json` and the four
`tokyo_run1_moving_block_relative_gate_{development,holdout}_*.json` files.

The mandatory identity revision retains the unshifted road mode regardless of
its absolute OSM rank and searches that mode to +/-3 m. Replaying 2520-2575 as
development expands the local pool from 4,056 to 5,424 candidates and now
supplies a 55/55 sub-50 cm point with median audit 0.274 m. The point ranks only
4,248th under the pre-refinement carrier/coherence/DDPR proposal score, and the
four retained identity hypotheses are no better than 0.80 m. This confirms
that identity mode fixes candidate supply but not ranking. No identity result
is promoted. The next ranker must add independent temporal/anchor evidence;
raising the per-parent top-K until it reaches the audited rank is prohibited.

### Independent wide-lane rank audit

The moving wide-lane auditor now accepts WP31 block-ambiguity artifacts
directly and supports a separate full-trajectory warm-up input. This preserves
the partial OSM route as the scored trajectory without using truth to process
the earlier wide-lane history.

On 3350-3405, only two epochs and six fixed wide-lane rows survive even at
stride 1. Seed 52 ranks second of 59 by wide-lane RMS (0.892 m), median
absolute residual (0.706 m), and Cauchy score (0.531). The wide-lane winner is
seed 22 at 0.814 m RMS, but it independently fails the frozen carrier gate
with 0.226 cycles. Thus carrier plus wide-lane ordering points to seed 52, but
the predeclared minimum of five evidence epochs is not met. The result remains
non-production and is stored in
`tokyo_run1_moving_block_widelane_stride1_3350_3405.json`.

### Adjacent-block temporal consensus and independent holdout

An exact adjacent, non-overlapping support block (3325-3350) independently
supplies one hypothesis within 0.499 m of development primary seed 52.  The
support has four integer arcs, 12 carrier rows, 64 DDPR rows, and 0.239-cycle
carrier RMS.  The pairing is unique in both directions among the retained
hypotheses, so the temporal-consensus auditor selects seed 52 on 3325-3405.
This remains `production_promoted: false` because both the relative gate and
the temporal rule were developed on this evidence.  The artifacts are
`tokyo_run1_moving_block_identitymode_adjacent_3325_3350.json` and
`tokyo_run1_moving_block_temporal_consensus_development_3325_3405.json`.

The first independent pair, support 2905-2960 and primary 2960-3015, fails
closed.  The primary has no frozen-relative-gate pass and the temporal auditor
finds no pair.  The complete local pools also contain no sub-50 cm point: their
best median audits are 14.812 m and 13.118 m.  This is candidate-supply failure,
not a false acceptance; no trajectory or FIX declaration changes.  See the
two `tokyo_run1_moving_block_temporal_holdout_{support,primary}_*.json`
artifacts, `tokyo_run1_moving_block_relative_gate_holdout_2960_3015.json`, and
`tokyo_run1_moving_block_temporal_consensus_holdout_2905_3015.json`.

### Spatial road-corridor enumeration

The OSM translation proposer now optionally partitions the +/-40 m translation
plane into fixed cells and farthest-point orders one measurement-only road
representative per cell.  Cells whose best road p95 is at most the already
frozen 1.0 m road gate are enumerated first.  This prevents repeated parallel
road matches from consuming all local-refinement parents.  The legacy
score-ordered behavior remains the default when the cell width is zero.

On the failed 2960-3015 development block, 24 parents with 10 m cells improve
the complete-pool best median audit from 13.118 m to 2.258 m.  Reducing the
predeclared cell width to 5 m moves the best to 0.675 m, but still supplies zero
sub-50 cm epochs; its carrier/DDPR proposal rank is 10,043 of 13,536.  The
frozen relative gate passes no retained hypothesis.  The remaining supply miss
is partly vertical: the existing local grid spans only +/-0.5 m while the
truth-only diagnostic height difference is about +1.1 m.  The runner now
exposes a reproducible height-offset list and records the best candidate's
within-parent proposal rank, but no broader-height result is promoted or used
to alter a gate.  Relevant artifacts are
`tokyo_run1_moving_block_spatialcorridor_development_osmroute_2960_3015.json`,
`tokyo_run1_moving_block_spatialcorridor5_development_osmroute_2960_3015.json`,
and `tokyo_run1_moving_block_relative_gate_spatialcorridor5_development_2960_3015.json`.

### Height-complete raw-pool temporal posterior

The local solver now optionally exports a separate truth-free candidate pool,
accepts an external seed portfolio, and can reuse a longer primary block's OSM
parent corridors on a short adjacent support block.  The pool contains only
offsets and measurement diagnostics; all `audit_*` fields are removed before
serialization.  A raw temporal auditor filters carrier RMS, DDPR improvement,
integer support, and retained rows, then pairs adjacent pools without reading
reference positions.

On primary development block 2960-3015, expanding height from +/-0.5 m to
`[-1.5,1.5] m` supplies a median-0.324 m candidate with 40/55 sub-50 cm epochs
in the 31,584-point pool.  Its original single-block proposal rank is 12,779
(878 within its road parent), proving again that top-K inflation is not a safe
selector.  The exported 19.5 MB pool has no audit keys.

The original 2905-2960 support fails supply.  A 2940-2960 support is locally
stable but its short route shape selects the wrong road corridors.  Reusing
the primary's truth-free corridors and widening only the short-support local
radius to 2.5 m at 0.5 m spacing supplies a median-0.263 m candidate and 20/20
sub-50 cm epochs; it is 0.223 m from the post-selection oracle offset.

With predeclared raw gates of carrier <=0.30 cycles, DDPR ratio <=0.80, offset
distance <=0.75 m, and mutual-nearest pairing, 1,267 temporal pairs survive.
The correct primary vicinity ranks 342, so a 512-seed development portfolio is
passed through independent integer refinement.  This recovers primary seed
347 at median 0.289 m and 40/55 sub-50 cm epochs (carrier 0.225 cycles, DDPR
ratio 0.703, road p95 0.211 m, four-block spread 0.151 m).

It is not selectable safely.  The relaxed measurement envelope also admits
260 primary hypotheses, including errors through 21.45 m.  Refining the same
512 seeds on the short support and requiring fitted-offset consistency still
leaves 170 hypotheses; correct seed 347 ranks only 121 by fitted-offset
distance.  Selecting the best support measurement within a 1.5 m drift ball
is also insufficient: the nearest primary-truth candidate ranks 1,534.  No
candidate is promoted and the accepted ten-anchor trajectory is unchanged.

The evidence shows that isolated moving blocks do not contain enough absolute
information to distinguish repeated road/carrier modes.  The next posterior
must propagate a distribution from the accepted epoch-2464 anchor through a
sequence of adjacent block states, allowing gradual route-offset drift while
penalizing corridor jumps.  This is a PF/Viterbi path problem, not another
single-block residual threshold.  Primary artifacts are the
`tokyo_run1_moving_block_rawpool_*`, `raw_temporal512_development_2940_3015`,
and `temporal512_{refined,support_refined}_development_*` JSON files.

### Anchor-propagated path posterior: fail closed

A truth-free Viterbi/PF-style selector was then run from the accepted epoch
2464 anchor through ten adjacent 55-epoch pools ending at epoch 3015.  Each
state uses only the OSM-road proposal, carrier/DDPR diagnostics, the preceding
offset, and an endpoint-independent motion prior.  Without integer lineage it
keeps all ten blocks reachable, but selects the nearly static OSM offset rather
than the required mid-route correction; every block has low posterior gamma
(`0.007--0.187`) and the production gate abstains.

Adding block-normalized integer signatures makes the posterior more selective
but not correct.  With at least four shared arcs, +/-1-cycle tolerance, and at
most two disagreements, the path becomes unreachable at 2905--2960.  Before
that failure its selected offset is already 1.7--19.3 m from the audit-only
reference offset.  The physically best supplied coarse transitions themselves
violate a single fixed-lineage model across pivot changes, cycle slips, and the
2773--3034 outage; some adjacent blocks share only two arcs.  Relaxing this
gate would restore the repeated-road ambiguity that the lineage was intended
to remove.

Therefore neither path posterior is promoted, no trajectory/FIX declaration
changes, and further block-score tuning is stopped.  The next mechanism is a
gap-local, endpoint-closed OSM road-graph recovery for the long gyro outage,
using independently accepted left/right boundaries.  Reproducible artifacts
are `tokyo_run1_path_posterior_development_2464_3015_manifest.json`,
`tokyo_run1_path_posterior_development_2464_3015.json`,
`tokyo_run1_path_posterior_integer_lineage_development_2464_3015.json`, and the
ten `tokyo_run1_pathpool_intsig_coarse_*` pools.

### Gap-local OSM graph proposal: boundary premise fails

The cached OSM geometries were converted to a 21,293-node/25,231-edge graph
and the shortest endpoint-closed road proposal was evaluated on the first long
gyro outage, 2773--3034.  Selection uses only accepted-anchor propagation,
Doppler length, graph topology, node snap, and road distance.  The best graph
path is 169.68 m versus 150.51 m Doppler length (scale 1.127), and the second
path is 25.13 m longer, so path supply itself is plausible.

The proposal correctly rejects: start/end node snaps are 13.72/25.03 m and
road-distance p95 is 24.38 m.  Post-selection audit explains the failure.  The
left boundary propagated 309 epochs from anchor 2464 is already 11.25 m wrong;
the right boundary propagated backward roughly 2,800 epochs from anchor 5866
is 74.13 m wrong.  Consequently the graph route has median error 68.15 m and
zero sub-50 cm epochs.  A local graph bridge cannot be endpoint-closed until
both boundary distributions are estimated rather than treated as known.

No trajectory or FIX declaration changes.  The implementation and fail-closed
artifact are `build_wp31_osm_graph_gap_proposal.py` and
`tokyo_run1_osm_graph_gap_development_2773_3034_summary.json`.  The next PF
must carry boundary position uncertainty across the full anchor-to-anchor
interval and score graph turn/length sequences, not clamp to dead-reckoned gap
boundaries.

### Full anchor-to-anchor road-edge PF: fail closed

A road-network particle filter was implemented over the complete accepted
2464--5866 interval.  Unlike the earlier continuous OSM particles, its state
is a directed OSM edge, distance along edge, Doppler scale, heading, and gyro
bias.  It proposes intersection branches from gyro turns, resets heading from
reliable TDCP, uses a backward graph-distance reachability field, and supports
forward/reverse filtering and bidirectional Cauchy dead-reckoning references.
Ground truth remains post-selection audit only.

The OSM graph supplies 94.36% of audit positions inside the independently
calibrated 5.105 m road band, so road geometry supply is not the main failure.
The initial implementation exposed and fixed three truth-free defects: gyro
sign was not applied, branch proposal probability was double-counted, and the
current TDCP heading was applied one epoch after an intersection.  These fixes
reduced the best 256-particle forward audit median from 741.6 m to 36.1 m, but
the lineage still ends 164.6 m from the right anchor and fails production.

Reverse filtering reaches within 38.5 m with the broad bias model.  Applying
the independently estimated -0.0166 deg/s gyro-bias prior gives 67.3 m at 256
particles and 44.2 m at 1,024 particles.  Thus fourfold candidate supply does
not reach the fixed 10 m endpoint gate.  Forward and reverse lineages never
meet: their closest same-epoch separation is 38.47 m.  Adding a truth-free
mixture of left- and right-integrated displacement references also fails
(forward/reverse endpoint errors 92.8/400.0 m).  Every variant is rejected;
no trajectory or FIX declaration changes.

The road-edge PF is retained as a measured negative implementation in
`build_wp31_osm_graph_particle_route.py`, with the principal artifacts named
`tokyo_run1_osm_graph_particle{256,1024}_*_development_2464_5866_summary.json`.
More particles or road-score tuning is not justified.  A future retry must
introduce an independently selected absolute mid-interval anchor or a genuine
two-filter state smoother rather than another terminal-conditioned lineage.

## WP106--WP107 global moving-supply restart

WP106 scans the full 11,924-epoch Tokyo production trajectory in 55-epoch
blocks using the same truth-free carrier/DDPR supply gate as Nagoya. Of 216
complete blocks, 214 pass supply and 150 are all-bad under post-scan audit. The
strongest block is 11495--11550 with 431 carrier and 286 DDPR rows.

WP107 evaluates that block. All three carrier bases have a 55/55 diagnostic
ceiling. After a 49-cell GSI-normalized grid, rank 2 supplies 57 absolute modes;
candidate 51 ranks 1/3/2 in road/carrier/CPPR evidence and beats the runner by
233%. The truth-free selected profile recovers 3 epochs. Both independent
holdouts fail closed, M4 remains exact, and the full-denominator audit reports
+3/0 gain/loss. Tokyo production advances to 3,268/11,924 = 27.4069%; FIX and
false FIX remain zero. A separate 55/55 diagnostic candidate is not substituted
after audit because it fails the frozen posterior margin.

## WP108--WP109 posterior basin rejections

WP108 evaluates the next global-scan block, 1100--1155. Rank 0 supplies a
55/55 diagnostic candidate, but the 49-cell grid produces 50 absolute modes
and the frozen posterior chooses candidate 10 with family ranks 1/12/12 and
only an 8% runner margin. The winner improves 0/55 epochs. Both posterior
gates fail, so no profile is promoted.

WP109 evaluates 990--1045. Rank 2 reaches a 55/55 diagnostic ceiling and its
49-cell grid supplies 54 strict candidates and 44 absolute modes. The frozen
truth-free posterior accepts candidate 52 with family ranks 1/2/2 and a 60%
runner margin, but that winner improves 0/55 epochs. The diagnostic 55/55
candidate is not substituted after audit. Production therefore remains
3,268/11,924 = 27.4069%; FIX and false FIX remain zero and M4 remains exact.

WP110 evaluates 11550--11605. Initial rank 1 and rank 2 both reach a 55/55
diagnostic ceiling; rank 1 is expanded because its initial winner itself is
55/55 but narrowly misses the frozen road-family and margin gates. The 49-cell
grid produces 52 absolute modes, but the posterior instead chooses candidate
50 with family ranks 18/3/4, an 8% runner margin, and 0/55 audit improvement.
The family and margin gates fail. No post-audit substitution is made and no
production state changes.

WP111 evaluates the globally ranked 1540--1595 block. Its independently
acquired target DEM height is available only from the 10 m GSI source, while
both accepted antenna-height calibration anchors use the 1 m laser source.
The frozen GSI gate prohibits mixing those sources, so candidate fitting fails
closed before ambiguity selection. No lower-quality Up prior is substituted
and no production state changes.

WP112 evaluates 11440--11495 and removes the single-median GSI supply
bottleneck. A new truth-free cache samples 17 fixed trajectory epochs, keeps
only the four points matching the accepted 1 m laser calibration source, and
selects the densest correction cluster bounded to 0.5 m. Epochs 11491 and
11494 agree within 0.0519 m and freeze the Up prior at +0.0567 m. Official API
acquisition now has bounded schema-validated retry; runtime remains offline.

With that prior, rank 0 has a 22/55 truth-seeded diagnostic ceiling. A 49-cell
grid supplies 59 hypotheses, 39 strict absolute modes, and a best diagnostic
candidate of 14/55. The frozen posterior accepts candidate 55 with family
ranks 1/1/4 and a 100% runner margin, but its audit gain is 0/55. No diagnostic
candidate is substituted after selection. Production remains 3,268/11,924 =
27.4069%; FIX and false FIX remain zero and M4 remains exact.

WP113 applies the multi-sample GSI gate to 11385--11440. Fifteen of 17 fixed
samples match the accepted 1 m laser source; the densest eight-point cluster
has 0.486 m spread and freezes the Up prior at -2.918 m. Rank 1 supplies a
three-mode posterior whose candidate 1 ranks 1/1/1 in road/carrier/CPPR
evidence and beats the runner by 133.3%. The winner reaches its 16/55
truth-seeded diagnostic ceiling without using truth in selection. Both
independent holdouts fail closed. Full-denominator application gains 16 epochs
with zero loss, advancing Tokyo to 3,284/11,924 = 27.5411%. FIX and false FIX
remain zero, production and shadow are byte-identical, and M4 remains exact.

WP114 rescans the WP113 production baseline. There are 146 remaining all-bad
supply-pass blocks; after excluding measured WP108--WP112 rejections, the next
highest carrier supply is 10890--10945. Its fixed multi-sample GSI cache has a
two-point 1 m laser cluster with 0.226 m spread and freezes Up at -7.593 m.
Rank 2 has the strongest 35/55 truth-seeded diagnostic ceiling, but every
initial hypothesis fails the 4.0 m DDPR RMS gate. A 49-cell grid expands supply
to 53 hypotheses without lowering the minimum DDPR RMS below 4.0324 m. The
gate is not relaxed; the block is rejected and WP113 production remains exact.

WP115 evaluates 935--990. All 17 fixed GSI samples match the 1 m laser source
and form one 0.265 m Up-correction cluster. Rank 0 and rank 1 both have a 55/55
diagnostic ceiling. The initial rank-0 winner itself is 55/55 but narrowly
fails road-family rank and runner-margin gates. A 49-cell grid produces 50
absolute modes and an accepted posterior winner with ranks 1/5/8 and a 42.9%
margin, but that winner improves 0/55 epochs. The 55/55 diagnostic candidate
is not substituted after audit. No production state changes.

WP116 evaluates 11605--11660. All 17 fixed GSI samples match the 1 m laser
source and form a tight 0.123 m Up-correction cluster. Rank 0 contains a 55/55
diagnostic candidate, but its initial posterior chooses a 0/55 basin and fails
the family gate. A 49-cell grid produces 57 absolute modes; candidate 54 wins
with family ranks 17/7/6 and only a 6.7% runner margin, while improving 0/55
epochs. Family and margin gates both fail. No post-audit substitution is made
and WP113 production remains unchanged.

WP117 evaluates 4840--4895. All 17 GSI samples match the 1 m laser source and
the densest ten-point cluster freezes Up at +5.345 m with 0.378 m spread.
Carrier supply remains strong, but every reference basis has zero strict
candidates: minimum DDPR RMS is 21.85--21.92 m against the frozen 4.0 m gate.
This is a structural code/carrier inconsistency rather than a seed-grid supply
problem, so no grid or threshold relaxation is attempted. Production remains
unchanged.

WP118 evaluates 10945--11000 with a six-point, 0.396 m GSI Up cluster. All
three initial carrier references have strict candidates but zero audit gain;
rank 0 has the highest 50/55 truth-seeded diagnostic ceiling. Its 49-cell grid
finds a 12/55 posterior winner, but carrier-family rank 11 exceeds the limit 10
and the 15.4% runner margin misses the frozen 20% minimum. An independent rank
1 grid yields 53 absolute modes and passes the posterior gates with ranks
1/3/9 and a 30.8% margin, but its winner improves 0/55 epochs while another
diagnostic candidate reaches 17/55. No post-audit substitution or gate change
is made; production remains unchanged.

WP119 evaluates 880--935. All 17 fixed GSI samples match the 1 m laser source
and form one 0.176 m Up-correction cluster. Every carrier-reference basis
independently produces a two-mode, ranks-1/1/1 posterior winner with a 66.7%
margin and reaches the 55/55 diagnostic ceiling. Rank 0 is frozen before audit
because its 270 CP/PR checked pairs are the largest truth-free supply. Both
independent holdouts fail closed. Full-denominator application gains all 55
epochs with zero loss, advancing Tokyo to 3,339/11,924 = 28.0023%. FIX and
false FIX remain zero, production and shadow are byte-identical, and M4 is
unchanged.

WP120 evaluates 1045--1100. All 17 fixed GSI samples match the 1 m laser
source and form one 0.161 m Up-correction cluster. Carrier-reference ranks 1
and 2 independently produce two-mode, ranks-1/1/1 posterior winners with 100%
runner margins and reach 55/55. Rank 1 is frozen before audit because its 217
CP/PR checked pairs exceed rank 2's 195. Both holdouts fail closed.
Full-denominator application gains all 55 epochs with zero loss, advancing
Tokyo to 3,394/11,924 = 28.4636%. FIX and false FIX remain zero, production
and shadow are byte-identical, and M4 is unchanged.

WP121 evaluates 1320--1375. All 17 GSI samples match the 1 m laser source and
form one tight 0.123 m Up-correction cluster. Carrier and block-stability
supply are healthy, but every reference basis has zero strict candidates:
minimum DDPR RMS is 14.88--14.92 m against the frozen 4.0 m gate. This is a
structural code/carrier inconsistency, so no grid or threshold relaxation is
attempted. WP120 production remains unchanged.

WP122 evaluates 1430--1485. All 17 GSI samples match the 1 m laser source and
form one 0.281 m Up-correction cluster. Every carrier reference has a 55/55
truth-seeded diagnostic ceiling, but every candidate fails the frozen DDPR RMS
gate: the per-basis floors are 4.741--4.877 m against the 4.0 m maximum. The
DDPR floor is not relaxed and no post-audit candidate is admitted. WP120
production remains unchanged.

WP126 evaluates 1375--1430. Fifteen fixed GSI samples match the 1 m laser
source and form one 0.422 m Up-correction cluster. Every carrier reference has
zero strict candidates because minimum DDPR RMS is 7.357--7.385 m against the
frozen 4.0 m gate. Together with WP121, WP122, and WP125 this measures a
continuous 1265--1485 code/carrier-inconsistent band. No grid or threshold
relaxation is attempted and WP120 production remains unchanged.

WP127 evaluates 1155--1210. All 17 fixed GSI samples match the 1 m laser
source and form a very tight 0.092 m Up-correction cluster. Rank 0 has a 53/55
truth-seeded diagnostic ceiling but misses the DDPR RMS gate by 0.0120 m. A
49-cell grid expands supply to 55 hypotheses without changing the 4.012024 m
DDPR floor. The frozen 4.0 m gate is not relaxed, so no candidate is admitted
and WP120 production remains unchanged.

WP123 evaluates 11275--11330. All 17 GSI samples match the 1 m laser source
and form one 0.146 m Up-correction cluster. Every carrier reference has a
55/55 truth-seeded diagnostic ceiling, but all candidates fail the frozen
DDPR RMS gate by a structural margin: the per-basis floors are 19.41--19.60 m
against the 4.0 m maximum. No grid or threshold relaxation is attempted and
WP120 production remains unchanged.

WP124 evaluates 4895--4950. All 17 GSI samples match the 1 m laser source;
the densest 11-point cluster has 0.486 m spread and freezes Up at +4.257 m.
Every carrier reference has zero strict candidates because minimum DDPR RMS is
6.457--6.495 m against the frozen 4.0 m gate. The diagnostic ceiling is 39/55,
but no inconsistent candidate is admitted and WP120 production is unchanged.

WP125 evaluates 1265--1320. All 17 GSI samples match the 1 m laser source and
form one 0.213 m Up-correction cluster. Every carrier reference has zero strict
candidates because minimum DDPR RMS is 11.44--11.47 m against the frozen
4.0 m gate. This continues the structural code/carrier inconsistency measured
in adjacent WP121. No grid or threshold relaxation is attempted and WP120
production remains unchanged.

WP128 evaluates 1595--1650. The source-matched GSI consensus uses the densest
four-point 1 m laser cluster from seven compatible samples, with 0.105 m spread,
and freezes Up at -1.441 m. All three carrier references reach DDPR floors below
the frozen 4.0 m gate, but every hypothesis fails the independent 0.5 m block
spread gate: the per-basis minima are 1.457--2.572 m. Even the truth-seeded
diagnostic has 4.293--5.191 m spread, so this is structural within-block
inconsistency rather than a seed-supply miss. No grid or threshold relaxation is
attempted and WP120 production remains unchanged.

WP129 evaluates 5225--5280. Thirteen of 17 fixed 1 m laser GSI samples form a
0.496 m Up-correction cluster and freeze Up at +4.196 m. All three carrier
references supply strict candidates. The initial rank-1 selector passes its
truth-free family and margin gates but selects a zero-gain basin. A predeclared
49-cell grid then supplies 59 hypotheses, including a 50/55 diagnostic candidate,
but the frozen selector chooses candidate 53 with zero gain and independently
fails both road-family rank (17 > 11) and runner margin (0.0625 < 0.2). The
diagnostic candidate is not substituted after audit, so WP120 production remains
unchanged and this block is locked as a posterior-selection failure.

WP130 evaluates 11660--11715. All 17 fixed GSI samples match the 1 m laser
source and form a 0.362 m Up-correction cluster, freezing Up at -0.540 m. Each
carrier reference supplies strict candidates, but every initial candidate has
zero gain. The rank-1 49-cell grid supplies 59 hypotheses and a 17/55 diagnostic
candidate at 3.996 m DDPR RMS and 0.202 m block spread. The frozen selector
instead chooses candidate 11 with zero gain and fails both family rank (11 > 9)
and runner margin (0.0 < 0.2). No post-audit candidate substitution is allowed,
so WP120 production remains unchanged and this block is locked as another
posterior-selection failure.

WP131 adds an independent cross-basis/CPPR selector without using audit labels.
It refits the same rank-1 grid seeds under carrier reference ranks 0 and 2, then
dense-ranks cross-reference convergence, maximum carrier RMS, and CP/PR evidence.
On 11660--11715 it selects candidate 24 with family ranks 2/3/1 and a 2.0 runner
margin. Nagoya WP53 abstains because its historical artifact lacks CP/PR evidence;
Tokyo WP129 fails the family gate instead of promoting its zero-gain winner. The
full 11,924-epoch audit gains 17 epochs and loses none, moving Tokyo production
to 3,411/11,924 (28.6062%) with FIX 0, false FIX 0, and M4 unchanged.

WP132 evaluates 1650--1705. All 17 fixed 1 m laser GSI samples form a tight
0.165 m cluster and freeze Up at -1.480 m. The initial rank-0/rank-1 pools and
a 49-cell rank-1 grid miss the useful basin. Rank 2 supplies candidate 10, and
cross-refits under ranks 0 and 1 show only 0.0395 m disagreement. The WP131
fusion ranks it CP/PR first but cross/carrier third of eight, so the frozen
top-20% family gate correctly abstains pending a separately validated rule.

WP133 adds a narrow CP/PR-anchor gate without changing WP131. It requires CP/PR
rank 1, both cross/carrier ranks within the top 40%, and a runner margin of at
least 20%, while retaining every WP131 absolute residual and stability gate.
Candidate 10 passes at ranks 3/3/1 with a 42.86% margin. The unsafe WP129 winner
fails because its CP/PR rank is 7, and Nagoya WP53 fails closed because CP/PR
evidence is unavailable. Full-denominator application gains 55 epochs and loses
none, moving Tokyo production to 3,466/11,924 (29.0674%). FIX and false FIX stay
at zero, runtime FGO remains disabled, and both M4 hashes remain exact.

WP134 evaluates the next strongest unassessed all-bad supply block, 1485--1540
(289 carrier rows and 239 DDPR rows). Fifteen of 17 GSI samples use a compatible
source and the densest 14-point cluster has 0.309 m spread, freezing Up at
-1.032 m. Nevertheless all three carrier references have zero strict candidates:
their minimum DDPR RMS values are 6.740--6.821 m against the frozen 4.0 m gate.
Even the truth-only diagnostic fits have 7.823--7.831 m DDPR RMS despite a 55/55
ceiling. This is structural code/carrier inconsistency, not a seed-supply miss;
the gate is not relaxed and WP133 production remains unchanged.

WP135 evaluates 1210--1265, the next strongest unassessed all-bad block after
WP134. All 17 fixed GSI samples match the 1 m laser source and form one 0.123 m
cluster, freezing Up at -0.903 m. All three carrier references have zero strict
candidates because their minimum DDPR RMS values are 9.949--10.002 m. The
truth-only diagnostic fits also measure 10.851--10.857 m DDPR RMS despite
52--54/55 ceilings. This is another structural code/carrier inconsistency, so
the frozen 4.0 m gate is not relaxed and WP133 production remains unchanged.

WP136 evaluates 11330--11385. The global supply scan selects stride phase 3;
an initial phase-0 diagnostic has no observations and is explicitly excluded.
At phase 3, all 17 fixed GSI samples form a 0.380 m cluster and freeze Up at
-2.925 m. The three carrier references retain 247--275 rows, but their minimum
DDPR RMS values are 16.313--16.490 m. Truth-only diagnostics likewise measure
17.622--17.628 m despite 53/55 ceilings. Block stability is not limiting; this
is structural code/carrier inconsistency. The 4.0 m gate remains fixed and
WP133 production remains unchanged.

WP137 evaluates 5005--5060 at the global scan's stride phase 1. All three
carrier references pass the absolute gates with 11--12 strict candidates and
1.023--1.088 m minimum DDPR RMS, but the initial pools have zero gain against a
55/55 diagnostic ceiling. A coarse 49-cell horizontal grid (3 m radius, 1 m
step) also supplies only zero-gain candidates. The grid builder's independently
fixed fine configuration (1.5 m radius, 0.5 m step, still 49 cells) supplies
three 55/55 candidates. All audit fields are then removed before selection.

WP138 reranks the resulting truth-free cross-basis modes using four direct
stability families: rank-0-to-rank-2 disagreement, within-basis block spread,
maximum cross-basis carrier RMS, and CP/PR rank sum. Candidate 26 wins with
ranks 6/3/10/3 and a 36.36% runner margin. The unsafe WP129 holdout ties at the
top and also exceeds its family-rank limit; Nagoya WP53 abstains because CP/PR
evidence is unavailable. Full-denominator application gains all 55 epochs and
loses none, moving Tokyo to 3,521/11,924 (29.5287%). FIX and false FIX remain
zero, runtime FGO remains disabled, and M4 remains unchanged.

WP139 evaluates the adjacent 4950--5005 phase-1 block. All three initial
carrier-reference pools pass the absolute gates near 1.40 m DDPR RMS but contain
only zero-gain candidates against a 55/55 diagnostic ceiling. The fixed fine
49-cell grid supplies candidate 25 at 55/55 and candidate 32 at 46/55. After
all audit fields are removed, cross-basis refits and the frozen WP138 selector
choose zero-gain candidate 55; its 18.75% runner margin also misses the fixed
20% minimum. No diagnostic candidate is substituted after audit. WP138
production remains unchanged and the block is locked as a posterior-selection
failure.

WP140 evaluates 11220--11275 at the global scan's stride phase 3. All 17 fixed
GSI samples match the 1 m laser source and form one 0.119 m cluster, freezing
Up at -2.880 m. The three carrier references retain 234--256 rows, but all
three have zero strict candidates: their minimum DDPR RMS values are
16.641--16.703 m against the frozen 4.0 m gate. Truth-only diagnostics likewise
measure 17.914--17.931 m despite 55/55 ceilings. Block stability is not
limiting; this is structural code/carrier inconsistency, matching the adjacent
WP136 block. The gate is not relaxed and WP138 production remains unchanged.

WP142 evaluates 2530--2585 at the global scan's stride phase 0. All 17 fixed
GSI samples match the 1 m laser source and form one 0.317 m cluster, freezing
Up at -0.550 m. The three carrier references retain 217--239 rows, but all
three have zero strict candidates: their minimum DDPR RMS values are
16.826--17.319 m against the frozen 4.0 m gate. Truth-only diagnostics measure
21.85--21.90 m with only 43/55 ceilings and an unstable rank-2 diagnostic
block spread of 11.08 m. This is structural code/carrier inconsistency; the
gate is not relaxed and WP138 production remains unchanged.

WP141 evaluates 5115--5170 at the global scan's stride phase 1. All 17 fixed
GSI samples form a 0.355 m cluster and freeze Up at +4.532 m. All three
carrier references pass the absolute gates with 2--5 strict candidates near
2.6 m DDPR RMS, but every initial-pool candidate is zero-gain against a 55/55
diagnostic ceiling. The fixed fine 49-cell grid supplies 53 hypotheses with 52
strict candidates, including diagnostic candidates 22 and 23 at 55/55. After
audit fields are removed, the cross-basis consensus selects candidate 26 with
a 35.6% margin, but the frozen WP138 stability selector's winner (zero-gain
candidate 45) beats runner 51 by only 13.3%, missing the fixed 20% minimum.
No diagnostic candidate is substituted after audit. WP138 production remains
unchanged and the block is locked as a posterior-selection failure with
demonstrated useful supply.

WP144 evaluates 4785--4840 at the global scan's stride phase 1. Seven of 17
GSI samples form the densest bounded cluster with 0.373 m spread, freezing Up
at +4.925 m. The three carrier references retain 234--266 rows, but all three
have zero strict candidates: their minimum DDPR RMS values are 45.42--45.52 m
against the frozen 4.0 m gate, the largest structural floor measured in the
chain so far. Truth-only diagnostics likewise measure 46.75 m with only
29--30/55 ceilings. This is structural code/carrier inconsistency; the gate is
not relaxed and WP138 production remains unchanged.

WP143 evaluates 5060--5115 at the global scan's stride phase 1. All 17 fixed
GSI samples form a 0.309 m cluster and freeze Up at +4.285 m. All three
carrier references pass the absolute gates with 9--12 strict candidates near
2.0--2.2 m DDPR RMS, but every initial-pool candidate is zero-gain against a
55/55 diagnostic ceiling. The fixed fine 49-cell grid supplies 60 hypotheses
with 59 strict candidates, including diagnostic candidate 24 at 55/55. After
audit fields are removed, the cross-basis consensus itself fails its gate with
a 3.5% runner margin, and the frozen WP138 stability selector's winner
(zero-gain candidate 22) carries a carrier-RMS family rank of 18 against the
limit of 12. Nothing is accepted, no diagnostic candidate is substituted, and
WP138 production remains unchanged; the block is locked as a
posterior-selection failure with demonstrated useful supply.

WP145 evaluates 11165--11220 at the global scan's stride phase 3. Nine of 17
GSI samples form the densest bounded cluster with 0.452 m spread, freezing Up
at -3.358 m. The three carrier references retain 246--263 rows, but all three
have zero strict candidates: their minimum DDPR RMS values are 16.846--17.068 m
against the frozen 4.0 m gate, and several hypotheses also exceed the 0.5 m
block-spread gate. Truth-only diagnostics measure 18.11--18.13 m with only
31/55 ceilings. This is structural code/carrier inconsistency, adjacent to the
WP136 and WP140 blocks; the gate is not relaxed and WP138 production remains
unchanged.

WP146 evaluates 3025--3080 at the global scan's stride phase 0. All 17 fixed
GSI samples form a 0.240 m cluster and freeze Up at -10.629 m. The block fails
both frozen gates structurally: minimum DDPR RMS is 10.414--11.085 m against
4.0 m, and minimum block spread is 3.552--10.860 m against 0.5 m. Even the
truth-seeded oracle diagnostics measure 10.47--12.58 m DDPR RMS with 5.96--14.29
m block spreads and recover 0 of 55 epochs, the first measured zero-ceiling
oracle in the chain. Constant-offset block recovery is structurally impossible
here; the gates are not relaxed and WP138 production remains unchanged.

WP147 evaluates 7095--7150 at the global scan's stride phase 1, the last
unassessed block of the WP106 all-bad ranking. Fourteen of 17 GSI samples form
a 0.465 m cluster and freeze Up at +0.844 m. The block fails both frozen gates
structurally: minimum DDPR RMS is 27.129--27.211 m against 4.0 m and minimum
block spread is 3.965--4.070 m against 0.5 m. Truth-only diagnostics measure
31.14--31.15 m DDPR RMS with 5.07--6.05 m block spreads and only 11/55
ceilings. This is structural code/carrier inconsistency compounded by block
instability; the gates are not relaxed and WP138 production remains unchanged.
This completes the assessment of all eight ranked all-bad blocks: six are
structural rejections (WP140, WP142, WP144, WP145, WP146, WP147, joining the
earlier WP134, WP135, and WP136 locks), and WP141 and WP143 are
posterior-selection failures with demonstrated useful supply, joining WP139 in
awaiting a separately named, holdout-validated selector.

With the ranked list exhausted, the campaign re-assesses pre-fine-grid-era
posterior failures under the current machinery, ordered by carrier supply.
WP149 re-assesses 990--1045 (WP109's block) with the 17-sample GSI cache;
twelve of 17 samples form a 0.454 m cluster freezing Up at -1.164 m. All three
carrier references pass the absolute gates with 6--7 strict candidates near
1.0 m DDPR RMS, and the initial pools carry diagnostic supply up to 55/55 at
rank 2 without any fine grid. After sanitization the cross-basis consensus
fails its own gate, and the frozen WP138 stability selector's winner (zero-gain
candidate 5) beats runner 2 by only 16.7%, missing the fixed 20% minimum. The
rejection is locked without expanded-supply reruns, and WP138 production
remains unchanged.

WP148 re-assesses 1100--1155, WP108's pre-fine-grid posterior failure and the
highest-carrier-supply block in the scan (400 rows). All 17 fixed GSI samples
form a 0.085 m cluster, the tightest in the chain, freezing Up at -0.981 m.
All three carrier references pass the absolute gates with 4--7 strict
candidates near 1.9--3.3 m DDPR RMS, and the rank-1 initial pool carries
direct supply without any fine grid. After sanitization the WP53 consensus
gate fails on its own margin, but the frozen WP138 stability selector accepts
candidate 1 as a unique cross-basis stability/CPPR mode with every family rank
first and a 100% runner margin. Both promotion holdouts fail closed, M4 stays
exact, and full-denominator application gains all 55 epochs and loses none,
moving Tokyo production to 3,576/11,924 (29.9899%). FIX and false FIX remain
zero and runtime FGO remains disabled. The post-freeze audit confirms the
selected candidate at 55/55 with 0.045 m median error. Production advances
from WP138 to WP148.

WP151 re-assesses 11440--11495, WP112's pre-fine-grid posterior failure,
against the WP148 production trajectory. Only four of 17 GSI samples are
compatible and two form the accepted cluster, freezing Up at +0.057 m. All
three carrier references pass the absolute gates but every initial candidate
is zero-gain, and the truth-seeded oracle itself tops out at 19--22/55 with a
0.61--0.63 m median error, so even a perfect constant offset recovers less
than half the block. The fixed fine 49-cell grid supplies 56 hypotheses with
34 strict candidates, the best diagnostic reaching only 20/55. The cross-basis
consensus fails its gate outright, and the frozen WP138 selector's winner
(candidate 35, 17/55 diagnostic) carries a cross-refit family rank of 3
against the limit of 2, so nothing is accepted. The rejection is locked;
WP148 production remains unchanged.

WP150 re-assesses 11550--11605, WP110's pre-fine-grid posterior failure,
against the WP148 production trajectory. All 17 fixed GSI samples form a
0.155 m cluster freezing Up at -0.298 m. All three carrier references pass
the absolute gates with 4--6 strict candidates near 0.7--0.9 m minimum DDPR
RMS, and the rank-1 initial pool carries direct supply without any fine grid.
The WP53 cross-basis consensus selects candidate 2 with a 34.4% margin, and
the frozen WP138 stability selector accepts the same candidate as a unique
cross-basis stability/CPPR mode with family ranks 1/1/2/1 and an 80% runner
margin. Both promotion holdouts fail closed, M4 stays exact, and
full-denominator application gains all 55 epochs and loses none, moving Tokyo
production to 3,631/11,924 (30.4512%). FIX and false FIX remain zero and
runtime FGO remains disabled. The post-freeze audit confirms the selected
candidate at 55/55 with 0.402 m median error. Production advances from WP148
to WP150.

WP152 re-assesses 935--990, WP115's pre-fine-grid posterior failure. All 17
GSI samples form a 0.265 m cluster freezing Up at -1.101 m. All three carrier
references pass the absolute gates; the rank-0 initial pool holds one 55/55
diagnostic candidate but the rank-1 source pool is zero-gain, so the fixed
fine 49-cell grid supplies 52 hypotheses, all 52 strict, including diagnostic
candidates 19 and 20 at 55/55. After sanitization the cross-basis consensus
fails its gate at a 7.2% margin, and the frozen WP138 stability selector's
winner (zero-gain candidate 41) beats runner 3 by only 7.1%, missing the fixed
20% minimum. Nothing is accepted, nothing is substituted, and WP150 production
remains unchanged; the block is locked as a posterior-selection failure with
demonstrated useful supply.

WP153 re-assesses 11605--11660, WP116's pre-fine-grid posterior failure,
against the WP150 production trajectory. All 17 GSI samples form a 0.123 m
cluster freezing Up at -0.396 m. All three carrier references pass the
absolute gates but the initial pools are zero-gain, so the fixed fine 49-cell
grid supplies 57 hypotheses, all strict. The WP53 consensus picks candidate 55
with a 39.9% margin, and the frozen WP138 stability selector accepts candidate
18 with family ranks 5/1/11/3 and a 35% runner margin. Both promotion holdouts
fail closed, M4 stays exact, and full-denominator application gains 5 epochs
and loses none, moving Tokyo production to 3,636/11,924 (30.4931%). The
post-freeze audit measures the selected candidate at 5/55 with a 0.54 m median
error while unselected diagnostic candidates 25 and 26 measure 55/55; no
substitution is made after audit, and the small promoted gain is recorded as
the frozen selector's own choice. FIX and false FIX remain zero. Production
advances from WP150 to WP153.

WP155 re-attempts 1540--1595, WP111's single-median GSI source mismatch,
with the 17-sample multi-source cache generation. All 17 fixed target samples
resolve to the 10 m DEM while the accepted-anchor calibration uses the 1 m
laser source, so the compatibility gate fails closed before any carrier refit
runs ("fewer than two compatible samples"). This is a permanent GSI 1 m laser
coverage gap at this location, not a pipeline-generation defect; the source
gate is not weakened and WP153 production remains unchanged.

WP154 re-assesses 10945--11000, WP118's pre-fine-grid posterior failure. Ten
of 17 GSI samples are compatible and six form a 0.396 m cluster freezing Up at
-2.018 m. All three carrier references pass the absolute gates but the initial
pools are zero-gain against 47--50/55 oracle ceilings, so the fixed fine
49-cell grid supplies 54 hypotheses with 53 strict, including diagnostic
candidate 29 at 46/55. After sanitization the cross-basis consensus fails its
gate at a 4.7% margin, and the frozen WP138 selector's winner (zero-gain
candidate 53) fails both the family-rank limit (cppr rank 17 against 15) and
the 20% margin (8.3%); the 46/55 candidate sits as runner and is not
substituted. The rejection is locked and WP153 production remains unchanged.

WP157 re-assesses 1155--1210, WP127's borderline 4.012 m DDPR floor. Under
the 17-sample GSI cache all 17 samples form a 0.092 m cluster freezing Up at
-0.936 m, and the minimum DDPR RMS drops to 3.84--3.91 m, re-entering the
frozen 4.0 m gate: the old +0.012 m structural call was generation-dependent.
The rank-1 initial pool carries direct supply, the WP53 consensus fails its
own margin gate, but the frozen WP138 stability selector accepts candidate 1
with family ranks 2/1/1/1 and a 120% runner margin. Both promotion holdouts
fail closed, M4 stays exact, and full-denominator application gains 53 epochs
and loses none, moving Tokyo production to 3,689/11,924 (30.9376%). The
post-freeze audit confirms 53/55 at 0.073 m median error. FIX and false FIX
remain zero. Production advances from WP153 to WP157.

WP156 re-assesses 10890--10945, WP114's borderline 4.032 m DDPR floor. Under
the 17-sample cache only two samples are compatible but they cluster at
0.226 m, freezing Up at -7.593 m, and the minimum DDPR RMS drops to
3.66--3.69 m, re-entering the gate as with WP157. The initial pools are
zero-gain against low 24--36/55 oracle ceilings, and the fixed fine grid's 52
hypotheses contain only one strict candidate and no diagnostic gain anywhere.
The WP53 consensus fails its margin gate, but the frozen WP138 selector
accepts candidate 18 with a 23.5% margin -- the first measured fine-grid-era
false acceptance: the post-freeze audit shows 0/55 at 5.75 m median error,
and full-denominator application measures gained 0, lost 0, failing the
frozen gained-epochs requirement. Production is not advanced, the rejected
application artifacts are retained as evidence, and the block's selection
chain is recorded as a new unsafe-acceptance holdout candidate alongside
WP129. WP157 production remains unchanged.

WP158 introduces the truth-free DDPR satellite screen. A per-satellite
residual anatomy at truth positions shows the 16.6--17.9 m DDPR floor of
11220--11275 (WP140's structural lock) is caused entirely by seven NLOS
satellites -- G06, G07, G13, G15 with stable 55--64 m biases and C26, C39,
C42 with intermittent 30--70 m switching -- while Galileo and QZSS sit at
0.47--0.48 m and the reference satellites are innocent. A truth-free
per-epoch triple-difference mutual-consistency clustering rule (edge 5 m,
outlier fraction 0.2) recovers the full culprit set from production
positions alone, flags nothing but one genuinely biased satellite on a
healthy control block, and is frozen as
experiments/build_wp158_ddpr_satellite_screen.py with an opt-in
--exclude-ddpr-satellites flag in the refit (default off; DDPR pairs only;
carrier untouched; recorded in every artifact). Screened refits re-enter
the frozen gates at 1.52--1.62 m minimum DDPR RMS with 55/55 diagnostic
supply in the rank-1 pool, and for the first time the frozen ranking places
the correct basin first: the WP138 winner (candidate 1) audits 55/55 at
0.103 m. The selection nevertheless fails, solely on the
cross-refit-disagreement family rank (10 against a limit of 4) and a 13.3%
runner margin. Nothing is substituted; the rejection is locked and WP157
production remains unchanged. The result isolates basis-swap stability as
the one remaining family that anti-correlates with correct basins, and any
successor selector must be separately named and validated against the
WP129, Nagoya WP53, and WP156 holdouts.

WP159 defines that successor: a screened-scope selector
(experiments/select_wp159_screened_stability_consensus.py, schema
wp159_screened_stability_consensus_v1) that ranks over block_spread_m,
max_cross_basis_carrier_rms_cycles, and cppr_rank_sum -- removing
cross_refit_disagreement_m from the RANKING only, while its 0.10 m absolute
eligibility gate and every other WP131/WP138 absolute gate stay unchanged
-- and that fails closed with screen_evidence_required unless the source
artifact carries a non-empty ddpr_excluded_satellites list. Because the
screen is opt-in and recorded, WP159 can never touch an unscreened chain:
replaying all ten stored chains (WP129, Nagoya WP53, WP139, WP141, WP143,
WP149, WP152, WP154, WP156, WP158) accepts only the screened WP158 chain.
The screened regime itself is holdout-tested by re-running the WP156
zero-gain block 10890--10945 through the FULL screened chain (its own
screen flags nine satellites; the screened rank-1 pool still audits 0/55
everywhere): WP159 rejects it on the family-rank gate, and a WP138
comparison run rejects it too, so the screen does not reintroduce the
WP156 false-acceptance mode.

WP160 promotes the WP158 screened chain under the WP159 selector. The
promoter (experiments/promote_wp159_screened_stability_consensus.py)
recomputes the selection from the hashed truth-free source and cross-basis
consensus, verifies the stored WP159 selection byte-for-byte on the
comparable keys, and requires both holdouts to fail closed (WP129 unsafe
pool and Nagoya WP53 missing-evidence, both screen_evidence_required).
Candidate 1 -- ranked 3/1/1 on the three screened families with a 1.0
runner margin -- is applied to the WP157 trajectory with the frozen
full-denominator application: gained 55, lost 0, FIX and false-FIX
unchanged at 0, M4 artifacts intact. Tokyo run1 production advances
3689 -> 3744 of 11924 (30.94% -> 31.40%), the first recovery of a
formerly structural DDPR-floor block, locked in
internal_docs/wp160_tokyo_screened_stability_promotion_2026_07_24.json
with production trajectory
results/wp31/tokyo_run1_wp160_screened_stability_full_trajectory.csv.

WP161 rolls the screened chain onto 11165-11220 (WP145's structural lock,
oracle ceiling 31/55 at that assessment). The screen flags seven satellites
(G07/G15/G30 at outlier fraction 1.0 with 33-84 m median residuals) and the
screened refits re-enter every per-rank truth-free gate at a 1.49-1.70 m
DDPR floor -- but the WP53 cross-basis consensus fails on a
disagreement/spread trade-off across all seven candidates (tight-agreement
candidates carry up to 4.57 m block spread; tight-spread candidates exceed
the 0.05 m disagreement gate), and WP159 therefore fails closed with
stability_cppr_evidence_unavailable (a single cross-basis mode). Nothing
is substituted; WP160 production is unchanged; locked in
internal_docs/wp161_tokyo_screened_consensus_rejection_2026_07_24.json.
The outcome is the expected honest failure mode for a low-oracle-ceiling
block: the screen restores evidence QUALITY, but cannot manufacture basin
CONSISTENCY that the underlying geometry does not support.

WP162 applies the same chain to 2530-2585 (WP142's structural lock, the
queue's highest recorded oracle ceiling at 43/55). The screen flags nine
satellites (seven at outlier fraction 1.0) and all three screened refit
ranks pass every truth-free gate with 1.43-1.53 m DDPR floors. This time
exactly ONE candidate (id 8) passes every absolute supply gate at the
WP53 stage (consensus 0.145 m, carrier 0.204 cycles, spread 0.482 m) while
all others fail on block spread -- but a single qualifying candidate has
no runner to establish the 0.2 minimum margin, WP53 fails closed at margin
0.0, and WP159 follows with stability_cppr_evidence_unavailable
(mode_count 1). Nothing is substituted; WP160 production is unchanged;
locked in
internal_docs/wp162_tokyo_screened_consensus_rejection_2026_07_24.json.
Unlike WP161 this block dies on the single-mode margin requirement rather
than on supply quality -- the frozen consensus machinery demands at least
two independently qualifying basins and this geometry yields one.

WP163 (7095-7150, WP147's structural lock, constant-model oracle bound
29/55) is the first screened block to fail at the REFIT stage: the screen
flags five satellites (E05 and G15 at outlier fraction 1.0) and drops the
DDPR floor to 1.27-1.38 m, but every hypothesis at every reference rank
fails the 0.5 m block-spread gate with spreads of 1.07-7.91 m while
passing the arc, row, carrier, and DDPR gates. The chain stops before any
selector runs. This matches the Track A affine probe (29 constant vs 35
affine): the block's residual limitation is intra-block drift -- an
offset-model question, not evidence quality. Locked in
internal_docs/wp163_tokyo_screened_refit_rejection_2026_07_24.json;
WP160 production unchanged.

WP164 (4785-4840, WP144's 45 m floor -- the campaign's worst) fails the
refit stage the opposite way. The screen flags FIFTEEN satellites (13 at
outlier fraction 1.0, median residuals 10-179 m, contamination reaching
Galileo for the first time), and the screened refits collapse the floor
by an order of magnitude -- but only to 5.33-5.44 m, still above the
frozen 4.0 m gate, with block spreads that are actually tight
(<= 0.14 m). Where WP163 died on spread with clean residuals, WP164 dies
on residuals with clean spread: with most of the visible constellation
excluded, the surviving evidence is thin and still biased. Locked in
internal_docs/wp164_tokyo_screened_refit_rejection_2026_07_24.json;
WP160 production unchanged.

WP166 (1430-1485, WP122's shallow 4.79 m floor, diagnostic ceiling 55)
is the deepest screened roll-out chain yet. A light screen (four
satellites) lets every refit and cross refit pass all gates, and the WP53
cross-basis consensus ACCEPTS candidate 1 with a 32.5% margin -- the
first screened roll-out block to reach the posterior stage. WP159 then
rejects on the single runner-margin check: its top-ranked candidate (10,
which differs from the WP53 pick) leads the runner (1) by only 16.7%
against the frozen 20% requirement, with family ranks passing. Under the
frozen discipline nothing is substituted, the audit fields are not
consulted, and the rejection is locked in
internal_docs/wp166_tokyo_screened_posterior_rejection_2026_07_24.json;
WP160 production unchanged. Any margin change is a selector change
requiring a new name and full holdout validation.

WP165 (11275-11330, WP123's structural lock, ceiling 55, adjacent to the
recovered WP160 block) produces the campaign's lowest screened floors
(1.017 m across all ranks after excluding eight satellites, most shared
with WP160's culprit set) and every refit passes cleanly -- but the
rank-1 pool collapses to two gate-passing hypotheses, WP53 finds exactly
one supply-passing candidate (three-basis agreement 0.0085-0.0143 m,
inside every absolute gate) with no runner for the 0.2 margin, and WP159
fails closed on a single mode, the same shape as WP162. Locked in
internal_docs/wp165_tokyo_screened_consensus_rejection_2026_07_24.json;
WP160 production unchanged. Together with WP166 this kills the "ceiling
55 implies promotable" working model: both ceiling-55 roll-out blocks
die honestly, and the frozen machinery's insistence on opposed, margined
consensus -- not evidence quality -- is now the binding constraint on
screened recovery.

WP167 (1375-1430, WP126's lock, ceiling 52) adds a third consensus-stage
failure shape. The screen excludes eight satellites, every refit and
cross refit passes cleanly (floors 1.05-1.16 m), and this time TWO of
four candidates pass every WP53 supply gate -- but they are separated by
a 0.6% margin against the frozen 20% requirement: two statistical twins
rather than one unopposed basin. WP53 fails closed, WP159 follows on a
single mode, and the rejection is locked in
internal_docs/wp167_tokyo_screened_consensus_rejection_2026_07_24.json;
WP160 production unchanged.

WP170 (1265-1320, WP125's lock, ceiling 32 -- the queue's lowest)
produces the campaign's lowest screened DDPR floor (0.871 m from 11.47 m
unscreened) and is the first roll-out chain to reach the WP159 ranking
stage with multiple modes (three). Both consensus stages still reject
independently: WP53 on a 0.76% runner margin, and WP159 on the
family-rank check (the winner carries a rank-3 family against the limit
of 2) with its margin check passing exactly at 0.2. Locked in
internal_docs/wp170_tokyo_screened_posterior_rejection_2026_07_24.json;
WP160 production unchanged.

WP168/WP171 (1320-1375, WP121's lock, ceiling 46) is the screened
regime's WP156: the FIRST measured zero-gain false acceptance of the
WP159 selector. The screened chain passes every gate, WP53 fails closed
on a 3.8% margin, but WP159 accepts candidate 2 with maximal confidence
-- stability family ranks 1/1/1 and a 1.33 runner margin. The promoter
passes (both holdouts fail closed as designed), and the full-denominator
application then measures gained 0, lost 0: the frozen gained>0 gate
refuses the advance and production stays at WP160. The lesson mirrors
WP156 one level deeper: on a screened chain a lone dominant-but-wrong
basin can be both unopposed and maximally stable, and only the
application gate catches it. The WP168 selection triple is now a
mandatory unsafe-acceptance holdout for any WP159 successor, alongside
WP129, Nagoya WP53, and WP156. Locked in
internal_docs/wp171_tokyo_screened_zero_gain_acceptance_rejection_2026_07_24.json
with the rejected promotion and application artifacts retained as
evidence.

WP169 (4895-4950, WP124's lock, ceiling 39, GSI inliers 11/17) closes
the WP121-126 re-assessment: a 13-satellite screen (neighbouring
WP164's contamination) yields clean refits at 0.84-1.01 m floors, WP53
ACCEPTS candidate 2 with a 58.7% margin, and WP159 nevertheless fails
closed at mode_count 0 -- the CPPR mode construction finds no
qualifying modes despite the clean WP53 winner. Under the frozen
conjunction discipline the block is rejected. Locked in
internal_docs/wp169_tokyo_screened_consensus_rejection_2026_07_24.json;
WP160 production unchanged.
