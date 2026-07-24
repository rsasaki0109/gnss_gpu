# WP31 Nagoya run1 static/motion chain (2026-07-19)

## Current result

The current production run covers **4,118/7,583 = 54.3057%** of the actual full
denominator at `<50 cm`. It declares no FIX, so false FIX is 0%. This is a
measured gain over the first-anchor dead-reckon result of 598/7,583 (7.8861%),
but it is not the 86% goal.

Artifact:
`results/wp31/nagoya_run1_tdcp_gyro_gapfill_eightanchor_secondary_durationp2_full_summary.json`.

## Accepted truth-free anchors

| segment | candidate | selection reason | audit error |
|---|---:|---|---:|
| 0--199 | 23 | clear wide-lane | 0.280 m |
| 805--922 | 37/60/65 | unique secondary-frequency top-3 posterior | 0.465 m |
| 1807--2326 | 49 | clear wide-lane after 15 m then 5 m coarse-to-fine search | 0.470 m |
| 2841--3271 | 52 | high-evidence temporal/WL consensus | 0.467 m |
| 3811--4273 | 47 | temporal-arc + road + prior-height consensus | 0.403 m |
| 4792--5014 | 49 | motion-supported compact child cluster | 0.451 m |
| 5287--5606 | 10 | compact tied-WL parent marginal | 0.280 m |
| 7529--7582 | 21/31 | multimode + absolute DDPR consensus | 0.247 m |

The motion smoother keeps 6,028 high-quality TDCP intervals, fills 1,554 weak
intervals with gyro-shaped Doppler speed, estimates gyro sign -1 and bias
-0.1540 deg/s, and measures 4.135 deg p95 TDCP/gyro heading residual. Endpoint
residual is applied only inside the longest filled outage between adjacent
accepted anchors. The exact closure ranges, residual norms and ECEF vectors
are in the summary artifact.

## Fail-closed evidence

- 1508--1610: a 20-evidence consensus selected a 1.879 m mode. This motivated
  the >=30-epoch high-evidence gate; the anchor is not used.
- 3811 and 4792 initial/fine grids showed that wide-lane rank is unstable among
  sub-metre local modes. 3811 was recovered by temporal-road-height consensus;
  4792 was recovered only after a prior motion path supported a compact child
  cluster.
- At 5287 the six-anchor predecessor motion path supplied a 1.5 m local shell.
  The two tied best absolute-WL children form a 0.430 m-radius parent mode; its
  marginalized position audits at 0.280 m. The fixed gate accepts only when
  evidence >=30, best absolute WL <=0.5 m, top-child ratio <=1.1 and cluster
  spread <=0.5 m. No earlier audited negative satisfies all four conditions.
- At 6014 the same six-anchor motion prior and a 5 m cube26 shell supplied no
  sub-50 cm candidate (best audit 1.108 m). The best absolute WL is 0.745 m,
  so the fixed parent-marginal gate rejects it. It is not used.
- OSM road distance alone is not a lane-level selector; correct candidates can
  lie about 1--2 m from the mapped centerline.

## Next gates

1. Refine 6014 only from truth-free top-parent evidence; do not follow its
   audit-best child. Add multiple radii only if the parent evidence becomes
   compact and the frozen absolute-WL gate passes.
2. Search post-6599 static/motion reset opportunities before relying on a
   single 6014 discriminator.
3. With each newly accepted anchor, rerun gyro-gap endpoint closure and report
   interval-level gain. Do not declare FIX from the smoother until posterior
   integrity is independently calibrated below 1% false FIX.
4. After Nagoya chain coverage is materially higher, apply exactly the same
   frozen settings to Tokyo run1; do not tune a separate Tokyo policy.

## Common endpoint-closure ablation (2026-07-22)

The six-anchor `longest` result was reproduced exactly at 3,073/7,583.  Keeping
the same anchors, TDCP/gyro inputs, and zero FIX declarations, three deterministic
truth-free residual-distribution rules were compared:

| closure rule | sub-50 cm | full rate | delta vs longest |
|---|---:|---:|---:|
| longest (current) | 3,073 | 40.5249% | 0 |
| all filled runs | 2,958 | 39.0069% | -115 |
| duration weighted, p=2 | 3,111 | 41.0251% | +38 |
| all intervals | 2,492 | 32.8629% | -581 |

The p=2 gain is concentrated between anchors 2326 and 2841 (+37 epochs), with
small changes of -1, +5, -2, and -1 in the other four inter-anchor regions.
The same predeclared p=2 rule also improves the existing Tokyo ten-anchor
result from 3,007 to 3,051 epochs, so it is retained as a common development
candidate rather than a city-specific audit fit.  It remains unpromoted until
the rule and its interval-level safety gate are frozen and checked on unseen
anchor gaps.  Nagoya artifacts are named
`nagoya_run1_sixanchor_{all_filled,duration_weighted,all_intervals}_development_*`.

## Post-6599 static-anchor supply (2026-07-22)

The two previously untested truth-free stops after the 6015--6597 stop were
evaluated from the full L1 basin trace using the same 0.2 m recurring-mode
extractor and static DD solver.  The analyzer now streams only the requested
epoch range instead of retaining the 476 MB trace in memory; numerical behavior
is unchanged.

At 6911--6958, 64 recurring candidates and ten carrier-evidence epochs supply
no sub-metre mode.  The best audit-only error is 2.601 m, while the measurement
winner audits at 4.042 m.  The stop fails candidate supply and is rejected.

At 7529--7583, all 54 stop epochs recur and the 64-candidate pool supplies four
sub-50 cm modes: candidates 31, 21, 36, and 44 audit at 0.269, 0.315, 0.417,
and 0.470 m.  Candidate 21 ranks third by wide-lane median and seventh by
temporal-arc Cauchy score, but the joint rank winner is candidate 8, which
audits at 1.145 m.  The frozen consensus therefore correctly fails closed.
OSM cannot resolve the set: the downloaded drive network is 33--36 m from all
candidates at this stop, and candidate 21 has road rank 21.  No seventh anchor
is accepted and the 3,073-epoch locked Nagoya trajectory is unchanged.

Artifacts are `nagoya_run1_static_{6911_6958,7529_7583}_development.json` and
the latter stop's `_widelane`, `_integrity`, and `_osm` JSON files.  The next
independent discriminator is an offline-cached GSI absolute-height observation
for 7529--7583; it must be calibrated from already accepted Nagoya anchors and
must select uniquely without using the known candidate audits.

The GSI follow-up was also completed.  Official DEM/geoid responses were
frozen for accepted anchors 0--199 and 1807--2326 and for the candidate-median
7529 center.  All three locations return `5m（レーザ）`, not the selector's
required `1m（レーザ）`, so the production selector rejects with
`unsupported_dem_source`.  Even an audit-only replay allowing the 5 m source
still rejects: the two accepted-anchor antenna-height offsets are 1.590 and
1.915 m, a 0.325 m spread versus the frozen 0.15 m maximum.  The DEM source
gate is therefore not the only blocker and must not be relaxed to select a
known-good child.  Cached provenance is in
`nagoya_run1_gsi_height_calibration_cache.json` and
`nagoya_run1_gsi_height_cache_7529.json`; the fail-closed selector result is
`nagoya_run1_static_7529_7583_development_gsi_height.json`.

Because the final stop is only about 28 m from the accepted first stop, the
next independent measurement is a cross-stop carrier ambiguity fingerprint:
resolve DD integers at the accepted first anchor, carry only demonstrably
continuous satellite arcs, and rank the supplied final-stop cluster without
refitting an independent integer for every candidate.

### Cross-stop continuity and multimode development anchor

The cross-stop assignment auditor matched 144 basin rows around accepted
candidate 23 at the first stop and 5,033 rows around all 64 final-stop modes.
Six raw DD satellite pairs recur, but none retain the same versioned ambiguity
generation.  Minimum generation change is 26 and maximum is 171 for every
right candidate.  Therefore no first-stop integer is carried to 7529 and the
fingerprint gate fails closed.  Artifact:
`nagoya_run1_cross_stop_assignment_continuity_0_7529_development.json`.

A separate development selector groups candidates at 0.5 m linkage and
marginalizes independent wide-lane-median and temporal-carrier ranks.  Fixed
gates require at least three members, cluster score >=0.5, and spread <=0.5 m.
It selects candidates 21/31/36 with score 0.715, spread 0.348 m, and a
time-aligned audit error of 0.245 m.  The known 1508 false mode, three prior
Nagoya shells, and nine Tokyo shell holdouts all fail closed under exactly the
same rule.

The selector is not production-promoted.  A 7529--7556 time-split evidence
holdout finds no eligible cluster, and an independently supplied MF-basin
64-candidate pool has no sub-50 cm point (best 0.746 m).  Thus the positive
L1 result has not reproduced on an independent supply/evidence split.

As an explicitly unpromoted seven-anchor shadow, the cluster raises Nagoya
from 3,073/7,583 (40.5249%) to 3,805 (50.1780%) with longest-gap closure.  The
common duration-weighted p=2 rule reaches 3,843/7,583 (50.6792%), with zero FIX
and zero false FIX.  These are development ceilings, not locked production
claims.  Artifacts are `nagoya_run1_static_7529_7583_multimode_cluster_development.json`
and `nagoya_run1_sevenanchor_multimode_{longest,duration_weighted}_development_*`.

Finally, the independent MF-basin top-24 recurring modes at the long
6015--6597 stop also fail supply: best audit is 9.190 m versus the prior L1
best near 1.1 m.  Expanding that MF mode family is not justified.

An additional interleaved evidence holdout was added to the wide-lane and
carrier-integrity analyzers (`epoch_modulus`, `epoch_remainder`).  Even epochs
retain six wide-lane evidence epochs distributed across the complete stop,
but the unchanged multimode gate finds no eligible cluster.  Together with
the failed contiguous first-half holdout and failed independent MF supply,
this is sufficient evidence not to promote or tune the 7529 rule.  The odd
split was not run after the even split failed the predeclared requirement that
both partitions select a consistent cluster.

### Independent DD pseudorange axis and production promotion

The earlier carrier/wide-lane-only development conclusion is superseded by a
new independent absolute-code discriminator. Eleven evidence epochs provide
double-difference pseudorange residuals for all 64 supplied candidates. In
the carrier/wide-lane cluster 21/31/36, candidates 21 and 31 independently
pass the frozen 0.5 m median-residual gate at 0.457 m and 0.494 m. Their
0.158 m spatial spread passes the two-member, 0.5 m consensus gate. Candidate
21 is selected, with a truth-only audit error of 0.247 m; truth is not an
input to either gate.

The composite selector is production-promoted with reason
`multimode_ddpr_consensus`. The normal `--position-anchor` path, not the
development override, produces a seven-anchor duration-weighted p=2 result of
3,847/7,583 (50.7319%), with zero declared FIX and zero false FIX. Artifacts
are `nagoya_run1_static_7529_7583_development_ddpr_integrity.json`,
`nagoya_run1_static_7529_7583_multimode_ddpr_consensus.json`, and
`nagoya_run1_tdcp_gyro_gapfill_sevenanchor_durationp2_full_summary.json`.

This result is locked together with the unchanged Tokyo p=2 comparison in
`internal_docs/wp32_pf_only_intermediate_benchmark_2026_07_22.json`. It is an
intermediate checkpoint; the 86% Nagoya target remains open.

### Next-anchor triage after the WP32 lock

Error mass under the seven-anchor p=2 trajectory is concentrated in two
gaps. Segment 200--1807 contains 1,196 failing epochs out of 1,607, while
5607--7529 contains 1,185 out of 1,922. The latter gap's known stops at
6015--6597 and 6911--6958 already fail candidate supply, so the first gap was
tested next without changing any production threshold.

At 805--923, the original 24-candidate supply has no sub-50 cm point (best
truth-only audit 0.528 m). Absolute DDPR also fails the fixed 0.5 m gate (best
median 0.557 m), and the carrier/WL multimode selector returns
`no_eligible_multimode_cluster`. The existing 1 m refinement contains two
sub-50 audit points, but still forms no eligible multimode cluster, so it is
not promoted. At 1508--1611, the supply fails decisively: best audit is
1.708 m and best DDPR median is 1.093 m. Artifacts are
`nagoya_run1_static_{805_923,1508_1611}_development_ddpr_integrity.json` and
the corresponding `*_multimode_cluster_*` files.

The next useful unit is therefore new truth-free candidate supply near
805--923 (for example a motion-conditioned local parent), not relaxation of
the selector or DDPR thresholds.

### Motion-parent PF resampling at 805 and 6073

A production-motion parent extractor now requires a production-promoted
trajectory, contiguous epoch coverage, at least 30 epochs, p95 position spread
<=0.25 m, and maximum spread <=0.5 m. It never reads the trajectory's audit
error. At 805--923 the seven-anchor p=2 trajectory supplies 118 epochs with
0.075 m p95 spread. A fixed 1/2/3/5 m cube26 shell supplies one sub-50 audit
candidate (candidate 71 at 0.49 m), but the unchanged multimode and absolute
DDPR gates both fail. OSM ranks that candidate only 20th. DD-code pair-bias
calibration from accepted anchor 1807 also fails transfer: the correct compact
component falls to calibrated DDPR ranks 26--46. Carrier ambiguity generations
from anchor 0 are discontinuous for every candidate.

The weak posterior was then resampled without choosing an audit-best parent.
All three components with at least three members and score >=0.4 received the
same 0.2/0.4 m cube26 child shell. Parents 0 and 1 supply no sub-50 child;
parent 2 supplies 12, best 0.30 m. The v1 single-link selector chains the dense
cloud and fails. A non-chaining compact-ball development selector admits both
parent 1 and parent 2 rather than a unique posterior, and their truth-only
center audits are 1.19 m and 0.87 m. Therefore neither is production-promoted.

The same fixed motion-parent shell was tested in the symmetric 10%-trimmed
core 6073--6539 of the long stop. The untrimmed span misses the fixed p95
spread gate (0.270 m); the core passes at 0.240 m. Its 129-candidate solve
supplies one sub-50 candidate (59 at 0.47 m) and three sub-1 m candidates, but
again produces no eligible v1 multimode cluster. This stop is also rejected.

New reproducible artifacts use stems
`nagoya_run1_static_805_923_{motion_parent,cluster_resample_parents}`,
`nagoya_run1_static_805_923_resample_parent{0,1,2}_*`, and
`nagoya_run1_static_motion_shell1235_{805_923,6073_6539}_*`. The next selector
must add an actually independent posterior discriminator; more spatial
resampling alone is now measured to improve supply but not selection.

### Secondary-frequency posterior production promotion

The resampled 805 cloud is now independently resolved with secondary-frequency
DD pseudorange. The implementation prohibits fallback to the primary code
family and uses GPS L2, Galileo E5, QZSS L2, and BeiDou B2. Of the three weak
parents expanded identically, parent 0 has no compact posterior, parent 1's
best three secondary medians are all above 0.5 m, and parent 2 uniquely passes.
Candidates 37/60/65 have medians 0.403/0.408/0.422 m and 0.248 m spread; their
mean position audits at 0.465 m.

The gate has a positive holdout at the already accepted 7529 stop: candidates
21/31/36 independently pass at 0.432/0.358/0.315 m, 0.264 m spread, and 0.257 m
audit error. Nagoya 3811/4792 fail closed at the absolute secondary gate, while
Nagoya 1508 and Tokyo 1--61 fail the earlier proposal gate. Hashes and exact
settings are locked in
`internal_docs/wp32_secondary_posterior_validation_2026_07_22.json`.

Adding the production-promoted 805 anchor to the unchanged common p=2 smoother
raises Nagoya from 3,847 to 4,118 sub-50 epochs, or from 50.7319% to 54.3057%.
FIX and false FIX remain zero. The checkpoint is locked in
`internal_docs/wp33_pf_secondary_posterior_benchmark_2026_07_22.json`.

### 6073--6539 direction success, radius rejection

The 129-candidate motion shell in the largest remaining 5607--7529 outage was
also evaluated with secondary-family DDPR. Across the four fixed radii, cube26
direction 8 (IDs 33/59/85/111) wins with about 33% margin and contains the sole
sub-50 supply candidate ID59 (0.473 m audit). Radius selection is unresolved:
secondary ranks IDs 85/111/59 first, their spread is 3 m, and their audits are
0.93/2.81/0.47 m. OSM distances within the direction are nearly identical.
The official GSI response is 5 m laser DEM, while the production selector
requires 1 m laser DEM, so it fails closed as `unsupported_dem_source`.

Artifact hashes: secondary
`92BA1D372FFB49355BB3DD908D51B3D6C851DE8E67B77485A1F98D0112D7434A`,
OSM `6BF84F6684B3326667ADCD6B8765D751A066CF99CE39B4DBA8265559F4865336`,
GSI cache `7918D0E6AA03A4007377225883391713547FF5391B8DDB81691FEB0D70A5D7CE`,
and fail-closed selector
`89DD4E5BB89BDA2EC39C63ABD71C85DCFBC1A3C7DA5BB32A1AC500952CC23CE2`.

### Tertiary-code radius holdout rejection

A third, non-fallback DD pseudorange family was added (GPS L5, Galileo
E5b/E5a, QZSS L5, BeiDou B3/B2a, and GLONASS G3). Within the secondary-winning
direction 8 at 6073--6539, tertiary residuals rank ID59 second across the four
radii: IDs 33/59/85/111 have tertiary medians 0.90/0.80/0.98/1.37 m and
truth-only audits 1.29/0.47/0.93/2.81 m. This is encouraging on the target but
does not establish a production selector.

The pre-existing accepted 805--923 positive holdout rejects the hierarchy.
Secondary direction 17 wins there by 31.5%, but tertiary selection within it
chooses a roughly 0.93 m audit point; the roughly 0.50 m direction is only the
runner. Secondary-direction then tertiary-radius selection is therefore
fail-closed and 6073 remains unpromoted. SHA-256 hashes are tertiary target
`D1BEA250F995A2E1503317F5C1CACCFE43C321B2BD9492AA4EB21E1082F5CAE3`,
805 secondary holdout
`913B0B8DD062802E3DBE392BF8DB31106F8A4F657D1CD550F46F8D386C422303`,
and 805 tertiary holdout
`828BED2798B70C6396090188176E36703335158B802F15F0D7DD789E5932DC2F`.

### Common fragmentation-gated outage closure

Unconditional all-filled closure improves Tokyo but reduces this eight-anchor
Nagoya result from 4,118 to 3,967 sub-50 cm epochs. The loss is concentrated in
three inter-anchor gaps (-62, -29, and -61 epochs) where one filled outage owns
60--91% of all filled duration.

A common truth-free majority gate now uses all-filled closure only when the
longest filled run owns at most half of filled duration; otherwise it retains
duration-weighted p=2. On Nagoya it chooses all-filled in two gaps and p=2 in
five, avoids every harmful all-filled interval, and reaches 4,120/7,583
(54.3321%). The identical policy keeps Tokyo at 3,265/11,924 (27.3818%). FIX
and false FIX remain zero. Exact artifacts and implementation hashes are in
`internal_docs/wp37_pf_fragmentation_gated_benchmark_2026_07_22.json`.

### Trifrequency DDPR rank consensus and 6073 production anchor

The earlier secondary-direction then tertiary-radius hierarchy is superseded
by a symmetric rank consensus across primary, secondary, and tertiary DDPR.
The fixed gate requires at least ten evidence epochs, every family rank within
the top 20% of the identical candidate pool, and at least 20% rank-sum runner
margin. It never reads candidate audit error at runtime.

At 6073--6539, ID59 ranks 2/3/2, has rank sum 7 versus runner 13, and passes
with 85.7% margin. Its truth-only audit is 0.473 m. Accepted positive holdouts
3811 and 7529 select 0.480 m and 0.269 m candidates at 30.0% and 27.3% margins.
Unsafe supply winners at 805 and 4792 audit at 0.638 m and 0.703 m but fail
closed at 3.0% and 7.0% margins. The hash-verified validation is locked in
`internal_docs/wp38_trifrequency_ddpr_rank_validation_2026_07_22.json`.

Adding the promoted 6073 anchor to the common fragmentation-gated smoother
raises Nagoya from 4,120 to 4,790 sub-50 cm epochs, or from 54.3321% to
63.1676%. All 670 changed epochs are gains and none are losses. FIX and false
FIX remain zero. The benchmark is locked in
`internal_docs/wp39_pf_trifrequency_benchmark_2026_07_22.json`.

### Post-WP39 805--1807 supply audit

The largest remaining error interval contains 804 failing epochs. The fixed
trifrequency gate was applied unchanged to the most promising existing 1508
resample parent. It fails closed: the winner ranks 19/15/13, misses the
top-20% family gate, has only 8.5% runner margin, and audits at 1.071 m. The
known best supplied point is 0.507 m but receives no unique three-family
support, so it is not used as a center or anchor. Selector SHA-256 is
`651CF7E387B60228823FDEA43149E2F8C8ACA13E7225FB45A601F64E4EEB7714`.

The previously unmodeled truth-free stop detector span 1441--1492 was also
extracted from the complete L1 basin trace with the fixed 0.2 m recurring-mode
extractor and 64-candidate cap. Ten DDPR evidence epochs yield no sub-metre
candidate; the best truth-only audit is 1.138 m. It fails candidate supply
before posterior selection. Candidate-supply SHA-256 is
`2F541BC26677B29864D4B62B8D7D9FDF52ED137FCE7E54A1D9F169883146798E`
and solved-candidate SHA-256 is
`67602EAA49A5318EB6DB560E70A264401F053CF67B4100824F3B0CC068065BC2`.

Static anchor supply inside 805--1807 is therefore exhausted under the current
L1 recurring and motion-resampled families. The next improvement unit for
this interval is a separately gated motion/outage bridge, not threshold
relaxation or audit-directed spatial refinement.

### WP40 motion/outage bridge rejection

The 805--1807 gap is highly fragmented but dominated by the 1055--1286
filled run (231/311 filled epochs and 59.8 s). A frozen duration-exponent
sweep peaks at the existing p=2 policy: p=1.25/1.5/1.75/2/2.25/2.5/3/4
produce 4645/4681/4765/4790/4761/4755/4753/4752 full-run sub-50 cm epochs.
Longest-only and all-interval closure produce 4745 and 3709. The existing
fragmentation gate is therefore retained.

The strict long-gyro route also fails closed at 805--1807. Across endpoint
stride 10--50, best heading p95 is 16.899 degrees against the fixed 15-degree
limit and speed scale is about 1.265 against the fixed 1.2 limit. A development
relaxation to 20 degrees and 1.3 adds only one full-run epoch, so neither the
gate nor production output changes.

Finally, deterministic 4,096-particle OSM bridges were tested with the same
calibrated road band and accepted truth-free boundary anchors. Both reliable-
TDCP and all-step routes fail at 922--1807. At 2326--2841, geometry, endpoint,
and scale gates pass, but the distinct-runner score gaps are only 0.0241 and
0.0165 against the frozen 2.0 minimum. At 3271--3811 the gaps are 0.1445 and
0.0196; the reliable-TDCP variant also exceeds the road p95 bound. Truth-only
audit reaches at most 66/516 and 40/540 sub-50 cm epochs, confirming that gate
relaxation would not be useful.

WP39 therefore remains production at 4,790/7,583 (63.1676%), with FIX=0 and
false FIX=0. Exact trials, cache and implementation hashes, and the 21-test
verification are locked in
`internal_docs/wp40_nagoya_motion_outage_rejection_2026_07_22.json`. The next
development family is moving carrier/DD posterior supply, not further inertial
or OSM-only tuning.

### WP41 moving carrier/DDPR supply audit

The first fixed 55-epoch block after the 1807--2327 accepted anchor contains
11 evidence epochs and 188 carrier rows. All supplied hypotheses satisfy the
carrier RMS and four-way bootstrap-spread gates, demonstrating unresolved
integer/road multimodality. Uncalibrated absolute DDPR rejects every hypothesis;
even its non-eligible truth-seeded ceiling has 82.73 m RMS.

Two truth-free bias transfers from the preceding accepted static anchor were
tested. Satellite-bias propagation lowers the truth-seeded ceiling to 20.68 m,
and exact DD-pair median bias yields 23.06 m. Neither approaches the frozen
4 m gate, so no motion block is selected and WP39 remains unchanged. Exact
artifact and implementation hashes are locked in
`internal_docs/wp41_nagoya_moving_ddpr_supply_rejection_2026_07_22.json`.
The next supply experiment must eliminate block-local DD pair biases as
nuisance parameters and demonstrate temporal holdout separation.

### WP42/WP43 moving temporal trifrequency promotion

Absolute DDPR is replaced by a block-local temporal score: for every candidate
and exact DD satellite pair, one robust constant is removed, and median
absolute residuals are ranked independently in primary, secondary, and
tertiary families. The frozen gate retains the WP38 top-20% family bound and
20% rank-sum runner margin, and additionally requires the source carrier arc,
row, RMS, and bootstrap-spread gates.

Nagoya 2327--2382 candidate 0 ranks 7/1/6, with rank sum 14 versus runner 17
and 21.43% margin. It changes the route by only 0.114 m, inside the 0.5 m
boundary-continuity gate. The adjacent weak-supply block fails closed. An
unsafe Tokyo holdout with no candidate better than 6.71 m also fails closed:
its two-family winner ranks 1/12/1 and misses the top-20% secondary gate.

The promoted candidate's four bootstrap offsets stay within 0.067 m of the
common solution. Linear interpolation between bootstrap centers improves the
full Nagoya trajectory from 4,790 to 4,814 sub-50 cm epochs, with 24 gains and
zero losses. Full denominator remains 7,583 and FIX / false FIX remain zero.
Validation is locked in
`internal_docs/wp42_moving_temporal_trifrequency_validation_2026_07_22.json`;
the full benchmark is locked in
`internal_docs/wp43_pf_moving_temporal_benchmark_2026_07_22.json`.

### WP44/WP45 direct anchor-boundary identity promotion

The 5-epoch evidence sampler now scans every modulo phase before selecting the
phase with maximum evidence supply. This corrects phase-alias false misses such
as Nagoya 5015--5070, where start phase 0 had no evidence but auto phase 2 has
11 epochs. The unchanged WP42 measurement gate still rejects that block.

WP44 adds a separate, narrow identity-profile gate for exactly one 55-epoch
block directly after an accepted static anchor. Nagoya 923--978 follows the
805--923 anchor and passes with 11 evidence epochs, maximum profile norm
0.162 m, spread 0.012 m, and the nearest runner 1.887 m away. Its production
profile gains 42 sub-50 cm epochs with no loss. WP45 therefore reaches
4,856/7,583 = 64.0380%, with FIX / false FIX still zero.

The gate cannot recurse. A hypothetical 978--1033 continuation passes the
local numeric thresholds but degrades 49/55 to 36/55 in the post-decision
audit, so a moving WP44 block is never accepted as the next anchor. Unsafe
Tokyo 2464--2519 independently fails the profile-norm and spread gates.
Validation and the full benchmark are locked in
`internal_docs/wp44_anchor_boundary_identity_validation_2026_07_22.json` and
`internal_docs/wp45_pf_anchor_boundary_benchmark_2026_07_22.json`.

### WP46/WP47 largest-gap supply and local-pool audit

The auto-phase supply scan covers Nagoya 1051--1806 in one dataset load.
Eleven of thirteen complete 55-epoch blocks pass fixed evidence/carrier/DDPR
minimums; 1051--1106 and 1216--1271 fail before candidate generation, and the
40-epoch tail fails closed. The strongest supplied window is 1601--1656 with
256 carrier rows.

Nagoya 1436--1491 was evaluated first. Its normal LAMBDA pool fails WP42 at
ranks 4/2/3 against the unchanged top-3 bound. Expanding five OSM road parents
to 2,025 local candidates does not solve identification: global runner margin
is 1.23%, and no parent exceeds 7.32% against the required 20%. No candidate
is promoted. Locks are
`internal_docs/wp46_nagoya_moving_supply_scan_2026_07_22.json` and
`internal_docs/wp47_nagoya_local_pool_rejection_2026_07_22.json`.

### WP48 long-window and reverse-anchor rejection

A 220-epoch 1436--1656 joint ambiguity window increases supply to 44 evidence
epochs and 1,011 carrier rows, but retains only three modes. The winner ranks
1/3/1 against the fixed top-1 family limit and is rejected. Independently,
reverse propagation from accepted anchor 1807 reaches epoch 1436 with 20.97 m
error and only one sub-50 cm epoch. It is rejected as direct output and as a
candidate seed. Evidence is locked in
`internal_docs/wp48_nagoya_long_window_reverse_rejection_2026_07_22.json`.

### WP49 LAMBDA and Up-prior sensitivity

Increasing the 220-epoch LAMBDA pool from 12 to 128 produces 25 fitted modes
but no sub-50 cm candidate; the best remains 7.45 m and the LAMBDA ratio is
1.0012. Relaxing the zero-centered Up sigma to 20 m worsens the best mode to
11.33 m. Tightening it to 0.5 m improves the best to 4.62 m, still with zero
sub-50 cm epochs. Candidate-count and zero-centered sigma tuning are therefore
rejected. The next supplier must center height from an independent truth-free
observable. The lock is
`internal_docs/wp49_nagoya_lambda_up_prior_sensitivity_2026_07_22.json`.

### WP50 independent GSI height rejection

WP50 adds an independently centered GSI DEM/geoid height prior over 1436--1656.
It improves the best 128-candidate basin to 3.331 m but supplies zero sub-50 cm
epochs, so it is rejected in
`wp50_nagoya_gsi_height_prior_rejection_2026_07_22.json`.

### WP51 latest GNSS++ DDPR FDE transfer

WP51 ports the latest GNSS++ one-row-exclusion DDPR FDE. On the same 44 evidence
epochs it improves DDPR anchor P50 from 4.650 m to 1.653 m and makes all 44
anchors land within 5 m. The resulting truth-free block proposal refines to
1.496 m median but still supplies zero sub-50 cm epochs, so production remains
unchanged. The next supplier experiment is covariance-aware partial AR; no
promotion threshold is relaxed.

### WP52 latest GNSS++ BSR partial AR rejection

The GNSS++ covariance-axis partial-AR heuristic drops the six worst loaded
ambiguity arcs progressively from the 29-arc 1436--1656 problem. Its subset
ratios stay between 1.021 and 1.107, and none of the twelve subset candidates
creates a position seed more than 5 cm from the existing full-AR pool. The best
posterior therefore remains the WP51 FDE seed at 1.496 m with zero sub-50 cm
epochs. The locked next direction is DD reference/arc diversity rather than a
larger same-basis candidate pool or a weaker gate.

### WP53 alternate-reference supply succeeds; posterior rejects

Using the second-highest carrier reference changes the block arc structure and
supplies the first near-correct 1436--1656 basin: 0.520 m median and 91/220
sub-50 cm epochs. Re-fitting the rank-1 pool under ranks 0 and 2 yields a shadow
linear profile that would add 84 full-run sub-50 cm epochs with zero loss.

It is not promoted. The truth-free geometric-consistency gate selects a 5.40 m
wrong mode, while the three-basis carrier-RMS sum selects the useful mode with
only 1.71% runner separation against the frozen 20% requirement. WP53 therefore
localizes the remaining problem to posterior identification rather than
candidate supply. Production stays at 4,856/7,583 and FIX/false FIX remain zero.

### WP54 CP/PR posterior identifies the WP53 supply in shadow

Rebasing primary DD pseudorange to the alternate carrier reference allows a
direct `DDPR - (DDCP - Nλ)` integer-consistency audit. The useful WP53 candidate
ranks 1/1/2 by median, p95, and bad-pair count, with rank sum 4 against runner
8. The unchanged top-20% family bound and 20% runner margin both pass. Its
shadow bootstrap profile adds 93 full-run sub-50 cm epochs with zero loss,
reaching 4,949/7,583 = 65.2644%.

WP54 remains development-only until the identical selector fails closed on the
predeclared unsafe holdouts. No production trajectory or FIX declaration is
changed yet.

### WP55 CP/PR posterior holdouts and production promotion

WP55 closes the missing safety gate without changing the target selector after
truth inspection. The CP/PR winner must satisfy the existing top-20% family
rank and 20% runner-margin gates plus at least 40 checked pairs, no more than 5%
innovations above 5 m, and four-block spread at or below 0.5 m. A failed winner
fails closed; there is no fallback candidate.

On unsafe Tokyo 2464--2519, the rank winner is rejected by its 0.571 m block
spread. On independent Nagoya 5015--5070, candidate 0 passes with 167 pairs,
zero bad pairs, 0.022 m spread, and is also the post-selection audit best at
55/55 sub-50 cm epochs. The target 1436--1656 candidate therefore promotes
through a source/validation hash-linked profile. It adds 93 full-denominator
epochs with zero loss, moving Nagoya to 4,949/7,583 = 65.2644%. FIX and false
FIX remain zero; runtime FGO remains disabled. Validation and production are
locked in `internal_docs/wp55_cppr_rank_validation_2026_07_22.json` and
`internal_docs/wp55_pf_cppr_rank_benchmark_2026_07_22.json`.

### WP56 dense-rank fix and adjacent-block rejection

Applying the frozen WP55 chain to Nagoya 1381--1436 exposed an implementation
bug in the posterior: exact `bad_pairs` ties were broken by seed ID. Dense
ranking fixes that without changing any threshold. The WP55 target and both
holdouts retain their prior pass/fail decisions and the +93 production gain.

On 1381--1436, candidates 6 and 12 become tied at rank sum 4, so the runner
margin is 0% and the block fails closed. This is the safe result: post-selection
audit shows 0/55 sub-50 cm epochs for candidate 6 but 35/55 for candidate 12.
Supply exists, but CP/PR alone cannot identify it. No WP56 promotion occurs;
the rejection is locked in
`internal_docs/wp56_cppr_dense_rank_rejection_2026_07_22.json`.

### WP57 long-anchor precursor boundary promotion

WP57 uses the accepted 220-epoch WP55 profile as a one-hop right-boundary
observable for the immediately preceding 1381--1436 block. Among candidates
that pass every frozen CP/PR family and absolute gate, candidate 12 is 0.139 m
from the right boundary versus 0.537 m for its runner, a 286% margin. The
truth-free selector therefore resolves the WP56 tie without weakening CP/PR.

The historical recursive 978--1033 holdout remains fail-closed: CP/PR gives a
0% runner margin, and its 55-epoch boundary-derived predecessor fails the new
anchor lineage and duration requirements. WP57 outputs may never seed another
WP57 boundary promotion. The accepted profile gains 36 full-denominator epochs
with no loss, moving Nagoya to 4,985/7,583 = 65.7392%. Validation and production
are locked in `internal_docs/wp57_precursor_boundary_validation_2026_07_22.json`
and `internal_docs/wp57_pf_precursor_boundary_benchmark_2026_07_22.json`.

### WP58--WP60 long-window rejection and two-block global path

WP58 and WP59 reject single ambiguity states over 1271--1436 and 1326--1436:
their best candidates remain 3.76 m and 1.08 m respectively with zero sub-50
cm epochs. WP60 preserves the 55-epoch ambiguity boundary instead. On
1326--1381, independent rank-0 and rank-1 CP/PR selectors choose profiles only
0.022 m apart, while rank 2 is over 1.03 m away. The two-basis median joins the
validated 1381--1436 candidate at 0.124 m versus 0.664 m for its runner.

This is one global path decision against the original WP55 anchor, not a
recursive extension of WP57, and its output cannot seed another path. The full
audit gains 77 epochs with no loss relative to WP55 and supersedes WP57 by 41,
reaching 5,026/7,583 = 66.2798%. Validation and production are locked in
`internal_docs/wp60_two_block_path_validation_2026_07_22.json` and
`internal_docs/wp60_pf_two_block_path_benchmark_2026_07_22.json`.

### WP61/WP62 supply rejection and backward-outage recovery

For 1271--1326, rank 0/1/2, DDPR-FDE seeding, and OSM road seeding all fail to
supply a sub-50 cm candidate. Two independent bases nevertheless show the same
observable failure shape: the leading bootstrap block diverges by 7.0--7.9 m,
while the final three remain stable within 0.03--0.05 m. CP/PR pair, bad-rate,
family-rank, and margin gates pass; only whole-block spread fails.

WP62 recomputes WP60 from every original hash-linked input and extends the
first path offset backward over exactly one 55-epoch predecessor. This is a
single global outage decision, not recursive anchoring, and cannot seed another
extension. The full audit gains 112 epochs with no loss versus WP55, moving
Nagoya to 5,061/7,583 = 66.7414%. Validation and production are locked in
`internal_docs/wp62_backward_outage_validation_2026_07_22.json` and
`internal_docs/wp62_pf_backward_outage_benchmark_2026_07_22.json`.

### WP63/WP64 standalone supply scan and rejection

WP63 first rejects 3336--3391: the default three-epoch arc gate yields no float
arcs for carrier-reference ranks 0--2, while relaxing it to two epochs retains
only 8--12 carrier rows and gives 15.5--43.4 m median error with zero sub-50 cm
epochs. A truth-free evidence scan over 3336--3794 then chooses 3666--3721 as
the strongest complete block, with 176 carrier and 144 DDPR rows.

On that block, rank 0/1/2 independently freeze candidate 0 after passing all
CP/PR family and absolute gates. Rank 0 and rank 2 agree within 0.084 m, but
post-selection audit gives zero sub-50 cm epochs for every frozen profile and
1.028--1.111 m median error. The consistent near-zero correction cannot repair
the displaced production trajectory, so WP64 is rejected and WP62 production
remains unchanged. The result is hash-locked in
`internal_docs/wp64_multibasis_standalone_rejection_2026_07_22.json`.

WP65 applies the same frozen experiment to the next-ranked supply block,
3721--3776. Rank 0 and rank 1 pass CP/PR individually but differ by 0.250 m;
rank 2 is tied at a 0% runner margin and fails closed. No pair of accepted
bases meets the 0.2 m agreement gate, and all three frozen winners audit at
0/55 sub-50 cm epochs. The non-promotion is locked in
`internal_docs/wp65_multibasis_standalone_rejection_2026_07_22.json`.

WP66 evaluates supply-ranked block 3611--3666. Only rank 0 passes, exactly at
the 40-pair minimum and 5% bad-pair ceiling; ranks 1 and 2 fail absolute or
posterior gates. The frozen rank-0 winner audits at 1.830 m median and 0/55
sub-50 cm epochs, so production again remains WP62. The rejection is locked in
`internal_docs/wp66_multibasis_standalone_rejection_2026_07_22.json`.

WP67 reaches supply-ranked block 3556--3611. Although ranks 0/1 retain 103 raw
carrier rows and rank 2 retains 82, slip segmentation creates 55--77 arcs and
none reaches the minimum support needed to form an ambiguity hypothesis. This
is a supply-level fail-closed result, locked in
`internal_docs/wp67_carrier_supply_rejection_2026_07_22.json`.

### WP68/WP69 common-mode DDPR rejection

WP68 calibrates 19 exact DDPR satellite-pair biases over 44 updates at the
accepted 1807--2327 static anchor. Carrying them 1,339 epochs forward makes the
3666--3721 winner's bad-pair fraction 19.5% and P95 innovation 86.0 m, proving
the pair biases are not stationary enough for this role.

WP69 adds an explicit `--trajectory-csv` input to the default-off DDPR
diagnostic so latest-GNSS++ one-row-exclusion FDE can start from the hash-locked
WP62 trajectory instead of a stale default `.pos`. All 11 DD evidence epochs
are accepted, but post-selection audit worsens from 1.05 m mean seed error to
2.94 m mean anchor error. The median FDE proposal refines to 3.565 m with zero
sub-50 cm epochs, while CP/PR gives it a 0% runner margin against the near-zero
candidate. Both common-mode attempts are rejected and locked in
`internal_docs/wp68_wp69_common_mode_rejection_2026_07_22.json`.

### WP70 road-translation observability gate

WP70 evaluates the calibrated OSM road-distance band over a fixed +/-5 m,
0.2 m common-translation grid before allowing any road proposal to enter the
PF/carrier pool. Acceptance requires at most 25 equivalent best cells, at most
1 m equivalent-posterior extent, and a 20% distinct-runner margin.

Nagoya 3666--3721, Nagoya transfer 3721--3776, and Tokyo unsafe holdout
2464--2519 all fail every gate. They contain 673, 2,062, and 1,559 exactly tied
best cells respectively, each spanning the full 10 m grid with a 0% runner
margin. OSM road bands therefore cannot identify the missing common
translation and are prohibited from supplying one. The implementation and
three truth-free evaluations are hash-locked in
`internal_docs/wp70_road_translation_observability_2026_07_22.json`.

### WP71--WP76 right-boundary affine multi-basis promotion

WP71 shows that replacing the fixed 0.5 m Up sigma with measured GSI antenna
height spread does not change the constant-offset basin. WP72 then changes the
state model: the accepted static anchor beginning at epoch 3811 fixes the
correction gauge to zero, while a three-dimensional affine correction is fit
backward over 3666--3721. This lowers the best supplied mode from 1.03 m to
0.55 m; affine oracle diagnostics reach 55/55, proving model sufficiency.

WP73 partial AR and WP74 final GSI-Up enforcement do not cross 0.5 m by
themselves. WP75 adds a fixed truth-free 49-cell East/North grid around each
GSI-normalized affine float solution. It supplies 7/8/8 sub-50 cm candidates
for reference ranks 0/1/2. WP76 clusters all three bases, applies the existing
absolute CP/PR gates, and ranks the resulting clusters by road-band violation
and summed carrier RMS. Candidate 27 forms a 0.072 m cluster, ranks 1/2, and
beats its runner by 33.3%.

The adjacent Nagoya transfer produces only one cluster and Tokyo unsafe
2464--2519 produces none; both fail closed. A hash-recomputing promoter then
applies the affine profile to WP62. The full audit gains 55 epochs with zero
loss, moving production to 5,116/7,583 = 67.4667%, with FIX and false FIX still
zero. Validation and production are locked in
`internal_docs/wp76_affine_multibasis_validation_2026_07_22.json` and
`internal_docs/wp76_pf_affine_multibasis_benchmark_2026_07_22.json`.

### WP79--WP80 predecessor affine rejection

The identical WP76 affine rule is evaluated on the preceding Nagoya block
3611--3666 using the WP76 production trajectory. The frozen execution settings
are explicit: automatic stride-phase selection chooses phase 2, the GSI Up
prior sigma is 0.5 m, and each carrier-reference basis receives the same
49-cell East/North grid. Rank 0 supplies 21 modes through the absolute CP/PR
and block-spread gates, but ranks 1 and 2 supply none. Consequently no
three-basis cluster exists and the selector fails closed before any road or
carrier family ranking can promote a profile.

Post-selection truth diagnostics also show that the grid supplies zero
sub-50 cm hypotheses in all three bases; the non-selectable affine oracle
reaches only 16/15/15 epochs. Production therefore remains unchanged at
5,116/7,583 = 67.4667%, with FIX and false FIX both zero. The rejection and
artifact hashes are locked in
`internal_docs/wp80_affine_multibasis_predecessor_rejection_2026_07_22.json`.

### WP81--WP82 fixed-boundary affine rejection

WP81 adds a default-off piecewise affine state that connects the predecessor
block directly to WP76's promoted correction at epoch 3666, instead of fitting
one line all the way to the zero correction at epoch 3811. This is a truth-free
fixed boundary: the source is the hash-locked WP76 production promotion. The
model improves the best rank-0 supplied audit from 0/55 to 28/55; after the
same 49-cell GSI-normalized grid, the best diagnostic audits are 36/29/36 for
carrier-reference ranks 0/1/2.

The absolute evidence result still fails closed. After CP/PR, block-spread,
and within-basis deduplication, the three bases supply 6/0/0 modes. No
three-basis cluster exists, so road/carrier family ranking and production
application are not reached. Production remains 5,116/7,583 = 67.4667%, with
FIX and false FIX both zero. The implementation, rejection, and artifact
hashes are locked in
`internal_docs/wp82_fixed_boundary_affine_rejection_2026_07_22.json`.

### WP83--WP87 single-basis posterior promotion

WP83 confirms that reducing the checked-pair floor from 40 to 24 does not
recover three-basis consensus: the fixed-boundary target supplies 11/0/1 modes
and zero clusters. WP84 tests a shorter 3636--3666 knot with a newly acquired,
segment-matched GSI cache, but its per-basis checked-pair maxima are only
20/20/8. WP85--WP86 split the carrier evidence into combined, L1/E1/B1, and
L5/E5a/B2a bases; the single-frequency bases do not pass the absolute spread
and pair gates. All three development branches fail closed without promotion.

WP87 retains the original 40-pair absolute gate and uses the strong combined
rank-0 basis. Six modes survive CP/PR, bad-pair, block-spread, and dedup gates.
A new posterior ranks them independently by calibrated OSM road-band
violation, carrier RMS, and CP/PR evidence. Candidate 58 ranks 1/1/1 with rank
sum 3; the runner has rank sum 7, giving a 133.3% margin. The Nagoya transfer
holdout fails with only 13.3% margin and Tokyo unsafe supplies only one mode.

A hash-recomputing promoter verifies target and both holdouts before applying
the fixed-boundary affine profile. The full audit gains 19 epochs with zero
loss, moving production to 5,135/7,583 = 67.7173%; FIX and false FIX remain
zero. Validation and production are locked in
`internal_docs/wp87_singlebasis_fixed_affine_validation_2026_07_22.json` and
`internal_docs/wp87_pf_singlebasis_fixed_affine_benchmark_2026_07_22.json`.

### WP88--WP93 outage rejection, global supply, and constant-profile promotion

WP88 and WP89 test the immediately preceding 3556--3611 block against the
promoted WP87 boundary. Its carrier supply is structurally insufficient: the
default run has 0/0/1 ambiguity arcs, and even `min_arc_epochs=2` produces
maximum checked-pair counts of only 4/2/3, far below the frozen 40-pair gate.
WP90's boundary-only outage propagation is lossless but gains zero epochs, so
none of WP88--WP90 is promoted.

WP91 scans 137 complete Nagoya blocks without truth input; 135 pass the supply
gate. The strongest all-bad block is 660--715 with 318 carrier rows and 230
DDPR rows. WP92 expands each of three carrier-reference solutions over a
49-cell GSI-normalized horizontal grid. The strict three-basis 0.12 m cluster
gate fails closed with zero clusters, but the predeclared WP87 single-basis
rule accepts rank 1: candidate 50 ranks first independently in calibrated OSM
road-band violation, carrier RMS, and CP/PR evidence, with a 366.7% runner
margin. The Nagoya transfer and Tokyo unsafe holdouts both fail closed.

WP93 applies the recomputed constant profile over epochs 660--715. The full
7,583-epoch audit gains 42 epochs with zero loss, moving production from
5,135/7,583 (67.7173%) to 5,177/7,583 (68.2711%). FIX and false FIX remain
zero, runtime FGO and production truth input remain disabled, and M4 retains
its exact hashes. Validation is locked in
`internal_docs/wp93_constant_singlebasis_validation_2026_07_22.json`.

### WP94--WP95 global-supply continuation

WP94 evaluates the next globally ranked all-bad block, 5720--5775. Its
49-cell grid supplies 58/55/56 strict modes and reaches the 55-epoch oracle
ceiling in two bases. Nevertheless, the frozen three-basis gate yields zero
clusters and all three single-basis posteriors fail the family or runner-margin
gate. WP94 is locked as supply-pass/posterior-reject and is not promoted.

WP95 evaluates 6710--6765. Carrier rank 0 alone has a 55/55 model ceiling, so
only that basis is expanded. Candidate 7 survives 57 absolute modes, ranks
2/7/5 in road/carrier/CPPR evidence (all within the frozen top-20% limit), and
beats the runner by 28.57%. The recomputing promoter confirms that the Nagoya
transfer and Tokyo unsafe holdouts fail closed and M4 remains exact. The full
audit gains 42 epochs with zero loss, moving production to 5,219/7,583 =
68.8250%; FIX and false FIX remain zero.

### WP96--WP97 ambiguity rejection

WP96 evaluates the adjacent 6765--6820 block. All three carrier bases reach a
55/55 diagnostic ceiling and supply 56/58/56 strict hypotheses. The frozen
three-basis selector finds only one cluster, so it cannot establish a runner
margin. Independently, rank 0 and rank 1 accept incompatible single-basis
profiles. The block therefore fails the cross-basis uniqueness requirement and
is not promoted.

WP97 evaluates 5775--5830. Rank 1 and rank 2 supply the full 55/55 ceiling.
The three-basis selector finds three clusters, but winner and runner tie and
the family-rank gate fails. Rank 0 and rank 2 single-basis selectors accept a
common alternative basin, while rank 1 rejects. With no unique safe posterior,
WP97 also remains unpromoted. These rejections preserve WP95 production at
5,219/7,583 and prevent post-audit basin choice from entering production.

### WP98--WP99 partial-model rejection

WP98 evaluates 5665--5720. Its best diagnostic model ceiling is 31/55, but the
rank-0 49-cell grid supplies no sub-50 candidate. The frozen posterior also
fails with an 11.1% runner margin. WP98 is rejected without promotion.

WP99 evaluates 5830--5885. Rank 0 and rank 2 grids both contain a 55/55
diagnostic candidate even though their local oracle fits have lower ceilings.
After truth fields are removed, rank 0 selects a different profile with zero
sub-50 gain; rank 2 fails its margin gate. The selected production-safe result
therefore has zero measurable benefit, so WP99 is not promoted and the
diagnostic candidates are not substituted post hoc.

### WP100 constant single-basis promotion

WP100 evaluates 5170--5225. All three carrier bases have a 55/55 model ceiling,
and rank 2 supplies a 55/55 mode before grid expansion, but its initial runner
margin is 16.67% and therefore fails closed. A 49-cell GSI-normalized grid
expands the independent truth-free candidate set to 59 absolute modes.
Candidate 50 ranks 7/2/1 in road/carrier/CPPR evidence, remains within the
frozen top-20% family limits, and reaches the exact 20% runner-margin floor.

The recomputing promoter confirms both independent holdouts fail closed and M4
is unchanged. Full-denominator shadow and production trajectories are
byte-identical; the audit gains 55 epochs with zero loss. Nagoya production
moves from 5,219/7,583 (68.8250%) to 5,274/7,583 (69.5503%); FIX and false FIX
remain zero.

### WP101 structural supply rejection

WP101 evaluates 4400--4455 from the WP100 production trajectory. Although the
raw scan reports 149 carrier and 149 DDPR rows, none of the three carrier bases
produces a hypothesis satisfying the frozen checked-pair, bad-fraction, and
block-spread gates. The diagnostic model ceilings are only 3/3/3 epochs.
WP101 is rejected before horizontal grid expansion and production remains
unchanged.

WP102 evaluates 5940--5995. All three bases again fail to produce any strict
mode; checked-pair maxima are only 28/25/9 and model ceilings are 16/10/0.
WP102 is rejected before grid expansion.

WP103 evaluates 5885--5940. Rank 0 has a 32/55 diagnostic ceiling and reaches
55 checked pairs, but neither the initial pool nor a 49-cell horizontal grid
produces any hypothesis satisfying all frozen absolute gates. WP103 is rejected
without posterior selection.

WP104 evaluates 1705--1760. Rank 0 reaches 64 checked pairs and has a 28/55
diagnostic ceiling, but the initial pool and 49-cell grid both produce zero
strict absolute modes and zero supplied sub-50 epochs. WP104 is rejected before
posterior selection.

WP105 evaluates 2695--2750. All bases retain only two integer arcs, checked-pair
maxima are 3/1/2, no strict mode survives, and no oracle solution is available.
WP105 is rejected as a structural carrier-arc outage.
