# WP28 Report — first recovery-supply increment

## Verdict

**WP28 recovery gate passes.** The final bounded satellite-arc live arm reaches
186/200 (93.0%) live sub-50 cm coverage and 37/40 (92.5%) correct proposal
anchors on Nagoya run3/200. Conditional live-span p90 is 85.5 epochs (17.1 s),
stale-generation holdover is zero, and the 128-candidate arc allocation
replaces the previous 128 DD-key replay allocation without increasing the
frozen 440-proposal envelope. Trusted FIX remains disconnected with zero
declared and false FIX; absolute selection is the next work package.

## Evidence

Single-center breadth alone reaches 28/200 epochs. Adding seven covariance-axis
position seeds reaches 93/200, and using up to 32 separated recent hypotheses
reaches 129/200. The best arm generates a correct proposal at 23/40 raw DDPR
anchors and sustains a live correct basin for at most 27 epochs. Its conditional
live-span p90 is 23.3 epochs, or 4.66 seconds at the 5 Hz replay rate.

Proposal rank diagnostics show why the old cap loses solutions: correct
hypotheses frequently appear hundreds of positions into the merged proposal
list. The result supports bounded temporal hypothesis memory, but not an
unbounded cap increase. The best diagnostic arm reaches 528 proposals and 512
live basins, so compute reduction is required before real-time promotion.

The frozen WP27 max-cost integrity selector returns 0/200 correct selections
even though a correct basin exists on 129 epochs. Thus supply and selection are
now cleanly separated: recovery improved substantially, while broad-bank
absolute ranking remains defective. No trusted FIX is declared and false FIX
is zero.

All features default off. A default-options replay is byte-identical to the
committed WP27 trajectory, with SHA-256
`C7B175C8EEF8690AFDE8B125D66B45DA161FCE52FD48B45DF5C67607075BF001`.
No FGO is used and truth remains diagnostic-only.

Primary artifacts:

- `csv/wp28_supply_ablation_run3_200_summary.json`;
- `csv/wp28_history_hist32_s1_k16_run3_200_epochs.csv`;
- `csv/wp28_history32_integrity_run3_200_epochs.csv`;
- `csv/wp28_default_neutrality_run3_200_summary.json`;
- `pos/wp28_default_neutrality_run3_200.csv`;
- `csv/wp28_arc_live_final_run3_200_epochs.csv`;
- `csv/wp28_arc_live_final_run3_200_summary.json`;
- `pos/wp28_arc_live_final_run3_200.csv`.

## Next gate

Continue WP28 with combined spatial/history proposals, source-aware retention,
and an explicit compute budget. Candidate recall must reach at least 90%,
conditional survival p90 at least 5 seconds, and incorrect holdover must remain
zero before recovery can advance. Candidate ranking and safety calibration are
separate later gates; this increment does not authorize production output.

## Round-2 addendum

Longer causal memory is modestly better than more simultaneous spatial seeds.
An age-50 history bank reaches 134/200 live epochs and 26/40 correct proposal
anchors, versus 122/200 and 24/40 for the combined axis/history arm. Lambda
integer-residual priors reach 133/200, so scalar proposal reweighting does not
solve retention.

Eight of the age-50 arm's 26 correct proposal anchors are absent from the live
bank immediately after pruning, while 14/40 anchors never generate a correct
proposal. WP28 therefore needs both source-aware cap allocation and a new
generation source. The fixed recall and survival gates remain unmet, and the
trusted output path remains unchanged.

A 1 m spatial deduplication radius was also tested and is trajectory-identical
to unrestricted assignment deduplication, with the same 134/200 recall and
8/26 supplied-then-absent anchors. This rejects cross-position moment merging
as the active bottleneck for this arm.

A 90% round-robin reserve across current position-proposal sources is also
trajectory-identical and leaves every recall/pruning metric unchanged. The
next implementation target is generation-aware replay of historical ambiguity
assignments; neither spatial deduplication nor source quotas is promoted.

## Assignment-replay addendum

Generation-versioned ambiguity replay is now implemented and causally audited.
It rejects slip-invalid generations, intersects assignments with current DD
carrier support, requires eight integers, and reconditions position from the
current DDPR guard.

The best 416-proposal allocation (eight historical positions × top-32 plus 128
assignment replays) reaches 141/200 live sub-50 cm epochs. It improves the
position-only age-50 arm by seven epochs while reducing immediate loss of a
correct proposal from 8/26 to 1/26. Longest survival rises from 27 to 80 epochs
and p90 from 3.52 to 10.5 seconds, passing the survival gate.

The 90% recall gate remains unmet at 70.5%, and no candidate is connected to
output or FIX. A wider 256-assignment bank and a deeper top-64/four-position
allocation both regress. The next audit must explain the 14/40 DD anchors where
no correct proposal is generated; retention is no longer the dominant failure
for the best arm.

The missing-anchor audit localizes failure to startup and a reset burst at
epochs 25–45, followed by missing proposals through epoch 90. A generation-safe
partial-assignment completion arm was tested and rejected: direct insertion
reduces live recall to 32/200 and proposal recall to 9/40. Completion proposals
must remain shadow-only until their independent oracle recall is established;
they are not eligible for the recovery bank or trusted output.

Shadow-only replay confirms the rejection without feedback confounding. It is
correct on 11/39 eligible anchors, but recovers zero of the best arm's 14
missing anchors; all 11 overlap already-supplied epochs. The shadow and control
trajectories are bit-identical. No completion proposal receives PF mass.

A 5 m covariance-axis shadow adds only one missing anchor and regresses to
111/200 when inserted directly. Respawn-time max-cost satellite exclusion is
fully neutral. These results reject broad position shells and single-satellite
subset exclusion as solutions to the remaining reset-recovery gap.

Multi-subset shadows likewise add only one missing anchor and are rejected.
The productive change is causal TDCP propagation of retained position
hypotheses. With a 12-position/top-24 allocation and 128 assignment replays,
cap-512 live recall reaches 149/200 (74.5%), proposal recall 31/40, longest
survival 87 epochs, and p90 8.72 seconds. Doppler propagation regresses.

Cap 768 raises the diagnostic ceiling to 154/200 (77.0%) and p90 to 14 seconds,
but remains below the 90% recall gate and is not a fixed-budget production
candidate. Higher respawn mass, age-100 memory, and neighboring 10×28/16×20
allocations regress. The trusted FIX path remains disconnected with zero FIX.

Using the existing truth-free LAMBDA residual as the proposal prior reaches the
best cap-512 recall, 150/200 (75.0%), at the cost of lower but still passing
7.16-second p90 survival. This does not change the WP28 gate verdict.

Farthest-point history selection is strongly negative both unrestricted
(15/200) and within a 10 m DDPR-guard radius (24/200). Weight-first history
selection remains the only advancing policy.

An opt-in pivot-invariant replay was also audited. It rebuilds satellite
integer potentials from historical DD assignments, rebases them to the current
pivot, and clears all history at a reported ambiguity reset. Replay supply
recovers to 128 candidates after the epoch-45 reset, but full-run live recall
is unchanged at 150/200 and proposal recall regresses from 31/40 to 29/40.
Whole-bank reset safety is too destructive: the float diagnostic does not
provide the per-satellite slip identity needed for selective invalidation.
This arm remains disconnected from production and FIX.

The next WP28 increment is therefore a per-satellite ambiguity-arc ledger:
store pivot-free integer potentials with causal arc identifiers, detect and
invalidate only slipped satellite arcs, and materialize DD assignments only at
proposal time. It must first exceed the frozen 31/40 anchor recall in shadow
mode before it may receive PF mass.

That shadow gate is now passed. A TDCP-referenced satellite-arc ledger plus
generation-safe partial completion reaches a **38/40 (95.0%)** union proposal
recall on Nagoya run3/200. It adds epochs 35 and 50–75 without carrying any
integer across the six BeiDou arc resets detected at epoch 45. Startup epochs 0
and 5 are the only remaining misses.

Eight completions per history source establish the ceiling but produce up to
1024 shadow candidates. Two completions per source preserve all seven
incremental anchors with at most 256 candidates; the incremental candidates all
rank within the first 108 by the same truth-free residual. A 128-candidate
replacement allocation therefore preserves the shadow gate while matching the
old assignment replay budget. Promotion is still withheld until completion is
computed within that cap and a live PF arm proves survival and runtime gates.

The full top-2/cap-128 replay confirms 38/40 union recall with exactly 128
maximum arc candidates and a hash-identical trusted trajectory. Fixed candidate
budget is therefore satisfied. Runtime is not: the 200-epoch shadow diagnostic
takes about 194 seconds on the current host, so repeated LAMBDA completion must
be cached or batched before live PF promotion.

## Final live promotion

The earlier 194-second wall-clock observation included the complete PF replay
and was incorrectly attributed to arc completion. Direct instrumentation after
prepared-search caching measures only 3.27 seconds of cumulative arc work
across 200 epochs and a 0.179-second maximum recovery epoch.

Replacing the old DD-key replay with the capped arc proposals raises live
sub-50 cm coverage from 150/200 to **186/200 (93.0%)**. Proposal recall is
37/40 (92.5%), the longest live span is 87 epochs, and live-span p90 is 85.5
epochs (17.1 s). Stale-generation holdover, declared FIX, false FIX, and
online/replay mismatches are all zero. WP28's recall, survival, fixed-budget,
compute, and holdover gates pass. WP29 must transfer this identical truth-free
configuration to Tokyo and solve absolute output selection without weakening
the FIX policy.
