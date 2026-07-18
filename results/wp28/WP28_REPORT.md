# WP28 Report — first recovery-supply increment

## Verdict

**Proposal-supply direction passes; WP28 production gate remains open and
unmet.** A bounded causal history bank raises Nagoya run3/200 live sub-50 cm
candidate coverage from 13 to 129 epochs without altering output or FIX. This
is a 9.9x coverage increase, but it remains below the predeclared 90% recall
gate and the existing integrity selector chooses none of the recovered correct
candidates.

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
- `pos/wp28_default_neutrality_run3_200.csv`.

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
