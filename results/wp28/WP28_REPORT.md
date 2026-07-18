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
