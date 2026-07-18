# WP27 Progress — satellite integrity and absolute basin evidence

## 2026-07-18 — multi-pivot DDPR consensus

Implemented a truth-free, pivot-invariant absolute evidence primitive. It
reconstructs constellation-local single-difference innovation coordinates from
the available DD pseudorange result, evaluates all within-constellation
satellite pairs with a robust Cauchy cost, and normalizes scores across live
ambiguity-basin candidates. Synthetic tests cover pivot changes, one biased
satellite, multiple constellations, insufficient support, and finite
probabilities.

### Tokyo run3/200 replay

Raw RINEX DDPR is available at 1 Hz, giving 40 anchor epochs in the 5 Hz,
200-epoch WP25 trace. Thirty-three anchor epochs contain a live sub-50 cm
candidate. Multi-pivot DDPR selects one on 19/33, versus 0/33 for the original
single-MAP selection, and lowers median error from 1.581 m to 0.545 m.

Combining those sparse anchors with WP26 TDCP holdover gives the first material
absolute-basin selection improvement in this campaign. The best diagnostic
configuration selects a sub-50 cm candidate on 91/200 epochs, versus 0/200 for
single MAP, and lowers median error from 1.583 m to 0.549 m. It is better than
single MAP on 180 epochs and worse on 20. Correct holdover is comparable at DD
anchors (19/40) and between anchors (72/160). The longest correct runs are 33
and 30 epochs; the initial 75 epochs remain wrong.

Adding the original per-epoch basin likelihood back into the filter collapses
the gain: the combined sweep reaches at most 15/200 sub-50 cm epochs. That
likelihood must therefore remain excluded from selection confidence until its
accumulation/calibration defect is resolved.

### Safety interpretation

Posterior gamma never exceeds 0.99, so this increment creates no trusted FIX.
The 91/200 figure is a truth-joined diagnostic candidate-selection result, not
a production output or a full-denominator `<50cm_full%` claim. Error is also
clustered in five-epoch RINEX anchor blocks, and no truth-free accept/reject
threshold yet separates the correct and wrong blocks.

## Online causal diagnostic arm

The offline winner is now integrated into `exp_wp23b_basin_ar.py` behind
`--enable-integrity-lineage`. It computes multi-pivot DDPR directly from the
current raw observation and live basins, computes robust TDCP from only the
current and previous epoch, and records both sources under the separate
`integrity_lineage` evidence target. Neither score is connected to the basin PF,
emitted position, or trusted commit policy.

Tokyo run3/200 reproduced the offline result exactly:

- 40/200 multi-pivot anchors and 199/199 TDCP intervals;
- 91/200 diagnostic sub-50 cm selections;
- evidence ledger 1,034 records/updates, zero beta errors;
- control versus enabled position/FIX/MAP/gamma mismatches: zero;
- control and enabled trajectory SHA-256 are identical;
- commit replay mismatches: zero, declared FIX: zero.

The online arm therefore passes causal integration and neutrality, but remains
diagnostic-only. Its maximum gamma is only 0.0627 and is not a FIX confidence.

## Frozen Tokyo three-run transfer

The unchanged run3 policy was run on the first 200 epochs of all Tokyo runs:

| Run | Oracle live | Selected sub-50 cm | Conditional selection | TDCP |
| --- | ---: | ---: | ---: | ---: |
| run1 | 200/200 | 33/200 | 16.5% | 197/199 |
| run2 | 200/200 | 130/200 | 65.0% | 199/199 |
| run3 | 165/200 | 91/200 | 55.2% | 199/199 |

Candidate supply is complete on run1/run2, so run1's failure is selection, not
respawn recall. The 0.5 m non-chaining ball does not manufacture confidence:
maximum mass is 0.237/0.388/0.166 and no epoch exceeds 0.99.

The existing trusted float/DDPR guard is not sufficient for integrity output.
It admits 16/38 false candidates on run1, 10/137 on run2, and 5/5 on run3.
A predeclared 315-cell gamma/dwell/guard audit finds 230 configurations with
zero *observed* false rate, but every one has zero accepted correct epochs on at
least one run. No configuration has a Wilson 95% false-rate upper bound below
1% on every run. No policy is promoted.

Calibration also varies materially: Brier/ECE are 0.167/0.148 (run1),
0.427/0.491 (run2), and 0.412/0.432 (run3). A single scalar temperature or
threshold cannot repair the measured run-dependent ranking failure.

## Next

Attribute anchor failures per satellite with leave-one-satellite-out
multi-pivot replay. Measure whether a stable satellite subset or persistent
innovation mode explains run1/run3 wrong blocks. Only then introduce causal
`clean/biased/blocked/recovering` latent states; do not tune a production
threshold against these windows.
