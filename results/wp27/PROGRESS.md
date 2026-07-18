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

## Next

Productionize the integrity anchor and TDCP holdover behind an opt-in diagnostic
arm, emit a causal evidence ledger, and validate calibration on held-out run1
and run2 before any FIX gate is permitted. Add per-satellite persistence states
if held-out failures show stable satellite-specific contamination.
