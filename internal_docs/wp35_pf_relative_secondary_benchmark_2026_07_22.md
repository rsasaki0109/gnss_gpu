# WP35 PF relative-secondary benchmark

Tokyo run1 6945--7076 is production-promoted through a two-stage truth-free
posterior: relative secondary-family DD pseudorange selects one of three
identically resampled parents, then a compact primary-DD top-three posterior
defines the position. The selected mean has a separate audit error of 0.432 m.

The fixed relative gate has two already accepted Tokyo positive holdouts:
9883--10248 at 33.8% runner margin and 5866--6021 at 57.5%. Tokyo 6865--6912
is the independent negative: its unsafe winner has 1.36 m audit error and only
2.17% runner margin, so the fixed 7.5% gate rejects it. Two earlier proposal
negatives also remain fail-closed. All inputs and M4 hashes are verified by
`wp34_relative_secondary_validation_2026_07_22.json`.

Adding only the promoted 6945 anchor to the reproduced common p=2 production
smoother raises Tokyo from 3,051/11,924 (25.5871%) to 3,184/11,924
(26.7024%): +133 epochs and +1.1154 percentage points. Declared FIX and false
FIX remain zero, runtime FGO remains disabled, and no development anchor is
used. Nagoya remains 4,118/7,583 (54.3057%).

The machine-readable hashes and metrics are locked in
`internal_docs/wp35_pf_relative_secondary_benchmark_2026_07_22.json`.
