# WP36 PF all-filled outage benchmark

Tokyo run1 keeps the same eleven production anchors and the same truth-free
inputs as WP35, while changing only motion-gap endpoint closure from p=2
duration weighting to all-filled weighting. Runtime FGO remains disabled and
the full 11,924-epoch denominator is unchanged.

The fixed smoother raises Tokyo from 3,184/11,924 (26.7024%) to
3,265/11,924 (27.3818%): +81 epochs and +0.6793 percentage points. There are
88 newly sub-50 cm epochs and seven losses. The largest gain is +64 epochs in
the 1867--2353 anchor-start interval; the only negative interval deltas are
-1, -1, and -2 epochs. Declared FIX and false FIX remain zero, no development
anchor is used, and unbounded fallback remains zero.

The same mode was then reproduced with Nagoya's eight production anchors. It
drops Nagoya from 4,118/7,583 (54.3057%) to 3,967/7,583 (52.3144%), a loss of
151 epochs. Therefore unconditional all-filled closure is rejected as a
common production policy, even though its Tokyo result remains a useful
ablation ceiling.

The exact artifacts, M4 hashes, invariants, interval deltas, and Nagoya
rejection are locked in
`internal_docs/wp36_pf_allfilled_benchmark_2026_07_22.json`.
