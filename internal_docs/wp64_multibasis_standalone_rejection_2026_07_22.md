# WP64 standalone multi-basis rejection

Nagoya run1 epochs 3666--3721 had the strongest truth-free carrier supply in
the 3336--3794 scan. Carrier-reference ranks 0, 1, and 2 independently selected
candidate 0 with valid CP/PR rank, checked-pair, bad-pair, and block-spread
gates. Rank 0 and rank 2 agreed within 0.084 m; the three pairwise distances
were 0.200, 0.084, and 0.202 m.

The profiles were frozen before truth was inspected. The post-selection audit
then found 0/55 sub-50cm epochs for every rank, with median errors of 1.028,
1.108, and 1.111 m. The consistent near-zero correction therefore does not
recover the displaced production trajectory and is not promoted.

Production remains WP62 at 5061/7583 (66.741395%) with zero FIX and zero false
FIX epochs. Exact inputs and selector outputs are hash-linked in
`wp64_multibasis_standalone_rejection_2026_07_22.json`.
