# WP60 PF-only two-block path benchmark (2026-07-22)

WP58 and WP59 first establish that treating 1271--1436 or 1326--1436 as one
ambiguity state is unsafe: their best candidates have 0 sub-50 cm epochs. WP60
instead keeps 1326--1381 and 1381--1436 as independent ambiguity blocks and
selects the complete path once, directly against the original WP55 anchor.

On the left block, rank-0 and rank-1 CP/PR selectors independently choose
profiles only 2.2 cm apart; rank 2 is about 1.04 m away. Their coordinatewise
median profile joins the already validated right candidate at 0.124 m, while
the right runner is 0.664 m away. The interface margin is 435.5%. No truth is
used in supply, selection, consensus, or promotion, and a WP60 output may not
seed another path promotion.

The full-denominator audit gains 77 epochs and loses none relative to WP55,
superseding WP57 by 41 epochs. Nagoya run1 reaches 5,026/7,583 = 66.2798%.
Runtime FGO remains disabled and FIX/false FIX remain 0/0. Another 1,496 epochs
are required for the 86% target; Tokyo remains 3,265/11,924 = 27.3818%.

Exact negative controls, gates, artifacts, and M4 hashes are recorded in
`wp60_two_block_path_validation_2026_07_22.json` and
`wp60_pf_two_block_path_benchmark_2026_07_22.json`.
