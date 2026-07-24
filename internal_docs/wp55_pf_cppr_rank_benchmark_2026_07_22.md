# WP55 PF-only CP/PR-rank benchmark (2026-07-22)

WP55 promotes the alternate-reference candidate supplied in WP53 after a
truth-free CP/PR posterior and two predeclared unsafe holdouts. The selector
ranks candidates independently by median innovation, p95 innovation, and the
count above 5 m. It then requires the winner to remain in the top 20% of every
family, beat the runner by 20%, provide at least 40 checked pairs, keep at most
5% bad pairs, and keep four-block spread at or below 0.5 m. A failed winner
causes fail-closed; the selector never falls back to another candidate. Exact
metric ties share a dense rank and are never broken by candidate ID.

Tokyo 2464--2519 rejects because its otherwise dominant candidate has 0.571 m
block spread. Nagoya 5015--5070 passes independently, and its truth-blind
winner is also the post-selection audit best at 0.421 m median and 55/55
sub-50 cm epochs. The target Nagoya 1436--1656 winner ranks 1/1/2 with a 100%
runner margin and passes every absolute gate.

The development artifact's post-selection audit fields are removed before
selection; the production source has `truth_usage: none` and contains no
`audit_*` or truth-oracle payload. The hash-linked promotion profile then adds
93 epochs and loses none over the full
7,583-epoch denominator. Nagoya run1 therefore moves from 4,856 (64.0380%) to
4,949 (65.2644%) sub-50 cm epochs. Runtime FGO remains disabled; declared FIX
and false FIX remain 0/0. Tokyo remains 3,265/11,924 (27.3818%). The campaign
targets remain unmet: Nagoya still needs 1,573 epochs to reach 86%, and Tokyo
still needs 6,394 epochs to reach 81%.

Exact gates, holdouts, artifacts, code hashes, and preserved M4 hashes are in
`wp55_cppr_rank_validation_2026_07_22.json` and
`wp55_pf_cppr_rank_benchmark_2026_07_22.json`.
