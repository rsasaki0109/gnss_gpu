# WP62 PF-only backward-outage benchmark (2026-07-22)

WP61 confirms that direct ambiguity supply over 1271--1326 is insufficient:
plain, DDPR-FDE-seeded, and OSM-seeded pools all contain zero sub-50 cm
candidates. Their failure has a repeatable truth-free shape, however. On two
independent carrier-reference bases, the first of four bootstrap offsets is
7.0--7.9 m from the tail, while the final three remain stable within 3--5 cm.
CP/PR checked-pair, bad-fraction, family-rank, and runner-margin gates all pass;
only whole-block spread fails.

WP62 classifies this as leading-instability outage recovery. It recomputes the
entire WP60 path from its original truth-free inputs, then propagates its first
offset backward for exactly one 55-epoch block. It never consumes WP60 output
as a new anchor, and the result cannot seed another outage or path promotion.

The full-denominator audit gains 112 epochs and loses none relative to WP55,
or 35 more than WP60. Nagoya run1 reaches 5,061/7,583 = 66.7414%. Runtime FGO
remains disabled and FIX/false FIX remain 0/0. Another 1,461 epochs are needed
for 86%; Tokyo remains 3,265/11,924 = 27.3818%.

Exact supply failures, shape gates, artifacts, and M4 hashes are recorded in
`wp62_backward_outage_validation_2026_07_22.json` and
`wp62_pf_backward_outage_benchmark_2026_07_22.json`.
