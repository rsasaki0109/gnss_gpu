# WP37 PF fragmentation-gated benchmark

WP36 showed that unconditional all-filled endpoint closure helps Tokyo but
costs Nagoya 151 sub-50 cm epochs. WP37 uses one common truth-free rule for
both cities: if no single filled outage owns more than half of the filled
duration between adjacent anchors, distribute the residual across all filled
runs; otherwise retain duration-weighted p=2 closure. The 0.5 boundary is the
structural majority boundary, not a city-specific setting.

All nine Tokyo gaps are fragmented (largest dominant share 0.4444), so Tokyo
keeps the WP36 gain at 3,265/11,924 (27.3818%), +81 epochs over p=2. Nagoya
uses all-filled in two fragmented gaps and p=2 in five dominant gaps, reaching
4,120/7,583 (54.3321%), +2 epochs over p=2. The four Nagoya intervals harmed
by all-filled have dominant shares 0.599--0.913 and are all rejected by the
gate. Declared FIX and false FIX remain zero in both cities, runtime FGO is
disabled, no development anchor is used, and the full denominators remain
unchanged.

The implementation, tests, exact artifacts, hashes, and gate audit are locked
in `internal_docs/wp37_pf_fragmentation_gated_benchmark_2026_07_22.json`.
