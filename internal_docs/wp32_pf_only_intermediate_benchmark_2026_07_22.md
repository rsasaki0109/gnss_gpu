# WP32 PF-only intermediate benchmark

This checkpoint locks the first common post-M4 comparison. It is an
intermediate benchmark, not completion of the Tokyo 81% / Nagoya 86% goal.
The M4 files are referenced by hash and were not modified.

Both runs use the full epoch denominator, no truth as production input, no
runtime FGO, and the same duration-weighted gap closure with exponent 2. FIX
is fail-closed; neither run declares a FIX in this checkpoint.

| Run | Anchors | <50 cm / full denominator | Rate | False FIX |
| --- | ---: | ---: | ---: | ---: |
| Tokyo run1 | 10 | 3,051 / 11,924 | 25.5871% | 0 / 0 |
| Nagoya run1 | 7 | 3,847 / 7,583 | 50.7319% | 0 / 0 |

Nagoya's seventh anchor is now production-promoted. The carrier/wide-lane
multimode cluster supplies candidates 21/31/36. An independent absolute DD
pseudorange axis admits candidates 21 and 31 at median residuals 0.457 m and
0.494 m; their 0.158 m spatial spread passes the frozen consensus gate. The
selected position has a truth-only audit error of 0.247 m, which is not used
by the selector.

The machine-readable lock, exact source paths, metrics, and SHA-256 hashes are
in `wp32_pf_only_intermediate_benchmark_2026_07_22.json`.

The next development unit is a Nagoya middle-anchor supply. It must pass a
truth-free production selector before being admitted, retain the common p=2
closure, and preserve declared FIX false rate at or below 1%. Once admitted,
the unchanged mechanism is evaluated on Tokyo rather than city-tuned there.
