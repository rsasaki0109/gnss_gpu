# WP44 direct anchor-boundary identity validation (2026-07-22)

WP44 fills exactly one 55-epoch moving block immediately after a truth-free
accepted static anchor. It evaluates only the zero-offset identity seed and
requires auto-phase evidence supply, carrier/DDPR support, a small common
offset, every bootstrap profile offset below 0.2 m, profile spread below
0.2 m, and the nearest competing seed at least 1.0 m away. Ground truth is
excluded from selection and is loaded only for the frozen post-decision audit.

Nagoya 923--978 follows the accepted 805--923 anchor. Auto phase 0 supplies
11 evidence epochs. The common and largest profile norms are 0.162 m, spread is
0.012 m, and the nearest runner is 1.887 m away. The full-denominator audit
improves this segment from 13/55 to 55/55, gaining 42 epochs with no loss.

Unsafe Tokyo 2464--2519 also follows an accepted static anchor, but its maximum
profile norm is 0.257 m and spread is 0.232 m. Both fixed gates reject it; its
truth-only audit has only 18/55 sub-50 cm epochs.

The gate is deliberately non-recursive. A hypothetical continuation at Nagoya
978--1033 looks locally excellent and would pass all numeric thresholds, yet
the authoritative audit degrades 49/55 to 36/55, losing 13 epochs. Because its
predecessor is a moving WP44 block rather than a static anchor, it is ineligible
by construction. This holdout freezes the direct-anchor-only scope and forbids
using WP44 output as the next anchor.
