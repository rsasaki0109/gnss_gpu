# WP40 Nagoya motion/outage bridge rejection (2026-07-22)

WP39 remains the production result: 4,790/7,583 sub-50 cm epochs
(63.1676117632%), FIX=0, false FIX=0. No motion/outage experiment in this work
package passes its truth-free production gate.

The duration-exponent ablation confirms p=2 as the best common closure policy.
Strict long-gyro routes fail the fixed heading/scale gates, while a relaxed
development-only route adds just one epoch. OSM particle routes are rejected
in all three evaluated motion gaps. The two shorter gaps satisfy most geometry
checks but remain posterior-multimodal: their best distinct-runner gap is
0.1445 against the frozen 2.0 minimum.

Ground truth is read only after route generation and production selection.
The audit does not justify relaxing any gate: best short-gap coverage is
66/516 (12.8%) and 40/540 (7.4%). Development should move to independent
moving carrier/DD evidence capable of resolving parallel-road ambiguity.

The machine-readable lock is
`internal_docs/wp40_nagoya_motion_outage_rejection_2026_07_22.json`, SHA-256
`4DF8140F828EAEBC9F30D04B1B564C0E6257BEE64D19184CAA7AEEF38B63228E`.
