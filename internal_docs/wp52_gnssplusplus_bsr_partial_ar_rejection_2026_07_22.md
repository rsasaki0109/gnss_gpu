# WP52 latest GNSS++ BSR partial AR rejection (2026-07-22)

WP52 mirrors the latest GNSS++ covariance-geometry partial-AR heuristic. The
three largest ambiguity-covariance eigenaxes are converted into per-arc
loadings, then the worst loaded arc is dropped progressively. The feature is
default-off and records every subset, dropped key, ratio, and unique supplied
position seed.

Nagoya run1 1436--1656 has 29 integer arcs. Dropping one through six arcs
produces ratios of only 1.021--1.107. Every two-candidate subset maps within
5 cm of an already supplied full-AR position seed, so all six steps contribute
zero distinct position modes. The best posterior remains the WP51 FDE seed at
1.496 m median and zero sub-50 cm epochs.

The failure localizes the next problem: this block needs a different DD basis
or reference-satellite/arc family, not more candidates from the same covariance
mode. Production settings, promotion thresholds, WP45, and M4 remain unchanged.

