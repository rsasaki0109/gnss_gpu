# WP49 Nagoya LAMBDA/up-prior sensitivity (2026-07-22)

On the 220-epoch Nagoya 1436--1656 window, increasing LAMBDA enumeration from
12 to 128 expands the fitted pool from 3 to 25 modes but does not improve its
best audit error (7.45 m, zero sub-50 cm epochs). Its LAMBDA ratio is 1.0012,
confirming severe integer ambiguity rather than a unique omitted runner.

Relaxing the zero-centered Up prior from 2 m to 20 m worsens the best mode to
11.33 m. Tightening it to 0.5 m improves the best mode to 4.62 m but still
supplies no sub-50 cm epoch. The tightened float seed remains displaced from
the truth-only oracle by approximately (-1.2, +4.5, +2.2) m in ENU.

No sensitivity variant is promoted. Further zero-centered sigma tuning or
larger integer enumeration is rejected. The next experiment should derive the
height-prior center from an independent truth-free source such as calibrated
GSI terrain height, then retain the existing trifrequency posterior gates.
