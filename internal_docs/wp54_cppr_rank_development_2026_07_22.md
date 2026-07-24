# WP54 CP/PR rank development (2026-07-22)

WP54 ports the GNSS++ CP/PR consistency concept into the alternate-reference
moving posterior. Primary DD pseudorange is rebased to each carrier reference,
then compared with the fixed carrier range `DDCP - Nλ`. The metric uses no
position truth and directly audits each integer assignment.

Across the nine rank-1 hypotheses, candidate 2 ranks 1/1/2 by median absolute
innovation, p95 absolute innovation, and count above 5 m. Its rank sum is 4
versus runner 8, so the existing top-20%-per-family and 20% runner-margin form
passes without threshold relaxation. Post-selection audit gives a 0.520 m
median candidate. Applying its linear bootstrap profile in shadow raises the
full Nagoya trajectory from 4,856 to 4,949 sub-50 cm epochs: +93 with zero loss.

This is not production yet. The selector must fail closed on predeclared unsafe
Tokyo and non-target Nagoya holdouts under the same configuration. Until that
validation is complete, WP45 remains the locked trajectory and M4 is unchanged.

