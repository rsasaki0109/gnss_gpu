# WP45 PF-only anchor-boundary benchmark (2026-07-22)

WP45 applies the truth-free WP44 promotion for Nagoya 923--978 to the locked
WP43 trajectory. On the complete 7,583-epoch denominator, sub-50 cm coverage
increases from 4,814 (63.4841%) to 4,856 (64.0380%): 42 epochs gained, zero
lost. FIX, false FIX, and declared false-FIX rate remain 0, 0, and 0.0%.

The 86% Nagoya target is 6,522 epochs, leaving 1,666 epochs. The largest
remaining continuous miss is 1051--1806 (755 epochs). WP44 recursive chaining
is not permitted: its 978--1033 holdout would lose 13 epochs despite passing
the local numeric profile checks.

The production trajectory, summary, promotion, WP44 validation lock, and both
M4 configuration ledgers are SHA-256 locked in the JSON companion file.
