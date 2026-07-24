# WP46 Nagoya moving evidence supply scan (2026-07-22)

WP46 scans the largest remaining Nagoya miss, 1051--1806, with one dataset
load. Every 55-epoch block evaluates all five stride phases. Selection uses
evidence epochs, carrier rows, DDPR epochs, then lowest phase; no ground truth
is read by the scanner.

Eleven of thirteen complete blocks pass the fixed pre-candidate supply gate.
1051--1106 fails with 7 evidence epochs and 22 carrier rows. 1216--1271 fails
with 9 evidence epochs. The final 1766--1806 span is only 40 epochs and fails
closed. The strongest carrier supplies are 1601--1656 (256 rows), 1546--1601
(254), and 1436--1491 (252).

This gate establishes where expensive ambiguity and posterior analysis is
justified; it does not select an offset or make a production promotion.
