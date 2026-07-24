# WP50 Nagoya moving GSI height prior rejection (2026-07-22)

WP50 calibrated the antenna height above mapped ground from two already
accepted anchors. The resulting cached GSI DEM/geoid prior centers Up at
-1.7605 m with a 0.5 m sigma; reference truth is not a runtime input and no
network request is required during replay.

This independent height cue improves the best 1436--1656 candidate from the
WP49 4.62 m result to 3.43 m with 12 LAMBDA candidates and 3.33 m with 128
candidates. It still supplies zero sub-50 cm epochs and the LAMBDA ratio is
only 1.0079. The candidate therefore fails closed and is not promoted.

Exact hashes and the unchanged M4 baseline are recorded in the adjacent JSON.

