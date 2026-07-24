# WP38 trifrequency DDPR rank validation

The selector combines only one-based median-residual ranks from independent
primary, secondary, and tertiary DD pseudorange families. A candidate must be
inside the top 20% in every family and beat the runner's rank sum by at least
20%. All three artifacts must be truth-free, uncalibrated, cover at least ten
evidence epochs, and share an identical candidate-source hash.

The 6073--6539 target selects candidate 59 with family ranks 2/3/2, rank sum
7, and 85.7% runner margin. Its separate truth audit is 0.473 m. Positive
holdouts 3811 and 7529 select 0.480 m and 0.269 m candidates with 30.0% and
27.3% margins. Unsafe supply winners at 805 and 4792 audit at 0.638 m and
0.703 m and fail closed at 3.0% and 7.0% margins.

The hash-verified gate, inputs, holdouts, and unchanged M4 baseline are locked
in `internal_docs/wp38_trifrequency_ddpr_rank_validation_2026_07_22.json`.

