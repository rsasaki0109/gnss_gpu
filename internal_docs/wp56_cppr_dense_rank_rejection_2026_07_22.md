# WP56 CP/PR dense-rank rejection (2026-07-22)

The frozen WP55 pipeline was applied to the untouched Nagoya 1381--1436 block
using the WP55 production trajectory. The first selector run exposed a ranking
bug: candidates with exactly equal `bad_pairs` were assigned different ranks
by seed ID. The implementation now uses dense ranks, so exact metric ties are
equal. WP55 target and both prior holdouts were rerun; their production outcome
is unchanged.

With corrected ranks, candidate 6 and its runner both have rank sum 4. The
runner margin is therefore 0%, below the frozen 20% gate, and the block fails
closed. Post-selection truth audit confirms why this is necessary: candidate 6
has 0/55 sub-50 cm epochs, while supplied candidate 12 has 35/55. No candidate
is promoted and Nagoya remains 4,949/7,583 = 65.2644%.

The next posterior experiment should add an independent observable capable of
separating the two CP/PR-consistent basins; it must not break exact ties by
candidate identity or relax the runner margin. Exact artifacts and hashes are
in `wp56_cppr_dense_rank_rejection_2026_07_22.json`.
