# WP33 PF secondary-posterior benchmark

This locked intermediate checkpoint adds the first production anchor selected
from a resampled particle cloud using an independent secondary-frequency code
posterior. It does not complete the Nagoya 86% or Tokyo 81% targets.

Nagoya run1 now reaches **4,118/7,583 = 54.3057%** at `<50 cm`, an increase of
271 epochs and 3.5738 percentage points over WP32. The run uses the full epoch
denominator, no runtime FGO, no truth as production input, the common
duration-weighted p=2 closure, and declares no FIX. False FIX remains 0%.

The new 805--923 anchor is produced by the following frozen chain:

1. production-motion parent and fixed 1/2/3/5 m cube26 proposal;
2. every weak component with at least three members and score >=0.4 is
   resampled with the same 0.2/0.4 m cube26 child shell;
3. compact non-chaining carrier/WL posterior gate;
4. GPS L2, Galileo E5, QZSS L2 and BeiDou B2 DD pseudorange ranking;
5. exactly one eligible parent, whose secondary top three each have median
   residual <=0.5 m and combined spread <=0.5 m.

Candidates 37/60/65 pass at 0.403/0.408/0.422 m secondary median and 0.248 m
spread. Their selected position audits at 0.465 m; audit truth is not an input
to selection. The independent positive holdout at 7529 selects three existing
correct modes and audits at 0.257 m. Nagoya 3811/4792 fail the absolute gate,
while Nagoya 1508 and Tokyo 1--61 fail before resampling.

Exact paths and SHA-256 hashes are in
`wp33_pf_secondary_posterior_benchmark_2026_07_22.json`.
