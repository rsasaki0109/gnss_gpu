# Experimental ordered backward-solve broadcast cache

Separate successor of the synchronized register LU prototype. Cache each
completed backward-solve row in registers after one full-warp broadcast.
For11 rows this replaces55 double shuffle calls by10. Every row retains its
original descending-i/ascending-k multiply-subtract order and final division.
No reduction tree, changed rounding, reciprocal approximation or FMA is used.

All32 warp lanes participate in each full-mask shuffle. A cached solution
is written before any subsequent row reads it and its source lane no longer
changes it. Both shared-status syncwarp barriers remain. Initial finite/range
guard, generic fallback, pivot order and row exchanges are unchanged. This
does not include the ineffective conditional identity exchange from D254.

Original numeric prefix, CPU body and original GPU kernel are literal copies.
Only dimension11 uses this register path;3..10 remain generic. Inherit the
explicit RegisterLUScorer and its original score method. No normal score
buffer, counter, transfer, product/default installation or new offload area.
The raw solve API is test-only. Preserve all observations/hypotheses/particles,
asymmetric priors, IRLS iterations, thresholds and RNG. FP64/fmad=false.

Use a new controlled experiment against the synchronized reference DLL and
original GPU/native CPU. Previous completed candidate timings are preserved;
their absolute times are not compared to this experiment.
