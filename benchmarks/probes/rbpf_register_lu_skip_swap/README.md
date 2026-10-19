# Experimental register LU with conditional row exchange

Separate successor of rbpf_register_lu_synced. Skip row and RHS shuffle
exchanges only when the pivot remains in its current row. The pivot index is
identical for all32 lanes, so every participant takes the same branch. Both
shared-status syncwarp barriers remain. All floating operations, comparisons,
per-element order, FP64 and fmad=false are unchanged. No observations,
hypotheses, particles, priors, IRLS iterations or thresholds are reduced.

Original numeric prefix, CPU body and original GPU kernel are literal copies.
The inherited explicit RegisterLUScorer selects original/generic/register
kernels;only dimension11 uses this new register path. The original range guard
and generic fallback remain. The same raw solve API is test-only. No normal
score buffer, counter, transfer, or default/product installation is added.

Compare against the corrected synchronized DLL in a new controlled experiment.
The rejected first register prototype's speed samples are not used. Whole-score
timing and bitwise accuracy evidence are reported separately from PF/FGO speed.
