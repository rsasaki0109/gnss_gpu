# Experimental block-element ordered11x11 LU

One128-thread block cooperates on each normal solve;up to121 threads own one
shared LU element each. Pivot/tie selection remains lane-zero ascending scan.
Row exchanges use one thread per column. Lower-column divisions run by row,
then trailing cells update independently with the original ascending-j order.
Explicit whole-block barriers order pivot,swap,division and update phases.
A local abort predicate passes through syncthreads_or,so all threads stop at
the same failed pivot without shared-status read/write races. Diagonal logs
are independent then added in original j order,including partial failures.
Forward/backward substitution is the literal original serial solve_lu.

All128 threads enter every collective;no per-warp return/participation rule.
Initial nonfinite or abs>2^500 normal/RHS entries uniformly dispatch original
generic solve. This is an implementation guard,not a changed model threshold.
Other dimensions3..10 stay generic. Preserve asymmetric prior,all observations,
hypotheses,particles,IRLS iterations,thresholds,RNG and FP64/fmad=false. No
tree sum,FMA reassociation,approximate reciprocal,jitter or symmetrization.

Original numeric prefix,CPU body and original GPU kernel are literal copies.
Standalone DLL only,no product/default installation or new offload domain.
Existing RegisterLUScorer and rbpf_register_select names are retained as a
legacy validation ABI:selector0 original,1 same-new-kernel generic control,
2("register" in the reused Python wrapper) selects this block-element LU.
No register-row/cache/conditional-shuffle optimization is combined here.
The raw solve probe uses128-thread blocks and unchanged arrays/allocation cap;
the synchronized register reference uses its own frozen32-thread raw probe.
No score buffer/counter/transfer is added. Whole-score timing is separate
from actual PF accuracy and does not establish PF/FGO speed or GPU utilization.
