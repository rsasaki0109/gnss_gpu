# Experimental fixed-size ordered normal LU

Explicit standalone DLL;no product/default change. Original D226 numeric prefix,
CPU body and original GPU kernel remain literal copies. New GPU kernel dispatches
dimension11 normal solves to fully unrolled compile-time-N functions;other sizes
retain generic solve_normal. Partial-pivot search/ties/row swaps,logdet accumulation,
division,ordered LU updates,pivot permutation,forward/back substitution and status
rules are retained. FP64,fmad=false;no fast math,reciprocal approximation,jitter,
symmetrization,observation/candidate/prior/IRLS/RNG/threshold change.

rbpf_fixed_select:0 literal original kernel,1 new kernel generic control,
2 new kernel fixed11 dispatch. The generic-control kernel has new code/resources;
it is a diagnostic contrast,not an equal-performance control to old code.
rbpf_fixed_solve_probe compares isolated generic/fixed LU,logdet,pivots,solution,
status for arbitrary dimension3..11 matrices including singular/nonfinite values.
This test-only kernel/API is never used by normal score.
