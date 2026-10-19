# Experimental register-row11 ordered LU and solve

Explicit standalone DLL;no default/product installation. Original D226 numeric
prefix,CPU body and original GPU kernel remain literal. New kernel dispatches
dimension11 normal solves to warp zero,one row in registers per lane. All32 lanes
participate in every full-mask shuffle/vote;shared state handoffs use syncwarp
and existing block barriers. Other dimensions use original generic solve.

Pivot comparison/tie scan remains ascending,rows exchange via shuffles,LU updates
keep per-element j order. Diagonal logs run independently then sum in original
order,including partial failures. Permutations retain original order. Forward
substitution interleaves rows by ascending k,retaining each row's subtract order.
Backward substitution retains descending i and ascending k. No tree sum,FMA,
approximate reciprocal,jitter,symmetrization,IRLS/prior/observation/candidate/RNG
or numerical-policy threshold changes. FP64,fmad=false. Initial nonfinite or
abs>2^500 normal/RHS values dispatch to literal original solve;this is a
performance guard and introduces no numerical approximation.

rbpf_register_select:0 literal original kernel,1 new kernel generic control,
2 new kernel register-row11 dispatch. Generic control has new code/resources
and is not assumed equal in time to old code. Optional Python scorer inherits
the original full score method unchanged. No new transfer/counter in score.
rbpf_register_solve_probe compares raw LU,pivots,solution,logdet,status for
dimension3..11;test-only temporary storage,never called by normal score.

Corrected successor of the frozen first register-row prototype. An additional
syncwarp after the shared status0/1 read orders that read before the final
lane-zero status3 write. Initial extreme finite-input racecheck reported9
warning groups (read0x13710/write0x16ed0 of shared0x7894);initial outputs matched.
This version has separate build/validation evidence. Initial speed samples do
not measure this corrected version;do not transfer their speed claim to it.
