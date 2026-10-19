# Adaptive row-capacity shared scratch

Capacity60 for input m<=60,64 for61..64;65..128 uses the literal corrected
register128-row kernel. No observation is dropped:dispatch uses the complete
input row count,not the count kept after marginalization. All row loops,
indices,strides,FP64 arithmetic,IRLS,asymmetric priors,pivots,ties,operation
order,guard values,status handoffs,RNG,particles and hypotheses are retained.
Original numeric prefix,CPU body,original/corrected kernels and register helper
stay literal controls. Namespace compact duplicates numeric bodies with only
row-array extents,type signatures and function qualification changed.

Register selector0 original serial,1 original generic,2 adaptive,3 literal
corrected128,4 templated128 control. RegisterLUScorer legacy default/register
selects2;CompactScratchScorer provides named explicit selections. Its inherited
score wrapper is unchanged. No additional numeric staging or timed counters.
Scratch footprints:16216/17080/30904 bytes for60/64/128 versus original30904.
The resource query uses function attributes and potential-occupancy API on
this process's device;it does not measure achieved occupancy or GPU utilization.

Raw4-variant original generic/compact60/64/128 solve validation is test-only,
bounded to256 matrices,with all buffers freed. No source from completed
experiments is edited. Standalone DLL only,no installation or default change;
whole-score and whole-PF speed/accuracy require separate validation.
