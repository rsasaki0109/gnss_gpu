# Corrected register LU phase diagnostics

Standalone experiment only. Literal original numeric prefix,CPU score body,
original kernel and corrected D253 register helper/kernel remain as controls.
No model/observation/hypothesis/particle/prior/FP64/RNG/operation-order change.

Selector0 uses the corrected register score,1 adds sentinel trace readback,
2 enables clocks and boundary barriers,3 retains boundary barriers and trace
initialization/writeback but compiles out clock reads. The inherited register
selector remains available independently. Each block returns48 uint64 words:
the existing16 coarse markers plus three10-word LU stages and two reserved
sentinels. LU slots0..7 bracket load+guard,factorization,LU stores+ordered logs
and status barriers,RHS permutation,forward,backward,validation+solution stores.
Slots8/9 describe dispatch(1 register,2 original guard fallback) and processed
pivots. Non11 dimensions use the original solve and leave fine markers empty.

Every fine boundary adds two full-mask syncwarp calls,including in selector3.
The two original status-ordering barriers are retained. Failed factorizations
return after slot3;unsafe inputs return after slot1 and original solve. Existing
block barriers order marker copying. A separate raw3-variant probe compares
original/barriers-only/marked full LU,pivot,solution,logdet,status bits;never
called by normal score. Temporary buffers are bounded and freed on every path.

Whole-score paired calibration is required before interpreting fractions;
even a pass does not prove an unperturbed kernel or exclusive instruction cost.
clock64 includes waits,stalls and time-slicing. No pure synchronization cost,
cross-SM timestamp difference,kernel wall time,GPU utilization or PF GPU share.
No runtime/default installation or optimization adoption from this profiler.
