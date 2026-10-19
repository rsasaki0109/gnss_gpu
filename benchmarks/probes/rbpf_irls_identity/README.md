# Exact repeated IRLS state

Explicit standalone FP64 CPU/CUDA likelihood DLL for the D236 caller. Retains
the original maximum of two IRLS updates. After computing the original weights,
compares every weighted matrix/right-hand-side binary64 bit, including signed
zero. If Aw/bw are identical to the solved system, its ordered normal/RHS,
partial-pivot LU, solution, subsequent weights and final output repeat exactly.
Only those redundant normal constructions/factorizations/solves are skipped.

No tolerance, approximate convergence, reduced precision, changed row order,
FMA, particle/candidate/prior/threshold/RNG change, or cross-hypothesis reuse.
When any bit changes, the original next solve executes. Original status/error,
diagnostics and fallback semantics remain. No prior rejected candidate or
profiling code is combined. Undoing the generated changes recovers the entire
pinned D226 source. Product/defaults stay unchanged pending complete tests.

Counters add one int32 per hypothesis to the existing host/device output
workspace/readback. They are read by `IdentityBatchScorer`; `IRLSIdentityStep`
keeps them separate from existing caller metadata. Whole-step timings include
all comparisons, synchronization, counter transfers/query and allocations.
Extra workspace and negative conditions must be reported, as well as any wins.
