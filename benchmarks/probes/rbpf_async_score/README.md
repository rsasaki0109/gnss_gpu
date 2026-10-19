# Owning score begin/finish transport

Private experiment. The entire corrected register-LU translation unit and
header precede a host-only API addition byte for byte. Numeric kernels,
FP64/order/IRLS/pivots and original synchronous exports remain unchanged.

`rbpf_async_create(legacy_workspace)` borrows one exclusively owned legacy
workspace and allocates a nonblocking stream plus pinned input/output staging.
`rbpf_async_begin` has the same argument signature as `rbpf_score`,requires GPU
mode,fully snapshots input values before returning,and retains output targets.
`rbpf_async_finish` waits,copies results,and releases the targets.
`rbpf_async_drain` waits without publishing output;it is idempotent.
Destroy the async object before the borrowed legacy workspace. Python retains
all NumPy buffers until finish/drain and rejects concurrent workspace reuse.

The blocked scheduling control uses the same stream/transfers but waits in
begin. A submitted job is not evidence that useful overlap or speed exists.
No additional numerical kernel,split batch,probability cache,or default change.
Foreign concurrent mutation/invalid raw ABI pointers,allocation/device failure
injection,and arbitrary private monkeypatches are outside this prototype.
