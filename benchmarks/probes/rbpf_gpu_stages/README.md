# Experimental GPU stage elapsed-cycle diagnostic

Explicit standalone DLL; no product/default changes. Original numeric functions,
CPU body and original GPU kernel are literal copies of D226. A separate kernel
adds lane-zero clock64 markers at existing synchronization boundaries and after
finish/write_result. No new block barriers or numerical changes are introduced.

rbpf_profile_select:0 original,1 original plus identical trace readback/query,
2 instrumented kernel. Buffers allocate lazily and each selection resets count.
rbpf_profile_read returns count with null output,or copies count*16 uint64 values.
Selection0 disables recording. CPU,fallback,empty/error calls yield no valid trace.
The Python profiled_score resets mode in finally and copies all returned storage.

Slots0..13 are elapsed-cycle boundaries:initialize0..1,prepare1..2,normal0=2..3,
solve0=3..4,reweight0=4..5,normal1=5..6,solve1=6..7,reweight1=7..8,
normal2=8..9,solve2=9..10,likelihood=last solve..11,finish11..12,
write_result12..13. Unvisited markers equal uint64 max;skipped stages are not0.
Slot14 is kept row count,slot15 low32 is dimension and high32 is status.
Total13-0 excludes final trace writeback and kernel prologue/epilogue.

clock64 measures per-block elapsed SM cycles including stalls/time slicing.
It is not instruction cycles,global kernel wall time,GPU utilization,or whole PF
GPU coverage. Do not compare absolute timestamps across blocks/SMs,or sum block
cycles as wall time. Profile overhead and copy-control must be measured separately.
Raw reports and failed profiler attempts remain private;publish only sanitized data.
