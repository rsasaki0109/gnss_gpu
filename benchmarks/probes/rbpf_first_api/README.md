# First CUDA operation after idle,diagnostic only

The complete D286 translation unit and register header remain byte-identical.
This is an optional Windows observation API,not an adopted optimization.
rbpf_first_api_prepare zeroes native host staging on the CPU before idle.
rbpf_first_api_probe runs one of:none,default-stream query,device synchronize,
H2D1byte,H2D65536bytes,device memset1byte. Transfers/memset are followed by
device synchronization to retain completed work. Query/sync/none add no copy.
No estimator kernel is added or changed;memset may use runtime device work.
All GPU work and completion stay inside the diagnostic interval.

Use a single exclusively owned workspace. prepare may overwrite host staging,
and transfer/memset may overwrite device input. The next ORIGINAL score must
freshly repack/upload its active input before calculation. No cached numerical
result is reused. Python inputs,graph,particles and RNG are never modified.
Data header and three raw QPC/thread-CPU rows describe API and completion spans.
Invalid args leave output untouched. These calls do not alter score TLS traces.

The caller compares a fresh process/workspace for each operation/trial and
keeps probe plus whole completed score time. Moving a wait to this preamble
does not establish a speedup. No power/keepalive/default/model change.
