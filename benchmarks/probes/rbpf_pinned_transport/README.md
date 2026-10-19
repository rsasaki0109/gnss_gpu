# Synchronous pinned host staging candidate

Optional Windows experiment derived from D286. Only Workspace host allocation
ownership and matching destruction change. Both whole score functions,all
numeric functions,kernels,memcpy APIs,launch order and host clock boundaries
remain byte-identical. Device SASS/resources must match before numeric use.

Call rbpf_host_staging_select once immediately after create,before any score:
0 selects original pageable staging;1 allocates pinned input/output with
cudaMallocHost. Allocation failure leaves original buffers owned for cleanup;
no silent performance fallback. Repeated selection is rejected. Raw ABI callers
must obey the pre-score/exclusive-owner contract;Python enforces selection
during construction. No pending-job API,stream/event/power/keepalive changes.

rbpf_host_staging_info copies version/selected/pinned/input_bytes/output_bytes.
rbpf_host_staging_validate explicitly checks both pinned CUDA pointer types
outside performance windows. Normal scoring does not call this diagnostic.
Pinned allocations are freed with cudaFreeHost;pageable with delete[].
Results are completed host values with original fallback and FP64/IRLS/order.
No optimization adoption or cold/steady speed claim until controlled results.
