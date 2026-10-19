# Explicit GPU stream transport experiment (D248)

The full numeric prefix,CPU formulas and likelihood kernel are identical to
the frozen D226 pageable source. Only workspace-owned nonblocking-stream
transfer/launch/completion and optional diagnostic CUDA events are added.
Default control uses the original synchronous copies/default stream.
Normal scores record no diagnostic events. CPU equations/FP64/IRLS/order,
all observations/candidates/priors/thresholds/RNG remain unchanged.
No runtime/default installation. Shared workspace concurrent use unsupported.

Build with CUDA12.8/MSVC x64,Release,fmad=false (no fast math):

```powershell
cmake -S benchmarks/probes/rbpf_gpu_stream -B dist/repro_rbpf_gpu_stream -G "Visual Studio 17 2022" -A x64
cmake --build dist/repro_rbpf_gpu_stream --config Release --parallel 2
```

Pass the explicit DLL to rbpf_gpu_stream.StreamBatchScorer(stream="nonblocking")
or stream="default" for the unchanged transport control in the same binary.
Both return completed host results. No assumed pageable-memory overlap;
async errors are drained before buffer reuse and destruction.

[CUDA12.8 API synchronization](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-runtime-api/api-sync-behavior.html)
and [stream semantics](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-runtime-api/stream-sync-behavior.html).
Profile event intervals include enqueue gaps/contention;diagnostic timing is
separate from noninstrumented whole-score acceptance.
