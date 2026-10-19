# Pinned three-node CUDA Graph experiment (D249)

Numeric prefix and CPU equations are literal D226 copies;the retained parent
transport comment above Workspace describes that parent source only.
Both new modes own pinned host input/output,device buffers and a nonblocking
stream. Graph mode owns one H2D -> unchanged kernel -> D2H graph executable.
Workspace setup instantiates a dummy shape without launching or scoring.
Real shape changes update all three node parameters inside score. Equal shapes
reuse only pointer/geometry/byte-count metadata;every call uploads fresh values.
There is no cross-epoch score cache,unbounded shape cache or extra profile graph.
All returned results are complete host arrays after explicit synchronization.
One workspace is single-threaded. Async errors drain before reuse/destruction.
Allocation failures clean up pinned/device buffers and graph resources.

Keep FP64,ordered sums,IRLS iterations,all observations/candidates/priors,
thresholds and RNG. No runtime/default installation. No diagnostic events.

```powershell
cmake -S benchmarks/probes/rbpf_gpu_graph -B dist/repro_rbpf_gpu_graph -G "Visual Studio 17 2022" -A x64
cmake --build dist/repro_rbpf_gpu_graph --config Release --parallel 2
```

Use an explicit DLL with rbpf_gpu_graph.GraphBatchScorer(transport="graph")
or transport="stream" for the pinned control in the same binary.
Build Release with fmad=false and no fast math.

[CUDA12.8 graph API](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-runtime-api/group__CUDART__GRAPH.html)
requires live original nodes and same-context 1D memory for executable memcpy
updates. Nodes,allocations and executable all live until workspace destruction.
