# Ordered structural-zero normal products (D250)

Standalone GPU experiment. Original numeric prefix,Scratch,LU/solve/IRLS,
CPU score and original GPU kernel remain literal D226 copies. The retained
parent transport comment refers to that source. GPU transport stays pageable
and synchronous;no Graph or dedicated-stream change is combined.

New GPU modes are dense control and group-indexed normal sums. A stable index
lists kept rows in each CP group. All Aw/bw must be finite with magnitude at
most2^500 and inactive group entries must be positive zero;otherwise later stages use the old dense
normal function. Active dot products preserve row order. Cross-group products
are structural zeros. FP64 round-to-nearest-even sums start at positive zero;
omitting signed-zero terms preserves the ordered sum,including subnormals.
The bound keeps every dot-product partial sum finite;extreme finite inputs
preserve the original overflow/NaN operations through the dense fallback.
No tree reduction,epsilon pruning,triangle copying,jitter or iteration changes.
Full asymmetric prior,candidates,observations,thresholds and RNG are retained.

Normal score records no stage counters. Separate count_score calls allocate
optional4uint64 per hypothesis on host/device and copy counters after scores.
The raw normal probe is a separate differential-test API,never a score path.

```powershell
cmake -S benchmarks/probes/rbpf_sparse_normal -B dist/repro_rbpf_sparse_normal -G "Visual Studio 17 2022" -A x64
cmake --build dist/repro_rbpf_sparse_normal --config Release --parallel 2
```

Explicit SparseNormalScorer(kernel="original"/"dense"/"sparse"). Build FP64,
fmad=false,no fast math. CPU always uses the original dense equations.
No product/default change. Shared workspace concurrent use unsupported.

[CUDA12.8 FP64 rounding](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__DOUBLE.html)
and [floating-point flags](https://docs.nvidia.com/cuda/archive/12.8.1/floating-point/index.html).
