# Fused RBPF likelihood prototype (D226)

This is an explicit component API, independent of the shipped RBPF runtime.
It scores an already requested list of hypotheses and does not replace `step`,
generate moves, update graphs, modify particles, or draw random numbers.
No backend or runtime default has changed.

The CUDA kernel performs residual construction, original uniform/Gaussian
unheld-row treatment, whitening, group nuisance columns, initial conditional
solve, both original Huber IRLS passes, and the mixture/Laplace likelihood in
one launch. Each hypothesis gets its own block, prior and normal matrix. The
same numerical functions provide a fused C++ CPU baseline. No new jitter,
variance floor, candidate limit or likelihood model is introduced.

Input, arithmetic and output are double precision; fast math is disabled and
CUDA FMA contraction is disabled. Partial-pivot LU uses full matrix entries,
including tiny asymmetries. This is the same mathematical model as the pinned
scalar `_loglik`, but reductions/LU/libm need not match NumPy/LAPACK bits.
The diagnostic tolerances in prototype tests do not change a positioning gate.

Three immutable experimental sources retain the same numeric functions and
identical GPU machine code:

| source | device transfers per normal score call | extra host staging |
|:---|:---|:---|
| `rbpf_likelihood_batch` | 6 H2D + 2 D2H | none |
| `../rbpf_likelihood_packed` | 1 H2D + 1 D2H | pinned |
| `../rbpf_likelihood_pageable` | 1 H2D + 1 D2H | pageable |

The common [Python API](../../rbpf_likelihood_batch.py) preserves first-request
order with gauge-invariant deduplication/scatter. Its capacity bounds are128
rows,128 pairs,8 groups and the explicitly allocated hypothesis capacity
(default512). Unsupported sizes/dtypes, strict/underflow-warning policies, bad values,
native errors and singular systems retain the original scalar calls. These
bounds do not truncate candidates or observations. Workspace objects own their
buffers and must be closed, normally through `with BatchScorer(...)`.

At capacity512 each source owns1,143,352 explicit device bytes. The packed and
pageable sources additionally own1,143,352 host staging bytes. These counts
exclude context/driver storage, Python packing and kernel stack residency.
The kernel uses30,904 shared bytes/block,56 registers/thread and32 stack
bytes/thread, with no register spills. Cold setup is excluded from steady-state
batch timing and retained separately in raw measurements.

## Build and run

Use CUDA12.8/MSVC x64, CMake, NumPy and the existing benchmark dependencies.
Run GPU builds/tests/benchmarks serially and use fresh output paths.

```powershell
cmake -S benchmarks/probes/rbpf_likelihood_pageable -B dist/repro_rbpf_pageable -G 'Visual Studio 17 2022' -A x64 -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build dist/repro_rbpf_pageable --config Release --parallel 2
$env:GNSS_GPU_RBPF_PROBE_DLL = "$PWD/dist/repro_rbpf_pageable/Release/rbpf_likelihood_probe.dll"
python -m pytest tests/test_rbpf_likelihood_batch_probe.py tests/test_rbpf_likelihood_packed_probe.py -q
```

For actual frozen-runtime objects, the explicit
[adapter](../../rbpf_likelihood_runtime_adapter.py) binds their original numeric
methods and rejects other source versions. Updating a newer model requires a
new numerical audit; a successful adapter binding is not a field validation.

```python
# Add the repository's benchmarks directory to the import path explicitly.
from rbpf_likelihood_batch import BatchScorer
from rbpf_likelihood_runtime_adapter import runtime_reference

reference = runtime_reference(rbpf)  # pinned model only; no monkey-patching
with BatchScorer(dll_path, capacity=512) as scorer:
    values, details = scorer.score(requested_hypotheses, epoch_problem,
                                   reference, mode='gpu')
    # mode='native_cpu' is the same fused C++ calculation without GPU transfers.
```

For benchmarking, create a JSON record containing `source_sha256` for the
chosen CMake/CUDA files and `dll_sha256` for the freshly built DLL. Pass it to
`bench_rbpf_likelihood_batch.py` with `--dll`, `--build-record`, `--output`.
`bench_rbpf_likelihood_transport.py --build-records <separate.json> <pinned.json>
<pageable.json> --output <new.json>` compares all three in the same rotating
experiment. In that helper each record's path without `.json` must be its
build directory containing `Release/rbpf_likelihood_probe.dll`.

Both helpers include deduplication, original DD/cluster lookup, packing,
epoch uploads, host return and scatter; allocation of retained capacity, warm
calls, hashing and validation are outside timing. They are complete likelihood
batches, not whole RBPF epochs or raw-data positioning evaluations.

The final source improves the component performance screen at32/64 unique
hypotheses with64 rows/4 groups;21 unique hypotheses miss the10% GPU-over-native
CPU threshold. Keep explicit selection pending actual caller, trajectory,
integer/RNG and original full accuracy gates. Do not infer a universal threshold
from these synthetic fixtures or remove failed/noisy conditions.

See [D226 results](../../RESULTS.md), the [complete report](../../rbpf_likelihood_gpu_prototype_20261016.json)
and [implementation notes](../../../internal_docs/rbpf_likelihood_gpu_prototype_2026_10_16.md).
