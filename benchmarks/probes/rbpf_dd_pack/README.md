# Exact native DD input expansion (D232)

Standalone CPU DLL for Windows x64 / CPython3.12. It uses the CPython C API
with the GIL held through `ctypes.PyDLL`, validates exact builtin containers
and ints before any write, and performs arbitrary-precision subtraction before
one FP64 conversion. Unsupported input delegates to the original Python DD
function. No input mutation, narrowing, DD cache, GPU math or CUDA allocation.

```powershell
cmake -S benchmarks/probes/rbpf_dd_pack -B dist/repro_rbpf_dd_pack -G 'Visual Studio 17 2022' -A x64 -DPython3_EXECUTABLE=C:/path/to/python312/python.exe
cmake --build dist/repro_rbpf_dd_pack --config Release --parallel 2
```

The measured v1 lives in `rbpf_dd_pack.py`. Full likelihood/step benchmark
helpers use it with the unchanged D226 pageable numerical DLL. Its strict
DD-bytecode guard accepts the isolated AST fixture, and rejects the imported
module call form. Keep this measured source/results frozen.

For actual imported frozen runtime, use the separate `rbpf_dd_pack_runtime`
variant. It derives both exact call forms from the SHA-pinned original source;
only this guard changes. All other AST, native code and numerical DLL are
unchanged. The fresh real-data runs are accuracy-only; the variant is unmeasured.

```python
from rbpf_dd_pack_runtime import PackedDdScorer, PackedDdStep

with PackedDdScorer(explicit_math_dll, pack_dll=explicit_pack_dll) as scorer:
    runtime.step = PackedDdStep(runtime, scorer, 'gpu')
    # Run while scorer/workspace and the selected caller are alive.
```

Add `benchmarks` to the import path and preserve the existing CUDA runtime
configuration for the unchanged score DLL. Use only explicit DLLs/callers;
no installed-extension replacement or automatic/default dispatch is provided.
The CPU-only pack DLL itself has no CUDA dependency. `native_cpu` keeps the
same fused C++ arithmetic; candidate limits and original scalar fallbacks apply.

Set `GNSS_GPU_RBPF_PROBE_DLL` to the frozen math DLL and
`GNSS_GPU_RBPF_DD_PACK_DLL` to this pack DLL, then run
`python -m pytest tests/test_rbpf_dd_pack_runtime.py -q`.
Run GPU tests/checkers/benchmarks serially and use fresh output paths.
The likelihood helper takes `--dll --pack-dll --build-record --output`;
the complete-step helper takes `--dll --pack-dll --output`.

The full-step CPU/GPU screens pass0of6 despite larger-batch gains. Defaults
remain unchanged. Preserve failed/noisy controls and do not retime to pass.
See [evaluation](../../../internal_docs/rbpf_dd_pack_2026_10_16.md),
[results](../../rbpf_dd_pack_20261016.json), and
[completion](../../rbpf_dd_pack_completion_check_20261016.json).
