# Native arbitrary-integer basin keys (D233)

Standalone Windows x64 / CPython3.12 extension. `canonical(H)` returns the
exact original builtin tuple key, or `None` for unsupported input. It validates
all dict/tuple/int types before arithmetic and never converts integers to
fixed-width or floating point. The C function keeps the GIL and retains no H/key.

```powershell
cmake -S benchmarks/probes/rbpf_canonical -B dist/repro_rbpf_canonical -G 'Visual Studio 17 2022' -A x64 -DPython3_EXECUTABLE=C:/path/to/python312/python.exe
cmake --build dist/repro_rbpf_canonical --config Release --parallel 2
```

Add `benchmarks` to the import path and choose both binaries explicitly:

```python
from rbpf_likelihood_batch import BatchScorer
from rbpf_native_canonical import NativeCanonicalStep

with BatchScorer(explicit_unchanged_math_dll) as scorer:
    runtime.step = NativeCanonicalStep(runtime, scorer, 'native_cpu',
        extension=explicit_canonical_pyd)
    # Keep the scorer/workspace alive while the caller is in use.
```

This is an opt-in experiment for the frozen source only. It keeps D229 moves,
state/RNG, original numerical DLL and strict/scalar fallbacks. No installed
extension replacement, global/runtime method patching, or default dispatch.
Unsupported numeric types/mappings and custom builtin bindings run the original
canonical function. Keys are fresh on every call; no extra key/DD cache is added.
D231 warp and D232 DD packing are not combined with this experiment.

Set `GNSS_GPU_RBPF_PROBE_DLL` to the unchanged D226 likelihood DLL and
`GNSS_GPU_RBPF_CANONICAL_EXT` to the new `.pyd`, then run
`python -m pytest tests/test_rbpf_native_canonical.py -q`.
The full-step benchmark takes `--dll --extension --build-record --output`,
where the build record pins extension/source SHA256. Use fresh output paths
and serial GPU jobs; retain every failed/noisy gate without timing retries.

CPU passes2of6 complete-step screens (10.27%/13.21%,5of5 rounds); GPU joint
screens pass0of6. Defaults remain unchanged. Actual CPU/GPU each preserve200
saved states/trajectory arrays; this is not full-route/multi-seed acceptance.
See [evaluation](../../../internal_docs/rbpf_native_canonical_2026_10_16.md),
[results](../../rbpf_native_canonical_20261016.json), and
[completion](../../rbpf_native_canonical_completion_check_20261016.json).
