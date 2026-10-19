# Exact DD expansion and ordered component mass (D235)

Standalone Windows x64 / CPython3.12 CPU extension. `component_mass(Hs,pairs,
target,weights)` preflights builtin integer maps/pairs, expands ALL DDs in
original particle/pair order using PyLong subtraction then the original double
conversion, and returns a fresh pair-index/mass dict. Each mass adds original
FP64 weights in particle order. No tree reduction, approximation or cache.
Python-int equality is preserved with finite-double truncation, including
integers beyond int64. Non-target conversion errors are not skipped.

Native buffers are 1D contiguous native `d` doubles, weights finite in[0,1].
Unsupported input returns None; the caller executes the literal original block.
Unaligned NumPy export uses `=d` and falls back. Aligned readonly is supported.
The GIL stays held; temporary N*nb*8byte matrix and all buffers are released.

```powershell
cmake -S benchmarks/probes/rbpf_component_mass -B dist/repro_rbpf_component_mass -G 'Visual Studio 17 2022' -A x64 -DPython3_EXECUTABLE=C:/path/to/python312/python.exe
cmake --build dist/repro_rbpf_component_mass --config Release --parallel 2
```

Add `benchmarks` to the import path and choose every binary explicitly:

```python
from rbpf_likelihood_batch import BatchScorer
from rbpf_component_mass import NativeComponentMassStep

with BatchScorer(explicit_unchanged_D226_math_dll) as scorer:
    runtime.step = NativeComponentMassStep(runtime,scorer,'native_cpu',
        extension=explicit_D233_key_pyd,
        parent_extension=explicit_D234_parent_pyd,
        mass_extension=explicit_mass_pyd)
    # Keep scorer/workspace alive while the caller is used.
```

This opt-in frozen-model experiment inherits D234. Only the component-mass
block changes; all original candidates, priors, thresholds, FIX/RNG, strict
fallback and graph remain. No installed extension or default dispatch changes.
Tests require `GNSS_GPU_RBPF_PROBE_DLL`, `GNSS_GPU_RBPF_CANONICAL_EXT`,
`GNSS_GPU_RBPF_PARENT_EXT`, `GNSS_GPU_RBPF_MASS_EXT`, then run
`python -m pytest tests/test_rbpf_component_mass.py -q`.
Complete-step benchmark arguments: `--dll --extension --build-record
--canonical-extension --parent-extension --output`, with new explicit mass
extension/source SHA256 in the build record. Use fresh paths/serial GPU jobs,
retain all failed/noisy controls and do not retry timings for acceptance.

CPU screens pass3of6 at21.58-23.16%;GPU joint0of6.86tests and17memcheck tests
pass;actual default/cluster-enabled CPU/GPU800paired steps preserve saved
state/trajectory bits. Full-route/multi-seed and whole-graph gates incomplete.
See [evaluation](../../../internal_docs/rbpf_component_mass_2026_10_16.md),
[results](../../rbpf_component_mass_20261016.json), and
[completion](../../rbpf_component_mass_completion_check_20261016.json).
