# Exact worst-CP release-subset selection probe

CPU-only CPython 3.12/64-bit Windows extension. Builds with `/O2 /fp:strict`; no CUDA build or installed extension replacement.
The D237 complete-step primary performance screens pass **0/7**. Keep D236/defaults; this is an explicit experimental candidate, not an accepted speedup.

```powershell
cmake -S benchmarks/probes/rbpf_shrink -B dist/rbpf_shrink_local_build -G "Visual Studio 17 2022" -A x64 -DPython3_EXECUTABLE=<python.exe>
cmake --build dist/rbpf_shrink_local_build --config Release --parallel 2
```

`worst_index(H,pairs,rows_v0,rows_iscp,rows_pair,rows_lam,rows_bmeas,rows_sig)` returns a Python pair index, `-1` for no eligible row, or `None` for original-Python fallback. All exact-integer DD pairs expand before scanning CP rows; first strict maximum and original FP64 arithmetic order remain. `rows_v0` supplies the row count only. Exact contiguous native-endian float64/bool/int32-or-int64 buffers, plain int maps, finite bounded CP inputs/DDs are supported. Unsupported formats/custom types/extremes/negative indices retain original Python semantics through the caller. No inputs or DD cache persist after a call.

```python
from rbpf_shrink import NativeShrinkStep

call = NativeShrinkStep(runtime, scorer, 'native_cpu',
    extension=canonical_extension,
    parent_extension=parent_extension,
    mass_extension=component_mass_extension,
    shrink_extension=shrink_extension)
runtime.step = call
```

Use a caller-owned `BatchScorer` lifetime and explicit extension paths. `scalar`, `native_cpu`, `gpu` select the unchanged D226 numerical backend; the new row selection runs on CPU in all three. The adapter only changes its local method view; runtime/module source and defaults remain unchanged. Source/method/binding guards and strict NumPy original-step fallback still apply. `last['shrink']` contains native/fallback call counts, rows and pairs, not timing fractions.

Whole-step helper arguments:

```powershell
python -B benchmarks/bench_rbpf_shrink.py --dll <D226.dll> --extension <mass.pyd> --build-record <mass-build.json> --canonical-extension <canonical.pyd> --parent-extension <parent.pyd> --shrink-extension <shrink.pyd> --shrink-build-record <shrink-build.json> --output <fresh-report.json>
```

`--extension` is the mass extension; `--canonical-extension` is separate. The actual-prefix helper is `bench_rbpf_shrink_field.py`, additionally requiring `--cluster-relin`, `--mass-extension`, `--extension` (canonical), `--parent-extension`, `--shrink-extension` and the inherited field/data/mode/output arguments. Whole-step fixtures exclude extraction/search/real graph solve. Actual prefixes are accuracy-only and never compared as speed samples. Preserve the completed D237 run; do not retime the same source/workload to force acceptance.

See [evaluation, all conditions and limits](../../../internal_docs/rbpf_shrink_2026_10_16.md), [raw report](../../rbpf_shrink_20261016.json), [completion check](../../rbpf_shrink_completion_check_20261016.json). No full-route/multi-seed, CPU ASan or general-Python-version guarantee is claimed.
