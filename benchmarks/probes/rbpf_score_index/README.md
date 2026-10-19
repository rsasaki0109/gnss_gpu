# Immutable score-index aliases (D239)

Explicit experimental CPU-only CPython3.12/64-bit Windows extension. D236/defaults remain:CPU primary0/7,GPU joint0/7. A same-backend GPU secondary1/7 does not establish GPU advantage over the new CPU.

```powershell
cmake -S benchmarks/probes/rbpf_score_index -B dist/rbpf_score_index_local_build -G "Visual Studio 17 2022" -A x64 -DPython3_EXECUTABLE=<python.exe>
cmake --build dist/rbpf_score_index_local_build --config Release --parallel 2
```

`IndexMemo(retained_identities)` receives the original epoch memo's exact identity dict,or None to bypass. `get(key,hypothesis,original_key_index)` returns the same original Python integer index. Misses execute the original key-index tail;only retained immutable builtin tuple/int key identities and builtin empty keys can alias. Canonical is called before every get. Retain key/index references to prevent id reuse. Custom hash/equality keys disable aliases for the rest of the call;errors propagate after cleanup. No floating-point math,probability/score/RNG cache or new CUDA code. GC traverse/clear/dealloc handles temporary reference ownership. `summary()` records requests/hits/source-tail calls/bypasses/entries;counts are not timings.

```python
from rbpf_score_index_memo import ScoreIndexMemoStep

call = ScoreIndexMemoStep(runtime, scorer, 'native_cpu',
    extension=canonical_extension,
    parent_extension=parent_extension,
    mass_extension=component_mass_extension,
    index_extension=index_extension)
runtime.step = call
```

Use explicit paths and a caller-owned BatchScorer lifetime. `scalar`/`native_cpu`/`gpu` select the unchanged numerical backend. No profiler is inherited/called in this controller;it reuses only the verified adapter redirect code. Original source/method/NumPy-policy guards remain. `last['score_index_memo']` is separate from unchanged prior metadata.

Full-step helper arguments:

```powershell
python -B benchmarks/bench_rbpf_score_index_memo.py --dll <D226.dll> --extension <mass.pyd> --build-record <mass-build.json> --canonical-extension <canonical.pyd> --parent-extension <parent.pyd> --index-extension <index.pyd> --index-build-record <index-build.json> --output <fresh-report.json>
```

Field helper `bench_rbpf_score_index_memo_field.py` additionally takes `--cluster-relin`,`--mass-extension`,`--extension`(canonical),`--parent-extension`,`--index-extension` and inherited data/mode/output arguments. Fine diagnostic helpers are `rbpf_move_phase_profile.py`,`bench_rbpf_move_phase_profile.py`,`bench_rbpf_move_phase_profile_field.py`;their spans/calibration are wall diagnostics,not optimization results.

See [evaluation/all conditions/limits](../../../internal_docs/rbpf_score_index_memo_2026_10_16.md),[raw results](../../rbpf_score_index_memo_20261016.json),[completion check](../../rbpf_score_index_memo_completion_check_20261016.json). Preserve existing measurements;do not retime this source/workload to force adoption. No full-route/multiseed,whole-graph speed,CPU ASan or general-version guarantee is claimed.
