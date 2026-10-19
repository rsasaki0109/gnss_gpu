# Exact native parent overlap (D234)

Standalone Windows x64 / CPython3.12 extension. `score(child,parent)` returns
`(agree,compared)` with exact Python arbitrary-integer DD subtraction and the
minimum COMMON satellite reference, or `None` for unsupported inputs. It keeps
the GIL and retains no input/cache. Exact dict/tuple/int inputs are preflighted
before arithmetic; custom types/helpers preserve the original Python behavior.

```powershell
cmake -S benchmarks/probes/rbpf_parent_score -B dist/repro_rbpf_parent_score -G 'Visual Studio 17 2022' -A x64 -DPython3_EXECUTABLE=C:/path/to/python312/python.exe
cmake --build dist/repro_rbpf_parent_score --config Release --parallel 2
```

Add `benchmarks` to the import path; explicitly select all binaries:

```python
from rbpf_likelihood_batch import BatchScorer
from rbpf_parent_score import NativeParentScoreStep

with BatchScorer(explicit_unchanged_math_dll) as scorer:
    runtime.step = NativeParentScoreStep(runtime, scorer, 'native_cpu',
        extension=explicit_D233_canonical_pyd,
        parent_extension=explicit_parent_score_pyd)
    # Keep scorer/workspace alive while the caller is used.
```

Only the parent score changes in the D233 per-call view. Original cluster
support, thresholds, Python division/rank/tie handling, returned state identity,
candidates, math, RNG and strict-policy original-step fallback remain. The
actual runtime/module and installed extensions are not patched. No D231/D232
combination or default dispatch change.

Tests require `GNSS_GPU_RBPF_PROBE_DLL` (D226 numerical DLL),
`GNSS_GPU_RBPF_CANONICAL_EXT` (unchanged D233 `.pyd`), and
`GNSS_GPU_RBPF_PARENT_EXT` (new `.pyd`). Run
`python -m pytest tests/test_rbpf_parent_score.py -q`.
The complete-step screen takes `--dll --extension --canonical-extension
--build-record --output`. The build record pins new source/binary SHA256.
Use fresh output paths, serial GPU jobs and retain failed/noisy controls.

The original cluster option is disabled by default. Default field checks make
zero parent-score calls. To exercise the new path, use
`bench_rbpf_parent_cluster_field.py --variant key` and `--variant parent` with
the same backend/data/seed. This helper enables original cluster_relin=1 for
BOTH, keeps all other configuration unchanged and saves configs for audit.
Both configurations preserve800paired actual steps in total; no full-route,
multi-seed positioning or actual graph-speed acceptance is claimed.

See [evaluation](../../../internal_docs/rbpf_parent_score_2026_10_16.md),
[results](../../rbpf_parent_score_20261016.json), and
[completion](../../rbpf_parent_score_completion_check_20261016.json).
