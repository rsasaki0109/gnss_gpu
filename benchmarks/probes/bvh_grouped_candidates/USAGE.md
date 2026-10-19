# D297 reproduction entry point

Use **generate_checked.py** to reproduce the measured register reuse candidate:

```powershell
python -B benchmarks/probes/bvh_grouped_candidates/generate_checked.py --output dist/d297_reproduction_source
```

The original generate.py is retained as the initial static projection failure.
Its reversible proof rejected an empty replacement before compilation or any
GPU or numerical execution. The checked version replaces that obsolete host
reset with a unique comment and otherwise preserves the candidate arithmetic.

Build the generated directory with the frozen CMake project under
`../bvh_tiled_geometry`, passing CODE_ROOT with forward slashes, CUDA/C++20
Release and the original FMA policy. Keep the extension separate from the
installed product. The current product does not expose this experimental API.

The measured large native cases passed both warm and cold controls, with exact
LOS/reflection outputs. The combined four-case gate failed because small cold
A/A comparisons were unstable. No default or public/urban pipeline adoption is
established. The complete result is in the D297 evaluation and JSON report.
