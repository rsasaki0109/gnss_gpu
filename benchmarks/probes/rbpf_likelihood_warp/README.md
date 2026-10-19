# Exact warp-cooperative RBPF likelihood experiment (D231)

This standalone DLL keeps the D226 pageable transport and literal CPU/serial
GPU formulas. A second GPU kernel cooperates on LU rows and diagonal logs.
Pivot choice, each element's arithmetic and final log addition order remain.
The explicit `WarpBatchScorer` selects `serial` or `warp` at workspace creation.
The shipped runtime and all defaults are unchanged; the primary performance
screen failed in all9 conditions. Do not auto-enable from this experiment.

Build with CUDA12.8/MSVC x64 and FP64/fmad=false:

```powershell
cmake -S benchmarks/probes/rbpf_likelihood_warp -B dist/repro_rbpf_warp -G 'Visual Studio 17 2022' -A x64 -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build dist/repro_rbpf_warp --config Release --parallel 2
```

Add `benchmarks` to the import path and pass that explicit DLL to
`WarpBatchScorer(path, kernel="warp")`; its inherited `score` API and lifetime
are the same as [BatchScorer](../../rbpf_likelihood_batch.py). `kernel="serial"`
selects the preserved kernel, and `mode="native_cpu"` keeps the original fused
C++ computation. All paths preserve scalar fallbacks and requested candidates.
The `rbpf_lu_check` export is only a differential-test harness, not scoring.

Set `GNSS_GPU_RBPF_PROBE_DLL` to the frozen pageable DLL and
`GNSS_GPU_RBPF_WARP_DLL` to this DLL to run
`python -m pytest tests/test_rbpf_warp_likelihood.py -q`.
Builds/tests/checkers/benchmarks use serial GPU jobs and fresh output paths.
The batch helper takes `--old-dll`, `--new-dll`, `--build-record` (SHA256 source
and DLL map), and `--output`; it retains all failed gates without retry.

All GPU score/diagnostic bits and200 actual saved PF states match the prior
GPU in the evaluated domain. This is not full-route/multi-seed acceptance or
an improvement of a newer external model. See the [evaluation](../../../internal_docs/rbpf_warp_likelihood_2026_10_16.md),
[full results](../../rbpf_warp_likelihood_20261016.json), and
[completion check](../../rbpf_warp_likelihood_completion_check_20261016.json).
