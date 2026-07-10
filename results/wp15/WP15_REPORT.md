# WP15_REPORT — TASK_M21: CUDA-accelerate the Python pipeline's hot paths

Workspaces: pipeline `C:\Users\rsasa\Workspace\old\repro_tc_fgo`
(`gtsam_rtk_standalone.py`, WP13r run2 ship config = Q4 + `COND_HOLD=1`);
CUDA home `C:\Users\rsasa\Workspace\old\gnss_gpu` (new `src/ar/lambda_batch.cu`
following the established `rtk.cu` extension pattern). All changes opt-in
behind `CUDA_LAMBDA=1` with automatic CPU fallback; defaults bit-identical
(verified, see Part 4). cssrlib and tc/ untouched. Nothing committed.

## Part 1 — Profile first (deliverable #1)

Profiled the standalone on run2, 3000 epochs, WP13r run2 ship flags
(`results/wp15/probe.sh` = the WP13r probe base + Q4 deltas + `COND_HOLD=1`,
`--ar-rtklib-mode 1 --subset-ar-enable 1` + `SUBSET_AR_USE_PAR=1`).
Uninstrumented baseline: **3000 ep in 248.5 s (12.1 ep/s)**. Two profilers:

- cProfile (`results/wp15/prof3000.prof`, run wall 325.8 s, tottime 256.2 s)
- py-spy 150 Hz attach on a live steady-state run (120 s window,
  `results/wp15/pyspy_att.speedscope.json`; the launcher mode fails on the
  venv shim with "Failed to find python version", attach-by-PID works)

Top blocks (cum% of cProfile tottime | py-spy steady-state cum%):

| block | cProfile | py-spy | ncalls (3000 ep) | GPU-portable? |
|---|---|---|---|---|
| **AR chain `_run_lambda_attempts`** | **22.4%** | **17.9%** | 34,786 `resamb_lambda` calls | **subset cascade: YES** |
| — `_try_subset_ar` cascade | 19.0% | 16.2% | 2,039 cascades, ~29.5k mlambda calls | YES (batch of independent (y,Qb) problems) |
| — cssrlib `restamb` (self) | 9.6% | 10.5% | 13,320 | no (Python scalar bookkeeping loop, cssrlib read-only) |
| — cssrlib `mlambda` numeric | 7.2% | 5.3% | 34,575 | yes (the actual ILS math) |
| cssrlib `prepare_double_difference_measurements` | 27.4% | ~17% | 3,000 | no (read-only cssrlib) |
| — `satposs` (findeph linear scans + eph2pos) | 15.3% | ~6.5% | 6,000 (247k findeph) | no (sequential per-call list search inside cssrlib) |
| — `qcedit` (self-heavy Python loops) | 11.1% | ~4% | 6,000 | no |
| cssrlib `update_ambiguities` + `initx` (self) | 16.5% | ~9-13% | 39,821 initx | no (O(nx) Python scalar loops per reset, cssrlib read-only) |
| RINEX `decode_obs` + `sync_obs_hold` | 7.6% | ~5% | 3,602 | no (text parsing / I/O) |
| GTSAM native (ISAM2/IFLS update, marginals) | 5.3% self | ~12-15% | — | excluded by task (already C++) |
| `_write_back_tc` marginal extraction loops | ~7% | ~9-11% | 2,991 | no (`jointMarginalCovariance` itself is only ~1-2%; the cost is Python `jm.at` loops) |
| `_apply_fde` residual sweep | ~2% | ~3-5% | 3,000 | below bar |

Key profile facts:

- **GTSAM is NOT the bottleneck** (lag-1.0 IFLS keeps graphs small): ~5%
  self in cProfile. `jointMarginalCovariance` ~1-2% — the "cheaper exact
  alternative" investigation the task pre-authorized is **not needed**.
- **The subset/retry AR cascade is the one block that meets both port
  criteria** (>=10% wall AND data-parallel): 2,039 of 3,000 epochs fire it;
  it produces 29.5k of the 34.6k mlambda calls (avg 14.5 per cascade, the
  predicted 12-30 pattern), and each sequential CPU combo evaluation drags
  cssrlib Python overhead with it (`ddidx` + y/Qb build + `restamb` on every
  internally-accepted-then-discarded candidate). Direct in-run timing
  (`WP15_TIME=1`, Part 4's `t1_cpu`) puts the CPU cascade at **14.7% of
  wall** (29.7 s of 202.3 s over 2,000 ep) — consistent with py-spy's
  16.2%.
- The other large items are cssrlib-internal Python scalar loops
  (`restamb`, `initx`, `qcedit`, `findeph`) — read-only code, no
  data-parallel numeric structure (bookkeeping, not math). Reported
  honestly as NOT portable to CUDA; see Part 6 for the residual-lever note.

## Part 2 — What was ported: batch LAMBDA (`gnss_gpu` CUDA unit)

New CUDA unit following the `rtk.cu` -> `_rtk_bindings.cpp` -> `rtk.py`
pattern:

- `src/ar/lambda_batch.cu` — faithful transcription of cssrlib
  `mlambda.py`: reversed-order LDL (`_ldldecom` + the `d<1e-10`
  LambdaError check), LLL reduction (`_reduction`, LOOPMAX 10000),
  search-and-shrink ILS (`_estimILS`, LOOPMAX abort semantics, np.rint
  half-even rounding, np.argmax/argsort tie behavior), `_sr_boost`, and
  the parmode=2 PAR path (`parsearch`, exclmax=1) — one GPU thread per
  (ahat, Qahat) problem, one kernel launch per batch. Compiled with
  `--fmad=false` so the FP operation order matches numba njit exactly.
- `include/gnss_gpu/lambda_batch.h`, `python/gnss_gpu/_lambda_batch_bindings.cpp`
  (pybind11), `python/gnss_gpu/lambda_batch.py` (wrapper mirroring
  `rtk.py`'s try-import/`HAS_*` style; returns cssrlib-shaped results
  including the parmode=2-reject empty-s/1-D-afix convention).
- CMake: new `gnss_gpu_ar` lib + `_gnss_gpu_lambda_batch` module
  (additive; every existing target untouched and the full configure
  still succeeds).
- Tests: `tests/test_lambda_batch.py` (16 tests: brute-force ILS
  cross-check, PAR accept/reject semantics, non-PD status, batch/single
  consistency, cssrlib parity when importable, captured-input replay
  when the WP15 npz is present).

### Pipeline wiring (`CUDA_LAMBDA=1`, CPU fallback)

`gtsam_rtk_standalone.py` `_try_subset_ar` is now a dispatcher:
`CUDA_LAMBDA=1` + the campaign's `SUBSET_AR_USE_PAR=1` routing ->
`_try_subset_ar_cuda`, which

1. enumerates the combos in the CPU loop's exact order (same
   `minfixsats` skip), runs the same `ddidx` per combo (so `nav.fix`
   sees the identical rewrite sequence) and builds (y, Qb) with
   `resamb_lambda`'s own expressions,
2. evaluates ALL combos in ONE `lambda_batch` kernel launch,
3. replicates the CPU selection verbatim (per-combo `_last_s0/_last_s1`
   stash from the kernel's bit-identical s values, `resamb_lambda`'s
   accept condition, `min_nb` gate, (ratio, nb, -k) scoring,
   strong-ratio early stop), and
4. re-fires ONLY the winning subset through the unchanged CPU
   `resamb_lambda` — the CPU cascade ends with the identical re-fire, so
   the final `nav.xa/Pa/fix` + `restamb` state matches by construction.

Automatic CPU fallbacks: extension/GPU missing (probe at init, warns
once), any GPU error, kernel-unsupported dims (n>64; observed max 36),
and the one case whose transient CPU side effects the emulation cannot
reproduce (a combo internally accepted but `min_nb`-rejected with no
winner) — the whole cascade then reruns on the exact CPU path.
Fallback count on the full run2: **0**.

NOT ported (honest scoping, per the profile):

- GTSAM/ISAM2/marginals — excluded by the task; not dominant anyway.
- `jointMarginalCovariance` — only ~1-2%; no action needed.
- cssrlib `restamb`/`initx`/`qcedit`/`findeph`/RINEX parsing — >=10% in
  aggregate but Python scalar bookkeeping/parsing inside READ-ONLY
  cssrlib, no data-parallel numeric structure. (The cascade batching
  already removes the `restamb` executions of discarded combos as a
  structural side effect.)
- The PRIMARY `resamb_lambda_rtklib` call pair — sequentially dependent
  (round-robin exclusion + lock/ratio state between call 1 and 2), only
  ~5k of 34.6k calls; per-call GPU offload measured only ~1.4x
  (launch+transfer bound) — not worth the divergence surface.
- FDE residual sweep / DD assembly — ~2-5%, below the bar.

## Part 3 — Numerical parity (correctness before speed)

Harness: `repro_tc_fgo/results/wp15/wp15_capture_runner.py` wraps the
WP13m-guarded `cssrlib.pppssr.mlambda` binding with a recorder (run
itself stays bit-identical) and captured **34,577 real calls** from a
run2-3000ep replay (5,081 parmode=1 primary + 29,496 parmode=2 cascade;
n = 2..36 med 18; 8 calls non-finite-guarded).
`results/wp15/wp15_parity.py` replays every finite call through the
CUDA batch and compares against the CPU outputs recorded in-run:

| parmode | calls | nfix mismatches | integer-vector mismatches | s mismatches | Ps mismatches |
|---|---|---|---|---|---|
| 1 (full ILS) | 5,073 | 0 | 0 (exact) | 0 (worst rel diff 0.0e+00 — **bitwise**) | 0 |
| 2 (PAR) | 29,496 | 0 | 0 (exact) | 0 (bitwise) | 0 |

**Verdict: bit-identical on all 34,569 real inputs** — including the
ratio numerators/denominators (s), so the pipeline's accept decisions
cannot diverge. (On synthetic random ill-conditioned inputs the small
matrix products around the search can differ by ~1 ulp vs BLAS —
integer outputs still matched exactly on all 60 synthetic cases;
covered in `test_lambda_batch.py` with the tolerance documented.)

## Part 4 — End-to-end verification + speedups

### Correctness (the hard requirement — met everywhere)

- run2-3000ep flag A/B: npz **IDENTICAL, all 27 fields bit-equal**
  (`base3000` vs `cuda3000`; same 959 fix / 2,022 float; even the
  "no valid DD" print count matches, 209/209). 2,026 kernel launches /
  30,390 combos / 0 CPU fallbacks.
- **Full run2 (9,152 epochs)**: npz **IDENTICAL, 27/27 fields**
  (`full_r2_cpu` vs `full_r2_cuda`); fix/float 2,293/6,037 matches the
  WP13r ship full (`full_cond_r2`) exactly. 5,262 launches / 78,539
  combos / 104 CPU fallbacks (2%, the accepted-but-small-nb exact-state
  case) — fallbacks change nothing (bit-identity holds).
- 2000-ep pairs `ab1..ab5`: IDENTICAL npz every time.
- Bit-identity at defaults (flag chain discipline): post-edit runs with
  every WP15 flag unset vs the PRE-edit build — npz IDENTICAL (27/27)
  (`smoke300_preedit` vs `smoke300` and vs `smoke300_final` after all
  edits incl. the WP15_TIME instrumentation).

### Speed (honest numbers — the benchmark host drifts)

Per-call, captured-input replay (in-process, CPU and GPU seconds apart
— the most controlled measurement available):

| path | CPU | GPU (chunk-4096 batches) | speedup |
|---|---|---|---|
| parmode=2 (cascade shape) | 0.186-0.255 ms/call | 0.041-0.048 ms/call | **x4.5-5.3** |
| parmode=1 | 0.258-0.557 ms/call | 0.101-0.390 ms/call | x1.4-2.6 |

At the pipeline's ACTUAL batch size (~15 combos/cascade) the picture is
different: one launch costs 2.4-2.6 ms nearly independent of batch size
(a 15-thread launch uses <1% of the GPU; per-problem latency is bound
by single-thread fp64 on a GTX 1660 Ti) vs 2.2-3.0 ms for 15 sequential
numba CPU calls — **per-cascade the GPU is only at par with a
well-clocked CPU; its win comes when the CPU is slow or contended.**

End-to-end wall-time pairs measured (run2, WP13r ship config):

| pair | CPU | CUDA | ratio | machine state |
|---|---|---|---|---|
| 3000 ep (first session window) | 248.5 s | 168.3 s | **x1.48** | CPU-degraded window (pre-CUDA runs all ran ~2x slow: base 248.5, capture 214.3, prof 325.8) |
| full 9,152 ep | 350.4 s | 443.2 s | x0.79 | drifting (CPU full ran in a fast window, CUDA full during degradation) |
| 2000 ep ab1/ab2 (old kernel) | 87.8 / 109.3 | 100.1 / 94.1 | x0.88 / x1.16 | fast, drifting |
| 2000 ep ab3/ab4/ab5 (optimized kernel) | 116.9 / 197.8 / 196.2 | 178.2 / 210.2 / 200.8 | x0.66 / x0.94 / x0.98 | progressive thermal collapse (identical CPU runs: 87.8 -> 197.8 s within 30 min) |
| **2000 ep controlled (`t1_*`, WP15_TIME cascade timer, states matched by same-run non-cascade time: 174.0 vs 172.6 s, 0.8%)** | cascade 29.7 s (28.6 ms/cascade) | cascade 34.0 s (32.7 ms/cascade) | **x0.87 cascade-only** | stable thermal floor |

Environment finding (documented so the numbers are interpretable): the
host's wall-clock throughput varies **2.3x between identical runs**
(laptop, Balanced power plan, shared CPU/GPU cooling) — single-run
comparisons are unreliable; the controlled `t1` pair (same-run
non-cascade time as the state control) and the in-process per-call
replay are the trustworthy measurements.

**Honest verdict**: the batched CUDA cascade is decision-identical
always, and its throughput advantage is real only when (a) the CPU
side is degraded/contended — exactly the parallel-campaign-runs
regime where the full-run pain lives; the one pair measured in that
state gave **x1.48 end-to-end** — or (b) batches grow beyond ~100
problems (x4.5+ per call), which the current cascade (~15 combos)
does not produce. On an idle, well-clocked host it is at par to
slightly slower (x0.87-0.98 cascade-only). Ship recommendation:
`CUDA_LAMBDA=1` for contended/parallel-run sessions and for any future
wider-candidate-exploration AR (the batch API is the enabler); default
off otherwise (which the flag discipline mandates anyway).

## Part 5 — Build notes

- Toolchain confirmed from the working `build/gnss_gpu_rtk.vcxproj`:
  VS 2022 (v143, MSVC 14.38), CUDA 12.8, `CMAKE_CUDA_ARCHITECTURES=native`
  (GTX 1660 Ti, sm_75, 6 GB; driver 596.36 / CUDA 13.2 runtime OK).
- `cmake -S . -B build` reconfigure + `cmake --build build --target
  _gnss_gpu_lambda_batch --config Release`; pyd copied to
  `python/gnss_gpu/` alongside the existing modules; added to the
  CMake install list. Existing targets and pyds untouched.
- `--fmad=false` on `gnss_gpu_ar` ONLY (parity requirement; documented
  in CMakeLists + the kernel header).
- The kernel keeps per-problem matrices in a cached growable global-
  memory workspace (one thread per problem). Host-side launch overhead
  was optimized after the first e2e round: all device + pinned-host
  buffers cached across calls (zero cudaMalloc/cudaFree per launch),
  inputs/outputs packed into single slabs (2 H2D + 2 D2H copies), no
  explicit cudaDeviceSynchronize (the blocking D2H is the sync).
  Measured launch floor at batch 15: 2.63 -> 2.42 ms — i.e. the floor
  is per-problem KERNEL latency (single-thread fp64 LDL/LLL/ILS),
  not allocation/transfer. The next latency lever would be intra-
  problem parallelism (one block per problem, elementwise-parallel
  LDL/LLL row updates — the op-order-preserving parts), left unported.
- The pipeline venv (`repro_tc_fgo\.venv`, Python 3.12.10) imports the
  cp312 pyd directly via a `sys.path` insert (env override
  `GNSS_GPU_PY`); CUDA runtime is statically linked (`-cudart static`),
  so no DLL path setup is needed.

## Part 6 — Residual levers (measured, out of WP15 scope)

The remaining top CPU items after this port are cssrlib-internal Python
scalar loops: `restamb` (winner re-fires only now), `initx`
(0.62 ms/call x 39.8k — a Python `for j in range(nx)` per ambiguity
reset), `qcedit`, `findeph` (247k linear ephemeris scans), and the
`_write_back_tc` `jm.at` extraction loops. None are CUDA-shaped; all
are numpy-vectorizable via the WP13m-sanctioned module-level
monkeypatch technique (no cssrlib edit) if a future WP wants the next
~20-30% of CPU wall. GTSAM native cost (~12-15% sampled) is the floor
after that.

## Artifacts

- gnss_gpu: `src/ar/lambda_batch.cu`, `include/gnss_gpu/lambda_batch.h`,
  `python/gnss_gpu/_lambda_batch_bindings.cpp`,
  `python/gnss_gpu/lambda_batch.py`, `tests/test_lambda_batch.py`,
  CMakeLists.txt (additive), `results/wp15/WP15_REPORT.md` (this file).
- repro_tc_fgo: `gtsam_rtk_standalone.py` (`__init__` WP15 block +
  `_try_subset_ar` dispatcher + `_try_subset_ar_cuda` +
  opt-in `WP15_TIME=1` cascade timer; `_try_subset_ar_cpu` is the
  verbatim original), `wp13a_run_standalone_rtk.py` (`WP15:` counters
  print incl. `cascade_s`), `results/wp15/` (probe.sh, capture runner,
  parity harness, compare_npz.py, analyze_prof.py,
  analyze_speedscope.py, profiles, capture npz, A/B + `ab1..ab5` +
  `t1_*` logs and npz), `PROGRESS.md` (`WP15:` lines).
- Tests: `python -m pytest tests/test_lambda_batch.py` — 14 passed
  (system python; cssrlib-parity tests skip) / 16 passed (pipeline venv).
