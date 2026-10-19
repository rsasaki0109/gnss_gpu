# D224: Certified FP64 division sharing component probes

These experimental DLLs are independent of the production particle filter.
All three candidates were rejected by their component performance gate. Neither
DLL is linked into the PF, and no whole-PF improvement is claimed.

The first source, `probe.cu` in this directory, tests two certificates for three
divisions sharing a positive denominator. Both start from one correctly rounded
reciprocal and a rounded product. Directed residual endpoints, or adjacent
endpoints around one RN residual, enclose the exact residual. Outward correction
products then enclose `(a - q*b)/b`. If adding both correction endpoints to `q`
with RN produces identical bits, monotonic rounding certifies that result.
All other cases use the original `__ddiv_rn`.

The separate [rounding-cell source](../rounding_cell_division/probe.cu) tests a
third certificate. It compensates the initial product with an RN FMA, then
computes half the smaller adjacent spacing of the candidate quotient. This
radius is a power of two; at powers of two it uses the smaller spacing. For
the guarded denominator and quotient ranges, multiplying that radius by the
denominator is exact, finite and normal. If the magnitude of the RN FMA residual
is strictly below this representable threshold, monotonicity implies the exact
residual is also below it. The exact quotient lies strictly inside the
candidate's RN rounding cell. Ties, unsupported ranges and failed certificates
retain the original division. Its mode 2 repeats the first source's directed
interval algorithm as a slower control; it is not a fourth unique method.

These certificates are our derivation. NVIDIA's [CUDA 12.8 Double Precision
Intrinsics documentation](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__DOUBLE.html)
specifies the primitive rounding contracts. No approximate answer is accepted
without a certificate. Signed zeros, exceptional denominators, NaNs and
infinities retain the original division path.

Each experiment compares 1,188,864 directly returned single-step quotient bit
patterns per variant, plus 393,216 twelve-step accumulated outputs. The latter
execute 4,718,592 internal divisions per variant but do not dump each internal
quotient. Each experiment independently checks 42,785 finite quotients against
Python exact rational arithmetic. Validation inputs and seed are the same in
both experiments. Timing compares every returned output hash, with validation
counters disabled. CUDA events cover one kernel for three quotient outputs per
row, or twelve repetitions with one RN accumulation per axis in each repetition. Setup, allocation,
transfers, hashes and the independent reference are outside event timing.

The gate was fixed before each experiment: both 65,536-row and 1,000,000-row
twelve-step conditions must improve at least 10%, with at least four of five
rounds improving; all A/A condition medians must be within 5%; outputs must match
exactly. Passing would only justify a separate whole-PF experiment. No candidate
passed. Preserve all rounds, including the noisy A/A rounds in the second
experiment, and do not retry a frozen candidate to obtain a passing result.

## Reproduction

From the repository root, use an x64 MSVC/CUDA 12.8 build environment, CMake and
the repository's Python dependencies. Use fresh output paths. These commands
build only the first standalone DLL and create the minimal build record consumed
by the comparison helper; they do not build or install the PF extension.

```powershell
$taskProbe = 'certified_division'
$taskBuild = 'dist/repro_certified_division'
cmake -S "benchmarks/probes/$taskProbe" -B $taskBuild -G 'Visual Studio 17 2022' -A x64 -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build $taskBuild --config Release --parallel 2
@'
from pathlib import Path
import hashlib
import json
import sys
root = Path.cwd()
probe, build = sys.argv[1:]
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
source = root / 'benchmarks/probes' / probe
record = root / build / 'probe_build_record.json'
assert not record.exists()
data = dict(
    source_sha256={p.relative_to(root).as_posix(): sha(p)
                   for p in (source / 'CMakeLists.txt', source / 'probe.cu')},
    dll_sha256=sha(root / build / 'Release/certified_division_probe.dll'))
record.write_text(json.dumps(data, indent=2) + '\n', encoding='utf8')
'@ | python - $taskProbe $taskBuild
python benchmarks/bench_pf_certified_division.py --dll "$taskBuild/Release/certified_division_probe.dll" --build-record "$taskBuild/probe_build_record.json" --output dist/repro_certified_division.json --rounds 5 --repeats 5
```

For the second experiment, set `taskProbe` to `rounding_cell_division`, choose a
different fresh build/output path, and use `bench_pf_rounding_cell.py` as the
helper. Run GPU jobs serially. The helpers pin only their own process to its last
allowed CPU at normal priority; the host and GPU are nonexclusive. Each sample
has one warm launch and three timed launches. Five rotating rounds of five
baseline/candidate/candidate/identical-baseline groups provide paired
percentages; the median within each round and then the median of five rounds is
the primary estimator. Overall absolute medians are separate diagnostics and
must not be used to reconstruct the paired estimator.

See [D224 results](../../RESULTS.md) and the [rejection report](../../pf_certified_division_rejected_20261016.json)
for the original source/DLL hashes, all measurements, register counts and limits.
