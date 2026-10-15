# RB-FGO-PF runtime (frozen copy)

This directory holds the runtime behind the RB-FGO-PF Tokyo PPC results in the
top-level README and `internal_docs/inuex35_tc_fgo_benchmark.md`. The files
are copied verbatim from the private research workspace `repro_tc_fgo` at
commit `fa57b7f` (2026-10-06), and WP43 (2026-10-10) and WP44–WP45
(2026-10-11/12) added default-off options to them (see below). They have not been refactored, so the code
still carries its campaign history (WP numbers, environment-variable
switches). `wp13m_lambda_diag.py` was never committed there; this is its first
archived copy.

| file | role |
|---|---|
| `wp16_run_rbpf.py` | runner: decodes one run, drives the filter, writes `.pos` / `.npz` |
| `rbpf_fgo.py` | Rao-Blackwellized PF over integer-ambiguity basins |
| `gtsam_rtk_standalone.py` | fixed-lag GNSS/IMU factor-graph backbone (shared float filter) |
| `wp13a_run_standalone_rtk.py` | data loading, reference and `.pos` helpers |
| `wp13m_lambda_diag.py` | finite-guarded MLAMBDA used by the backbone |
| `relabel_nb9_pos.py` | report floor: relabels fixes with fewer than `nb` resolved ambiguities as float |

## Reusing component DDs in feedback (2026-10-16)

`step` passes its component-agreement DD arrays to `_feedback` while the
particle hypotheses remain unchanged. Feedback reuses those arrays for its
held-integer mass decisions. Its optional trailing `dd_all=None` argument
preserves legacy calls and empty-MAP fallback: missing arrays are computed
at the original point in feedback. No DD cache persists across maintenance,
particle changes, resampling or epochs. Release/commit policy is unchanged.

Actual Tokyo run1 inputs provide **24 feedback calls, 1,536 particle visits
and 5 graph events**, N=64, K=12, seed 20260710, from the 600-epoch prefix.
The observer checks that already available component DD bytes equal every
DD recomputed by the original feedback and that H stays unchanged. Feedback
DD calls fall **1,536 → 0** in these recorded cases. Original call/particle/
pair order and count are retained, without synthetic or expanded hypotheses.

| Wall-time replay scope | Baseline median ms | Candidate median ms | Paired reduction median | Paired range | Improved pairs |
|---|---:|---:|---:|---:|---:|
| Feedback policy with recorded graph transitions | 87.78 | 69.33 | **17.61%** | 13.20–27.62% | **10/10** |

**This is policy replay, not live GTSAM graph/feedback latency.** Real graph
methods run unchanged in the observer and pipeline controls. Replay replaces
them with recorded transitions of the graph fields read by feedback, with
transition playback included in the clocks. Before timing, graph call
arguments/order and before/after states are checked against the recording.
Fresh state restoration, existing component DD preparation, imports, loading,
decode and output validation stay outside the clocks. The original component
calculation is unchanged. No full-filter, full-route or GPU speed gain is
established. Percentages are medians of paired reductions, distinct from
ratios of per-policy medians; do not combine them with earlier results.

Each policy retains actual epoch/particle order. Policies run adjacent
within epoch windows, alternating first policy with trial and epoch. Two
full warmups per source are excluded. Windows, Python 3.12.10, NumPy 2.4.6,
one OpenBLAS/OMP thread; local jobs/checks finish before timing, with no
overlapping edits, scans, tests or builds.

Policy return, every feedback-owned field and projected graph state match
the observed results. Particle H, weights and supplied DDs stay unchanged.
The policy and step ASTs match the baseline after removing only the optional
DD parameter, recomputation guard and call keyword. The observer matches
the frozen PR #247 control. Candidate TRACK=2 600 epochs and TRACK=1 350
epochs match their controls: position bytes and all 90 arrays, zero
exclusions. These runs exercise the real graph methods and supply correctness
evidence only. The full suite has 6,047 passed / 3,778 skipped; 59 new cases
cover release/commit, carry protection, support/DDPR resets, dual/witness/
commit gates, missing/empty/large DDs, legacy freshness, snapshot types and
in-place committed-dictionary identity during transition playback.

To reproduce with fresh directories, save the baseline from `6a7af5e` and
forward the WP56 flags/environment from `benchmark_shared_moves_cpu.py`:

```bash
python capture_feedback_dd.py --source BASELINE.py --output NEW_CAPTURE \
  --tc-src tc/src -- 1 --max-ep 600 --out-dir NEW_RUN [WP56 flags]
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmark_feedback_dd.py \
  --manifest NEW_CAPTURE/manifest.json --baseline BASELINE.py \
  --candidate rbpf_fgo.py --output NEW_REPLAY.json --pairs 10
```

The observer calls original feedback/graph methods and returns original
values unchanged. Its elapsed time is not speed evidence. Preserve completed
controllers and records. See the [raw trials](../../benchmarks/rbpf_feedback_dd_tokyo_prefix_20261016.json)
and [independent audit](../../benchmarks/rbpf_feedback_dd_audit_20261016.json).

## Bulk preparation of conditional-position rows (2026-10-16)

`_position_rows` prepares FP64 conditional-position inputs with NumPy: keep
code rows and finite held carrier rows in their original order, calculate
carrier residuals/sigma, copy Jacobian rows and assign nuisance groups by
first retained carrier occurrence. Other numeric dtypes retain the original
scalar preparation and promotion rules. The robust solve, information,
covariance, conditional cache and cluster lookup are unchanged. Outputs are
fresh arrays; no hypothesis-dependent data persists across calls.

Tokyo run1's 600-epoch prefix supplies **126 actual LS calls inside 24 sampled
steps**, N=64, K=12, seed 20260710. Raw problems, DDs and resolved cluster
states retain their actual epoch/call order/count. No hypotheses or calls
are synthesized, duplicated or expanded for performance measurement.

| Wall-time replay scope | Baseline median ms | Candidate median ms | Paired reduction median | Paired range | Improved pairs |
|---|---:|---:|---:|---:|---:|
| Position-row preparation, including copies and sqrt(beta) | 31.26 | 10.48 | **66.72%** | 65.13–68.59% | **10/10** |
| LS evaluator with recorded resolved cluster state | 101.65 | 76.76 | **25.14%** | 18.32–29.16% | **10/10** |

The LS scope includes preparation, robust solution, minimum information and
covariance. **Live cluster lookup, conditional cache and its outer wrapper
are excluded.** These results do not establish full-filter, full-route or
GPU speed gains. Percentages are medians of paired reductions, distinct
from ratios of per-policy medians; do not combine them with earlier results.

Each policy retains actual epoch/call order. Policies run adjacent within
epoch windows, alternating the first policy with epoch and trial. Two full
warmups per source/stage are excluded. Imports, loading, decode, validation
and hash checks stay outside the clocks. Windows, Python 3.12.10, NumPy 2.4.6,
one OpenBLAS/OMP thread; local jobs/checks finish before timing, with no
overlapping edits, scans, tests or builds.

All recorded residual/sigma/J/group bytes, group counts, solutions, minimum
information and covariance match both exact source bodies; inputs remain
unchanged. The baseline observer matches the frozen PR #246 control.
Candidate TRACK=2 600 epochs and TRACK=1 350 epochs match their controls:
position bytes and all 90 arrays, zero exclusions. These runs are correctness
evidence only. The full suite has 5,988 passed / 3,778 skipped; 67 new cases
cover FP64/FP32 and mixed dtype promotion, empty/code-only/carrier-only rows,
missing and nonfinite values, group/row order, sigma scaling, recorded prior
variants, robust solves and changed DD/observations on repeated calls.

To reproduce with fresh directories, save the baseline from `97e5eb9` and
forward the WP56 flags/environment from `benchmark_shared_moves_cpu.py`:

```bash
python capture_position_rows.py --source BASELINE.py --output NEW_CAPTURE \
  --tc-src tc/src -- 1 --max-ep 600 --out-dir NEW_RUN [WP56 flags]
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmark_position_rows.py \
  --manifest NEW_CAPTURE/manifest.json --baseline BASELINE.py \
  --candidate rbpf_fgo.py --output NEW_REPLAY.json --pairs 10
```

The observer records actual LS calls and returns original results unchanged.
Its elapsed time is not speed evidence. Preserve completed controllers and
records. See the [raw trials](../../benchmarks/rbpf_position_rows_tokyo_prefix_20261016.json)
and [independent audit](../../benchmarks/rbpf_position_rows_audit_20261016.json).

## Reusing basin IDs during posterior aggregation (2026-10-16)

The posterior computes each particle's canonical basin ID once and reuses it
for mass grouping, first-representative selection and tracked MAP output,
including optional tracking variants. The canonical-ID function is unchanged.
Hypotheses remain read-only between grouping and tracked output; IDs are local
to this step and are recomputed after particle or epoch changes. Scalar mass
accumulation order, stable ties, first representative identity and non-empty
tracking's `bool(H)` behavior are preserved.

Tokyo run1's 600-epoch prefix supplies **24 actual aggregation inputs and
1,536 particle visits**, N=64, K=12, seed 20260710. Captured H, normalized
weights and tracked states retain their original order/count. No hypotheses
or states are synthesized, duplicated or expanded for performance measurement.

| Wall-time replay scope | Baseline median ms | Candidate median ms | Paired reduction median | Paired range | Improved pairs |
|---|---:|---:|---:|---:|---:|
| Basin grouping, ranking and representatives | 83.69 | 52.58 | **35.37%** | 19.73–40.59% | **10/10** |
| Grouping plus tracked-output selection and weighted means | 210.49 | 105.77 | **49.63%** | 37.50–53.48% | **10/10** |

Canonical-ID calls fall **3,072 → 1,536** in the first scope and
**4,608 → 1,536** in the second. Both include ID construction, ordered mass
sums, ranking, first representatives and output construction. The second
also includes MAP/non-empty tracked-output selection and weighted means.
**Conditional top-position solves and cluster-state publication are excluded
from both replay scopes.** Their production statements are unchanged and run
in pipeline controls. Weight normalization is outside the clocks because
normalized weights are captured inputs. These results do not establish full
posterior, full-filter, full-route or GPU speed gains. Percentages are medians
of individual paired reductions, distinct from ratios of per-policy medians;
do not combine them with previous study percentages.

Each policy retains the actual epoch/particle order. Policies run adjacent
within epoch windows, alternating the first policy with trial and epoch.
Two full warmups per source/stage are excluded. Imports, loading, decode,
validation and hash checks stay outside the clocks. Windows, Python 3.12.10,
NumPy 2.4.6, one OpenBLAS/OMP thread; local jobs/checks finish before timing,
with no overlapping edits, scans, tests or builds.

All observed mass/ranking/representative/tracked-output bytes match both
sources. First representative object identity and unchanged input bytes are
also checked. The baseline observer matches the frozen PR #245 control.
Candidate TRACK=2 600 epochs and TRACK=1 350 epochs match their controls:
position bytes and all 90 arrays, zero exclusions. These runs are correctness
evidence only. The full suite has 5,921 passed / 3,778 skipped; 151 new cases
cover all tracking modes, missing states/variants, gauge-equivalent H, empty
groups, stable ties, zero/unequal mass, first representative identity, large
integers and H changes/reordering on repeated calls. CI includes these cases.

To reproduce with fresh directories, save the baseline from `5dab2b1` and
forward the WP56 flags/environment from `benchmark_shared_moves_cpu.py`:

```bash
python capture_posterior_ids.py --source BASELINE.py --output NEW_CAPTURE \
  --tc-src tc/src -- 1 --max-ep 600 --out-dir NEW_RUN [WP56 flags]
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmark_posterior_ids.py \
  --manifest NEW_CAPTURE/manifest.json --baseline BASELINE.py \
  --candidate rbpf_fgo.py --output NEW_REPLAY.json --pairs 10
```

The observer inserts input/output callbacks around unchanged original
posterior statements, including live conditional solves. Its elapsed time is
not speed evidence. Preserve completed controllers and records. See the
[raw trials](../../benchmarks/rbpf_posterior_ids_tokyo_prefix_20261016.json)
and [independent audit](../../benchmarks/rbpf_posterior_ids_audit_20261016.json).

## Reusing DD arrays within component agreement (2026-10-16)

The component-agreement phase reuses its already computed per-particle DD
arrays for the non-empty diagnostic. When the best non-empty basin is the MAP
basin, its representative also reuses `dd_map`. If the MAP hypothesis is
empty, the diagnostic still computes the DDs it needs. Reuse ends with this
phase: particle changes and later epochs always read fresh hypotheses.
Scalar weight sums, integer comparisons, pair order and strict thresholds
are unchanged; the existing arrays remain available to tracked output.

Tokyo run1's 600-epoch prefix supplies **24 actual phases and 1,536 particle
visits**, N=64, K=12, seed 20260710. DD calls fall **3,120 → 1,560**.
Inputs and natural repetitions retain their original order/count, without
synthetic or expanded hypotheses. Both studies are retained:

| Wall-time replay scope | Baseline median ms | Candidate median ms | Paired reduction median | Paired range | Improved pairs |
|---|---:|---:|---:|---:|---:|
| Complete component agreement and non-empty diagnostic | 326.55 | 292.53 | 11.34% | -11.85–23.76% | 7/10 |
| DD materialization, including availability guards | 37.24 | 18.87 | **48.35%** | 45.46–50.29% | **10/10** |

Adoption is supported by reduced DD work and the consistent materialization
gain. **Stable complete-phase speed improvement remains unproven.** The
second study retains the exact source DD assignments, conditional guards,
non-empty index/mass preparation and array construction, checked against
each observed complete phase. It excludes scalar component-agreement sums
and diagnostic thresholds. Basin ranking is outside both scopes. These
measurements do not establish full-filter, full-route or GPU speed gains.
Percentages are medians of individual paired reductions, distinct from the
ratio of per-policy medians; do not combine them with earlier percentages.

Each policy visits the original epochs/particles/pairs in order. Policies
run adjacent within each epoch window, alternating the first policy with
trial and epoch. Two full warmups per source are excluded. Imports, loading,
decode, validation and hash checks stay outside the clocks. Windows, Python
3.12.10, NumPy 2.4.6, one OpenBLAS/OMP thread; local jobs/checks finish before
timing, with no overlapping edits, scans, tests or builds.

Original DD bytes, ordered mass sums, pair decisions and diagnostic outputs
match both sources at all 24 epochs. The baseline observer matches the
frozen PR #244 control. Candidate TRACK=2 600 epochs and TRACK=1 350 epochs
match their controls: position bytes and all 90 arrays, zero exclusions.
These runs are correctness evidence only. The full suite has 5,770 passed /
3,778 skipped. All 88 focused cases also pass with the final materialization
projection, covering empty MAP fallback, empty groups, missing support,
zero mass, pair permutations, strict thresholds, large integers and fresh
H/pair changes on repeated calls. CI includes these cases.

To reproduce with fresh directories, save the baseline from `fafc91c` and
forward the WP56 flags/environment from `benchmark_shared_moves_cpu.py`:

```bash
python capture_agreement_dd.py --source BASELINE.py --output NEW_CAPTURE \
  --tc-src tc/src -- 1 --max-ep 600 --out-dir NEW_RUN [WP56 flags]
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmark_agreement_dd.py \
  --manifest NEW_CAPTURE/manifest.json --baseline BASELINE.py \
  --candidate rbpf_fgo.py --output NEW_FULL.json --pairs 10
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmark_agreement_dd_materialization.py \
  --manifest NEW_CAPTURE/manifest.json --baseline BASELINE.py \
  --candidate rbpf_fgo.py --full-replay NEW_FULL.json --output NEW_DD.json --pairs 10
```

The observer inserts input/output callbacks around the unchanged original
phase statements. Its elapsed time is not speed evidence. Preserve completed
controllers and records. See the [complete-phase raw trials](../../benchmarks/rbpf_agreement_dd_tokyo_prefix_20261015.json),
[complete-phase audit](../../benchmarks/rbpf_agreement_dd_audit_20261015.json),
[DD materialization raw trials](../../benchmarks/rbpf_agreement_dd_materialization_20261015.json)
and [materialization audit](../../benchmarks/rbpf_agreement_dd_materialization_audit_20261015.json).

## Bulk preparation of scalar likelihood rows (2026-10-15)

`_likelihood_rows` prepares held carrier residuals and sigma scaling together
with NumPy for the default uniform-lattice model. Unheld lattice terms still
accumulate with the original scalar `math.log` calls in original row order.
The nonuniform model keeps the original scalar calculation. The likelihood
cache, current DD generation, nuisance-group ordering, robust solve and mixture
evaluation are unchanged. Outputs are fresh arrays, without persistent H data.

Tokyo run1's 600-epoch prefix supplies **657 actual scalar cache misses inside
24 sampled steps**, with N=64, K=12 and seed 20260710. Cache hits are excluded;
no hypotheses are synthesized, duplicated or expanded for measurement.

| Wall-time replay scope | Baseline median ms | Candidate median ms | Paired reduction median | Paired range | Improved pairs |
|---|---:|---:|---:|---:|---:|
| Row preparation, including output copies | 46.39 | 10.00 | **78.93%** | 76.75–80.03% | 10/10 |
| Miss evaluator with recorded resolved cluster state | 307.67 | 255.53 | **16.35%** | 13.18–25.28% | 10/10 |

The second scope includes canonical ID construction, current DD generation,
row preparation, the robust matrix solve and mixture evaluation. Each call
uses a fresh cache and an already resolved recorded cluster state. **Live
cluster-state lookup is excluded.** These figures do not establish live
weight-stage, full-filter, full-route or GPU latency gains. Percentages are
medians of individual paired percentages, distinct from the ratio of the
per-policy time medians; do not combine them with earlier study percentages.

Policies retain the original epoch/call order, run adjacent within epoch
windows and alternate which goes first with epoch and trial. Two full warmups
per source/stage are excluded. Imports, capture, decode, validation and hash
checks stay outside the clocks. Conditions: Windows, Python 3.12.10, NumPy
2.4.6, one OpenBLAS/OMP thread; all local jobs/checks finish before timing.
No local edits, scans, tests or builds overlap the clocks.

Every captured residual/scale/mask byte, ordered lattice sum and original
likelihood value matches both exact source bodies. The original observer
TRACK=2 run matches the frozen PR #243 control, and fresh candidate TRACK=2
600 epochs and TRACK=1 350 epochs match their respective controls: position
bytes and all 90 arrays, zero exclusions. These runs supply correctness
evidence only. The final suite has 5,682 passed / 3,778 skipped. The 129 new
cases cover float32/64, partial/empty/full support, row permutations,
nonfinite values, sigma scaling, lattice floors, both models, recorded
position priors and cache hits.

To reproduce with fresh directories, save the baseline from `6bbd63b` and
forward the WP56 flags/environment from `benchmark_shared_moves_cpu.py`:

```bash
python capture_scalar_rows.py --source BASELINE.py --output NEW_CAPTURE \
  --tc-src tc/src -- 1 --max-ep 600 --out-dir NEW_RUN [WP56 flags]
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmark_scalar_rows.py \
  --manifest NEW_CAPTURE/manifest.json --baseline BASELINE.py \
  --candidate rbpf_fgo.py --output NEW_REPLAY.json --pairs 10
```

The observer records actual scalar misses and raw problem/cluster data,
returning the original values unchanged. Its elapsed time is not speed
evidence. Preserve completed controllers/records. See [all raw trials](../../benchmarks/rbpf_scalar_rows_tokyo_prefix_20261015.json)
and the [independent audit](../../benchmarks/rbpf_scalar_rows_audit_20261015.json).

## Preparing integer-DD row layouts once per epoch (2026-10-15)

`EpochProblem.set_pairs` freezes pair triples and prepares contiguous
group/reference runs once when normal or wide-lane problems are extracted.
`_pair_dd` reads the group's reference once per run and then visits its target
rows. The layout contains row indices and satellite identifiers, without any
hypothesis values or reusable DD arrays. Every call reads the current H.
The prepared path requires the same immutable pair tuple. Replacing `pairs`
directly uses the original scalar fallback; preparing again creates a fresh
layout. Legacy unprepared problems retain scalar behavior.

Actual Tokyo run1 inputs provide 24 sampled epochs, N=64, K=12, seed 20260710,
and 1,536 original structural particle visits. Per-DD-replay group lookups
fall **63,104 → 12,288**, and satellite dictionary reads **124,016 → 73,544**.
No hypotheses are synthesized or expanded. Two approaches were measured;
both complete studies are retained:

| Study / wall metric | Baseline median ms | Candidate median ms | Paired reduction median | Improved pairs |
|---|---:|---:|---:|---:|
| Initial per-row branch reuse / DD | 74.40 | 71.87 | 3.22% | 6/10 |
| Initial per-row branch reuse / paired moves | 197.56 | 207.10 | -2.66% | 2/10 |
| Prepared epoch layout / DD, preparation included | 30.14 | 25.98 | **12.90%** | **10/10** |
| Prepared epoch layout / paired moves, preparation included | 309.41 | 297.78 | 4.87% | 7/10 |

The initial approach is deferred. The prepared layout is adopted for the
consistent **DD-only** improvement (paired range **9.63–25.47%**). Paired
moves vary from -11.41 to 11.75%, so their improvement remains unproven.
Percentages are medians of individual paired percentages, distinct from the
ratio of per-policy time medians. Absolute timings from separate studies
must not be compared or their speedups combined. Full-filter, full-route
and GPU latency gains are not established.

Each policy retains the recorded epoch/particle order. Policies run adjacent
within each epoch window, alternating which goes first with epoch and trial.
Two full warmups per source/stage are excluded. Candidate layout freezing
and construction are charged once per timed epoch window in both stages.
Output construction is included; imports, decode and validation are excluded.
Windows, Python 3.12.10, NumPy 2.4.6, one OpenBLAS/OMP thread; no other local
jobs or edits overlap timing. Process CPU observations are retained as
secondary data, with coarse quantization and possible zero windows; adoption
uses DD wall observations. Final pipeline/full-suite checks follow the final
study on the same source bytes.

Every recorded DD byte (including NaNs), structural output and dictionary
order matches the baseline. TRACK=2's 600 epochs and TRACK=1's 350 epochs
match their frozen same-source controls: position bytes and all 90 arrays,
zero exclusions. These runs supply correctness evidence only. The 101
regression cases cover prepared and legacy problems, row/reference changes,
missing support, nonfinite arithmetic, mutation of H, independent outputs,
input-row freezing and safe fallback after direct pair replacement.

Use new output paths and the existing real structural capture schema. Save
the baseline from `511aa35` and use WP56 flags from
`benchmark_shared_moves_cpu.py` when recording fresh actual visits. Replay:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmark_dd_layout.py \
  --manifest NEW_CAPTURE/manifest.json --baseline BASELINE.py \
  --candidate rbpf_fgo.py --output NEW_REPLAY.json --pairs 10
```

The initial candidate is retained as `dd_reference_candidate.patch` against
that baseline. Apply it to a separate full runtime copy and replay using
`benchmark_dd_lookups.py` with the same arguments. Completed controllers and
records remain frozen. Records: [initial study](../../benchmarks/rbpf_dd_lookup_probe_tokyo_prefix_20261015.json),
[prepared layout study](../../benchmarks/rbpf_dd_layout_tokyo_prefix_20261015.json),
[independent audit](../../benchmarks/rbpf_dd_layout_audit_20261015.json).

## Canonical basin identity candidate: adoption deferred (2026-10-15)

The experimental `basin_id_candidate.py` sorts satellite numbers once per
nonempty group. The first
number supplies the gauge anchor, and the already sorted numbers determine
the output order. This replaces a separate minimum scan and sorting allocated
satellite/difference tuples. Empty groups, singleton groups, integer values,
group order and the hashable identity remain unchanged. There is no persistent
cache: a mutated hypothesis always produces a fresh identity. **The live
runtime retains its original canonicalization.**

Tokyo run1's 600-epoch prefix supplies **11,434 actual calls inside 24
sampled steps**, with the original N=64, K=12 and seed 20260710. Naturally
repeated hypotheses retain their original count/order; no hypotheses are
synthesized or expanded. Two warmups per source precede 10 alternating pairs.
Conditions: Windows, Python 3.12.10, NumPy 2.4.6, one OpenBLAS/OMP thread;
other local correctness jobs/checks finish before either timed study.

| Study / metric | Baseline median ms | Candidate median ms | Paired reduction median | Paired range | Improved pairs |
|---|---:|---:|---:|---:|---:|
| First / wall | 503.16 | 452.02 | 35.43% | -73.93–49.09% | 6/10 |
| Additional / process CPU | 351.56 | 265.63 | 15.24% | -61.90–68.57% | 6/10 |
| Additional / wall | 359.86 | 268.10 | 24.65% | -40.91–66.95% | 6/10 |

Paired percentages are computed separately from the per-policy time medians.
Both studies have substantial regressions in four pairs, so the positive
medians do **not** establish a stable speed improvement. Process CPU time
excludes time when this process is not scheduled; observed CPU times have
15.625 ms quantization despite the clock's nominal resolution. Both clocks
and all raw trials remain available. No cause is assigned to the variability.
There is no full-filter, full-route or GPU speed claim.

All recorded identities match both functions. The observed baseline 600-epoch
run matches the frozen PR #241 control, and candidate TRACK=2 600 epochs and
TRACK=1 350 epochs match their respective controls: position bytes and all 90
arrays are identical, with zero exclusions. Correctness-run timings are not
speed evidence. The candidate function retained here has exactly the same AST
as the function measured in the archived full candidate runtime. Production
`rbpf_fgo.py` is restored byte-for-byte to the baseline. The 43 regression cases
cover insertion order, empty/singleton groups, independent gauge shifts,
large and NumPy integers, hash/type preservation and mutation of the same H.

Records: [first wall study](../../benchmarks/rbpf_basin_id_tokyo_prefix_20261015.json),
[additional CPU/wall study](../../benchmarks/rbpf_basin_id_cpu_tokyo_prefix_20261015.json),
and [independent audit and adoption decision](../../benchmarks/rbpf_basin_id_audit_20261015.json).

To reproduce on new output paths, save the baseline runtime from commit
`98f0a89`, use the WP56 flags/environment in `benchmark_shared_moves_cpu.py`
(N=64, K=12, seed 20260710), and forward runner arguments after `--`:

```bash
python capture_basin_ids.py --source BASELINE.py --output NEW_CAPTURE \
  --tc-src tc/src -- 1 --max-ep 600 --out-dir NEW_RUN [WP56 flags]
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmark_basin_id.py \
  --manifest NEW_CAPTURE/manifest.json --baseline BASELINE.py \
  --candidate basin_id_candidate.py --output NEW_REPLAY.json --pairs 10
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmark_basin_id_cpu.py \
  --manifest NEW_CAPTURE/manifest.json --baseline BASELINE.py \
  --candidate basin_id_candidate.py --output NEW_CPU_REPLAY.json --pairs 10
```

The observer records all canonical calls inside sampled filter steps and
returns original identities unchanged. Replay verifies source/input hashes,
every identity and input immutability before timing exact function bodies.
Loading, imports, decoding and validation stay outside the clocks; identity
construction and the resulting tuples are included. Preserve completed
controllers and records, and use fresh directories for later studies.

## Reusing DDs within a structural move pair (2026-10-15)

`_move_pair` builds the current particle's integer double differences once
and passes that array to extension and release. Both helpers read it and
produce separate hypothesis copies. The array lives only within this call;
maintenance, changes to the hypothesis and later epochs always construct
fresh DDs. Individual two-argument helper calls still compute their own DDs.
The scalar and tracked proposal paths retain move order, priors and RNG calls.

On original Tokyo run1 inputs, 24 sampled epochs with 64 particles give
1,536 particle visits and 3,072 structural calls. DD construction drops from
3,072 to 1,536 calls. The paired structural stage has **19.72% median paired
reduction**, improving all 10 alternating pairs (range 8.43–45.17%). Its
per-policy time medians are **203.14 → 160.84 ms**; the percentage is the
median of the individual paired percentages. The baseline is PR #240's
vectorized helpers, and these figures are not combined with earlier studies.

Timing uses the exact method bodies, the original epoch/particle order and
both output constructions. Two warmups per source, imports, decoding,
output validation and DD counters are excluded. The counters are removed
before timing. Conditions: Windows, Python 3.12.10, NumPy 2.4.6, one
OpenBLAS/OMP thread; other local checks finish before timing. Hypotheses are
recorded from actual calls with no synthesized or expanded workload. This
measures the structural stage, without establishing full-filter, full-route,
ensemble or GPU latency gains.

All captured integer outputs and dictionary insertion orders match the
separate baseline helpers. TRACK=2's 600-epoch run matches the frozen PR #240
control, and fresh TRACK=1 baseline/candidate 350-epoch runs match each other:
position bytes and all 90 arrays are identical in both comparisons, with no
exclusions. These are correctness runs; their elapsed times are not speed
evidence. The 28 new regression cases check one DD allocation, read-only
use, independent output ownership, fresh computation after hypothesis/epoch
changes, recovery after an exception and legacy direct helper calls.

For current runtimes, `capture_move_pair_inputs.py` observes `_move_pair`;
it also supports a legacy source that calls the two helpers separately.
Use the WP56 flags/environment from `benchmark_shared_moves_cpu.py`, N=64,
K=12 and seed 20260710, forwarding runner arguments after `--`:

```bash
python capture_move_pair_inputs.py --source rbpf_fgo.py --output NEW_CAPTURE \
  --tc-src tc/src -- 1 --max-ep 600 --out-dir NEW_RUN [WP56 flags]
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmark_move_pair.py \
  --manifest NEW_CAPTURE/manifest.json --baseline PR240_BASELINE.py \
  --candidate rbpf_fgo.py --output NEW_REPLAY.json --pairs 10
```

Save the baseline from commit `53cd866` before applying this change. Use new
output paths; replay verifies hashes, adjacent call pairing, every integer
output, input immutability and DD call counts before timing. Complete
[timings](../../benchmarks/rbpf_move_pair_tokyo_prefix_20261015.json) and
[independent audit](../../benchmarks/rbpf_move_pair_audit_20261015.json).

## Vectorized move selection (2026-10-15)

Extension computes the held-pair mask once, returns early on empty/full
support, and checks group references only for missing pairs. Release scores
the held carrier rows with NumPy and selects the first maximum. NaNs are
skipped, equal scores (including infinity) retain the first observation, and
the sigma floor stays double precision even with float32 sigma inputs.
The conditional solve, integer rounding and copied hypothesis order are
unchanged. Both the scalar and tracked proposal paths use these helpers.

On actual Tokyo run1 inputs, the combined helpers took **383.46 → 215.48 ms**
over 3,072 recorded calls: **44.14% median paired reduction**, improving all
10 alternating pairs (range 38.21–48.58%). This is the sum of 24 sampled
epochs' helper calls, with 64 original particles per epoch: 1,536 extension
and 1,536 release calls. Of the extension calls, 1,301 produce a hypothesis
and 235 return `None`. Problems contain 24–54 pairs and 48–108 measurement
rows. Naturally repeated particle hypotheses are kept at their original
count; no workload is synthesized or expanded.

The replay runs the exact original/candidate method bodies without GTSAM,
including hypothesis copies and output construction. Two full replay
warmups per source are excluded. Input loading/decoding, imports, parity and
hash checks are outside the clock. Conditions: Windows, Python 3.12.10,
NumPy 2.4.6, one OpenBLAS/OMP thread; no local edits, tests, builds or scans
during timing. This measures the two helpers, and establishes no whole-filter,
full-route, ensemble or GPU speedup.

Every recorded output matches the original integer values and dictionary
insertion order exactly. The original observer run and a fresh candidate
600-epoch run also match the previous frozen control's position bytes,
all 88 diagnostic arrays and both row-dump arrays, with no exclusions.
Correctness runs may overlap other checks and their elapsed times are not
used for performance claims. The 61 selection regression cases cover empty
and partial support, singular covariance, ties-to-even rounding, row ties,
duplicate carrier pairs, nonfinite values and the float32 sigma-floor edge.

For a new capture, save the original runtime as `BASELINE.py`, then use
`capture_move_inputs.py --source BASELINE.py --output NEW_CAPTURE --tc-src
tc/src -- 1 --max-ep 600 --out-dir NEW_RUN [WP56 flags]`. Use the WP56 flags
and environment listed in `benchmark_shared_moves_cpu.py` (N=64, K=12,
seed=20260710). The capture observer returns the original results. Replay:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmark_move_selection.py \
  --manifest NEW_CAPTURE/manifest.json --baseline BASELINE.py \
  --candidate rbpf_fgo.py --output NEW_REPLAY.json --pairs 10
```

Use a fresh output path each time. Sources and captured problem arrays are
hash checked; every output is checked before timing. Complete records:
[all trials](../../benchmarks/rbpf_move_selection_tokyo_prefix_20261015.json)
and [independent parity/measurement audit](../../benchmarks/rbpf_move_selection_audit_20261015.json).

## Shared proposal scores (2026-10-15)

The scalar proposal path (`RBPF_TRACK=0/2`) scores each shared consensus,
LAMBDA and spawn move once per epoch instead of once per particle. With
64 particles, each shared move needs one score instead of 64. Release uses
the already computed empty-hypothesis score. Move order, prior arithmetic,
particle copies and RNG calls are preserved; tracked proposals (`TRACK=1`)
keep their particle-specific priors.

Two separate studies used the original Tokyo run1 RINEX/IMU/reference files,
600 epochs, 64 particles, K=12, seed 20260710 and WP56 flags. Each study used
one full warmup per source, then three alternating pairs in fresh processes,
with CPU thread counts set to one. No edits or other local checks ran during
timing. Results are mixed:

| Study | Full-run CPU time reduction | PF elapsed reduction | Full-run elapsed reduction | Improving pairs |
|---|---:|---:|---:|---:|
| [Initial](../../benchmarks/rbpf_shared_moves_tokyo_prefix_20261015.json) | not recorded | -4.47% | -12.67% | 1/3 |
| [Additional CPU-time study](../../benchmarks/rbpf_shared_moves_cpu_tokyo_prefix_20261015.json) | 12.94% | 22.03% | 12.98% | 3/3 |

Values are medians of paired percentage reductions, with negative values
meaning slower. Initial PF results range from -8.38% to +11.52%; additional
CPU results range from +6.19% to +22.44%. The first study triggered the
additional CPU measurement; both full records include all trials and warmups.
This adopts the redundant-work removal, with no stable end-to-end speedup
claim across the two studies. It does not establish full-route or ensemble
latency, or a change in positioning accuracy.

Across all 16 runs the position bytes, all 88 diagnostic arrays and both row
dump arrays match exactly, with no exclusions. Each has 400 PF steps (397
measurement updates, three skips), 397 PF fixes and 59 resamples. The prefix
is mostly fixed and does not cover the difficult parts of the route. The
46 CPU-only proposal regression cases also compare weights, selected moves,
RNG state, gauge-equivalent candidate entries, particle ownership and epoch
boundaries. The runtime environment was Windows/Python 3.12.10/NumPy 2.4.6
with custom GTSAM and cssrlib; exact dependency hashes are in the records.

Reproduce on a separate branch, saving the baseline before applying this
optimization. Use a new output directory for every study:

```bash
python benchmark_shared_moves.py --baseline BASELINE.py --tc-src tc/src \
  --data-root /data/PPC-Dataset-data/tokyo --output NEW_INITIAL --pairs 3
python benchmark_shared_moves_cpu.py --baseline BASELINE.py --tc-src tc/src \
  --data-root /data/PPC-Dataset-data/tokyo --output NEW_CPU --pairs 3
```

PF time uses the existing `step_seconds` accumulator. Full-run elapsed and
CPU times cover runner imports, input loading, filtering and output writes;
process startup, metadata hashing and parity checks are outside those clocks.
CPU time uses `time.process_time()` and still depends on CPU clock and system
conditions. The controllers copy both source snapshots, verify hashes, and
fail if any output array differs.

## Experimental likelihood replay on CUDA

`capture_likelihood_inputs.py` records actual scalar weight-cache misses and
tracked-conditional batches while returning the original CPU results. It forwards
runner arguments after `--`; supply the same WP56 flags from `run_ship.sh`, a
fresh runner `--out-dir`, and a fresh capture `--output`. `--tc-src` points at the
pinned dependency's `src` directory. Capture overhead is not timing evidence.

```bash
python capture_likelihood_inputs.py --output NEW_CAPTURE --tc-src tc/src -- \
    1 --max-ep 600 --out-dir NEW_RUN [normal WP56 runner flags]
python benchmark_likelihood_replay.py --manifest NEW_CAPTURE/manifest.json \
    --output NEW_TIMING.json
```

The replay controller checks input/source hashes and original CPU values before
timing. Its CPU reference compiles the unchanged `_batch_rows` and
`_loglik_batch` method bodies without importing GTSAM. Both policies receive the
same packed design, integers, means and covariance arrays. CUDA uses an FP64
kernel with growing work buffers, and its clock includes all input uploads,
output readbacks and synchronization. Imports, JIT, packing and parity checks
are outside the clock. The optional replay backend requires a working
Numba/CUDA installation; unsupported/singular systems raise explicitly.

This is an isolated experiment. The live filter's candidate loop, GTSAM graph,
random-number consumption, resampling, FIX gates and default CPU dispatch retain
their existing code. Scalar-weight recordings verify GPU likelihoods against
actual scalar results, but replay CPU timings use the existing batch method;
they do not measure the scalar proposal loop or positioning throughput.

The packed-transfer variant is `benchmark_likelihood_packed.py`. It uses the
same arithmetic kernel, combines inputs into two uploads and reads results in
one transfer. Host payload concatenation is part of the excluded preparation;
these timings cannot establish live-filter latency.

On a Tokyo run1 600-epoch WP56 prefix (seed 20260710), recording preserved the
position file and all 88 saved arrays byte-for-byte. Sampling every eighth
phase-2 epoch captured 32 weight-cache-miss batches (811 hypotheses, 15–39 per
batch) and 32 conditional batches (755 hypotheses, 5–53 per batch), with 48–100
measurement rows and state size 11. No hypotheses were replicated. This is an
early, predominantly fixed prefix, not a difficult-canyon or six-route study.

| Replay/pass | Kind, 32 batches | NumPy total median, ms | CUDA total median, ms | Faster paired totals |
| --- | --- | ---: | ---: | ---: |
| Separate transfers / initial | Conditional | 140.94 | 325.63 | 0/10 |
| Separate transfers / initial | Weight | 137.68 | 337.05 | 0/10 |
| Separate transfers / final | Conditional | 162.29 | 375.51 | 0/10 |
| Separate transfers / final | Weight | 160.95 | 380.02 | 0/10 |
| Packed transfers / initial | Conditional | 118.50 | 202.01 | 0/10 |
| Packed transfers / initial | Weight | 114.90 | 208.89 | 0/10 |
| Packed transfers / final | Conditional | 100.39 | 195.36 | 0/10 |
| Packed transfers / final | Weight | 103.61 | 204.86 | 0/10 |

Each controller/pass uses a fresh process, two warm-up pairs and ten alternating
timed pairs per recorded batch. The final process reverses the first policy.
The table reports medians of per-trial sums over 32 distinct recorded calls,
not one artificial larger batch. Both variants match all captured CPU values
at `rtol=atol=1e-9`, and compare all three returned arrays to the unchanged NumPy
batch method. CUDA timings include the same full output readback for weight
records, although the live scalar weight method only needs the likelihood.

Neither variant wins on this workload: packed CUDA takes about 1.7–2.0 times
the NumPy batch duration. Separate and packed studies ran sequentially with
changing CPU medians, so their ratios do not establish a paired speedup between
the two GPU variants. No default/live GPU backend is enabled. Further work must
measure actual proposal generation, packing and same-epoch ensemble scheduling;
increasing particle counts or batching future observations would change the
workload. Replay uses Windows, Python 3.12.10, NumPy 2.4.6, Numba 0.66.0, CUDA
12.8 and compute capability 7.5, with one OpenBLAS/OpenMP thread. Production
does not force these settings.

All observations, source hashes, capture provenance and numeric errors are in
the [separate initial](../../benchmarks/rbpf_likelihood_tokyo_prefix_20261015_initial.json),
[separate final](../../benchmarks/rbpf_likelihood_tokyo_prefix_20261015_final.json),
[packed initial](../../benchmarks/rbpf_likelihood_packed_tokyo_prefix_20261015_initial.json)
and [packed final](../../benchmarks/rbpf_likelihood_packed_tokyo_prefix_20261015_final.json)
records. The 29 replay tests exercise both backends on CUDA, input ownership,
validation, unheld/partial carrier hypotheses, IRLS, workspace growth/shrink,
singular failure/recovery and explicit close. CPU-only CI covers the parser and
NumPy reference while skipping the optional CUDA cases.

## Dependencies

All pinned; `setup_deps.sh` fetches them.

- Python 3.12 with the packages in `requirements.txt`, including the
  `inuex35/cssrlib-numba` fork at `dd90eb0`.
- `inuex35/tightly-coupled-gnss-imu-fgo` at `5cbfec4`, checked out as `tc/`
  (BSD-3-Clause). The runtime imports `gnss_fgo` from `tc/src`.
- `inuex35/gtsam` at `3c2f54c`, built with its Python wrapper. The PyPI
  `gtsam` wheel lacks the `DoubleDifference*Factor[Arm]` factors.

```bash
bash setup_deps.sh                    # venv + pip + tc + GTSAM build
bash setup_deps.sh --python PY --skip-pip --skip-gtsam   # reuse an existing environment
```

The tc stage was exercised on 2026-10-06. The GTSAM stage replays the CMake
options of the original build (Visual Studio 17 2022, Release, Python wrapper
on, TBB/Boost features off) but has not yet been run on a clean machine. The original build used plain
MSVC with the build tree on a data drive. It did not use vcpkg; a vcpkg/MKL
attempt filled the system drive.

## Running

The data root defaults to the original workstation path; set `PPC_DATA_ROOT`
to the `tokyo` directory of the PPC dataset (`run{1,2,3}/` with `rover.obs`,
`base.obs`, `base.nav`, `imu.csv`, `reference.csv`).

```bash
PPC_DATA_ROOT=.../PPC-Dataset-data/tokyo bash run_ship.sh ship_r3 3 16000 --checkpoint-every 1200
python relabel_nb9_pos.py results/ship_r3/run3.pos results/ship_r3/run3.npz ship_r3_nb12.pos 12
python ../score_vs_inuex35.py --traj ship_r3_nb12.pos --city tokyo --run run3 --format pos \
  --data-root .../PPC-Dataset-data
```

A full run takes 30–70 min (single process, about 5–8 epochs/s). The filter
is stochastic: the seed is 20260710 unless `--rb-seed` is given, and results
move by up to about 3 pp between seeds (see the WP39 section of the benchmark
record). Compare variants over several seeds.

Optional map-aided setting (WP39, not the default): `--env NLOS_ENABLE=1
--env NLOS_AR_EXCLUDE=1 --env NLOS_MASK_DIR=<dir>`. The directory holds
`tokyo_run{N}_per_epoch_nlos.csv` masks from
`experiments/build_per_epoch_nlos_csv.py`. Build them with
`--receiver-pos-file` pointing at a first-pass output. Without that flag the
builder ray-traces at the ground truth, which is an oracle.

## WP43 additions (2026-10-10)

The copy is no longer verbatim: WP43 added the options below. Each defaults
to off, and with all of them off the output is bit-identical to the
original copy (checked with `check_frozen_copy.py` on 1,200 run3 epochs).

| option | effect |
|---|---|
| `--env RBPF_TRACK=2` | shadow tracking: every particle lineage carries a Gaussian belief of the antenna position, predicted with the shared graph's IMU motion increment and updated with the epoch's DD rows under the lineage's own integers. Weights, holds and FIX decisions are unchanged. |
| `--env RBPF_TRACK_OUT=2` | report the MAP basin's mass-weighted tracked belief on every epoch where it exists (FIX/FLOAT labels unchanged) |
| `--env RBPF_TRACK_EMPTY=1` | also track float (empty-basin) lineages with a code-only belief |
| `--env RBPF_TRACK_Q`, `RBPF_TRACK_FLOOR` | random-walk sigma per graph epoch (0.02 m) and posterior floor (0.01 m) |
| `--env RBPF_TRACK_VARIANTS=q:floor:empty,...` | extra shadow variants on the same filter, recorded in `trk_var_xyz` for offline comparison |
| `--dr-fill 1` | on rover epochs the graph abandons, output the last reported position plus the IMU-predicted antenna motion (smode 6, written as float) |
| `--env RBPF_TRACK=1` | experimental: score the moves against the tracked beliefs too (one batched solve per particle x move). Not adopted: more fixes, more false fixes. |

`run_ship.sh` now runs the WP43 configuration (`--dr-fill 1 --env
RBPF_TRACK=2 --env RBPF_TRACK_EMPTY=1 --env RBPF_TRACK_OUT=2` on top of
WP18); `RBPF_SHIP=wp18 bash run_ship.sh ...` reproduces the WP18 output.
The FIX decisions are identical in both. Results and the reasoning are in
the WP43 section of `internal_docs/inuex35_tc_fgo_benchmark.md`.

Tools added with WP43:

- `run_queue.sh`: run a queue file of `run_ship.sh` jobs with bounded
  concurrency (3–4 jobs fit in 16 GB).
- `score_runs.py`: official matched-3D score of `results/<prefix>_r{1,2,3}`
  with the `nb >= 12` floor, and the mean/std over seeds.
- `output_policies.py`: offline comparison of output policies on saved runs
  (shipped float, tracked belief, fix-offset hold, offline interpolation).
- `error_budget.py`: splits the `<50cm_full%` denominator into fix / float /
  false fix / lost / missing epochs.
- `test_loglik_batch.py`: parity check of the batched likelihood kernel
  against the scalar one.

## WP44–WP45 additions (2026-10-11/12)

`run_ship.sh` now runs WP45 by default (`RBPF_SHIP=wp43` and `wp18`
reproduce the earlier releases). WP45 runs one configuration on all three
runs; the run1-only recovery flags of WP43/WP18 are not used.

| option | effect |
|---|---|
| `--nf 2` | two bands per system. With `--nf 3`, cssrlib's `qcedit` drops every satellite that lacks any of the three bands, although the DD factors use two: about 15 of 25 usable satellites survive on Tokyo run1 |
| `--band-priority 'C:2,6;R:-'` | per-system band order: BeiDou B1I + B3I (the default L1-first order picked B1C, which only BDS-3 transmits), GLONASS dropped (no FDMA channel table) |
| `--env RBPF_KEEP=1` | basin persistence: only real slips (slip / MW / CMC / FDE on carrier) remove an integer from the basins; graph re-seeds keep them and an unseen (sat, f) ages out after `RBPF_KEEP_MAXOUT_S` (10 s) |
| `--fix-rule t,F,t2,F2,M,B` | causal FIX label for the `.pos` flag and the npz `rule_fix` column: `nb_at[t] >= F`, or `nb_at_ne[t2] >= F2` with non-empty mass `>= M`, or the shared filter's own LAMBDA fix with `>= B` ambiguities, behind the PF output guards. Positions and the filter are unchanged. WP45 ships `3,5,3,12,0.1,5`, the rule `fix_rule_loro.py` selects in every leave-one-run-out fold |
| `--env RBPF_AFV=1` | diagnostic: share of the epoch's DD carrier rows within `RBPF_AFV_THR` (0.15) cycle of an integer at the reported position (`rb_afv_n`, `rb_afv_frac`) |
| `--env RBPF_DUMP_ROWS=1` | diagnostic: every phase-observed DD row of each epoch with the MAP basin's integer and agreement mass (`runN_rows.npz`) |
| `--env RBPF_TRACK_PROBATION_S=s` | `RBPF_TRACK=1` only: score on the loose float prior for `s` seconds after a belief reset. Not adopted |

WP46 (2026-10-12) ships a stricter label, `--fix-rule 3,15,3,15,0.1,99`,
selected over Tokyo 1–3 and Nagoya 1–2 (`RBPF_SHIP=wp45` keeps the
Tokyo-selected `3,5,3,12,0.1,5`). Other WP46 additions:

| option | effect |
|---|---|
| `PPC_CITY=nagoya` | `run_ship.sh` passes the Nagoya lever arm (`LEVER_ARM=0.593,0.670,1.216`, body FLU); the scorers read Nagoya references |
| `--env QC_BASE_MASK=1` | apply the rover elevation / C/N0 masks to the base QC too (cssrlib copies the base Nav before they are set, so `ELMIN_DEG` below 15° had no effect). Default off |
| `--env QC_ANY_BAND=1` | keep a satellite with at least one usable band and blank only the failed bands. Default off |
| `fix_rule_loro.py --set city:run:prefix` | leave-one-run-out over any mix of Tokyo and Nagoya runs |

WP47 (2026-10-13) ships `--fix-rule 5,15,3,99,0.1,9` with one-band
satellites as code-only rows and a 10° mask on both receivers
(`RBPF_SHIP=wp46` keeps WP46). Pairs with a one-band satellite charge their
residual to that satellite only (never to the reference), and such a
satellite is never the DD reference; both only apply with `QC_ANY_BAND=1`.

| option | effect |
|---|---|
| `--env QC_ANY_BAND_CODE_ONLY=1` | with `QC_ANY_BAND`: a one-band satellite (rover or base) keeps its pseudorange only; its carrier is blanked, so it adds no ambiguity. Shipped |
| `--env QC_ANY_BAND_PR_SCALE=s` | DD-pseudorange sigma multiplier for pairs with a one-band satellite. WP47 ships 10 |
| `--env QC_ANY_BAND_MAX_FULL=n` | keep one-band satellites only while fewer than `n` satellites pass the full QC. Not adopted |
| `--env QC_ANY_BAND_CNR=c` | keep a one-band satellite only if its usable bands reach `c` dB-Hz. Not adopted |

WP48 (2026-10-13) ships the measurement-model options below with
`--fix-rule 5,15,5,15,0.1,5,1.0` (`RBPF_SHIP=wp47` keeps WP47). The
optional 7th `--fix-rule` field vetoes every label while the PF's DDPR
quality exceeds it.

| option | effect |
|---|---|
| `--env DOP_V2=1` | Doppler velocity prior from an own per-satellite solve: no GLONASS / BeiDou GEO (GEO Doppler is 3–22 m/s off), C/N0 weights, residuals above `DOP_V2_OUTLIER_M_S` (0.5) rejected, covariance from the solve (`DOP_COV_SCALE`, floor `DOP_SIG_FLOOR` 0.05 m/s). Shipped, scale 1.5 |
| `--env DOPPLER_LEVER_COMP=1` | subtract the lever-arm rotation term `R_nb (w x l)` from the Doppler (antenna) velocity before it constrains the body velocity. Shipped |
| `--env DOPPLER_EXCL_BDS_GEO=1` | drop BeiDou GEO rows from tc/'s Doppler LS (DOP_V2 always does). Not needed with DOP_V2 |
| `--env PR_CNR_ERR=e` / `PR_CNR_REF=c` | add `e^2 * 10^((c - C/N0) / 10)` per rover satellite to the DD-PR variance (default c = 45 dB-Hz). Shipped e = 0.5 |
| `--env HUBER_PR=k` / `HUBER_CP=k` | Huber loss on DD-PR / DD-CP (whitened threshold). Shipped 1.345 / 2.0 |

WP49 (2026-10-13) diagnostics, not shipped (D-047):

| option | effect |
|---|---|
| `--env RBPF_WL=1` | per-epoch wide-lane fix: DD WL integers from the float, LAMBDA ratio, WL carrier position (`rb_wl_n`, `rb_wl_ratio`, `rb_wl_rms`, `wl_xyz`) |
| `--env RBPF_WLPF=1` | a second PF on the wide-lane problem (no feedback, main output unchanged): `wlpf_ok`, `wlpf_nb_at`, `wlpf_nb_at_ne`, `wlpf_ne_mass`, `wlpf_npairs`, `wlpf_xyz` |

WP50 (2026-10-13) ships a hysteresis label and an integer carry on the WP48
filter (`RBPF_SHIP=wp48` keeps WP48):

| option | effect |
|---|---|
| `--fix-hyst L,G,t,F,t2,F2,M,B[,Q]` | FIX once `--fix-rule` held L consecutive epochs, then while this continuation rule holds, bridging G failed epochs. Shipped `5,2,2,6,2,6,0.1,5` |
| `--carry 1` with `--env RBPF_CARRY=1` | forward integer carry (`integer_carry.IntegerCarry`): an unfixed epoch is labelled FIX at the carried-integer position while the last FIX's single-difference integers fit the raw DD carrier (8+ pairs, 0.15 cycle). npz columns `hyst_fix`, `carry_fwd` |
| `integer_smoother.py runN.npz --max-s 15` | backward carry from later FIX epochs, a fixed-lag output (`runN_smooth.npz` / `.pos`, column `carry_bwd`); needs the row dump (`RBPF_DUMP_ROWS=1`, on in WP50) |
| `replay_carry.py [--flt-max 0.5]` | replays the WP50 label layer on a row-dump run (the PF does not see it, so the output equals `RBPF_SHIP=wp50`); `--flt-max` adds the WP56 float check |
| `score_label.py --data-root .../PPC-Dataset runN.npz [runN_smooth.npz ...]` | correct FIX (% of rover epochs) and wrong / FIX of the `rule_fix` label as written to the `.pos`, the FIX columns of [the report](../../docs/rbpf_fgo_report.md); `PPC_CITY` selects the city |

WP51 (2026-10-13) adds a post-processed output from a second, backward pass
(the real-time and 15 s lag outputs are unchanged):

| option | effect |
|---|---|
| `--reverse 1` | run on a time-mirrored copy of the run (`reverse_time.py`): epochs reversed, Doppler / satellite velocity / gyro negated, satellite states at the true transmission time, LLI moved to the previous observation. `tow`, truth and outputs stay in true time; npz rows are in processing (reverse) order and carry `reverse=1`. Also sets a 100-epoch Phase-1 bootstrap (`--env PHASE1_EPOCHS=...` overrides), `PHASE1_STUCK_N=5` and (WP57) `IMU_STATIC_SEARCH=1` |
| `--env IMU_STATIC_SEARCH=1` | WP57: take the Phase-2 IMU bias from the first 5 s window whose per-axis gyro / accel std stay under `IMU_STATIC_GYRO_STD` (0.0087 rad/s) / `IMU_STATIC_ACC_STD` (0.2 m/s²), not from the first 5 s (a backward pass starts at the run's end, mid-turn on Tokyo 3). Default 0 (forward) |
| `--env PHASE1_STUCK_N=N` | hand over to Phase 2 after N Phase-1 epochs in a row fail to commit, once 20 have committed (the Phase-1 smoother can wedge on `map::at`). Default 0 (off) |
| `combine_fb.py fwd/runN.npz bwd/runN.npz [--k 5] [--d 0.3] [--af-half 5] [--af-margin 0.1]` | the WP51 output; WP54 adds `pos_src`: on unfixed epochs the position is the one of forward / backward / mean / forward float / backward float that best fits the DD carrier modulo whole cycles (ambiguity function over ±`--af-half` epochs, forward kept unless beaten by `--af-margin`; `--af-half -1` keeps forward positions). Then (`runN_fb.npz` / `.pos`, column `fb_src`): backward carry from the forward label without a lag limit, backward FIX segments that the forward label confirms (k+ epochs within d m, never further), then carry both ways. Needs the forward row dump |

WP55 (2026-10-14; `RBPF_SHIP=wp55` keeps it, `wp50` keeps WP50): the
same filter and label, plus a causal choice of the reported position on
unfixed epochs.

| option | effect |
|---|---|
| `--af-select N [--af-margin 0.1]` | with RBPF_CARRY=1: on epochs the label leaves unfixed, report whichever of the reported position, graph float and MAP / mixture tracked beliefs best fits the DD carrier modulo whole cycles, averaged over the last N epochs (`af_select.CausalSelector`); npz columns `pos_src` (0 reported, 1 float, 2 MAP belief, 3 mixture belief) and `pre_af_xyz`. Shipped N = 6 |
| `af_select.py runN.npz [--past 6] [--margin 0.1]` | the same choice replayed on a saved run with its row dump (`runN_af.npz`) |

WP56 (2026-10-14) is the default: WP55 plus a float check on the label.

| option | effect |
|---|---|
| `--fix-flt-max D` | no `--fix-rule` / `--fix-hyst` label (entry or stay) while the reported position is more than D m from the shared graph's float; the integer carry is not vetoed. Shipped D = 0.5 (the correct-FIX tolerance). On the backward pass, the worst wrong share falls from 5.28% to 0.41% (Nagoya 2's wrong basin aside) |

WP52 (negative result, default off):

| option | effect |
|---|---|
| `--env RBPF_CARRY_FB=1` | with `--carry 1`: hold the carried integer set in the shared graph on carried epochs and keep the PF's mass release off those pairs (`=2`: also on labelled FIX epochs). Carry episodes last a median of 2 epochs, so it changes little (benchmark record, WP52) |

Tools: `fix_integrity.py` (correct FIX / wrong FIX on one scorer for our
npz, `.pos`, inuex35 npz and libgnss++ CSV), `fix_rules.py` and
`fix_rule_loro.py` (label-rule sweeps and leave-one-run-out selection;
`--afv` and `--witness-pos` add the ambiguity-function and an external
RTK witness term), `fix_witness.py` (ambiguity-function study on row
dumps).

Scoring a WP45 run (the `.pos` flag is already the rule label, so no
relabelling):

```bash
python fix_integrity.py --data-root .../PPC-Dataset --traj ship:3:results/ship_r3/run3.pos:pos
```

On Linux (Ubuntu 24.04, Python 3.12, 4 cores, 16 GB) `setup_deps.sh` builds
the GTSAM stage when the compile is limited to two jobs
(`CMAKE_BUILD_PARALLEL_LEVEL=2`); four jobs ran out of memory.

## WP59–WP60 additions (2026-10-14)

Real-time ensemble and PLATEAU NLOS. All options default off; the shipped
WP56 output is unchanged.

| option / file | effect |
|---|---|
| `rt_ensemble.py` | scores an ensemble of forward runs on the same epochs the PPC2024 way: per epoch a member's candidate (reported, float, MAP, mixture) chosen by the WP55 ambiguity-fit rule (`--select af`) or a classifier trained leave-one-run-out (`--select learned`); `--shift` lowers unfixed outputs by a height bias chosen leave-one-run-out; `--gaps` adds rover-gap rows; `--label-members` gives the ensemble FIX label (2+ of them fixed within 0.2 m). Needs a seed-20260710 row dump (`RBPF_DUMP_ROWS=1`) as `dmp_s10_<ct><r>` |
| `--gap-fill 1` | rover observation gaps: IMU dead-reckoned positions at 5 Hz (IMU up to each output time only), written to the npz `gap_rows` |
| `--env QC_ANY_BAND_CP_CNR=40` | with `QC_ANY_BAND_CODE_ONLY=0`, a rover one-band satellite keeps its carrier only at this C/N0 or more (the ensemble's one-band member) |
| `plateau_los.py` | CPU line-of-sight tracer for PLATEAU meshes (2D grid + DDA, numba) |
| `--env NLOS_ONLINE=1 --env NLOS_TRI_NPZ=tri.npz` | traces every satellite each epoch from the IMU-predicted antenna position over the `load_plateau` triangle cache, and feeds the NLOS set to the NLOS layers below |
| `--env NLOS_MEAS_MODE=3` | NLOS pairs leave the double differences while `NLOS_HYBRID_MIN_LOS` (6) satellites stay in line of sight; with fewer, their sigma is multiplied by `NLOS_MEAS_SIGMA_MULT` (5) |
| `--env NLOS_REF_LOS=1` | each system's DD reference is its highest line-of-sight satellite |
| `--env NLOS_MASK_FILE=...csv` | read a precomputed mask (`build_per_epoch_nlos_csv.py` format) for any city |
| `--env P2_MIN_NS=n`, `P2_IMU_ONLY=1`, `DOP_LOWSAT_SIG=s` | probes, not adopted: commit phase-2 epochs with `n` common satellites (default 6), commit GNSS-less epochs from IMU/NHC/ZUPT, Doppler velocity prior from 2–4 satellites |

Results and the negative findings are in the technical report (WP59, WP60).

## Checking the copy

`check_frozen_copy.py` compares the shipped mode, resolved-ambiguity count
and position over the common prefix of two runs. On 2026-10-06, 400-epoch
prefixes of run1 and run3 from this directory were bit-identical to the
archived WP18 full runs:

```bash
RBPF_SHIP=wp18 bash run_ship.sh smoke_r3 3 400 --checkpoint-every 0
python check_frozen_copy.py results/smoke_r3/run3.npz <archived full_r3/run3.npz>
```
