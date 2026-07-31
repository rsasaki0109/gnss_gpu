# GNSS-only MultiSD FGO / integer-aperture research audit (2026-08-01)

## Scope and hard constraints

- Inputs are PPC `rover.obs`, `base.obs`, and `base.nav` only.
- IMU, LiDAR, camera, map, and external-route training data are excluded.
- The production-library target is correct FIX >=70% for Tokyo and >=80% for
  Nagoya, false/FIX <=0.1%, zero >1 m false fixes, and end-to-end p95 <=100 ms.
- All new acquisition paths remain default-off until nested blocked CV, injected
  faults, CPU/CUDA parity, and latency gates pass.

## Primary literature and OSS audit

1. Teunissen and Verhagen's ratio-test analysis shows that a fixed ratio is not
   itself a correctness test. Hou, Verhagen, and Wu's FFRT implementation makes
   the acceptance threshold conditional on a required failure rate. Therefore
   the regular LAMBDA ratio remains a candidate-quality feature; the authority
   to publish FIX belongs to the independent aperture/holdout validator.
   Sources: [ratio-test revisited](https://doi.org/10.1179/003962609X390058),
   [efficient FFRT](https://doi.org/10.3390/s16070945).
2. Success-rate PAR fixes a high-confidence subset rather than requiring full
   ambiguity resolution. The two-step data-driven PAR work uses the integer
   bootstrapping success-rate lower bound. The 2026 entropy-weighted MPAR work
   ranks with ambiguity variance, carrier residual, and signal strength. We can
   implement the variance and residual terms from the existing PPC graph; any
   SNR term must come from the same RINEX observations and needs a missing-value
   fail-safe.
   Sources: [two-step success-rate PAR](https://doi.org/10.1016/j.asr.2016.07.029),
   [entropy-weighted MPAR](https://doi.org/10.3390/s26144388).
3. GraphGNSSLib demonstrates GNSS-only FGO with DD pseudorange, DD carrier, and
   Doppler followed by LAMBDA. `gtsam_gnss` exposes pseudorange, Doppler, TDCP,
   robust error, and ambiguity examples. GICI-LIB is broader and includes
   inertial/visual factors, but its useful comparison here is only the GNSS
   temporal/spatial factor organization and outlier rejection. No IMU/camera
   code or measurements are adopted.
   Sources: [GraphGNSSLib](https://github.com/weisongwen/GraphGNSSLib),
   [gtsam_gnss](https://github.com/taroz/gtsam_gnss),
   [GICI-LIB](https://github.com/chichengcn/gici-open),
   [RTKLIB](https://github.com/tomojitakasu/RTKLIB).
4. Window carrier phase FGO uses temporal carrier correlation together with
   pseudorange and Doppler. This supports the current fixed-lag MultiSD graph
   and TDCP/arc continuity direction without requiring inertial aiding.
   Source: [window carrier-phase FGO](https://arxiv.org/abs/2109.00683).

This is an algorithm/design audit, not source-code transplantation. OSS code is
used for behavioral comparison; implementation remains native to gnssplusplus.

## Measured PPC blocker before validator-aware PAR fallback

The production-library baselines are 5,984/11,928 correct FIX (50.1677%) for
Tokyo and 5,047/7,602 (66.3904%) for Nagoya, both with zero false fixes.

With causal MultiSD FGO, constellation PAR, candidate ratio 1.0, and the strict
disjoint validator, the baseline-priority union reached:

| City | Correct FIX | Total | Rate | False FIX |
|---|---:|---:|---:|---:|
| Tokyo | 6,151 | 11,928 | 51.5677% | 0 |
| Nagoya | 5,304 | 7,602 | 69.7711% | 0 |

The full-route FLOAT ledger showed candidate supply, not only the validator
threshold, as the limiting stage. More importantly, native PAR stopped at the
first subset/pool whose LAMBDA search succeeded. If every top-K hypothesis from
that group failed the independent validator, it never tried the next valid PAR
subset. Candidate generation and validation were therefore sequential but not
integrated.

## Implemented experiment: validator-aware candidate groups

- Generate up to `multisd_max_candidate_groups` successful LAMBDA top-K groups.
- Evaluate one group at a time using only disjoint satellite/time observations.
- If a group has zero passing hypotheses, evaluate the next PAR group.
- Accept only one passing hypothesis within a group.
- If multiple hypotheses pass in any group, fail closed and do not search for a
  more convenient later group.
- Keep the default at one group, preserving production behavior.
- Expose `--multisd-fgo-shadow-candidate-groups` (1..32) and include it in the
  PPC CV policy/sidecar command identity.

The first comparison is locked to candidate ratio 1.0, eight groups, top-K 4,
constellation PAR, three-epoch causal history, 0.75 carrier pass fraction, four
holdout satellites, and minimum six fixed ambiguities. It must first beat the
one-group six-route 300-epoch result with zero false fixes before full-route and
fault testing.

### Result: groups=8 is diagnostic-only and rejected for promotion

The six-route 300-epoch probe improved Tokyo from 565 to 569 correct shadow
fixes and Nagoya from 444 to 476. It had zero false fixes, zero >1 m fixes, and
a worst-route p95 of 69.12 ms. Nagoya run1 reproduced 208 correct fixes with
`groups=1`, while `groups=8` produced 219; all 11 additions had selected ranks
at least four and therefore exercised the intended fallback.

The complete routes exposed the safety failure that the short probe missed:

| City | Shadow correct | Shadow false | >1 m false | p95 (ms) |
|---|---:|---:|---:|---:|
| Tokyo | 1,236 | 6 | 2 | 51.19 |
| Nagoya | 1,692 | 2 | 0 | 49.78 |

After production-library priority, Tokyo still had 6 false rescues (2 above
1 m), while Nagoya had 269 correct rescues and zero false rescues. Therefore
groups=8 is not an admissible policy. All six Tokyo false candidates came from
later groups (selected ranks 5--23); a two-group cap is also unsafe because one
false candidate had rank 5. The feature remains defaulted to one group.

An oracle-only diagnostic found that a later-group condition such as seed
separation <=0.18 m OR maximum integer distance <=0.18 cycles removed these
full-route false candidates, but it is not adopted: those thresholds were
observed on the final route and must be treated only as candidates for blocked
nested CV. The next implementation should prefer cross-subset consensus or an
FFRT-calibrated fallback aperture over a route-fitted scalar cutoff.

### Cross-subset consensus and fallback aperture

A second default-off experiment requires later PAR groups to be corroborated
by multiple subset solutions whose latest positions agree pairwise. The first
group retains its original unique-top-K decision. With eight candidate groups,
two-group consensus, and 0.1 m maximum pairwise separation, the complete Tokyo
route still produced two >1 m false fixes: the same wrong basin was supported
by two or three subsets. Both had large separation from the independent GNSS
seed (0.391 m and 0.485 m).

The follow-up adds a fallback-only seed aperture. It does not tighten or alter
the first candidate group. A four-group budget, two-group/0.1 m consensus, and
0.25 m fallback seed aperture is the best safe latency-bounded candidate so
far:

| City | Production baseline correct | Union correct | Rate | False | p95 |
|---|---:|---:|---:|---:|---:|
| Tokyo | 5,984 | 6,161 | 51.6516% | 0 | 55.13 ms |
| Nagoya | 5,047 | 5,305 | 69.7843% | 0 | 47.70 ms |

Relative to the safe one-group/candidate-ratio-1.0 experiment, this adds ten
Tokyo and one Nagoya production-FLOAT rescues. This is a small, safe diagnostic
gain, not the 70%/80% target and not a production promotion. The 0.25 m value
must still survive blocked policy selection because it was motivated after the
complete-route failure analysis.

Eight groups had the same safe Nagoya union only after the fallback aperture,
but its full-route CPU p95 was 142.94 ms. Reducing the budget to four retained
all 258 Nagoya rescues and reduced p95 to 47.70 ms.

### CUDA parity and performance result

The CUDA Release build and explicit CUDA MultiSD smoke passed. Across all six
300-epoch routes, CPU and CUDA had identical FIX epoch sets and zero ECEF
coordinate difference for common fixes. CUDA reported 77,647/77,647 successful
solves and zero fallback. Nevertheless, the current fine-grained serial GPU
path is slower: route p95 ranged from 84.88 to 407.76 ms, versus 31.41 to
85.05 ms for the CPU eight-group policy. GPU is therefore parity-qualified but
performance-rejected. A future GPU change must batch candidate/group RHS solves
instead of launching each small dense optimization separately.

### Raw-RINEX fault matrix

The four-group candidate was replayed for 1000-epoch prefixes after deterministic
GNSS rover-RINEX injection. Injection used production FIX anchors, never truth.
Tokyo outage/NLOS used four events because eight events could not satisfy the
anchor-spacing constraint; all other cases used eight.

| City | Fault | Events | Correct FIX | False | >1 m | p95 ms |
|---|---|---:|---:|---:|---:|---:|
| Tokyo | outage | 4 | 332 | 0 | 0 | 63.12 |
| Tokyo | cycle slip | 8 | 360 | 0 | 0 | 65.62 |
| Tokyo | satellite loss | 8 | 324 | 0 | 0 | 61.10 |
| Tokyo | NLOS | 4 | 315 | 0 | 0 | 97.30 |
| Nagoya | outage | 8 | 523 | 0 | 0 | 37.46 |
| Nagoya | cycle slip | 8 | 651 | 0 | 0 | 49.39 |
| Nagoya | satellite loss | 8 | 553 | 0 | 0 | 39.13 |
| Nagoya | NLOS | 8 | 505 | 0 | 0 | 95.05 |

The matrix contains 5,785 correct fixes, zero false fixes, zero >1 m false
fixes, and all case-level p95 values below 100 ms.

### Two-policy outer CV

The six 300-epoch routes were rerun together with two policies: the safe
one-group candidate-ratio-1.0 baseline and the four-group consensus/aperture
candidate. Outer leave-one-run-out selection chose the four-group policy in
all six folds. Its selected holdout aggregate was 1,020/1,798 correct fixes,
zero false fixes, and zero >1 m false fixes, versus 1,009 correct fixes for the
one-group policy. This validates the small PPC-prefix gain without using the
complete-route KPI for selection. It does not yet constitute an independent
external-data generalization claim.

### PPC quality-ranked PAR and city-stratified outer CV

The literature-motivated quality experiment adds a default-off candidate order
using only quantities already present in the causal PPC graph: ambiguity
marginal variance, distance to the nearest integer, and the windowed RMS of the
DD carrier residual normalized by its factor sigma. Each term is mapped to a
bounded unitless interval before summation. SNR was deliberately not fabricated
because it is not retained in the current FGO carrier factor.

Across six 300-epoch routes, quality-ranked `q4` produced 1,047 correct shadow
fixes versus 1,020 for `g4`, with zero false and zero >1 m false fixes. The gain
was heterogeneous: Tokyo improved by 55 fixes while Nagoya lost 28. A
city-stratified outer leave-one-run-out audit selected `q4` in all three Tokyo
folds; the selected holdouts reached 623/900 (69.22%) for Tokyo and 423/898
(47.10%) for Nagoya, with zero false fixes. This stratification uses only the
known city label and the other two runs of that city; truth remains
post-subprocess scoring-only.

On the production Tokyo run, `q4` increased the production-priority union from
6,161 to 6,193 correct fixes:

| City | Policy | Union correct | Total | Rate | False | >1 m | p95 |
|---|---|---:|---:|---:|---:|---:|---:|
| Tokyo | `q4` | 6,193 | 11,928 | 51.9199% | 0 | 0 | 63.16 ms |
| Nagoya | `g4` | 5,305 | 7,602 | 69.7843% | 0 | 0 | 47.70 ms |

Thus residual-aware ordering is a safe measured Tokyo gain (+32 production
rescues over `g4`) but remains far below the 70% stretch target.

The same fixed raw-RINEX fault inputs were then replayed with `q4`:

| City | Fault | Correct FIX | False | >1 m | p95 ms |
|---|---|---:|---:|---:|---:|
| Tokyo | outage | 403 | 0 | 0 | 48.33 |
| Tokyo | cycle slip | 451 | 0 | 0 | 46.26 |
| Tokyo | satellite loss | 385 | 0 | 0 | 43.72 |
| Tokyo | NLOS | 389 | 0 | 0 | 65.20 |
| Nagoya | outage | 533 | 0 | 0 | 24.38 |
| Nagoya | cycle slip | 662 | 0 | 0 | 25.43 |
| Nagoya | satellite loss | 561 | 0 | 0 | 22.11 |
| Nagoya | NLOS | 513 | 0 | 0 | 53.85 |

All 3,897 shadow fixes were correct, with zero >1 m errors and every case p95
below 100 ms. The inputs were the already frozen injected rover RINEX files;
no truth or additional sensor was used by the solver. `q4` passes this fault
gate but remains default-off until ambiguity-arc blocked CV is complete.

### Constellation-pool interleave

The validator group collector previously exhausted successively smaller
prefixes of the first constellation pool before visiting another pool. A
default-off interleave now visits every unique constellation pool at the same
subset depth before shrinking again. On six 300-epoch routes, the combined
quality/interleave policy produced 1,048 correct fixes, zero false fixes, and
zero >1 m false fixes. It also cut worst-route p95 to about 22 ms by reaching
four diverse groups with fewer failed LAMBDA attempts.

The production run showed that speed did not translate to higher availability:

| City | Union correct | Total | Rate | False | >1 m | p95 |
|---|---:|---:|---:|---:|---:|---:|
| Tokyo | 6,186 | 11,928 | 51.8612% | 0 | 0 | 34.90 ms |
| Nagoya | 5,304 | 7,602 | 69.7711% | 0 | 0 | 36.13 ms |

Interleave is therefore rejected as the FIX-rate policy, but retained as a
default-off performance primitive for future batched CPU/GPU evaluation.

The full Tokyo run also exposed a harness-resume edge case: after a parent
timeout, its solver child overlapped a resumed process for 37 final rows. The
scientific columns were identical and only runtime differed. The scorer now
accepts only scientifically identical duplicate rows and keeps the conservative
maximum runtime; any conflicting duplicate remains fail-closed.

### Bootstrap-success/ADOP gates and prefix-CV failure

The existing FGO core already computed and enforced optional bootstrap success
rate (BSR) and ADOP gates, but the PPC shadow CLI left both at zero and exposed
only diagnostics. The PPC CLI and policy identity now expose default-off
`--multisd-fgo-shadow-min-bsr` and `--multisd-fgo-shadow-max-adop` values.

Three predeclared policies were compared on all six 300-epoch prefixes: BSR
>=0.9999, ADOP <=0.10 cycles, and both. BSR-only was selected in all six global
outer folds and produced 1,052 correct fixes with zero false fixes, versus
1,047 for `q4`; worst-route p95 was 34.34 ms. ADOP-only and the combined gate
produced 1,037 and 1,043 fixes and were rejected at this stage.

The complete production routes contradicted the prefix selection:

| City | BSR union correct | Total | Rate | False | p95 |
|---|---:|---:|---:|---:|---:|
| Tokyo | 6,172 | 11,928 | 51.7438% | 0 | 31.60 ms |
| Nagoya | 5,283 | 7,602 | 69.4949% | 0 | 22.10 ms |

BSR-only is therefore performance-useful but availability-rejected. More
importantly, this is direct evidence that first-300-epoch route prefixes are
not a sufficient blocked-CV surrogate. Subsequent policy selection must cover
multiple continuous-time blocks and explicitly audit ambiguity-arc overlap;
the full-route KPI remains evaluation-only.

### Ambiguity-arc split audit

`experiments/audit_ppc_ambiguity_arcs.py` now streams the six rover RINEX files
and starts a new `(satellite, carrier signal)` ambiguity arc on an LLI event or
a gap above 1.5 s. It assigns complete arcs to deterministic five-fold groups
using SHA-256 of route/satellite/signal/arc-start; no reference truth is read.

| Route | Epochs | Arcs | LLI-started | Longest arc epochs | Arcs crossing naive time boundaries |
|---|---:|---:|---:|---:|---:|
| Tokyo/run1 | 11,928 | 5,591 | 4,619 | 5,519 | 285 |
| Tokyo/run2 | 9,151 | 5,020 | 4,044 | 3,822 | 350 |
| Tokyo/run3 | 15,301 | 7,925 | 6,687 | 3,078 | 344 |
| Nagoya/run1 | 7,602 | 3,970 | 3,450 | 3,326 | 328 |
| Nagoya/run2 | 9,451 | 6,823 | 5,812 | 4,987 | 291 |
| Nagoya/run3 | 5,201 | 4,718 | 3,917 | 4,702 | 250 |

There are 34,047 arcs total. Route-namespaced leave-one-run-out overlap is zero
for all 15 route pairs, but 250--350 arcs per route cross naive equal-time
block boundaries. Therefore ordinary contiguous blocks leak ambiguity state.
The generated arc folds are the required mask identity for the next solver
replays; this audit alone is not claimed as a completed arc-held-out score.

`experiments/mask_ppc_ambiguity_arc_fold.py` connects that identity to solver
input by blanking only the carrier fields of non-selected complete arcs. Code,
Doppler, SNR, epoch structure, base observations, and navigation remain
unchanged. A five-fold keep-only Tokyo/run1 prefix retained 153,798 of 808,380
carrier fields and correctly failed closed with zero shadow fixes; it is too
sparse for policy comparison. A predeclared two-fold split retained
401,434 fields and produced 190/300 correct `q4` shadow fixes, zero false,
zero >1 m, and 16.93 ms p95. Therefore the executable arc-held-out matrix will
use two fully disjoint folds rather than weakening the six-ambiguity minimum.

The complete 6-route x 2-fold x 300-epoch replay then compared `g4` and `q4`.
Both produced 820 correct fixes, but `q4` emitted one 0.654 m false fix at
Nagoya/run2 fold 1 (TOW 555777.6). The candidate came from fallback rank 6 and
had BSR 0.9844, ADOP 0.199 cycles, and maximum integer distance 0.441 cycles.
`q4` is therefore rejected despite its clean full-route and fault results.

A follow-up `qf` policy leaves the first `q4` candidate group unchanged but
requires BSR >=0.9999 for later validator groups. This removes the arc-held-out
false candidate without applying the availability-reducing gate to first-group
fixes:

| Policy | Arc-held-out correct | False | >1 m | Worst p95 |
|---|---:|---:|---:|---:|
| `g4` | 820 | 0 | 0 | 54.34 ms |
| `q4` | 820 | 1 | 0 | 25.15 ms |
| `qf` | 820 | 0 | 0 | 26.69 ms |

Because the `qf` fallback threshold was designed after seeing the first split's
`q4` failure, it was not considered independently validated on that split. The
arc assignment therefore gained an explicit hash salt, and a predeclared fresh
`outer-v2` two-fold matrix was rerun after freezing `qf`. On that independent
split, both `g4` and `qf` produced 463 correct fixes, zero false, and zero >1 m;
`qf` worst-route p95 was 19.30 ms. The two independent arc assignments thus
show no `qf` availability regression or false fix relative to `g4`.

On the production routes, `qf` retains almost all of the Tokyo quality gain
and the full safe Nagoya result:

| City | Production baseline | `qf` union | Rate | False | >1 m | p95 |
|---|---:|---:|---:|---:|---:|---:|
| Tokyo | 5,984 | 6,191/11,928 | 51.9031% | 0 | 0 | 31.31 ms |
| Nagoya | 5,047 | 5,305/7,602 | 69.7843% | 0 | 0 | 25.23 ms |

The frozen eight-case fault replay also passes for `qf`:

| City | Fault | Correct FIX | False | >1 m | p95 ms |
|---|---|---:|---:|---:|---:|
| Tokyo | outage | 403 | 0 | 0 | 35.84 |
| Tokyo | cycle slip | 451 | 0 | 0 | 38.35 |
| Tokyo | satellite loss | 385 | 0 | 0 | 41.65 |
| Tokyo | NLOS | 389 | 0 | 0 | 83.54 |
| Nagoya | outage | 529 | 0 | 0 | 26.66 |
| Nagoya | cycle slip | 655 | 0 | 0 | 25.90 |
| Nagoya | satellite loss | 554 | 0 | 0 | 26.34 |
| Nagoya | NLOS | 507 | 0 | 0 | 41.96 |

All 3,873 fault-replay fixes are correct. `qf` is the highest measured
non-regressing candidate at this point. It remains a default-off shadow policy:
51.90%/69.78% is still far below the 70%/80% stretch objective.

### Fixed-lag window ablation

Longer fixed-lag windows were tested as a predeclared candidate-supply probe,
without changing the frozen `qf` quality ranking or fallback BSR gate. The same
six 300-epoch route prefixes were replayed with minimum epochs held at 10, so
the comparison isolates windows 10, 15, and 25 rather than adding a longer
warm-up. Reference truth remained post-solver scoring only.

| Window policy | Correct FIX | False | >1 m | Tokyo correct | Nagoya correct | Worst route p95 |
|---|---:|---:|---:|---:|---:|---:|
| `qf10` | 1,047 | 0 | 0 | 623 | 424 | 60.54 ms |
| `qf15` | 1,036 | 0 | 0 | 623 | 413 | 113.11 ms |
| `qf25` | 1,032 | 0 | 0 | 614 | 418 | 189.05 ms |

Nested leave-one-run-out selection chose `qf10` for every outer fold. The
longer windows therefore reduce candidate supply and exceed the 100 ms budget;
they are rejected without a full-route promotion run. This closes fixed-lag
length as the immediate stretch lever. Further availability work must create
new ambiguity hypotheses or new independent validation evidence, not retain
the same state longer.

The next predeclared ablation lowered the minimum PAR dimension from six to
five and four while retaining `qf10`, four holdout satellites, quality-ranked
groups, and the fallback-only BSR >=0.9999 gate. All three policies produced
exactly 1,047 correct fixes (Tokyo 623, Nagoya 424), zero false fixes, and zero
>1 m fixes on the same six-route probe. Their worst route p95 values were
35.14, 35.04, and 33.86 ms for dimensions six, five, and four. Therefore every
accepted subset was already at least six-dimensional: lowering the PAR floor
creates no additional hypotheses and is rejected as an availability lever.

### Dual holdout-partition union

Reducing the validator holdout from four satellites to three was previously
unsafe before quality ranking and the fallback BSR gate. With frozen `qf`, the
six-route 300-epoch replay produced 997 correct fixes and zero false fixes for
the three-satellite partition, versus 1,047/0 for the four-satellite partition.
Although the smaller holdout is worse alone, their fail-closed union accepts a
single-partition result or two results agreeing within 0.1 m. It produced 683
Tokyo and 523 Nagoya correct fixes, zero false, zero >1 m, and no conflicts.

A new `outer-v3-holdout` salted two-fold ambiguity-arc replay then produced
452 correct fixes for holdout four, 474 for holdout three, and 481 for their
union. All had zero false and zero >1 m fixes; the union added 29 correct fixes
without a position conflict. Sparse arc folds supplied candidates only on two
Tokyo fold/routes in this salt, so this is independent safety evidence rather
than a Nagoya availability claim.

On the full production routes, baseline-priority dual holdout improves the
highest safe result again:

| City | Baseline | Dual union | Rate | Delta vs `qf` | False | >1 m | Conflicts rejected | Sequential p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Tokyo | 5,984 | 6,237/11,928 | 52.2887% | +46 | 0 | 0 | 1 | 58.08 ms |
| Nagoya | 5,047 | 5,344/7,602 | 70.2973% | +39 | 0 | 0 | 0 | 47.15 ms |

The same union passed all eight frozen raw-RINEX fault replays with 4,547
correct fixes, zero false, and zero >1 m; two Nagoya partition conflicts were
rejected. Seven sequential p95 values are below 100 ms. Tokyo NLOS is 145.84
ms sequential, while the measured per-epoch parallel lower-bound p95 is 84.90
ms because each individual partition remains below budget. Parallel partition
execution is therefore required before this default-off candidate can satisfy
the runtime gate. `experiments/audit_multisd_fgo_dual_holdout.py` records the
fail-closed union, runtimes, truth-use boundary, and artifact hashes.

A direct two-`FGOProcessor` CPU experiment was performance-rejected. Leaving
each partition's internal top-K `std::async` enabled caused nested
oversubscription; disabling it made the repeated fixed-hypothesis
reoptimizations serial and a 300-epoch Tokyo probe exceeded six minutes. The
solver experiment was reverted with zero tracked gnssplusplus diff. The next
implementation must batch/shared-factorize hypothesis RHS solves or separate
CPU/CUDA work; spawning a second outer CPU task is not an accepted speedup.

### Common-normal CUDA multi-RHS top-K evaluation

The next implementation uses the existing cuSOLVER `potrf`/multi-column
`potrs` backend rather than adding another GPU framework. Within one PAR group,
top-K hypotheses constrain the same ambiguity/reference columns, so their first
Gauss--Newton normal matrix is common and only the right-hand sides differ.
The CUDA path now submits those RHS columns together, uses each solved column
as the corresponding hypothesis's first iteration, and runs one fewer ordinary
iteration. A pattern mismatch, non-finite result, or CUDA failure falls back to
the unchanged per-hypothesis optimizer. The independent holdout validator is
still the only FIX publication authority. CSV/JSON diagnostics expose batch
attempts, successes, and RHS columns.

On the six 300-epoch `qf` replay, explicit CUDA executed 2,750 successful
hypothesis batches containing 9,300 RHS columns. It retained exactly the same
1,047 accepted FIX epochs as the CPU artifact, with zero false and zero >1 m
fixes; accepted ECEF coordinates differed by at most 10 micrometres. A Tokyo
run1 A/B also reduced CUDA solve calls from 8,865 to 8,575 and measured wall
sum from 41.96 s to 20.32 s in the same loaded-session comparison. These
figures establish that the multi-RHS branch is real and scientifically
equivalent, not that forced GPU is universally faster.

The GTX 1660 Ti forced-CUDA route p95 values were 135.29, 109.12, 79.39,
60.60, 78.30, and 82.42 ms. Small PPC windows have about 105 states, so PCIe
and launch overhead still dominate on the first two Tokyo routes. Production
therefore retains the pre-existing heterogeneous `auto` threshold: state sizes
below 2,048 use Eigen/CPU, while genuinely large dense problems use CUDA. The
same CUDA-enabled binary in `auto` mode produced route p95 values 39.63, 38.07,
31.43, 28.36, 25.34, and 29.69 ms (maximum epoch 65.14 ms), again with
1,047/1,047 correct and no false fixes. Forced CUDA remains available for
large-problem and parity audits; it is not the PPC small-window production
policy.

## Next ranked experiments

1. Run the holdout-three/four FGO partitions concurrently (shared input
   preparation, independent candidate graphs) and require the same 0.1 m
   fail-closed union; verify Tokyo NLOS wall p95 <=100 ms.
2. Audit dual-frequency WL/NL candidates as an additional source only where
   they use observations disjoint from each selected holdout partition; the
   existing L1/L5 source alone supplied no Nagoya prefix candidates.
3. Use the already-recorded bootstrap success rate and ADOP to order whole PAR
   subsets, rather than adding more route-fitted scalar thresholds.
4. Calibrate an FFRT/IA lookup or Monte-Carlo boundary per ambiguity dimension
   and covariance-quality band, while retaining disjoint validation as final
   publication authority.
5. Exercise the implemented multi-RHS CUDA path on state sizes above the 2,048
   auto threshold and batch the two independent holdout partitions without
   weakening their fail-closed union; repeat parity and p95 <=100 ms gates.

Failure to reach 70%/80% is reported as a measured candidate/oracle boundary;
it does not authorize relaxing the false-fix integrity limits.
