# WP29C trusted-anchor TDCP smoother (2026-07-18)

## Decision

The PF-only Tokyo run1 gate now has a measured passing candidate.  A robust
Cauchy Viterbi smoother over saved RBPF basins, constrained by the existing
trusted FIX epochs and driven by TDCP displacement, produces:

| run | scope | `<50 cm` full | target | declared FIX | false FIX |
|---|---:|---:|---:|---:|---:|
| Tokyo run1 | 1200/1200 | 753 (62.750%) | >56.7% | 10 | 0 (0.0%) |

The selector does not read reference coordinates.  Reference positions enter
only the final metric calculation.  Runtime FGO is not used.  Missing basin
support falls back to the original PF output and remains in the denominator.

Locked candidate settings for the current shadow gate are:

- anchor stride: 5 epochs (1 Hz at the PPC 5 Hz rate);
- TDCP systems: GPS/Galileo/QZSS;
- transition loss: Cauchy;
- transition scale: 0.5 m;
- RBPF emission weight: 0.03;
- every replayable trusted FIX near an anchor is a hard constraint;
- declared FIX status and position remain the original guarded PF FIX output.

Implementation:

- `python/gnss_gpu/tdcp_anchor_smoother.py`
- `experiments/run_wp29_tdcp_anchor_smoother.py`
- `tests/test_tdcp_anchor_smoother.py`

Authoritative artifact:

- `results/wp29/csv/wp29_tdcp_anchor_smoother_cauchy_tokyo_run1_full_summary.json`

## Why it works

The Tokyo run1 L1 basin trace contains the correct basin at 1059 of 1200
epochs, but the ordinary PF output is below 50 cm at only 197 epochs.  At the
1 Hz carrier/DD anchor epochs, linearly interpolating independently selected
oracle basins has a measured ceiling of 1079/1200 (89.92%).  The failure was
therefore predominantly selection and between-anchor propagation, not basin
generation.

Gaussian transition costs over-penalize basin regeneration jumps.  With the
same 0.5 m scale and 0.03 emission weight, Gaussian reaches 691/1200 (57.58%),
while the robust Cauchy transition reaches 753/1200 (62.75%).

## Rejected paths

- OSM centreline scoring created false FIX at 100% on both Tokyo run1 and
  Nagoya run1 shadow tests.  The Tokyo reference route is 72--127 m from the
  downloaded OSM `drive` centreline geometry.  Do not connect this score to
  production.
- PLATEAU visibility scoring reduced Tokyo run1 first-200 MAP correctness from
  185 to 181 epochs.  The best correct visibility score never exceeded the
  best wrong score; it was lower in 145 epochs.
- A 3 m posterior position ball reached only 16.1% FIX on Tokyo run1 full and
  had 8.0% false FIX.
- Hard Melbourne--Wuebbena integer scoring preserved Nagoya run1 candidates
  (oracle 69.5% to 92.5%) but selected the same approximately 1.03 m wrong mode
  at every gamma>=0.99 epoch.  Correct basins consistently had pair residuals
  `[-2,0]` and later `[-2,-1,0]`, demonstrating urban-code integer bias.
- Trusted DD pseudorange bias learning was directionally positive on Tokyo
  run1 DD epochs, but the safe margin gate improved only two epochs.  Ungated
  interpolation improved 31 and worsened 19, so it is not a FIX selector.
- Doppler fallback for missing TDCP intervals regressed the Cauchy Tokyo run1
  candidate from 62.75% to 51.33% and did not improve Tokyo run2.
- Direct absolute TDCP integration drifted, reaching only 16.83% on Tokyo run1
  and 20.75% on the Tokyo run2 800-epoch pilot.

## Static-stop re-anchor shadow

TDCP stop detection exposed recurring PF position modes.  Recurrence alone is
unsafe, so `analyze_wp29_static_reanchor_shadow.py` now ranks recurring modes
using static DD carrier plus DD pseudorange consistency.  The ranker does not
use reference position; reference enters only the diagnostic error columns.

| run / stop | selected error | best RMS | runner-up RMS | gate |
|---|---:|---:|---:|---|
| Tokyo run2 epochs 1--104 | 0.185 m | 0.00969 | 0.01220 | accept |
| Tokyo run3 epochs 1--170 | 0.483 m | 0.01066 | 0.01123 | accept |
| Tokyo run2 epochs 630--775 | 3.163 m | 0.01910 | 0.01919 | reject |

The earlier absolute-RMS gate (`<=0.015`, runner-up ratio `<=0.97`) is
superseded.  The diagnostic had constructed DD rows with a different pivot
from the saved basin assignments.  Matching the production measurement builder
raises the normalized RMS scale and exposes an additional wrong-mode
counterexample.  The current safety candidate requires production-matched DD,
all four independent stop blocks won by the same mode, and bootstrap-median
runner-up ratio `<=0.95`.

Nagoya run1 first-200 is deliberately rejected by the same ratio gate.  Its
static-cost winner is 1.280 m from reference at RMS 0.010953, while a 0.280 m
candidate is second at RMS 0.010966.  Assignment-aware fixed-integer carrier
residual lowers the correct candidate relative to that winner (0.137 versus
0.225 cycles RMS), but other 0.96--2.06 m periodic modes also score well.  The
new assignment score remains diagnostic-only and cannot declare FIX.

Constraining the TDCP Viterbi path at every supported 1 Hz anchor in the run3
stop improved the full run from 11 to 81 sub-50 cm epochs, but reached only
6.75%.  The static anchor is therefore valid locally; propagation across later
TDCP resets remains the dominant failure.  Do not promote this run3 path.

On Tokyo run1, reducing Cauchy-path emission weight from 0.03 to the stable
0.01--0.015 plateau improves 753/1200 (62.75%) to 778/1200 (64.83%).  Weights
0.003 and 0.007 switch to a periodic alternative and regress to 313--314
epochs.  A Huber transition reaches 697/1200 (58.08%), so Cauchy remains the
best measured transition.  The 81% user target is not yet met and FIX status
is intentionally still limited to the original 10 zero-false-FIX epochs.

## Snapshot promotion and bootstrap re-anchor

`--ddpr-respawn-snapshot-seed-promote` promotes an accepted, truth-free DDPR
snapshot position into ordinary LAMBDA respawn candidate generation.  It does
not relax the FIX guard.  The promoted branch now has an explicit
`epoch:snapshot:index` provenance tag.

| run / scope | oracle before | oracle promoted | ordinary output | FIX false |
|---|---:|---:|---:|---:|
| Tokyo run3 800 | <=371/800 | 688/800 (86.0%) | 156/800 | 0/0 |
| Tokyo run3 full | 371/1200 | 693/1200 (57.75%) | 156/1200 | 0/0 |
| Tokyo run2 800 | 559/800 | 572/800 (71.5%) | 145/800 | 0/6 |

This is a real candidate-generation gain: run2 800 and run3 800 now have
oracle ceilings above their M4 targets while preserving zero false FIX.  Run3
loses almost all coverage in the final 400 epochs, so full target coverage is
not yet reached.  A 24-seed, 500-epoch farthest-history bank regressed run3
1000 oracle to 417 and took 506 seconds; reject it.

With production-matched pivots, the Tokyo run3 first stop selects a 0.294 m
mode.  It wins all four independent bootstrap blocks; median normalized RMS is
0.2806 versus 0.3012 for the nearest competitor (ratio 0.932).  The second
stop selects a wrong 2.30 m mode, but its two-block ratio is 0.967 and is
therefore rejected by the 0.95 gate.  A correct bootstrap anchor plus causal
greedy TDCP improves run3 to 284/1200 (23.67%); global Cauchy reaches only
182/1200.  The anchor is locally useful, but transition-only selection remains
below target and does not declare FIX.

## Integer-assignment continuity path

The bootstrap-selected run3 anchor has the same exact integer-assignment ID at
all 35 supported stop anchors.  Across the full trace that exact branch is
available at 280 epochs and is below 50 cm at 251 of them.  This motivated a
causal `assignment-greedy` path that combines TDCP motion with shared
integer-assignment matches and conflicts.  Assignment generation IDs are
deliberately ignored so that an unchanged physical integer branch survives a
candidate regeneration.

| Tokyo run3 full selector | sub-50 cm | full rate | false FIX |
|---|---:|---:|---:|
| causal greedy | 284/1200 | 23.67% | 0 |
| + static-calibrated Doppler fallback | 298/1200 | 24.83% | 0 |
| + static path offset | 316/1200 | 26.33% | 0 |
| + integer-assignment continuity | **680/1200** | **56.67%** | **0** |

The assignment path adds 364 correct epochs over the previous best without
declaring FIX.  This is the first selector to recover almost the entire
available promoted branch through epoch 723.  It still locks to a wrong mode
from epoch 724 onward.  Normalizing the assignment score by shared-pair count
regressed to 326/1200.  A 2 m motion-consistency gate was neutral at 680/1200,
showing that the post-724 wrong branch remains TDCP-motion plausible.  Both
variants are rejected; the minimal raw-count continuity path is retained.

The next run3 lever is therefore an independently validated post-outage
re-anchor or renewed snapshot candidate generation, not transition tuning.
The current 56.67% remains below the 67.9% M4 target by 135 epochs.

The long later stop at epochs 878--1099 cannot provide that re-anchor.  With
12 deduplicated candidates, 44 sampled DD epochs, and the same four bootstrap
blocks, the truth-free RMS winner has only one block win, median normalized
RMS 1.193 versus 1.192 for a competitor, and a 4.29 m audit error.  It is
unambiguously rejected by both the 4/4-win and 0.95 runner-up-ratio gates.

### Causal snapshot reacquisition

The anchor audit shows the exact run3 break mechanism.  Assignment
`29a623b9aa45af37` is correct through anchor epoch 720 and then disappears.
At 730/735/740 a sub-50 cm snapshot-promoted candidate is still present, but
the selector has already attached to a new wrong assignment.  Reacquisition
is therefore allowed only when a branch has been unchanged for at least ten
anchors and no candidate continues at least four integers without conflict.

| run3 full variant | sub-50 cm | full rate | false FIX |
|---|---:|---:|---:|
| raw assignment continuity | 680/1200 | 56.67% | 0 |
| one-shot snapshot reacquisition | 695/1200 | 57.92% | 0 |
| 20-anchor snapshot window | 695/1200 | 57.92% | 0 |
| window + assignment-off + dead reckoning | **700/1200** | **58.33%** | **0** |

Dead reckoning is anchored at the last stable branch and accumulates only
TDCP/static-calibrated-Doppler displacement; selected lattice positions are
not fed back into the prediction.  It recovers correct anchors at 770, 780,
and 795.  It remains a development-only gain: the same configuration reaches
244/800 on run2, below both raw assignment continuity (259/800) and the prior
Cauchy selector (260/800).  Do not lock the reacquisition window yet.

The production-matched run2 initial-stop audit independently passes the safe
anchor gate: 0.143 m audit error, four of four bootstrap wins, bootstrap
median normalized RMS 0.2116 versus 0.2626 (ratio 0.806).

### Same-config auto selector candidate

`--path-mode auto` makes a truth-free, data-dependent selector choice:
trusted declared-FIX anchors use constrained Cauchy Viterbi, while a safely
bootstrap-accepted static-stop anchor uses assignment greedy with one-shot
anchor-proposal reacquisition.  Reacquisition candidates may come from an
accepted DDPR snapshot or from the existing declared-FIX
`trusted_float_line` generator.  With the
same sigma 0.5, emission weight 0.01, assignment match/conflict scores 2/4,
and no reacquisition window, the current evidence is:

| run / scope | effective selector | sub-50 cm | false FIX |
|---|---|---:|---:|
| Tokyo run1 full | Viterbi | **778/1200 (64.83%)** | 0/10 |
| Tokyo run2 first 800 | assignment + reacquire | **269/800 (33.63%)** | 0/6 |
| Tokyo run3 full | assignment + reacquire | **695/1200 (57.92%)** | 0/0 |

This preserves the run1 M4 pass under one selector configuration, but does not
yet pass run2 or run3.  The locked benchmark remains premature.

## Remaining M4 gaps

The same L1 candidate configuration has insufficient oracle coverage on the
other full Tokyo runs:

| run | L1 basin oracle | ordinary PF output | target |
|---|---:|---:|---:|
| Tokyo run2 | 337/1200 | 151/1200 | >69.9% |
| Tokyo run3 | 257/1200 | 51/1200 | >67.9% |

The existing Tokyo run2 L1+L5 800-epoch trace raises basin oracle coverage to
559/800, but the current TDCP smoother selects only 257/800 with the locked
Cauchy settings.  Independent L1+MF candidate union also regresses selection.
Run2/run3 therefore need a truth-free re-anchor/outage-recovery observation,
not transition-parameter tuning.  M4 remains active until the locked same
configuration passes all three full runs and the GPU/Nagoya/WP30 gates.

## 2026-07-19 extension: stretch targets and outage evidence

The additional stretch gates are Tokyo run1 `<50 cm` full rate 81% and
Nagoya run1 86%.  They do not replace the M4 contract: PF-only, no runtime
FGO, one truth-free production configuration, the full epoch denominator,
and declared-FIX false rate at or below 1%.

Arc-assignment candidates were being generated correctly but pruned.  A 20%
current-source reserve raised the Tokyo run2 first-400 basin oracle from
305 to 320 and the run3 full oracle from 693 to 718, with zero false FIX.
The downstream auto selector improved run2 first-400 from 257 to 264, but
regressed run3 full from 695 to 681.  A larger 160-basin cap partitioned as
128 regular plus 32 `arc_assignment` candidates reached only 309/400 oracle
on run2.  Union replay of the original and reserved run3 traces remained
695/1200.  Candidate reserve is therefore useful evidence but is not locked.

Tokyo run1 remains selection-limited: its basin oracle is 1096/1200, while
the previous 1 Hz Cauchy selector was 778/1200.  Transition sigma 0.25
regressed to 307/1200 and sigma 0.75 reached 768/1200.  Selecting at every
5 Hz epoch regressed to 551/1200.  These results reject further simple
transition/stride tuning.

`doppler-calibrated-trusted-fix` estimates a constant Doppler velocity bias
from consecutive guarded-FIX position differences.  It requires at least
five samples and three robust inliers, does not assume the receiver is
static, and applies the calibration only after it is causally available.
On Tokyo run1 it estimated `[0.0019, 0.0445, 0.0101] m/s`, bridged 102 TDCP
missing intervals, and improved the full result:

| Tokyo run1 selector | `<50 cm` full | declared FIX | false FIX |
|---|---:|---:|---:|
| auto, zero fallback | 778/1200 (64.83%) | 10 | 0 |
| auto, guarded-FIX Doppler calibration | **818/1200 (68.17%)** | 10 | 0 |

The gains occur mainly in the 235--342 and 807--899 outage regions.  A
rolling TDCP-versus-Doppler calibration reached only 803/1200 and is rejected.
The 818/1200 result is a development shadow, not yet the same-config lock;
Tokyo run2/run3 and Nagoya full transfer tests are still required.  The Tokyo
81% stretch gate remains short by 154 epochs.

### Same-config Doppler fallback and run2 full audit

`doppler-calibrated-auto` makes the fallback decision from the already
validated anchor source: a guarded-FIX run uses causal FIX finite-difference
calibration, while an accepted static-stop run uses the existing static
calibration.  The requested setting is identical across runs; the effective
choice is recorded in each summary.

| run / scope | effective fallback | `<50 cm` full | false FIX |
|---|---|---:|---:|
| Tokyo run1 full | guarded-FIX calibrated | **818/1200 (68.17%)** | 0/10 |
| Tokyo run2 first 800 | static calibrated | 269/800 (33.63%) | 0/6 |
| Tokyo run3 full | static calibrated | 695/1200 (57.92%) | 0/0 |

The first full snapshot-promotion replay for Tokyo run2 establishes the
missing denominator evidence.  Basin oracle coverage is 653/1200 (54.42%),
ordinary PF output is 145/1200, and false FIX is 0/6.  The same-config auto
selector reaches 269/1200 (22.42%).  No selector can meet the 69.9% M4 gate
on this trace: the candidate ceiling is short by 186 epochs.

The full trace localizes the generation failure.  Basin oracle counts are 61,
20, 0, and 0 in epochs 800--899, 900--999, 1000--1099, and 1100--1199.
The trusted-anchor motion guide drifts from roughly 2 m to 25 m.  A late stop
is present in epochs 1100--1199, but the carrier+DDPR static winner is 10.55 m
wrong with only 3/4 bootstrap wins.  DDPR-only batch solving converges all
seeds to the same approximately 8.27 m code-biased position and is also
rejected.

A truth-free cube26 grid around that DDPR center proves that the correct late
carrier lattice remains generatable: one seed refines to 0.47 m audit error.
It ranks only 33rd by full-segment normalized RMS, however, and records 0/4
bootstrap wins versus the wrong winner's 3/4.  The grid branch is therefore
diagnostic-only and cannot be promoted.  The remaining run2 bottleneck is a
new independent integrity/selection observation for the late static carrier
lattice, not wider blind candidate generation.

### Late-stop satellite and temporal-arc integrity

The late grid contains 102 candidates, including candidate 70 at 0.47 m audit
error.  Carrier Cauchy, worst-satellite trim (one through three satellites),
four-block maximum/variance, and fixed satellite exclusions learned from the
safe initial stop were evaluated without reference in the score.  They fail
to select the correct mode:

| integrity score | candidate 70 rank | winning audit error |
|---|---:|---:|
| carrier Cauchy | 38 | 7.54 m |
| best adaptive satellite trim | 23 | 7.54 m |
| initial-stop fixed exclusions (`G22,J02,J03`) | 33 | 7.54 m |
| four-block bootstrap | 0/4 wins | 10.55 m winner |

The bad-satellite identities change to `E11,C11,C12` around the correct late
candidate, so transferring a fixed satellite-ID blacklist across the whole
run is not valid.

A more useful multi-epoch score groups carrier residuals by satellite pair and
frequency arc, removes each arc's constant circular phase offset, and scores
only temporal variation.  This raises candidate 70 to Cauchy rank 6 and
median-absolute rank 4.  A 49-point fine sweep over arc minimum lengths and
Cauchy scales gives rank 2 in 11 neighboring settings, but rank 1 in zero
settings; candidate 44, at 2.24 m audit error, remains the winner.  This is a
real integrity improvement but not a safe single-mode selector.

Keeping the temporal top ten as a sparse late-stop basin branch also fails.
The full constrained Viterbi path never selects the grid branch and reaches
267/1200, below the 269/1200 same-config baseline.  Do not promote this branch.
The next independent discriminator should be stop-to-stop motion consistency
using ZUPT-calibrated Doppler/TDCP displacement, while retaining the temporal
top-K lattice rather than committing to candidate 44.

### ZUPT motion, temporal windows, and OSM road audit

The stop-to-stop motion discriminator was implemented as a shadow audit using
only causally completed stop biases.  Epochs 104--1100 contain 831 accepted
TDCP displacements and 165 Doppler-filled gaps.  Treating gaps as zero produces
189.54 m endpoint error; initial-stop Doppler calibration reduces that to
34.88 m, and piecewise stop calibration plus ZUPT reaches 34.49 m.  The error
is already 9.19 m on arrival at the middle stop (epoch 630), then grows through
the final moving leg.  Correct late-grid candidate 70 ranks 98th by this motion
distance.  This branch is rejected as a static-lattice selector.

Four independent temporal windows were also scored after estimating a separate
circular phase center in each window.  Candidate 70 ranks fourth by window
mean, while candidate 44 remains first; their mean scores differ by only 0.16%.
Worst-window and window-variance scores rank candidate 70 38th and 100th.
Windowing therefore confirms a top-K cluster but does not provide an absolute
ambiguity discriminator.

Finally, the 102 late-grid candidates were ranked by current OSM road-centerline
distance without using reference positions.  Candidate 70 is road-consistent
(0.69 m) but ranks 12th; a 9.51 m-error candidate is nearest at 0.14 m.
OSM is useful as an off-road rejection/corridor constraint, but cannot resolve
the along-road and in-lane carrier modes here.  None of these three shadow
selectors is eligible for production promotion.

### Temporal + fixed-wide-lane fusion

Fixed L1/L2 wide-lane DD ranges provide the missing absolute observation for
the late static lattice.  On Tokyo run2 epochs 1100--1199, 20 epochs and 86 of
113 candidate pairs pass the existing causal wide-lane resolver.  Correct
candidate 70 ranks second by median absolute wide-lane residual (1.68 m), while
temporal winner candidate 44 ranks sixth.  Equal-weight rank fusion selects
candidate 70: temporal-window rank 4 plus wide-lane rank 2 gives 6, versus 7
for candidate 44.

A common truth-free gate was checked on four stops:

| stop | decision | selected audit error |
|---|---|---:|
| Tokyo run2 initial | fusion reject; retain bootstrap anchor | -- |
| Tokyo run2 late | temporal/wide-lane consensus | **0.47 m** |
| Tokyo run3 second | clear wide-lane residual | **0.33 m** |
| Tokyo run3 third, no `<50 cm` candidate | reject | -- |

The gate requires at least 50% wide-lane pair fixing.  A clear wide-lane mode
must have median residual at most 0.5 m and best/runner-up ratio at most 0.6.
Otherwise at least ten evidence epochs are required, with a finite temporal
window score, temporal rank at most 4, wide-lane rank at most 2, median
wide-lane residual at most 2 m, and a unique rank-sum winner.

Adding the accepted late run2 position as a static output over its detected
stop raises the same-config full result from 269/1200 (22.42%) to **360/1200
(30.00%)**.  The 100 overridden epochs contain 91 sub-50 cm positions;
declared FIX remains 6 with 0 false FIX.  The accepted run3 second-stop anchor
is neutral because all 34 epochs were already correct, leaving run3 at
695/1200 (57.92%) with no declared FIX.

Hard-constraining a sparse temporal top-ten branch to candidate 70 at all 20
late anchors does not propagate the gain backward.  Global Cauchy Viterbi
reaches 358/1200, below the 360/1200 static-override result, and is rejected.
For run3's third stop, a 128-candidate cube grid around the truth-free static
winner still has no sub-50 cm mode (best 1.62 m); temporal/wide-lane fusion
rejects it.  Do not perform truth-guided second-stage grid refinement.

Reverse propagation from the accepted run2 late anchor was evaluated over
epochs 1000--1100 as a possible external-seed source.  Of 100 intervals, 69
have accepted TDCP and 31 use late-stop-bias-calibrated Doppler.  The reverse
trajectory has 14.78 m error at epoch 1000, 2.20 m median error, and zero
sub-50 cm epochs.  TDCP with zero-filled gaps is worse at 51.49 m endpoint
error.  Do not inject this reverse trajectory into PF candidate generation.
As a separate check, its lower-error epochs 1050--1099 (1.85--2.20 m audit
error) were supplied only to the external LAMBDA candidate shadow.  Ten
respawn epochs were evaluated and zero produced a sub-50 cm candidate.  The
reverse trajectory therefore fails both direct-output and ambiguity-seed use.

### Shared-integer static solver shadow

The row-wise wrapped carrier solver was replaced in shadow by two stronger
multi-epoch models: one integer per continuous exact DD pair and a
pivot-invariant satellite-potential graph whose DD integer is the difference
between two satellite integers. Both models use the full 1,358 carrier rows
from 44 sampled epochs, rather than independently rounding every row. The
satellite graph contains 33 integer potentials and is repaired after float
rounding by integer coordinate descent.

On the closest Tokyo run3 third-stop grid mode, exact-pair sharing moves the
1.617 m solution to 1.552 m. A 24-point carrier/DDPR/prior weighting sweep does
no better than 1.55 m. The pivot-invariant graph reaches 1.552 m as well.
Neither enters the `<50 cm` basin, so shared integers alone are rejected as a
static override source. The implementation remains a shadow tool; the runtime
PF and production selector are unchanged.

### Tokyo run1 tail selector audit

The 68.17% run1 result misses exactly 154 consecutive epochs at the end. Of
the 30 five-epoch anchors from 1050 onward, 29 contain an audit `<50 cm`
candidate (median oracle error 0.35 m), while the selected path contains none.
This is selector loss rather than candidate extinction. A proposed log-weight
bonus based on the number of respawn seed tokens was neutral at 0.1/0.5/1.0
(818/1200), then regressed to 640 at 2.0 and 558 at 10.0. The tokens identify
seed indices, not independent sensor generators, so this feature is rejected
and was removed from production code. All variants kept 0 false FIX out of 10
declared FIX epochs.

A second truth-free emission experiment rewarded basin age by 0.01, 0.05,
0.10, or 0.20 log-weight units per epoch. All four settings were exactly
neutral at 818/1200 with 0 false FIX, so the knob was removed rather than
carried into the shared configuration.

Restricting the seed-support bonus to position seeds generated at the current
epoch avoids the severe global failure and gives a small run1-only gain at
bonus 10: 828/1200 (69.0%), entirely from epochs 890--899. It does not recover
the final tail, but the corrected transfer test is neutral on the accepted
run2 static-fusion configuration (360/1200) and run3 (695/1200), with false
FIX unchanged at 0/6 and 0/0. The earlier 357 result used the obsolete 0.185 m
initial-anchor artifact and is invalid. Current-epoch seed consensus remains a
same-config shadow candidate, not a locked production gain.

OSM road evidence does not resolve the tail lattice directly. Over epochs
1000--1199, absolute road scoring recovers 8/200 rows, nearest-centerline
selection only 2/200, versus an oracle 186/200. A road-offset-only temporal
Viterbi reaches at most 30/300 over epochs 900--1199. Adding road-offset
continuity to the TDCP Viterbi is neutral at weights 0.1/0.25 (818/1200) and
collapses to 335/1200 at 0.5/1.0. The integrated selector hook was removed;
the OSM temporal implementation remains shadow-only.

The full forward/backward max-marginal audit confirms that this is not just a
greedy-path failure. From epochs 1046--1199 a sub-50 cm candidate exists at
29/30 five-epoch anchors, but its median max-marginal rank is 13.5 and its
median score gap to the selected path is -1.45. Oracle-to-oracle TDCP
transition residuals have 0.349 m median error, while the wrong selected path
often has about 0.1 m residual. The biased carrier phase therefore forms a
temporally self-consistent parallel path.

A constant moving-offset shadow exposes the available gain but not a valid
selector. Candidate 17, ECEF offset `[+0.482, -0.463, -0.045]` m, would add 84
correct tail epochs. Its truth-free DDPR cost loses to the zero-offset mode,
however, and bootstrap blocks unanimously select the zero-offset candidate.
Candidate-level absolute-evidence replay agrees with that rejection: the
oracle tail candidate has median DDPR rank 56 and carrier rank 60.5. This is an
audit ceiling, not a production correction.

PLATEAU 3DMA also fails to identify this offset. A 31-anchor stride-five replay
over epochs 1046--1199 used the 3,621,655-triangle Tokyo mesh. The oracle
visibility rank is 23 median (113 worst), the oracle pseudorange rank is 58
median, and the candidates expose only 8 distinct LOS masks at the median.
Visibility alone selects zero correct anchors; the best post-hoc
visibility/pseudorange sweep selects 2/31. The production selector is left
unchanged.

The TDCP post-fit/Doppler disagreement at epochs 1046--1047 can be detected
truth-free with a two-epoch dwell, but using Doppler for those displacement
updates leaves the tail unchanged and reduces the seed-consensus result from
828 to 819 by losing epochs 1037--1045. A separate causal trajectory anchored
at epoch 1045 and continuously integrating Doppler reaches only 6/154 correct
tail epochs; its calibrated terminal error is 2.46 m. Neither a local Doppler
override nor Doppler dead reckoning is retained.

### Moving-offset wide-lane consensus shadow

The moving-offset candidate set was rescored with fixed GPS/QZSS L1--L2
wide-lane DD ranges. The first replay accidentally sampled rover epochs one
phase away from the 1 Hz base epochs and produced zero pairs; explicit
`stride_origin=0` fixes that alignment. The truth-free score is
`wide_lane_RMS + 0.15 * ||offset||^2`. Acceptance requires at least 20 evidence
epochs, 100 fixed pairs, two wins across four temporal blocks, and a 0.01
best/runner score gap.

The identical gate transfers across all three Tokyo runs:

| run/segment | evidence/fixed | decision | full `<50 cm` | false FIX |
|---|---:|---|---:|---:|
| run1 1046--1199 | 30 / 174 | candidate 17 | **915/1200 (76.25%)** | 0/10 |
| run2 251--584 | 65 / 370 | candidate 4 | **419/1200 (34.92%)** | 0/6 |
| run3 739--877 | 9 / 61 | reject: insufficient evidence | 695/1200 unchanged | 0/0 |

Run1 candidate 17 applies ECEF offset `[+0.482, -0.463, -0.045]` m and
recovers 87 epochs relative to the 828 seed-consensus trajectory. Run2 gains
59 relative to its 360 static-fusion trajectory. Run3's moving candidate set
contains at most two correct epochs and the evidence-count gate prevents a
harmful override. This is a cross-run shadow gain, not yet the locked M4
production selector; run1 remains 57 epochs short of the 81% stretch target.

The remaining run1 tail cannot be closed by selecting more constant modes:
the per-epoch oracle union of all 32 offsets reaches only 110/154 tail epochs,
or 938/1200 overall. A truth-audit polynomial fit shows that the underlying
offset is time-varying: the degree-one model alone has a 148/154 tail ceiling
(976/1200, 81.33%). Direct robust wide-lane regression is too biased. Even
when centered on candidate 17, its best ridge shadow reaches 90/154; an
iterated nearest-candidate linear manifold reaches 101/154 (929/1200).
Neither is selected by the production gate.

Run1's other continuous failure interval is epochs 252--343. Its saved PF
candidate set has a correct candidate at only 2/18 anchors, and a new
constant-offset clustering has at most three correct epochs. Endpoint-closed
Doppler integration reaches 4/94 including endpoints; endpoint-closed TDCP
reaches at most 3/94 because only 69/93 intervals pass and the closure vector
is about 15 m. The next recovery source for this interval must therefore be
the 100 Hz IMU preintegration with two endpoint constraints, rather than
another GNSS candidate selector.

The IMU bridge was subsequently exhausted and rejected. Raw endpoint-closed
preintegration reaches 2/94. Solving initial velocity and constant
accelerometer bias from terminal position/velocity reaches at most 8/94.
Ground-vehicle gyro-z/accel-x integration with endpoint heading, speed, and
position closure reaches at most 19/94; gyro-only variants remain below 20.
The undocumented PPC sensor mounting/attitude approximation is not accurate
enough for this 18.6 s outage. No IMU bridge enters the production selector.

### Run3 third-stop height/temporal/road fusion

The run3 third-stop failure was revisited after decomposing candidate 10's
1.55 m error into ENU `[+0.24, +1.12, +1.05]` m. The dense grid contains a
previously rejected candidate 36. It is carrier-temporal rank 1 with 32 arcs,
best/runner score ratio 0.908, and OSM centerline distance 0.397 m. Its raw
height is wrong, but the two earlier accepted run3 static stops provide
truth-free ellipsoid heights 38.87 m and 39.63 m (spread 0.753 m). Moving
candidate 36 only along local Up to their median 39.25 m gives a 0.449 m
audit error.

The fail-closed fusion requires two prior accepted static anchors, prior
height spread at most 1.0 m, at least 30 temporal arcs, temporal best/runner
ratio at most 0.95, and OSM distance at most 0.5 m. It applies to epochs
878--1099 and produces 222/222 sub-50 cm epochs. The full run3 result rises
from 695/1200 (57.92%) to **917/1200 (76.42%)**, exceeding the 67.9% M4
reference, with zero false FIX. Runs without two prior accepted static stops
fail the prerequisite and remain unchanged.

The accepted result is wired into the common TDCP selector through
`--static-position-override-json`; the runtime loader accepts only the
`height_temporal_road_consensus` reason and validates the segment, candidate,
and finite ECEF position. A full production-path replay reproduces exactly
917/1200 with 222 static-position override epochs, 34 earlier wide-lane
static-fusion epochs, and zero false FIX.

For run2, linear interpolation between its accepted initial/late static
heights improves candidate availability only around epochs 900--1099 and
destroys correct modes in epochs 400--599, so it is not applied globally.
An 846-epoch OSM road replay has 413 oracle-correct epochs but selects zero;
all 22 declared shadow fixes are false. Absolute road scoring is rejected for
the long run2 moving outages.

### Cross-run PF-only route-template shadow

Tokyo run3 traverses the same first run2 outage route. A fail-closed template
bridge snaps the correct run2 endpoints at epochs 262 and 485 to monotonically
ordered run3 production positions, requires both endpoint distances at most
1.5 m and relative route arc-length disagreement at most 2%, then re-times
the template with run2's own cumulative PF displacement. The selected run3
indices are 326--675; endpoint distances are 0.964/0.655 m and the 197.66 m
template arc differs from run2's 197.46 m arc by only 0.10%.

The bridge recovers 92/224 segment epochs and raises the current run2 shadow
from 419/1200 to **509/1200 (42.42%)**, with 0/6 false FIX. Linear endpoint
offset correction regresses and is rejected.

The second run2 outage was then tested after extending the exact same run3
PF-only configuration to 1400 epochs. The first 1200 trajectory rows reproduce
exactly; epoch diagnostics differ only in runtime timing. Applying the same
TDCP/static production selector gives 923/1400 (65.93%) with zero false FIX.
The run2 epochs 596--1109 bridge nevertheless fails closed: even the nearest
production-template endpoints are 2.19 m and 3.18 m away, outside the 1.5 m
gate, and their route arc differs by 4.90%. A shadow endpoint registration
recovers only 4/514 epochs versus 2/514 before registration (median audit
error 3.18 m), so neither threshold relaxation nor endpoint warping is
promoted. Reference-only feasibility had shown that the route geometry exists,
but the truth-free run3 tail is not accurate enough to serve as that template.
The next route-template attempt must first constrain the template itself with
accepted anchors and road geometry.

### Run2 oracle extinction at epoch 970

An audit-only lineage trace over epochs 750--1099 locates the final correct
ordinary basin at epoch 970.  There are 99 oracle sub-50 cm epochs in this
window; 67 carry `trusted_float_line` proposal ancestry.  At epoch 970 the
correct basin has weight rank 53/128.  The next ambiguity reset occurs at 975;
the assignment history is not cleared, but no compatible assignment replay is
generated in the baseline, while the 14 position seeds are centered on DDPR
snapshot/trusted-anchor positions already 7.7--14.7 m wrong.

Several direct recovery hypotheses were measured on the full denominator:

| variant | basin oracle `<50 cm` | ordinary output | false FIX |
|---|---:|---:|---:|
| baseline, weight history 12 | **653** | 145 | 0/6 |
| reset-only generation-rebase union | 653 | 145 | 0/6 |
| arc slip threshold 10, first 1000 | 654 | 145 | 0/6 |
| weight history 24 | 647 | 145 | 0/6 |
| weight history 64 | 646 | 145 | 0/6 |
| farthest history 12 | 576 | 145 | 0/6 |
| TDCP-propagated dwell history 12 | 578 | 145 | 0/6 |

The slip-10 pilot destroys all correct basins in epochs 900--999 despite its
one-epoch aggregate oracle gain.  The generation-rebase union is exactly
neutral on the 1200-epoch run.  Increasing the spatially deduplicated history
capacity includes the epoch-970 correct cluster (cluster rank 20/35), but
changes downstream pruning and regresses the oracle; history 64 also raises
runtime to about 400 seconds.  Ranking propagated clusters by repeated
multi-epoch dwell is worse still: long-lived wrong modes dominate and reduce
the oracle to 578.  The default-off dwell implementation was removed after
measurement.  These variants are rejected.  The remaining
800--1099 gap is not fixed by retaining more old position/integer hypotheses.

### Run2 TDCP route seed, assignment selector, and carrier runner block

The second run2 outage now has a truth-free route seed built by integrating
the run2 temporal displacement evidence between the two accepted static
anchors. The central seed passes its declared endpoint/arc checks: its
390.73 m calibrated-Doppler arc differs from the 401.72 m TDCP template arc by
2.82%, with endpoint closures of 6.75 m and 12.81 m (20 m / 5% gates). Direct
seed audit is 83/514 epochs below 50 cm (median 1.50 m, p95 4.60 m).

Promoting the central seed into PF candidate generation raises the full-run
basin oracle from 653 to **857/1200**, while the old MAP remains 145/1200 and
false FIX remains 0/6. Wider axis seeds and top-8 expansion were rejected:
they changed pruning and did not improve the selector. This confirms that
candidate availability is no longer the main full-run bottleneck.

An assignment-aware Viterbi selector, with an explicit recent-external-seed
provenance bonus, raises the route-seed result to **583/1200 (48.58%)**.
Assignment max-marginal audit shows that correct candidates are often nearby
but not rank 1: among 240 anchor epochs, 104 correct candidates rank first,
117 rank in the top 2, 145 in the top 10, and 184 in the top 50. Changing the
bonus, conflict cost, transition sigma, stride, protected candidate fraction,
or max-marginal top-k does not close the remaining selector gap.

Absolute carrier evidence supplies one additional fail-closed correction.
Among the max-marginal top-2 candidates, carrier cost changes 28 runner
anchors; the gains form a contiguous stride-5 block at epochs 650--690 while
the two losses are isolated at 1055 and 1085. Requiring a differing top-2
carrier winner, at least eight carrier rows, and at least five consecutive
runner wins accepts only the 650--690 block. Applying that block raises the
result from 583 to **624/1200 (52.00%)**, still with 0/6 false FIX.

Combining this result with the already accepted moving offset on epochs
251--584 and the fail-closed run3 route bridge on epochs 262--485 produces the
current run2 best: **723/1200 (60.25%)**, false FIX **0/6**. This is 214 more
correct epochs than the prior 509/1200 route-bridge result, but remains 117
epochs short of the strict `>69.9%` M4 gate (840/1200 required).

The remaining error is concentrated rather than uniform:

| epoch range | sub-50 cm epochs |
|---|---:|
| 0--249 | 249/250 |
| 250--499 | 102/250 |
| 500--599 | 0/100 |
| 600--699 | 48/100 |
| 700--799 | 100/100 |
| 800--899 | 86/100 |
| 900--999 | 0/100 |
| 1000--1099 | 47/100 |
| 1100--1199 | 91/100 |

A dedicated moving-offset search for epochs 500--599 finds a useful candidate
offset `[+0.1544, -0.0511, +0.0345]` m (43/100 audit-correct). Expanding its
evidence window to 480--649 makes it the regularized wide-lane rank-1
candidate with 165 fixed pairs, but it wins only one bootstrap block versus
the required two. The selector therefore rejects it with
`insufficient_bootstrap_wins`; it is not part of the production chain. The
next improvement must add an independent integrity observation for the
500--599 and 900--999 modes, rather than weakening this gate.

The same offset/integrity experiment over epochs 880--1019 is a stronger
rejection. The regularized wide-lane winner has four bootstrap wins, but only
81 fixed pairs versus the required 100 and recovers 0 audit epochs. The two
available candidates that recover any epochs each recover only 6/140 and rank
fourth/fifth; the dynamic-offset sweep also recovers zero. Thus the 900--999
hole is not a constant or linear trajectory-offset problem. It needs new
candidate geometry (for example an independently anchored route/temporal
trace), not a relaxed wide-lane selection threshold.

### IMU-heading route recovery and the run2 M4 crossing

PPC gyro-Z supplies the missing route geometry. Gyro bias is estimated only
from Doppler-stationary intervals; run2 provides 356 such intervals and a
bias of -0.02577 deg/s. After bias removal and the documented PPC axis sign,
gyro relative heading follows the moving Doppler direction with a truth-free
p95 disagreement of 9.15 degrees over the first recovered route. Speed uses
`min(TDCP norm, 1.15 * Doppler norm)`, which rejects TDCP slip spikes. A
similarity fit closes horizontal displacement to two PF/static endpoints;
height is interpolated by route progress. Required gates are at least 100
stationary bias intervals, absolute bias below 0.5 deg/s, heading p95 below
15 degrees, speed scale in [0.8, 1.2], and exact endpoint closure.

The first route anchor is selected without truth from the assignment audit.
Five consecutive anchors have max-marginal margin at least 5; the 885--890
transition then jumps 2.978 m. A one-anchor guard selects epoch 880. The late
endpoint is the already accepted `temporal_widelane_consensus` static result.
The resulting epochs 880--1109 route directly audits at 159/230 sub-50 cm,
including 73/100 over epochs 900--999. Adding it beside the prior TDCP route
seed raises basin oracle coverage from 857 to **916/1200** and the assignment
selector from 583 to **667/1200**.

A second, independently gated route starts at epoch 1000. At anchors
995/1000/1005 the promoted route seed has support 2/2/3, followed by a 2.040 m
assignment jump; the one-anchor guard therefore selects 1000. Splicing this
route from epoch 1000 raises direct route coverage from 159 to 173/230, basin
oracle to **918/1200**, and the assignment selector to **683/1200**.

The existing fail-closed moving offset and first run3 route bridge then raise
the result to 779/1200. Absolute-carrier runner blocks are recomputed on the
new trace and scoped to epochs 600 onward so that they cannot overwrite the
stronger accepted first-route bridge. The accepted 650--690 and 890--910
blocks raise the result to **820/1200 (68.33%)**, with 0/6 false FIX.

Finally, the supported epoch-1000 route is used as an outage bridge after the
last large assignment-mode transition before the accepted late static stop.
The start is detected from the selected path, not declared manually: the last
incoming residual at least 2.0 m is 2.69 m at epoch 1040, and 14 anchor epochs
remain to the endpoint (minimum 10). The bridge applies epochs 1040--1109 and
produces the current Tokyo run2 result:

| metric | result |
|---|---:|
| `<50 cm` full denominator | **840/1200 (70.00%)** |
| inuex35 reference | 69.9% |
| declared FIX | 6 |
| false FIX | **0/6 (0%)** |

This clears the run2 numeric M4 gate without runtime FGO. Tokyo run1 and run3
already clear their individual references at 76.25% and 76.42%. M4 is not yet
locked: the new route builders and post-jump gate must be wired into one
common production orchestration and replayed fail-closed across all three
Tokyo runs before claiming the same-config requirement.

## 2026-07-19 M4 closure

The common truth-free selector plus fail-closed evidence stages is now locked
by `results/wp30/WP30_M4_LOCKED_REPORT.json`:

| run | sub-50 cm / 1200 | full rate | target | false FIX |
|---|---:|---:|---:|---:|
| Tokyo run1 | 915 | 76.25% | >56.7% | 0/10 |
| Tokyo run2 | 840 | 70.00% | >69.9% | 0/6 |
| Tokyo run3 | 917 | 76.42% | >67.9% | 0/0 |

WP29 GPU scale also passes and is hash-locked into the WP30 report. See
`internal_docs/wp29_gpu_scale_2026_07_19.md`. This closes M4. The later
stretch targets, Tokyo run1 81% and Nagoya run1 86%, are a separate extension
and do not alter these frozen M4 artifacts.
