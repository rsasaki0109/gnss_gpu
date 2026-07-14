# Structural GNSS methods milestone (2026-07-13)

## DD-RTK/IMU ESKF

The former design-only `libgnss++/fusion/dd_imu_bridge.hpp` is now operational.
It preserves the existing 15-state INS layout while appending live
double-difference ambiguity states to a dynamic covariance matrix, including
INS/ambiguity cross-covariances. It provides joint DD code/carrier Joseph
updates, innovation gating, slip-generation retirement, propagation of cross
covariance, quality-ordered sequential partial LAMBDA, and an innovation-gated
soft SPP reset that inflates covariance instead of overwriting a valid
propagated position. A DD ambiguity key includes both rover/reference
constellation and PRN plus their arc generations, preventing G01/E01 or
reference-change collisions in multi-GNSS operation. Twelve directly linked
C++ tests pass.

The bridge is no longer limited to hand-authored observation rows.
`RTKProcessor::formTightlyCoupledObservations` evaluates the existing real
rover/base RINEX double-difference machinery at an INS-propagated ECEF
position and exports code/carrier innovations, ENU Jacobians, body-frame
azimuth, signal-aware wavelength, elevation/lock/slip metadata, and separate
code/carrier variances. Ambiguity identity now also includes the concrete
`SignalType`, so a tracking-code change on one band cannot reuse the previous
arc. `LooseCouplingProcessor::processTightlyCoupledDD` now synchronizes the
propagated 15-state covariance and accumulated transition with the augmented
ambiguity covariance, applies joint DD updates/PAR/soft reset, and returns the
result to the live fusion state. The opt-in `gnss fuse --tight-dd-imu` path
assembles and applies those rows after each aligned rover/base epoch while
recording accepted/rejected updates, row counts, fixed ambiguity counts, and
soft resets. The default remains unchanged, and earlier loose-only results are
not being relabeled as tight-coupling evidence.

The complete CLI now builds with MSVC 19.44, Eigen 5.0.1, and C++20. On a
Tokyo/run1 50-epoch real-RINEX/IMU smoke, 40 initialized tight epochs were all
accepted, producing 1,440 DD rows, one partial-AR epoch with 36 fixed
ambiguities, and zero soft resets. Against the matching reference timestamps,
baseline versus tight p50/p95/p99 error was
0.05211/0.05613/0.06054 m versus 0.05193/0.05465/0.05562 m; every emitted
position in both variants was within 0.5 m. Coverage was 40/50 because the
first ten epochs initialize the fusion state. This is an operational smoke,
not the still-pending full-six/blocked-span adoption comparison.

The matched Tokyo/run1 development replay improves honest score
0.7122→1.5300% and p95 81.53→57.63 m.  More importantly, the first independent
full holdout Tokyo/run2 gives a smaller, mixed transfer: score
62.5366→62.8991% and p99 24.37→23.44 m improve, while pass@0.5 m
65.999→65.430%, p50 0.248→0.253 m, and p95 9.896→10.419 m worsen.  The tight
variant accepts/rejects 7,832/1,218 DD epochs, falls back from carrier to code
3,835 times, performs 26 soft resets, and activates zero PAR.  It remains
opt-in while the other full holdouts and blocked matrix run.

Tokyo/run3 provides a second independent full-holdout transfer.  Tight DD/IMU
improves score 59.7994→62.2919%, pass@0.5 m 61.710→63.135%, p50
0.261→0.244 m, p95 19.21→16.96 m and p99 35.92→33.05 m.  It
accepts/rejects 12,079/2,915 DD epochs, performs 6,471 carrier-to-code
fallbacks and 68 soft resets, and again activates zero PAR.  The two completed
holdouts are positive, but neither reaches Phase71 and the remaining Nagoya
runs are still active.

Nagoya/run1 is the third independent full holdout. Tight DD/IMU improves score
0.5839→0.9629% and p95/p99 97.15/231.96→93.81/173.38 m, while pass@0.5 m
19.05→18.27% and p50 5.66→5.77 m worsen.  It accepts/rejects 2,376/4,954
epochs, records 144,641 DD rows, 3,534 carrier-to-code fallbacks and 142 soft
resets, with zero PAR. Nagoya/run2 has started; the three holdouts improve
score but the mixed central/tail metrics and zero PAR keep adoption pending.

The completed all-six replay changes that provisional picture only modestly.
Distance-pooled score improves 32.246385→33.726454% (+1.480069), but Nagoya/run2
regresses 5.741607→5.567837%. Four of five untouched full holdouts improve
score; blocked holdouts give three positive and one zero score delta. Across
full runs the bridge processes 57,004 DD epochs, accepts/rejects
33,729/23,275, emits 1,197,961 rows, falls back 21,766 times, performs 621 soft
resets, and fixes zero ambiguities. The result is 52.479038 points below
Phase71, so both tight flags are omitted from production and retained only as
research ablations.

## WCP and switchable pseudorange factors

`wcp_factor.py` implements whitened left-nullspace projection of
`r = J dx + A dN`, eliminating shared carrier ambiguities without estimating
them. TC-FGO groups continuous carrier rows by satellite pair and explicit slip
generation. When a receiver does not provide generations, it now derives them
conservatively from jumps in the geometry-corrected float ambiguity at the
initial trajectory. It projects each sufficiently long slip-free arc and
inserts the resulting multi-epoch factor. `switchable_factor.py` analytically eliminates the switch
from `(s r)^2 + lambda (1-s)^2`; its signed reduced residual and exact Jacobian
are inserted into TC-FGO, with total and downweighted factor diagnostics.
The official PPC TC-FGO runner exposes both features independently and records
WCP, switchable, and downweighted factor counts in per-epoch telemetry. It also
selects the matching city/run RTK seed automatically and writes the correct GPS
week for all six runs. The focused WCP/switch/TC-FGO suite includes an automatic
5-cycle slip split test.

On a fixed Tokyo/run1 20-epoch real-data smoke, baseline / WCP / WCP+switch had
p50 errors 1.157 / 1.123 / 1.092 m and p95 errors 1.245 / 1.224 / 1.204 m.
WCP contributed 621 projected residual rows and the combined run inserted 801
switchable DD pseudorange rows. No switch crossed the strict `<0.5` integrity
threshold in this benign prefix. These are implementation smokes, not evidence
for production adoption; full-run and blocked-span replay remains required.

`run_tcfgo_full_runs.py` provides a resumable four-way
baseline/WCP/switch/WCP+switch replay across all six official runs, using the
same coverage, 0.5/1/3 m pass, p50/p95/p99 tail, factor-count, integrity, and
runtime schema as the blocked-span runner. The blocked matrix is complete.
Tokyo/run2 is the first completed independent full holdout: WCP improves the
honest score from 3.388952% to 7.888571%, p50 from 5.210 m to 3.370 m, p95 from
68.219 m to 67.275 m and p99 from 118.442 m to 111.949 m, while adding 221,810
WCP factors and 548.6 s wall time. This supports transfer but remains far below
Phase71; the other four full holdouts remain active.

Tokyo/run3 is the second completed full holdout: WCP improves score
0.082123→4.400715%, p50 12.849→2.551 m, p95 90.571→49.070 m and p99
116.498→110.149 m with 389,318 factors.  Its 800.0 s wall-time increase is
recorded but not used for ranking under concurrent load. Both completed full
holdouts transfer positively, while the other three remain active and all are
still far below Phase71.

Nagoya/run1 is the third independent full holdout. WCP improves score
0.354327→4.071689% (+3.717362) and p99 252.95→252.62 m with 192,581 factors,
while p50/p95 worsen 29.42/135.19→30.67/138.38 m. Runtime rises from 678.9 to
1,158.9 s. Thus all three completed holdouts improve score, but the Nagoya
central/tail transfer is mixed and remains far below Phase71.

Nagoya/run2 is the fourth full holdout. WCP improves score
0.275110→2.327408% (+2.052298), p95/p99 208.71/257.12→207.70/251.15 m and
pass@0.5 m 2.15→4.68%, while p50 worsens 46.78→47.14 m. It adds 223,322
factors and 402.8 s.

Nagoya/run3 completes the matrix under an explicit data-feasibility protocol:
all four available static FIX epochs are used, versus five on every other run.
None exceeds the 1 m/s heading threshold, so `phase2@5200` records that yaw is
unobservable from RTK-FIX velocity. WCP changes score 0→0.000127% and p50/p95
100.66/312.41→89.69/303.34 m, but p99 worsens 368.62→368.76 m. Across all six
runs, pooled score improves 0.644545→3.448119% with 1,341,642 WCP rows, still
82.757373 points below Phase71. WCP is therefore research-only and omitted
from production.

All full and blocked baseline/switch and WCP/WCP+switch pairs preserve exact
position SHA-256 and metrics. Full shadows evaluate 2,275,766 rows across
291,254 epochs without committing a factor. The shadow flag remains useful
opt-in telemetry; committed switching remains rejected after its earlier
catastrophic regressions.

Both features are omitted from Phase71 production. Existing explicit-ambiguity
carrier and pseudorange behavior remains the default.

## Signal-aware Doppler

`doppler_signals.py` resolves RINEX Doppler bands for GPS, Galileo, QZSS,
BeiDou, and GLONASS. GLONASS G1/G2 use the supplied FDMA channel; an unknown
channel abstains instead of silently applying GPS L1. Scalar-wavelength PF
interfaces receive an equivalent Doppler frequency that exactly preserves each
row's range rate. The Python velocity solver also accepts per-row wavelengths.
A multi-constellation WLS adds one clock-drift column per observed system, then
normalizes inter-system drift to the most-supported reference clock before the
single-clock PF update.

On the first 50 usable Tokyo/run1 epochs with all constellations loaded, 30.88
rows/epoch on average had known signal wavelength, 5.00 GLONASS rows/epoch were
safely excluded for missing channel metadata, and all 50 RBPF Doppler updates
ran. With clock normalization enabled, four clock groups were fitted at every
epoch; median fit RMS was 0.0337 m/s and median inter-system drift span was
0.0266 m/s. The Doppler suites pass 28 tests.

The matched full-six and declared blocked-span replay is complete. On 57,398
holdout epochs, GEJ and GEJCR both produced Doppler diagnostics/updates at
100% coverage. GEJCR increased mean fitted clock groups from 2.9970 to 3.9970
and changed standalone position p50/p95/p99 by +0.0649/-1.0008/-1.6493 m.
However, fit-RMS p95 worsened from 1.4273 to 1.4931 m/s and inter-system
drift-span p95 from 1.4924 to 1.7231 m/s. Both standalone RBPF configurations
had a 0% official PPC score and large absolute tails, so those position deltas
are diagnostic rather than production evidence. The signal-aware machinery is
retained, but unconditional C/R enablement is rejected; staged GEJ remains the
default pending evidence in a competitive estimator. The paired artifact is
`experiments/results/rbpf_doppler_gej_vs_gejcr_comparison.csv`.

## Recurrence Vector

`candidate_3dma.recurrence_vector_scores` implements the method missing from
the earlier residual-only approximations:

1. solve an initial position and receiver clock independently for each usable
   four-satellite subset;
2. difference every map candidate from each subset position;
3. project that recurrence vector onto the subset LOS directions;
4. convert projected ranging errors to LOS/NLOS probabilities and compare them
   with the candidate visibility map;
5. accumulate classification log probability across subsets.

The solver, projection, visibility probability, and exact-candidate behavior
pass 18 candidate-3DMA tests. The implementation follows the public method
description in Lee et al., ION GNSS+ 2025; the full paper is access-controlled,
so unpublished parameter details are not claimed.

On the Nagoya/run2 20-epoch +1.7 m controlled-offset pilot, ungated recurrence
selection moved to the grid corner and worsened p50 from 1.70 m to 5.58 m.
Four-satellite positions were 90.2 m median and 366.4 m p95 from the local
source, while the local grid radius was only 3 m. A truth-free consistency gate
now abstains when projected source error is excessive or the optimum lies on
the grid boundary. It preserved 1.70 m on the pilot and on the Nagoya/run3
20-epoch regression span.

The predeclared blocked-span replay is now complete. It evaluated 595 epochs
across all six official runs (100, 100, 96, 100, 100, and 99 usable epochs;
five skipped). Nagoya/run1 supplied the initial development settings, and
Tokyo/run1 later supplied the fixed 0.05 confidence gate; those two are labelled
development and the remaining four runs are holdout. The consistency gate abstained on every evaluated epoch,
so selected p50/p95 exactly matched the source on every span, with zero
improved and zero worsened epochs. Runtime ranged from 601 to 1813 ms per
evaluated epoch. Therefore Recurrence Vector remains a safe but inactive
experimental diagnostic and is not promoted over Phase71.

The first independent safe full holdout, Tokyo/run2, evaluates 6,466/9,151
epochs (70.66% coverage), abstains on all 6,466, and therefore preserves the
40.322580% source score and 0.496/4.278/55.205 m p50/p95/p99 exactly. Runtime
is 3,651.2 s. Tokyo/run3 independently evaluates 12,833/15,301 epochs (83.87%
coverage), abstains on all of them, and preserves its 40.106488% source score
and 0.487/13.659/23.250 m p50/p95/p99 exactly; runtime is 8,663.2 s. The
separately named ungated raw counterfactual directly tests the paper argmax so
safe-gate inactivity is not mistaken for positive evidence.

The resumable replay now validates policy provenance before reusing a chunk,
not merely its epoch count.  This audit found 24 early Tokyo/run1 chunks and
the first Tokyo/run2 chunk whose JSON predated the recorded 0.05 threshold.
They are excluded from the safe aggregate and queued for correction.  Raw
chunks additionally must record a zero source-error gate, zero probability
gate, and boundary-maxima permission.  The same fail-closed rule now applies
to blocked spans, preventing safe/raw artifact reuse across policies.
Coverage, abstention, tails, runtime normalization, and honest scoring use the
same finite-error row population, so a non-finite row cannot alter only one
denominator.

The corrected safe full-six matrix evaluates 38,749 epochs and abstains on all
of them, preserving a distance-pooled source score of 29.533785%. Per-run
coverage ranges from 48.01% to 83.87%. Safe blocked spans similarly preserve
56.656641% over 595 evaluated epochs. The ungated raw blocked counterfactual
accepts all 595, worsens 593, improves one, and collapses the pooled score to
0.276064%; five of six spans score zero.

The raw full-six counterfactual is also complete. It accepts all 38,749 finite
epochs, improves 2,726, worsens 35,972, and drops the same pooled source score
from 29.533785% to 0.059943%. Per-run scores are 0.118453, 0.005099, 0.055105,
0.079475, 0.055732, and 0%. Thus the safe mode is non-regressing only because
it never emits, while the ungated paper argmax is rejected independently on
both full-six and blocked evidence. The runner's resume contract checks raw
policy provenance and epoch-CSV populations, and permits only declared
terminal gaps where the official reference outlasts source `.pos` epochs.

## Production decision

All new emission paths are default-off or abstaining. The canonical Phase71
reference remains 86.205492% official across six runs. A fresh Phase71 replay
cannot currently be launched from this checkout. The base Phase11fa manifests
are in fact preserved under `experiments/results/rtkdiag_manifest`; the Phase71
script now falls back to those tracked files instead of requiring volatile
`/tmp/{run}_phase11fa_{labels,dirs}.txt` copies. A fail-fast preflight nevertheless
finds the three production ranker prediction CSVs, the Phase57 Nagoya/run2
internal diagnostics, three Phase10 FGO candidate directories, and all 19
Phase19 GICI candidate directories absent. Git history confirms that only the
small Phase70/71 summaries were formerly tracked; these candidate and ranker
inputs are not recoverable from the repository. This is an artifact
reproducibility gap, not an estimator failure.

The production script explicitly fixes `--pf-mode-policy off` and
`--doppler-systems G,E,J`; its source-contract audit rejects accidental FFBSi,
tight-DD, experimental-carrier, WCP, switch, or Recurrence production flags.

The fixed blocked-span manifest and
`experiments/summarize_structural_ablation.py` now impose one output schema on
pooled, per-run, and predeclared 100-epoch urban-blocked evaluations. It records
reference coverage, 0.5/1/3 m pass rates, p50/p95/p99 tail error, mode and FFBSi
abstention, signal-aware Doppler availability, and runtime without fitting any
threshold on the evaluated rows.

## Final verification

- Current core changed-area suite after reverse-time Doppler and resumable
  evaluation-runner additions: **103 passed**.
- Changed-area Python suite: **136 passed** in the final expanded item-1--7
  regression; Ruff passes and repository-wide discovery collects **3,606**
  tests with zero collection errors.
- Latest focused item-1--7 and evaluation-runner audit: **105 passed**; Ruff
  passes after renaming the two ambiguous FFBSi test variables.
- After adding the honest PPC scorer and switch shadow/integrity gate, the
  expanded focused audit is **107 passed** and Ruff remains clean.
- Structural-ablation summarizer suite: **2 passed**.
- PF backward-smoother compatibility regression: **14 passed** after restoring
  lazy public-class construction for test doubles and subclasses.
- Changed-area Ruff check: **all checks passed**.
- Standalone C++ DD/IMU bridge suite: **12 passed**.
- Live fusion-state C++ DD/IMU integration test: **1 passed**.
- Tight-DD ablation runner test: **1 passed**; Ruff passes.
- Full `gnss_fuse` CLI: **MSVC 19.44 / Eigen 5.0.1 / C++20 build passed** and
  the real-RINEX/IMU 50-epoch tight-DD smoke passed.
- Repository-wide Python suite excluding the pre-existing untracked
  `test_reproduce_urbannav_external_baseline.py` collection error:
  **2725 passed, 48 skipped, 41 failed**. The failures are existing
  Windows-path assumptions, optional pyproj/PLATEAU environment failures,
  installed CUDA binary/API mismatches, and unrelated GSDC artifact tests;
  none occur in the changed-area 122-test suite.
