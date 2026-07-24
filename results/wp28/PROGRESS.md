# WP28 Progress — outage recovery and active hypothesis management

## 2026-07-18 — proposal-supply diagnostic

Nagoya run3/200 isolates a candidate-generation failure: the frozen WP27 arm
contains a live sub-50 cm basin on only 13/200 epochs. WP28 now separates
proposal supply from basin selection and trusted output. All additions are
opt-in; the default operational trajectory is bit-identical to WP27.

### Delivered

- covariance-axis position seeds around the causal float estimate;
- a bounded, age-limited bank of past position hypotheses;
- multi-position DDPR integer proposals with proposal-stage recall/rank
  diagnostics;
- an optional spatial-diversity reserve during basin pruning;
- a generic ablation summarizer for proposal recall, survival, and selection;
- unit coverage for seed geometry, history expiry, diversity retention, and
  evaluator output.

No truth enters proposal generation, pruning, selection, output, or FIX.
Truth is joined only in post-decision diagnostics.

### Frozen Nagoya run3/200 supply ablation

| Arm | Live sub-50 cm | Correct proposal anchors | Longest live span | Live-span p90 |
| --- | ---: | ---: | ---: | ---: |
| WP27 control | 13/200 (6.5%) | 0/40 | 10 epochs | 8.4 epochs |
| Single seed, top 256 | 28/200 (14.0%) | 4/40 | 25 | 20.4 |
| Float covariance axes, 5 m, top 32/seed | 57/200 (28.5%) | 9/40 | 26 | 15.8 |
| Float covariance axes, 5 m, top 64/seed | 93/200 (46.5%) | 15/40 | 26 | 13.0 |
| History 32, 1 m separation, top 16/seed | **129/200 (64.5%)** | **23/40** | **27** | **23.3** |

The correct proposals often rank deep in the combined list. In the single-seed
top-256 arm their ranks are 101–251; in the history arm they reach rank 431.
This identifies generation breadth and pruning capacity as the dominant cause
of the original 13/200 recall, rather than a missing final acceptance rule.

Dense 26-direction position grids and a spatial-diversity reserve do not beat
the axis/history arms. Those negative arms remain diagnostic evidence and are
not promoted as defaults.

### Selection and safety

Feeding the best history arm to the frozen WP27 max-cost integrity selector
leaves candidate supply unchanged at 129/200, but selects a sub-50 cm candidate
on 0/200 epochs. Declared FIX and false FIX are both zero. Candidate recovery
has therefore exposed the next failure: absolute candidate ranking does not
scale to the broader bank.

The production gate remains closed. Recall is 64.5%, below the required 90%,
and live-span p90 is 23.3 epochs = 4.66 s at 5 Hz, below the required 5 s.
No recovered candidate is connected to output or trusted commit.

### Neutrality

With all new options at their defaults, the Nagoya run3/200 trajectory SHA-256
is exactly the WP27 hash:

`C7B175C8EEF8690AFDE8B125D66B45DA161FCE52FD48B45DF5C67607075BF001`

### Next

Combine sparse covariance-axis and bounded-history seeds under a fixed compute
budget, then test longer history age and source-aware retention. Advance only
if frozen Nagoya run3 candidate recall approaches 90% without changing output
or FIX. Once supply passes, redesign absolute ranking for a broad candidate
bank before any safety calibration or Tokyo headline run.

## 2026-07-18 — recovery round 2

Two frozen follow-up arms tested whether more spatial sources or longer causal
memory closes the recall gap. Combining the 5 m covariance-axis shell with the
32-entry history bank produces 24/40 correct proposal anchors, but live recall
falls to 122/200 because the 624 proposals compete for the same 512 slots.
More breadth without source-aware retention is therefore counterproductive.

Extending history age from 25 to 50 epochs, with proposal count fixed at 528,
improves correct proposal anchors from 23/40 to 26/40 and live recall from
129/200 to **134/200 (67.0%)**. It does not improve survival: p90 falls from
23.3 to 17.6 epochs. Integer-search residual priors produce 133/200 and do not
materially change that verdict.

The new supplied-then-pruned audit finds that 8 of the 26 correct age-50
proposal anchors have no live sub-50 cm basin immediately after pruning.
Correct proposals reach rank 509. The remaining gap is now quantitatively split
between generation (14/40 anchors have no correct proposal) and retention
(8/26 generated-correct anchors are immediately absent). The next implementation
must preserve proposal-source coverage under a fixed cap; another unstructured
increase in basin count is not justified.

### Spatial deduplication check

An opt-in 1 m deduplication radius preserves distinct conditional-position
modes that share the same integer assignment. On the age-50 arm it is exactly
neutral: live recall remains 134/200, supplied-then-absent remains 8/26, and
the trajectory hash is unchanged. Cross-position Gaussian merging is therefore
not the measured loss mechanism in this window. The option remains disabled by
default; source-aware fixed-cap allocation is still the next diagnostic.

### Source-aware cap check

Each respawn position now carries a causal proposal-source identifier, and an
opt-in cap policy can reserve slots round-robin across current sources. A 90%
source reserve on the age-50 arm is again exactly neutral: 134/200 live epochs,
8/26 supplied-then-absent anchors, and a trajectory hash identical to the
ordinary cap. Source monopolization is not the measured retention failure.

The next recovery source should replay compatible historical ambiguity
assignments, not only historical positions. It must enforce ambiguity
generation IDs and current active-key support, and remain diagnostic until the
90% recall and 5 s survival gates pass.

## 2026-07-18 — generation-safe assignment replay

Implemented a bounded causal ambiguity-assignment bank. Every stored integer
is keyed by satellite pair and float-KF generation. Replay projects only keys
whose generation is still active and which occur in the current DD carrier
observation, then requires at least eight compatible integers. The assignment
is reconditioned from the current DDPR guard; no stale position state or
truth-derived choice is reused.

Assignment replay exposes a useful supply/retention trade. With 32 historical
positions and top-16 per position, 128 replay assignments raise correct
proposal anchors from 26/40 to 32/40 (65% to 80%), but 656 candidates competing
for 512 basins lower live coverage from 134 to 128 epochs. Reducing position
history to 16 keeps 32/40 proposal anchors with only 400 candidates and raises
survival p90 to 24.8 epochs, just below five seconds.

The best fixed-budget split is eight historical positions, top-32 per
position, and 128 assignment replays:

| Metric | Position-only age-50 | Assignment replay best |
| --- | ---: | ---: |
| Maximum proposals | 528 | 416 |
| Correct proposal anchors | 26/40 | 26/40 |
| Supplied then immediately absent | 8/26 | **1/26** |
| Live sub-50 cm epochs | 134/200 | **141/200** |
| Longest live span | 27 epochs | **80 epochs** |
| Live-span p90 | 17.6 epochs | **52.5 epochs (10.5 s)** |

This is the first WP28 arm to pass the predeclared five-second survival gate.
It does not pass the 90% live-recall gate: coverage is 70.5%. Increasing the
assignment bank to 256 lowers recall, and reallocating to four position sources
with top-64 lowers it to 53.5%; both are rejected. Declared and false FIX remain
zero because recovery is still disconnected from trusted output.

With assignment replay disabled, the default Nagoya run3/200 trajectory remains
bit-identical to WP27 (`C7B175...F001`).

The remaining failure is generation rather than immediate pruning: the best
arm produces a correct candidate on only 26/40 anchors, while losing only one
of those after insertion. Next, audit the 14 missing anchors by ambiguity
generation/reset state and per-source oracle rank before adding another
proposal mechanism.

### Reset audit and conditional-completion rejection

The best arm's 14 missing proposal anchors are epochs 0, 5 and a contiguous
recovery interval at 30/35/40/50–90. The latter follows resets at epochs 25,
35, 40, and 45; reset age grows to 45 epochs before proposal recovery. This
confirms that generation-change recovery, rather than steady-state depth, is
the remaining supply bottleneck.

An opt-in completion arm preserved four or more unchanged, generation-exact
historical integers and searched top-4 values only for new/reset dimensions.
Although it never copies an old integer across a generation boundary, direct
insertion is strongly negative: proposal recall falls to 9/40 and live recall
to 32/200 (16%). Wrong completion basins feed the position-history bank and
degrade later proposals. Direct conditional completion is rejected. Any
follow-up must first run as a shadow proposal source with no PF/history
feedback, and earn a recall gain before receiving nonzero mass.

The shadow audit is now complete. With the 70.5% best arm frozen, completion
finds a correct candidate on 11/39 eligible anchors, but every one is already a
correct proposal anchor in the control. Incremental recovery is **0/14 missing
anchors**. The shadow trajectory is hash-identical to the best arm. Conditional
completion is therefore fully rejected: it adds no recall where needed and its
direct insertion creates harmful feedback.

### Position-shell and satellite-subset audits

A 5 m covariance-axis shell was evaluated shadow-only against the frozen best
arm. It contains a correct candidate on 9/40 anchors but adds only one of the
14 missing anchors (epoch 40); the other eight overlap existing supply. Direct
insertion lowers live recall to 111/200, so the shell remains shadow-only.

Applying WP27's instantaneous max incident-cost satellite exclusion before
respawn subset selection is exactly neutral: all proposal, survival, live
recall, and trajectory metrics match the 141/200 best arm. The excluded
satellites (C19/E24/J03 across this window) do not control the selected
low-variance ambiguity subset. Satellite exclusion is therefore useful for
absolute selection but not this generation failure.

## 2026-07-18 — multi-subset and causal motion recovery

One-swap and four shifted-window ambiguity subsets were evaluated shadow-only.
They produce correct candidates on 19/40 and 18/40 anchors respectively, but
each adds only epoch 90 among the then-14 missing anchors. A dense 26-direction
5 m position shell is identical to the six-axis shell at the oracle level and
also adds only epoch 40. Neither subset diversity nor angular shell density
explains the reset gap.

Position-history motion is decisive. Per-basin Doppler velocity propagation is
negative (live recall 129/200), while robust causal TDCP displacement improves
the eight-position/top-32 arm from 141 to 143 epochs and removes its sole
same-anchor pruning loss. TDCP uses 199/199 causal intervals and never touches
trusted output.

Reallocating the fixed proposal budget to 12 historical positions × top-24,
plus 128 generation-safe assignment replays, gives the best cap-512 arm:

| Metric | Previous best | TDCP 12×24 |
| --- | ---: | ---: |
| Maximum proposals | 416 | 440 |
| Correct proposal anchors | 26/40 | **31/40** |
| Live sub-50 cm epochs | 141/200 | **149/200 (74.5%)** |
| Longest live span | 80 | **87 epochs** |
| Live-span p90 | 52.5 | 43.6 epochs (8.72 s) |

The neighboring 10×28 and 16×20 allocations reach only 142 and 144 live
epochs, so 12×24 is the frozen fixed-budget point. Extending history age to 100
regresses to 131; Doppler extrapolation and long stale memory are rejected.

A diagnostic cap increase from 512 to 768 reaches 154/200 (77.0%), a longest
span of 115 epochs, and p90 of 70 epochs, but still misses the 90% recall gate
and increases compute/state. Raising respawn cohort mass from 0.05 to 0.20 does
not help live recall at either cap. The cap-768 result is a retention ceiling,
not a production promotion. Declared and false FIX remain zero in every arm.

LAMBDA-residual proposal priors add one live epoch at cap 512 (150/200, 75.0%)
without changing proposal-anchor recall, but reduce live-span p90 from 43.6 to
35.8 epochs. The gain is real but marginal; both arms remain above the 5 s
survival gate and below the 90% recall gate.

Farthest-point history selection is also rejected. Unbounded spatial coverage
collapses live recall to 15/200; restricting candidates to 10 m from the DDPR
guard still reaches only 24/200. Low-weight spatial extremes are not useful
recovery modes. The frozen history selector remains weight-first with 1 m
separation.

### Pivot-invariant assignment replay audit

The assignment bank can now reconstruct satellite integer potentials from old
DD edges and express them under the currently observed pivot. The mode is
opt-in and clears the bank whenever the float filter reports an ambiguity
reset, so an integer is never deliberately carried across a detected slip
generation.

The mechanism restores proposal supply after a pivot transition: at epochs 50
and 55 the compatible replay count rises from 4 and 5 to 128, and the epoch-55
proposal oracle improves from 0.822 m to 0.547 m. It does not cross the 0.5 m
gate. On the full Nagoya run3/200 replay, live recall, longest span, and p90 are
unchanged at 150/200, 87 epochs, and 35.8 epochs, while correct proposal anchors
regress from 31/40 to 29/40. The bank is cleared 11 times.

This arm is rejected for production. `ambiguities_reset` identifies DD-track
residual resets, not a per-satellite slip arc. Clearing the whole bank is safe
but discards useful unaffected integer relations, including correct anchors at
epochs 25 and 180. The next assignment design must use pivot-invariant
per-satellite arc identities and invalidate only relations incident to a
causally detected slipped satellite.

### Per-satellite arc ledger and selective completion

A pivot-free satellite-arc ledger now reconstructs integer node potentials
from DD assignments. Arc continuity is observed against a TDCP-propagated
position reference, the constellation gauge is aligned from common satellites,
and only a satellite whose continuity residual exceeds 2 cycles receives a new
generation. A 50-epoch maximum gap matches the bounded assignment memory. The
ledger and every candidate in this section remain shadow-only.

At epoch 45 the detector identifies six BeiDou arcs while leaving the other
constellations intact. Requiring eight unchanged DD dimensions therefore
remains too strict. Holding at least four unchanged arc dimensions fixed and
using current-epoch LAMBDA to complete only the reset dimensions recovers the
entire epoch-50–75 burst. With eight diagnostic completions per historical
source, the union of frozen proposals and arc proposals reaches **38/40
(95.0%)** anchors. The seven incremental anchors are epochs 35, 50, 55, 60,
65, 70, and 75; only startup epochs 0 and 5 remain missing. This passes the
predeclared 90% shadow proposal-recall gate for the first time.

The unrestricted diagnostic arm generates as many as 1024 candidates. Keeping
only two current-generation completions per historical source preserves the
same 38/40 union with at most 256 arc candidates. All seven incremental correct
candidates have truth-free residual rank at most 108, so a 128-candidate cap
preserves the recall result and can replace, rather than augment, the existing
128 DD-key replay allocation. Global top-72 residual pruning is rejected; it
retains only 5/15 eligible correct shadow anchors in the first 80 epochs.

WP28B's proposal-recall gate is complete. WP28C must avoid computing discarded
LAMBDA completions, verify the 128-candidate arm end to end without shadow
feedback, and measure wall-clock cost before any PF-mass promotion.

The bounded full replay is now complete. Requesting only two LAMBDA completions
per source, ranking before position conditioning, and applying a 128-candidate
cap preserves the same **38/40 (95.0%)** union recall and all seven incremental
anchors. Maximum arc supply is exactly 128, so it can replace the old 128 DD-key
replay allocation without increasing the frozen 440-proposal envelope. The
trajectory remains hash-identical (`C7B175...F001`).

Compute is not yet promotable: the capped shadow replay takes about 194 seconds
for this 200-epoch diagnostic on the current host. Candidate conditioning is
bounded, but repeated per-history LAMBDA completion still dominates. WP28C must
cache or batch completions and demonstrate an acceptable runtime before the arc
source is connected to PF mass.

### Final bounded live arm

The 194-second wall-clock observation above included the complete PF replay and
was not an arc-compute measurement. Prepared-search caching plus direct timing
shows 3.27 seconds of cumulative arc work across the final 200 epochs, peaking
at 0.179 seconds on one recovery epoch.

Promoting the capped arc proposals as a replacement for the old DD-key replay
raises live sub-50 cm coverage from 150/200 to **186/200 (93.0%)**. Proposal
recall is 37/40 (92.5%), the longest span is 87 epochs, and p90 is 85.5 epochs
(17.1 s). Stale-generation holdover, declared FIX, false FIX, and commit replay
mismatches are all zero. WP28's recall, survival, fixed-budget, compute, and
holdover gates pass. The next increment is WP29 Tokyo transfer and absolute
truth-free selection.
