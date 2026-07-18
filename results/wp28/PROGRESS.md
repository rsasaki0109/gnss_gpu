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
