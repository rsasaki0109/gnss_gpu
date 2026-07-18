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
