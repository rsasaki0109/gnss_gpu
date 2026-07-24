# WP42 moving temporal trifrequency validation (2026-07-22)

WP42 removes one robust constant bias per exact DD satellite pair inside each
55-epoch moving block. Candidate ranking uses median absolute temporal residual,
not RMS, independently for primary, secondary, and tertiary pseudorange. The
5-epoch sampler now scans all five modulo phases and deterministically chooses
maximum evidence epochs, then raw carrier rows, then DDPR epochs, then the
lowest phase. This prevents block start from accidentally hiding 1 Hz RINEX
evidence.

The Nagoya 2327--2382 positive selects candidate 0 at ranks 7/1/6, rank sum
14 versus runner 17, and 21.43% margin. Frozen audit improves from 27/55 to
43/55 sub-50 cm epochs for the common offset. Its four truth-free bootstrap
offsets remain within 0.067 m; continuous interpolation reaches 51/55. Adjacent
Nagoya 2382--2437 fails upstream carrier
supply. Unsafe Tokyo 3350--3405 has a 6.71 m best supplied mode and is rejected
because its apparent winner ranks 1/12/1, outside the top-20% family bound.

Four further Nagoya boundary blocks were audited unchanged. 2786--2841 and
4274--4329 had zero evidence on their originally sampled phases. The old
5015--5070 zero-evidence result was a phase alias: auto phase selects modulo 2
with 11 evidence epochs, but WP42 still rejects at ranks 2/2/5 against a
top-3 family limit. 3272--3327 has 11 evidence epochs but rejects at ranks
11/6/11 and 14.3% runner margin. No threshold is relaxed.

Nagoya 923--978 supplies 11 epochs on auto phase 0 and contains a useful
identity mode, but WP42 correctly rejects it because the rank-sum runner margin
is only 18.18%. It is handled separately by the stricter direct-anchor WP44
gate; it is not used to weaken WP42.

Nagoya 5607--5662 supplies 11 evidence epochs and a truth-only 55/55 candidate,
but the unchanged selector still rejects. Its measurement winner ranks 2/2/11
with only 6.7% runner margin, while the audit-best candidate ranks 12/1/12.
This is retained as a positive-supply fail-closed case rather than an excuse to
relax the gate.

No ground truth enters candidate scoring, ranking, or promotion. Selected
candidate audit is attached only after the decision is frozen.
