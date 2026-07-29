# Phase 1 truth-free Evidence API

`gnss_gpu.evidence` separates candidate-basin safety evidence from estimator
implementation. A PF, grid search, affine DDPR solver, or future estimator can
all emit the same `BasinEvidence` records.

## Evidence families

The API defines seven independent families:

- TDCP;
- Doppler;
- IMU;
- carrier continuity;
- satellite arc continuity;
- road/height consistency;
- LOS/NLOS consistency.

Each `EvidenceSample` contains a candidate residual, its physical acceptance
scale, reliability, epoch, sample count, and provenance. Its support is a
bounded Gaussian kernel. Ground truth, reference trajectories, post-audit
position error, sub-50 cm results, and gained/lost epochs are rejected from
production metadata, including when nested.

`EvidenceBuilder` supplies typed adapters for all seven families with physical
units (metres, m/s, cycles, epochs, or mismatch fraction). This keeps producer
code explicit while preserving one normalized downstream contract.

Scoring is family-balanced: samples are averaged inside a family and families
then receive one vote each. A channel with many satellites or high sample rate
cannot overwhelm independent evidence.

## Unsafe-acceptance detector

The default detector fails closed on:

- too few independent families;
- evidence drawn from too few of motion, carrier, and context groups;
- insufficient temporal span or unstable support;
- weak total support;
- a close runner (ambiguous basin identity);
- an unopposed basin without substantially stronger coverage, support, and
  temporal stability.

The output retains the winner, runner, margin, per-family support, and every
unsafe reason even when selection is rejected. It never uses audit truth.

`TemporalEvidenceTracker` records winner continuity, basin switches, and
median runner margin in a bounded window. Epochs must be strictly increasing.

## Historical negative controls

Run:

```text
python experiments/audit_phase1_holdout_detector.py
```

The adapter reads only truth-free selector-time fields from the immutable WP53,
WP129, WP156, and WP168 locks. It does not feed their later sub-50 cm or
gained/lost audit into the detector. The Phase 1 gate requires recovery of at
least two historical rejections; the current conservative policy rejects all
four. This is a safety foundation, not yet evidence that a positive candidate
can recover lost epochs. Positive recovery remains a separate promotion gate.
