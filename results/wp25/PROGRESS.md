# WP25 Progress — multi-epoch ambiguity lineage

## 2026-07-18 — first diagnostic posterior

Implemented a diagnostic-only temporal assignment filter with:

- per-basin current-epoch log-likelihood increments, separate from cumulative
  evidence;
- normalized stay/adopt/release/birth transitions over versioned integer
  assignments;
- an explicit death state, preventing an extinct lineage from transferring all
  mass to an unrelated candidate;
- constant-velocity Gaussian motion transitions;
- immediate ancestors, dwell time, bounded Viterbi history, and replay from a
  truth-free compressed per-basin trace.

Synthetic tests pass for persistent-lineage recovery against alternating
single-epoch distractors, partial assignment adoption, generation reset,
normalization, empty-epoch restart, and ancestry backtrace.

### Real-data result

Tokyo run3 without top-64 respawn, 400 epochs:

- single-epoch and temporal MAP both produced zero sub-50 cm positions;
- temporal selection was better on 82 epochs but worse on 305;
- median MAP error worsened from 3.552 m to 5.098 m;
- all 21 temporal `gamma>0.99` epochs were wrong.

Tokyo run3 with top-64 DDPR respawn, 200 epochs:

- a sub-50 cm live basin existed on 165/200 epochs;
- single-epoch MAP selected it zero times;
- default temporal selection selected it 7 times;
- a 12-cell replay sweep selected it at most 10 times;
- all temporal `gamma>0.99` epochs remained wrong.

This cleanly separates the bottleneck: candidate supply is now broad, and the
temporal plumbing can occasionally recover it, but DDCP/DDPR-conditioned basin
motion is not an independent discriminator and creates inverse calibration.
The temporal arm remains disabled for output and FIX.

Default Tokyo run2/200 regression is exact versus WP24: zero position/gamma/FIX
difference and 14/14 correct, zero false FIX.

Evidence is in `csv/wp25_temporal_ablation.json`, the replay sweep JSON, and the
compressed truth-free basin trace.

## Next

Continue WP25 together with WP26: add independent TDCP/Doppler/IMU relative
innovation to the transition likelihood, then repeat the frozen trace/calibration
evaluation. A posterior gamma cannot reach the production gate merely by
retuning birth or motion sigma.
