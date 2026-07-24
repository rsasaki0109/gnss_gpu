# WP26 Progress — independent relative motion

## 2026-07-18 — robust TDCP displacement

Implemented a causal FGO-free TDCP displacement primitive with:

- explicit carrier and receiver-motion signs;
- receiver clock-delta nuisance state;
- satellite motion and clock-drift correction;
- iterative leave-one-out slip/outlier rejection;
- displacement covariance and input/used/rejected row counts;
- external displacement/covariance transitions in the WP25 lineage replay.

The existing PPC TDCP runner had drifted from the solver API and could not pass
its sign/max-speed arguments. The API is repaired with backward-compatible
defaults. Synthetic tests recover a known displacement after rejecting an
8 m phase slip and verify positive-definite covariance.

### Tokyo run3/200

Truth-free acceptance used all 199 intervals and rejected 161 satellite rows.
Truth was joined only for evaluation:

- displacement error median 0.0028 m, p90 0.0056 m, RMS 0.0040 m;
- post-fit RMS median 0.0020 m;
- NIS median 0.213, so the covariance is conservative.

### Basin-selection result

External TDCP did not select a sub-50 cm basin despite 165/200 oracle-live
epochs. The reason is measured translation invariance, not poor TDCP:

- correct persistent edges: residual median 9.76 mm, p90 29.48 mm;
- wrong persistent edges: residual median 8.77 mm, p90 28.40 mm.

Both correct and coherently shifted basins reproduce the same relative motion.
Doppler and IMU can strengthen survival/holdover but share this inability to
observe a constant absolute shift. Therefore relative motion is retained for
WP28 recovery, but is not a FIX-selection confidence source.

## Next

Prioritize WP27 absolute satellite integrity and coherent-bias modeling:
multi-pivot DD consistency, satellite latent modes, or deployable map/road
evidence. Those can break the translation symmetry that TDCP cannot.
