# WP26 Report — robust TDCP relative-motion evidence

## Verdict

**Motion primitive pass; RTK basin-selection gate fail.** The new causal TDCP
stream is millimeter-accurate on the measured window, but relative motion alone
cannot distinguish a correct absolute basin from a coherently translated one.

## Delivered

- robust slip-aware TDCP displacement and covariance;
- repaired PPC TDCP sign/max-speed API compatibility;
- external motion/covariance support in temporal ambiguity transitions;
- truth-free motion trace plus post-decision accuracy/NIS audit;
- synthetic slip rejection and motion-transition tests.

No FGO is used. The production FIX path is unchanged.

## Evidence

Tokyo run3/200 produced 199/199 accepted TDCP intervals. Median displacement
error was 2.8 mm and p90 was 5.6 mm. The solver rejected 161 individual
satellite rows using only post-fit consistency.

Nevertheless, TDCP replay selected zero of the 165 epochs containing a live
sub-50 cm basin. Correct and wrong persistent basin edges have nearly identical
motion residual distributions. This is expected for a coherent constant
position shift: differencing across time removes the absolute offset.

The result changes the campaign ordering. Doppler and IMU remain valuable for
propagation, outage bridging, and lineage survival, but adding more relative
sources cannot by itself solve the WP25 selection failure. WP27 must add an
absolute, truth-free discriminator before temporal gamma can become a FIX gate.

Artifacts:

- `csv/wp26_tdcp_run3_200_motion.csv`;
- `csv/wp26_tdcp_run3_200_summary.json`;
- `csv/wp26_tdcp_temporal_run3_200_scale1.json`;
- `csv/wp26_relative_motion_ablation.json`.
