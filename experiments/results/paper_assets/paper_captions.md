# Paper Captions

## Table 1
Main quantitative summary used in the paper. PPC holdout is reported as a design-discipline result rather than a headline accuracy claim. UrbanNav external uses fixed `trimble + G,E,J` settings without UrbanNav-specific retuning. `PF+RobustClear-10K` is the strongest external method, improving mean RMS horizontal error from 93.25 m (`EKF`) to 66.60 m and mean p95 from 178.18 m to 98.53 m while reducing the >100 m rate from 16.29% to 4.80% and the >500 m rate from 0.161% to 0.000%. `WLS+QualityVeto` is shown as a promoted core utility, not as the main external method. BVH systems rows isolate runtime on a real PLATEAU subset and show unchanged PF3D accuracy with large acceleration.

## Figure 1
Segment-wise PPC holdout comparison between the safe baseline `always_robust` and the best exploratory gate `entry_veto_negative_exit_rescue_branch_aware_hysteresis_quality_veto_regime_gate`. The gain survives holdout but remains modest: mean RMS decreases from 66.92 m to 65.54 m and mean p95 decreases from 81.69 m to 81.22 m. This figure should be used to support the paper's experiment-discipline claim, not as the main accuracy result.

## Figure 2
UrbanNav external validation on Odaiba and Shinjuku using trimble observations and `G,E,J` measurements. Left: empirical CDF of 2D error. Right: rates of large failures above 100 m and catastrophic failures above 500 m. `PF+RobustClear-10K` achieves 66.60 m mean RMS and 98.53 m mean p95, outperforming `EKF` at 93.25 m and 178.18 m. Relative to `EKF`, the >100 m rate falls from 16.29% to 4.80% and the >500 m rate falls from 0.161% to 0.000%. `PF-10K` follows closely, while `WLS+QualityVeto` improves raw multi-GNSS WLS tails but remains far worse in RMS.

## Figure 3
Runtime and accuracy on the real PLATEAU PF3D subset. `PF3D-BVH-10K` preserves the same accuracy as `PF3D-10K` (55.50 m RMS, 58.39 m p95) while reducing runtime from 1028.29 ms/epoch to 17.78 ms/epoch, a 57.8x speedup. This figure should be framed as a systems contribution rather than a real-data accuracy gain from explicit 3D reasoning.

## In-Text Placement

- Table 1: first paragraph of the Results section as the paper-wide summary.
- Figure 1: PPC holdout ablation subsection.
- Figure 2: UrbanNav external validation subsection as the main accuracy figure.
- Figure 3: systems / implementation subsection for the BVH result.

## Supporting Numbers

- `PF-10K` UrbanNav external mean RMS/p95: 67.61 / 101.46 m.
- `WLS+QualityVeto` UrbanNav external mean RMS/p95: 2933.77 / 175.38 m.
