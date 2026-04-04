| section | method | rms_2d_m | p95_m | outlier_rate_pct | catastrophic_rate_pct | time_ms_per_epoch | note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PPC holdout | Safe baseline | 66.92 | 81.69 | 5.83 | 0.0 |  | always_robust |
| PPC holdout | Exploratory gate | 65.54 | 81.22 | 5.83 | 0.0 |  | entry_veto_negative_exit_rescue... |
| UrbanNav external | EKF | 93.25 | 178.18 | 16.29 | 0.161 | 0.031 | trimble + G,E,J |
| UrbanNav external | PF-10K | 67.61 | 101.46 | 5.44 | 0.0 | 1.367 | trimble + G,E,J |
| UrbanNav external | PF+RobustClear-10K | 66.6 | 98.53 | 4.8 | 0.0 | 1.401 | trimble + G,E,J |
| UrbanNav external | WLS+QualityVeto | 2933.77 | 175.38 | 10.13 | 2.552 | 0.195 | promoted core hook |
| HK supplemental | EKF | 69.49 | 95.19 | 2.99 | 0.0 | 0.028 | ublox + G (GPS-only) |
| HK supplemental | PF+AdaptiveGuide-10K | 66.85 | 97.45 | 3.85 | 0.0 | 1.494 | ublox + G,C (adaptive guide) |
| BVH systems | PF3D-10K | 55.5 | 58.39 | 0.0 | 0.0 | 1028.29 | real PLATEAU subset |
| BVH systems | PF3D-BVH-10K | 55.5 | 58.39 | 0.0 | 0.0 | 17.78 | 57.8x faster |
