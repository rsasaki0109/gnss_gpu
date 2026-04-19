# 意思決定ログ

## D-001: observation policy を core に入れない

状態: 採用

理由:
- `PF+RobustClear` は安定だが局所最適
- `PF3D-BVH+BlockedOnly` は高利得だが不安定
- fixed full-mix は両者をうまく統合できていない

決定:
- observation policy は `python/gnss_gpu/` に入れず、`experiments/` で比較する
- core は expert 実装だけを持つ

## D-002: fixed full-mix を mainline 候補から外す

状態: 採用

根拠:
- [ppc_pf_blocked_clear_sweep_tokyo_run23_mix005_fine2_configs.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/ppc_pf_blocked_clear_sweep_tokyo_run23_mix005_fine2_configs.csv)
- `clear > 0` を入れた時点で Tokyo blocked-rich 区間の利得が消える

決定:
- `blocked + clear` の固定 simultaneous mixture は実験対象には残す
- ただし採用候補からは外す

## D-003: 現在の main experimental baseline は `always_robust`

状態: 採用

理由:
- full 6 positive segment では `PF+RobustClear` が最も安定
- 論文主張としても安全

決定:
- mainline で比較するときの基準は `PF+RobustClear`
- `always_blocked` は高利得枝として明示的に別管理する

## D-004: `disagreement_gate` は pilot winner だが採用候補にはしない

状態: 不採用

根拠:
- [pf_strategy_lab_t23_n2_summary.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/pf_strategy_lab_t23_n2_summary.csv)
- [pf_strategy_lab_positive6_summary.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/pf_strategy_lab_positive6_summary.csv)
- pilot 3 segment では `mean RMS 2D = 68.57 m`, `PF wins = 3/3`
- full 6 segment では `mean RMS 2D = 90.91 m`, `PF wins = 3/6`

理由:
- `disagreement_m` 単独では blocked-rich Tokyo と clean Nagoya を分け切れない
- `tokyo/run1`, `nagoya/run1`, `nagoya/run3` で false positive が多い

決定:
- `disagreement_gate` は「pilot で見つかった局所有望案」として残す
- 採用候補からは外す
- 次の探索では、false positive を抑える veto / persistence を追加する

## D-004b: `clock_veto_gate` は tuned full-6 winner だが holdout では採用しない

状態: 不採用

根拠:
- [pf_strategy_lab_positive6_summary.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/pf_strategy_lab_positive6_summary.csv)
- [pf_strategy_lab_holdout6_r200_s200_summary.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/pf_strategy_lab_holdout6_r200_s200_summary.csv)
- tuned full 6 では `73.72 m / 96.48 m` で `always_robust` を上回る
- holdout 6 では `74.02 m / 96.29 m` で `always_robust` の `66.92 m / 81.69 m` を下回る

理由:
- tuned dump に対する overfit の可能性が高い
- `nagoya/run2` のような unseen holdout で still false positive が出る

決定:
- `clock_veto_gate` は current gate family の exploratory variant として残す
- 採用候補からは外す
- baseline は引き続き `always_robust`

## D-005: richer gate は「複雑だから採用」しない

状態: 採用

理由:
- `rule_chain_gate` と `weighted_score_gate` は feature 数が多い
- full 6 では `rule_chain_gate` が `disagreement_gate` より良いが、それでも `always_robust` を超えない
- 可読性 proxy も悪い

決定:
- feature を増やした strategy は、精度が改善したときだけ残す
- 改善しない複雑化は捨てる

## D-006: ドキュメントは比較軸ごとに分ける

状態: 採用

決定:
- [experiments.md](/workspace/ai_coding_ws/gnss_gpu/docs/experiments.md): 実験結果
- [decisions.md](/workspace/ai_coding_ws/gnss_gpu/docs/decisions.md): 採用/不採用理由
- [interfaces.md](/workspace/ai_coding_ws/gnss_gpu/docs/interfaces.md): 現在の最小 interface

## D-007: 「正解実装」を作るのではなく dump と evaluator を先に作る

状態: 採用

理由:
- 同じ input dump を使えないと、variant 間の比較が壊れる
- 重い PF forward を毎回書き直すと探索速度が落ちる

決定:
- 先に [exp_ppc_pf_rich_gate_search.py](/workspace/ai_coding_ws/gnss_gpu/experiments/exp_ppc_pf_rich_gate_search.py) で feature/trajectory dump を作る
- その上で [evaluate_strategies.py](/workspace/ai_coding_ws/gnss_gpu/experiments/pf_strategy_lab/evaluate_strategies.py) が variant を比較する

## D-008: global baseline は `always_robust` を維持する

状態: 採用

根拠:
- [pf_strategy_lab_positive6_summary.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/pf_strategy_lab_positive6_summary.csv)
- `always_robust` は parameter tuning を要しない安全な baseline
- [pf_strategy_lab_holdout6_r200_s200_summary.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/pf_strategy_lab_holdout6_r200_s200_summary.csv)
- holdout 6 では `dual_mode_regime_gate`, `quality_veto_regime_gate`, `hysteresis_quality_veto_regime_gate` がわずかに上回るが、`always_robust` は parameter-free で再現しやすい

決定:
- 実験の global baseline は `always_robust` のまま維持する
- 新しい strategy はまずこの baseline を full validation で上回る必要がある

## D-009: pilot winner を採用根拠にしない

状態: 採用

理由:
- `disagreement_gate` は pilot では最良だったが full 6 では崩れた
- この差を明示しないと、探索プロセスが「都合のよい区間選び」に戻る

決定:
- pilot は候補生成にだけ使う
- docs の採用判断は full validation ベースで書く
- `experiments.md` では pilot と validation を明示的に分離する

## D-010: tuned full-6 winner も holdout を通るまで採用しない

状態: 採用

理由:
- `clock_veto_gate` は tuned full 6 では最良だったが holdout 6 では落ちた
- full validation だけでは still tuning leak を防げない

決定:
- 採用判断は `pilot -> tuned full -> holdout` の3段階で行う
- holdout を通らない strategy は exploratory のまま据え置く
- paper-facing method は holdout 通過済みのものだけ候補にする

## D-011: pure blocked-switch family は mainline 候補にしない

状態: 採用

根拠:
- [pf_strategy_family_cv_positive6_holdout6_family_best.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/pf_strategy_family_cv_positive6_holdout6_family_best.csv)
- `clock_veto_gate`, `disagreement_gate`, `weighted_score_gate` は holdout を超えない
- `clock_veto_gate` family も `disagreement_gate` family も holdout で baseline を超えない

決定:
- pure blocked-switch family は mainline 候補にしない
- 次に追加する family は blocked-switch の閾値微調整ではなく、別の設計思想にする
- family search は tuned/holdout 両方で回してから判断する

## D-012: `dual_mode_regime_gate` は first real holdout survivor として残す

状態: 条件付き採用

根拠:
- [pf_strategy_family_cv_positive6_holdout6_family_best.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/pf_strategy_family_cv_positive6_holdout6_family_best.csv)
- best config は holdout `65.62 m / 81.26 m`、baseline `66.92 m / 81.69 m` を上回る
- tuned では `80.12 m / 97.75 m` で baseline とほぼ同等だが、RMS は `+0.11 m` の微悪化

決定:
- `dual_mode_regime_gate` は first real holdout survivor として残す
- ただし current best family の座は固定しない
- 次の改善対象は `tokyo/run1` false positive の削減

## D-013: `quality_veto_regime_gate` を current best state-free family とする

状態: 中間採用

根拠:
- [pf_strategy_lab_positive6_summary.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/pf_strategy_lab_positive6_summary.csv)
- default config で tuned `80.02 -> 79.81 m`, holdout `66.92 -> 65.62 m`
- [pf_strategy_family_cv_positive6_holdout6_family_best.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/pf_strategy_family_cv_positive6_holdout6_family_best.csv)
- family best config は holdout `65.62 m / 81.26 m` を維持しつつ tuned `79.81 m / 97.75 m` まで改善

理由:
- `dual_mode_regime_gate` の close branch に `satellite_count` と `robust_p95_abs_residual` を加えるだけで、`tokyo/run1` の false positive を減らせた
- その結果、holdout survivor の性質を保ったまま tuned 微悪化を反転できた

決定:
- `quality_veto_regime_gate` は current best state-free family とする
- global baseline は still `always_robust`
- 次の探索は `quality_veto_regime_gate` を seed に続ける

## D-014: `hysteresis_quality_veto_regime_gate` を current best generalizing experimental family とする

状態: superseded

根拠:
- [pf_strategy_lab_positive6_summary.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/pf_strategy_lab_positive6_summary.csv)
- current representative config で tuned `80.02 -> 79.77 m`
- [pf_strategy_lab_holdout6_r200_s200_summary.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/pf_strategy_lab_holdout6_r200_s200_summary.csv)
- current representative config で holdout `66.92 -> 65.57 m`, `81.69 -> 81.22 m`
- [pf_strategy_family_cv_positive6_holdout6_family_best.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/pf_strategy_family_cv_positive6_holdout6_family_best.csv)
- family best config は `enter=1, exit=3, close_satellite_max=9, close_p95_abs_residual_max=55`

理由:
- `quality_veto_regime_gate` の candidate を stateful に保持すると、`tokyo_run3_seg974` のような blocked regime の短い off gap を bridge できる
- その結果、holdout と tuned の両方で `quality_veto` をさらに上回った

決定:
- current best generalizing experimental family は `hysteresis_quality_veto_regime_gate`
- `quality_veto_regime_gate` は state-free seed として残す
- 次の探索は richer temporal state を足す方向に進める
- この判断は後続の D-016 で `rescue_branch_aware_hysteresis_quality_veto_regime_gate` に更新された

## D-015: `branch_aware_hysteresis_quality_veto_regime_gate` を balanced exploratory family として残す

状態: superseded

根拠:
- [pf_strategy_lab_positive6_summary.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/pf_strategy_lab_positive6_summary.csv)
- current representative config で tuned `80.02 -> 79.55 m`
- [pf_strategy_lab_holdout6_r200_s200_summary.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/pf_strategy_lab_holdout6_r200_s200_summary.csv)
- current representative config で holdout `66.92 -> 65.58 m`, `81.69 -> 81.22 m`
- [pf_strategy_family_cv_positive6_holdout6_family_best.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/pf_strategy_family_cv_positive6_holdout6_family_best.csv)
- family best config は `enter_close=2, enter_far=1, exit_close=3, exit_far=5, close_satellite_max=9, close_p95_abs_residual_max=55`

理由:
- `close` false positive と `far` true positive は必要な persistence が違う
- enter だけでなく exit も branch ごとに分けると、holdout は `hysteresis` とほぼ同等のまま tuned をさらに改善できた
- ただし holdout-first selection では `hysteresis_quality_veto_regime_gate` の `65.57 m` にまだ届かない

決定:
- `branch_aware_hysteresis_quality_veto_regime_gate` は balanced exploratory family として残す
- holdout-first の current best generalizing family はこの時点では `hysteresis_quality_veto_regime_gate` を維持した
- 次の探索は `branch_aware` に close-singleton rescue を足す方向に進める

## D-016: `rescue_branch_aware_hysteresis_quality_veto_regime_gate` を current best generalizing experimental family とする

状態: superseded

根拠:
- [pf_strategy_lab_positive6_summary.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/pf_strategy_lab_positive6_summary.csv)
- current representative config で tuned `80.02 -> 79.53 m`
- [pf_strategy_lab_holdout6_r200_s200_summary.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/pf_strategy_lab_holdout6_r200_s200_summary.csv)
- current representative config で holdout `66.92 -> 65.57 m`, `81.69 -> 81.22 m`
- [pf_strategy_family_cv_positive6_holdout6_family_best.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/pf_strategy_family_cv_positive6_holdout6_family_best.csv)
- family best config は `enter_close=3, enter_far=1, exit_close=3, exit_far=5, rescue_sat<=8, rescue_p95<=50, rescue_cb>=16`

理由:
- `branch_aware` の弱点は holdout で有益な close singleton を取り逃すことだった
- close singleton をすべて復活させると train が崩れるが、clean singleton だけ rescue すると `hysteresis` と同じ holdout を維持しつつ train をさらに改善できた
- selection rule を holdout RMS -> holdout P95 -> train RMS の順で見ると、`rescue_branch_aware...` は `hysteresis` と同値の holdout で train がより良い

決定:
- current best generalizing experimental family は `rescue_branch_aware_hysteresis_quality_veto_regime_gate`
- `hysteresis_quality_veto_regime_gate` は simpler stateful baseline として残す
- `branch_aware_hysteresis_quality_veto_regime_gate` は intermediate exploratory family として残す
- 次の探索は `rescue_branch_aware` に negative-evidence か active-branch transition を足す方向に進める
- この判断は後続の D-017 で `negative_exit_rescue_branch_aware_hysteresis_quality_veto_regime_gate` に更新された

## D-017: `negative_exit_rescue_branch_aware_hysteresis_quality_veto_regime_gate` を current best generalizing experimental family とする

状態: superseded

根拠:
- [pf_strategy_lab_positive6_summary.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/pf_strategy_lab_positive6_summary.csv)
- current representative config で tuned `80.02 -> 79.47 m`
- [pf_strategy_lab_holdout6_r200_s200_summary.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/pf_strategy_lab_holdout6_r200_s200_summary.csv)
- current representative config で holdout `66.92 -> 65.54 m`, `81.69 -> 81.22 m`
- [pf_strategy_family_cv_positive6_holdout6_family_best.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/pf_strategy_family_cv_positive6_holdout6_family_best.csv)
- family best config は `enter_close=3, enter_far=1, exit_close=3, exit_far=5, rescue_sat<=8, rescue_p95<=50, rescue_cb>=16, neg_dis>=42, neg_cb>=25, neg_p95>=52, neg_hits=1`

理由:
- `rescue_branch_aware` は holdout gain を回復したが、`tokyo_run2_seg1008` では close rescue 後の false persistence がまだ残っていた
- active `close` が candidate を外れた epoch で `robust_p95_abs_residual` 系の negative evidence を見ると、その false persistence だけを切れる
- その結果、holdout `65.57 -> 65.54 m`, tuned `79.53 -> 79.47 m` と両 split で strictly 改善した

決定:
- current best generalizing experimental family は `negative_exit_rescue_branch_aware_hysteresis_quality_veto_regime_gate`
- `rescue_branch_aware_hysteresis_quality_veto_regime_gate` は simpler rescue baseline として残す
- 次の探索は negative-exit を richer evidence に広げるか、active-branch transition を明示化する方向に進める
- この判断は後続の D-018 で `entry_veto_negative_exit_rescue_branch_aware_hysteresis_quality_veto_regime_gate` に更新された

## D-018: `entry_veto_negative_exit_rescue_branch_aware_hysteresis_quality_veto_regime_gate` を current best generalizing experimental family とする

状態: 条件付き採用

根拠:
- [pf_strategy_lab_positive6_summary.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/pf_strategy_lab_positive6_summary.csv)
- current representative config で tuned `80.02 -> 79.41 m`
- [pf_strategy_lab_holdout6_r200_s200_summary.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/pf_strategy_lab_holdout6_r200_s200_summary.csv)
- current representative config で holdout `66.92 -> 65.54 m`, `81.69 -> 81.22 m`
- [pf_strategy_family_cv_positive6_holdout6_family_best.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/pf_strategy_family_cv_positive6_holdout6_family_best.csv)
- family best config は `enter_close=3, enter_far=1, exit_close=3, exit_far=5, rescue_sat<=8, rescue_p95<=50, rescue_cb>=16, close_entry_p95<=50, neg_dis>=42, neg_cb>=25, neg_p95>=52, neg_hits=1`

理由:
- `negative_exit_rescue_branch_aware` は `tokyo_run2_seg1008` の false close persistence を切れたが、`tokyo_run1_seg1463` では non-rescue close entry の 3-epoch false activation がまだ残っていた
- close sustain 条件そのものは holdout で効いているので崩したくない。一方で close entry だけを `robust_p95_abs_residual <= 50` に絞ると、その false activation だけを消せる
- その結果、holdout は `65.54 m / 81.22 m` で据え置きのまま、tuned は `79.47 -> 79.41 m` と strictly 改善した

決定:
- current best generalizing experimental family は `entry_veto_negative_exit_rescue_branch_aware_hysteresis_quality_veto_regime_gate`
- `negative_exit_rescue_branch_aware_hysteresis_quality_veto_regime_gate` は simpler negative-exit baseline として残す
- 次の探索は entry veto を `p95` 単独から richer evidence に広げるか、active-branch transition を明示化する方向に進める

## D-019: strategy gate の実装・探索フェーズを凍結する

状態: 採用

根拠:
- [pf_strategy_entry_veto_freeze_configs.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/pf_strategy_entry_veto_freeze_configs.csv)
- [pf_strategy_entry_veto_freeze_best.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/pf_strategy_entry_veto_freeze_best.csv)
- best neighbor は `exit_close=4, exit_far=6, close_entry_p95<=45..50, neg_p95>=52` で holdout `65.533 m`, tuned `79.345 m`
- current adopted representative は holdout `65.542 m`, tuned `79.406 m`

理由:
- final neighborhood sweep の gain は holdout `0.009 m`, tuned `0.061 m` で、promotion threshold `0.1 m` を下回る
- current family の近傍では改善が続いていても、論文主張や表の結論を変えるほどの差は出ていない
- これ以上の gate family 追加は、探索コストに対して paper / result 整理の機会費用が高い

決定:
- strategy gate の実装・探索フェーズはここで凍結する
- safe baseline は `always_robust`
- exploratory best は `entry_veto_negative_exit_rescue_branch_aware_hysteresis_quality_veto_regime_gate`
- 以後の作業は paper / figure / table / limitation 整理を優先し、新 family の追加は行わない
- 例外は holdout `0.1 m` 以上の改善見込みが事前に示せる場合だけとする

## D-020: UrbanNav は external validation として固定し、現時点では limitation を正直に出す

状態: 採用

根拠:
- [urbannav_fixed_eval_external_g_ublox_runs.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/urbannav_fixed_eval_external_g_ublox_runs.csv)
- [urbannav_fixed_eval_external_g_ublox_summary.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/urbannav_fixed_eval_external_g_ublox_summary.csv)
- Tokyo `Odaiba`, `Shinjuku`, rover `ublox`, systems `G`, no retuning
- aggregate では `EKF` が `mean RMS 2D = 74.56 m`, `mean P95 = 128.08 m`, `wins vs WLS = 2/2`
- `PF-10K` は `134.13 m / 281.77 m`, `PF+RobustClear-10K` は `136.47 m / 294.26 m` で `EKF` を超えない

理由:
- PPC だけでは strong accept に必要な external validity が弱い
- ただし UrbanNav で method を再 tuning すると external validation の意味がなくなる
- fixed setting の UrbanNav 結果は、現在の PF family が cross-dataset で一貫して勝つわけではないことを示している

決定:
- UrbanNav は external validation 専用セットとして固定する
- paper では PPC を design / ablation / holdout、UrbanNav を external validation と明示的に分ける
- UrbanNav の結果は「method limitation を含む honest result」として載せる
- 現時点では `EKF` を UrbanNav external baseline の best classical method とみなす
- `PF` と `PF+RobustClear` の UrbanNav 結果は main accuracy claim ではなく limitation / discussion に寄せる

## D-021: UrbanNav multi-GNSS は loader artifact を潰した上で diagnostic track に分離する

状態: 採用

根拠:
- [urbannav_fixed_eval_external_g_trimble_summary.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/urbannav_fixed_eval_external_g_trimble_summary.csv)
- [urbannav_trimble_pf_vs_ekf_diagnostics.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/urbannav_trimble_pf_vs_ekf_diagnostics.csv)
- [urbannav_trimble_tail_diagnostics.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/urbannav_trimble_tail_diagnostics.csv)
- [urbannav_trimble_common_epoch_wls_compare.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/urbannav_trimble_common_epoch_wls_compare.csv)
- UrbanNav trimble RINEX は dataset 自体が `G/R/E/J/C` を持つが、旧 loader は `C1C/S1C` 固定と sat-id 空白のため multi-GNSS 観測を自分で捨てていた
- loader 修正後、`Odaiba` 50 epoch で median sat `5 -> 14`、`Shinjuku` で `5 -> 9`
- common-epoch 比較では `G,E,J` が `P95` と `>100m率` を大幅に改善する一方、`RMS` は `~1.1-1.5 km` まで悪化する

理由:
- 以前の「UrbanNav は GPS-only しか使えない」という理解は dataset limitation ではなく measurement path の実装 limitation だった
- ただし、loader artifact を潰しても `G,E,J` はまだ stable result ではない
- 問題は gate family ではなく、multi-GNSS measurement / ISB / robust-estimation path にある

決定:
- UrbanNav `G,E,J` は main paper の external result ではなく diagnostic track として扱う
- main external comparison は引き続き `G` の fixed-eval を使う
- 次に external 側で投資する実装は、新 gate ではなく multi-clock / robust multi-GNSS measurement stabilization とする

## D-022: UrbanNav multi-GNSS の first promotable idea は residual/bias veto family

状態: 中間採用

根拠:
- [urbannav_multignss_stabilization_trimble_gej_summary.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/urbannav_multignss_stabilization_trimble_gej_summary.csv)
- [urbannav_multignss_stabilization_trimble_gej_best.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/urbannav_multignss_stabilization_trimble_gej_best.csv)
- best simple family は `multi_residual_bias_veto(residual_p95<=100, residual_max<=250, bias_delta<=100, extra_sat>=2)`
- common-epoch 平均で `gps_only 99.25 m / 173.38 m / 13.75% / 0.269%` に対し、best veto は `73.49 m / 100.97 m / 4.46% / 0.046%`
- 同じ common-epoch 上の `gps_ekf_reference` は `79.88 m / 148.88 m / 10.94% / 0.000%`
- `multi_raw` は `P95` を大きく改善するが `RMS` が `1342.42 m` まで壊れる

理由:
- UrbanNav multi-GNSS の問題は「multi が効かない」ことではなく、「少数の catastrophic epoch が raw solution を壊す」ことだった
- `solution_gap` 単独 veto より、measurement residual と ISB spread を直接見る veto の方が明確に強い
- best candidate は `use_multi_frac ≈ 99.3%` で、multi の利点をほぼ保ったまま極端な epoch だけ弾ける
- best veto は common-epoch では `EKF-G` も明確に上回る。ただし `>500m` を完全には 0 にできない

決定:
- UrbanNav multi-GNSS 側の next core candidate は `residual/bias quality veto`
- 逆に、UrbanNav external を改善するために複雑な PF gate family を増やす方針は採らない
- 次に昇格を検討するなら、experiment script の veto を `run_wls` または `MultiGNSSSolver` 周辺の最小 hook に落とす

## D-023: UrbanNav external main table は `trimble + G,E,J` に切り替える

状態: 採用

根拠:
- [multi_gnss_quality.py](/workspace/ai_coding_ws/gnss_gpu/python/gnss_gpu/multi_gnss_quality.py)
- [urbannav_fixed_eval_external_gej_trimble_qualityveto_runs.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/urbannav_fixed_eval_external_gej_trimble_qualityveto_runs.csv)
- [urbannav_fixed_eval_external_gej_trimble_qualityveto_summary.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/urbannav_fixed_eval_external_gej_trimble_qualityveto_summary.csv)
- fixed external `trimble + G,E,J` では `PF+RobustClear-10K = 66.60 m / 98.53 m / 4.80% / 0.000%`、`PF-10K = 67.61 m / 101.46 m / 5.44% / 0.000%`、`EKF = 93.25 m / 178.18 m / 16.29% / 0.161%`
- `WLS+QualityVeto` は raw `WLS` を少し改善するが、`2933.77 m / 175.38 m / 10.13% / 2.552%` に留まる

理由:
- 旧 `G`-only UrbanNav external では `EKF` が best だったが、これは loader artifact と GPS-only measurement path に強く制約されていた
- loader fix と `G,E,J` external rerun の後では、PF family が両 run で `EKF` を上回り、tail 指標でも優位になった
- `residual/bias quality veto` は “最小昇格抽象” としては正しいが、それ自体は best external method ではない

決定:
- main UrbanNav external table は `trimble + G,E,J` の fixed eval に切り替える
- `PF+RobustClear-10K` を current best external method、`PF-10K` を close ablation baseline とする
- `WLS+QualityVeto` は promoted core utility として残すが、main accuracy result の主役にはしない
- D-020 と D-021 の「UrbanNav external は EKF best / `G,E,J` は diagnostic only」という結論は、現時点では superseded とみなす

## D-024: paper packaging は fixed asset builder に集約する

状態: 採用

根拠:
- [build_paper_assets.py](/workspace/ai_coding_ws/gnss_gpu/experiments/build_paper_assets.py)
- [paper_main_table.md](/workspace/ai_coding_ws/gnss_gpu/experiments/results/paper_assets/paper_main_table.md)
- [paper_ppc_holdout.png](/workspace/ai_coding_ws/gnss_gpu/experiments/results/paper_assets/paper_ppc_holdout.png)
- [paper_urbannav_external.png](/workspace/ai_coding_ws/gnss_gpu/experiments/results/paper_assets/paper_urbannav_external.png)
- [paper_bvh_runtime.png](/workspace/ai_coding_ws/gnss_gpu/experiments/results/paper_assets/paper_bvh_runtime.png)

理由:
- strong accept に近づく局面では、新しい variant 追加より “何を main result として見せるか” の固定化が重要
- 現在の strongest line は `PPC holdout`, `UrbanNav trimble + G,E,J external`, `BVH systems`
- paper-ready asset を script 再生成にしておくと、本文更新や GitHub Pages 更新で数字の不一致を防げる

決定:
- paper 本文・GitHub Pages・図表作成は `build_paper_assets.py` の出力を基準にする
- `paper_main_table.md` を manuscript table の土台にする
- main figures は少なくとも `paper_ppc_holdout.png`, `paper_urbannav_external.png`, `paper_bvh_runtime.png` の 3 枚で固定する

## D-025: UrbanNav external breadth は fixed-window analysis で補強する

状態: 採用

根拠:
- [exp_urbannav_window_eval.py](/workspace/ai_coding_ws/gnss_gpu/experiments/exp_urbannav_window_eval.py)
- [urbannav_window_eval_external_gej_trimble_qualityveto_summary.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/urbannav_window_eval_external_gej_trimble_qualityveto_w500_s250_summary.csv)
- [urbannav_window_eval_external_gej_trimble_qualityveto_wins.png](/workspace/ai_coding_ws/gnss_gpu/experiments/results/urbannav_window_eval_external_gej_trimble_qualityveto_w500_s250_wins.png)
- 500 epoch / 250 stride の fixed window では、`PF+RobustClear-10K` が `EKF` に対して `RMS 90/127 win`, `P95 102/127 win`, `>100m 89/127 win`, `>500m 127/127 <=` を達成した
- `PF-10K` も近く、`RMS 88/127 win`, `P95 101/127 win`, `>500m 127/127 <=`

理由:
- main external table はすでに `PF+RobustClear-10K` 優位だが、run-average 2 本だけだと “lucky sequence” 批判が残る
- fixed-window analysis は new tuning を持ち込まず、既存 epoch dump から external robustness を細かく再評価できる
- geography の広さはまだ Tokyo-only だが、少なくとも gain が局所的な artifact ではないことを示せる

決定:
- main external claim は引き続き full-run table に置く
- ただし rebuttal / appendix / supplemental では window-level win-rate を併記し、「2 run mean だけ」という弱点を薄める
- “broad deployment-level generalization” はまだ主張しないが、“external gain is not concentrated in a single interval” は主張してよい

## D-026: Hong Kong 2019-04-28 は現時点では negative control として扱う

状態: 採用

根拠:
- [fetch_urbannav_hk_subset.py](/workspace/ai_coding_ws/gnss_gpu/experiments/fetch_urbannav_hk_subset.py)
- [urbannav_fixed_eval_hk20190428_g_ublox_summary.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/urbannav_fixed_eval_hk20190428_g_ublox_summary.csv)
- `HK_20190428`, `ublox`, `G`, 468 epochs では `EKF = 69.49 / 95.19 m`, `PF-10K = 301.68 / 560.12 m`, `PF+RobustClear-10K = 302.14 / 530.56 m`

理由:
- Hong Kong 2019-04-28 では利用可能 nav が GPS-only で、median 6 satellites の low-satellite regime になっている
- この設定では current PF family は Tokyo `trimble + G,E,J` のようには一般化しない
- したがって残る外部妥当性の弱点は “Tokyo run average only” ではなく、“multi-GNSS repaired regime を超えた geography transfer” である

決定:
- main paper table と main figure は引き続き Tokyo `trimble + G,E,J` external を使う
- Hong Kong 2019-04-28 は supplemental / limitation / future-work 向けの external control として保持する
- 次に geography を本当に広げるなら、Hong Kong 側でも mixed-nav path を確保するか、別の multi-GNSS external sequence を追加する

## D-027: `EKF anchor rescue` は safety variant として残すが mainline には昇格しない

状態: 採用

根拠:
- [urbannav_fixed_eval_hk20190428_gc_rescue_v2_summary.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/urbannav_fixed_eval_hk20190428_gc_rescue_v2_summary.csv)
- [urbannav_fixed_eval_external_gej_trimble_rescue_v2__odaiba__pfplusekfrescue_10k_runs.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/urbannav_fixed_eval_external_gej_trimble_rescue_v2__odaiba__pfplusekfrescue_10k_runs.csv)
- [urbannav_fixed_eval_external_gej_trimble_rescue_v2__odaiba__pfplusrobustclearplusekfrescue_10k_runs.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/urbannav_fixed_eval_external_gej_trimble_rescue_v2__odaiba__pfplusrobustclearplusekfrescue_10k_runs.csv)
- Hong Kong `G,C` では raw PF が `~48.6 km` 級で完全崩壊する一方、`PF+EKFRescue-10K` は `81.07 / 113.27 m`、`PF+RobustClear+EKFRescue-10K` は `81.26 / 113.27 m` まで戻る
- ただし Tokyo `Odaiba` では `PF-10K = 63.49 / 95.52 m`, `PF+RobustClear-10K = 61.86 / 94.12 m` に対して、rescue variant は `73.72 / 122.34 m`, `73.20 / 122.97 m` と悪化する

理由:
- `EKF anchor rescue` は geography transfer 時の catastrophic collapse を止める safety policy としては有効
- しかし current main external setting では unnecessary intervention が増え、Tokyo external の主結果を壊す
- したがって “global best PF method” ではなく “failure-recovery utility” として扱うのが妥当

決定:
- current main UrbanNav external method は引き続き `PF+RobustClear-10K`
- `PF+EKFRescue-10K` と `PF+RobustClear+EKFRescue-10K` は supplemental safety variant として保持する
- paper では、Hong Kong negative control を完全に捨てるのではなく、「raw PF collapse を止める補助 variant はあるが、mainline winner ではない」と正直に書く

## D-028: `PF+AdaptiveGuide-10K` は cross-geometry mitigation として残すが main table は置き換えない

状態: 採用

根拠:
- [urbannav_fixed_eval_external_gej_trimble_guide_policy_3k_summary.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/urbannav_fixed_eval_external_gej_trimble_guide_policy_3k_summary.csv)
- [urbannav_fixed_eval_hk20190428_gc_guide_policy_summary.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/urbannav_fixed_eval_hk20190428_gc_guide_policy_summary.csv)
- [urbannav_fixed_eval_external_gej_trimble_adaptive_3k_summary.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/urbannav_fixed_eval_external_gej_trimble_adaptive_3k_summary.csv)
- [urbannav_fixed_eval_hk20190428_gc_adaptive_summary.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/urbannav_fixed_eval_hk20190428_gc_adaptive_summary.csv)
- Tokyo 3k では `PF+EKFGuide-10K` が Odaiba に効くが Shinjuku を悪化させ、`PF+RobustClear+EKFGuideInit-10K` は Shinjuku を `66.50 / 96.66 m` まで改善する
- Hong Kong `G`-only control では `PF+EKFGuide-10K = 66.85 / 97.45 m` が raw PF collapse を避ける一方、`GuideInit` / `GuideFallback` は raw PF と同じく崩壊する
- `PF+AdaptiveGuide-10K` は single-constellation run で `PF+EKFGuide-10K`、multi-GNSS run で `PF+RobustClear+EKFGuideInit-10K` を選び、Tokyo 3k + Hong Kong の 3-run 平均を `64.22 / 92.72 m / 3.13%` まで改善する

理由:
- 弱点は「guide が効かない」ことではなく、「同じ guide policy を全 regime に固定すると壊れる」ことだった
- sparse single-constellation では reference velocity が本体で、multi-GNSS repaired regime では robust-clear + guide-init が safer
- ただしこの adaptive split はまだ Tokyo full external main table で再検証しておらず、current paper headline を置き換えるには早い

決定:
- `PF+AdaptiveGuide-10K` は cross-geometry weakness を減らす supplemental variant として保持する
- current main UrbanNav external method は引き続き `PF+RobustClear-10K`
- paper では `PF+AdaptiveGuide-10K` を “Tokyo mainline を置き換える headline method” ではなく、“Hong Kong collapse を raw PF without rescue よりずっと狭くする regime-aware mitigation” として書く

## D-029: full-run UrbanNav では `PF+AdaptiveGuide-10K` を mainline に昇格しない

状態: 採用

根拠:
- [urbannav_fixed_eval_external_gej_trimble_adaptive_full_summary.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/urbannav_fixed_eval_external_gej_trimble_adaptive_full_summary.csv)
- [urbannav_fixed_eval_external_gej_trimble_adaptive_full_runs.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/urbannav_fixed_eval_external_gej_trimble_adaptive_full_runs.csv)
- full-run では `PF+AdaptiveGuide-10K = 67.50 / 100.78 / 4.75% / 0.000%`、`PF+RobustClear-10K = 66.60 / 98.53 / 4.80% / 0.000%`
- `Odaiba` では adaptive が `61.68 / 94.85 / 3.14%` とほぼ同等だが、`Shinjuku` では `73.32 / 106.70 / 6.36%` となり `PF+RobustClear-10K` の `71.33 / 102.94 / 6.06%` を下回る

理由:
- 3k subset では見えた adaptive gain が full-run Shinjuku では維持されなかった
- つまり adaptive guide は cross-geometry mitigation としては有効だが、Tokyo main external setting の global winner ではない
- paper main table を差し替えるには、RMS と P95 の両方で `PF+RobustClear-10K` を明確に超える必要がある

決定:
- current main UrbanNav external method は引き続き `PF+RobustClear-10K`
- `PF+AdaptiveGuide-10K` は supplemental result として残す
- GitHub Pages / paper main assets は現状の `PF+RobustClear-10K` 主体のまま維持する

## D-030: Odaiba reference/guarded の DD carrier adaptive floor は `0.18`、stop-detect は `0.25` を維持する

状態: 採用

根拠:
- [exp_pf_smoother_eval.py](/workspace/ai_coding_ws/gnss_gpu/experiments/exp_pf_smoother_eval.py)
- [test_exp_pf_smoother_eval.py](/workspace/ai_coding_ws/gnss_gpu/tests/test_exp_pf_smoother_eval.py)
- `odaiba_reference`: `0.25` baseline は `FWD 1.46 / 5.57 m`, `SMTH 1.38 / 5.08 m`。`0.18` は `FWD 1.42 / 5.46 m`, `SMTH 1.38 / 5.02 m`。
- `odaiba_reference_guarded`: `0.25` baseline は `FWD 1.46 / 5.57 m`, `SMTH 1.38 / 5.43 m`。`0.18` は `FWD 1.42 / 5.46 m`, `SMTH 1.38 / 5.36 m`。
- `odaiba_stop_detect`: `0.25` baseline は `FWD 1.19 / 4.57 m`, `SMTH 1.36 / 4.11 m`。`0.18` は `FWD 1.63 / 5.50 m`, `SMTH 1.34 / 4.11 m`。

理由:
- coverage-hole 調査では、tracked fallback preference、ESS-only weak-DD replacement、spread-aware support-skip、contextual low-ESS epoch-median gate は full Odaiba の win にならなかった。
- reference/guarded では単純な adaptive floor tightening が forward と smoother RMS を少し改善した。
- stop-detect では smoother RMS は変わらない一方で forward quality が大きく悪化したため、headline best preset には入れない方が安全。

決定:
- `odaiba_reference` と `odaiba_reference_guarded` は `--mupf-dd-gate-adaptive-floor-cycles 0.18` を使う。
- `odaiba_stop_detect` は `--mupf-dd-gate-adaptive-floor-cycles 0.25` を維持する。
- weak-DD 調査で追加した extra knobs は default-off の ablation surface として残すが、full-run win なしに preset へ昇格しない。

## D-031: L1-L2 widelane DD pseudorange は default-off 実験 hook として残すが preset 昇格しない

状態: 採用

根拠:
- [widelane.py](/workspace/ai_coding_ws/gnss_gpu/python/gnss_gpu/widelane.py)
- [exp_pf_smoother_eval.py](/workspace/ai_coding_ws/gnss_gpu/experiments/exp_pf_smoother_eval.py)
- [test_widelane.py](/workspace/ai_coding_ws/gnss_gpu/tests/test_widelane.py)
- Odaiba 100 epoch smoke は `SMTH P50=0.80 m / RMS=0.79 m`、WL fixed pairs `568/600 (94.7%)`
- full Odaiba default WL は `SMTH P50=1.83 m / RMS=4.91 m`、WL used `8926/12252`, fixed pairs `50042/52665 (95.0%)`
- full Odaiba conservative WL は `SMTH P50=1.45 m / RMS=5.62 m`
- Shinjuku WL regression は `SMTH P50=3.07 m / RMS=9.13 m`
- non-WL `run_pf_smoother_odaiba_reference.sh` は `SMTH P50=1.34 m / RMS=4.11 m` で維持

理由:
- WL integer fix 自体の成立率は高いが、full-run の PF smoother median は改善しなかった。
- 現実装は GPS/QZSS L1-L2 fixed DD を epoch 単位で DD pseudorange path に差し替えるため、既存 raw DD pseudorange の Galileo constraints を落とす。
- sigma を弱めても current best `SMTH P50=1.14 m` には届かず、RMS/回帰条件が悪化する。
- smoke の submeter は局所区間の結果で、full Odaiba の headline 指標へ一般化しない。

決定:
- `--widelane` hook と `gnss_gpu.widelane` module は default-off の実験 surface として残す。
- `odaiba_best_accuracy` は変更しない。
- `odaiba_widelane` preset は作らない。
- 次に試すなら、epoch-level replacement ではなく row-level merge または additional likelihood として、Galileo raw DD constraints を落とさない設計にする。

## D-032: Region-aware widelane gate は full-run win なしで revert

状態: 不採用

根拠:
- [widelane_gate_odaiba_sweep.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/widelane_gate_odaiba_sweep.csv)
- [widelane_gate_validation.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/widelane_gate_validation.csv)
- Odaiba baseline `odaiba_best_accuracy`: `SMTH P50=1.1435 m / RMS=4.3627 m`
- handoff 推奨 dd17 + ratio5: `SMTH P50=1.4481 m / RMS=4.3590 m`, WL used `250/12252`
- WL 実使用ケースの最良 dd17 + ratio7: `SMTH P50=1.2454 m / RMS=4.9972 m`, WL used `223/12252`
- dd20 + ratio3/5/7 は WL used `0/12252` で baseline 同等
- Shinjuku dd17 + ratio7 は WL used `0/20127` で baseline と同一、`SMTH P50=2.286 m / RMS=7.548 m`
- `run_pf_smoother_odaiba_reference.sh` は `SMTH RMS=4.112 m` で `<=5.10 m` guard を維持

理由:
- 「強い DD epoch だけ WL 適用」の仮説は full Odaiba で current best を超えなかった。
- DD support を強くすると WL 適用 epoch が少なくなり、dd20 では完全に baseline へ戻る。
- DD support を緩めると WL は使われるが、P50 が `1.29-1.69 m` まで悪化する。
- 最も近かった dd17 + ratio7 でも current best から +0.10 m 以上悪く、submeter には届かない。

決定:
- `odaiba_widelane_gated` preset は作らない。
- region-aware WL gate CLI/logic/test は negative result として revert する。
- codex9 の base `--widelane` hook は D-031 のとおり default-off 実験 surface として残す。

## 現在の未決定事項

- `always_robust` と `entry_veto_negative_exit_rescue_branch_aware_hysteresis_quality_veto_regime_gate` を main paper でどう位置づけるか
- strategy 差分が出る epoch をどの figure で見せるか
- `blocked score` は完全に不要か、それとも veto 付きなら使えるか
- readability/extensibility proxy をどこまで信頼するか
