# 意思決定ログ

> これは研究用の完全な意思決定ログ。README からリンクしているのは公開向けの要約
> [`docs/decisions.md`](../docs/decisions.md)（README の公開 contract に関わる判断のみ）。
> 両ファイルの D 番号は独立しており一致しない。他文書の素の D 番号（D-037 など）はこのファイルを指す。

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
- [ppc_pf_blocked_clear_sweep_tokyo_run23_mix005_fine2_configs.csv](gnss_gpu/experiments/results/ppc_pf_blocked_clear_sweep_tokyo_run23_mix005_fine2_configs.csv)
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
- [pf_strategy_lab_t23_n2_summary.csv](gnss_gpu/experiments/results/pf_strategy_lab_t23_n2_summary.csv)
- [pf_strategy_lab_positive6_summary.csv](gnss_gpu/experiments/results/pf_strategy_lab_positive6_summary.csv)
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
- [pf_strategy_lab_positive6_summary.csv](gnss_gpu/experiments/results/pf_strategy_lab_positive6_summary.csv)
- [pf_strategy_lab_holdout6_r200_s200_summary.csv](gnss_gpu/experiments/results/pf_strategy_lab_holdout6_r200_s200_summary.csv)
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
- [experiments.md](gnss_gpu/docs/experiments.md): 実験結果
- [decisions.md](gnss_gpu/docs/decisions.md): 採用/不採用理由
- [interfaces.md](gnss_gpu/docs/interfaces.md): 現在の最小 interface

## D-007: 「正解実装」を作るのではなく dump と evaluator を先に作る

状態: 採用

理由:
- 同じ input dump を使えないと、variant 間の比較が壊れる
- 重い PF forward を毎回書き直すと探索速度が落ちる

決定:
- 先に [exp_ppc_pf_rich_gate_search.py](gnss_gpu/experiments/exp_ppc_pf_rich_gate_search.py) で feature/trajectory dump を作る
- その上で [evaluate_strategies.py](gnss_gpu/experiments/pf_strategy_lab/evaluate_strategies.py) が variant を比較する

## D-008: global baseline は `always_robust` を維持する

状態: 採用

根拠:
- [pf_strategy_lab_positive6_summary.csv](gnss_gpu/experiments/results/pf_strategy_lab_positive6_summary.csv)
- `always_robust` は parameter tuning を要しない安全な baseline
- [pf_strategy_lab_holdout6_r200_s200_summary.csv](gnss_gpu/experiments/results/pf_strategy_lab_holdout6_r200_s200_summary.csv)
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
- [pf_strategy_family_cv_positive6_holdout6_family_best.csv](gnss_gpu/experiments/results/pf_strategy_family_cv_positive6_holdout6_family_best.csv)
- `clock_veto_gate`, `disagreement_gate`, `weighted_score_gate` は holdout を超えない
- `clock_veto_gate` family も `disagreement_gate` family も holdout で baseline を超えない

決定:
- pure blocked-switch family は mainline 候補にしない
- 次に追加する family は blocked-switch の閾値微調整ではなく、別の設計思想にする
- family search は tuned/holdout 両方で回してから判断する

## D-012: `dual_mode_regime_gate` は first real holdout survivor として残す

状態: 条件付き採用

根拠:
- [pf_strategy_family_cv_positive6_holdout6_family_best.csv](gnss_gpu/experiments/results/pf_strategy_family_cv_positive6_holdout6_family_best.csv)
- best config は holdout `65.62 m / 81.26 m`、baseline `66.92 m / 81.69 m` を上回る
- tuned では `80.12 m / 97.75 m` で baseline とほぼ同等だが、RMS は `+0.11 m` の微悪化

決定:
- `dual_mode_regime_gate` は first real holdout survivor として残す
- ただし current best family の座は固定しない
- 次の改善対象は `tokyo/run1` false positive の削減

## D-013: `quality_veto_regime_gate` を current best state-free family とする

状態: 中間採用

根拠:
- [pf_strategy_lab_positive6_summary.csv](gnss_gpu/experiments/results/pf_strategy_lab_positive6_summary.csv)
- default config で tuned `80.02 -> 79.81 m`, holdout `66.92 -> 65.62 m`
- [pf_strategy_family_cv_positive6_holdout6_family_best.csv](gnss_gpu/experiments/results/pf_strategy_family_cv_positive6_holdout6_family_best.csv)
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
- [pf_strategy_lab_positive6_summary.csv](gnss_gpu/experiments/results/pf_strategy_lab_positive6_summary.csv)
- current representative config で tuned `80.02 -> 79.77 m`
- [pf_strategy_lab_holdout6_r200_s200_summary.csv](gnss_gpu/experiments/results/pf_strategy_lab_holdout6_r200_s200_summary.csv)
- current representative config で holdout `66.92 -> 65.57 m`, `81.69 -> 81.22 m`
- [pf_strategy_family_cv_positive6_holdout6_family_best.csv](gnss_gpu/experiments/results/pf_strategy_family_cv_positive6_holdout6_family_best.csv)
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
- [pf_strategy_lab_positive6_summary.csv](gnss_gpu/experiments/results/pf_strategy_lab_positive6_summary.csv)
- current representative config で tuned `80.02 -> 79.55 m`
- [pf_strategy_lab_holdout6_r200_s200_summary.csv](gnss_gpu/experiments/results/pf_strategy_lab_holdout6_r200_s200_summary.csv)
- current representative config で holdout `66.92 -> 65.58 m`, `81.69 -> 81.22 m`
- [pf_strategy_family_cv_positive6_holdout6_family_best.csv](gnss_gpu/experiments/results/pf_strategy_family_cv_positive6_holdout6_family_best.csv)
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
- [pf_strategy_lab_positive6_summary.csv](gnss_gpu/experiments/results/pf_strategy_lab_positive6_summary.csv)
- current representative config で tuned `80.02 -> 79.53 m`
- [pf_strategy_lab_holdout6_r200_s200_summary.csv](gnss_gpu/experiments/results/pf_strategy_lab_holdout6_r200_s200_summary.csv)
- current representative config で holdout `66.92 -> 65.57 m`, `81.69 -> 81.22 m`
- [pf_strategy_family_cv_positive6_holdout6_family_best.csv](gnss_gpu/experiments/results/pf_strategy_family_cv_positive6_holdout6_family_best.csv)
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
- [pf_strategy_lab_positive6_summary.csv](gnss_gpu/experiments/results/pf_strategy_lab_positive6_summary.csv)
- current representative config で tuned `80.02 -> 79.47 m`
- [pf_strategy_lab_holdout6_r200_s200_summary.csv](gnss_gpu/experiments/results/pf_strategy_lab_holdout6_r200_s200_summary.csv)
- current representative config で holdout `66.92 -> 65.54 m`, `81.69 -> 81.22 m`
- [pf_strategy_family_cv_positive6_holdout6_family_best.csv](gnss_gpu/experiments/results/pf_strategy_family_cv_positive6_holdout6_family_best.csv)
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
- [pf_strategy_lab_positive6_summary.csv](gnss_gpu/experiments/results/pf_strategy_lab_positive6_summary.csv)
- current representative config で tuned `80.02 -> 79.41 m`
- [pf_strategy_lab_holdout6_r200_s200_summary.csv](gnss_gpu/experiments/results/pf_strategy_lab_holdout6_r200_s200_summary.csv)
- current representative config で holdout `66.92 -> 65.54 m`, `81.69 -> 81.22 m`
- [pf_strategy_family_cv_positive6_holdout6_family_best.csv](gnss_gpu/experiments/results/pf_strategy_family_cv_positive6_holdout6_family_best.csv)
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
- [pf_strategy_entry_veto_freeze_configs.csv](gnss_gpu/experiments/results/pf_strategy_entry_veto_freeze_configs.csv)
- [pf_strategy_entry_veto_freeze_best.csv](gnss_gpu/experiments/results/pf_strategy_entry_veto_freeze_best.csv)
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
- [urbannav_fixed_eval_external_g_ublox_runs.csv](gnss_gpu/experiments/results/urbannav_fixed_eval_external_g_ublox_runs.csv)
- [urbannav_fixed_eval_external_g_ublox_summary.csv](gnss_gpu/experiments/results/urbannav_fixed_eval_external_g_ublox_summary.csv)
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
- [urbannav_fixed_eval_external_g_trimble_summary.csv](gnss_gpu/experiments/results/urbannav_fixed_eval_external_g_trimble_summary.csv)
- [urbannav_trimble_pf_vs_ekf_diagnostics.csv](gnss_gpu/experiments/results/urbannav_trimble_pf_vs_ekf_diagnostics.csv)
- [urbannav_trimble_tail_diagnostics.csv](gnss_gpu/experiments/results/urbannav_trimble_tail_diagnostics.csv)
- [urbannav_trimble_common_epoch_wls_compare.csv](gnss_gpu/experiments/results/urbannav_trimble_common_epoch_wls_compare.csv)
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
- [urbannav_multignss_stabilization_trimble_gej_summary.csv](gnss_gpu/experiments/results/urbannav_multignss_stabilization_trimble_gej_summary.csv)
- [urbannav_multignss_stabilization_trimble_gej_best.csv](gnss_gpu/experiments/results/urbannav_multignss_stabilization_trimble_gej_best.csv)
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
- [multi_gnss_quality.py](gnss_gpu/python/gnss_gpu/multi_gnss_quality.py)
- [urbannav_fixed_eval_external_gej_trimble_qualityveto_runs.csv](gnss_gpu/experiments/results/urbannav_fixed_eval_external_gej_trimble_qualityveto_runs.csv)
- [urbannav_fixed_eval_external_gej_trimble_qualityveto_summary.csv](gnss_gpu/experiments/results/urbannav_fixed_eval_external_gej_trimble_qualityveto_summary.csv)
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
- [build_paper_assets.py](gnss_gpu/experiments/build_paper_assets.py)
- [paper_main_table.md](gnss_gpu/experiments/results/paper_assets/paper_main_table.md)
- [paper_ppc_holdout.png](gnss_gpu/experiments/results/paper_assets/paper_ppc_holdout.png)
- [paper_urbannav_external.png](gnss_gpu/experiments/results/paper_assets/paper_urbannav_external.png)
- [paper_bvh_runtime.png](gnss_gpu/experiments/results/paper_assets/paper_bvh_runtime.png)

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
- [exp_urbannav_window_eval.py](gnss_gpu/experiments/exp_urbannav_window_eval.py)
- [urbannav_window_eval_external_gej_trimble_qualityveto_summary.csv](gnss_gpu/experiments/results/urbannav_window_eval_external_gej_trimble_qualityveto_w500_s250_summary.csv)
- [urbannav_window_eval_external_gej_trimble_qualityveto_wins.png](gnss_gpu/experiments/results/urbannav_window_eval_external_gej_trimble_qualityveto_w500_s250_wins.png)
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
- [fetch_urbannav_hk_subset.py](gnss_gpu/experiments/fetch_urbannav_hk_subset.py)
- [urbannav_fixed_eval_hk20190428_g_ublox_summary.csv](gnss_gpu/experiments/results/urbannav_fixed_eval_hk20190428_g_ublox_summary.csv)
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
- [urbannav_fixed_eval_hk20190428_gc_rescue_v2_summary.csv](gnss_gpu/experiments/results/urbannav_fixed_eval_hk20190428_gc_rescue_v2_summary.csv)
- [urbannav_fixed_eval_external_gej_trimble_rescue_v2__odaiba__pfplusekfrescue_10k_runs.csv](gnss_gpu/experiments/results/urbannav_fixed_eval_external_gej_trimble_rescue_v2__odaiba__pfplusekfrescue_10k_runs.csv)
- [urbannav_fixed_eval_external_gej_trimble_rescue_v2__odaiba__pfplusrobustclearplusekfrescue_10k_runs.csv](gnss_gpu/experiments/results/urbannav_fixed_eval_external_gej_trimble_rescue_v2__odaiba__pfplusrobustclearplusekfrescue_10k_runs.csv)
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
- [urbannav_fixed_eval_external_gej_trimble_guide_policy_3k_summary.csv](gnss_gpu/experiments/results/urbannav_fixed_eval_external_gej_trimble_guide_policy_3k_summary.csv)
- [urbannav_fixed_eval_hk20190428_gc_guide_policy_summary.csv](gnss_gpu/experiments/results/urbannav_fixed_eval_hk20190428_gc_guide_policy_summary.csv)
- [urbannav_fixed_eval_external_gej_trimble_adaptive_3k_summary.csv](gnss_gpu/experiments/results/urbannav_fixed_eval_external_gej_trimble_adaptive_3k_summary.csv)
- [urbannav_fixed_eval_hk20190428_gc_adaptive_summary.csv](gnss_gpu/experiments/results/urbannav_fixed_eval_hk20190428_gc_adaptive_summary.csv)
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
- [urbannav_fixed_eval_external_gej_trimble_adaptive_full_summary.csv](gnss_gpu/experiments/results/urbannav_fixed_eval_external_gej_trimble_adaptive_full_summary.csv)
- [urbannav_fixed_eval_external_gej_trimble_adaptive_full_runs.csv](gnss_gpu/experiments/results/urbannav_fixed_eval_external_gej_trimble_adaptive_full_runs.csv)
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
- [exp_pf_smoother_eval.py](gnss_gpu/experiments/exp_pf_smoother_eval.py)
- [test_exp_pf_smoother_eval.py](gnss_gpu/tests/test_exp_pf_smoother_eval.py)
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
- [widelane.py](gnss_gpu/python/gnss_gpu/widelane.py)
- [exp_pf_smoother_eval.py](gnss_gpu/experiments/exp_pf_smoother_eval.py)
- [test_widelane.py](gnss_gpu/tests/test_widelane.py)
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
- [widelane_gate_odaiba_sweep.csv](gnss_gpu/experiments/results/widelane_gate_odaiba_sweep.csv)
- [widelane_gate_validation.csv](gnss_gpu/experiments/results/widelane_gate_validation.csv)
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

## D-033: Phase 1 per-particle NLOS rejection は preset 昇格しない

状態: 不採用

根拠:
- [per_particle_nlos_phase1_summary.csv](gnss_gpu/experiments/results/per_particle_nlos_phase1_summary.csv)
- [per_particle_nlos_phase1_sweep.csv](gnss_gpu/experiments/results/per_particle_nlos_phase1_sweep.csv)
- Odaiba baseline `odaiba_best_accuracy`: `SMTH P50=1.14 m / RMS=4.36 m`
- default Phase 1 (`undiff PR 30 m`, `DD PR 10 m`, `DD carrier 0.5 cyc`): `SMTH P50=64.07 m / RMS=91.98 m`
- specified all-gate sweep (`DD PR in {5,10,15,20} m`, `DD carrier in {0.3,0.5,0.7} cycles`): best `SMTH P50=59.62 m / RMS=63.12 m`
- no-undiff variant (`DD PR 10 m`, `DD carrier 0.5 cyc`): `SMTH P50=6.75 m / RMS=8.92 m`
- safest DD-carrier-only variant (`DD carrier 0.3 cyc`): `SMTH P50=1.38 m / RMS=4.61 m`
- Shinjuku DD-carrier-only regression: `SMTH P50=2.35 m / RMS=7.95 m`, below the `<9.5 m` RMS guard but not an Odaiba win
- `run_pf_smoother_odaiba_reference.sh`: `SMTH RMS=4.11 m`, existing reference guard maintained

理由:
- Per-particle rejection is not free: particles can explain different satellite subsets, which preserves diversity but also lets wrong modes survive in weak-DD sections.
- A minimum-inlier fallback is required to avoid reject-all particles, but even with that guard the Odaiba median degrades.
- DD carrier-only is stable enough as an ablation surface, but it does not beat current best or reach submeter.

決定:
- `--per-particle-nlos-gate` remains a default-off experimental hook.
- `odaiba_rbpf_nlos` preset は作らない。
- `odaiba_best_accuracy` は変更しない。
- 次に進むなら Phase 2 の per-particle velocity/RBPF 側で mode survival を制約する。

## D-034: Solver-state lightweight wrapper を opt-in research module として追加

状態: 採用 (interface only)

根拠:
- PR #42 (`docs: tokyo/run2 8.13pp residual is an accepted modeling ceiling`) が tokyo/run2 8.13 pp 残差の3つ取り組み (linear correction / hidden-high classifier / abstention) を null result としてクローズし、ceiling lift の3つの out-of-scope path を列挙した。
- `PLATEAU_BRIDGE_INTEGRATION.md` "Why the residual is not closeable": 1番目の path "Solver-state lightweight wrapper" は medium-size PR であり model input contract を変更するもの。
- 現状の product 契約 (`product_deliverable/README.md` §1) は `rtk_*` / `solver_demo5_*` を runtime feature から除外、6 列はすべて training target としてのみ使用。
- `experiments/_common._is_metadata_or_label` が `rtk_*` / `solver_*` を gate して runtime feature 入りを防いでいる。

決定:
- `experiments/solver_state_wrapper.py` を追加: 6列 (`solver_demo5_ratio_{mean,p90,p95,mean_past_delta}`, `rtk_lock_p90_p50{,_past_delta}`) の curated allowlist + `SolverStateWrapper.validate` / `curate` / `runtime_feature_columns`。
- `experiments/augment_window_csv_with_solver_state_wrapper.py` を追加: window CSV → curated subset を sanitise した CSV を出す CLI (research input prep).
- `tests/test_solver_state_wrapper.py` で allowlist / non-finite handling / default gatekeeper 不変性 を検証。
- 既存の `train_ppc_solver_transition_surrogate_stack.py` / `product_inference_model.py` は wrapper を import せず、deployed run-MAE 1.79 pp は維持。
- ceiling lift を実証する training script + LORO 評価は follow-up PR に分離 (target leakage 注意点を wrapper module docstring に記載済み)。

理由:
- Path 1 を opt-in interface として確保することで、follow-up での research path を ready にしつつ、deployed contract と adopted artefact の両方を変更しない。
- 既に existing 6 列が training target として材料化されているため、wrapper は新規 column 計算でなく allowlist + sanitiser に責任が限られ、medium-size PR として収まる。

## D-035: Path 1 LORO 評価で ceiling 解消、ただし predictor の役割が "post-demo5 QA" にシフト

状態: 採用 (parallel model class として; deployed §7.16 stack は併存)

根拠:
- `experiments/train_ppc_solver_state_wrapper_loro.py` で 197 windows / 6 routes の strict LORO を実行。3 variants (default GBR、no hyperparameter tuning):
  - `baseline_no_solver_state` (5873 features, no rtk_*/solver_*): run-MAE 12.39 pp / window-MAE 25.07 pp
  - `treatment_with_curated_six` (5879 features, baseline + curated 6): run-MAE **1.43 pp** / window-MAE **5.05 pp**
  - `curated_six_only` (6 features only): run-MAE **1.66 pp** / window-MAE 4.76 pp
  - 参考: deployed §7.16 + isotonic75 + phaseguard = run-MAE 1.79 pp / window-MAE 15.85 pp
- Tokyo/run2 (PR #42 で accepted ceiling とした 8.13 pp 残差): treatment 3.39 pp / `curated_six_only` **1.50 pp** で ceiling lifted。
- `curated_six_only` ≈ `treatment_with_curated_six` (差 0.23 pp)。残り 5873 features は signal 追加せず、curated 6 が dominant feature。
- 詳細: `internal_docs/product_deliverable/D034_PATH1_LORO_RESULTS.md`。

注意 (tautology):
- curated 6 は demo5 自身の ambiguity-fix-state confidence。"demo5 confidence → demo5 success" の予測は半 tautology であり、predictor は **post-demo5 QA tool** に役割が変わる (pre-demo5 estimator としては使えない)。
- 用途別影響: README §2 の offline-QA / route-ranking はどちらも demo5 を実行済み前提なので両立可能。real-time / online は不可。

決定:
- D-034 で ship した interface を活用する evaluation script `experiments/train_ppc_solver_state_wrapper_loro.py` + 結果 CSV `experiments/results/d034_loro_results.csv` + analysis doc `D034_PATH1_LORO_RESULTS.md` を follow-up PR (#44 想定) として ship。
- Path 1 を **parallel model class** として位置づける: deployed §7.16 stack (pre-demo5 estimator) は retire しない。post-demo5 QA は別の deliverable として将来追加可能。
- README §1 contract 文言は変更しない (default contract は依然 rtk_*/solver_* を runtime から除外)。新しい model class は wrapper を opt-in した派生として記録のみ。

理由:
- 1.66 pp run-MAE は deployed 1.79 pp を上回るが、tautology を踏まえると "別問題への新解" であって "同問題の改善" ではない。両モデルを並立させることで ambiguity を残さない。
- §7.16 stack の elaborate transition-surrogate scaffolding (6列を target として使う) は、6列が runtime feature でないことを前提にした正当な設計。Contract change で stack の存在意義が消えるわけではない (pre-demo5 use case は依然 stack のみ対応)。

## D-036: Path 1 (post-demo5 QA) を product deliverable bundle に opt-in 列として exposure

状態: 採用 (deployed §7.16 stack の側に並列、`build_product_deliverable.py --path1-prediction-csv` で opt-in)

根拠:
- D-035 で path 1 を parallel model class として採用したが、product deliverable bundle (`route_level_fix_rate_prediction.csv`, `window_level_details.csv`) への露出形式は未決定だった。
- 別ファイル (`post_demo5_qa_route.csv` 等) にすると operator が adopted と path1 を相互参照しづらくなる。同ファイルに `path1_*` 列を追加する形なら 1 行で adopted と post-demo5 QA の値を比較できる。
- Path 1 input (`d034_path1_per_window_predictions.csv`) が無いケース (deployed flow で `wrapper.curate(df)` を回していない場合) との互換性のため、`--path1-prediction-csv` を渡したときだけ列が増える `optional column` schema にする。

実装:
- `experiments/train_ppc_solver_state_wrapper_loro.py` に `--per-window-output-csv` flag を追加。`curated_six_only` variant の per-window 予測 (197 行) を `experiments/results/d034_path1_per_window_predictions.csv` に書き出す。
- `experiments/build_product_deliverable.py` に `--path1-prediction-csv` flag を追加。指定時のみ:
  - `window_level_details.csv` に `path1_pred_fix_rate_pct`, `path1_abs_error_pp` を追加
  - `route_level_fix_rate_prediction.csv` に `path1_pred_fix_rate_pct`, `path1_abs_error_pp`, `path1_lift_pp` (= `adopted_abs_error_pp - path1_abs_error_pp`、正なら path1 の方が正確), `path1_role_note` (固定文字列で role caveat) を追加
- 既存 deployed flow (`predict.py`) は `--path1-prediction-csv` を渡さないので、列も増えない (backward compatible)。

確認:
- 6 routes 中 5 routes で path1 lift が −0.04 〜 −2.34 pp (path1 の方が adopted より少し悪い)。
- Tokyo/run2 だけ path1 lift **+6.63 pp** (8.13 pp ceiling を 1.50 pp に圧縮)。focus windows w7/w9 (false_high)、w23-w25 (hidden_high) すべて 1-2 pp 以内に解消。
- 全 6 routes 平均 path1 abs error: 1.66 pp (D-035 報告値と一致)。

理由:
- 同ファイル列追加は ambiguity を最小化 (operator が `lift_pp` 列を見るだけで "ここでは path1 を信用すべきか" が判定できる)。
- `path1_role_note` 列で各 row に "post-demo5 QA tool であって pre-demo5 estimator ではない" という caveat を埋め込み、tautology を忘れさせない。
- `--path1-prediction-csv` を opt-in にすることで、real-time / online 用途の deployed flow は contract 変更なしで運用できる。

## D-037: per-particle 3DMA NLOS measurement likelihood は mainline に採用しない

決定: 不採用。陰影判定を各粒子で行う per-particle 3DMA 測定尤度は mainline lever から外し、elevation バイアス・仰角条件付き prior・重裾尤度は **opt-in capability** としてのみ保持する（既定は従来挙動）。ray tracing は feature / coverage / scenario（NLOS 率、C/N0 予測、DOP 地図）用途に限定する。

背景:
- GPU ビルド後、合成 PLATEAU シーンでは per-particle BVH 3DMA は oracle 相当（P50 0.59 m / RMS 0.70 m、hard 既定は P50 20 m）だった。
- ただしその合成観測は同じ elevation バイアス式で生成しており循環的。実データで検証したところ転移しなかった。

確認（実 UrbanNav Odaiba + PLATEAU 382,724 tri、G-only、300 epoch）:
| PF3D-BVH 設定 | RMS 2D |
|---|---|
| all-LOS (`blocked_nlos_prob=0`、素の PF と厳密一致) | **37.63 m** |
| soft+elev（Step 1） | 81.1 m |
| empirical（仰角 prior + Laplace） | 80.4 m |
| 対称 down-weight（バイアス無し） | 83.2 m |
| hard（旧既定） | 85.3 m |

- NLOS 枝をどう構成しても ~80–85 m に悪化。モデル再校正では直らない。
- 診断 `experiments/diag_nlos_label_and_bias.py`: レイラベル自体は健全（ray-LOS 92.3% clean、ray-NLOS 78.4% が excess≥5 m、低仰角 median 21.8 m / p90 149 m、高仰角 median 0 m で 89.4% が near-zero）。つまりラベル不良ではなくモデル/構造の問題。
- 根因: per-particle で遮蔽を判定すると「影に入る粒子ほど寛容な NLOS 枝で測定を説明できる」ため、粒子が geometry を gaming して影側へ収束する。`blocked_nlos_prob=0` が素の PF と厳密一致することがこれを裏付ける。
- 推定位置で 1 回だけレイ判定し全粒子へ共通適用する方式は既存 per-epoch mask（`nlos_pf_measurement_wiring.md`、PPC で Δ=0）と等価で、既に閉じている。

実装:
- `pf_weight_3d_bvh` に elevation 依存バイアス、仰角条件付き blocking prior、一般化ガウス（`nlos_beta=1` で Laplace）を追加。すべて opt-in、既定は従来挙動。
- `ParticleFilter3DBVH` の既定は本変更前の値に戻す（`sigma_nlos=30`, `nlos_bias=20`, `blocked_nlos_prob=1.0`, slope 0, `prob_high=-1`, `beta=2`）。
- 診断ツール: `experiments/diag_pf_3dma_likelihood.py`, `experiments/diag_nlos_label_and_bias.py`（真値ラベル検証・バイアス分布は再利用可能）。

理由: ラベルが健全でも per-particle 測定尤度は構造的に有害であり、CPU/PPC 側の既往の負の結果（D-033、`nlos_pf_measurement_wiring.md`）と整合する。単一 pilot の合成成功で既定を昇格させない（開発ポリシー）。

未決定: この負の結果を踏まえた pivot 先（OSS/ベンチマーク公開、学習誤差モデル、GSDC/PPC の伸びしろ）。候補の整理は末尾「現在の未決定事項」を参照。

## D-038: `experiments/gsdc2023_*` 共有モジュールは今はパッケージ化しない

決定: 保留（現状維持）。40 個の共有モジュール（`experiments/gsdc2023_*.py`、計 19,737 行）を `experiments/gsdc2023/` パッケージへ移す作業は行わない。D-001 に従い `python/gnss_gpu/` にも入れない。

判断材料（2026-10-03 計測）:
- 参照: tests / experiments から 335 import 文（ほぼ全て `experiments.gsdc2023_xxx` 形式）、107 本の `*_gsdc2023_*` スクリプト。
- tests は `import experiments.gsdc2023_raw_bridge as raw_bridge` 等のモジュールオブジェクトへ `monkeypatch.setattr` を 17 か所行っている。旧パスに re-export shim を置いて移動すると、patch は shim の属性だけを書き換え実装側に効かず、テストが無言で空振りし得る。安全に移すには 335 import の一括書き換えが必要。
- GSDC トラックは 2026-06 中旬以降ほぼ休眠（最終の機能 commit は 2026-07-02 の DD-carrier gate、best は v13 public 3.224 / private 3.783 m のまま）。移動の便益（発見性）に対し、churn と blame 断絶のコストが勝つ。

再開条件: GSDC トラックを再開するとき、または gsdc2023 モジュールを experiments 外（ライブラリ・別ツール）から使う必要が出たとき。その場合は shim ではなく import の一括書き換え＋ monkeypatch 対象の確認を 1 PR で行う。

## D-039: PF device のリサンプラを正しい Megopolis に切り替える（旧版は `megopolis_legacy`）

**更新（2026-10-03、同日）**: 既定を切替えた。`resampling="megopolis"` は Chesser et al. (2021) の正しい Megopolis（`megopolis_coalesced` は別名）、`megopolis_iterations` の既定は 60。旧カーネルは `"megopolis_legacy"` として旧結果の再現用にのみ残す。根拠: PPC 6 route（P50 −0.7 m、`resampler_ablation_ppc_2026_10_03.md`）と UrbanNav Odaiba PF smoother 複数 seed（前向き RMS −0.6 m、smoother 出力は noise 内で同等、`resampler_ablation_urbannav_2026_10_03.md`）で、noise を超える悪化が無く前向きは改善、かつ正しく 1M 粒子で速い（B=60 12 ms vs 旧 B=15 28 ms）。ホスト側 `ParticleFilter` の megopolis（`resampling.cu`）も同じ欠陥を持つが未評価のため変更していない。

以下は同日の当初判断（opt-in 追加時点）。


決定: 既定値は変えない。`resampling="megopolis_coalesced"`（Chesser et al. 2021）を opt-in として追加する。

判断材料（2026-10-03、合成重み N=4096、400 seed の子孫数平均と重みの total variation。下限は numpy 多項で 0.008–0.015）:
| 方式 | σ=1.5, B=15 | B=60 | B=240 | σ=3.0, B=240 |
|---|---:|---:|---:|---:|
| 既定 `megopolis` | 0.106 | 0.124 | 0.242 | 0.153 |
| `megopolis_coalesced` | 0.084 | 0.019 | 0.015 | 0.066 |

- 既定 `megopolis` は受理判定を「現在の祖先」ではなく「スロットの元の重み」と比べ、反復を増やすと偏りが**増える**。リサンプラとして重み分布に収束しない。
- `megopolis_coalesced` は B を増やすと noise floor に収束し、1M 粒子で既定より 3–13 倍速い（Turing、単精度の受理判定）。
- ただし既定を切り替えると PF の全結果が変わる。採用は holdout を通してから（D-009, D-010）。
- PPC 6 route 全区間での ablation（2026-10-03、`resampler_ablation_ppc_2026_10_03.md`）: `megopolis_coalesced` B=60 だけが noise 幅（P50 ±0.05 m）を超えて改善（PF+PU / RBPF+PU で P50 −0.65〜−0.72 m、RMS −0.8〜−1.0 m）。ただし route 差が大きい（tokyo3 −2.9 m、nagoya3 +0.17 m）、`<10m` は −0.3 pp、B=15 は既定と同等、seed 更新は効果なし。PF 単体の PPC 誤差は multipath 支配（P50 ≈31 m）で公式 score は動かず、正規 59.22% パイプラインはこのリサンプラを使わない。→ 既定切替の判断は README headline の UrbanNav PF smoother（既定 `megopolis`）で B=60 を検証してから。UrbanNav データは dev 機に無い。

同時に見つかった既存の問題（未修正、数値が変わるため別判断）:
- `pf_device_resample_systematic` は `u0=(seed%1e6)/(1e6·N)` をさらにカーネルで `/N` しており、起点が [0, 1/N²) に縮んで実質決定的・有偏（TV 0.06–0.17）。FFBSi 系 variant が使う。
- `ParticleFilterDevice._resample` は毎回同じ `self.seed` を渡すため、全 epoch のリサンプリングが同じ乱数列を使う。

## D-040: RB-FGO-PF の report floor を `nb >= 12` に上げる

**決定（2026-10-06）**: RB-FGO-PF（Tokyo PPC）で FIX として報告する下限を、解いた ambiguity 数 `nb >= 9` から `nb >= 12` に上げる。位置は変えず float に relabel するだけなので `<50cm_full%`（59.6/78.7/78.1）と PPC OFFICIAL は不変。false-fix 0.59/0.37/3.09% → 0.29/0.00/0.05%、FixRMS 0.104/0.121/0.150 → 0.062/0.042/0.059 m、fix 率 43.1/75.1/73.7 → 37.8/71.1/69.4%。

判断材料: WP18 は「run ごとに後付けで floor を選ぶのは cherry-picking」として `nb >= 11` を見送った。今回は保存済み WP18 full run の npz で leave-one-run-out（他 2 run で `kept good − λ·kept false` 最大の floor を選び残りで評価）を行い、run1/run2 を評価する場合は λ=5–20 で `nb >= 12`（λ=50 で 12/13）。run3 を評価する場合は run1/run2 に誤 FIX が 51 個しかなく λ ≤ 10 では `nb >= 9` のまま、λ=50 で `nb >= 13`。つまり「誤 FIX 1 個 = 正しい FIX 約 50 個」という integrity 重視の価値判断を含む。LAMBDA 単独 AR（`ar_raw_nb` / `ar_ratio`）を使う規則も探索したが floor 単独を上回らなかった。

未解決: run3 の 0.53 m ずれ区間は float 表示になっただけで位置は 0.53 m のまま。`<50cm` を上げるには runtime 側の変更が必要（WP19–37 の試みは不採用）。詳細 `inuex35_tc_fgo_benchmark.md`。

追記（2026-10-06, WP38）: FIX と FLOAT が食い違ったら FLOAT 位置を出す規則は不採用（leave-one-run-out で −8/0/+2 エポック）。区間内で FIX と FLOAT は中央値 0.09 m で一致しており、float も同じバイアスを持つ（誤差 0.48 m、50 cm 閾値のすぐ内側にいるだけ）。

## D-041: RB-FGO-PF の NLOS-AR 除外は既定にしない（地図あり 2 パスのオプション扱い）

**決定（2026-10-06, WP39）**: PLATEAU NLOS 衛星を AR 候補から外す設定（`NLOS_AR_EXCLUDE=1`）は既定にせず、地図あり 2 パスのオプションとして記録する。マスクは 1 パス目の推定位置でレイトレースしたものを使う（`NLOS_MASK_DIR`）。

判断材料:
- run3 の 0.53 m ずれの引き金は、E04（仰角 56° で SNR 27–29、DD コード約 4.4 m）と G09（L1 搬送波 +0.47 サイクル）の NLOS 化。既存の除外閾値をすり抜けていた。
- 既存の phase33 マスクは真値位置でレイトレースされていたので不採用。推定位置で作り直したマスクは真値マスクと 94–99% 一致する。
- 3 シードで見ると、OFFICIAL の改善は run1 +2.95 pp（全シードで正）、run2 +0.83、run3 +0.30。run2/run3 はシード差の範囲内。FixRMS は 1–3 cm、誤 FIX は 0.1–0.2 pp 悪化する。
- 0.53 m ずれ自体は出荷版シード（20260710）でしか起きない（誤 FIX 261 / 0 / 0）。README の RB-FGO-PF の数値はシード 1 本分で、±1–3 pp の幅がある。3 シード平均でも inuex35 は全 run で上回る。

今後: RB-FGO-PF の変種比較は複数シードで行う。

## D-042: RB-FGO-PF の出力を lineage ごとの追跡位置と IMU 推測航法の穴埋めにする（WP43）

**決定（2026-10-10）**: RB-FGO-PF の出荷構成（`experiments/rbpf_fgo/run_ship.sh`）を WP43 にする。重み・hold・FIX 判定は WP18 のまま。変えるのは出力の 2 点だけ:

- `RBPF_TRACK=2 RBPF_TRACK_EMPTY=1 RBPF_TRACK_OUT=2`:
  - 粒子 lineage ごとにアンテナ位置のガウス belief を持つ。
  - エポック間は共有グラフの IMU 予測移動量で予測する（random walk 0.02 m/epoch）。
  - その lineage の整数で DD 行を更新する。
  - MAP basin の belief を報告位置にする（FIX/FLOAT のラベルは不変）。
- `--dr-fill 1`: グラフが捨てた rover エポック（各 run の 5–9%）に、直前の報告位置 + IMU 予測移動量を出す。

判断材料:
- 3 シード（20260710/11/12）の公式 matched-3D スコア（`nb >= 12`）:
  - `<50cm_full%`: 60.10/78.91/81.86 → 63.07/82.65/86.65。
  - OFFICIAL: 57.13/82.22/83.47 → 63.68/85.08/86.94。
- 9 個の (seed, run) すべてで +2.3–4.8 pp 改善。libgnss++ GTSAM（56.8/80.5/72.8）を全 run で上回る。
- 寄与はほぼ加算的: 追跡位置だけで +1.4/+2.1/+2.6 pp、穴埋めだけで +1.3/+1.1/+2.1 pp。
- FIX 位置も少し良くなる（FixRMS 0.187/0.075/0.060 → 0.178/0.073/0.055 m、誤 FIX 0.47/0.17/0.11 → 0.33/0.16/0.10%）。
- ノブ（Q、floor、空 lineage の追跡）はシード 20260710 の 3 run で選び、残り 2 シードは 1 回だけ実行した。
- CPU 1 コアで 10.7 epoch/s（WP18 は 11.6、5 Hz の 2.1 倍）。

不採用:
- 追跡位置を重みにも入れる `RBPF_TRACK=1`: 粒子 × move ごとの batched solve、1 エポック約 400–500 回。FIX は増えるが、トンネル後に 0.72 m の誤 basin を固定して誤 FIX 2.39%（run1）。
- N = 256: run1 で精度は変わらず、run2 では 1.9 pp 悪化。PF 時間は 3.5 倍。

WP19–22/WP42 の cluster shadow（−3.1 pp）との違い: あれは記憶した位置を重みに入れ、しかも次エポックの事前分布を前エポックの絶対位置に置いていた（車両の移動を予測していない）。WP43 は IMU で予測し、重みには触れない。

GPU: この改善は計算量を増やさずに得られた。粒子を増やしても効かないので、GPU 化はまだ精度面の理由がない。

詳細: `inuex35_tc_fgo_benchmark.md` の WP43 節。

## D-043: RB-FGO-PF の出荷構成を WP45 にする（2 周波の衛星集合・basin 保持・LORO 選択の FIX ラベル）

**決定（2026-10-12）**: `experiments/rbpf_fgo/run_ship.sh` の既定を WP45 にする。目標は「誤 FIX（3D 誤差 0.5 m 以上）を各 run で FIX の 1% 以下に保ったまま、正しい FIX の割合を上げる」。3 run とも同じ構成で、WP43 の run1 専用復帰フラグは使わない。

- `--nf 2 --band-priority 'C:2,6;R:-'`: 1 衛星あたり 2 バンド、BeiDou は B1I + B3I、GLONASS は使わない。
  - `--nf 3` では、cssrlib の `qcedit` が 3 バンドのどれかが欠けた衛星を丸ごと捨てていた（DD は 2 バンドしか使わないのに）。L5 のない衛星と、B1C のない BeiDou（BDS-2 や大半の IGSO/GEO）が落ちていた。
  - run1 の取りこぼしエポックで、libgnss++ KF-RTK は中央値 15 衛星、共有グラフは 7–8 衛星だった。サンプルの 1 エポックで 15 → 25 衛星。
- `RBPF_KEEP=1`: グラフの再シードでは basin を消さない（slip / MW / CMC / 搬送波 FDE の release だけ消す）。
- `--fix-rule 3,5,3,12,0.1,5`: FIX ラベルは `nb_at[0.95] >= 5`、または `nb_at_ne[0.95] >= 12` かつ非空質量 0.1 以上、または共有グラフ自身の LAMBDA FIX（5 個以上）。どれも PF 出力ガードを通ったときだけ。
  - `fix_rule_loro.py` の leave-one-run-out で 3 fold とも同じ規則が選ばれた。
  - 実行時の判定はオフラインのラベルと全行で一致する。
  - 位置は WP43 の追跡 belief のまま。

判断材料（3 シード、同一採点）:
- 正しい FIX（rover エポック比）と誤 FIX（FIX 比）:
  - WP45: 66.6/83.9/84.3%、誤 0.56/0.09/0.04%（最悪シード 0.63/0.17/0.06%）。
  - WP43: 35.4/64.1/68.5%、誤 0.33/0.16/0.10%。
- OFFICIAL 80.8/92.7/90.4%、`<50cm_full%` 80.1/90.2/94.7。
- 比較対象（同一採点）:
  - libgnss++ KF-RTK 生出力: 68.0/76.7/71.6%（誤 8.76/1.80/3.01%）。その run 自身で誤 1% 以下にゲートした楽観値: 52.8/76.4/68.3%。
  - libgnss++ GTSAM FGO: 37.7/77.3/68.2%（誤 29.9/1.57/1.56%）。
  - inuex35: 46.7/59.7/58.2%（誤 5.72/1.78/2.06%）。
- 寄与の大半は衛星集合。旧ラベル（`nb >= 12`）のままでも run1 は 35.4 → 58.7%、誤 0.00%。`--nf 2` で B1C + B1I のままだと 44.1%。
- run1 の sanity リセットあり 67.1%（0.54%）となし 66.6%（0.56%）はシード差の範囲。同じフラグを run2/3 に入れると 0.1–2.3 pp 下がる。

不採用:
- `RBPF_TRACK=1` + リセット後 20 秒の猶予: run1 41.0%（誤 1.17%）。
- ambiguity function（報告位置で DD 搬送波の 95% 以上が整数から 0.15 cycle 以内）のラベル: 12 行以上で +0.3–1.6 pp。8–10 行では run3 の誤 FIX が 1.56%（保持行はそれ自身で解いた位置で必ず整数に乗るため、行数が少ないと誤った整数の組も通る）。診断として残す。
- 独立衛星による検証（海洋大 Fredeluces ら 2024）: 1 エポックあたり 0.4–1.7 衛星しかなく分離できない。libgnss++ も同じ理由で不採用にしている。
- 外部 KF-RTK を FIX の証人にする案: 51.7/74.8/74.5% になったが、PF ではなく 2 エンジン合議なので不採用。run1 の取りこぼし原因の特定にだけ使った。

詳細: `inuex35_tc_fgo_benchmark.md` の WP44–WP45 節。

## D-044: RB-FGO-PF の FIX ラベル規則を東京＋名古屋で選び直す（WP46）

**決定（2026-10-12）**: `run_ship.sh` の既定を WP46 にする。フィルタと位置は WP45 のまま、ラベル規則だけを変える。

- 新しい規則（`--fix-rule 3,15,3,15,0.1,99`）: `nb_at[0.95] >= 15`、または `nb_at_ne[0.95] >= 15` かつ非空質量 0.1 以上。共有グラフ自身の FIX 項は使わない。
- `PPC_CITY=nagoya` で名古屋のレバーアーム（FLU 0.593, 0.670, 1.216 m）を入れる。

背景:
- WP45 の規則は東京 3 run だけで選んでいて、名古屋では誤 FIX が 0.70 / 8.53 / 16.2% だった。
  - 主因は `nb_at[0.95] >= 5` の項（名古屋 2/3 でその項が付けた FIX の 9.7 / 11.8% が誤り）。
- 名古屋の DD 残差が東京の約 3 倍あった原因も調べた。電離層・対流圏・基準局座標・基準局の時刻ずれではなかった。参照点がアンテナ位相中心から車体固定で約 4 cm ずれているだけで、0.5 m 判定には影響しない。

判断材料（leave-one-run-out、東京 1–3 と名古屋 1–2、名古屋 3 は除外、3 シード、訓練上限 0.7%）:
- 5 fold 中 4 fold が同じ規則を選んだ。
  - 名古屋 2 を外した fold だけは、訓練に難しい run がないので WP45 の規則を選んだ（名古屋 2 で誤 8.53%）。
- WP46 の正しい FIX:
  - 東京 56.1/76.4/75.8%、名古屋 61.0/46.5/12.3%。
  - 誤 FIX は 0.02/0.02/0.00/0.03/0.21/6.15%（最悪シード 0.37%、名古屋 3 を除く）。
- 上限 1.0% で選ぶ規則（`nb_at[0.98] >= 12` ...）は東京で 3 pp 高い。ただし名古屋 2 の最悪シードが 1.45% になるので採らない。
- 代償: 東京の正しい FIX は WP45 より 8–10 pp 下がる。位置・OFFICIAL は変わらない。

保留（既定 off）:
- `QC_BASE_MASK=1`: 基準局側の QC にローバーのマスクを入れる。これまで `ELMIN_DEG` を 15° 未満にしても効いていなかった。
- `QC_ANY_BAND=1`: 1 バンドだけ使える衛星も残す。
- どちらも 1 シードでは混在した結果だった。名古屋はシード間で 7 pp ぶれる。

未解決: 名古屋 3（栄）は最も厳しいラベルでも誤 6% で、フィルタ側の対策が必要。

## D-045: RB-FGO-PF の出荷構成を WP47 にする（1 バンド衛星を code のみで使う・10° マスク・6 run で選んだラベル）

**決定（2026-10-13）**: `run_ship.sh` の既定を WP47 にする。

- 1 バンドしか使えない衛星も残す。ただし code のみ（`QC_ANY_BAND=1 QC_ANY_BAND_CODE_ONLY=1`）で、DD-PR の sigma は 10 倍（`QC_ANY_BAND_PR_SCALE=10`）。DD の基準衛星には選ばない。
- ローバーと基準局の両方に 10° マスク（`QC_BASE_MASK=1 ELMIN_DEG=10`）。
- ラベル規則 `--fix-rule 5,15,3,99,0.1,9`: `nb_at[0.99] >= 15`、または共有グラフ自身の LAMBDA FIX（9 個以上）。

背景:
- 名古屋 3（栄）では 2 バンドが揃う衛星が少なく、共有グラフが解けないエポックが多かった（PF 有効 79%）。1 バンド衛星を足すと 94% に上がる。
- 1 バンド衛星を足すと東京 1 が最大 12 pp 下がっていた。原因は衛星ごとの DD-PR 残差の付け方。ペアの残差を基準衛星にも付けていたので、マルチパスの大きい 1 バンド衛星がいると全バンドの基準衛星がゲート／隔離され、その衛星系の搬送波ペアが全部落ちた。1 バンド衛星を含むペアの残差はその衛星にだけ付けるように直した。`QC_ANY_BAND` が off なら挙動は変わらない。
- 10° マスクなしだと、東京 2 の難所で浮動解が数分発散するシードがある（誤 FIX 2.40 / 1.56%）。マスクありでは起きない。

判断材料（6 run × 3 シード）:
- 正しい FIX: 東京 56.9 / 78.0 / 76.1%、名古屋 65.4 / 46.3 / 26.3%（WP46: 56.1 / 76.4 / 75.8 / 61.0 / 46.5 / 12.3%）。
- 誤 FIX はすべての run・シードで 1% 以下（最悪は名古屋 2 の 0.99%）。WP46 は名古屋 3 で 6.15%。
- PPC OFFICIAL: 名古屋 3 で 20.2 → 71.0%。東京 3 は 90.4 → 88.3%、東京 2 は 92.7 → 91.8% に下がる。
- WP46 のフィルタのままではどのラベル規則でも名古屋 3 の誤 FIX が約 5% 残るので、フィルタ側の変更が必要。

注意:
- ラベルは 6 run の in-sample 選択。leave-one-run-out では名古屋 2 の fold だけが緩い規則を選んで誤 3.5–4.7% になる。名古屋 2 の 556840–556862 s に、浮動解ごと約 1 m ずれた自己整合な basin があり、今の特徴量では分けられない（WP46 でも一部で誤 FIX になる）。

## D-046: RB-FGO-PF の共有グラフの測定モデルを改善する（WP48）

**決定（2026-10-13）**: `run_ship.sh` の既定を WP48 にする。

- ドップラー速度事前分布を作り直す（`DOP_V2=1 DOPPLER_LEVER_COMP=1 DOP_COV_SCALE=1.5`）。GLONASS と BeiDou GEO を除き、C/N0 で重み付けし、0.5 m/s を超える残差を棄却し、解の共分散を使い、レバーアームの回転項を差し引く。
- DD 擬似距離と DD 搬送波に Huber 損失（`HUBER_PR=1.345 HUBER_CP=2.0`）。
- 擬似距離の分散に C/N0 項（`PR_CNR_ERR=0.5`）。
- ラベル規則 `--fix-rule 5,15,5,15,0.1,5,1.0`（7 番目は新設の DDPR 品質による拒否）。

背景:
- 東京では 5 cm 以内のエポックはほぼすべて FIX ラベルが付いていた。ラベル側の改善（割合ルール、学習した分類器）は held-out で +0.3〜1.4 pp か、かえって悪化した。
- BeiDou GEO のドップラーは参照速度から 3〜22 m/s ずれていて、ドップラー速度を上下に約 0.4 m/s 偏らせていた。事前分布はレバーアームの回転項なしで機体速度に入っていた。直した後の水平誤差 p95 は 2.0 → 0.4〜0.5 m/s。

判断材料（6 run × 3 シード）:
- 正しい FIX: 東京 56.4 / 78.7 / 74.8%、名古屋 66.5 / 47.8 / 27.4%（WP47: 56.9 / 78.0 / 76.1 / 65.4 / 46.3 / 26.3%）。
- 誤 FIX は全 run・全シードで 0.7% 以下（最悪 0.62%）。WP47 は最悪 0.99%。
- PPC OFFICIAL: 東京 84.3 / 93.8 / 91.8%、名古屋 74.5 / 56.1 / 78.8%（WP47 より 1〜8 pp 高い）。
- 上限を 1 シード 1.0% にすると平均 +1.1 pp だが、最悪 0.99% になるので採らない。

注意: leave-one-run-out では名古屋 2 の fold が緩い規則を選び、誤 5.5% になる。名古屋 2 の約 0.95 m ずれた自己整合 basin は未解決。

## D-047: ワイドレーン FIX は出荷しない（WP49、否定結果）

**決定（2026-10-13）**: ワイドレーン（WL）の FIX は FIX ラベルにも別区分にも使わない。診断オプション `RBPF_WL` / `RBPF_WLPF`（既定 off）だけ残す。

根拠（6 run、seed 20260710、WP48 構成）:
- WL 位置が 0.5 m 以内のエポックは FIX ラベルより 7〜30 pp 多く、伸びしろはある。
- 1 エポックごとの WL 比検定では正しい WL 整数と誤った整数を分けられない（誤 1〜12%、名古屋 2 は最大 48%）。
- WL の整数仮説を時間方向に保持する 2 本目の PF（`RBPF_WLPF`）でも、cm 級の PF が解けない区間では WL 基盤も同じくらい誤る。誤 1% 以下で足せるのは +1 pp 前後。
- 名古屋 2 の約 0.95 m ずれた誤 basin は WL 位置でも拒否できない（WL 位置も同じ向きにずれ、差の中央値 0.23 m）。
- 原因はどれも、マルチパスで共有の浮動解自体が偏ること。3D 地図（PLATEAU）による NLOS 除去はこの環境から建物データに届かないため未検証。

## D-048: FIX ラベルにヒステリシスと整数の持ち越しを入れる（WP50）

**決定（2026-10-13）**: `run_ship.sh` の既定を WP50 にする。フィルタは WP48 のまま、ラベルの層だけを変える。

- ヒステリシス（`--fix-hyst 5,2,2,6,2,6,0.1,5`）: WP48 の規則が 5 エポック続いたら FIX に入り、0.90 超の合意ペアが 6 組以上ある間は FIX を続ける（2 エポックまでの途切れは許す）。
- 前方の整数持ち越し（`--carry 1`、`RBPF_CARRY=1`）: ラベルが外れたエポックでも、直前の FIX の一重差整数が生の DD 搬送波に合う間（8 ペア以上、0.15 サイクル以内）は、その整数で解いた位置を FIX として出す。
- 後方の持ち越し（`integer_smoother.py`、15 秒固定遅延）は別出力として提供する。既定の出力は実時間のまま。

背景:
- 誤った basin は真の軌跡を一定量ずらしたもので、配置が変わらない間は搬送波で見分けられず、擬似距離も同じ向きに偏る。libgnss++ の WP176 も同じ壁に当たっている（東京 1 54.4%、名古屋 1 69.4%）。
- 誤 FIX は短いちらつきで起き、正しい整数はロックが続く限り有効という、時間方向の情報を使っていなかった。

判断材料（6 run × 3 シード）:
- 実時間: 東京 61.8 / 82.5 / 77.9%、名古屋 70.4 / 51.1 / 29.7%（WP48 より 2.3〜5.4 pp 高い）。全シードで誤 FIX 0.20% 以下（WP48 は最悪 0.62%）。
- 15 秒遅延: 65.3 / 85.0 / 81.1 / 74.6 / 54.2 / 38.4%。
- 持ち越しの設定は決め打ちで、変えても結果はほぼ変わらない。持ち越し元を WP48 の規則のままにすると誤 FIX が伝播する（名古屋 2）。ヒステリシスのラベルを元にすると、誤った元からの持ち越しは 0。

注意: ヒステリシスの規則は 6 run の in-sample 選択。leave-one-run-out では名古屋 2 の fold だけが入り口 1 エポックを選んで誤 2.6% になる。1 秒の入り口は設計として固定する。

訂正（2026-10-13、WP51）: 上の数値は、持ち越したエポックにも PF の出力ガード（報告位置がコードのみの位置から 5 m 超）をかけて採点していた。`.pos` 出力はそのエポックも FIX のまま出す。出力どおりに採点すると、実時間 63.1 / 82.8 / 78.8 / 70.6 / 52.2 / 31.0%、15 秒遅延 67.3 / 85.7 / 82.5 / 75.3 / 55.4 / 41.6%。誤 FIX 率の差は 0.01 pp 以内。

## D-049: 後処理出力に逆向きの PF パスを足す（WP51）

**決定（2026-10-13）**: 後処理（全区間を使える）出力として、同じ設定のフィルタを時間反転したデータで逆向きに走らせ、前向きの出力と組み合わせる（`--reverse 1`、`combine_fb.py`）。実時間の既定（WP50）と 15 秒遅延の出力は変えない。

- 時間反転: 時刻を τ = t_first + t_last − t に写し、ドップラー・衛星速度・ジャイロの符号を反転、衛星の位置と時計は本当の送信時刻で計算、LLI はその衛星の一つ前の観測へ移す。フィルタ本体は前向きのまま。
- 組み合わせ: (1) 前向きのラベルから遅延制限なしの後方持ち越し、(2) 逆向きの FIX 区間は、(1) のラベルが 0.3 m 以内で 5 エポック以上一致し、食い違いが一度もないときだけ加える（食い違いは前向きを優先）、(3) 全 FIX エポックから前後へ持ち越し。

判断材料（6 run × 3 シード、出力どおりの採点）:
- 正解 FIX: 69.3 / 87.1 / 83.6 / 79.9 / 56.8 / 55.8%（15 秒遅延より 1.1〜14.2 pp 高い。名古屋 3 が最大）。誤 FIX は全シードで 0.35% 以下。
- 逆向きのパス自体の寄与は 0.5〜1.1 pp、名古屋 3 は 8.9 pp。残りは遅延制限なしの持ち越しと、持ち越したエポックからの再持ち越し。
- PPC OFFICIAL も 0.2〜2.1 pp 上がる。

却下した案:
- 逆向きの FIX を確認なしで足す: 東京 3 で誤 1.7%、名古屋 2 で 1.1%。
- 前向きと逆向きの PF の FIX が数 cm で一致すれば FIX: 名古屋 2 で誤 4.5〜4.8%。両方向が同じ誤った basin に収束するため、二つのフィルタの一致は根拠にならない。
- 二つの float 位置の平均: 0.5 m 以内の割合が下がる run がある。

注意: 逆向きのデータでは、前向きで選んだラベル規則の誤 FIX が名古屋 1・2 で 0.8〜1.2%、東京 3 の 1 シードで 5.3% になる。規則の held-out での誤り率は in-sample の 0.2% より高いと見込むべき。計算量は 2 倍。

## D-050: 整数の持ち越しを共有グラフに戻す案は採らない（WP52、否定結果）

**決定（2026-10-13）**: `RBPF_CARRY_FB`（持ち越した整数の組を共有グラフに hold として入れ、持ち越しの間は PF の質量による解放を止める）は既定オフのまま出荷しない。

判断材料（シード 20260710）:
- 東京 1: 63.4 → 63.6%（誤 0.05 → 0.09%）、名古屋 2: 52.4 → 52.6%（誤 0.12 → 0.48%）、名古屋 3: 32.0 → 29.0%（FIX 時にも入れる版も 29.0%）。
- 前向きの持ち越しは全エポックの 2〜3.5%、長さの中央値 2 エポック（0.4 秒）しか続かない。難しい区間では持ち越し自体がすぐ切れるので、グラフを支える機会がほとんどない。

## D-051: 停車中は FIX に入らない案は採らない（WP53、否定結果）

**決定（2026-10-13）**: 名古屋 2 の誤った basin は 45 秒の停車中に選ばれる（停車中はコードのマルチパスが一定で、配置もほとんど変わらない）。IMU で停車を検出してヒステリシスの入り口を止める案をラベル再生で試したが採らない。

- 停車は全 run のエポックの 33〜55%。名古屋では正解 FIX が 5〜15 pp 落ちる（名古屋 1: 70.6 → 55.4%）。
- 名古屋 2 の誤 FIX は減らない（最悪 0.20 → 0.26%）。誤った basin は停車中に選ばれ、FIX のラベルが付くのは動き出してから。

## D-052: 後処理出力の float 位置を搬送波の整数適合で選ぶ（WP54）

**決定（2026-10-13）**: `combine_fb.py` の既定に、FIX にならないエポックの位置選択を加える。候補は前向き・逆向き・その平均・前向きの float・逆向きの float の 5 つ。生の DD 搬送波が整数サイクルにどれだけ合うか（ambiguity function、前後 ±5 エポックで平均）が最も高い候補を出す。ただし前向きより 0.1 以上良いときだけ入れ替える。FIX のラベルと位置は変えない。

判断材料（6 run × 3 シード、PPC OFFICIAL）:
- 92.0 / 95.6 / 93.0 / 79.5 / 64.2 / 90.9%、平均 85.9%（WP51 は 81.5%）。PPC2024 優勝（TURING）の 85.6% と同水準（D-057 で訂正: 後処理は大会ルール外）。シード間のばらつき（名古屋 1 で ±3 pp）を考えると「上回った」とは言わない。
- 学習不要で真値を使わない。試した 9 通りの設定はすべて 85.1〜85.8% の範囲。
- 学習器（leave-one-run-out）でも 85.2%、整数適合を特徴量に足すと 86.3%。差は小さく、学習データに依存しない規則を採る。
- 候補の中から理想的に選べたときの上限は 90.1%。

## D-053: 実時間でも位置を搬送波の整数適合で選ぶ（WP55）

**決定（2026-10-14）**: `run_ship.sh` の既定を WP55（WP50 + `--af-select 6`）にする。FIX にならないエポックで、報告位置・グラフの float・PF の MAP / 混合の追跡位置のうち、直近 6 エポックで搬送波に最もよく合う位置を出す（報告位置より 0.1 以上良いときだけ入れ替え）。未来のデータは使わない。FIX のラベルとフィルタは変えない。

判断材料（6 run × 3 シード、PPC OFFICIAL）:
- 実時間 85.5 / 93.7 / 92.3 / 77.2 / 57.5 / 83.7%、平均 81.6%（WP50 は 79.9%）。東京 2 だけ −0.1 pp（シード差の範囲）。
- 実時間で使える候補から理想的に選べたときの上限は 84.0%。窓と閾値を変えた 7 通りはすべて 81.7〜82.1%。
- 実行中の選択はオフラインの因果的な再生と一致する（名古屋 3）。

却下した案:
- 直近 1 秒の軌跡全体を共通のずれで動かして搬送波に合わせる探索: ±3 m の範囲に偽のピークが多く、前向きの位置より悪い（0.5 m 以内 18% 対 22%）。
- 搬送波の少ない区間で前向きと逆向きを FIX からの時間で重み付け平均: 名古屋 2 は最大 +1 pp だが東京 3 で −2〜5 pp。

名古屋 2 の残り: 6.5 分間、衛星 2〜7・搬送波ペア 0〜8 の区間（555966–556367 s）で、どちらのパスも IMU の推測航法で 3〜20 m ずれる。距離の 17% が後処理でも 2 m 以上ずれたまま（シード 20260710）。この区間はデータ側の限界として扱う。

## D-054: FIX 位置がグラフの float から 0.5 m 以上離れたら FIX ラベルを出さない（WP56）

**決定（2026-10-14）**: `run_ship.sh` の既定を WP56（WP55 + `--fix-flt-max 0.5`）にする。ラベルの入りと継続の両方で、報告位置と共有グラフの float の距離が 0.5 m を超えるエポックは FIX にしない。閾値は正しい FIX の判定基準（0.5 m）そのもので、データから選んだ値ではない。整数の持ち越し（carry）は搬送波だけで判定するので止めない。

判断材料（6 run × 3 シード、ラベルの再生。フィルタは同じ）:
- 逆向きパス（ラベルを選ぶのに使っていないデータ）で、WP50 のラベルは東京 3 で 4.0〜5.3%、名古屋 1 で 1.2%、名古屋 2 で 0.9% の誤 FIX を出していた。逆向きの誤 FIX 1,894 エポックのうち 1,582 は、FIX 位置が float から 0.5 m 以上離れていた（順向きは 38 中 17）。
- 東京 3 と名古屋 1 では、逆向きの float が順向きよりずっと悪い（東京 3: 0.5 m 以内 52% 対 94.5%、名古屋 1 は 3 分間 90〜190 m ずれる）。時刻ずれは無い（進行方向の誤差 / 速度 ≈ 0.002 s）。原因は未解明。
- WP56 の結果: 順向きの正しい FIX 62.9 / 82.2 / 78.6 / 70.5 / 52.0 / 30.8%（平均 −0.25 pp）、最悪シードの誤 FIX 0.20% → 0.08%。逆向きは名古屋 2 の誤 basin（1.02%）を除いて最悪 0.41%（WP50: 5.28%）。
- 15 秒遅延: 平均 −0.1 pp、最悪 0.19% → 0.08%。後処理: 平均 72.1% のまま、最悪 0.35% → 0.24%。PPC OFFICIAL の変化は 0.07 pp 以内（実時間 81.6%、後処理 85.9%）。
- 閾値 0.4〜0.6 m は結果がほぼ同じ。走行単位の leave-one-run-out（順向きと逆向きをまとめて外す）で閾値を選ぶと、5 fold は 0.6 m、東京 3 の fold は 1.0 m を選び、逆向き東京 3 で 2.4% になる。0.6〜1.0 m ずれる失敗が東京 3 にしか無いため。データで選ぶより、設計で固定した 0.5 m の方が頑健。
- 実行中の出力は `replay_carry.py --flt-max 0.5` と一致する（東京 1・東京 3・名古屋 3 の先頭 1,500 エポック、チェックが効いた 3・4 エポックを含む）。

却下した案:
- carry にも同じ距離チェックをかける: 逆向き東京 3 が 59.9% → 46.7% になり、誤 FIX は減らない（0.41% → 0.48%）。

残る課題: 名古屋 2 の誤 basin は float も同じ basin にある（FIX との距離 0.05 m）ため、このチェックでは消えない。逆向きパスはこのチェックを見つけるのに使ったので、今後は独立した検証データにならない。

訂正（2026-10-14、D-055）: 上の東京 3 の逆向きの数字（誤 FIX 4.0〜5.3%、float 0.5 m 以内 52%）は、逆向きパスの IMU バイアスのバグが原因だった。バグを直すと、WP50 のラベルでも逆向き東京 3 の誤 FIX は 0.06%。修正後の逆向きデータでは、このチェックの根拠は名古屋 1（1.23% → 0.04%）と順向き（0.20% → 0.08%）。閾値の leave-one-run-out は、5 fold が 1.0 m、名古屋 1 の fold は「チェックなし」を選び、逆向き名古屋 1 で 1.16% になる。設計で固定した 0.5 m を維持する。

## D-055: 逆向きパスの IMU バイアスを静止区間から取る（WP57）

**決定（2026-10-14）**: `--reverse` のとき `IMU_STATIC_SEARCH=1` を既定にする。Phase 2 への切り替えで使う IMU バイアスを、先頭 5 秒ではなく、ジャイロの標準偏差が 0.5 deg/s 未満かつ加速度の標準偏差が 0.2 m/s² 未満になる最初の 5 秒窓から取る。順向きは変えない（オプション無効。全 run が静止状態で始まるので、有効にしても同じ窓になる）。

判断材料:
- 逆向きパスは run の終わりから始まる。東京 3 は旋回中に終わる（ヨーレート 3.99 deg/s）ため、逆向きのジャイロ z バイアスが −4.5 deg/s（正しくは約 +0.02）になった。バイアスのランダムウォークが小さいので、走行の最後までそのままだった。
- 修正後の逆向き東京 3（3 シード）: float が 0.5 m 以内に入る割合 52.1% → 93.6%（順向き 94.5%）、正しい FIX（WP56 ラベル）59.9% → 80.0%、最悪の誤 FIX 0.41% → 0.07%。東京 1 は 60.8% → 61.9%。他の 4 run は同じ窓になるので変わらない。
- 後処理: 東京 3 で 84.6% → 85.3%、東京 1 で 69.3% → 69.6%。PPC OFFICIAL は 92.1 / 95.6 / 95.2 / 79.5 / 64.3 / 90.9%、平均 86.3%（WP56 は 85.9%）。PPC2024 優勝 85.6% との比較は D-057 で訂正（後処理は大会ルール外）。

却下した案（名古屋 1 の逆向き）:
- 名古屋 1 の逆向きは、コード誤差 10〜20 m の衛星（41、24）を新しいアークとして迎えて 3〜4 m ずれ、18 秒の欠測を挟んで 330 m ずれたまま 3 分間搬送波を使えなくなる。バグではなく、方向によるデータの非対称。
- DD 擬似距離の sanity reset を逆向きで有効にする: この区間は 20 秒で復帰するが、先頭 3,300 エポックで正しい FIX が 72.8% → 59.4% に下がる。
- アークの終わり際の観測を捨てる: 順向きの残差で見ると、終わり際の誤差は始まりより大きくない（擬似距離 p90 2.1 m 対 2.7 m）。根拠が無い。

## D-056: 少ないペアでの FIX 入りと 1 周波衛星の搬送波は採らない（WP58、否定結果）

**決定（2026-10-14）**: どちらも出荷しない。FIX 率の残りの伸びしろは、ラベルではなく測定（使える搬送波ペアと float の質）で決まっている。

判断材料:
- リアルタイムで FIX になっていないのに位置が 0.5 m 以内のエポックは、東京 1 で 22.6%、名古屋 3 で 46.3% ある。そこの搬送波ペアは中央値 9〜13（FIX のエポックは 24〜40）、一致度 0.99 超は 0〜6 ペア、誤差の多くは 5〜50 cm で、float の品質。入りに必要な 15 ペアに届かない。
- 少ないペア（8 / 10 / 12）で 1 秒続いたら FIX に入る: 平均 +2〜4 pp だが、名古屋 2 で誤 FIX 4.6〜13.5%。5 秒以上続くことを求めると +0.1 pp 以下。
- 1 周波衛星の搬送波を使う（シード 20260710、4 run）: 正しい FIX が全 run で下がる（東京 1 63.2 → 57.2%、名古屋 3 32.0 → 15.1%）、名古屋 2 で誤 2.1%。1 周波の搬送波は 32〜42% のエポックで整数から 0.25 サイクル以上ずれ、整数の飛びは 2 周波の 5〜11 倍。
- C/N0 40 dB-Hz 以上だけ使う: 名古屋 3 の OFFICIAL は 82.7 → 87.9% に上がるが、正しい FIX は下がり（東京 1 63.2 → 58.7%）、名古屋 2 で誤 2.05%、東京 1 の OFFICIAL は 86.1 → 79.7%。
- GLONASS は基地局が Trimble、移動局が Septentrio で、整数を解くには周波数間バイアスの扱いが要る。

## D-057: PPC2024 との比較はリアルタイム出力で行う（訂正）

**決定（2026-10-14）**: PPC2024 の優勝スコア（TURING 85.6%）と比べてよいのはリアルタイム出力だけとする。後処理（WP51/54/57）と 15 秒遅延の出力は未来のデータを使うので、大会ルールの外として扱う。D-052・D-055 で「後処理 85.9 / 86.3% は優勝と同水準」とした記述は撤回する。

根拠（公式資料: http://taroz.net/data/PPC2024.pdf 、 http://taroz.net/data/PPC2024_results.pdf ）:
- ルール: リアルタイム処理を想定し、その時刻の過去の観測値は使ってよいが、未来の観測値は使ってはいけない。
- 評価: 走行距離に対する 3 次元誤差 50 cm 以下の距離の割合（停止中は数えない）を 6 走行で平均。Public は 6 走行の 50%、Private は残りの 50% で計算し、順位は Private。
- 結果: 1 位 TURING（Public 78.7% / Private 85.6%）、2 位 東京海洋大（77.6% / 80.3%）、3 位 産総研（71.6% / 78.0%）。
- 大会中、参加者が真値を持っていたのは別日の 2 走行（train）だけ。こちらは 6 走行の真値を見ながら開発している。

現状: リアルタイム出力は 6 走行全体で 81.6%（3 シード平均）。Public / Private の分け方が分からないので厳密には比べられないが、半々なら TURING の全体スコアは 82% 前後と推定され、こちらは同程度。優勝チームに勝つには、リアルタイムで 85.6% 以上を目標にする。

## D-058: リアルタイム出力は複数の前向き実行のアンサンブルから選ぶ（WP59、地図なし）

**決定（2026-10-14）**: リアルタイム出力の評価・比較には、前向き実行を並列に走らせたアンサンブル（`rt_ensemble.py`）を使う。メンバーは WP56 の 3 シードと、1 周波衛星の搬送波を 40 dB-Hz 以上だけ使う変種（`QC_ANY_BAND_CODE_ONLY=0 QC_ANY_BAND_CP_CNR=40`、シード 20260710）。各エポックで、各メンバーの候補（報告位置・グラフ float・MAP・mixture）から 1 つを因果的に選ぶ。単一実行（81.6%）の既定出力は変えない。

根拠（PPC OFFICIAL、6 走行の平均、すべて `rt_ensemble.py`）:
- 3 シードを WP55 の曖昧度関数で選ぶ: 83.0%。1 周波メンバーを足す: 83.8%（6 走行すべて上がる、名古屋 2 は +2.4 pp）。
- 選択を勾配ブースティングの候補スコアにする（他の 5 走行で学習する leave-one-run-out）: 84.2%。5/6 走行で上がり、名古屋 2 だけ −1.0 pp。
- FIX でない出力を 0.10–0.15 m 下げる（leave-one-run-out、全 fold が 0.10 か 0.15 を選ぶ）、移動局の観測欠落区間に IMU の推測位置を出す（`--gap-fill`）: 84.7%。高さの補正は、ビル街で float 解が上にずれる（高さだけで失った区間の中央値 +0.46〜+1.02 m）ことへの対処。
- FIX ラベルは地図なしメンバーの 2 本以上が 0.2 m 以内で FIX のときだけ。誤 FIX は全走行 0.04% 以下で、正 FIX は WP56 単体とほぼ同じ。

採らなかったもの: 衛星 4 機以上でエポックを確定（`P2_MIN_NS=4`、全体では効かない）、IMU だけでエポックを確定（`P2_IMU_ONLY=1`、純 IMU 予測より誤差が早く広がる）、2〜4 衛星の Doppler 速度拘束（ほとんど発火しない）、メンバー候補の medoid / 中央値（−1.5〜−2.1 pp）、最後の FIX のオフセットの持ち越し、libgnss++ KF-RTK メンバー（+0.1 pp、外部エンジン）。

注意: 選択器と高さの補正はこの 6 走行の真値で（leave-one-run-out で）決めている。PPC2024 の参加者が真値を持っていたのは別の 2 走行だけなので、大会の結果とは条件が違う。

## D-059: PLATEAU の 3D 都市モデルで NLOS を判定するメンバーを足す（WP60、地図あり）

**決定（2026-10-14）**: 地図ありのリアルタイム出力として、PLATEAU の建物・橋モデルで NLOS を判定する前向き実行 3 シードを D-058 のアンサンブルに足す。地図なしの数値とは分けて報告する。

方式:
- 各エポックで IMU 予測のアンテナ位置から全衛星へ光線を飛ばし、建物に当たれば NLOS（`plateau_los.py`、CPU の格子トレーサ、`NLOS_ONLINE=1`）。そのエポックの予測位置だけを使うのでリアルタイム。
- 見通し内の衛星が 6 機以上残るときは NLOS のペアを二重差から外し、残らないときは雑音 σ を 5 倍にするだけ（`NLOS_MEAS_MODE=3`）。各システムの基準衛星は見通し内で最も高い衛星（`NLOS_REF_LOS=1`）。
- 受信機から 3 m 以内の面は無視する。橋の上を走ると、橋桁の面がアンテナの 1.1 m 上に来て天頂がふさがった扱いになっていた（東京 2 で 9,151 エポック中 1,354、建物だけなら 7）。

根拠（PPC OFFICIAL、6 走行の平均）:
- 単体（3 シード平均）: 81.6% → 82.3%（東京 1 は 85.2 → 89.7%、名古屋 1 は 77.2 → 75.4%）。
- アンサンブル: 地図なし 84.7% → 地図ありメンバー 1 本で 86.1%、3 本で 86.3%（92.2 / 96.2 / 94.4 / 81.6 / 62.1 / 91.0%）。各走行を時間で前後半に分けると 83.3% / 90.3%。
- 地図ありメンバー自身の FIX ラベルは名古屋 2 で誤り 5.84%（最悪シード 7.06%）。0.95 m ずれた誤 basin がより強く残る（地図が外す衛星に、それに反対していた衛星が含まれる）。そのためラベルは地図なしメンバーから作る（誤 FIX は全走行 0.06% 以下）。

比較: 6 走行全体の 86.3% は、優勝チーム（TURING）の Private 85.6% を上回る。ただし Private は非公開の半分で、選択器はこの 6 走行で学習し、地図という追加データも使っている。名古屋 2 の 375 秒のビル街区間（距離の 21%）は、どのメンバーのどの候補も 0.5 m 以内に入らない（最良候補の中央値 2.35 m）。90% にはここを解く別の拘束が要る。

## 現在の未決定事項

棚卸し: 2026-09-26。旧リストの各項目を「解決済み / 休眠 / 未決定」に分類した。解決済み・休眠の項目は、再開条件が満たされるまで議論しない。

### 未決定（active）

| 項目 | 現状 | 判断材料 / 次の一手 |
|---|---|---|
| D-037 後の pivot 先 | 粒子ごとの 3DMA 尤度は否定済み（D-033, D-037）。(c) の「RTK の除外衛星候補」は D-059 で採用（予測位置からのオンライン判定、地図ありアンサンブル 86.3%） | 残りの候補: (a) 推定位置の近傍で集約した遮蔽確率を全粒子共通の重みにする、(b) レイラベル（`diag_nlos_label_and_bias.py`）を教師データにした学習誤差モデル、(d) UrbanNav+PLATEAU+GPU BVH パイプラインをベンチマークとして公開 |

### 解決済み（closed）

| 旧項目 | 結論 | 根拠 |
|---|---|---|
| `blocked score` は完全に不要か、veto 付きなら使えるか | 尤度・切替の駆動には使わない。遮蔽情報は feature（ranker 特徴量、coverage、シナリオ）用途に限定 | D-011（blocked-switch は holdout を超えない）、D-033・D-037（粒子ごとの遮蔽判定は構造的に有害）、`nlos_pf_measurement_wiring.md`（epoch 共通 mask は PPC で Δ=0） |
| Path 3（architectural pivot / sequence-latent model）の優先度 | D-034 文脈の Path 3 は kill criteria 到達で終了 | `product_deliverable/D034_PATH3_PREGATE_NULL.md`, `D034_PATH3_HMM_NULL.md`（2026-05-01、全 HMM variant が deployed run-MAE 1.79 pp に負け） |

### 休眠（dormant: paper track 再開時に再評価）

paper track（D-019, D-024, `paper_draft_2026-04-01.md`）は 2026-04 以降止まっている。以下は paper 再開時まで凍結する。

| 旧項目 | 暫定の扱い | 再開条件 |
|---|---|---|
| `always_robust` と `entry_veto_..._regime_gate` を main paper でどう位置づけるか | D-019 のとおり: safe baseline = `always_robust`、exploratory best = `entry_veto_...` | paper 本文の執筆を再開したとき |
| strategy 差分が出る epoch をどの figure で見せるか | `build_paper_assets.py` の固定 3 図（D-024）を維持 | 同上 |
| readability/extensibility proxy をどこまで信頼するか | 補助指標扱い（`experiments.md` の proxy 定義）。採用判断の根拠にはしない | 同上 |
| Path 2（PPC データの追加）の優先度 | 追加データの取得計画はなし。既存 6 route で LORO / disjoint holdout を継続 | 新しい PPC 相当のデータセットが入手可能になったとき |

2026-10-11 CPU robust SPP最適化: `python/gnss_gpu/robust_spp.py` の `H.T @ diag(w)` をC-orderの作業バッファへの列スケーリングへ変更し、HとHTWを反復間で再利用した。旧版 `c09dc38` と同じ倍精度・重み・収束・有効衛星判定で比較し、合成15条件の座標はbit一致、入力非変異。clean/Cauchy/Huber、ゼロ重み・有効衛星不足・特異配置を含む53回帰テストPASS。局所CPU計測100call×7交互roundの速度比1.08〜1.53×、別の長時間field検証が同時稼働しておりwhole latencyやfield精度の証拠にはしない。再現sourceと全round・source hashesは `benchmarks/bench_robust_spp.py` / `benchmarks/robust_spp_cpu_20261011.json`、解釈は `benchmarks/RESULTS.md`。元private220とqueue7544/222b/223bの数学・source・閾値・凍結runtimeは変更しない。公開物のGPU製品名非記載ルールを維持。

## Device PF gather高速化（2026-10-11）

精度を維持したGPU高速化は、既定coalesced/B=60のボトルネックから進める。
ブロック乱数の事前共有案は100万粒子で約1.01×に留まり破棄。状態コピーの共分散9成分を
隣接laneへ連続割当し、祖先・乱数・反復・粒子数・状態精度を維持した。追加確保/launchなし。
同一CUDA 12.8/MSVC Releaseで旧PF `c09dc38` と比較し、100万粒子のresampleは
11.6016→4.8157 ms（2.41×）、12衛星のnative predict/weight/ESS/resample/estimateは
21.7690→14.1663 ms（1.54×）。20回×5交互ラウンド、stream同期あり、状態/重み復元は測定外。
1,274合成ケースで全16状態/重みと連続pipeline/混合resampler出力のhashがliteral一致、
統計・wrapper・固定lagの41テストPASS。小粒子数のpipelineは速度改善を確認できず、
非専有ホストの変動とともに全行を公開。実データ精度の改善やwhole GNSS/IMU速度とは区別する。
詳細と各サンプルは `benchmarks/RESULTS.md` と `benchmarks/pf_gather_gpu_20261011.json`、
再現は `benchmarks/bench_pf_coalesced_compare.py`。次はraytrace反射候補の並列化、静的geometryの
常駐、signal channel中間配列/launch削減を、同値候補順・加算順・seedの比較付きで検証する。

## BVH反射候補の並列評価（2026-10-11）

rayごとのthreadが全leafを巡る反射計算を、512三角形以上・4,096 ray以下で候補tileの
並列評価と最小delay選択へ変更。FP64/閾値/正delay判定は維持し、同値は元のright-first
訪問ordinalで選ぶ。元の64-entry stack guardも再現する。GPU勝者bufferは最大160 KiB、
orderはint32×訪問三角形数。三角形×rayの候補全配列は作らず、小mesh・大ray数・
共有/重複leafでscratchが増える入力には元の経路を残す。外部APIは変更しない。
同一CUDA 12.8/MSVC Releaseで旧BVH `c09dc38` と同プロセス内の呼び出し順を毎回交互にし、
20回×5ラウンドを比較。GPU確保/転送/同期/取得/解放込みで12,000三角形・12rayは
64.4258→7.2077 ms（8.94×）、1rayは56.5891→3.8205 ms（14.81×）。事前BVH構築は測定外。
90合成ケースで全反射点/delayのuint64とLOSが一致（80ケースは非zero反射）、入力非変更を
確認。複数epoch/ECEF/scale/退化/非finite/deep/custom tree/同値の別反射点を含み、
BVH/raytrace/wrapper/urban multipathの60テストPASS。全サンプルとhashは
`benchmarks/bvh_multipath_gpu_20261011.json`、再現は `benchmarks/bench_bvh_multipath_compare.py`。
非専有ホストの時間変動、並列化しない範囲も `benchmarks/RESULTS.md` に全行を記録。
元モデルの計算結果を維持する変更であり、実データ位置精度やwhole GNSS/IMUの改善とは区別する。
次はsignal simulationのchannel中間配列/launch削減を、float加算順とseedを維持して比較する。

## Signal simulationのIQ中間配列削減（2026-10-11）

衛星ごとの生成・合成を1 kernelへ統合し、チャネル数×サンプル数のfloat IQ中間配列を
除いた。input順の加算、元の保存時の丸めを `__fmul_rn` / `__fadd_rn` で維持する。
double位相計算/コード表/trig/Philox/ノイズ式は維持。32有効チャネル以下は2,560 byteの
設定をkernel引数で渡し、4 KiB制約をstatic_assert、33以上は設定bufferを使う。
nav bitsは1 bufferへまとめ、static/空/clamp/無効PRNの動作は維持する。
32ch×2,600,000sampleのIQ用cudaMalloc要求は654.60→19.84 MiB（ソースから算出、
全runtimeの実測peakではない）。全nav bitsの総長に比例したhost/device bufferを追加する。
旧 `c09dc38` と同一CUDA 12.8/MSVC Release/compute capability 7.5で20回×5交互roundを
比較し、11ch×26,000sampleは0.6886→0.1639 ms（4.20×）、32ch×2,600,000sampleは
46.2629→13.6752 ms（3.38×）。native検証/変換/GPU確保/転送/同期/取得/解放込み、
初回コード表初期化とPython class wrapperは除外。263ケース全IQ bit一致・入力非変更、
独立single-channel kernelとの合成/航法bit比較、seed、信号捕捉、urban/E2Eの88テストPASS。
非専有ホストの変動と未変更noise-only経路も含む全25条件を公開。
source/binary/input/output hashesと各callは `benchmarks/signal_fused_gpu_20261011.json`、
再現は `benchmarks/bench_signal_sim_compare.py`、解釈は `benchmarks/RESULTS.md`。
既存信号モデルの出力同値を示す検証であり、実測位置精度/whole GNSS/IMUの主張とは区別する。
公開物にGPU製品名を記載しない。次はgeometry等の再利用で呼び出しごとの確保/転送を
減らす案を、寿命・更新時の挙動と全体時間を含めて比較する。

## BVH形状・作業bufferの常駐（2026-10-11）

`BVHAccelerator` の反復ray呼び出しで、meshの検査/unpack/GPU確保/転送を毎回行う
コストを除く。CPU meshは所有read-only snapshot、最初のray callでnative `DeviceBVH` を
作り、geometryと用途別の最大要求サイズのworkspaceを保持する。rx/sat転送/同期/新しい
出力取得は毎回行う。CPU訪問順は一回求め、GPU orderは初回parallel multipathまで遅延転送。
FP64/閾値/tie/stack/512tri・4,096rayの経路選択とkernelは直前版と同じ。
stateless native functionsを維持。`update_triangles` は新CPU build成功後に旧GPUを解放し、
次回新版を転送。不正更新では既存scene維持。`release_gpu` で明示解放/次回再転送。
pickle/deepcopyはCPU形状だけを保存し、CUDA ownerは元deviceで操作/解放しdeviceを戻す。
native ownerの並行C++利用は対象外、Python bindingはGIL維持。

baselineは直前のparallel multipath版（退避source/binary hashが前の公開JSONと一致）。
同一CUDA 12.8/MSVC Release/compute capability 7.5、20回×5交互roundで、12,000tri/12rayの
warm LOS+反射は4.8805→0.8216 ms（5.94×）、LOSは2.4148→0.3713 ms（6.50×）。
native検証/転送/同期/取得込み、CPU BVH build/Python wrapper除外。coldはscene作成からclose
まで含み、同条件LOS+反射は4.6620→3.4101 ms（1.37×）。cold LOSの全条件は0.72〜0.86×で
遅く、反復利用の改善として扱う。4,097ray反射の計算は元のまま。非専有ホストの変動を明記し、
35条件を全公開。前版との速度比の積、実データ位置精度/whole GNSS/IMUの利得は主張しない。
100ケース＋7測定入力で全反射/遅延bit一致、LOSは106入力で一致。大小batch/異なるray値/並列と直列/LOSを
混ぜて再利用し、入力破棄と旧出力の所有も照合。更新失敗/解放/serialization、coverage/
candidate/PF3D互換を含む155テストPASS、既存1skip。最終8所有/解放テストのCUDA memcheckは
0 errors/0 leaked allocations。12,000tri/12ray使用後のowner確保は1,457,928 byte（約1.39 MiB）、
runtime全peakではなく保持したcudaMalloc要求量。sceneは形状更新/明示解放/破棄まで保持する。
測定/hash/再現は `benchmarks/bvh_resident_gpu_20261011.json` /
`benchmarks/bench_bvh_resident_compare.py` / `benchmarks/RESULTS.md`。
公開物のGPU製品名非記載を維持。次はLOSと反射を続けて呼ぶ際の同じrx/sat転送、または
PF内の同期/取得のコストを、呼び出し全体の時間と出力同値を含めて調べる。

## LOSと一次反射の座標転送・同期の共有（2026-10-11）

常駐geometryでもLOS→multipathに同じrx/sat転送と同期が二回あったため、
`DeviceBVH.trace` / `trace_batch` と公開wrapperを追加した。両kernelの起動前に必要な
workspace/orderを用意し、座標は一回転送、同じstreamで従来kernelを実行して一回同期。
FP64計算/閾値/tie/stack/経路選択を維持。独立配列を返し、既存の別々のAPIも維持。
`UrbanSignalSimulator` の通常一次反射をcombined APIに接続。詳細reflection pathsは
従来経路、APIのないmodelは従来二call、不可視衛星はray対象外。

baselineは直前の常駐版で、source/binary hashを公開resident JSONのcandidateと照合。
CUDA 12.8/MSVC Release/compute capability 7.5、20回×5の同一プロセス交互比較で、
1,200tri/12rayのwarm single LOS+反射は0.1808→0.1426 ms（1.27×）。native検証/転送/
同期/取得込み。18 native条件（coldは作成/close込み）と6都市条件を全公開。
実際の都市compute_epoch（可視判定/チャネル作成/CUDA IQまで、同じ信号生成版）は
1.05〜1.13×、12,000tri/12ray/10msは1.1870→1.1311 ms（1.05×）。
計算が重い大batchは1.00〜1.01×で差が小さい。非専有ホストでのwall time比較。
以前の速度比を掛け合わせず、実データ位置精度/whole GNSS/IMUの利得は主張しない。

100ケース＋7 native測定入力でLOS/反射点/遅延bit一致、有限105入力でsingleも一致。
reuse/入力変更/旧出力保持、padding/退化/tie/深い木/共有leafを含む。6都市入力はIQ/
全配列byte一致、metadata一致。273テストPASS、既存1skip。拡張/混合利用/shape/更新/
解放/IQの最終11テストのCUDA memcheckは0 errors/0 bytes leaked。
新しいdevice bufferは追加せず、同じquery列の後のowner確保量は全7入力で旧新版同じ。
再現/hash/各callは `benchmarks/bench_bvh_combined_compare.py` /
`benchmarks/bvh_combined_gpu_20261011.json` / `benchmarks/RESULTS.md`。
公開物へのGPU製品名非記載を維持。

## Systematic PFの重み準備の重複除去（2026-10-11）

`resampling="systematic"` のglobal max後のexp(lw-max)とblock sumを一kernelにまとめ、
正規化で保存済みFP64重みを再利用した。expを2N→N回、起動を一回、N doubleのreadを除く。
同じblock/最終加算順、division、Thrust CDF、target/seed/探索/gatherを維持。
追加device buffer/API変更なし。端数threadもshared reductionに参加する。
baselineは直前のcoalesced covariance gather版で、source/binaryを公開PF JSONと照合。

CUDA 12.8/MSVC Release/compute capability 7.5、別プロセス・5交互round×20回の同期込み比較。
100万粒子normal resampleは2.5514→2.3493 ms（1.09×）、native predict/weight/ESS/条件付き
resample/estimateは9.1824→8.8376 ms（1.04×）。公開predict/update/estimateは9.0897→8.8628 ms
（1.03×、約2.5%短縮）。3重み/4粒子数/全体step/未変更Megopolis controlの24条件を全公開。
小粒子数には遅い条件と未変更controlの変動があり、非専有hostのwall timeとして扱う。
実データ位置精度/whole GNSS/IMUの利得や前版速度比の積を主張しない。

実際の旧新版handleを読む独立probeで、948ケースのnormalized weightsとmax/sumが全bit一致。
全16状態は元配列[取得祖先]とbit一致、reset weights一致。3epoch/mixedを含む986中977の
旧新版出力hashが一致し、全24測定入力の最終出力は全round/両buildで一致した。
残る9件は大Nのuniform/sparse・seed 0/1000000で、同じnormalized weightsからCDF境界の
祖先選択が異なった。旧版自身の8回比較でもCDFが最大6.66e-16動き、100万uniformでは最大
158,006祖先が変わる（index差最大1）。sparseでは最大17。数ulpのCDF下降も旧新版にある。
未変更の並列scanの丸め/実行順の制約と判断し、例外/自己比較/下降をJSONで明示する。
全状態bit再現を達成したとはしない。既存target/seedや統計的性質の修正も今回の範囲外。
CPU dyadic参照/partial/underflow/連続利用/既存統計/wrapper/smoother/SMC/IMUの104テストPASS。
最終38テストのCUDA memcheckは0 errors/0 bytes leaked。
測定/hash/再現は `benchmarks/pf_systematic_gpu_20261011.json` /
`benchmarks/bench_pf_systematic_compare.py` / `benchmarks/probes/` / `benchmarks/RESULTS.md`。
公開物へのGPU製品名非記載を維持。

## 既定PFの共通共分散保持と候補先読み（2026-10-11）

単体kernelの微小な利得に留まらず、既定coalesced Megopolis（B=60）の公開update全体を測定した。
初期化/共通process noiseで保証される同一共分散は、resample時に現在の配列を保持する。
9成分のread/writeを省き、100万粒子の一gatherで論理的144 MBを除く。import/Doppler KFは
保証を無効化し全copyへ戻す。再初期化で回復、他resamplerとの混合も維持。device確保は同じ。
Philoxの同じ4 drawと4候補を先読みし、元の受理順/FP64差分/FP32 draw・受理式/seedを維持。
端数はscalar draw。共通乱数のwarp共有のみの試行はほぼ1.00×で破棄した。

baselineの5 source/binaryは直前のsystematic JSON candidate hashと照合。CUDA 12.8/MSVC
Release/compute capability 7.5/C++20/fast mathなし、別processの旧新版を5交互round×20回比較。
100万粒子の公開predict/update/estimateは14.2177→12.6016 ms（1.13×、約11.4%短縮）、native
全体14.1212→12.5580 ms（1.12×）、尤度後のresample7.1227→5.6140 ms（1.27×）。
全60条件/生sampleを公開。未変更predict/weight/systematicは100万粒子でほぼ同じだが、
Doppler KF全体は48.8948→48.2415 ms（1.01×）、10,000粒子の公開stepもほぼ同じ。
小粒子数には遅い条件とcontrol変動がある。非専有hostの同期込みwall timeとして扱う。
実データ位置精度/whole GNSS/IMUの利得や過去速度比の積を主張しない。

1,659検証ケースと全60測定入力/全roundの状態・重み・返却値出力hashが旧新版で一致。
共通covと同じ状態をimportした全copy経路、common KF predict、Doppler KF後の無効化、
再初期化/混合利用、B=0〜9/15/60/240/257、seed/端数、大shift/微小差/±Inf/NaN/極端重みを含む。
180テストPASS、追加76テストのCUDA memcheckは0 errors/0 bytes leaked。以前のsystematic
CDF境界変動は残っており、今回はその解消やすべての入力のbit再現性を主張しない。
測定/hash/再現は `benchmarks/pf_covariance_gpu_20261011.json` /
`benchmarks/bench_pf_covariance_compare.py` / `benchmarks/RESULTS.md`。
公開物へのGPU製品名非記載を維持。残る大きな計算はDoppler KFで、次の候補を測定する。

## Doppler KFの二pass間のgeometry再利用（2026-10-12）

前回のDoppler KF残存コストを測定し、weighted meanとKF更新で重複していた方向/観測値を
32衛星まで第一passのFP64値で再利用。64衛星まではsigma²/weightをblock内で一度計算する。
centering順/衛星順/速度・covariance更新/symmetrization/floor/S guard/log likelihood/精度を維持。
33衛星以上はgeometry再計算、65衛星以上は追加shared領域なしで従来workspaceへ戻す。
ownerの追加cudaMallocはないがcompiled kernel stackは512/1,024 bytes/threadで旧版0から増える。
16/32衛星版86 registers、fallback94、旧版92。dynamic sharedは64衛星まで8n→9n double。
実行時peak memoryは未測定。128threadへの変更は追加利得がなく、従来256を維持した。

baselineの5 source/binaryは直前covariance JSON candidateと照合。CUDA 12.8/MSVC Release/
compute capability 7.5/C++20/fast mathなし、別processの旧新版を5交互round×20回比較。
100万粒子12衛星のnative KFは34.5045→27.1058 ms（1.27×）、32衛星91.0176→71.3388 ms（1.28×）。
PRとKFの両log likelihoodを使う公開predict/PR/KF/ESS/条件付きB60 resample/estimate全体は
50.0132→42.7602 ms（1.17×、約14.5%短縮）。native全体49.6287→42.5168 ms（1.17×）。
未変更PR controlは100万粒子14.1337→14.3793 ms（0.98×）、64/65衛星KFは1.03×/0.99×。
全40条件/生sampleを公開。小粒子数にはcontrol変動があり、非専有hostのsteady-state wall time。
初回local memory準備/初期化/復元/出力検査は測定外。過去速度比の積や実データ精度/whole
GNSS/IMUの利得を主張しない。systematicの既存CDF境界変動も維持される。

1,566ケースの全16状態/log weightと、全40測定入力/全roundの出力hashが旧新版で一致。
端数粒子・0〜128衛星のcache/shared/growth境界・mask・近接range・不正S・全zero weight・
sigma² underflow・NaN/Inf weight・二連続KF・観測変更/成長/resample混合・大粒子数を含む。
独立CPU FP64 matrix参照（rtol/atol 2e-10）、48 KiB shared境界768衛星、入力/出力所有権と
既存統計/wrapper/smoother/SMC/IMUの214テストPASS。追加34テストのCUDA memcheckは
0 errors/0 bytes leaked、racecheckは0 hazards/0 errors/0 warnings。
測定/hash/再現は `benchmarks/pf_doppler_gpu_20261012.json` /
`benchmarks/bench_pf_doppler_compare.py` / `benchmarks/RESULTS.md`。
公開物へのGPU製品名非記載を維持。

## IQ simulationのowner単位work buffer再利用（2026-10-12）

前回までのfused IQ版が毎epoch行っていたGPU output/nav bit/32channel超descriptorの確保と解放を、
DeviceSignalWorkspaceでowner単位に再利用。CPU packing vectorもclearしてcapacityを再利用する。
毎callの入力はpack/uploadし直す。3 CUDA kernel body/float丸め/加算順/nav clamp/固定seedの
Philox系列/seed=0 fresh noise/PRN処理を維持し、device-wide同期とhost IQ copyも同じ。
既存native単発APIは一時ownerで維持。公開SignalSimulatorは遅延ownerを利用し、旧extensionには
fallback。release_gpu/close/destructorで解放し、再作成可能。copy/pickleはownerを共有しない。

baseline source/binaryは直前fused JSON candidate hashと照合。CUDA 12.8/MSVC Release/
compute capability 7.5/C++20/fast mathなし。実binaryを同一processで5round×20call、旧新版順序を
交互にして比較。同期込みwall timeで、初回code table/CPU BVH build/出力hash検査は測定外。
BVH source/binaryは今回変更せず、都市全体は実際のLOS/反射/channel構築/CUDA IQ/metadataを含む。
26万sample/12可視衛星の1,200 triangle都市全体は2.5379→1.8947 ms（1.34×、約25.3%短縮）、
12,000 triangleは3.3476→2.7244 ms（1.23×、約18.6%短縮）、12 triangleは1.33×。
公開12channel/26万sampleは7.0985→5.9108 ms（1.20×）。1msの都市全体は0.99〜1.04×、
26,000sampleもほぼ1.00×。cold単発controlは0.99〜1.08×で、再利用の利得はない。
全54条件/生sampleを公開し、非専有host/desktopによる変動と小さい条件の限界も記録する。
過去速度比の積、全GNSS/IMUや実データ位置精度の向上を主張しない。

最大requested capacityを保持するtradeoffがある。outputだけで26万sample=2,080,000 bytes、
260万sample=20,800,000 bytes。65channel/nav/26万sampleの3 buffer計は2,086,060 bytes。
allocated_bytesはownerのbuffer capacityで、global code table/CPU packing vector/driver領域/
grow時のpeakを含まない。global code tableの既存単一device前提/instance非thread-safeを維持。
明示release/GCでownerのwork bufferは全解放する。

271ケースの旧新版IQがbit一致、うち263件は新単発APIも旧版と一致。8件はgrowth/shrink/empty/
入力変更後の連続利用と返却配列所有権。全54測定入力/全round/各callのIQ・urban metadata hashも
一致。信号/urban/reflection/diffraction/BVH/wrapperの243テストPASS、追加27テストのCUDA
memcheckは0 errors/0 bytes leaked、racecheckは0 hazards/0 errors/0 warnings。
close/destructor/独立owner/入力と設定の再反映/invalid後の再利用/release再作成/copy/deepcopy/
pickle/旧extension fallback/fresh noiseを含む。
測定/hash/再現は `benchmarks/signal_workspace_gpu_20261012.json` /
`benchmarks/bench_signal_workspace_compare.py` / `benchmarks/RESULTS.md`。
公開物へのGPU製品名非記載を維持。

## PF診断集計の同じFP64 treeでのwarp化（2026-10-12）

ESS/estimate/RMS spreadと各finalizeの6 kernelで、256thread treeのstride128/64/32をsharedに
残し、stride16/8/4/2/1だけwarp shuffle/registerで実行。加算node/FP64/exp/partial読み順/
division/guard/host copy/stream同期を維持し、各kernelのblock同期を8回減らす。
追加cudaMalloc/owner bufferなし、dynamic sharedもESS/spread6,144・estimate12,288 bytesのまま。
query registersは20→19/28→30/20→22、finalize24/38/24は同じ。6 kernelのcompiled stack/localは0。

直前Doppler-KF candidateの5 source/binary hashへbaselineを固定。CUDA12.8/MSVC Release/
compute capability7.5/C++20/fast mathなし、別processで5交互round×20回の同期込みwall time。
100万粒子normal weightのESS0.4884→0.4074 ms（1.20×）、estimate0.5948→0.5361（1.11×）、
spread0.5746→0.5089（1.13×）。公開ESS/estimate/default-center spread一式は
2.4147→2.0137 ms（1.20×、約16.6%短縮）。公開PR更新全体12.6661→12.3835（1.02×）と
PR+KF41.9750→41.8294（1.003×）は差が小さい。未変更weight controlにも1.10×の変動があり、
小粒子数/100,000粒子KFの測定悪化も全60条件・raw sampleとともに公開する。
非専有hostでの更新全体の小さい差を安定した利得と断定せず、過去速度比の積や実データ精度/
whole GNSS・IMUの向上を主張しない。

2,372検証ケースの2,281件はbit一致。91件の差は両版ともNaNの符号bitだけで、NaN以外の値は
すべてbit一致。160 unchanged-stateと12件の5epoch軌跡は全16状態/log weightも一致。
NaNを正規化せず各caseの生bit/hashと差を公開する。1〜100万粒子/warp・block境界/大shift/
±Inf/NaN/微小・巨大・相殺座標などを含む。全60測定条件・全round・各回の返却値hashと
各stage最後の状態snapshotが一致。systematic Thrust CDFの既存境界変動は対象外。
独立NumPy参照/返却配列所有権/状態不変/NaN guard/公開threshold/default centerと既存PF/
wrapper/smoother/SMC/IMUの343テストPASS。追加129件のmemcheckは0 errors/0 bytes leaked、
racecheck0 hazards/0 errors/0 warnings、synccheck0 errors。
測定/hash/再現は `benchmarks/pf_reduction_gpu_20261012.json` /
`benchmarks/bench_pf_reduction_compare.py` / `benchmarks/RESULTS.md`。
公開物へのGPU製品名非記載を維持。

## Default Gaussian PR専用化（2026-10-12）

proposal列の別kernel生成/既存scratch共有と8件先読みを先に試したが、全体の安定利得がなく棄却。
8件案の5交互round×20回/72条件/4,558bit一致ケースとcandidate patchを公開し、resamplerは
元の4件先読みを維持した。採用した変更は既定Gaussian PRの専用kernelとblock共通GPU逆分散、
2衛星unroll。元のrange/residual/積/衛星累積順/FP64/overwriteを保ち、Student/Huber/正gateは
元のgeneral kernelへdispatchする。無効robust optionも元の条件で判定する。
追加owner buffer/cudaMallocなし、launch/block256/転送/stream同期/APIは同じ。新register38/旧44、
stack/local0。static sharedはcompiled16 bytes増え、dynamic5*n_sat*8を維持。1,228衛星でも
49,136 bytesで48 KiB内に収まり、成長/縮小とfull kernelとのbit比較をテストした。

直前query版の5 source/binary hashへbaselineを固定。CUDA12.8/MSVC Release/compute capability7.5/
C++20/fast mathなし。別processの5交互round×20回、同期込みwall timeで全68条件/raw sampleを保存。
100万粒子/12衛星PR3.5309→3.2065 ms（1.10×）、公開PR更新全体12.3487→12.0579 ms
（1.024×、約2.4%短縮）、高ESSのnative全体6.6986→6.3755 ms（1.05×）。4衛星単独1.18×、
64/65衛星1.06×。100万粒子の未変更predict/resample/KFとrobust controlはほぼ1.00×、PR+KF全体は
1.007×。小粒子数の悪化/host変動と10,000粒子public/nativeの乖離も公開し、過去速度比の積や
実データ位置精度/whole GNSS・IMUの利得を主張しない。

1,205ケースの全16状態/log weight/返却値・5epoch軌跡がbit一致。1〜100万粒子・1〜128衛星/
奇数・端数・大/微小sigma・非有限粒子・zero/near range・zero weight・robust/fallback混合と
buffer成長を含む。全68測定条件/全round/各回の返却値hashと各stage最後のstate snapshotも一致。
関連432テストPASS、追加89件は独立NumPy参照/full Gaussianとのbit比較と48 KiB境界も検証。
CUDA memcheck0 errors/0 bytes leaked、racecheck0 hazards/0 errors/0 warnings。
測定/hash/再現は `benchmarks/pf_gaussian_gpu_20261012.json` /
`benchmarks/bench_pf_gaussian_compare.py` / `benchmarks/RESULTS.md`。
棄却8件案は `benchmarks/pf_prefetch8_rejected_20261012.json` / `benchmarks/probes/pf_prefetch8.patch`。
systematic Thrust CDFの既存変動は範囲外。公開物のGPU製品名非記載を維持。

## PF予測専用化と速度ガイド引数化（2026-10-12）

直前Gaussian PR版をbaselineとし、通常/速度ノイズ/KFのcompile-time専用kernelへdispatch。
ガイド3 FP64値をlaunchで捕捉し、独立24-byte H2D copyと共通pinned-result buffer書換えを省いた。
乱数列/normal draw順/FP64/Cholesky/clamp/Qv/速度sample条件を保持。引数化だけで生じた1 ULP差は
旧binaryのalpha*guide丸め→prior側FMAをintrinsicsで明示して修正し、本比較ではbit一致。
register66→48/58/60、stack/shared/local0。追加owner buffer/launch/syncなし、既存d_vel予約/APIは維持。

CUDA12.8/MSVC Release/compute capability7.5/C++20/fast mathなし、5交互round×20回、同期込みwall time。
5 source/binary baseline hashを直前Gaussian PR JSON candidateに固定。全76条件/raw sample保存。
100万粒子通常native予測2.2659→2.1475 ms（1.055×）、公開8回連続予測
17.9598→17.0453 ms（1.054×）、公開PR更新全体12.1421→12.0994 ms（1.004×）。
10万粒子公開queue8は2.4208→1.8286 ms（1.324×）、公開PR全体は
1.1346→1.2430 ms（0.913×）と悪化。100万通常予測/queue8と10万queue8のround範囲は分離するが、
全更新の安定した利得はこの測定では示せない。
controlと小粒子数の変動/悪化を全て残し、初回大きな速度比や過去の速度比の積を代表値にしない。
速度ノイズ/KF・全体利得の限界はRESULTS/JSONに記録。実データ位置精度/whole GNSS・IMUは未評価。

1,836ケースの全16状態/log weight/5epoch軌跡がbit一致。相関SPD/zero/非SPD/非対称/
非有限covと速度、alpha境界/負sigma_vel/zero dt/負Qv/小大sigma、small/ECEF、partial block/100万粒子含む。
全76測定条件/全round/各回返却値hashとstage最後の状態も一致。関連510テストPASS、追加78件の
独立分布momentとnative/public48回異なるguideの非同期列を検証。memcheck0 errors/0 bytes leaked、
racecheck0 hazards/errors/warnings。既存systematic Thrust CDF変動は範囲外、GPU製品名非記載を維持。
測定/再現は `benchmarks/pf_predict_gpu_20261012.json` / `benchmarks/bench_pf_predict_compare.py` /
`benchmarks/RESULTS.md` / `tests/test_pf_predict_specialization.py`。

## Doppler KFのblock形状調整（2026-10-12）

32衛星以下のgeometry-cache KFだけ256→128 threads/blockへ変更。32衛星超は256を維持し、
他kernelのstate.grid_size256基準/stream/転送/owner buffer/APIは同じ。変更はhost起動指定だけで、
CUDA12.8 buildの全device SASS dumpがbyte-identical。FP64/除算/FMA/累積順/geometry/KF/clamp/乱数を維持。
register cache16/cache32/fallback86/86/94、stack512/1024/0、shared/localも同じ。実occupancy/peak未測定。

直前予測版の5 source/binary hashへbaselineを固定。CUDA12.8/MSVC Release/compute capability7.5/
C++20/fast mathなし、別process実productionの5交互round×20回、同期込みwall time、全72条件保存。
2,048粒子単独KF12 0.2009→0.1173 ms（1.713×）、KF32 0.4775→0.2568 ms（1.859×）。
12衛星全体はnative 0.4205→0.3266 ms（1.288×）、public
0.4565→0.3586 ms（1.273×）。単独KF12の全round範囲は分離するが、public全体は重なりと
未変更PR control1.177×の変動もあり固定利得/全帰属を主張しない。native PR control1.018×。
10万単独約1.028×/control変動、100万public全体約1.00×/KF32の0.992×悪化も記録。
少数粒子での実効性を根拠に採用し、大粒子数の大きな利得/実データ位置精度/whole GNSS・IMUは未評価。

2,070ケースの全16状態/log weight/14件5epoch軌跡がbit一致。cache/block境界、非対称/非SPD/
非有限/微小大cov、masked/near/invalid-S/empty/growth/resampleを含み、全72条件の返却値/最後の状態も一致。
関連570テストPASS、追加60件の独立行列/分割cloud bit/後続ESS・estimate・resample/所有権を検証。
memcheck0 errors/0 bytes leaked、racecheck0 hazards/errors/warnings、synccheck0 errors。
既存systematic CDF境界変動は範囲外。GPU製品名非記載を維持。

対称積再利用/共分散load後退/64-thread版は不採用。各1,728bit一致でも大粒子数の安定利得なし、
前2案はregister88/90へ増加。非交互1round×20の予備測定を採用証拠にせず、probe raw timings/
manifest/source・binary hashes/元AST同一scriptと3復元patchを公開した。
測定/再現は `benchmarks/pf_kf_launch_gpu_20261012.json` / `benchmarks/bench_pf_kf_launch_compare.py` /
`benchmarks/RESULTS.md` / `tests/test_pf_kf_launch_layout.py`。予備案は
`benchmarks/pf_kf_probes_20261012.json` / `benchmarks/probes/bench_pf_kf_launch_probe.py` と3 patch。

## Student-t PRの専用化（2026-10-12）

KF loop抑制/固定12衛星/分母共有は各1,728bit一致でも大粒子数の安定利得なし/悪化で不採用。
分母共有の区間certificateはPF検証のみで独立の全域division検査ではなく、予備KF12は約7%遅かった。
非交互1round×20のraw/manifest/source・binary hashと復元patch/headerを
`benchmarks/pf_kf_math_rejected_20261012.json` に保存し、元の128-thread KFを維持。

採用はnu>0/particle gate無効のStudent-t PR専用kernel。GPU上で元の逆スケール/half係数をblockごとに
一度計算し、2衛星unroll。元のrange/residual/積/log/累積順/FP64/overwrite/zero-weight動作を保持。
Student優先のためHuber指定時も専用経路、正gateはgeneral、nu<=0は既存経路。
追加owner buffer/cudaMalloc0、3H2D/同期/block256/launch/API同じ。register48/旧44、stack/local0、
static shared16/dynamic5*n_sat*8。1,228衛星49,136 bytesで48 KiB内の境界を検証した。

直前KF-launch版の5 source/binary hashへbaselineを固定。CUDA12.8/MSVC Release/compute capability7.5/
C++20/fast mathなし、別process5交互round×20、同期込みwall time、全80条件保存。
100万粒子nu3 Student12 8.4165→7.9962 ms（1.053×）、公開Student全体
17.1683→16.7782 ms（1.023×）（約2.3%短縮）。4衛星1.109×/32衛星1.034×、public PR+KF1.008×。
単独12/公開全体のround範囲は分離。未変更control/小粒子数の変動も公開し、予備大きな速度比や
過去速度比の積、全条件固定利得/実データ位置精度/whole GNSS・IMUの利得を主張しない。

2,185ケースの全16状態/log weight/12件5epoch軌跡がbit一致。nu/scale/非有限/zero-weight/near/gate/
Huber優先/growth境界、全80測定条件の返却値と最後の状態も検証。関連647テストPASS、追加77件は
独立NumPy/元general bit参照/最大sharedもPASS。memcheck0 errors/0 bytes leaked、racecheck0 hazards/
errors/warnings、synccheck0 errors。systematic CDFの既存境界変動は範囲外、GPU製品名非記載を維持。
測定/再現は `benchmarks/pf_student_gpu_20261012.json` / `benchmarks/bench_pf_student_compare.py` /
`benchmarks/RESULTS.md` / `tests/test_pf_student_weight.py`。

## BVH構築のgeometry/scratch再利用（2026-10-12）

採用: CPU SAH構築中だけtriangle AABB/3軸centroidをcacheし、prefix/suffix作業領域を再帰で共有。
FP64式/累積順/3軸12bin/比較/tie/sort/分割位置/leaf4/ノード順/三角形順は同じ。
cacheは各buildで再生成して終了時破棄。GPU section source bytes/5関数SASS一致、API/常駐buffer同じ。
一時geometry72*n_tri、prefix/suffix各48*n_tri（>4）、計168*n_tri bytes。
元の各nodeの2 vectorを置き換え、全体peakメモリの実測値ではない。

直前combined BVH source5件/binaryをhash固定、実旧新版同一process順次5交互round×20。
12,000三角形/12衛星/260,000サンプルの構築→IQ→close全体は
48.8338→38.0132 ms（1.285×）、
120,000三角形では479.2411→341.4804 ms（1.403×）。
全28条件/raw/hash/warm controlを保存し、初期化/メッシュ更新の利得と位置づける。
host非専有のround変動、warm epoch/small meshの変動、既存の完全退化大mesh stack上限を明記。

97構築のnode/index bits、87GPU ray結果と全timed output/Urban IQ・metadata一致。
関連312PASS/0SKIP、追加44件の独立tree bounds/coverage/入力変更/linear rays参照をPASS。
memcheck0 errors/0 leaked。PF/信号generator sourceとbuild/installed binaryは直前版のhash維持。
測定/再現: `benchmarks/bvh_build_gpu_20261012.json` / `benchmarks/bench_bvh_build_compare.py` /
`benchmarks/bvh_build_device_code_20261012.json` / `tests/test_bvh_build_cache.py` / `benchmarks/RESULTS.md`。

## IQ搬送波係数のGPU block共有（2026-10-12）

採用: 32ch以下で元のGPU FP64 kTwoPi*(IF+doppler)をblockごとに一度計算し、shared256 bytesから参照。
全threadがbarrierを通ってからpartial blockの範囲外をreturn。reg32/stack・local0を維持。
時間division/chip式/整数剰余/phase最終演算/sincos/nav/order/Philox/noiseは同じ。
owner buffer/packing/転送/launch/block256/API同じ。>32/global/noise-only/singleのSASS完全一致。
PF/直前BVHのsource/build/installed binary hashも保持した。

直前signal workspace版source5件/binaryをhash固定、別processのpersistent workerでGPU処理順次。
stageごと5交互round×20。public12ch/2.6m samplesは24.3171→19.8292 ms
（1.226×）、Urban12k三角形/12sat/260k samples全体は
5.8920→5.4444 ms（1.082×）。CPU BVH構築除外。
全54条件/raw/hash、短epoch初回負荷変動と追加5交互round×20の3条件を公開し、小幅低下も記録。
377IQ bit hashes/各build内warm-cold uint32/全timed IQ・metadata一致、361関連PASS。
新49件のsingle/global参照・cancellation・partial blocks・reuseをPASS、mem/race/syncすべて0。
整数剰余の案は3交互round×10で安定した全体利得なし、bit一致でも棄却してpatch/rawを保存。
測定/再現: `benchmarks/signal_frequency_gpu_20261012.json` / `benchmarks/bench_signal_kernel_compare.py` /
`benchmarks/signal_frequency_short_check_20261012.json` / `benchmarks/signal_modulo_rejected_20261012.json` /
`benchmarks/signal_frequency_device_code_20261012.json` / `tests/test_signal_frequency_cache.py`。

## Global descriptorの搬送波係数共有（2026-10-12）

採用: 33〜256chも元のGPU FP64搬送波係数をblockごとに計算してshared2048 bytesから参照。
256threadでwriterを覆うcompile-time assert、barrier後のpartial return、reg42/旧44/stack・local0。
CPU packing/転送/owner buffer/launch/block256/API、時間/chip/nav/累積/Philox/noiseは同じ。
既存inline/noise-only/single/global fallbackの4関数SASS完全一致。PF/直前BVHのsource/binary保持。

直前inline-frequency版のsource5件/binaryへbaseline固定。GPU順次のpersistent workerで
stageごと5交互round×20、全48条件/raw/hash/実Urban生成ch数を保存。
public256ch/2.6m samples 77.9126→61.3208 ms（1.271×）、
追加5交互round×20でも76.9202→59.0877 ms（1.302×）。
本測定の1roundで低下、短epoch/未変更fallbackの変動を記録。Urban全体は追試の中央値比較で
0.864×へ逆転（round対の4/5では新版が速い）し、安定した全体利得は未確認。
改善を多chの長い信号生成に限定して位置づけ、過去速度比を積算しない。

457IQ bit hashes/各build内uint32/全timed IQ・metadata一致、416関連PASS。
新55件のsingle/uncached参照/境界/cancellation/filter/更新をPASS、既存49件のpaddingを257に移して
独立参照を維持。計104件mem/race/syncすべて0。測定/再現:
`benchmarks/signal_global_frequency_gpu_20261012.json` / `benchmarks/signal_global_frequency_check_20261012.json` /
`benchmarks/bench_signal_global_frequency_compare.py` / `benchmarks/signal_global_frequency_device_code_20261012.json` /
`tests/test_signal_global_frequency_cache.py` / `tests/test_signal_frequency_cache.py`。

## 常駐BVHのGPU法線cache（2026-10-12）

採用: parallel反射の三角形法線をGPU上で初回のみ計算し、ownerの訪問順cacheで再利用。
元のFP64 cross/sqrt/division/strict cutoffを維持し、反射式・tie ordinal・fallbackを保持。
cache32 bytes/訪問三角形、12kで384,000/120kで3,840,000 bytes追加。LOS-only/serial-onlyはlazy未生成。
allocated_bytes計上/close解放/update時再生成。cached reg78/shared5120/stack・local0、準備reg36。
既存5関数のSASS instruction body完全一致、PF/signal直前source/built・installed binaryも保持。

直前BVH build-cache版source5件とpydへbaseline固定、順次5交互round×20/48 timed条件。
公開12k/192 rays 5.6700→3.7431 ms（1.515×）、120k/192 rays 51.7678→32.0369 ms（1.616×）。
120k/12sat/260k samples実Urban全体5.4601→3.9921 ms（1.368×、全5 round改善1.321〜1.407×）。
小規模・少数rayのcoldは1未満もあり、追加malloc/kernelのコストも全rawへ記録。
未変更LOS/小mesh/4097-ray serial controlsを測定、全timed出力hash一致。過去速度比を掛け合わせない。
110 LOS/反射bit一致、459関連PASS、新43件mem/race/syncすべて0・leak0。
測定・再現: `benchmarks/bvh_normal_gpu_20261012.json` / `benchmarks/bench_bvh_normal_compare.py` /
`benchmarks/bvh_normal_device_code_20261012.json` / `benchmarks/probes/bvh_resident_normal_cache_20261012.patch` /
`tests/test_bvh_normal_cache.py`。

## 常駐BVHの受信機鏡像点共有（2026-10-12）

採用: 同epochのsatellitesへ元のFP64鏡像点を共有。GPUで毎回再計算し、input更新で古い値を使わない。
visited>=8192/satellites>=16/total rays<=4096/scratch<=32MiBに限定。
方向正規化/MT交差/距離/tie/serial/stateless維持。24 bytes×visited×epochsのscratchを再利用・bytes計上・close解放。
shared候補reg77/旧78/shared5120/stack・local0、準備reg30。既存7関数SASS instruction body完全一致。
PF/signal直前source/built・installed binary保持。

GPU順次5交互round×20/54 timed条件、132 LOS/反射bit一致、全timed IQ・metadata hash一致。
公開120k/192 rays 32.0752→28.7006 ms（1.118×）、12k/192 rays 3.7584→3.4069 ms（1.103×）、
両条件5 roundとも改善。120k/32sat/260k samples実Urban全体10.2990→9.8646 ms（1.044×）だが
round比1.005〜1.066×/baseline8.60〜12.27msと変動が大きく、一律の都市利得は主張しない。
484関連PASS、新25件mem/race/syncすべて0・leak0。過去速度比を掛け合わせない。

見送り: edges/v0/normal104-byte geometry recordは旧32-byte cacheの3.25倍、3×10探索で反射約1.075×でも
初回・全Urban利得が安定せず撤回。8衛星からmirror共有する案は5×20で12sat全Urban0.997×/0.984×、
16sat以上へdispatchを限定。raw/各exact CRLF patch/当時runnerを残し、sanitizer結果を不採用案へ移さない。
測定・再現: `benchmarks/bvh_mirror_gpu_20261012.json` / `benchmarks/bench_bvh_mirror_compare.py` /
`benchmarks/bvh_mirror_device_code_20261012.json` / `tests/test_bvh_mirror_sharing.py` /
`benchmarks/bvh_geometry_rejected_20261012.json` / `benchmarks/bvh_mirror_dispatch_probe_20261012.json`。

## BVH追加3案の探索・不採用（2026-10-12）

実効性重視でwarp選択/tile上限解除/block距離共有を実ビルド順次3交互round×10で探索。
全timed出力hash一致、case数132/158/158、timed条件28/32/28。
warp shared5120→160/barrier8→1でも公開12k/192 rays 0.994×、reg増78→82/77→78。
tileは全9関数SASS一致でも公開120k/1〜2 rays 0.945×/0.946×。
distance共有は158bit一致、公開12k/192 rays 0.998×、512/4096 raysの1.026×のみ。
GPU profilerで原因を確定せず、3案とも正式採用を見送り。前mirror-sharing source/pydをbyte一致で復元し、
stale objectを防ぐため復元sourceを再ビルド。484関連PASS、PF/signal source/built・installed binaryも保持。
不採用案に既存sanitizer結果を移さず、測定raw/CRLF再現patch/runner/resource/SASS auditを保存。
`benchmarks/bvh_warp_selection_rejected_20261012.json` / `benchmarks/bvh_tiles_rejected_20261012.json` /
`benchmarks/bvh_distance_rejected_20261012.json` / `benchmarks/bvh_probes_device_code_20261012.json`。

## DD搬送波のGPU逆波長共有（2026-10-12）

採用: DD CP/jointのshared観測load最終columnで元のFP64 1/wavelengthを計算し再利用。
残差/rint/gate/Huber/集計順/CP加算・joint上書き保持、CPU packing/H2D/owner bytes/launch/barrier1/block256/APIも同じ。
reg48旧と同じ/static shared・stack・local0、42/44関数SASS instruction body一致、BVH/signal source/built・installed保持。

Student版source5件/pydへbaseline固定、GPU順次persistent workerでstageごと5交互round×20/64条件。
公開1m/12DD joint全step17.2490→16.4796ms（1.047×）、追加7条件5×20でも17.0426→16.3059ms（1.045×）、
両方の全5 round改善。CP12は両方約1.15×だが追試1round0.984×。
未変更controlの大きな変動、CP4 1.464×→0.925×、joint12 0.988×→1.096×、gate1.844×→1.109×を全raw記録。
最大値・一律の利得は主張せず、安定した公開12DD全stepを採用根拠とする。過去速度比を掛け合わせない。
2316 state/log/trajectory bit一致、別16小geometry/gate条件の各CP/joint/splitも旧新版bit一致。
joint/split間の旧からのFMA丸め差は数値比較で許容し、変更前後bit比較を別checkerで確認。
759関連PASS、新59 mem/race/syncすべて0・leak0。独立単独AFV/DD PR/48KiB/partial/更新を検証。
測定・再現: `benchmarks/pf_dd_wavelength_gpu_20261012.json` / `benchmarks/pf_dd_wavelength_check_20261012.json` /
`benchmarks/pf_dd_wavelength_gate_check_20261012.json` / `benchmarks/pf_dd_wavelength_device_code_20261012.json` /
`benchmarks/bench_pf_dd_wavelength_compare.py` / `tests/test_pf_dd_wavelength_cache.py`。

## PF DD共通参照の距離再利用を採用 (2026-10-12)

DD PR・搬送波の単独更新では、有限な全参照ECEFがbit一致する2組以上の呼び出しだけ、
GPUで粒子ごとの参照距離を1回計算する。mixed/signed-zero/nonfinite/single-DDの元の命令列を保持する。
100万粒子/12 DDの公開PR→CP分割PF全stepは21.0943→17.1320 ms（約18.8%短縮）、
5交互round全て1.222〜1.232×。先行3×5も約18.7%短縮。
4,338+144ケースの旧新版bit一致、809関連tests、新50testsのCUDA mem/race/sync0・leak0、
既存44 GPU instruction bodies一致を確認した。
joint試作は38ケースのbit不一致のため見送り、元のjointのsource/命令列を保持。
一括PF・混在参照・全サイズへの一律の利得は主張しない。過去の速度比は掛け合わせない。
source/build/installed binary・raw計測・棄却probe・再現patchを
`benchmarks/pf_dd_reference_{gpu,device_code,probe,precision_check,joint_rejected}_20261012.json` と
`benchmarks/RESULTS.md` に記録。

## PF DD jointの共通参照を丸め保存ガード付きで採用 (2026-10-12)

Gaussian CPの最終乗算を `__dmul_rn` で明示。carrier-only Huberはbit差が出たため元のjointカーネルを使う。
既存46 GPU instruction bodiesを保持し、共通参照の公開100万粒子/12 DD PF全stepは 16.3814→14.4195 ms（1.136×、約12.0%短縮）。
追加測定も 16.3794→14.4223 ms（1.136×、約11.9%短縮）、両方全5組で改善した。
5,382+288旧新版bitケース、881tests、新72testsのCUDA mem/race/sync0・leak0を確認。
前回差が出た粒子239の正確な重みをfixtureへ保存し、1/31/257粒子でbit照合した。
共通参照以外・全サイズ・全Huber構成への利得は主張しない。過去の速度比は掛け合わせない。
ガード前の計測は別保存。最終source/build/installed binary・raw・SASS・再現patchを
`benchmarks/pf_dd_joint_reference_{gpu,check,device_code}_20261012.json`、
`benchmarks/pf_dd_joint_options_check_20261012.json`、`benchmarks/RESULTS.md` に記録。

## D-153: DD係数共有・Gaussian gate geometry統合・epoch一括生成を採用（2026-10-12）

状態: 採用

精度を保つ3候補を全て実装・比較した。DD係数共有は100万粒子の全stepで約0.7〜1.1%、
gate統合はゲート付き全stepで17.87→14.73ms（約17.6%）、
epoch一括生成は32epoch公開APIで7.57→4.16ms（約45.0%）、都市全体で26.05→23.47ms（約9.9%）短縮。
5交互round×20で実production旧新buildを順次実行し、入力/全timed出力を照合。
特殊波長を含む未ガード版はsigma48/gate30のbit差で棄却し、従来経路へのguardを採用。
sigma・gate各6,102、最終PFの極端値360、signal IQ/metadata524ケースでbit一致。
関連1,218 tests、追加111 testsの3 CUDA sanitizersがpass（errors/hazards/leaks0）。
既存device命令はsigma46/gate49/signal5を保持。追加kernelにstack/local spillなし。
都市batchはopt-in、既定のepoch単位callback/yieldと単一epoch APIを維持。
scene callbackの先読み・IQ32MiB chunk上限・seed/lifetime/fallbackをdocとtestで固定。
単独kernelの揺れや変更対象外の利得は主張しない。直接のray tracing利得としても扱わない。
phaseごとのsource/pyd/raw/helper/SASS/resource/SHA・再構成patchは
`benchmarks/pf_dd_sigma_*_20261012.json`、`benchmarks/pf_dd_gate_*_20261012.json`、
`benchmarks/signal_batch_*_20261012.json`、`benchmarks/RESULTS.md` に記録。

## D-154: 両HuberのDD gate geometry統合を採用しGaussian partial-NaN条件を修正（2026-10-12）

状態: 採用

共通参照/両Huber/正有限k/2〜511DD/gate有効/有限観測/normal正波長だけ単一geometry loopを使う。
100万粒子・12DDの公開PF全stepは21.7786→18.2760ms（約16.1%短縮）、
10万粒子も約15.8%短縮し、いずれも5交互round×20の全5組で改善。
初期版はweighted Huber costの乗算/accumulator加算の丸めが変わり451ケースで棄却。
PR FMA、CPの乗算後の丸め+加算を明示し、最終6,102ケースでbit一致。
小粒子数・変更対象外controlの揺れは新処理の利得として扱わない。
さらにGaussian fused gateのpartial NaN条件を従来jointの `<=` に修正。
有限sat座標のrange overflowで4DD中3DDが残るfixtureのNaN誤加算を直し、元の+0.0とbit一致。
不足時fallbackはNaNを含む元の全観測和を保つ。前回の極端値検査にこのpartial-kept条件が不足していた。
関連1,280 tests、追加62 testsの3 CUDA sanitizersがpass（errors/hazards/leaks0）。
既存49 instruction bodiesを維持、Gaussian predicate1関数変更、Huber59register/no spillsの1関数を追加。
API/既定設定、BVH本体・signal batchのsource/binaryは維持。
raw/device/initial rejected/partial-NaN fixture/再構成patchを
`benchmarks/pf_dd_huber_gate_*_20261012.json`、`benchmarks/RESULTS.md` に保存。

## D-155: 個別Gaussian DD gateも単一geometry loopへ統合（2026-10-12）

状態: 採用

共通参照・Gaussian・正gate・有限観測（CPはnormal正波長）の個別PR/CP更新へ適用。
100万粒子の公開PR→CP全stepは23.1002→17.4678ms（約24.4%短縮）、5交互round×20の全5組で改善。
native全stepも約24.4%短縮、公開10万粒子も全5組で改善。小粒子数は一律の改善を主張しない。
元のPR/CPそれぞれの積結合・FMA・観測順・最少kept数・fallbackを保つ。
PR上書き/CP加算、個別APIのskip-if-greaterに従うNaNのcount/filter動作も元のGPU式とbit一致。
6,102比較ケース、関連1,334 tests、追加54 testsの3 CUDA sanitizersがpass（errors/hazards/leaks0）。
既存51 instruction bodiesを全て維持。新PR46/CP40register、stack/local0の2 kernelsを追加。
shared/H2D/allocation/barrier/launch数/API/既定設定、joint/BVH本体/signal batchを維持。
raw/device/再構成patch/最終整合確認を `benchmarks/pf_dd_single_gate_*_20261012.json`、`benchmarks/RESULTS.md` に保存。

## D-156: 個別Huber DD gateも単一geometry loopへ統合（2026-10-12）

状態: 採用

共通参照・有限正Huber k・正gate・有限観測（CPはnormal正波長）の個別PR/CP更新へ適用。
100万粒子の公開PR→CP全stepは26.1994→20.5405ms（約21.6%短縮）、5交互round×20の全5組で改善。
native全stepも全5組で改善。粒子数別の実測と変更対象外controlも公開し、一律の利得を主張しない。
元のHuber正規化・weight-cost FMA・観測順・最少kept数・fallbackを保つ。
PR上書き/CP加算、個別APIのskip-if-greaterに従うNaNのcount/filter動作も元のGPU式とbit一致。
6,486比較ケース、関連1,396 tests、追加62 testsの3 CUDA sanitizersがpass（errors/hazards/leaks0）。
既存53 instruction bodiesを全て維持。新PR52/CP53register、stack/local0の2 kernelsを追加。
H2D/allocation/barrier/launch数/API/既定設定、個別Gaussian/joint/BVH本体/signal batchを維持。
raw/device/再構成patch/最終整合確認を `benchmarks/pf_dd_single_huber_gate_*_20261012.json`、`benchmarks/RESULTS.md` に保存。

## D-157: 混在参照の個別DD gateを単一loopへ統合しsignal descriptor cacheは棄却（2026-10-13）

状態: 採用（PF）/ 棄却（signal試作）

mixed referenceでもper-pair距離を維持して重複するgate/尤度loopを統合。
100万粒子の公開全stepはGaussian 31.4688→21.5671ms（約31.5%短縮）、
Huber 34.8032→24.6530ms（約29.2%短縮）。公開/nativeとも全5交互roundで改善。
有限観測/正gate/2組以上DD/Huber正有限k/CP normal正波長へ限定し、特殊入力と他の経路は維持。
PR/CP各積の結合・sigma GPU式・FMA・観測順・NaN/kept/fallback・上書き/加算を保つ。
6,678旧新版bit比較、関連1,456 tests、新60 testsの3 CUDA sanitizersでerrors/hazards/leaks0。
partial gateは採用subsetの元のungated GPU計算と独立にbit照合。既存55 SASS全同一、新4 kernelsにspillsなし。
signal batch descriptor cacheは633bit一致だが、公開/都市全体の安定した利得がなく棄却。
signal source/build/installedを元に復元し、BVHも保持。棄却source/pyd/helper/raw/patchも保存。
raw/device/patch/completionと棄却記録を `benchmarks/pf_dd_mixed_gate_*_20261012.json`、
`benchmarks/signal_batch_descriptor_*_20261012.json`、`benchmarks/RESULTS.md` に記録。

## D-158: 混在参照のjoint DD gateを単一geometry loopへ統合（2026-10-13）

状態: 採用

mixed jointのper-DD距離・独立gate/fallback・PR FMA・CP multiply/add丸めを保ち、重複geometry loopを除く。
100万粒子の公開全stepはGaussian 22.0870→16.7508ms（約24.2%短縮）、
Huber 25.8809→20.1617ms（約22.1%短縮）。公開/nativeとも全5交互roundで改善。
有限観測/少なくとも片側正gate/2組以上DD/normal正波長、Huber両側正有限kへ限定。
元のjoint `<=` NaN filterと不足時の全観測和、log weights上書きを保つ。
7,098旧新版bit比較、関連1,528 tests、新72 testsの3 CUDA sanitizersでerrors/hazards/leaks0。
partial gateは各sideの採用subsetと元のungated joint GPU costを独立にbit照合。
既存59 SASS全同一、新Gaussian50/Huber54 registers、spillsなし。
12*nDD sharedを保ち512DDも48KiB内。H2D/allocation/launch/API/既定設定、common joint/個別mixed gates/BVH/signalを維持。
raw/device/patch/completionを `benchmarks/pf_dd_mixed_joint_gate_*_20261013.json`、`benchmarks/RESULTS.md` に記録。

## D-204: RB-FGO-PFの共有候補をエポックごとに一度採点 (2026-10-15)

状態: 重複削減を採用、安定した全体高速化の主張は保留

TRACK=0/2のconsensus/LAMBDA/spawn候補は粒子共通のfloat priorで採点するため、
64粒子で同じ候補のID生成とlogaddexpを64回呼ぶ処理を1回にする。
粒子別survive/extend/shrink、移動順、prior計算、乱数消費、Hのcopyは維持する。
TRACK=1の粒子別prior経路は対象外。永続する可変Hのキャッシュを追加しない。

Tokyo run1実入力600epoch、64粒子/K12/seed20260710、各source 1warmup後3交互pair。
先行studyはPF時間のpaired短縮中央値-4.47%、全体経過-12.67%、1/3pair改善。
追加のCPU時間studyは全体CPU 12.94%、PF経過22.03%、全体経過12.98%短縮、3/3改善。
両studyを残し、時刻による大きな変動から全体の安定した高速化には一般化しない。
16実行全てで位置bytes、診断88配列、行dump 2配列が完全一致し、除外0。
全体5320 passed/3778 skipped、新規46 cases、型検査errors0。
これは序盤のFIX優勢区間であり、full-route/ensemble性能や測位精度改善の証明ではない。
記録: benchmarks/rbpf_shared_moves_tokyo_prefix_20261015.json、
benchmarks/rbpf_shared_moves_cpu_tokyo_prefix_20261015.json、experiments/rbpf_fgo/README.md。

## D-203: 厳密CDF逆検索・全16状態gather・重みresetの統合 (2026-10-15)

状態: 採用

- D202のdirect経路で同じRNしきい値/整数逆検索/left/clampを使い、全16状態copyと+0.0 resetまで1kernel化。共有ancestorは全laneが初期化し無条件barrierを経由、共分散は元の連続component/lane対応を維持。
- D202のsourceN>=262144/全weight検査/policy/identity/容量fallback/診断CDF/cache無効化/完了同期を維持。新CUDA owner/確保/layoutなし。全87既存GPU body/resource同一。
- D202比、100万↔50万切替3.419→3.055ms（10.66%短縮）、切替込み5epoch62.800→61.386ms（2.25%短縮）。posteriorは28.904→28.955ms（0.17%悪化）。
- 候補6,904件freshがD202の6,817参照+87追加freshと一致。最終回帰11,375、GPU87件のCUDA検査3種0error/leak/hazard。
- 計測外のclose末尾でpool/peerの循環参照を解放。残り全AST/検証関数/演算/timer/CUDA source/PEは同一で、弱参照とhash/patchにより検証。修正前の不安定な全測定も保持。
- 修正後36条件/225pairに加え、反復を増やした135pair/15snapshotで全体への効果を確認。先行5epochの1.05%/3round改善も明記。6peakと初期75pair/15snapshotを保存。長時間実データ誤差曲線・multi-device/他archは未追加。
- [測定](../benchmarks/pf_fused_uniform_gpu_20261015.json)、[コード](../benchmarks/pf_fused_uniform_device_code_20261013.json)、[完了](../benchmarks/pf_fused_uniform_completion_check_20261015.json)。

## D-202: GPU全重み検査後の厳密なuniform CDF逆検索 (2026-10-15)

状態: 採用

- D200の整数逆数/high64/積の補正による厳密な逆検索を、D201のGPU全重み検査と組み合わせて再評価。逐次FP64 CDF/元のRNしきい値/left同値境界/全16状態/重み/乱数/モデル/粒子数を維持。
- sourceN>=262144でCDF全体の書込みを通常省略し、小粒子はD201の元の本体を実行。診断出力は従来通り全ビット生成。追加metadataがscratchに収まらなければ元のCDF/探索へ戻る。新CUDA owner/確保/layout変更なし。全86既存GPU body/resource同一。
- D201比、100万↔50万切替4.242→3.618ms（14.72%短縮）、切替込み5epoch65.076→62.572ms（3.85%短縮）。
- 最終候補6,817件freshがD201の6,799参照+18追加freshと一致、新旧530unit、最終回帰11,287。最終境界18件のCUDA検査3種0error/leak/hazard。制限前237件とsource reduction probe12件の3種検査を保持し、全87GPU body/resource同一。
- 36batch条件/900pair/180snapshot/6peak/初期75pairを保存。batchで悪化した10万粒子などを追加675pairで確認し全結果を保持。長時間実データ誤差曲線・multi-device/他archは追加評価なし。
- [測定](../benchmarks/pf_scan_inverse_gpu_20261015.json)、[コード](../benchmarks/pf_scan_inverse_device_code_20261013.json)、[完了](../benchmarks/pf_scan_inverse_completion_check_20261015.json)。

## D-201: 重み全要素の有限・同値検査をGPUへ移し読み戻しを4byteに縮小 (2026-10-15)

状態: 採用

- 毎回全weightをIEEE754整数bitで有限・同値検査し、正負zeroを同値扱い、整数ORで全blockの判定を集約する。cache/部分sample/近似なし。
- sourceの既存reduction scratchを借り、4byte flagだけ読み戻して完了同期。Trueは元のD199逐次FP64 CDF/left/clamp/all16 gather/reset、Falseは元のCPU経路。
- 元のPython/native getter、uniform/checked CDF pair、state type/public-native count/NumPy policyを確認し、差替え・旧backend・strict policyはfallback。PF layout/全84GPU body/resource/粒子数/乱数/モデル/許容誤差を保つ。
- 100万↔50万は12.27→4.23ms、約65.5%短縮、posteriorは25.35→21.64ms、約14.6%短縮、連続5epochは105.08→64.39ms、約38.7%短縮。
- 候補全6,799件fresh、baselineはhash確認済みD199の6,587参照+212追加fresh、旧新137unit、回帰11,269、CUDA検査3種各3,148件0error/leak/hazard。
- 36条件+36probe/225pair/45snapshot/6peakと悪化した対照も保存。長時間実データ誤差曲線とmulti-device/他archは追加評価していない。
- [測定](../benchmarks/pf_uniform_scan_gpu_20261015.json)、[コード](../benchmarks/pf_uniform_scan_device_code_20261013.json)、[完了](../benchmarks/pf_uniform_scan_completion_check_20261015.json)。

## D-200: uniform CDF区間表の直接逆検索 (2026-10-15)

状態: 保留

- 厳密なCDF区間表からleft検索を整数で逆算する候補。元のFP64 thresholdとclampを維持し、整数除算版・整数逆数補正版の両方を保存。
- CPU約1,051万threshold/約19万ceil division一致、旧新80test通過。native20往復は約18.3%短縮したが、5epoch全体で安定した実効改善を示せず採用しない。
- 全回帰/CUDA検査へは進めず、全source/PE/helper/測定を保存し、通常buildで全84GPU一致を確認、D199の正確なPE/sourceを復元してD201を開始。
- [全結果](../benchmarks/pf_uniform_inverse_gpu_20261015.json)、[候補patch](../benchmarks/pf_uniform_inverse_device_code_20261013.json)。

## D-199: 逐次uniform CDFの全ビットを保つGPU整数再構成 (2026-10-15)

状態: 採用

- D198の6source/PE/helperを凍結。有限かつ同値の重みでpool切替時、元の逐次FP64 cumsum(1/N)を同じ丸め結果の整数区間表からGPU再構成する。halfway tie/指数境界はCPUで元の加算を実行する。
- 既存source CDF/target ancestor-B scratchを再利用し、CPU CDF生成と全要素H2Dを省く。新CUDA owner/確保なし。元の83GPU body/resource、検索/clamp/gather/重みkernelと全16状態を維持し、整数CDF kernelを1個追加。
- 元のchecked CDF hook pair/粒子数/NumPy policy/有限同値/丸め/容量/stream寿命を確認。小粒子、非nearest、in-place/cross-device/他arch、差替え・旧backendは変更前declineと元のfallback。
- 100万↔50万切替は18.71→12.13ms、約35.1%短縮、posteriorは26.52→24.45ms、約7.8%短縮、連続5epochは134.25→104.93ms、約21.8%短縮。
- 6,587record bit一致、旧新各232+63test、回帰11,052、2,931caseのCUDA検査3種0error/leak/hazard。CDF全要素bit/queued scratch/丸めmode/例外を確認し、粒子数/乱数/測位モデル/許容誤差を変えない。
- 36条件+36probe/225交互pair/45snapshot/6peakと悪化した対照も保存。長時間実データの誤差曲線とmulti-device/他archは追加評価していない。
- 証跡: [測定](../benchmarks/pf_uniform_cdf_gpu_20261015.json)、[コード](../benchmarks/pf_uniform_cdf_device_code_20261013.json)、[完了](../benchmarks/pf_uniform_cdf_completion_check_20261015.json)。

## D-198: host CDF検証の全条件を対応CPUのSIMDで確認 (2026-10-15)

状態: 採用

- D197の6source/PE/helperを凍結。変更はC++ bindingのhost CDF検証だけ。CU/header/Python/runtime/config/realtime/全83GPU bodies/resources/layoutは同じ。
- x86-64のbaseline SSE2、N>=65536でCPU/OS対応AVX2は未整列8要素、未対応x86-64は元のSSE2で4要素ずつ。整数指数maskでInf/qNaN/sNaNをrejectし、全隣接組を元と同じFP64 `<` で確認。先頭/末尾と小粒子/他archはscalar。
- 検証を省かず、native/state/scratch/stream/GPU操作の前にdeclineする。API/owner/array cache/動的確保/転送量/粒子選択/CDF/モデル/乱数/粒子数/許容誤差を保つ。実host命令のCPU/OS guardとAVX2分離も監査。
- 100万↔50万の切替は18.75→17.92ms、約4.5%短縮、posteriorは27.19→26.01ms、約4.3%短縮、連続5epochは139.47→133.68ms、約4.1%短縮。
- 6,305recordを照合、旧新追加2,520test通過、回帰10,757、2,636caseのCUDA検査3種0error/leak/hazard。
- 36条件+36probe/225交互pair/45snapshot/6peakを保存。未変更controlの変動と悪化した測定も保持。
- 長時間実データ誤差曲線とmulti-device/non-x86は追加評価していない。CDF検証を通るpool処理への効果。
- 証跡: [測定](../benchmarks/pf_cdf_validate_gpu_20261015.json)、[コード](../benchmarks/pf_cdf_validate_device_code_20261013.json)、[完了](../benchmarks/pf_cdf_validate_completion_check_20261015.json)。

## D-197: 均一重みのpool正規化を同じFP64の1/Nへ直接fill (2026-10-14)

状態: 採用

- D196の6source/PE/全helperを凍結。変更はrealtimeのCPU正規化だけ。native PE/全83GPU bodies/resources/layout/config/PF runtimeは同じ。
- 既存の全有限値1D N>=65536と普通のwarn/ignore policy gateで、private copy/maxの後に先頭・中央・末尾==maxならminで均一を確認。
- 等しい有限値のsubtract→exp→sum→divisionが与える正確なFP64の1/Nをfillし、元のNumPy逐次cumsumを保つ。
- 非均一は元の演算、小粒子/matrix/nonfinite/strict/warning-errorは元のeager fallback。caller入力/CDF検索/粒子選択/モデル/乱数/粒子数/許容誤差を維持。
- 100万↔50万の切替は25.59→19.68ms、約23.1%短縮、PR posteriorは35.11→32.41ms、約7.7%短縮、連続5epochは178.57→144.97ms、約18.8%短縮。
- 3,785recordを照合し、追加1058direct CDF/policy testは旧新通過。回帰8,237、実pool90caseのCUDA検査3種は0error/leak/hazard。
- 24条件+24probe/150交互pair/30snapshot/6peak/12CPU対照を保存。未変更controlの変動と非均一経路の追加min負担も保持。
- GPU転送量/確保は同じ。均一weightを含むpool切替への改善。通常PF/raytrace/signal全体の改善へ一般化しない。
- 長時間実データ誤差曲線と実機multi-deviceは追加評価していない。
- 証跡: [測定](../benchmarks/pf_pool_uniform_gpu_20261014.json)、[コード](../benchmarks/pf_pool_uniform_device_code_20261013.json)、[完了](../benchmarks/pf_pool_uniform_completion_check_20261014.json)。

## D-196: CPUで求めた均一log-weightをFP64 bitsのままGPUへbroadcast (2026-10-14)

状態: 採用

- D195を凍結。元CPU -math.log(N)を8registerのstore kernelで全weightへ写す。浮動小数点演算/新GPU owner/確保/layout変更なし。
- 元82GPU SASS/resources、CDF/検索/選択/測位モデル/FP64/粒子数/乱数/許容誤差は維持。ESS無効化/stream完了は元setter同様。
- optional APIはlarge standard Python/native setterだけ。元identityでinstance/class/module/native hook/custom uniform overrideを守り、old/small/unpaired/count mismatchは元array setter。
- 100万↔50万切替は42.56→35.40ms、約16.8%短縮、PR posteriorは50.59→46.48ms、約8.1%短縮、連続5epochは306.19→270.55ms、約11.6%短縮。3対象5/5round改善。
- 2727照合、baseline183+4skip/candidate187、回帰7179、1215case CUDA検査3種0error/leak/hazard。
- 32条件+32probe/200pair/40snapshot/6peakを保存。setter microだけで採用せず、全体改善を確認。未変更control変動は本変更へ帰属しない。
- 最初のsetter measurement fixtureが旧版側で未初期化wrapperを参照した。source/PE/helper/logを保存し、prefix修正後全検査/測定を再実行。期待値は緩めない。
- 長時間実データ誤差曲線/multi-deviceは追加評価していない。次は残るsource weight read/transferと全体profileを確認する。
- 証跡: [測定](../benchmarks/pf_uniform_weights_gpu_20261014.json)、[コード](../benchmarks/pf_uniform_weights_device_code_20261013.json)、[完了](../benchmarks/pf_uniform_weights_completion_check_20261014.json)。

## D-195: 全有限値のpool正規化とCDFをprivate CPU bufferで再利用 (2026-10-14)

状態: 採用

- D194 native/PF Python/realtimeを凍結。変更はrealtimeのCPU正規化だけでnative PE/全82GPU bodies/resources/layout/config/PF runtimeは同じ。
- 全有限値1D FP64かつN>=65536、通常warn/ignore policyだけprivate copyを再利用する。不要floor/min/whereを省く。
- max→subtract→NumPy exp→元sum→division→逐次cumsumを保つ。caller配列/viewを変更しない。
- 小粒子/matrix/非有限/strict/callback/log/print/warning-errorは元式。CDF検索/粒子選択/測位モデル/FP64/乱数/粒子数/許容誤差は維持。
- 100万↔50万の切替は48.83→32.16ms、約34.1%短縮、PR posteriorは54.98→39.50ms、約28.2%短縮、連続5epochは294.02→227.87ms、約22.5%短縮。3対象5/5round改善。
- 2544recordとdirect CDF全要素/警告/例外/handler textを照合、旧新追加368pass、回帰6992、実pool90のCUDA検査3種0error/leak/hazard。
- 24条件+24probe/150交互pair/30snapshot/6peakを保存。未変更controlの変動を本変更へ帰属しない。
- native printがsys.stderrを迂回することを確認し、fd2捕捉/worker stderr fileで計測基盤を修正。停止時source/PE/helper/logを保存し期待値は緩めない。
- 長時間実データ誤差曲線とmulti-deviceは追加評価していない。次は再profileして、残るweight read/transferの効果を調べる。
- 証跡: [測定](../benchmarks/pf_pool_normalize_gpu_20261014.json)、[コード](../benchmarks/pf_pool_normalize_device_code_20261013.json)、[完了](../benchmarks/pf_pool_normalize_completion_check_20261014.json)。

## D-194: 元のCPU CDF上のleft検索だけをGPUへ移す (2026-10-14)

状態: 採用

- D193のnative/PF Python/realtimeを凍結し、CPU finite処理・weight正規化・exp/sum・逐次cumsumを保った。
- FP64加算→除算の閾値とleft二分探索、source上限clampをGPUで実行し、元の全16状態gather/resetを使う。
- warm同一backend/deviceかつmax(source,target)>=65536。小粒子/cold/旧版/mismatch/decline/非有限CDFは従来経路。
- source stream完了後にsource d_cdfをtarget streamで借用し、H2D/search/gather/reset/syncを順序付ける。新規GPU owner/layout変更なし。
- 100万↔50万uniform切替は120.85→55.24ms、約54.3%短縮、PR posteriorは110.08→59.74ms、約45.7%短縮、連続5epochは562.36→304.40ms、約45.9%短縮。3対象5/5round改善。
- 24stage条件と150交互pair/30snapshot、6peakを保存。未変更controlの変動は本変更へ帰属しない。
- 2176照合、baseline2203+17skip/candidate2220、追加guard旧新各8、最終回帰6624、pool1020caseと追加guard8caseのCUDA検査3種0error/leak/hazard。
- getter差替でCDF長さがnative source Nと違う場合も元CPU選択→resident ancestor gatherへ戻る。変更前の拒否を旧新で再現/保存後、最終版の全照合/測定/回帰を再実行した。native source/PEは同じ。
- 既存81GPU SASS/resources同一、新検索1kernel/25register/stack/shared/local0。測位モデル・FP64・粒子数・乱数・許容誤差を保つ。
- 長時間実データ測位誤差曲線と実機multi-deviceは追加評価していない。
- 次は切替内のCPU weight正規化が残る。全体profileを取り、順序/端点/NumPy error policyを変えず効果が再現する案だけ比較する。
- 証跡: [測定](../benchmarks/pf_pool_cdf_gpu_20261014.json)、[コード](../benchmarks/pf_pool_cdf_device_code_20261013.json)、[完了](../benchmarks/pf_pool_cdf_completion_check_20261014.json)。

## D-193: CPU粒子選択を保つresident PF pool gather (2026-10-14)

採用。同一backend/deviceで初期化済みtargetへ切り替える際、元CPU正規化/CDF/searchsorted/clamp/−log(N) weightを保ち、選ばれた全16状態を元gather kernelで直接source→targetコピーする。
indexだけを既存target Megopolis scratchへH2D。source producer sync→target既存work/index/gather/reset/sync。新GPU kernel/buffer/state fieldはなく、全81GPU SASS/resources一致。
source状態/weight/cacheは不変。target ESS/covariance certificateは元importと同様に無効化。旧module/injected/mismatched/cold/declined fallbackを保持。cold mean/initは元CPU。
in-place/cross-deviceは変更前にdecline。実機multi-deviceは未検証。C++ callerのsourceは戻りまで同時変更しない。

100万→50万→100万の往復は378.671→144.991ms (61.71%短縮)。
PR posteriorの切替は363.482→124.589ms (65.72%短縮)。
PR＋各epoch切替の連続5epochは1601.293→643.762ms (59.80%短縮)。3対象5/5round改善。
150旧新交互pair/30両state snapshot、18stagebatch条件、6traced peak caseを保存。主PCIe転送408→30MB。traced peak222.21→39.10MiB（C++/GPUは対象外）。
全状態getter/host import/PR対照の時間差は帰属しない。通常PF/RT/simulation全般へ一般化しない。

1567一致=1531出力36例外。追加387は288native全bit/cache/reset/所有権、9独立queued履歴、72warm original CDF、18cold/decline fallback。
source getter/target host setterを禁止したwarmケースも通過。source/target counts不一致、尾、重複/逆順/random、IEEE payload、KF/predict/次gatherを確認。
追加15 native guard/alias/coercion/使用完了 tests。baseline1587pass+15newAPI skip/candidate1602pass、回帰5998pass、3CUDA検査各1602pass、error/leak/hazard0。
初回旧版fixtureでtarget予測step不一致を確認し保存。条件を揃えて全照合/測定をやり直した。数値許容・既存skip・期待値を緩めない。
実データ長時間位置誤差は新規評価していない。FP64算術/モデル/粒子数/乱数/選択CDFは変更しない。
根拠: `benchmarks/pf_pool_gather_gpu_20261014.json`, `benchmarks/pf_pool_gather_device_code_20261013.json`,
`benchmarks/pf_pool_gather_completion_check_20261014.json`, `benchmarks/probes/pf_pool_gather_20261013.patch`,
`benchmarks/bench_pf_pool_gather_compare.py`, `benchmarks/pf_pool_gather_cases.py`, `tests/test_pf_pool_gather.py`。
private baseline/adopted/queue fixture trialにsource/PE/helper/raw log/sample/hashを保存する。
次候補: CPUの正規化/逐次CDFは維持し、同じFP64閾値とside=leftのbinary searchだけをGPUで行う。既存sourceCDF/target index scratchを使い、ties/末尾/clampの祖先と全stateのbit一致を検証してから判断する。

## D-192: 大粒子の全16状態importを既存inactive covarianceのchunkで処理 (2026-10-14)

採用。N>=65536の `set_particle_states` だけを変更。16double×floor(9N/16)行ずつH2D→元kernelで処理し、最後に元weight resetを1回/syncする。
100万粒子の追加128MB確保を除去。getter/小粒子import/layout/binding/runtime/全81GPU SASS/resources/算術/モデル/RNG/許容誤差を保つ。
復元単体は28.215→22.835ms、19.07%短縮、5/5round改善。
resident persistent pool100万→50万→100万の全切替は331.380→313.442ms、5.41%短縮、4/5round改善。
125旧新交互pair/25全状態snapshot、28stagebatch条件を保持。通常PF/readout全般への速度向上へ一般化しない。
初回のgetter/小粒子分割trialはprecision1156一致だが遅延を確認して不採用。最終限定probeの切替悪化も保持し、fresh worker最初の切替を交互確認。
PR対照round中央値比−10.63%とper-pair差中央値−0.25%の双方を保存。非独占hostの変動があり、原因を断定しない。
1180一致=1144出力36例外。追加247件は128bit/ESS/reset、116checkpointなしPF履歴、3input形式。
gate境界/端数、IEEE payload、queued predict/gather/KF、common/individual、systematic/legacy/coalesced、geometry scratchを検査。
baseline/candidate1200pass、回帰5596pass、3CUDA検査各1200pass、error/leak/hazard0。長時間実データ位置誤差は新規評価していない。
根拠: `benchmarks/pf_state_chunks_gpu_20261014.json`, `benchmarks/pf_state_chunks_device_code_20261013.json`,
`benchmarks/pf_state_chunks_completion_check_20261014.json`, `benchmarks/probes/pf_state_chunks_20261013.patch`,
`benchmarks/bench_pf_state_chunks_compare.py`, `benchmarks/pf_state_chunks_cases.py`, `tests/test_pf_state_chunks.py`。
private baseline/adopted/export trialへsource/PE/helper/raw sample/log/hashを保存する。
次の検討対象: poolのCPU CDF/indicesは同じに保ち、warm source/target間の全16状態を元gather kernelでGPU上コピーすれば、状態D2H/CPU選択/H2Dをまとめて省ける可能性がある。cold target・旧binding・injected backendのfallbackと全bit照合が必要。

## D-191: PF粒子exportの一時GPU allocationを既存inactive covarianceで置換 (2026-10-14)

採用。対象は `pf_device_get_particles` / public `get_particles()` のpack済み4N FP64 readout。
元のpack kernelと同期/D2Hを維持し、inactive9N covarianceを使って毎回のCUDA allocate/freeを省く。
100万粒子で追加32MB確保が不要。owner buffer・state layout・Python/native binding・転送量・算術は変えない。
全81GPU SASS/resources一致。active covariance/logweight/ESS/cache certificateを変更しない。

粒子取得付きの100万粒子・PF連続5epochで105.959→93.620ms (11.64%短縮、5/5round改善)。
5round×8repeat×7conditions=280pairと35全状態snapshotが一致。
追加150pairの旧新/同一PE対照を経て、同じ論理CPU・stageごとfresh workerの125pairも確認。
同じCPUで粒子取得付き5epochは101.661→90.484ms、10.99%短縮・5/5round改善。
初回/追加のK4対照遅延は保持。同じCPUのK4対照のper-pair差中央値−0.38%、同一PE対照−1.42%であり、CPU条件/host変動の影響を示唆する。
初回round中央値比とper-pair差は同じ統計量ではないので双方を記録し、対照速度差の原因は断定しない。
21stagebatch条件×3round×5repeatも保持。単体取得・1epoch・変更箇所を通らない対照のばらつき/負roundを削らない。
K8/K16混合尤度のgetterにも適用されるが、CPU尤度処理に支配される条件の速度差は限定的。

933旧新照合=897出力36同じ例外。追加80export bit/所有権検査と74scratch履歴は独立PFの全16state/weightと一致。
queued predict/resample、common/individual、systematic/legacy/coalesced、10万粒子individual B60 packed gather、KF、D190geometry tableとの併用を検査。
baseline/candidate953pass、回帰5343pass、CUDA memcheck/racecheck/synccheck各953pass、error/leak/hazard0。
fixture API名修正前の試行はprivateへ保存し、修正後に照合/測定を再実行。数値許容誤差・期待値・テストskipは緩めない。
実データ長時間position-error curveは新規評価していない。粒子exportを呼ぶPF処理の増分であり、PR単体/RT/simulationへの一般化はしない。

根拠: `benchmarks/pf_particle_readout_gpu_20261014.json`, `benchmarks/pf_particle_readout_device_code_20261013.json`,
`benchmarks/pf_particle_readout_completion_check_20261014.json`, `benchmarks/probes/pf_particle_readout_20261013.patch`,
`benchmarks/bench_pf_particle_readout_compare.py`, `benchmarks/pf_particle_readout_cases.py`, `tests/test_pf_particle_readout.py`。
private frozen baseline/adoptedへsource/PE/helper/raw logs/checks/hash manifestを保管する。

## D-190: 混合位置尤度のgeometryを元FP64丸めでGPUへ（2026-10-14）

状態: 採用

2-4候補・大粒子の座標差分/二乗/x-y-z加算/0.5乗算/σ²除算/log_mix減算だけをexplicit FP64 RN kernelへ移す。
NumPy exp/log/成分sumと粒子数/候補/モデル/乱数/許容誤差を保持する。元78GPU body/resourceとnative state layoutが一致。
既存inactive 9N covarianceとscalar bufferを使い、追加GPU allocateなく全粒子getterの一時allocate/freeを省く。
非有限座標・small/unsupported/strict NumPy・旧native/injected backendは元D189 CPU経路。
779旧新records、216独立3D table/27scratch historiesが一致。旧536tests、新799tests、regression5189、CUDA3sanitizers0。
27stage条件/15host peaksと200旧新pairs/40snapshotを保存。負の対照と非専有host制限を保持する。
効果は2-4候補のsynthetic mixture-guided PF。実データ長時間位置誤差は新規評価していない。
根拠: [measurement](../benchmarks/pf_mixture_geometry_gpu_20261014.json)、
[device proof/patch](../benchmarks/pf_mixture_geometry_device_code_20261013.json)、
[completion](../benchmarks/pf_mixture_geometry_completion_check_20261014.json)、[RESULTS](../benchmarks/RESULTS.md)。

## D-189: 混合位置尤度の元FP64演算を2Dの粒子chunkへ（2026-10-14）

状態: 採用

大きなN×K×3差分/二乗とlogsumexpの中間配列を、ordered x/y/zの2D FP64作業配列と粒子chunkへ移す。
小入力とstrict NumPy error/callback/warning-errorは元のeager経路。全候補/粒子・元exp/log・reduction順を維持する。
初期trialの値は一致したがstrict error順が違ったため保存し、guard追加後に全536validation/測定を再実行した。
他Python/sourceと全native PEはbyte一致、78GPU bodies/resourcesと全GPU転送を保持する。
旧新536recordsと元3D oracleがbit一致、旧新536tests、regression4926、CUDA3sanitizers errors/hazards/leaks0。
200旧新pair/40snapshotと24stage条件/12traced host peakを保存。負の対照も保持する。
効果の対象はCPU mixture position likelihoodを使うsynthetic PF。実データ位置誤差曲線は新規評価していない。
根拠: [measurement](../benchmarks/pf_mixture_coordinates_gpu_20261014.json)、
[device proof/patch](../benchmarks/pf_mixture_coordinates_device_code_20261013.json)、
[completion](../benchmarks/pf_mixture_coordinates_completion_check_20261014.json)、[RESULTS](../benchmarks/RESULTS.md)。

## D-188: 重みが不変なら計算済みの厳密ESSを再利用する（2026-10-14）

状態: 採用

公開IMU/RBPFのESS取得→条件付きresampleで、同じ重みを2度GPU reductionしていた。
最初の元GPU ESS値だけを独立host slotへ保持し、再照会は同じstream同期後にそのFP64値を返す。
全16native weight mutation入口で無効化。predict/shifts/covariance、共有query buffer再利用でも同期/値を維持。
全78GPU body/resourceとGPU sourceが一致し、粒子/観測/FP64/reduction/乱数/resamplingを変更しない。
25,227旧新records一致、240新casesはexact weight再uploadで元GPU reductionを強制する独立oracleと一致。
旧版・候補240tests、regression4390、CUDA3sanitizers errors/hazards/leaks0。
100万粒子の8全体conditionsを1call交互800pairsで確認し、負の対照も保持する。
効果は重み間の重複ESS照会に限る。実データ位置誤差曲線/全pipeline割合は新規評価していない。
native stateのhost field追加は対応headerで再build。直接device weight pointerを書き換えるclientはcacheを無効化する。
根拠: [measurement](../benchmarks/pf_ess_cache_gpu_20261014.json)、
[device audit/patch](../benchmarks/pf_ess_cache_device_code_20261013.json)、
[completion](../benchmarks/pf_ess_cache_completion_check_20261014.json)、
[RESULTS](../benchmarks/RESULTS.md)。

## D-187: 共分散9値をGPUへ直接broadcastして状態/重み272MB往復を省く（2026-10-14）

状態: 採用

毎epoch用 `set_velocity_covariance` の全N×16 state/N weights往復を、9FP64 inline係数・flat9N writeの1kernelへ置換。
7state/全weights/finite reshape3x3/最後のstate stream同期を保持、古いnative/injected backendは元fallback。
同じ9値を配りuniform certificateを付け、元D184 predictor/共通covariance resamplerを使える。
共分散値の対称化/対角化/近似や観測/粒子/Philox/祖先変更はしない。元77GPU SASS/resources/struct layout保持、追加1kernel26regs/shared80/stack/local0。
1call交互5×20で100万粒子setter100.7809→0.3496ms、
共分散設定を含む5epoch公開PR全体595.5055→54.6984ms、
KF併用公開step163.6148→41.9690ms。全21+6条件/600pairs/30snapshots/予備12条件を保持。
24,987records一致（新1,024 actual old/new、retained roundtrip/individual predictor oracle、signed zero/NaN payload/非対称/negative/extreme/partial blocks）。
public/injected257tests old/new合格、new native13casesは旧moduleにAPIがなくskip、候補270合格。4150回帰/3sanitizers270すべてclean。
初期zero sigma_pos fixtureの旧API拒否はarchiveし、fixtureだけaccepted tiny-positiveへ修正。
BVH D169/signal D185を保持。共分散を設定しないPFへの同率の利得と長期実データIMUの測位誤差は未評価。
根拠: `benchmarks/pf_covariance_broadcast_gpu_20261014.json`、`benchmarks/pf_covariance_broadcast_device_code_20261013.json`、
`benchmarks/probes/pf_covariance_broadcast_20261013.patch`、`tests/test_pf_covariance_broadcast.py`。

## D-186: Doppler尤度専用prefixは資源削減に留まり全体利得不足で不採用（2026-10-14）

状態: 不採用、元D184 source/PEへ復元

gain<=0/<=64観測だけ元kernelの尤度prefixを切り出し、元77GPU SASS/resourcesを保持して6kernel追加。
regs70→56/76→58、stackからWLS160bytesを除けたが、3×5全14条件で100万粒子12観測公開step約0.8%、
5epoch約0.1%悪化、32観測step約0.3%に留まった。資源差だけで有効な高速化とせず、複雑さを増やさない。
1,925records old/new bit一致、保持した元producerをNaN gainのGPU clampで選ぶoracleも両binaryで確認。
非専有hostの予備測定までで不採用を決め、追加1call交互/候補回帰/sanitizersは未実行。
全正/負のtiming、validation、candidate source/pyd/helpers/device-code/byte patchをarchive。
元sourceを通常rebuildし77GPU bodies/resources一致後に以前のexact PEへ戻し、installed既存781tests合格。
PF D184、BVH D169、signal D185を保持。長期実データの測位誤差分布は未評価。
根拠: `benchmarks/pf_doppler_likelihood_rejected_20261014.json`、
`benchmarks/pf_doppler_likelihood_device_code_20261013.json`、
`benchmarks/probes/pf_doppler_likelihood_20261013.patch`、
`benchmarks/pf_doppler_likelihood_rejected_completion_check_20261014.json`。

## D-185: 256超の静的NAVも元の未cache演算のまま専用producerへ（2026-10-14）

状態: 採用

valid channel maximum>256かつpacked NAV空のsingle/batchだけ、既存static templateの未cache版を選ぶ。
host2dispatchのみ変更、元11GPU functionsのSASS/resourcesを保持。新single33/batch43regs、stack/local/shared0。
サンプル/channel数、FP64 time/chip/phase、ordered FP32 sum、Philox draw/seed/sequence、返却ownership、転送/API/streamを保持。
元dynamic producerへ同じconstant1bitを渡すoracleを含み1,215records bit一致、92新tests old/new、3,864回帰、3sanitizers全clean。
1call交互5×20で257channels/260,000samples×4epoch公開single群54.1514→45.5011ms、
公開batch50.9313→42.4090ms。全14非交互/10交互/8予備条件、負の対照、短い32epochも保存。
不変65channelの初回悪化も保存し、旧版同士200pairと旧新の対照再検査200pairを追加。再検査結果も全件保存。
非交互native batchでの悪化は記録し、広いチャネル数全般/都市trajectory/測位全体への同率の効果は一般化しない。
初期0Hz oracle fixtureの旧版入力拒否はarchiveしpositive1Hzへ修正、productには影響しない。
PF D184/BVH D169を保持。長期実データ測位誤差分布は新規測定していない。
根拠: `benchmarks/signal_wide_static_gpu_20261014.json`、`benchmarks/signal_wide_static_device_code_20261013.json`、
`benchmarks/probes/signal_wide_static_20261013.patch`、`tests/test_signal_wide_static.py`。

## D-184: 共通速度共分散の元FP64 Cholesky計算をblock共有（2026-10-14）

状態: 採用

velocity KF predictかつ既存covariance_uniform保証時のみ、block内最初の粒子から元式で6coeffを作りshared48bytesへ。
全tail threadがbarrier前参加。global particle0を読むraceを避け、元RNG/guideFMA/粒子別Q/位置/clock更新を保持。
全76旧GPU body/resourceとkernel sourceを維持、新1kernel58register/stack/local0。
allocation/transfer/launch/API/stream/粒子・観測数/祖先/query tree/非KF・個別predictは元のまま。
100万粒子・共通RBPF/PR5epoch公開処理は1call交互で57.7528→51.5635ms（約10.7%短縮）。
Doppler KF後は個別になるためsteady individualへの利得を主張しない。小粒子数の負値/対照/paired差分も記録。
23,963一致、追加464実旧新/保持原producer照合、512原版/新版tests、3,772回帰、512testsの3CUDA sanitizers clean。
27条件5×20、8条件800pair＋N2048 4条件400pair。長時間実データの測位誤差分布は未追加評価。
詳細はRESULTS.md冒頭とpf_common_predict_gpu/device_code/completion_check。BVH/signal static single/batchを保持。

## D-183: 静的NAV単一epochの33..256チャネルだけ専用producerを使う（2026-10-14）

状態: 採用

元static templateをsingle globalとして実体化し、packed NAV空・valid33..256の場合だけhost選択1箇所を変更。
全10既存GPU SASS/resourceとkernel sourceを保持、新1kernel31register/stack/local0/shared2048。
inline/dynamic/空/>256/all batchとFP64/chip整数剰余/ordered FP32/RNG/sample数/所有権/確保/転送/APIは元のまま。
65ch×各260,000samplesの8個別生成＋np.stack全体は1group交互で88.3894→66.1128ms（約25.2%短縮）。
1,045比較一致、205実原版/新版tests、3,260回帰、205testsの3CUDA sanitizers clean。
13条件5×20、9条件900pair、全negative/control保存。小channel/短いepoch/都市へ一律一般化せず、長時間実データ測位誤差は未追加評価。
unrestricted1..256案は12chの全体安定利得がなく棄却・archive。棄却版に採用版の追加回帰/sanitizer結果を流用しない。
詳細はRESULTS.md冒頭とsignal_static_single_gpu/device_code/completion_check/unrestricted_rejected報告。

## D-182: 静的NAV batchだけ到達しないdescriptor分岐を省き高速化（2026-10-14）

状態: 採用

packed NAV空かつmax valid channel1..256のbatchに専用2producer。元kernel-copyから到達しないdynamic NAV分岐のみ除く。
FP64時間/code/phase・ordered FP32積/加算・RNG/seed・sample/channel数・API/owning output・確保/転送を保つ。
dynamic/single/空all/257以上は元経路。旧8GPU body/resource保持、新2は40register/stack/local0/shared256or2048。
65chの8epoch×260,000 samples公開全体は1call交互で53.2742→40.0439ms（約24.8%短縮）。
1,200tri都市全体は27.8604→23.1097ms（約17.1%短縮）。短いepoch/未変更経路へ一般化しない。
843旧新一致、104原版/新版tests、3,055 regression、104casesの3 CUDA sanitizers clean。
12条件5×20、6条件600pair、全negative/controlも保存。長時間実データの測位誤差分布は未新規評価。
事前D2H activityが約半分でも、pinned stagingとtransient registrationは全体で遅くなり棄却。各737比較/8条件3×5/source/pyd/patchをarchive。
棄却2案へ採用案のregression/sanitizer結果は流用せず、元転送へ正確に戻した。
詳細はRESULTS.md冒頭、signal_static_nav_gpu/device_code/completion_checkとsignal_pinned_*_rejected報告。

## D-181: BVH構築3案を全体利得不足で不採用（2026-10-14）

状態: 不採用

3axisのbin population走査統合、誤差boundで同一整数bucketを保証したreciprocal分類、左子prefix bounds再利用を実ビルドで比較。
各97caseのtree/index/trace・全timed出力はbit一致。各GPU10body/resource全同一、新kernel0。
前2案は12k/120k公開build/rebuildに安定した利得なし。左子boundsは同source/pyd5×10確認で12k約1%差、120k公開buildは悪化。
非占有hostの変動と区別するためown-thread cycle diagnosticsも記録したが、大きな費用削減は確認できない。
境界のguardは誤差許容ではなく同じ整数分類の保証で、境界/非finiteは元除算。左boundsは同じordered union、右/rootはsigned-zero tieを保つため元scan。
各21条件3×5、左bounds追加6条件5×10、raw/source/pyd/helper/patch/証明をarchive。試案に新規full regression/sanitizersは実施しない。
BVHをparallel mirror chunksのsource/build/installed exact SHAへ復帰。PF individual gather/signal epoch batchも保持。
詳細はbenchmarks/RESULTS.md冒頭、bvh_build_trials_rejected_20261014.jsonとcompletion check。

## D-180: individual covariance PFの元sourceをpacked gather領域へ再利用（2026-10-13）

状態: 採用

個別共分散の全9*Nを元と同じmapping/lane順でdestinationへcopyした後、同じstreamでformer source9*Nを8*N packed状態行へ再利用。
N≥100,000かつB≥60のみを新経路とし、小規模/短反復は元のgather/reset。common positive経路を保持。
全75 GPU body/resource・FP64/RNG/受理/祖先/query/API・pointer swapを保つ。新規確保なし、個別経路のみ3→4起動。
100万粒子の公開PR+KF全stepを1call交互で42.6055→39.7989ms（約6.6%短縮）、独立round中央値は4/5で改善。
対応pair差の中央値は各5roundで正、77/100pairでcandidateが速い。悪化したroundも残し、stage-batch5/5と区別する。
27条件5×20、追加500交互pair、無制限試行の小規模/短反復の悪化も保存。10万の全step効果は小さい。
23,499旧新一致、31新規独立tests、398原版/新版tests、2,951 regression PASS、398casesの3 CUDA sanitizers clean。
コピー後のsource再利用・閾値/端数/最大seed/NaNpayload/signedzero・queued/resampler/import切替をbit検証。
長時間実データの測位誤差分布は新規評価していない。BVH/signalの採用版も保持。
詳細はbenchmarks/RESULTS.md冒頭、pf_individual_gather_gpu/device_code/completion_check_20261013.json。

## D-179: common covariance PFの状態gatherを既存buffer内のpacked行で高速化（2026-10-13）

状態: 採用

事前GPU activityでは100万粒子/common B60のgatherが3.5515ms、kernel時間の約64%だった。
未使用destination covariance9*N内の8*Nに7状態を64byte行へcopyしてdouble2でgatherする。primary SoA/active covarianceは保持、追加allocationなし。
copy-only pack/gather2kernelを追加し、祖先選択完了後の同じzero resetをgatherへ統合。元と同じ3launch。
共分散uniform保証あり/positive iterationsのみ。import/KFやB0/negativeは元のgather/resetを保持。
元73SASS/resource一致、23,499旧新比較一致（23,067出力等+432入力棄却）。361新規/367targetは原版/新版で通過。
2,920regression、367memcheck/racecheck/synccheckすべてclean。8*N≤9*Nのbounds、tail barrier、padding、所有/再初期化/無効化を確認。
30条件・5×20と、100万粒子4条件の1call交互5×20を検証。公開PR step12.5106→9.8135ms（約21.6%短縮、5/5）。
native PR14.6844→13.0132ms（5/5）、共通B60 resample6.1303→4.6006ms（4/5）。
小粒子数には遅い条件があり、一律の改善を主張しない。個別共分散のPR+KFは改善対象外。全control/全roundを保存。
PFの数学/FP64/粒子数/観測数/RNG/祖先/共分散/ESS条件/APIは同じ。BVH/signal採用版も保持。長時間実測位誤差は新規評価していない。
記録: benchmarks/pf_packed_gather_gpu_20261013.json、RESULTS.md冒頭。

## D-178: block Philox proposal共有を全体利益不足で不採用（2026-10-13）

状態: 不採用

候補列を同じblock subsequence/absolute offsetから協力生成し1KB sharedに保存。受理順/own RNG/FP64差分/既存FP32式を保持。
元73GPUbody/resource保持、追加kernel38register。23,499一致、181独立testsは原版/試行で通過。
30条件・5×20で100万粒子公開PR12.4705→12.4706ms、公開PR+KF41.9774→41.9595ms、利益なし。
GPU activity selection1.9795→1.9784msも差が小さい。全raw/source/pyd/helper/patch/testをarchiveに固定して不採用。
D-179は元のproposal kernelを使い、実測律速のgatherを変更した。不採用案の追加regression/sanitizerなし。
記録: benchmarks/pf_shared_proposals_rejected_20261013.json。

## D-177: undifferenced AFVの逆数共有/2件展開を全体効果不足で不採用（2026-10-13）

状態: 不採用

粒子ごとの波長・分散逆数をblock/thread0で同じGPU FP64式により共有し、次にordered loopの2件展開を追加。
距離・残差・AFV/rint・観測順・prior加算・API/input/RNGを保持。ソース再構成でscalar位置/pragma以外同一。
各案73中72のSASS/resource同一、AFV42register/0stack、static shared16byte追加のみ。
各案19,531比較一致（19,099出力等+432入力棄却）、独立DD/half-cycle/IEEE/tail/queued入力所有57testsは原版と両試行で通過。
24条件×2案を5交互round×20で測定。共有案の公開MUPFは22.3039→23.7366ms、3/5改善。
共有+2件展開案の公開MUPFは21.9035→21.7602ms、3/5改善。control変動が大きいため100万粒子3条件を1call交互5×20で確認。
単独AFVは6.2938→6.1382ms（4/5）、公開MUPFは21.6386→21.5297ms（5/5）。
公開全体の利益は約0.5%で未変更PRも約0.8%短縮し、他条件の利益は不安定。効果的な全体改善を立証できず両案不採用。
全時間/bit比較/source/binary/helper・2patchを保存し、57draft testsはarchiveのみ。追加のfull regression/sanitizerなし。
PF source/build/installed exact復元、D-175とBVH/signal採用版を維持。実データの長時間測位誤差は新規評価していない。
記録: benchmarks/pf_afv_constants_unroll_rejected_20261013.json、RESULTS.md冒頭。

## D-176: signal bindingの空array/診断名生成削減を全経路の安定利益不足で不採用（2026-10-13）

状態: 不採用

default py::array_t nav参照slotのチャネル別空NumPy配列をnull objectへ置換し、converted arrayを同期generation終了までmove保持。
正常値の診断名は元predicateが失敗する時だけ生成。検査内容・cast回数/順序・例外型/文言・FP32/FP64演算を保つ。全8 SASS/resource同一。
slot-only633比較/20timings・3×10を固定し、組合せ案737一致（657出力等+80入力例外）、旧新42draft testsも通過。
20条件・5 stage-batch round×20で公開経路の改善pairが3〜4/5に留まるため、10条件を1callずつ交互の5×20で追加確認。
nativeと一部公開batchは改善したが、12チャネル固定navの公開32epoch batchは6.6880→7.3727ms、2/5のみ改善。
urbanは30.5099→29.0162msでも3/5のみ改善。通常経路の悪化と全体の不安定さを優先して不採用。
全条件・control・source/binary/helper・2patchを保存。draft42testsはarchiveのみ、追加のfull regression/sanitizerは行わない。
signal source/build/installed exact復元、PF D-175/BVH採用版を維持。長時間実測位誤差は新規評価していない。
記録: benchmarks/signal_input_packing_rejected_20261013.json、RESULTS.md冒頭。

## D-175: 33〜64衛星の速度KF geometry再利用をexact FP64で拡張（2026-10-13）

状態: 採用

33〜48 cache48、49〜64 cache64、block128。原cached bodyと関数名/64bit integer maskだけが異なる。
FP64 geometry/平均/逐次KF/P更新/symmetrization/clamp/尤度/RNG/観測順・粒子/観測数・APIを保持。1〜32と65以上の既存dispatchを維持。
公開100万粒子・64衛星は196.5619→161.9534ms（17.6%短縮）。対象16条件は全5交互round×20で改善。39条件・対照も保存。
18,941旧新一致（18,509出力+432入力棄却）、58独立retained GPU oracleを旧新版双方で確認、2,559回帰、58testsの3sanitizersはerrors/hazards/warnings/leaks0。
既存71 SASS/resource同一、新2。register94→86、stack0→1536/2048 bytesをcomplete callの測定に含める。
owner容量/alloc/H2D/shared/launch/streamを保持。BVH早期棄却の不採用trialはexact復元、BVH/Signal採用版を保持。
小規模/既定PRのみ/初回/長時間全実データへ速度比を一般化せず、測位誤差分布を新規評価したとはしない。
記録: benchmarks/pf_kf_wide_cache_*_20261013.json、RESULTS.md冒頭。

## D-174: 反射分子の保守的early rejectをwhole-call悪化で不採用（2026-10-13）

状態: 不採用

512以上のordered parallel反射で、元FP64分子のfinite wide missだけを逆数前に棄却。
逆符号|numerator|≥1のsigned-zero underflowと、同符号|numerator|/2>|a|の丸めを考慮。境界/非有限値は元の商/述語へ戻る。
ray正規化/FP64式/三角形順/tie/閾値/候補数/owner payloadを保持。既存10 SASS/resource同一、新2kernelの資源も元と同じ。
466bit比較、64追加sceneのretained stateless GPU oracleも一致。48条件を3交互round×5で測定。
公開8192三角形・129x32は36.3267→39.5976msで全3pair悪化。不採用trialの回帰/sanitizerは追加せず、直前BVH source/build/installedをexact復元。
記録: benchmarks/bvh_early_intersection_rejected_20261013.json、RESULTS.md冒頭。

## D-173: 共分散component planesとsignal normal2を実効速度の不足で不採用（2026-10-13）

状態: 不採用

共分散を9成分planeへ変更し、粒子間の予測/KF load/storeを連続化。公開Nx16形式・容量・全FP64式/RNG/観測順を保持。
共分散indexとcopy-only gather変更だけでsource全体をexact再構成。71 GPU関数中62 SASS body同一、9関数が変更。
18,591旧新比較一致、189入力/output NPZ bit差0。28 timed条件を3交互round×5で測定。
公開100万粒子PR＋KFは41.8795→46.6076ms、3pair全て悪化。予測/systematic単独の改善で全体悪化を正当化しない。
非占有hostで小差を一般化せず、complete public stepに採用価値がないと判断。候補をarchiveし、直前PF source/build/installed exact復元。
signal normal2置換は全8 SASS body/resource同一で不採用。device変更がないのでtiming/bit出力比較を追加せずexact復元。
不採用trialには新規回帰/sanitizerを実行しない。採用済みD-172/BVH/Signal batchを保持し、精度toleranceは変更しない。
長時間実データの測位誤差分布は新規評価していない。
記録: benchmarks/pf_covariance_planes_rejected_20261013.json、signal_noise_pair_rejected_20261013.json、RESULTS.md冒頭。

## D-172: 64bit validityで33〜64衛星の元FP64 geometryを再利用（2026-10-13）

状態: 採用

33〜48衛星cache48、49〜64衛星cache64、block128。整数maskをunsigned64へ広げ、bit63まで定義されたshiftで扱う。
専用bodyは元cached bodyと関数名/整数mask幅以外完全同一、全FP64式・加算順・range/NaN/solver/gain/clippingを保持。
1〜32の既存cacheと65以上の元uncachedを維持。公開100万粒子・64衛星・gain0.25は153.6282→91.6412ms（40.3%短縮）。
対象32timed条件は全5交互round×20で改善。63条件の全raw・対照を固定し、初回/小規模/全実データへ一般化しない。
15,189旧新一致（14,757出力+432入力棄却）、追加75+既存160の独立retained GPU oracle、関連2501tests、235testsの3sanitizersでerrors/hazards/warnings/leaks0。
oracleの65paddingと既存2fileのexact修正patch、324input/output診断を保存。48/49/64/65・最高mask bit・predict/queue/所有/容量も確認。
既存69 SASS/resource同一、新2。register90→76、stack160→1312/1696 bytesのtradeoffを測定に含める。
owner payload/H2D/shared/launch回数/API/既定値/乱数/観測/粒子/stream ownershipを保持、BVH/Signal source/build/installedも保持。
長時間実データの測位誤差分布は新規評価していない。
記録: benchmarks/pf_doppler_wide_cache_*_20261013.json、RESULTS.md冒頭。

## D-171: 17〜32衛星でも元のDoppler geometryを一度だけ計算（2026-10-13）

状態: 採用

17〜24衛星cache24、25〜32衛星cache32を既存template本体のまま追加し、時計drift/尤度/WLSで方向3成分を再利用する。
block128、32bit maskのbit31まで有効。33以上は元のuncached/block256、1〜16は直前cache8/cache16。
公開100万粒子・32衛星・gain0.25は83.5292→52.1874ms（37.5%短縮）。
10万〜100万粒子の対象32timed条件は全5交互round×20で改善。63timed条件のrawと対照を保持し、小規模/初回/全実データへ一般化しない。
14,202旧新一致（13,770出力+432入力棄却）、追加75+既存85の独立retained GPU oracle、関連2426tests、160testsの3sanitizersでerrors/hazards/warnings/leaks0。
oracle paddingを17→33へ更新したexact patchを保存。24/25/32/33・最高mask bit・predict/queue・ECEF・scalar異常・所有/容量も確認。
既存67 SASS/resource全同一、新2。register90→70、stack160→736/928 bytesのtradeoffを測定に含める。
FP64/全観測/全粒子/乱数/加算順/solver/gain/clipping/API/既定値/owner payload/H2D/shared/launch回数/stream ownershipを保持。
時計drift+WLS loop統合は13,215bit一致でも主なpublic2〜3%と対照変動/94registerを理由に不採用、216input/outputを含むexact試案を固定。
BVH parallel mirror chunksとsignal batchのsource/build/installedを保持。長時間測位誤差分布は新規評価していない。
記録: benchmarks/pf_doppler_large_cache_*_20261013.json、pf_doppler_wls_fusion_rejected_20261013.json、RESULTS.md冒頭。

## D-170: Doppler尤度とWLSで粒子別FP64方向を再利用（2026-10-13）

状態: 採用

通常Dopplerの時計drift/尤度/WLSで重複するsqrtと方向の除算を最初のloopにまとめる。
1〜8衛星cache8、9〜16衛星cache16の3成分+valid maskを再利用、block128。17衛星以上は元のkernel/block256。
元のrange<1/NaN semantics、ordered sums、4x4 solver、gain/clipping、PRや他の観測、既定設定を保持。
100万粒子の公開predict/PR/Doppler/ESS/必要時B60/estimate（gain0.25）:
8衛星: 32.0242→24.2402ms（24.3%短縮）。
12衛星: 40.7091→28.9450ms（28.9%短縮）。
16衛星: 55.2839→39.2947ms（28.9%短縮）。
100,000/1,000,000粒子の対象28timed条件は全5交互round×20回で改善。57条件全rawを固定、小規模/対照の揺れと初回処理へ一般化しない。
13,215旧新一致（12,783出力+432入力棄却）、独立retained GPU oracleの85tests、関連2351tests、85testsの3sanitizersでerrors/hazards/warnings/leaks0。
既存65 SASS/resource完全同一、新2。register90→70、stack160→352/544bytesのtradeoffを実測に含める。
owner payload・H2D・shared・launch回数・ABI/stream ownership維持、BVH parallel mirror chunksとsignal batchのsource/build/installedを保持。
予備8衛星publicの逆転を受け本計測を増やし確認。予備helperを終了後に固定してからlintのみ整理、両SHAを保存。
長時間実データの測位誤差分布は新しく測定していない。
記録: benchmarks/pf_doppler_weight_cache_*_20261013.json、RESULTS.md冒頭。

## D-169: 大batchの並列候補評価で32MiB内のmirrorをepoch分割して再利用（2026-10-13）

状態: 採用

parallel residentの8,192 visits以上・16衛星以上で、全epochのmirrorが32MiBを超えても共有を続ける。
floor(32MiB/24/visits) epochずつ既存mirror/candidate kernelを起動し、partialをray_offset*tilesへ書く。
同じdefault streamで再利用順序を保ち、winner finalizeは全rayに一度。1epoch超過は元のfallback。
公開12,000三角形は4,128 raysで60.6382→55.0109ms（9.3%短縮）、8,224 raysで122.3819→111.4445ms（8.9%短縮）。
8,192三角形・8,224 raysは81.2450→73.2454ms（9.8%短縮）。初回trace/closeも6.6〜7.6%短縮。
対象9timed条件は全5交互round×20回で改善。変更対象外の小batch初回+4.9%の変動もRESULTSに記録し、全処理の利得へ一般化しない。
402旧新bit条件、独立stateless oracle追加62tests、関連2266tests、62testsの3sanitizersでerrors/hazards/warnings/leaks0。
10既存GPU関数のSASS/resource完全同一、新0。FP64・全候補・訪問順/tie・sp<62・nonfiniteを保持。
mirror最大32MiB、従来winner最大640KiB、lazy/reuse/close。複数tile・1epoch容量超過・ECEF・serial切替も検証。
64thread block/compact64winner/2ray blockの3案は353bit条件を保持したが実効利得不足で棄却、exact source/pyd/helper/raw/patchを保存。
PF packedとsignal batchのsource/build/installedを保持。長時間実データの測位誤差分布は新規計測していない。
記録: benchmarks/bvh_parallel_mirror_chunks_*_20261013.json、bvh_candidate_layouts_rejected_20261013.json、RESULTS.md冒頭。

## D-168: 頂点配置の2案を棄却し、候補評価の時間支配を計測（2026-10-13）

状態: 両案不採用、元floor64へ正確に復帰。実timelineの根拠を保存。

座標plane cacheは純FP64 vertex copyと同じ式でも、15診断中8条件でpoints/delayのbit差。
toleranceを緩めず性能計測前に棄却。元10 SASS/resourceは保持だが新consumerを同値とは扱わない。
元Triangle recordを訪問順にコピーする版は308 bit条件・全timed出力一致、追加72bytes/visit/<=8MiB。
36 timed条件3交互round×5では12k tri/4128 raysの公開trace 59.4905→59.4978msとほぼ不変、
16 raysの初回は2.9011→3.6725msへ悪化し、追加cacheを正当化できず不採用。
両source/pyd/frozen helper/raw/patchをarchive、追加regression/sanitizersは実行しない。PF packed/BVH floor64/signal batchをSHAで保持。
nvprofで復帰版12k tri/4128 raysを実計測。warm候補72.626961ms、LOS0.965870ms、finalize0.002144ms、
コピーGPU合計0.023102ms。候補はkernel時間98.68%で、次は候補本体を優先。
profiler下の単一timelineで通常wall timeへ一般化しない。Nsight Computeは一覧取得で異常終了しcounter未取得、FP64/memory律速は未断定。
nvprofの不足CUPTI DLLは既存folderを計測子process PATHへ追加して解決、永続変更なし。
記録: bvh_vertex_planes_precision_rejected_20261013.json、bvh_ordered_triangles_rejected_20261013.json、bvh_floor64_kernel_profile_20261013.json。

## D-167: host同期の2試案はwhole-call利得不足で不採用（2026-10-13）

状態: 不採用、直前版へ正確に復帰

PF pinned source再利用をH2D completion eventへ限定する案は、stream上のdevice順序とgrowth/destroyのfull同期を維持。
11,895比較（入力棄却432含む）、Doppler全1,566条件と33本の10回queued/mutated入力列、旧65 SASS/resource同一。
16条件3交互round×5ではpublic PR/MUPFやqueueの利得が安定せず、100k粒子4回DD jointは全3組で悪化。
BVH blocking D2H直前の4 device wait除去も、292 bit条件・全timed出力・旧10 SASS/resource同一だが、
32 timed条件3交互round×8で複数LOS/反射が悪化。同期回数の削減だけを高速化として採用しない。
両source/pyd/frozen helper/raw/patchをarchive。PF helperの重複import整理は計測後に凍結版を保存して行う。
不採用の試案に追加regression/sanitizersは実行せず、PF packed/BVH floor64 source/build/installedの正確なSHA復帰を確認。
signal batchも保持。記録: pf_copy_event_rejected_20261013.json、bvh_copy_sync_rejected_20261013.json、RESULTS.md冒頭。

## D-166: 64〜127三角形でも元のGPU法線と全訪問候補を再利用（2026-10-13）

状態: 採用

resident referencedのorder capture下限と大型candidateのorder下限を128→64へ変更。
元の10 SASS/resource全同一、FP64/全候補/訪問ordinal/tie/sp<62保持。stateless下限512と不正order fallbackを維持。
公開16 raysは64〜79%、4,128 raysは33〜43%、16,384 raysは25〜32%短縮。
127三角形の公開16 raysは0.6532→0.1386ms。
対象36条件で全5交互roundが改善。初回traceも改善、64三角形・上限batchの初回は3.7%の小幅短縮。
274 bit条件、独立stateless oracle新41tests、関連2204tests、3 sanitizers clean。
order/normal36bytes/visit lazy、winner<=640KiB、既存mirror/tile/batch上限とcapacity reuse/closeを保持。
下限32の無条件案は初回悪化、候補下限64を残す案は4096-ray境界の利得不足、1024-ray制限案は時間変動で不採用。
3試案のsource/pyd/helper/raw/patchと292/304/312 bit検証をarchive。不採用後のregression/sanitizersは実行しない。
PF packedとsignal batchの最新source/build/installedをSHAで保持。
記録: benchmarks/bvh_tiny_order_*_20261013.json、bvh_compact_order_rejected_20261013.json、RESULTS.md冒頭。

## D-165: 小メッシュにも元のGPU法線と全訪問候補の再利用を適用（2026-10-13）

状態: 採用

resident referencedのorder capture下限と大型candidateのorder下限を512→128へ変更。
元の10 SASS/resource全同一、FP64/候補数/訪問ordinal/tie/sp<62保持。stateless下限512と不正order fallbackを維持。
511三角形の公開16rayは2.2595→0.1373ms（約93.9%短縮）、
4,128rayは6.6267→3.0106ms（約54.6%短縮）。
対象3メッシュ×4batch×3stageの全5交互roundが改善、256 bit条件、独立stateless oracle新40tests、関連2163tests、3 sanitizers clean。
order/normal36bytes/visit lazy、winner<=640KiB、既存mirror/tile/batch上限とcapacity reuse/closeを保持。
PF scan temporary再利用は単体短縮に対しwhole/controlの揺れを十分識別できず不採用・archive。
Systematic境界8 state差は旧新版の同一buildでも変動する元CDFの診断として保存、bit一致へ丸めない。
PF packed source/build/installedへ正確に復帰しsignal batchも保持。draft24testsは不採用後に実行していない。
記録: benchmarks/bvh_small_order_*_20261013.json、pf_scan_cache_rejected_20261013.json、RESULTS.md冒頭。

## D-164: 中小メッシュの大型reflection batchにも既存候補並列を適用（2026-10-13）

状態: 採用

ordered visitsの下限を8,192から512へ変更、4,097〜16,384 raysで既存candidate/finalizeを使う。
同じ全候補・FP64式・訪問ordinal/tie・sp<62を維持。元の10 SASS/resource全同一、新device関数0。
512三角形・公開4,128 raysは4.0883→3.0084ms（約26.4%短縮）、
4,096三角形は30.4804→20.7920ms（約31.8%短縮）。
下限4,096/2,048/512を順次検証。対象45条件中44条件の全5交互roundが改善。
512三角形/16,384 raysのcoldだけ12.0086→12.1063ms（0.10ms/0.8%増）。同条件warmは約0.56ms短縮し、反復ownerを主対象に採用。
上限16,384 raysは小メッシュほど利得が小さく、上限を超える既存serialは維持。
220 bit条件、独立stateless oracle新45tests、関連2123tests、3 sanitizers clean。
新対象の追加winner<=640KiB、normal/order reuse、既存mirror>=8192 visits/<=32MiB、LOS-only lazy/close解放を維持。
既存2メモリtestsはwinner容量を含む正確な期待値へ更新。PF packed/signal source/build/installedを保持。
記録: benchmarks/bvh_medium_parallel_*_20261013.json、probes/bvh_medium_parallel_*patch、RESULTS.md冒頭。

## D-163: PF PR/GMM/AFVの観測転送と所有を一括化（2026-10-13）

状態: 採用

PR/GMM/undifferenced AFVの同じ5*nSat doublesを1 device slabへ1 H2Dで転送、3→1 allocation/free。
各callでalias offset3*nSat/4*nSatを更新、growth/closeはbaseのみfree、pinned上書き前のstream同期を保持。
GPU65 SASS/resource全同一、ABI/declaration同一、FP64/粒子数/観測数/RNG/ゲート/AFV prior additionを維持。
2,048粒子12衛星MUPF全体は起動順反転・call交互再確認で0.4501→0.4045ms（約10.1%短縮）、全5round改善。
Gaussian/GMM再確認も全5round改善。本計測4N×12stage・5×20と独立再確認6stage・5×20を保存。
Studentの本計測利得は再確認で再現せず、10万/100万粒子とcoldも大きな利得を主張しない。
10,296比較（9,864 bit output+432 PR棄却）、new25 ownership tests、関連2078 tests、3 sanitizers clean。
要求payload不変、DD/Doppler scratch独立。最新BVH/signalのsource/build/installedを保持。
記録: benchmarks/pf_packed_satellite_*_20261013.json、RESULTS.md冒頭。

## D-162: 大型reflection batchへ上限付き候補並列評価を適用（2026-10-13）

状態: 採用

ordered visits>=8192、4,097〜16,384 raysに既存1 tile/rayのcandidate/finalizeを適用。
元の10 SASS/resource全同一、FP64式・候補数・元訪問ordinal/tie・stack guard・paddingを保持。
12,000三角形・公開4,128 raysは89.0147→59.9158ms（約32.7%短縮）、
8,224 raysは175.6105→120.3161ms（約31.5%短縮）。
2メッシュ×3batch×3stage、全5交互roundで改善。187 bit条件、独立stateless oracle新24tests、関連2053tests、3 sanitizers clean。
winner scratch<=640KiB、既存mirror<=32MiBをlazyに確保してowner reuse/closeで解放。対象外のdispatchは維持。
PF Student除外log省略、PR係数共有、epoch mirror分割、warp mirror共有は全体の効果がなく棄却・archive。
PF/Signalのsource/build/installedは直前のSHA一致を保持。都市/PF全体の速度比へ一般化しない。
記録: benchmarks/bvh_large_parallel_*_20261013.json、今回4種rejected JSON、RESULTS.md冒頭。

## D-161: 大型reflection batchでもGPU法線と元の訪問順を再利用（2026-10-13）

状態: 採用

4,097 rays以上のordered resident反射を1 thread/rayのcached serial kernelへ振り分け、法線再計算とper-ray木走査を省く。
owner構築で元のright-first/sp<62訪問順を捕捉。候補数・FP64式・normal cutoff・strict winner/tie・paddingを維持する。
12,000三角形・公開4,128 raysは148.6805→88.5833ms（約40.4%短縮）、
8,224 raysは298.7955→174.3706ms（約41.6%短縮）。
3メッシュ×3大型batchの公開/native/初回全5組で改善。通常batch/LOSは旧命令列を維持。
162旧新bit条件、独立stateless oracle新31tests、関連2029tests、3 sanitizersでerrors/hazards/leaks0。
旧9 SASS/resource全同一、新85register/stack/local/shared0、旧serial90register/stack256bytes。
36bytes/visit entryをlazy allocation、owner reuse/closeで解放。stateless/小規模/unorderedは旧経路。
PFとsignalは最新source/build/installed保持。都市/PF全体の速度向上へ一般化せず、過去の比率と掛け合わせない。
記録: benchmarks/bvh_serial_cache_*_20261013.json、RESULTS.md冒頭。

## D-160: undifferenced PR gateの二重geometryを省く（2026-10-13）

状態: 採用

Gaussian/Huber/Studentのpositive PR gateを1 geometry loopへ振り分け、元の観測順でall/filtered sumを累積。
min(nSat,4) fallback、<= countとskip-if-greater filter、Student優先、元のFP64/FMA/ordinary log、PR上書きを保つ。
100万粒子の公開全stepはgaussian **14.7978→12.4090ms（約16.1%短縮）**、huber **16.9270→13.8134ms（約18.4%短縮）**、student **19.6077→17.1278ms（約12.6%短縮）**。公開/nativeの3種類は全5交互roundで改善。
10,158旧新比較（9,726出力bits、432入力拒否）、関連1,719 tests、新109 testsの3 CUDA sanitizersでerrors/hazards/leaks0。
ungated GPU subsetを独立oracleに使い、threshold境界・粒子別fallback・容量上限をtoleranceなしで確認。
旧62 SASS/resource全同一、新42/51/44register、stack/local/static shared0。dynamic shared/H2D/launch/API/既定値は同じ。
explicit rounded multiply/subtract案はptxasのFMA融合を妨げ85bit不一致となり却下し、正確なsource/pyd/helper/raw/patchを固定。
採用版はSASSに合わせた元のFMAを保つ。小粒子数・対照経路の揺れを一般化しない。
BVH native/signal batchを維持。計測・device・最終整合はbenchmarks/pf_pr_gate_*_20261013.json、詳細はRESULTS.md冒頭。

## D-159: GMMの共通係数とfinite-logit exp(0)を再利用（2026-10-13）

状態: 採用

GPUの元のsigma reciprocal2/log4をblockで共有し、finite logitsのmax成分exp(0)=1を再利用。
他方expの入力、正数加算、log、元のFP64 geometry/weight/logit/ordered sumを保つ。
非有限logitsは元の2expとoperand順、特殊metadata/容量上限1,228衛星は元のkernelへ戻す。
100万粒子の公開GMM全stepは24.8145→21.0389ms（約15.2%短縮）。
公開/native/equal-component全stepは全5交互roundで改善。小粒子数と変更対象外の揺れは一律の利得としない。
8,631旧新bit比較、関連1,610 tests、新82 testsの3 CUDA sanitizersでerrors/hazards/leaks0。
元のkernel+neutral paddingと同順序のFP64加算を独立oracleに使い、toleranceなしでbit照合。
既存61 SASS/resource全同一、新GMM48register/stack/local0、shared追加48byte・barrier既存1回。
H2D/allocation/launch/API/既定設定、前回DD gates/BVH/signalを維持。
raw/device/patch/completionを `benchmarks/pf_gmm_*_20261013.json`、`benchmarks/RESULTS.md` に記録。
