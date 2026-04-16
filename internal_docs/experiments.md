# 実験ログ

## 目的

このリポジトリでは、設計を先に固定しない。
まず複数の具体実装を同じ入力・同じ指標で比較し、そこから最小抽象を見つける。

現時点の探索対象は、`PF+RobustClear` と `PF3D-BVH+BlockedOnly` のどちらを epoch ごとに採用するかという
`PF observation/gating policy` 問題である。

安定領域:
- `python/gnss_gpu/` 以下の測位・CUDA実装

実験領域:
- `experiments/`
- `experiments/pf_strategy_lab/`

## 問題定義

同一の PPC real-PLATEAU segment に対して、

- `robust expert`: `PF+RobustClear`
- `blocked expert`: `PF3D-BVH+BlockedOnly`

の 2 本を用意し、epoch ごとにどちらの推定を採用するかを strategy として比較する。

重要なのは「どの strategy が理論上きれいか」ではなく、
同じ dump を使ってどの strategy が実際に勝つかである。

## 共通入力

すべての strategy は同じ feature dump と trajectory dump を使う。
dump は pilot 用と full validation 用で分かれているが、schema は同一である。

pilot dump:
- feature dump:
  [ppc_pf_rich_gate_t23_n2_v2_features.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/ppc_pf_rich_gate_t23_n2_v2_features.csv)
- trajectory dump:
  [ppc_pf_rich_gate_t23_n2_v2_trajectories.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/ppc_pf_rich_gate_t23_n2_v2_trajectories.csv)
- base metrics:
  [ppc_pf_rich_gate_t23_n2_v2_bases.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/ppc_pf_rich_gate_t23_n2_v2_bases.csv)

pilot 対象 segment:
- `tokyo/run2 @ 808`
- `tokyo/run3 @ 774`
- `nagoya/run2 @ 983`

full validation dump:
- feature dump:
  [ppc_pf_rich_gate_positive6_v2_features.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/ppc_pf_rich_gate_positive6_v2_features.csv)
- trajectory dump:
  [ppc_pf_rich_gate_positive6_v2_trajectories.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/ppc_pf_rich_gate_positive6_v2_trajectories.csv)
- base metrics:
  [ppc_pf_rich_gate_positive6_v2_bases.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/ppc_pf_rich_gate_positive6_v2_bases.csv)

full validation 対象 segment:
- `tokyo/run1 @ 1463`
- `tokyo/run2 @ 808`
- `tokyo/run3 @ 774`
- `nagoya/run1 @ 0`
- `nagoya/run2 @ 983`
- `nagoya/run3 @ 235`

holdout segment spec:
- [ppc_holdout_segments_r200_s200_best.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/ppc_holdout_segments_r200_s200_best.csv)

holdout 対象 segment:
- `tokyo/run1 @ 1663`
- `tokyo/run2 @ 1008`
- `tokyo/run3 @ 974`
- `nagoya/run1 @ 200`
- `nagoya/run2 @ 1183`
- `nagoya/run3 @ 35`

## 現在の variants

実装は [strategies.py](/workspace/ai_coding_ws/gnss_gpu/experiments/pf_strategy_lab/strategies.py) に集約する。

1. `always_robust`
   style: `constant`
   何も推論せず常に `PF+RobustClear` を使う。

2. `always_blocked`
   style: `constant`
   何も推論せず常に `PF3D-BVH+BlockedOnly` を使う。

3. `disagreement_gate`
   style: `oop-threshold`
   `disagreement_m` が閾値以上なら blocked expert を使う。

4. `clock_veto_gate`
   style: `pipeline-veto`
   `disagreement_m` と `cb_disagreement_m` が十分大きいときだけ blocked expert を候補にし、
   `mean_weighted_blocked_frac` が大きすぎる epoch は veto する。

5. `dual_mode_regime_gate`
   style: `regime-branch`
   blocked を使う regime を 2 つに分ける。
   `close mode`: moderate blocked frac, low disagreement, low clock disagreement, low residual
   `far mode`: very low blocked frac, positive residual mass, high disagreement, high clock disagreement

6. `quality_veto_regime_gate`
   style: `regime-quality-veto`
   `dual_mode_regime_gate` の close mode に quality veto を足す。
   `satellite_count` と `robust_p95_abs_residual` が悪い epoch では close branch を踏まない。

7. `hysteresis_quality_veto_regime_gate`
   style: `stateful-hysteresis`
   `quality_veto_regime_gate` の candidate をそのまま使い、短い off gap をまたいで blocked state を保持する。
   current best config は `enter=1, exit=3`。

8. `mode_aware_hysteresis_quality_veto_regime_gate`
   style: `stateful-branch-hysteresis`
   `close` と `far` で enter persistence を分ける。
   `close` false positive を削りつつ、`far` regime は 1 epoch で blocked に入れる。

9. `branch_aware_hysteresis_quality_veto_regime_gate`
   style: `stateful-branch-exit-hysteresis`
   `mode_aware` に加えて exit persistence も branch ごとに分ける。
   現在の balanced config は `enter_close=2, enter_far=1, exit_close=3, exit_far=5`。

10. `rescue_branch_aware_hysteresis_quality_veto_regime_gate`
   style: `stateful-branch-rescue-hysteresis`
   `branch_aware` に「clean な close singleton だけ即時採用する rescue」を足す。
   current best config は `enter_close=3, enter_far=1, exit_close=3, exit_far=5, rescue_sat<=8, rescue_p95<=50, rescue_cb>=16`。

11. `negative_exit_rescue_branch_aware_hysteresis_quality_veto_regime_gate`
   style: `stateful-branch-rescue-negative-exit`
   `rescue_branch_aware` に、active `close` が candidate を外れた直後の negative-evidence exit を足す。
   current best config は `neg_dis>=42 or neg_cb>=25 or neg_p95>=52` の 1-hit exit。

12. `entry_veto_negative_exit_rescue_branch_aware_hysteresis_quality_veto_regime_gate`
   style: `stateful-branch-entry-veto-rescue-negative-exit`
   `negative_exit_rescue_branch_aware` に、non-rescue の `close` entry だけを `robust_p95_abs_residual <= 50` で絞る entry veto を足す。
   current best config は `close_entry_p95<=50`。

13. `rule_chain_gate`
   style: `pipeline-rule-chain`
   `blocked score`, `positive residual ratio`, `expert disagreement` の 3 条件をすべて満たすときだけ blocked expert を使う。

14. `weighted_score_gate`
   style: `functional-score`
   同じ 3 特徴量を線形スコア化して blocked expert を使う。

## 共通評価指標

accuracy:
- `mean_rms_2d`
- `mean_p95`
- `mean_outlier_rate_pct`
- `pf_rms_wins`

readability proxy:
- `readability_loc`
- `readability_branch_count`
- `readability_proxy`

extensibility proxy:
- `extensibility_param_count`
- `extensibility_proxy`

proxy 指標は [evaluate_strategies.py](/workspace/ai_coding_ws/gnss_gpu/experiments/pf_strategy_lab/evaluate_strategies.py) が自動生成する。
これは人間レビューの代替ではなく、比較の初期ふるいとして使う。

## pilot 結果

strategy summary:
[pf_strategy_lab_t23_n2_summary.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/pf_strategy_lab_t23_n2_summary.csv)

run-wise metrics:
[pf_strategy_lab_t23_n2_runs.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/pf_strategy_lab_t23_n2_runs.csv)

主要結果:

| strategy | style | mean RMS 2D | mean P95 | PF勝ち数 | readability proxy | extensibility proxy |
|---|---|---:|---:|---:|---:|---:|
| `always_robust` | constant | 93.17 m | 108.99 m | 2/3 | 85.0 | 100.0 |
| `always_blocked` | constant | 69.82 m | 91.31 m | 2/3 | 85.0 | 100.0 |
| `disagreement_gate` | oop-threshold | **68.57 m** | 93.42 m | **3/3** | 70.5 | 98.0 |
| `clock_veto_gate` | pipeline-veto | 74.49 m | 99.85 m | **3/3** | 14.5 | 94.0 |
| `dual_mode_regime_gate` | regime-branch | 92.50 m | 108.99 m | 2/3 | 0.0 | 114.0 |
| `quality_veto_regime_gate` | regime-quality-veto | 92.50 m | 108.99 m | 2/3 | 0.0 | 110.0 |
| `hysteresis_quality_veto_regime_gate` | stateful-hysteresis | 92.18 m | 108.99 m | 2/3 | 0.0 | 122.0 |
| `mode_aware_hysteresis_quality_veto_regime_gate` | stateful-branch-hysteresis | 92.07 m | 108.99 m | 2/3 | 0.0 | 128.0 |
| `branch_aware_hysteresis_quality_veto_regime_gate` | stateful-branch-exit-hysteresis | 91.95 m | 108.99 m | 2/3 | 0.0 | 134.0 |
| `rescue_branch_aware_hysteresis_quality_veto_regime_gate` | stateful-branch-rescue-hysteresis | 91.95 m | 108.99 m | 2/3 | 0.0 | 146.0 |
| `negative_exit_rescue_branch_aware_hysteresis_quality_veto_regime_gate` | stateful-branch-rescue-negative-exit | 91.95 m | 108.99 m | 2/3 | 0.0 | 176.0 |
| `entry_veto_negative_exit_rescue_branch_aware_hysteresis_quality_veto_regime_gate` | stateful-branch-entry-veto-rescue-negative-exit | 91.95 m | 108.99 m | 2/3 | 0.0 | 182.0 |
| `rule_chain_gate` | pipeline-rule-chain | 90.88 m | 108.99 m | 2/3 | 11.5 | 94.0 |
| `weighted_score_gate` | functional-score | 90.28 m | 114.01 m | 3/3 | 35.5 | 100.0 |

解釈:
- pilot では `disagreement_gate` が最も強い。
- `rule_chain_gate` と `weighted_score_gate` は特徴量を増やしたが、今の閾値では blocked expert の利得を十分拾えていない。
- ここで重要なのは「pilot winner をそのまま採用しない」こと。pilot は有望候補を絞るために使い、採用判断は full validation に持ち越す。

## full 6 validation 結果

strategy summary:
[pf_strategy_lab_positive6_summary.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/pf_strategy_lab_positive6_summary.csv)

run-wise metrics:
[pf_strategy_lab_positive6_runs.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/pf_strategy_lab_positive6_runs.csv)

主要結果:

| strategy | style | mean RMS 2D | mean P95 | PF勝ち数 | mean blocked epoch frac |
|---|---|---:|---:|---:|---:|
| `always_robust` | constant | 80.02 m | 97.75 m | **5/6** | 0.000 |
| `always_blocked` | constant | 95.16 m | 123.35 m | 2/6 | 1.000 |
| `disagreement_gate` | oop-threshold | 90.91 m | 124.40 m | 3/6 | 0.625 |
| `clock_veto_gate` | pipeline-veto | **73.72 m** | **96.48 m** | **5/6** | 0.293 |
| `dual_mode_regime_gate` | regime-branch | 80.12 m | 97.75 m | 4/6 | 0.043 |
| `quality_veto_regime_gate` | regime-quality-veto | 79.81 m | 97.75 m | **5/6** | 0.023 |
| `hysteresis_quality_veto_regime_gate` | stateful-hysteresis | 79.77 m | 97.75 m | 4/6 | 0.038 |
| `mode_aware_hysteresis_quality_veto_regime_gate` | stateful-branch-hysteresis | 79.63 m | 97.75 m | **5/6** | 0.030 |
| `branch_aware_hysteresis_quality_veto_regime_gate` | stateful-branch-exit-hysteresis | 79.55 m | 97.75 m | **5/6** | 0.030 |
| `rescue_branch_aware_hysteresis_quality_veto_regime_gate` | stateful-branch-rescue-hysteresis | 79.53 m | 97.75 m | **5/6** | 0.028 |
| `negative_exit_rescue_branch_aware_hysteresis_quality_veto_regime_gate` | stateful-branch-rescue-negative-exit | 79.47 m | 97.75 m | **5/6** | 0.025 |
| `entry_veto_negative_exit_rescue_branch_aware_hysteresis_quality_veto_regime_gate` | stateful-branch-entry-veto-rescue-negative-exit | 79.41 m | 97.75 m | **5/6** | 0.020 |
| `rule_chain_gate` | pipeline-rule-chain | 83.14 m | 109.23 m | 3/6 | 0.097 |
| `weighted_score_gate` | functional-score | 90.06 m | 118.88 m | 4/6 | 0.305 |

解釈:
- pilot winner の `disagreement_gate` は full 6 では一般化しなかった。
- `clock_veto_gate` は pilot では勝者ではないが、full 6 の single-split では最良になった。
- これは「pilot で勝つもの」と「validation で残るもの」が違うことを示している。
- `rule_chain_gate` は `disagreement_gate` より conservative で被害は減ったが、それでも `always_robust` を超えない。
- 今の lab が正しく機能している証拠は、「pilot で勝った戦略を full validation で落とせたこと」にある。
- `quality_veto_regime_gate` は `dual_mode` の close branch に quality veto を足しただけで、tuned full-6 では `80.02 -> 79.81 m` に改善する。
- `hysteresis_quality_veto_regime_gate` はその candidate を stateful に保持し、`tokyo/run2` の blocked run を少し伸ばして `79.77 m` まで改善した。
- `mode_aware_hysteresis_quality_veto_regime_gate` は `close` enter を遅くして `tokyo/run1` の false positive を減らしつつ、tuned `79.63 m` まで改善した。
- `branch_aware_hysteresis_quality_veto_regime_gate` は close/far で exit も分け、tuned `79.55 m` と現時点の best train score を出した。
- `rescue_branch_aware_hysteresis_quality_veto_regime_gate` は `branch_aware` に clean close singleton の即時採用を足し、tuned `79.53 m` までさらに改善した。
- `negative_exit_rescue_branch_aware_hysteresis_quality_veto_regime_gate` は rescue の後に `p95>=52` 系の negative-evidence exit を足し、tuned `79.47 m` までさらに改善した。
- `entry_veto_negative_exit_rescue_branch_aware_hysteresis_quality_veto_regime_gate` は non-rescue の close entry だけを `p95<=50` に絞り、`tokyo/run1_seg1463` の 3-epoch false activation を消して tuned `79.41 m` までさらに改善した。

失敗の内訳:
- `disagreement_gate` は `tokyo/run1`, `nagoya/run1`, `nagoya/run3` で false positive が多い。
- `disagreement_m` 単独では blocked-rich Tokyo と clean Nagoya を分け切れない。
- `nagoya/run1` と `nagoya/run3` でも disagreement が高く、blocked expert への誤切替が起きる。
- `clock_veto_gate` は `nagoya/run1` に still false positive を残すが、`tokyo/run2/run3` の利得を large enough に拾えて全体平均では勝つ。
- `dual_mode_regime_gate` は holdout を通す first seed として有効だったが、`tokyo/run1` false positive を抱えたままだった。
- `quality_veto_regime_gate` はその seed を quality veto で削った後継 family とみなす。
- `hysteresis_quality_veto_regime_gate` は `quality_veto` の true-positive regime を gap 越しに繋ぐ stateful variant とみなす。
- `mode_aware_hysteresis_quality_veto_regime_gate` は `close` false positive を減らす intermediate family とみなす。
- `branch_aware_hysteresis_quality_veto_regime_gate` は `close` と `far` の persistence 非対称性を explicit にした balanced family とみなす。
- `rescue_branch_aware_hysteresis_quality_veto_regime_gate` は `branch_aware` の train gain を保ったまま、holdout で必要だった close singleton を rescue する後継 family とみなす。
- `negative_exit_rescue_branch_aware_hysteresis_quality_veto_regime_gate` は rescue の holdout gain を保ったまま、false close persistence だけを切る後継 family とみなす。
- `entry_veto_negative_exit_rescue_branch_aware_hysteresis_quality_veto_regime_gate` は negative-exit の holdout gain を保ったまま、non-rescue close entry の false activation だけを切る後継 family とみなす。

## holdout 結果

holdout candidate search:
- [ppc_holdout_candidates_tokyo_run1_r200_s200_best.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/ppc_holdout_candidates_tokyo_run1_r200_s200_best.csv)
- [ppc_holdout_candidates_tokyo_run2_r200_s200_best.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/ppc_holdout_candidates_tokyo_run2_r200_s200_best.csv)
- [ppc_holdout_candidates_tokyo_run3_r200_s200_best.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/ppc_holdout_candidates_tokyo_run3_r200_s200_best.csv)
- [ppc_holdout_candidates_nagoya_run1_r200_s200_best.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/ppc_holdout_candidates_nagoya_run1_r200_s200_best.csv)
- [ppc_holdout_candidates_nagoya_run2_r200_s200_best.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/ppc_holdout_candidates_nagoya_run2_r200_s200_best.csv)
- [ppc_holdout_candidates_nagoya_run3_r200_s200_best.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/ppc_holdout_candidates_nagoya_run3_r200_s200_best.csv)

holdout dump:
- [ppc_pf_rich_gate_holdout6_r200_s200_features.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/ppc_pf_rich_gate_holdout6_r200_s200_features.csv)
- [ppc_pf_rich_gate_holdout6_r200_s200_trajectories.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/ppc_pf_rich_gate_holdout6_r200_s200_trajectories.csv)
- [ppc_pf_rich_gate_holdout6_r200_s200_bases.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/ppc_pf_rich_gate_holdout6_r200_s200_bases.csv)

strategy summary:
- [pf_strategy_lab_holdout6_r200_s200_summary.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/pf_strategy_lab_holdout6_r200_s200_summary.csv)

run-wise metrics:
- [pf_strategy_lab_holdout6_r200_s200_runs.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/pf_strategy_lab_holdout6_r200_s200_runs.csv)

family cross-validation:
- [pf_strategy_family_cv_positive6_holdout6_configs.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/pf_strategy_family_cv_positive6_holdout6_configs.csv)
- [pf_strategy_family_cv_positive6_holdout6_family_best.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/pf_strategy_family_cv_positive6_holdout6_family_best.csv)

主要結果:

| strategy | style | mean RMS 2D | mean P95 | PF勝ち数 | mean blocked epoch frac |
|---|---|---:|---:|---:|---:|
| `always_robust` | constant | **66.92 m** | **81.69 m** | 3/6 | 0.000 |
| `always_blocked` | constant | 86.98 m | 111.68 m | 2/6 | 1.000 |
| `disagreement_gate` | oop-threshold | 83.84 m | 104.89 m | 1/6 | 0.318 |
| `clock_veto_gate` | pipeline-veto | 74.02 m | 96.29 m | 2/6 | 0.090 |
| `dual_mode_regime_gate` | regime-branch | **65.62 m** | **81.26 m** | **4/6** | 0.183 |
| `quality_veto_regime_gate` | regime-quality-veto | 65.62 m | **81.26 m** | **4/6** | 0.183 |
| `hysteresis_quality_veto_regime_gate` | stateful-hysteresis | **65.57 m** | **81.22 m** | **4/6** | 0.207 |
| `mode_aware_hysteresis_quality_veto_regime_gate` | stateful-branch-hysteresis | 65.57 m | **81.22 m** | **4/6** | 0.200 |
| `branch_aware_hysteresis_quality_veto_regime_gate` | stateful-branch-exit-hysteresis | 65.58 m | **81.22 m** | **4/6** | 0.197 |
| `rescue_branch_aware_hysteresis_quality_veto_regime_gate` | stateful-branch-rescue-hysteresis | **65.57 m** | **81.22 m** | **4/6** | 0.207 |
| `negative_exit_rescue_branch_aware_hysteresis_quality_veto_regime_gate` | stateful-branch-rescue-negative-exit | **65.54 m** | **81.22 m** | **4/6** | 0.203 |
| `entry_veto_negative_exit_rescue_branch_aware_hysteresis_quality_veto_regime_gate` | stateful-branch-entry-veto-rescue-negative-exit | **65.54 m** | **81.22 m** | **4/6** | 0.203 |
| `rule_chain_gate` | pipeline-rule-chain | 67.41 m | 83.95 m | 3/6 | 0.013 |
| `weighted_score_gate` | functional-score | 69.67 m | 91.76 m | 4/6 | 0.612 |

baseline も含めた holdout mean:
- `PF = 67.83 m / 81.03 m`
- `PF+RobustClear = 66.92 m / 81.69 m`
- `PF3D-BVH+BlockedOnly = 86.98 m / 111.68 m`
- `EKF = 71.90 m / 89.65 m`
- `WLS = 93.24 m / 118.41 m`

解釈:
- `clock_veto_gate` は tuned full 6 では良かったが、holdout 6 では `always_robust` を超えなかった。
- `disagreement_gate` は holdout でも再び崩れた。
- `rule_chain_gate` は holdout では surprisingly close まで戻るが、それでも baseline を超えない。
- `dual_mode_regime_gate` は current lab で初めて non-trivial に holdout を超えた family で、`66.92 -> 65.62 m`, `81.69 -> 81.26 m` を達成した。
- `quality_veto_regime_gate` は `dual_mode` より train を改善し、holdout も維持した state-free seed である。
- `hysteresis_quality_veto_regime_gate` は short off-gap を橋渡しすることで、holdout `66.92 -> 65.57 m`, `81.69 -> 81.22 m`、tuned `80.02 -> 79.77 m` を達成した。
- `mode_aware_hysteresis_quality_veto_regime_gate` は holdout をほぼ維持しつつ tuned を `79.63 m` まで押し下げたが、holdout-first selection では `hysteresis` を超えなかった。
- `branch_aware_hysteresis_quality_veto_regime_gate` は holdout `65.58 m / 81.22 m` を維持しつつ tuned `79.55 m / 97.75 m` を出し、rescue 導入前の best train/holdout balance だった。
- `rescue_branch_aware_hysteresis_quality_veto_regime_gate` は holdout `65.57 m / 81.22 m` を `hysteresis` と同値で維持しつつ、tuned を `79.53 m / 97.75 m` までさらに下げた。
- `negative_exit_rescue_branch_aware_hysteresis_quality_veto_regime_gate` は holdout `65.54 m / 81.22 m`, tuned `79.47 m / 97.75 m` まで進め、`tokyo_run2_seg1008` の false close persistence を削った。
- `entry_veto_negative_exit_rescue_branch_aware_hysteresis_quality_veto_regime_gate` は holdout `65.54 m / 81.22 m` を維持したまま、tuned を `79.41 m / 97.75 m` までさらに下げ、`tokyo_run1_seg1463` の false close entry を消した。
- 現時点の global safe baseline は引き続き `PF+RobustClear`。current best generalizing experimental family は `entry_veto_negative_exit_rescue_branch_aware_hysteresis_quality_veto_regime_gate` に更新する。

cross-validation の追加結論:
- `dual_mode_regime_gate` は first real survivor だった。
- その後の family grid search では `quality_veto_regime_gate` が `dual_mode` と同じ holdout を維持しつつ train を改善した。
- さらに `hysteresis_quality_veto_regime_gate` がその上を取り、family best config は `close_blocked_low=0.1`, `close_blocked_high=0.5`, `close_disagreement_max=40`, `close_cb_max=20`, `close_residual_max=22`, `close_satellite_max=9`, `close_p95_abs_residual_max=55`, `far_blocked_max=0.01`, `far_positive_min=0.15`, `far_disagreement_min=90`, `far_cb_min=45`, `enter_confirm_epochs=1`, `exit_confirm_epochs=3`。
- この config は holdout `65.57 m / 81.22 m`、tuned `79.77 m / 97.75 m`。
- `mode_aware_hysteresis_quality_veto_regime_gate` は `enter_close=2`, `enter_far=1`, `exit=4` で holdout をほぼ維持したまま tuned `79.63 m` を達成した。
- `branch_aware_hysteresis_quality_veto_regime_gate` は `enter_close=2`, `enter_far=1`, `exit_close=3`, `exit_far=5` で holdout `65.58 m / 81.22 m`, tuned `79.55 m / 97.75 m` を達成した。
- `rescue_branch_aware_hysteresis_quality_veto_regime_gate` は `enter_close=3`, `enter_far=1`, `exit_close=3`, `exit_far=5`, `rescue_sat<=8`, `rescue_p95<=50`, `rescue_cb>=16` で holdout `65.57 m / 81.22 m`, tuned `79.53 m / 97.75 m` を達成した。
- `negative_exit_rescue_branch_aware_hysteresis_quality_veto_regime_gate` は `rescue` の上に `neg_dis>=42 or neg_cb>=25 or neg_p95>=52` の 1-hit exit を足し、holdout `65.54 m / 81.22 m`, tuned `79.47 m / 97.75 m` を達成した。
- `entry_veto_negative_exit_rescue_branch_aware_hysteresis_quality_veto_regime_gate` は `negative_exit_rescue` の上に `close_entry_p95<=50` を足し、holdout `65.54 m / 81.22 m`, tuned `79.41 m / 97.75 m` を達成した。
- `clock_veto_gate` family の best は holdout を超えず、current best generalizing family は `entry_veto_negative_exit_rescue_branch_aware_hysteresis_quality_veto_regime_gate` に更新する。

freeze sweep:
- [pf_strategy_entry_veto_freeze_configs.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/pf_strategy_entry_veto_freeze_configs.csv)
- [pf_strategy_entry_veto_freeze_best.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/pf_strategy_entry_veto_freeze_best.csv)
- `entry_veto` 近傍を最後に narrow sweep した結果、best neighbor は `exit_close=4, exit_far=6, close_entry_p95<=45..50, neg_p95>=52` で holdout `65.533 m`, tuned `79.345 m` だった。
- ただし現行 representative (`65.542 m`, `79.406 m`) との差は holdout `0.009 m`, tuned `0.061 m` に留まり、promotion threshold `0.1 m` を下回る。
- よってこの neighborhood は plateau と判断し、実装・探索フェーズは `entry_veto_negative_exit_rescue_branch_aware_hysteresis_quality_veto_regime_gate` を exploratory winner に据えたまま凍結する。

## richer gate の探索結果

grid search summary:
[ppc_pf_rich_gate_t23_n2_v2_best.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/ppc_pf_rich_gate_t23_n2_v2_best.csv)

現在の best pilot config:
- `PF-RichGate(b=0, p=0, d=80)`
- `mean RMS 2D = 68.57 m`
- `mean P95 = 93.42 m`
- `PF wins = 3/3`

ただしこの結果は pilot 限定であり、full 6 validation では一般化しなかった。
したがって現時点では「次の候補を掘るための seed」としてのみ扱う。

## 実行手順

1. feature/trajectory dump を作る

```bash
PYTHONPATH=python:experiments python3 -u experiments/exp_ppc_pf_rich_gate_search.py \
  --data-root /tmp/PPC-real/PPC-Dataset \
  --cache-root /tmp/plateau_segment_cache \
  --systems G \
  --segments tokyo/run2,tokyo/run3,nagoya/run2 \
  --max-epochs 100 \
  --n-particles 10000 \
  --sample-particles 64 \
  --blocked-nlos-prob 0.05 \
  --clear-nlos-prob 0.01 \
  --blocked-grid 0,0.001,0.002,0.005,0.01,0.02 \
  --positive-grid 0,0.25,0.5,0.75 \
  --disagreement-grid 0,5,10,20,40,80 \
  --results-prefix ppc_pf_rich_gate_t23_n2_v2
```

2. strategy lab evaluator を回す

```bash
PYTHONPATH=python:experiments python3 -u experiments/pf_strategy_lab/evaluate_strategies.py \
  --feature-csv experiments/results/ppc_pf_rich_gate_t23_n2_v2_features.csv \
  --trajectory-csv experiments/results/ppc_pf_rich_gate_t23_n2_v2_trajectories.csv \
  --base-csv experiments/results/ppc_pf_rich_gate_t23_n2_v2_bases.csv \
  --results-prefix pf_strategy_lab_t23_n2
```

3. full 6 validation を回す

```bash
PYTHONPATH=python:experiments python3 -u experiments/exp_ppc_pf_rich_gate_search.py \
  --data-root /tmp/PPC-real/PPC-Dataset \
  --cache-root /tmp/plateau_segment_cache \
  --systems G \
  --max-epochs 100 \
  --n-particles 10000 \
  --sample-particles 64 \
  --blocked-nlos-prob 0.05 \
  --clear-nlos-prob 0.01 \
  --blocked-grid 0 \
  --positive-grid 0 \
  --disagreement-grid 80 \
  --results-prefix ppc_pf_rich_gate_positive6_v2
```

```bash
PYTHONPATH=python:experiments python3 -u experiments/pf_strategy_lab/evaluate_strategies.py \
  --feature-csv experiments/results/ppc_pf_rich_gate_positive6_v2_features.csv \
  --trajectory-csv experiments/results/ppc_pf_rich_gate_positive6_v2_trajectories.csv \
  --base-csv experiments/results/ppc_pf_rich_gate_positive6_v2_bases.csv \
  --results-prefix pf_strategy_lab_positive6
```

4. holdout candidate を作る

```bash
PYTHONPATH=python:experiments python3 -u experiments/scan_ppc_holdout_candidates.py \
  --data-root /tmp/PPC-real/PPC-Dataset \
  --subset-root /tmp/plateau_segment_cache \
  --systems G \
  --segment-length 100 \
  --search-radius 200 \
  --step 200 \
  --min-offset 200 \
  --mesh-radius 1 \
  --sample-stride 5 \
  --results-prefix ppc_holdout_candidates_tokyo_run1_r200_s200 \
  --segments tokyo/run1
```

5. holdout dump を作る

```bash
PYTHONPATH=python:experiments python3 -u experiments/exp_ppc_pf_rich_gate_search.py \
  --data-root /tmp/PPC-real/PPC-Dataset \
  --cache-root /tmp/plateau_segment_cache \
  --segment-spec-csv experiments/results/ppc_holdout_segments_r200_s200_best.csv \
  --systems G \
  --max-epochs 100 \
  --n-particles 10000 \
  --sample-particles 64 \
  --blocked-nlos-prob 0.05 \
  --clear-nlos-prob 0.01 \
  --blocked-grid 0 \
  --positive-grid 0 \
  --disagreement-grid 80 \
  --results-prefix ppc_pf_rich_gate_holdout6_r200_s200
```

6. holdout evaluator を回す

```bash
PYTHONPATH=python:experiments python3 -u experiments/pf_strategy_lab/evaluate_strategies.py \
  --feature-csv experiments/results/ppc_pf_rich_gate_holdout6_r200_s200_features.csv \
  --trajectory-csv experiments/results/ppc_pf_rich_gate_holdout6_r200_s200_trajectories.csv \
  --base-csv experiments/results/ppc_pf_rich_gate_holdout6_r200_s200_bases.csv \
  --results-prefix pf_strategy_lab_holdout6_r200_s200
```

7. family cross-validation を回す

```bash
PYTHONPATH=python:experiments python3 -u experiments/pf_strategy_lab/cross_validate_families.py \
  --train-feature-csv experiments/results/ppc_pf_rich_gate_positive6_v2_features.csv \
  --train-trajectory-csv experiments/results/ppc_pf_rich_gate_positive6_v2_trajectories.csv \
  --holdout-feature-csv experiments/results/ppc_pf_rich_gate_holdout6_r200_s200_features.csv \
  --holdout-trajectory-csv experiments/results/ppc_pf_rich_gate_holdout6_r200_s200_trajectories.csv \
  --results-prefix pf_strategy_family_cv_positive6_holdout6
```

## 次にやること

- UrbanNav fixed external validation を main table 用に整理する
- PPC tuned / holdout と UrbanNav external の位置づけを paper 上で明確に分離する
- 新 family の追加は止め、`always_robust` を safe baseline、`entry_veto_negative_exit_rescue_branch_aware_hysteresis_quality_veto_regime_gate` を exploratory best として固定する
- `negative_exit_rescue_branch_aware_hysteresis_quality_veto_regime_gate` と `entry_veto_negative_exit_rescue_branch_aware_hysteresis_quality_veto_regime_gate` の差が出る epoch を figure 化する
- strategy ごとに runtime ではなく「追加 feature 計算コスト」を整理して table 化する
- paper の method / ablation / limitations をこの fixed set に合わせて書く

## UrbanNav external validation

PPC は design / tuning / holdout に使い、UrbanNav は external validation に使う。
ここでは UrbanNav 上で threshold や gate を再調整しない。

使用データ:
- Tokyo `Odaiba`, `Shinjuku`
- rover: `ublox`
- constellation: `G`
- methods: `WLS`, `EKF`, `PF-10K`, `PF+RobustClear-10K`

データ取得:
- subset fetcher:
  [fetch_urbannav_subset.py](/workspace/ai_coding_ws/gnss_gpu/experiments/fetch_urbannav_subset.py)
- fixed evaluator:
  [exp_urbannav_fixed_eval.py](/workspace/ai_coding_ws/gnss_gpu/experiments/exp_urbannav_fixed_eval.py)

結果ファイル:
- clean aggregate runs:
  [urbannav_fixed_eval_external_g_ublox_runs.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/urbannav_fixed_eval_external_g_ublox_runs.csv)
- clean aggregate summary:
  [urbannav_fixed_eval_external_g_ublox_summary.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/urbannav_fixed_eval_external_g_ublox_summary.csv)
- Shinjuku fresh rerun:
  [urbannav_fixed_eval_shinjuku_g_ublox_runs.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/urbannav_fixed_eval_shinjuku_g_ublox_runs.csv)

run-wise highlights:

| run | WLS RMS 2D | EKF RMS 2D | PF-10K RMS 2D | PF+RobustClear RMS 2D |
|---|---:|---:|---:|---:|
| Odaiba | 87.92 m | **82.54 m** | 126.83 m | 130.22 m |
| Shinjuku | 149.78 m | **66.58 m** | 141.42 m | 142.73 m |

aggregate summary:

| method | mean RMS 2D | mean P95 | mean >100m rate | wins vs WLS RMS |
|---|---:|---:|---:|---:|
| `WLS` | 118.85 m | 138.06 m | 10.97% | 0/2 |
| `EKF` | **74.56 m** | **128.08 m** | **11.10%** | **2/2** |
| `PF-10K` | 134.13 m | 281.77 m | 40.04% | 1/2 |
| `PF+RobustClear-10K` | 136.47 m | 294.26 m | 39.69% | 1/2 |

解釈:
- UrbanNav external validation では、current PF family は accuracy claim を強めない。
- `EKF` が 2/2 run で `WLS` を上回り、mean RMS 2D でも best。
- `PF-10K` と `PF+RobustClear-10K` は `Odaiba` と `Shinjuku` のどちらでも `EKF` を超えない。
- したがって、UrbanNav は「PF が cross-dataset で一貫して改善する」証拠ではなく、現時点では method limitation の証拠として扱うべき。
- 一方で、UrbanNav subset fetch と fixed-eval pipeline 自体は揃ったので、今後は同じ条件で追加 run / trimble / G,E,J を検証できる。

注意:
- `Odaiba + Shinjuku` を 1 プロセスで連続実行した初回 run では、Shinjuku の `PF+RobustClear` が CUDA OOM で CPU fallback に落ちた。
- clean aggregate は `Odaiba` を multi-run job の先頭結果から、`Shinjuku` を fresh single-run から合成している。

## UrbanNav trimble follow-up

`ublox` だけだと receiver quality と method weakness が混ざるので、`trimble` でも fixed eval を実施した。
さらに `exp_urbannav_fixed_eval.py` を拡張し、`--isolate-methods` と `--save-epoch-errors` で full-run の epoch diagnostics を OOM なしに保存できるようにした。

結果ファイル:
- [urbannav_fixed_eval_external_g_trimble_runs.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/urbannav_fixed_eval_external_g_trimble_runs.csv)
- [urbannav_fixed_eval_external_g_trimble_summary.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/urbannav_fixed_eval_external_g_trimble_summary.csv)
- [urbannav_fixed_eval_external_g_trimble_diag_epochs.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/urbannav_fixed_eval_external_g_trimble_diag_epochs.csv)
- [urbannav_trimble_pf_vs_ekf_diagnostics.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/urbannav_trimble_pf_vs_ekf_diagnostics.csv)
- [urbannav_trimble_tail_diagnostics.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/urbannav_trimble_tail_diagnostics.csv)

trimble aggregate summary:

| method | mean RMS 2D | mean P95 | mean >100m rate |
|---|---:|---:|---:|
| `EKF` | **79.67 m** | **154.58 m** | **11.24%** |
| `PF+RobustClear-10K` | 99.53 m | 179.00 m | 16.16% |
| `PF-10K` | 100.52 m | 179.13 m | 16.33% |
| `WLS` | 104.91 m | 174.08 m | 14.16% |

epoch-level diagnostics:
- `trimble` にすると PF は `ublox` よりかなり改善するが、`EKF` は still best。
- `Odaiba` では PF 系が全体として負ける。`PF-10K` の `method_bad_ekf_ok` は `1267` epoch で、`>500 m` も `58` epoch 出る。
- `Shinjuku` では PF は約半分の epoch で `EKF` より良いが、`>100 m` の長い failure segment が残る。
- `PF+RobustClear` は `Shinjuku` で win-rate を `49.28% -> 51.16%` まで押し上げるが、`P95` はむしろ悪化する。

loader 修正:
- UrbanNav trimble RINEX は dataset 自体が `G/R/E/J/C` を持つが、旧 loader は `C1C/S1C` 固定と sat-id の空白 (`E 1`, `G 5`) のため multi-GNSS をかなり捨てていた。
- `python/gnss_gpu/io/urbannav.py` で constellation ごとの L1-like obs code fallback を追加し、`python/gnss_gpu/ephemeris.py` で sat-id を `E 1 -> E01`, `G 5 -> G05` に正規化した。
- その結果、UrbanNav trimble の smoke では `Odaiba` 50 epoch で median sat が `5 -> 14`、`Shinjuku` で `5 -> 9` まで増えた。

common-epoch multi-GNSS WLS 診断:
- [urbannav_fixed_eval_external_gej_trimble_wls_v2_runs.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/urbannav_fixed_eval_external_gej_trimble_wls_v2_runs.csv)
- [urbannav_trimble_common_epoch_wls_compare.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/urbannav_trimble_common_epoch_wls_compare.csv)

| run | config | common epochs | RMS 2D | P95 | >100m rate | >500m rate |
|---|---|---:|---:|---:|---:|---:|
| Odaiba | `G` | 10805 | 96.24 m | 165.74 m | 13.92% | 0.24% |
| Odaiba | `G,E,J` | 10805 | 1154.16 m | **106.14 m** | **5.35%** | 0.36% |
| Shinjuku | `G` | 16139 | 102.26 m | 181.02 m | 13.58% | 0.30% |
| Shinjuku | `G,E,J` | 16139 | 1530.69 m | **98.03 m** | **4.16%** | 0.59% |

解釈:
- UrbanNav multi-GNSS は「効かない」のではなく、「tail を強く改善する一方で少数の catastrophic outlier を増やす」挙動を示す。
- つまり strong accept に効く次の軸は、新しい gate family ではなく、multi-GNSS の measurement / ISB / robust-estimation path を安定化すること。
- 現時点では `G,E,J` の external result を main table にそのまま昇格させるべきではない。diagnostic result として扱う。

## UrbanNav multi-GNSS stabilization lab

「tail は良くなるが catastrophic で壊れる」状態をそのまま放置せず、same-input / same-metric で simple veto variants を比較した。

script:
- [exp_urbannav_multignss_stabilization.py](/workspace/ai_coding_ws/gnss_gpu/experiments/exp_urbannav_multignss_stabilization.py)

結果ファイル:
- [urbannav_multignss_stabilization_trimble_gej_features.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/urbannav_multignss_stabilization_trimble_gej_features.csv)
- [urbannav_multignss_stabilization_trimble_gej_runs.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/urbannav_multignss_stabilization_trimble_gej_runs.csv)
- [urbannav_multignss_stabilization_trimble_gej_summary.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/urbannav_multignss_stabilization_trimble_gej_summary.csv)
- [urbannav_multignss_stabilization_trimble_gej_best.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/urbannav_multignss_stabilization_trimble_gej_best.csv)

variants:
- `gps_only`
- `multi_raw`
- `multi_residual_bias_veto`
- `multi_comparative_residual`
- `multi_solution_gap_veto`

aggregate baseline:

| variant | mean RMS 2D | mean P95 | mean >100m rate | mean >500m rate |
|---|---:|---:|---:|---:|
| `gps_ekf_reference` | 79.88 m | 148.88 m | 10.94% | **0.000%** |
| `gps_only` | 99.25 m | 173.38 m | 13.75% | 0.269% |
| `multi_raw` | 1342.42 m | **102.08 m** | **4.75%** | 0.478% |

best simple family:

| variant | parameters | mean RMS 2D | mean P95 | mean >100m rate | mean >500m rate | mean use_multi_frac |
|---|---|---:|---:|---:|---:|---:|
| `multi_residual_bias_veto` | `residual_p95<=100`, `residual_max<=250`, `bias_delta<=100`, `extra_sat>=2` | **73.49 m** | **100.97 m** | **4.46%** | **0.046%** | 99.33% |

run-wise:

| run | variant | RMS 2D | P95 | >100m rate | >500m rate |
|---|---|---:|---:|---:|---:|
| Odaiba | `gps_only` | 96.24 m | 165.74 m | 13.92% | 0.241% |
| Odaiba | `multi_raw` | 1154.16 m | 106.14 m | 5.35% | 0.361% |
| Odaiba | `multi_residual_bias_veto` | **75.53 m** | **104.65 m** | **5.23%** | **0.056%** |
| Shinjuku | `gps_only` | 102.26 m | 181.02 m | 13.58% | 0.297% |
| Shinjuku | `multi_raw` | 1530.69 m | 98.03 m | 4.16% | 0.595% |
| Shinjuku | `multi_residual_bias_veto` | **71.45 m** | **97.29 m** | **3.70%** | **0.037%** |

解釈:
- UrbanNav multi-GNSS は raw のままだと危険だが、simple residual/bias veto だけで catastrophic tail をほぼ GPS-only 以下まで下げられる。
- しかも best veto は `use_multi_frac ≈ 99.3%` で、multi をほとんど維持したまま極端な epoch だけ切っている。
- `solution_gap` 単独 veto は効くが弱い。主因は estimator disagreement より measurement residual / bias の方にある。
- common-epoch 比較では best veto が `gps_ekf_reference` も上回る。`79.88 m / 148.88 m / 10.94%` に対して `73.49 m / 100.97 m / 4.46%`。ただし `>500m` は `0.000% -> 0.046%` なので、integrity を最優先する table では EKF を残す理由もある。
- したがって次に core 側へ昇格を検討すべき最小抽象は、「multi-GNSS measurement quality veto」であって、複雑な paper-only gate family ではない。

## UrbanNav fixed external after hook promotion

common-epoch lab で選んだ `residual/bias veto` を reusable hook に落として、fixed external eval を `trimble + G,E,J` で再実行した。

script / hook:
- [multi_gnss_quality.py](/workspace/ai_coding_ws/gnss_gpu/python/gnss_gpu/multi_gnss_quality.py)
- [exp_urbannav_baseline.py](/workspace/ai_coding_ws/gnss_gpu/experiments/exp_urbannav_baseline.py)
- [exp_urbannav_fixed_eval.py](/workspace/ai_coding_ws/gnss_gpu/experiments/exp_urbannav_fixed_eval.py)

結果ファイル:
- [urbannav_fixed_eval_external_gej_trimble_qualityveto_runs.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/urbannav_fixed_eval_external_gej_trimble_qualityveto_runs.csv)
- [urbannav_fixed_eval_external_gej_trimble_qualityveto_summary.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/urbannav_fixed_eval_external_gej_trimble_qualityveto_summary.csv)

aggregate summary:

| method | mean RMS 2D | mean P95 | mean >100m rate | mean >500m rate | mean ms/epoch |
|---|---:|---:|---:|---:|---:|
| `WLS` | 3179.37 m | 194.55 m | 10.37% | 2.909% | 0.024 |
| `WLS+QualityVeto` | 2933.77 m | 175.38 m | 10.13% | 2.552% | 0.195 |
| `EKF` | 93.25 m | 178.18 m | 16.29% | 0.161% | 0.031 |
| `PF-10K` | 67.61 m | 101.46 m | 5.44% | **0.000%** | 1.367 |
| `PF+RobustClear-10K` | **66.60 m** | **98.53 m** | **4.80%** | **0.000%** | 1.401 |

run-wise:

| run | method | RMS 2D | P95 | >100m rate | >500m rate |
|---|---|---:|---:|---:|---:|
| Odaiba | `EKF` | 89.42 m | 165.29 m | 16.36% | 0.000% |
| Odaiba | `PF-10K` | 63.49 m | 95.52 m | 3.43% | 0.000% |
| Odaiba | `PF+RobustClear-10K` | **61.86 m** | **94.12 m** | **3.54%** | 0.000% |
| Shinjuku | `EKF` | 97.07 m | 191.07 m | 16.23% | 0.321% |
| Shinjuku | `PF-10K` | 71.72 m | 107.39 m | 7.46% | 0.000% |
| Shinjuku | `PF+RobustClear-10K` | **71.33 m** | **102.94 m** | **6.06%** | 0.000% |

解釈:
- これは以前の `G`-only UrbanNav external 結論を更新する。loader fix と multi-GNSS measurement path 修正の後では、`trimble + G,E,J` の fixed external で PF family が明確に勝つ。
- `WLS+QualityVeto` は core に昇格した最小 hook としては有効だが、final external method としてはまだ弱い。raw multi-WLS の catastrophic tail を少し削るだけで、PF family の精度には届かない。
- `PF-10K` と `PF+RobustClear-10K` は両 run とも `EKF` を上回り、しかも `>500m` を 0 に抑えている。external validation の main table はこの設定へ切り替えてよい。
- したがって current strongest paper story は、「PPC で design / holdout、UrbanNav `trimble + G,E,J` で external validation、BVH で systems acceleration」である。

## UrbanNav external window robustness

2 run 平均だけだと “external breadth がまだ弱い” という批判は残る。そこで、同じ
fixed external dump を 500 epoch window / 250 epoch stride で切り、window 単位の
勝率を集計した。

script:
- [exp_urbannav_window_eval.py](/workspace/ai_coding_ws/gnss_gpu/experiments/exp_urbannav_window_eval.py)

結果ファイル:
- [urbannav_window_eval_external_gej_trimble_qualityveto_windows.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/urbannav_window_eval_external_gej_trimble_qualityveto_w500_s250_windows.csv)
- [urbannav_window_eval_external_gej_trimble_qualityveto_comparisons.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/urbannav_window_eval_external_gej_trimble_qualityveto_w500_s250_comparisons.csv)
- [urbannav_window_eval_external_gej_trimble_qualityveto_summary.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/urbannav_window_eval_external_gej_trimble_qualityveto_w500_s250_summary.csv)
- [urbannav_window_eval_external_gej_trimble_qualityveto_wins.png](/workspace/ai_coding_ws/gnss_gpu/experiments/results/urbannav_window_eval_external_gej_trimble_qualityveto_w500_s250_wins.png)

window summary:

| method | n windows | mean RMS 2D | mean P95 | mean >100m rate | mean >500m rate |
|---|---:|---:|---:|---:|---:|
| `EKF` | 127 | 85.52 m | 139.15 m | 16.38% | 0.205% |
| `PF-10K` | 127 | 65.10 m | 85.54 m | 5.91% | 0.000% |
| `PF+RobustClear-10K` | 127 | **63.59 m** | **84.61 m** | **5.11%** | **0.000%** |

win rates vs `EKF`:

| method | RMS wins | P95 wins | >100m wins | `>500m <= EKF` |
|---|---:|---:|---:|---:|
| `PF-10K` | 88 / 127 (69.29%) | 101 / 127 (79.53%) | 89 / 127 (70.08%) | 127 / 127 (100%) |
| `PF+RobustClear-10K` | **90 / 127 (70.87%)** | **102 / 127 (80.31%)** | **89 / 127 (70.08%)** | **127 / 127 (100%)** |

解釈:
- これで UrbanNav external の win は “Odaiba と Shinjuku の run 平均だけ” ではなくなった。固定 external 条件の下で、`PF+RobustClear-10K` は 127 個の window のうち 90 個で `EKF` より RMS が良く、102 個で `P95` が良い。
- 特に catastrophic tail は強く、`PF-10K` と `PF+RobustClear-10K` は全 127 window で `EKF` 以下の `>500m` 率を維持した。
- もちろん geography 自体はまだ Tokyo-only だが、「勝ちが 1 本の lucky interval に依存している」という弱点はかなり薄くなった。

## UrbanNav Hong Kong external control

Tokyo-only の geography 弱点をさらに崩すため、UrbanNav-HK-Data20190428 の最小 subset
support を追加し、GPS-only external control を実施した。

script / support:
- [fetch_urbannav_hk_subset.py](/workspace/ai_coding_ws/gnss_gpu/experiments/fetch_urbannav_hk_subset.py)
- [urbannav.py](/workspace/ai_coding_ws/gnss_gpu/python/gnss_gpu/io/urbannav.py)

結果ファイル:
- [urbannav_fixed_eval_hk20190428_g_ublox_summary.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/urbannav_fixed_eval_hk20190428_g_ublox_summary.csv)

summary:

| run | systems | rover | epochs | EKF RMS / P95 | PF-10K RMS / P95 | PF+RobustClear RMS / P95 |
|---|---|---|---:|---:|---:|---:|
| `HK_20190428` | `G` | `ublox` | 468 | **69.49 / 95.19 m** | 301.68 / 560.12 m | 302.14 / 530.56 m |

解釈:
- これは positive result ではなく、negative control である。Hong Kong 2019-04-28 の `G` 単独・median 6 satellites では current PF family は `EKF` を大きく下回る。
- したがって geography 弱点は完全には解消していない。ただし弱点の位置はかなり明確になった。current strongest claim は “UrbanNav Tokyo trimble + G,E,J external” であり、GPS-only low-satellite regime まで自動で一般化するわけではない。
- 言い換えると、いま残っている未解決点は “external gain が lucky interval かどうか” ではなく、“multi-GNSS repaired regime を超えた geographic/domain generalization” である。

## Hong Kong safety rescue variants

Hong Kong `G`-only negative control のあと、`G,C` を使う HK external で PF の完全崩壊を safety
variant で止められるかを見た。ここでは new headline method を作るのではなく、同じ fixed evaluator
の中で `EKF anchor + multi-GNSS quality veto` を足した PF variant を比較した。

script / hook:
- [exp_urbannav_pf3d.py](/workspace/ai_coding_ws/gnss_gpu/experiments/exp_urbannav_pf3d.py)
- [exp_urbannav_fixed_eval.py](/workspace/ai_coding_ws/gnss_gpu/experiments/exp_urbannav_fixed_eval.py)
- [test_urbannav_pf_stabilization.py](/workspace/ai_coding_ws/gnss_gpu/tests/test_urbannav_pf_stabilization.py)

結果ファイル:
- [urbannav_fixed_eval_hk20190428_gc_rescue_v2_summary.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/urbannav_fixed_eval_hk20190428_gc_rescue_v2_summary.csv)
- [urbannav_fixed_eval_external_gej_trimble_rescue_v2__odaiba__pf_10k_runs.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/urbannav_fixed_eval_external_gej_trimble_rescue_v2__odaiba__pf_10k_runs.csv)
- [urbannav_fixed_eval_external_gej_trimble_rescue_v2__odaiba__pfplusrobustclear_10k_runs.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/urbannav_fixed_eval_external_gej_trimble_rescue_v2__odaiba__pfplusrobustclear_10k_runs.csv)
- [urbannav_fixed_eval_external_gej_trimble_rescue_v2__odaiba__pfplusekfrescue_10k_runs.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/urbannav_fixed_eval_external_gej_trimble_rescue_v2__odaiba__pfplusekfrescue_10k_runs.csv)
- [urbannav_fixed_eval_external_gej_trimble_rescue_v2__odaiba__pfplusrobustclearplusekfrescue_10k_runs.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/urbannav_fixed_eval_external_gej_trimble_rescue_v2__odaiba__pfplusrobustclearplusekfrescue_10k_runs.csv)

HK `G,C` summary:

| method | mean RMS 2D | mean P95 | mean >100m rate | mean >500m rate |
|---|---:|---:|---:|---:|
| `EKF` | **76.95 m** | **95.35 m** | **3.49%** | **0.000%** |
| `WLS+QualityVeto` | 82.06 m | 95.51 m | 3.70% | **0.000%** |
| `PF+EKFRescue-10K` | 81.07 m | 113.27 m | 9.45% | **0.000%** |
| `PF+RobustClear+EKFRescue-10K` | 81.26 m | 113.27 m | 9.45% | **0.000%** |
| `PF-10K` | 48585.75 m | 48872.05 m | 100.00% | 100.000% |
| `PF+RobustClear-10K` | 48588.11 m | 48858.40 m | 100.00% | 100.000% |

Tokyo `Odaiba` side check:

| method | RMS 2D | P95 | >100m rate | >500m rate |
|---|---:|---:|---:|---:|
| `PF-10K` | 63.49 m | 95.52 m | 3.43% | 0.000% |
| `PF+RobustClear-10K` | **61.86 m** | **94.12 m** | **3.54%** | 0.000% |
| `PF+EKFRescue-10K` | 73.72 m | 122.34 m | 8.30% | 0.000% |
| `PF+RobustClear+EKFRescue-10K` | 73.20 m | 122.97 m | 8.38% | 0.000% |

解釈:
- `EKF anchor rescue` は Hong Kong `G,C` で raw PF の catastrophic collapse を止める。`>500m` を `100% -> 0%` まで落とせたので、safety variant としては意味がある。
- ただし main external setting の Tokyo `Odaiba` では明確に悪化する。つまりこの variant は global winner ではない。
- したがって current main method は引き続き `PF+RobustClear-10K`。`PF+EKFRescue-10K` 系は “geography failure を完全崩壊から recover する supplemental safety variant” として保持する。
- これで弱点の位置はさらに狭まった。残る問題は “PF family が Hong Kong でも常に勝つか” であって、“Hong Kong で完全に壊れるか” ではなくなった。

## Cross-geometry guide-policy mitigation

Hong Kong rescue の次に、`EKF` を anchor にした guide family を切り分けた。狙いは
`Hong Kong sparse single-constellation` と `Tokyo repaired multi-GNSS` で、何が効くかを
同じ fixed evaluator で比べることだった。

script / hook:
- [exp_urbannav_pf3d.py](/workspace/ai_coding_ws/gnss_gpu/experiments/exp_urbannav_pf3d.py)
- [exp_urbannav_fixed_eval.py](/workspace/ai_coding_ws/gnss_gpu/experiments/exp_urbannav_fixed_eval.py)
- [test_urbannav_pf_stabilization.py](/workspace/ai_coding_ws/gnss_gpu/tests/test_urbannav_pf_stabilization.py)

guide policy variants:
- `PF+EKFGuide-10K`
  `always` に reference velocity を使う
- `PF+EKFGuideInit-10K`
  初期化だけ `EKF` に寄せる
- `PF+EKFGuideFallback-10K`
  quality-veto fallback epoch だけ reference velocity を使う
- `PF+RobustClear+EKFGuide*`
  上と同じ policy を robust-clear PF に載せる
- `PF+AdaptiveGuide-10K`
  single-constellation run では `PF+EKFGuide-10K`、multi-GNSS run では `PF+RobustClear+EKFGuideInit-10K` を選ぶ

結果ファイル:
- [urbannav_fixed_eval_external_gej_trimble_guide_policy_3k_summary.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/urbannav_fixed_eval_external_gej_trimble_guide_policy_3k_summary.csv)
- [urbannav_fixed_eval_external_gej_trimble_guide_policy_3k_runs.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/urbannav_fixed_eval_external_gej_trimble_guide_policy_3k_runs.csv)
- [urbannav_fixed_eval_hk20190428_gc_guide_policy_summary.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/urbannav_fixed_eval_hk20190428_gc_guide_policy_summary.csv)
- [urbannav_fixed_eval_hk20190428_gc_guide_policy_runs.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/urbannav_fixed_eval_hk20190428_gc_guide_policy_runs.csv)
- [urbannav_fixed_eval_external_gej_trimble_adaptive_3k_summary.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/urbannav_fixed_eval_external_gej_trimble_adaptive_3k_summary.csv)
- [urbannav_fixed_eval_external_gej_trimble_adaptive_3k_runs.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/urbannav_fixed_eval_external_gej_trimble_adaptive_3k_runs.csv)
- [urbannav_fixed_eval_hk20190428_gc_adaptive_summary.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/urbannav_fixed_eval_hk20190428_gc_adaptive_summary.csv)
- [urbannav_fixed_eval_hk20190428_gc_adaptive_runs.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/urbannav_fixed_eval_hk20190428_gc_adaptive_runs.csv)

Tokyo `trimble + G,E,J`, 3000-epoch summary:

| method | mean RMS 2D | mean P95 | mean >100m rate |
|---|---:|---:|---:|
| `EKF` | 67.85 m | 102.70 m | 6.53% |
| `PF+RobustClear-10K` | 64.58 m | 96.52 m | 4.98% |
| `PF+EKFGuide-10K` | **62.49 m** | 97.22 m | 5.65% |
| `PF+AdaptiveGuide-10K` | 62.90 m | **90.35 m** | **2.77%** |

run-wise highlights:

| run | best RMS-oriented guide variant | RMS 2D | P95 | >100m rate |
|---|---|---:|---:|---:|
| `Odaiba` | `PF+EKFGuide-10K` | **51.71 m** | **73.60 m** | **0.67%** |
| `Shinjuku` | `PF+AdaptiveGuide-10K` | **66.50 m** | **96.66 m** | **4.00%** |

Hong Kong `G,C` control:

| method | RMS 2D | P95 | >100m rate | >500m rate |
|---|---:|---:|---:|---:|
| `EKF` | 69.49 m | **95.19 m** | **2.99%** | **0.000%** |
| `PF+EKFGuide-10K` | **66.85 m** | 97.45 m | 3.85% | **0.000%** |
| `PF+AdaptiveGuide-10K` | **66.85 m** | 97.45 m | 3.85% | **0.000%** |
| `PF-10K` | 301.68 m | 560.12 m | 68.16% | 8.974% |

解釈:
- `always guide` の本体は `EKF`-derived velocity で、`init-only` や `fallback-only` だけでは Hong Kong sparse regime を救えない。
- 一方で Tokyo multi-GNSS では `always guide` が Shinjuku を壊しやすく、`PF+RobustClear+EKFGuideInit-10K` が `71.56 -> 66.50 m`, `116.71 -> 96.66 m`, `10.53% -> 4.00%` と明確に改善する。
- つまり弱点は「guide が無効」ではなく、「同じ guide policy を全 regime に固定すると壊れる」ことだった。
- `PF+AdaptiveGuide-10K` はその弱点を一段薄くする。Tokyo 3k と Hong Kong を合わせた 3-run 平均では `EKF = 68.40 m / 100.20 m / 5.35%` に対し、`PF+AdaptiveGuide-10K = 64.22 m / 92.72 m / 3.13%` まで改善する。
- ただしこれは current paper mainline を置き換える結果ではまだない。Tokyo full external main table の主役は引き続き `PF+RobustClear-10K` で、`PF+AdaptiveGuide-10K` は cross-geometry weakness を減らす supplemental variant として扱うのが妥当。

full-run confirmation:
- [urbannav_fixed_eval_external_gej_trimble_adaptive_full_summary.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/urbannav_fixed_eval_external_gej_trimble_adaptive_full_summary.csv)
- [urbannav_fixed_eval_external_gej_trimble_adaptive_full_runs.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/urbannav_fixed_eval_external_gej_trimble_adaptive_full_runs.csv)

full-run summary:

| method | mean RMS 2D | mean P95 | mean >100m rate | mean >500m rate |
|---|---:|---:|---:|---:|
| `PF+RobustClear-10K` | **66.60 m** | **98.53 m** | 4.80% | **0.000%** |
| `PF+AdaptiveGuide-10K` | 67.50 m | 100.78 m | **4.75%** | **0.000%** |
| `EKF` | 93.25 m | 178.18 m | 16.29% | 0.161% |

run-wise:

| run | `PF+RobustClear-10K` RMS / P95 / >100m | `PF+AdaptiveGuide-10K` RMS / P95 / >100m |
|---|---:|---:|
| `Odaiba` | 61.86 / 94.12 / 3.54% | **61.68 / 94.85 / 3.14%** |
| `Shinjuku` | **71.33 / 102.94 / 6.06%** | 73.32 / 106.70 / 6.36% |

full-run解釈:
- `PF+AdaptiveGuide-10K` は full-run でも `EKF` を大きく上回るが、current mainline `PF+RobustClear-10K` は超えない。
- `Odaiba` ではほぼ横並びで、`RMS` と `>100m` はわずかに良いが `P95` は悪化する。
- `Shinjuku` では 3k subset で見えた改善が full-run では維持されず、ここが mainline 置換に失敗した理由である。
- したがって `PF+AdaptiveGuide-10K` は supplemental cross-geometry mitigation として残し、main UrbanNav external table は引き続き `PF+RobustClear-10K` を使う。

## Paper-ready assets

paper packaging を固定するため、main table と figure を既存 CSV から再生成する builder を追加した。

script:
- [build_paper_assets.py](/workspace/ai_coding_ws/gnss_gpu/experiments/build_paper_assets.py)

outputs:
- [paper_main_table.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/paper_assets/paper_main_table.csv)
- [paper_main_table.md](/workspace/ai_coding_ws/gnss_gpu/experiments/results/paper_assets/paper_main_table.md)
- [paper_ppc_holdout.png](/workspace/ai_coding_ws/gnss_gpu/experiments/results/paper_assets/paper_ppc_holdout.png)
- [paper_urbannav_external.png](/workspace/ai_coding_ws/gnss_gpu/experiments/results/paper_assets/paper_urbannav_external.png)
- [paper_bvh_runtime.png](/workspace/ai_coding_ws/gnss_gpu/experiments/results/paper_assets/paper_bvh_runtime.png)

中身:
- main table は `PPC holdout`, `UrbanNav external`, `BVH systems` を 1 枚に並べる
- `paper_ppc_holdout.png` は holdout 6 segment で `always_robust` と exploratory gate の paired comparison
- `paper_urbannav_external.png` は `trimble + G,E,J` external の CDF と tail rate
- `paper_bvh_runtime.png` は `PF3D` と `PF3D-BVH` の runtime / accuracy 比較

解釈:
- これで paper の主結果は “その場で集めた表” ではなく、fixed CSV から再生成できる asset になった
- reviewers 向けには、main table の主役は `PF+RobustClear-10K`, `PF-10K`, `EKF`, `PF3D-BVH` で十分で、`WLS+QualityVeto` は promoted utility として補助的に置くのが自然

## Odaiba weak-DD adaptive-floor confirmation

`internal_docs/plan.md` の weak-DD / coverage-hole 方針に沿って、Odaiba PF smoother preset の
DD carrier adaptive floor を full-run で確認した。

採用:
- `odaiba_reference`: `--mupf-dd-gate-adaptive-floor-cycles 0.25 -> 0.18`
- `odaiba_reference_guarded`: `0.25 -> 0.18`

不採用:
- `odaiba_stop_detect`: `0.18` は入れず、`0.25` を維持

full-run 結果:

| preset | floor | FWD P50 | FWD RMS | SMTH P50 | SMTH RMS | 判定 |
|---|---:|---:|---:|---:|---:|---|
| `odaiba_reference` baseline | 0.25 | 1.46 m | 5.57 m | 1.38 m | 5.08 m | baseline |
| `odaiba_reference` updated | 0.18 | 1.42 m | 5.46 m | 1.38 m | 5.02 m | 採用 |
| `odaiba_reference_guarded` baseline | 0.25 | 1.46 m | 5.57 m | 1.38 m | 5.43 m | baseline |
| `odaiba_reference_guarded` updated | 0.18 | 1.42 m | 5.46 m | 1.38 m | 5.36 m | 採用 |
| `odaiba_stop_detect` baseline | 0.25 | 1.19 m | 4.57 m | 1.36 m | 4.11 m | 維持 |
| `odaiba_stop_detect` trial | 0.18 | 1.63 m | 5.50 m | 1.34 m | 4.11 m | 不採用 |

解釈:
- coverage-hole は単純な “DD carrier absent” ではなく、mediocre DD carrier を信頼しすぎる epoch が混じる問題だった。
- tracked fallback preference、ESS-only weak-DD replacement、spread-aware support-skip、contextual low-ESS epoch-median gate は promoted しない。
- accepted change は、reference/guarded preset だけ DD carrier adaptive pair-floor を `0.18` に締めること。
