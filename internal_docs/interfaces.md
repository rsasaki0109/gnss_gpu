# 最小インターフェース

## 目的

このファイルは「今ある実験で共通化できた最小面」だけを書く。
将来必要になるかもしれない interface は書かない。

## 1. Strategy interface

場所:
- [interfaces.py](/media/sasaki/aiueo/ai_coding_ws/gnss_gpu/experiments/pf_strategy_lab/interfaces.py)

```python
@dataclass(frozen=True)
class StrategyContext:
    segment_label: str
    epoch: int
    features: Mapping[str, float]

@dataclass(frozen=True)
class StrategyDecision:
    use_blocked: bool
    score: float
    rationale: str

class GateStrategy(Protocol):
    name: str
    style: str
    def required_features(self) -> tuple[str, ...]: ...
    def parameters(self) -> Mapping[str, float]: ...
    def decide(self, context: StrategyContext) -> StrategyDecision: ...
```

意味:
- `StrategyContext`: strategy が見てよい入力
- `StrategyDecision`: strategy の出力
- `GateStrategy`: variant 比較の最小契約

## 2. 共通入力フォーマット

### feature dump

生成元:
- [exp_ppc_pf_rich_gate_search.py](/media/sasaki/aiueo/ai_coding_ws/gnss_gpu/experiments/exp_ppc_pf_rich_gate_search.py)

必須列:
- `segment_label`
- `epoch`
- `mean_weighted_blocked_frac`
- `blocked_positive_frac_gt5`
- `disagreement_m`

任意列:
- `max_weighted_blocked_frac`
- `n_sat_blocked_gt_005`
- `n_sat_blocked_gt_010`
- `blocked_positive_mean_residual`
- `robust_positive_frac`
- `robust_positive_frac_gt5`
- `robust_mean_residual`
- `robust_mean_abs_residual`
- `robust_p95_abs_residual`
- `cb_disagreement_m`
- `satellite_count`

備考:
- `quality_veto_regime_gate` は `robust_p95_abs_residual` と `satellite_count` を close branch の veto に使う
- したがってこの 2 列は「常に必須」ではないが、current best family を比較する dump では実質必須
- `hysteresis_quality_veto_regime_gate`, `mode_aware_hysteresis_quality_veto_regime_gate`, `branch_aware_hysteresis_quality_veto_regime_gate`, `rescue_branch_aware_hysteresis_quality_veto_regime_gate`, `negative_exit_rescue_branch_aware_hysteresis_quality_veto_regime_gate`, `entry_veto_negative_exit_rescue_branch_aware_hysteresis_quality_veto_regime_gate` も同じ feature 列を使う
- 違いは strategy 内部の temporal state と、close/far branch ごとの persistence 規則だけ

### trajectory dump

必須列:
- `segment_label`
- `epoch`
- `gps_tow`
- `truth_x`, `truth_y`, `truth_z`
- `robust_x`, `robust_y`, `robust_z`
- `blocked_x`, `blocked_y`, `blocked_z`

意味:
- strategy は feature dump だけを見る
- evaluator は trajectory dump を使って accuracy を計算する

## 3. evaluator interface

場所:
- [evaluate_strategies.py](/media/sasaki/aiueo/ai_coding_ws/gnss_gpu/experiments/pf_strategy_lab/evaluate_strategies.py)

入力:
- `--feature-csv`
- `--trajectory-csv`
- `--base-csv` optional
- `--results-prefix`

出力:
- `*_runs.csv`
- `*_decisions.csv`
- `*_summary.csv`

実行上の約束:
- evaluator は segment ごとに fresh strategy instance を使う
- これにより、将来 stateful strategy を足しても segment 間の状態汚染を避ける
- 現在の `hysteresis_quality_veto_regime_gate`, `mode_aware_hysteresis_quality_veto_regime_gate`, `branch_aware_hysteresis_quality_veto_regime_gate`, `rescue_branch_aware_hysteresis_quality_veto_regime_gate`, `negative_exit_rescue_branch_aware_hysteresis_quality_veto_regime_gate`, `entry_veto_negative_exit_rescue_branch_aware_hysteresis_quality_veto_regime_gate` はこの前提に依存している

### family cross-validation

場所:
- [cross_validate_families.py](/media/sasaki/aiueo/ai_coding_ws/gnss_gpu/experiments/pf_strategy_lab/cross_validate_families.py)

入力:
- `--train-feature-csv`
- `--train-trajectory-csv`
- `--holdout-feature-csv`
- `--holdout-trajectory-csv`
- `--results-prefix`

出力:
- `*_configs.csv`
- `*_family_best.csv`

## 4. segment spec interface

場所:
- [exp_ppc_pf_rich_gate_search.py](/media/sasaki/aiueo/ai_coding_ws/gnss_gpu/experiments/exp_ppc_pf_rich_gate_search.py)

用途:
- tuned 6 segment 以外でも同じ dump-generation pipeline を再利用するための入力

必須列:
- `city`
- `run`
- `start_epoch`
- `nlos_fraction`
- `subset_key`
- `plateau_zone`

例:
- [ppc_holdout_segments_r200_s200_best.csv](/media/sasaki/aiueo/ai_coding_ws/gnss_gpu/experiments/results/ppc_holdout_segments_r200_s200_best.csv)

## 5. core と experiments の境界

## 6. UrbanNav fixed-eval interface

場所:
- [exp_urbannav_fixed_eval.py](/media/sasaki/aiueo/ai_coding_ws/gnss_gpu/experiments/exp_urbannav_fixed_eval.py)

入力:
- `--data-root`
- `--runs`
- `--systems`
- `--urban-rover`
- `--methods`
- `--isolate-methods`
- `--save-epoch-errors`
- `--quality-veto-residual-p95-max`
- `--quality-veto-residual-max`
- `--quality-veto-bias-delta-max`
- `--quality-veto-extra-sat-min`
- `--results-prefix`

出力:
- `*_runs.csv`
- `*_summary.csv`
- `*_epochs.csv` (`--save-epoch-errors` のとき)

実行上の約束:
- `--isolate-methods` は `(run, method)` ごとに fresh subprocess を使う
- これにより UrbanNav full-run の GPU method でも long-lived CUDA allocation を避ける
- `*_epochs.csv` の `error_2d` / `error_3d` は `compute_metrics` と同じ ENU-based 定義に揃える

## 7. UrbanNav loader invariants

場所:
- [urbannav.py](/media/sasaki/aiueo/ai_coding_ws/gnss_gpu/python/gnss_gpu/io/urbannav.py)

約束:
- loader は sat-id の空白入り表現 (`E 1`, `G 5`) を許容する
- default の `obs_code=C1C`, `snr_code=S1C` でも constellation ごとの L1-like code fallback を使う
- つまり external experiment script 側は `G,E,J` を指定してよく、個別 obs code の知識を毎回持つ必要はない

## 8. UrbanNav multi-GNSS stabilization lab interface

場所:
- [exp_urbannav_multignss_stabilization.py](/media/sasaki/aiueo/ai_coding_ws/gnss_gpu/experiments/exp_urbannav_multignss_stabilization.py)

入力:
- `--data-root`
- `--runs`
- `--urban-rover`
- `--gps-systems`
- `--multi-systems`
- `--results-prefix`

出力:
- `*_features.csv`
- `*_runs.csv`
- `*_summary.csv`
- `*_best.csv`

実行上の約束:
- 比較は common epoch に揃える
- `gps_only` と `multi_raw` は必ず baseline として入れる
- veto family は same feature dump 上でだけ比較し、data loading や solver 実行を variant ごとに繰り返さない
- ranking は `mean_catastrophic_rate_pct -> mean_p95 -> mean_rms_2d` の順に行う

## 9. Multi-GNSS quality veto interface

場所:
- [multi_gnss_quality.py](/media/sasaki/aiueo/ai_coding_ws/gnss_gpu/python/gnss_gpu/multi_gnss_quality.py)
- [exp_urbannav_baseline.py](/media/sasaki/aiueo/ai_coding_ws/gnss_gpu/experiments/exp_urbannav_baseline.py)

core contract:
- `MultiGNSSQualityVetoConfig`
- `compute_multi_gnss_quality_metrics(...)`
- `accept_multi_gnss_solution(...)`
- `select_multi_gnss_solution(...)`

入力:
- reference-system-only epoch solution `[x, y, z, cb]`
- multi-GNSS epoch solution `[x, y, z]` と `bias_by_system`
- `sat_ecef`, `pseudoranges`, `system_ids`
- fixed threshold config

出力:
- chosen `position`
- chosen `clock_bias_m`
- `use_multi`
- metrics:
  `reference_satellite_count`, `multi_satellite_count`, `extra_satellite_count`,
  `reference_residual_p95_abs_m`, `reference_residual_max_abs_m`,
  `multi_residual_p95_abs_m`, `multi_residual_max_abs_m`, `multi_bias_range_m`

実行上の約束:
- hook 自体は epoch-local decision だけを行う
- GPS-only fallback solve は caller 側が用意する
- したがって core に入るのは “variant zoo” ではなく “quality metrics + accept/reject” だけ

`run_wls` hook:
- `run_wls(..., quality_veto_config=...)` を使うと、multi-constellation epoch で reference-system fallback を並列計算し、veto に通らない epoch だけ fallback に戻す

experiment hook:
- [exp_urbannav_pf3d.py](/media/sasaki/aiueo/ai_coding_ws/gnss_gpu/experiments/exp_urbannav_pf3d.py) の `run_pf_standard(..., quality_veto_config=None, guide_reference_positions=None, guide_initial_from_reference=False, guide_mode=\"always\", guide_satellite_max=None, rescue_reference_positions=None, rescue_distance_m=None)` と `run_pf3d_variant(...)` は、同じ veto config を PF update にも適用できる
- `guide_mode` は experiment-only の guide policy を切り替えるための最小 hook で、現在は `always`, `init_only`, `fallback_only`, `fallback_or_low_sat` を受ける
- `guide_reference_positions` は通常 `EKF` 軌跡を渡し、`guide_initial_from_reference` は初期粒子群だけを `EKF` に寄せる
- `rescue_reference_positions` は通常 `EKF` 軌跡を渡し、PF/EKF gap が `rescue_distance_m` を超えた epoch だけ PF を再中心化する
- これは still core には昇格しない。Tokyo external では悪化するため、現時点では supplemental safety variant としてだけ扱う

fixed evaluator method:
- [exp_urbannav_fixed_eval.py](/media/sasaki/aiueo/ai_coding_ws/gnss_gpu/experiments/exp_urbannav_fixed_eval.py) は experiment-only method として `PF+AdaptiveGuide-10K` を持つ
- この method は observable regime だけで分岐する
  - single-constellation run: `PF+EKFGuide-10K`
  - multi-GNSS run: `PF+RobustClear+EKFGuideInit-10K`
- まだ core abstraction には入れない。現時点では cross-geometry mitigation の比較対象であり、main paper table を置き換える stable method ではない

## 10. core と experiments の境界

core:
- `python/gnss_gpu/`
- `src/`
- `include/`

ここには「確立した expert 実装」だけ置く。

experiments:
- `experiments/*.py`
- `experiments/pf_strategy_lab/`

ここには以下を置く。
- variant
- feature dump
- search harness
- evaluator
- proxy metrics

## 11. paper asset builder interface

場所:
- [build_paper_assets.py](/media/sasaki/aiueo/ai_coding_ws/gnss_gpu/experiments/build_paper_assets.py)

入力:
- [pf_strategy_lab_holdout6_r200_s200_runs.csv](/media/sasaki/aiueo/ai_coding_ws/gnss_gpu/experiments/results/pf_strategy_lab_holdout6_r200_s200_runs.csv)
- [urbannav_fixed_eval_external_gej_trimble_qualityveto_summary.csv](/media/sasaki/aiueo/ai_coding_ws/gnss_gpu/experiments/results/urbannav_fixed_eval_external_gej_trimble_qualityveto_summary.csv)
- [urbannav_fixed_eval_external_gej_trimble_qualityveto_runs.csv](/media/sasaki/aiueo/ai_coding_ws/gnss_gpu/experiments/results/urbannav_fixed_eval_external_gej_trimble_qualityveto_runs.csv)
- [urbannav_fixed_eval_external_gej_trimble_qualityveto_epochs_epochs.csv](/media/sasaki/aiueo/ai_coding_ws/gnss_gpu/experiments/results/urbannav_fixed_eval_external_gej_trimble_qualityveto_epochs_epochs.csv) or isolated `__*__*_epochs.csv`
- [ppc_pf3d_tokyo_run1_g_100_plateau_summary.csv](/media/sasaki/aiueo/ai_coding_ws/gnss_gpu/experiments/results/ppc_pf3d_tokyo_run1_g_100_plateau_summary.csv)

出力:
- `experiments/results/paper_assets/paper_main_table.csv`
- `experiments/results/paper_assets/paper_main_table.md`
- `experiments/results/paper_assets/paper_ppc_holdout.png`
- `experiments/results/paper_assets/paper_urbannav_external.png`
- `experiments/results/paper_assets/paper_bvh_runtime.png`

実行上の約束:
- main table は `PPC holdout`, `UrbanNav external`, `BVH systems` の 3 section を持つ
- UrbanNav figure は CDF と tail rate を同じ raw epoch dump から作る
- builder は raw experiment を再実行せず、固定 CSV だけを見る

## 12. 今はまだ interface に入れないもの

以下はまだ共通化しない。

- runtime estimator interface
- training / learned gate interface
- multi-constellation gate interface
- paper-facing method abstraction

理由:
- まだ比較結果が足りない
- 共通面が発見されていない
