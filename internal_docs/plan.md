# gnss_gpu Claude 引き継ぎメモ

**最終更新**: 2026-04-04 JST  
**現在の HEAD**: `41ccb98` (`refine teaser media layout`)  
**ブランチ状態**: `main`, worktree clean  
**現フェーズ**: 実装・探索フェーズ凍結済み。いまは artifact / README / GitHub Pages / 原稿パッケージングの段階。  

---

## 0. 最初に読むもの

1. `README.md`
2. `docs/experiments.md`
3. `docs/decisions.md`
4. `docs/interfaces.md`
5. `docs/paper_draft_2026-04-01.md`
6. `experiments/results/paper_assets/paper_main_table.md`
7. `docs/assets/results_snapshot.json`

この `docs/plan.md` は「何が frozen で、何が exploratory で、Claude が次にどこを触るべきか」を 1 本で分かるように書いている。

---

## 1. いまの結論を先に書く

### 1.1 frozen mainline

- **mainline method は `PF+RobustClear-10K`**
- これは **UrbanNav external** の full-run で一番安全に勝っている構成
- README, GitHub Pages, paper assets, snapshot JSON はこの前提に揃っている

### 1.2 exploratory / supplemental

- **PPC gate family** は残しているが exploratory
- **`entry_veto_negative_exit_rescue_branch_aware_hysteresis_quality_veto_regime_gate`** は PPC holdout で surviving gate だが gain は小さい
- **`PF+AdaptiveGuide-10K`** と **`PF+EKFRescue-10K`** は supplemental
- **explicit 3D PF / PF3D-BVH** は accuracy の headline ではなく **systems result**

### 1.3 safe headline

いま安全に言えるのは次の 3 本だけ。

1. **UrbanNav external では multi-GNSS PF path が EKF を明確に上回る**
2. **Hong Kong 3シーケンスでも PF+AdaptiveGuide が EKF を上回る（cross-geography breadth）**
3. **BVH は PF3D の runtime を大幅に削る**
4. **PPC では holdout-surviving な小さい gate gain があるが、headline ではない**

### 1.4 unsafe headline

以下は今も危ない。

- `world first`
- `3D map aided PF improves real-data accuracy` を主張の中心に置くこと
- `guaranteed strong accept`
- `geography-independent general win`
- `adaptive / rescue` を mainline に昇格させること

---

## 2. 現在の数値

### 2.1 paper main table の固定値

出典: `experiments/results/paper_assets/paper_main_table.md`

| section | method | RMS 2D [m] | P95 [m] | >100m [%] | >500m [%] | time [ms/epoch] | note |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| PPC holdout | Safe baseline | 66.92 | 81.69 | 5.83 | 0.0 |  | `always_robust` |
| PPC holdout | Exploratory gate | 65.54 | 81.22 | 5.83 | 0.0 |  | `entry_veto_negative_exit_rescue...` |
| UrbanNav external | EKF | 93.25 | 178.18 | 16.29 | 0.161 | 0.031 | `trimble + G,E,J` |
| UrbanNav external | PF-10K | 67.61 | 101.46 | 5.44 | 0.0 | 1.367 | `trimble + G,E,J` |
| UrbanNav external | PF+RobustClear-10K | **66.60** | **98.53** | **4.80** | **0.0** | 1.401 | frozen mainline |
| UrbanNav external | WLS+QualityVeto | 2933.77 | 175.38 | 10.13 | 2.552 | 0.195 | promoted core hook |
| BVH systems | PF3D-10K | 55.50 | 58.39 | 0.0 | 0.0 | 1028.29 | real PLATEAU subset |
| BVH systems | PF3D-BVH-10K | 55.50 | 58.39 | 0.0 | 0.0 | **17.78** | **57.8x faster** |

### 2.2 UrbanNav external の補強

出典:

- `experiments/results/urbannav_fixed_eval_external_gej_trimble_qualityveto_summary.csv`
- `experiments/results/urbannav_window_eval_external_gej_trimble_qualityveto_w500_s250_summary.csv`

重要なのは run 平均 2 本だけではないこと。

- fixed full-run average:
  - `EKF = 93.25 / 178.18 / 16.29% / 0.161%`
  - `PF-10K = 67.61 / 101.46 / 5.44% / 0.000%`
  - `PF+RobustClear-10K = 66.60 / 98.53 / 4.80% / 0.000%`
- fixed window evaluation (`500 epoch / 250 stride`):
  - `PF+RobustClear-10K` は `EKF` に対して
    - `RMS 90/127 win`
    - `P95 102/127 win`
    - `>100m 89/127 win`
    - `>500m 127/127 <=`

つまり Tokyo external は「たまたま Odaiba / Shinjuku の run 平均で勝った」だけではない。

### 2.3 PPC holdout の位置づけ

出典:

- `experiments/results/pf_strategy_lab_positive6_summary.csv`
- `experiments/results/pf_strategy_lab_holdout6_r200_s200_summary.csv`
- `experiments/results/pf_strategy_family_cv_positive6_holdout6_family_best.csv`

PPC では gate family をかなり掘ったが、最終的な結論はこう。

- tuned split では gain がある
- holdout でも一応 survival する
- ただし gain は **小さい**
- したがって paper / README / Pages の main headline にしてはいけない

安全な表現は「design-space / ablation / experiment-first process の証拠」まで。

### 2.4 Hong Kong の位置づけ

出典:

- `experiments/results/urbannav_fixed_eval_hk20190428_gc_adaptive_summary.csv`
- `experiments/results/urbannav_fixed_eval_hk_tst_gc_ublox_summary.csv`
- `experiments/results/urbannav_fixed_eval_hk_whampoa_gc_ublox_summary.csv`

Hong Kong は **supplemental positive result** に昇格した。3シーケンス全てで `PF+AdaptiveGuide-10K` が `EKF` を上回る:

- HK-20190428 (GC): 66.85 m vs 69.49 m (EKF)
- HK TST (GC): 152.37 m vs 301.04 m (EKF) — 49% 改善
- HK Whampoa (GC): 413.68 m vs 463.09 m (EKF) — 11% 改善

ただし frozen mainline `PF+RobustClear-10K` は HK multi-GNSS で崩壊する。
勝つのは supplemental variant (`PF+AdaptiveGuide-10K`)。

paper での位置づけ: cross-geography breadth evidence として使う。
headline claim にはしない（winning method が mainline と異なるため）。

---

## 3. method freeze

### 3.1 mainline

**`PF+RobustClear-10K`**

理由:

- UrbanNav external full-run の frozen winner
- `PF-10K` との差は大きくはないが、tail 指標まで含めて最も安定
- README / Pages / paper assets をこの method に揃え済み

### 3.2 exploratory gate

**`entry_veto_negative_exit_rescue_branch_aware_hysteresis_quality_veto_regime_gate`**

理由:

- PPC holdout で生き残る non-trivial gate
- ただし improvement は小さい
- mainline ではなく appendix / lab result 扱いが妥当

### 3.3 promoted core hook

**`WLS+QualityVeto`**

場所:

- `python/gnss_gpu/multi_gnss_quality.py`
- `experiments/exp_urbannav_baseline.py`
- `experiments/exp_urbannav_fixed_eval.py`

意味:

- multi-GNSS stabilization policy を reusable hook として core 側へ押し上げた
- ただし best external method ではない

### 3.4 supplemental variants

- `PF+AdaptiveGuide-10K`
- `PF+EKFRescue-10K`
- `PF+RobustClear+EKFRescue-10K`

役割:

- Hong Kong や sparse regime の mitigation
- cross-geometry weakness の応急処置
- Tokyo full-run frozen mainline の置換ではない

### 3.5 3D path

- `PF3D-BVH` は **systems contribution**
- explicit blocked/NLOS likelihood を headline accuracy result にしないこと

理由:

- real PLATEAU + NLOS で explicit 3D likelihood は安定勝ちしていない
- hard / mixture / gate を掘ったが、mainline にはなっていない
- 一方で runtime gain は非常に強い

---

## 4. ここまでに試して、主役から降ろしたもの

### 4.1 PF strategy zoo

`experiments/pf_strategy_lab/` 以下でかなり多くの gate family を試した。

例:

- `always_robust`
- `always_blocked`
- `disagreement_gate`
- `rule_chain_gate`
- `weighted_score_gate`
- `clock_veto_gate`
- `dual_mode_regime_gate`
- `quality_veto_regime_gate`
- `hysteresis_quality_veto_regime_gate`
- `branch_aware_hysteresis_quality_veto_regime_gate`
- `rescue_branch_aware_hysteresis_quality_veto_regime_gate`
- `entry_veto_negative_exit_rescue_branch_aware_hysteresis_quality_veto_regime_gate`

結論:

- best surviving family は最後の `entry_veto_negative_exit...`
- それでも gain は small
- これ以上 strategy family を増やすのは return が薄い

### 4.2 adaptive guide

`PF+AdaptiveGuide-10K` は 3-run mixed regime では良く見えたが、full Tokyo external では frozen mainline を超えなかった。

出典:

- `experiments/results/urbannav_fixed_eval_external_gej_trimble_adaptive_3k_summary.csv`
- `experiments/results/urbannav_fixed_eval_external_gej_trimble_adaptive_full_summary.csv`

結論:

- supplemental に留める
- README / Pages / paper main table を差し替えない

### 4.3 rescue variants

Hong Kong では効くが Odaiba で悪化する。

結論:

- safety option としては有用
- mainline には昇格させない

### 4.4 3D map accuracy story

real PLATEAU path は integration / systems 的には重要だが、accuracy headline には使わない。

結論:

- runtime figure は main figure で良い
- accuracy 主張は `PF+RobustClear-10K` に寄せる

---

## 5. 重要ファイルの地図

### 5.1 main results / artifact builders

- `experiments/build_paper_assets.py`
- `experiments/build_githubio_summary.py`
- `experiments/build_site_media.py`
- `experiments/results/paper_assets/paper_main_table.md`
- `docs/assets/results_snapshot.json`
- `docs/assets/results_snapshot.js`

### 5.2 website / README

- `README.md`
- `docs/index.html`
- `docs/site.css`
- `.github/workflows/pages.yml`
- `tests/site/playwright.config.cjs`
- `tests/site/site.spec.cjs`

### 5.3 frozen evaluation entry points

- `experiments/exp_urbannav_fixed_eval.py`
- `experiments/exp_urbannav_baseline.py`
- `experiments/exp_urbannav_pf.py`
- `experiments/exp_urbannav_pf3d.py`

### 5.4 loaders

- `python/gnss_gpu/io/ppc.py`
- `python/gnss_gpu/io/urbannav.py`
- `python/gnss_gpu/io/plateau.py`
- `python/gnss_gpu/ephemeris.py`

### 5.5 strategy lab

- `experiments/pf_strategy_lab/strategies.py`
- `experiments/pf_strategy_lab/evaluate_strategies.py`
- `experiments/pf_strategy_lab/cross_validate_families.py`

### 5.6 docs for process

- `docs/experiments.md`
- `docs/decisions.md`
- `docs/interfaces.md`
- `docs/paper_draft_2026-04-01.md`

---

## 6. README / GitHub Pages / media の現状

### 6.1 README

README はすでに artifact-first に更新済み。

載せているもの:

- poster
- teaser GIF
- teaser `mp4` / `webm`
- main figures
- reproduce commands
- method freeze
- safe / unsafe claim の整理

### 6.2 GitHub Pages

Pages は `docs/index.html` から静的表示する。

特徴:

- `results_snapshot.js` を読む
- `noscript` fallback あり
- main figures と extra charts を表示
- teaser video は controls なし、`preload="metadata"`
- Playwright smoke test あり

### 6.3 teaser 修正

直近の `41ccb98` は teaser 修正。

何を直したか:

- 変な crop / composition をやめた
- paper figure をそのまま preview に使うように変更
- poster 風の full-frame slide にした
- `video.controls` を外して browser UI の被りを避けた

関係ファイル:

- `experiments/build_site_media.py`
- `docs/index.html`
- `docs/assets/media/site_teaser.gif`
- `docs/assets/media/site_teaser.mp4`
- `docs/assets/media/site_teaser.webm`

### 6.4 Pages workflow

`.github/workflows/pages.yml` は以下を通す。

1. `python3 experiments/build_paper_assets.py`
2. `python3 experiments/build_githubio_summary.py`
3. `npm ci`
4. `npm run site:smoke`

この順にしてあるので、paper assets と Pages assets のズレが起きにくい。

---

## 7. validation 状態

### 7.1 freeze validation

出典: `experiments/results/freeze_validation_summary.json`

- headline: `440 passed, 7 skipped`
- full summary: `440 passed, 7 skipped, 17 warnings`
- command: `PYTHONPATH=python python3 -m pytest tests/ -q`

warning の中身:

- `pytest.mark.slow`
- `datetime.utcnow()`
- plotting / matplotlib

いまのところ freeze を止める性質の warning ではない。

### 7.2 site validation

- `npm run site:smoke`
- Playwright 2 tests pass

これは desktop / mobile の smoke で、main sections, figures, video, overflow を見ている。

### 7.3 current repo state

- branch: `main`
- HEAD: `41ccb98`
- worktree: clean

---

## 8. data / loaders の整理

### 8.1 PPC

役割:

- design split
- ablation
- holdout gate evaluation

主ファイル:

- `python/gnss_gpu/io/ppc.py`
- `experiments/exp_ppc_wls_sweep.py`
- `experiments/exp_ppc_outlier_analysis.py`
- `experiments/exp_ppc_pf_gate_sweep.py`
- `experiments/exp_ppc_pf_blocked_clear_sweep.py`

### 8.2 UrbanNav Tokyo

役割:

- external validation の主戦場
- main paper claim の source

主ファイル:

- `python/gnss_gpu/io/urbannav.py`
- `experiments/fetch_urbannav_subset.py`
- `experiments/exp_urbannav_fixed_eval.py`

### 8.3 UrbanNav Hong Kong

役割:

- cross-geometry weakness の確認
- supplemental mitigation の testbed

主ファイル:

- `experiments/fetch_urbannav_hk_subset.py`
- `experiments/exp_urbannav_fixed_eval.py`

### 8.4 PLATEAU

役割:

- PF3D / BVH systems path
- real mesh integration

主ファイル:

- `python/gnss_gpu/io/plateau.py`
- `experiments/fetch_plateau_subset.py`
- `experiments/scan_ppc_plateau_segments.py`

---

## 9. まだ残る弱点

全部は潰れていない。いま残っている弱点はかなり限定的。

### 9.1 geography breadth

- Tokyo external は強い（2シーケンス、PF+RobustClear mainline）
- Hong Kong 3シーケンスで PF+AdaptiveGuide が EKF を上回る
- ただし HK の winning method は mainline と異なる
- 5シーケンス/2都市の cross-geography breadth がある

### 9.2 3D map accuracy headline

- 3D path は systems 的に強い
- でも explicit 3D likelihood が real-data accuracy を押し上げた、とはまだ言いにくい

### 9.3 PPC gate gain の小ささ

- holdout-surviving だが small gain
- algorithm novelty の主役に据えるには弱い

### 9.4 PF vs PF+RobustClear の差の小ささ

- `PF-10K` も close ablation
- だから robust-clear story は「real but not huge」

これは弱点でもあるが、同時に誠実さでもある。過大主張しない方がいい。

---

## 10. Claude が次にやるなら

### 10.1 いちばん安全な路線

**新しい method を増やさない。**

やること:

1. manuscript source へ fixed assets を移植
2. bibliography / citation 整理
3. figure / table の caption を仕上げる
4. README / Pages と paper の wording を揃える

### 10.2 もし追加実験をするなら

優先順位:

1. **multi-GNSS external breadth の追加**
2. **Hong Kong でも headline が立つ regime の探索**
3. **3D path の systems benchmark 拡充**

やらない方がいい:

- 新しい PPC gate family をさらに量産
- 3D likelihood の headline accuracy 化を急ぐ
- adaptive / rescue を mainline へ無理に昇格

### 10.3 artifact / infra で触るなら

候補:

- Pages に captions や downloadable CSV 導線を追加
- CI warnings の軽減
- media の圧縮や alt text 改善

ただし main story 自体はもう固定でよい。

---

## 11. Claude への注意事項

### 11.1 変えない方がいいもの

- `PF+RobustClear-10K` を mainline とする freeze
- `paper_main_table.md` の headline table
- Pages / README / snapshot JSON の mainline wording

### 11.2 変えてよいもの

- paper wording
- captions
- bibliography
- asset presentation
- supplemental section の整理

### 11.3 避けるべき主張

- `strong accept は確定`
- `3D map が real-data accuracy を押し上げた`
- `global / geography-independent win`
- `world first`

### 11.4 安全な主張

- `UrbanNav external では frozen PF path が EKF を大きく上回った`
- `BVH keeps PF3D accuracy while delivering a large runtime reduction`
- `PPC gate work is exploratory but holdout-surviving`
- `the package now supports honest, reproducible artifact-level evaluation`

---

## 12. 最低限の再生成コマンド

artifact 層だけならこれで足りる。

```bash
python3 experiments/build_paper_assets.py
python3 experiments/build_site_media.py
python3 experiments/build_githubio_summary.py
npm run site:smoke
PYTHONPATH=python python3 -m pytest tests/ -q
```

### 12.1 key outputs

- `experiments/results/paper_assets/paper_main_table.md`
- `experiments/results/paper_assets/paper_captions.md`
- `docs/assets/results_snapshot.json`
- `docs/assets/results_snapshot.js`
- `docs/assets/media/site_teaser.gif`
- `docs/assets/media/site_teaser.mp4`
- `docs/assets/media/site_teaser.webm`

---

## 13. 一言でまとめると

この repo はもう「新しい gate を探す場所」ではない。  
いまは **`PF+RobustClear-10K` を frozen mainline として提示し、PPC は design-space、BVH は systems、Hong Kong は limitation/control として正直に並べる段階** である。

Claude が次に入るなら、仕事は exploration ではなく **curation / packaging / manuscript integration** が中心になる。
