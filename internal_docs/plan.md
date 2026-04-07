# gnss_gpu / gnss_gpu_ws Claude 引き継ぎメモ

**ドキュメント先頭の読み方**

- **セクション A（このファイルの上の方）**: 2026-04-07 時点の **RTKLIB demo5 と gnss_gpu の SPP（単点）観測モデル整合**、`export_spp_meas`、`gtsam_gnss` 公開 RINEX での FGO 検証。**いまコードを触るなら基本はここ。**
- **付録 B（ファイル後半）**: 2026-04-04 スナップショットの **UrbanNav / PF / paper assets frozen mainline**。別系統の実験パイプライン記録として残している。

---

## A. RTKLIB demo5 × gnss_gpu（SPP / FGO / 公開 RINEX）— 2026-04-07 更新

### A.0 目的（ユーザーが選んだ「1」）

- **demo5 の `pntpos` / `rescode` と同じ観測項**を gnss_gpu の WLS・FGO に流したい（Python 側で `correct_pseudoranges` + 自前エフェネリスだけだと **約 35 m 級のモデル不整合**が残っていた）。
- 結論: **ずれの本体は FGO ソルバではなく gnss_gpu 側の SPP 前処理**だった。観測を **RTKLIB が使っている分解（`prange`、ブロードキャストイオノ・トロポ、衛星時計）**に揃えると、公開クリップでは **FGO の reference 2D RMS が ~36 m → ~1 m 前後**に改善（下記数値）。

### A.1 リポジトリ・パスの前提

- ワークスペース例: `…/gnss_gpu_ws/`
- **gnss_gpu（Python + 実験）**: `gnss_gpu_ws/gnss_gpu/`
- **RTKLIB demo5**: `gnss_gpu_ws/ref/RTKLIB-demo5/`（clone 例: `rtklibexplorer/RTKLIB` の `demo5` ブランチ）
- **公開テストデータ（gtsam_gnss 同梱例）**: `gnss_gpu_ws/ref/gtsam_gnss/examples/data/`  
  - `rover_1Hz.obs`, `base.nav`, `reference.csv`

### A.2 観測モデル（RTKLIB `pntpos` / `rescode` と整合）

`pntpos.c` の疑似距離残差（単一星座・受信機時計を状態に含む）は次の形。

- `P = prange(obs, nav, opt, …)` … 生コード観測に **TGD / コードバイアス**などを反映した **幾何に乗せる前の擬似距離**
- 残差:  
  `v = P - ( r + dtr_rx - CLIGHT*dts_sat + d_iono + d_trop )`  
  ここで `r = geodist(rs, rr)`、`dts_sat` は `satposs` が出す衛星時計（秒）、`dtr_rx` は受信機時計バイアス（**メートル**として状態に入る）

**gnss_gpu の WLS が「幾何距離 + 受信機時計」だけを見る形**にするには、観測を次にするのが一貫する。

- `pseudorange_for_wls = P - d_iono - d_trop - sat_clk_m`
- ここで **`sat_clk_m = - CLIGHT * dts_sat`**（CSV でそう定義して出力している）

衛星位置は **RTKLIB の `satposs` がそのエポックで使った ECEF** を使うと、`geodist` と一致させやすい。

### A.3 新規 C ツール `export_spp_meas`

**場所**: `ref/RTKLIB-demo5/app/consapp/rnx2rtkp/gcc/export_spp_meas.c`

**役割**

- RINEX OBS + NAV を読み、エポックごとに `pntpos` を実行（`PMODE_SINGLE`, GPS, `IONOOPT_BRDC`, `TROPOPT_SAAS`, 仰角マスクは CLI で指定）。
- **`ssat[sat-1].vs`** で最終的に使われた衛星に限定し、`rescode` 相当のフィルタ（仰角・`testsnr` 等）に寄せたうえで CSV 出力。

**CSV ヘッダ（1 行 1 衛星）**

`gps_week,gps_tow,prn,sat_id,prange_m,r_m,iono_m,trop_m,sat_clk_m,satx,saty,satz`

- `sat_clk_m` = `-CLIGHT * dts`（上式の「衛星時計項」をメートルで足し引きしやすいように）
- `satx/y/z` = そのエポックの `satposs` ECEF（m）

**ビルド**（`rnx2rtkp` と同じ `gcc` ディレクトリ）

```bash
make -C ref/RTKLIB-demo5/app/consapp/rnx2rtkp/gcc
```

生成物: 同ディレクトリに `export_spp_meas` と `rnx2rtkp`。

**実行例**

```bash
./export_spp_meas rover_1Hz.obs base.nav -m 15
```

**実装上の注意**

- `traceopen(NULL)` は **呼ばない**（`trace.c` が `strcpy` で落ちる）。
- `postpos.o` リンクのため、`rnx2rtkp.c` と同様に **`showmsg` / `settspan` / `settime` のスタブ**を同 C ファイルに定義。

**Makefile 変更**

- 共有オブジェクトを `RTK_OBJS` にまとめ、`rnx2rtkp` / `export_spp_meas` の両ターゲットからリンク。
- `make all` で **両方**ビルド。

### A.4 Python 側: `gtsam_public_dataset.py`

**`build_public_gtsam_arrays`**

- 新オプション:  
  `rtklib_export_spp_exe: Path | None = None`, `el_mask_deg: float = 15.0`
- `rtklib_export_spp_exe` 指定時:
  - 一時 CSV に `export_spp_meas` を **subprocess** で流し、パース。
  - キー: `(gps_week, round(tow, 4), sat_id)`（例: `G10`）。
  - 各エポックで RINEX 側の衛星順を保ちつつ、`pseudorange` に  
    `prange_m - iono_m - trop_m - sat_clk_m`  
    を代入、`sat_ecef` に CSV の `satx/y/z` を代入。
  - **重み**は従来どおり `correct_pseudoranges(...)[1]` の elevation 由来を流用（`pr_raw` はダミー扱いで、仰角・マスクのためだけに使う）。完全な RTKLIB `varerr` との一致は未実装。
- **エポックメタデータ**: `epochs_data` の型を  
  `(tow, sats, pr_array, gps_week)` に拡張（既存コードが `epochs_data[t][0]` で TOW だけ見る用法は互換）。

**関連 imports**

- `_datetime_to_gps_week` を `gnss_gpu.io.nav_rinex` から使用。

### A.5 実験スクリプト: `compare_fgo_rtklib_demo5.py`

- **既定**: `rnx2rtkp` と同じディレクトリの `export_spp_meas` があれば **自動で SPP 整合パスを使用**（環境変数 `RTKLIB_EXPORT_SPP_MEAS` でも指定可）。
- **`--rtklib-export-spp PATH`**: バイナリ明示。
- **`--no-rtklib-spp-export`**: 従来の `correct_pseudoranges` + `Ephemeris` のみ（モデル不整合再現に有用）。
- **elevation mask** は `--elev` と `build_public_gtsam_arrays(..., el_mask_deg=...)` で一致させる。

**実行例（リポジトリルート想定）**

```bash
cd gnss_gpu
PYTHONPATH=python python3 experiments/compare_fgo_rtklib_demo5.py --max-epochs 60
```

### A.6 代表的な数値（公開クリップ・60 epoch・GPS・15°）

`export_spp_meas` **オン**（上記 compare が自動採用した場合の一例）:

- **FGO RMS 2D vs `reference.csv`**: およそ **0.96 m**（以前、gnss_gpu 単独 SPP 経路では **~36 m** 級）
- **RTKLIB `rnx2rtkp` 単点 vs reference**: およそ **1.67 m**
- **||FGO − RTKLIB|| 2D**: およそ **0.82 m**（観測は揃えたが、重み・反復・時刻・vsat の厳密一致などで差は残りうる）

### A.7 まだやっていない / 次にやるとよいこと

1. **残差 ~0.8 m の ||FGO − RTKLIB||**  
   - RTKLIB `varerr` と gnss_gpu 重みの統一、最終 `vsat` 集合の完全一致、受信時刻補正まわりの確認。
2. **`validate_fgo_gtsam_public_dataset.py`** に  
   `--rtklib-export-spp` / 環境変数連携を足し、CI・ドキュメントの一本化。
3. **運用・依存削減**  
   - subprocess + CSV の代わりに、共有ライブラリ + pybind などで `export_spp_meas` 相当を呼ぶ。
4. **別 RINEX / マルチ GNSS** で同パイプラインの回归テスト。

### A.8 主要ファイル一覧（RTKLIB ライン）

| 役割 | パス |
| --- | --- |
| RTKLIB CSV 出力 | `gnss_gpu_ws/ref/RTKLIB-demo5/app/consapp/rnx2rtkp/gcc/export_spp_meas.c` |
| RTKLIB ビルド | `…/rnx2rtkp/gcc/makefile` |
| バッチ組み立て | `gnss_gpu/experiments/gtsam_public_dataset.py` |
| FGO vs demo5 比較 | `gnss_gpu/experiments/compare_fgo_rtklib_demo5.py` |
| FGO 検証（未: RTKLIB フラグ） | `gnss_gpu/experiments/validate_fgo_gtsam_public_dataset.py` |
| 旧・モデルギャップ診断 | `gnss_gpu/experiments/diagnose_spp_rtklib_gap.py`（概念確認用） |
| Python SPP（レガシー経路） | `gnss_gpu/python/gnss_gpu/spp.py`（`correct_pseudoranges`） |

### A.9 Claude への注意（RTKLIB ライン）

- **`export_spp_meas` をビルドしていない環境**では compare は `export_spp_meas` を付けずに `build_public_gtsam_arrays` を呼ぶ（`batch_kw` が空 → レガシー `correct_pseudoranges` 経路）。**意図せず古い SPP 経路に戻る**ので、バイナリの有無を README かスクリプト先頭で明示するとよい。
- RTKLIB の `prange` は **`pntpos.c` 内 static**。現状、GPS L1 単頻度向けに **必要最小限を `export_spp_meas.c` 側に複製**している。マルチシステム・IF 組合せ까지揃えるなら **RTKLIB 本体の関数を EXPORT するリファクタ**が必要。
- `correct_pseudoranges` の重みは **RTKLIB と同一ではない**（セクション A.4）。厳密一致が必要なら重みもエクスポートするか、Python で `varerr` を移植する。

---

## 付録 B: UrbanNav / PF / paper assets frozen mainline（2026-04-04 スナップショット）

以下は **別ライン**（粒子フィルタ・都市外乱データ・論文 asset）のフリーズ記録。**HEAD / ブランチ名は当時のメモ**であり、gnss_gpu_ws の現在の git と一致しない場合がある。必要なら倉庫履歴で照合すること。

**最終更新（当時）**: 2026-04-04 JST  
**現在の HEAD（当時）**: `41ccb98` (`refine teaser media layout`)  
**ブランチ状態（当時）**: `main`, worktree clean  
**現フェーズ（当時）**: 実装・探索フェーズ凍結済み。いまは artifact / README / GitHub Pages / 原稿パッケージングの段階。  

---

### B.0 最初に読むもの

1. `README.md`
2. `docs/experiments.md`
3. `docs/decisions.md`
4. `docs/interfaces.md`
5. `docs/paper_draft_2026-04-01.md`
6. `experiments/results/paper_assets/paper_main_table.md`
7. `docs/assets/results_snapshot.json`

この節以下の「`docs/plan.md`」参照は歴史的表記。**現行の引き継ぎは本ファイルのセクション A を優先。**

---

### B.450 いまの結論を先に書く

#### B.450.1 frozen mainline

- **mainline method は `PF+RobustClear-10K`**
- これは **UrbanNav external** の full-run で一番安全に勝っている構成
- README, GitHub Pages, paper assets, snapshot JSON はこの前提に揃っている

#### B.450.2 exploratory / supplemental

- **PPC gate family** は残しているが exploratory
- **`entry_veto_negative_exit_rescue_branch_aware_hysteresis_quality_veto_regime_gate`** は PPC holdout で surviving gate だが gain は小さい
- **`PF+AdaptiveGuide-10K`** と **`PF+EKFRescue-10K`** は supplemental
- **explicit 3D PF / PF3D-BVH** は accuracy の headline ではなく **systems result**

#### B.450.3 safe headline

いま安全に言えるのは次の 3 本だけ。

1. **UrbanNav external では multi-GNSS PF path が EKF を明確に上回る**
2. **Hong Kong 3シーケンスでも PF+AdaptiveGuide が EKF を上回る（cross-geography breadth）**
3. **BVH は PF3D の runtime を大幅に削る**
4. **PPC では holdout-surviving な小さい gate gain があるが、headline ではない**

#### B.450.4 unsafe headline

以下は今も危ない。

- `world first`
- `3D map aided PF improves real-data accuracy` を主張の中心に置くこと
- `guaranteed strong accept`
- `geography-independent general win`
- `adaptive / rescue` を mainline に昇格させること

---

### B.451 現在の数値

#### B.451.1 paper main table の固定値

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

#### B.451.2 UrbanNav external の補強

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
    - `>500m 127/127 ≤`

つまり Tokyo external は「たまたま Odaiba / Shinjuku の run 平均で勝った」だけではない。

#### B.451.3 PPC holdout の位置づけ

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

#### B.451.4 Hong Kong の位置づけ

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

### B.452 method freeze

#### B.452.1 mainline

**`PF+RobustClear-10K`**

理由:

- UrbanNav external full-run の frozen winner
- `PF-10K` との差は大きくはないが、tail 指標まで含めて最も安定
- README / Pages / paper assets をこの method に揃え済み

#### B.452.2 exploratory gate

**`entry_veto_negative_exit_rescue_branch_aware_hysteresis_quality_veto_regime_gate`**

理由:

- PPC holdout で生き残る non-trivial gate
- ただし improvement は小さい
- mainline ではなく appendix / lab result 扱いが妥当

#### B.452.3 promoted core hook

**`WLS+QualityVeto`**

場所:

- `python/gnss_gpu/multi_gnss_quality.py`
- `experiments/exp_urbannav_baseline.py`
- `experiments/exp_urbannav_fixed_eval.py`

意味:

- multi-GNSS stabilization policy を reusable hook として core 側へ押し上げた
- ただし best external method ではない

#### B.452.4 supplemental variants

- `PF+AdaptiveGuide-10K`
- `PF+EKFRescue-10K`
- `PF+RobustClear+EKFRescue-10K`

役割:

- Hong Kong や sparse regime の mitigation
- cross-geometry weakness の応急処置
- Tokyo full-run frozen mainline の置換ではない

#### B.452.5 3D path

- `PF3D-BVH` は **systems contribution**
- explicit blocked/NLOS likelihood を headline accuracy result にしないこと

理由:

- real PLATEAU + NLOS で explicit 3D likelihood は安定勝ちしていない
- hard / mixture / gate を掘ったが、mainline にはなっていない
- 一方で runtime gain は非常に強い

---

### B.453 ここまでに試して、主役から降ろしたもの

#### B.453.1 PF strategy zoo

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

#### B.453.2 adaptive guide

`PF+AdaptiveGuide-10K` は 3-run mixed regime では良く見えたが、full Tokyo external では frozen mainline を超えなかった。

出典:

- `experiments/results/urbannav_fixed_eval_external_gej_trimble_adaptive_3k_summary.csv`
- `experiments/results/urbannav_fixed_eval_external_gej_trimble_adaptive_full_summary.csv`

結論:

- supplemental に留める
- README / Pages / paper main table を差し替えない

#### B.453.3 rescue variants

Hong Kong では効くが Odaiba で悪化する。

結論:

- safety option としては有用
- mainline には昇格させない

#### B.453.4 3D map accuracy story

real PLATEAU path は integration / systems 的には重要だが、accuracy headline には使わない。

結論:

- runtime figure は main figure で良い
- accuracy 主張は `PF+RobustClear-10K` に寄せる

---

### B.454 重要ファイルの地図

#### B.454.1 main results / artifact builders

- `experiments/build_paper_assets.py`
- `experiments/build_githubio_summary.py`
- `experiments/build_site_media.py`
- `experiments/results/paper_assets/paper_main_table.md`
- `docs/assets/results_snapshot.json`
- `docs/assets/results_snapshot.js`

#### B.454.2 website / README

- `README.md`
- `docs/index.html`
- `docs/site.css`
- `.github/workflows/pages.yml`
- `tests/site/playwright.config.cjs`
- `tests/site/site.spec.cjs`

#### B.454.3 frozen evaluation entry points

- `experiments/exp_urbannav_fixed_eval.py`
- `experiments/exp_urbannav_baseline.py`
- `experiments/exp_urbannav_pf.py`
- `experiments/exp_urbannav_pf3d.py`

#### B.454.4 loaders

- `python/gnss_gpu/io/ppc.py`
- `python/gnss_gpu/io/urbannav.py`
- `python/gnss_gpu/io/plateau.py`
- `python/gnss_gpu/ephemeris.py`

#### B.454.5 strategy lab

- `experiments/pf_strategy_lab/strategies.py`
- `experiments/pf_strategy_lab/evaluate_strategies.py`
- `experiments/pf_strategy_lab/cross_validate_families.py`

#### B.454.6 docs for process

- `docs/experiments.md`
- `docs/decisions.md`
- `docs/interfaces.md`
- `docs/paper_draft_2026-04-01.md`

---

### B.455 README / GitHub Pages / media の現状

#### B.455.1 README

README はすでに artifact-first に更新済み。

載せているもの:

- poster
- teaser GIF
- teaser `mp4` / `webm`
- main figures
- reproduce commands
- method freeze
- safe / unsafe claim の整理

#### B.455.2 GitHub Pages

Pages は `docs/index.html` から静的表示する。

特徴:

- `results_snapshot.js` を読む
- `noscript` fallback あり
- main figures と extra charts を表示
- teaser video は controls なし、`preload="metadata"`
- Playwright smoke test あり

#### B.455.3 teaser 修正

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

#### B.455.4 Pages workflow

`.github/workflows/pages.yml` は以下を通す。

1. `python3 experiments/build_paper_assets.py`
2. `python3 experiments/build_githubio_summary.py`
3. `npm ci`
4. `npm run site:smoke`

この順にしてあるので、paper assets と Pages assets のズレが起きにくい。

---

### B.456 validation 状態

#### B.456.1 freeze validation

出典: `experiments/results/freeze_validation_summary.json`

- headline: `440 passed, 7 skipped`
- full summary: `440 passed, 7 skipped, 17 warnings`
- command: `PYTHONPATH=python python3 -m pytest tests/ -q`

warning の中身:

- `pytest.mark.slow`
- `datetime.utcnow()`
- plotting / matplotlib

いまのところ freeze を止める性質の warning ではない。

#### B.456.2 site validation

- `npm run site:smoke`
- Playwright 2 tests pass

これは desktop / mobile の smoke で、main sections, figures, video, overflow を見ている。

#### B.456.3 current repo state（当時）

- branch: `main`
- HEAD: `41ccb98`
- worktree: clean

---

### B.457 data / loaders の整理

#### B.457.1 PPC

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

#### B.457.2 UrbanNav Tokyo

役割:

- external validation の主戦場
- main paper claim の source

主ファイル:

- `python/gnss_gpu/io/urbannav.py`
- `experiments/fetch_urbannav_subset.py`
- `experiments/exp_urbannav_fixed_eval.py`

#### B.457.3 UrbanNav Hong Kong

役割:

- cross-geometry weakness の確認
- supplemental mitigation の testbed

主ファイル:

- `experiments/fetch_urbannav_hk_subset.py`
- `experiments/exp_urbannav_fixed_eval.py`

#### B.457.4 PLATEAU

役割:

- PF3D / BVH systems path
- real mesh integration

主ファイル:

- `python/gnss_gpu/io/plateau.py`
- `experiments/fetch_plateau_subset.py`
- `experiments/scan_ppc_plateau_segments.py`

---

### B.458 まだ残る弱点

全部は潰れていない。いま残っている弱点はかなり限定的。

#### B.458.1 geography breadth

- Tokyo external は強い（2シーケンス、PF+RobustClear mainline）
- Hong Kong 3シーケンスで PF+AdaptiveGuide が EKF を上回る
- ただし HK の winning method は mainline と異なる
- 5シーケンス/2都市の cross-geography breadth がある

#### B.458.2 3D map accuracy headline

- 3D path は systems 的に強い
- でも explicit 3D likelihood が real-data accuracy を押し上げた、とはまだ言いにくい

#### B.458.3 PPC gate gain の小ささ

- holdout-surviving だが small gain
- algorithm novelty の主役に据えるには弱い

#### B.458.4 PF vs PF+RobustClear の差の小ささ

- `PF-10K` も close ablation
- だから robust-clear story は「real but not huge」

これは弱点でもあるが、同時に誠実さでもある。過大主張しない方がいい。

---

### B.459 Claude が次にやるなら（PF / paper ライン）

#### B.459.1 いちばん安全な路線

**新しい method を増やさない。**

やること:

1. manuscript source へ fixed assets を移植
2. bibliography / citation 整理
3. figure / table の caption を仕上げる
4. README / Pages と paper の wording を揃える

#### B.459.2 もし追加実験をするなら

優先順位:

1. **multi-GNSS external breadth の追加**
2. **Hong Kong でも headline が立つ regime の探索**
3. **3D path の systems benchmark 拡充**

やらない方がいい:

- 新しい PPC gate family をさらに量産
- 3D likelihood の headline accuracy 化を急ぐ
- adaptive / rescue を mainline へ無理に昇格

#### B.459.3 artifact / infra で触るなら

候補:

- Pages に captions や downloadable CSV 導線を追加
- CI warnings の軽減
- media の圧縮や alt text 改善

ただし main story 自体はもう固定でよい。

---

### B.460 Claude への注意事項（PF / paper）

#### B.460.1 変えない方がいいもの

- `PF+RobustClear-10K` を mainline とする freeze
- `paper_main_table.md` の headline table
- Pages / README / snapshot JSON の mainline wording

#### B.460.2 変えてよいもの

- paper wording
- captions
- bibliography
- asset presentation
- supplemental section の整理

#### B.460.3 避けるべき主張

- `strong accept は確定`
- `3D map が real-data accuracy を押し上げた`
- `global / geography-independent win`
- `world first`

#### B.460.4 安全な主張

- `UrbanNav external では frozen PF path が EKF を大きく上回った`
- `BVH keeps PF3D accuracy while delivering a large runtime reduction`
- `PPC gate work is exploratory but holdout-surviving`
- `the package now supports honest, reproducible artifact-level evaluation`

---

### B.461 最低限の再生成コマンド（PF / paper artifact）

artifact 層だけならこれで足りる。

```bash
python3 experiments/build_paper_assets.py
python3 experiments/build_site_media.py
python3 experiments/build_githubio_summary.py
npm run site:smoke
PYTHONPATH=python python3 -m pytest tests/ -q
```

#### B.461.1 key outputs

- `experiments/results/paper_assets/paper_main_table.md`
- `experiments/results/paper_assets/paper_captions.md`
- `docs/assets/results_snapshot.json`
- `docs/assets/results_snapshot.js`
- `docs/assets/media/site_teaser.gif`
- `docs/assets/media/site_teaser.mp4`
- `docs/assets/media/site_teaser.webm`

---

### B.462 一言でまとめると（PF / paper）

この repo はもう「新しい gate を探す場所」ではない。  
いまは **`PF+RobustClear-10K` を frozen mainline として提示し、PPC は design-space、BVH は systems、Hong Kong は limitation/control として正直に並べる段階** である。

Claude が **PF/paper ライン**に入るなら、仕事は exploration ではなく **curation / packaging / manuscript integration** が中心になる。

**RTKLIB / 単点観測 / FGO の作業はセクション A を参照。**
