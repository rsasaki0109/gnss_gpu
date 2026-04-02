# gnss_gpu 完全引き継ぎドキュメント（Codex向け）

**最終更新**: 2026-04-02 20:23 JST
**ステータス**: コア実装完了（414テストpass）→ 実装・探索フェーズ凍結 / 結果整理・論文化フェーズ
**前任**: Claude Opus 4.6 (1Mコンテキスト)
**引き継ぎ先**: Codex

---

## 重要: まず読むべきファイル

1. `docs/design.md` — アーキテクチャ、研究動機、新規性分析
2. この `docs/plan.md` — 現状、問題、次のアクション
3. `docs/paper_draft_2026-04-01.md` — abstract / intro / related work の草案
4. `README.md` — プロジェクト概要
5. `benchmarks/RESULTS.md` — 性能ベンチマーク結果

---

## 1. プロジェクト現状

### 1.1 数値サマリー

| 指標 | 値 |
|------|-----|
| テスト | **414 passed**, 7 skipped |
| CUDAファイル | 23 (.cu) |
| Pythonファイル | 88 (.py) |
| pybind11 bindings | 20 (.cpp) |
| コミット | 11 (main) |
| ベンチマーク | 1M particles @ 12Hz, WLS @ 9.6M epoch/s |
| 検証 | gnss_lib_py (Stanford) 比較済み、31衛星サブnm精度 |

### 1.2 直近の更新

- `tests/test_multi_gnss.py::test_insufficient_satellites` と `tests/test_raim.py::test_raim_insufficient_satellites` は修正済み。
- PPC-Dataset (`taroz/PPC-Dataset`) 用ローダーを追加済み。`python/gnss_gpu/io/ppc.py` から `rover.obs`, `base.obs`, `base.nav`, `reference.csv`, `imu.csv` を読める。
- `exp_urbannav_baseline.py` は PPC 実データを直接読める。`--systems` で `G`, `G,J`, `G,E`, `G,E,J` を切り替え可能。
- Galileo 混合 nav については RINEX 3 の `data_sources`, `BGD E5a/E1`, `BGD E5b/E1` を parser/ephemeris 側で保持するように修正済み。
- 一次ソースベースの literature audit を `docs/literature_audit_2026-04-01.md` に追加済み。`docs/design.md` の「世界初」主張は監査結果に合わせて修正済み。
- `docs/paper_draft_2026-04-01.md` を追加済み。現時点で安全に言える abstract / intro / related work の草案をまとめた。
- `exp_urbannav_pf.py` と `exp_urbannav_pf3d.py` は PPC-Dataset を直接読めるように更新済み。`--max-epochs`, `--systems`, `--results-prefix` を追加し、PF系 summary に tail-aware 指標も保存する。
- `experiments/fetch_plateau_subset.py` を追加済み。PLATEAU の multi-GB ZIP を全部落とさず、PPC の `reference.csv` から trajectory mesh を計算して必要な `udx/bldg/*.gml` だけを HTTP range で抜き出せる。
- `experiments/fetch_urbannav_subset.py` を追加済み。UrbanNav Tokyo `Tokyo_Data.zip` から `Odaiba` / `Shinjuku` の `reference.csv`, `base.nav`, `base_trimble.obs`, `rover_*.obs`, `imu.csv` だけを HTTP range で抜き出せる。
- `experiments/fetch_urbannav_hk_subset.py` を追加済み。UrbanNav-HK-Data20190428 の GNSS tar.gz と IMU/reference zip から `rover_ublox.obs`, `base_hksc.obs`, `base.nav`, `reference.csv`, `imu.csv` を正規化配置できる。
- `python/gnss_gpu/io/urbannav.py` を拡張済み。UrbanNav run directory 判定と `load_experiment_data()` を追加し、`reference.csv`, `base.nav`, `base_trimble.obs`, `rover_ublox.obs` / `rover_trimble.obs` から PPC と同じ experiment dict を組める。
- `exp_urbannav_baseline.py`, `exp_urbannav_pf.py`, `exp_urbannav_pf3d.py` は UrbanNav 実データも自動判定できる。`--urban-rover` で `ublox` / `trimble` を切り替え可能。
- `experiments/exp_urbannav_fixed_eval.py` を追加済み。UrbanNav を fixed setting で評価し、per-run / aggregate CSV を保存できる。現在は `--isolate-methods` と `--save-epoch-errors` に対応し、full-run の epoch diagnostics を OOM なしで保存できる。
- `python/gnss_gpu/io/urbannav.py` は constellation ごとの L1-like obs code fallback を持つ。`C1C/S1C` 固定で `E/J/C` を捨てない。
- `python/gnss_gpu/ephemeris.py` は空白入り sat-id (`E 1`, `G 5`) を `E01`, `G05` に正規化する。UrbanNav trimble の 1 桁 PRN が nav と正しく結びつく。
- `python/gnss_gpu/io/plateau.py` を修正済み。real PLATEAU の一部 GML は中身が `lat/lon/h` なのに、従来 loader は plane-rect metres とみなして約 37 km ずらしていた。現在は degree 範囲を自動検出して geodetic として ECEF 化する。
- `experiments/scan_ppc_plateau_segments.py` を追加・更新済み。PPC run の mesh-boundary segment を real PLATEAU subset で走査し、segment ごとの ray-traced `NLOS fraction` を CSV に保存できる。各 segment 後に partial save し、`--stop-on-positive` にも対応。
- `experiments/exp_ppc_pf3d_residual_analysis.py` を追加済み。WLS / PF / PF3D-BVH を同一区間で再実行し、epoch・sat 単位の pseudorange residual、LOS/NLOS 判定、4D state (`x,y,z,cb`) を CSV に保存できる。
- 開発プロセス自体を experimental-convergence 型へ寄せるため、[experiments.md](/workspace/ai_coding_ws/gnss_gpu/docs/experiments.md), [decisions.md](/workspace/ai_coding_ws/gnss_gpu/docs/decisions.md), [interfaces.md](/workspace/ai_coding_ws/gnss_gpu/docs/interfaces.md) を追加済み。
- strategy 比較専用の実験領域として [pf_strategy_lab](/workspace/ai_coding_ws/gnss_gpu/experiments/pf_strategy_lab) を追加済み。ここでは `GateStrategy` 共通 interface 上で variant を比較し、readability / extensibility proxy も自動生成する。

### 1.3 PPC / UrbanNav 実データの現状

- PPC は design / ablation / holdout 用、UrbanNav は external validation 用として分離する。
- `tokyo/run1` の 300 epoch では GPS-only (`G`) が `RMS 2D = 54.59 m`、`G,E,J` は `71.77 m`。ただし `E=0.1,J=2.0` の固定重みは `58.12 m` まで改善。
- その固定重みは全 run には一般化しなかった。6 run sweep (`tokyo/nagoya × run1-3`, 各300 epoch) の平均では:
  - `G,E`: `mean RMS 2D = 51.49 m`
  - `G,E,J`: `51.61 m`
  - `G,E,J scaled (E=0.1,J=2.0)`: `60.82 m`
  - `G,J`: `63.97 m`
  - `G`: `64.27 m`
- run ごとの best config は `G,E` が 2/6, `G,E,J` が 3/6, `G` が 1/6。QZSS単独追加 (`G,J`) が最良になった run はない。
- sweep の出力は `experiments/results/ppc_wls_sweep_300_runs.csv`, `ppc_wls_sweep_300_configs.csv`, `ppc_wls_sweep_300_best.csv`。
- full-epoch でも `G,E` と `G,E,J` を全6 runで再評価済み。出力は `experiments/results/ppc_wls_sweep_ge_full_runs.csv`, `ppc_wls_sweep_ge_full_configs.csv`, `ppc_wls_sweep_ge_full_best.csv`。
- full-epoch では `p95` は概ね `74–158 m` に収まる一方、`RMS 2D` は `561–3503 m` まで跳ねる。少数の catastrophic outlier が RMS を支配しており、論文化前に外れ値率や連続スリップ区間を別指標で整理する必要がある。
- full-epoch 平均では `G,E` が `mean RMS 2D = 1221.12 m`, `mean P95 = 102.72 m`、`G,E,J` が `1080.25 m`, `102.67 m`。P95 はほぼ同等で、RMS では `G,E,J` がやや良い。
- `experiments/exp_ppc_outlier_analysis.py` を追加済み。best config を入力に、per-epoch error、100m超 epoch、連続 outlier segment を CSV に保存できる。
- full-best 解析の出力は `experiments/results/ppc_outlier_ge_full_best_summary.csv`, `ppc_outlier_ge_full_best_epochs.csv`, `ppc_outlier_ge_full_best_outliers.csv`, `ppc_outlier_ge_full_best_segments.csv`。
- catastrophic outlier は約 `20 km` 級で、`>=500 m` epoch は 329 件。その衛星数は `n_sat=4,5,6` がほとんど（`4:87, 5:130, 6:99`）。
- もっとも悪いのは `tokyo/run1 (G,E,J)` で、`outliers > 100 m = 1397 epoch`, `>=500 m = 247 epoch`, 最長連続区間は `191 epoch / 38.0 s`。`nagoya/run2` も `360 epoch` の長い outlier 区間がある。
- 標準 PF の初回 PPC 実験も開始済み。`G` 単独・300 epoch では `tokyo/run1` で `PF-10K RMS 2D = 55.60 m, P95 = 58.09 m`, `PF-100K = 55.71 m, 58.73 m`、`nagoya/run1` で `PF-10K = 57.27 m, 78.34 m`, `PF-100K = 58.03 m, 81.60 m`。少なくともこの2 run では WLS/EKF に対して RMS はわずかに悪化、P95 は run によって改善するが決定的ではない。
- その結果は `experiments/results/ppc_pf_tokyo_run1_g_300_summary.csv` と `experiments/results/ppc_pf_nagoya_run1_g_300_summary.csv` に保存済み。
- PF3D / PF3D-BVH は PPC 経路で動作確認済み。ただし現状は対応 PLATEAU モデル未接続のため synthetic building model fallback を使っている。`tokyo/run1` 100 epoch / 10K particles では `PF3D = PF3D-BVH = RMS 2D 37.87 m, P95 41.36 m` と大きく改善したが、これは論文用結果としては扱わない。出力は `experiments/results/ppc_pf3d_tokyo_run1_g_100_synthmodel_summary.csv`。
- real PLATEAU との接続も開始済み。`fetch_plateau_subset.py --run-dir /tmp/PPC-real/PPC-Dataset/tokyo/run1 --preset tokyo23 --max-rows 300` で、`tokyo/run1` 先頭区間に対応する `53394603_bldg_6697_2_op.gml` を抽出し、`/tmp/plateau_subsets/tokyo_run1_300_bldg` を作成した。
- ただし、2026-04-02 に real PLATEAU loader の座標解釈バグを修正したため、これ以前の real PLATEAU accuracy / `NLOS fraction` の結論は無効。`0.00%` 判定の主因は 3D model 側が約 37 km ずれていたことだった。旧結果は runtime の目安としてだけ扱うべき。
- loader 修正後の segment scan で、`tokyo/run1`, `systems=G`, `segment_length=100`, `mesh_radius=1` の real PLATEAU 走査を実施済み。出力は `experiments/results/ppc_plateau_segment_scan_tokyo_run1_g.csv`。`start_epoch=1463`, `mesh=53394613`, `subset_key=a5844cfc41fbda08` が最初の positive 区間で、`4 / 170` 観測が NLOS、`NLOS fraction = 2.35%`。
- その positive 区間を real PLATEAU + PF3D-BVH で再評価済み。結果は `experiments/results/ppc_pf3dbvh_tokyo_run1_seg1463_g_100_plateau_r1_summary.csv` と `..._nlos_stats.csv`。`WLS RMS 2D = 54.72 m`, `EKF = 54.69 m`, `PF-10K = 96.02 m`, `PF3D-BVH = 158.01 m`、ray-traced `NLOS fraction = 2.68%`。つまり 3D尤度は初めて実際に発火したが、現状の likelihood / bias 設定ではこの区間で明確に悪化する。
- `exp_urbannav_pf3d.py` には tuning 用の `--sigma-pr`, `--sigma-los`, `--sigma-nlos`, `--nlos-bias` を追加済み。同区間で quick sweep も実施し、`experiments/results/ppc_pf3dbvh_tokyo_run1_seg1463_g_100_param_sweep.csv` に保存した。`soft_nlos (sigma_nlos=100, bias=0)` でも `RMS 2D = 153.78 m`, `P95 = 224.48 m`、`moderate_nlos (60, 5)` でも `155.12 m`, `224.46 m` で、単に bias を弱めるだけでは改善しない。
- 旧 real PLATEAU subset 上で観測した `PF3D 1028.29 ms/epoch`, `PF3D-BVH 17.78 ms/epoch` という runtime 差自体は、systems 比較の目安としてはまだ有用。BVH の高速化効果は引き続き大きい。
- `PPCDatasetLoader.load_experiment_data` と各 experiment CLI に `start_epoch` / `start_row` 相当の offset 指定を追加済み。これで later segment を直接切り出して評価できる。
- `tokyo/run1` later segment（`start_epoch=1600`, `max_epochs=100`）について、`fetch_plateau_subset.py --mesh-radius 1` で周辺 9 mesh を含む real PLATEAU subset を抽出済み。subset は `1,490,984` triangles。
- `start_epoch=1600` の旧結果 (`experiments/results/ppc_pf3dbvh_tokyo_run1_seg1600_g_100_plateau_r1_summary.csv`) も loader 修正前のため、accuracy / `NLOS fraction` の結論としては再計測が必要。
- residual dump (`experiments/results/ppc_pf3d_residual_tokyo_run1_seg1463_g_100_summary.csv`) により、`start_epoch=1463` 区間の GT-NLOS residual は `22/22` 件すべて負だった。`truth_ref` で `NLOS mean residual = -34.55 m`, `positive_frac = 0.0`。つまり「NLOSなら正の擬似距離 excess」という仮定はこの区間では成立していない。
- 同 residual dump では、PF3D-BVH 推定位置での ray tracing が `22 TP / 0 FN / 827 FP / 11 TN` となり、LOS satellite の大半を NLOS と誤分類していた。これが hard NLOS model の正帰還を起こしている。
- CUDA kernel (`src/particle_filter/weight_3d.cu`, `weight_3d_bvh.cu`) は修正済み。`nlos_bias` は常時ではなく「residual > 0` のときだけ引く」positive-only bias に変更した。対応する low-level test も `tests/test_pf3d.py`, `tests/test_pf3d_bvh.py` に追加済み。
- その修正後、`start_epoch=1463` の real PLATEAU 再実験では `PF3D-BVH RMS 2D` が `158.01 m -> 154.96 m`、`mean residual` が `20.02 m -> 9.49 m`、`mean (pf3d_cb - pf_cb)` が `-19.00 m -> -12.32 m` に改善した。ただし根本改善には至っていない。主因は bias より hard LOS/NLOS 判定側。
- その後、hard LOS/NLOS switch を soft mixture likelihood に置き換えた。`P(NLOS | blocked)=blocked_nlos_prob`, `P(NLOS | clear)=clear_nlos_prob` を kernel / Python binding / experiment CLI に追加し、ray tracing 結果を「確率 0/1 の判定」ではなく事前分布として扱えるようにした。
- `start_epoch=1463` の quick sweep は `experiments/results/ppc_pf3dbvh_tokyo_run1_seg1463_g_100_mixture_sweep.csv` に保存済み。`hard_default (1.0, 0.0)` の `RMS 2D = 154.96 m`, `P95 = 224.44 m`, `outlier_rate = 72%` に対し、`mix_010_001` で `97.30 m`, `114.44 m`, `53%`、`mix_005_001` で `96.15 m`, `117.47 m`, `50%` まで改善した。
- best 設定 `blocked_nlos_prob=0.05`, `clear_nlos_prob=0.01` の validation は `experiments/results/ppc_pf3dbvh_tokyo_run1_seg1463_g_100_plateau_mix005_001_summary.csv` と `experiments/results/ppc_pf3d_residual_tokyo_run1_seg1463_g_100_mix005_001_summary.csv` に保存済み。`PF3D-BVH = 96.15 m` で、plain `PF-10K = 96.02 m` とほぼ同等まで戻った。hard switch の catastrophic collapse はほぼ解消。
- residual / classification も改善した。PF3D-BVH の LOS/NLOS confusion は `TP/FN/FP/TN = 22/0/827/11 -> 6/16/150/688`。false positive が大幅に減り、`pf3d_bvh` の GT-NLOS mean residual は `41.39 m -> 7.42 m`、`mean (pf3d_cb - pf_cb)` は `-12.32 m -> -9.79 m`。一方で true NLOS の取り逃し (`FN`) は増えており、現状は「3D尤度で勝つ」より「悪化しない」設定が見つかった段階。
- `fetch_plateau_subset.select_bldg_entries()` も修正済み。Nagoya PLATEAU ZIP は `udx/bldg/...` で始まり、Tokyo 側の `/.../udx/bldg/...` と path 形が違っていたため、旧コードは Nagoya の建物 GML を 0 件選んでいた。現在は両方に対応し、回帰 test は `tests/test_fetch_plateau_subset.py` に追加済み。
- soft mixture 設定を real PLATEAU の positive segment 6 本へ横展開済み。segment scan 出力は `experiments/results/ppc_plateau_segment_scan_tokyo_run2_g.csv`, `...tokyo_run3_g.csv`, `...nagoya_run1_g.csv`, `...nagoya_run2_g.csv`, `...nagoya_run3_g.csv`。最初の positive 区間は `tokyo/run2 start=808 frac=6.49%`, `tokyo/run3 start=774 frac=1.36%`, `nagoya/run1 start=0 frac=0.67%`, `nagoya/run2 start=983 frac=6.67%`, `nagoya/run3 start=235 frac=1.95%`。
- 同設定で PF3D-BVH を再評価済み。summary は `experiments/results/ppc_pf3dbvh_tokyo_run2_seg808_g_100_plateau_mix005_001_summary.csv`, `...tokyo_run3_seg774...`, `...nagoya_run1_seg0...`, `...nagoya_run2_seg983...`, `...nagoya_run3_seg235...` に保存済み。`PF3D-BVH` は 6 区間中 3 区間で `PF-10K` の `RMS 2D` を改善した。特に `nagoya/run2` は `78.76 -> 74.04 m`, `P95 88.08 -> 81.92 m`、`tokyo/run2` は `107.48 -> 106.42 m`, `133.04 -> 130.47 m`、`nagoya/run1` は `73.44 -> 71.61 m`。一方で `tokyo/run1`, `tokyo/run3`, `nagoya/run3` はほぼ同等か微悪化。
- 6 区間平均では `PF-10K RMS 2D = 81.67 m`, `PF3D-BVH = 80.48 m` で、mean 改善量は `1.20 m`。ただし `P95` 平均はほぼ横ばいで、効果は「NLOSが比較的強い区間で効きやすいが一様ではない」段階。
- `nagoya/run2 start=983` の residual dump も追加済み。出力は `experiments/results/ppc_pf3d_residual_nagoya_run2_seg983_g_100_mix005_001_summary.csv` ほか。ここでは GT-NLOS residual が `mean = +91.48 m` と強く正側で、`tokyo/run1` の `-34.55 m` と対照的。つまり Nagoya 側では NLOS bias 仮定が比較的成り立っており、これが `PF3D-BVH` 改善の一因と考えられる。
- ただし同 residual dump での final-state LOS/NLOS confusion は `TP/FN/FP/TN = 0/43/0/559` で、最終推定位置で ray tracing すると全 satellite が LOS 判定だった。改善は final-state の hard classification より、particle 群全体に対する soft prior / path pruning の寄与である可能性が高い。
- `experiments/exp_ppc_pf3d_particle_diagnostics.py` を追加済み。これは指定 epoch まで PF / PF3D-BVH を再走し、post-update 粒子群について sat-wise `blocked_frac`, residual, mixture-vs-LOS log-likelihood 差を CSV 化する。`nagoya/run2 start=983` の最大改善 epoch `63` の出力は `experiments/results/ppc_pf3d_particle_diag_nagoya_run2_seg983_epoch63_mix005_001_meta.csv` と `..._satellites.csv`。
- この particle diagnostics では、epoch 63 の surviving particles は `PF`, `PF3D-BVH` とも sampled 128 粒子すべて `blocked_frac = 0` だった。にもかかわらず centroid error は `PF 81.34 m -> PF3D-BVH 72.51 m` に改善していた。つまりこの区間の利得は「blocked ray を NLOS 扱いしたこと」ではなく、`clear_nlos_prob > 0` による clear-ray 側の heavy-tail / robustification が主因と考えられる。
- その仮説を `nagoya/run2 start=983` で ablation した。`blocked only (0.05, 0.0)` は `PF3D-BVH RMS 2D = 91.02 m, P95 = 96.44 m` まで悪化。一方で `clear only (0.0, 0.01)` は `76.22 m, 84.29 m` まで改善し、full mixture `(0.05, 0.01)` の `74.04 m, 81.92 m` にかなり近い。出力は `experiments/results/ppc_pf3dbvh_nagoya_run2_seg983_g_100_blocked005_clear000_summary.csv` と `...blocked000_clear001_summary.csv`。
- `experiments/exp_ppc_pf_ablation_sweep.py` を追加済み。positive-NLOS 6 区間 (`tokyo/run1-3`, `nagoya/run1-3`) について、`PF`, `PF+RobustClear`, `PF3D-BVH+BlockedOnly`, `PF3D-BVH+FullMix` を一括再評価し、run-wise / config-wise の CSV を `experiments/results/ppc_pf_ablation_positive6_mix005_001_runs.csv` と `..._configs.csv` に保存した。
- 4-way ablation の結論はかなり重要。`PF+RobustClear` は 6 区間平均で `81.67 -> 80.02 m`、`pf_rms_wins = 5/6`、`pf_p95_wins = 3/6` と最も安定していた。つまり reviewer 向けに一番守りやすい主張は「generic robust mixture observation model」であって、3D blocked-ray 単独ではない。
- `PF3D-BVH+BlockedOnly` は平均では `95.16 m` と悪いが、`tokyo/run2` で `107.48 -> 65.11 m`, `P95 133.04 -> 105.08`、`tokyo/run3` で `98.05 -> 53.32 m`, `109.40 -> 72.40` と大勝ちする区間がある。一方で `tokyo/run1`, `nagoya/run1-3` では大きく崩れる。つまり blocked-ray 寄与は「ある区間では非常に強いが、不安定」。
- `PF3D-BVH+FullMix` は平均 `80.48 m`, `pf_rms_wins = 3/6` で、catastrophic collapse を避けるが blocked-only の大勝ちも相殺する。`clear_nlos_prob=0.01` が全 clear ray を平坦化し、Tokyo `run2/run3` の強い blocked 効果を弱めている可能性が高い。
- `experiments/exp_ppc_pf_blocked_clear_sweep.py` を追加済み。これは selected segment に対して `PF`, `PF+RobustClear(clear=c)`, `PF3D-BVH+BlockedClear(clear=c)` をまとめて sweep し、run-wise / config-wise / best-per-segment の CSV を保存する。
- Tokyo の blocked-rich 2 区間 (`tokyo/run2 start=808`, `tokyo/run3 start=774`) について coarse sweep を実施済み。出力は `experiments/results/ppc_pf_blocked_clear_sweep_tokyo_run23_mix005_runs.csv`, `..._configs.csv`, `..._best.csv`。結果はかなり強く、`blocked_only (clear=0)` が `run2: 107.48 -> 65.11 m`, `run3: 98.05 -> 53.32 m` と大勝ちする一方、`clear=0.001` を入れた時点で `run2: 109.91 m`, `run3: 100.61 m` までほぼ PF 水準に戻る。つまり fixed full-mix は Tokyo blocked 区間では利得を壊している。
- さらに fine sweep (`clear = 0, 1e-4, 2e-4, 5e-4, 1e-3`) も実施済み。出力は `experiments/results/ppc_pf_blocked_clear_sweep_tokyo_run23_mix005_fine2_runs.csv`, `..._configs.csv`, `..._best.csv`。結論は「共存帯域なし」に近い。`PF3D-BVH+BlockedClear` は `clear=1e-4` ですでに `run2: 107.30 m`, `run3: 103.93 m` まで崩れ、`clear=2e-4` でも `107.56 m`, `99.11 m`。少なくともこの 2 区間では `clear > 0` を入れるだけで blocked-only の利得が消える。
- 一方、`PF+RobustClear` 単独は Tokyo では大きくない。best でも `run2: 107.48 -> 106.30 m (clear=5e-4)`, `run3: 98.05 -> 96.23 m (clear=1e-4)` に留まる。つまり `clear_nlos_prob` は Nagoya 系では効いても、Tokyo blocked-rich 区間では blocked-ray 寄与を食ってしまう。
- 重要な結論として、今の `blocked` と `clear` の fixed simultaneous mixture は reviewer 向けの main method にはしにくい。ここから先はパラメータ探索ではなく設計変更が必要。候補は「clear-mixture を blocked 判定と独立な robust baseline として分離して論文に出す」「map-aided blocked prior は別枝の method として出す」「あるいは clear-mixture を全衛星一律ではなく adaptive / residual-aware / blocked-aware に掛ける」。
- `experiments/exp_ppc_pf_gate_sweep.py` を追加済み。これは `PF+RobustClear` と `PF3D-BVH+BlockedOnly` を並走し、epoch ごとの blocked-evidence score が threshold を超えたときだけ blocked-only estimate を採る hybrid gate を評価する。score は sampled particles の ray-traced `mean_weighted_blocked_frac` で、`score-source` は `blocked` または `robust` を選べる。
- 最初の gate 実装 (`score-source=blocked`) は自己正当化になって失敗した。blocked-only expert 自身の particles で score を作ると Nagoya 側でも `mean_gate_score ≈ 0.99` となり、threshold を上げてもほぼ全 epoch で blocked-only を採ってしまう。`experiments/results/ppc_pf_gate_positive6_mix005_clear01_runs.csv` を参照。
- `score-source=robust` に切り替えると少しは分離する。pilot 3 区間 (`tokyo/run2`, `tokyo/run3`, `nagoya/run2`) の出力は `experiments/results/ppc_pf_gate_t23_n2_robustscore_runs.csv`, `..._configs.csv`, `..._scores.csv`。`mean_weighted_blocked_frac` は `tokyo/run2 mean=0.00686`, `tokyo/run3=0.00329`, `nagoya/run2=0.00349` で、完全分離ではないが `th=0.001` や `0.005` で部分的に Tokyo を blocked 側へ寄せられる。
- ただし full 6 区間に広げた結果、simple gate でもまだ全体勝ちにはならない。出力は `experiments/results/ppc_pf_gate_positive6_robustscore_mix005_clear01_runs.csv`, `..._configs.csv`, `..._scores.csv`。best は `th=0.05` で `mean RMS 2D = 86.80 m`, `mean P95 = 109.29 m`、`PF+RobustClear` の `80.02 m / 97.75 m` には届かない。
- 失敗理由は score の誤反応。`tokyo/run1` は `mean score = 0.13866`, `nagoya/run3` も `0.03811` と高く、ここで blocked-only が悪いのに gate が引っ張られる。実際 `th=0.05` でも `tokyo/run1` は `94.78 -> 103.10 m`, `nagoya/run3` は `35.02 -> 67.55 m` に悪化した。一方で `tokyo/run2` と `tokyo/run3` では同 threshold でもほぼ robust-only に近く、blocked の大勝ちを十分拾えていない。
- よって、単純な `mean weighted blocked fraction` gate も main method には不十分。必要なのは `blocked score` 単独ではなく、residual sign / residual magnitude / blocked persistence / expert disagreement を含む richer gate か、あるいは blocked-only を selective post-correction として別設計にすること。
- 次アクションは、`PF`, `PF+RobustClear`, `PF3D-BVH+BlockedOnly` を主軸に据えたまま、gate を「blocked score だけ」で作らないこと。具体的には `(blocked score, positive residual ratio, expert disagreement)` の 2-3変数 gate を試すか、論文主張を robust baseline と systems contribution に寄せるのが現実的。
- `experiments/exp_ppc_pf_rich_gate_search.py` を更新し、per-epoch feature dump に加えて trajectory dump も保存するようにした。これにより重い PF forward と strategy 評価を分離できる。出力例は `experiments/results/ppc_pf_rich_gate_t23_n2_v2_features.csv` と `..._trajectories.csv`。
- [evaluate_strategies.py](/workspace/ai_coding_ws/gnss_gpu/experiments/pf_strategy_lab/evaluate_strategies.py) を追加済み。共通 dump を入力に、`always_robust`, `always_blocked`, `disagreement_gate`, `clock_veto_gate`, `dual_mode_regime_gate`, `quality_veto_regime_gate`, `hysteresis_quality_veto_regime_gate`, `mode_aware_hysteresis_quality_veto_regime_gate`, `branch_aware_hysteresis_quality_veto_regime_gate`, `rescue_branch_aware_hysteresis_quality_veto_regime_gate`, `negative_exit_rescue_branch_aware_hysteresis_quality_veto_regime_gate`, `entry_veto_negative_exit_rescue_branch_aware_hysteresis_quality_veto_regime_gate`, `rule_chain_gate`, `weighted_score_gate` を同一 interface で比較できる。
- pilot 3 segment (`tokyo/run2`, `tokyo/run3`, `nagoya/run2`) の strategy lab 出力は `experiments/results/pf_strategy_lab_t23_n2_summary.csv`。pilot best は `disagreement_gate` で、`mean RMS 2D = 68.57 m`, `mean P95 = 93.42 m`, `PF wins = 3/3`。
- ただし full 6 positive segment (`tokyo/run1-3`, `nagoya/run1-3`) に広げた validation では、`disagreement_gate` は一般化しなかった。出力は `experiments/results/pf_strategy_lab_positive6_summary.csv` と `..._runs.csv`。`always_robust` は `80.02 m / 97.75 m`, `PF wins = 5/6` で安全 baseline を維持する。single-split winner は `clock_veto_gate` の `73.72 m / 96.48 m` だが holdout は通らない。current best generalizing family は `entry_veto_negative_exit_rescue_branch_aware_hysteresis_quality_veto_regime_gate` で、tuned `79.41 m`, holdout `65.54 m / 81.22 m`。
- `disagreement_gate` の失敗区間は `tokyo/run1`, `nagoya/run1`, `nagoya/run3`。`disagreement_m` 単独では blocked-rich Tokyo と clean Nagoya を分離できず、false positive が多い。`rule_chain_gate` はこれを少し抑えるが、それでも `always_robust` を超えない。
- その後、veto 型の新 variant `clock_veto_gate` も追加済み。これは `disagreement_m`, `cb_disagreement_m`, `mean_weighted_blocked_frac` だけを使う `pipeline-veto` で、tuned full 6 では `73.72 m / 96.48 m` まで改善した。
- ただし holdout 6 segment を別途切り出して再評価すると、`clock_veto_gate` は `74.02 m / 96.29 m` に留まり、`always_robust` の `66.92 m / 81.69 m` を超えなかった。holdout dump は `experiments/results/ppc_holdout_segments_r200_s200_best.csv` と `experiments/results/pf_strategy_lab_holdout6_r200_s200_summary.csv` を参照。
- holdout の baseline 平均は `PF = 67.83 m / 81.03 m`, `PF+RobustClear = 66.92 m / 81.69 m`, `PF3D-BVH+BlockedOnly = 86.98 m / 111.68 m`。つまり current gate family は tuned dump に過学習している可能性が高い。
- `experiments/scan_ppc_holdout_candidates.py` を追加済み。これは tuned positive anchor の近傍から holdout candidate を検索し、best-per-run の segment spec CSV を生成する。`exp_ppc_pf_rich_gate_search.py` も `--segment-spec-csv` 対応に更新済みで、tuned 6 以外でも同一 dump-generation pipeline を再利用できる。
- さらに [cross_validate_families.py](/workspace/ai_coding_ws/gnss_gpu/experiments/pf_strategy_lab/cross_validate_families.py) を追加済み。これは tuned dump と holdout dump の両方に対して strategy family の grid search を一括で回し、best-per-family を保存する。出力は `experiments/results/pf_strategy_family_cv_positive6_holdout6_configs.csv` と `..._family_best.csv`。
- その後、別設計の [dual_mode_regime_gate](/workspace/ai_coding_ws/gnss_gpu/experiments/pf_strategy_lab/strategies.py) も追加済み。これは `close mode` と `far mode` の 2 branch で blocked expert を使い分ける `regime-branch` family。
- さらに [quality_veto_regime_gate](/workspace/ai_coding_ws/gnss_gpu/experiments/pf_strategy_lab/strategies.py) を追加済み。これは `dual_mode_regime_gate` の close branch に `satellite_count` と `robust_p95_abs_residual` の quality veto を足した `regime-quality-veto` family。
- `quality_veto_regime_gate` の representative config (`close_satellite_max=9`, `close_p95_abs_residual_max=55`) では tuned `80.02 -> 79.81 m`, holdout `66.92 -> 65.62 m` と `always_robust` を両 split で上回る。これは current best state-free family。
- さらに [hysteresis_quality_veto_regime_gate](/workspace/ai_coding_ws/gnss_gpu/experiments/pf_strategy_lab/strategies.py) を追加済み。これは `quality_veto` の candidate を stateful に保持する `stateful-hysteresis` family で、representative config (`close_satellite_max=9`, `close_p95_abs_residual_max=55`, `enter=1`, `exit=3`) では tuned `80.02 -> 79.77 m`, holdout `66.92 -> 65.57 m`, `81.69 -> 81.22 m` を達成した。
- [mode_aware_hysteresis_quality_veto_regime_gate](/workspace/ai_coding_ws/gnss_gpu/experiments/pf_strategy_lab/strategies.py) も追加済み。これは `close` と `far` で enter persistence を分ける `stateful-branch-hysteresis` family で、representative config (`enter_close=2`, `enter_far=1`, `exit=4`) では tuned `79.63 m`, holdout `65.57 m` を達成した。
- [branch_aware_hysteresis_quality_veto_regime_gate](/workspace/ai_coding_ws/gnss_gpu/experiments/pf_strategy_lab/strategies.py) も追加済み。これは `mode_aware` に branch別 exit を足す `stateful-branch-exit-hysteresis` family で、representative config (`enter_close=2`, `enter_far=1`, `exit_close=3`, `exit_far=5`) では tuned `79.55 m`, holdout `65.58 m` を達成した。holdout-first では `hysteresis` に僅差で及ばないが、rescue 導入前の best train/holdout balance だった。
- [rescue_branch_aware_hysteresis_quality_veto_regime_gate](/workspace/ai_coding_ws/gnss_gpu/experiments/pf_strategy_lab/strategies.py) も追加済み。これは `branch_aware` に「clean な close singleton を即時採用する rescue」を足す `stateful-branch-rescue-hysteresis` family で、representative config (`enter_close=3`, `enter_far=1`, `exit_close=3`, `exit_far=5`, `rescue_sat<=8`, `rescue_p95<=50`, `rescue_cb>=16`) では tuned `79.53 m`, holdout `65.57 m / 81.22 m` を達成した。
- [negative_exit_rescue_branch_aware_hysteresis_quality_veto_regime_gate](/workspace/ai_coding_ws/gnss_gpu/experiments/pf_strategy_lab/strategies.py) も追加済み。これは `rescue_branch_aware` に active `close` の negative-evidence exit を足す `stateful-branch-rescue-negative-exit` family で、representative config (`neg_dis>=42`, `neg_cb>=25`, `neg_p95>=52`, `neg_hits=1`) では tuned `79.47 m`, holdout `65.54 m / 81.22 m` を達成した。
- [entry_veto_negative_exit_rescue_branch_aware_hysteresis_quality_veto_regime_gate](/workspace/ai_coding_ws/gnss_gpu/experiments/pf_strategy_lab/strategies.py) も追加済み。これは `negative_exit_rescue_branch_aware` に non-rescue の `close` entry veto (`close_entry_p95<=50`) を足す `stateful-branch-entry-veto-rescue-negative-exit` family で、representative config では tuned `79.41 m`, holdout `65.54 m / 81.22 m` を達成した。
- family cross-validation の current best も `entry_veto_negative_exit_rescue_branch_aware_hysteresis_quality_veto_regime_gate(close_blocked_low=0.1, close_blocked_high=0.5, close_disagreement_max=40, close_cb_max=20, close_residual_max=22, close_satellite_max=9, close_p95_abs_residual_max=55, far_blocked_max=0.01, far_positive_min=0.15, far_disagreement_min=90, far_cb_min=45, enter_confirm_close_epochs=3, enter_confirm_far_epochs=1, exit_confirm_close_epochs=3, exit_confirm_far_epochs=5, close_rescue_satellite_max=8, close_rescue_p95_abs_residual_max=50, close_rescue_cb_min_m=16, close_entry_p95_abs_residual_max_m=50, negative_exit_disagreement_min_m=42, negative_exit_cb_min_m=25, negative_exit_p95_abs_residual_min_m=52, negative_exit_hits_required=1)` に更新された。
- 最後に [sweep_entry_veto_family.py](/workspace/ai_coding_ws/gnss_gpu/experiments/pf_strategy_lab/sweep_entry_veto_family.py) を追加して current best family の近傍を narrow sweep した。出力は `experiments/results/pf_strategy_entry_veto_freeze_configs.csv` と `..._best.csv`。
- best neighbor は `exit_close=4, exit_far=6, close_entry_p95<=45..50, neg_p95>=52` で holdout `65.533 m`, tuned `79.345 m` だったが、現行 representative との差は holdout `0.009 m`, tuned `0.061 m` に留まり、promotion threshold `0.1 m` を下回った。
- つまり、現時点の repository-level decision は「pilot winner を採用する」でも「tuned full-6 winner を採用する」でもなく、「同一 dump / 同一 metrics / 同一 interface の上で variants を増やし、holdout を通り、かつ tuned で大きく落ちないものだけ残す」ことである。今の safe global baseline は `always_robust`、current best generalizing family は `entry_veto_negative_exit_rescue_branch_aware_hysteresis_quality_veto_regime_gate`、`negative_exit_rescue_branch_aware_hysteresis_quality_veto_regime_gate` は simpler negative-exit baseline、`rescue_branch_aware_hysteresis_quality_veto_regime_gate` は simpler rescue baseline、`hysteresis_quality_veto_regime_gate` は simpler stateful baseline、`quality_veto_regime_gate` は state-free seed、`dual_mode_regime_gate` は first survivor seed、`clock_veto_gate` は exploratory variant、`disagreement_gate` は local high-upside variant に留める。
- その上で、strategy gate の実装・探索フェーズはここで凍結する。今後は `always_robust` と `entry_veto_negative_exit_rescue_branch_aware_hysteresis_quality_veto_regime_gate` を固定セットとして、table / figure / limitations / paper text を詰める。
- UrbanNav external validation も開始済み。`/tmp/UrbanNav-Tokyo/Odaiba` (`6173` epoch) と `/tmp/UrbanNav-Tokyo/Shinjuku` (`10346` epoch) の `ublox`, `G` で fixed eval を実施し、clean aggregate を `experiments/results/urbannav_fixed_eval_external_g_ublox_runs.csv` と `..._summary.csv` に保存した。
- UrbanNav aggregate の `mean RMS 2D / mean P95` は `EKF = 74.56 m / 128.08 m`, `WLS = 118.85 m / 138.06 m`, `PF-10K = 134.13 m / 281.77 m`, `PF+RobustClear-10K = 136.47 m / 294.26 m`。現状の PF family は UrbanNav では accuracy claim を強めない。
- initial multi-run job では Shinjuku の `PF+RobustClear` が CUDA OOM で CPU fallback に落ちたため、clean aggregate は `Odaiba` を先頭結果から、`Shinjuku` を fresh single-run (`experiments/results/urbannav_fixed_eval_shinjuku_g_ublox_runs.csv`) から合成している。
- `trimble`, `G` の fixed eval も clean に再実行済み。aggregate は `experiments/results/urbannav_fixed_eval_external_g_trimble_summary.csv` にあり、`EKF = 79.67 m / 154.58 m`, `PF-10K = 100.52 m / 179.13 m`, `PF+RobustClear = 99.53 m / 179.00 m`, `WLS = 104.91 m / 174.08 m`。receiver quality を上げても EKF が still best。
- `experiments/results/urbannav_fixed_eval_external_g_trimble_diag_epochs.csv` から per-epoch diagnostics を作成済み。`experiments/results/urbannav_trimble_pf_vs_ekf_diagnostics.csv` と `urbannav_trimble_tail_diagnostics.csv` により、PF は `Shinjuku` で約半分の epoch では EKF と競るが、`method_bad_ekf_ok` 区間で `~180-250 m` 級の extra error を出して RMS を落とすと分かった。
- UrbanNav trimble は dataset 自体が multi-GNSS (`G/R/E/J/C`) を持つ。旧 `G` 固定は dataset limitation ではなく loader artifact だった。修正後の smoke では `Odaiba` 50 epoch で median sat `14`, `Shinjuku` で `9`、constellations は `('E','G','J')`。
- ただし `G,E,J` の common-epoch WLS 比較 (`experiments/results/urbannav_trimble_common_epoch_wls_compare.csv`) では、`p95` と `>100m率` は大きく改善する一方、`RMS` は catastrophic outlier により `Odaiba 96.24 -> 1154.16 m`, `Shinjuku 102.26 -> 1530.69 m` まで悪化した。つまり UrbanNav external の次の本命は new gate family ではなく、multi-GNSS measurement / ISB / robust-estimation stabilization。
- `experiments/exp_urbannav_multignss_stabilization.py` を追加済み。これは UrbanNav trimble の common epoch 上で `gps_only`, `multi_raw`, `residual/bias veto`, `comparative veto`, `solution-gap veto` を同一 feature dump で比較する小さい experiment lab。
- best simple family は `multi_residual_bias_veto(residual_p95<=100, residual_max<=250, bias_delta<=100, extra_sat>=2)`。出力は `experiments/results/urbannav_multignss_stabilization_trimble_gej_summary.csv` と `..._best.csv`。
- この best veto は `use_multi_frac ≈ 99.3%` のまま、common-epoch 平均を `gps_only 99.25 m / 173.38 m / 13.75% / 0.269%` から `73.49 m / 100.97 m / 4.46% / 0.046%` まで改善した。`multi_raw` の tail gain を保ちつつ catastrophic だけほぼ切れている。
- 同じ common-epoch 上の `gps_ekf_reference` は `79.88 m / 148.88 m / 10.94% / 0.000%`。つまり best veto は accuracy/tail では EKF を上回るが、integrity 最優先なら `>500m` を完全に 0 にできていない。
- run-wise でも両方改善している。`Odaiba` は `96.24 -> 75.53 m`, `165.74 -> 104.65 m`, `0.241% -> 0.056%`、`Shinjuku` は `102.26 -> 71.45 m`, `181.02 -> 97.29 m`, `0.297% -> 0.037%`。
- したがって external validation 側の次アクションは、新しい PF gate ではなく、この simple residual/bias veto を reusable な multi-GNSS WLS / EKF hook に落とし込むこと。
- その hook を [multi_gnss_quality.py](/workspace/ai_coding_ws/gnss_gpu/python/gnss_gpu/multi_gnss_quality.py) と [exp_urbannav_baseline.py](/workspace/ai_coding_ws/gnss_gpu/experiments/exp_urbannav_baseline.py) に昇格済み。`run_wls(..., quality_veto_config=...)` が使える。
- fixed external eval も `trimble + G,E,J` で再実行済み。aggregate は [urbannav_fixed_eval_external_gej_trimble_qualityveto_summary.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/urbannav_fixed_eval_external_gej_trimble_qualityveto_summary.csv) にあり、`PF+RobustClear-10K = 66.60 m / 98.53 m / 4.80% / 0.000%`、`PF-10K = 67.61 m / 101.46 m / 5.44% / 0.000%`、`EKF = 93.25 m / 178.18 m / 16.29% / 0.161%`、`WLS+QualityVeto = 2933.77 m / 175.38 m / 10.13% / 2.552%`。
- run-wise でも `PF+RobustClear-10K` が両 run で best。`Odaiba` は `61.86 m / 94.12 m / 3.54% / 0.000%`、`Shinjuku` は `71.33 m / 102.94 m / 6.06% / 0.000%`。
- したがって現時点の UrbanNav external main table は `G`-only ではなく `trimble + G,E,J` に切り替える。`WLS+QualityVeto` は promoted core utility、main external method は `PF+RobustClear-10K` と `PF-10K` の比較になる。
- [build_paper_assets.py](/workspace/ai_coding_ws/gnss_gpu/experiments/build_paper_assets.py) を追加済み。`PPC holdout`, `UrbanNav external`, `BVH systems` をまとめた [paper_main_table.md](/workspace/ai_coding_ws/gnss_gpu/experiments/results/paper_assets/paper_main_table.md) と、[paper_ppc_holdout.png](/workspace/ai_coding_ws/gnss_gpu/experiments/results/paper_assets/paper_ppc_holdout.png), [paper_urbannav_external.png](/workspace/ai_coding_ws/gnss_gpu/experiments/results/paper_assets/paper_urbannav_external.png), [paper_bvh_runtime.png](/workspace/ai_coding_ws/gnss_gpu/experiments/results/paper_assets/paper_bvh_runtime.png) を再生成できる。
- 同 builder は [paper_captions.md](/workspace/ai_coding_ws/gnss_gpu/experiments/results/paper_assets/paper_captions.md) も生成する。Table 1 と Figure 1-3 の caption、本文中での配置メモ、補助数値を 1 ファイルにまとめた。
- [exp_urbannav_window_eval.py](/workspace/ai_coding_ws/gnss_gpu/experiments/exp_urbannav_window_eval.py) を追加済み。fixed external epoch dump を 500 epoch / 250 stride の window に切って external robustness を再集計できる。結果は [urbannav_window_eval_external_gej_trimble_qualityveto_w500_s250_summary.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/urbannav_window_eval_external_gej_trimble_qualityveto_w500_s250_summary.csv) と [urbannav_window_eval_external_gej_trimble_qualityveto_w500_s250_wins.png](/workspace/ai_coding_ws/gnss_gpu/experiments/results/urbannav_window_eval_external_gej_trimble_qualityveto_w500_s250_wins.png) に保存済み。
- window 集計では `PF+RobustClear-10K` が `EKF` に対して `RMS 90/127 win`, `P95 102/127 win`, `>100m 89/127 win`, `>500m 127/127 <=` を達成した。`PF-10K` もほぼ同等で、run 平均 2 本だけに依存した external gain ではないことが確認できた。
- UrbanNav-HK-Data20190428 の GPS-only control も実施済み。結果は [urbannav_fixed_eval_hk20190428_g_ublox_summary.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/urbannav_fixed_eval_hk20190428_g_ublox_summary.csv) にあり、`EKF = 69.49 / 95.19 m` に対して `PF-10K = 301.68 / 560.12 m`, `PF+RobustClear-10K = 302.14 / 530.56 m`。つまり geography 弱点はまだ完全には解消しておらず、現在の勝ち筋は Tokyo `trimble + G,E,J` に依存する。
- 追加で PF runner に `quality_veto_config` と `EKF anchor rescue` hook を入れ、[test_urbannav_pf_stabilization.py](/workspace/ai_coding_ws/gnss_gpu/tests/test_urbannav_pf_stabilization.py) で unit test を追加した。これは experiment-only safety hook で、core にはまだ昇格しない。
- Hong Kong `G,C` fixed eval は [urbannav_fixed_eval_hk20190428_gc_rescue_v2_summary.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/urbannav_fixed_eval_hk20190428_gc_rescue_v2_summary.csv) に保存済み。raw `PF-10K` / `PF+RobustClear-10K` は `~48.6 km` 級で完全崩壊するが、`PF+EKFRescue-10K = 81.07 / 113.27 m`, `PF+RobustClear+EKFRescue-10K = 81.26 / 113.27 m` まで戻る。`>500m` はどちらも `0.000%`。
- ただし Tokyo `Odaiba` では rescue variant が悪化する。[urbannav_fixed_eval_external_gej_trimble_rescue_v2__odaiba__pf_10k_runs.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/urbannav_fixed_eval_external_gej_trimble_rescue_v2__odaiba__pf_10k_runs.csv) と [urbannav_fixed_eval_external_gej_trimble_rescue_v2__odaiba__pfplusrobustclear_10k_runs.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/urbannav_fixed_eval_external_gej_trimble_rescue_v2__odaiba__pfplusrobustclear_10k_runs.csv) に対し、rescue は [urbannav_fixed_eval_external_gej_trimble_rescue_v2__odaiba__pfplusekfrescue_10k_runs.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/urbannav_fixed_eval_external_gej_trimble_rescue_v2__odaiba__pfplusekfrescue_10k_runs.csv) と [urbannav_fixed_eval_external_gej_trimble_rescue_v2__odaiba__pfplusrobustclearplusekfrescue_10k_runs.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/urbannav_fixed_eval_external_gej_trimble_rescue_v2__odaiba__pfplusrobustclearplusekfrescue_10k_runs.csv) の通り `73 m` 台まで悪化した。したがって rescue は mainline ではなく safety variant として保持する。
- さらに guide policy も切り分けた。Tokyo `trimble + G,E,J` 3k と Hong Kong `G,C` control の比較は [urbannav_fixed_eval_external_gej_trimble_guide_policy_3k_summary.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/urbannav_fixed_eval_external_gej_trimble_guide_policy_3k_summary.csv) と [urbannav_fixed_eval_hk20190428_gc_guide_policy_summary.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/urbannav_fixed_eval_hk20190428_gc_guide_policy_summary.csv) に保存済み。
- この比較で重要だったのは、`always guide` と `init-only/fallback-only` が別 regime で勝つこと。`PF+EKFGuide-10K` は Hong Kong `G` sparse control を `66.85 / 97.45 m` まで戻す一方、Shinjuku 3k では `73.26 / 120.84 m` と悪い。逆に `PF+RobustClear+EKFGuideInit-10K` は Shinjuku 3k を `66.50 / 96.66 m` に改善するが、Hong Kong では raw PF と同じく崩壊する。
- そこで [exp_urbannav_fixed_eval.py](/workspace/ai_coding_ws/gnss_gpu/experiments/exp_urbannav_fixed_eval.py) に `PF+AdaptiveGuide-10K` を追加した。single-constellation run では `PF+EKFGuide-10K`、multi-GNSS run では `PF+RobustClear+EKFGuideInit-10K` を選ぶ simple regime split である。
- adaptive 結果は [urbannav_fixed_eval_external_gej_trimble_adaptive_3k_summary.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/urbannav_fixed_eval_external_gej_trimble_adaptive_3k_summary.csv) と [urbannav_fixed_eval_hk20190428_gc_adaptive_summary.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/urbannav_fixed_eval_hk20190428_gc_adaptive_summary.csv) にある。Tokyo 3k では `62.90 / 90.35 / 2.77%`、Hong Kong では `66.85 / 97.45 / 3.85%` で、3-run 平均は `EKF = 68.40 / 100.20 / 5.35%` に対して `PF+AdaptiveGuide-10K = 64.22 / 92.72 / 3.13%`。
- ただし main paper table を置き換えるにはまだ弱い。full Tokyo external の headline は引き続き [urbannav_fixed_eval_external_gej_trimble_qualityveto_summary.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/urbannav_fixed_eval_external_gej_trimble_qualityveto_summary.csv) の `PF+RobustClear-10K` で、adaptive guide は cross-geometry weakness を減らす supplemental variant として扱うのが安全。
- full-run confirmation も実施済み。結果は [urbannav_fixed_eval_external_gej_trimble_adaptive_full_summary.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/urbannav_fixed_eval_external_gej_trimble_adaptive_full_summary.csv) と [urbannav_fixed_eval_external_gej_trimble_adaptive_full_runs.csv](/workspace/ai_coding_ws/gnss_gpu/experiments/results/urbannav_fixed_eval_external_gej_trimble_adaptive_full_runs.csv) にある。`PF+AdaptiveGuide-10K = 67.50 / 100.78 / 4.75% / 0.000%` で `EKF` には大勝するが、current mainline `PF+RobustClear-10K = 66.60 / 98.53 / 4.80% / 0.000%` は超えない。
- run-wise では `Odaiba` がほぼ横並び (`61.68 / 94.85 / 3.14%` vs `61.86 / 94.12 / 3.54%`) だが、`Shinjuku` で `73.32 / 106.70 / 6.36%` と悪化する。したがって adaptive は mainline 置換ではなく supplemental のまま固定する。
- [paper_draft_2026-04-01.md](/workspace/ai_coding_ws/gnss_gpu/docs/paper_draft_2026-04-01.md) もこの fixed asset set に同期済み。abstract は `PF+RobustClear-10K` の UrbanNav external win と `BVH 57.8x` を反映し、Results / Discussion 草案も追加した。
- `paper_draft_2026-04-01.md` には venue-shaped section plan と caption draft も追加済み。したがって次のアクションは algorithm search ではなく、manuscript source へこの fixed asset set と caption pack を移植することである。

### 1.4 skipped 7件の内訳

| 件数 | 理由 |
|------|------|
| 2 | mpl_toolkits.mplot3d 環境依存（matplotlib 3D投影が壊れている） |
| 5 | SVGD large-scale テスト（`@pytest.mark.slow`） |

---

## 2. 全ファイル構成

```
gnss_gpu/
├── CMakeLists.txt                 ← ビルド設定（23 CUDAライブラリ + 20 pybind11モジュール）
├── pyproject.toml                 ← scikit-build-core, pip install .
├── README.md                      ← プロジェクト概要（英語）
├── LICENSE                        ← Apache-2.0
├── MANIFEST.in
├── .github/workflows/ci.yml      ← GitHub Actions CI
│
├── docs/
│   ├── design.md                  ← 設計書・研究分析（441行）
│   └── plan.md                    ← 本文書
│
├── include/gnss_gpu/              ← C++ヘッダ（17ファイル）
│   ├── cuda_check.h               ← CUDA_CHECK / CUDA_CHECK_LAST マクロ（throw on error）
│   ├── coordinates.h              ← WGS84定数、ECEF↔LLA、satellite_azel
│   ├── positioning.h              ← wls_position, wls_batch
│   ├── ekf.h                      ← EKFState, EKFConfig, ekf_initialize/predict/update/batch
│   ├── rtk.h                      ← rtk_float, rtk_float_batch, lambda_integer
│   ├── multi_gnss.h               ← wls_multi_gnss, wls_multi_gnss_batch
│   ├── doppler.h                  ← doppler_velocity, doppler_velocity_batch
│   ├── raim.h                     ← RAIMResult, raim_check, raim_fde
│   ├── particle_filter.h          ← pf_initialize/predict/weight/ess/resample/estimate
│   ├── pf_3d.h                    ← pf_weight_3d（レイトレース統合尤度）
│   ├── pf_3d_bvh.h                ← pf_weight_3d_bvh（BVH加速版）
│   ├── pf_device.h                ← PFDeviceState（GPU常駐メモリ + CUDA Streams）
│   ├── svgd.h                     ← pf_svgd_step, pf_estimate_bandwidth
│   ├── raytrace.h                 ← Triangle, raytrace_los_check, raytrace_multipath
│   ├── bvh.h                      ← AABB, BVHNode, bvh_build, raytrace_los_check_bvh
│   ├── multipath.h                ← simulate_multipath, apply_multipath_error
│   ├── skyplot.h                  ← GridQuality, compute_grid_quality, compute_sky_visibility
│   ├── acquisition.h              ← AcquisitionResult, generate_ca_code, acquire_parallel
│   ├── tracking.h                 ← ChannelState, TrackingConfig, batch_correlate, VTL
│   ├── interference.h             ← InterferenceType, compute_stft, detect/excise_interference
│   ├── ephemeris.h                ← EphemerisParams, compute_satellite_position/_batch
│   └── atmosphere.h               ← tropo_saastamoinen, iono_klobuchar, _batch versions
│
├── src/                           ← CUDAソース（23ファイル）
│   ├── positioning/wls.cu         ← n_sat<4ガード済み、ゼロ除算ガード済み
│   ├── positioning/multi_gnss.cu  ← ISB推定WLS、MAX_STATE=8
│   ├── ekf/ekf.cu                ← 8状態EKF、mat8ヘルパー、GPU batchカーネル
│   ├── rtk/rtk.cu                ← DD RTK + Schnorr-Euchner LAMBDA
│   ├── doppler/doppler.cu        ← ドップラー速度WLS
│   ├── raim/raim.cu              ← χ²検定 + FDE、gnss_gpu_coreにリンク
│   ├── particle_filter/
│   │   ├── predict.cu             ← cuRAND Philox、定速モデル
│   │   ├── weight.cu              ← shared memory衛星、log-space尤度
│   │   ├── weight_3d.cu           ← Möller-Trumbore + NLOS-aware尤度（線形探索）
│   │   ├── weight_3d_bvh.cu       ← BVH加速版weight_3d（**NEW**）
│   │   ├── resampling.cu          ← Systematic + Megopolis（double-buffer）
│   │   ├── svgd.cu               ← K=32近傍、common-mode除去、メモリリーク修正済み
│   │   └── pf_device.cu          ← GPU常駐 + CUDA Streams + pinned memory
│   ├── raytrace/
│   │   ├── raytrace.cu            ← 2フェーズmultipath（race condition修正済み）
│   │   └── bvh.cu                ← SAH構築 + GPUスタックトラバーサル
│   ├── multipath/multipath.cu
│   ├── skyplot/skyplot.cu         ← DOP計算、ecef_to_lla_inlineにリテラル定数使用
│   ├── acquisition/acquisition.cu ← cuFFT、PRN 1-32 C/Aコード
│   ├── interference/interference.cu ← STFT + ノッチフィルタ
│   ├── tracking/tracking.cu       ← __constant__ C/Aコード、ループフィルタ永続化
│   ├── ephemeris/ephemeris.cu     ← IS-GPS-200（31衛星サブnm検証済み）
│   ├── atmosphere/atmosphere.cu   ← Saastamoinen + Klobuchar
│   └── utils/coordinates.cu       ← ECEF↔LLA、satellite_azel
│
├── python/gnss_gpu/               ← Pythonパッケージ
│   ├── __init__.py                ← 全モジュールエクスポート
│   ├── _version.py                ← __version__ = "0.1.0"
│   ├── _bindings.cpp              ← core（WLS、座標変換）
│   ├── _*_bindings.cpp            ← 各モジュールのpybind11（19ファイル）
│   ├── particle_filter.py         ← ParticleFilter クラス
│   ├── particle_filter_3d.py      ← ParticleFilter3D（weight_3d使用）
│   ├── particle_filter_3d_bvh.py  ← ParticleFilter3DBVH（BVH加速、**NEW**）
│   ├── particle_filter_device.py  ← ParticleFilterDevice（GPU常駐、__del__でリークなし）
│   ├── svgd.py                    ← SVGDParticleFilter
│   ├── ekf.py                     ← EKFPositioner（_NativeState経由でnumpy in-place操作）
│   ├── rtk.py                     ← RTKSolver
│   ├── multi_gnss.py              ← MultiGNSSSolver
│   ├── doppler.py                 ← doppler_velocity, doppler_velocity_batch
│   ├── raim.py                    ← raim_check, raim_fde
│   ├── raytrace.py                ← BuildingModel（create_box, from_obj）
│   ├── bvh.py                     ← BVHAccelerator
│   ├── multipath.py               ← MultipathSimulator
│   ├── skyplot.py                 ← VulnerabilityMap
│   ├── acquisition.py             ← Acquisition
│   ├── tracking.py                ← ScalarTracker, VectorTracker
│   ├── interference.py            ← InterferenceDetector
│   ├── ephemeris.py               ← Ephemeris（CPU fallback付き）
│   ├── atmosphere.py              ← AtmosphereCorrection
│   ├── cycle_slip.py              ← detect_geometry_free, detect_melbourne_wubbena, detect_time_difference
│   ├── sbas.py                    ← SBASCorrection, QZSSAugmentation
│   ├── io/
│   │   ├── __init__.py
│   │   ├── rinex.py               ← RINEX 3.x OBS parser
│   │   ├── nav_rinex.py           ← RINEX 2/3 NAV parser
│   │   ├── nmea.py                ← NMEA GGA/RMC reader
│   │   ├── nmea_writer.py         ← NMEA GGA/RMC/GSA/GSV/VTG writer
│   │   ├── plateau.py             ← PlateauLoader（CityGML → BuildingModel）
│   │   ├── citygml.py             ← 汎用CityGMLパーサ
│   │   └── urbannav.py            ← UrbanNavLoader（**NEW**）
│   ├── viz/
│   │   ├── __init__.py
│   │   ├── plots.py               ← matplotlib 9関数
│   │   └── interactive.py         ← plotly 3関数
│   └── ros2/
│       ├── __init__.py
│       ├── gnss_node.py           ← ROS2 GNSSPositioningNode
│       ├── launch/gnss_gpu_launch.py
│       └── rviz_config.py
│
├── tests/                         ← 399+テスト（27ファイル）
│   ├── test_positioning.py, test_raytrace.py, test_multipath.py
│   ├── test_skyplot.py, test_acquisition.py, test_interference.py
│   ├── test_tracking.py, test_particle_filter.py, test_pf3d.py
│   ├── test_pf3d_bvh.py          ← **NEW**: BVH weight_3d テスト
│   ├── test_svgd.py, test_cuda_streams.py
│   ├── test_ekf.py, test_rtk.py, test_multi_gnss.py
│   ├── test_doppler.py, test_raim.py, test_atmosphere.py
│   ├── test_ephemeris.py, test_cycle_slip.py, test_sbas.py
│   ├── test_io.py, test_nmea_writer.py, test_plateau.py
│   ├── test_urbannav.py           ← **NEW**: UrbanNav loader テスト
│   ├── test_ros2_node.py, test_viz.py
│   └── test_verification.py      ← gnss_lib_py比較検証
│
├── examples/                      ← 7デモスクリプト
│   ├── demo_rinex.py, demo_acquisition.py, demo_interference.py
│   ├── demo_full_pipeline.py, demo_visualization.py
│   ├── demo_real_data.py          ← 実NAVファイル使用E2Eデモ
│   ├── demo_plateau_urban.py      ← PLATEAU+PF3D都市実験
│   └── demo_plateau_urban_results.md
│
├── experiments/                   ← 論文実験スクリプト（**NEW**）
│   ├── evaluate.py                ← 共通評価ユーティリティ
│   ├── exp_urbannav_baseline.py   ← WLS/EKFベースライン
│   ├── exp_urbannav_pf.py         ← GPU-PFスケーリング
│   ├── exp_urbannav_pf3d.py       ← 3D-aware PF
│   ├── exp_ablation_particles.py  ← パーティクル数アブレーション
│   ├── exp_ablation_svgd.py       ← SVGD vs リサンプリング
│   └── results/                   ← 結果出力先（自動作成）
│
├── benchmarks/                    ← 8ベンチマーク
│   ├── bench_all.py, bench_wls.py, bench_particle_filter.py
│   ├── bench_pf_device.py         ← **NEW**: PF Device vs 標準PF比較
│   ├── bench_acquisition.py, bench_skyplot.py, bench_raytrace.py
│   └── RESULTS.md
│
├── data/
│   ├── sample_building.obj, sample_satellites.json
│   ├── sample_plateau.gml, gps_ca_codes.py
│   └── (NAVファイル: /tmp/gnss_eph/rinex/nav/brdc0150.24n にダウンロード済み)
│
└── output/
    └── plateau_vulnerability_map.geojson
```

---

## 3. 絶対に守るべきルール

### 3.1 pybind11の配列作成
```cpp
// ❌ 絶対にダメ（全要素が[0]の値になるバグ）
auto arr = py::array_t<double>(n);

// ✅ 正しい
auto arr = py::array_t<double>(std::vector<ssize_t>{n});
```
**理由**: `py::array_t<T>(int)` は意図しないコンストラクタオーバーロードにマッチする。この修正に丸1日かかった。

### 3.2 shape validation について
**現在、bindings に shape validation は入っていない**。以前入れたが既存テスト（flat配列渡し）が100件壊れたため全て除去した。

将来 validation を入れる場合は:
- flat配列（N*3）と2D配列（N,3）の**両方**を受け付けるようにする
- `if (buf.size % 3 != 0) throw ...` のようなsize-basedチェックのみ
- ndim/shapeチェックは**しない**

### 3.3 CUDA_CHECK
全`.cu`ファイルに `#include "gnss_gpu/cuda_check.h"` が入っている。

```cpp
CUDA_CHECK(cudaMalloc(&ptr, size));  // 失敗時 std::runtime_error throw
CUDA_CHECK_LAST();                   // カーネル起動後に呼ぶ
```

**注意**: `std::runtime_error`をthrowするため、pybind11がPython `RuntimeError`に変換する。

### 3.4 GPU型番
コード・ドキュメント・コミットメッセージに**具体GPU型番（RTX 4070 Ti SUPER等）を書かない**。「Ada Lovelace世代消費者向けGPU (16GB VRAM)」と記載する。

### 3.5 Co-Authored-By
gitコミットに Co-Authored-By は**付けない**（ユーザー設定）。

### 3.6 ビルド後の.soコピー
```bash
cp build/*.so python/gnss_gpu/   # ビルド後に必ず実行
PYTHONPATH=python python3 -m pytest tests/ -q  # テスト時
```

---

## 4. 既知の問題（優先度順）

### P0: すぐ直すべき（2件のテスト失敗）

#### 4.1 test_insufficient_satellites (2件)
```
FAILED tests/test_multi_gnss.py::TestMultiGNSSCPU::test_insufficient_satellites
FAILED tests/test_raim.py::test_raim_insufficient_satellites
```
**原因**: n_sat < 4 のエッジケース。C++ `wls_position` は `-1` を返すが、Python CPU fallback は例外を投げる。テスト側の `pytest.raises` が合っていない。

**修正**: テストファイルを読み、C++ path と CPU fallback path の両方で正しいエラーハンドリングを確認・修正。

### P1: 論文化ブロッカー

#### 4.2 実データ未取得
**最大の問題**。合成データのみ。実データでの評価なしではどのトップ会議でも100%リジェクト。

#### 4.3 weight_3d_bvh.cu の大規模テスト未実施
BVH版weight_3dは実装・テスト済みだが、PLATEAU LOD2レベル（10万三角形+）での性能テストが未実施。

### P2: 技術的負債

| # | 問題 | 影響 | 備考 |
|---|------|------|------|
| D1 | `predict.cu`等の毎回cudaMalloc/Free | 性能劣化 | `ParticleFilterDevice`で解決済み。論文ではDevice版の性能を報告 |
| D2 | EKF GPU bindings | 動作するが_NativeState経由 | state構造体のpybind11コピー問題を回避済み |
| D3 | RTK solve_fixed | ambiguity共分散がハードコード | `Q_amb = np.eye(n) * 0.1` |
| D4 | 高緯度ECEF→LLA | cos(lat)→0で高度発散 | 極付近のみ |
| D5 | tracking VTL | 衛星16機超で配列オーバーフロー | S[1024]制限 |

---

## 5. 論文計画（詳細は design.md 参照）

### 5.1 タイトル
"GPU-Resident Large-Scale Particle Filtering with 3D Ray-Tracing Likelihoods for Urban GNSS Positioning"

### 5.2 守れる貢献
1. GPU常駐の大規模GNSS粒子推論を実装し、実測ランタイムを示す
2. 3D建物モデル由来のレイトレース尤度をGNSS PFへ統合する
3. clock-bias common-mode除去を含むGNSS向け粒子更新を示す
4. 実データで tail-aware 指標と catastrophic failure を解析する

### 5.3 ターゲット
- IROS 2027 (締切 2027年3月頃)
- ION GNSS+ 2026 (締切 2026年6月頃)

---

## 6. 次のアクション（Codexがやるべきこと）

### Phase 1: すぐやること（今日〜1週間）

#### 6.1 PPC catastrophic outlier の抑制
候補は `n_sat <= 5` epoch の multi-WLS回避、GPS-only fallback、RAIM/FDE連携。まず `tokyo/run1` と `nagoya/run2` を対象に、`experiments/results/ppc_outlier_ge_full_best_*.csv` を基準に改善量を測る。

#### 6.2 tail-aware 指標でベースライン再整理
`RMS` だけでは full-run の挙動を誤るため、`p95`, `>100m率`, `>500m率`, `最長failure segment` を baseline CSV に追加する。WLS/EKF/RTK-like を同じ指標で比較する。

#### 6.3 related work / claim 整理
`docs/literature_audit_2026-04-01.md` と `docs/design.md` を起点に、論文本文では個別要素の「世界初」を避ける。主張は組み合わせ・規模・実証に限定する。

### Phase 2: 実験（1〜4週間）

#### 6.4 PPCでPF / PF3D / BVH評価
`experiments/exp_urbannav_pf.py` と `experiments/exp_urbannav_pf3d.py` を PPC-Dataset に接続し、WLS tail 改善が PF 系で再現できるか確認する。

#### 6.5 PLATEAUモデル取得・統合
1. https://www.geospatial.jp/ckan/dataset?q=plateau から PPC / UrbanNav 対応エリアのLOD2取得
2. `PlateauLoader(zone=9).load_directory(path)` で読み込み
3. `experiments/exp_urbannav_pf3d.py` で3D-PF評価

#### 6.6 アブレーション実験
```bash
PYTHONPATH=python python3 experiments/exp_ablation_particles.py
PYTHONPATH=python python3 experiments/exp_ablation_svgd.py
```

### Phase 3: 論文化（4〜8週間）

#### 6.7 実験結果の整理
- `experiments/results/` にCSV保存
- CDF曲線、エラー時系列、outlier segment の図生成
- `evaluate.py` にユーティリティ関数実装済み

#### 6.8 論文ドラフト
構成案は `docs/design.md` セクション8に記載。related work は `docs/literature_audit_2026-04-01.md` の制約を守る。

---

## 7. ビルド・テスト手順

### 7.1 クリーンビルド
```bash
cd /workspace/ai_coding_ws/gnss_gpu
rm -rf build && mkdir build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=native
make -j$(nproc)
cd ..
cp build/*.so python/gnss_gpu/
```

### 7.2 テスト実行
```bash
PYTHONPATH=python python3 -m pytest tests/ -q               # 全テスト
PYTHONPATH=python python3 -m pytest tests/ -q -k "not slow"  # slow除外
PYTHONPATH=python python3 -m pytest tests/test_verification.py -v  # 検証テスト
```

### 7.3 デモ実行
```bash
PYTHONPATH=python python3 examples/demo_real_data.py         # 実NAVファイルE2E
PYTHONPATH=python python3 examples/demo_full_pipeline.py     # フルパイプライン
PYTHONPATH=python python3 examples/demo_plateau_urban.py     # PLATEAU+PF3D
```

### 7.4 ベンチマーク
```bash
PYTHONPATH=python python3 benchmarks/bench_all.py            # 全ベンチマーク
PYTHONPATH=python python3 benchmarks/bench_pf_device.py      # PF Device比較
```

### 7.5 依存関係
```
必須: CUDA 12.0+, CMake 3.18+, pybind11, Python 3.9+, NumPy
検証: gnss-lib-py (pip install gnss-lib-py) — インストール済み
可視化: matplotlib, plotly
```

---

## 8. 重要な設計判断の記録

### 8.1 SoA (Structure of Arrays)
パーティクル状態を `px[], py[], pz[], pcb[]` の4配列で保持。GPU coalesced memory access のため。

### 8.2 double精度
GNSS擬似距離は ~2.6×10^7 m。float32では1m丸め誤差。prefix-sumの累積誤差もdoubleで回避。

### 8.3 SVGD common-mode除去
GNSSの擬似距離残差にはクロックバイアスの共通成分がある。これが位置勾配にリークする問題を、全衛星の残差平均を減算して解決。`svgd.cu` L166-178。

### 8.4 EKF bindingsの回避策
`EKFState` 構造体の `double x[8], P[64]` をpybind11が正しくコピーできない問題。回避策: `_NativeState` クラスでnumpy配列として保持し、`ekf_predict`/`ekf_update`はnumpy配列をin-placeで操作する形に変更（`_ekf_bindings.cpp` L60-97）。

### 8.5 pf_device のダブルフリー修正
`unique_ptr<PFDeviceState, PFDeviceStateDeleter>` でpybind11のGC + 明示的destroyのダブルフリーを防止。`pf_device_destroy_resources()` でGPUメモリだけ解放、`delete`はpybind11に任せる。

### 8.6 BVH weight_3d
`weight_3d.cu`（線形探索 O(n_tri)）と `weight_3d_bvh.cu`（BVH O(log n_tri)）の2バージョン。少数三角形ならlinear、大規模メッシュならBVH。`ParticleFilter3DBVH` クラスで使い分け。

### 8.7 tracking.cu のC/Aコード
初期実装はハッシュベース疑似コード（テストで精度が出ない）→ 正規GPS C/Aコードを`__constant__`メモリに格納（32KB、PRN 1-32 × 1023チップ）。ホスト側で初期化→`cudaMemcpyToSymbol`。

---

## 9. 外部データ・リソース

| リソース | 場所 | 備考 |
|---------|------|------|
| GPS NAVファイル | `/tmp/gnss_eph/rinex/nav/brdc0150.24n` | 2024-01-15、gnss_lib_pyでダウンロード済み |
| PLATEAU sample | `data/sample_plateau.gml` | 東京駅付近3建物、zone=9 |
| PPC-Dataset | `/tmp/PPC-real/PPC-Dataset` | 実データ取得・6 run WLS sweep済み |
| UrbanNav | `/tmp/UrbanNav-Tokyo` | Tokyo `Odaiba` / `Shinjuku` subset 抽出済み。元ZIPは `Tokyo_Data.zip` |
| gnss_lib_py | pip installed | Stanford NavLab、検証テスト用 |

---

## 10. 性能ベンチマーク結果（最新）

**GPU**: Ada Lovelace消費者GPU (16GB VRAM), CUDA 12.0

| Module | Input | Time (ms) | Throughput |
|--------|-------|-----------|------------|
| WLS Batch | 10K epochs | 1.04 | 9.60M epoch/s |
| Particle Filter | 1M particles | 81.44 | 12.28M part/s |
| PF Device | 1M particles | **未計測** | **要ベンチマーク** |
| Acquisition | 32 PRN, 1ms | 142.50 | 224.6 PRN/s |
| Vulnerability Map | 100x100 grid | 0.62 | 16.14M pts/s |
| Ray Tracing | 1008 tri, 8 sats | 0.71 | 11.32M checks/s |

---

## 11. 研究コンテキスト（Codexが論文を書く場合に参照）

### 11.1 直接競合
| 研究 | 年/会議 | 手法 | パーティクル数 | GPU | 限界 |
|------|---------|------|-------------|-----|------|
| MegaParticles (Koide) | ICRA 2024 | SVGD PF, LiDAR | 1M | ✅ | GNSSなし |
| Suzuki | ICRA 2024 | Multiple Update PF, GNSS | ~2K | ❌ | CPU、少パーティクル |
| Groves (UCL) | 複数 | Shadow Matching | - | ❌ | グリッドベース |
| Hsu & Wen (PolyU) | T-ITS 2021 | FGO + 3DMA | - | ❌ | CPU |
| Neamati (Stanford) | ION GNSS+ 2024 | Neural City Maps | - | ✅ | NeRF、計算コスト |

### 11.2 主張すべきポイント
1. 「既存のGNSS PF / 3DMA GNSS / shadow matching を踏まえた上で、GPU常駐の大規模GNSS PF を構築した」
2. 「3D建物モデル由来のLOS/NLOSレイトレース尤度を粒子ごとに評価する実装を統合した」
3. 「SVGD/粒子更新をGNSS状態空間向けに整理し、common-mode除去を明示した」
4. 「RMSだけでなく p95, outlier rate, catastrophic rate, 最長failure segment で頑健性を評価する」

### 11.3 レビュアー対策
- 「MegaParticlesの単純適用」→ GNSS固有の工夫（クロックバイアス、3D尤度、スコア関数）を強調
- 「新規性主張が強すぎる」→ 個別要素の「初」は避け、組み合わせ・規模・実証に主張を限定
- 「実データがない」→ PPC-Dataset で一次評価は解決済み。次は UrbanNav 追加か、PPC の full-epoch / RTK / PF3D 評価を進める。
- 「計算コスト大」→ 消費者GPUで12Hz、Jetsonベンチ追加

---

## 12. よくある作業パターン

### 12.1 新CUDAモジュール追加
1. `include/gnss_gpu/foo.h` にヘッダ作成
2. `src/foo/foo.cu` に実装（`#include "gnss_gpu/cuda_check.h"` 忘れない）
3. `python/gnss_gpu/_foo_bindings.cpp` にpybind11
4. `python/gnss_gpu/foo.py` にPythonラッパー
5. `CMakeLists.txt` にライブラリ + bindingsターゲット追加
6. `python/gnss_gpu/__init__.py` にimport追加
7. `tests/test_foo.py` にテスト
8. `rm -rf build && mkdir build && cd build && cmake .. -DCMAKE_CUDA_ARCHITECTURES=native && make -j$(nproc) && cd .. && cp build/*.so python/gnss_gpu/`
9. `PYTHONPATH=python python3 -m pytest tests/test_foo.py -v`

### 12.2 テスト失敗のデバッグ
```bash
# 1. 失敗の詳細確認
PYTHONPATH=python python3 -m pytest tests/test_xxx.py::TestClass::test_method --tb=long

# 2. Python直接実行でデバッグ
PYTHONPATH=python python3 -c "from gnss_gpu import ...; ..."

# 3. CUDAエラーの場合
# CUDA_CHECK がthrowするので、Pythonの RuntimeError メッセージにCUDAエラー名が含まれる
```

### 12.3 コミット
```bash
git add -A
git commit -m "説明的なメッセージ"
git push
```
Co-Authored-Byは付けない。GPU型番を書かない。
