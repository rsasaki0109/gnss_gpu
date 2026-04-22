# gnss_gpu 引き継ぎメモ

**最終更新**: 2026-04-21 JST
**現在の HEAD**: `7952456` (`feature/carrier-phase-imu`, origin 同期)
**ブランチ**: `feature/carrier-phase-imu`
**作業ツリー**: clean (未追跡の `.codex_handoff*.md` を除く、gitignore 済)
**PR #4**: CLOSED (not merged、2026-04-16)。現在の 28+ commits はどの PR にも入っていない

**北極星目標 (2026-04-19 設定)**:
**A Continuous-Time Rao-Blackwellized Particle Filter with Factor Graph Optimization** (CT-RBPF-FGO)

- **CT**: B-spline trajectory (control points)、任意時刻で (R, p, v, a) を解析的に query (参考: https://qiita.com/NaokiAkai/items/dc77f8dd7fb514a75add)
- **RB**: per-particle velocity を KF で marginalize (Doppler observation を per-particle linear-Gaussian update)
- **PF**: per-particle NLOS rejection (satellite LOS/NLOS 判定を各 particle の hypothesized position で独立実施)
- **FGO**: weak-DD window で two-step FGO overlay (velocity FGO → position+TDCP FGO、太郎式 https://github.com/taroz/gsdc2023)

段階目標: Odaiba SMTH P50 < 1.00m (現 1.14m から)

**FGO**: メイン engine には使わない。ただし **PF の weak-DD window だけ局所 FGO** で救うハイブリッドは OK (2026-04-18 緩和)。

## 0. 現状サマリ (2026-04-21)

- **Best Odaiba SMTH P50 = 1.14m** (preset `odaiba_best_accuracy`、200K + anchor σ 0.15 + stop_sigma 0.1 + guarded tail guard)
- **Submeter (<1m) は未達**。2026-04-17〜21 の 10 セッション (codex4-14) で多数の algo/architecture 試行、いずれも 1.14m を超えられず
- **proper RBPF (codex14) の subset テストでのみ 0.89m (submeter) を達成**、full Odaiba では 1.20m 付近
- MAP-collapse 問題は観測共通のため、FGO/LAMBDA の単独導入では解消しないことが empirical に判明

---

## 0. 最初に読む順

1. 本ファイルの **§1 現在の要約** と **§10 次にやるべきこと**
2. `internal_docs/pf_smoother_api.md`
3. `experiments/exp_pf_smoother_eval.py` (UrbanNav 主戦場)
4. `experiments/exp_gsdc2023_pf.py` (Kaggle GSDC 評価)
5. `experiments/exp_gsdc2023_submission.py` (Kaggle submission 生成)
6. `tests/test_exp_pf_smoother_eval.py` (18 tests, 全 pass)

---

## 1. 現在の要約

### 1.1 headline numbers

#### UrbanNav Tokyo Odaiba (dual-frequency Trimble, submeter 挑戦本戦場)

| preset / 手法 | particles | SMTH P50 | SMTH RMS | 備考 |
|---|---:|---:|---:|---|
| `odaiba_reference` (100K baseline) | 100K | 1.38 | 5.08 | 以前の headline |
| `odaiba_reference` (stop_sigma 昇格後) | 100K | 1.34 | 4.11 | 2026-04-17 改善 |
| `odaiba_best_accuracy` | 200K | **1.14** | **4.36** | **current best** (2026-04-17〜) |
| proper RBPF (subset 3k) | 200K | 0.89 | — | 部分区間のみ submeter 達成 |
| proper RBPF (full Odaiba) | 200K | 1.20 | 4.29 | baseline 超え無し、subset 以外で悪化相殺 |
| RTKLIB demo5 (reference) | — | 4.20 | 13.08 | 外部 baseline |

#### UrbanNav Tokyo Shinjuku (cross-site 検証)

| config | FWD P50 | FWD RMS | SMTH P50 | SMTH RMS |
|---|---:|---:|---:|---:|
| odaiba_reference (0.25 old floor) | 2.63 | 10.18 | 2.58 | 9.93 |
| odaiba_reference (0.18 new floor) | 2.53 | 6.41 | 2.61 | 6.87 |
| odaiba_best_accuracy (200K + stop_sigma) | 2.49 | 7.06 | 2.29 | 7.55 |

#### UrbanNav HK (supplemental, single-frequency ublox)

| Method | P50 | RMS | >100m |
|---|---:|---:|---:|
| RTKLIB demo5 | 16.18m | 26.80m | 0.2% |
| **PF 100K** | **14.21m** | **22.53m** | **0%** |

#### Kaggle GSDC 2023 (supplemental, smartphone)

| Version | 手法 | Public Score |
|---|---|---:|
| **v3 (best)** | PF + smoother | **4.128m** |
| v1 | pseudorange only | 4.207m |
| v2 | + TDCP + Hatch | 10.150m (悪化) |
| 1st place (参考) | FGO+TDCP+DGNSS | 0.789m |

### 1.2 PF が勝つ環境 vs 負ける環境

| 環境 | PF vs Baseline | 理由 |
|---|---|---|
| **Urban canyon (UrbanNav)** | **PF 圧勝** | NLOS outlier を temporal filtering で排除 |
| Open-sky smartphone (GSDC) | WLS が勝つ | NLOS 少ない → 時間平滑化の恩恵なし |
| Extreme urban (HK TST/Whampoa) | 両方壊滅 | SPP 自体が >300m → PF でも救えない |

### 1.3 1m 切りの状況

DD+IMU の両方が効く 7099 epoch (58%) は **P50=1.107m** で 1m に近い。
全体 P50=1.36m を引き上げているのは:
- DD pairs 少ない区間 (epoch 2445-4890, base station coverage の穴)
- DD pairs≥17 のエポックは **P50=0.899m (1m 切り達成)**

**構造的限界**: base station coverage 改善 or TDCP predict 改善が必要だが、どちらもデータ/前処理の制約。

---

## 2. 実装済み技術スタック (全体)

### 2.1 CUDA カーネル

| カーネル | ファイル | 機能 |
|---|---|---|
| pf_device_position_update | pf_device.cu | SPP soft constraint |
| pf_device_shift_clock_bias | pf_device.cu | per-epoch cb re-centering |
| DD pseudorange weight | pf_device.cu | base station DD PR update |
| DD carrier AFV weight | pf_device.cu | base station DD carrier update |
| spread stat | pf_device.cu | particle spread 計測 |

### 2.2 Python API (particle_filter_device.py)

| メソッド | 機能 |
|---|---|
| position_update() | SPP position-domain soft constraint |
| correct_clock_bias() | per-epoch cb correction |
| shift_clock_bias() | 低レベル cb shift |
| update_carrier_afv() | DD carrier AFV weight update |
| enable_smoothing() / store_epoch() / smooth() | forward-backward smoother |

### 2.3 DD (Double Difference) スタック

| モジュール | ファイル | 機能 |
|---|---|---|
| DD pseudorange | dd_pseudorange.py | base station DD PR 計算 |
| DD carrier AFV | dd_carrier.py | base station DD carrier phase |
| DD quality gate | dd_quality.py | adaptive threshold + ESS/spread scaling |

### 2.4 gnssplusplus API 拡張

CorrectedMeasurement に追加: prn, carrier_phase, doppler, snr, satellite_velocity, clock_drift

### 2.5 観測スタック効果 (Odaiba)

| 手法 | 効果 | 状態 |
|---|---|---|
| DD carrier AFV | P50 1.65→1.38m | 実装済み、主力 |
| DD pseudorange | RMS 改善 | 実装済み |
| Forward-backward smoother | RMS 5.04→4.81m | 実装済み |
| IMU stop-detection | P50 1.38→1.36m, RMS 5.08→4.11m | 実装済み、現best |
| cb_correct | HK で必須 (168→22m) | 実装済み |
| position_update | P50 4.5→1.65m | 実装済み |
| Doppler velocity | P50 -0.05m | 実装済み |
| Elevation/SNR weighting | P95 改善 | 実装済み |
| RAIM satellite exclusion | P95 改善 (HK) | 実装済み |

---

## 3. 正直なネガティブ結果 (全部)

### 3.1 Odaiba (1.14m 超えを目指した試行、全て 1.14m を超えられず)

| 手法 | 実装担当 | 結果 | 原因 |
|---|---|---|---|
| Huber DD soft downweight | codex5 | 1.22m | 既存 adaptive gate が binary 版 Huber 相当、上積み無し |
| OSM road constraint (soft) | codex6 | 1.30m | 2D road は urban で wrong match、Odaiba でも HK 同様悪化 |
| Local FGO (window 2400:3500) | codex7 | 1.14m (同値) | PF smoother と同じ MAP 解に収束、factor error は削減するが GT 改善なし |
| LAMBDA L1 integer fix | codex8 | 1.14m (同値) | partial fix 22 seg / 1093 obs、weak-DD 区間で fixable 少ない |
| Widelane (region-less) | codex9 | 1.83m (悪化) | WL fix rate 高い (95%) が weak-DD で wrong fix 入って全体悪化 |
| Widelane region-aware gate | codex10 | 1.29m | DD pairs / ratio gate 効くが baseline 未満 |
| Per-particle hard NLOS reject | codex11 | 64m (壊滅) | reject に penalty 無く particle 漂流、density 崩壊 |
| Per-particle Huber soft | codex12 | 1.21m (k=1.5 最良) | 既存 adaptive gate と等価、上積み無し |
| Naive sampled velocity RBPF | codex13 | 1.22m | state 7D 化で curse of dimensionality、200K で密度不足 |
| **Proper RBPF (velocity KF marginalize)** | codex14 | **1.20m full / 0.89m subset** | 局所的に効くが full で相殺、region-aware 化の余地あり |

### 3.2 過去のネガティブ (引き続き NG)

| 手法 | データ | 結果 | 原因 |
|---|---|---|---|
| Student's t likelihood | 全データ | 悪化 | urban canyon で Gaussian が安定、重い尻尾で情報不足 |
| RTK carrier phase (integer) | Odaiba | 改善なし | NLOS で integer fix 不可 |
| Float carrier phase | HK | 効果なし | single-freq + NLOS で ambiguity 収束せず |
| OSM map constraint | HK | 悪化 | wrong road matching |
| DGNSS (NOAA CORS) on GSDC | GSDC | 改善なし | coverage 不足、daily 30s RINEX では GSDC 1Hz rover に対応しきれない |
| Hatch filter | Odaiba | 悪化 | urban canyon で carrier phase 途切れ diverge |
| TDCP predict | Odaiba | IMU に負ける | IMU (wheel+gyro) の方が高精度 |
| DD PR base interpolation | Odaiba | RMS 暴発 | 1Hz→10Hz 補間の品質が低い |
| 1M + small sigma_pos | Odaiba | 崩壊 | particle depletion (sp<1) |
| DD gate 緩和 | Odaiba | 悪化 | 品質の悪い pair を通すと P50 悪化 |
| sigma_pos < 1.0 (100K) | Odaiba | P50 悪化 | predict noise 不足 |
| 500K particles | Odaiba | 1.40 (悪化) | 200K が sweet spot、過多は Smoother で overshoot |
| tracked fallback preference | Odaiba weak-DD | 悪化 | coverage hole で fallback 品質不足 |
| ESS-only weak-DD replacement | Odaiba | 悪化 | 過度に fallback |
| Robust WLS (Huber) for GSDC | Kaggle | P50 -0.0004 (実質ゼロ) | WLS 既に良く Huber の余地少ない |
| TDCP (GSDC smartphone) | Kaggle | 30.76m | smartphone ADR 品質低 |
| Hatch filter (GSDC) | Kaggle | 10.15m | 頻繁 cycle slip |
| Carrier phase smoothing (GSDC) | Kaggle | 全悪化 | smartphone carrier 信頼不可 |

---

## 4. Kaggle GSDC 2023 詳細

### 4.1 データ

- train: 80+ run × 複数 phone (146 run/phone 組み合わせ)
- test: 40 trips
- スマホ: Pixel 4/4XL/5/6Pro/7Pro, Samsung
- 環境: Mountain View, San Jose, LA — 郊外/highway

### 4.2 Train 評価結果

| | Mean P50 | Median P50 | Mean RMS |
|---|---:|---:|---:|
| WLS (Android) | **2.62m** | **2.42m** | **5.14m** |
| PF-100K | 2.83m | 2.62m | 5.36m |

PF wins: 21% (P50), 26% (RMS)

### 4.3 Test submission 結果

| Version | Public | Private | 手法 |
|---|---:|---:|---|
| v1 | 4.207m | **5.144m** | pseudorange only |
| v3 | 4.128m | — | + smoother |
| v2 | 10.150m | — | + TDCP + Hatch (悪化) |
| v11 | 4.223m | 5.255m | reset-safe segmented smoother |
| v12 | 4.133m | 5.242m | reset-safe smoother-only |
| v13 | 4.117m | 5.268m | reset-safe smoother-only + Gaussian backward |
| v15 | 4.116m | 5.268m | reset-safe smoother-only + Gaussian backward + alpha 0.45 |
| **v22** | **4.112m** | 5.200m | shared TDCP soft-only, no TDCP predict, ultra-conservative gates |

### 4.4 なぜ GSDC で PF が勝てないか

1. **スマホの pseudorange ノイズが大きい** (15-20m std) → PF の PR update の情報量が少ない
2. **WLS が既に良い** (Google 最適化済み) → temporal filtering の余地が少ない
3. **PF の predict noise (sigma_pos=10)** が邪魔 → open-sky ではノイズを足すだけ
4. **carrier phase 自体が悪いのではなく coupling が悪かった** → `TDCP predict + Hatch` は悪化したが、shared TDCP を `soft-only` で厳しく gate すると改善余地が残る
5. **smoother が divergence reset をまたぐと hidden/private で壊れる** → reset-safe segmentation で private は 5.255m まで回復
6. **TDCP/Hatch の direct coupling が public 悪化の主因** → reset-safe smoother-only (`v12`) では `4.133m / 5.242m` まで戻り、public best `4.128m` にかなり近づいた
7. **backward smoother の実装差も効く** → `Gaussian + current-step transition` に寄せた `v13` で public は `4.117m` まで改善
8. **blend weight も数 mm 単位で効く** → `alpha=0.45` の `v15` で public は `4.116m` に微改善、private は `5.268m` で据え置き
9. **shared TDCP を predict に入れず soft-only factor 風に使うとさらに改善** → `v22` は `4.112m / 5.200m` で public best 更新、private も `v15` より改善

### 4.5 ファイル

- `experiments/exp_gsdc2023_pf.py` — 全 train 評価 (146 run)
- `experiments/exp_gsdc2023_submission.py` — test submission 生成
- `experiments/results/gsdc2023_eval.csv` — train 評価結果
- `experiments/results/gsdc2023_submission.csv` — v1 submission
- `experiments/results/gsdc2023_submission_v2.csv` — v2 submission
- `experiments/results/gsdc2023_submission_v3.csv` — v3 submission
- `experiments/results/gsdc2023_submission_v12.csv` — reset-safe smoother-only submission
- `experiments/results/gsdc2023_submission_v13.csv` — reset-safe smoother-only + Gaussian backward
- `experiments/results/gsdc2023_submission_v15.csv` — reset-safe smoother-only + Gaussian backward + alpha 0.45
- `experiments/results/gsdc2023_submission_v22.csv` — shared TDCP soft-only + strict gates, current public best

---

## 5. UrbanNav 詳細

### 5.1 frozen presets

| Preset | particles | P50 | RMS | 用途 |
|---|---:|---:|---:|---|
| odaiba_reference | 100K | 1.34 | 4.11 | DD floor `0.18` + stop detection 入り、smoother-first baseline |
| odaiba_stop_detect | 100K | 1.36 | 4.11 | legacy stop-detection comparison、DD floor `0.25`、forward-stable |
| odaiba_reference_guarded | 100K | 1.38 | 5.36 | low-ESS tail guard + DD floor `0.18`、weak tail 対策 |
| **odaiba_best_accuracy** | **200K** | **1.14** | **4.36** | **current best**: guarded base + stop_sigma + carrier-anchor σ 0.15 |

新 preset は 2026-04-17 追加。details:
```
--runs Odaiba --n-particles 200000 --sigma-pos 1.2 --position-update-sigma 1.9
--predict-guide imu --imu-tight-coupling --imu-stop-sigma-pos 0.1
--residual-downweight --pr-accel-downweight --smoother
--dd-pseudorange --dd-pseudorange-sigma 0.5 (+ adaptive floor 4.0 / mad 3.0 / ess 0.9-1.1)
--mupf-dd --mupf-dd-sigma-cycles 0.20 --mupf-dd-base-interp
--mupf-dd-gate-adaptive-floor-cycles 0.18 --mupf-dd-gate-adaptive-mad-mult 3.0
--mupf-dd-skip-low-support-ess-ratio 0.01 --mupf-dd-skip-low-support-max-pairs 4
--mupf-dd-fallback-undiff --mupf-dd-fallback-sigma-cycles 0.10
--carrier-anchor --carrier-anchor-sigma-m 0.15 --carrier-anchor-max-residual-m 0.80
--smoother-tail-guard-ess-max-ratio 0.001 --smoother-tail-guard-min-shift-m 4.0
```

### 5.2 DD carrier 統計 (Odaiba, 100K)

- DD-AFV used: 11208/12252 (91.5%)
- DD skip: 1044 (8.5%)
  - gate epoch_skip: 292
  - support_skip: 261
  - undiff fallback: 766
  - carrier anchor: 3
- DD pseudorange used: 1214/12252 (10%)
- IMU used: 12251/12252 (100%, stop detect 込み)
- Stop detect: 4177 epochs

### 5.3 エポック別診断

| 条件 | エポック数 | P50 |
|---|---:|---:|
| DD=yes + IMU=yes | 7099 (58%) | **1.107m** |
| DD=yes + IMU=no | 4109 (34%) | 3.883m |
| DD=no + IMU=yes | 951 (8%) | 3.840m |
| DD=no + IMU=no | 69 (0.6%) | 8.291m |

DD pair 数と P50:
- pairs≥17: **P50=0.899m**
- pairs=14: P50=1.191m
- pairs=10: P50=1.418m
- pairs=0: P50=3.854m

worst epoch は TOW 273836-274261 に集中 (NLOS 区間)。DD=yes でも 20m 級誤差。

### 5.4 試行した改善と結果

| 手法 | P50 | RMS | 結果 |
|---|---:|---:|---|
| frozen baseline (sp=1.2) | 1.38m | 5.08m | baseline (2026-04-14 rerun) |
| + stop detect (σ=0.1) | 1.36m | 4.11m | P50/RMS 改善 ✅ |
| + sp=1.0 | 1.42m | 4.71m | P50 悪化 |
| + sp=0.8 | 1.50m | 4.64m | P50 悪化 |
| support skip 緩和 (max-pairs=2) | 1.37m | 5.14m | P50 微改善 |
| + smoother tail guard (ESS≤0.001, shift≥4m) | 1.38m | 5.43m | full Odaiba では悪化 |
| DD gate 緩和 | 1.82m | — | 悪化 |
| TDCP predict | 1.92m | — | IMU に負ける |
| DD PR base interpolation | — | 11.05m | RMS 暴発 |
| Doppler PU sigma=1.5 | 1.74m | 4.93m | RMS 改善のみ |

2026-04-17 追記:
- `odaiba_reference` と `odaiba_reference_guarded` は `--mupf-dd-gate-adaptive-floor-cycles 0.18` を採用した。full Odaiba で reference は `SMTH RMS 5.08m -> 5.02m`、guarded は `5.43m -> 5.36m`。
- `odaiba_stop_detect` は `0.25` のまま固定する。`0.18` は `SMTH RMS=4.11m` を維持したが、forward が `P50 1.19m / RMS 4.57m` から `P50 1.63m / RMS 5.50m` へ悪化した。
- weak-DD 調査で追加した ESS replacement / low-support max-spread / low-ESS epoch-median gate の追加 knob は default-off の ablation surface として残す。preset に昇格した変更は reference/guarded の adaptive floor だけ。
- B-2 coverage-hole diagnostics で `epoch 2445-4890` 近傍を再確認。問題は DD pair 数 0 自体ではなく、
  DD-PR 不在 + high-support DD carrier の stationary/near-stationary collapse。`--imu-stop-sigma-pos 0.1` は
  full Odaiba `SMTH P50 1.38m -> 1.34m`、`SMTH RMS 5.02m -> 4.11m` に改善したため
  `odaiba_reference` preset に昇格した。

---

## 6. HK 詳細

### 6.1 clock bias 問題と解決

- ublox cb ≈ -960,000m (drift +65 m/s) → PF の random walk で追従不能
- correct_clock_bias() で per-epoch re-centering → 168m→22m
- Trimble cb ≈ -99,000m (drift ~6 m/s) → PF で自然に追従可能

### 6.2 best config

| Config | P50 | P95 | RMS |
|---|---:|---:|---:|
| RTKLIB demo5 | 16.18m | 60.85m | 26.80m |
| SPP | 15.27m | 43.72m | 23.71m |
| PF + cb + el20 + RAIM + Dop | 14.21m | 41.60m | 22.53m |

---

## 7. CI/テスト状態

### 7.1 CI (GitHub Actions)

- **lint**: pass (ruff, F841/E741 修正済み)
- **build-cuda**: pass
- **test-python**: pass (CUDA/gnssplusplus 依存テストは ignore)

### 7.2 ローカルテスト

```bash
PYTHONPATH=python python3 -m pytest tests/test_exp_pf_smoother_eval.py -q
# 25 passed (2026-04-17 rerun)
```

```bash
ruff check experiments/exp_pf_smoother_eval.py tests/test_exp_pf_smoother_eval.py --ignore=E402,E501,F401
# All checks passed
```

### 7.3 frozen reference 再現

```bash
PYTHONPATH="python:third_party/gnssplusplus/build/python:third_party/gnssplusplus/python" \
python3 experiments/exp_pf_smoother_eval.py --data-root /tmp/UrbanNav-Tokyo --preset odaiba_reference
# SMTH P50=1.38m RMS=5.02m
```

---

## 8. データセット場所

| データ | パス | 受信機 | 用途 |
|---|---|---|---|
| Odaiba | /tmp/UrbanNav-Tokyo/Odaiba | Trimble (L1+L2+L5) | headline |
| Shinjuku | /tmp/UrbanNav-Tokyo/Shinjuku | Trimble (L1+L2+L5) | headline |
| HK-20190428 | /tmp/UrbanNav-HK/HK_20190428 | ublox M8 (L1) | supplemental |
| GSDC 2023 | /tmp/gsdc_data/gsdc2023/sdc2023/ | Pixel etc (L1+L5) | supplemental |

---

## 9. ビルド

```bash
# gnss_gpu CUDA
cd build && make -j$(nproc)
cp build/python/gnss_gpu/_gnss_gpu_pf_device.cpython-312-x86_64-linux-gnu.so python/gnss_gpu/

# gnssplusplus
cd third_party/gnssplusplus/build && cmake --build . -j$(nproc)

# テスト
PYTHONPATH=python python3 -m pytest tests/test_exp_pf_smoother_eval.py -q
```

---

## 10. 次にやるべきこと (2026-04-21 大幅更新)

### 10.1 submeter 突破の優先順位

現状 1.14m、submeter (<1m) 未達。codex14 の proper RBPF が **subset で 0.89m 達成**しており、これを full Odaiba に拡張できるかが鍵。

#### AAA (最優先): Region-aware proper RBPF
- codex14 の proper RBPF は 3k subset で 0.89m を達成、full で 1.20m (相殺悪化)
- 強い DD 区間でのみ Doppler KF update を有効化する region gate を追加
- gate 候補: DD pair 数 ≥ N、ESS ≥ threshold、Doppler residual median ≤ threshold
- 実装: `python/gnss_gpu/particle_filter_device.py` の Doppler KF hook に epoch gate を追加
- CLI: `--rbpf-velocity-kf-gate-min-dd-pairs 15`, `--rbpf-velocity-kf-gate-min-ess-ratio 0.02` など
- subset 成功の再現と full 展開を検証

#### BBB: Phase 3 — Continuous-Time B-spline trajectory
- B-spline control points で軌道を連続化、IMU 残差は spline 解析微分で計算
- GNSS 観測は観測時刻で評価 (epoch snap 不要)
- 参考: https://qiita.com/NaokiAkai/items/dc77f8dd7fb514a75add
- 工数大、PF/FGO 両方に効く、北極星 CT 層

#### CCC: Phase 4 — Two-step FGO overlay (太郎式)
- 既存 `python/gnss_gpu/local_fgo.py` を 2 段階に分解
  1. velocity-first FGO (Doppler + IMU only)
  2. position FGO + TDCP を state-to-state constraint として (velocity は loose prior)
- 現在の single joint FGO と構造が違う、submeter を目指す
- 参考: Suzuki 2023 Sensors https://www.mdpi.com/1424-8220/23/3/1205, https://github.com/taroz/gsdc2023

#### 既に試し済み (再試行しない)
- 上記 §3.1 表を参照。Huber / OSM / local FGO single-joint / LAMBDA L1 / widelane (region-less/aware) / per-particle hard NLOS / per-particle Huber / naive sampled velocity は全て 1.14m 超え不可

### 10.2 既存 PF 機構の改善 (続けるなら小改善余地)

1.14m から届かないが、以下の細部チューニングは試す価値あり:
- carrier anchor sigma の re-sweep (0.15 が現 best、0.12-0.18 の fine sweep)
- smoother tail guard の threshold 再調整
- IMU stop detection の robustness 改善 (信号停止判定精度)
- sigma_doppler / Q_v sweep (proper RBPF 文脈で、region-aware 前提)

### 10.3 GSDC 改善の方向 (副次目標)

- **DGNSS (高 rate CORS)** — 実装検討した (codex F)、daily 30s では不足。1Hz / high-rate source 取得から必要
- **Robust WLS (Huber)** — 試済、negligible 差
- carrier phase は smartphone では使えない

### 10.4 論文/artifact 整備 (完結させるなら)

- README は 1.34m headline のまま、更新されていない → 1.14m (best_accuracy) へ更新要
- 全 10+ negative 結果を supplemental として整理 (この plan.md §3 をベースに)
- 北極星 CT-RBPF-FGO は論文の future work として記載
- PR は feature/carrier-phase-imu → main が unrelated history (main と独立)、要判断

### 10.5 PR の扱い

- PR #4 は CLOSED かつ not merged (2026-04-16)
- 2026-04-17 以降の 28+ commit はどの PR にも属さない
- main と feature/carrier-phase-imu は共通祖先なしの独立履歴 — PR 作るには base 判断必要

---

## 11. 重要ファイル一覧

### 11.1 CUDA コア
- `src/particle_filter/pf_device.cu`
- `include/gnss_gpu/pf_device.h`
- `python/gnss_gpu/_pf_device_bindings.cpp`

### 11.2 Python API
- `python/gnss_gpu/particle_filter_device.py`
- `python/gnss_gpu/imu.py`
- `python/gnss_gpu/dd_pseudorange.py`
- `python/gnss_gpu/dd_carrier.py`
- `python/gnss_gpu/dd_quality.py`
- `python/gnss_gpu/tdcp_velocity.py`

### 11.3 gnssplusplus (submodule)
- `third_party/gnssplusplus/` (feature/expose-corrected-pseudoranges)

### 11.4 実験スクリプト
- `experiments/exp_pf_smoother_eval.py` — UrbanNav 主戦場 (preset 対応)
- `experiments/exp_gsdc2023_pf.py` — GSDC train 評価
- `experiments/exp_gsdc2023_submission.py` — GSDC test submission
- `experiments/exp_position_update_eval.py` — position_update 評価
- `experiments/exp_hk_visualization.py` — HK GIF 生成
- `experiments/exp_particle_visualization.py` — OSM 可視化

### 11.5 CI
- `.github/workflows/ci.yml` — lint + test-python + build-cuda

---

## 12. ユーザーからの指示

- **FGO はメイン engine には NG** (PF/smoother が軸)。ただし weak-DD window など局所救済に限って FGO を使うハイブリッドは OK (2026-04-18 緩和)
- **PR #4 は merge 不可** — 明示許可が必要 (現状 CLOSED なので moot)
- **コミットに Co-Authored-By は付けない** (私自身名義のみ)
- **PR に AI 生成表記は入れない**
- **完了時刻は具体的な時刻で答える**
- **2026-04-20 redact 実施**: commit author を全て `gnss-gpu contributors <redacted@example.com>` に書き換え済み。以後のコミットも同じ author 情報で (設定するなら `git config user.name "gnss-gpu contributors"` + `user.email "redacted@example.com"`)
- **ファイル内容の redact**: `(16GB VRAM)` は全履歴から削除済み、`redacted@example.com` / `gnss-gpu contributors` (pyproject.toml) → `gnss-gpu contributors` / `redacted@example.com` に置換済み

---

## 13. 次セッション向けメモ (2026-04-21 現在)

### バックグラウンド codex セッション履歴 (2026-04-17〜21)
最新のコミット群は `bee364f` (redact) を最終として、その前に以下の試行 (いずれも revert or negative note のみ残存):
- codex4 algorithmic smoother tweaks → negative
- codex5 Huber DD likelihood → negative (revert)
- codex6 OSM road constraint → negative (revert)
- codex7 local FGO hybrid → 同 MAP 解
- codex8 LAMBDA L1 integer fix → partial fix、改善なし
- codex9 widelane → negative
- codex10 widelane region-aware → negative (revert)
- codex11 per-particle hard NLOS → 壊滅 (revert)
- codex12 per-particle Huber → negative (revert)
- codex13 naive sampled velocity RBPF → 1.22m 悪化 (残存、default off)
- codex14 proper RBPF (velocity KF) → 1.20m full / 0.89m subset (残存、default off、`--rbpf-velocity-kf`)

### 本プランの読み方 (引き継ぐ codex へ)

1. まず §0 現状サマリと §10 「次にやるべきこと」を読む
2. §3 全ネガティブ結果を必ず確認 (再試行禁止)
3. §5 Odaiba presets の `odaiba_best_accuracy` を baseline として使う
4. AAA (region-aware proper RBPF) が最も promising、subset 0.89m の再現と full への展開が次タスク
5. 北極星 (冒頭) の CT-RBPF-FGO を意識しつつ、Phase 毎に独立コミット

### 既存 default-off 機能 (任意活用可)
- `--rbpf-velocity-kf`: proper RBPF (codex14)、Q_v, Doppler sigma はまだ未最適
- `--doppler-per-particle`: naive sampled velocity (codex13)、proper RBPF と排他
- `--per-particle-huber` / `--per-particle-nlos-gate`: per-particle gate 系 (codex11/12)、全 negative
- `--widelane`: widelane DD (codex9)、region-aware gate 付き (codex10)、default off
- `--fgo-local-window` / `--fgo-local-lambda`: local FGO + LAMBDA (codex7/8)

### 重要: codex への指示テンプレート
各セッションは `.codex_handoff{N}.md` に詳細を書いて渡す形式。過去の例は gitignore されているため参照不可だが、plan.md §3 / §10 を元に構成すれば同等。
