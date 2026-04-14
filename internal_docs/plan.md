# gnss_gpu 引き継ぎメモ

**最終更新**: 2026-04-14 JST
**現在の HEAD**: `ca73867` (`feature/carrier-phase-imu`)
**ブランチ**: `feature/carrier-phase-imu` (PR #4 open, CI 全 pass)
**作業ツリー**: dirty (DD pseudorange/carrier/quality/stop-detect/GSDC 評価が積み上がっている)
**FGO**: 使わない。PF の枠内で詰める。
**PR #4**: 許可があるまで merge しない。

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

#### UrbanNav Tokyo (dual-frequency Trimble, 100K particles, smoother)

| 場所 | PF P50 | PF RMS | Baseline | Baseline RMS | PF RMS 改善 |
|---|---:|---:|---|---:|---:|
| **Odaiba** | **1.36m** | **4.11m** | RTKLIB demo5 | 13.08m | **69%** |
| **Shinjuku** | **2.52m** | **8.92m** | SPP | 18.12m | **51%** |

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

| 手法 | データ | 結果 | 原因 |
|---|---|---|---|
| Student's t likelihood | 全データ | 悪化 | urban canyon で Gaussian が安定 |
| RTK carrier phase | Odaiba | float-only, 改善なし | NLOS で integer fix 不可 |
| Float carrier phase | HK | 効果なし | single-freq + NLOS で ambiguity 収束せず |
| OSM map constraint | HK | 悪化 | wrong road matching |
| DGNSS (base station 差分) | Odaiba | 改善なし | gnssplusplus 補正が既に十分 |
| Hatch filter | Odaiba | 悪化 | urban canyon で carrier phase 途切れ diverge |
| TDCP predict | Odaiba | IMU に負ける | IMU (wheel+gyro) の方が高精度 |
| DD PR base interpolation | Odaiba | RMS 暴発 | 1Hz→10Hz 補間の品質が低い |
| 1M + small sigma_pos | Odaiba | 崩壊 | particle depletion (sp<1) |
| DD gate 緩和 | Odaiba | 悪化 | 品質の悪い pair を通すと P50 悪化 |
| DD sigma support scaling | Odaiba | 変化なし | 閾値 sweep で改善しない |
| sigma_pos < 1.0 (100K) | Odaiba | P50 悪化 | predict noise 不足 |
| TDCP (GSDC smartphone) | Kaggle | 30.76m (悪化) | smartphone ADR の品質が低い |
| Hatch filter (GSDC) | Kaggle | 10.15m (悪化) | 頻繁な cycle slip で diverge |
| Carrier phase smoothing (GSDC) | Kaggle | 全て悪化 | smartphone carrier phase は信頼できない |

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
| **v3** | **4.128m** | — | + smoother |
| v2 | 10.150m | — | + TDCP + Hatch (悪化) |
| v11 | 4.223m | 5.255m | reset-safe segmented smoother |

### 4.4 なぜ GSDC で PF が勝てないか

1. **スマホの pseudorange ノイズが大きい** (15-20m std) → PF の PR update の情報量が少ない
2. **WLS が既に良い** (Google 最適化済み) → temporal filtering の余地が少ない
3. **PF の predict noise (sigma_pos=10)** が邪魔 → open-sky ではノイズを足すだけ
4. **carrier phase がスマホで信頼できない** → TDCP/Hatch が全滅
5. **smoother が divergence reset をまたぐと hidden/private で壊れる** → reset-safe segmentation で private は 5.255m まで回復したが、public best 更新には未達

### 4.5 ファイル

- `experiments/exp_gsdc2023_pf.py` — 全 train 評価 (146 run)
- `experiments/exp_gsdc2023_submission.py` — test submission 生成
- `experiments/results/gsdc2023_eval.csv` — train 評価結果
- `experiments/results/gsdc2023_submission.csv` — v1 submission
- `experiments/results/gsdc2023_submission_v2.csv` — v2 submission
- `experiments/results/gsdc2023_submission_v3.csv` — v3 submission

---

## 5. UrbanNav 詳細

### 5.1 frozen presets

| Preset | P50 | RMS | 用途 |
|---|---:|---:|---|
| odaiba_reference | 1.38m | 5.08m | frozen baseline (2026-04-14 rerun) |
| odaiba_stop_detect | 1.36m | 4.11m | + stop detection, current best |
| odaiba_reference_guarded | 1.38m | 5.43m | low-ESS tail guard。full Odaiba では悪化 |

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
# 18 passed (5m45s)
```

### 7.3 frozen reference 再現

```bash
PYTHONPATH="python:third_party/gnssplusplus/build/python:third_party/gnssplusplus/python" \
python3 experiments/exp_pf_smoother_eval.py --data-root /tmp/UrbanNav-Tokyo --preset odaiba_reference
# SMTH P50=1.38m RMS=5.08m
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

## 10. 次にやるべきこと

### 10.1 PR #4 の merge 準備 (最優先)

CI 全 pass。README 最新。merge はユーザーの明示許可待ち。

### 10.2 精度改善の方向 (もし続けるなら)

**1m 切りに到達するには:**
- DD pair 数を増やす (base station coverage 改善 — データの制約)
- base station interpolation の品質改善 (1Hz→10Hz、現状は品質不足で RMS 暴発)
- weak-DD epoch の fallback を DD=yes 並みの品質にする

**攻めて微改善が出る可能性:**
- support_skip 閾値緩和 (max-pairs=2 で P50 0.01m 改善済み)
- epoch 2445-4890 の特別扱い (base coverage 穴の区間)

**攻めても改善しないもの:**
- sigma_pos 縮小 (100K では 1.2 が最適、これ以下は particle depletion)
- DD gate 緩和 (品質の悪い pair を通すと P50 悪化)
- TDCP predict (IMU に負ける)
- smoother tail guard (ESS/shift guard は weak segment の局所改善はあるが、full Odaiba では悪化)

### 10.3 GSDC 改善の方向 (もし続けるなら)

- **DGNSS (基準局差分)** — CORS station のデータが必要。gsdc2023 1st place の主要因。
- **Robust WLS (Huber)** — PF の代わりに robust WLS で前処理
- carrier phase は smartphone では使えない (honest negative result)

### 10.4 論文/artifact 整備

- README は最新結果に更新済み
- GSDC 結果を README に supplemental として追記
- GIF は Odaiba/Shinjuku/HK 生成済み

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

- **FGO は NG** — PF/smoother の枠組みで攻める
- **PR #4 は merge 不可** — 明示許可が必要
- **コミットに Co-Authored-By は付けない**
- **PR に AI 生成表記は入れない**
- **完了時刻は具体的な時刻で答える**
