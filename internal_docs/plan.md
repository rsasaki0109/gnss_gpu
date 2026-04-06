# gnss_gpu 引き継ぎメモ

**最終更新**: 2026-04-07 JST  
**現在の HEAD**: `b361006` (feature/carrier-phase-imu)  
**ブランチ**: `feature/carrier-phase-imu` (PR #4 open, main から分岐)  
**現フェーズ**: carrier phase / TDCP 実装中。Odaiba P50=1.65m で SPP と同等。1m 切りを目指す。

---

## 0. 最初に読むもの

1. `README.md` — 最新の全結果
2. `internal_docs/plan.md` (これ)
3. `internal_docs/experiments.md`
4. `internal_docs/decisions.md`
5. `python/gnss_gpu/particle_filter_device.py` — PF の Python API (position_update, cb_correct, smoother 含む)
6. `include/gnss_gpu/pf_device.h` — CUDA API

---

## 1. 現在の結論

### 1.1 headline results (Odaiba, dual-frequency Trimble)

| Method | P50 | P95 | RMS | >100m |
|---|---:|---:|---:|---:|
| RTKLIB demo5 | 2.67m | 32.41m | 13.08m | — |
| SPP (gnssplusplus) | 1.66m | 12.96m | 63.25m | 0.08% |
| **PF 100K (full stack)** | **1.65m** | **12.60m** | **6.45m** | **0%** |

PF が RTKLIB demo5 に全指標で勝利。SPP の P50 にも匹敵 (1.65 vs 1.66)。

### 1.2 Shinjuku (deep urban canyon)

| Method | P50 | RMS |
|---|---:|---:|
| SPP | 3.01m | 18.12m |
| **PF 100K** | **3.13m** | **13.88m** |

RMS 23% 改善。P50 は SPP にやや負け。

### 1.3 HK-20190428 (supplemental, single-frequency ublox)

| Method | P50 | P95 | RMS |
|---|---:|---:|---:|
| RTKLIB demo5 | 16.18m | 60.85m | 26.80m |
| SPP | 15.27m | 43.72m | 23.71m |
| **PF 100K** | **14.21m** | **41.60m** | **22.53m** |

RTKLIB demo5 に P50 12%, P95 32%, RMS 16% 勝利。single-freq の限界あり → supplemental 扱い。

### 1.4 GSDC 2023 (smartphone, supplemental)

| Method | P50 | RMS |
|---|---:|---:|
| WLS (Android) | **1.92m** | **2.34m** |
| PF 100K | 2.19m | 2.63m |

PF は WLS に負け。GSDC は open-sky 寄りで NLOS 少ない → PF の temporal filtering の恩恵が少ない。PF は urban canyon で強い。

---

## 2. 実装済みの技術スタック

### 2.1 CUDA カーネル (src/particle_filter/pf_device.cu)

| カーネル | 機能 |
|---|---|
| pf_device_initialize | GPU 上でパーティクル初期化 |
| pf_device_predict | ランダムウォーク predict (velocity 対応) |
| pf_device_weight | pseudorange Gaussian 尤度 (Student's t 対応) |
| **pf_device_position_update** | SPP soft constraint (Gaussian) |
| **pf_device_shift_clock_bias** | per-epoch cb re-centering |
| pf_device_ess | ESS 計算 |
| pf_device_resample_systematic | systematic resampling |
| pf_device_resample_megopolis | Metropolis resampling |
| pf_device_estimate | 重み付き平均 |

### 2.2 Python API (python/gnss_gpu/particle_filter_device.py)

| メソッド | 機能 |
|---|---|
| position_update() | SPP position-domain soft constraint |
| correct_clock_bias() | per-epoch cb correction (median residual) |
| shift_clock_bias() | cb shift (低レベル) |
| enable_smoothing() / store_epoch() / smooth() | forward-backward smoother |

### 2.3 gnssplusplus API 拡張

CorrectedMeasurement に追加:
- `prn` — 衛星 PRN 番号
- `carrier_phase` — 搬送波位相 [cycles]
- `doppler` — Doppler [Hz]
- `snr` — C/N0 [dB-Hz]
- `satellite_velocity` — 衛星速度 ECEF [m/s]
- `clock_drift` — 衛星 clock drift [s/s]

### 2.4 観測スタック (Python レベル)

| 手法 | 効果 (Odaiba) | 状態 |
|---|---|---|
| gnssplusplus corrections | baseline | 実装済み |
| cb_correct | RMS -0.7m | 実装済み |
| position_update | P50 -2m | 実装済み |
| Doppler velocity | P50 -0.05m | 実装済み |
| Dual-freq iono-free (L1+L2) | P50 -0.02m | 実装済み |
| Elevation weighting | P95 改善 | 実装済み |
| SNR weighting | P95 改善 | 実装済み (HK) |
| Carrier phase NLOS detection | P95 改善 | 実装済み (HK) |
| RAIM satellite exclusion | P95 -1m (HK) | 実装済み (HK) |
| Carrier-phase-derived PR (float) | P50 -0.1m | 実装済み、効果限定的 |
| Forward-backward smoother | RMS -1m, P50 +0.3m | 実装済み (トレードオフ) |

---

## 3. 次にやるべきこと (carrier phase / TDCP)

### 3.1 TDCP (Time-Differenced Carrier Phase) — 最優先

**目的**: cm 級の inter-epoch 変位推定で predict を劇的に改善

**前回の問題**: 衛星移動分を引いていなかった → 修正コード作成済みだが未テスト

**正しい TDCP**:
```
delta_L_m = (L_current - L_prev) * wavelength
sat_range_change = los · avg_sat_vel * dt
sat_clock_change = avg_clock_drift * C * dt
corrected = delta_L - sat_range_change + sat_clock_change
WLS: [los, 1] * [delta_rx; delta_cb_rx] = corrected
velocity = delta_rx / dt
```

**実装場所**: 実験スクリプト内（Python レベル）。CUDA カーネル変更不要。

**必要なデータ**: gnssplusplus の carrier_phase + satellite_velocity + clock_drift (拡張済み)

**期待効果**: Doppler velocity (0.1 m/s 精度) → TDCP (0.01 m/s 精度)。predict noise が 10 倍改善。

### 3.2 Wide-lane integer ambiguity resolution

**目的**: carrier-phase-derived pseudorange の精度向上 (float ~1.5m → integer ~0.1m)

**手法**: Melbourne-Wübbena combination
```
N_wl = L1_cycles - L2_cycles - P_nl / lambda_wl
P_nl = (f1*P1 + f2*P2) / (f1+f2)
lambda_wl = 0.862m
→ 数エポック平均して integer に round
```

**前回の問題**: N_wl は固定できたが、N1 = N_wl + N2 の N2 が PF 位置精度に依存 → 循環依存

**解決策**: TDCP で PF 位置を改善 → N2 推定精度が向上 → carrier phase PR 精度向上の好循環

### 3.3 TDCP + carrier phase の統合パイプライン

```
1. predict: TDCP velocity (cm 級変位)
2. update: carrier-phase-derived PR (wide-lane fixed N_wl + float N2)
3. position_update: SPP soft constraint
4. cb_correct: per-epoch
```

TDCP と carrier phase PR の両方が carrier_phase を使うが、異なる用途:
- TDCP: **エポック間の差分** (ambiguity cancel)
- CP PR: **絶対距離** (ambiguity 必要)

---

## 4. 正直なネガティブ結果

| 手法 | 結果 | 理由 |
|---|---|---|
| Student's t likelihood | 全データセットで悪化 | urban canyon では Gaussian が安定 |
| RTK carrier phase | float-only、改善なし | urban NLOS で integer fix 不可 |
| Float carrier phase (HK) | 効果なし | single-freq + NLOS で ambiguity 収束しない |
| OSM map constraint | 悪化 | wrong road matching で引きずられる |
| TDCP (初回実装) | 動作せず | 衛星移動分の未補正 |
| Hatch filter | Odaiba で悪化 | urban canyon で carrier phase が途切れ diverge |
| DGNSS (base station 差分) | 改善なし | gnssplusplus 補正が既に十分 |
| GSDC PF | WLS に負け | open-sky で temporal filtering 不要 |
| 1M particles + small sigma_pos | 崩壊 | particle depletion (sp<1 で追従不能) |

---

## 5. ブランチ・PR 状態

| ブランチ | PR | 状態 |
|---|---|---|
| feature/carrier-phase-imu | #4 (open) | position_update + cb_correct + gnssplusplus API 拡張 + smoother |
| main | — | old mainline (PF+RobustClear-10K) |

**PR #4 には merge 許可が出ていない** (ユーザーが "mada merge sinaidene" と指示)。

---

## 6. データセット

| データ | 場所 | 受信機 | 周波数 | 用途 |
|---|---|---|---|---|
| Odaiba | /tmp/UrbanNav-Tokyo/Odaiba | Trimble | L1+L2+L5 | headline |
| Shinjuku | /tmp/UrbanNav-Tokyo/Shinjuku | Trimble | L1+L2+L5 | headline |
| HK-20190428 | /tmp/UrbanNav-HK/HK_20190428 | ublox M8 | L1 only | supplemental |
| HK TST/Whampoa | /tmp/UrbanNav-HK-New/ | ublox | L1 | extreme (SPP >300m) |
| GSDC 2023 | /tmp/gsdc_data/gsdc2023/ | Pixel 4 | L1+L5 | supplemental |

---

## 7. 重要ファイル

### 7.1 CUDA コア
- `src/particle_filter/pf_device.cu` — 全カーネル
- `include/gnss_gpu/pf_device.h` — API 宣言
- `python/gnss_gpu/_pf_device_bindings.cpp` — pybind11

### 7.2 Python API
- `python/gnss_gpu/particle_filter_device.py` — ParticleFilterDevice クラス
- `python/gnss_gpu/imu.py` — IMU ローダー + ComplementaryHeadingFilter

### 7.3 gnssplusplus (submodule)
- `third_party/gnssplusplus/` — feature/expose-corrected-pseudoranges ブランチ
- CorrectedMeasurement に PRN, carrier_phase, doppler, satellite_velocity 追加済み

### 7.4 実験スクリプト
- `experiments/exp_urbannav_fixed_eval.py` — メイン評価
- `experiments/exp_position_update_eval.py` — position_update 評価
- `experiments/exp_hk_visualization.py` — HK GIF 生成
- `experiments/exp_gsdc2023_pf.py` — GSDC 評価
- `experiments/exp_particle_visualization.py` — OSM 可視化 (baseline_label 対応)

### 7.5 RINEX パース
- dual-frequency RINEX パースは実験スクリプト内に inline 実装
- gnssplusplus は L1 のみ。L2 は RINEX から直接読む必要あり
- RINEX TOW → gnssplusplus TOW のオフセット: `tow_offset = gnssplusplus_first - rinex_first` (Odaiba: 259200s = 3 days)

---

## 8. 技術メモ

### 8.1 Clock bias の問題
- Trimble: cb ≈ -99,000m, drift ~6 m/s → PF の random walk (sigma_cb=300) で追従可能
- ublox: cb ≈ -960,000m, drift ~65 m/s → per-epoch correct_clock_bias() が必須
- correct_clock_bias(): median(PR - range) で cb を推定し、全パーティクルを shift

### 8.2 Position update の仕組み
- CUDA カーネルで log_weight += -0.5 * dist^2 / sigma^2
- SPP 位置に近いパーティクルに高い重み → resampling で SPP 近傍に集約
- sigma が小さいほど SPP に強く引かれる (sigma=3 が Odaiba 最適)
- PU なしだと P50 が 2.68m に悪化 (SPP アンカーが必要)

### 8.3 Smoother (forward-backward)
- enable_smoothing() → store_epoch() × N → smooth()
- backward pass: 新しい PF インスタンスで逆順に走る
- 結合: (forward + backward) / 2
- 効果: RMS 改善 (6.51→5.35), P50 悪化 (1.67→2.94)
- position_update が両方向で SPP に引くため、差が小さい

### 8.4 Doppler velocity
- gnssplusplus の satellite_velocity で衛星運動を補正
- range_rate = -doppler * C / freq
- WLS: los · rx_vel + clock_drift = sat_vel · los - range_rate + sat_clock_drift * C
- SPP 位置差分 (0.5 m/s) より高精度 (0.1 m/s)

### 8.5 GSDC で PF が WLS に勝てない理由
- GSDC は open-sky 寄り → pseudorange ノイズが Gaussian に近い → WLS が最適解
- PF の predict noise (sigma_pos) が余計 → 精度劣化の主因
- PF は NLOS が多い urban canyon で temporal filtering の恩恵が大きい

---

## 9. ビルド

```bash
# gnss_gpu CUDA
cd build && make -j$(nproc)
cp build/python/gnss_gpu/_gnss_gpu_pf_device.cpython-312-x86_64-linux-gnu.so python/gnss_gpu/

# gnssplusplus
cd third_party/gnssplusplus/build && cmake --build . -j$(nproc)

# テスト
PYTHONPATH=python python3 -m pytest tests/ -q
```

---

## 10. 残課題 (Cursor に引き継ぐ)

### 10.1 最優先: TDCP + carrier phase で 1m 切り

1. TDCP velocity の正しい実装 (衛星移動補正済みコードは書いた、テスト未完了)
2. TDCP velocity → predict で sigma_pos を下げられるか検証
3. Wide-lane integer fix → narrow-lane float → carrier-phase PR
4. TDCP + CP PR の相乗効果を Odaiba で検証

### 10.2 FGO は NG

ユーザーが明示的に FGO を拒否。PF / smoother の枠組みで攻める。

### 10.3 PR #4 の merge

ユーザーが "merge するな" と言っている。merge は明示的に許可を得てから。

### 10.4 README / PR 更新

carrier phase / TDCP の結果が出たら README と PR #4 の description を更新。

### 10.5 GIF

HK の GIF は生成済み (`experiments/results/paper_assets/particle_viz_hk20190428.gif`)。
Odaiba/Shinjuku の GIF は既存のまま (gnssplusplus + RTKLIB demo5 比較)。
