# gnss_gpu 引き継ぎメモ

**最終更新**: 2026-04-07 JST（本稿を Claude 向けに長めに更新）  
**現在の HEAD**: `d0c7436` (feature/carrier-phase-imu) — 作業ツリーは **多くの変更が未コミット**（実験スクリプト・`tdcp_velocity.py`・FFBSi 系・CUDA 祖先バッファ等）。コミット前に `git status` を必ず確認すること。  
**ブランチ**: `feature/carrier-phase-imu` (PR #4 open, main から分岐)  
**現フェーズ**: **TDCP ガイド付き PF** は Python 実装・単体テストまで完了。**系譜（genealogy）スムーザ** と公平比較スクリプトあり。仰角重み付き TDCP WLS を試したが効果はデータ区間依存。**1 m 切り**は引き続き carrier / wide-lane 統合が主戦場。  
**FGO**: ユーザー方針で **やらない**（既存どおり）。

### Claude へ（最初に読む順）

1. 本ファイルの **§1.5**（TDCP ガイドの短い実測）と **§10**（残課題）
2. `python/gnss_gpu/tdcp_velocity.py` — TDCP WLS（衛星運動・衛星時計補正済み）
3. `python/gnss_gpu/particle_ffbsi.py` — genealogy / marginal FFBSi
4. `experiments/exp_pf_smoother_eval.py` — `run_pf_with_optional_smoother`、**`--skip-valid-epochs`**
5. `experiments/exp_gnss_compare_pf_ffbsi.py` — forward vs FFBSi の門番比較（`resampling` 一致が必須）
6. PR #4 は **merge 禁止**（許可があるまで）

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

### 1.5 短い実測メモ: PF + TDCP 速度ガイド + 仰角重み（100k 粒子, PU=1.95, σ_pos_tdcp=1.0）

**前提**: 以下は `experiments/exp_pf_smoother_eval.py` 一行評価。**全区間フル run の headline 表（§1.1）とは条件が違う**（エポック数・ウィンドウ・PU 等）。相対比較用。

| 設定 | 区間 | 仰角重み | P50 | RMS 2D |
|---|------|---------|-----|--------|
| Odaiba | 先頭 500 有効 ep | off | 0.78 m | 0.86 m |
| Odaiba | 先頭 500 有効 ep | on (`--tdcp-elevation-weight`) | 0.75 m | 0.84 m |
| Shinjuku | 先頭 500 有効 ep | off | **1.79 m** | 2.77 m |
| Shinjuku | 先頭 500 有効 ep | on | 1.87 m | 2.81 m |
| Odaiba | **中盤** skip=4000 後 500 ep | off | **6.80 m** | 9.30 m |
| Odaiba | 同上 | on | 6.88 m | 9.58 m |

**解釈（事実と推論を分ける）**

- **事実**: 仰角重みは区間によって **改善も悪化も**あり、一貫した勝ちではない。
- **確認済み (2026-04-08)**: `CorrectedMeasurement.elevation` は **radians**（`navigation.cpp` の `atan2` → `spp.cpp` でそのまま代入）。`tdcp_velocity.py` の `np.sin(el)` は正しい。効果が区間依存なのは `sin²(el)` モデル自体の限界。

**ウィンドウ評価**: `--skip-valid-epochs K` で先頭 K 有効エポックは **PF では burn-in のみ**（メトリクスには含めない）。`--max-epochs N` と併用で **合計 K+N 有効 ep 処理後に打ち切り**。

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
- `elevation` — 仰角 [rad]（`navigation.cpp:atan2` 由来。TDCP 仰角重みで使用。**単位確認済み 2026-04-08**）

### 2.4 TDCP 速度（Python・コア外）

| 項目 | 内容 |
|---|---|
| 実装 | `python/gnss_gpu/tdcp_velocity.py` |
| 内容 | 衛星 LOS・衛星平均速度・衛星 clock drift で補正した TDCP、WLS で `delta_rx/dt`；L1 波長仮定、多周波行の SNR で 1 行に圧縮 |
| テスト | `tests/test_tdcp_velocity.py` |
| オプション | `elevation_weight=True` で行重みに `max(sin el, floor)²`（両エポックで有限の `elevation` がある行のみ） |

### 2.5 系譜スムーザ・FFBSi（Python + デバイス）

| 項目 | 内容 |
|---|---|
| 動機 | systematic 再サンプル時の親インデックスを無視した marginal backward は、ESS 等と組み合わせたとき整合が崩れうる |
| CUDA | systematic resample カーネルが **祖先インデックス GPU バッファ**に書き込み。`pf_device_get_resample_ancestors` / Python `get_resample_ancestors()` |
| デフォルト実験 | FFBSi 評価は **genealogy**（祖先一致）を優先。marginal は明示指定 |
| resampling | genealogy 比較では forward も **`systematic`** に揃える必要あり（`exp_gnss_compare_pf_ffbsi.py` の `--resampling`） |

### 2.6 観測スタック (Python レベル)

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
| **TDCP predict ガイド** | 条件依存（§1.5） | **実装済み**・実験スクリプト複数から利用可 |
| **Genealogy スムーザ** | FFBSi 公平比較用 | 実装済み（祖先 export 要 systematic） |

---

## 3. 次にやるべきこと (carrier phase / TDCP)

### 3.1 TDCP (Time-Differenced Carrier Phase) — 最優先（更新）

**目的**: cm 級の inter-epoch 変位推定で predict を劇的に改善し、**σ_pos / tight-RMS 連動**や wide-lane 側の好循環を作る。

**実装状況（2026-04）**: 衛星運動・衛星時計補正を含む WLS は **`tdcp_velocity.py` に実装済み**。単体テストあり。実験では `predict_guide=tdcp` と `--sigma-pos-tdcp` 等で制御。

**残り（優先順）**

1. ~~**仰角 `elevation` の単位・値域の確認**~~ → **確認済み (2026-04-08): radians**。`navigation.cpp:atan2` → `spp.cpp` → Python バインディング。`tdcp_velocity.py` の `sin(el)` は正しい。
2. **TDCP 採用時の σ_pos カーブ**（RMS・衛星本数・都市区間で切り替え）— いまの `sigma_pos_tdcp` / `tdcp_tight_rms_max` の整理・拡張。
3. Odaiba **全区間**で TDCP ガイドの headline 行を README 水準で更新するか判断。

**正しい TDCP（整理・数式は従来どおり）**:
```
delta_L_m = (L_current - L_prev) * wavelength
sat_range_change = los · avg_sat_vel * dt
sat_clock_change = avg_clock_drift * C * dt
corrected = delta_L - sat_range_change + sat_clock_change
WLS: [los, 1] * [delta_rx; delta_cb_rx] = corrected
velocity = delta_rx / dt
```

**実装場所**: **`python/gnss_gpu/tdcp_velocity.py`**（実験は import して利用）。CUDA カーネル変更不要。

**必要なデータ**: gnssplusplus の carrier_phase + satellite_velocity + clock_drift（**elevation はオプション**）

**期待効果**: Doppler ガイドより強い predict が取れる区間では σ_pos を下げられる → 1 m 帯への寄与を全体評価で確認。

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
| TDCP (初回実装) | 動作せず | 衛星移動分の未補正（**現行 `tdcp_velocity.py` では補正済み**） |
| TDCP WLS `sin²(el)` 重み | 区間次第で悪化あり | モデルが粗い（仰角単位は rad で正しい、§1.5） |
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
- `experiments/exp_urbannav_fixed_eval.py` — メイン評価（headline に近い全区間）
- `experiments/exp_pf_smoother_eval.py` — PF + optional smoother；**`run_pf_with_optional_smoother`**；**`--skip-valid-epochs`**；TDCP 仰角重み **`--tdcp-elevation-weight`**
- `experiments/exp_ffbsi_eval.py` — FFBSi / genealogy；`resampling=systematic` 固定
- `experiments/exp_gnss_compare_pf_ffbsi.py` — forward vs smoother **公平比較**（同一 `dataset`、`--resampling`）
- `experiments/exp_submeter_sweep.py` / `exp_zenbu_sweep.py` / `exp_fine_pu_sweep.py` — 設定スイープ
- `experiments/exp_position_update_eval.py` — position_update；TDCP ガイド・仰角重み CLI あり
- `experiments/exp_fgo_benchmark_hook.py` — FGO ベンチ外部コマンド用フック（**実装しない**；参照用）
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

### 8.3 Smoother (forward-backward) と FFBSi
- **デフォルト smoother（exp_pf_smoother_eval）**: enable_smoothing() → store_epoch() × N → smooth()；backward は別 PF で逆順、**forward/backward の平均**。
- **効果（過去メモ）**: RMS 改善例 (6.51→5.35), P50 悪化例 (1.67→2.94)。PU が両方向で SPP に引くと差が縮む。
- **FFBSi / genealogy（exp_ffbsi_eval）**: forward 履歴Weights/States/**祖先**を使い、**backward で祖先パスをサンプル**（genealogy）または legacy カーネル（marginal）。
- **公平比較の注意**: forward が **Megopolis**、smoother 側が **systematic 前提**だと再サンプルが一致せず指標が歪む。比較時は **`--resampling systematic`** 等で揃える（`exp_gnss_compare_pf_ffbsi.py`）。
- **実装メモ**: systematic 再サンプル直後に `get_resample_ancestors()` で親インデックス取得。テスト: `tests/test_ffbsi.py`, `tests/test_cuda_streams.py`。

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

# CUDA / バインディングを直したら .so を python 配下へコピー（環境依存ファイル名）
# cp build/python/gnss_gpu/_gnss_gpu_pf_device*.so python/gnss_gpu/
```

---

## 10. 残課題 (Claude / Cursor に引き継ぐ)

### 10.1 最優先: 1 m 切り（TDCP + carrier、FGO なし）

1. ~~**仰角の単位確認**~~ → **確認済み: radians** (2026-04-08)。
2. ~~**TDCP × σ_pos スイープ**~~ → **完了 (2026-04-08)**。全区間では TDCP ガイド単体の恩恵は小さい（Odaiba P50=1.672m、ベースライン 1.65m とほぼ同等）。σ_pos_tdcp=1.5-3.0, PU=1.5 が安定。Shinjuku は 3.24m（ベースライン 3.13m よりやや悪化）。短区間 (500ep) では P50=0.78m と効くが、全区間では carrier phase 途切れが悪影響。
3. ~~**TDCP adaptive + smoother 組み合わせ**~~ → **完了 (2026-04-08)**。結論: **効果なし**。
   - **Adaptive TDCP**: postfit RMS が常に閾値以下で fallback 0 回。always-on と同一結果。TDCP の品質は良好だが PF 全体への寄与が小さい。
   - **Smoother + TDCP**: Odaiba P50=1.66m (ベスト)、RMS は smoother で一貫改善 (6.33→6.27m)。ただし P50 改善は微小。
   - **Shinjuku**: tdcp_adaptive 単体が P50 ベスト (3.28m) だがベースライン 3.13m より悪化。
   - 詳細: `results/tdcp_adaptive_sweep.csv`, `results/pf_smoother_eval.csv`
4. **Wide-lane → carrier-phase PR** パイプライン（§3.2）が **次の主戦場**。`wide_lane.py` 実装済み (12テスト全パス)、実験統合が必要。
5. **README / headline**：条件を揃えた Odaiba 全区間の数値を更新するか、短評価と役割分担を文書化。

### 10.2 Cursor タスク（2026-04-08 起票、上から順に実行）

#### ~~Task A: tdcp_adaptive スイープ~~ → **完了 (2026-04-08)**
80 runs 実行。TDCP postfit RMS が常に全閾値 (1.0-8.0m) 以下で fallback 0 回。adaptive と always-on で結果同一。ベースラインを上回る設定なし。`results/tdcp_adaptive_sweep.csv`。

#### ~~Task B: Wide-lane integer ambiguity resolution~~ → **実装完了 (2026-04-08)**
`python/gnss_gpu/wide_lane.py` + `tests/test_wide_lane.py` (12 tests pass)。Melbourne-Wübbena N_wl 推定 + integer fix + wide-lane pseudorange 出力。**実験統合は未着手**（下記 Task D）。

#### ~~Task C: Smoother + TDCP 組み合わせ~~ → **完了 (2026-04-08)**
8 runs 実行。smoother + tdcp_adaptive: Odaiba P50=1.66m (微改善), Shinjuku P50=3.30m (悪化)。効果なし。`results/pf_smoother_eval.csv`。

#### ~~Task D: Wide-lane / carrier-phase smoothing 統合~~ → **完了・ネガティブ (2026-04-08)**
**実装**: `exp_widelane_eval.py` + `urbannav.py` L2 拡張 (Codex 実装)。3 段階で試行:
1. **N_wl 固定 wide-lane PR 直接置換** → P50=35.75m。N1 未解決でバイアス発散。
2. **median bias 除去** → P50=4.11m。改善だがまだ悪い。
3. **Iono-free divergence-free Hatch filter** (carrier delta smoothing, alpha=0.2, L1 fallback) → P50=1.71m, RMS=6.31m。ベースライン 1.67m より微悪化。カバレッジ 42.6%。

**結論**: carrier-phase PR 改善は urban canyon の cycle slip + iono divergence で限界。§4 の Hatch filter ネガティブ結果と一致。**1m 切りには carrier-phase 以外のアプローチが必要**。

**ネガティブ結果に追加**:
| Wide-lane N_wl → PR 置換 | P50=35m 崩壊 | N1 未解決でバイアス発散 |
| Iono-free Hatch filter (dual-freq) | P50=1.71m (微悪化) | urban canyon cycle slip + iono divergence |

### 10.3 実装・実験の片付け

- ~~未コミットの実験スクリプト・`tdcp_velocity.py`・FFBSi・祖先バッファ変更を **論理単位でコミット**~~ → **完了 (2026-04-08)**。7 コミットに整理済み。
- `skip_valid_epochs` を **`exp_gnss_compare_pf_ffbsi.py` 等**にも渡したい場合は `run_pf_with_optional_smoother` の引数をスレッドする（未対応なら明記して実装）。

### 10.4 FGO は NG

ユーザーが明示的に FGO を拒否。PF / smoother / 粒子 backward の枠で攻める。

### 10.5 PR #4

**ユーザー許可まで merge しない**。

### 10.6 可視化

HK: `experiments/results/paper_assets/particle_viz_hk20190428.gif`（ほか `.mp4` が未追跡で残っている場合あり）。Odaiba/Shinjuku GIF は従来どおり。

---

## 11. 付録: コマンド例（UrbanNav）

```bash
cd /path/to/gnss_gpu/experiments
export PYTHONPATH=".:../python:../third_party/gnssplusplus/build/python:../third_party/gnssplusplus/python"

# TDCP ガイド、先頭 500 ep、PU 例
python3 exp_pf_smoother_eval.py --data-root /tmp/UrbanNav-Tokyo --runs Odaiba \
  --n-particles 100000 --max-epochs 500 --predict-guide tdcp \
  --position-update-sigma 1.95 --sigma-pos-tdcp 1.0

# 仰角重みオン
#  ... 上に加え --tdcp-elevation-weight [--tdcp-el-sin-floor 0.1]

# burn-in 4000 後に 500 ep だけメトリクス
python3 exp_pf_smoother_eval.py --data-root /tmp/UrbanNav-Tokyo --runs Odaiba \
  --n-particles 100000 --skip-valid-epochs 4000 --max-epochs 500 \
  --predict-guide tdcp --position-update-sigma 1.95 --sigma-pos-tdcp 1.0
```

**注意**: `exp_pf_smoother_eval` の CSV は既定パスで**上書き**される。複数 run を残すなら実行後にファイルをリネームすること。
