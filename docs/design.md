# gnss_gpu 設計書

## 1. プロジェクト概要

GPU加速GNSS測位ライブラリ。MegaParticles (Koide et al., ICRA 2024) のGPU大規模パーティクルフィルタ+SVGD設計を参照しつつ、GNSS都市測位向けに GPU常駐の大規模粒子推論、3D都市モデル（PLATEAU CityGML）からのレイトレース尤度、GNSS向けのSVGD/重み設計を統合する。研究上の主張は個別要素の「世界初」ではなく、組み合わせ、規模、実データでのロバスト性評価に置く。

### 1.1 研究的動機

#### MegaParticles (Koide et al., ICRA 2024)
- 論文: "MegaParticles: Range-based 6-DoF Monte Carlo Localization with GPU-Accelerated Stein Particle Filter"
- **LiDAR** 6-DoF localization に100万パーティクル + SVGD をGPU上で実現
- Gauss-Newton ベースSVGDによるパーティクル更新
- リサンプリングをSVGDで置き換え → サンプル枯渇回避
- A100で約91ms/frame (11Hz)
- 被引用: Nakao et al. (ICRA 2025) がSLAMに拡張

#### 関連研究の位置づけ（2026-04-01 literature audit 反映）

| 観点 | 既存研究の状況 | 本プロジェクトの狙い |
|---------|--------------|-------------|
| GNSS particle filter | Suzuki 2024, Gupta & Gao 2021 など先行あり | GPU常駐・大規模化・実用ランタイム化 |
| 3D map aided GNSS | Groves 2011, Adjrad & Groves 2017, Suzuki & Kubo 2016 など先行あり | パーティクルごとのレイトレース尤度を大規模PFに統合 |
| GPU/SVGD large-scale PF | MegaParticles 2024 がLiDARで先行 | GNSS状態空間・擬似距離尤度向けの適応 |
| 日本の公開3D都市モデル活用 | PLATEAU自体は公開基盤 | 3DMA GNSS評価パイプラインへの実装統合 |
| robustness / integrity analysis | PF系GNSSで議論あり | catastrophic tail を含む失敗解析を明示 |

#### 既存GNSS PF研究との比較
- **Suzuki (2024)**: 都市GNSSの直接PF推定。既に urban GNSS PF 自体は先行あり。
- **Gupta & Gao (2021)**: GNSS fault-aware PF。integrity / availability の論点は既に存在。
- **Suzuki & Kubo (ION 2016)**: 3DモデルとPFの統合。3D map aided GNSS PF 自体は新規ではない。
- **Groves (2011), Adjrad & Groves (2017)**: shadow matching と 3DMA GNSS ranging の系譜。3D building model 利用や shadow matching+ranging 統合は先行あり。
- **Hsu & Wen (PolyU)**: Factor Graph + 3DMA。PF以外のベイズ推論系との比較対象。
- **Zhong (UCL thesis, 2024)**: 3DMA GNSS の multi-epoch filtering/PF 系の整理があるため、研究方向そのものの新規性主張は避ける。

#### 競合研究動向
- **Deep Learning系**: Graph Transformer (NAVIGATION 2024), STL-NLOS (GPS Solutions 2024) — 学習データ依存
- **Neural表現**: Neural City Maps (Stanford, ION GNSS+ 2024) — NeRFベース、計算コスト大
- **集合ベース**: Zonotope Shadow Matching (TU Munich, 2026) — 確率的表現なし

### 1.2 守れる貢献の3本柱

1. **GPU常駐の大規模GNSS粒子推論**: 100万粒子級まで拡張可能な実装と実測ランタイム評価
2. **3Dレイトレース尤度の統合**: 各パーティクル×各衛星のLOS/NLOSを評価する NLOS-aware likelihood をPFに統合
3. **GNSS向けの粒子更新と頑健性評価**: クロックバイアスcommon-mode除去を含むSVGD/粒子更新設計と、catastrophic tail を含む失敗解析

#### 1.3 避けるべき主張

- 「world first GNSS particle filter」
- 「world first urban GNSS particle filter」
- 「3D building model を使ったGNSS尤度が初」
- 「shadow matching と ranging の統合が初」
- 「real-time 3D-map-aided GNSS が初」

詳細な整理は `docs/literature_audit_2026-04-01.md` を参照。

---

## 2. アーキテクチャ

### 2.1 全体構成

```
Python API (gnss_gpu)
    │
    ├── gnss_gpu.particle_filter      ← メガパーティクルフィルタ（1M+粒子）
    ├── gnss_gpu.particle_filter_3d   ← 3Dレイトレース統合PF
    ├── gnss_gpu.svgd                 ← Stein Variational Gradient Descent
    ├── gnss_gpu.particle_filter_device ← デバイスメモリ常駐 + CUDA Streams
    ├── gnss_gpu.positioning          ← WLS/EKF/RTK/Multi-GNSS
    ├── gnss_gpu.raytrace             ← 3Dレイトレース + BVH加速
    ├── gnss_gpu.multipath            ← マルチパスシミュレーション
    ├── gnss_gpu.skyplot              ← 脆弱性マップ (DOP)
    ├── gnss_gpu.acquisition          ← GPU信号捕捉 (cuFFT)
    ├── gnss_gpu.tracking             ← ベクトルトラッキング (EKF)
    ├── gnss_gpu.interference         ← 干渉検出・除去 (STFT)
    ├── gnss_gpu.ephemeris            ← 衛星軌道計算 (IS-GPS-200)
    ├── gnss_gpu.atmosphere           ← 大気補正 (Saastamoinen/Klobuchar)
    ├── gnss_gpu.doppler              ← ドップラー速度推定
    ├── gnss_gpu.rtk                  ← RTK + LAMBDA整数アンビギュイティ
    ├── gnss_gpu.multi_gnss           ← マルチGNSS ISB推定
    ├── gnss_gpu.raim                 ← RAIM/FDE故障検出排除
    ├── gnss_gpu.cycle_slip           ← サイクルスリップ検出
    ├── gnss_gpu.sbas                 ← SBAS/QZSS補強
    ├── gnss_gpu.bvh                  ← BVH空間分割木レイトレース加速
    └── gnss_gpu.io                   ← RINEX/NMEA/PLATEAU CityGML
    │
pybind11
    │
CUDA Kernels (20+ .cu files)
    ├── particle_filter/  ← predict, weight, weight_3d, resampling, svgd, pf_device
    ├── positioning/      ← wls, multi_gnss
    ├── raytrace/         ← raytrace, bvh
    ├── acquisition/      ← acquisition (cuFFT)
    ├── interference/     ← interference (cuFFT STFT)
    ├── tracking/         ← tracking (correlator + EKF VTL)
    ├── ephemeris/        ← ephemeris (IS-GPS-200)
    ├── atmosphere/       ← atmosphere (Saastamoinen + Klobuchar)
    ├── doppler/          ← doppler velocity
    ├── ekf/              ← EKF positioning
    ├── rtk/              ← RTK + LAMBDA
    ├── raim/             ← RAIM/FDE
    └── utils/            ← coordinates (ECEF↔LLA)
```

### 2.2 統合（ROS2 / 可視化）

```
ROS2 Integration
    ├── gnss_gpu.ros2.gnss_node     ← リアルタイム測位ノード
    ├── NavSatFix, PointCloud2      ← パーティクル群の可視化
    └── RViz config                 ← 3D表示設定

Visualization
    ├── gnss_gpu.viz.plots          ← matplotlib (skyplot, DOP, trajectory)
    └── gnss_gpu.viz.interactive    ← plotly (3D particles, heatmap)
```

---

## 3. コアアルゴリズム設計

### 3.1 パーティクルフィルタ

#### 状態ベクトル
```
x = [x, y, z, clock_bias]  (ECEF, 4次元)
```
- メモリレイアウト: **SoA** (Structure of Arrays) — coalesced memory access
- 各成分を独立配列で持つ → GPU帯域幅を最大活用

#### Predict Kernel
```cuda
__global__ void predict_kernel(
    double* px, double* py, double* pz, double* pcb,
    const double* vx, const double* vy, const double* vz,
    double dt, double sigma_pos, double sigma_cb,
    int N, unsigned long long seed, int step
);
```
- 1スレッド = 1パーティクル
- cuRAND Philox: `(seed, particle_index, step)` でステートレス
- 運動モデル: `x_new = x_old + v * dt + noise`

#### Weight Kernel (擬似距離尤度)
```cuda
__global__ void weight_kernel(
    const double* px, const double* py, const double* pz, const double* pcb,
    const double* sat_ecef, const double* pseudoranges, const double* weights_sat,
    double* log_weights,
    int N, int n_sat, double sigma_pr
);
```
- 衛星データは **shared memory** にロード（全スレッドで共有、帯域節約）
- **log-space** で重み計算（アンダーフロー防止）
- 尤度: `log p = -0.5 * Σ_sat w_s * ((pr_obs - pr_pred)² / sigma²)`

#### 3D-Aware Weight Kernel (本研究の核心)
```cuda
__global__ void weight_3d_kernel(
    // particle states, satellite data, building triangles...
    double sigma_los, double sigma_nlos, double nlos_bias
);
```
- 各パーティクル × 各衛星ペアでMöller-Trumboreレイトレース
- LOS衛星: `log_w += -0.5 * (residual / sigma_los)²`
- NLOS衛星: `log_w += -0.5 * ((residual - nlos_bias) / sigma_nlos)²`
- NLOSバイアス補正（反射パスは直接パスより長い → 正のバイアス）

#### SVGD (Stein Variational Gradient Descent)
```cuda
__global__ void svgd_gradient_kernel(
    // particles, satellite data, gradients...
    double bandwidth, int n_neighbors
);
```
- リサンプリングの代替 → サンプル枯渇回避
- スコア関数: `∇_x log p(x|y) = Σ_sat [(pr_obs - pr_pred) / σ²] * (x - sat) / r`
- **クロックバイアスcommon-mode除去**: 全衛星の残差平均を位置スコアから減算
- K=32ランダム近傍でO(N*K)計算量
- バンド幅: メディアンヒューリスティック（M=1000ペアのサブサンプル）

#### リサンプリング
2方式をサポート:

**A. Systematic Resampling**
- `thrust::inclusive_scan` でCDF計算（**double精度** — 100万パーティクル以上で数値安定性に必要）
- 各スレッドがCDFをbinary searchで祖先インデックスを取得

**B. Megopolis Resampling** (Chesser et al., 2021)
- Metropolis系列の最新手法
- チューニングパラメータ不要
- prefix sum不要 → 数値的に安定
- Double-buffering実装でレースコンディション回避

#### Adaptive Resampling
- ESS (Effective Sample Size) = `1 / Σ(w_i²)` を毎ステップ計算
- ESS < N/2 のときのみリサンプリング実行

### 3.2 デバイスメモリ常駐 + CUDA Streams

```cpp
struct PFDeviceState {
    double* d_px, *d_py, *d_pz, *d_pcb;  // パーティクル状態（GPU常駐）
    double* d_log_weights;
    cudaStream_t stream;                   // 非同期実行用ストリーム
    double* h_sat_pinned;                  // Pinned memory（非同期H2D転送）
    double* h_result_pinned;
};
```
- パーティクル状態はGPUに常駐（初期化時に1回だけcudaMalloc）
- 衛星データのみ毎エポックH2D転送（~1KB、非同期）
- CUDA Streamsでカーネル起動とメモリ転送をオーバーラップ
- 推定結果のD2Hも非同期（32バイトのみ）

### 3.3 BVH加速レイトレース

```cpp
struct BVHNode {
    AABB bbox;      // Axis-Aligned Bounding Box
    int left, right;
    int tri_start, tri_count;
};
```
- CPU側でSAH (Surface Area Heuristic) によるトップダウンBVH構築
- GPU側でスタックベーストラバーサル（最大深度64）
- slab method によるAABB-ray交差判定
- 三角形数10K+の都市モデルに対応

---

## 4. CUDA設計方針

| 項目 | 方針 |
|------|------|
| メモリレイアウト | SoA (Structure of Arrays) |
| ブロックサイズ | 256スレッド |
| 乱数生成 | cuRAND Philox4_32_10（ステートレス） |
| 重み計算 | log-space |
| prefix sum精度 | double |
| 衛星データ | shared memory broadcast |
| エラーチェック | CUDA_CHECK マクロ（std::runtime_error throw） |
| メモリ管理 | ParticleFilterDevice: GPU常駐 + CUDA Streams |
| レイトレース | Möller-Trumbore + BVH加速 |
| FFT | cuFFT (捕捉、干渉検出) |

---

## 5. 実装済みモジュール一覧

### 5.1 測位エンジン (6モジュール)
| モジュール | ファイル | 説明 |
|-----------|---------|------|
| WLS Batch | wls.cu | Gauss-Newton重み付き最小二乗、GPU並列バッチ |
| EKF | ekf.cu | 8状態拡張カルマンフィルタ（pos+vel+clk+drift） |
| RTK | rtk.cu | 二重差分RTK + LAMBDA整数アンビギュイティ（Schnorr-Euchner探索） |
| Multi-GNSS | multi_gnss.cu | GPS/GLONASS/Galileo/BeiDou + ISB推定 |
| Doppler | doppler.cu | ドップラー速度WLS推定、GPU batch |
| RAIM/FDE | raim.cu | χ²検定 + 衛星除外再計算 + HPL/VPL |

### 5.2 パーティクルフィルタ (6モジュール)
| モジュール | ファイル | 説明 |
|-----------|---------|------|
| Predict | predict.cu | 定速モデル + cuRAND noise |
| Weight | weight.cu | 擬似距離尤度（shared memory衛星データ） |
| Weight 3D | weight_3d.cu | レイトレース統合NLOS-aware尤度 |
| Resampling | resampling.cu | Systematic + Megopolis |
| SVGD | svgd.cu | Stein Variational Gradient Descent |
| PF Device | pf_device.cu | GPU常駐メモリ + CUDA Streams |

### 5.3 信号処理 (3モジュール)
| モジュール | ファイル | 説明 |
|-----------|---------|------|
| Acquisition | acquisition.cu | cuFFTベースGPS L1 C/A捕捉（PRN 1-32） |
| Tracking | tracking.cu | GPU correlator + DLL/PLL + VTL EKF |
| Interference | interference.cu | STFT + ノッチフィルタ除去 |

### 5.4 都市環境 (4モジュール)
| モジュール | ファイル | 説明 |
|-----------|---------|------|
| Ray Tracing | raytrace.cu | Möller-Trumbore LOS/NLOS判定 |
| BVH | bvh.cu | SAH構築 + GPU スタックトラバーサル |
| Multipath Sim | multipath.cu | DLLエラーモデル + 反射計算 |
| Vulnerability Map | skyplot.cu | DOP/可視衛星数グリッド並列評価 |

### 5.5 補正 (3モジュール)
| モジュール | ファイル | 説明 |
|-----------|---------|------|
| Ephemeris | ephemeris.cu | IS-GPS-200衛星位置計算（31衛星サブnm精度検証済み） |
| Atmosphere | atmosphere.cu | Saastamoinen対流圏 + Klobuchar電離層 |
| SBAS/QZSS | sbas.py | WAAS/MSAS/EGNOS + みちびきCLAS（Python） |

### 5.6 品質管理 (2モジュール)
| モジュール | ファイル | 説明 |
|-----------|---------|------|
| Cycle Slip | cycle_slip.py | Geometry-free / Melbourne-Wubbena / Time-diff |
| RAIM/FDE | raim.cu | (5.1と共有) |

### 5.7 I/O (6モジュール)
| モジュール | ファイル | 説明 |
|-----------|---------|------|
| RINEX OBS | rinex.py | RINEX 3.x観測ファイルパーサ |
| RINEX NAV | nav_rinex.py | RINEX 2/3 航法ファイルパーサ |
| NMEA Reader | nmea.py | GGA/RMC パーサ |
| NMEA Writer | nmea_writer.py | GGA/RMC/GSA/GSV/VTG生成 |
| PLATEAU | plateau.py | CityGML → BuildingModel（ガウスクリューゲル逆変換） |
| CityGML | citygml.py | 汎用CityGMLパーサ |

---

## 6. 性能ベンチマーク

### 6.1 測定環境
- **GPU**: NVIDIA Ada Lovelace世代 消費者向けGPU (16GB VRAM)
- **CUDA**: 12.0

### 6.2 結果

| Module | Input Size | Time (ms) | Throughput |
|--------|-----------|-----------|------------|
| WLS Batch | 10K epochs | 1.04 | **9.60M epoch/s** |
| Particle Filter | 1M particles | 81.44 | **12.28M part/s** |
| Signal Acquisition | 32 PRN, 1ms | 142.50 | **224.6 PRN/s** |
| Vulnerability Map | 100x100 grid | 0.62 | **16.14M pts/s** |
| Ray Tracing | 1008 tri, 8 sats | 0.71 | **11.32M checks/s** |

### 6.3 リアルタイム性
- 100万パーティクルフルパイプライン: **12Hz**
- ParticleFilterDevice（メモリ常駐）使用時: **推定100-200Hz**（H2D/D2H削減）
- WLS: 1エポック0.1μs → **リアルタイムに対して9,600倍の余裕**

---

## 7. 検証結果

### 7.1 gnss_lib_py (Stanford NavLab) との比較

| 検証項目 | 結果 | 許容誤差 |
|----------|------|---------|
| GPS定数 (WGS84, c, μ, ωe) | 完全一致 | 0 |
| ECEF ↔ LLA変換 | 一致 | < 1e-5° |
| WLS測位（ノイズなし） | 一致 | < 0.01m |
| DOP (HDOP/VDOP/PDOP) | 一致 | < 10% |
| 対流圏遅延 | 妥当 | ~2.3m at zenith |
| 電離層遅延 | 昼>夜 | 正の遅延 |
| C/Aコード (PRN 1-32) | IS-GPS-200準拠 | 完全一致 |
| Multi-GNSS ISB | 復元 | < 1m |

### 7.2 実ブロードキャストエフェメリスでの衛星位置検証

- **データ**: 2024-01-15 GPS broadcast NAV (brdc0150.24n)
- **検証衛星数**: 31/31 (G01-G32, G27欠)
- **平均位置誤差**: **0.020 nm** (ナノメートル)
- **最大位置誤差**: **0.056 nm**
- **IS-GPS-200参照計算とサブナノメートルで一致**

### 7.3 テストカバレッジ
- **359テスト全パス**, 0 failed, 7 skipped
- 24件のgnss_lib_py比較検証テスト
- 15件のCUDA Streamsテスト
- CUDA_CHECKマクロによる全CUDA API呼び出しのエラー検出

---

## 8. 論文計画

### 8.1 ターゲット論文

**Title**: "GPU-Resident Large-Scale Particle Filtering with 3D Ray-Tracing Likelihoods for Urban GNSS Positioning"

**Key Claim**: GPU常駐の大規模GNSS粒子フィルタに、3D建物モデルからのレイトレース尤度とGNSS向け粒子更新を統合する。既存のGNSS PF / 3DMA GNSS / shadow matching を踏まえた上で、規模、計算時間、外れ値耐性の観点から WLS・EKF・RTK-like・PF系ベースラインと比較評価する。

**ターゲット会議**: IROS 2027 (締切 2027年3月頃) / ION GNSS+ 2026 (締切 2026年6月頃)

### 8.2 必要な実験（論文化の最大ボトルネック）

#### 実データセット（致命的に不足）
1. **PPC-Dataset** (`taroz/PPC-Dataset`) — 現在の主実データ。real-data evaluation と tail解析を継続
2. **UrbanNav** (Hong Kong/Tokyo) — 外部妥当性のため追加候補
3. **Google Smartphone Decimeter Challenge** — スマホGNSS系の追加候補
4. **自前収集** — 東京都市部（u-blox F9P + RTK真値）3コース以上
5. **PLATEAUモデル** — 対応エリアのCityGML LOD2

#### ベースライン
| 手法 | 実装状態 | 備考 |
|------|---------|------|
| WLS | ✅ 実装済み | |
| EKF | ✅ 実装済み | |
| RTKLIB (SPP) | 未実装 | pyrtklib経由で比較 |
| CPU-PF 2K (Suzuki再現) | 部分実装 | パーティクル数制限版 |
| GPU-PF 1M (3Dモデルなし) | ✅ 実装済み | アブレーション |
| GPU-PF 1M (3Dモデルあり) | ✅ 実装済み | 提案手法 |
| Shadow Matching | 未実装 | Miura手法の再現 |

#### アブレーション実験
1. パーティクル数スケーリング: 1K, 10K, 100K, 500K, 1M
2. 3Dモデル有無: weight.cu vs weight_3d.cu
3. SVGD有無: SVGD vs Megopolis vs Systematic
4. NLOS尤度パラメータ感度: sigma_nlos, nlos_bias
5. BVH有無: 大規模三角形メッシュでのスケーリング
6. GPU種類: RTX 4070, Jetson Orin

#### 評価指標
- 2D/3D RMS誤差
- 50/67/95パーセンタイル誤差
- CDF曲線
- 最大誤差
- 100m超外れ値率
- 500m超catastrophic rate
- 最長failure segment長・時間
- 処理時間/epoch
- ESS推移

#### 統計的有意性
- 複数走行の平均 ± 標準偏差
- Wilcoxon符号順位検定
- 5回以上の独立試行（乱数シード変更）

### 8.3 想定されるレビュアの懸念と対策

| 懸念 | 対策 |
|------|------|
| MegaParticlesの単純適用 | GNSS固有の状態空間、擬似距離尤度、clock-bias common-mode除去、3Dレイトレース統合を切り分けて説明 |
| 新規性主張が強すぎる | 個別要素の「初」は避け、組み合わせ・規模・実証の貢献として記述する |
| 3Dモデルの入手性・精度に依存 | PLATEAUで日本全国LOD2が無料、LOD精度の感度分析を含める、モデルなしでも動作 |
| 計算コストが大きい | 消費者向けGPUで12Hz、Jetsonベンチマーク追加、ParticleFilterDeviceで5-10ms見込み |
| 実データでの検証不十分 | PPC-Dataset全runで評価済み。UrbanNav / 自前収集で外部データ追加 |

### 8.4 アクションプラン

```
Week 1:   PPC-Dataset full-run の catastrophic outlier 抑制（n_sat閾値, fallback, RAIM連携）
Week 2:   WLS/EKF/RTK-like の tail-aware 再評価（RMSだけでなく p95 / >100m / >500m）
Week 3-4: PLATEAU対応エリアの3Dモデル取得 + 3D-PF評価
Week 5-6: アブレーション実験（パーティクル数、3D有無、SVGD vs リサンプリング）
Week 7:   UrbanNav など外部データで再評価、Jetson Orinベンチマーク
Week 8:   論文ドラフト
```

---

## 9. ディレクトリ構成

```
gnss_gpu/
├── CMakeLists.txt
├── pyproject.toml              ← scikit-build-core, pip install対応
├── README.md
├── LICENSE                     ← Apache-2.0
├── MANIFEST.in
├── .github/workflows/ci.yml   ← GitHub Actions CI
│
├── docs/
│   └── design.md              ← 本文書
│
├── include/gnss_gpu/
│   ├── cuda_check.h            ← CUDA_CHECK マクロ（throw on error）
│   ├── positioning.h, ekf.h, rtk.h, multi_gnss.h, doppler.h, raim.h
│   ├── particle_filter.h, pf_3d.h, pf_device.h, svgd.h
│   ├── acquisition.h, tracking.h, interference.h
│   ├── raytrace.h, bvh.h, multipath.h, skyplot.h
│   ├── ephemeris.h, atmosphere.h
│   └── coordinates.h
│
├── src/
│   ├── positioning/            ← wls.cu, multi_gnss.cu
│   ├── ekf/                    ← ekf.cu
│   ├── rtk/                    ← rtk.cu (+ LAMBDA Schnorr-Euchner)
│   ├── doppler/                ← doppler.cu
│   ├── raim/                   ← raim.cu
│   ├── particle_filter/        ← predict.cu, weight.cu, weight_3d.cu,
│   │                              resampling.cu, svgd.cu, pf_device.cu
│   ├── raytrace/               ← raytrace.cu, bvh.cu
│   ├── multipath/              ← multipath.cu
│   ├── skyplot/                ← skyplot.cu
│   ├── acquisition/            ← acquisition.cu (cuFFT)
│   ├── interference/           ← interference.cu (cuFFT STFT)
│   ├── tracking/               ← tracking.cu (C/Aコード定数メモリ)
│   ├── ephemeris/              ← ephemeris.cu (IS-GPS-200)
│   ├── atmosphere/             ← atmosphere.cu (Saastamoinen + Klobuchar)
│   └── utils/                  ← coordinates.cu
│
├── python/gnss_gpu/
│   ├── __init__.py, _version.py
│   ├── _*_bindings.cpp         ← 15+ pybind11バインディングファイル
│   ├── particle_filter.py, particle_filter_3d.py, particle_filter_device.py
│   ├── svgd.py, ekf.py, rtk.py, multi_gnss.py, doppler.py
│   ├── raytrace.py, bvh.py, multipath.py, skyplot.py
│   ├── acquisition.py, tracking.py, interference.py
│   ├── ephemeris.py, atmosphere.py, raim.py, sbas.py, cycle_slip.py
│   ├── io/                     ← rinex.py, nav_rinex.py, nmea.py,
│   │                              nmea_writer.py, plateau.py, citygml.py
│   ├── viz/                    ← plots.py (matplotlib), interactive.py (plotly)
│   └── ros2/                   ← gnss_node.py, launch/, rviz_config.py
│
├── tests/                      ← 359テスト (25ファイル)
├── examples/                   ← 6デモスクリプト
├── benchmarks/                 ← 6ベンチマーク + RESULTS.md
├── data/                       ← sample_building.obj, sample_plateau.gml, etc.
└── output/                     ← 生成ファイル出力先
```

---

## 10. 参考文献

### 直接関連（本研究の基盤）
- Koide et al., "MegaParticles: Range-based 6-DoF Monte Carlo Localization with GPU-Accelerated Stein Particle Filter", ICRA 2024. arXiv:2404.16370
- Nakao et al., "Range-based 6-DoF Monte Carlo SLAM with Gradient-guided PF on GPU", ICRA 2025. arXiv:2504.18056
- Chesser et al., "Megopolis Resampling", 2021. arXiv:2109.13504
- Liu & Wang, "Stein Variational Gradient Descent", NeurIPS 2016. arXiv:1608.04471

### GNSS パーティクルフィルタ
- Suzuki, "Multiple Update Particle Filter: Position Estimation by Combining GNSS Pseudorange and Carrier Phase Observations", 2024. arXiv:2403.03394
- Gupta and Gao, "Reliable GNSS Localization Against Multiple Faults Using a Particle Filter Framework", 2021. arXiv:2101.06380
- Suzuki & Kubo, "Integration of GNSS and 3D Map using Particle Filter", ION GNSS 2016
- Murray et al., "Parallel Resampling in the Particle Filter", 2013. arXiv:1301.4019

### 3D建物モデル × GNSS
- Groves, "Shadow Matching: A New GNSS Positioning Technique for Urban Canyons", Journal of Navigation, 2011
- Adjrad & Groves, "Intelligent Urban Positioning: Integration of Shadow Matching with 3D-Mapping-Aided GNSS Ranging", Journal of Navigation, 2017
- Hsu & Wen, "3D Mapping Database Aided GNSS Based Collaborative Positioning Using FGO", IEEE T-ITS 2021
- Zhong, "Investigation of 3D-Mapping-Aided GNSS Navigation in Urban Canyons", UCL PhD thesis, 2024
- Neamati et al., "Neural City Maps for GNSS Shadow Matching", ION GNSS+ 2024
- Ketzler & Althoff, "Zonotope Shadow and Reflection Matching", TU Munich 2026. arXiv:2601.10727

### GNSS マルチパス・NLOS
- O'Connor et al., "Low-latency GNSS multipath simulation in urban environments", Simulation 2024
- Zheng et al., "GNSS Satellite Visibility Graph Transformer", NAVIGATION 2024
- 3D Map-Aided Smartphone GNSS + FGO, Satellite Navigation 2025

### GPU GNSS
- GNSS-SDR (gnss-sdr.org) — オープンソースSDR
- gps-sdr-sim (osqzss) — GPS信号シミュレータ

### 標準・データ
- IS-GPS-200 (GPS Interface Specification)
- RTKLIB (tomojitakasu/RTKLIB)
- gnss_lib_py (Stanford NavLab)
- Project PLATEAU (mlit.go.jp/plateau)
- UrbanNav Dataset (Hong Kong/Tokyo)
