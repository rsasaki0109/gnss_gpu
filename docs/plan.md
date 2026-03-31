# gnss_gpu 開発計画・Codex引き継ぎドキュメント

**最終更新**: 2026-04-01
**ステータス**: コア実装完了 → 論文化フェーズへ移行

---

## 目次

1. [プロジェクト現状サマリー](#1-プロジェクト現状サマリー)
2. [実装完了モジュール一覧](#2-実装完了モジュール一覧)
3. [テスト・検証状況](#3-テスト検証状況)
4. [既知の問題・技術的負債](#4-既知の問題技術的負債)
5. [論文計画](#5-論文計画)
6. [次のアクション（優先度順）](#6-次のアクション優先度順)
7. [実データ実験計画](#7-実データ実験計画)
8. [コードベースガイド](#8-コードベースガイド)
9. [ビルド・テスト手順](#9-ビルドテスト手順)
10. [設計判断の記録](#10-設計判断の記録)

---

## 1. プロジェクト現状サマリー

### 一言で
MegaParticles (Koide, ICRA 2024) のGPU大規模PF+SVGDをGNSS測位に世界初適用するライブラリ。3D都市モデル(PLATEAU)からのリアルタイムレイトレースをPF尤度に統合。コア実装は完了、**論文化に必要な実データ実験が未着手**。

### 数値で

| 指標 | 値 |
|------|-----|
| テスト数 | 359 passed, 0 failed, 7 skipped |
| ファイル数 | 155+ |
| コード行数 | ~30,000 |
| CUDAモジュール | 20+ (.cu files) |
| Python モジュール | 25+ (.py files) |
| pybind11 バインディング | 15+ (.cpp files) |
| コミット数 | 12 (main branch) |
| ベンチマーク | 1M particles @ 12Hz, WLS @ 9.6M epoch/s |
| 検証 | gnss_lib_py (Stanford) 比較済み、31衛星サブnm精度 |

### gitログ
```
a71fec6 Remove GPU model references from docs and benchmarks
XXXXXXX Update design.md with full architecture, research analysis, and paper plan
XXXXXXX Fix pf_device crashes, add real-data demos and benchmarks
XXXXXXX Add verification tests against gnss_lib_py (Stanford NavLab)
XXXXXXX Fix cycle slip test thresholds and pseudorange noise levels
XXXXXXX Add cycle slip, RAIM, Doppler, SBAS/QZSS, BVH, CUDA streams, pip support
XXXXXXX Add full GPU-accelerated GNSS processing library
XXXXXXX Add Phase 1: WLS positioning, coordinate utils, RINEX/NMEA parsers
XXXXXXX Initial project structure with CUDA + Python bindings
XXXXXXX Initial commit
```

---

## 2. 実装完了モジュール一覧

### 2.1 測位エンジン

| モジュール | CUDA | Python | Bindings | テスト | 状態 |
|-----------|------|--------|----------|-------|------|
| WLS Batch | `src/positioning/wls.cu` | `_bindings.cpp` | ✅ | ✅ | 完成・検証済み |
| EKF | `src/ekf/ekf.cu` | `ekf.py` | ✅ | ✅ | 完成（GPU bindingsにstate copy問題あり→CPU fallback使用中） |
| RTK + LAMBDA | `src/rtk/rtk.cu` | `rtk.py` | ✅ | ✅ | 完成（Schnorr-Euchner探索実装済み） |
| Multi-GNSS ISB | `src/positioning/multi_gnss.cu` | `multi_gnss.py` | ✅ | ✅ | 完成 |
| Doppler速度 | `src/doppler/doppler.cu` | `doppler.py` | ✅ | ✅ | 完成 |
| RAIM/FDE | `src/raim/raim.cu` | `raim.py` | ✅ | ✅ | 完成 |

### 2.2 パーティクルフィルタ（コア研究対象）

| モジュール | ファイル | 状態 | 備考 |
|-----------|---------|------|------|
| Predict | `predict.cu` | ✅ 完成 | cuRAND Philox、定速モデル |
| Weight | `weight.cu` | ✅ 完成 | shared memory衛星データ、log-space |
| Weight 3D | `weight_3d.cu` | ✅ 完成 | **論文の核心**：Möller-Trumboreレイトレース統合尤度 |
| Resampling | `resampling.cu` | ✅ 完成 | Systematic + Megopolis（double-buffer） |
| SVGD | `svgd.cu` | ✅ 完成 | K=32ランダム近傍、common-mode除去 |
| PF Device | `pf_device.cu` | ✅ 完成 | GPU常駐メモリ + CUDA Streams + pinned memory |

### 2.3 信号処理

| モジュール | ファイル | 状態 | 備考 |
|-----------|---------|------|------|
| Acquisition | `acquisition.cu` | ✅ 完成 | cuFFT、PRN 1-32 C/Aコード |
| Tracking | `tracking.cu` | ✅ 完成 | GPU correlator + DLL/PLL + VTL EKF、正規C/Aコード（constant memory） |
| Interference | `interference.cu` | ✅ 完成 | STFT + ノッチフィルタ除去 |

### 2.4 都市環境

| モジュール | ファイル | 状態 | 備考 |
|-----------|---------|------|------|
| Ray Tracing | `raytrace.cu` | ✅ 完成 | Möller-Trumbore、2フェーズmultipath（race condition修正済み） |
| BVH | `bvh.cu` | ✅ 完成 | SAH構築 + GPU スタックトラバーサル |
| Multipath Sim | `multipath.cu` | ✅ 完成 | DLLエラーモデル |
| Vulnerability Map | `skyplot.cu` | ✅ 完成 | DOP並列計算 |

### 2.5 補正・品質管理

| モジュール | ファイル | 状態 |
|-----------|---------|------|
| Ephemeris | `ephemeris.cu` | ✅ 完成（IS-GPS-200、31衛星サブnm検証） |
| Atmosphere | `atmosphere.cu` | ✅ 完成（Saastamoinen + Klobuchar） |
| SBAS/QZSS | `sbas.py` (Python) | ✅ 完成 |
| Cycle Slip | `cycle_slip.py` (Python) | ✅ 完成 |

### 2.6 I/O

| モジュール | ファイル | 状態 |
|-----------|---------|------|
| RINEX OBS | `io/rinex.py` | ✅ 完成 |
| RINEX NAV | `io/nav_rinex.py` | ✅ 完成 |
| NMEA R/W | `io/nmea.py`, `nmea_writer.py` | ✅ 完成 |
| PLATEAU CityGML | `io/plateau.py`, `citygml.py` | ✅ 完成 |

### 2.7 統合・インフラ

| モジュール | 状態 | 備考 |
|-----------|------|------|
| ROS2 Node | ✅ 完成 | NavSatFix, PointCloud2 |
| Visualization | ✅ 完成 | matplotlib 9関数 + plotly 3関数 |
| CI/CD | ✅ 完成 | GitHub Actions (Python test, CUDA build, lint) |
| pip install | ✅ 完成 | scikit-build-core |
| CUDA_CHECK | ✅ 完成 | 全.cuファイルにthrow-on-errorマクロ適用 |
| Benchmarks | ✅ 完成 | 6ベンチマーク + RESULTS.md |

---

## 3. テスト・検証状況

### 3.1 テスト概要
```
359 passed, 0 failed, 7 skipped
```

**skipped の内訳**:
- 2件: mpl_toolkits.mplot3d 環境依存（matplotlib 3D投影がこの環境で壊れている）
- 5件: SVGD large-scale テスト（`@pytest.mark.slow` — 実行に時間がかかる）

### 3.2 gnss_lib_py (Stanford NavLab) 比較検証

`tests/test_verification.py` に24件の比較テスト:

| 項目 | 比較結果 |
|------|---------|
| GPS定数 | 完全一致 |
| ECEF ↔ LLA | < 1e-5° |
| WLS測位 | < 0.01m |
| DOP | < 10% |
| 対流圏遅延 | 妥当（~2.3m @ zenith） |
| 電離層遅延 | 昼 > 夜（正の遅延） |
| C/Aコード | IS-GPS-200完全一致 |
| Multi-GNSS ISB | 15m注入 → < 1m誤差で復元 |

### 3.3 実ブロードキャストエフェメリス検証

- **データ**: 2024-01-15 GPS broadcast NAV (`brdc0150.24n`, `/tmp/gnss_eph/`にダウンロード済み)
- **検証衛星**: 31/31
- **平均誤差**: 0.020 nm
- **最大誤差**: 0.056 nm
- **結論**: IS-GPS-200参照計算とサブナノメートルで一致

### 3.4 未検証事項（論文化のブロッカー）

- ❌ **実GNSSデータでの測位精度評価**（合成データのみ）
- ❌ **実都市環境でのPF3D vs WLS/EKF比較**
- ❌ **UrbanNavデータセットでの評価**
- ❌ **Jetson Orinでのベンチマーク**

---

## 4. 既知の問題・技術的負債

### 4.1 Critical（論文前に修正すべき）

| # | 問題 | 影響 | 対策 |
|---|------|------|------|
| C1 | `predict.cu`/`weight.cu`/`resampling.cu`の毎回cudaMalloc/Free | 性能劣化（81ms中の大半が転送オーバーヘッド） | `ParticleFilterDevice`（pf_device.cu）を使用すれば解消。論文ではDevice版の性能を報告すべき |
| C2 | `weight_3d.cu`の三角形線形探索 | 大規模メッシュで遅い | `bvh.cu`のBVH版weight_3dカーネルが未実装。bvh.cuのlos_checkは実装済みなので、weight_3dにBVHトラバーサルを統合する必要あり |
| C3 | EKF GPU bindings のstate copy問題 | GPU EKFが使えない（CPU fallbackで動作中） | pybind11のEKFState holder typeを修正（py::array_tのコンストラクタ問題は修正済みだが、EKFState構造体の値コピーが不完全）。`_HAS_NATIVE = False`で回避中 |

### 4.2 Important（品質向上）

| # | 問題 | 影響 |
|---|------|------|
| I1 | pybind11全bindings: 入力配列のshapeバリデーションなし | 不正shapeでセグフォの可能性 |
| I2 | 高緯度でのECEF→LLA数値不安定（cos(lat)→0） | 極付近で高度計算が発散 |
| I3 | tracking.cuのVTL: 衛星16機超でS行列オーバーフロー | n_locked > 16でinvert_matrixが拒否 |
| I4 | RTK solve_fixed: ambiguity共分散が単位行列ハードコード | LAMBDA結果が不正確 |
| I5 | スレッドセーフティなし | ROS2マルチスレッドで問題 |

### 4.3 Minor（改善推奨）

| # | 問題 |
|---|------|
| M1 | 一部.cuファイルでCUDA_CHECK適用がsed由来で不完全な箇所あり（手動修正で対応済みだが網羅性要確認） |
| M2 | `multipath.cu`のFresnel係数が定数0.5（入射角依存なし） |
| M3 | `io/rinex.py`で`import re`が未使用 |
| M4 | デモスクリプトの一時ファイルが削除されない |

### 4.4 pybind11 py::array_t問題（解決済み・記録用）

**発見した重大バグ**: `py::array_t<double>(n)`（int変数を渡す）が意図しないオーバーロードにマッチし、配列の全要素が[0]の値になる。

**修正**: 全bindings（8ファイル）で `py::array_t<double>(std::vector<ssize_t>{n})` に統一。EKFの`get_state()`/`get_covariance()`も`request().ptr`でmemcpyに変更。

**影響範囲**: `_multi_gnss_bindings.cpp`, `_ekf_bindings.cpp`, `_rtk_bindings.cpp`, `_skyplot_bindings.cpp`, `_ephemeris_bindings.cpp`, `_interference_bindings.cpp`, `_multipath_bindings.cpp`, `_raytrace_bindings.cpp`, `_tracking_bindings.cpp`

---

## 5. 論文計画

### 5.1 ターゲット

**Title**: "GPU-Accelerated Mega-Particle Filter with Real-Time 3D Ray Tracing for Robust Urban GNSS Positioning"

**ターゲット会議**:
- **第一候補**: IROS 2027 (締切 2027年3月頃)
- **第二候補**: ION GNSS+ 2026 (締切 2026年6月頃)
- **ジャーナル**: IEEE RA-L (随時投稿可)

### 5.2 新規性（2026年3月時点の網羅的サーベイに基づく）

**5つの「世界初」**:

| # | 世界初の主張 | 根拠 |
|---|------------|------|
| 1 | GPU PF × GNSS | GPUパーティクルフィルタのGNSS適用は皆無（LiDARのみ） |
| 2 | SVGD × GNSS | SVGDのGNSS測位適用は皆無 |
| 3 | 100万パーティクル × GNSS | メガスケールPFはLiDARのみ（Koide ICRA2024） |
| 4 | PLATEAU × GNSS | PLATEAUの3D都市モデルのGNSS測位利用なし |
| 5 | GPU Ray Tracing × GNSS PF尤度 | GPUレイトレースとPF尤度の統合なし |

**論文で主張する3本柱**:
1. GNSS初の100万パーティクルGPUフィルタ（12Hz、消費者向けGPU）
2. PF尤度関数内リアルタイム3Dレイトレース（NLOS-awareヘテロジニアスガウシアン尤度）
3. SVGDのGNSSスコア関数導出（クロックバイアスcommon-mode除去）

### 5.3 競合研究（最新）

| 研究 | 会議/年 | 手法 | 限界 |
|------|---------|------|------|
| MegaParticles (Koide) | ICRA 2024 | GPU SVGD PF, LiDAR | GNSSなし |
| Nakao et al. | ICRA 2025 | GPU PF SLAM | GNSSなし |
| Suzuki | ICRA 2024 | Multiple Update PF, GNSS | CPU, 2K particles |
| Suzuki & Kubo | ION 2016 | 3D model + PF, GNSS | CPU, レイトレース遅い |
| Groves (UCL) | 複数 | Shadow Matching | グリッドベース, CPUのみ |
| Hsu & Wen (PolyU) | T-ITS 2021 | Factor Graph + 3DMA | CPUのみ |
| Neamati (Stanford) | ION GNSS+ 2024 | Neural City Maps | NeRFベース, 計算コスト大 |
| Ketzler (TU Munich) | 2026 | Zonotope Shadow/Reflection | 集合ベース, 確率的表現なし |
| DL系 (多数) | 2024-2025 | Graph Transformer, CNN等 | 学習データ依存 |

### 5.4 想定レビュアー懸念と対策

| 懸念 | 対策 |
|------|------|
| 「MegaParticlesの単純適用」 | GNSS固有: クロックバイアス状態空間、3Dレイトレース統合尤度、SVGDスコア関数のcommon-mode除去を強調 |
| 「3Dモデルに依存しすぎ」 | PLATEAUで日本全国LOD2が無料、LOD精度の感度分析を含める、モデルなしでも標準PFとして動作 |
| 「計算コスト大」 | 消費者GPUで12Hz、Jetsonベンチ追加、ParticleFilterDeviceで5-10ms見込み |
| 「実データ不足」 | UrbanNav + 自前収集で3都市5走行以上 |
| 「DLに劣る」 | DLは学習データ依存・汎化問題。PFはモデルベースで環境変化にロバスト。相補的 |

---

## 6. 次のアクション（優先度順）

### P0: 論文化ブロッカー（実データ実験）

#### 6.1 UrbanNavデータセットのダウンロードと評価
- **URL**: https://github.com/weisongwen/UrbanNavDataset
- **対象**: Tokyo/Hong Kong の都市走行データ
- **やること**:
  1. データローダー実装（`python/gnss_gpu/io/urbannav.py`）
  2. WLS/EKFベースライン評価
  3. GPU-PF（3Dモデルなし）評価
  4. GPU-PF（3Dモデルあり）評価
- **出力**: CDF曲線、エポック別誤差、統計量

#### 6.2 自前データ収集
- **機材**: u-blox F9P (or ZED-F9R) + RTK基準局
- **場所**: 東京都市部3コース（新宿/渋谷/丸の内）
- **走行**: 各コース3回以上
- **真値**: RTK-fixed解（基準局から1km以内）

#### 6.3 PLATEAUモデル取得
- **URL**: https://www.geospatial.jp/ckan/dataset?q=plateau
- **対象**: UrbanNav/自前収集エリアのLOD2建物モデル
- **処理**: CityGML → BuildingModel → weight_3d.cu用三角形メッシュ

### P1: 実験基盤整備

#### 6.4 BVH統合weight_3dカーネル
- **現状**: `weight_3d.cu`はO(n_tri)線形探索。`bvh.cu`のLOS checkは別モジュール
- **やること**: `weight_3d_bvh.cu`を作成。BVHトラバーサルをweight計算内に統合
- **理由**: PLATEAU LOD2の三角形数が数万〜数十万になるため必須

#### 6.5 ParticleFilterDeviceのベンチマーク
- **現状**: `pf_device.cu`は完成・テスト通過。ベンチマーク未実施
- **やること**: `benchmarks/bench_pf_device.py`を作成。標準PFとの性能比較
- **期待**: 81ms → 5-10ms（H2D/D2H削減効果）

#### 6.6 Jetson Orinベンチマーク
- **理由**: 消費者GPUだけでなくエッジデバイスでのリアルタイム性を論文で主張
- **やること**: Jetson Orin NX/Nanoでの全ベンチマーク実行

### P2: アブレーション実験

#### 6.7 パーティクル数スケーリング
- 1K, 10K, 100K, 500K, 1M → 精度 vs 計算時間のパレートフロント
- 実データ上で実施

#### 6.8 3Dモデル有無比較
- `weight.cu`（標準尤度）vs `weight_3d.cu`（3D-aware尤度）
- NLOS環境での精度差を定量化

#### 6.9 SVGD vs リサンプリング比較
- SVGD vs Megopolis vs Systematic
- マルチモーダル環境でのモード捕捉能力
- 少パーティクル数での効率比較

#### 6.10 NLOS尤度パラメータ感度
- sigma_los: 1, 3, 5, 10 m
- sigma_nlos: 10, 20, 30, 50 m
- nlos_bias: 5, 10, 20, 40 m
- グリッドサーチ → 最適パラメータ報告

### P3: 論文執筆

#### 6.11 論文構成（案）
```
I.   Introduction
     - Urban GNSS multipath problem
     - Particle filters for non-Gaussian GNSS
     - GPU acceleration for mega-scale particles
II.  Related Work
     - GNSS particle filters (Suzuki)
     - 3D model GNSS (Shadow matching, FGO)
     - GPU particle filters (MegaParticles)
III. Proposed Method
     A. GPU Mega-Particle Filter for GNSS
        - State space, motion model, pseudorange likelihood
     B. 3D Ray Tracing Integrated Likelihood
        - NLOS-aware heterogeneous Gaussian
        - BVH-accelerated Möller-Trumbore
     C. SVGD for GNSS
        - Score function derivation
        - Clock bias common-mode removal
IV.  Implementation
     - CUDA kernel design (SoA, shared memory, log-space)
     - CUDA Streams + persistent device memory
V.   Experiments
     A. Datasets: UrbanNav + self-collected
     B. Baselines: WLS, EKF, CPU-PF, Shadow Matching
     C. Results: accuracy, real-time performance, ablation
VI.  Conclusion
```

---

## 7. 実データ実験計画

### 7.1 データセット

| データセット | 場所 | 取得方法 | 3Dモデル | 備考 |
|------------|------|---------|---------|------|
| UrbanNav | Hong Kong TST | ダウンロード | OSM/Google 3D | 公開、即利用可 |
| UrbanNav | Tokyo | ダウンロード | PLATEAU LOD2 | 公開、即利用可 |
| 自前収集 | 新宿 | u-blox F9P | PLATEAU LOD2 | RTK真値必要 |
| 自前収集 | 渋谷 | u-blox F9P | PLATEAU LOD2 | RTK真値必要 |
| 自前収集 | 丸の内 | u-blox F9P | PLATEAU LOD2 | RTK真値必要 |
| Google SDC | 米国各地 | ダウンロード | なし | スマホデータ |

### 7.2 UrbanNavデータローダー（実装予定）

```python
# python/gnss_gpu/io/urbannav.py

class UrbanNavLoader:
    def __init__(self, data_dir):
        """Load UrbanNav dataset from directory."""
        ...

    def load_gnss(self):
        """Load GNSS observations (pseudoranges, carrier phase, Doppler)."""
        ...

    def load_ground_truth(self):
        """Load RTK ground truth trajectory."""
        ...

    def load_imu(self):
        """Load IMU data (optional, for comparison)."""
        ...

    def epochs(self):
        """Iterator over synchronized GNSS+truth epochs."""
        ...
```

### 7.3 実験スクリプト構成（実装予定）

```
experiments/
├── exp_urbannav_baseline.py     ← WLS/EKF/RTKLIBベースライン
├── exp_urbannav_pf.py           ← GPU-PF (3Dモデルなし)
├── exp_urbannav_pf3d.py         ← GPU-PF (3Dモデルあり)
├── exp_ablation_particles.py    ← パーティクル数スケーリング
├── exp_ablation_3d.py           ← 3Dモデル有無
├── exp_ablation_svgd.py         ← SVGD vs リサンプリング
├── exp_ablation_nlos_params.py  ← NLOS尤度パラメータ
├── exp_benchmark_jetson.py      ← Jetson Orinベンチマーク
└── results/                     ← 結果CSV/図/テーブル
```

### 7.4 評価指標

```python
def evaluate(estimated_positions, ground_truth):
    """Compute all evaluation metrics."""
    errors_2d = np.sqrt((est[:, 0] - gt[:, 0])**2 + (est[:, 1] - gt[:, 1])**2)
    errors_3d = np.linalg.norm(est - gt, axis=1)

    return {
        'rms_2d': np.sqrt(np.mean(errors_2d**2)),
        'rms_3d': np.sqrt(np.mean(errors_3d**2)),
        'mean_2d': np.mean(errors_2d),
        'std_2d': np.std(errors_2d),
        'p50': np.percentile(errors_2d, 50),
        'p67': np.percentile(errors_2d, 67),
        'p95': np.percentile(errors_2d, 95),
        'max': np.max(errors_2d),
        'cdf': np.sort(errors_2d),  # for CDF curve
    }
```

---

## 8. コードベースガイド

### 8.1 重要ファイル（変更頻度が高い）

| ファイル | 役割 | 変更時の注意 |
|---------|------|------------|
| `CMakeLists.txt` | ビルド設定 | 新モジュール追加時に更新必要 |
| `python/gnss_gpu/__init__.py` | パッケージエクスポート | 新クラス追加時に更新必要 |
| `include/gnss_gpu/cuda_check.h` | CUDAエラーチェック | `std::runtime_error`をthrowする。変更注意 |
| `src/particle_filter/weight_3d.cu` | **論文の核心** | レイトレース+尤度統合カーネル |
| `src/particle_filter/svgd.cu` | **論文の核心** | SVGDグラディエントカーネル |

### 8.2 CUDAカーネルの共通パターン

全ホスト関数が以下のパターンを踏襲:
```cpp
void some_function(const double* input, double* output, int n) {
    double *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_in, input, n * sizeof(double), cudaMemcpyHostToDevice));

    int block = 256;
    int grid = (n + block - 1) / block;
    kernel<<<grid, block>>>(d_in, d_out, n);
    CUDA_CHECK_LAST();

    CUDA_CHECK(cudaMemcpy(output, d_out, n * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
}
```

**例外**: `pf_device.cu`はGPU常駐メモリ+CUDA Streams使用。

### 8.3 pybind11バインディングの注意点

**重要**: 配列作成時に `py::array_t<double>(n)` は使わない！必ず:
```cpp
auto arr = py::array_t<double>(std::vector<ssize_t>{n});
```
理由: `py::array_t<T>(int)` は意図しないコンストラクタオーバーロードにマッチし、全要素が同じ値になるバグが発生する。

### 8.4 テストの実行方法

```bash
# フルテスト
PYTHONPATH=python python3 -m pytest tests/ -q

# 特定モジュール
PYTHONPATH=python python3 -m pytest tests/test_particle_filter.py -v

# 検証テスト（gnss_lib_py必要）
PYTHONPATH=python python3 -m pytest tests/test_verification.py -v

# デモ実行
PYTHONPATH=python python3 examples/demo_full_pipeline.py
PYTHONPATH=python python3 examples/demo_real_data.py
PYTHONPATH=python python3 examples/demo_plateau_urban.py

# ベンチマーク
PYTHONPATH=python python3 benchmarks/bench_all.py
```

---

## 9. ビルド・テスト手順

### 9.1 依存関係

```
必須:
- CUDA Toolkit 12.0+
- CMake 3.18+
- pybind11
- Python 3.9+
- NumPy

オプション:
- matplotlib (可視化)
- plotly (インタラクティブ可視化)
- gnss-lib-py (検証テスト)
- rclpy (ROS2ノード)
```

### 9.2 ビルド

```bash
# クリーンビルド
rm -rf build && mkdir build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=native
make -j$(nproc)

# Pythonパッケージとしてインストール
pip install .

# 開発モード（.soをpythonディレクトリにコピー）
cp build/*.so python/gnss_gpu/
PYTHONPATH=python python3 -m pytest tests/ -q
```

### 9.3 テスト実行時の注意

- `PYTHONPATH=python` を必ず指定（`pip install`していない場合）
- `build/*.so` を `python/gnss_gpu/` にコピーする必要がある
- CUDA Stream テスト（`test_cuda_streams.py`）はGPU必須
- 検証テスト（`test_verification.py`）は`gnss-lib-py`必須

---

## 10. 設計判断の記録

### 10.1 なぜSoA（Structure of Arrays）か
パーティクル状態を`px[], py[], pz[], pcb[]`の4配列で保持。AoS（`{x,y,z,cb}[]`）と比較して:
- GPU coalesced memory access が可能（同一成分が連続メモリ）
- 帯域幅利用効率が高い
- predict/weight各カーネルが必要な成分のみアクセスできる

### 10.2 なぜdouble精度か
- GNSS擬似距離は数千万メートル（~2.6×10^7 m）
- float32の有効桁数7桁では1mオーダーの丸め誤差
- prefix-sum (CDF計算) は100万パーティクルで累積誤差が問題
- クロックバイアスは光速オーダー（~3×10^8 m/s × ~10μs = ~3km）

### 10.3 なぜMegopolisリサンプリングか
- prefix-sum不要 → double精度の累積誤差問題を回避
- coalesced memory access実現（ランダムアクセスパターンがSoAと相性良い）
- チューニングパラメータ不要
- 理論的収束保証あり (Chesser et al., 2021)

### 10.4 なぜSVGDのスコア関数でcommon-mode除去するか
GNSSの擬似距離残差は全衛星で共通のクロックバイアス誤差を含む。この共通成分が衛星方向ベクトルの重み付き和（≠ゼロ、受信機は衛星群の重心にいないため）を通じて位置勾配にリークする。デミーニング処理で共通成分を除去することで、位置スコアがクロックバイアスの影響を受けなくなり、SVGDの収束が安定化。

### 10.5 なぜCUDA_CHECKでthrowするか
当初はfprintf+continueだったが、レビューで指摘:
- cudaMalloc失敗後のNULLポインタ使用でセグフォ
- エラーが無言で伝播し、デバッグ困難
- pybind11がstd::runtime_errorをPython例外に自動変換するため、throwが適切

### 10.6 GPU型番をぼかす理由
論文・公開リポジトリではGPUの具体型番（RTX 4070 Ti SUPER等）を記載しない方針。理由:
- 特定ハードウェアへの依存印象を避ける
- 「消費者向けGPU」で十分なリアルタイム性を達成できることを強調
- 型番を出すとそのGPU固有の最適化と誤解される可能性
- benchmarks/RESULTS.md, docs/design.md, READMEでは「Ada Lovelace世代消費者向けGPU (16GB VRAM)」と記載

---

## 付録: Codexへの引き継ぎメモ

### このプロジェクトでの作業の注意点

1. **ビルド後に`.so`コピーが必要**: `cp build/*.so python/gnss_gpu/` を忘れない
2. **pybind11の`py::array_t`**: 絶対に`py::array_t<T>(n)`を使わない。`std::vector<ssize_t>{n}`を使う
3. **CUDA_CHECK**: 全新規.cuファイルに`#include "gnss_gpu/cuda_check.h"`を追加
4. **テスト実行**: `PYTHONPATH=python python3 -m pytest tests/ -q`
5. **GPU型番**: コード・ドキュメント・コミットメッセージに具体GPU型番を書かない
6. **Co-Authored-By**: gitコミットにCo-Authored-Byは付けない（ユーザー設定）

### すぐ始められるタスク

**最優先**: UrbanNavデータローダー（`python/gnss_gpu/io/urbannav.py`）
1. https://github.com/weisongwen/UrbanNavDataset からTokyo/HK dataをダウンロード
2. GNSS pseudorange + ground truth を読むローダーを実装
3. `experiments/exp_urbannav_baseline.py` でWLS/EKFベースライン評価
4. `experiments/exp_urbannav_pf3d.py` でGPU-PF3D評価

**次に**: BVH統合weight_3dカーネル（`src/particle_filter/weight_3d_bvh.cu`）
- `weight_3d.cu`の線形探索を`bvh.cu`のBVHトラバーサルに置き換え
- 大規模PLATEAUメッシュ（10K+三角形）でのスケーラビリティ確保

### 開発環境

- Ubuntu, bash shell
- CUDA 12.0, CMake
- Python 3.12
- GPU: 16GB VRAM
- gnss-lib-py インストール済み（pip）
- NAVファイル: `/tmp/gnss_eph/rinex/nav/brdc0150.24n` にダウンロード済み
