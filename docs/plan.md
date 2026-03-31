# gnss_gpu 完全引き継ぎドキュメント（Codex向け）

**最終更新**: 2026-04-01 21:00 JST
**ステータス**: コア実装完了（399テストpass）→ 論文化フェーズへ移行
**前任**: Claude Opus 4.6 (1Mコンテキスト)
**引き継ぎ先**: Codex

---

## 重要: まず読むべきファイル

1. `docs/design.md` — アーキテクチャ、研究動機、新規性分析
2. この `docs/plan.md` — 現状、問題、次のアクション
3. `README.md` — プロジェクト概要
4. `benchmarks/RESULTS.md` — 性能ベンチマーク結果

---

## 1. プロジェクト現状

### 1.1 数値サマリー

| 指標 | 値 |
|------|-----|
| テスト | **399 passed**, 2 failed, 7 skipped |
| CUDAファイル | 23 (.cu) |
| Pythonファイル | 88 (.py) |
| pybind11 bindings | 20 (.cpp) |
| コミット | 11 (main) |
| ベンチマーク | 1M particles @ 12Hz, WLS @ 9.6M epoch/s |
| 検証 | gnss_lib_py (Stanford) 比較済み、31衛星サブnm精度 |

### 1.2 残り2件のテスト失敗

```
FAILED tests/test_multi_gnss.py::TestMultiGNSSCPU::test_insufficient_satellites
FAILED tests/test_raim.py::test_raim_insufficient_satellites
```

**原因**: 衛星不足（n_sat < 4）のエッジケーステスト。`wls_position`のC++側は`n_sat < 4`で`-1`を返すが、テストがPython CPU fallback経由で呼ばれている場合に例外の種類が不一致。

**修正方法**: テストファイルを確認し、`pytest.raises(RuntimeError)` or `pytest.raises(ValueError)` のどちらが来るか確認して合わせる。簡単な修正。

### 1.3 skipped 7件の内訳

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
"GPU-Accelerated Mega-Particle Filter with Real-Time 3D Ray Tracing for Robust Urban GNSS Positioning"

### 5.2 新規性（5つの「世界初」）
1. GPU PF × GNSS（既存研究なし）
2. SVGD × GNSS（既存研究なし）
3. 100万パーティクル × GNSS（既存研究なし）
4. PLATEAU × GNSS（既存研究なし）
5. GPU Ray Tracing × GNSS PF尤度（既存研究なし）

### 5.3 ターゲット
- IROS 2027 (締切 2027年3月頃)
- ION GNSS+ 2026 (締切 2026年6月頃)

---

## 6. 次のアクション（Codexがやるべきこと）

### Phase 1: すぐやること（今日〜1週間）

#### 6.1 2件のテスト失敗を修正
```bash
PYTHONPATH=python python3 -m pytest tests/test_multi_gnss.py::TestMultiGNSSCPU::test_insufficient_satellites tests/test_raim.py::test_raim_insufficient_satellites --tb=short
```
テストファイルを読み、エラーの種類を合わせる。

#### 6.2 UrbanNavデータのダウンロード
```bash
# Tokyo dataset
git clone https://github.com/weisongwen/UrbanNavDataset /tmp/urbannav
```
`python/gnss_gpu/io/urbannav.py` のローダーは実装済み。実データで動作確認。

#### 6.3 ベンチマーク再実行
```bash
PYTHONPATH=python python3 benchmarks/bench_all.py
PYTHONPATH=python python3 benchmarks/bench_pf_device.py
```
`benchmarks/RESULTS.md` を更新。

### Phase 2: 実験（1〜4週間）

#### 6.4 UrbanNavベースライン評価
```bash
PYTHONPATH=python python3 experiments/exp_urbannav_baseline.py --data-dir /tmp/urbannav/Tokyo
```
実験スクリプトは全て `experiments/` に実装済み。合成データfallback付き。

#### 6.5 PLATEAUモデル取得・統合
1. https://www.geospatial.jp/ckan/dataset?q=plateau からUrbanNavエリアのLOD2取得
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
- CDF曲線、エラー時系列、パレートフロントの図生成
- `evaluate.py` にユーティリティ関数実装済み

#### 6.8 論文ドラフト
構成案は `docs/design.md` セクション8.3に記載。

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
| UrbanNav | https://github.com/weisongwen/UrbanNavDataset | 未ダウンロード |
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
1. 「GNSSにGPU大規模PFを適用した世界初の研究」
2. 「3D建物モデルをPF尤度関数にリアルタイム統合」
3. 「SVGDのGNSS向けスコア関数を新規導出（common-mode除去）」
4. 「消費者GPUで100万パーティクル12Hzリアルタイム処理」

### 11.3 レビュアー対策
- 「MegaParticlesの単純適用」→ GNSS固有の工夫（クロックバイアス、3D尤度、スコア関数）を強調
- 「実データがない」→ UrbanNav + 自前収集で解決（最優先タスク）
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
