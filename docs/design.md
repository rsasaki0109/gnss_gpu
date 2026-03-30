# gnss_gpu 設計書

## 背景・動機

### MegaParticles (Koide et al., ICRA 2024)
- 論文: "MegaParticles: Range-based 6-DoF Monte Carlo Localization with GPU-Accelerated Stein Particle Filter"
- **LiDAR** 6-DoF localization に100万パーティクル + SVGD をGPU上で実現
- リサンプリングをSVGD（Stein Variational Gradient Descent）で置き換え → サンプル枯渇回避
- A100で約91ms/frame (11Hz)

### 研究ギャップ
**GNSS測位にGPUメガパーティクルを適用した研究は存在しない。**

- GNSS PF の既存研究 (Suzuki et al.) はCPU上で2000パーティクル程度
- GPU PF は LiDAR/SLAM 分野に偏っている
- GNSS都市部マルチパス問題はマルチモーダル → 大量パーティクルが本質的に有効

## アーキテクチャ

```
Python API (gnss_gpu)
    │
    ├── gnss_gpu.particle_filter  ← メガパーティクルフィルタ
    ├── gnss_gpu.positioning      ← WLS/LSQ 測位エンジン
    └── gnss_gpu.io               ← RINEX/NMEA パーサ (Pure Python)
    │
pybind11
    │
CUDA Kernels (libgnss_gpu_core)
    ├── predict_kernel      ← 状態遷移 + ノイズ
    ├── weight_kernel       ← 擬似距離尤度計算
    ├── resampling          ← Megopolis or Prefix-sum systematic
    ├── wls_kernel          ← Weighted Least Squares バッチ
    └── utils               ← 座標変換, cuRAND
```

## モジュール設計

### 1. Particle Filter (`src/particle_filter/`)

#### 状態ベクトル
```
x = [x, y, z, clock_bias]  (ECEF, 4次元)
```
- メモリレイアウト: **SoA** (Structure of Arrays)
- 各成分を独立配列で持つ → coalesced memory access

#### Predict Kernel
```cuda
__global__ void predict_kernel(
    float* x, float* y, float* z, float* cb,
    const float* velocity,       // GNSS Doppler由来
    float dt, float sigma_pos, float sigma_cb,
    curandStatePhilox4_32_10_t* rng_states,
    int N
);
```
- 1スレッド = 1パーティクル
- cuRAND Philox: `(seed, particle_index, timestep)` でステートレス
- 運動モデル: `x_new = x_old + v * dt + noise`

#### Weight Kernel (尤度計算)
```cuda
__global__ void weight_kernel(
    const float* px, const float* py, const float* pz, const float* pcb,
    const float* sat_x, const float* sat_y, const float* sat_z,
    const float* pseudoranges,
    float* log_weights,
    int N_particles, int N_satellites,
    float sigma_pr    // 擬似距離の標準偏差
);
```
- 各パーティクル位置から各衛星への予測擬似距離を計算
- 尤度: `log p = -0.5 * Σ_sat ((pr_obs - pr_pred)^2 / sigma^2)`
- 衛星データは **shared memory** にロード（全スレッドで共有）
- **log-space** で重み計算（アンダーフロー防止）

#### リサンプリング
2方式をサポート:

**A. Prefix-sum Systematic Resampling**
- `thrust::inclusive_scan` でCDF計算
- **double精度** prefix sum（100万パーティクル以上で数値安定性に必要）
- 各スレッドがCDFをbinary searchで祖先インデックスを取得

**B. Megopolis Resampling** (推奨)
- Metropolis系列の最新手法 (Chesser et al., 2021)
- チューニングパラメータ不要
- prefix sum不要 → 数値的に安定
- coalesced memory access を実現
- 参考実装: https://github.com/AdelaideAuto-IDLab/Megopolis

#### Adaptive Resampling
- ESS (Effective Sample Size) = `1 / Σ(w_i^2)` を毎ステップ計算
- ESS < N/2 のときのみリサンプリング実行

### 2. Positioning Engine (`src/positioning/`)

#### WLS Batch Kernel
```cuda
__global__ void wls_batch_kernel(
    const float* sat_pos,       // [N_epoch, N_sat, 3]
    const float* pseudoranges,  // [N_epoch, N_sat]
    const float* weights,       // [N_epoch, N_sat]
    float* positions,           // [N_epoch, 4] (x,y,z,cb)
    int N_epochs, int N_sat, int max_iter
);
```
- 数千エポックを一括GPU並列処理
- 1スレッド = 1エポック
- 内部でGauss-Newton反復

### 3. I/O (`python/gnss_gpu/io/`)
- Pure Python（GPUに載せる意味がない）
- RINEX 3.x observation file パーサ
- NMEA パーサ
- NumPy配列 → CUDA転送

## CUDA設計方針

| 項目 | 方針 |
|---|---|
| メモリレイアウト | SoA |
| ブロックサイズ | 256スレッド（occupancy calculator で自動調整） |
| 乱数生成 | cuRAND Philox4_32_10 |
| 重み計算 | log-space |
| prefix sum精度 | double |
| 衛星データ | shared memory broadcast |
| ホスト-デバイス転送 | pinned memory + CUDA streams |

## Python API 設計

```python
import gnss_gpu
import numpy as np

# WLS測位 (バッチ)
positions = gnss_gpu.wls_batch(sat_positions, pseudoranges, weights)

# メガパーティクルフィルタ
pf = gnss_gpu.ParticleFilter(
    n_particles=1_000_000,
    sigma_pos=1.0,      # m
    sigma_cb=1e-7,       # s
    sigma_pr=5.0,        # m
    resampling="megopolis",  # or "systematic"
)

for epoch in observations:
    pf.predict(velocity=epoch.doppler_velocity, dt=epoch.dt)
    pf.update(
        sat_positions=epoch.sat_ecef,
        pseudoranges=epoch.pseudoranges,
    )
    estimate = pf.estimate()  # weighted mean position
    particles = pf.get_particles()  # for visualization
```

## ディレクトリ構成

```
gnss_gpu/
├── CMakeLists.txt
├── pyproject.toml
├── docs/
│   └── design.md
├── include/gnss_gpu/
│   ├── particle_filter.h
│   ├── positioning.h
│   └── utils.h
├── src/
│   ├── particle_filter/
│   │   ├── predict.cu
│   │   ├── weight.cu
│   │   └── resampling.cu
│   ├── positioning/
│   │   └── wls.cu
│   └── utils/
│       ├── coordinates.cu
│       └── random.cu
├── python/gnss_gpu/
│   ├── __init__.py
│   ├── _bindings.cpp
│   ├── particle_filter.py
│   ├── positioning.py
│   └── io/
│       ├── __init__.py
│       ├── rinex.py
│       └── nmea.py
└── tests/
    ├── test_particle_filter.py
    ├── test_positioning.py
    └── test_io.py
```

## 実装順序

1. **Phase 1**: WLS batch positioning + RINEX parser
2. **Phase 2**: Particle filter (predict + weight + systematic resampling)
3. **Phase 3**: Megopolis resampling + adaptive resampling
4. **Phase 4**: 3D建物モデル × レイトレース NLOS 尤度 (将来)

## 参考文献

- Koide et al., "MegaParticles: Range-based 6-DoF Monte Carlo Localization with GPU-Accelerated Stein Particle Filter", ICRA 2024. arXiv:2404.16370
- Chesser et al., "Megopolis Resampler", 2021. arXiv:2109.13504
- Suzuki, "Multiple Update Particle Filter for GNSS", 2024. arXiv:2403.03394
- Murray et al., "Parallel Resampling in the Particle Filter", 2013. arXiv:1301.4019
- Liu & Wang, "Stein Variational Gradient Descent", 2016. arXiv:1608.04471
