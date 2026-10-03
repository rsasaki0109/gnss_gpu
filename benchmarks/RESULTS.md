# gnss_gpu Performance Benchmark Results

`python benchmarks/bench_all.py` の結果。各値は 3 回実行の中央値。

## 2026-10-03 (current)

**GPU**: Turing 世代 consumer GPU, 6 GB VRAM (compute capability 7.5)
**CUDA**: 12.8 / Windows 11, MSVC 2022, Release build (`CMAKE_CUDA_ARCHITECTURES=75`)

| Module | Input Size | Time (ms) | Throughput |
|--------|-----------|-----------|------------|
| WLS Batch | 10K epochs | 2.95 | **3.39M epoch/s** |
| Particle Filter (host-buffered) | 1M particles | 106.45 | **9.39M part/s** |
| Particle Filter Device (predict+update) | 1M particles | 31.79 | **31.46M part/s** |
| Signal Acquisition | 32 PRN, 1ms | 135.00 | **237.0 PRN/s** |
| Vulnerability Map | 100x100 grid | 1.03 | **9.71M pts/s** |
| Ray Tracing | 1008 tri, 8 sats | 1.04 | **7.72M checks/s** |

- `Particle Filter` 行は毎回 `initialize → predict → update → estimate` を host 経由の
  バッファで実行する。`Particle Filter Device` 行はデバイス常駐の `predict → update`
  （ESS 判定とリサンプリングを含む）の定常ループ。同じ `predict → update` で比べると
  Device は host-buffered PF（約 85 ms）の **2.7–2.8 倍速**。
- 1M 粒子の Device 段階別内訳: predict 2.2 ms、weight 3.0 ms、ESS 0.5 ms、
  Megopolis resample 30 ms、estimate 0.6 ms。残りの律速はリサンプリング。

### Device Megopolis resampling (2026-10-03 fix)

このリサンプリングは以前、15 回の反復ごとに粒子状態（16 double/粒子：位置・速度・
速度共分散 9・clock）を丸ごとコピーしていた。そのため 1M 粒子で **225 ms** かかり、
Device PF は host-buffered PF より遅かった（153 ms、0.5 倍）。

受理判定はスロットの log-weight と乱数だけで決まり、粒子の中身には依存しない。
そこで反復を int32 の祖先インデックス配列だけで回し、最後に 1 回だけ状態を gather
するよう変更した。結果はビット単位で同一のまま、リサンプリングは 30 ms（7.4 倍）、
Device の `predict → update` は 153 → 32 ms（4.8 倍）になった。

## 2026-04-01 (historical)

**GPU**: Ada 世代 consumer GPU, 16 GB VRAM（型番は記録せず）
**CUDA**: 12.0

| Module | Input Size | Time (ms) | Throughput |
|--------|-----------|-----------|------------|
| WLS Batch | 10K epochs | 1.04 | **9.60M epoch/s** |
| Particle Filter | 1M particles | 81.44 | **12.28M part/s** |
| Signal Acquisition | 32 PRN, 1ms | 142.50 | **224.6 PRN/s** |
| Vulnerability Map | 100x100 grid | 0.62 | **16.14M pts/s** |
| Ray Tracing | 1008 tri, 8 sats | 0.71 | **11.32M checks/s** |

当時のボトルネック分析は、`cudaMalloc/cudaFree` を毎回呼ぶことだった。
`ParticleFilterDevice` で 5–10 ms まで改善する見込みとしていたが、上記のとおり、
実際の律速は Megopolis リサンプリングのコピー量だった。
