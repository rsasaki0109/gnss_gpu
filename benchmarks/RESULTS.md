# gnss_gpu Performance Benchmark Results

**Date**: 2026-04-01
**GPU**: NVIDIA Ada Lovelace consumer GPU (16GB VRAM)
**CUDA**: 12.0

## Summary

| Module | Input Size | Time (ms) | Throughput |
|--------|-----------|-----------|------------|
| WLS Batch | 10K epochs | 1.04 | **9.60M epoch/s** |
| Particle Filter | 1M particles | 81.44 | **12.28M part/s** |
| Signal Acquisition | 32 PRN, 1ms | 142.50 | **224.6 PRN/s** |
| Vulnerability Map | 100x100 grid | 0.62 | **16.14M pts/s** |
| Ray Tracing | 1008 tri, 8 sats | 0.71 | **11.32M checks/s** |

## Key Findings

- **WLS Batch**: 10,000エポックを1msで処理。リアルタイム1Hz処理に対して9,600倍の余裕
- **Particle Filter**: 100万パーティクルのフルパイプライン（predict+weight+resample+estimate）が81ms。12Hzリアルタイム可能
- **Signal Acquisition**: 全32 PRNの並列捕捉が143ms。1秒間隔の捕捉に十分
- **Vulnerability Map**: 10,000グリッド点のDOP計算が0.62ms。リアルタイムマップ更新に十分
- **Ray Tracing**: 1008三角形×8衛星のLOS判定が0.71ms。都市環境リアルタイムNLOS判定可能

## Bottleneck Analysis

現在のボトルネックは `cudaMalloc/cudaFree` の毎呼び出し実行。
`ParticleFilterDevice`（デバイスメモリ常駐版）を使用すればH2D/D2H転送を削減し、
パーティクルフィルタの処理時間を5-10ms（100-200Hz）まで改善見込み。
