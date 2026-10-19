# RBPF-FGO GPU化の検討と既存FGO経路の再測定 (2026-10-16)

広い窓・多数仮説の連続行列計算は有望。ただし「RBPF全体がGPUで速くなった」という新実装/採用結果はまだない。
対象は公開凍結版`experiments/rbpf_fgo`と現行`gnss_gpu.fgo`。精度改善した別の最新版の所在は確認を依頼中で、今回の数値をその版の測定とは扱わない。
製品コード/バイナリ/既定off・auto閾値512は変更しない。

## 今回の測定

既存scalar CPU Cholesky指定off、既存cuSOLVER指定on、既存auto、off同士A/Aを5round×3groupで回転する。
12衛星、stationary Sagnac補正PR、固定seed、外れ値1つ、Huber、line searchあり、最大5反復。
VDは速度/clock drift/Doppler/IMU delta-p/delta-vも使う。入力作成・state初期copy/hashは時間外、公開関数内のallocation・transfer・全実行反復・同期host返却を含む。
120groups/480timed calls、240cross-mode比較/120A/A比較。主指標はpair短縮率のround内中央値→5round中央値。絶対p50の比とは別。全raw/auto/個別round/p95/p99をJSONへ保持した。

| 構成 | epochs / 状態数 | CPU solve要求 p50 ms | GPU要求 p50 ms | paired短縮 | 改善round | 同一CPU A/A | 扱い |
|:---|---:|---:|---:|---:|---:|---:|:---|
| basic | 5 / 20 | 0.660 | 3.118 | -339.33% | 0/5 | -4.77% | 初期値のまま・有効解の利得に数えない |
| basic | 50 / 200 | 2.107 | 3.837 | -85.08% | 0/5 | -0.02% | 初期値のまま・有効解の利得に数えない |
| basic | 128 / 512 | 24.553 | 7.670 | 68.47% | 5/5 | -0.11% | 初期値のまま・有効解の利得に数えない |
| basic | 200 / 800 | 88.292 | 11.863 | 86.40% | 5/5 | -0.76% | 初期値のまま・有効解の利得に数えない |
| VD/Doppler/IMU | 5 / 40 | 0.705 | 4.238 | -500.92% | 0/5 | -2.51% | 更新あり |
| VD/Doppler/IMU | 20 / 160 | 3.861 | 6.324 | -56.62% | 0/5 | -1.57% | 更新あり |
| VD/Doppler/IMU | 64 / 512 | 114.128 | 40.295 | 64.69% | 5/5 | -0.48% | 更新あり |
| VD/Doppler/IMU | 100 / 800 | 431.337 | 78.180 | 82.69% | 5/5 | +0.66% | 更新あり |

**basic全4条件は初期値のまま終了するため、有効な最適化の高速化には数えない。** sourceではPRのRHSをnegateしており、VDにはそれと異なりH*delta=gを直接解く注記がある。今回GPU調査でこの既存挙動を勝手に修正せず、別の正しさ課題として保存する。過去のbasic一反復benchmarkも、解が改善したことの証明には使わない。
VD全4条件は5反復で位置を更新し、最初のnative accepted-step objective減少を確認。CPU/GPU双方で合成位置RMSEが改善した。ここでは最終ECEF位置bitと反復数/返却PR MSEが一致し、全状態では最大8.53e-14の差がある。無条件のbit一致、実データ精度保証やRBPF整数選択/RNG一致とは扱わない。
GPU forced-onでVD512/800状態は約65%/83%短縮・各5/5round改善。小窓では逆効果。autoは小窓CPU/大窓GPUを選ぶ既存方針と整合するが、今回の少数fixtureから新しい最適閾値を決めない。

全A/A条件中央値はabs5%以内。ただし非占有host/GPUで個別roundは揺れる。15標本のp95/p99は締切保証ではない。
別の10秒Nsight診断でDpotrf/Dpotrs各6callとFP64 GPU kernelsを確認。trace時間は利得に使わない。最初のoff trace終了停止も保存し、専用PIDだけをidentity確認後に終了した。
FGO/native wrapper既存91tests PASS、pytest設定/書込cacheのwarning2件。wrapperはPASS記録後のlog表示でencoding errorとなったため、元cp932 bytesを別途読むだけに留め、試験は再実行しなかった。

## CPUの比較対象

今回の大きな比率は既存の素朴なCPU Choleskyに対する値。最適化CPUや疎行列法への勝利とはしない。
同じVD first-step H/gを保存し、元と同じlower triangle・1e-3 jitterを使う1BLAS threadのSciPy/LAPACKも計測した。
40/160/512/800状態のcopy＋分解＋solve p50は0.097/0.344/5.537/15.327ms。
最大native first-step差は約2e-13。これは最初の固定行列1回だけであり、5反復FGO全体と直接割り算しない。元行列の約4e-14..6e-14の非対称は残し、勝手に平均しない。
GPU prototypeはこのCPU backendを統合した版も比較相手にし、転送/同期を含むwhole操作で評価する。

## RBPFで優先する実装

1. **尤度の一括処理**。実際に要求されたunique仮説だけを一度packし、共有観測/JacobianをGPUへ置く。residual更新・whitening・group nuisance・既存IRLS各pass・小行列solve・尤度までまとめる。結果は元の要求順へ戻し、extend/shrinkの再生成や候補削減で速度を作らない。元scalar cache、optimized CPU native batch、GPU fused batchでunique16/32/64/128/512・異なるheld maskを比較する。
2. **履歴別FGOの連続root**。問題が大きくなった段階でFP64 QR/triangular solveとworkspace再利用を使う。Jacobian・weight・prior・historyが同じ場合だけ分解共有する。IRLSやhard holdで行列が変わる場合は独立に解く。数値安定性のために使うsquare-root QRを、GPU化の都合でnormal equationsへ置き換えない。
3. **広窓FGOの常駐化/疎構造**。basicはdense HをGPU→host→GPU、VDはhost assemblyしたdense Hを反復ごとGPUへ転送する。800状態のHは5,120,000B。適用できる因子だけGPU側へまとめ、元のaccumulation順とtrial規則を保つ。CPU/SIMD/LAPACK・band/sparse版との比較も必要。
4. **独立整数探索のbatch**。既存GPU LAMBDAは単発共有問題を多数粒子が参照してもbatchが増えない。真に独立な問題だけまとめ、元の探索順/全候補/予算/fallbackを保つ。

公開版の過去WP41 profileはISAM2 update2.6%、likelihood27%、unique仮説約21/epoch。そこでISAM2だけを無限に速くしても全体時間の削減上限は2.6%。新しい履歴別graph構成にはその割合を流用しない。
過去のNumPy likelihood batchは約20%遅かったが、move生成を重複させない融合CUDA実装の実測ではない。これだけを理由に今後のGPU化を却下しない。
厳密整数/rationalのnormalizer・全整数domain・tail/background massの証明はCPU側へ保ち、FP64近似で置き換えない。

採用前は最新版source/inputを固定し、全ての整数/採否/候補順/seed・RNGを照合する。全平均/共分散/evidenceと既存数値guardを維持し、同一seedのCPU/GPU軌跡を比較する。丸め差がbranchを変える場合のCPU再確認も含める。元の全区間3seed OFFICIAL/coverage/false-FIX条件を維持し、転送/同期/pack込みのwhole RBPF p50/p95/p99とmemoryで判定する。
現在のPythonにはGTSAM/cssrlibがなく、今回新しいPPC full-run測位精度を試験したとは主張しない。新しい精度改善版へはこの手順で接続する。

## 再現と一次資料

```text
python benchmarks/bench_rbpf_fgo_gpu_feasibility.py --output dist/new_fgo_feasibility.json --rounds 5 --repeats 3
python benchmarks/bench_fgo_lapack_probe.py --matrix-dir <saved_VD_first_step_directory> --output dist/new_lapack_probe.json
```

GPUジョブは直列、出力は新規pathを使う。既存GPU経路の指定は`GNSS_GPU_FGO_GPU_SOLVER=0|auto|1`、公開RBPFの既存GPU LAMBDA入口は`RBPF_CUDA=1`。両者は別の経路であり、このFGO環境変数だけでGTSAM RBPFがGPU化されるわけではない。
[全測定/診断/計画JSON](../benchmarks/rbpf_fgo_gpu_feasibility_20261016.json)、[完了検証](../benchmarks/rbpf_fgo_gpu_feasibility_completion_check_20261016.json)。
API: [cuSOLVER12.8](https://docs.nvidia.com/cuda/archive/12.8.0/cusolver/index.html)、[cuBLAS12.8](https://docs.nvidia.com/cuda/archive/12.8.0/cublas/index.html)、[GTSAM ISAM2](https://borglab.github.io/gtsam/isam2/)、[SciPy Cholesky](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.cho_factor.html)。
