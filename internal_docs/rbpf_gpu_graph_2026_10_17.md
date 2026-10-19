# D249: RBPF GPU likelihoodを3ノードCUDA Graphで実行

2026-10-17。入力転送→既存尤度kernel→結果転送を1つのGraph launchで投入する実験を実装した。速度の共同gateは0/2で、製品/既定には採用しない。GPUで数値計算したうえで完成したhost結果を比較し、非同期投入だけの時間を高速化としない。D248や以前の不採用結果は保全する。

## 実装と精度

両新方式は固定メモリのhost staging、同じdevice staging、専用nonblocking streamを持つ。Graph方式はH2D→kernel→D2Hの3nodeと1executableを保持。workspace初期化時に小さい仮形状をinstantiateし、計算/launchはしない。実scoreでcount/rows/pairs/diagnosticが変わると全3nodeの引数・転送量を更新する。この更新費用はscore内に含む。同形状ではポインタ/形状だけ再利用し、全入力値は毎回転送する。形状cacheを増やしたり、epochを越えて尤度値を再利用しない。1workspaceの同時使用は対象外。通常scoreはeventを記録せず、今回event profileを速度指標に使わない。

D226の数値prefix bytes・CPU bodyを維持し、GPUのnormalized SASS 5144命令/opcode/operandも一致した。56registers/shared30904B/stack32B/spill0。FP64、元の加算順、pivot/tie、2回のIRLS再重み付け、全観測/候補/prior/閾値/RNGを保持した。Graphのkernel引数配列はAPI呼出中にcopyされ、元node/graph、固定host/device allocation、executableはworkspace終了まで保持する。[使用したCUDA12.8 Graph API](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-runtime-api/group__CUDART__GRAPH.html)。

83tests PASS。3seed×3size×6optionsで元GPU/同DLL固定メモリstream/Graphを同backendで比較し、score/normal/solution/logdet/statusがbit一致。rows0/1/63/127、diagnostic有無、count/rows/pairsの変更、同形状で新しい値を送る場合を含む。capacity/rows/groups/dtype/singular/strict/warning/invalid/emptyの元fallback・例外・警告を保持する。stream/Graph各24workspaceのclose/doublecloseと返却array所有権、3seed×4PF設定×4epochの全保存state/RNG/graph eventも確認した。CUDA memcheck/racecheck/synccheckは各15対象subsetでPASS、error/hazard/leak0。既存pytest設定のasyncio option warning1件は継承。CUDA allocation失敗/非同期障害の注入試験は行っていない。

## 実入力全batchの比較

D243で保存したrun3/seed20260710、cluster_relin0/1の各197batch、計394入力を全て使用。7mode（最適化CPU、元GPU、同DLL固定メモリstream、Graph、元GPU A/A、CPU A/A、Graph A/A）×5round×3group、41370timed/2758warm。全warmの同backend score/全native diagnostic/status/inputと全timedの結果hashが一致した。Graphのcompleted launchは12608、固定メモリstreamは6304で、別経路へのすり替えをcounterで確認した。

workspace/Graph初期化、warm、入力復元/copy、hashはtimer外。dedup/DD/cluster/packing、fresh input staging、形状node更新、H2D/kernel/D2H/完了待ち、fallback/scatterは元scoreのtimer内。allowed CPU1本、BLAS1、normal priority、非exclusive、GPU処理は直列。各group全197batchの整数ns合計→round内3groupのpaired短縮率median→5round median。事前条件はGraphが元GPU・同DLL固定メモリstream・最適化CPUの全てに10%以上、各4/5round以上正、3種A/Aのabsmedian<5%。完了した同一候補を測り直さず、全sampleを保存した。

時間列は197batch合計のmedian÷197で、個別batchのp50ではない。短縮率は別のpaired統計量。

| cluster | CPU ms | 元GPU ms | 固定メモリstream ms | Graph ms | 対元GPU短縮 | 対固定stream短縮 | 対CPU短縮 |
|:---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1.642 | 3.475 | 3.614 | 3.665 | -5.084% | -0.943% | -122.200% |
| 1 | 2.524 | 4.352 | 4.516 | 4.496 | -3.313% | +0.208% | -75.841% |

| cluster | 元GPU A/A | CPU A/A | Graph A/A | control gate |
|:---:|---:|---:|---:|:---:|
| 0 | +0.036% | -0.923% | -0.202% | True |
| 1 | +0.102% | +1.275% | -1.812% | True |

各設定の全roundと採否は結果JSONに残す。別環境時刻のD248絶対時間との比較でGraphの効果を認定しない。全score共同条件が成立しないため、新しい全PF/RTK/FGOの性能採用試験へは進めない。既存のCPU component改善やGPU kernel単独の差と加算しない。

両設定で3種A/Aは基準内。Graphは元GPUより約5.08/3.31%遅く、固定メモリstream比でも-0.94/+0.21%で、期待した改善は確認できなかった。CPUに対するpaired時間比は約2.22/1.76倍。今回の全score比較だけでは遅さをkernel・Python・driverへ分解できない。D248/D249の結果を踏まえ、転送方式だけを変える試行の優先度を下げる。次に進む際はGPU kernel内で共有できる計算の同定を優先する。

## 実GTSAM+Graph GPU

元D236の同backend GPU基準と同じrun3/seed/400epoch、cluster0/1各1prefixを実行した。計400保存PF state行、全trajectory配列dtype/shape/bytes、全cfg/全既存caller metadataが一致した。各197valid/3skip、394nonempty GPU score、completed Graph launch394、scalar fallback0。Graphのinstantiateは各1個のみ。元fieldのRBPF_CUDA=0は整数探索側の設定で、尤度scorerのmode=gpuとは別。source/input/外部tc/cssrlib/3CPU拡張もhash固定。新helperは元field/baseのbindingを書き換えずproxy/別globalsでscorerを選択する。元field自身のruntime adapter一時install/restoreは踏襲する。追加counter/保存のある実走は精度確認だけに用い、call_secondsを速度証拠にしない。

元NumPy対nativeの既存微小差、保存state codecのscalar NaN payload正規化は継承し、新しい許容差はない。1route/1seedのprefix検証であり、GPUの全route/multiseed/既定採用gateは未完了。新しい数値処理領域のGPU化やGTSAM solverのGPU化は含まず、FGOはCPUのまま。GPU利用率・全体時間のGPU比率は未測定。

capacity512のstagingは1contextにつきhost/device各1143352B。比較時4contextsでdevice計4573408B、固定host2286704B、通常host2286704B。新規stream2本、Graph1個/executable1個/3nodes。driver管理領域/CPU heap peakは未測定。36製品source、5installed PEs、PF build PE、元数値DLL/3拡張、58既報、28archive/4375filesを保全した。

[結果](../benchmarks/rbpf_gpu_graph_20261017.json)、[完了照合](../benchmarks/rbpf_gpu_graph_completion_check_20261017.json)、[scorer](../benchmarks/rbpf_gpu_graph.py)、[probe/API](../benchmarks/probes/rbpf_gpu_graph/README.md)、[実入力比較](../benchmarks/bench_rbpf_gpu_graph.py)。
