# RBPF尤度のLUと対角logをwarp内で並列化する (D231, 2026-10-16)

**実験用の明示選択として保存し、既定適用は見送る。** 小行列の直列LUをwarp内で並列化した。代表的な64rows/21uniqueでは、準備・転送・返却込みで旧GPU比7.19%、融合C++比12.66%短縮、各5/5round改善。ただし事前の旧GPU比10%gateには届かず、全9条件の主screenは不合格だった。D229と製品/凍結runtimeの既定動作を維持し、D230の不採用判断も変更しない。

## 数値と並列化

独立したstandalone DLLに旧serial kernelと新warp kernelを同居させ、同じpageable転送とCPU計算を使う。`WarpBatchScorer`はworkspace作成時だけ明示kernelを選び、通常のPython dedup/DD/cluster/pack/scatterコードをそのまま継承する。全入力と2IRLS、FP64、元のpivot LU、full covariance/asymmetry、likelihood式/閾値、全候補を維持する。fast mathなし、FMA contractionなし。

normal matrixのcopyをwarp内で分担し、pivotの最大値比較とtie順はlane0が元の順で実行する。row swapは各columnを分担する。各elimination rowは独立なlaneが元と同じdivision→column順のmultiply/subtractを実行する。列間の依存は全32laneの`__syncwarp()`で区切る。forward/back substitutionは元のserial関数のまま。
各pivot diagonalは後続columnのpivotで変更されない。LU後に対角の`log(abs(diag))`だけを並列計算し、lane0が元のcolumn順で加算する。途中でpivotが失敗した場合も、処理済みcolumnのpartial logdetを同じ順で返す。sign/status/初期rhs/diagnosticも維持する。tree reduction、近似逆数、反復数削減、対称化やjitterは追加しない。
同期の前提は[CUDA12.8.1 Programming Guide](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-c-programming-guide/index.html#synchronization-functions)に従う。全warp laneがbarrierへ参加し、他warpとの受渡しは元のblock barrierを維持する。暗黙lockstepだけに依存しない。

## 準備と転送を含めた比較

9条件×5round×3group×5mode、675timed calls。既存融合C++、旧GPU、同じ旧serial kernelを含む新DLL、warp、新旧GPU A/Aを直列回転した。warp比較135、A/A135、新serial対照135。全675返却hashとwarm diagnosticを照合する。
canonical/dedup・元DD/cluster lookup・NumPy pack・pageable全upload・native/kernel計算・同期readback/scatterを含む。workspace作成/warm/hash/検証は時間外で、graphや完全PFの時間ではない。自processの1許可CPU・1BLAS thread、通常priority、host/GPU非占有。
paired短縮率は同じgroup内比のround内中央値→5round中央値。warp対旧GPUと対融合C++の両方で10%以上・4/5round以上改善し、旧GPU A/A中央値abs5%未満、新serial対旧GPU中央値abs5%未満を要求する。N64利用への関連性は64rows/21uniqueと64uniqueの両方が必要。全raw/失敗を残し、再試行しない。

| rows/groups/unique | C++ ms | 旧GPU ms | warp ms | 対旧GPU短縮 | 対C++短縮 | A/A短縮 | 新serial対旧GPU | 主screen |
|:---|---:|---:|---:|---:|---:|---:|---:|:---|
| rows64/groups4/unique4 | 0.388 | 0.596 | 0.551 | +7.93% | -43.01% | -1.04% | -1.89% | FAIL |
| rows64/groups4/unique21 | 1.179 | 1.080 | 0.997 | +7.19% | +12.66% | +3.66% | -0.85% | FAIL |
| rows64/groups4/unique32 | 1.815 | 1.516 | 1.452 | +3.57% | +17.69% | +1.43% | -6.71% | FAIL |
| rows64/groups4/unique64 | 3.416 | 2.799 | 2.678 | +5.60% | +23.14% | +2.54% | +0.52% | FAIL |
| rows64/groups4/unique128 | 7.156 | 5.814 | 5.321 | +9.28% | +24.36% | -5.70% | +8.96% | FAIL |
| rows64/groups4/unique512 | 30.218 | 23.994 | 22.977 | -3.17% | +26.21% | +13.52% | -4.20% | FAIL |
| rows128/groups8/unique21 | 4.178 | 2.996 | 2.789 | +6.41% | +33.46% | +5.50% | -40.65% | FAIL |
| rows128/groups8/unique64 | 12.399 | 7.913 | 8.060 | -19.76% | +39.93% | -6.90% | -8.71% | FAIL |
| rows128/groups8/unique128 | 25.532 | 18.895 | 17.650 | -6.57% | +29.76% | +9.16% | -2.65% | FAIL |

64rows/21uniqueの7.19%/12.66%、64uniqueの5.60%/23.14%は有効な副指標として残すが、旧GPU比10%の不足をC++比の大きな値で埋め合わせない。9条件中5条件はA/A不合格、4条件は新serial対照不合格だった。既存と同じserial機械語でも時間差が揺れており、新kernelやホスト処理へ原因を断定しない。独立p50の比やkernelだけの時間へ判定を置き換えない。

旧serial kernelの全5,144 SASS instructionはopcode/operand正規化後も新DLL内のserial kernelと同一。warpは5,080 instruction。両者とも56register、shared30,904bytes/block、stack32bytes/thread、spill0。命令数やresource数の差を速度利得とは扱わない。これはstandalone likelihoodの静的disassemblyで、製品PF/FGOの新しいdisassembly/traceではない。

## 精度と実走

新79tests PASS、設定warning1。1〜11次元それぞれ1,040件、合計11,440行列のLU結果・pivot・partial logdet・solution/statusを旧serial GPUとbit比較した。一般/非対称/row swap/広いscale/zero/NaN/Inf/subnormal/negative determinant/失敗途中を含む。54種の尤度入力で全LL、normal matrix、solution、logdetとC++計算の返却bitsを旧DLLへ比較し、部分row/odd count・scalar/strict fallbackも確認した。完全PFは3seed×3構成×4epochで全保存stateとRNGが一致した。
memcheck/racecheck/synccheckは各11targeted tests PASS、各設定warning1。各checkerは3,120件の直接LUを含み、error/hazard/leak0。これらは79testsのsubsetであり、新規unique testsを112と数えない。

Tokyo run3/seed20260710/400epochの実GTSAM+GPU prefixでも、旧D229 GPUの凍結対照と全200PF stepの保存state・軌跡NPZ全配列が一致した。位置/P/FIX/hold/粒子/重み/RNG/feedbackを含む。有効197batch、要求move206,522、unique move4,363、scalar fallback0、extend/shrink各2,110callで、callerはD229と同じ。旧対照は精度だけに使い、過去の時間との差を速度結果にしない。
NumPy配列dtype/shape/bytesと有限scalarのbitsを保存し、Python scalar NaN payloadの従来codec canonical化は除く。native/元NumPy間の既存微小差に新しい許容差を加えない。1prefixは全route/multi-seed測位精度gateを完了しない。新しい精度改善版や別GPUの評価とも扱わない。

## メモリと利用

capacity512のdevice/host stagingは各1,143,352bytesで旧pageable版と同じ。追加のkernel workspaceはなく、既存Scratchを使う。benchmarkは旧/新serial/warpの3所有contextを作るため合計device staging3,430,056bytes、hostも同量。context/driver/JIT/コード/stack・Python packは含めず、heap peak実測ではない。直接LUのtest専用APIはそのcallだけ一時allocを使い、通常scoreからは呼ばない。
36製品source、5installed extensions、PF build PE、D226の旧DLL、過去22report/9archiveは不変。新DLLはstandaloneで製品へinstallしていない。

```python
from rbpf_warp_batch import WarpBatchScorer

with WarpBatchScorer(explicit_new_dll_path, kernel="warp") as scorer:
    scores, details = scorer.score(hypotheses, problem, oracle, mode="gpu")
    # kernel="serial" is the unchanged-kernel control in the same new DLL.
```

再現: [build/API](../benchmarks/probes/rbpf_likelihood_warp/README.md)、[完全バッチ比較](../benchmarks/bench_rbpf_warp_likelihood.py)、[実走wrapper](../benchmarks/bench_rbpf_warp_field.py)。記録: [全結果](../benchmarks/rbpf_warp_likelihood_20261016.json)、[完了照合](../benchmarks/rbpf_warp_likelihood_completion_check_20261016.json)。
