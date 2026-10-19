# D253: レジスタに行を分散するGPU LU・求解

2026-10-17。初版の速度は旧GPU比約9.27%/5.00%で10%基準未達。追加の極端値検査でstatusの共有read/writeに9件のRacecheck警告群が出たため、初版は不採用とする。別source版でstatus読出し後のfull-mask syncwarpを追加し、265testsとCUDA各22subsetで検証した。修正版の速度は未計測で、初版の時間を流用しない。

11×11のLU行列をwarpの各laneのレジスタに置き、共有メモリへの逐次読み書きを減らした。前進代入もlaneごとに分担し、各rowの減算順と後退代入の順序を保つ。旧GPUに対する全score短縮はcluster0 +9.267%、cluster1 +4.997%。GPU同士のcomponent gateは0/2、CPUを含むjoint gateは0/2。製品・既定は変更せず、全PF/FGO速度の採用試験は未実施。

## 対象と演算順

D251の凍結warm traceでは、全394入力のunique hypothesis8229件中6685件（約81.2%）がdimension11だった。今回も同サイズを対象とし、3〜10は元solve_normalを使う。D231のshared-row warp-LUやD252のlane0固定サイズ展開とは別の実装で、それらの完了した速度採用判断は更新しない。D226のpageable/synchronous転送を維持し、Graph/stream/構造的ゼロ省略を併用しない。

11×11は32laneのwarp zeroで処理する。最初の11laneが各1rowの11要素とRHSを持ち、残りもshuffle/voteには参加する。pivot比較は元のascending i順、tie時の選択を保つ。row交換とpivot row取得はshuffleで行い、各要素は元のascending j順にdivision/multiply/subtractを受ける。対角logを独立計算して元column順に加算し、途中失敗のpartial logdet/sign/statusも維持する。

RHSのpivot置換を元順に適用する。前進代入はascending kごとに解済みRHSをbroadcastし、row iのk=0..i-1の減算順を保って複数rowを並行更新する。後退代入はdescending i・row内ascending kを維持し、順序を反転したtree和には変えない。FP64/fmad=false、full asymmetric prior、全候補・観測、IRLS回数、閾値/RNGを維持。FMA/fast math/近似逆数/対称化/jitterは使わない。

初期normal/RHSが非有限またはabs>2^500のときは、全warpで判定して元のsolve_normalへ戻す。これは数値を近似・切捨てる閾値ではなく、元の結果を返すための実装選択guard。範囲内でも悪条件によるstatusは元どおり返す。warp内shared pivot/logdetの受渡しは明示syncwarp、他warpとの受渡しは元のblock barrierで区切る。全mask32laneが同期intrinsicへ参加し、暗黙lockstepに依存しない。[CUDA12.8.1 shuffle/synchronization](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-c-programming-guide/index.html#warp-shuffle-functions)。

元numeric prefix/CPU body/通常kernelはliteral copy。元kernelの5144 normalized SASS命令/hashは旧DLLと一致する。初版kernelは18112命令、修正版は18336命令。両者64register/shared30904B/stack32B/spill0。元は56registerでshared/stack同じ。raw probeは初版13128命令、修正版13136命令。両者96register/shared30904B/stack0/spill0。命令数やregister数を速度利得へ換算しない。新kernelのgeneric-controlは同じcode/resourcesでregister solveを無効にし、旧kernelとの等速は仮定しない。

## 差分検証

修正版265tests PASS（main261＋追加4）。3seed×3size×6optionの同backend全score/normal/solution/logdet/status、row0/1/63/127、元のfallback/例外/警告、24context lifetime/返却array所有権、3seed×4PF設定×4epochの全保存state/RNG/graph eventを比較した。

raw solve APIは5760行列（dimension3〜11×20種類×32）を比較。SPD、非対称、row pivot、tie/後段tie、特異/重複row/負det、NaN/Inf、overflow、最小subnormal、signed-zero、桁差、Hilbert型の悪条件、ほぼ重複したrow、RHSのNaN/Inf/巨大値/dense行列でのsigned-zeroを含む。dimension11の640行列にはguard内でregister分岐へ進むケースと元solveへのguard fallbackの両方がある。追加384行列では、guard内の有限入力から小さい対角の除算でInfが生じ、NaNが後退代入へ伝播するケースと有限の極端値対照を検査した。計6144行列、うちdimension11は1024行列。LU/pivot/解/logdet/statusのdtype/shape/bytesが一致し、数値許容差は追加しない。raw probeは通常scoreで呼ばない。

修正版のCUDA memcheck/racecheck/synccheckは各22対象subset PASS、error/hazard/leak0。66個の追加固有testとは数えない。初版main各18subsetは通ったが、追加4testのRacecheckは9警告群（0errors）でexit99となった。初版追加synccheckはその失敗で未実施。初版追加test/memcheckの4件はPASS。値の一致と競合なしは別条件として扱う。各suiteに既存pytest asyncio設定warning1件。allocation/非同期障害の注入試験は未実施。実行したsource/DLLは凍結する。元NumPy/native間の既存丸め差と保存state codecのscalar NaN payload正規化は継承する。

## 初版の実入力全score時間（修正版は未計測）

以下は初版DLLだけの全394入力、8mode×5round×3group、47280timed/3152warm。最適化CPU、旧GPU、同DLL元kernel、新generic、新register、旧GPU/CPU/register A/Aを順序rotate/reverseで比較。全warmの同backend score/135diagnostic/status/input bitsと全timed返却hashが一致。GPU warmは過去の凍結GPU基準とも照合した。

workspace/warm/復元copy/hashはtimer外。dedup/DD/cluster/pack/guard、register dispatch/vote/shuffle/solve、全転送/kernel/完了同期/返却、fallback/scatterはtimer内。allowed CPU1本/BLAS1/normal priority/非exclusive、GPU処理は直列。全197batchの整数ns合計→round内3group paired率median→5round median。事前gateは旧GPU・同DLL元GPU・新generic各10%以上、各4/5round以上正。GPU/CPU/register A/Aと同DLL元対照のabsmedian<5%。CPU比も10%以上をjoint gateとし別に報告する。全raw/roundを保持し、完了した候補は再測定しない。

時間列は197batch合計median÷197で、個別batchのp50ではない。短縮率は別のpaired統計量。絶対時間を過去実験と引き比べて速度を主張しない。

| cluster | CPU ms | 旧GPU ms | 新generic ms | 新register ms | 対旧GPU短縮 | 対generic短縮 | 対CPU短縮 |
|:---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2.131 | 2.195 | 2.284 | 2.034 | +9.267% | +9.067% | +9.764% |
| 1 | 3.329 | 3.212 | 3.237 | 3.167 | +4.997% | +4.826% | +8.202% |

| cluster | GPU A/A | CPU A/A | register A/A | 同DLL元対照 | controls |
|:---:|---:|---:|---:|---:|:---:|
| 0 | -0.566% | +1.550% | -2.125% | +2.147% | True |
| 1 | -0.370% | +0.452% | -1.507% | -2.171% | True |

事前のmedian対照条件は両設定で通ったが、個別roundではcluster0 register A/A第5round -5.27%、cluster1同DLL元対照第1round +5.61%の変動がある。CPU比も最後のroundは約-52%/-46%と大きく振れ、CPUに対する優位性を一般化しない。全roundを保持し、10%基準未達の判定を維持する。

GPU同士の短縮と、最適化CPUからGPUへ切り替える利得を区別する。全RTK/PF/FGO性能、GPU利用率やGPU化率には換算しない。GTSAM solver/整数探索はCPUのままで、新たな数値offload領域はない。

## 競合の修正

初版raw probeのSASSは0x13710でshared0x7894をreadし、0x16ed0でstatus3をwriteしていた。status0/1を読んで進む分岐と最後のnonfinite解判定の間にshared memoryのorderingが必要だった。shuffle/voteの収束はshared memory barrierの代わりにならない。元のstatus読出し直後にfull-mask syncwarpを1つ追加した別source/buildを作り、数値式は変更しない。初版source/DLL/失敗checker/raw timingは全て保全した。

## 初版と修正版の実GTSAM+GPU精度

run3/seed20260710/400epoch、cluster0/1各1prefixを初版と修正版それぞれで実行。元D236 GPU基準に対し各versionで計400保存PF state、全trajectory配列のdtype/shape/bytes、全cfgと従来caller metadataが一致。各197valid/3skip、計394nonempty GPU score、scalar fallback0。fieldのみ全diagnosticを取得してdimension11を両設定で確認し、呼出元には元metadataだけを返す。guard通過回数を直接計数するcounterは追加しない。追加score callはなく、diagnostic取得付きfield時間を性能根拠にしない。同じ400保存状態を2versionで照合したもので、800unique状態とは数えない。全route/multiseed精度gateとGPU既定採用は未完了。

stagingはhost/device各1143352B/context、timing5contextsで各5716760Bと元と同じ。通常scoreに追加buffer/counter/転送はない。raw probe最大256行列のdevice一時確保は863232Bで、通常scoreにはない。driver管理領域/CPU heap peakは未測定。36製品source、5installed PEs、PF build PE、元DLL/拡張、66既報、32archives/5080filesを保全する。

[結果](../benchmarks/rbpf_register_lu_20261017.json)、[完了照合](../benchmarks/rbpf_register_lu_completion_check_20261017.json)、[scorer](../benchmarks/rbpf_register_lu.py)、[全score比較](../benchmarks/bench_rbpf_register_lu.py)、[probe/API](../benchmarks/probes/rbpf_register_lu_synced/README.md)。
