# D251: GPU尤度の工程別サイクル診断

2026-10-17。D250で多数の0積を省いてもscore時間が変わらなかったため、元のD226 GPU尤度に任意の工程別計測を追加した。今回は診断機能で、速度改善の採用ではない。通常の数値領域・製品・既定は変更しない。実入力394件の全score/diagnosticを元GPU基準と照合した。

計測では、block内経過サイクルのLU分解・求解3回が約74.1%、正規行列3回が約10.5%、初期化が約8.5%だった。全scoreに対する計測の上乗せはcluster0約1.98%、cluster1約1.28%で、両設定が校正条件を満たした。次の実装候補は3回の正規行列のLU分解・求解を優先し、元のpivotと演算順を維持する固定dimension向けのデータ配置を調べる。D231の既存warp-LU候補の再測定や、74.1%を全scoreの速度利得へ読み替えることはしない。

## 測定範囲

Nsight SystemsのCUDA限定・自分の子processだけを対象とした1入力smokeは、入力再生完了後もcapture出力が終わらず、180秒で自分のprofiler PIDだけを停止した（実待ち183.94秒）。その出力・失敗を保持し、時間の根拠や成功したtimelineとして使わない。管理者権限・system-wide tracingは使わず、再試行もしない。

代替の別kernelで、既存block barrier後とfinish/write_result後にlane0がclock64を記録する。元数値関数・通常kernel・CPU bodyはliteral copy。演算順、候補/観測、全prior、FP64、IRLS回数、pivot/閾値/RNGは保持。新しいbarrierはない。原kernelは5144 normalized SASS命令とhashが同一。計測kernelは5232命令、元と同じ56register/stack32B/spill0、sharedは30904から31032Bへ128B増える。

clock64はSM上の経過サイクルで、stall/time-slicingを含む。[CUDA Programming Guide 7.13](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-c-programming-guide/index.html#time-function)。別SMの絶対timestampを引かず、各block内の差だけを使う。総区間はinitialize開始からwrite_result直後までで、kernel prologue/epilogueと最後のtrace書出しは含まない。スキップした区間はuint64 maxのままで0扱いしない。

## 校正と実入力

CPU実走から凍結したcluster0/1各197batchの全394入力。6mode（旧GPU、同DLL元GPU、元GPU＋同じtrace転送/query、計測GPU、旧GPU A/A、計測GPU A/A）×5round×3group、35460timed/2364warm。順序をrotate/reverseし、全rawを保存した。workspace/warm/復元copy/hash/trace解析はtimer外。dedup/DD/cluster/pack、全転送/kernel/完了同期/返却、任意計測のselect/query/resetはtimer内。allowed CPU1本/BLAS1/normal priority/非exclusive、GPU処理は直列。

各group全197batchの整数ns合計からpaired overheadを出し、round内3groupのmedian→5round medianを使う。事前条件は全6contrastのabsmedian<5%。今回2/2設定が通過。条件不通過の比率は未校正の参考値に留める。完了した計測の再実行や、速度採用への読み替えはしない。

| cluster | 計測全体の上乗せ | marker対copy | copyのみ | 同DLL元対照 | GPU A/A | 計測 A/A | 校正 |
|:---:|---:|---:|---:|---:|---:|---:|:---:|
| 0 | +1.978% | -0.648% | +2.633% | -0.419% | +0.536% | -0.817% | True |
| 1 | +1.284% | +0.511% | +0.943% | +0.441% | -0.148% | -0.241% | True |

全timed計測callの各blockについて区間差を検査し、block内の総経過サイクルに対する各区間比率を算出。batch内block比率のmedianを取り、全batch/round/group/計測2modeのmedianを示す。独立したmedian同士なので列の合計は100%になるとは限らない。block時間の総和、kernelの排他的wall時間配分、GPU利用率、全PFのGPU化率ではない。境界付近のwarp schedulingと計測markerの影響も含む。

| 区間 | cluster0 | cluster1 |
|:---|---:|---:|
| 初期化 | 8.48% | 8.48% |
| 行準備 | 0.93% | 0.93% |
| 正規行列3回 | 10.50% | 10.49% |
| LU分解・求解3回 | 74.07% | 74.08% |
| IRLS重み2回 | 1.56% | 1.56% |
| 行尤度 | 1.90% | 1.90% |
| 最終和・prior | 2.54% | 2.55% |
| 結果書出し | 0.06% | 0.06% |

この診断は、工程の優先順位を考えるための入力である。区間比率をそのまま全scoreの短縮上限に換算しない。D248のCUDA event間隔にもhost enqueue gapが入り得るため、その値と本表を混ぜてCPU/GPU時間を分割しない。今回GPU実時間のtimelineを得ておらず、packing/driver/kernelの全wall配分は未確定。

## 数値と検査

チェック版83tests PASS。3seed×3size×6optionの同backend score/normal/solution/logdet/status、元のfallback/例外/警告、row0/1/63/127、24context lifetime/返却array所有権、3seed×4PF設定×4epochの全保存state/RNG/graph eventを比較した。計測のvisited marker、huber有無、失敗status、copy/CPU/空入力/unsupported後の非staleとfinally解除も確認した。新たな数値許容差はない。NumPyとnativeの既存丸め差は継承する。

初版testは81PASS/2FAIL。原因はstrict NumPy設定で元から発生するlogaddexp underflow例外をテストが捕捉しなかったこと。初版source/logを凍結し、別test fileで元版/計測版の例外型・message一致を確認した。数値source/DLLは変更しない。修正版83件と初版81件を別件として加算しない。各suiteには既存pytest asyncio設定warning1件がある。

CUDA memcheck/racecheck/synccheckは各15対象subset PASS、error/hazard/leak0。45件の新規固有testとは数えない。allocation/非同期障害の注入試験は未実施。実入力全warmのscore/全135diagnostic/status/input bitsと、全35460timed返却hashが元GPU基準と一致した。実入力の計測traceは246870blockを全件検査した。今回は新しいGTSAM実走や全route/multiseed精度gateを実施しておらず、全PF/FGO性能の採用判断もしない。

capacity512の元stagingはhost/device各1143352B/context、timing4contextsで各4573408B。copy/trace2contextsのlazy trace buffersが各host/device合計131072B追加され、計4704480Bとなる。driver管理領域/CPU heap peakは未測定。通常scoreはtraceを記録・転送せず元kernelを呼ぶ。36製品source、5installed PEs、PF build PE、元DLL/拡張、62既報、30archives/4744filesを保全した。

[結果](../benchmarks/rbpf_gpu_stages_20261017.json)、[完了照合](../benchmarks/rbpf_gpu_stages_completion_check_20261017.json)、[計測scorer](../benchmarks/rbpf_gpu_stages.py)、[実入力driver](../benchmarks/bench_rbpf_gpu_stages.py)、[probe/API](../benchmarks/probes/rbpf_gpu_stages/README.md)。
