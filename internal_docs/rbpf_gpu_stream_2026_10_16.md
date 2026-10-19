# RBPF GPU尤度を専用ストリームで実行する (D248, 2026-10-16)

**GPUの尤度経路へ戻り、専用ストリーム版を実装して検証した。全実入力で旧GPUとbit一致したが、速度改善は1%未満、最適化CPUより約2.1〜2.6倍遅いため採用しない。** GPU joint component gate 0/2、対照は全てPASS。製品・既定・D236 PF基準は維持する。

## GPU化の範囲

この実験はRBPF尤度の既存CUDA経路を対象とする。残差、whitening、group nuisance、初期solveと元2IRLS、小行列LU、mixture/Laplace尤度はGPUで実行する。dedup/pack/要求順へのscatter、厳密な整数処理、particle choice/RNG、FIX/feedback、GTSAM FGOはCPUのまま。新しい数値kernelを追加してGPUへ載せる処理範囲を増やした変更ではなく、既存GPU経路の転送・launch・完了待ちを見直したもの。GPU使用率や全体のGPU時間比率を測ったものではない。

## 変更と同期

D226 pageable版の数値関数とkernelをworkspace定義直前までbyte単位で複製した。GPU機械語5144命令のopcode/operand正規化hashも元と一致、56registers、shared30904B、stack32B/thread、spill0。数値式、full covarianceと微小な非対称、pivot/tie、行順の逐次和、FP64、元2IRLS、全観測/仮説/prior/閾値/RNGを変えない。

workspaceがnonblocking streamを1本所有し、H2D→元kernel→D2Hを同じstreamへ投入し、最後にそのstreamの完了を待ってからhost結果を読む。同じDLLのdefault controlは元の同期copy/既定streamを使う。選択は最初のscore以前だけ可能。未完了のqueued workが残るerror pathとdestructorでもstreamを待ち、host/device stagingの再使用・解放前に完了させる。正常のscoreは元のPython packing/fallbackを継承する。通常scoreでCUDA diagnostic eventはrecordしない。

pageable host stagingは元のまま。AsyncというAPI名だけからhost非同期化やcopy/kernelのoverlapを仮定しない。pageable copyはhostに対し同期になる場合があり、nonblocking streamはlegacy default streamとの暗黙同期を避ける指定。[CUDA12.8 API同期仕様](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-runtime-api/api-sync-behavior.html)、[stream仕様](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-runtime-api/stream-sync-behavior.html)を根拠にし、既存の遅さの原因がその暗黙同期だとは断定しない。新しいpinned allocation、数値cache、device常駐跨epoch結果の使い回しはない。同一workspaceへの並行呼出しは対象外。

## 精度と検査

81tests PASS、既知pytest設定warning1。3seed×3rows/groups×6数値条件で、CPU/GPUそれぞれ元DLL・同じ新DLLのdefault・nonblockingのscore/solution/normal/logdet/statusと入力bytesを比較した。CPU対GPUをbit一致とは扱わず、それぞれの元backendに一致させる。partial/odd alignment、0行、通常の返却arrayの所有権、次callによるworkspace上書き、capacity/rows/groups/dtype/singular/strict/警告/例外/emptyの元fallback、profile counterの非stale、選択の不変性、24回context生成/完了/二重closeも確認した。

3seed×4構成×4epoch、旧GPU/default/専用streamの全PF state/RNG/graph eventを比較しbit一致した。memcheck/racecheck/synccheck各13targeted tests PASS（81testsのsubset、各既知設定warning1）、error/hazard/leak0。native CPU拡張や一般versionのASan保証ではない。

## 全394実入力での転送込み比較

D243で保存したTokyo run3/seed20260710、cluster0/1各197バッチの全入力をSHA付きで再生し、速度による選別をしない。旧fused CPU、旧GPU、新DLL default GPU、新DLL専用stream GPU、GPU A/A、CPU A/Aの6modeを5round×3groupで巡回/反転した。35460timed、2364warm。全warmのscore・途中行列・statusが同一backendの元とbit一致し、CPUは保存した元実走結果にも一致。全timedの返却hashもwarmに一致する。

1許可CPU/BLAS1/通常priority/非占有、GPU処理の重複なし。setup/restore/copy/hash/IOは時間外、score内のdedup/DD/cluster/packing/upload/kernel/完了待ち/readback/scatter/fallbackは全て時間内。各groupの全197バッチの整数ns合計からpaired短縮率を求め、round内3group中央値→5round中央値で判定する。旧GPU・同じDLL default GPU・CPUの全てに10%以上、4/5改善、GPU A/A・CPU A/A・default controlの中央値abs5%未満を要求する。全raw/負値/対照を保持し、再測定しない。

時間列は197バッチ合計時間の中央値/197による平均batch時間、個別batchのp50ではない。短縮率は別のpaired統計量。

| cluster | CPU ms | 旧GPU ms | stream GPU ms | 対旧GPU短縮 | 対同DLL default短縮 | 対CPU短縮 | GPU A/A | CPU A/A |
|:---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1.226 | 3.212 | 3.177 | +0.942% | +0.014% | -160.065% | +0.445% | +0.603% |
| 1 | 1.838 | 3.839 | 3.786 | +0.765% | +0.407% | -106.847% | -0.277% | +1.236% |

旧GPU比+0.94/+0.76%、同じDLLのdefault比+0.014/+0.407%。stream切替による大きな改善は確認できない。GPU/CPU A/Aとdefault controlは両設定で条件内だが、速度条件を満たさず0/2。D243の別候補のreject判断やD247のCPU component改善を上書き/加算しない。PF全体/FGO全体の新しい速度比較は実施せず、componentの失敗を全体の成功へ読み替えない。

## 別のevent診断

全speed計測終了後、同DLL default/streamを全394入力へ別途適用し、計788callでeventを記録した。全出力bitsは変わらない。eventはhost input staging後、H2D前、kernel前後、D2H後に置く。event区間にはenqueue gapや他処理との競合が入り得る。以下の列はそれぞれ個別callのp50で、足して全体時間やCPU/GPUの排他分割にはできない。計測ありのprofile wallを速度採否へ使わない。

| cluster | mode | profile全score wall p50 ms | upload event ms | kernel event ms | readback event ms |
|:---:|:---|---:|---:|---:|---:|
| 0 | default_gpu | 2.794 | 0.376 | 1.322 | 0.155 |
| 0 | stream_gpu | 2.895 | 0.483 | 1.323 | 0.136 |
| 1 | default_gpu | 3.335 | 0.371 | 1.273 | 0.162 |
| 1 | stream_gpu | 3.431 | 0.478 | 1.273 | 0.140 |

kernel event区間は約1.27〜1.32msで、streamを変えてもほぼ同じ。今回の転送方式変更ではCPUとの差が埋まらなかった。GPUで実行する仕事のまとめ方やkernel内部の並列化/共有できる計算が次の検討対象となるが、このevent値だけから原因・改善上限・CUDA Graphの効果を断定しない。

## GPU実走

元D236のGPU基準と同じrun3/seed/400epoch、cluster0/1各1prefixで明示stream GPUを実行。計400保存PF state行、全trajectory配列dtype/shape/bytes、全cfg/全既存caller metadataが凍結旧GPUと一致した。各197valid/3skip、計394callでGPU scoreが選択され、全batchのunique数が正、scalar fallback0。source/input/外部tc/cssrlib/3CPU拡張のhashも確認した。FGO/整数探索を新しくGPUへ移したという意味ではない。

field helperは元D236 bodyを別globals/proxyで使い、GPU scorerだけを置換する。field/base helperのmain・scorer・selector bindingは元moduleへ代入せずproxy内に閉じ、既存NativeFeedbackMassStepが選択される時点でそのbindingを渡す。元field自身が行うruntime adapterの一時install/restoreは踏襲する。transport.jsonのoriginal_modules_not_mutatedは新helperによるfield/base bindingの差替えがないことを示す。実走の追加counter/保存は診断のみで、call_secondsから速度を認定しない。新しいscore ABI/DLLのhashが旧DLLと異なることを明示し、数値prefix/SASS/返却bitsは別に照合した。nativeと元NumPy間の既存微小差、scalar NaN codecの既存payload正規化は継承し、新しい数値許容差は加えない。全route/multiseed/GPU既定への精度gateは未完了。

capacity512のhost/device stagingはそれぞれ1143352Bで元と同じ。比較時4contextsで各4573408B。通常のstream context1本、profile用eventは後段の2contexts各4個。runtime/driverが持つstream/event/context領域とCPU heap peakは未測定。workspace objectへhandles/controlを追加したが、新しいdevice数値bufferはない。

36製品source、5installed PEs、PF build PE、旧数値DLL/3拡張、56既報、27archive/4196filesを保全する。実験APIだけを保存し、製品/既定dispatchのGPU化は今回も見送る。

[結果](../benchmarks/rbpf_gpu_stream_20261016.json)、[完了照合](../benchmarks/rbpf_gpu_stream_completion_check_20261016.json)、[scorer](../benchmarks/rbpf_gpu_stream.py)、[probe/API](../benchmarks/probes/rbpf_gpu_stream/README.md)、[実入力比較](../benchmarks/bench_rbpf_gpu_stream.py)。
