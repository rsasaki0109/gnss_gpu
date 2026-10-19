# D289: FGO移行とGPU待機の重ね合わせは安定性で不採用 (2026-10-18)

元FGOがPhase 2へ移行する時点で一度だけ別threadから1B転送を始め、必須CPU処理と待ちを重ねる候補を実装した。実FGO比較の候補側でアクセス違反が発生し、追加のsanitizer検査も終了処理で停止したため不採用とした。速度の判定は未成立。成功した試行だけを使う比較、既完了gateの再試験、数値条件の緩和は行わない。製品・既定・精度設定は維持する。

27unique tests PASS、保存済み実score394入力の値/全diagnostic/statusとPython入力は元GPUにbit一致した。新しい400epoch accuracy prefixも200個のPF/RNG/宣言済みRTK境界と元metadataが一致する。完了したaccuracy/controlの2prefixは800RTK時計、400PF呼出し、394score区間、2最終境界と2軌跡114配列が一致した。ただし次の候補primaryが異常終了しており、候補全体の安全性・測位精度を確立したとは扱わない。

## 実装と事前宣言

新しい数値DLLやkernelを作らずD288 DLLをそのまま再利用した。元RegisterLUScorerの完全score、FP64、IRLS回数、prior、観測/仮説/閾値、乱数とgraph処理順は維持する。元RTK processは同じ関数のまま。instance-localな移行wrapperが元_transition_to_phase2を同じ引数で一度呼ぶ直前にhookを呼ぶ。移行policyにepoch番号や保存データの将来値を埋め込まない。

hookはprivate host staging先頭1byteをCPU初期化し、non-daemon threadを一度起動する。workerは作成元deviceをcudaSetDeviceで選び、既存D288のH2D1B+device synchronizeを完了する。通常scoreのpackより前、またはclose時にjoinし、worker障害を伝えた後もnative workspaceを解放する構造とした。次の元scoreが全active inputsをpack/uploadし直す。継続keepalive、clock/power/driver設定、GC/priority方針は変更しない。

[CUDA device API](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-runtime-api/group__CUDART__DEVICE.html)のcurrent deviceはhost thread単位なので、新threadでもdeviceを明示した。[Python ctypes](https://docs.python.org/3.12/library/ctypes.html)のCDLL呼出しはGILを解放する。これは実装上の前提であり、今回の異常終了の原因を特定する説明ではない。

数値source/test/caller/runner/入力/PE/gateは最初の実行前に凍結した。独立processのaccuracy1本とprimary15本、計16本を宣言し、primaryはA/overlap/AAの5巡にした。方式間でGPUを温め合うことを避けるため各試行を別processとし、各fresh NUMBA cache、CPU1/BLAS1、normal GC/priority、host/GPU非占有とした。初回ticket準備後に共通15秒idleを1回置き、driver全経過に保持する。移行hook/CPU prepare/thread開始/device選択/完了/joinはRTK呼出し時計内に含む。共通scorer作成はdriver時計より前で、process全経過は別のrun記録に残る。

一次指標は移行RTK呼出し開始から、初めてscoreしたRTK呼出し終了までのwall時間とし、全400epoch/driver全体も残す予定だった。事前gateはA/A絶対差中央値<5%かつ4/5巡の絶対差<5%、両controlに対して短縮中央値>=5%かつ4/5巡で正、である。データが揃わず、このgateは計算していない。D282/D284/D287の既完了FAILも保持する。

## 実走の失敗

| trial | 方式 | 結果 |
|---|---|---|
| 0 | overlap、accuracy専用 | 400epoch完了、200状態/元metadata/最終境界/57NPY member一致 |
| 1 | A、primary control | 400epoch完了、最終境界/57NPY member一致 |
| 2 | overlap、primary | ep300進捗の後にaccess violation、exit3221225477 (0xC0000005)。完全result/final proofなし |
| 3〜15 | 残る13本 | 元runnerが異常終了で停止し、未実行 |

trial2のlogには元native scoreを含む60秒周期trace dumpと、その後のアクセス違反、component mass周辺のstackが記録された。どの処理が原因か、watchdogとの因果、正確な最終epochや未保存時計は特定していない。新しい数値sourceへの変更や、再走して成功結果へ置換することは行わなかった。partial directory/log/run記録をそのまま保全する。

trial0はaccuracy専用で状態保存費用があり、trial1とは速度比較できない。候補primaryは未完了であり、これら2成功例から短縮率を作らない。全体RTK throughput・GPU利用率・driver/power原因や全route/多seed精度も未確立。

## メモリ検査の停止と後片付け

別の安全性診断として元27件のうち7subsetをcompute-sanitizer/memcheck下で実行した。pytestは7passed/20deselectedまで進んだが、Pythonと検査ツールが終了せず、最終summaryも出ない状態が続いた。これは正常終了の確認にならない。

processのPID・作成時刻・親子関係・command lineを保存し、同じ実体であることを再確認した。自分が起動した検査対象Python38580だけを終了し、sanitizer40104は自らexit15で終了した。runner26788も失敗guardで終了した。無関係なprocessを止めず、3個の所有processが残っていないことを再確認した。

強制終了後のlogにはerror0/leak0bytesのsummaryが出たが、target error/exit15があり、clean memcheck PASSとして採用しない。停止前logも別に残した。続けて予定していたfull RTK sanitizer診断はguardで開始されていない。停止原因や新threadとの因果は未特定で、速度再試験や修正済みという主張には使わない。

## 精度の範囲と保全

27testsはone-shot/owner/開始失敗/prepare失敗/worker障害/完了順/close時drain/元transitionの引数・返値・例外/none追加作業0、元CPU/GPUの完全返却、両cluster全394実入力/入力不変、結果所有権とscreen拒否条件を検証する。sanitizer下の7件を別のunique数へ足さない。200状態はPF/RNGと宣言済みRTK境界の照合で、GTSAM全内部graphのraw snapshotではない。既存state encoderはscalar NaN payloadをcanonicalizeする制約がある。arrayとNPZはdtype/shape/dataをraw bytesで照合する。

独立failure auditorは数値helperをimportせず、完了2本の全800時計の整数和/順序/完了/包含、394score区間、200状態、2最終境界と展開した全114NPY memberを再照合した。trial2の失敗/完全出力の欠落/残13本の未実行、sanitizerの停止前後log/exit15と所有processのcleanupも照合する。成功用に事前作成したauditor/publisher/evaluatorは未実行のまま保存し、今回の合格記録として扱わない。

138既報、68archives/16608files、36製品source、5installed PEとPF buildを保全する。新しい数値source/test/PEを実行後に編集せず、製品の追加変更・既定採用・精度条件変更は0。[結果](../benchmarks/rbpf_transition_wake_20261018.json)、[完了照合](../benchmarks/rbpf_transition_wake_completion_check_20261018.json)、[候補scorer](../benchmarks/rbpf_transition_wake.py)、[実FGO caller](../benchmarks/bench_rbpf_transition_wake_field.py)、[tests](../tests/test_rbpf_transition_wake.py)。
