# D291: 親側だけの監視で400epochの終了を確認 (2026-10-19)

基準`none`と一度だけ事前転送を重ねる`overlap`を、各1回・独立したfresh worker/cacheで直列に実行した。両方とも400epochを完走し、Python起動用プロセス・計算worker・観測したOS version照会`cmd.exe /c ver`の計6processがsignalled exit0。timeout・強制終了・監視error・残存workerは0。数値処理内のperiodic stack dump、native debugger、sanitizerは使わない。`faulthandler.enable()`によるfatal出力は維持する。

全800RTK clocks、400PF calls、394score intervalsを保持した。最終の宣言済みstate boundary（rootRTKを含む）2件、57NPZ arrays×2=114配列のdtype/shape/raw bytes、57raw NPY members×2=114件が旧goldと完全一致した。元のFP64数学・IRLS・prior・乱数・測位条件・通常GC・CUDA/CPU PEsは変更0。今回も1route/seedの400epoch prefixであり、各epochの全state captureやfull-route/multiseed精度保証ではない。

親の監視は、数値import前のready/go待ち合わせでvenv launcherと実workerのPID/PPID/command/creation timeを確認し、開いたhandleを終了まで保持する。親の0.1s pollingは子のPython frameを読まない。結果file・completed markerの有無では待機を終えず、handleのsignalled状態と実exit codeを確認する。timeout時は確認済みhandleのみを停止し、終了を再確認する。任意のdetached/短命子processをすべて観測・封じ込める方式ではない。

新native monitor回帰6件と、保存済みprocess evidenceのportable検証15件、計21件PASS。正常/非0/259 exit、launcher/worker分離、結果・marker保存後のhang、観測した子のcleanupと無関係なprocessの維持を検証した。検証policyは追加子の非0/未終了/異常command/親子関係不一致も拒否する。Windows専用テストは非Windowsでmodule skipし、portable検証はnative APIを使用しない。Linux上の実行は今回未実施。

初回launch2件は出力親directory作成漏れによるFileNotFoundErrorで、scorer作成・測位開始より前にexit1となった。旧runner/worker/helper/tests/plan/logは保持し、新runnerで出力祖先だけを測位前に準備した。数値prefixを開始したのは修正版の2件だけで、完成した失敗速度setの取り直しはしていない。

初回auditorは「processは2つ」という誤った想定でlifecycleをfalseにした。この失敗も保持した。実記録には各worker直下の`cmd.exe /c ver`があり、すべてexit0だった。ローカルCPython `platform.py`の`_syscmd_ver`には一致するshell照会があるが、その呼出元import stackは今回記録しておらず断定しない。新validatorは正確な追加command・PID/PPID・生成順・3件すべてのsignalled exit0を要求し、観測processを除外しない。実行済みhelper/docstring・plan内の「子processを作らない」という記述はこのOS照会により不正確であり、元bytesを保全してここに訂正する。新auditorのための数値再走は0。

今回確認できたのは親だけの監視条件での2回の正常終了と保存結果一致である。D289の古いcrash原因、D290 case5の失われた終了履歴、sanitizer終了hangの原因が解明したとは扱わない。速度gateは未実施で、過去の不合格・未完了gateを維持し、overlap候補は引き続き未採用。今回のtimingを速度差・GPU使用率・GPU化率として公表しない。cold startを含む全epochを残した。

Windows終了確認の根拠は[WaitForSingleObject](https://learn.microsoft.com/en-us/windows/win32/api/synchapi/nf-synchapi-waitforsingleobject)と[TerminateProcess](https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-terminateprocess)の公式仕様。停止要求の返り値だけでは終了確認としない。

[結果](../benchmarks/rbpf_lifecycle_20261019.json)、[完了照合](../benchmarks/rbpf_lifecycle_completion_check_20261019.json)、[親monitor](../benchmarks/rbpf_process_lifecycle.py)、[保存記録validator](../benchmarks/rbpf_lifecycle_validation.py)、[native回帰](../tests/test_rbpf_process_lifecycle.py)、[portable回帰](../tests/test_rbpf_lifecycle_validation.py)。144既報・71archives・18058filesと製品36sources/5installedPE/PFbuildを保全する。
