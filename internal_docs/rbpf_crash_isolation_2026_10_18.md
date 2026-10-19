# D290: 定期stack dump側のアクセス違反を分離 (2026-10-18)

実FGO/GPUの式を変えず、検査用の周期stack dumpを行うC thread側のアクセス違反を再現した。今回の2件はどちらもpython312.dll内で、唯一追加されたwatchdog thread上で発生した。CUDA/GTSAMのDLLを読み込む前であり、GPU追加threadを無効にした条件でも再現する。今回の再現にはGPU計算・FGO数値計算は不要だった。一方、native fault情報のなかったD289の旧障害を同じ原因と断定しない。

14unique tests PASS (native捕捉11件と親timeout回帰3件)。周期出力なしの2本は400epochの数値結果を保存し、2最終PF/RNG/宣言済みRTK境界と2軌跡114NPY memberが元結果とbyte一致した。ただし最後の候補側は数値完了後に親観測器で別エラーが出て、最終process exit/native eventsが不明である。正常終了まで確認した実FGOはbaseline1本。前回の不採用や未成立の速度setは維持する。

## 新しい診断条件

WindowsのDEBUG_PROCESSで新規childとvenv redirectorの子だけを観測し、例外のfirst/second chance、process/thread ID、moduleの実image範囲、fault address、x64整数registerを読む。初回loader breakpoint以外はDBG_EXCEPTION_NOT_HANDLEDで通常の例外処理へ渡し、コード/register/数値状態を書かない。[Windows debug loop](https://learn.microsoft.com/en-us/windows/win32/debug/writing-the-debugger-s-main-loop)と[ContinueDebugEvent](https://learn.microsoft.com/en-us/windows/win32/api/debugapi/nf-debugapi-continuedebugevent)に従う。process/threadのdebug handleはEXIT後のContinueDebugEventでOSが閉じ、image file handleだけを明示closeする。

数値source/PE/seed/FP64/IRLS/prior/観測/仮説/閾値/順序はD289のまま。新6ケースを事前に宣言し、同じCPU1/BLAS1/通常GCで直列に実行した。Pythonだけの4秒frame/callback churnを周期出力なし/1ms指定で行い、実FGO callerはGPU追加thread none/overlapと周期出力なし/20ms指定の4条件にした。周期指定はサンプル数や実際の頻度を保証せず、debugger/頻繁なdumpの観測費用も大きい。速度の数字・改善率・gateには使わない。

[faulthandler仕様](https://docs.python.org/3.12/library/faulthandler.html)ではdump_traceback_laterはwatchdog threadを使う。arm前後のnative thread IDを保存し、追加が1個だったこととfault TIDの一致を確認した。周期logのfileはcancelまで保持し、fatal handlerはstderrへ残す。この出力先・周期とdebugger追加はD289とは異なる条件であり、旧障害の厳密な再現ではない。

| trial | workload | GPU追加thread | 周期指定 秒 | 結果 | 保存dump数 |
|---|---|---|---:|---|---:|
| 0 | pure | none | 0 | exit 0 | 0 |
| 1 | pure | none | 0.001 | exit 0 | 186 |
| 2 | field | none | 0 | exit 0 | 0 |
| 3 | field | none | 0.02 | exit 3221225477 | 8 |
| 4 | field | overlap | 0.02 | exit 3221225477 | 19 |
| 5 | field | overlap | 0 | 数値完了、親側失敗、exit不明 | 0 |

## native捕捉で分かったこと

trial3はmain TID12820に対し、追加watchdog TID33588上でpython312.dll+0x27EA51のread access violationを起こした。読取先は9、RDX=1、fault instructionはmov rax,[rdx+8]だった。trial4はmain TID25124に対しwatchdog TID48212、同DLL+0x27E990でread access violation、読取先0x10000006F、RSI=0xFFFFFFFF、instructionはmov rax,[rsi+0x70]だった。first/second chanceの4recordsを保存し、各RIPとExceptionAddress/module rangeの一致を再検算した。

両caseとも、その時点のloaded-module一覧にnvcuda/cudart/元scorer DLL/GTSAMがない。周期traceはmodule import途中で切れている。追加GPU処理を始める前の障害であり、今回の2件をGPUの計算誤差やFGOの数値変更の結果と扱わない。ただしPython内部でどの不変条件が壊れたか、該当source行/既知修正との同一性、D289旧障害やsanitizer終了停止との因果までは証明していない。

## 観測器の別エラーと修正

trial5は400epoch、最終境界/57NPY memberの一致とworker完了markerを保存した。その後、親の240秒timeout分岐がTerminateProcessを呼び、WinError5で例外になったため、case5のnative event recordと全体aggregateをpublishできなかった。この履歴は欠落のまま保持し、fault0/正常exit0を補わない。全6数値診断の再走もしない。

[TerminateProcess仕様](https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-terminateprocess)では、終了済みprocessに対する呼出しでもerror5となり得る。別の親API adapterを作り、error5のときGetExitCodeProcessがSTILL_ACTIVE以外を返した場合だけ、既終了として扱うよう修正した。元debug loopのcode objectはそのままで、module globalsを書き換えずexplicitなcopyへAPIだけ渡す。終了済みchildの元exit37維持、稼働中childのtimeout124、handled AVのnative捕捉/通常処理維持の3件が通った。これは対応するOS edgeへの修正であり、失われたcase5のエラーがこの分岐だった確証ではない。

## 試験と精度の範囲

最初の捕捉器テストはSIGSEGV fixtureをWindows AVと取り違え1fail/10passとなった。元source/logを保存し、別checked suiteでWindowsの無効thread-startによるAVと、ctypesでhandledとなるAVを分けた。11件のchecked testは構造体/owner native TID/通常exit/first-second chance/module範囲/register/例外を飲み込まないこと/venv相当のnested child/timeoutを確認する。14件は重複する旧10passを加算していない。

凍結した初版fixtureを通常の収集で再実行しないよう、診断終了後にpyproject.tomlのpytest ignoreへその1pathだけ追加した。旧config bytesを保存し、他の数値source/test/runner/PEは未変更。checked11+回帰3が通常収集されることをcollect-onlyで確認した。最初の収集起動はWindows command-line長制限でchild作成前に失敗し、別JSON argv/script経由で収集だけを確認した。これらの失敗も保存し、数値診断を再実行していない。

独立auditorは数値/観測helperをimportせず、保存5native event treeの全exit/disposition/module範囲、追加watchdogとの対応、欠落したcase5と親側例外、800RTK時計/394score区間、2最終境界と展開した114NPY memberを照合した。宣言済みstate encoderはscalar NaN payloadをcanonicalizeする制約があり、GTSAMの全内部graph snapshotではない。array/NPZはraw bytesで比較する。残る所有workerは0。

140既報、69archives/17212files、36製品source、5installed PEとPF buildを保全した。製品source/精度/実行時既定は維持し、変更した設定はpytestの初版fixture収集除外だけ。GPU使用率、継続throughput、全route/多seedの精度、D289の速度gate修復を主張しない。[結果](../benchmarks/rbpf_crash_isolation_20261018.json)、[完了照合](../benchmarks/rbpf_crash_isolation_completion_check_20261018.json)、[元捕捉器](../benchmarks/rbpf_windows_debug.py)、[親timeout修正](../benchmarks/rbpf_windows_debug_checked.py)、[診断caller](../benchmarks/bench_rbpf_crash_isolation.py)、[checked tests](../tests/test_rbpf_windows_debug_checked.py)、[回帰tests](../tests/test_rbpf_windows_debug_timeout.py)。
