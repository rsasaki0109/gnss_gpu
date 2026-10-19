# 同じepochを独立RTKで交互に測る (D271, 2026-10-17)

計測方式のA/A安定性gateは**PASS**。28tests PASS。独立した元D236CPUを12prefix、計4800RTKepoch/2400PFcall実行し、全軌跡array・設定・入力が元にbits一致した。最適化候補や新しい高速化率はなく、製品/既定は変更しない。

## 実行ごとの時間変動を小さな比較間隔へ区切る

D270では同じ処理の別run間にも大きな差が出た。今回の新しい計測では、独立した2つのRTK processを同じepochまで進め、Aのprocess呼出しとAAのprocess呼出しを直列に測る。最初に測る側をepoch/pairごとに交代し、各phaseで両側が100回ずつ先行する。

両側がreadyになるまで計測を始めず、両側がdoneになるまでadvanceを送らない。先に終わった側の次epoch準備も相手のtimerに重ならない。標準入出力のready/go/done/advanceは3つのclockの外。親は小さなstdout messageを読み、各子の全logと全raw sampleを保存する。タイムアウトや片方の異常終了では、その実験で所有する子だけを終了する。

凍結D270 field mainのcode objectをそのまま別globalsで実行し、clock factoryだけを差し替える。元のprocess関数を同じself/引数で1回呼び、全FP64/順序/観測/粒子/候補/prior/閾値/RNG/FIX/feedbackを保持する。2つのRTK stateは別processなので共有しない。Numba cacheもlabelごとに分け、同時cache書込みを避ける。

28試験が最初の実行でPASS。通信I/Oがtimer外であること、両readyと両doneのbarrier、先行順、戻り値identity、例外identity/部分変更、再入拒否、不正command/sample/phase/欠落、実子processの完走・途中終了・不正JSON・timeout・残余event検出を確認した。試験launcher作成時のshell引用符エラーと欠落file起動は数値実行前で、保存launcherの試験結果には含めない。

## 実データ対照

Tokyo run3/seed20260710、nativeCPU/cluster0、各400epoch。各子は許可CPU1本/BLAS1/通常優先度、非占有環境。1完全warm pairの後、5対照pairを測る。初期の遅いepochや不利な組も除かず、phase2全200epoch合計と全400epoch合計を評価する。全12軌跡の全array dtype/shape/bytes、設定、入力、source/PE/DLLを別auditで照合した。全9600protocol eventと親sampleが一致した。

事前gateは、各全400/phase2について、5組の**絶対差の中央値が5%未満、かつ4/5組以上の絶対差が5%未満**。符号付き中央値の相殺を合格に使わない。CPU時計は補助診断で、integer tick合計後に比を取る。下表は `(A-AA)/A` の差で、同じ処理のため高速化率ではない。

| 対照組 | 全400 wall差 | phase2 wall差 | 全400 threadCPU差 | phase2 threadCPU差 |
|---|---:|---:|---:|---:|
| 1 | +0.359% | +0.583% | -1.051% | -1.662% |
| 2 | +0.101% | +0.771% | -0.632% | +0.798% |
| 3 | +0.809% | +2.033% | +0.000% | +0.106% |
| 4 | +1.442% | +2.660% | +1.969% | +2.162% |
| 5 | +0.906% | +4.300% | +0.611% | +3.439% |

全400の絶対差中央値 0.809%、5%未満 5/5組。phase2は 2.033%、5/5組。両方を満たす判定はPASS。全warm/対照/負値を保存し、完了済み計測の取り直しは行っていない。D269以前の候補をこの方式で再測定して採用する実験は行っていない。

phase2では5/5組でAA側が短く、方向の偏りも残っている。原因は確定していない。5%の校正条件を通っても、小さな候補差を無条件に効果へ数えない。次の候補評価では同じ処理の対照を残し、候補と対照を割り当てるlabelもpair間で交換する必要がある。

## 適用範囲

これは強制待機を含む新しい比較方法の校正である。待機はtimer外でもcache/working set/OS schedulingに影響し得るため、連続routeの実時間性能や計測overheadの校正ではない。新しい候補に使う場合も、元の連続実行で全体の速度と精度を確認する必要がある。今回の対照が安定しても、高速化候補の採用や過去のFAILの変更にはならない。

1route/seed/clusterのCPU診断に限る。新しい保存PF state照合、GPU演算計測/使用率、全route/多seedの精度・速度は確立していない。時計の量子化、stdout readerの小さな負荷、診断snapshotを除く既存codecの制限を引き継ぐ。36製品source・5installed PE・PF build PE・102既報・50archive/9588filesを保全する。

[結果](../benchmarks/rbpf_epoch_lockstep_20261017.json)、[完了照合](../benchmarks/rbpf_epoch_lockstep_completion_check_20261017.json)、[barrier/clock](../benchmarks/rbpf_epoch_lockstep.py)、[field runner](../benchmarks/bench_rbpf_epoch_lockstep_field.py)、[試験](../tests/test_rbpf_epoch_lockstep.py)。
