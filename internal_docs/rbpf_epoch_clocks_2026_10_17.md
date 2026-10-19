# 全RTK時間の揺れを3つの時計で診断する (D270, 2026-10-17)

18tests PASS。元D236CPUの同じ400epochを8回（warm2回と対照3組）、計3200RTKepoch/1600PFcall実行し、全軌跡arrayのdtype/shape/bytes、設定、入力hash、source/PE/DLLが元処理と一致した。候補の高速化実験ではなく、全体計測の揺れを調べる診断である。製品/既定・測位式は変更しない。

## 前回の時間差を保存データから確認

D269の全18runの各epoch時間をhash検証して再集計した。元処理と同じ元処理のtimed対照10runでも、phase2全200epochの合計は12.592〜23.165秒だった。差は最初の2epochだけでなくphase2の50epoch区間にも分布する。全データを残し、起動直後や遅い区間を除いてgateを通す処理は行わない。D269の全RTK gate FAILは維持する。このオフライン再集計は新しい速度測定ではない。

## 追加した診断

`EpochClocks`は元のprocess関数を同じself/引数で1回だけ呼び、戻り値/例外をそのまま渡す。RTK/PF stateへ計測値を入れず、各processの外側に経過時間、プロセスCPU時間、実行スレッドCPU時間を記録する。CPU時計の区間をwall区間が覆う順序で読む。PF/IMU/FGO/AR/全観測/粒子/候補/prior/FP64/閾値/RNG/FIX/feedbackの処理は元のまま。

凍結した既存field adapterを使い、診断state/deepcopy/JSON/gzipだけをprocessから除くAST undoを確認する。入力decode/setup/出力保存もprocessの外。全400epochとphase1/phase2を別々に集計し、全rawを保存する。例外・部分変更・戻り値identity・super closure・元関数/globals不変、時計を読む順、再帰/欠落/順序/不正phase/逆行/別thread、ゼロtick/多thread CPU/符号付き差の18試験が最初の実行で通った。

## 実データのA/A

Tokyo run3/seed20260710、nativeCPU/cluster0、許可CPU1本/BLAS1/通常優先度/非占有。AとAAの処理は同じで、ラベルは出力先と比較順だけに使う。warm2完全prefixの後に3組を順序回転する。最初の遅いepochも全400epoch集計に残す。前回候補の取り直し、採用gateや勝敗は設けない。

phase2の全200epoch合計（単位秒）:

| 組 | ラベル | 経過時間 | プロセスCPU | スレッドCPU | スレッドCPU/経過時間 |
|---|---|---:|---:|---:|---:|
| warm | A | 19.179 | 17.344 | 17.344 | 90.43% |
| warm | AA | 13.399 | 12.188 | 12.188 | 90.96% |
| 1 | A | 14.750 | 13.047 | 13.047 | 88.46% |
| 1 | AA | 17.478 | 16.031 | 16.031 | 91.72% |
| 2 | AA | 14.162 | 13.219 | 13.219 | 93.34% |
| 2 | A | 15.090 | 13.672 | 13.672 | 90.60% |
| 3 | A | 15.147 | 13.547 | 13.547 | 89.44% |
| 3 | AA | 13.367 | 12.672 | 12.672 | 94.80% |

対応するAに対するAAの差 `(A-AA)/A`。正の値はAAの時間が短いことを表す。処理が同じため高速化率ではない。

| 組 | 経過時間差 | プロセスCPU時間差 | スレッドCPU時間差 |
|---|---:|---:|---:|
| 1 | -18.499% | -22.874% | -22.874% |
| 2 | +6.146% | +3.314% | +3.314% |
| 3 | +11.749% | +6.459% | +6.459% |

3組の差の中央値はphase2経過時間 +6.146%、プロセスCPU +3.314%、スレッドCPU +3.314%。timed6prefixのphase2スレッドCPU/経過時間は88.46〜94.80%。CPU時間にも時間差がある場合、wallから待ち時間を引くだけでは対照差を解決できない。これらはこの診断runの観測であり、前回D269のCPU時間や原因を復元したものではない。

## 解釈と次の判断

CPU時計の正のepoch差の観測GCDは結果JSONに保存する。この環境ではCPU値の刻みが粗いため、短いepochのCPU/wall比の中央値やゼロ値除去を使わず、整数CPU tickを全epoch合計してから比を取る。観測GCDは時計精度や誤差上限の保証ではない。量子化による符号付き差をclipせず保持する。プロセスCPUには別worker threadが含まれ得る。wallとCPUの差だけでOS scheduling、blocking、CPU周波数、GC、JITのどれが原因かは判定できない。

計測追加のoverheadは未校正であり、非計測時のCPU占有率、bottleneck share、GPU使用率や高速化上限として使わない。3組/1route/seed/clusterだけで速度を採用しない。新しい保存PF state比較は行っておらず、ここでの精度確認は全軌跡array/設定/入力/同一source/PE/DLLに限る。D269以前の保存state検証やFAILは変更しない。

細かいcacheやkernel共有を追加する前に、この対照差を確認できる計測を次の全体評価にも残す。次の候補は既存D245/D246の粗い診断にあるPF/ARの重い処理を対象とし、全体wallの対照が安定しない限り速度採用は保留する。ARはすでにNumba実装であり、再JIT化だけの候補にはしない。観測削減や精度条件緩和も行わない。

36製品source・5installed PE・PF build PE・100既報・49archive/9345filesを保全する。[結果](../benchmarks/rbpf_epoch_clocks_20261017.json)、[完了照合](../benchmarks/rbpf_epoch_clocks_completion_check_20261017.json)、[clock helper](../benchmarks/rbpf_epoch_clocks.py)、[field診断](../benchmarks/bench_rbpf_epoch_clocks_field.py)、[試験](../tests/test_rbpf_epoch_clocks.py)。
