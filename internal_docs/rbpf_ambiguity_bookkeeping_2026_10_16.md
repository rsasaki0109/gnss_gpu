# 曖昧度の整数管理をまとめる (D247, 2026-10-16)

**全400実入力で曖昧度更新全体の短縮を比較。RTK全体はgate FAIL、RTK全体の効果が確認できないため全体の高速化として採用しない。製品・既定は変えない。**

## 変更の対象

D246後に残る全衛星×周波数のPython走査を減らす。事前の非計測400入力調査ではedt/slipはint64の3周波数配列、行のnonzero集約呼出しは45174回だった。小さな行の判定だけを置き換える案から、整数カウンター更新とリセット対象の抽出を周波数ごとにまとめる案へ進めた。FP演算をGPUへ移す変更ではなく、ホストの独立した管理処理をまとめる変更。

元の最初のfor loopはoutcを各衛星について増やし、outage/edt/slipとsystemとx[j]から対象を判定してinitxを呼ぶ。候補は通常の非重複int64配列についてoutc[:,f]を増やし、同じ整数条件を配列で評価して、対象のinitxとoutc/slipのresetを元と同じ昇順で実行する。各周波数の観測loop、波長・cp-pr/lam等のFP式、全観測、全粒子/候補/prior、FP64、閾値、FIX/feedback/graph、RNGは元のまま。各周波数でflagsを再評価し、前の周波数のslip消費を反映する。観測を間引かず、すでにゼロと仮定してリセットを省略しない。

D246の保護版initxと組み合わせる。D246単体の全RTK比較のFAILは維持し、今回の対照に使っただけで採用済みとは扱わない。新しい全RTK比較は処理の異なるD247候補であり、D246の同じ候補を再測定したものではない。

## 制限と同一性

元rtk.py/pppssr.pyのhashとliteral codeを検証し、変更した最初のloopを戻すと元update_ambiguitiesのASTが一致する。観測/FP loopはそのまま保持する。完全な元Nav schema、通常のndarray/int64/float64/dim/shape/stride/writeability/index条件、配列間の非alias、literal IB/row predicate/initx、通常の辞書keys等を確認する。float xの該当範囲にNaN/Infがある場合、int64カウンターが上限の場合、重複要素/alias、read-only、不適合bufferやcustom callbackの場合は、元の衛星loopへ戻す。global binding/codeが変われば元update全体へ戻す。巨大整数/custom scalarのinitxはD246保護版から元initializerへ戻る。整数の切り詰め、飽和、許容差の追加はない。

まとめたoutc更新をresetより前へ動かせるのは、元initxがx/Pだけを触り、配列間のaliasとcustom callbackがない検証済み領域だから。custom initxが途中のoutcを読むtestでは元の順序へ戻り、callごとの全outc bytesも一致させる。OOM、同じ状態への並行書込み、任意cssrlib versionの互換性を一般保証するものではない。constructorでのsource検証と固定system index作成はprocessタイマー外、各更新のguards/対象抽出/counters/一括store/resetは時間内。

新31testsと凍結した保護版38testsの計69tests PASS（既知pytest設定warning1）。nf=1/2/3/5の4条件×3seed×4epochの全変更array bits/全initx引数順序、F-order、符号付きゼロ、NaN/Inf、overflowとstrict-policy warning/途中state、read-only、strided/要素重複/alias、dtype、index/bounds、custom callback、binding変更を確認した。新しいCUDA/native kernelはなく、既存の数値DLL/3拡張やPF/FGO modelは変えない。

## 全400入力のcomponent比較

元のcluster0/1各200phase2入力を全て再生し、元処理、保護版initxのみ、今回の組合せ、元処理A/A、initx A/Aの5modeを5round×3groupで巡回した。計30000timed/2000warm。roundごとに全200入力を合算し、5roundのpaired短縮率中央値を使う。1許可CPU/BLAS1/通常priority/非占有。restore/deepcopy/IO/出力照合は時間外。全modeの全x/P/outc/edt/slip dtype/shape/bytes、観測、initx counterを確認し、candidateの計測・warm全6400update/19200scanでfallbackなしだった。独立した非計測再生400update/1200scanでもfallbackはなく、全400入力の全変更配列と全5189initx引数の順序/bytesが一致した。

| cluster | 元処理から短縮 | initxのみから追加短縮 | 元処理 A/A | initx A/A | gate |
|:---:|---:|---:|---:|---:|:---:|
| 0 | +81.26% | +48.78% | +1.34% | +1.59% | PASS |
| 1 | +80.64% | +47.35% | -0.23% | -0.78% | PASS |

事前gateは元処理から30%以上、initxのみから20%以上、両方4/5round以上改善、両A/A中央値abs5%未満。新しい組合せのcomponent gateは2/2。良いroundだけを選ばず、不利な対照/全raw sampleも保存する。ここまでの数字は曖昧度更新全体のもの。

## 実走の精度と全RTK比較

native_cpuの元D236 PFを使い、cluster0/1各400RTK epochを実行。計400保存PF state、全trajectory配列のdtype/shape/bytes、全cfgと既存metadataがD236と一致した。各prefix197valid/3skip。全400component captureも、今回の入力/出力/obs/signal metadata/initx counterまでD246初版captureと一致。capture helperは元bodyから選択updateだけを差し替え、parent呼出しを明示した2箇所のAST undoを検証した。継承metadataのoriginal_update_source_retainedは外部sourceの保全を表し、実行updateの変更はbookkeeping.jsonに記録した。

全RTK速度は、診断snapshot/deepcopy/encoding/gzip/metadataだけを元field helperから除き、AST undoを確認した元D246のタイミングbodyを別globalsで再利用した。variant名bulkは今回の組合せを選択する。D246のsource/globalsは変えない。cluster0/run3/seed20260710/400epoch、3fullprefix warmの後、reference/combined/AAを5roundで巡回し、計18prefixの全7200epochのtrajectory/cfg/source/input/PE/DLLを照合した。速度判定に全phase2の200epochと全400epochを含める。

| round | phase2短縮 | phase2 A/A | 全400epoch短縮 | 全400 A/A |
|:---:|---:|---:|---:|---:|
| 1 | +53.87% | +50.76% | +52.07% | +47.94% |
| 2 | +20.69% | +18.93% | +22.29% | +23.39% |
| 3 | +4.39% | -27.77% | +9.29% | -31.29% |
| 4 | +18.23% | -8.36% | +20.48% | -2.71% |
| 5 | +7.21% | -66.28% | +13.35% | -39.48% |

phase2中央値+18.23%、5/5改善、A/A -8.36%。全400epoch中央値+20.48%、5/5改善、A/A -2.71%。事前gateはphase2で5%以上、全400epochで3%以上、各4/5改善、各A/A中央値abs5%未満。全体gate FAIL、RTK全体の効果が確認できないため全体の高速化として採用しない。今回はphase2のA/A中央値がabs5%条件を超えた。全400epochのA/A中央値だけでは基準内でも、各roundの対照は大きく変動しており、観測した全体短縮率を候補の効果として確定しない。 全roundを保存し、同じ候補を再測定して採用へ寄せない。

これは1route/1seedのprefix比較。cluster1は実走state/component精度を確認したが、全RTK速度は比較していない。全application起動/I/O、fullroute/multiseed、GPU backendの速度/精度gateは未完了。製品/既定は維持する。小さいCPU一時配列はO(MAXSAT×nf)、固定system indexはO(MAXSAT)。新device/native workspaceなし、CPU heap peakは未測定。既存scalar NaN codecの制限は継承し、array bitsは許容差なしで照合した。

36製品source、5installed PEs、PF build PE、数値DLL/3拡張、54既報、26archive/3849filesを保全する。新しいGPU実走やsanitizerを実行したとは扱わない。

[結果](../benchmarks/rbpf_ambiguity_bookkeeping_20261016.json)、[完了照合](../benchmarks/rbpf_ambiguity_bookkeeping_completion_check_20261016.json)、[実入力再生](../benchmarks/bench_rbpf_ambiguity_bookkeeping.py)、[候補factory](../benchmarks/rbpf_ambiguity_bookkeeping.py)。
