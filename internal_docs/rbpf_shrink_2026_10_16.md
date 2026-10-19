# RBPF release-subsetの最悪CP行をnativeで選ぶ (D237, 2026-10-16)

**CPU/GPUとも主screenは0/7で、既定への採用を見送る。前回D236を維持する。** `_shrink_worst`のDD展開とCP行ループをC++へまとめたが、完全PF-step比較では10%/4of5round/A/A条件を満たす追加効果を確立できなかった。A/A中央値が7条件中5条件でabs5%外。すべての負値・失敗を保持し、完了した速度測定は繰り返さない。

## 実装と精度の条件

凍結`_shrink_worst`の3statement（DD展開・初期最大値・行走査）だけをcallbackへ置換し、置換を戻したASTが元methodと一致することをassertする。後半の`_copy_H`、対象satの削除、2sat未満groupの削除、空H/None処理は文字どおり元のcodeを使う。候補生成回数、particle/move/prior、conditional/FIX/hold/graph/RNG、FP64・2IRLS・閾値は変えない。

nativeでは全pairを元順に展開し、任意精度PyLongを減算してから元のFP64へ変換する。CPで使わないpairの変換overflowも飛ばさない。CP行は元順で、`bmeas-dd` → `lam*residual` → `abs` → `max(sig,1e-6)` → 除算を行い、厳密な`wr > worst_r`だけで更新する。同点は最初の行、最大値の初期値は−1.0のまま。FP32、tree reduction、FMA/reassociation、新しい許容差を導入しない。MSVC `/O2 /fp:strict`、GIL保持、call中だけのDD vectorを使う。

元のplain source object、dict/tuple/int、1次元連続exact NumPy arraysを適格域にする。CPのlam/bmeas/sigと定義済みDDのabs1e100以下はoverflow/invalid演算を避ける**dispatch安全域**であり、モデル上の値制限ではない。それ以外、custom型/binding、unaligned/strided配列、非有限値、負index等は元のPython blockを1回実行する。strict NumPy policyは既存caller同様に元step全体へ戻し、途中state/RNG/例外を保つ。underflow ignoreも既存条件のまま。入力は元caller同様、block中に外部から変更されない前提。

実runtime/moduleをpatchせずD236のlocal viewだけにmethodを登録する。H/DD/problem cacheは新設しない。D226数値DLLとD233 canonical/D234 parent/D235 component-mass拡張はbyte単位で不変。D230 proposal、D231 warp、D232 packは併用しない。製品source・既定dispatchは維持する。

## 完全PF-stepの速度

D236 CPU/候補CPU/候補GPU/D236 GPU/CPU A/Aの5modeを直列回転し、7条件×5round×3group=525timed callsと35warm snapshotを保存した。D236と同じ6feedback有効/事前hold条件に1feedback無効条件を加える。**shrinkは7条件すべてで実際に使用する。** 候補や粒子を減らさない。新guard、buffer preflight、vector確保、DD/行走査と元のcopy/drop、既存のmove/score packing・転送・conditional/FIX/feedback/resamplingをtimerに含む。

許可CPUを自processの1coreへpin、BLAS1thread、通常priority、host/GPUは非占有。restore/warm/binding/hash/検証は時間外。deepcopyしたRTKのcallbackは元fixtureとAST同一bodyをコピー先へ再束縛し、毎callでtemplate digest不変をassertする。extract/time/satellite/integer-searchとgraph callbackはfixture境界で、GTSAM graph solve/実注入costは測っていない。

paired group短縮率のround内中央値→5round中央値を使い、p50同士の比とは区別する。CPU採用には同backend D236比10%以上、4/5round以上改善、CPU A/A中央値abs5%未満を要求する。GPU主screenはD236 GPUと候補CPUの双方へ同じ改善基準を要求する。丸め前で判定する。

| particles/initial_unique/rows/groups | feedback | CPU p50 ms D236→候補 | CPU同士paired短縮 | 改善round | GPU同士短縮 | GPU対候補CPU短縮 | CPU A/A短縮 | CPU主screen | GPU主screen |
|:---|:---|---:|---:|:---|---:|---:|---:|:---|:---|
| 64/6/64/4 | on | 18.445 → 18.127 | +7.728% | 3/5 | +1.794% | -0.270% | +13.173% | FAIL | FAIL |
| 64/32/64/4 | on | 30.314 → 26.083 | +21.147% | 5/5 | +3.423% | -21.901% | +22.542% | FAIL | FAIL |
| 64/64/64/4 | on | 88.184 → 55.050 | +22.482% | 3/5 | +15.598% | -10.858% | +7.943% | FAIL | FAIL |
| 128/32/64/4 | on | 40.563 → 39.609 | +0.431% | 3/5 | +12.145% | -4.371% | -7.538% | FAIL | FAIL |
| 64/6/128/8 | on | 24.263 → 23.952 | -0.230% | 2/5 | -1.361% | -24.476% | +0.854% | FAIL | FAIL |
| 128/128/128/8 | on | 423.701 → 387.785 | +6.505% | 5/5 | +3.502% | -22.464% | +0.918% | FAIL | FAIL |
| 64/6/64/4 | off | 21.220 → 15.305 | +14.021% | 4/5 | +7.682% | -28.658% | +12.266% | FAIL | FAIL |

CPU0/7、GPU同backend副screen0/7、GPU主screen0/7、A/A失敗5/7。CPUの+21.15%、+22.48%、feedback無効の+14.02%はA/A基準外で採用しない。+22.48%は改善roundも3/5で不足する。A/A合格の2条件は−0.23%と+6.51%で10%未達。観測された揺れの原因を断定せず、時間の短い箇所だけを根拠にした採用・過去の短縮率との合算も行わない。

## 正確性と実走

84tests PASS、既知設定warning1。450random selection比較、同点/1ULP差、巨大gauge・int64超の整数差、欠損support、no-CP/空行、non-target overflow、custom mapping/abs callback/型/buffer/index/policyのfallback、10,000call参照数/更新反映を確認した。3backend×3seed×4構成×4epochで全保存state/位置/P/RNG/graph-event/既存metadataがD236と一致した。525timed state hashと35warm snapshotも同backend D236に一致し、新しい数値許容差は使わない。

CUDA memcheckは18targeted tests PASS、設定warning1、error/leak0。今回の追加はCPU C++であり、新GPU codeの検証ではない。CPU参照数/RAII試験は別に行っており、CPU ASanや一般Python version保証とは扱わない。

実GTSAM/cssrlibのTokyo run3/seed20260710/400epoch、元のcluster_relin0/1×CPU/GPUの4prefixを実行した。各200、計800paired PF stepで、凍結D236の同backend/同構成に対し保存state JSONL全行と軌跡NPZ全配列dtype/shape/bytesが一致する。全cfg・source/input/external tc/binary hashを照合した。新native経路の正の呼出数/fallback0に加え、従来の全caller metadataも一致した。

| prefix | paired steps | valid steps | native shrink calls | fallback | scanned rows | peak new DD payload bytes |
|:---|---:|---:|---:|---:|---:|---:|
| r3_seed0_shrink_cpu_c0 | 200 | 197 | 2110 | 0 | 116160 | 432 |
| r3_seed0_shrink_gpu_c0 | 200 | 197 | 2110 | 0 | 116160 | 432 |
| r3_seed0_shrink_cpu_c1 | 200 | 197 | 2007 | 0 | 112092 | 432 |
| r3_seed0_shrink_gpu_c1 | 200 | 197 | 2007 | 0 | 112092 | 432 |

実走counterは速度内訳ではない。各prefixの単発wall timeは比較に使わない。これらは1route/seedのprefixで、全route/multi-seed測位精度やwhole-graph速度gateは未完了。有限scalarと全数値配列bitsは一致するが、Python scalar NaN payloadをcanonical化する既存codecの制約は残る。

## 失敗と保存

初回testは80PASS/2FAIL。型fallback試験で巨大なgaugeの一部だけを0へ置換し、意図しない巨大DDを作っていた。初回test source/plan/logを保存し、有限値の型試験を修正、元入力が同じOverflowErrorへfallbackする2試験も追加して84PASSとなった。実装/C++/binaryは変更・再buildしていない。その後のtest整形は実行済みsourceとのAST完全一致と双方SHAを保存した。

最初のbenchmark helperはtarget辞書に旧mode名が残り、最初の条件の5warm snapshot後、timed groupの開始前にKeyErrorで終了した。元helper/plan/5snapshot/失敗logを保存する。対象labelだけ修正したv2を別pathへ凍結し、実装/binary/workloadを維持して525timed比較を1回完了した。完了した測定の失敗条件を再測定していない。

最初の実走監査は「全valid stepでshrink呼出し正」と誤って期待した。元実装も最初のvalid stepはHが空で、held particle/新旧shrink呼出しは全て0になる。初回監査source/エラー記録を保持し、v2ではこの正しい0call条件とprefix全体の正のnative呼出数を別々に照合した。実装・実走を再実行していない。

## メモリと再現

新一時DD payloadはnb×8bytes、合成最大512bytes。native永続work buffer/CUDA allocation/cacheは増やさない。上記bytesはlogical payloadで、Python guard list、PyLong/result、buffer export、vector bookkeeping/allocator等を除きheap peakを実測していない。既存D226 score workspaceはdevice/host各1,143,352bytes/context、合成benchmarkの2contextは各合計2,286,704bytes。従来FIX/feedbackの一時DD行列も維持する。

[build/API/再現コマンド](../benchmarks/probes/rbpf_shrink/README.md)、[全rawと独立監査](../benchmarks/rbpf_shrink_20261016.json)、[完了照合](../benchmarks/rbpf_shrink_completion_check_20261016.json)。私有raw/archiveは`dist/rbpf_shrink_*20261016*`へ保持する。36production sources/5installed extensions/PF build PE/数値DLL/従来3拡張、34prior reports/16archives/1903filesを再照合する。

次は別の実測ボトルネックを検討する。今回のsource/workloadの再測定で採用基準を上書きせず、D236と製品・既定設定を維持する。
