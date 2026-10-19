# 全候補登録をネイティブループへまとめる (D241, 2026-10-16)

**完全PF-stepのCPU/GPU主screenは0/7。D236/製品/既定を維持する。** 小さな関数呼出しをまとめるため、候補準備全体をC++ Python C APIへ移したが、有効な短縮は実証できなかった。候補登録の大きなspanを、そのまま移植効果へ読み替えない。

## コピーだけの変更は見送る

最初に候補/commit/resamplingでの`_copy_H`を調べた。元の処理は既に組込み`dict(zmap)`であり、3fixture条件×5回、15profiled whole-stepと3従来参照で呼出し内wall spanを測った。D236のscalar数値backendを使い、全保存state/RNG/従来metadataは一致する。

| particles/unique/rows | copy wall fraction median | all5samples range |
|:---|---:|---:|
| 64/6/64 | 0.913% | 0.909–0.931% |
| 64/6/128 | 1.040% | 0.988–1.108% |
| 128/128/128 | 0.281% | 0.263–0.331% |

clock/計測wrapper/schedulingを含む診断で、CPU cycleの帰属や確定的な削減上限ではない。A/A/速度採用試験ではなく、GPU性能へ外挿しない。約0.3〜1.1%という規模から、コピー単独のC++化は実装せず、全rawを保存して候補登録全体へ対象を変えた。

## 元の登録順・確率演算を保持する

baselineはD236。D239 score番号alias、D240 key/type書換え、D230 common-record/確率cache、profilerは併用しない。native loopは各粒子を元順に読み、heldならsurvive/extend/shrink/release、emptyならstay、その後consensus/adopt/spawnを元順で登録する。元のextend/shrinkを同じ位置で呼び、releaseは毎粒子で新しい空dictを作る。各粒子の全recordも別々に作る。

**全moveで元のcanonical callbackを呼び、元の値hash/lookupも全て実行する。** `lookup`contains→新規index登録→key/H append→`cache.get(key) is None`判定→pending append→最終lookup、という順を保つ。同値の別key object、hash衝突、custom hash/equality callback/例外にも元の順で応答する。既存epoch memo/純粋move再利用は元のままで、新たなalias/分布/key cacheは作らない。

stay priorの左結合加算とadopt/spawnの毎move除算には、元と同じ順序のPython number操作を使う。C++のdouble式へ置換せず、共通priorやrecordを共有しない。score packing/尤度/weight/normalization/`rng.choice`を含む元の後半はliteralのまま残す。FP64/2IRLS、全粒子・全候補・prior・閾値・整数値/順序・乱数draw、conditional/FIX/feedback/graph処理を保ち、新しい数値許容差を加えない。

原staging blockを別関数へそのまま取り出してfallbackに使う。native guard対象外のcaller/list/cache/prior型や`len`bindingでは、元blockを一度実行する。元のstrict NumPy whole-step fallbackも残す。source/function code/call targetを照合し、private helper scopeを毎step更新する。3adapter redirectと1staging置換だけを戻すと4helperの元ASTへ一致することをassertする。元module/runtime/classへのpatchはない。

## 完全PF-stepの速度

従来CPU/候補CPU/候補GPU/従来GPU/CPU A/Aを直列回転し、7条件×5round×3group、525timed callと35warm snapshotを保存した。1許可CPU/BLAS1thread/通常priority、host/GPU非占有。新guard/preflight/scope更新、native list/tuple/参照確保・解放、canonical/hash/lookup/callback、Python prior算術、元PF全処理を時間に含める。restore/warm/callback再束縛/hash/検証/保存は時間外。copied RTK callback bodyのAST一致と、各callのtemplate不変を確認した。

extract/time/satellite/整数探索/graph callbackはfixtureで、GTSAM graph solveの速度ではない。短縮率はgroup内paired比率のround中央値→5round中央値。p50比とは区別する。CPUは同backend D236比10%以上、4/5round以上改善、CPU A/A中央値abs5%未満。GPU主screenはD236 GPUと候補CPUの双方へ同じ短縮率/round条件を要求する。

| particles/unique/rows/groups | feedback | CPU p50 ms D236→候補 | CPU paired短縮 | 改善round | GPU対D236 GPU | GPU対候補CPU | CPU A/A | CPU/GPU主screen |
|:---|:---:|---:|---:|---:|---:|---:|---:|:---:|
| 64/6/64/4 | 1 | 12.271→12.143 | -4.70% | 1/5 | +3.07% | +0.32% | +2.41% | FAIL/FAIL |
| 64/32/64/4 | 1 | 22.438→22.750 | -1.78% | 2/5 | -0.70% | -14.73% | +2.39% | FAIL/FAIL |
| 64/64/64/4 | 1 | 37.078→36.728 | +2.96% | 3/5 | +6.10% | -12.95% | -5.74% | FAIL/FAIL |
| 128/32/64/4 | 1 | 34.431→32.625 | +0.04% | 3/5 | -9.42% | -15.25% | -2.52% | FAIL/FAIL |
| 64/6/128/8 | 1 | 23.621→22.503 | +7.79% | 3/5 | +3.64% | -31.43% | -18.63% | FAIL/FAIL |
| 128/128/128/8 | 1 | 473.663→482.966 | +0.49% | 4/5 | -2.29% | -10.39% | +5.17% | FAIL/FAIL |
| 64/6/64/4 | 0 | 14.176→14.639 | +4.38% | 4/5 | +1.65% | -20.00% | -3.01% | FAIL/FAIL |

CPU主0/7、GPU同backend副0/7、GPU主0/7、A/A失敗3/7。CPU最大+7.79%も10%未満、改善3/5かつA/A−18.63%で不合格。default相当条件は−4.70%、1/5roundで、少数の好都合なp50やroundへ置き換えない。全小利得/負値/失敗controlを保持し、同source/workloadの完了測定を繰り返さない。この粒度のPython制御移植の優先度を下げ、既定を維持する。

## 正確性・所有・実走

最終60tests PASS、既知pytest設定warning1。9条件の全候補値/順/キー/first-seen H/pending/全callback、custom hash/equality衝突とcanonical/hash/eq/extend/shrink途中例外、Python加算の結合順・負のゼロ/非正規化数bits、`2**16000`の整数、callbackによるlist append、empty粒子/list、release dict/recordの非共有、5,000callのinput/function参照数を確認した。

3backend×3seed×4構成×4epochで全保存state/位置/共分散/RNG/graph eventと全従来caller metadataがD236に一致する。特殊bindingの原staging一回fallback、strict NumPy途中state/error、helper変更拒否も確認した。全525timed state hash、35warm snapshotと既存metadataの一致を独立再計算した。追加は外部`native_staging`counterだけ。

初回suiteは59PASS/1FAIL。custom `len`試験が従来側だけsetup/constructorの100callbackを含めていた。別診断で処理中は双方370callback、全state bitsとfallback一回が一致することを確認した。初版test/source/plan/logを保存し、**試験の記録開始位置だけ**を修正して60testsを通した。C++/controller/binaryは初回から不変で、速度測定前に修正を完了した。

CUDA memcheckは24targeted tests PASS、既知warning1、errors/leaks0。追加codeはCPU-onlyで、CPU所有は別の参照数試験でも検証する。CUDA検査をCPU ASan/一般version保証とは扱わない。

実GTSAM/cssrlib Tokyo run3/seed20260710、400epoch入力、元cluster_relin0/1×CPU/GPU各200、計800paired PF stepを凍結D236同backend/同構成と照合した。全保存state JSONL行、軌跡NPZ全配列dtype/shape/bytes、全cfg、全従来caller metadataが一致する。source/input/external tc/全binary hashも照合する。

| prefix | paired steps | native staging blocks | original move requests handled | fallback blocks |
|:---|---:|---:|---:|---:|
| r3_seed0_staging_cpu_c0 | 200 | 197 | 206522 | 0 |
| r3_seed0_staging_gpu_c0 | 200 | 197 | 206522 | 0 |
| r3_seed0_staging_cpu_c1 | 200 | 197 | 206341 | 0 |
| r3_seed0_staging_gpu_c1 | 200 | 197 | 206341 | 0 |

各prefix197valid/3skipで、全valid blockと全論理moveが新loopを通り、fallback0。canonical/moveの原要求数は変わらない。counterは速度比ではない。全route/multi-seed精度・whole-graph速度gateは未完了。有限scalar/全数値配列bitsを照合し、Python scalar NaN payloadをcanonical化する既存codec制約は残る。

## 保存と再現

native関数はGILを保持し、引数と元の辞書/list/memoを使う。取得したPython参照をRAIIで解放し、元と同じ一時record/key/H/pendingを保持する。永続native/CUDA workspaceやcross-epoch cacheは追加しない。Python/native wrapper/tuple/list/allocator等のheap peakは実測しない。元score stagingはdevice/host各1,143,352bytes/context、比較2context、各実走1contextで不変。

42prior reports、20archives/2499files、36production sources、5installed extensions、PF build PE、元数値DLLと3拡張を再照合した。source/docsはUTF8 CRLF、tests/JSONはLF。Ruff/compile/diff/公開名scan/artifact policyを確認する。失敗した初版試験、コピー診断、全速度control、実走state/binary/sourceを保存する。

[probe/API](../benchmarks/probes/rbpf_native_staging/README.md)、[全測定・独立監査](../benchmarks/rbpf_native_staging_20261016.json)、[完了照合](../benchmarks/rbpf_native_staging_completion_check_20261016.json)。候補は未採用として保存し、D236/製品/既定を維持する。
