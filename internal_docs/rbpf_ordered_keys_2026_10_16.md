# 順序付き整数keyと型確認をネイティブ化する (D240, 2026-10-16)

**PF全体のCPU主screen0/7、GPU主screen0/7で、D236と既定設定を維持する。** キー作成と型確認をCPU-only C++へ移し、型確認だけの経路で不要なタプル構築を省いた。精度に関わる演算を変えず実装したが、今回の速度測定から有効な高速化とは判断しない。全小利得・負値・不安定controlを保持する。

## 変更範囲

D239の粗いmove診断で大きかったstaging領域から、既存`_ordered_integer_key`の型検査とタプル構築を対象にした。大きな領域の比率をそのまま改善可能率へ換算しない。比較基準はD236で、D237 shrink、D239 score番号alias、D230確率cache、profilerは併用しない。

`_MoveView._key`には元と同じ完全なkeyが必要なので、外側groupと内側satelliteの**挿入順、空group、絶対整数gauge、元のPython int object**を保つタプルを作る。exact dict/tuple/intだけを受け付け、元の型判定と同じ領域を事前確認する。任意精度整数をそのまま保持し、int64/float変換・加減算・ソート・gauge正規化は行わない。元のidentity/extend/shrink memoのscopeと保持objectを変更しない。

`_EpochMemo.canonical`では、元のsource canonicalを元の位置で実行した後、`_ordered_integer_key(hypothesis) is None`の判定だけを、同じ型領域を確認するnative boolean呼出しに置換する。捨てるだけのkeyタプルを作らない。empty/identity hit/source call/bypassの分岐・counter・canonical key保持を全て残す。

原helperのfunction codeと`type/dict/tuple/int/any/len`のglobal/builtin bindingをnative側で**毎要求**確認する。source canonical callback後にbuiltin bindingが変わった場合も原helperを使い、元のcallback/例外順を維持する。unsupported型は元と同じNone/Falseになり、subclassのitems/iter/int callbackを新しく実行しない。helper bindingの差し替えは各stepのscope更新で反映し、literal method/call target変更はguardで拒否する。

adapter3か所のcall target redirectとcanonical1条件だけを戻すと、5methodのASTが元sourceに一致する。private clone/classのscopeを毎step更新し、元module/runtime/classをpatchしない。raw move helperのstaging/index/weight/normalization/choice本体は無変換。全粒子・全候補・prior・FP64・2IRLS・閾値・乱数呼出しと順序、conditional/FIX/feedback/graph処理、strict NumPy原step fallbackを保つ。新しい数値許容差はない。

## 完全PF-stepの測定

従来CPU/候補CPU/候補GPU/従来GPU/CPU A/Aの5modeを直列回転し、7条件×5round×3group、525timed callと35warm snapshotを保存した。6feedback有効・事前hold条件と1無効条件で、今回のkey処理は全7条件で使用する。1許可CPU、BLAS1thread、通常priority、host/GPU非占有。guard/scope更新・native code/binding/type確認・タプル確保/解放を含む、PF全体のstaging・packing/転送・尤度・確率/choice・conditional・FIX・feedback・resamplingを時間に含めた。

restore/warm/hash/検証/保存は時間外。copied RTK callback bodyは元ASTと同一で、試行先RTKへtimer外で再束縛し、templateが不変なことを毎call検証する。extract/time/satellite/整数探索/graph callbackはfixtureで、GTSAM graph solveの速度ではない。

paired短縮率はgroup内比率のround中央値→5round中央値で、p50の比ではない。CPU主screenは同backend D236比10%以上、4/5round以上改善、CPU A/A中央値の絶対値5%未満。GPU主screenは同backend D236 GPUと候補CPUの双方へ同じ短縮率/round条件を要求する。

| particles/unique/rows/groups | feedback | CPU p50 ms D236→候補 | CPU paired短縮 | 改善round | GPU対D236 GPU paired短縮 | GPU対候補CPU paired短縮 | CPU A/A | CPU/GPU主screen |
|:---|:---:|---:|---:|---:|---:|---:|---:|:---:|
| 64/6/64/4 | 1 | 21.032→18.801 | +18.23% | 4/5 | -0.34% | -21.95% | -18.49% | FAIL/FAIL |
| 64/32/64/4 | 1 | 41.933→37.028 | +10.70% | 3/5 | +8.10% | +15.85% | +1.97% | FAIL/FAIL |
| 64/64/64/4 | 1 | 45.889→47.092 | -1.90% | 2/5 | +6.05% | -24.95% | +4.80% | FAIL/FAIL |
| 128/32/64/4 | 1 | 36.718→33.722 | +7.10% | 3/5 | -21.07% | -33.96% | +1.09% | FAIL/FAIL |
| 64/6/128/8 | 1 | 32.636→27.412 | +10.78% | 5/5 | -0.54% | -17.30% | -20.21% | FAIL/FAIL |
| 128/128/128/8 | 1 | 501.143→535.260 | +5.48% | 4/5 | -3.09% | -3.18% | -8.49% | FAIL/FAIL |
| 64/6/64/4 | 0 | 13.758→13.417 | +5.40% | 4/5 | +7.77% | -28.29% | -6.61% | FAIL/FAIL |

CPU0/7、GPU同backend副screen0/7、GPU主screen0/7、A/A失敗4/7。CPU+18.23%/4roundと+10.78%/5roundの条件は、A/Aが−18.49%/−20.21%で基準外。+10.70%の別条件は改善3/5round。個別roundの大きな揺れも全rawに保持し、短縮率やp50だけで採用へ読み替えない。同じsource/workloadの完了測定を再試行せず、既定とD236を維持する。

## 正確性・所有・実走

75tests PASS、既知のpytest設定warning1。9条件×1000要求で元helperとkey値/型領域/元integer object identityが一致し、`2**16000`の整数、挿入順の違い、空group、絶対gauge、入力変更後の新規keyを確認した。bool/NumPy integer/各種subclass等13unsupported型、6binding×通常/例外のcallback順、helper code変更fallback、canonical callbackによるbinding変更、10,000native instanceのinput/type参照数、GC cycle/再初期化も検査する。

3backend×3seed×4構成×4epochで全保存state/位置/共分散/RNG/graph eventと全従来caller metadataがD236に一致した。strict NumPy途中state/errorとhelper変更guardも確認する。全525timed state hashと35warm snapshot、既存metadataの一致を独立再計算した。新しい外部`ordered_keys`counterだけを加える。

CUDA memcheckは39targeted tests PASS、既知設定warning1、errors/leaks0。追加codeはCPU-onlyなので、このCUDA検査をCPU ASanや一般version保証とは扱わない。CPU参照所有は別途GC/refcount試験でも確認する。

実GTSAM/cssrlib Tokyo run3/seed20260710、400epoch入力、元cluster_relin0/1×CPU/GPU各200、計800paired PF stepを、凍結D236の同backend/同構成と照合した。全保存state JSONL行、軌跡NPZ全配列のdtype/shape/bytes、全cfg、全従来caller metadataが一致する。source/input/external tc/全binary hashも照合する。

| prefix | paired steps | native full-key calls | native support checks | fallback keys/support |
|:---|---:|---:|---:|---:|
| r3_seed0_ordered_cpu_c0 | 200 | 12504 | 18027 | 0/0 |
| r3_seed0_ordered_gpu_c0 | 200 | 12504 | 18027 | 0/0 |
| r3_seed0_ordered_cpu_c1 | 200 | 12508 | 17815 | 0/0 |
| r3_seed0_ordered_gpu_c1 | 200 | 12508 | 17815 | 0/0 |

各prefixは197valid/3skipで、新経路は両構成/両backendで実際に使用する。counterは旧Python helperの代わりに処理した要求数で、速度比ではない。全route/multi-seed測位精度・whole-graph速度gateは未完了。有限scalarと全数値配列bitsを照合し、Python scalar NaN payloadをcanonical化する既存codecの制約は残る。

## 保存と再現

GILを保持してexact builtinの辞書を順に読む。native engineはsource/code/builtin refsとcounterを持ち、GC traverse/clear/deallocで所有を管理する。key自体は元と同じ既存move memoで保持し、新しい永続hypothesis/DD/key cacheやnative/CUDA workspaceを追加しない。type-only経路では完全keyの一時構築を省く。guardのbinding参照等も含めた一時確保が全てゼロという意味ではない。native object/GC/type、Python tuple/allocator等のheap peakは実測しない。元score stagingはdevice/host各1,143,352bytes/context、比較2context、各実走1contextで不変。

初回buildはCPython3.12に存在しないbuiltin参照macroでcompile failureになった。初版source/config・build plan/log/recordを残し、3.12のfunction構造体によるbuiltin参照へ修正し、別のv2 buildへ保存した。テスト・速度・実走で使ったv2 source/binaryは不変。失敗buildやA/A不合格を削除しない。

40prior reports、19archives/2345files、36production sources、5installed extensions、PF build PE、既存数値DLLと3拡張を再照合した。製品source・凍結model・既定は今回変更しない。公開source/docsはUTF8 CRLF、tests/JSONはLF、Ruff/compile/diff/公開名scan/artifact policyを検証する。

[probe/API](../benchmarks/probes/rbpf_ordered_keys/README.md)、[全測定・独立監査](../benchmarks/rbpf_ordered_keys_20261016.json)、[完了照合](../benchmarks/rbpf_ordered_keys_completion_check_20261016.json)。今回の候補は未採用として保存し、次は異なる処理変更の効果を調べる。
