# 不変keyのスコア番号を呼出し内で再利用する (D239, 2026-10-16)

**CPU主screen0/7、GPU主screen0/7のため既定採用を見送る。D236を維持する。** move区間を細分化した後、既に計算済みのimmutable key objectからスコア番号への対応をC++で再利用した。canonical要求・全候補・各粒子の確率計算・乱数抽選を残し、実走800stepは同backendのD236に全保存state/軌跡bitsが一致する。対GPU旧版の副screenは1/7だが、候補CPUを含む主screenを置き換えない。

## move区間の細分化

D238で大きかったmove残部を、元のD227/D228/D229/D233 helperの式を保持して3区間へ分け、scorerもpacking/native call/返却の3区間へ分けた。計測境界とadapter targetの変更だけを戻すと、5helper全てのASTが元sourceに一致することをassertする。function code/source SHAとcall targetを固定し、private cloneのglobalsを毎call更新する。実runtime/module/classをpatchせず、RNG proxyや粒子ごとの細かいtraceは使わない。

計測器41tests PASS。3backend×3seed×4構成×4epochで全保存state/位置/P/RNG/graph-event/既存metadata、strict-policy途中state/error、empty-batch early return、capacity scalar fallback、helper変更拒否を確認した。実Tokyo run3/seed20260710/default cluster0のCPU/GPU各200、計400stepも凍結D236の全保存state/全軌跡配列bits・全cfg・既存metadataに一致した。

全197valid step（各3skip）を使用し、遅い/不利な区間の事後除外はしない。下表はexclusive wall spanの中央値と、各stepのroot時間に対する比の中央値で、両中央値の比ではない。tracing/schedulingを含む粗い診断で、CPU cycle帰属・CPU/GPU速度比・確定的な短縮可能率ではない。

| phase exclusive | CPU median ms | CPU median fraction | GPU median ms | GPU median fraction |
|:---|---:|---:|---:|---:|
| `_move_staging` | 9.708 | 39.08% | 12.023 | 33.39% |
| `_score_pack` | 0.836 | 3.12% | 1.073 | 2.83% |
| `_score_native` | 0.599 | 2.33% | 4.843 | 13.05% |
| `_score_finish` | 0.019 | 0.07% | 0.040 | 0.11% |
| `_move_scores` | 0.150 | 0.59% | 0.282 | 0.73% |
| `_move_choice` | 2.996 | 11.78% | 3.779 | 10.85% |
| `_moves` | 0.420 | 1.51% | 0.537 | 1.48% |

`_move_staging`はextend/shrink等の計測済み子を除いた候補登録/lookup等の残部。`_move_scores`はpacking/native/finish等の子を除いたscore整備の残部。`_moves`は3区間の外側のadapter/context処理等。GPUの`_score_native`はkernelだけでなく転送・完了待ち等を含むDLL呼出し全体であり、CPUと単発時間を比較してGPU性能の根拠にはしない。

細分化計測器の135timed/9warm較正も保存した。低観測overhead補助flag1/3、A/A失敗1/3。paired slowdownは−9.08/+2.96/+0.70%、A/Aの高負荷条件は+11.01%で基準外。−9.08%を計測器による高速化とは扱わず、広い個別roundを保持し、再測定しない。extract/searchがfixtureの較正なので実走の2wrapper分は対象外。計測比率は候補領域の選定だけに用い、以下の採否は**計測器を外した完全PF-step**で判定する。

## 番号再利用の実装

従来のlocal `index(hypothesis)`は、毎moveでcanonical keyを要求した後、同じ深いtupleを`lookup`辞書で繰り返しhashする。今回はそのcanonical要求を元の位置で毎回実行し、同じimmutable key objectの番号だけをcall内C++ pointer mapで覚える。新しい数値演算はない。

miss時は元のindex tailをそのまま実行し、最初に現れたkey/Hの順、`keys/hypotheses/pending/cache`処理を保持する。元の`_EpochMemo`が既に保持するkey objectとbuiltin empty keyに限りaliasを登録する。異なる同値key objectの初回は元の値lookupを行う。key objectを強参照するためid再利用は起きず、immutableなbuiltin tuple/intだけを適格とする。

custom hash/equality keyは元のcallback/例外を実行した後、aliasをblock終了まで無効化する。型・callback・未知canonical域では元lookupを実行する。source callbackが例外ならaliasを解放し、その例外を伝える。元のcanonical callbackを省略しないので、既存canonical/move/reuse/native countersを含む**全ての従来caller metadata**も同じになる。新しい`score_index_memo`だけを外部metadataへ加える。

元のweight/normalization/`rng.choice`式は一切置換しない。D230の共通候補record/proposal分布cacheは併用せず、提案確率・score・priorを再利用しない。粒子/候補数、順序、FP64/2IRLS、閾値、FIX/feedback/graph処理も不変。元のstrict NumPy whole-step fallbackを保つ。cacheはmove blockで解放され、commit/feedback/次epochへ残らない。

adapterはD239でAST照合済みの3つのtarget redirectを使うが、profilerを継承せずspan/timer codeを呼ばないことを計測前に検証した。全guard/scope更新・cache構築・key検証・map確保/解放を速度timerに含める。D226数値DLL、D233/D234/D235拡張はbyte単位で不変。追加はCPython3.12/64-bit WindowsのCPU-only index拡張で、新CUDA code/製品source/既定変更はない。

## 完全PF-stepの速度

D236 CPU/候補CPU/候補GPU/D236 GPU/CPU A/Aの5modeを直列回転し、7条件×5round×3group=525timed calls、35warm snapshotを保存した。6feedback有効/事前hold条件と1feedback無効条件で、番号再利用は全7条件で使用する。自processの1許可CPU/BLAS1thread/通常priority、host/GPU非占有。restore/warm/callback再束縛/hash/検証は時間外。fixture callback bodyは元ASTと同一で、毎callのRTK template不変を検証する。

PF全体のstaging/score packing・転送・確率/choice/conditional/FIX/feedback/resamplingを含む。extract/time/satellite/整数探索/graph callbackは入力fixtureで、GTSAM graph solveの速度ではない。paired group短縮率のround内中央値→5round中央値を使い、p50比とは区別する。CPUは同backend D236比10%以上、4/5round以上改善、CPU A/A中央値abs5%未満を要求する。GPU主screenは旧GPUと候補CPUの双方へ同じ基準を要求する。

| particles/initial_unique/rows/groups | feedback | CPU p50 ms D236→候補 | CPU paired短縮 | 改善round | GPU同士短縮 | GPU対候補CPU短縮 | CPU A/A短縮 | CPU主screen | GPU主screen |
|:---|:---|---:|---:|:---|---:|---:|---:|:---|:---|
| 64/6/64/4 | on | 16.426 → 17.414 | +1.641% | 3/5 | +12.917% | +2.990% | -1.120% | FAIL | FAIL |
| 64/32/64/4 | on | 28.842 → 28.870 | -2.389% | 1/5 | -4.427% | -18.948% | +1.382% | FAIL | FAIL |
| 64/64/64/4 | on | 37.206 → 37.223 | -3.512% | 1/5 | +1.408% | -9.917% | -0.081% | FAIL | FAIL |
| 128/32/64/4 | on | 34.832 → 32.311 | +8.061% | 4/5 | +10.767% | -12.728% | +15.191% | FAIL | FAIL |
| 64/6/128/8 | on | 24.620 → 23.205 | +4.058% | 3/5 | +3.828% | -17.045% | -3.933% | FAIL | FAIL |
| 128/128/128/8 | on | 390.794 → 394.522 | +0.648% | 3/5 | -0.024% | -4.714% | -0.062% | FAIL | FAIL |
| 64/6/64/4 | off | 13.552 → 12.624 | +8.096% | 5/5 | -2.831% | -21.565% | +8.395% | FAIL | FAIL |

CPU0/7、GPU同backend副screen1/7、GPU主screen0/7、A/A失敗2/7。CPUの+8.06/+8.10%は10%未達かつA/A基準外。他の小利得・負値・不安定roundも全保持する。主screen不合格を副screenやcall数削減で採用へ読み替えず、完了測定を繰り返さない。D236と製品・既定を維持する。

## 正確性と実走

候補54tests PASS、設定warning1。9条件×1000requestのbare原処理/Python参照/native比較、最初のkey/H順、異なる同値H・gauge変更・任意精度整数、custom hash/equality/例外順、unknown/unretained域、10,000instanceのinput/type参照数、GC cycleと再初期化を検証した。3backend×3seed×4構成×4epochで全保存state/位置/P/RNG/graph-event/全既存metadataがD236と一致する。strict-policy途中state/errorとhelper変更拒否も確認した。fine profilerの41testsとは別suiteで、候補/C++/binaryは初回から不変。

全525timed state hashと35warm snapshot、既存metadataも同backend D236に一致する。新しい数値許容差は追加しない。CUDA memcheckは18targeted tests PASS、設定warning1、error/leak0。新codeはCPUのみであり、CUDA検査をCPU ASan/一般version保証とは扱わない。CPU object所有はGC/refcount試験でも別に確認する。

実GTSAM/cssrlib Tokyo run3/seed20260710/400epoch、元cluster_relin0/1×CPU/GPU各200、計800paired stepを実行した。凍結D236の同backend/同構成に対して保存state JSONL全行、軌跡NPZ全配列dtype/shape/bytes、全cfg、全既存caller metadataが一致し、source/input/external tc/全binary hashを照合した。

| prefix | paired steps | requested indices | original index-tail calls | identity hits | bypassed | peak logical pointer bytes |
|:---|---:|---:|---:|---:|---:|---:|
| r3_seed0_index_cpu_c0 | 200 | 206522 | 18224 | 188298 | 0 | 1872 |
| r3_seed0_index_gpu_c0 | 200 | 206522 | 18224 | 188298 | 0 | 1872 |
| r3_seed0_index_cpu_c1 | 200 | 206341 | 18012 | 188329 | 0 | 1984 |
| r3_seed0_index_gpu_c1 | 200 | 206341 | 18012 | 188329 | 0 | 1984 |

ここで減るのは元のindex tail（値hash/lookup）実行で、canonical要求数や論理move要求数は減らさない。counter削減は実行時間の比ではない。診断用400stepと候補用800stepは同じroute/seed prefixの別実行で、1200個の独立epochやmulti-seed精度coverageへ数え直さない。有限scalar/全数値配列bitsを照合し、Python scalar NaN payloadをcanonical化する既存codecの制約は残る。全route/multi-seed測位精度・whole-graph速度gateは未完了。

## メモリ・保存・再現

native alias mapはcall中だけkey/indexの強参照を持ち、元memoが保持する入力をdeep copyしない。表のbytesは2個の64-bit pointer×entry数のlogical payloadで、unordered_map node/bucket、State、Python object/allocator/closure/stack等を除く。heap peakは実測しない。GIL保持、GC traverse/clear/deallocとPython heap type所有を実装する。永続native/CUDA workspaceやcross-epoch H/DD/key cacheは増やさない。既存score workspaceはdevice/host各1,143,352bytes/context、比較benchmarkは2context、各実走は1context。

全raw・negative・失敗control・source/binary/独立監査を`dist/rbpf_score_index_memo_*20261016*`と`dist/rbpf_move_phase_profile_*20261016*`へ保持する。38prior reports/18archives/2158files、36production sources/5installed extensions/PF build PE/従来数値DLLと3拡張も再照合する。

[build/API](../benchmarks/probes/rbpf_score_index/README.md)、[全結果/較正/独立監査](../benchmarks/rbpf_score_index_memo_20261016.json)、[完了照合](../benchmarks/rbpf_score_index_memo_completion_check_20261016.json)。候補準備全体の大きさを、この一部分の利得へ外挿しない。今回の番号再利用は未採用とし、別の処理変更を検討する。
