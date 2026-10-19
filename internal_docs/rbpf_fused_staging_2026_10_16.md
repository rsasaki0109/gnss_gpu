# 候補登録内で同一メモと検索をまとめる (D244, 2026-10-16)

**PF全体では採用基準に届かず、D236/製品/既定を維持する。** 同じ仮説の既存メモhitと、完了済みのbuiltin-keyスコア番号検索をC++の候補登録内でまとめた。735timed完全PF step/49warmでCPU主screen0/7、GPU主screen0/7、A/A失敗3/7。全不利な条件を保持し、測定を繰り返さない。

## 対象と実装

D243ではIRLS省略が実走スコア単体でも旧版比−0.45〜+0.82%だった。次にD239の粗い実走診断で大きかった候補登録を調べた。D239はcanonicalのPython呼出しを残したまま一部検索を省略、D241は全登録をC++化して全canonical/hashを残し、どちらも採用基準未達だった。今回は新しい一体化した候補として、同じC++登録call内で両者の境界をまたぐ重複処理を省く。過去の不採用候補を実証済みの改善として足し合わせない。

既存_EpochMemoが既に保持する同じH/keyと、builtin tuple/intだけの完了済みlookupに限定する。hit時は元と同じlogical canonical_requests/identity_hits/empty_hitsを更新する。miss時のsource canonical callback、全初出key/H/pending/cache判定、candidate/prior順は元の処理を行う。全粒子・全候補・元Python FP演算・重み正規化・各rng.choice・conditional/FIX/feedback/graphは維持し、確率cacheや精度/粒子数/閾値変更を加えない。

同じ拡張にoffの元登録関数とonのfusion関数を置き、両者を比較する。onはmethod/function code、globals/builtins/class/lookupのdict version、retained H/key、counts/identitiesの通常型とbindingを確認する。callbackによるlookup変更、特殊keyやbindingはaliasを無効化して元処理へ戻す。整数gaugeはPyLongのままで任意精度を保ち、同値だが異なるHは初回の元処理で登録する。mapはcall中だけH/key/indexを強参照し、例外後もRAIIで解放する。

公開D241 C++からの生成変更を全て戻すとsource全文が一致することを独立再計算した。最初の未build source、初回buildのsource/PE、test前のidentity guard整理を保存した。測定は最終v3の1binaryだけで行い、最初のtestから実装/Python controller/PEは変えない。数値DLLはD226、3CPU拡張はD233/D234/D235のまま。今回GPU solver/kernelは変更しない。

## 正確性

101tests PASS、既知pytest設定warning1。3backend×3seed×4構成×4epoch×on/offで全保存state/位置/P/RNG/graph events/従来metadataがD236に一致する。component比較ではcandidate値/順序/prior bits/元callbackとcounter、3seed×3shape、巨大整数、signed zero/subnormal prior、custom hash/equality/途中例外、candidate listのcallback追加、lookup/identities/counts/method/class変更時の無効化、refcount解放、strict-policy途中stateとhelper変更拒否を確認した。

初回95PASS/6FAILはcustom-key試験の観測側の不備だった。D241のbare登録fixtureは全keyに.valueがある前提だったが、元の_EpochMemoは空Hにbuiltin ()を返すため、比較元自身がlabel取り出しで停止した。初版source/plan/logを保全し、fixtureのlabel取り出しだけ空tupleも受け入れるよう修正した。観測変更を戻すASTは元fixtureと一致する。実装/PEと数値判定は不変、許容差は追加しない。

今回のCUDA memcheckは24targeted GPU tests PASS、既知設定warning1、0error/0leak。新codeはCPUのみで、これをCPU ASanや他Python versionの保証にしない。以前のrace/sync結果を今回の新規PASSに加えない。CPython3.12/64bit Windows専用で、dict version fieldの3.12 deprecation warningをbuild logに残す。安定ABI・将来versionの実装とは扱わない。

## 完全PF stepの比較

元CPU/候補CPU/候補GPU/元GPU/同一拡張off CPU/off GPU/元CPU A/Aの7modeを直列回転。7条件×5round×3group×7mode=735timed、49warm snapshot。型/binding/version確認、scope更新、native alias確保/解放、全候補構築、likelihood pack/転送/同期、weight/RNG、conditional/FIX/feedback/resamplingを計測する。restore/setup/warm/callback再束縛/状態hash/検証/serializationは時間外。callback bodyは元fixtureとAST一致し、RTK template不変も毎callで検証する。1許可CPU/BLAS1thread/通常priority、host/GPU非占有。extract/search/time/satellite/graphはfixtureで、whole FGO solverの速度ではない。

paired短縮率のgroup→round中央値→5round中央値で、p50比とは別の指標。10%以上、4/5以上改善、CPU A/A abs5%未満を要求し、CPUは元CPUと同一拡張off CPUの両方、GPUは元GPU・同一拡張off GPU・候補CPUの全てを満たす必要がある。

| particles/initial_unique/rows/groups | feedback | CPU p50 ms 元→候補 | CPU対元 | CPU対off | 改善round | GPU対元 | GPU対off | GPU対候補CPU | CPU A/A |
|:---|:---:|---:|---:|---:|:---:|---:|---:|---:|---:|
| 64/6/64/4 | on | 13.511→14.732 | -1.09% | -2.80% | 2/5 | -3.83% | +23.57% | -4.22% | -1.25% |
| 64/32/64/4 | on | 22.749→26.176 | -4.44% | +3.70% | 1/5 | -1.27% | +0.03% | -14.04% | -4.09% |
| 64/64/64/4 | on | 44.487→38.376 | +3.02% | +5.30% | 3/5 | -7.04% | -0.22% | -20.66% | -5.11% |
| 128/32/64/4 | on | 42.668→50.020 | +3.74% | -12.67% | 4/5 | +4.88% | -3.09% | -5.75% | -0.43% |
| 64/6/128/8 | on | 21.220→21.185 | -0.85% | +6.14% | 2/5 | +1.73% | -17.30% | -16.52% | -6.81% |
| 128/128/128/8 | on | 371.187→381.610 | +2.64% | -2.60% | 3/5 | -0.72% | +2.50% | -3.46% | +1.37% |
| 64/6/64/4 | off | 11.187→11.606 | -3.27% | -1.47% | 2/5 | +0.44% | -0.45% | -24.00% | -5.30% |

CPU/GPU同backend副/主とも0/7。CPU最大+3.74%は基準未達、default-likeは−1.09%。A/A失敗3/7と個別roundの揺れを残す。有利なroundやcounter削減から採用へ読み替えず、遅延の原因を特定の管理処理だけに断定しない。全735timed state hash、49warm state、全従来metadataは同じbackendのD236と一致し、毎timed callのfusion counterも保存・照合した。

## 実走

Tokyo run3/seed20260710/400epoch、元cluster_relin0/1×CPU/GPU、各200paired PF stepで全800を比較した。保存state JSONL全行、軌跡NPZ全配列のdtype/shape/bytes、全cfg、全従来caller metadataがD236に一致した。全prefixの197valid blockで新しいfusionが動作する。source/input/external tc/全PE/DLLのhashも照合した。下表のcounterは作業量であり、速度比ではない。

| prefix | paired steps | move要求 | fused memo/index hit | 元index実行 | 無効化 | peak alias logical上界bytes |
|:---|---:|---:|---:|---:|---:|---:|
| r3_seed0_fused_cpu_c0 | 200 | 206522 | 188298 | 18224 | 0 | 3744 |
| r3_seed0_fused_gpu_c0 | 200 | 206522 | 188298 | 18224 | 0 | 3744 |
| r3_seed0_fused_cpu_c1 | 200 | 206341 | 188329 | 18012 | 0 | 3968 |
| r3_seed0_fused_gpu_c1 | 200 | 206341 | 188329 | 18012 | 0 | 3968 |

実走は精度のみの単発prefixであり、そのcall_secondsや過去runとのwall差を速度基準へ使わない。1route/seed・prefixの範囲で、全route/multiseedの測位精度とwhole graph速度gateは未完了。既存codecのscalar NaN payload正規化を引き継ぎ、新たな数値許容差は加えない。

## メモリと保存

aliasはmove call内だけで解放し、次epochへH/キー/解を持ち越さない。64bitでH/key/indexとmap keyのlogical payloadは32bytes/entry、empty indexは8bytes。表はentry数×32の上界で、map node/bucket/allocator/Python closure/import guard code等は含まず、heap peakを測定した値ではない。新しい永続native/CUDA bufferはなく、元scorerのhost/device staging各1143352bytes/contextを保つ。benchmarkは元1+候補1scorer、実走は1scorer/process。

36製品source、5installed extension、PF build PE、D226 DLL、3CPU拡張、48既報、23archive/3018filesを保全する。この方式の速度採用を見送り、D236/製品/既定は維持する。次は個々のPython loopの置換を続ける前に、実走のsampled call stackやallocationを含む、より広い処理境界で律速を確認する。

[全結果](../benchmarks/rbpf_fused_staging_20261016.json)、[完了照合](../benchmarks/rbpf_fused_staging_completion_check_20261016.json)、[API/probe](../benchmarks/probes/rbpf_fused_staging/README.md)、[benchmark](../benchmarks/bench_rbpf_fused_staging.py)。
