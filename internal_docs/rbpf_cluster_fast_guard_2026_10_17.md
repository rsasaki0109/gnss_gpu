# D261: 一括状態検索のhot pathからfilesystem解決を除去 (2026-10-17)

cluster有効時の全scoreは元CPU版より8.850%、元GPU版より5.282%短縮し、両方5/5roundで改善した。一方、事前の10%採用基準には届かず、既定・製品は維持する。新たなCUDA数値処理の移行はない。入力・数値と保存PF状態の照合は通過した。

## 初版からの独立した変更

D260の一括親状態検索はguardのsource/code/path確認が重く、CPU/GPUとも遅く不採用だった。初版source・PE・全timingは固定して保存した。今回の別wrapperは、module初期化で元のsource SHAとcode familyを確認し、呼出しごとは既知の元filename表記とcode/bindingを調べる。guard内のPath.resolveを6回から0回へ減らし、未知の表記は元parameter blockへ戻す。許す表記の集合は旧resolve判定の部分集合である。

_eligibleのAST本文は元のまま、_literalだけclosureで渡す。他のglobal名は初版と同じlive dictを参照し、code/defaults/builtins/globals/native export/object/buffer/policyの確認を残す。仮説や結果、モデル値をcacheしていない。C++状態選択、任意精度整数差分、Python int true division、support・閾値・exact hit・順位・同点時None・state identity・logical counters・元のNumPy centre/prior式は変更なし。CPU extensionと同期修正版D253 CUDA DLLも同じPEで、新たなbuildはない。

## 完全scoreの対照計測

全394保存実入力（各cluster197）、original/off/on CPU/GPU、CPU/GPU A/A、凍結初版GPUの9モードで、5round×3group、53190timed/3546warm。各roundは197件integer-ns合計のpaired短縮率を3group中央値にし、採否は5round中央値。元版と同一extension無効版の両方に>=10%・>=4改善round、対応A/A絶対中央値<5%を要求した。GPU jointはさらにonCPUと両A/Aも要求する。初版GPUとの対比だけで採用しない。

正は短縮、負は遅化。全round値と対照も保存した。

| 対比 | c0 | c1 |
|---|---:|---:|
| cpu_prior | -2.638% | +8.850% |
| cpu_control | -1.216% | +9.078% |
| gpu_prior | -0.716% | +5.282% |
| gpu_control | -0.400% | +5.045% |
| gpu_vs_cpu | -107.728% | -83.526% |
| cpu_AA | -0.214% | -0.011% |
| gpu_AA | +0.320% | +0.173% |
| gpu_initial | +0.507% | +37.855% |

c1 CPUは元版比8.850%、無効版比9.078%、GPUは元版比5.282%、無効版比5.045%、すべて5/5改善。初版GPU比37.855%も同一計測内で確認した。ただしCPU/GPU component各0/2、GPU joint0/2で、10%基準未達を保持する。c0は元blockへ戻るが全scoreはCPU-2.638%、GPU-0.716%で、無効時の上積みはない。この入力では新GPU版も新CPU版より遅く、GPUへの一律移行を支持しない。

全score時計はdedup・全guard・C API・全整数処理・logical counter更新・parameter式・元数値処理・転送・kernel・完了待ち・返却を含む。復元/setup/warm/hash/count検証/serializationは外。BLAS1・許可CPU1個・通常優先度・非占有。D260の絶対時間との引算や改善率の加算はしない。完成した計測のやり直しなし。

全warmのscore/status/135診断/input bitsと全timed返却hash、全logical countはそれぞれの凍結backend参照と一致、scalar fallback0。全rawの列順・重複欠落・新path/counter・paired集計を独立auditで再照合。c1は197new C API batchesで4063仮説・4063canonical logical requests・7299parent logical requests、335exact・2674parent選択・1030ties。元counterのnative_callsはlogical数であり、融合後の実C API回数ではない。

## 精度と検証範囲

144tests PASS、既存pytest設定warning1件。D260の139test bodyはimport先以外のASTをそのまま移し、追加5testで同じlive globals・closure・Path.resolve/read_bytes禁止中のnative packing・未知filenameの元block fallbackを確認した。270random batchesと順序/16000bit gauge、support/exact/tie/隣接float閾値・state identity・refcount、54fixtureのCPU/GPU score/135診断/status/input bits、callback/binding/strict/例外fallback、12設定の4epoch全PF state/RNG/graph/prior metadataが一致。数値許容差を追加していない。

実走は元D236 GPUのcluster0/1、run3/seed20260710、各400 RTK epoch prefix。各200保存PF state、計400行と全trajectory配列のdtype/shape/bytes、全cfg、従来caller metadataが凍結同設定GPU参照と一致した。各197valid/3skipで、c1新batch197回、c0は元block197回を確認。score内部の135診断を毎回取得してreadbackも確認した。この実走時間は診断付きの単発走行なので速度比較には使わない。

初回fieldは数値処理・出力保存後、外側の回数確認で停止した。candidate constructorがinstanceへscoreをbindするため、subclassの同名counter methodが呼ばれないという検証wrapperの誤りだった。失敗run/log/sourceとc0の200保存行・軌跡の一致を別auditで保持し、成功した経路確認としては数えない。別checked driverは先にinstanceのcandidate scoreを捕まえ、その外側に回数・診断取得wrapperをbindした。数値wrapper/C++/PE/performance sourceは修正せず、checked fieldを別dirで実行した。

fullroute/multiseed測位精度、CPU側の新しいactual field、allocation/async障害注入、CUDA sanitizer/CPU ASanの新実行はない。元NumPy/native差とscalar NaN codecの限界を引き継ぐ。元の整数探索・GTSAM FGOはCPUのまま。ここでの結果はこの入力・試験範囲の非劣化確認で、全PF/FGOの速度やGPU化率ではない。

complete PF-stepの別screen sourceは準備したが、componentが事前10%基準を通らず実行していない。全PF throughputへの効果は未確立。次に検討するなら、選択済みの同じstateに対するcentre/priorの同一式をcall内で繰り返す箇所が候補になる。state identityと全12値のbits、callback/strict/error fallbackを保つ必要があり、cross-call cacheへ広げない。この追加案は未実装・未計測である。

## 資源と保存

timing4contextsの既存numeric stagingはhost/device各4573408B（各1143352B）、actualは1context。新しいCUDA bufferはない。C++の選択listは8*unique bytesのpointer payloadとheader/reference、key一時object等を持ち、packing後に入力参照を保持しない。追加の既知filename集合は2個のimmutable stringで、数値cacheではない。Python heap peakは未計測。36製品source・5installed PE・PF build PE・82過去reports・40archives6612filesを照合した。初版の拒否とfailed fieldを含む全証拠を保存する。

[結果](../benchmarks/rbpf_cluster_fast_guard_20261017.json)、[完了照合](../benchmarks/rbpf_cluster_fast_guard_completion_check_20261017.json)、[wrapper](../benchmarks/rbpf_cluster_fast_guard.py)、[driver](../benchmarks/bench_rbpf_cluster_fast_guard.py)、[checked field](../benchmarks/bench_rbpf_cluster_fast_guard_field_checked.py)。
