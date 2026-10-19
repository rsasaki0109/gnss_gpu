# RBPF候補登録の属性・ID再利用 (D265, 2026-10-17)

完全PFのCPU主screen 0/8、GPU同backend副screen 0/8、GPU joint screen 0/8。明示研究callerとして保存し、D236/製品/既定を維持する。過去D239/D241/D244の不採用結果やD264の限定されたCPU結果は変更せず、速度比を合算しない。D263の校正不足の区間比率から速度を予測しない。

## 変更と精度

D244と同じrequest単位のguardを保持し、同じmemo/context属性の取得とID用Python整数の生成/検索をcall内で再利用した。CPythonの辞書slotとversion、type/bases/コード/globals/builtins/lookupを毎request確認する。辞書/型の差し替えは次のhit前にdeoptし、元canonical/indexを実行する。identity辞書versionが変わると保持したH/keyを再確認する。countsへの自分の更新を追跡し、外からの異常なkey/valueの変更も検出する。

全source canonical miss callbackと全logical counter更新、任意精度整数canonical、Python prior加算/除算、first-seen key/H/pending、全particle/move/候補順を保持する。スコア、FP64・IRLS・weight/normalization/choice/RNG、位置/P、FIX/feedback/resample/graphは元の式のまま。確率/common-record cache、観測・候補の間引き、数値許容差、CUDA source/転送を追加しない。D241の元4method AST復元/guard/scope刷新はそのまま使う。

同一拡張内の別namespaceにD241/D244 C++本文を変更せず残し、off対照は旧D244を使う。新方式への明示置換12個を戻すと旧C++全体へ一致し、下記型辞書修正を戻すと初版へ一致することを独立auditした。既存caller metadataは全て比較し、新しい作業量counterは別stream/追加metadataで保存する。counterを速度やGPU化率へ換算しない。

63tests PASS、既知pytest設定warning1。候補順/prior/callback/全memo counter、custom hash/eqと部分例外、候補追加、辞書の差し替え/entry編集/counter subclass/異種key/class変更、0粒子/空H/ID再利用、GCと参照数を確認した。CPU/GPU/scalar×2seed×3構成×4epochで元/旧D244 off/新方式の全状態・RNG・graph/prior metadataがbits一致する。0/1/64粒子の回帰と1000回の型辞書参照数試験も通過した。

CPU AddressSanitizerは新しい41targeted tests PASS/22deselected、既知warning1、ASan error診断0。Python全体はinstrumentしておらず、LeakSanitizer/全Python heap/GPU sanitizerの保証ではない。一般version、fault injection、全heap peakは未測定。release/ASanのC++source bytesは同一で、速度にはreleaseだけを使用した。

## 初版失敗の保全

初版は最初の試験でWindows access violation。0粒子でも再現し、ASanは`prepare_cached`のnull辞書参照を示した。組み込み型の`tp_dict`を直接読む箇所が原因だった。別pathのchecked source/buildで[型辞書取得API](https://docs.python.org/3.12/c-api/type.html#c.PyType_GetDict)へ替え、[実行中versionのsource](https://github.com/python/cpython/blob/v3.12.10/Objects/typeobject.c#L223)で確認したnew-referenceをRAIIで解放し、null/typeを検査する。数学式/試験の数値判定は変更していない。

失敗source/PE/plan/log/4最小再現と初版ASanの負例を保全した。初回pytest processはfatal例外後に終了待ちになり、確認済みの当該child PIDだけを終了した。以降の隔離childはprocess-local error modeとwatchdogを設定し、crash時にGUI待ちをしない。初版は速度/fieldへ使わず、これらは全て修正版のみで実行した。完了済み速度測定の再試行はない。

## 完全PFの速度

8構成、元CPU/旧D244 off CPU/新CPU/元GPU/off GPU/新GPU/CPU A/A/GPU A/Aの8mode、4RNG初期状態seed20260710〜20260713、5round×3group。3840timed/256warm。観測/config/初期粒子は構成ごと固定で、4seedはRNG初期状態を変える。元の各粒子choiceを毎回実行する。

各groupの4完全PF integer-ns合計からpaired短縮率を計算し、3group中央値→5round中央値を使う。CPUは元CPUとoff CPU双方へ10%以上/4of5改善round/CPU A/A中央値abs5%未満。GPU副は元GPUとoff GPUの双方へ同条件、GPU jointは候補CPU比にも同条件と両A/Aを要求する。丸め前で判定し、全負値/不利/対照失敗/揺れを保存する。

| particles/unique/rows/groups,cluster,feedback | CPU対元 | CPU対off | GPU対元 | GPU対off | GPU対候補CPU | CPU A/A | GPU A/A | CPU/GPU副/joint |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 0: 64/6/64/4,c0,fb1 | +21.857% | +0.587% | +3.218% | -13.828% | -5.207% | +11.705% | +2.716% | False/False/False |
| 1: 64/32/64/4,c0,fb1 | -27.906% | -2.821% | +6.412% | +1.474% | +1.506% | +0.956% | +4.511% | False/False/False |
| 2: 64/64/64/4,c0,fb1 | +1.476% | -2.360% | -4.670% | +3.110% | -10.129% | -4.166% | -1.354% | False/False/False |
| 3: 128/32/64/4,c0,fb1 | +4.251% | +1.531% | +3.647% | -0.353% | -19.114% | -12.286% | +1.260% | False/False/False |
| 4: 64/6/128/8,c1,fb1 | +6.851% | -0.046% | +0.318% | +2.391% | -12.326% | +3.893% | -8.781% | False/False/False |
| 5: 128/128/128/8,c1,fb1 | +11.173% | +1.796% | -7.332% | +1.151% | +2.330% | -0.766% | -3.004% | False/False/False |
| 6: 64/6/64/4,c0,fb0 | +8.264% | +1.590% | -10.100% | +7.432% | +8.004% | -8.920% | -10.980% | False/False/False |
| 7: 64/6/64/4,c1,fb1 | +6.208% | +15.020% | +11.657% | +2.004% | -2.335% | +5.215% | +8.960% | False/False/False |

全modeで所有register-LU scorer context1個を共有。guard/native一時確保/元callback/prior算術/score転送・readback/choice/RNG/conditional/FIX/feedback/resampleを含む。restore/deepcopy/RNG生成/RTK callback再bind/hash/検証/保存は外。許可CPU1本/BLAS1/normal priority/非占有、CPU/GPUは直列。extract/search/time/satellite/graph solveはfixture境界で、実走/FGO全体の速度やGPU化率へ換算しない。独立auditで全4096 raw記録の順序/件数/hash/counterと256warm全状態/metadata、offと新方式の作業量一致を確認した。

## 実データ

Tokyo run3/seed20260710/400RTK epoch、CPU/GPU×元cluster0/1の4条件、各200保存PF状態（197valid/3skip）、計800を凍結D236同backendへ照合した。全state JSONL行、全trajectory配列dtype/shape/bytes、全cfg/従来metadataが一致する。診断引数は元どおり転送し、追加GPU診断転送0。全800件の新counter/登録counterを別streamで照合し、全valid stepでnative登録が動作、fallback/guard deopt/lookup invalidation0。

| backend/config | requested moves | 属性取得再利用 | ID version hit | ID再確認 | 最大alias entries |
|---|---:|---:|---:|---:|---:|
| native_cpu/c0 | 206522 | 206522 | 1289 | 174598 | 117 |
| gpu/c0 | 206522 | 206522 | 1289 | 174598 | 117 |
| native_cpu/c1 | 206341 | 206341 | 1224 | 174694 | 124 |
| gpu/c1 | 206341 | 206341 | 1224 | 174694 | 124 |

単発field wall timeは速度基準へ使わない。1route/seed prefixの4条件で、full-route/multiseed測位精度・whole GTSAM速度は未完了。既存codecのPython scalar NaN payload正規化制約を引き継ぎ、新しい数値許容差はない。

## メモリと保存

aliasのlogical map payloadはH/key/index/ID各pointerとuint64 version/map keyで48B/entry、さらにID用PyLong参照を保持する。empty indexは8Bで表のentriesにも1件含むため、48×entriesはlogical上界。C++map node/buckets/allocator/Python object/型辞書materialization/guard用定数領域は別で、heap peakや総差分は未実測。辞書/型basesへの追加所有参照もcall内だけに解放し、数値/H/problem値のcacheは次callへ持ち越さない。新しい永続native/CUDA workspaceは0、既存scorer host/device stagingは各1143352B/context。

CPython 3.12/Win64専用で、私有slot/辞書version APIへの依存を明示する。製品36source・5installed PE・PF build PE・90過去reports・44archives7654filesを保全した。

[結果](../benchmarks/rbpf_versioned_staging_20261017.json)、[完了照合](../benchmarks/rbpf_versioned_staging_completion_check_20261017.json)、[checked probe](../benchmarks/probes/rbpf_versioned_staging_checked/README.md)。

最終静的checkの初回は、slots回帰試験でimportしたpytest fixture `native`と同名の引数にRuff F811が2件出た。実行済みの試験sourceを固定したまま、この試験fileだけF811を除外して他のRuff rule/compile/EOL/diff/policyを再確認した。初回closing script・未完report/doc・診断を保全し、数値試験/完了済み速度は再実行していない。
