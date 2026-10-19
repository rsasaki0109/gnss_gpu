# RBPF extendのsupport抽出と整数辞書組み立て (D264, 2026-10-17)

完全PFのCPU主screen 1/8、GPU同backend副screen 0/8、GPU joint screen 0/8。明示研究callerとして保存し、製品・既定・D236を維持する。D263の候補生成/登録41〜44%は校正不足の区間診断で、今回の速度予測や達成率へ換算しない。過去D230/D237/D241/D244の不採用結果も変更しない。

## 変更と精度

元の`_extend`のsupport前半と辞書組み立て後半だけを別CPU CPython拡張へまとめた。2置換を戻したASTは元method全体と一致する。全pairを元順で展開し、任意精度Python整数の差を取ってからFP64へ変換する。後でextend不能になる場合も変換例外を省略しない。held/missingのindex順、`np.ix_`、`np.linalg.solve`、残差差分、行列積、条件付き平均の差分は元の式/呼出し順のまま。

有限の連続exact float64 vectorだけは元`np.rint` ufuncをvectorへ1回適用する。整数への変換・元gaugeからの任意精度減算・sat上書きは元順を保ち、group/sat挿入順と独立copyも一致する。C++による近似rounding、FP32、FMA/solver変更、許容差の追加、候補・粒子・priorの間引きはない。非標準binding/型/buffer/非有限conditionalは元blockへ1回fallbackし、scalarのwarning/例外順を維持する。strict NumPy policyは元whole stepへ戻す。scopeは既存per-instance viewのみで、元module/runtime methodをpatchしない。

51tests PASS、既知pytest設定warning1。8seedのsupport/組み立て比較、巨大gaugeと任意精度整数、半整数/1ULP/負零/subnormal/最大有限値、全pair変換overflow、非有限値のscalar例外、custom mapping/numeric/binding/buffer、1000回参照数/入力更新、AST復元、CPU/GPU×2seed×3構成×4epochの全状態・位置/P・RNG・graph/prior metadata、strict例外時部分状態とcleanupを照合した。試験・native source・PEは初回実行後固定。

別の147組のDD境界照合では、2**52/53/63/64/100/500/1023の±1近傍、最大有限binary64整数、overflowを3gaugeで検査し、元変換bits/例外が一致した。solveを観測するcallbackでは呼出回数・全引数の順序/bytesが一致し、LinAlgError→None経路も同じ。元methodのliteral codeも別照合した。この検査に速度sampleは含めない。

最初の2つのliteral-code追加検査はboundary比較前に停止した。元moduleを全体compileすると`np`のimportがcompilerへ分かり、import文を含めない既存fixtureと12か所のLOAD_GLOBAL/LOAD_ATTR呼出用flagが異なる。全const/name/variable/flag/例外table/位置と190命令を照合し、差をその12bitへ限定した。元classのASTを試験と同じimport条件でcompileするとcode全体が一致することを確認した。失敗検査sourceと診断を保持し、数値実装/試験fixture/PEは変更していない。

## 完全PFの速度

8構成、元CPU/候補CPU/元GPU/候補GPU/CPU A/A/GPU A/Aの6mode、4個の事前RNG初期状態seed20260710〜20260713、5round×3group。2880timed/192warmを1回完了し、全3072 raw記録の順序/件数/hash/counterと192warm全状態/metadataを独立auditした。各構成の観測・初期粒子/configは固定し、4seedはRNG初期状態を変える。元の各粒子choiceを毎回実行する。

1groupは4完全PFのinteger-ns合計。paired短縮率の3group中央値→5round中央値を用い、短縮10%以上かつ4/5改善round、同backend A/A中央値abs5%未満を要求する。GPU jointは候補CPU比にも同じ速度条件と両A/Aを要求する。合否は丸め前で判定し、負値/不利/揺れも全保存、完了済み計測の再試行なし。

| particles/unique/rows/groups,cluster,feedback | CPU短縮 | GPU同士短縮 | GPU対候補CPU | CPU A/A | GPU A/A | CPU/GPU副/joint |
|---|---:|---:|---:|---:|---:|---|
| 0: 64/6/64/4,c0,fb1 | -0.827% | +3.769% | -3.350% | +0.756% | -5.559% | False/False/False |
| 1: 64/32/64/4,c0,fb1 | +6.794% | +2.514% | +6.300% | +7.384% | -4.012% | False/False/False |
| 2: 64/64/64/4,c0,fb1 | +12.232% | +3.101% | -25.275% | -4.106% | -1.634% | True/False/False |
| 3: 128/32/64/4,c0,fb1 | +8.118% | -5.596% | -25.674% | -1.034% | -5.022% | False/False/False |
| 4: 64/6/128/8,c1,fb1 | +5.317% | -0.331% | -16.157% | -1.254% | +4.977% | False/False/False |
| 5: 128/128/128/8,c1,fb1 | -9.823% | +5.973% | +7.536% | -0.240% | +9.793% | False/False/False |
| 6: 64/6/64/4,c0,fb0 | -5.727% | +4.277% | -7.446% | +4.828% | -10.582% | False/False/False |
| 7: 64/6/64/4,c1,fb1 | -1.603% | -8.259% | -15.618% | -0.001% | +2.560% | False/False/False |

全modeで同じ所有register-LU scorer context1個を使う。新guard/一時配列/native呼出し/元solve・組み立て、全move/score転送・readback/weight/RNG/位置/FIX/feedback/resampleを含む。restore/deepcopy/RNG生成/callback再bind/hash/検証/保存は外。許可CPU1本/BLAS1/normal priority/非占有、CPU/GPUは直列。extract/search/time/satelliteとgraph solveはfixture境界で、実RTK/FGO全体の速度やGPU化率は未確立。GPU同士の追加短縮はGPU kernel変更を意味しない。

## 実データ

Tokyo run3/seed20260710/400RTK epochのCPU/GPU×元cluster0/1、各200PF状態・197valid/3skip、計800保存状態を凍結D236同backendと照合した。全state JSONL行、trajectory全配列dtype/shape/bytes、全cfgと全従来caller/prior metadataが一致する。元の診断引数を転送し、追加GPU診断転送0。extend native/support/assembly counterは別streamで全800件記録し、全valid stepで元extend call数と一致、各prefixで両native経路使用/fallback0を確認した。

| backend/config | paired PF | native support | native assembly | assembled integers | fallback support/assembly |
|---|---:|---:|---:|---:|---:|
| native_cpu/c0 | 200 | 2110 | 852 | 2272 | 0/0 |
| gpu/c0 | 200 | 2110 | 852 | 2272 | 0/0 |
| native_cpu/c1 | 200 | 2007 | 739 | 1826 | 0/0 |
| gpu/c1 | 200 | 2007 | 739 | 1826 | 0/0 |

最初のfield wrapperは計算・200状態/trajectory保存後、下位helperが`sys.argv`を置換済みのため終了時`--mode`検索で停止した。失敗source/plan/log/200状態/trajectoryを保存した。別pathのchecked wrapperは既に記録したscorer counterからbackendを取得し、数値実装/PE/入力を変えず上の4条件のaccuracy証拠を完了した。これは記録wrapperの修復で、完了した速度測定のやり直しではない。失敗runの200状態は上の800件に加算しない。

## メモリと保存

一時DDはnb×8B、held index arrayはheld数×sizeof(intp)、assembly時の一時rint arrayはmissing数×8B。Python lists/整数/辞書copy/buffer bookkeeping/allocatorを除くlogical payloadで、heap peakや総差分は未実測。入力/DD/problem値cache・永続native/CUDA work bufferを新設せず、GIL保持、call後の入力参照保持なし。元contextのhost/device stagingは各1143352B。新CUDA source/転送/sanitizer実行0で、CPU C++ ASanや一般version保証は行っていない。

36製品source・5installed PE・PF build PE・88過去reports・43archives7341filesを保全。実走は同じroute/seed prefixの4条件で、full-route/multiseed測位精度・全GTSAM速度gateは未完了。既存codecのPython scalar NaN payload正規化制約を継承し、新しい数値許容差はない。

[結果](../benchmarks/rbpf_extend_support_20261017.json)、[完了照合](../benchmarks/rbpf_extend_support_completion_check_20261017.json)、[probe/API](../benchmarks/probes/rbpf_extend_support/README.md)。
