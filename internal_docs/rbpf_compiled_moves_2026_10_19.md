# D293: RBPF候補制御4メソッドのリテラルコンパイル (2026-10-19)

**標準条件のCPU/GPUとも速度screen不合格。採用しない。** 状態・乱数・metadataの一致は確認できたが、効果的な高速化を示せなかった。製品/default・測位モデル・精度条件は維持する。

## 実装

MoveBatchStep、MoveReuseStep、MoveKeyMemoStep、NativeCanonicalStepの`_moves`をASTのままplain classへ投影し、Cython3.1.4でCPython C APIへコンパイルした。4メソッドのASTを元sourceと完全照合する。Cythonはworkspace内の専用directoryへpinして導入し、既存Python/field環境を変更していない。`infer_types=False`、`annotation_typing=False`、`cdivision=False`、bounds/wraparound check有効、MSVC `/O2 /fp:strict`。生成C++/新PE/全compiler fileのhashとinstall reportを保存した。

候補数、prior、NumPyの式・reduction・scorer、粒子のmutation順、各粒子のRNG.choice呼出しを維持した。新しい数値C型・CUDA kernel・GPU転送方式は導入していない。既存scorer/3CPU extensionも維持する。entryのmethod/code/global/builtin/NumPy/math bindingを確認し、対象外なら元実装へ戻す。同期した通常bindingが対象で、任意のcallbackによる実行途中のnamespace変更やframe introspectionまで一般互換と主張しない。

## 状態検証と全体時間

33tests PASS。portable AST投影4件とWindows専用29件で、scalar/nativeCPU/GPU、2seed、4configuration、4epochのoriginal/off/on状態・乱数・feedback・metadataを比較した。binding変更時のliteral fallbackとstrict NumPy fallbackも確認した。追加の速度screenでは4seed、2条件、8mode、5round×3groupで**64warm/960timed=1024全PF呼出し**を保存し、同backendの状態・乱数・metadataが完全一致した。warm全64状態も保存した。

各groupは4seedの全PF整数nsを合計し、3group中央値→5round中央値で判定する。候補準備/entry guard/packing/scoring/transfers/choice/conditional/FIX/feedback/resamplingを含み、入力復元・乱数生成・callback再bind・hash/保存は計時外。一つのRegisterLUScorerを共有し、CPU1/BLAS1/通常GC/priority、非専有環境で直列に測った。mode順とseed順を事前固定して交互に回し、cold/slow/負値も残した。

事前条件は両controlそれぞれに対し短縮中央値>=10%かつ4/5roundで正、同じbackendのA/A絶対差中央値<5%かつ4/5roundで絶対差<5%。**標準条件が通ったbackendだけが次の実データ検証に進める。** 以下は観測差であり採用利得ではない。

| 条件 | 元実装比 | 機能off比 | A/A絶対中央値 | A/A絶対差5%未満 |
|---|---:|---:|---:|---:|
| 標準64粒子/初期6仮説 / CPU | +3.854% | +3.564% | 4.945% | 3/5 |
| 標準64粒子/初期6仮説 / GPU | +8.200% | -0.268% | 11.205% | 1/5 |
| 大規模128粒子/初期128仮説 / CPU | -3.597% | +5.962% | 3.726% | 2/5 |
| 大規模128粒子/初期128仮説 / GPU | +1.460% | +4.112% | 5.250% | 1/5 |

全raw grid/order/counter/64状態hash/1024call hashと統計を、速度実行前に固定した独立auditorでも照合した。標準CPU/GPU0/2のため、新しい実RTK走行は行っていない。fixtureの一致を実測位精度・fullroute・多seedの新保証やGTSAM内部graphの全bytes一致へ拡張しない。scalar NaN payloadをcanonicalizeする既存encoderの制約も維持する。GPU使用率・GPU化率・FGO速度は測っていない。

## 不備と保全

最初のlaunchは入力freeze planと出力measurement planの名前衝突を検出し、scorer作成/全PF/タイマー開始前に終了した。失敗source/plan/log/exitを残し、別版orchestratorとauditorでは出力先文字列だけを変更した。計測benchmark・数値実装・tests・PE・gateは同一bytesのまま、完了済み測定の取り直しはない。

148既報/73archives/19087files、製品36source/5installedPE/PFbuild、旧FAIL/aborted setを保全する。新しい数値kernel/sanitizer runは0。次の最適化は、この候補を軽微に変えて再測定するのではなく、全PFの実行量や待ち時間を構造的に減らす根拠がある案を対象にする。

[結果](../benchmarks/rbpf_compiled_moves_20261019.json)、[完了照合](../benchmarks/rbpf_compiled_moves_completion_check_20261019.json)、[controller](../benchmarks/rbpf_compiled_moves.py)、[projection/build](../benchmarks/probes/rbpf_compiled_moves/README.md)、[full PF screen](../benchmarks/bench_rbpf_compiled_moves.py)、[native tests](../tests/test_rbpf_compiled_moves.py)、[portable AST tests](../tests/test_rbpf_compiled_moves_projection.py)。
