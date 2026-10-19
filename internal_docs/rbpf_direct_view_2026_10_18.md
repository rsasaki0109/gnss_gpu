# 同じ数値関数を使うクラス経由のPF呼出し (D277, 2026-10-18)

call-local viewの明示的なcached MethodType生成を実入力200callで2972件から600件へ79.812%減らした。54unique tests PASS、実GPU200保存状態・全測位軌跡/cfg/従来metadataと、完全PF2880timed/192warmの全同backend状態・RNG・feedbackが参照値へ一致した。速度screenはCPU 0/2、GPU同backend 0/2、joint 0/2。対照の不安定さも残り、速度向上は実証せず製品/既定を維持する。

## 実装と同一性

D233の元viewは__getattr__内で数値functionを自分へboundし、dictへ保存する。今回の明示callerは元D236をそのまま構築し、その同じ数値function objectをcontroller専用クラスへ一度登録する。元factoryが作った同じview objectを、同一slot layoutのクラスへcompiled body入口で切り替え、最初のoracle3methodが入った通常cacheをclearする。oracleが保持する元の3bound methodもそのまま有効である。GenericGetAttrによる通常のclass method-call経路を使い、以降の元__getattr__/cache経由生成を避ける。

元の数値function/code、全粒子/候補、FP64、prior、IRLS、閾値、choice順序、RNG、FIX、feedback、cluster更新を変更しない。state属性はdescriptorから実runtimeを毎回読み、setterも元と同じ実runtimeへ渡す。view/input/H/DD/matrixのepoch間共有は追加しない。従来callerのbinding/global/error-policy guardと全tailを保つ。

exact元view、元functions dict/value identity、専用classの関数/field identity、通常cacheのstr key/MethodType/self/function identityだけを対象にする。未対応cache/map/変更済みclass/fieldは元viewで元bodyを1回実行する。class切替がTypeErrorならcacheを触らずdeclineする。固定CPython3.12の私有API実験であり、任意private namespace/audit hook/同時書換え/OOMの一般保証ではない。

layout preflightで両型のbasicsize40/itemsize0/dictoffset0/weakrefoffset0と同一object identity、元function identity、既に保持したmethodの呼出し、属性書戻しを確認した。型構築はsetup、全guard/class切替/cache clear/counterは完全PF timer内に含める。実runtime/moduleへのmethod差替えは行わない。

## 実データの処理量と精度

Tokyo run3/seed20260710/400RTK epoch、GPU/cluster0の新候補だけを実行した。元PF-return boundaryのgoldへ200state JSONL全行をliteral比較し、全trajectory arrayのdtype/shape/bytes、cfg、従来metadata、入力/外部source/PEを照合した。D276のpost-process snapshotと境界を混ぜず、rtk keyの除外や数値許容差拡大はしない。scalar NaN payload codecの既存制約は継承する。

sys profileは元private __getattr__のcode entryだけを数え、既存profilerがあれば拒否し、終了/例外時に所有したcallbackを復帰する。観測は数値sourceをpatchしない。通常GC policyを保ち、追加GPU転送0、GPU score197/skip3、scalar fallback0である。

| 200callでの監視対象 | 保全済みD275元 | 新候補 |
|---|---:|---:|
| 元view cache経由の明示MethodType生成 | 2972 | 600 |
| 元__getattr__ entry | 未計測 | 600（3/call） |
| compiled終了時cache | 元のまま | 全200callで0 |
| return後の監視元method残留 | 200/200 | 0/200 |
| 新class適用/decline | 対象外 | 200/0 |

約80%はこのcache経由の生成件数だけで、全Python method/全allocation/heap bytesの減少率ではない。元D275の保全済みwork counterを再利用し、元/過去候補の性能測定は取り直さない。profile付きwall値はuncalibratedで、速度比や上限へ換算しない。

## 試験

新本体24件と凍結D275回帰21件、observer5件、集計gate4件で54unique tests PASS。本体はscalar/nativeCPU/GPU×2seed×2cluster×4連続epoch、同一view/元関数/保持済みmethod、属性書戻し、通常以外のcache/map/field/class decline、GC policy、strict/skip、元RNG例外の途中状態とcleanupを確認した。observerはlegacy metadata一致、3lookup/call、既存profile保持、例外時cleanupを検査した。

集計testsでは、A/Aの正負差の相殺を拒否し、中央値が小さくても4/5安定round未満なら不合格にする。整数tickを足してから比を計算する。observer testのimport済みpytest fixture名と引数名によるF811だけをfile/SHA付き例外とし、数値実行済みのtest本文を維持する。他の選択Ruff/compile/EOL/diffを検査する。

## 完全PFの速度screen

新候補の実入力精度が一致し、全200callでclass適用/decline0/cache0/残留0、元__getattr__600、明示生成50%以上削減という事前triageを満たしたため実施した。GC時間はtriage条件に使わない。元/新CPU/GPUとCPU/GPU A/Aの6mode、4RNG初期seed、2条件（64粒子/6unique、64rows/4groups/c0と128rows/8groups/c1）、4連続epoch、5round×3groupで2880timed/192warm。

各epochで全6streamの入力準備を済ませ、mode順をepoch/seed/round/groupでrotate/reverseして6個の完全PFをserial計時し、その後に全6状態をcopy/hash/検証/保存する。stream内の粒子/RNG/holdは連続し、epoch problemは独立copyにする。通常GCのままcallback/forced collect/threshold変更なし。共有scorer1context、許可CPU1本/BLAS1/normalpriority/非占有。restore/準備/copy/hash/照合/gzipはtimer外で、元tail・転送/完了待ちと新guard/cleanupを全て含む。

1groupは4seed×4epochの完全PF integer-ns合計。速度はgroup paired短縮の3group中央値→5round中央値10%以上、4/5改善roundを要求する。対応A/Aは各roundの**3group差絶対値中央値**を取り、その5round中央値5%未満かつ4/5round以上が5%未満を要求し、正負相殺で合格にしない。GPU jointは候補CPU比と両A/Aも必要。正は短縮。

| 条件 | CPU対元 | GPU対元 | CPU A/A絶対中央値 | GPU A/A絶対中央値 | CPU/GPU同backend/joint |
|---|---:|---:|---:|---:|---|
| c0 | +2.524% | -1.349% | 10.476% | 5.149% | 0/0/0 |
| c1 | -2.085% | -2.958% | 13.477% | 13.006% | 0/0/0 |

両条件で速度採用gateを満たさず、両A/Aも不成立だった。c0のCPU符号付き+2.524%を有効な短縮とせず、全負値/全group/全roundを保持する。全3072rawの順序/count/hash/metadata/counterと192warm全保存状態を別auditで再照合した。これはsnapshotを挟むfixture PF call和であり、抽出/整数探索/時刻/衛星/graph solveは境界外である。連続実RTK速度・GPU使用率を主張しない。

## 次の調査対象を件数で絞る

別の計時なしprobeで元D236 nativeCPUの4epoch fixtureだけを調べた。profile callbackはframeからHのlenを読むだけで、数値sourceと入力を変更しない。

| fixture | _copy_H呼出し | 複製する内側dict | 複製する整数entry | 元view lookup |
|---|---:|---:|---:|---:|
| c0 /4epoch | 536 | 2144 | 19152 | 2737 |
| c1 /4epoch | 542 | 4336 | 38795 | 3982 |

仮説複製の処理量があることを確認したが、件数だけで実時間の主要因と断定しない。次の候補では、実routeの複製量/所有権/全writerの不変条件を確認してから、その費用を調べる。共有で粒子間のHが書き換わる変更や、候補/精度/RNGを減らす近道は採らない。

## 保全

36製品source・5installed PE・PF build PE・114過去reports・56archives/11676filesを保全する。開始時は前回独立full-byte照合を再利用し、全manifest/report/source/PEと最新374archive filesを照合、最終時に過去archive bytesを1回再確認する。新しいnative/CUDA buildやsanitizerはなく、製品/既定の変更と速度採用はない。

[結果](../benchmarks/rbpf_direct_view_20261018.json)、[完了照合](../benchmarks/rbpf_direct_view_completion_check_20261018.json)、[caller](../benchmarks/rbpf_direct_view.py)、[実走observer](../benchmarks/bench_rbpf_direct_view_field.py)、[速度screen](../benchmarks/bench_rbpf_direct_view.py)。
