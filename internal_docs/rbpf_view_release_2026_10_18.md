# RBPF呼出し後の内部メソッド参照を解放する (D275, 2026-10-18)

呼出し終了後に循環参照の回収を待つ内部メソッド保持を解消する明示callerを保存する。26tests PASS、実GPUの元/候補200状態ずつ計400と、4epoch連続PF2880timed/192warmの全同backend状態・RNG・feedback・従来metadataがbits一致した。速度screenはCPU 0/2、GPU同backend 0/2、joint 0/2。速度向上は確立せず、製品/既定を維持する。

## 対象と実装

D233からのcall-local `_CanonicalView` は、自分にboundされたメソッドを内部dictへ保持する。元処理は数値計算終了後にcontrollerのview/oracle参照を外すが、その内部dictとbound methodの循環は残る。今回のwrapperは元D236のcompiled callableをそのまま呼び、終了finallyで通常の内部method cacheだけをclearする。元の数値function、全粒子/候補/prior/FP64/IRLS/threshold/weight/RNG/commit/FIX/feedback/resampleを変更しない。

viewとcacheが元のexact型、全keyがstr、全valueがそのview自身にboundされたexact MethodTypeの領域だけを解放し、未対応は旧lifetimeへ戻す。外部が保持したmethodを無効化しない。GCのenable/threshold、forced collection、epoch間のview共有、数値cacheやsource moduleの差し替えは追加しない。これは元の私有cacheを使う固定API実験で、任意の私有cache/namespace変更や同時state書換え/OOMの一般保証ではない。

## 実入力で先に確認した処理量

Tokyo run3/seed20260710/400RTK epoch、GPU/元cluster0で、元と候補各200PF状態・197valid/3skipを保存し、凍結D236同backendへ照合した。全state JSONL行、全trajectory配列のdtype/shape/bytes、cfgと全従来caller metadataが一致する。元のscorer diagnostic flagを転送し、追加GPU転送0、各197GPU score・scalar fallback0を確認した。

観測は各callの元 `_pair_dd` bound methodへの弱参照と、処理終了時のcache件数、GC callbackだけを使う。元/candidateの数値計算/入力は同じで、observerを両方へ加える。GC policyは変えない。計測・記録overheadの校正はなく、wall/GC値を速度比や厳密な上限へ換算しない。

| 診断値 | 元 | 候補 |
|---|---:|---:|
| return直後に監視methodが残るcall | 200/200 | 0/200 |
| 監視methodの同時残留peak | 11 | 0 |
| 内部method解放 | なし | 2972件/200call |
| 観測GC ns / PF診断ns | 10.685% | 14.924% |

候補は全200callで内部cache0、監視method残留0、decline0だった。元のcache件数合計と候補の解放件数2972が一致する。これは参照保持の改善で、heap peakや総メモリbytesを測った結果ではない。候補はgen2 GC19回、元10回で、候補のGC比率が高い値もそのまま残す。GC時間差の原因をこの変更だけに帰属しない。

事前triageは元の残留callが100以上かつ観測GC比率5%以上であること。実入力200/200・約10.7%だったため、次の通常GCでの完全PF screenへ進んだ。このtriage自体を速度採用基準にはしない。

## 数値・lifetime試験

21本の本体testsではCPU/GPU/scalar×2seed×2cluster×4epochの全state/位置/P/RNG/feedback/従来metadata、GC policy維持、外部が保持したmethodの再呼出し、未対応cache型/値/key/別selfのdecline、skip/strict fallback、元RNG例外時の途中stateとfinally解放を確認した。unitの一部だけで自動GCを一時停止してreturn後の参照を検査し、終了時に元policyへ復帰する。さらに別fileのobserver5testsで、実GC eventの記録・callback除去・skip/strict/例外cleanupを検査した。両fileとも初回数値実行後固定し、26unique tests PASS。

実走とfixtureの一致は、fullroute/multiseed測位精度の保証とは区別する。元の数値許容差は広げず、scalar NaN payload codecの既存制約を継承する。新しいnative/CUDA sourceやbuild/sanitizerは追加しない。

## 通常GCでの連続epoch速度screen

observerを外した6mode（元/候補CPU/GPU、CPU/GPU A/A）、cluster0/1の2条件、4RNG初期seed×4連続epoch×5round×3groupで2880timed/192warm。粒子/RNG/RTK holdをstream内で引き継ぎ、各epoch problemは別copyにして古いDDPR cacheを再利用しない。snapshotはstep間のtimer外に取り、全warm状態・全timed hash/metadata/counterを保持する。これはPF call時間の和であり、snapshot間のGCを含むstream全体や実RTK throughputではない。

1groupは4seed×4epochの16完全PF integer-ns合計。paired短縮の3group中央値→5round中央値で10%以上・4/5改善round・対応A/A abs中央値5%未満を要求する。GPU jointは候補CPU比と両A/Aも必要。GCは通常enable/thresholdのまま、callback/強制回収/手動閾値変更なし。許可CPU1本/BLAS1/normal priority/非占有、mode/seedをrotate/reverse、同じscorer1contextを使う。正は短縮。

| particles/unique/rows/groups,cluster | CPU対元 | GPU対元 | CPU A/A | GPU A/A | CPU/GPU同backend/joint |
|---|---:|---:|---:|---:|---|
| 0: 64/6/64/4,c0 | -6.689% | -3.947% | -14.655% | +11.318% | False/False/False |
| 1: 64/6/128/8,c1 | -5.442% | -2.863% | -8.425% | +7.071% | False/False/False |

CPU/GPU A/Aはいずれも0/2だった。この較正不成立を含めて速度未確立とし、次の速度評価ではGCの発生位置、restore/snapshotとの境界、実行待ちのばらつきを先に切り分ける。現時点で原因を1つに断定しない。

元の型/binding guard、cache判定/clear/counter、全score転送と完了待ち、元の各choice/conditional/feedback/PF tailを含む。restore/RNG初期化/callback再bind/入力epoch更新/deepcopy/hash/照合/gzipは外。全3072rawの順序/件数/epoch/seed/hash/counterと192warm全状態を別auditで再確認する。抽出/整数探索/時刻/衛星/graph solveはfixture境界である。対照不成立・負値・全roundを保持し、完了済み測定は取り直さない。

## 保存と制約

保持methodの変更だけなので、元host/device staging各1143352bytes、GPU数値source/転送は同じ。heap peak、controller自身の他の循環、任意custom private cacheのdestructorや他versionへの一般保証は未測定である。製品/既定変更、新しいGPU化率/全FGO速度の主張はない。

保全用preparer初版は数字prefixの重複置換によりassertで止まり、その後の未作成preserver起動も数値実行前に停止した。別preparerでassert全文を原子的に置換した。field runner初版は存在しないABI付きGTSAM filenameのSHA読出しでplan作成前に停止し、別runnerで実在するgtsam.pydへ直した。元sourceと失敗理由/修正の復元proofを残し、数値helper/試験は変更していない。

36製品source・5installed PE・PF build PE・110過去reports・54archives/11067filesを保全する。実走GTSAM PE/外部tc/cssrlibと入力hashも同一。測定済みbenchmarkのflush2文だけE702をfile/SHA付きで除外し、他のRuff/compile/EOL/diff/artifact/public text検査を行う。

[結果](../benchmarks/rbpf_view_release_20261018.json)、[完了照合](../benchmarks/rbpf_view_release_completion_check_20261018.json)、[caller](../benchmarks/rbpf_view_release.py)、[実走observer](../benchmarks/bench_rbpf_view_release_field.py)、[連続PF](../benchmarks/bench_rbpf_view_release.py)。
