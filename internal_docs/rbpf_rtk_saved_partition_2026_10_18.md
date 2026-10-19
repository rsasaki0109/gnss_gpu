# 保存済みRTK計測をPF内外へ分ける (D283, 2026-10-18)

PFの割合が小さいだけでは前回の不通過を説明できない。D282の連続RTK保存記録では、PFは後半wallの中央値で旧版53.697%、同期対照47.922%、非同期候補46.979%、旧版A/A51.769%を占める。同じ旧版同士でもPF区間に絶対差中央値16.908%、PFが動かない前半に35.038%の変動があった。D282の速度gateはFAILのまま保持する。

## 実行を増やさない検証ツール

新しいhelperが、D282全24prefixの9600個のRTK時計と4800個のPF時計を読み取る。元field sourceは`perf_counter_ns`の整数差を`1e9`で割って`call_seconds`へ保存していた。3600秒以下の有限非負floatに限定し、最も近い整数nsへ戻した値を再び`1e9`で割って元floatとexactに一致することを要求する。元時刻を丸め直して有利な値へ調整せず、全4800件で元の整数を復元できた。別auditorではDecimalのhalf-upで独立に同じ整数を得た。元のfloat、最初のcall、3skipの時間も残る。

元のselected call/timer/appendの4文をASTで照合し、diagnostic suffixの除去を戻したASTも一致する。凍結modelの`phase != 2`ならreturnというguardと唯一のPF dispatchを別照合した。全24の既存adapter proof/source SHAと結び、PFはphase2のRTK呼出しの内側にある。call数だけから個々のepochとの対応は推測せず、全200callを合計して後半200epochの合計から引く。

各prefixで`後半wall = 全PF wall + PF外wall`、`driver wall = 全RTK wall + driver外wall`へ整数nsで正確に足し戻る。PF外はFGO/AR/前処理/後処理/時計wrapper等の全てを含み、GTSAMだけを表さない。wall−CPU等の負の差もWindows時計の量子化を含む元の符号で保存する。CPU時計を速度合格の代用にしない。

## 保存された全5roundの対照

4warmを全て残し、20timedの5roundを元の順・4armで集計する。全prefixの400RTK/200PFを使用し、遅いcallやstartupを除外しない。正は候補の短縮、負は増加。ここでの再集計は診断であり、元の不通過gateを再判定・置換しない。

| 境界 | 旧版対候補の中央値 | 同期対照対候補の中央値 | 旧版A/Aの絶対差中央値 |
|---|---:|---:|---:|
| PF区間 | +16.289% | +2.203% | 16.908% |
| PFなしの前半RTK | +12.523% | -10.161% | 35.038% |
| 後半RTK全体 | +3.716% | -0.344% | 15.663% |
| 後半のPF外 | -10.842% | -2.683% | 20.283% |
| 前半のthread CPU | +1.611% | -11.303% | 24.012% |
| 後半のthread CPU | +4.664% | -0.921% | 14.941% |

PF区間の候補差+16.289%だけでは利得を確定できない。PF自体の旧版A/Aも16.908%あり、同期対照差は+2.203%。PF外は旧版対候補で-10.842%、旧版A/Aは20.283%だった。各中央値は分母が異なり、足してRTKの短縮率にはしない。PFが全体で無視できる量だという説明も、この観測とは合わない。

前半にはPF dispatchがなく、そこでもwall絶対A/A35.038%、thread CPU24.012%の差がある。後半thread CPUにも14.941%の絶対A/A差がある。このためwallの待機差だけを差し引く方法では解決できない。scheduler、GC、CPU周波数、GPU待ち、特定の他processのどれが原因かは今回の資料では特定していない。外部processの完全な一覧/非稼働も証明していない。

## 次に使う優先順位

IMU積分のGPU化は今回の試作対象から外す。別のD246 nativeCPU/cluster0,1の粗い観測では、`build_pim`200callがそれぞれ48.353/49.985ms、snapshotを明示的に引いた観測workに対して0.325/0.385%だった。現在のGPU処理割合や因果的な短縮上限ではなく、instrumentationを含む過去1route/seedの優先順位資料としてだけ使う。

次の前提は実RTK対照の安定性を確認できる別の計測protocol。PF比率を減らす候補だけを追加し続けたり、前回の完了済み測定を取り直したりはしない。今回追加したhelperは、そのprotocolで保存するPF/RTKの時計整合性にも再利用できる。GPU実稼働率、新しい高速化の採用、full route/多seed精度の主張はない。

## 検証と保全

33unique tests PASS。元source変更拒否、整数ns復元の端点/4096整数例、不正float/負のzero、prefix欠落/再順序化、元summaryとの不一致、PF包含違反、CPUの符号保持、warmの保持を検査した。別auditorは全24prefix/9600RTK/4800PF、13系列・39対照・195個別差を独立照合した。推定器実行/GPU起動/新しい速度測定は0。D282の200PF戻り状態・全24軌跡/最終状態の一致は保存済み精度証拠であり、新しい実走精度試験には数えない。

126過去reports・62archives/13670files・36製品source・5installed PE・PF buildを保全する。粒子/候補/FP64/prior/RNG/閾値/feedbackの数値source、製品/既定は変更していない。analysis JSONのlegacy key `original_RTK_speed_gate_unchanged: false`は元gateの値Falseを保持する意味で、新しいreport/auditでは`original_RTK_gate_preserved: true`と分けて明記した。

[結果](../benchmarks/rbpf_rtk_saved_partition_20261018.json)、[完了照合](../benchmarks/rbpf_rtk_saved_partition_completion_check_20261018.json)、[解析helper](../benchmarks/rbpf_rtk_saved_partition.py)、[tests](../tests/test_rbpf_rtk_saved_partition.py)。
