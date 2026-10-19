# D292: 子側watchdogを外した独立processの初回更新latency比較 (2026-10-19)

初回移行から有効更新までのlatency screenはFAIL。以下の差は保存した観測差であり、有効な速度利得として採用しない。 数値実装・FP64・IRLS・prior・乱数・粒子/仮説数・FGO処理順・通常GC・priority・power設定は維持した。製品/defaultへの採用は引き続き0。新しい連続RTK throughput gate、sanitizer再試験は実施していない。

## 新しい固定計測と判定

D289は3試行目の異常終了で15primaryが揃わず、速度gateは未評価だった。D290でperiodic stack dump threadのfaultがCUDA/GTSAM load前にも再現し、D291で親のみの監視下の2prefixが正常終了した。そこで新しい計測instrumentationとして、子側periodic dump/debugger/sanitizerを使わず、凍結したD291親monitor/workerとD289数値caller/evaluatorをそのまま再利用した。古いaborted setやD282/D284/D287の既完了FAILを置換・取り直ししていない。古いD289 crashの正確な原因、D290 case5の失われた終了履歴、sanitizer hangの解消を証明するものではない。

最初にoverlap専用accuracy400epochを1本、その後A/overlap/AAを5巡・計15本、合計16prefixをすべて別process/freshNUMBAcacheで直列実行した。各400epoch、共通15秒idleをdriver全経過に含める。CPU1/BLAS1、非専有host/GPU。primaryにはepoch間のstate encodingを挟まない。方式の順は元D289と同じ5つの順序で事前固定し、途中の利得に基づく追加・除外・再走はしなかった。

一次指標は移行RTK開始から最初の有効scoreを含むRTK終了までのwall時間。元D289と同じく、A/A絶対差中央値<5%かつ4/5巡の絶対差<5%、両controlそれぞれに対し短縮中央値>=5%かつ4/5巡で正を要求する。全15本が完走・精度一致・launcher/worker正常終了することも必須。全6400RTK clocks、全400epoch/phase/driver時間は残すが、全体高速化の合否は判定しない。

| 比較 | 短縮率の中央値 | 改善した巡 |
|---|---:|---:|
| A → overlap | 33.255% | 5/5 |
| AA → overlap | 38.668% | 5/5 |

A/A絶対差中央値は15.977%、絶対差5%未満は2/5巡、control validはFalse。初回latency screenは**FAIL**。5巡のlatency中央値はA 2507.587ms、overlap 1617.321ms、AA 2752.988ms。代表値だけからpaired gateを作らず、各巡の2controlとの差を使う。whole400/phase/driverの全生時計と観測差はJSONに保持し、連続throughputやGPU使用率/GPU化率へ換算しない。

## 精度・終了・照合

16prefix/6400RTK、3200PF calls、3152score intervalsが完全に保存された。accuracy prefixの200postRTK statesは、旧PF-return境界とrootRTKだけを事前宣言どおり除いて全タグ/array bytes/RNG等が一致した。全16最終boundaryはrootRTKを含む旧post-driver基準と一致し、57NPZ arrays×16=912rawNPY membersが展開後bytesまで一致した。宣言済みstate境界でありGTSAM内部graph全bytesではない。scalar NaN payloadのcanonicalizationという既存encoderの制約は維持する。今回の精度確認は1route/seedの400epoch prefixで、full route/多seedの新保証ではない。

16numeric workersと全38観測processがsignalled exit0。親は0.1sで開いたowned handleを確認し、結果fileやcompleted markerだけで終了と判定しない。短いOS version照会`cmd.exe /c ver`はpoll間に終わる場合があるため、測定前から観測数2または3を許容した。launcher/workerは必須、追加processが観測されたら正確なcommand/親子関係/生成順/exit0をすべて要求する。未観測の任意のdetached/短命子を封じ込める方式ではない。metadata/親全経過の`speed_evidence=false`はその記録自体を速度指標にしない意味で維持し、今回の新RTK clocksだけを事前宣言したgateに使う。D291の安全診断timingを速度評価へ流用しない。

新portable exit-evidence testsは24PASS、Ruff/diff check成功。旧D28927testsの結果と394実入力のbit一致証拠はhashを保全して再利用し、新しいunique test数へ足さない。数値source/test/caller/runner/PE/入力/gateは実行前に凍結した。別auditorを実行前に凍結し、全整数時計/包含/順序/200状態/16最終境界/912NPY members/6one-shot native probe ticks/親子関係/正常終了を再照合した。変更しない元D289 evaluatorの全metricと独立計算が完全一致することも確認した。146既報、72archives/18513files、製品36sources/5installedPE/PFbuildを保全する。

[結果](../benchmarks/rbpf_parent_latency_20261019.json)、[完了照合](../benchmarks/rbpf_parent_latency_completion_check_20261019.json)、[新exit policy](../benchmarks/rbpf_parent_measurement.py)、[24tests](../tests/test_rbpf_parent_measurement.py)、[元scorer/evaluator](../benchmarks/rbpf_transition_wake.py)、[D291正常終了確認](rbpf_lifecycle_2026_10_19.md)。
