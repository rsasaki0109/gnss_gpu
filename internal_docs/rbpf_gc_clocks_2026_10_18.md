# GPU実RTKの交互実行とGC/CPUサイクル観測 (D276, 2026-10-18)

同じD236 GPU処理同士をepochごとに交互実行し、速度を判断する前の対照安定性を調べた。51unique tests PASS、18prefix/7200RTK epoch/3600PF状態の粒子・重み・全永続PF状態・RNG・出力が保存goldへ完全一致し、全測位軌跡arrayのdtype/shape/bytesも一致した。事前の全400/phase2安定性gateは **FAIL**。高速化候補の測定や採用は行っていない。

## 新しい観測境界

D271の二つの独立processを使うready/go/done/advance protocolを維持し、両ready後に各bodyを順に実行、両done後に次epochへ進む。先行childをepoch/pairごとに交代し、各phaseのA/AA先行を100回ずつにする。process作成順もpairごとに反転する。全400epoch・初回compilation・遅い/負の差を保持する。

元D246 adaptのAST復元proofを維持し、D236数値関数と既存の訂正済みRegisterLUScorer(register)を使う。元RTK processを1回だけ呼び、結果/例外をそのまま返す。wall/process CPU/thread CPUの時計区間を保持し、完了後のhookでsnapshot/metadataを保存してからdoneを送る。GCのenable/thresholdは変更せず、forced collectionは実走で行わない。候補削減・FP64変更・許容差拡大・RNG追加draw・graph処理省略はない。

PF return後に親RTKがhold/navを後処理できるため、新保存状態はRTK process完了境界と明記した。事前planどおり、root `rtk` だけを除いた全PF状態/粒子/重み/RNG/出力を従来PF-return goldへ照合する。root `rtk` を含む全新状態はA/AA同士で完全一致を要求する。全trajectory・cfg・従来caller metadata・入力・外部source/PEも照合する。保存境界の違いを数値許容差へ吸収せず、実行後に対象keyを変えない。既存scalar NaN payload codecの制約は継承し、array bytesと有限scalar hexは保持する。

## 観測なしの主要A/A

1warm pairのaccuracy確認後、GC callback/CPU-cycle/nested PF probeを入れない5primary pairを一度だけ実行した。各pairの相対wall差の**絶対値**中央値が5%未満、かつ4/5pair以上で絶対差5%未満を、全400とphase2の両方へ要求する。CPU時間は診断に保持し、安定性判定への後付け代用はしない。

| 対象 | wall絶対差中央値 | abs差5%未満 | 個別gate |
|---|---:|---:|---|
| 全400 | 7.092% | 2/5 | False |
| phase1 200 | 10.133% | 2/5 | False |
| phase2 200 | 3.450% | 4/5 | True |

warmを含む各child/pairでNumba cache directoryを別にしたため、warm pairは後続childのJIT cacheを温めない。全400はそれぞれの初回compilationも含むcold-start寄りの観測である。phase2の判定も併記し、startupを後付けで除いて全400gateを合格扱いにはしない。待機・別GPU context切替・snapshot/serializationは連続実走のcache/GC scheduleを変える。このgateは連続route throughputや過去FAILの再評価ではない。

## GC区間とscheduled CPU cycles

primary完了後に1pairだけ両childへGC callback・QueryThreadCycleTime・nested PF clocksを加えた。GC eventのstart/end区間、generation、epoch、collected/uncollectable、thread、境界跨ぎを保存する。process、pf、snapshot、serialize、barrier、outsideを区別する。process集計の内側GCはprocess+pf区間のclean eventだけで、PFはpf区間だけを使う。境界跨ぎ/他thread eventは別記し、無理に割り当てない。transition時のallocationやobserver自体の影響は残り、未観測実走の厳密なGC費用や因果効果ではない。

| child | phase2 wall | phase2 thread CPU | phase2 scheduled cycles | body内へ帰属したGC/wall | PF内GC/PF wall |
|---|---:|---:|---:|---:|---:|
| A | 17.899s | 16.984s | 43769617294 | 6.261% | 12.207% |
| AA | 17.668s | 17.391s | 45113863170 | 6.550% | 12.209% |

CPU cyclesは呼出しthreadが実行されたcycle数であり、nsへ変換しない。CPU clockはWindowsの量子化を保持し、integer tickを集計してから比率を出す。wall−thread CPUには待機/割込などが混在し、差だけでOS schedulerやGPU待ちが原因と断定しない。cycle数も周波数変動/実行経路の影響を受け、GPU稼働率やCPU実行待ち時間の直接測定ではない。

## probeを入れる影響の参考値

観測あり/なしを2pairだけ比較し、label役割を反転した。正は観測ありのwall増加。両方ともpost-clock詳細snapshotは同じで、この比較が見るのは追加GC/cycle/PF probeだけである。

| pair | 観測ありlabel | 全400相対増加 | phase2相対増加 |
|---|---|---:|---:|
| 7 | A | +7.342% | +1.597% |
| 8 | AA | +2.477% | +0.624% |

2contrastは一般的なoverhead校正には不十分で、primaryからこの値を引いたり、観測GC比率を高速化上限へ換算したりしない。全raw/全round/両役割を保存する。GCをPF区間外へ移しただけで速度改善とせず、生成object数と観測なしの完全PF時間、別の連続RTK実行を合わせて次の変更を評価する。

## 試験・保全・結論

新規21testsとD271回帰28testsが初回実行でPASS。phase復元、GC policy/callback所有権、実GC、thread跨ぎ/不正event記録、実Windows cycles、元return/例外同一性、clock外hook、実独立process IPCの順序を検査した。14400protocol eventをchild raw/parent/保存clocksへ別auditで全照合し、snapshot3600状態も独立再照合した。

仕上げに実Windows APIを使う2件へ他OS時のskip指定がないことを確認した。凍結済みtest本体と数値helperは変更せず、別collection pluginとconftestのplugin binding1文で対象2件だけをskipする。元conftest bytesを保存し、その1文を除いたfixture ASTが元と完全一致するproofを残す。新しいplatform分岐2testsと49回帰を変更後に実施し、51unique tests PASS。Linuxそのものの実行ではなく、両platform分岐と他test非変更をunitで確認した結果である。最初のcollection設定はbinding位置がimport前のためRuff E402となり、同じbindingをimport後へ移した設定だけを別proof/checked試験で確認した。最初の51PASS・設定bytes・任意pytest cache書込み拒否warning1件を保存し、checked設定試験はcache providerを使わない。数値test本体/benchmark/helperの変更や完了済み性能測定の再実行はない。

親runner初版はD275 planにD271形式の外部SHA keyがあると仮定してKeyErrorで停止した。plan作成・推定処理前の失敗を保持し、別checked runnerでD271の絶対path manifestを使った。field/helper/判定は変更していない。新test内のno-op callback lambda1行だけE731をSHA付きで除外し、初回実行sourceを維持する。

36製品source・5installed PE・PF build PE・112過去reports・55archives/11302filesを保全する。新しい数値kernel/build/sanitizerや製品/既定変更はない。今回の対象は一つのroute/seed/GPU/cluster0 prefixであり、fullroute/multiseed精度や速度は新たに主張しない。主要gateがFAILであることをそのまま記録し、完了済み高速化候補の測定は取り直さない。

[結果](../benchmarks/rbpf_gc_clocks_20261018.json)、[完了照合](../benchmarks/rbpf_gc_clocks_completion_check_20261018.json)、[clock helper](../benchmarks/rbpf_gc_clocks.py)、[実走caller](../benchmarks/bench_rbpf_gc_clocks_field.py)、[tests](../tests/test_rbpf_gc_clocks.py)。
