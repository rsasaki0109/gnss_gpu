# 実走入力でIRLS再計算省略を制御比較する (D243, 2026-10-16)

実走から全394スコア入力を保存し、同じDLLで省略オン・オフを比較した。CPU component screen 0/2、GPU joint component screen 0/2、A/A失敗0/2。これはスコア計算単体の比較で、PF全体・FGO全体の高速化は認定しない。D236の実験基準、製品、既定設定を維持する。

## 入力と制御

Tokyo run3/seed20260710/400epochのD236 native CPUを、元のcluster_relin0/1の両設定で実行した。各設定200PF stepのうち197回のスコア入力を全件保存し、速度や省略回数による選別を行わない。仮説の挿入順、全観測配列のdtype/shape/bytes、全cfg、クラスタH/pos/Pをコピーする。元のscoreは1回だけ実行し、追加のcanonical/DD/cluster callbackは呼ばない。

保存処理あり・なしで400paired stepの全state行、軌跡の全配列bits、全設定、既存metadataが一致した。保存のcopy/serializationは診断step timer内にあり、そのcall_secondsから速度は比較しない。scalar NaN payloadの正規化という既存codecの限界を引き継ぐ。route全体・複数seedの精度試験ではない。

別の明示的DLLはD242 sourceから作成し、変更を逆適用するとD242 source全体に戻ることを独立に再計算した。オフはD226の元のreweight関数と全solve、オンはD242の全Aw/bw binary64 bit比較と同一状態の再計算省略を実行する。FP64/行順/strict partial-pivot/事前分布/閾値/仮説数は保ち、epsilon打切りはない。同じbinary/workspace/extra counter転送/query/allocationを使い、制御はworkspace生成時だけ明示設定する。DLLの初期値はoffで、製品への自動導入はない。

replayは元の数値method codeとD236のnative canonical/parent viewを復元する。グラフobjectは復元せず、対象は完成したscore呼び出しだけ。CPU・GPU間のbit一致は要求せず、それぞれの同一backendの元実装と比較した。

## 精度と検査

122tests PASS/既知pytest設定warning1。両制御でscore・solution・normal・logdet・status・入力bits、uniform/Huber/cluster/shape、signed zeroの同一状態、empty/partial block、asymmetric/singular/tiny/huge prior、unsupported/capacity/stale counterを検査した。CPU/GPU×2seed×4構成×4epoch×両制御で全PF state/RNG/graph event/既存metadataも一致する。CUDA memcheck90、racecheck20、synccheck20 PASS（各既知設定warning1）、error/leak/hazard0。一般versionやASanの保証へは拡張しない。

実走全394バッチのwarm比較は7modeで2758呼び出し。CPUは保存した実走出力にbit一致し、各backendの旧版/off/onのscore・solution・normal・logdet・statusがbit一致した。timed全41370呼び出しも同一backendのwarm結果にbit一致する。

各backendで、cluster0は4166unique score仮説のうち1401で2802solveを省略し、cluster1は4063のうち1329で2658を省略した。全timed on/off呼び出しの仮説別省略vectorを保存し、warmと同一であることを再検証した。offは全件0。counterは作業量であり速度比ではない。

## 全件のペア比較

元CPU/off CPU/on CPU/元GPU/off GPU/on GPU/元CPU A/Aの7modeを直列で回転・反転する。全197batch×2設定×5round×3group×7mode=41370timed call。対象inputを毎callコピー・復元してから計測し、score内のdedup/packing/compute/transfers/readback/fallbackとcounter query/array確保をすべて含む。setup/control、input復元、diagnostics/warmup、hash/比較/記録は時間外。1許可CPU/BLAS1thread/通常priority、非占有host/GPU、GPU処理の重複なし。新しい実走入力の初回測定であり、D242の既測定合成入力を再測定していない。

各groupの197batch合計時間のpaired短縮率→round内3group中央値→5round中央値を使う。10%以上、4/5round以上改善、CPU A/A abs5%未満が条件。CPUは同一DLL offと元CPUの双方、GPUは同一DLL offと元GPUとon CPUの全てに条件を要求する。以下のp50時間比とは別の統計量である。全不利なroundと失敗対照を保持し、測定をやり直さない。

| cluster_relin | CPU対off | CPU対元版 | GPU対off | GPU対元版 | GPU対on CPU | CPU A/A | CPU screen | GPU joint |
|:---:|---:|---:|---:|---:|---:|---:|:---:|:---:|
| 0 | +2.30% | -0.45% | +0.74% | -0.50% | -158.76% | +0.15% | FAIL | FAIL |
| 1 | +3.87% | +0.82% | -0.24% | -0.36% | -102.73% | -0.01% | FAIL | FAIL |

初回の独立auditは、通常の逐次加算とPython3.12 sum()の補償加算の丸め差で停止した。初版auditと診断を保存し、測定値/source/DLL/数値bitsの判定を変えず、照合側をlist集約後のsumへ修正した。全210groupでmath.fsumと一致し、整数nanosecondの再集計とも照合した。全screen/A/A判定は元から同じで、測定の再実行はない。

197batch合計時間の中央値を197で割った平均batch時間（ms、個別batch p50ではない）:

| cluster_relin | 元CPU | off CPU | on CPU | 元GPU | off GPU | on GPU |
|:---:|---:|---:|---:|---:|---:|---:|
| 0 | 1.844 | 1.885 | 1.903 | 4.437 | 4.457 | 4.459 |
| 1 | 2.557 | 2.634 | 2.552 | 6.193 | 6.256 | 6.274 |

本比較は同一DLLの制御と元版の両方を用い、D242で未分離だった再compile/layoutの影響を制御比較から除いた。新DLLのbranch/control/query等の費用は全て候補側に含める。全PFやgraphの速度へ換算したり、異なるworkloadの過去の改善率に足し合わせたりしない。

## 境界と保存

capacity512のhost/device stagingは各1145400bytes/contextでD242と同じ、D226比各+2048bytes。workspaceにbool制御を追加し、64bit ABI上のobject layoutはpadding込みで8bytes増えると算定した（heap peakの測定ではない）。追加device allocation/cross-H/epoch cacheはない。benchmarkは元1+制御2contexts、Python last_skipped最大2048bytes/context。入力corpus・全timed vector・2758warm記録はprivate archiveに保存する。

36製品source、5installed extension、PF build PE、元尤度DLLと3CPU拡張、46既報、22archive/2871filesを照合して保全した。実装とDLLは初回testから不変。PF/graphの既定へ採用するには別の全体性能gateが必要であり、このcomponent結果だけでは採用しない。

[全結果](../benchmarks/rbpf_irls_replay_20261016.json)、[完了照合](../benchmarks/rbpf_irls_replay_completion_check_20261016.json)、[replay harness](../benchmarks/bench_rbpf_irls_replay.py)、[probe/API](../benchmarks/probes/rbpf_irls_replay/README.md)。
