# RBPFのcanonical keyとDD展開を呼出内で再利用する (D229, 2026-10-16)

D228の同一move生成の再利用に、canonical key生成の再利用を追加した。軽量な`MoveKeyMemoStep`は候補を並べるmove blockだけで共有する。重複の多い合成2条件で、D228のC++ callerからさらに12.19%/12.72%短縮した。残る4条件には約7%の遅延もあり、製品・凍結RBPF・既定設定は変更しない。

先に試した`EpochMemoStep`はstep全体のcanonicalとDD配列を共有する。効果が安定せず、DDだけの追加効果は全6条件で10%基準を通らなかった。両実装と全rawを明示選択の実験として残す。後の軽量版実験に置いた同じfull-epoch版は固定対照であり、先のA/A不合格や採否を上書きしない。

## 何を共有するか

canonical keyの初回は元の関数で計算する。通常のplain dict/builtin intからなるHだけをobject identityで再利用し、Hも保持してid再使用を防ぐ。空plain dictは元の結果と同じ空tuple。異なる辞書をidentityで混同せず、非標準mapping/integerは毎回元処理を呼ぶ。凍結codeではそのscopeの初回lookup後に同じHを変更しない。keyはimmutable tuple、次のepochへ持ち越さない。

軽量版はD228のmove/weight/choiceコードをそのまま実行し、そのstaging中のcanonical lookupだけを置く。実runtimeのbound method/globalは元のまま。DDを保存せず、元のcommitの前にmemoが終了する。D228のordered/absolute integer keyによるextend/shrink再利用、粒子ごとのdeep copy、strict-policy完全step fallback、全候補scalar fallbackを維持する。

full-epoch版は元methodと同じcode objectを呼出専用のstate viewへbindし、専用global dict中のcanonicalだけを差し替える。元moduleのglobalや実runtimeのmethodを一時変更しない。DDはplain整数Hのcanonical値と同じproblem object identityに限って共有する。整数差はgaugeで不変で、元のDD生成式を変更しない。凍結consumerは配列を変更しない。problemも保持してid再使用を防ぎ、全method/source guardを確認する。正常・skip・例外のいずれもfinallyでmemo参照を外す。任意のcallback副作用や同一filterの並行呼出しは実験domain外。

全粒子・要求move・重複候補のprior・辞書順・絶対整数値・選択順・RNG順を保つ。FP64、元の2IRLS、尤度式、探索範囲、FIX/hold閾値、graph solverを変更しない。native CUDA sourceとDLLはD226のpageable版のまま。

## 軽量版の完全step比較

6条件×5round×3group×5mode、450 timed calls。D228 C++、軽量C++/GPU、固定full-epoch C++、同一D228 C++ A/Aを回転して比較した。新候補比較180、固定対照90、A/A90。別途untimed D228 GPUを同じbackendの精度対照にする。graph抽出・integer-search・時刻/衛星入力はfixture、候補生成/key・pack/転送/返却・条件付き位置・mass/FIX・resamplingは時間内。restore/warm/検証/保存は時間外。

自processの1許可CPU・1BLAS thread、通常priority、host/GPU非占有。短縮率は同じgroup内の比をround内中央値→5round中央値にする。D228 C++比10%以上・4/5round改善・A/A中央値abs5%未満を要求。独立p50同士の比をpaired値と混同しない。条件やroundを捨てず、再試行しない。

| 粒子 / 初期固有H / rows / groups | D228 C++ ms | 軽量C++ ms | paired短縮 | 改善round | A/A短縮 | screen |
|:---|---:|---:|---:|---:|---:|:---|
| 64 / 6 / 64 / 4 | 17.163 | 14.991 | +12.19% | 5/5 | -4.59% | PASS |
| 64 / 32 / 64 / 4 | 27.010 | 26.661 | -0.61% | 2/5 | +0.08% | FAIL |
| 64 / 64 / 64 / 4 | 37.659 | 39.976 | -7.15% | 1/5 | -2.95% | FAIL |
| 128 / 32 / 64 / 4 | 39.620 | 37.138 | +8.62% | 5/5 | -0.98% | FAIL |
| 64 / 6 / 128 / 8 | 33.640 | 29.110 | +12.72% | 5/5 | +1.17% | PASS |
| 128 / 128 / 128 / 8 | 966.407 | 966.236 | +1.16% | 4/5 | +0.46% | FAIL |

軽量GPU対軽量C++は全6条件で不合格。GPU既定化や自動選択を追加しない。最初の条件のA/Aは-4.59%で合格範囲の端に近く、単一環境のscreenとして扱う。

## full-epoch版の最初の比較

同じ6条件・5round×3groupで別の450 timed calls。D228 C++、全step canonicalのみ、全step canonical+DD、同GPU、同一D228 A/A。新候補比較270、A/A90。canonicalのみもstate viewを通るため、軽量版と同じ実装ではない。先の結果を合格へ寄せる再測定は行わない。

| 粒子 / 初期固有H / rows / groups | 全step canonicalのみ vs D228 | canonical+DD vs D228 | DD追加分 vs canonicalのみ | A/A短縮 | full CPU screen |
|:---|---:|---:|---:|---:|:---|
| 64 / 6 / 64 / 4 | +9.69% | +6.74% | +2.40% | -10.07% | FAIL |
| 64 / 32 / 64 / 4 | +11.56% | +8.75% | +1.74% | +3.49% | FAIL |
| 64 / 64 / 64 / 4 | +8.41% | +10.51% | +1.51% | -5.03% | FAIL |
| 128 / 32 / 64 / 4 | +6.63% | +12.13% | +4.93% | -3.47% | PASS |
| 64 / 6 / 128 / 8 | +7.46% | +16.94% | +8.89% | -9.49% | FAIL |
| 128 / 128 / 128 / 8 | +1.69% | +0.97% | +0.31% | -0.65% | FAIL |

full版は128粒子/初期32HだけPASS。canonicalのみは64粒子/初期32HだけPASS。3条件のA/A不合格は保持し、-5.034%を丸めて合格にしない。DD追加分とGPU対full C++はいずれも0/6合格。別実験の固定full版が速く出た条件も、この最初の判断を変更する根拠に使わない。
最初の合成条件のfull memoではcanonical 777要求→元関数131call、DD 96要求→元生成18call。保持DD payloadは4,608bytes。これはNumPy配列payloadであり、Python辞書/H/key/viewやprocess最大heapを含むpeak測定ではない。管理費用があるためcall削減率を速度倍率にしない。

両実験計900timed calls、新候補比較450、固定full版対照90、A/A180、保存warm state72。全保存stateと全timed state hashは同じbackendのD228と一致し、再利用変更に数値許容差を設けていない。元NumPyとnative自体の微小差はD226/D227のまま。別backendの完全bit一致や新しい全route測位誤差の合格は主張しない。

## 実走での照合

実GTSAM/cssrlibとTokyo run1/run2/run3、seed20260710/20260711のprefixを使う。軽量C++は計1,200 PF step、full C++と軽量GPUは各200 stepで、同じbackendのD228と全保存state・軌跡NPZの全配列bytesが一致した。位置/P/FIX/hold/粒子/重み/RNG/feedbackを含む。NumPyのdtype/shape/bytesと有限Python scalarのbitsを保存し、Python scalar NaN payloadのcanonical化は従来codecの制限として残す。

| prefix / 実装 | PF step | memo内canonical要求 → 元関数call | 呼出削減 | 新規対照PF秒 → 候補PF秒 |
|:---|---:|---:|---:|---:|
| local_r3_seed0 | 200 | 206,522 → 18,027 | 91.27% | 15.816 → 10.794 |
| local_r1_seed0 | 200 | 211,977 → 17,966 | 91.52% | 8.814 → 5.888 |
| local_r2_seed0 | 200 | 207,168 → 16,960 | 91.81% | 凍結対照（精度のみ） |
| local_r3_seed1 | 600 | 628,030 → 55,641 | 91.14% | 凍結対照（精度のみ） |
| full_epoch_r3_seed0 | 200 | 273,526 → 30,795 | 88.74% | 15.816 → 18.478 |
| local_GPU_r3_seed0 | 200 | 206,522 → 18,027 | 91.27% | 凍結対照（精度のみ） |

軽量C++4prefixでは全1,253,697要求moveを保持し、memo対象canonical 1,253,697要求に対して元関数108,594call。これはstaging内の計数であり、step全体のcanonical総callや全処理時間の削減率ではない。各extend/shrink call、unique score数、fallback数もD228と同じ。run2/別seedの事前指定したprimary対照は凍結D227 native callerであり、move call数には保存state一致を確認済みのD228再利用companionを使う。両record/snapshot hashと既存の一致auditも照合する。入力データhash、公開source、readonly外部tc Python、DLLを前後照合した。

各新規実走は1回の直列・unpinned・非占有診断。snapshot保存はPF timerの外だが、compile/cache/負荷/実行順の影響を統制していない。凍結対照は精度照合だけに使い、過去の時間と比較しない。これらの時間を統制されたwhole-graph速度gateに数えない。全route/multi-seed精度gateは未完了で、既定適用は引き続き保留する。

## 検証と利用

full版68tests・軽量版29testsの計97新規tests PASS。各pytest設定warning1。3seed×4epoch、cluster/uniform/Huber/IRLS、shadow/spawn/resampling、maintenance/skip、strict例外時の部分RNG/state、feedback commit/release、巨大整数gauge/辞書順、problem分離、非標準integer bypassを検証した。full版は正常/例外後のmemo参照解放と元method code objectも確認する。
CUDA memcheckは12 targeted tests PASS、設定warning1、0 error/0 leak。native数値source/DLLは同一で再buildなし。以前のracecheck/synccheckを今回の新規PASSに足さない。36製品source、5installed extensions、PF build PE、過去18reportと7archiveを照合した。

```python
from rbpf_likelihood_batch import BatchScorer
from rbpf_move_key_memo import MoveKeyMemoStep

saved_step = pf.step
with BatchScorer(explicit_dll_path) as scorer:
    pf.step = MoveKeyMemoStep(pf, scorer, mode="native_cpu")
    try:
        result = pf.step(rtk, epoch_context)
    finally:
        pf.step = saved_step
```

再現: [軽量比較](../benchmarks/bench_rbpf_move_key_memo.py)、[full比較](../benchmarks/bench_rbpf_epoch_memo.py)、[実走wrapper](../benchmarks/bench_rbpf_epoch_memo_field.py)（軽量は`--move-key-memo`、fullは`--epoch-memo`、省略でD228）。記録: [全結果](../benchmarks/rbpf_epoch_memo_20261016.json)、[完了照合](../benchmarks/rbpf_epoch_memo_completion_check_20261016.json)。
