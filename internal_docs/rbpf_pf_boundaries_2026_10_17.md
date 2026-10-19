# D263: PF本体の疎な境界計測 (2026-10-17)

今回は次の高速化箇所を選ぶための診断追加。新しい速度改善・GPU移行・既定採用はない。D262の全score短縮を完全PFの短縮へそのまま読み替えず、同じD262 scorerを使うPF本体の候補生成/score/抽選/位置/feedbackを分けて調べた。負荷校正は2/6に留まり、実走のoverheadも未校正のため、以下の比率は参考診断値として扱う。

実GPUの全197有効step/clusterで、requested全scoreを含む区間はc0 14.838%、c1 17.031%。move生成・canonical/index/登録は43.837/40.613%、粒子ごとの元重み式・抽選は11.343/11.036%。独立区間のwall観測であり、純GPU時間・CPU cycleや因果的な短縮上限ではない。

## 実装と精度

D235の完全step ASTとD236のfeedback/counter wrapperを保持し、top-levelに15個、元move本体に4個の直接境界markを追加した。outer2個を含む通常21mark/step。粒子ごとのtracing、cProfile、methodごとのcontext managerやRNG proxyは使わない。3個の既存adapterはcalleeだけをclosureに差し替え、戻すと元ASTになる。計5個のAST復元を検査した。clone対照は元inner code object、元の2個のhelper所有wrapper code objectをそのまま使う。

数値globalは同じlive dict。既存guard、型/例外/警告、全観測・候補・粒子・prior・FP64演算順・threshold・RNG・FIX/feedback/graph callbackを保持する。新しいnative/CUDA source、PE、buffer、追加転送はない。正常/途中例外/skip/strict元step fallbackでもfinallyで記録を公開し、callbackや入力参照をtraceに保持しない。traceはstring/intのimmutable tupleだけで、次callに数値を再利用しない。

23tests PASS（既存pytest config warning1）。CPU/GPU×2seed×3構成×4epochでoriginal/clone/profileの全PF状態・RNG・graph/prior/caller metadata、D262 countersがbits一致。AST・live globals/closure・wrapper同一性、skip/strict、抽出/RNG/feedback例外、部分state、次callへのtrace更新、helper変更拒否、reentry、壊れたpartitionも検査した。試験済みsourceはLFで固定したまま保存し、後からEOLを書き換えなかった（SHA付き例外を結果へ記録）。

## 完全PFの負荷校正

元PF/同code clone/profile/元A/A/profile A/AをCPU/GPUで計10mode、3構成、16個の事前RNG初期状態、5round×3group。7200timed/480warm。構成は64/6/64/4 c0、同c1、128/32/128/8 c1（particles/unique/rows/groups）。観測/config/初期particlesは構成ごと固定、RNG初期状態はseed20260710〜20260725。元の各粒子のchoiceは全回そのまま実行する。

各groupの16完全PF stepのinteger-ns合計からpaired overheadを計算し、3group中央値→5round中央値。全5contrastのabsmedian<5%を事前校正条件とし、2/6構成/backendが通過。全失敗/負値/揺れを保存し、完了済み計測のやり直しはない。これは新しい診断の校正であり、D262最適化の速度を再判定する試験ではない。

| 構成/backend | profile/元 | profile/clone | clone/元 | 元A/A | profile A/A | 校正 |
|---|---:|---:|---:|---:|---:|---|
| 0/cpu | +8.291% | +0.470% | +4.712% | +1.035% | -0.674% | False |
| 0/gpu | +0.119% | +2.297% | +1.187% | +2.018% | +3.034% | True |
| 1/cpu | +4.634% | +10.222% | -8.815% | -5.729% | -1.326% | False |
| 1/gpu | -2.255% | -6.881% | +0.754% | +4.947% | -0.087% | False |
| 2/cpu | -6.527% | -3.418% | -2.758% | +5.847% | +3.193% | False |
| 2/gpu | +0.505% | +0.295% | +1.425% | -2.212% | -0.470% | True |

外部timerは完全PFのguard・scope準備・clock/記録・trace tuple公開/cleanupを含む。restore/deepcopy/callback再bind/hash/validation/JSON/gzipは外。許可CPU1本/BLAS1/normal priority/非占有、CPU/GPU処理は直列。観測抽出/整数探索/時刻・衛星入力/graph solveはfixture境界であり、全GTSAMの性能ではない。全7680記録の順序・件数・保存状態/metadata hashと全3072traceのinteger partitionを独立auditした。480warmの完全state/metadataもdecodeしてbits再照合した。

## 実走の区間

Tokyo run3/seed20260710/400epoch、cluster0/1各1prefix。D262のscore引数diagnosticsをそのまま転送し、追加診断転送を要求しない。元D236 GPUの同設定参照へ、計400保存PF状態・全trajectory配列dtype/shape/bytes・全cfg・従来prior/caller metadataが一致。197有効/3skipずつ、計400traceを全件順序/状態/partition照合し、394有効traceをすべて使う。skip6件も保存し、未到達区間は0として作らない。途中省略や有利なepoch選択はない。

表は各有効stepの区間/記録root比の中央値。項目ごと独立中央値なので合計100%とは限らない。recorded rootには親controller準備/metadata返却を含み、trace tuple公開等の一部末尾作業は含まない（校正の外部timerには含む）。snapshot/gzipはPF呼出し後で区間外だがcache/scheduling影響は残る。fixtureの校正通過を実走の無計測対照での負荷校正と扱わない。実走のoverhead、全RTK/FGO速度、GPU化率は未確立。

| 区間 | c0 | c1 |
|---|---:|---:|
| caller guard・scope準備 | 0.138% | 0.149% |
| epoch初期化 | 0.083% | 0.078% |
| 観測抽出・witness/support準備 | 6.197% | 5.895% |
| maintenance・spawn | 0.282% | 0.280% |
| DDPR位置/品質 | 0.709% | 0.689% |
| burst基準・shadow評価 | 1.020% | 1.012% |
| consensus/adopt候補 | 1.097% | 1.038% |
| move adapter/memo準備 | 0.053% | 0.056% |
| 全move生成・canonical/index/登録 | 43.837% | 40.613% |
| requested全score・cache反映・burst混合 | 14.838% | 17.031% |
| 元の重み式・粒子ごとの抽選 | 11.343% | 11.036% |
| move metadata・adapter返却 | 1.391% | 1.275% |
| 選択Hのcopy/commit | 0.926% | 0.824% |
| 正規化・basin mass/rank/rep | 5.251% | 5.049% |
| top basin条件付き位置 | 6.041% | 7.090% |
| FIX component・追加条件付き位置・witness | 1.741% | 2.051% |
| hold release/commit・graph callback | 1.804% | 1.703% |
| resample・cluster publish準備 | 0.118% | 0.149% |
| 結果publish | 0.010% | 0.010% |
| caller metadata・scope解放 | 0.594% | 0.542% |

各stepの隣接区間nsはrootへ厳密に足し戻る。区間にはallocation/wait/schedulingとmarkerの負荷が入り、計測なしのCPU/GPU内訳ではない。c0の大きい4区間は全move生成・canonical/index/登録, requested全score・cache反映・burst混合, 元の重み式・粒子ごとの抽選, 観測抽出・witness/support準備、c1は全move生成・canonical/index/登録, requested全score・cache反映・burst混合, 元の重み式・粒子ごとの抽選, top basin条件付き位置。これらの中を次の候補としてcode/再利用可能性を確認する。比率から速度倍率を予告したり、過去に不採用だった案を再採用したりしない。

## 保存

1所有D262 numeric contextの既存host/device stagingは各1143352B。追加は通常21個のstring/int markとtrace tuple、実走記録listは保存用で200call分。新CUDA buffer・cross-call数値cacheはなく、Python heap peakは未測定。既存codecのPython scalar NaN payload正規化は継承し、新しい数値許容差は追加していない。新しいsanitizer実行・CPU実走/fullroute/multiseed accuracy試験・fault injectionはない。36製品source・5installed PE・PF build PE・86過去reports・42archives7092filesを照合した。D262の集計末尾停止と復元rawも過去archiveに保持する。

[結果](../benchmarks/rbpf_pf_boundaries_20261017.json)、[完了確認](../benchmarks/rbpf_pf_boundaries_completion_check_20261017.json)、[計測source](../benchmarks/rbpf_pf_boundaries.py)。
